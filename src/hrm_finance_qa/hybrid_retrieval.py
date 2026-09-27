"""Budgeted lexical/dense/structural retrieval over unchanged source candidates.

Graph edges guide retrieval only. Source contracts still establish all facts.
"""
from collections import defaultdict, OrderedDict
from dataclasses import replace
from .document_sources import rank_chunks, expand_neighbor_chunks
from .program_data import merge_candidates
from .evidence_schema import build_evidence_graph, EvidenceRelation


def physical_key(c):
    return c.source_name, c.line_number, tuple(c.source_span or ()), c.source_line_sha256


class HybridEvidenceRetriever:
    def __init__(self, encoder=None):
        self.encoder = encoder
        self.cache = OrderedDict()

    def trace_existing(self, candidates, catalog, facts, slots, matches, limit):
        """Trace an already sufficient lexical pool without another rerank."""
        ids = {physical_key(c): c.index for c in catalog}
        by_local = {c.index: ids.get(physical_key(c)) for c in candidates}
        kept = {i for i in by_local.values() if i is not None}
        relevant = {f.candidate_index for f in facts if any(matches(f,slot) for slot in slots)}
        graph = build_evidence_graph(candidates)
        paths = []
        for edge in graph.edges:
            left, right = by_local.get(edge.source), by_local.get(edge.target)
            if left is not None and right is not None:
                paths.append({'from':left,'to':right,'relation':edge.relation.value})
        paths.sort(key=lambda p:(not(p['from'] in relevant or p['to'] in relevant),p['from'],p['to'],p['relation']))
        return {'mode':'adaptive_graph','strategy':'trace_sufficient_lexical_pool',
                'dense':'not_needed_complete_lexical_pool','budget':limit,
                'catalog_candidates':len(catalog),'bm25_candidates':len(candidates),
                'selected_candidates':len(candidates),'graph_nodes':len(candidates),
                'graph_edges':len(graph.edges),'graph_index_space':'catalog',
                'graph_paths':paths[:256],'graph_paths_truncated':len(paths)>256,
                'graph_added_selected':[],
                'slots':[{'id':slot.id,'available':sum(matches(f,slot) for f in facts),
                          'retained':sum(f.candidate_index in kept and matches(f,slot) for f in facts)} for slot in slots],
                'candidates':[{'index':c.index,'catalog_index':ids.get(physical_key(c)),
                               'source_name':c.source_name,'line_number':c.line_number,
                               'source_span':list(c.source_span or ()),'origins':['bm25']} for c in candidates]}

    def retrieve(self, question, chunks, catalog, facts, slots, matches, limit, *, use_graph=True):
        by_key = {physical_key(c): c for c in catalog}
        by_id = {c.index: c for c in catalog}
        origins = defaultdict(set)
        ranks = defaultdict(float)
        lexical = expand_neighbor_chunks(rank_chunks(question, chunks, top_k=8), chunks, neighbor_window=1, max_chunks=12)
        legacy = merge_candidates(question, lexical, max_candidates=limit)

        def add_rank(candidates, origin):
            ids = []
            for c in candidates:
                canonical = by_key.get(physical_key(c))
                if canonical is not None and canonical.index not in ids:
                    ids.append(canonical.index)
            for rank, i in enumerate(ids, 1):
                origins[i].add(origin)
                ranks[i] += 1 / (60 + rank)
            return ids

        lexical_ids = add_rank(legacy, 'bm25')
        dense_state = 'unavailable'
        if self.encoder is not None and chunks:
            import numpy as np
            # Cache exactly the source chunk texts; no question-derived facts.
            key = tuple((c.source_path, c.line_start, c.text) for c in chunks)
            vectors = self.cache.get(key)
            if vectors is None:
                vectors = self.encoder.encode([c.text for c in chunks], role='evidence')
                self.cache[key] = vectors
                while len(self.cache) > 3:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            query = self.encoder.encode([question], role='query')[0]
            scores = vectors @ query
            order = np.argsort(-scores, kind='stable')[:8]
            dense_chunks = [chunks[int(i)] for i in order]
            add_rank(merge_candidates(question, dense_chunks, max_candidates=limit), 'dense')
            dense_state = 'used_existing_encoder'

        slot_ids = []
        for slot in slots:
            ids = sorted({f.candidate_index for f in facts if matches(f, slot)},
                         key=lambda i: (-ranks[i], i))
            slot_ids.append((slot.id, ids))
        mandatory = []
        # Round-robin slots before truncation; "other" is protected by metric
        # identity like every requested metric, not by a global keyword boost.
        reserve = max(len(slots), limit // 2)
        for offset in range(max((len(ids) for _, ids in slot_ids), default=0)):
            for slot_id, ids in slot_ids:
                if offset < len(ids):
                    i = ids[offset]
                    origins[i].add('required_slot:' + slot_id)
                    if i not in mandatory:
                        mandatory.append(i)
            if len(mandatory) >= reserve:
                break
        seed_ids = list(dict.fromkeys(mandatory + sorted(ranks, key=lambda i: (-ranks[i], i))))[:limit]
        paths = []
        graph_ids = []
        graph_edges = 0
        graph_priority = {}
        if use_graph and seed_ids:
            row_keys = {(by_id[i].source_name, by_id[i].table_id, by_id[i].row_id) for i in seed_ids if by_id[i].table_id and by_id[i].row_id}
            table_keys = {(by_id[i].source_name, by_id[i].table_id) for i in seed_ids if by_id[i].table_id}
            neighbors = [c.index for c in catalog if c.index not in seed_ids and (c.source_name,c.table_id,c.row_id) in row_keys]
            neighbors += [c.index for c in catalog if c.index not in seed_ids and c.index not in neighbors and (c.source_name,c.table_id) in table_keys]
            graph_ids = list(dict.fromkeys(seed_ids + neighbors))[:max(limit, 2 * limit)]
            graph = build_evidence_graph([replace(by_id[i], index=n) for n,i in enumerate(graph_ids)])
            graph_edges = len(graph.edges)
            seed_set = set(seed_ids)
            allowed = {EvidenceRelation.SAME_ROW, EvidenceRelation.SAME_TABLE,
                       EvidenceRelation.NEXT_PERIOD, EvidenceRelation.PREVIOUS_PERIOD,
                       EvidenceRelation.PART_OF_TOTAL}
            for edge in graph.edges:
                left, right = graph_ids[edge.source], graph_ids[edge.target]
                if left not in seed_set or edge.relation not in allowed or by_id[left].source_name != by_id[right].source_name:
                    continue
                if edge.relation in {EvidenceRelation.NEXT_PERIOD,EvidenceRelation.PREVIOUS_PERIOD}:
                    from .post_hrm_source import metric_key
                    if metric_key(by_id[left].row_label)!=metric_key(by_id[right].row_label):
                        continue
                if edge.relation==EvidenceRelation.SAME_TABLE and abs(by_id[left].line_number-by_id[right].line_number)>2:
                    continue
                priority = 0 if edge.relation==EvidenceRelation.SAME_ROW else 1 if edge.relation in {EvidenceRelation.NEXT_PERIOD,EvidenceRelation.PREVIOUS_PERIOD} else 2
                if right not in seed_set:
                    graph_priority[right] = min(graph_priority.get(right,99),priority)
                origins[right].add('graph:' + edge.relation.value)
                if len(paths) < 256:
                    paths.append({'from': left, 'to': right, 'relation': edge.relation.value})
                ranks[right] = max(ranks[right], 1 / (90 + graph_ids.index(right)))
        graph_reserved = sorted(graph_priority,key=lambda i:(graph_priority[i],-ranks[i],i))[:min(limit//4,max(0,limit-len(mandatory)))]
        selected_ids = list(dict.fromkeys(mandatory + graph_reserved + sorted(ranks, key=lambda i: (-ranks[i], i))))[:limit]
        selected = [replace(by_id[i], index=n) for n,i in enumerate(selected_ids)]
        retained = set(selected_ids)
        trace = {'mode': 'hybrid_graph' if use_graph else 'hybrid', 'dense': dense_state,
                 'budget': limit, 'catalog_candidates': len(catalog), 'bm25_candidates': len(lexical_ids),
                 'selected_candidates': len(selected), 'graph_nodes': len(graph_ids), 'graph_edges': graph_edges,
                 'graph_index_space':'catalog','graph_paths': [p for p in paths if p['to'] in retained],
                 'graph_added_selected': [i for i in graph_reserved if i in retained],
                 'slots': [{'id': name, 'available': len(ids), 'retained': len(retained.intersection(ids))} for name,ids in slot_ids],
                 'candidates': [{'index': n, 'catalog_index': i, 'source_name': by_id[i].source_name,
                                 'line_number': by_id[i].line_number, 'source_span': list(by_id[i].source_span or ()),
                                 'origins': sorted(origins[i])} for n,i in enumerate(selected_ids)]}
        return selected, trace
