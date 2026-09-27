"""Trace HRM source ranking while source verification controls publication.

The extractors propose an answer and physical spans.  The neural pointer may
select a span, but the source snapshot and fact catalog retain authority over
the value, period, unit, and final publication decision.
"""
from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import re

from .document_sources import rank_chunks
from .program_data import NumericCandidate


def _physical(candidate):
    return candidate.source_name, candidate.line_number, tuple(candidate.source_span or ())


def _citation_key(citation):
    return citation['source_name'], citation['line_number'], tuple(citation['source_span'])


def _source_line(snapshot, citation):
    name, number, _ = _citation_key(citation)
    original = snapshot.lines[name][number - 1]
    for line in (original, re.sub(r'\s+', ' ', original).strip()):
        if sha256(line.encode()).hexdigest() == citation.get('source_line_sha256'):
            return line
    return original


def _line_candidate(snapshot, citation):
    name, number, span = _citation_key(citation)
    line = _source_line(snapshot, citation)
    raw = line[span[0]:span[1]]
    return NumericCandidate(
        index=0, value='0', raw=raw, context=f'ROW: {line}\nVALUE_LINE: {line}',
        line_number=number, char_start=span[0], char_end=span[1],
        unit='', is_percent=False, is_year_like=False, relevance=0.0,
        source_name=name, page=snapshot.pages[name][number],
        extraction_method='mineru_markdown', row_label=line[:160],
        source_span=span, source_span_scope='line_v2',
        source_line_sha256=sha256(line.encode()).hexdigest(),
    )


def _valid_citations(snapshot, citations):
    if not citations or not snapshot.unchanged():
        return False
    for citation in citations:
        try:
            name, number, (start, end) = _citation_key(citation)
            line = _source_line(snapshot, citation)
            if not (0 <= start < end <= len(line)):
                return False
            if citation['document_sha256'] != snapshot.hashes[name]:
                return False
            if citation['source_line_sha256'] != sha256(line.encode()).hexdigest():
                return False
            if citation.get('page') != snapshot.pages[name][number]:
                return False
            if citation.get('raw_value') and citation['raw_value'] not in line[start:end]:
                return False
            if citation.get('text_value') and citation['text_value'] != line[start:end]:
                return False
        except (KeyError, IndexError, TypeError, ValueError):
            return False
    return True


def select_source_proposal(bundle, question, proposal, snapshot, candidates, facts, chunks):
    """Rank proposed source spans against nearby evidence; fail closed on disagreement."""
    citations = proposal.get('citations') or []
    if not _valid_citations(snapshot, citations):
        return None, {'reason': 'proposal_source_binding_failed'}
    targets = {_citation_key(citation) for citation in citations}
    by_key = {_physical(candidate): candidate for candidate in candidates}
    selected = [by_key.get(key) or _line_candidate(snapshot, citation)
                for citation in citations for key in [_citation_key(citation)]]
    selected = list({_physical(candidate): candidate for candidate in selected}.values())
    primary = selected[0]
    limit = max(1, min(64, bundle.model.config.max_candidates))
    # Similar source facts are genuine competitors for numeric disclosures.
    if proposal.get('answer_type') == 'published_ratio':
        target_facts = [fact for fact in facts if fact.source_name == primary.source_name
                        and fact.line_number == primary.line_number
                        and tuple(fact.source_span) == tuple(primary.source_span)]
        keys = {fact.metric for fact in target_facts}
        alternatives = [candidate for candidate in candidates
                        if candidate.index in {fact.candidate_index for fact in facts
                                               if fact.metric in keys and fact.source_verified}]
    else:
        # Text spans and unresolved numeric tokens need a common HRM input.
        # The source line is represented as a zero-valued evidence candidate;
        # its number is never used for an answer or computation.
        alternatives = []
        for chunk in rank_chunks(question, chunks, top_k=3):
            for number in range(chunk.line_start, chunk.line_end + 1):
                name = chunk.source_name
                if name not in snapshot.lines or number > len(snapshot.lines[name]):
                    continue
                line = snapshot.lines[name][number - 1]
                if not line.strip() or line.startswith('## Page'):
                    continue
                alternatives.append(_line_candidate(snapshot, {
                    'source_name': name, 'line_number': number,
                    'source_span': [0, len(line)],
                }))
                if len(alternatives) >= limit:
                    break
            if len(alternatives) >= limit:
                break
    present = {_physical(item) for item in selected}
    for candidate in alternatives:
        if _physical(candidate) not in present:
            selected.append(candidate)
            present.add(_physical(candidate))
        if len(selected) >= limit:
            break
    selected = [replace(candidate, index=index) for index, candidate in enumerate(selected[:limit])]
    if not selected:
        return None, {'reason': 'proposal_candidates_missing'}
    from .post_hrm_engine import _rank_with_timing
    ranking = _rank_with_timing(bundle, question, selected)
    scores = ranking.get('select') or (ranking.get('pointers') or [[]])[0]
    if len(scores) < len(selected):
        return None, {'reason': 'invalid_hrm_pointer_shape'}
    # HRM tensors are padded to the checkpoint's candidate capacity.
    scores = scores[:len(selected)]
    ordered = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    winner = selected[ordered[0]]
    # A full-line text competitor is equivalent to a cited subspan only when
    # it is the same physical line.  Numeric table cells retain exact spans.
    def proposed(candidate):
        return _physical(candidate) in targets or (
            proposal.get('answer_type') == 'text'
            and any(candidate.source_name == name and candidate.line_number == number
                    for name, number, _ in targets)
        )
    # The pointer supplies a short proposal list; the source verifier rejects
    # candidates without the required physical citation.
    shortlist = ordered[:min(8, len(ordered))]
    verified = next((selected[index] for index in shortlist
                     if proposed(selected[index])), None)
    selected_target = verified is not None
    trace = {
        'mode': 'source_proposal_hrm', 'candidate_count': len(selected),
        'segments': ranking.get('segments', 0),
        'selected': {'source_name': winner.source_name,
                     'line_number': winner.line_number,
                     'source_span': list(winner.source_span)},
        'proposed_citations': len(citations),
        'selected_proposed_evidence': selected_target,
        'verified_selected': ({'source_name': verified.source_name,
                               'line_number': verified.line_number,
                               'source_span': list(verified.source_span)}
                              if verified is not None else None),
        'proposal_best_rank': next((rank + 1 for rank,index in enumerate(ordered)
                                    if proposed(selected[index])), None),
        'shortlist_size': len(shortlist),
    }
    if not selected_target:
        trace['reason'] = 'hrm_selected_unverified_evidence'
        # The pointer shortlist is diagnostic. Valid source citations remain
        # eligible even when the checkpoint ranks them below eight.
    return proposal, trace


def verify_source_proposal(proposal, refreshed, snapshot):
    """Recheck the extractor's metric, time, unit and physical span after HRM."""
    if refreshed is None:
        return False
    for field in ('status', 'answer_type', 'canonical_value', 'unit',
                  'unit_state', 'rendered'):
        if proposal.get(field) != refreshed.get(field):
            return False
    citations = proposal.get('citations') or []
    if citations != (refreshed.get('citations') or []):
        return False
    return _valid_citations(snapshot, citations)
