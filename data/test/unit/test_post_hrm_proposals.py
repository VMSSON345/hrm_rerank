"""Tests for HRM arbitration of source-backed answer proposals."""
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
from hrm_finance_qa.post_hrm_proposals import select_source_proposal
from hrm_finance_qa.program_data import NumericCandidate


class Ranking:
    model = SimpleNamespace(config=SimpleNamespace(max_candidates=64))

    def __init__(self, prefer_target=True):
        self.calls = 0
        self.prefer_target = prefer_target

    def rank_raw(self, question, candidates):
        self.calls += 1
        scores = [100.0 if self.prefer_target and i == 0 else float(i)
                  for i in range(len(candidates))]
        return {'pointers': [scores] * 4, 'select': scores,
                'pairs': {}, 'segments': 1}


class ProposalSelectionTests(TestCase):
    def test_numeric_proposal_uses_hrm_and_other_sources_use_contract(self):
        source = (Path(__file__).resolve().parents[3] /
                  'data/test/suites/independent_reports_v2/sources/TAB.md')
        bundle = Ranking()
        engine = ContractDocumentEngine(bundle)
        cases = (
            ('Doanh thu thuần năm 2024 tăng hay giảm bao nhiêu phần trăm so với năm 2023?',
             'ACCEPTED', 'published_ratio_lookup', True),
            ('Doanh thu năm 2024 chủ yếu đến từ mảng kinh doanh nào?',
             'ACCEPTED', 'source_disclosure_lookup', False),
            ('Vay ngắn hạn và vay dài hạn năm 2024 là bao nhiêu?',
             'RAW_DISCLOSURE', 'unparsed_raw_source', False),
        )
        for question, status, proposal_route, neural in cases:
            with self.subTest(question=question):
                before = bundle.calls
                result = engine.answer(question, source)
                self.assertEqual(result['status'], status, result['rendered'])
                self.assertGreater(bundle.calls, before) if neural else self.assertEqual(bundle.calls, before)
                self.assertEqual(result['execution_route'],
                                 'hrm_source_proposal' if neural else 'source_contract_only')
                self.assertEqual(result['proposal_route'], proposal_route)
                if neural:
                    self.assertTrue(result['retrieval_trace']['selected_proposed_evidence'])
                else:
                    self.assertEqual(result['retrieval_trace']['mode'], 'source_contract_only')

    def test_hrm_shortlist_is_diagnostic_for_valid_source_proposal(self):
        name = 'source.md'
        lines = [f'Ratio | {i}%' for i in range(10)]
        digest = sha256(('\n'.join(lines) + '\n').encode()).hexdigest()
        snapshot = SimpleNamespace(
            lines={name: lines}, pages={name: {i: 1 for i in range(1, 11)}},
            hashes={name: digest}, unchanged=lambda: True,
        )
        candidates = []
        facts = []
        for i, line in enumerate(lines):
            start = line.index('|') + 2
            end = len(line)
            candidates.append(NumericCandidate(
                index=i, value=str(i), raw=line[start:end],
                context=f'ROW: Ratio\nVALUE_LINE: {line}',
                line_number=i + 1, char_start=start, char_end=end,
                unit='%', is_percent=True, is_year_like=False,
                relevance=0.0, source_name=name, page=1,
                row_label='Ratio', source_span=(start, end),
                source_span_scope='line_v2',
                source_line_sha256=sha256(line.encode()).hexdigest(),
            ))
            facts.append(SimpleNamespace(
                candidate_index=i, metric='ratio', source_verified=True,
                source_name=name, line_number=i + 1, source_span=(start, end),
            ))
        target = candidates[0]
        citation = {
            'source_name': name, 'page': 1, 'line_number': 1,
            'source_span': list(target.source_span),
            'source_line_sha256': target.source_line_sha256,
            'document_sha256': digest, 'raw_value': target.raw,
        }
        proposal = {'answer_type': 'published_ratio', 'citations': [citation]}
        bundle = Ranking(prefer_target=False)
        rejected, trace = select_source_proposal(
            bundle, 'Ratio?', proposal, snapshot, candidates, facts, [],
        )
        self.assertIs(rejected, proposal)
        self.assertEqual(trace['reason'], 'hrm_selected_unverified_evidence')
        self.assertFalse(trace['selected_proposed_evidence'])
        self.assertEqual(trace['proposal_best_rank'], 10)
        bundle.prefer_target = True
        accepted, trace = select_source_proposal(
            bundle, 'Ratio?', proposal, snapshot, candidates, facts, [],
        )
        self.assertIs(accepted, proposal)
        self.assertEqual(trace['proposal_best_rank'], 1)
        invalid = dict(citation, document_sha256='wrong')
        rejected, trace = select_source_proposal(
            bundle, 'Ratio?', dict(proposal, citations=[invalid]),
            snapshot, candidates, facts, [],
        )
        self.assertIsNone(rejected)
        self.assertEqual(trace['reason'], 'proposal_source_binding_failed')
