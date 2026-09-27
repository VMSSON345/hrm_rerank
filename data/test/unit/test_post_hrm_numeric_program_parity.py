"""Parity and fact-pool tests for the experimental numeric program path."""
from pathlib import Path
from dataclasses import replace
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
from hrm_finance_qa.post_hrm_numeric_program import _pool
from hrm_finance_qa.post_hrm_formulas import margin_operand_plan
from hrm_finance_qa.post_hrm_source import metric_key
from hrm_finance_qa.program_data import NumericCandidate
from hrm_finance_qa.question_planner import compile_question_plan, OperandSlot
from test_post_hrm_contract import fact


class SourceOnlyRanking:
    model = SimpleNamespace(config=SimpleNamespace(max_candidates=64))

    def rank_raw(self, question, candidates):
        return {'pointers': [[0.0] * len(candidates)] * 4,
                'select': [0.0] * len(candidates), 'pairs': {}, 'segments': 0}


class NumericProgramParityTests(TestCase):
    def test_multiple_fact_interpretations_survive_one_physical_candidate(self):
        question = 'Tài sản ngắn hạn năm 2031 là bao nhiêu đồng?'
        plan = compile_question_plan(question)
        plan = replace(plan, operand_slots=(OperandSlot(
            'lookup', plan.metric_text, '2031', 'ACTUAL', plan.entity_mode,
            'currency_or_source', ('explicit_source_metric',)),))
        candidate = NumericCandidate(
            index=0, value='100', raw='100', context='ROW: Tài sản ngắn hạn',
            line_number=1, char_start=0, char_end=3, unit='VND',
            is_percent=False, is_year_like=False, relevance=0.0,
            source_name='synthetic.md', source_span=(0, 3))
        metric = metric_key('Tài sản ngắn hạn')
        common = dict(metric_label='Tài sản ngắn hạn', period='2031',
                      source_name='synthetic.md', candidate_index=0)
        first = fact(metric, 100, fact_id='first', scope='consolidated', **common)
        second = fact(metric, 100, fact_id='second', scope='separate', **common)
        candidates, facts, available = _pool(
            question, plan, [candidate], [first, second], [], 64)
        self.assertEqual((len(candidates), len(facts), available), (2, 2, 2))
        self.assertEqual({c.scope for c in candidates}, {'consolidated', 'separate'})
        self.assertEqual({f.fact_id for f in facts}, {'first', 'second'})

    def test_complete_verified_pool_is_not_pruned_by_retrieval(self):
        question = 'Tài sản ngắn hạn năm 2031 là bao nhiêu đồng?'
        plan = compile_question_plan(question)
        plan = replace(plan, operand_slots=(OperandSlot(
            'lookup', plan.metric_text, '2031', 'ACTUAL', plan.entity_mode,
            'currency_or_source', ('explicit_source_metric',)),))
        metric = metric_key('Tài sản ngắn hạn')
        candidates = [
            NumericCandidate(
                index=i, value=str(100 + i), raw=str(100 + i),
                context='ROW: Tài sản ngắn hạn', line_number=i + 1,
                char_start=0, char_end=3, unit='VND',
                is_percent=False, is_year_like=False, relevance=0.0,
                source_name='synthetic.md', source_span=(0, 3))
            for i in range(3)
        ]
        facts = [
            fact(metric, 100 + i, fact_id=f'fact-{i}',
                 metric_label='Tài sản ngắn hạn', period='2031',
                 source_name='synthetic.md', candidate_index=i)
            for i in range(3)
        ]
        complete = _pool(question, plan, candidates, facts, [], 3, [0])
        self.assertEqual((complete[2], len(complete[1])), (3, 3))
        budgeted = _pool(question, plan, candidates, facts, [], 2, [0, 2])
        self.assertEqual((budgeted[2], len(budgeted[1])), (3, 2))
        self.assertEqual({item.fact_id for item in budgeted[1]},
                         {'fact-0', 'fact-2'})

    def test_crowded_printed_ratios_do_not_remove_margin_operands(self):
        question = 'Biên lợi nhuận ròng năm 2031 là bao nhiêu phần trăm?'
        plan = margin_operand_plan(question)
        metrics = ['printed_margin'] * 10 + [
            metric_key('Lợi nhuận sau thuế'), metric_key('Doanh thu thuần')]
        candidates = [
            NumericCandidate(
                index=i, value=str(10 + i), raw=str(10 + i),
                context='ROW: source fact', line_number=i + 1,
                char_start=0, char_end=2, unit='%',
                is_percent=True, is_year_like=False, relevance=0.0,
                source_name='synthetic.md', source_span=(0, 2))
            for i in range(12)
        ]
        facts = [
            fact(metric, 10 + i, fact_id=f'fact-{i}',
                 metric_label=metric, period='2031', role='actual',
                 source_name='synthetic.md', line_number=i + 1,
                 candidate_index=i)
            for i, metric in enumerate(metrics)
        ]
        ratio_options = [(item, []) for item in facts[:10]]
        _, kept, available = _pool(
            question, plan, candidates, facts, ratio_options, 8,
            list(range(12)))
        self.assertEqual(available, 12)
        self.assertEqual(len(kept), 8)
        self.assertIn('printed_margin', {item.metric for item in kept})
        self.assertIn(metric_key('Lợi nhuận sau thuế'),
                      {item.metric for item in kept})
        self.assertIn(metric_key('Doanh thu thuần'),
                      {item.metric for item in kept})

    def test_retrieval_trace_keeps_complete_pool(self):
        source = ('## Page 1\nBảng cân đối kế toán hợp nhất\nĐơn vị: đồng\n'
                  'Chỉ tiêu | Số cuối kỳ 31/12/2031\n--- | ---\n'
                  'Tài sản ngắn hạn | 110.000.000\n')
        result = self._answer_pair(source)[1]
        trace = result['program_trace']['retrieval']
        self.assertEqual(trace['pool_selection'], 'verified_pool_fits_budget')
        self.assertEqual(trace['eligible_fact_count'],
                         trace['retained_typed_fact_count'])
        self.assertEqual(trace['dense'], 'skipped_complete_pool')
        self.assertEqual(trace['graph_edges'], 0)

    def test_over_budget_pool_uses_hybrid_retrieval(self):
        source = ('## Page 1\nBảng cân đối kế toán hợp nhất\nĐơn vị: đồng\n'
                  'Chỉ tiêu | TH 2031\n--- | ---\n' +
                  ''.join(f'Tài sản ngắn hạn | {100000000 + i}\n'
                          for i in range(12)))
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.md'
            path.write_text(source, encoding='utf8')
            bundle = SourceOnlyRanking()
            bundle.model = SimpleNamespace(config=SimpleNamespace(max_candidates=8))
            class FixedEncoder:
                def encode(self, texts, *, role):
                    import numpy as np
                    return np.ones((len(texts), 4), dtype=np.float32)
            bundle.semantic_encoder = FixedEncoder()
            result = ContractDocumentEngine(
                bundle, numeric_program_mode='baseline').answer(
                    'Tài sản ngắn hạn năm 2031 là bao nhiêu đồng?', path)
        trace = result['program_trace']['retrieval']
        self.assertEqual(trace['pool_selection'], 'hybrid_retrieval_budgeted')
        self.assertGreater(trace['eligible_fact_count'], 8)
        self.assertLessEqual(trace['retained_typed_fact_count'], 8)
        self.assertEqual(trace['mode'], 'hybrid_graph')
        self.assertEqual(trace['dense'], 'used_existing_encoder')
        self.assertIn('bm25_candidates', trace)

    def test_empty_retrieval_pool_halts_without_old_planner_fallback(self):
        source = ('## Page 1\nBảng cân đối kế toán hợp nhất\nĐơn vị: đồng\n'
                  'Chỉ tiêu | TH 2031\n--- | ---\n' +
                  ''.join(f'Tài sản ngắn hạn | {100000000 + i}\n'
                          for i in range(12)))
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.md'
            path.write_text(source, encoding='utf8')
            bundle = SourceOnlyRanking()
            bundle.model = SimpleNamespace(config=SimpleNamespace(max_candidates=8))
            engine = ContractDocumentEngine(bundle, numeric_program_mode='baseline')
            engine.retriever.retrieve = lambda *args, **kwargs: (
                [], {'mode': 'hybrid_graph', 'candidates': [],
                     'graph_added_selected': []})
            result = engine.answer(
                'Tài sản ngắn hạn năm 2031 là bao nhiêu đồng?', path)
        self.assertEqual(result['status'], 'ABSTAIN')
        self.assertEqual(result['program_selector'], 'baseline')
        self.assertGreater(result['program_trace']['candidate_available'], 8)
        self.assertIn('numeric_retrieval_pool_empty',
                      [check['code'] for check in result['validation']['checks']])

    def _answer_pair(self, source, question='Tài sản ngắn hạn năm 2031 là bao nhiêu đồng?'):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.md'
            path.write_text(source, encoding='utf8')
            return [ContractDocumentEngine(SourceOnlyRanking(), numeric_program_mode=mode)
                    .answer(question, path) for mode in ('off', 'baseline')]

    def test_distinct_closing_dates_abstain_with_specific_reason(self):
        source = ('## Page 1\nBảng cân đối kế toán hợp nhất\nĐơn vị: đồng\n'
                  'Chỉ tiêu | Số cuối kỳ 30/06/2031 | Số cuối kỳ 31/12/2031\n'
                  '--- | --- | ---\nTài sản ngắn hạn | 110.000.000 | 90.000.000\n')
        responses = self._answer_pair(source)
        self.assertEqual(responses[1]['program_trace']['mode'], 'baseline')
        for result in responses:
            self.assertEqual(result['status'], 'ABSTAIN')
            self.assertIn('source_period_ambiguous',
                          [check['code'] for check in result['validation']['checks']])

    def test_conflicting_same_date_disclosures_abstain(self):
        source = ('## Page 1\nBảng cân đối kế toán hợp nhất\nĐơn vị: đồng\n'
                  'Chỉ tiêu | Số cuối kỳ 31/12/2031\n--- | ---\n'
                  'Tài sản ngắn hạn | 110.000.000\n'
                  'Tài sản ngắn hạn | 90.000.000\n')
        responses = self._answer_pair(source)
        self.assertEqual(responses[1]['program_trace']['mode'], 'baseline')
        for result in responses:
            self.assertEqual(result['status'], 'ABSTAIN')
            self.assertIn('source_values_conflict',
                          [check['code'] for check in result['validation']['checks']])

    def test_explicit_date_resolves_two_closing_dates(self):
        source = ('## Page 1\nBảng cân đối kế toán hợp nhất\nĐơn vị: đồng\n'
                  'Chỉ tiêu | Số cuối kỳ 30/06/2031 | Số cuối kỳ 31/12/2031\n'
                  '--- | --- | ---\nTài sản ngắn hạn | 110.000.000 | 90.000.000\n')
        responses = self._answer_pair(
            source, 'Tài sản ngắn hạn ngày 30/06/2031 là bao nhiêu đồng?')
        self.assertEqual(responses[1]['program_trace']['mode'], 'baseline')
        self.assertEqual([r['canonical_value'] for r in responses],
                         ['110000000', '110000000'])

    def test_unrequested_scope_conflict_abstains(self):
        source = ('## Page 1\nBảng cân đối kế toán hợp nhất\nĐơn vị: đồng\n'
                  'Chỉ tiêu | Số cuối kỳ 31/12/2031\n--- | ---\n'
                  'Tài sản ngắn hạn | 110.000.000\n\n## Page 2\n'
                  'Bảng cân đối kế toán riêng\nĐơn vị: đồng\n'
                  'Chỉ tiêu | Số cuối kỳ 31/12/2031\n--- | ---\n'
                  'Tài sản ngắn hạn | 90.000.000\n')
        responses = self._answer_pair(source)
        self.assertEqual(responses[1]['program_trace']['mode'], 'baseline')
        for result in responses:
            self.assertEqual(result['status'], 'ABSTAIN')
            self.assertIn('source_scope_ambiguous',
                          [check['code'] for check in result['validation']['checks']])

    def test_unresolved_unit_keeps_raw_explanation(self):
        source = ('## Page 1\nBảng cân đối kế toán hợp nhất\n'
                  'Ngày 31/12/2031\nChỉ tiêu | Số cuối kỳ\n--- | ---\n'
                  'Tài sản ngắn hạn | 19765\n')
        responses = self._answer_pair(
            source, 'Tài sản ngắn hạn cuối năm 2031 là bao nhiêu đồng?')
        self.assertEqual(responses[1]['program_trace']['mode'], 'baseline')
        for result in responses:
            self.assertEqual(result['status'], 'ABSTAIN')
            self.assertEqual(result['answer_type'], 'explained_halt')
            self.assertIsNone(result['canonical_value'])
            self.assertTrue(result['raw_disclosures'])
            self.assertIn('source_unit_scale_missing',
                          [check['code'] for check in result['validation']['checks']])

    def test_requested_display_scale_matches_default_contract(self):
        source = ('## Page 1\nKết quả kinh doanh năm 2031\nĐơn vị: đồng\n'
                  'Chỉ tiêu | TH 2031\n--- | ---\nLợi nhuận sau thuế | 2500000\n')
        responses = self._answer_pair(
            source, 'Lợi nhuận sau thuế năm 2031 là bao nhiêu triệu đồng?')
        self.assertEqual(responses[1]['program_trace']['mode'], 'baseline')
        for result in responses:
            self.assertEqual(result['status'], 'ACCEPTED')
            self.assertEqual(result['canonical_value'], '2500000')
            self.assertEqual((result['display_value'], result['display_unit']),
                             ('2.5', 'triệu đồng'))
