"""Source-bound margin operands in the experimental numeric program path."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
from hrm_finance_qa.post_hrm_formulas import margin_operand_plan
from hrm_finance_qa.program_data import OPERATIONS


class ControlledRanking:
    model = SimpleNamespace(config=SimpleNamespace(max_candidates=64))

    def rank_raw(self, question, candidates):
        count = len(candidates)
        operation_scores = [0.0] * len(OPERATIONS)
        operation_scores[OPERATIONS.index('RATIO_PERCENT')] = 2.0
        matrix = [[0.0] * count for _ in range(count)]
        profit = [i for i, c in enumerate(candidates)
                  if 'profit_after_tax' in c.context]
        revenue = [i for i, c in enumerate(candidates)
                   if 'net_revenue' in c.context]
        for left in profit:
            for right in revenue:
                matrix[left][right] = 5.0
        return {'operation_scores': operation_scores, 'select': [0.0] * count,
                'pointers': [[0.0] * count] * 4,
                'pairs': {'RATIO_PERCENT': matrix}, 'segments': 1}


class MarginProgramTests(TestCase):
    question = 'Biên lợi nhuận ròng năm 2031 là bao nhiêu phần trăm?'

    def _answers(self, source):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.md'
            path.write_text(source, encoding='utf8')
            digest = sha256(path.read_bytes()).hexdigest()
            lines = source.splitlines()
            answers = []
            for mode in ('baseline', 'hrm'):
                answer = ContractDocumentEngine(
                    ControlledRanking(), numeric_program_mode=mode).answer(
                        self.question, path)
                for citation in answer.get('citations') or []:
                    self.assertEqual(citation['document_sha256'], digest)
                    line = lines[citation['line_number'] - 1]
                    self.assertEqual(citation['source_line_sha256'],
                                     sha256(line.encode()).hexdigest())
                    start, end = citation['source_span']
                    self.assertIn(citation['raw_value'], line[start:end])
                answers.append(answer)
            return answers

    def test_named_margin_creates_slots_without_planner_operation_override(self):
        plan = margin_operand_plan(self.question)
        self.assertEqual(plan.operation, 'SELECT')
        self.assertEqual([slot.metric for slot in plan.operand_slots],
                         ['Lợi nhuận sau thuế', 'Doanh thu thuần'])
        self.assertIsNone(margin_operand_plan(
            'Biên lợi nhuận năm 2031 là bao nhiêu phần trăm?'))

    def test_both_selectors_compute_from_sourced_operands(self):
        source = ('## Page 1\nBáo cáo kết quả hoạt động kinh doanh năm 2031\n'
                  'Đơn vị: đồng\nChỉ tiêu | TH 2031\n--- | ---\n'
                  'Doanh thu thuần | 100000000\n'
                  'Lợi nhuận sau thuế | 25000000\n')
        for answer in self._answers(source):
            self.assertEqual(answer['status'], 'ACCEPTED')
            self.assertEqual(answer['canonical_value'], '25')
            self.assertEqual(answer['program']['operation'], 'RATIO_PERCENT')
            self.assertEqual(len(answer['citations']), 2)
            self.assertIn('Doanh thu thuần', answer['rendered'])
            self.assertEqual(answer['program_trace']['choice_class'],
                             'single_valid_program')

    def test_printed_margin_stays_a_selectable_fact(self):
        source = ('## Page 1\nBáo cáo kết quả hoạt động kinh doanh năm 2031\n'
                  'Chỉ tiêu | TH 2031\n--- | ---\n'
                  'Biên lợi nhuận ròng | 25%\n')
        for answer in self._answers(source):
            self.assertEqual(answer['status'], 'ACCEPTED')
            self.assertEqual(answer['canonical_value'], '25')
            self.assertEqual(answer['program']['operation'], 'SELECT')
            self.assertEqual(len(answer['citations']), 1)

    def test_disagreeing_printed_margin_blocks_both_programs(self):
        source = ('## Page 1\nBáo cáo kết quả hoạt động kinh doanh năm 2031\n'
                  'Đơn vị: đồng\nChỉ tiêu | TH 2031\n--- | ---\n'
                  'Doanh thu thuần | 100000000\n'
                  'Lợi nhuận sau thuế | 25000000\n'
                  'Biên lợi nhuận ròng | 23%\n')
        for answer in self._answers(source):
            self.assertEqual(answer['status'], 'ABSTAIN')
            self.assertIsNone(answer['canonical_value'])
            self.assertIn('calculation_disagrees_with_printed_ratio',
                          {item['reason'] for item in
                           answer['program_trace']['proposals']})

    def test_different_scopes_do_not_supply_a_pair(self):
        source = ('## Page 1\nBáo cáo kết quả hoạt động kinh doanh hợp nhất năm 2031\n'
                  'Đơn vị: đồng\nChỉ tiêu | TH 2031\n--- | ---\n'
                  'Doanh thu thuần | 100000000\n\n## Page 2\n'
                  'Báo cáo kết quả hoạt động kinh doanh riêng năm 2031\n'
                  'Đơn vị: đồng\nChỉ tiêu | TH 2031\n--- | ---\n'
                  'Lợi nhuận sau thuế | 25000000\n')
        for answer in self._answers(source):
            self.assertEqual(answer['status'], 'ABSTAIN')
            self.assertIsNone(answer['canonical_value'])
            self.assertIn('operand_identity_or_scope_mismatch',
                          {item['reason'] for item in
                           answer['program_trace']['proposals']})

    def test_same_table_unknown_scale_cancels_only_for_percent(self):
        source = ('## Page 1\nBáo cáo kết quả hoạt động kinh doanh năm 2031\n'
                  'Chỉ tiêu | TH 2031\n--- | ---\n'
                  'Doanh thu thuần | 100000000\n'
                  'Lợi nhuận sau thuế | 25000000\n')
        for answer in self._answers(source):
            self.assertEqual(answer['status'], 'ACCEPTED')
            self.assertEqual(answer['canonical_value'], '25')
            self.assertEqual({c['unit_state'] for c in answer['citations']},
                             {'UNIT_UNRESOLVED_RAW'})
            self.assertIn('same_table_common_scale',
                          {check['code'] for check in
                           answer['validation']['checks']})

    def test_different_table_unknown_scales_do_not_cancel(self):
        source = ('## Page 1\nBáo cáo kết quả hoạt động kinh doanh năm 2031\n'
                  'Chỉ tiêu | TH 2031\n--- | ---\n'
                  'Doanh thu thuần | 100000000\n\n## Page 2\n'
                  'Chỉ tiêu | TH 2031\n--- | ---\n'
                  'Lợi nhuận sau thuế | 25000000\n')
        for answer in self._answers(source):
            self.assertEqual(answer['status'], 'ABSTAIN')
            self.assertIsNone(answer['canonical_value'])
            self.assertIn('source_unit_scale_missing',
                          {item['reason'] for item in
                           answer['program_trace']['proposals']})

    def test_missing_denominator_never_becomes_a_margin(self):
        source = ('## Page 1\nBáo cáo kết quả hoạt động kinh doanh năm 2031\n'
                  'Đơn vị: đồng\nChỉ tiêu | TH 2031\n--- | ---\n'
                  'Lợi nhuận sau thuế | 25000000\n')
        for answer in self._answers(source):
            self.assertEqual(answer['status'], 'ABSTAIN')
            self.assertIsNone(answer['canonical_value'])
            self.assertEqual(answer['program_trace']['choice_class'],
                             'no_valid_program')
