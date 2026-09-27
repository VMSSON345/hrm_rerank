"""A shared table does not by itself prove a shared money scale."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
from test_post_hrm_margin_program import ControlledRanking


class CommonScaleAdversarialTests(TestCase):
    question = 'Biên lợi nhuận ròng năm 2031 là bao nhiêu phần trăm?'

    def _results(self, revenue, profit):
        source = ('## Page 1\nBáo cáo kết quả hoạt động kinh doanh năm 2031\n'
                  'Chỉ tiêu | TH 2031\n--- | ---\n'
                  + revenue + '\n' + profit + '\n')
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.md'
            path.write_text(source, encoding='utf8')
            return [ContractDocumentEngine(
                ControlledRanking(), numeric_program_mode=mode).answer(
                    self.question, path) for mode in ('baseline', 'hrm')]

    def test_row_specific_scale_breaks_cancellation(self):
        for answer in self._results(
                'Doanh thu thuần | 100000000',
                'Lợi nhuận sau thuế | 25 triệu đồng'):
            self.assertEqual(answer['status'], 'ABSTAIN')
            self.assertIsNone(answer['canonical_value'])
            self.assertIn('source_unit_scale_missing',
                          {item['reason'] for item in
                           answer['program_trace']['proposals']})

    def test_parallel_currencies_need_fx_input(self):
        for answer in self._results(
                'Doanh thu thuần | 100000000 VND',
                'Lợi nhuận sau thuế | 25000000 USD'):
            self.assertEqual(answer['status'], 'ABSTAIN')
            self.assertIsNone(answer['canonical_value'])
            self.assertTrue(any(
                'Currency conversion requires an explicit dated FX input'
                in item['reason'] for item in
                answer['program_trace']['proposals']))

    def test_different_year_columns_cannot_be_paired(self):
        for answer in self._results(
                'Doanh thu thuần | Năm 2030: 100000000',
                'Lợi nhuận sau thuế | Năm 2031: 25000000'):
            self.assertEqual(answer['status'], 'ABSTAIN')
            self.assertIsNone(answer['canonical_value'])
            self.assertEqual(answer['program_trace']['choice_class'],
                             'no_valid_program')

    def test_ambiguous_separator_does_not_create_a_ratio(self):
        for answer in self._results(
                'Doanh thu thuần | 1.000',
                'Lợi nhuận sau thuế | 250'):
            self.assertEqual(answer['status'], 'ABSTAIN')
            self.assertIsNone(answer['canonical_value'])
