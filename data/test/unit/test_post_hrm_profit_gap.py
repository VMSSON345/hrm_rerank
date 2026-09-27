"""The same-year profit gap is a verified absolute difference."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from hrm_finance_qa.exact_dag_executor import verify_dag_proof_certificate
from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine


class FlatRanking:
    model = SimpleNamespace(config=SimpleNamespace(max_candidates=64))

    def rank_raw(self, question, candidates):
        return {'pointers': [[0.0] * len(candidates)] * 4,
                'select': [0.0] * len(candidates), 'pairs': {}, 'segments': 0}


class ProfitGapTests(TestCase):
    def setUp(self):
        self.engine = ContractDocumentEngine(FlatRanking())
        self.alt = (Path(__file__).resolve().parents[3] /
                    'data/test/suites/alt_independent_v1/ALT_25CN_BCTN.md')

    def test_alt_a07_answers_from_two_explicit_vnd_cells(self):
        result = self.engine.answer(
            'Chênh lệch lợi nhuận trước thuế và lợi nhuận sau thuế năm 2025 là bao nhiêu đồng?',
            self.alt,
        )
        self.assertEqual(result['status'], 'ACCEPTED', result['rendered'])
        self.assertEqual(result['canonical_value'], '1747649993')
        self.assertIn('1.747.649.993 đồng', result['rendered'])
        self.assertEqual(len(result['citations']), 2)
        self.assertEqual({cite['page'] for cite in result['citations']}, {20})
        self.assertEqual({cite['unit_state'] for cite in result['citations']}, {'EXPLICIT_BOUND'})
        self.assertTrue(verify_dag_proof_certificate(result['proof']['execution_certificate']))

    def test_gap_is_absolute_but_explicit_subtraction_keeps_its_sign(self):
        gap = self.engine.answer(
            'Chênh lệch lợi nhuận sau thuế và lợi nhuận trước thuế năm 2025 là bao nhiêu đồng?',
            self.alt,
        )
        signed = self.engine.answer(
            'Lợi nhuận sau thuế năm 2025 trừ lợi nhuận trước thuế năm 2025 là bao nhiêu đồng?',
            self.alt,
        )
        self.assertEqual(gap['canonical_value'], '1747649993')
        self.assertEqual(signed['canonical_value'], '-1747649993')
        self.assertTrue(verify_dag_proof_certificate(gap['proof']['execution_certificate']))

    def test_missing_unit_still_halts(self):
        source = ('## Page 1\nChỉ tiêu | Thực hiện 2032\n--- | ---\n'
                  'Lợi nhuận trước thuế | 120\nLợi nhuận sau thuế | 90\n')
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'report.md'
            path.write_text(source, encoding='utf8')
            result = self.engine.answer(
                'Chênh lệch lợi nhuận trước thuế và lợi nhuận sau thuế năm 2032 là bao nhiêu đồng?',
                path,
            )
        self.assertEqual(result['status'], 'ABSTAIN')
        self.assertIsNone(result['canonical_value'])
        self.assertIn('`120`', result['rendered'])


if __name__ == '__main__':
    import unittest
    unittest.main()
