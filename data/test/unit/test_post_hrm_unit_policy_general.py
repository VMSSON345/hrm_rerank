"""Unit provenance and metric-link evidence remain independent in fallback answers."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase

from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine


class FlatRanking:
    model = SimpleNamespace(config=SimpleNamespace(max_candidates=64))

    def rank_raw(self, question, candidates):
        return {'pointers': [[0.0] * len(candidates)] * 4,
                'select': [0.0] * len(candidates), 'pairs': {}, 'segments': 0}


class GeneralUnitFallbackTests(TestCase):
    def _answers(self, peer_label, *, peer_year='2032', question=None):
        question = question or 'Tài sản ngắn hạn sau điều chỉnh năm 2032 là bao nhiêu?'
        source = ('## Page 1\nBÁO CÁO THƯỜNG NIÊN NĂM 2032\n'
                  'I. Chỉ tiêu chính\nĐơn vị: đồng\n'
                  f'Chỉ tiêu | TH {peer_year}\n--- | ---\n{peer_label} | 1200\n'
                  '\n## Page 2\nII. Thuyết minh\nChỉ tiêu | TH 2032\n'
                  '--- | ---\nTài sản ngắn hạn sau điều chỉnh | 1200\n')
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.md'
            path.write_text(source, encoding='utf8')
            digest = sha256(path.read_bytes()).hexdigest()
            lines = path.read_text(encoding='utf8').splitlines()
            results = []
            for mode in ('off', 'baseline'):
                result = ContractDocumentEngine(
                    FlatRanking(), numeric_program_mode=mode).answer(question, path)
                for citation in result.get('citations') or []:
                    self.assertEqual(citation['document_sha256'], digest)
                    line = lines[citation['line_number'] - 1]
                    self.assertEqual(citation['source_line_sha256'],
                                     sha256(line.encode()).hexdigest())
                    start, end = citation['source_span']
                    self.assertIn(citation['raw_value'], line[start:end])
                results.append(result)
            return results

    def test_related_nonrevenue_label_is_disclosed_separately(self):
        for result in self._answers('Tài sản ngắn hạn'):
            self.assertEqual(result['status'], 'RAW_DISCLOSURE')
            self.assertEqual(result['unit_state'], 'UNIT_UNRESOLVED_RAW')
            self.assertEqual(result['metric_link_state'], 'NAME_COLLISION_AMBIGUOUS')
            self.assertIsNone(result['canonical_value'])
            self.assertEqual({item['metric_label'] for item in result['raw_disclosures']},
                             {'Tài sản ngắn hạn sau điều chỉnh', 'Tài sản ngắn hạn'})
            self.assertEqual({cite['unit_state'] for cite in result['citations']},
                             {'UNIT_UNRESOLVED_RAW', 'EXPLICIT_BOUND'})

    def test_explicit_money_request_halts_with_both_disclosures(self):
        for result in self._answers(
                'Tài sản ngắn hạn',
                question='Tài sản ngắn hạn sau điều chỉnh năm 2032 là bao nhiêu đồng?'):
            self.assertEqual(result['status'], 'ABSTAIN')
            self.assertEqual(result['answer_type'], 'explained_halt')
            self.assertEqual(result['metric_link_state'], 'NAME_COLLISION_AMBIGUOUS')
            self.assertIsNone(result['canonical_value'])
            self.assertEqual(len(result['raw_disclosures']), 2)

    def test_unrelated_equal_value_does_not_create_metric_link(self):
        for result in self._answers('Doanh thu thuần'):
            self.assertEqual(result['status'], 'RAW_DISCLOSURE')
            self.assertEqual(result['metric_link_state'], 'EXACT_METRIC')
            self.assertEqual(len(result['raw_disclosures']), 1)

    def test_other_year_does_not_lend_unit_or_metric_identity(self):
        for result in self._answers('Tài sản ngắn hạn', peer_year='2031'):
            self.assertEqual(result['status'], 'RAW_DISCLOSURE')
            self.assertEqual(result['metric_link_state'], 'EXACT_METRIC')
            self.assertEqual(len(result['raw_disclosures']), 1)
