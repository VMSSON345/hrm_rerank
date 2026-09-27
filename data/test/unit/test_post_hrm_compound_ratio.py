"""Aligned OCR percentage facts and common-scale margin safeguards."""
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from hrm_finance_qa.document_sources import OCRConfig, build_chunks
from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
from hrm_finance_qa.post_hrm_source import SourceSnapshot, build_source_facts
from hrm_finance_qa.post_hrm_proposals import _valid_citations
from hrm_finance_qa.program_data import merge_candidates
from test_post_hrm_margin_program import ControlledRanking


QUESTION = 'Biên lợi nhuận ròng năm 2031 là bao nhiêu phần trăm?'
BASE = ('## Page 1\nBáo cáo kết quả hoạt động kinh doanh năm 2031\n'
        'Chỉ tiêu | Năm 2031\n--- | ---\n'
        'Doanh thu thuần | Năm 2031: 100000000\n'
        'Lợi nhuận sau thuế | Năm 2031: 25000000\n')
COMPOUND = ('4. Chỉ tiêu khả năng sinh lời+ '
            'Hệ số Lợi nhuận sau thuế/Doanh thu thuần+ '
            'Hệ số Lợi nhuận sau thuế/Vốn chủ sở hữu'
            ' | Năm 2030: 20%4% | Năm 2031: 25%5% |\n')


class CompoundRatioTests(TestCase):
    def _catalog(self, source):
        directory = TemporaryDirectory()
        path = Path(directory.name) / 'synthetic.md'
        path.write_text(source, encoding='utf8')
        snapshot = SourceSnapshot([path])
        candidates = merge_candidates('', build_chunks(
            [path], ocr_config=OCRConfig(enabled=False)),
            max_candidates=10000)
        facts, errors = build_source_facts(candidates, snapshot)
        return directory, path, snapshot, candidates, facts, errors

    def test_exact_alignment_yields_hashed_fact_spans(self):
        directory, path, snapshot, _, facts, _ = self._catalog(BASE + COMPOUND)
        try:
            recovered = [f for f in facts if f.metric == 'net_margin'
                         and f.period_binding == 'aligned_compound_ratio_column']
            self.assertEqual([(f.period, str(f.value)) for f in recovered],
                             [('2030', '20'), ('2031', '25')])
            for fact in recovered:
                citation = fact.citation()
                line = snapshot.lines[path.name][fact.line_number - 1].strip()
                start, end = fact.source_span
                self.assertEqual(line[start:end], fact.raw_value)
                self.assertEqual(citation['source_line_sha256'],
                                 sha256(line.encode()).hexdigest())
                self.assertEqual(citation['document_sha256'],
                                 sha256(path.read_bytes()).hexdigest())
            answer = ContractDocumentEngine(
                ControlledRanking(), numeric_program_mode='off').answer(
                    QUESTION, path)
            self.assertEqual(answer['status'], 'ACCEPTED')
            self.assertEqual(answer['canonical_value'], '25')
            self.assertEqual(answer['citations'][0]['source_span'],
                             list(recovered[1].source_span))
            self.assertEqual(len(answer['citations']), 3)
            self.assertTrue({'crosscheck_profit', 'crosscheck_revenue'} <=
                            {c['operand_ref'] for c in answer['citations']})
            self.assertTrue(_valid_citations(snapshot, answer['citations']))
        finally:
            directory.cleanup()

    def test_malformed_count_does_not_align_by_position(self):
        directory, _, _, _, facts, _ = self._catalog(
            BASE + COMPOUND.replace('25%5%', '25%'))
        try:
            self.assertFalse(any(f.metric == 'net_margin'
                                 and f.period == '2031'
                                 and f.period_binding == 'aligned_compound_ratio_column'
                                 for f in facts))
        finally:
            directory.cleanup()

    def test_ambiguous_percent_token_is_not_normalized(self):
        directory, _, _, _, facts, _ = self._catalog(
            BASE + COMPOUND.replace('25%5%', '1.000%5%'))
        try:
            self.assertFalse(any(f.metric == 'net_margin'
                                 and f.period == '2031'
                                 and f.period_binding == 'aligned_compound_ratio_column'
                                 for f in facts))
        finally:
            directory.cleanup()

    def test_reconstructed_ratio_requires_independent_operands(self):
        source = ('## Page 1\nBáo cáo kết quả hoạt động kinh doanh năm 2031\n'
                  + COMPOUND)
        directory, path, _, _, facts, _ = self._catalog(source)
        try:
            self.assertTrue(any(f.metric == 'net_margin'
                                and f.period_binding == 'aligned_compound_ratio_column'
                                for f in facts))
            for mode in ('off', 'baseline', 'hrm'):
                answer = ContractDocumentEngine(
                    ControlledRanking(), numeric_program_mode=mode).answer(
                        QUESTION, path)
                self.assertEqual(answer['status'], 'ABSTAIN', mode)
                self.assertIsNone(answer['canonical_value'])
        finally:
            directory.cleanup()

    def test_fuzzy_ocr_alias_needs_nearby_exact_revenue_row(self):
        typo = COMPOUND.replace('Doanh thu thuần', 'Doanh thu thuunan')
        directory, _, _, _, facts, _ = self._catalog(
            BASE.replace('Doanh thu thuần', 'Doanh thu hoạt động') + typo)
        try:
            self.assertFalse(any(f.metric == 'net_margin'
                                 and f.period_binding == 'aligned_compound_ratio_column'
                                 for f in facts))
        finally:
            directory.cleanup()
        directory, _, _, _, facts, _ = self._catalog(BASE + typo)
        try:
            recovered = next(f for f in facts if f.metric == 'net_margin'
                             and f.period == '2031'
                             and f.period_binding == 'aligned_compound_ratio_column')
            self.assertTrue(any(w.get('basis') == 'bounded_ocr_metric_alias'
                                and w.get('metric') == 'net_revenue'
                                and w.get('source_line_sha256')
                                == sha256(w['text'].encode()).hexdigest()
                                for w in recovered.source_context))
        finally:
            directory.cleanup()

    def test_disagreement_blocks_default_and_both_selectors(self):
        source = BASE + COMPOUND.replace('25%5%', '23%5%')
        directory, path, _, _, _, _ = self._catalog(source)
        try:
            for mode in ('off', 'baseline', 'hrm'):
                answer = ContractDocumentEngine(
                    ControlledRanking(), numeric_program_mode=mode).answer(
                        QUESTION, path)
                self.assertEqual(answer['status'], 'ABSTAIN', mode)
                self.assertIsNone(answer['canonical_value'])
        finally:
            directory.cleanup()

    def test_unknown_scale_never_becomes_absolute_money(self):
        directory, path, _, _, _, _ = self._catalog(BASE + COMPOUND)
        try:
            answer = ContractDocumentEngine(
                ControlledRanking(), numeric_program_mode='baseline').answer(
                    'Doanh thu thuần năm 2031 là bao nhiêu đồng?', path)
            self.assertEqual(answer['status'], 'ABSTAIN')
            self.assertIsNone(answer['canonical_value'])
        finally:
            directory.cleanup()
