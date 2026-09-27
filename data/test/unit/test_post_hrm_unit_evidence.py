from unittest import TestCase
from tempfile import TemporaryDirectory
from pathlib import Path
from decimal import Decimal
from hrm_finance_qa.document_sources import discover_documents,build_chunks,OCRConfig
from hrm_finance_qa.program_data import merge_candidates
from hrm_finance_qa.post_hrm_source import SourceSnapshot,build_source_facts

class UnitEvidenceTests(TestCase):
 def facts(self,body):
  with TemporaryDirectory() as d:
   p=Path(d)/'UNSEEN.md';p.write_text('## Page 1\nIII. Báo cáo và đánh giá của Ban Giám đốc\n2. Tình hình tài chính\n'+body,encoding='utf8');docs=discover_documents(p);cs=merge_candidates('',build_chunks(docs,ocr_config=OCRConfig(enabled=False)),max_candidates=10000)
   return build_source_facts(cs,SourceSnapshot(docs))
 def test_row_unit_precedes_table(self):
  fs,es=self.facts('Đơn vị: triệu đồng\nChỉ tiêu | Đơn vị tính | TH 2032 | TH 2031\n--- | --- | --- | ---\nDoanh thu thuần | tỷ đồng | 12 | 10\n')
  self.assertEqual([f.canonical_value for f in fs],[Decimal('12000000000'),Decimal('10000000000')]);self.assertTrue(all(not f.unit_assumption for f in fs));self.assertTrue(all(any(w['basis']=='row_or_cell' and 'tỷ đồng' in w['text'] for w in f.unit_evidence) for f in fs))
 def test_column_unit_precedes_table(self):
  fs,es=self.facts('Đơn vị: triệu đồng\nChỉ tiêu | TH 2032 (tỷ đồng) | TH 2031 (tỷ đồng)\n--- | --- | ---\nDoanh thu thuần | 12 | 10\n');self.assertEqual([f.unit for f in fs],['tỷ đồng','tỷ đồng']);self.assertTrue(all(any(w['basis']=='column_header' for w in f.unit_evidence) for f in fs))
 def test_each_column_has_own_scale(self):
  fs,es=self.facts('Chỉ tiêu | TH 2032 (tỷ đồng) | TH 2031 (triệu đồng)\n--- | --- | ---\nDoanh thu thuần | 12 | 10000\n');self.assertEqual([f.canonical_value for f in fs],[Decimal('12000000000'),Decimal('10000000000')])
 def test_currency_only_column_keeps_explicit_table_scale(self):
  fs,es=self.facts('Đơn vị: triệu đồng\nChỉ tiêu | TH 2032 (VND) | TH 2031 (VND)\n--- | --- | ---\nDoanh thu thuần | 12 | 10\n');self.assertEqual([f.unit for f in fs],['triệu đồng','triệu đồng'])
 def test_row_column_scale_conflict_cannot_default(self):
  fs,es=self.facts('Chỉ tiêu | Đơn vị tính | TH 2032 (tỷ đồng) | TH 2031 (tỷ đồng)\n--- | --- | --- | ---\nDoanh thu thuần | triệu đồng | 12 | 10\n');self.assertFalse(fs);self.assertTrue(any(e['reason']=='source_specific_unit_conflict' for e in es))
 def test_inline_unit_is_most_specific(self):
  fs,es=self.facts('Đơn vị: triệu đồng\nChỉ tiêu | TH 2032 | TH 2031\n--- | --- | ---\nDoanh thu thuần | 12 tỷ đồng | 10 tỷ đồng\n');self.assertEqual([f.unit for f in fs],['tỷ đồng','tỷ đồng']);self.assertTrue(all(any(w['basis']=='row_or_cell' for w in f.unit_evidence) for f in fs))
 def test_glued_vnd_date_header_has_physical_unit_witness(self):
  fs,es=self.facts('Chỉ tiêu | 31/12/2032VND | 31/12/2031VND\n--- | --- | ---\nDoanh thu thuần | 12 | 10\n')
  self.assertEqual(len(fs),2)
  self.assertTrue(all(f.unit=='VND' and f.unit_state=='EXPLICIT_BOUND' for f in fs))
  self.assertTrue(all(any(w['basis']=='column_header' and 'VND' in w['text'] for w in f.unit_evidence) for f in fs))
 def test_missing_unit_remains_unresolved(self):
  fs,es=self.facts('Chỉ tiêu | TH 2032 | TH 2031\n--- | --- | ---\nDoanh thu thuần | 12 | 10\n');self.assertTrue(fs);self.assertTrue(all(not f.unit and not f.unit_resolved and f.canonical_value is None and f.citation()['scale'] is None and not f.citation()['unit_evidence'] for f in fs))
 def test_percent_not_scaled_as_money(self):
  fs,es=self.facts('Đơn vị: triệu đồng\nChỉ tiêu | TH 2032 | TH 2031\n--- | --- | ---\nBiên lợi nhuận | 12% | 10%\n');self.assertTrue(fs);self.assertTrue(all(f.unit=='%' and not f.unit_assumption for f in fs))
 def test_foreign_row_not_overwritten_by_vnd_table(self):
  fs,es=self.facts('Đơn vị: triệu đồng\nChỉ tiêu | Đơn vị tính | TH 2032 | TH 2031\n--- | --- | --- | ---\nDoanh thu thuần | USD | 12 | 10\n');self.assertTrue(fs);self.assertTrue(all(f.currency=='USD' for f in fs))



 def test_standalone_vnd_immediately_above_table_header(self):
  fs,es=self.facts('VND\nChỉ tiêu | TH 2032 | TH 2031\n--- | --- | ---\nDoanh thu thuần | 12.000.000 | 10.000.000\n')
  self.assertEqual(len(fs),2)
  self.assertTrue(all(f.unit=='VND' and f.unit_state=='EXPLICIT_BOUND' for f in fs))
  self.assertTrue(all(any(w['basis']=='table_declaration' and w['text'].strip()=='VND'
                          for w in f.unit_evidence) for f in fs))
 def test_standalone_vnd_does_not_cross_page(self):
  fs,es=self.facts('VND\n## Page 2\nChỉ tiêu | TH 2032 | TH 2031\n--- | --- | ---\nDoanh thu thuần | 12.000.000 | 10.000.000\n')
  self.assertEqual(len(fs),2)
  self.assertTrue(all(f.unit_state=='UNIT_UNRESOLVED_RAW' for f in fs))
 def test_standalone_vnd_separated_by_prose_not_bound(self):
  fs,es=self.facts('VND\nBảng khác đã kết thúc.\nChỉ tiêu | TH 2032 | TH 2031\n--- | --- | ---\nDoanh thu thuần | 12.000.000 | 10.000.000\n')
  self.assertEqual(len(fs),2)
  self.assertTrue(all(f.unit_state=='UNIT_UNRESOLVED_RAW' for f in fs))