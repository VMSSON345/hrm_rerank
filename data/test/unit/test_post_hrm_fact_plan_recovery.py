from unittest import TestCase
from pathlib import Path
from tempfile import TemporaryDirectory
from decimal import Decimal
from hrm_finance_qa.question_planner import compile_question_plan,resolve_public_question
from hrm_finance_qa.program_data import _is_table_header_or_separator,_iter_number_matches,_is_table_metadata_cell_code
from hrm_finance_qa.document_sources import discover_documents,build_chunks,OCRConfig
from hrm_finance_qa.program_data import merge_candidates
from hrm_finance_qa.post_hrm_source import SourceSnapshot,build_source_facts
from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
from test_post_hrm_semantic_transfer import FlatRanking

SOURCE="""## Page 1
BÁO CÁO THƯỜNG NIÊN 2032
III. Báo cáo và đánh giá của Ban Giám đốc
2. Tình hình tài chính
Đơn vị: triệu đồng
Mục | Nội dung | Năm 2032 | Năm 2031 | Chênh lệch | Tỷ lệ (%)
--- | --- | --- | --- | --- | ---
1 | Tổng tài sản | 9.481 | 8.710 | 771 | 109%
1.1 | Tài sản ngắn hạn | 7.391 | 6.810 | 581 | 109%
Scanned with CamScanner
## Page 2
 | Tài sản ngắn hạn khác | 2.345 | 1.574 | 771 | 149%
--- | --- | --- | --- | --- | ---
1.2 | Tài sản dài hạn | 2.090 | 1.900 | 190 | 110%
 | Nợ dài hạn | 12,25 | 12,25 | 0 |
"""

class FactPlanRecoveryTests(TestCase):
 def setUp(self):
  self.t=TemporaryDirectory();self.addCleanup(self.t.cleanup);self.p=Path(self.t.name)/'NEW_REPORT.md'
 def facts(self,text):
  self.p.write_text(text,encoding='utf8');docs=discover_documents(self.p);cs=merge_candidates('',build_chunks(docs,ocr_config=OCRConfig(enabled=False)),max_candidates=10000)
  fs,err=build_source_facts(cs,SourceSnapshot(docs));return cs,fs,err
 def test_endpoint_default_is_idempotent_and_preserves_slots(self):
  for phrase,role in [('cuối năm','closing'),('đầu năm','opening'),('cuối kỳ','closing'),('đầu kỳ','opening')]:
   q=f'Tài sản ngắn hạn khác hợp nhất {phrase} 2032 là bao nhiêu đồng?';resolved=resolve_public_question(q);p=compile_question_plan(resolved)
   self.assertEqual(resolve_public_question(resolved),resolved);self.assertEqual(p.metric_text,'tai san ngan han khac');self.assertEqual(p.requested_period_roles,(role,));self.assertEqual(p.requested_accounting_scope,'consolidated');self.assertEqual(p.requested_years,('2032',))
 def test_explicit_plan_and_total_other_remain_distinct(self):
  for metric in ['Tài sản ngắn hạn','Tài sản ngắn hạn khác']:
   p=compile_question_plan(resolve_public_question(metric+' kế hoạch cuối năm 2032 là bao nhiêu?'))
   self.assertEqual(p.column_roles,('PLAN',));self.assertEqual(p.requested_period_roles,('closing',));self.assertEqual(p.metric_text,__import__('hrm_finance_qa.ontology',fromlist=['normalize_text']).normalize_text(metric))
 def test_value_row_before_repeated_header_kept(self):
  rows=['Mã số: 150 | V. Tài sản ngắn hạn khác | | Số cuối năm: 346.820 | Số đầu năm: 26.481','Mã số | TÀI SẢN | Thuyết minh | Số cuối năm | Số đầu năm','--- | --- | --- | --- | ---']
  self.assertFalse(_is_table_header_or_separator(rows,0));self.assertTrue(_is_table_header_or_separator(rows,1))
 def test_numeric_band_and_year_headers_stay_headers(self):
  for row in ['Tài sản | Dưới 1.000 | Trên 2.000','Chỉ tiêu | Năm 2032 | Năm 2031']:
   self.assertTrue(_is_table_header_or_separator([row,'--- | --- | ---'],0))
 def test_grouped_amount_not_note_but_explicit_note_is(self):
  for row,expected in [('Tài sản ngắn hạn khác | 2.345 | 11.543 | (10.323)',False),('Thuyết minh | 2.345 | 11.543 | (10.323)',True)]:
   m=next(x for x in _iter_number_matches(row) if '2.345' in x[0]);self.assertEqual(_is_table_metadata_cell_code(row,m,Decimal(2345)),expected)
 def test_continuation_binds_year_unit_and_value(self):
  cs,fs,err=self.facts(SOURCE);f=next(f for f in fs if f.raw_value=='2.345');self.assertEqual((f.period,f.unit,f.canonical_value),('2032','triệu đồng',Decimal(2345000000)))
  first=next(f for f in fs if f.raw_value=='7.391');self.assertEqual(first.canonical_value,Decimal(7391000000))
 def test_continuation_does_not_cross_new_section(self):
  cs,fs,err=self.facts(SOURCE.replace('## Page 2','## Page 2\n3. Kế hoạch kinh doanh'))
  self.assertFalse(any(c.table_semantics.get('numbered_continuation') for c in cs if c.raw.strip()=='2.345'))
 def test_continuation_does_not_cross_ordinal_gap(self):
  cs,fs,err=self.facts(SOURCE.replace('1.2 |','1.4 |'))
  self.assertFalse(any((c.table_semantics or {}).get('numbered_continuation') for c in cs if c.raw.strip()=='2.345'))
 def test_continuation_does_not_cross_unit_change(self):
  cs,fs,err=self.facts(SOURCE.replace('## Page 2','## Page 2\nĐơn vị: tỷ đồng'))
  self.assertFalse(any((c.table_semantics or {}).get('numbered_continuation') for c in cs if c.raw.strip()=='2.345'))
 def test_continuation_does_not_override_new_header(self):
  cs,fs,err=self.facts(SOURCE.replace(' | Tài sản ngắn hạn khác','Mục | Nội dung | Năm 2035 | Năm 2034 | Chênh lệch | Tỷ lệ (%)\n--- | --- | --- | --- | --- | ---\n | Tài sản ngắn hạn khác'))
  self.assertFalse(any((c.table_semantics or {}).get('numbered_continuation') for c in cs if c.raw.strip()=='2.345'))
 def test_full_answer_uses_source_not_requested_year(self):
  self.p.write_text(SOURCE,encoding='utf8');e=ContractDocumentEngine(FlatRanking())
  r=e.answer('Tài sản ngắn hạn khác năm 2032 là bao nhiêu đồng?',self.p);self.assertEqual(r['canonical_value'],'2345000000')
  r=e.answer('Tài sản ngắn hạn khác năm 2039 là bao nhiêu đồng?',self.p);self.assertNotEqual(r['status'],'ACCEPTED')

 def balance_source(self, date='vào ngày 31 tháng 12 năm 2032', middle=''):
  return '## Page 1\nBảng cân đối kế toán hợp nhất\n'+date+'\n'+middle+'Đơn vị: đồng\nMã số | TÀI SẢN | Thuyết minh | Số cuối năm | Số đầu năm\n--- | --- | --- | --- | ---\nMã số: 150 | V. Tài sản ngắn hạn khác | | Số cuối năm: 346.820.500 | Số đầu năm: 126.810.200\n'
 def test_balance_title_date_closing_opening_and_scope(self):
  self.p.write_text(self.balance_source(),encoding='utf8');e=ContractDocumentEngine(FlatRanking())
  for endpoint,value in [('cuối','346820500'),('đầu','126810200')]:
   r=e.answer(f'Tài sản ngắn hạn khác hợp nhất {endpoint} năm 2032 là bao nhiêu?',self.p);self.assertEqual(r['canonical_value'],value)
  for q in ['Tài sản ngắn hạn khác hợp nhất cuối năm 2039 là bao nhiêu?','Tài sản ngắn hạn khác riêng lẻ cuối năm 2032 là bao nhiêu?']:
   self.assertNotEqual(e.answer(q,self.p)['status'],'ACCEPTED')
 def test_balance_date_cannot_be_signing_or_unrelated_caption(self):
  for text in [self.balance_source(date='Ngày ký 31 tháng 12 năm 2032'),self.balance_source(middle='Kế hoạch kinh doanh\n'),self.balance_source(date='vào ngày 30 tháng 6 năm 2032')]:
   cs,fs,err=self.facts(text);self.assertFalse(any(f.period=='2032' and f.scope=='consolidated' for f in fs))
 def test_section_prefix_preserves_metric_qualifiers(self):
  from hrm_finance_qa.metric_semantics import identity
  from hrm_finance_qa.unit_defaults import monetary_label
  self.assertEqual(identity('V. Tài sản ngắn hạn khác'),identity('Tài sản ngắn hạn khác'))
  self.assertNotEqual(identity('V. Tài sản ngắn hạn khác'),identity('Tài sản ngắn hạn'))
  self.assertTrue(monetary_label('V. Tài sản ngắn hạn khác'))
  self.assertFalse(monetary_label('V. Số cổ phiếu'))

 def test_initial_capital_is_not_opening_balance(self):
  for phrase in ['ban đầu','ban dau']:
   p=compile_question_plan(resolve_public_question('Vốn điều lệ '+phrase+' năm 2006 là bao nhiêu đồng?'))
   self.assertEqual(p.metric_text,'von dieu le');self.assertEqual(p.requested_event,'initial');self.assertEqual(p.requested_years,('2006',))
