import unittest,tempfile
from pathlib import Path
from hrm_finance_qa.document_sources import discover_documents,build_chunks,OCRConfig
from hrm_finance_qa.program_data import merge_candidates
from hrm_finance_qa.post_hrm_source import SourceSnapshot,build_source_facts
class SourcePeriodTests(unittest.TestCase):
 def facts(self,text):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'UNSEEN.md';p.write_text(text,encoding='utf8');ds=discover_documents(p);cs=merge_candidates('',build_chunks(ds,ocr_config=OCRConfig(enabled=False)),max_candidates=10000);return build_source_facts(cs,SourceSnapshot(ds))[0]
 def test_opening_closing_same_year_are_distinct(self):
  fs=self.facts('## Page 1\n2. Tình hình tài chính\nChỉ tiêu | Số dư đầu(01/01/2031) | Số dư cuối kỳ(31/12/2031)\n--- | --- | ---\nTài sản ngắn hạn | 110.000.000 | 90.000.000\n')
  self.assertEqual([(f.period,f.period_role,f.as_of_date) for f in fs],[('2031','opening','2031-01-01'),('2031','closing','2031-12-31')]);self.assertTrue(all(not f.comparison_role for f in fs))
 def test_prior_is_not_opening(self):
  fs=self.facts('## Page 1\n2. Tình hình tài chính\nChỉ tiêu | Cuối kỳ 30/06/2031 | Cuối kỳ 30/06/2030\n--- | --- | ---\nTài sản ngắn hạn | 110.000.000 | 90.000.000\n')
  self.assertTrue(fs);self.assertTrue(all(f.period_role=='closing' for f in fs));self.assertEqual([f.as_of_date for f in fs],['2031-06-30','2030-06-30'])
 def test_balance_title_year_does_not_invent_calendar_date(self):
  fs=self.facts('## Page 1\na. Bảng cân đối kế toán hợp nhất năm 2031.\nChỉ tiêu | Số cuối kỳ | Số đầu năm\n--- | --- | ---\nTài sản ngắn hạn | 110.000.000 | 90.000.000\n')
  self.assertEqual([f.period for f in fs],['2031','2031']);self.assertTrue(all(not f.as_of_date and not f.period_end for f in fs));self.assertEqual([f.period_role for f in fs],['closing','opening'])
 def test_adjacent_profile_report_year(self):
  fs=self.facts('## Page 1\nBÁO CÁO THƯỜNG NIÊN NĂM 2031\nI. Thông tin chung\nVốn điều lệ: 37.000.000 đồng\n')
  self.assertEqual(fs[0].period,'2031');self.assertEqual(fs[0].period_binding,'profile_report_title');self.assertFalse(fs[0].as_of_date)
 def test_new_heading_stops_balance_year(self):
  fs=self.facts('## Page 1\nBảng cân đối kế toán hợp nhất năm 2031\nKế hoạch năm tới\nChỉ tiêu | Số cuối kỳ | Số đầu năm\n--- | --- | ---\nTài sản ngắn hạn | 110.000.000 | 90.000.000\n')
  self.assertTrue(fs);self.assertTrue(all(not f.period for f in fs))
 def test_profile_without_title_does_not_get_year(self):
  fs=self.facts('## Page 1\nI. Thông tin chung\nVốn điều lệ: 37.000.000 đồng\n');self.assertEqual(fs[0].period,'')

 def test_numbered_continuation_keeps_own_title_year(self):
  text='## Page 1\nBảng cân đối kế toán hợp nhất năm 2031\nĐơn vị: đồng\nSTT | Chỉ tiêu | Số cuối kỳ | Số đầu năm\n--- | --- | --- | ---\n1 | Tài sản ngắn hạn | 110.000.000 | 90.000.000\n\n## Page 2\n2 | Tài sản ngắn hạn khác | 7.000.000 | 5.000.000\n3 | Hàng tồn kho | 9.000.000 | 6.000.000\n'
  fs=self.facts(text);other=[f for f in fs if 'khac' in f.metric]
  self.assertEqual(len(other),2);self.assertTrue(all(f.period=='2031' for f in other));self.assertEqual([f.period_role for f in other],['closing','opening']);self.assertTrue(all(any(w['page']==1 and '2031' in w['text'] for w in f.source_context) for f in other))
 def test_distinct_dates_not_one_conflict_key(self):
  from hrm_finance_qa.post_hrm_engine import _logical_key
  from test_post_hrm_contract import fact
  a=fact('total_assets',100,period='2031',period_role='closing',as_of_date='2031-06-30')
  b=fact('total_assets',80,period='2031',period_role='closing',as_of_date='2031-12-31')
  self.assertNotEqual(_logical_key(a),_logical_key(b))
  self.assertEqual(_logical_key(a),_logical_key(a.model_copy(update={'value':b.value})))
 def test_relative_role_not_relabelled_endpoint(self):
  fs=self.facts('## Page 1\n2. Tình hình tài chính\nChỉ tiêu | Năm 2031 | Năm 2030\n--- | --- | ---\nDoanh thu thuần | 110.000.000 | 90.000.000\n')
  self.assertTrue(all(f.period_role not in {'opening','closing'} for f in fs))

 def test_planner_distinguishes_endpoint_from_relative_column(self):
  from hrm_finance_qa.question_planner import compile_question_plan,resolve_public_question
  cases=[('Tài sản ngắn hạn cuối kỳ năm 2030 là bao nhiêu?', 'closing'),('Tài sản ngắn hạn đầu năm 2031 là bao nhiêu?', 'opening')]
  for q,role in cases:
   self.assertEqual(compile_question_plan(resolve_public_question(q)).operand_slots[0].period_role,role)

 def test_unknown_date_relative_roles_remain_separate(self):
  from hrm_finance_qa.post_hrm_engine import _logical_key
  from test_post_hrm_contract import fact
  a=fact('total_assets',100,period='',comparison_role='current')
  b=a.model_copy(update={'comparison_role':'prior'})
  self.assertNotEqual(_logical_key(a),_logical_key(b))
 def test_endpoint_render_preserves_date(self):
  from hrm_finance_qa.post_hrm_engine import _fact_time_text
  from test_post_hrm_contract import fact
  self.assertEqual(_fact_time_text(fact('total_assets',100,period='2031',period_role='opening',as_of_date='2031-07-01')),' đầu kỳ ngày 01/07/2031')

 def test_known_annual_period_not_split_by_table_relative_position(self):
  from hrm_finance_qa.post_hrm_engine import _logical_key
  from test_post_hrm_contract import fact
  a=fact('profit_after_tax',100,period='2031',comparison_role='current')
  self.assertEqual(_logical_key(a),_logical_key(a.model_copy(update={'comparison_role':'prior'})))
  self.assertNotEqual(_logical_key(a),_logical_key(a.model_copy(update={'period_basis':'quarter_1'})))

 def test_adjacent_dated_balance_title_beats_prior_year_narrative(self):
  text=('## Page 1\nBáo cáo tài chính năm 2034 đã được kiểm toán.\n'
        '## Page 2\nBẢNG CÂN ĐỐI KẾ TOÁN vào ngày 31 tháng 12 năm 2035\n'
        'VND\nMã số | Chỉ tiêu | Số cuối năm | Số đầu năm\n'
        '--- | --- | --- | ---\n'
        '270 | Tổng tài sản | 1.200.000 | 1.000.000\n'
        'BẢNG CÂN ĐỐI KẾ TOÁN (tiếp theo) vào ngày 31 tháng 12 năm 2035\n'
        'VND\nMã số | Nguồn vốn | Số cuối năm | Số đầu năm\n'
        '--- | --- | --- | ---\n'
        '300 | Nợ phải trả | 500.000 | 400.000\n')
  fs=self.facts(text)
  closing=[f for f in fs if f.metric in {'total_assets','total_liabilities'}
           and f.period_role=='closing']
  self.assertEqual(len(closing),2)
  self.assertTrue(all(f.period=='2035' and f.unit=='VND'
                      and f.unit_state=='EXPLICIT_BOUND' for f in closing))
  self.assertTrue(all(any('2035' in w['text'] and 'CÂN ĐỐI' in w['text']
                          for w in f.source_context) for f in closing))