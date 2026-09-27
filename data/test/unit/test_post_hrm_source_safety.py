import unittest
from types import SimpleNamespace
import test_post_hrm_source_period as source_period_tests
from test_post_hrm_contract import fact
from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine, _logical_key

class SourceSafetyTests(unittest.TestCase):
 facts=source_period_tests.SourcePeriodTests.facts
 def test_profile_plan_heading_does_not_inherit_report_year(self):
  fs=self.facts('## Page 1\nBÁO CÁO THƯỜNG NIÊN NĂM 2031\nI. Thông tin chung\nKế hoạch năm 2032\nVốn điều lệ: 37.000.000 đồng\n')
  self.assertTrue(fs)
  self.assertFalse(any(f.period=='2031' for f in fs))
 def test_profile_effective_date_after_value_does_not_inherit_report_year(self):
  fs=self.facts('## Page 1\nBÁO CÁO THƯỜNG NIÊN NĂM 2031\nI. Thông tin chung\nVốn điều lệ: 37.000.000 đồng (thay đổi từ ngày 15/02/2032)\n')
  self.assertTrue(fs)
  self.assertFalse(any(f.period=='2031' for f in fs))
 def test_profile_next_line_effective_date(self):
  fs=self.facts('## Page 1\nBÁO CÁO THƯỜNG NIÊN NĂM 2031\nI. Thông tin chung\nVốn điều lệ: 37.000.000 đồng\nCó hiệu lực từ ngày 15/02/2032\n')
  self.assertTrue(fs)
  self.assertFalse(any(f.period=='2031' for f in fs))
 def test_title_and_column_same_date_have_same_key(self):
  title=self.facts('## Page 1\nBảng cân đối kế toán hợp nhất\nNgày 31/12/2031\nĐơn vị: đồng\nChỉ tiêu | Số cuối kỳ | Số đầu năm\n--- | --- | ---\nTài sản ngắn hạn | 110.000.000 | 90.000.000\n')
  column=self.facts('## Page 1\nBảng cân đối kế toán hợp nhất\nĐơn vị: đồng\nChỉ tiêu | Số cuối kỳ 31/12/2031 | Số đầu năm 01/01/2031\n--- | --- | ---\nTài sản ngắn hạn | 120.000.000 | 90.000.000\n')
  self.assertEqual(_logical_key(title[0]),_logical_key(column[0]))
 def answer(self, question, fs, known=None):
  engine=object.__new__(ContractDocumentEngine)
  engine.bundle=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(max_candidates=100)))
  return engine._scalar(question,[],[],fs,known_facts=known or fs)[0]
 def dates(self):
  return [fact('literal:tai san ngan han',110000000,metric_label='Tài sản ngắn hạn',period='2031',period_role='closing',as_of_date='2031-06-30',fact_id='june'),fact('literal:tai san ngan han',90000000,metric_label='Tài sản ngắn hạn',period='2031',period_role='closing',as_of_date='2031-12-31',fact_id='dec')]
 def test_two_closing_dates_require_clarification_even_if_only_one_retrieved(self):
  fs=self.dates()
  for pool in [fs,fs[::-1],fs[:1],fs[1:]]:
   b=self.answer('Tài sản ngắn hạn năm 2031 là bao nhiêu đồng?',fs,pool)
   self.assertEqual(b['status'],'ABSTAIN')
   self.assertIn('source_period_ambiguous',[c['code'] for c in b['validation']['checks']])
 def test_explicit_date_selects_its_own_value(self):
  b=self.answer('Tài sản ngắn hạn ngày 30/06/2031 là bao nhiêu đồng?',self.dates())
  self.assertEqual(b['status'],'ACCEPTED')
  self.assertEqual(b['canonical_value'],'110000000')
 def test_same_date_conflict_detected_across_metadata_forms(self):
  fs=self.dates()[:1]
  fs.append(fs[0].model_copy(update={'fact_id':'other','value':fs[0].value+1,'as_of_date':'','period_end':'2031-06-30'}))
  b=self.answer('Tài sản ngắn hạn năm 2031 là bao nhiêu đồng?',fs)
  self.assertEqual(b['status'],'ABSTAIN')
  self.assertIn('source_values_conflict',[c['code'] for c in b['validation']['checks']])

 def test_midyear_title_preserves_june_date(self):
  fs=self.facts('## Page 1\nBảng cân đối kế toán hợp nhất\nNgày 30/06/2031\nĐơn vị: đồng\nChỉ tiêu | Số cuối kỳ | Số đầu năm\n--- | --- | ---\nTài sản ngắn hạn | 110.000.000 | 90.000.000\n')
  self.assertEqual(fs[0].as_of_date,'2031-06-30')
  self.assertEqual(fs[0].period_end,'2031-06-30')
  self.assertEqual(fs[1].as_of_date,'2031-01-01')

 def test_yearless_lookup_does_not_publish_future_or_planned_profile(self):
  for tail in ['Kế hoạch năm 2032\nVốn điều lệ: 37.000.000 đồng\n','Vốn điều lệ: 37.000.000 đồng (thay đổi từ ngày 15/02/2032)\n']:
   fs=self.facts('## Page 1\nBÁO CÁO THƯỜNG NIÊN NĂM 2031\nI. Thông tin chung\n'+tail)
   self.assertTrue(fs)
   b=self.answer('Vốn điều lệ của công ty là bao nhiêu?',fs)
   self.assertEqual(b['status'],'ABSTAIN')

 def test_registration_date_is_not_capital_effective_date(self):
  fs=self.facts('## Page 1\nBÁO CÁO THƯỜNG NIÊN NĂM 2031\nI. Thông tin chung\nGiấy phép hoạt động kinh doanh chứng khoán cấp ngày 26/12/2006\nVốn điều lệ: 37.000.000 đồng\n')
  capital=next(f for f in fs if f.metric=='charter_capital')
  self.assertEqual(capital.period,'2031')
  self.assertTrue(capital.is_profile)
  self.assertEqual(self.answer('Vốn điều lệ của công ty là bao nhiêu?',fs)['canonical_value'],'37000000')

 def test_compatible_year_only_disclosure_is_not_another_day(self):
  fs=self.dates()[:1]
  fs.append(fs[0].model_copy(update={'fact_id':'annual','as_of_date':''}))
  self.assertEqual(self.answer('Tài sản ngắn hạn năm 2031 là bao nhiêu đồng?',fs)['canonical_value'],'110000000')
 def test_incompatible_year_only_disclosure_remains_unresolved(self):
  fs=self.dates()[:1]
  fs.append(fs[0].model_copy(update={'fact_id':'annual','as_of_date':'','value':fs[0].value+100}))
  self.assertEqual(self.answer('Tài sản ngắn hạn năm 2031 là bao nhiêu đồng?',fs)['status'],'ABSTAIN')
 def test_equal_amounts_on_distinct_dates_still_require_clarification(self):
  fs=self.dates();fs[1]=fs[1].model_copy(update={'value':fs[0].value})
  b=self.answer('Tài sản ngắn hạn năm 2031 là bao nhiêu đồng?',fs)
  self.assertIn('source_period_ambiguous',[c['code'] for c in b['validation']['checks']])
 def test_digits_inside_amount_are_not_a_context_year(self):
  fs=self.facts('## Page 1\nBÁO CÁO THƯỜNG NIÊN NĂM 2031\nI. Thông tin chung\nVốn điều lệ: 12000000 đồng\n')
  capital=next(f for f in fs if f.metric=='charter_capital')
  self.assertEqual(capital.period,'2031')
