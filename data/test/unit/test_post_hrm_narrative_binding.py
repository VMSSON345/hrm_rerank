from unittest import TestCase
from decimal import Decimal
import test_post_hrm_semantic_transfer as fixtures
import test_post_hrm_source_period as source_fixtures

class NarrativeBindingTests(TestCase):
 setUp=fixtures.SemanticTransferTests.setUp
 answer=fixtures.SemanticTransferTests.answer
 facts=source_fixtures.SourcePeriodTests.facts
 def test_inline_year_value_growth_and_other_metric(self):
  s='## Page 1\nNăm 2032, Công ty ghi nhận doanh thu thuần 731,4 tỷ đồng, tăng 8,2%; lợi nhuận sau thuế đạt 29,1 tỷ đồng.\n'
  r=self.answer(s,'Doanh thu thuần năm 2032 là bao nhiêu?');self.assertEqual(r['canonical_value'],'731400000000',r['rendered'])
 def test_adjacent_heading_and_split_card(self):
  for heading in ['Phân tích tài chính năm 2032','Điểm nhấn tài chính 2032','Kết quả kinh doanh năm 2032']:
   s='## Page 1\n'+heading+'\nDOANH THU THUẦN\n731,4 tỷ đồng\n'
   r=self.answer(s,'Doanh thu thuần năm 2032 là bao nhiêu?');self.assertEqual(r['canonical_value'],'731400000000',r['rendered'])
 def test_report_title_not_data_year(self):
  r=self.answer('## Page 1\nBáo cáo thường niên 2032\nDOANH THU THUẦN\n731,4 tỷ đồng\n','Doanh thu thuần năm 2032 là bao nhiêu?');self.assertNotEqual(r['status'],'ACCEPTED')
 def test_new_section_or_page_stops_heading(self):
  for boundary in ['Kế hoạch phát triển','## Page 2']:
   r=self.answer('## Page 1\nPhân tích tài chính năm 2032\n'+boundary+'\nDOANH THU THUẦN\n731,4 tỷ đồng\n','Doanh thu thuần năm 2032 là bao nhiêu?');self.assertNotEqual(r['status'],'ACCEPTED')
 def test_explicit_year_overrides_heading(self):
  s='## Page 1\nPhân tích tài chính năm 2032\nNăm 2031, doanh thu thuần đạt 412,5 tỷ đồng.\n'
  r=self.answer(s,'Doanh thu thuần năm 2032 là bao nhiêu?');self.assertNotEqual(r['status'],'ACCEPTED')
  r=self.answer(s,'Doanh thu thuần năm 2031 là bao nhiêu?');self.assertEqual(r['canonical_value'],'412500000000')
 def test_plan_not_actual(self):
  s='## Page 1\nNăm 2033, doanh thu thuần kế hoạch đạt 850,5 tỷ đồng.\n'
  r=self.answer(s,'Doanh thu thuần thực hiện năm 2033 là bao nhiêu?');self.assertNotEqual(r['status'],'ACCEPTED')
  r=self.answer(s,'Doanh thu thuần kế hoạch năm 2033 là bao nhiêu?');self.assertEqual(r['canonical_value'],'850500000000',r['rendered'])
 def test_explicit_scope(self):
  s='## Page 1\nNăm 2032, doanh thu thuần hợp nhất đạt 731,4 tỷ đồng.\nNăm 2032, doanh thu thuần riêng lẻ đạt 512,8 tỷ đồng.\n'
  r=self.answer(s,'Doanh thu thuần hợp nhất năm 2032 là bao nhiêu?');self.assertEqual(r['canonical_value'],'731400000000',r['rendered'])
  r=self.answer(s,'Doanh thu thuần năm 2032 là bao nhiêu?');self.assertNotEqual(r['status'],'ACCEPTED');self.assertIn('731.400.000.000',r['rendered']);self.assertIn('512.800.000.000',r['rendered'])
 def test_coarse_card_cannot_hide_precise_scope_difference(self):
  s='## Page 1\nĐiểm nhấn tài chính 2032\nDOANH THU THUẦN\n731 tỷ đồng\nNăm 2032, doanh thu thuần hợp nhất đạt 731,2 tỷ đồng.\nNăm 2032, doanh thu thuần đạt 731,3 tỷ đồng.\n'
  r=self.answer(s,'Doanh thu thuần năm 2032 là bao nhiêu?');self.assertNotEqual(r['status'],'ACCEPTED');self.assertIn('731.200.000.000',r['rendered']);self.assertIn('731.300.000.000',r['rendered'])
 def test_same_scope_conflicting_values(self):
  s='## Page 1\nNăm 2032, doanh thu thuần đạt 731,2 tỷ đồng.\nNăm 2032, doanh thu thuần đạt 731,3 tỷ đồng.\n'
  r=self.answer(s,'Doanh thu thuần năm 2032 là bao nhiêu?');self.assertNotEqual(r['status'],'ACCEPTED');self.assertIn('chưa thống nhất',r['rendered'])
 def test_local_review_topic_can_reference_prior_year(self):
  s='## Page 1\nNhìn lại hoạt động năm 2032, công ty phục hồi sau một năm 2031 khó khăn.\nVề kết quả kinh doanh, doanh thu thuần hợp nhất đạt 731,2 tỷ đồng.\n'
  r=self.answer(s,'Doanh thu thuần hợp nhất năm 2032 là bao nhiêu?');self.assertEqual(r['canonical_value'],'731200000000',r['rendered'])
 def test_quarter_not_annual(self):
  r=self.answer('## Page 1\nQuý 2 năm 2032, doanh thu thuần đạt 731,2 tỷ đồng.\n','Doanh thu thuần năm 2032 là bao nhiêu?');self.assertNotEqual(r['status'],'ACCEPTED')
 def test_unrelated_metric_not_inherited(self):
  r=self.answer('## Page 1\nNăm 2032, tổng doanh thu đạt 731,2 tỷ đồng.\n','Doanh thu thuần năm 2032 là bao nhiêu?');self.assertNotEqual(r['status'],'ACCEPTED')
 def test_missing_year_not_taken_from_question(self):
  r=self.answer('## Page 1\nDoanh thu thuần hợp nhất đạt 731,2 tỷ đồng.\n','Doanh thu thuần năm 2032 là bao nhiêu?');self.assertNotEqual(r['status'],'ACCEPTED')

 def test_precise_asof_date_precedes_heading_year(self):
  s='## Page 1\nPhân tích tài chính năm 2032\nVốn điều lệ\n731.200.000 đồng\nSố liệu tính đến 31/12/2032\n'
  r=self.answer(s,'Vốn điều lệ tại ngày 31/12/2032 là bao nhiêu?');self.assertEqual(r['canonical_value'],'731200000',r['rendered']);self.assertIn('31/12/2032',r['rendered'])
