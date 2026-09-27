from unittest import TestCase
import test_post_hrm_semantic_transfer as fixtures
class SourcePeriodBindingTests(TestCase):
 setUp=fixtures.SemanticTransferTests.setUp
 answer=fixtures.SemanticTransferTests.answer
 def test_date_headers_keep_code_and_note_columns(self):
  s='## Page 1\nBALANCE SHEET\nASSETS | Code | Note | 31/12/2032VND | 01/01/2032VND\n--- | --- | --- | --- | ---\nCURRENT ASSETS | 100 | 4 | 12500 | 8300\n'
  for endpoint,value in [('cuối','12500'),('đầu','8300')]:
   r=self.answer(s,f'Tài sản ngắn hạn {endpoint} năm 2032 là bao nhiêu đồng?');self.assertEqual(r['canonical_value'],value,r['rendered'])
 def test_year_alone_is_not_closing(self):
  s='## Page 1\nTình hình tài chính\nĐơn vị: đồng\nChỉ tiêu | Năm 2032 | Năm 2031\n--- | --- | ---\nTài sản ngắn hạn | 12500 | 8300\n'
  self.assertNotEqual(self.answer(s,'Tài sản ngắn hạn cuối năm 2032 là bao nhiêu đồng?')['status'],'ACCEPTED')
  self.assertEqual(self.answer(s,'Tài sản ngắn hạn năm 2032 là bao nhiêu đồng?')['canonical_value'],'12500')
 def test_prior_closing_is_not_next_opening(self):
  s='## Page 1\nBALANCE SHEET\nASSETS | Code | Note | 31/12/2032VND | 31/12/2031VND\n--- | --- | --- | --- | ---\nCURRENT ASSETS | 100 | 4 | 12500 | 8300\n'
  self.assertNotEqual(self.answer(s,'Tài sản ngắn hạn đầu năm 2032 là bao nhiêu đồng?')['status'],'ACCEPTED')
 def test_profile_local_title_not_remote_report_year(self):
  s='## Page 1\nBÁO CÁO THƯỜNG NIÊN NĂM 2032\nI. Thông tin chung\n1. Thông tin khái quát\nVốn điều lệ: 810000 đồng\n## Page 2\nBÁO CÁO THƯỜNG NIÊN NĂM 2031\n'
  self.assertEqual(self.answer(s,'Vốn điều lệ năm 2032 là bao nhiêu đồng?')['canonical_value'],'810000')
 def test_profile_change_is_not_report_year(self):
  s='## Page 1\nBÁO CÁO THƯỜNG NIÊN NĂM 2032\nI. Thông tin chung\n1. Thông tin khái quát\nKế hoạch tăng vốn năm 2033\nVốn điều lệ: 810000 đồng\n'
  self.assertNotEqual(self.answer(s,'Vốn điều lệ năm 2032 là bao nhiêu đồng?')['status'],'ACCEPTED')
 def test_flat_profile_unique_binding(self):
  s='## Page 1\nGENERAL INFORMATION\nCompany Name\nCharter Capital\nAddress\nExample Company\n810000 VND\nSome Street\nCompany History\nANNUAL REPORT 2032\n'
  self.assertEqual(self.answer(s,'Vốn điều lệ năm 2032 là bao nhiêu đồng?')['canonical_value'],'810000')
 def test_flat_profile_two_amounts_not_guessed(self):
  s='## Page 1\nGENERAL INFORMATION\nCompany Name\nCharter Capital\nAddress\nExample Company\n810000 VND\n820000 VND\nSome Street\nCompany History\nANNUAL REPORT 2032\n'
  self.assertNotEqual(self.answer(s,'Vốn điều lệ năm 2032 là bao nhiêu đồng?')['status'],'ACCEPTED')
 def test_adjacent_financial_heading_binds_actual_without_closing_date(self):
  s='## Page 1\nII. Báo cáo và đánh giá của Ban Giám đốc\nTrong năm 2032, kế hoạch đầu tư được thảo luận.\n2/ Tình hình tài chính:\na/ Tình hình tài sản:\nChỉ tiêu | Đvt | Năm 2032 | Năm 2031\n--- | --- | --- | ---\nTài sản ngắn hạn | Tỷ đồng | 12 | 10\n'
  r=self.answer(s,'Tài sản ngắn hạn năm 2032 là bao nhiêu đồng?')
  self.assertEqual(r['canonical_value'],'12000000000',r['rendered'])
  self.assertTrue(any('Tình hình tài sản' in w['text'] for c in r['citations'] for w in c['source_context']))
  self.assertNotEqual(self.answer(s,'Tài sản ngắn hạn cuối năm 2032 là bao nhiêu đồng?')['status'],'ACCEPTED')
 def test_adjacent_plan_heading_does_not_bind_actual(self):
  s='## Page 1\nII. Báo cáo và đánh giá của Ban Giám đốc\n2/ Kế hoạch tài chính:\nChỉ tiêu | Đvt | Năm 2032\n--- | --- | ---\nTài sản ngắn hạn | Tỷ đồng | 12\n'
  self.assertNotEqual(self.answer(s,'Tài sản ngắn hạn thực hiện năm 2032 là bao nhiêu đồng?')['status'],'ACCEPTED')
 def test_audited_result_column_overrides_remote_plan_heading(self):
  s=('## Page 1\n1. Kế hoạch kinh doanh năm 2032\n'
     '## Page 2\nDVT:VND\n(Nguồn: Báo cáo Tài chính hợp nhất đã kiểm toán năm 2032)\n'
     'Chỉ tiêu | Năm 2031 | Kết quả năm 2032\n--- | --- | ---\n'
     'Doanh thu thuần | 1000 | 1200\n')
  actual=self.answer(s,'Doanh thu thuần thực hiện năm 2032 là bao nhiêu đồng?')
  self.assertEqual(actual['canonical_value'],'1200',actual['rendered'])
  self.assertTrue(any('Kết quả năm 2032' in w['text'] for c in actual['citations'] for w in c['source_context']))
  previous=self.answer(s,'Doanh thu thuần thực hiện năm 2031 là bao nhiêu đồng?')
  self.assertEqual(previous['canonical_value'],'1000',previous['rendered'])
  plan=self.answer(s,'Doanh thu thuần kế hoạch năm 2032 là bao nhiêu đồng?')
  self.assertNotEqual(plan['status'],'ACCEPTED')
 def test_approved_target_column_paired_with_actual_is_plan(self):
  s=('## Page 1\nĐánh giá kết quả kinh doanh năm 2032\n'
     'Chỉ tiêu | Đơn vị tính | Chỉ tiêu được thông qua 2032 | TH 2032\n'
     '--- | --- | --- | ---\nLợi nhuận trước thuế | Đồng | 800 | 900\n'
     'Lợi nhuận sau thuế | Đồng | 600 | 700\n')
  actual=self.answer(s,'Lợi nhuận trước thuế thực hiện năm 2032 là bao nhiêu đồng?')
  self.assertEqual(actual['canonical_value'],'900',actual['rendered'])
  plan=self.answer(s,'Lợi nhuận trước thuế kế hoạch năm 2032 là bao nhiêu đồng?')
  self.assertEqual(plan['canonical_value'],'800',plan['rendered'])
  reverse=self.answer(s,'Lấy lợi nhuận sau thuế thực hiện năm 2032 trừ lợi nhuận trước thuế thực hiện năm 2032 thì bằng bao nhiêu đồng?')
  self.assertEqual(reverse['canonical_value'],'-200',reverse['rendered'])
