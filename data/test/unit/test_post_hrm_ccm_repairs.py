from unittest import TestCase
from tempfile import TemporaryDirectory
from pathlib import Path
from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
from hrm_finance_qa.question_planner import compile_question_plan,resolve_public_question
from test_post_hrm_semantic_transfer import FlatRanking

class CCMRepairTests(TestCase):
 def setUp(self):
  self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)/'SOURCE.md';self.engine=ContractDocumentEngine(FlatRanking())
 def ask(self,text,q):
  self.p.write_text(text,encoding='utf8');return self.engine.answer(q,self.p)
 def test_scope_unknown_different_values_requires_clarification(self):
  text='## Page 1\nb. Kết quả sản xuất kinh doanh hợp nhất năm 2030.\nĐơn vị: đồng\nChỉ tiêu | Năm 2030 | Năm 2029\n--- | --- | ---\nLợi nhuận sau thuế | 120.000.000 | 100.000.000\n## Page 2\n2. Báo cáo kết quả hoạt động sản xuất kinh doanh.\nĐơn vị: đồng\nChỉ tiêu | Năm 2030 | Năm 2029\n--- | --- | ---\nLợi nhuận sau thuế | 120.000.000 | 90.000.000\n'
  b=self.ask(text,'Lợi nhuận sau thuế năm 2029 là bao nhiêu?');self.assertEqual(b['status'],'ABSTAIN');self.assertIn('phạm vi',b['rendered'])
  b=self.ask(text,'Lợi nhuận sau thuế hợp nhất năm 2029 là bao nhiêu?');self.assertEqual(b['canonical_value'],'100000000')
  b=self.ask(text.replace('90.000.000','100.000.000'),'Lợi nhuận sau thuế năm 2029 là bao nhiêu?');self.assertEqual(b['status'],'ACCEPTED',b['rendered'])
 def balance(self,tail):
  return '## Page 1\na. Bảng cân đối kế toán hợp nhất năm 2030.\nĐơn vị: đồng\nChỉ tiêu | Mã chỉ tiêu | Thuyết minh | Số cuối kỳ | Số đầu năm\n--- | --- | --- | --- | ---\nTài sản ngắn hạn | 100 | | 90.000.000 | 120.000.000\n'+tail
 def test_coded_continuation_keeps_year_and_column(self):
  text=self.balance('## Page 2\nTỔNG CỘNG TÀI SẢN | 270 | | 150.000.000 | 200.000.000\n')
  for when,value in [('cuối','150000000'),('đầu','200000000')]:
   b=self.ask(text,f'Tổng tài sản hợp nhất {when} năm 2030 là bao nhiêu?');self.assertEqual(b['canonical_value'],value,b['rendered']);self.assertEqual(b['citations'][0]['page'],2)
 def test_new_balance_title_does_not_inherit_old_year(self):
  text=self.balance('## Page 2\nBảng cân đối kế toán riêng lẻ năm 2031.\nTổng tài sản | 270 | | 150.000.000 | 200.000.000\n')
  b=self.ask(text,'Tổng tài sản hợp nhất cuối năm 2030 là bao nhiêu?');self.assertNotEqual(b['status'],'ACCEPTED')
 def test_unmatched_width_and_large_page_gap_not_carried(self):
  for tail in ['## Page 2\nTổng tài sản | 270 | 150.000.000 | 200.000.000\n','## Page 5\nTổng tài sản | 270 | | 150.000.000 | 200.000.000\n']:
   b=self.ask(self.balance(tail),'Tổng tài sản hợp nhất cuối năm 2030 là bao nhiêu?');self.assertNotEqual(b['status'],'ACCEPTED')
 def test_plan_sentence_value_unit_and_actual_guard(self):
  text='## Page 1\n- Kế hoạch lợi nhuận trước thuế của Toàn công ty năm 2031 : 70 tỷ đồng.\n'
  b=self.ask(text,'Kế hoạch lợi nhuận trước thuế năm 2031 của toàn công ty là bao nhiêu?');self.assertEqual(b['canonical_value'],'70000000000',b['rendered'])
  b=self.ask(text,'Lợi nhuận trước thuế thực hiện năm 2031 là bao nhiêu?');self.assertNotEqual(b['status'],'ACCEPTED')
 def test_heading_year_is_not_operand_year(self):
  q='Theo bảng kết quả sản xuất kinh doanh hợp nhất năm 2031, lợi nhuận sau thuế năm trước là bao nhiêu?'
  plan=compile_question_plan(resolve_public_question(q));self.assertEqual(plan.requested_years,('2030',));self.assertEqual(plan.metric_text,'loi nhuan sau thue');self.assertIn('2031',plan.requested_source_heading);self.assertEqual(plan.requested_accounting_scope,'consolidated')
 def test_employee_date_is_checked(self):
  text='## Page 1\n- Tổng số nhân viên trong Công ty là 376 nhân viên tính đến thời điểm 31/12/2030.\n'
  b=self.ask(text,'Công ty có bao nhiêu nhân viên tại ngày 31/12/2030?');self.assertEqual(b['status'],'ACCEPTED');self.assertIn('376 nhân viên',b['rendered'])
  b=self.ask(text,'Công ty có bao nhiêu nhân viên tại ngày 30/12/2030?');self.assertEqual(b['status'],'ABSTAIN')
 def test_salary_preserves_compound_unit_and_year(self):
  text='## Page 1\n- Tiền lương bình quân trong năm 2030 : 10.000.000đ/người/tháng.\n'
  b=self.ask(text,'Tiền lương bình quân năm 2030 là bao nhiêu đồng mỗi người mỗi tháng?');self.assertEqual(b['status'],'ACCEPTED');self.assertIn('10.000.000đ/người/tháng',b['rendered'])
  b=self.ask(text,'Tiền lương bình quân năm 2031 là bao nhiêu đồng mỗi người mỗi tháng?');self.assertEqual(b['status'],'ABSTAIN')
 def test_hours_does_not_confuse_daily_with_weekly(self):
  q='Nhân viên làm việc bao nhiêu giờ mỗi tuần?'
  b=self.ask('## Page 1\nCông ty thực hiện chế độ làm việc 44 giờ/tuần và 08 giờ/ngày.\n',q);self.assertEqual(b['status'],'ACCEPTED');self.assertIn('44 giờ/tuần',b['rendered'])
  b=self.ask('## Page 1\nCông ty thực hiện chế độ làm việc 08 giờ/ngày.\n',q);self.assertEqual(b['status'],'ABSTAIN')
 def test_bare_followup_year_asks_meaning_before_retrieval(self):
  b=self.ask('## Page 1\n','Còn năm 2030?');self.assertEqual(b['status'],'ABSTAIN');self.assertIn('xem riêng',b['rendered']);self.assertEqual(b['timings_seconds']['retrieval'],0);self.assertEqual(b['timings_seconds']['hrm'],0)
