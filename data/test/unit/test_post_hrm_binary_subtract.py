from unittest import TestCase
from unittest.mock import patch
from dataclasses import replace
from decimal import Decimal
import test_post_hrm_semantic_transfer as fixtures
from hrm_finance_qa.question_planner import compile_question_plan,resolve_public_question

class BinarySubtractTests(TestCase):
 setUp=fixtures.SemanticTransferTests.setUp
 answer=fixtures.SemanticTransferTests.answer
 def source(self,before=1500,after=900):
  return f'## Page 1\nKết quả kinh doanh\nĐơn vị: đồng\nChỉ tiêu | Thực hiện 2032 | Thực hiện 2031\n--- | --- | ---\nLợi nhuận trước thuế | {before} | 800\nLợi nhuận sau thuế | {after} | 500\n'
 def test_swapping_operands_swaps_sign(self):
  for left,right,expected in [('trước','sau',600),('sau','trước',-600)]:
   r=self.answer(self.source(),f'Lấy lợi nhuận {left} thuế trừ lợi nhuận {right} thuế năm 2032 được bao nhiêu đồng?')
   self.assertEqual(r['status'],'ACCEPTED',r['rendered']);self.assertEqual(Decimal(r['canonical_value']),expected)
   self.assertEqual(r['verified_program'][0][2],['subtract_left','subtract_right'])
 def test_explicit_year_per_operand(self):
  r=self.answer(self.source(),'Lợi nhuận trước thuế năm 2031 trừ lợi nhuận sau thuế năm 2032 là bao nhiêu đồng?');self.assertEqual(r['canonical_value'],'-100',r['rendered'])
 def test_changed_source_is_read_again(self):
  q='Lấy lợi nhuận trước thuế trừ lợi nhuận sau thuế năm 2032 được bao nhiêu đồng?'
  from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
  engine=ContractDocumentEngine(fixtures.FlatRanking())
  self.p.write_text(self.source(),encoding='utf8');a=engine.answer(q,self.p)
  self.p.write_text(self.source(2200,1300),encoding='utf8');b=engine.answer(q,self.p,previous_question=q)
  self.assertEqual(a['canonical_value'],'600');self.assertEqual(b['canonical_value'],'900')
 def test_missing_source_operand_does_not_compute(self):
  r=self.answer(self.source().replace('Lợi nhuận sau thuế | 900 | 500','Chi phí bán hàng | 900 | 500'),'Lợi nhuận trước thuế trừ lợi nhuận sau thuế năm 2032 là bao nhiêu đồng?')
  self.assertNotEqual(r['status'],'ACCEPTED');self.assertNotIn('operands',r['rendered'])
 def test_missing_question_operand_stops_before_retrieval(self):
  r=self.answer(self.source(),'Lấy lợi nhuận trước thuế năm 2032 trừ bao nhiêu đồng?')
  self.assertNotEqual(r['status'],'ACCEPTED');self.assertIn('hai đầu vào',r['rendered']);self.assertEqual(r['timings_seconds']['retrieval'],0)
 def test_missing_year_does_not_invent_year(self):
  r=self.answer(self.source(),'Lấy lợi nhuận trước thuế trừ lợi nhuận sau thuế được bao nhiêu đồng?');self.assertNotEqual(r['status'],'ACCEPTED');self.assertIn('năm',r['rendered'])
 def test_distinct_metrics_have_distinct_slots(self):
  p=compile_question_plan(resolve_public_question('Lợi nhuận trước thuế năm 2032 trừ lợi nhuận sau thuế năm 2032 là bao nhiêu đồng?'))
  self.assertEqual(len(p.operand_slots),2);self.assertNotEqual(p.operand_slots[0].metric,p.operand_slots[1].metric)
 def test_tampered_program_order_is_blocked(self):
  def bad(q):
   p=compile_question_plan(q)
   if tuple(s.id for s in p.operand_slots)==('subtract_left','subtract_right'):p=replace(p,program_dag=tuple(replace(n,inputs=tuple(reversed(n.inputs))) for n in p.program_dag))
   return p
  with patch('hrm_finance_qa.post_hrm_engine.compile_question_plan',side_effect=bad):
   r=self.answer(self.source(),'Lợi nhuận trước thuế trừ lợi nhuận sau thuế năm 2032 là bao nhiêu đồng?')
  self.assertNotEqual(r['status'],'ACCEPTED');self.assertIn('thứ tự',r['rendered'])

 def test_plan_and_actual_stay_with_their_operands(self):
  s=self.source()+'\nKế hoạch tài chính\nĐơn vị: đồng\nChỉ tiêu | Kế hoạch 2033\n--- | ---\nLợi nhuận sau thuế | 600\n'
  r=self.answer(s,'Lợi nhuận trước thuế thực hiện năm 2032 trừ lợi nhuận sau thuế kế hoạch năm 2033 là bao nhiêu đồng?');self.assertEqual(r['canonical_value'],'900',r['rendered'])
 def test_executor_error_is_not_public(self):
  with patch('hrm_finance_qa.post_hrm_engine.compute_records',side_effect=ValueError('SUBTRACT needs 2 operands, got 1')):
   r=self.answer(self.source(),'Lợi nhuận trước thuế trừ lợi nhuận sau thuế năm 2032 là bao nhiêu đồng?')
  self.assertNotEqual(r['status'],'ACCEPTED');self.assertNotIn('SUBTRACT',r['rendered']);self.assertNotIn('operands',r['rendered']);self.assertIn('Vui lòng',r['rendered'])

 def test_address_word_is_not_subtraction(self):
  r=self.answer('## Page 1\nThông tin doanh nghiệp\nĐịa chỉ: 25 Nguyễn Huệ, Thành phố Hồ Chí Minh\n','Địa chỉ trụ sở của công ty là gì?')
  self.assertNotIn('hai đầu vào',r['rendered']);self.assertEqual(r['status'],'ACCEPTED',r['rendered'])
 def test_deduction_noun_is_not_operator(self):
  from hrm_finance_qa.question_planner import _explicit_subtract_slots
  for q in ['cac khoan giam tru doanh thu nam 2032','thue duoc khau tru nam 2032','dia chi tru so']:
   self.assertIsNone(_explicit_subtract_slots(q,('2032',)))

 def test_elided_second_metric_keeps_temporal_subtraction(self):
  s='## Page 1\nKết quả kinh doanh hợp nhất\nĐơn vị: tỷ đồng\nChỉ tiêu | Năm 2032 | Năm 2031\n--- | --- | ---\nDoanh thu thuần | 1500 | 800\n'
  r=self.answer(s,'Lấy doanh thu thuần 2032 trừ 2031, đơn vị tỷ đồng')
  self.assertEqual(r['status'],'ACCEPTED',r['rendered']);self.assertEqual(r['canonical_value'],'700000000000')
 def test_elided_endpoint_keeps_existing_grammar(self):
  from hrm_finance_qa.question_planner import _explicit_subtract_slots
  self.assertIsNone(_explicit_subtract_slots('tai san ngan han cuoi nam 2032 tru dau nam',('2032',)))
 def test_same_metric_two_explicit_years_then_equals(self):
  s='## Page 1\nTình hình tài chính\nĐơn vị: đồng\nChỉ tiêu | Thực hiện 2032 | Thực hiện 2031\n--- | --- | ---\nVốn chủ sở hữu | -900 | -800\n'
  for question,value in [
   ('Lấy vốn chủ sở hữu năm 2032 trừ vốn chủ sở hữu năm 2031 thì bằng bao nhiêu đồng?','-100'),
   ('Lấy vốn chủ sở hữu năm 2031 trừ vốn chủ sở hữu năm 2032 thì bằng bao nhiêu đồng?','100'),
  ]:
   r=self.answer(s,question)
   self.assertEqual(r['canonical_value'],value,r['rendered'])
   self.assertEqual(r['verified_program'][0][2],['subtract_left','subtract_right'])
