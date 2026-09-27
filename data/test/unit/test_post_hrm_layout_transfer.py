"""Independent layout and semantic counterexamples for ALT fixes."""
import test_post_hrm_semantic_transfer as fixtures
from unittest import TestCase
from decimal import Decimal

class LayoutTransferTests(TestCase):
 setUp=fixtures.SemanticTransferTests.setUp
 answer=fixtures.SemanticTransferTests.answer
 source=fixtures.SemanticTransferTests.source
 def dated(self,card='',unit=''):
  return '## Page 1\nTình hình tài chính\n'+card+'\n'+unit+'\nChỉ tiêu | 01/01/2032 | 31/12/2032\n--- | --- | ---\nTổng tài sản | 400.000 | 350.000\nTài sản ngắn hạn | 200.000 | 150.000\n'
 def test_unknown_scale_money_is_blocked(self):
  r=self.answer(self.dated().replace('.', ''),'Tài sản ngắn hạn cuối năm 2032 là bao nhiêu đồng?')
  self.assertNotEqual(r['status'],'ACCEPTED');self.assertIn('đơn vị',r['rendered'])
 def test_unknown_shared_scale_cancels_in_same_row_percent_change(self):
  r=self.answer(self.dated().replace('.', ''),'Tài sản ngắn hạn cuối năm 2032 giảm bao nhiêu phần trăm so với đầu năm 2032?')
  self.assertEqual(r['status'],'ACCEPTED',r['rendered']);self.assertEqual(Decimal(r['canonical_value']),Decimal('-25'))
  self.assertEqual(len({c['line_number'] for c in r['citations']}),1)
  self.assertTrue(all(not c['unit_resolved'] and c['scale'] is None for c in r['citations']))
  self.assertTrue(any(c['code']=='same_row_scale_cancellation' and c['status']=='PASS' for c in r['validation']['checks']))
 def test_scale_is_not_inferred_from_numeric_agreement(self):
  for unit,factor in [('tỷ đồng',10**6),('triệu đồng',1000)]:
   with self.subTest(unit=unit):
    r=self.answer(self.dated('Tổng tài sản\n350,0 '+unit),'Tài sản ngắn hạn cuối năm 2032 là bao nhiêu đồng?')
    self.assertNotEqual(r['status'],'ACCEPTED');self.assertIn('đơn vị',r['rendered'])
 def test_ambiguous_scale_stays_blocked(self):
  r=self.answer(self.dated('Tổng tài sản\n350,0 tỷ đồng\nTổng tài sản\n350,0 triệu đồng'),'Tài sản ngắn hạn cuối năm 2032 là bao nhiêu đồng?')
  self.assertNotEqual(r['status'],'ACCEPTED')
 def test_actual_direction_overrides_word_decrease(self):
  for target,expected,word in [(900,300,'giảm'),(1500,-300,'tăng')]:
   r=self.answer(self.source(metric='Lợi nhuận trước thuế',columns='Thực hiện 2032 | Kế hoạch 2033',numbers=f'1200 | {target}'),'Lợi nhuận trước thuế kế hoạch năm 2033 giảm bao nhiêu đồng so với thực hiện năm 2032?')
   self.assertEqual(r['status'],'ACCEPTED',r['rendered']);self.assertEqual(Decimal(r['canonical_value']),expected);self.assertIn(word+' 300',r['rendered'])
 def test_footer_does_not_cross_page(self):
  s='## Page 1\nThông tin doanh nghiệp\nVốn điều lệ\n123.000.000 đồng\n## Page 2\nSố liệu tính đến 31/12/2032\n'
  r=self.answer(s,'Vốn điều lệ tại ngày 31/12/2032 là bao nhiêu đồng?');self.assertNotEqual(r['status'],'ACCEPTED')
 def test_june_not_december(self):
  s=self.dated(unit='Đơn vị: triệu đồng').replace('31/12/2032','30/06/2032')
  r=self.answer(s,'Tài sản ngắn hạn cuối năm 2032 là bao nhiêu đồng?');self.assertNotEqual(r['status'],'ACCEPTED')

 def test_wrong_program_order_is_rejected(self):
  from unittest.mock import patch
  from dataclasses import replace
  from hrm_finance_qa.question_planner import compile_question_plan
  def reversed_plan(q):
   p=compile_question_plan(q)
   if p.comparison_target and p.operation=='SUBTRACT':
    p=replace(p,program_dag=tuple(replace(n,inputs=tuple(reversed(n.inputs))) for n in p.program_dag))
   return p
  with patch('hrm_finance_qa.post_hrm_engine.compile_question_plan',side_effect=reversed_plan):
   r=self.answer(self.source(metric='Lợi nhuận trước thuế',columns='Thực hiện 2032 | Kế hoạch 2033',numbers='1200 | 900'),'Lợi nhuận trước thuế kế hoạch năm 2033 giảm bao nhiêu đồng so với thực hiện năm 2032?')
  self.assertNotEqual(r['status'],'ACCEPTED');self.assertIn('chiều phép tính',r['rendered'])
 def test_separate_share_categories(self):
  s='## Page 1\nThông tin cổ phiếu\n1.234.567\nCổ phiếu đang lưu hành\n12.345\nCP quỹ\nSố liệu tính đến 31/12/2032\n'
  for q,n in [('Số cổ phiếu đang lưu hành năm 2032 là bao nhiêu?',1234567),('Số cổ phiếu quỹ năm 2032 là bao nhiêu?',12345)]:
   r=self.answer(s,q);self.assertEqual(r['status'],'ACCEPTED',r['rendered']);self.assertEqual(Decimal(r['canonical_value']),n)
 def test_contact_label_split(self):
  s='## Page 1\nThông tin doanh nghiệp\nEmail / Website\ninfo@example.test · www.example.test\n'
  for q,v in [('Email của công ty là gì?','info@example.test'),('Website của công ty là gì?','www.example.test')]:
   r=self.answer(s,q);self.assertEqual(r['status'],'ACCEPTED',r['rendered']);self.assertIn(v,r['rendered'])
