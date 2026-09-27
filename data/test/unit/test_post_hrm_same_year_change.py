from unittest import TestCase
from pathlib import Path
from tempfile import TemporaryDirectory
from decimal import Decimal
from hrm_finance_qa.question_planner import compile_question_plan,resolve_public_question
from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
from test_post_hrm_semantic_transfer import FlatRanking

class SameYearChangeTests(TestCase):
 def test_original_p24_and_paraphrase(self):
  for q in ['Cuối năm 2023, tài sản ngắn hạn giảm bao nhiêu đồng so với đầu năm?','Tài sản ngắn hạn cuối năm 2023 giảm bao nhiêu đồng so với đầu năm 2023?']:
   p=compile_question_plan(resolve_public_question(q))
   self.assertEqual(p.operation,'SUBTRACT');self.assertEqual(len(p.operand_slots),2)
   self.assertEqual([(s.metric,s.year,s.period_role) for s in p.operand_slots],[('tai san ngan han','2023','opening'),('tai san ngan han','2023','closing')])
 def test_amount_direction_and_percentage_base_from_source(self):
  with TemporaryDirectory() as d:
   source=Path(d)/'TEST.md'
   text='## Page 1\nBảng cân đối kế toán\nĐơn vị: đồng\nChỉ tiêu | Số đầu năm 01/01/2031 | Số cuối năm 31/12/2031\n--- | --- | ---\nTài sản ngắn hạn | 120.000.000 | 90.000.000\n'
   source.write_text(text,encoding='utf8');engine=ContractDocumentEngine(FlatRanking())
   for wording,expected,unit in [('giảm bao nhiêu đồng','30000000','VND'),('tăng bao nhiêu đồng','-30000000','VND'),('giảm bao nhiêu phần trăm','-25','%'),('tăng bao nhiêu phần trăm','-25','%')]:
    b=engine.answer('Tài sản ngắn hạn cuối năm 2031 '+wording+' so với đầu năm 2031?',source)
    self.assertEqual(b['status'],'ACCEPTED',b['rendered']);self.assertEqual(Decimal(b['canonical_value']),Decimal(expected));self.assertEqual(b['unit'],unit)
    self.assertEqual({c['as_of_date'] for c in b['citations']},{'2031-01-01','2031-12-31'})
   source.write_text(text.replace('90.000.000','150.000.000'),encoding='utf8')
   b=engine.answer('Tài sản ngắn hạn cuối năm 2031 tăng bao nhiêu phần trăm so với đầu năm 2031?',source)
   self.assertEqual(b['canonical_value'],'25')
 def test_single_endpoint_not_duplicated(self):
  p=compile_question_plan(resolve_public_question('Tài sản ngắn hạn cuối năm 2031 là bao nhiêu?'))
  self.assertEqual(len(p.operand_slots),1)
 def test_missing_opening_never_subtracts_closing_from_itself(self):
  with TemporaryDirectory() as d:
   source=Path(d)/'TEST.md';source.write_text('## Page 1\nBảng cân đối kế toán\nĐơn vị: đồng\nChỉ tiêu | Số cuối năm 31/12/2031\n--- | ---\nTài sản ngắn hạn | 90.000.000\n',encoding='utf8')
   b=ContractDocumentEngine(FlatRanking()).answer('Tài sản ngắn hạn cuối năm 2031 giảm bao nhiêu đồng so với đầu năm 2031?',source)
   self.assertNotEqual(b['status'],'ACCEPTED')

 def test_cross_years_stay_attached_to_endpoints(self):
  for target,base in [('2024','2023'),('2023','2024')]:
   p=compile_question_plan(resolve_public_question(f'Tài sản ngắn hạn cuối năm {target} giảm bao nhiêu đồng so với đầu năm {base}?'))
   self.assertEqual([(s.year,s.period_role) for s in p.operand_slots],[(base,'opening'),(target,'closing')])
 def test_cross_year_source_computation_and_public_dates(self):
  with TemporaryDirectory() as d:
   source=Path(d)/'TEST.md';source.write_text('## Page 1\nBảng cân đối kế toán\nĐơn vị: đồng\nChỉ tiêu | Số đầu năm 01/01/2031 | Số cuối năm 31/12/2031 | Số cuối năm 31/12/2032\n--- | --- | --- | ---\nTài sản ngắn hạn | 120.000.000 | 90.000.000 | 100.000.000\n',encoding='utf8')
   b=ContractDocumentEngine(FlatRanking()).answer('Tài sản ngắn hạn cuối năm 2032 giảm bao nhiêu đồng so với đầu năm 2031?',source)
   self.assertEqual(b['canonical_value'],'20000000',b['rendered'])
   self.assertIn('01/01/2031',b['rendered']);self.assertIn('31/12/2032',b['rendered'])
 def test_same_endpoint_does_not_create_second_operand(self):
  p=compile_question_plan(resolve_public_question('Tài sản ngắn hạn cuối năm 2031 giảm bao nhiêu đồng so với cuối năm 2031?'))
  self.assertLess(len(p.operand_slots),2)
