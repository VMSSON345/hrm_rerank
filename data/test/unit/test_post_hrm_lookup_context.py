import unittest,tempfile
from pathlib import Path
from hrm_finance_qa.document_sources import discover_documents,build_chunks,OCRConfig
from hrm_finance_qa.program_data import merge_candidates,_best_row_label
from hrm_finance_qa.post_hrm_source import SourceSnapshot,build_source_facts
from hrm_finance_qa.question_planner import compile_question_plan,resolve_public_question
class LookupContextTests(unittest.TestCase):
 def facts(self,body):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'UNSEEN.md';p.write_text('## Page 1\nI. Thông tin chung\n'+body,encoding='utf8');docs=discover_documents(p);cs=merge_candidates('',build_chunks(docs,ocr_config=OCRConfig(enabled=False)),max_candidates=10000);return build_source_facts(cs,SourceSnapshot(docs))
 def test_key_value_amount_in_words_not_label(self):
  self.assertEqual(_best_row_label(['Vốn điều lệ:','37.500.000.000 đồng (Ba mươi bảy tỷ năm trăm triệu đồng)'],1),'Vốn điều lệ:')
 def test_key_value_without_colon(self):
  self.assertEqual(_best_row_label(['Tài sản ngắn hạn khác','47.000 đồng (Bốn mươi bảy nghìn đồng)'],1),'Tài sản ngắn hạn khác')
 def test_profile_fact_from_two_cells(self):
  fs,es=self.facts('Tên công ty: | CÔNG TY THỬ\n--- | ---\nVốn điều lệ: | 37.500.000.000 VNĐ (Ba mươi bảy tỷ năm trăm triệu đồng)\n');f=next(f for f in fs if f.metric=='charter_capital');self.assertTrue(f.is_profile);self.assertEqual(f.role,'profile');self.assertEqual(str(f.canonical_value),'37500000000')
 def test_yearless_default_actual(self):
  p=compile_question_plan(resolve_public_question('Vốn điều lệ của công ty là bao nhiêu?'));self.assertEqual(p.operand_slots[0].column_role,'ACTUAL');self.assertEqual(p.operand_slots[0].year,'')
 def test_explicit_plan_remains_plan(self):
  p=compile_question_plan(resolve_public_question('Vốn điều lệ kế hoạch là bao nhiêu?'));self.assertEqual(p.operand_slots[0].column_role,'PLAN')
 def test_two_cell_plan_not_profile(self):
  fs,es=self.facts('Chỉ tiêu | Kế hoạch 2034\n--- | ---\nVốn điều lệ | 47.000.000 đồng\n');f=next(f for f in fs if f.metric=='charter_capital');self.assertFalse(f.is_profile);self.assertEqual(f.role,'plan');self.assertEqual(f.period,'2034')

 def test_default_role_filters_unequal_values_and_column_order(self):
  from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
  from test_post_hrm_contract import fact
  engine=object.__new__(ContractDocumentEngine)
  p=compile_question_plan(resolve_public_question('Vốn điều lệ là bao nhiêu?'))
  actual=fact('charter_capital',37000000,role='actual')
  planned=fact('charter_capital',99000000,role='plan')
  for values in [(actual,planned),(planned,actual)]:
   selected=[f for f in values if engine._matches(f,p.operand_slots[0],p)]
   self.assertEqual(selected,[actual])
 def test_explicit_yearless_plan_cannot_take_profile(self):
  from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
  from test_post_hrm_contract import fact
  engine=object.__new__(ContractDocumentEngine)
  p=compile_question_plan(resolve_public_question('Vốn điều lệ kế hoạch là bao nhiêu?'))
  profile=fact('charter_capital',37000000,role='profile',is_profile=True,period='')
  self.assertFalse(engine._matches(profile,p.operand_slots[0],p))
