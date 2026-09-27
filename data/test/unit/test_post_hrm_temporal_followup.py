from unittest import TestCase
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path
from hrm_finance_qa.conversation_context import resolve_followup
from hrm_finance_qa.question_planner import compile_question_plan,resolve_public_question
from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
from test_post_hrm_semantic_transfer import FlatRanking

class TemporalFollowupTests(TestCase):
 def plan(self,q,previous):
  result=resolve_followup(q,previous);self.assertIsNotNone(result)
  return compile_question_plan(resolve_public_question(result['resolved_question']))
 def test_original_percentage_followups(self):
  for q,previous,metric in [
   ('So với năm 2023 tăng bao nhiêu phần trăm?','Doanh thu thuần năm 2024 là bao nhiêu đồng?','doanh thu thuan'),
   ('Có giảm so với năm 2023 không, giảm bao nhiêu phần trăm?','Lợi nhuận sau thuế năm 2024 là bao nhiêu đồng?','loi nhuan sau thue')]:
   p=self.plan(q,previous)
   self.assertEqual(p.operation,'PERCENT_CHANGE');self.assertEqual(set(p.requested_years),{'2023','2024'})
   self.assertEqual(p.metric_text,metric)
 def test_original_endpoint(self):
  p=self.plan('Còn đầu năm?','Tài sản ngắn hạn năm 2023 là bao nhiêu đồng?')
  self.assertEqual(p.operand_slots[0].year,'2023');self.assertEqual(p.operand_slots[0].period_role,'opening')
 def test_explicit_endpoint_year_overrides_previous(self):
  p=self.plan('Còn đầu năm 2035?','Tài sản ngắn hạn năm 2023 là bao nhiêu đồng?')
  self.assertEqual(p.operand_slots[0].year,'2035')
 def test_explicit_reference_year_overrides_previous(self):
  p=self.plan('So với năm 2022 tăng bao nhiêu phần trăm?','Doanh thu thuần năm 2024 là bao nhiêu đồng?')
  self.assertEqual(set(p.requested_years),{'2024','2022'})
 def test_scope_inherited_or_explicitly_replaced(self):
  previous='Tài sản ngắn hạn hợp nhất năm 2023 là bao nhiêu đồng?'
  self.assertEqual(self.plan('Còn đầu năm?',previous).requested_accounting_scope,'consolidated')
  self.assertEqual(self.plan('Còn đầu năm 2024 riêng lẻ?',previous).requested_accounting_scope,'separate')
 def test_new_explicit_metric_year_or_entity_is_not_overwritten(self):
  previous='Doanh thu thuần năm 2024 là bao nhiêu đồng?'
  for q in ['Lợi nhuận sau thuế năm 2022 là bao nhiêu?','Còn lợi nhuận sau thuế năm 2022?','Còn đầu năm của công ty con?']:
   self.assertIsNone(resolve_followup(q,previous))
 def test_missing_or_multiyear_context_cannot_supply_target_year(self):
  for prev in ['', 'Doanh thu thuần là bao nhiêu?', 'Doanh thu thuần năm 2024 tăng bao nhiêu phần trăm so với năm 2023?']:
   self.assertIsNone(resolve_followup('Còn đầu năm?',prev))
 def test_same_year_comparison_requires_clarification(self):
  self.assertIsNone(resolve_followup('So với năm 2024 tăng bao nhiêu phần trăm?','Doanh thu thuần năm 2024 là bao nhiêu?'))
 def test_missing_context_stops_before_source_and_model(self):
  engine=ContractDocumentEngine(FlatRanking())
  with patch('hrm_finance_qa.post_hrm_engine.discover_documents',side_effect=AssertionError('source accessed')),patch('hrm_finance_qa.post_hrm_engine.raw_hrm_ranking',side_effect=AssertionError('model called')):
   for q in ['Còn đầu năm?','So với năm 2023 tăng bao nhiêu phần trăm?','Có giảm so với năm 2023 không, giảm bao nhiêu phần trăm?']:
    b=engine.answer(q,Path('not-accessed.md'))
    self.assertEqual(b['status'],'ABSTAIN');self.assertTrue(b['intent_clarification_required'])
    self.assertEqual(b['timings_seconds']['hrm'],0)
    self.assertEqual(b['timings_seconds']['retrieval'],0)
    self.assertNotIn('missing_reference_base',b['rendered'])
 def test_missing_base_message_is_public(self):
  b=ContractDocumentEngine(FlatRanking()).answer('Doanh thu thuần năm 2024 tăng bao nhiêu phần trăm?',Path('not-accessed.md'))
  self.assertNotIn('missing_reference_base',b['rendered'])
  self.assertEqual(b['timings_seconds']['source_catalog'],0)
 def test_new_query_reads_source_and_preserves_date(self):
  with TemporaryDirectory() as d:
   p=Path(d)/'NEW.md'
   text='## Page 1\nBảng cân đối kế toán hợp nhất\nĐơn vị: đồng\nChỉ tiêu | Số đầu năm 01/01/2031 | Số cuối kỳ 31/12/2031\n--- | --- | ---\nTài sản ngắn hạn | 110.000.000 | 90.000.000\n'
   p.write_text(text,encoding='utf8');engine=ContractDocumentEngine(FlatRanking())
   q='Tài sản ngắn hạn hợp nhất năm 2031 là bao nhiêu đồng?'
   self.assertEqual(engine.answer(q,p)['canonical_value'],'90000000')
   b=engine.answer('Còn đầu năm?',p,previous_question=q)
   self.assertEqual(b['canonical_value'],'110000000')
   self.assertEqual(b['citations'][0]['as_of_date'],'2031-01-01')
   p.write_text(text.replace('110.000.000','130.000.000'),encoding='utf8')
   self.assertEqual(engine.answer('Còn đầu năm?',p,previous_question=q)['canonical_value'],'130000000')
 def test_percentage_uses_both_source_values(self):
  with TemporaryDirectory() as d:
   p=Path(d)/'NEW.md';p.write_text('## Page 1\n1. Tình hình tài chính\nĐơn vị: đồng\nChỉ tiêu | TH 2032 | TH 2031\n--- | --- | ---\nDoanh thu thuần | 1800 | 1200\n',encoding='utf8')
   b=ContractDocumentEngine(FlatRanking()).answer('So với năm 2031 tăng bao nhiêu phần trăm?',p,previous_question='Doanh thu thuần năm 2032 là bao nhiêu đồng?')
   self.assertEqual(b['canonical_value'],'50');self.assertEqual(len(b['citations']),2)

 def test_repeated_comparison_both_word_orders_and_reference_override(self):
  previous='Doanh thu thuần năm 2025 là bao nhiêu?'
  for q,base in [('So với năm 2024 tăng hay giảm bao nhiêu phần trăm?','2024'),('So với năm 2024 giảm hay tăng bao nhiêu phần trăm?','2024'),('So với năm 2023 giảm hay tăng bao nhiêu phần trăm?','2023')]:
   result=resolve_followup(q,previous);self.assertIsNotNone(result)
   plan=compile_question_plan(resolve_public_question(result['resolved_question']))
   self.assertEqual(next(s.year for s in plan.operand_slots if s.id==plan.reference_base),base)
   self.assertEqual(next(s.year for s in plan.operand_slots if s.id!=plan.reference_base),'2025')
   previous=result['resolved_question']
 def test_comparison_target_is_not_maximum_year(self):
  p=self.plan('So với năm 2022 giảm hay tăng bao nhiêu phần trăm?','Doanh thu thuần năm 2023 tăng bao nhiêu phần trăm so với năm 2024?')
  self.assertEqual(next(s.year for s in p.operand_slots if s.id!=p.reference_base),'2023')
 def test_pending_year_keeps_metric_and_reference(self):
  from hrm_finance_qa.conversation_context import complete_followup_year
  q='So với năm 2024 giảm hay tăng bao nhiêu phần trăm?'
  result=complete_followup_year('năm 2025',q,'Doanh thu thuần năm 2024 là bao nhiêu?')
  p=compile_question_plan(resolve_public_question(result))
  self.assertEqual(p.metric_text,'doanh thu thuan');self.assertEqual(set(p.requested_years),{'2024','2025'})
  self.assertIsNone(complete_followup_year('năm 2025',q,''))
  self.assertIsNone(complete_followup_year('năm 2024',q,'Doanh thu thuần năm 2024 là bao nhiêu?'))
 def test_bare_year_without_context_does_not_retrieve(self):
  b=ContractDocumentEngine(FlatRanking()).answer('năm 2025',Path('not-accessed.md'))
  self.assertTrue(b['intent_clarification_required']);self.assertEqual(b['timings_seconds']['retrieval'],0)

 def test_elliptical_comparison_keeps_known_operation_only(self):
  previous='Lợi nhuận sau thuế năm 2024 tăng bao nhiêu phần trăm so với năm 2023?'
  p=self.plan('Vậy so với năm 2022 thì sao?',previous)
  self.assertEqual(p.operation,'PERCENT_CHANGE');self.assertEqual(set(p.requested_years),{'2024','2022'})
  for prior in ['', 'Lợi nhuận sau thuế năm 2024 là bao nhiêu?']:
   self.assertIsNone(resolve_followup('Vậy so với năm 2022 thì sao?',prior))
