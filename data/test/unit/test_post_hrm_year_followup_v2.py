from unittest import TestCase
import test_post_hrm_semantic_transfer as fixtures
from hrm_finance_qa.conversation_context import resolve_followup
from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
class YearFollowupTests(TestCase):
 setUp=fixtures.SemanticTransferTests.setUp
 def test_inherits_metric_scope_and_requeries_changed_source(self):
  source='## Page 1\nTình hình tài chính hợp nhất\nĐơn vị: đồng\nChỉ tiêu | Năm 2033 | Năm 2032\n--- | --- | ---\nTài sản ngắn hạn | 1800 | 1200\n'
  engine=ContractDocumentEngine(fixtures.FlatRanking());self.p.write_text(source,encoding='utf8')
  previous='Tài sản ngắn hạn hợp nhất năm 2033 là bao nhiêu đồng?'
  self.assertEqual(engine.answer(previous,self.p)['canonical_value'],'1800')
  r=engine.answer('Còn năm 2032?',self.p,previous_question=previous)
  self.assertEqual(r['canonical_value'],'1200',r['rendered']);self.assertEqual(r['context_resolution']['inherited']['scope'],'consolidated')
  self.p.write_text(source.replace('1200','1739'),encoding='utf8')
  self.assertEqual(engine.answer('Còn năm 2032?',self.p,previous_question=previous)['canonical_value'],'1739')
 def test_comparison_requires_clarification(self):
  self.assertIsNone(resolve_followup('Còn năm 2031?','Tài sản ngắn hạn năm 2033 tăng bao nhiêu phần trăm so với năm 2032?'))
 def test_missing_new_year_not_reused(self):
  self.p.write_text('## Page 1\nTình hình tài chính\nĐơn vị: đồng\nChỉ tiêu | Năm 2033\n--- | ---\nTài sản ngắn hạn | 1800\n',encoding='utf8')
  r=ContractDocumentEngine(fixtures.FlatRanking()).answer('Còn năm 2032?',self.p,previous_question='Tài sản ngắn hạn năm 2033 là bao nhiêu đồng?')
  self.assertNotEqual(r['status'],'ACCEPTED');self.assertIn('2032',r['rendered'])
