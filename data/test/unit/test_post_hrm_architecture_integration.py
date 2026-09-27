from unittest import TestCase
from pathlib import Path
from tempfile import TemporaryDirectory
from decimal import Decimal
from types import SimpleNamespace
import numpy as np
from hrm_finance_qa.conversation_context import resolve_followup
from hrm_finance_qa.number_formats import alternatives,resolve
from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
from hrm_finance_qa.post_hrm_contract import ValidationReport,accounting_checks
from test_post_hrm_semantic_transfer import FlatRanking
from test_post_hrm_contract import fact

class Encoder:
 def __init__(self):self.calls=[]
 def encode(self,texts,*,role):
  self.calls.append((role,len(texts)))
  return np.array([[1.,0.] for t in texts])

class ArchitectureIntegrationTests(TestCase):
 def setUp(self):
  self.t=TemporaryDirectory();self.addCleanup(self.t.cleanup);self.p=Path(self.t.name)/'UNSEEN.md'
 def source(self):
  self.p.write_text('## Page 1\nBÁO CÁO THƯỜNG NIÊN 2032\n1. Tình hình tài chính\nĐơn vị: đồng\nChỉ tiêu | TH 2032 | TH 2031\n--- | --- | ---\nDoanh thu thuần | 1800 | 1200\nLợi nhuận sau thuế | 250 | 200\nTài sản ngắn hạn | 1000 | 900\nTài sản ngắn hạn khác | 74 | 56\n',encoding='utf8')
 def test_followup_preserves_operation_years(self):
  for q in ['Doanh thu 2031 và 2032 tăng bao nhiêu?','Doanh thu thuần năm 2032 tăng bao nhiêu phần trăm so với 2031?','So sánh doanh thu thuần năm 2031 và 2032']:
   r=resolve_followup('Còn lợi nhuận sau thuế?',q);self.assertIsNotNone(r);self.assertIn('2031',r['resolved_question']);self.assertIn('2032',r['resolved_question']);self.assertIn('lợi nhuận sau thuế',r['resolved_question'])
 def test_no_silent_inheritance_with_new_qualifiers(self):
  prev='Doanh thu thuần năm 2032 tăng bao nhiêu phần trăm so với 2031?'
  for q in ['Lợi nhuận sau thuế là bao nhiêu?','Còn lợi nhuận sau thuế năm 2034?','Còn lợi nhuận sau thuế của công ty con?','Còn lợi nhuận?']:
   self.assertIsNone(resolve_followup(q,prev),q)
  self.assertIsNone(resolve_followup('Còn lợi nhuận sau thuế?',''))
 def test_followup_end_to_end_and_next_turn(self):
  self.source();e=ContractDocumentEngine(FlatRanking());r=e.answer('Còn lợi nhuận sau thuế?',self.p,previous_question='Doanh thu thuần năm 2032 tăng bao nhiêu phần trăm so với 2031?');self.assertEqual(r['canonical_value'],'25');self.assertEqual(r['context_resolution']['inherited']['years'],['2031','2032'])
  next_r=e.answer('Còn doanh thu thuần?',self.p,previous_question=r['question_resolution']['resolved_question']);self.assertEqual(next_r['canonical_value'],'50')
 def test_numeric_guard_retains_two_interpretations(self):
  for raw,values in [('12,345,678.000',{'12345678.000','12345678000'}),('12.345.678,000',{'12345678.000','12345678000'})]:
   self.assertEqual({v['value'] for v in alternatives(raw).values()},values);self.assertEqual(resolve(raw)['state'],'UNKNOWN')
 def test_regular_and_negative_numbers_unchanged(self):
  self.assertEqual({Decimal(v['value']) for v in alternatives('(1.234.567)').values()},{Decimal('-1234567')})
  self.assertEqual({Decimal(v['value']) for v in alternatives('1,234.50').values()},{Decimal('1234.50')})
 def test_mixed_number_not_published_without_source_convention(self):
  self.p.write_text('## Page 1\nVốn điều lệ: 12,345,678.000 đồng\n',encoding='utf8');r=ContractDocumentEngine(FlatRanking()).answer('Vốn điều lệ là bao nhiêu?',self.p);self.assertEqual(r['status'],'ABSTAIN')
 def test_check_applicability_and_required_missing(self):
  checks=accounting_checks([fact('total_assets',100)]);self.assertEqual(next(c.status for c in checks if c.code=='balance_sheet'),'INSUFFICIENT_EVIDENCE');self.assertTrue(ValidationReport(checks=checks).can_publish)
  r=ValidationReport();r.add('data_contract','required_period','INSUFFICIENT_EVIDENCE','Missing period',blocking=True);self.assertFalse(r.can_publish)
 def test_dense_encoder_and_graph_trace_are_real_routes(self):
  self.source();bundle=FlatRanking();bundle.semantic_encoder=Encoder();r=ContractDocumentEngine(bundle,retrieval_mode='hybrid_graph').answer('Tài sản ngắn hạn khác năm 2032 là bao nhiêu?',self.p)
  self.assertEqual(r['canonical_value'],'74');trace=r['retrieval_trace'];self.assertEqual(trace['dense'],'used_existing_encoder');self.assertGreater(trace['graph_edges'],0);self.assertTrue(trace['graph_paths']);self.assertTrue(any(x[0]=='query' for x in bundle.semantic_encoder.calls));self.assertTrue(all(s['retained']>0 for s in trace['slots']))
 def test_graph_off_has_no_edges_same_executor(self):
  self.source()
  for mode in ['legacy','hybrid','hybrid_graph']:
   r=ContractDocumentEngine(FlatRanking(),retrieval_mode=mode).answer('Lợi nhuận sau thuế năm 2032 tăng bao nhiêu phần trăm so với 2031?',self.p);self.assertEqual(r['canonical_value'],'25');self.assertEqual(r['retrieval_trace']['mode'],mode)
   if mode=='hybrid':self.assertEqual(r['retrieval_trace']['graph_edges'],0)
 def test_graph_does_not_supply_missing_period(self):
  self.source();r=ContractDocumentEngine(FlatRanking()).answer('Tài sản ngắn hạn khác năm 2039 là bao nhiêu?',self.p);self.assertEqual(r['status'],'ABSTAIN')

 def test_required_other_row_survives_lexical_miss(self):
  from unittest.mock import patch
  self.source();bundle=FlatRanking();bundle.model=SimpleNamespace(config=SimpleNamespace(max_candidates=2))
  with patch('hrm_finance_qa.hybrid_retrieval.rank_chunks',return_value=[]):
   r=ContractDocumentEngine(bundle,retrieval_mode='hybrid_graph').answer('Tài sản ngắn hạn khác năm 2032 là bao nhiêu?',self.p)
  self.assertEqual(r['canonical_value'],'74');trace=r['retrieval_trace'];self.assertEqual(trace['bm25_candidates'],0);self.assertLessEqual(trace['selected_candidates'],2);self.assertTrue(any(any(o.startswith('required_slot:') for o in c['origins']) for c in trace['candidates']))
 def test_two_period_slots_survive_budget(self):
  from unittest.mock import patch
  self.source();bundle=FlatRanking();bundle.model=SimpleNamespace(config=SimpleNamespace(max_candidates=2))
  with patch('hrm_finance_qa.hybrid_retrieval.rank_chunks',return_value=[]):
   r=ContractDocumentEngine(bundle).answer('Lợi nhuận sau thuế năm 2032 tăng bao nhiêu phần trăm so với 2031?',self.p)
  self.assertEqual(r['canonical_value'],'25');self.assertEqual(len(r['retrieval_trace']['slots']),2);self.assertTrue(all(s['retained']==1 for s in r['retrieval_trace']['slots']))
 def test_graph_reserves_real_neighbors_under_budget(self):
  self.source();bundle=FlatRanking();bundle.model=SimpleNamespace(config=SimpleNamespace(max_candidates=4))
  r=ContractDocumentEngine(bundle,retrieval_mode='hybrid_graph').answer('Lợi nhuận sau thuế năm 2032 là bao nhiêu?',self.p)
  self.assertEqual(r['canonical_value'],'250');trace=r['retrieval_trace'];self.assertGreater(len(trace['graph_added_selected']),0);self.assertLessEqual(trace['selected_candidates'],4)

 def test_adaptive_keeps_complete_pool_without_dense_call(self):
  self.source();bundle=FlatRanking();bundle.semantic_encoder=Encoder()
  r=ContractDocumentEngine(bundle).answer('Lợi nhuận sau thuế năm 2032 là bao nhiêu?',self.p)
  self.assertEqual(r['canonical_value'],'250');self.assertEqual(bundle.semantic_encoder.calls,[]);self.assertEqual(r['retrieval_trace']['strategy'],'trace_sufficient_lexical_pool');self.assertGreater(r['retrieval_trace']['graph_edges'],0)
