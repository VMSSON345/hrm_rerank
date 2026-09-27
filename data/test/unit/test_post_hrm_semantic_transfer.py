"""Mechanism tests: synthetic sources, independent values, no company fixtures."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from types import SimpleNamespace
from decimal import Decimal
from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
from hrm_finance_qa.metric_semantics import identity

class FlatRanking:
 model=SimpleNamespace(config=SimpleNamespace(max_candidates=64))
 def rank_raw(self,q,cs):return {'pointers':[[0.0]*len(cs)]*4,'select':[0.0]*len(cs),'pairs':{},'segments':0}

class SemanticTransferTests(TestCase):
 def setUp(self):
  self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)/'REPORT.md'
 def answer(self,text,q):
  self.p.write_text(text,encoding='utf8');return ContractDocumentEngine(FlatRanking()).answer(q,self.p)
 def source(self,heading='Financial Performance',metric='Net Revenue',columns='2032 | 2031',numbers='1800 | 1200',unit='VND'):
  return f'## Page 1\nANNUAL REPORT 2032\n1. {heading}\nUnit: {unit}\nIndicator | {columns}\n--- | --- | ---\n{metric} | {numbers}\n'
 def test_language_and_changed_amounts(self):
  for heading,label in [('Financial Performance','Net Revenue'),('Tình hình tài chính','Doanh thu thuần'),('Business Performance','Net Sales')]:
   for a,b in [(1800,1200),(3179,1487)]:
    with self.subTest(heading=heading,a=a):
     out=self.answer(self.source(heading,label,numbers=f'{a} | {b}'),'Doanh thu thuần năm 2032 tăng bao nhiêu phần trăm so với 2031')
     self.assertEqual(out['status'],'ACCEPTED',out['rendered']);self.assertAlmostEqual(Decimal(out['canonical_value']),(Decimal(a)-Decimal(b))/Decimal(b)*100,places=20)
     self.assertIn('Doanh thu thuần',out['rendered'])
     self.assertTrue(any('Performance' in w['text'] or 'Tình hình' in w['text'] for c in out['citations'] for w in c['source_context']))
 def test_explicit_tax_labels(self):
  for label in ['Profit After Corporate Income Tax','Profit After Income Tax','Earnings After Tax','Lợi nhuận sau thuế']:
   with self.subTest(label=label):
    out=self.answer(self.source(metric=label),'Lợi nhuận sau thuế năm 2032 là bao nhiêu đồng');self.assertEqual(out['canonical_value'],'1800',out['rendered'])
 def test_qualified_metrics_are_not_collapsed(self):
  for label in ['Net Profit','Profit After Tax Attributable to Owners','Profit Before Tax of Subsidiary','Net Revenue from Exports']:
   self.assertTrue(identity(label).startswith('literal:'),label)
 def test_missing_second_year_does_not_erase_other_metric(self):
  source=self.source(metric='Profit Before Tax')+'Profit After Corporate Income Tax | N/A | 900\n'
  out=self.answer(source,'Lợi nhuận năm 2032 tăng bao nhiêu phần trăm so với 2031')
  self.assertNotEqual(out['status'],'ACCEPTED');self.assertIn('sau thuế',out['rendered']);self.assertIn('trước thuế',out['rendered'])
 def test_missing_role_does_not_erase_other_metric(self):
  source=self.source(metric='Profit Before Tax',columns='Actual 2032 | Actual 2031')+'\n2. Selected Indicators\nIndicator | 2032 | 2031\n--- | --- | ---\nProfit After Corporate Income Tax | 1500 | 900\n'
  out=self.answer(source,'Lợi nhuận năm 2032 tăng bao nhiêu phần trăm so với 2031')
  self.assertNotEqual(out['status'],'ACCEPTED');self.assertIn('sau thuế',out['rendered'])
 def test_plan_heading_is_not_actual(self):
  for title in ['Financial Plan','Financial Forecast','Kế hoạch tài chính']:
   with self.subTest(title=title):
    out=self.answer(self.source(title),'Doanh thu thuần năm 2032 là bao nhiêu đồng');self.assertNotEqual(out['status'],'ACCEPTED',out['rendered'])
 def test_explicit_plan_column_overrides_actual_heading(self):
  out=self.answer(self.source(columns='Plan 2032 | Actual 2032'),'Doanh thu thuần năm 2032 là bao nhiêu đồng');self.assertEqual(out['canonical_value'],'1200',out['rendered'])
  out=self.answer(self.source(columns='Plan 2032 | Actual 2032'),'Doanh thu thuần kế hoạch năm 2032 là bao nhiêu đồng');self.assertEqual(out['canonical_value'],'1800',out['rendered']);self.assertIn('kế hoạch',out['rendered'])
 def test_unknown_heading_does_not_invent_actual(self):
  out=self.answer(self.source('Selected Indicators'),'Doanh thu thuần năm 2032 là bao nhiêu đồng');self.assertNotEqual(out['status'],'ACCEPTED')
 def test_missing_year_not_from_question(self):
  out=self.answer(self.source(),'Doanh thu thuần năm 2045 là bao nhiêu đồng');self.assertNotEqual(out['status'],'ACCEPTED');self.assertIn('2045',out['rendered'])
 def test_new_section_stops_prior_actual_context(self):
  source=self.source()+'\n2. Financial Plan\nIndicator | 2035 | 2034\n--- | --- | ---\nNet Revenue | 9800 | 7300\n'
  out=self.answer(source,'Doanh thu thuần năm 2035 là bao nhiêu đồng');self.assertNotEqual(out['status'],'ACCEPTED')
 def test_order_and_filename_invariance(self):
  source=self.source(columns='2031 | 2032',numbers='1200 | 1800')
  for name in ['UNSEEN.md','abc123def0-OTHER.md']:
   self.p=Path(self.tmp.name)/name;out=self.answer(source,'Doanh thu thuần năm 2032 là bao nhiêu đồng');self.assertEqual(out['canonical_value'],'1800');self.assertNotIn('abc123def0-',out['rendered'])
 def test_rejection_messages_not_repeated(self):
  out=self.answer(self.source('Selected Indicators'),'So sánh doanh thu thuần năm 2031 và 2032')
  parts=out['rendered'].split('\n\n');self.assertEqual(len(parts),len(set(parts)));self.assertIn('Doanh thu thuần',out['rendered'])
 def test_conflicting_disclosures_remain_blocked(self):
  out=self.answer(self.source()+self.source(numbers='1801 | 1200'),'Doanh thu thuần năm 2032 là bao nhiêu đồng');self.assertNotEqual(out['status'],'ACCEPTED')

 def test_unmapped_profit_label_prevents_false_unique_choice(self):
  source=self.source(metric='Profit Before Tax')+'Net Profit | 1500 | 900\n'
  out=self.answer(source,'Lợi nhuận năm 2032 tăng bao nhiêu phần trăm so với 2031')
  self.assertNotEqual(out['status'],'ACCEPTED');self.assertIn('Net Profit',out['rendered'])
 def test_explicit_scale_is_preserved(self):
  out=self.answer(self.source(unit='triệu đồng'),'Doanh thu thuần năm 2032 là bao nhiêu đồng')
  self.assertEqual(out['canonical_value'],'1800000000',out['rendered'])

 def test_unmapped_clarification_preserves_operation_and_years(self):
  from finhrm_web.clarification import pending_metric_choice,complete_metric_choice
  source=self.source(metric='Profit Before Tax')+'Net Profit | 1500 | 900\n'
  question='Lợi nhuận năm 2032 tăng bao nhiêu phần trăm so với 2031'
  out=self.answer(source,question);digest=out['question_resolution']['document_sha256']
  pending=pending_metric_choice(out,question,digest);self.assertIsNotNone(pending)
  completed=complete_metric_choice('Net Profit',pending,digest)
  self.assertEqual(completed,'Net Profit thực hiện năm 2032 tăng bao nhiêu phần trăm so với 2031')
  final=self.answer(source,completed);self.assertEqual(final['status'],'ACCEPTED',final['rendered'])
  self.assertAlmostEqual(Decimal(final['canonical_value']),Decimal(200)/3,places=20)
  self.assertIsNone(complete_metric_choice('Net Profit',pending,'different-document'))
