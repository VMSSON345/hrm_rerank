from decimal import Decimal
from unittest import TestCase
import test_post_hrm_semantic_transfer as fixtures
class SourceScaleTests(TestCase):
 setUp=fixtures.SemanticTransferTests.setUp
 answer=fixtures.SemanticTransferTests.answer
 def table(self,unit,value,year=2032,scope=''):
  return f'## Page 1\nKết quả kinh doanh {scope}\n{unit}\nChỉ tiêu | TH {year}\n--- | ---\nLợi nhuận trước thuế | {value}\n'
 def test_unknown_summary_cannot_create_false_conflict(self):
  s=self.table('', '18.427')+self.table('Đơn vị: đồng','18.427.234.120')
  r=self.answer(s,'Lợi nhuận trước thuế năm 2032 là bao nhiêu đồng?')
  self.assertEqual(r['status'],'ACCEPTED',r['rendered']);self.assertEqual(r['canonical_value'],'18427234120')
 def test_unknown_only_is_blocked(self):
  r=self.answer(self.table('','18427'),'Lợi nhuận trước thuế năm 2032 là bao nhiêu đồng?')
  self.assertNotEqual(r['status'],'ACCEPTED');self.assertIn('đơn vị',r['rendered'])
 def test_true_conflict_still_blocks(self):
  r=self.answer(self.table('Đơn vị: triệu đồng','18427')+self.table('Đơn vị: đồng','20.427.234.120'),'Lợi nhuận trước thuế năm 2032 là bao nhiêu đồng?')
  self.assertNotEqual(r['status'],'ACCEPTED');self.assertIn('chưa thống nhất',r['rendered'])
 def test_same_printed_number_with_different_units_blocks_after_conversion(self):
  s=self.table('Đơn vị: triệu đồng','18427')+self.table('Đơn vị: đồng','18427')
  r=self.answer(s,'Lợi nhuận trước thuế năm 2032 là bao nhiêu đồng?')
  self.assertEqual(r['status'],'ABSTAIN')
  self.assertIn('chưa thống nhất',r['rendered'])
  self.assertIn('18.427.000.000 đồng',r['rendered'])
  self.assertIn('18.427 đồng',r['rendered'])
  self.assertIsNone(r['canonical_value'])
  self.assertFalse(r['disclosure_conflict']['computation_performed'])
  self.assertEqual({x['raw_value'] for x in r['disclosure_conflict']['disclosures']},{'18427'})

 def test_explicit_scales_rounding(self):
  r=self.answer(self.table('Đơn vị: triệu đồng','18427')+self.table('Đơn vị: đồng','18.427.234.120'),'Lợi nhuận trước thuế năm 2032 là bao nhiêu đồng?')
  self.assertEqual(r['canonical_value'],'18427234120',r['rendered'])
 def test_different_period_not_conflict(self):
  r=self.answer(self.table('Đơn vị: đồng','100')+self.table('Đơn vị: đồng','200',2031),'Lợi nhuận trước thuế năm 2032 là bao nhiêu đồng?')
  self.assertEqual(r['canonical_value'],'100',r['rendered'])
 def test_different_scope_not_conflict(self):
  r=self.answer(self.table('Đơn vị: đồng','100',scope='hợp nhất')+self.table('Đơn vị: đồng','200',scope='riêng lẻ'),'Lợi nhuận trước thuế hợp nhất năm 2032 là bao nhiêu đồng?')
  self.assertEqual(r['canonical_value'],'100',r['rendered'])
 def test_true_conflict_with_unknown_companion_does_not_crash_or_default_unit(self):
  s=(self.table('Đơn vị: triệu đồng','18427')
     +self.table('Đơn vị: đồng','20.427.234.120')
     +self.table('','18427'))
  r=self.answer(s,'Lợi nhuận trước thuế năm 2032 là bao nhiêu đồng?')
  self.assertNotEqual(r['status'],'ACCEPTED')
  self.assertIn('chưa thống nhất',r['rendered'])
  self.assertIn('chưa xác định đơn vị',r['rendered'])
  self.assertNotIn('quy ước mặc định đồng',r['rendered'])
