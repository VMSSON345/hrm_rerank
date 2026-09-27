"""Source mutation and request tests independent of HRM weights."""
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase,main
from unittest.mock import patch
import json
from hrm_finance_qa.document_sources import build_chunks,OCRConfig
from hrm_finance_qa.program_data import merge_candidates
from hrm_finance_qa.post_hrm_source import SourceSnapshot,build_source_facts
from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine

SOURCE='''## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2032
I. Thông tin chung
- Vốn điều lệ: 1.000.000 đồng (Một triệu đồng)
II. Tình hình kinh doanh
Đơn vị: đồng
Chỉ tiêu | TH 2032 | TH 2031 | KH 2032
--- | --- | --- | ---
Lợi nhuận sau thuế | 800 | 1000 | 1500
Lợi nhuận trước thuế | 1000 | 1250 | 1800
Tài sản ngắn hạn | 9000 | 8500 | 9500
Tài sản ngắn hạn khác | 300 | 250 | 400
'''
class RawFixture:
 model=SimpleNamespace(config=SimpleNamespace(max_candidates=64))
 def rank_raw(self,q,cs):return {'pointers':[[0.0]*len(cs)]*4,'select':[0.0]*len(cs),'pairs':{},'segments':0}
class SourceContractTests(TestCase):
 def setUp(self):
  self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'ENTITY.md';self.path.write_text(SOURCE,encoding='utf8');self.snap=SourceSnapshot([self.path]);self.candidates=merge_candidates('Lợi nhuận sau thuế năm 2032',build_chunks([self.path],ocr_config=OCRConfig(enabled=False)),max_candidates=128)
  self.candidate=next(c for c in self.candidates if c.row_label=='Lợi nhuận sau thuế' and c.period=='2032' and c.value=='800')
 def rejected(self,c):
  fs,errors=build_source_facts([c],self.snap);self.assertFalse(fs,errors)
 def test_valid_source_bound_value(self):
  fs,e=build_source_facts([self.candidate],self.snap);self.assertEqual(len(fs),1,e);self.assertEqual(fs[0].canonical_value,800)
 def test_wrong_hash(self):self.rejected(replace(self.candidate,source_line_sha256='0'*64))
 def test_wrong_page(self):self.rejected(replace(self.candidate,page=2))
 def test_wrong_document(self):self.rejected(replace(self.candidate,source_name='OTHER.md'))
 def test_partial_number_span(self):
  a,b=self.candidate.source_span;self.rejected(replace(self.candidate,source_span=(a,b-2)))
 def test_wrong_year(self):self.rejected(replace(self.candidate,period='2035',context=self.candidate.context.replace('PERIOD: 2032','PERIOD: 2035')))
 def test_wrong_column_role(self):self.rejected(replace(self.candidate,column_role='plan'))
 def test_wrong_metric(self):self.rejected(replace(self.candidate,row_label='Tài sản ngắn hạn khác',context=self.candidate.context.replace('ROW: Lợi nhuận sau thuế','ROW: Tài sản ngắn hạn khác')))
 def test_wrong_witness(self):
  meta=dict(self.candidate.table_semantics);hs=[dict(w) for w in meta['header_evidence']];hs[0]['text']=hs[0]['text'].replace('2032','2035');meta['header_evidence']=hs;self.rejected(replace(self.candidate,table_semantics=meta))
 def test_model_value_cannot_override_source(self):
  fs,e=build_source_facts([replace(self.candidate,value='123456789')],self.snap);self.assertTrue(not fs or fs[0].canonical_value==800,e)
 def test_default_missing_accounting_checks_do_not_block(self):
  with patch('hrm_finance_qa.symbolic_verifier.verify_program',side_effect=AssertionError('old verifier called')):
   b=ContractDocumentEngine(RawFixture()).answer('Vốn điều lệ là bao nhiêu',self.path)
  self.assertEqual(b['canonical_value'],'1000000');self.assertTrue(all(not c['blocking'] for c in b['validation']['checks'] if c['status']=='NOT_APPLICABLE'))
 def test_growth_uses_actual_and_correct_order(self):
  b=ContractDocumentEngine(RawFixture()).answer('Lợi nhuận sau thuế năm 2032 tăng bao nhiêu phần trăm so với 2031',self.path);self.assertEqual(Decimal(b['canonical_value']),Decimal('-20'))
 def test_other_assets_not_total(self):
  b=ContractDocumentEngine(RawFixture()).answer('Tài sản ngắn hạn khác năm 2032 là bao nhiêu',self.path);self.assertEqual(Decimal(b['canonical_value']),300)
 def test_unsupported_year_kept_missing(self):
  b=ContractDocumentEngine(RawFixture()).answer('Lợi nhuận sau thuế năm 2040 là bao nhiêu',self.path);self.assertNotEqual(b['status'],'ACCEPTED');self.assertIn('2040',b['rendered'])
 def test_ambiguity_and_previous_metric(self):
  engine=ContractDocumentEngine(RawFixture());q='Lợi nhuận năm 2032 tăng bao nhiêu phần trăm so với 2031';b=engine.answer(q,self.path);self.assertIn('?',b['rendered']);b=engine.answer(q,self.path,previous_question='Lợi nhuận sau thuế năm 2032 là bao nhiêu');self.assertEqual(Decimal(b['canonical_value']),-20)
 def test_unknown_scale_divergence_requires_review(self):
  self.path.write_text(SOURCE+'\nIII. Kết quả tài chính\nChỉ tiêu | TH 2032 | TH 2031\n--- | --- | ---\nLợi nhuận sau thuế | 900 | 1000\n',encoding='utf8')
  b=ContractDocumentEngine(RawFixture()).answer('Lợi nhuận sau thuế năm 2032 là bao nhiêu',self.path)
  self.assertEqual(b['status'],'ABSTAIN',b['rendered'])
  self.assertIn('unresolved_competing_disclosure',[c['code'] for c in b['validation']['checks']])
  self.assertIn('chưa xác định đơn vị',b['rendered'])
 def test_eps_and_pe_end_to_end_source_binding(self):
  self.path.write_text("## Page 1\nII. Kết quả kinh doanh\nChỉ tiêu | TH 2032\n--- | ---\nLợi nhuận phân bổ cho cổ đông phổ thông | 120000 đồng\nSố cổ phiếu phổ thông bình quân gia quyền | 100\nLãi cơ bản trên cổ phiếu | 1200 đồng\n\nGiá cổ phiếu | 31/12/2032\n--- | ---\nGiá cổ phiếu | 24000 đồng\n",encoding='utf8')
  engine=ContractDocumentEngine(RawFixture())
  for q,expected in [('Tính EPS năm 2032',1200),('EPS năm 2032 là bao nhiêu',1200),('Tính P/E năm 2032 tại ngày 31/12/2032',20)]:
   with self.subTest(q=q):
    b=engine.answer(q,self.path);self.assertEqual(b['status'],'ACCEPTED',b['rendered']);self.assertEqual(Decimal(b['canonical_value']),expected)
 def test_printed_percent_is_not_reinterpreted_as_growth(self):
  self.path.write_text('## Page 1\nII. Kết quả kinh doanh\nĐơn vị: đồng\nChỉ tiêu | TH 2032 | TH 2031 | SS CK (%)\n---|---|---|---\nTổng doanh thu | 8700 | 10000 | 87%\n',encoding='utf8')
  b=ContractDocumentEngine(RawFixture()).answer('Tổng doanh thu ở cột “SS CK (%)” là bao nhiêu?',self.path);self.assertEqual(b['status'],'ACCEPTED',b['rendered']);self.assertEqual(Decimal(b['canonical_value']),87);self.assertEqual(b['unit'],'%')
 def test_opening_and_closing_dates_come_from_source(self):
  self.path.write_text('## Page 1\nBảng cân đối kế toán\nĐơn vị: đồng\nChỉ tiêu | Số đầu năm 01/01/2032 | Số cuối năm 31/12/2031\n--- | --- | ---\nTổng tài sản | 9000 | 9000\n',encoding='utf8')
  cs=merge_candidates('Tổng tài sản',build_chunks([self.path],ocr_config=OCRConfig(enabled=False)),max_candidates=128);fs,e=build_source_facts(cs,SourceSnapshot([self.path]));self.assertTrue(any(f.period_start=='2032-01-01' for f in fs),e);self.assertTrue(any(f.period_end=='2031-12-31' for f in fs),e)
 def test_number_convention_survives_small_candidate_pool(self):
  self.path.write_text('## Page 1\nI. Thông tin chung\nVốn điều lệ: 50.000.000.000 đồng\nII. Kết quả kinh doanh\nĐơn vị: triệu đồng\nChỉ tiêu | TH 2032 | TH 2031\n--- | --- | ---\nLợi nhuận sau thuế | 60.847 | 4.838\n',encoding='utf8')
  raw=RawFixture();raw.model=SimpleNamespace(config=SimpleNamespace(max_candidates=4))
  b=ContractDocumentEngine(raw).answer('Lợi nhuận sau thuế năm 2032 tăng bao nhiêu phần trăm so với 2031',self.path)
  self.assertEqual(b['status'],'ACCEPTED',b['rendered']);self.assertEqual([Decimal(c['value'])*Decimal(c['scale']) for c in b['citations']],[Decimal(60847000000),Decimal(4838000000)])
 def test_source_revision_invalidates_catalog(self):
  engine=ContractDocumentEngine(RawFixture());q='Lợi nhuận sau thuế năm 2032 là bao nhiêu'
  self.assertEqual(Decimal(engine.answer(q,self.path)['canonical_value']),800)
  self.path.write_text(SOURCE.replace('| 800 |','| 900 |'),encoding='utf8')
  self.assertEqual(Decimal(engine.answer(q,self.path)['canonical_value']),900)
 def test_quarter_is_not_an_annual_result(self):
  self.path.write_text(SOURCE.replace('TH 2032','Quý IV 2032').replace('TH 2031','Quý IV 2031'),encoding='utf8')
  b=ContractDocumentEngine(RawFixture()).answer('Lợi nhuận sau thuế năm 2032 là bao nhiêu',self.path);self.assertNotEqual(b['status'],'ACCEPTED')
 def test_wrong_calendar_date_cannot_use_same_year(self):
  self.path.write_text('## Page 1\nII. Tình hình tài chính\nNợ phải trả tại ngày 30/06/2032 là 9.000.000 đồng.\n',encoding='utf8')
  b=ContractDocumentEngine(RawFixture()).answer('Nợ phải trả tại ngày 31/12/2032 là bao nhiêu đồng?',self.path);self.assertNotEqual(b['status'],'ACCEPTED')
 def test_cross_scope_inputs_not_combined(self):
  self.path.write_text('## Page 1\nII. Kết quả kinh doanh hợp nhất\nĐơn vị: đồng\nChỉ tiêu | TH 2032\n---|---\nLợi nhuận sau thuế | 800\nIII. Kết quả kinh doanh riêng lẻ\nĐơn vị: đồng\nChỉ tiêu | TH 2031\n---|---\nLợi nhuận sau thuế | 1000\n',encoding='utf8')
  b=ContractDocumentEngine(RawFixture()).answer('Lợi nhuận sau thuế năm 2032 tăng bao nhiêu phần trăm so với 2031',self.path);self.assertNotEqual(b['status'],'ACCEPTED')
 def test_indirect_cashflow_check_is_wired_to_source(self):
  from hrm_finance_qa.post_hrm_contract import cross_statement_checks
  self.path.write_text('## Page 1\nBáo cáo kết quả hoạt động kinh doanh\nĐơn vị: đồng\nChỉ tiêu | TH 2032\n---|---\nLợi nhuận trước thuế | 1000\nBáo cáo lưu chuyển tiền tệ (phương pháp gián tiếp)\nĐơn vị: đồng\nChỉ tiêu | TH 2032\n---|---\nLợi nhuận trước thuế | 1000\n',encoding='utf8')
  cs=merge_candidates('',build_chunks([self.path],ocr_config=OCRConfig(enabled=False)),max_candidates=128);fs,e=build_source_facts(cs,SourceSnapshot([self.path]));checks=cross_statement_checks(fs)
  self.assertTrue(any(c.code=='indirect_cashflow_start' and c.status=='PASS' for c in checks),[(c.code,c.status) for c in checks])
 def test_narrative_accounting_scope_is_preserved(self):
  self.path.write_text('## Page 1\nII. Kết quả kinh doanh\nNăm 2032, Doanh thu hợp nhất đạt 4.608,7 tỷ đồng.\n',encoding='utf8')
  cs=merge_candidates('',build_chunks([self.path],ocr_config=OCRConfig(enabled=False)),max_candidates=128);fs,e=build_source_facts(cs,SourceSnapshot([self.path]))
  money=[f for f in fs if f.metric=='revenue_unspecified' and f.period=='2032'];self.assertTrue(money,e);self.assertEqual(money[0].scope,'consolidated')
if __name__=='__main__':main()
