from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from hrm_finance_qa.post_hrm_source import SourceSnapshot
from hrm_finance_qa.report_text_fields import answer_fields,source_fields,verify_text_fact
from hrm_finance_qa.report_context import undated_monetary_field
from hrm_finance_qa.metric_semantics import identity
class NewLayoutTests(TestCase):
 def setUp(self):
  self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.p=Path(self.tmp.name)/'unknown.md'
 def snap(self,t):self.p.write_text(t,encoding='utf8');return SourceSnapshot([self.p])
 def test_toc_does_not_close_later_explicit_profile(self):
  s=self.snap('## Page 1\nMỤC LỤC\nNgành nghề kinh doanh\nQuá trình phát triển\n## Page 2\nI. THÔNG TIN CHUNG:\nWebsite: new.test\n2. Chi nhánh X\nWebsite: wrong.test\n');self.assertEqual(answer_fields('Website là gì?',s)['answer'],'new.test')
 def test_inline_long_labels_and_number_preserved(self):
  s=self.snap('## Page 1\nSố điện thoại: (012) 345 678 Số fax: 012 345 679 Website: https://new.test/path\n');r=answer_fields('Số điện thoại và fax và website là gì?',s);self.assertEqual([x['answer'] for x in r['outputs']],['(012) 345 678','012 345 679','https://new.test/path'])
 def test_equivalence_not_different_digits_or_path(self):
  for text,question in [('Điện thoại: 01 234\nĐiện thoại: 01 235','Số điện thoại là gì?'),('Website: https://www.new.test/a\nWebsite: https://new.test/b','Website là gì?')]:
   self.assertEqual(answer_fields(question,self.snap('## Page 1\n'+text))['status'],'ABSTAIN')
  s=self.snap('## Page 1\nĐiện thoại: (01) 234\nĐiện thoại: 01234\n');self.assertEqual(answer_fields('Số điện thoại là gì?',s)['status'],'ACCEPTED')
 def test_source_owned_list_and_continuation(self):
  s=self.snap('## Page 1\n3. Ngành nghề kinh doanh\n- Sản xuất thiết bị\n## Page 2\nđiện và phụ kiện;\n- Dịch vụ sửa chữa;\n4. Địa bàn kinh doanh\n- Quận khác\n')
  r=answer_fields('Ngành nghề kinh doanh là gì?',s);self.assertIn('thiết bị điện',r['answer']);self.assertNotIn('Quận khác',r['answer']);self.assertEqual(len(r['citations']),2)
  f=source_fields(s)[0];self.assertTrue(verify_text_fact(f,s));self.assertFalse(verify_text_fact(f.model_copy(update={'source_context':()}),s))
 def test_pair_row_and_unsupported_multicolumn(self):
  r=answer_fields('Ngành nghề kinh doanh là gì?',self.snap('## Page 1\nNgành nghềkinh doanh | Sản xuất giấy\n'));self.assertEqual(r['answer'],'Sản xuất giấy')
  r=answer_fields('Địa chỉ là gì?',self.snap('## Page 1\nĐịa chỉ | Công ty con | nơi khác\n'));self.assertEqual(r['status'],'ABSTAIN')
 def test_narrative_count_year_and_plan(self):
  for prefix,status in [('Tính đến ngày 31/12/2031,','ACCEPTED'),('Dự kiến năm 2031,','ABSTAIN')]:
   s=self.snap('## Page 1\n'+prefix+' số lượng CBNV tại ABC là 172 nhân sự. Trong đó 12 người quản lý.\n');r=answer_fields('Năm 2031 công ty có bao nhiêu nhân viên?',s);self.assertEqual(r['status'],status)
   if status=='ACCEPTED':self.assertEqual(r['answer'],'172 nhân sự');self.assertEqual(answer_fields('Năm 2032 công ty có bao nhiêu nhân viên?',s)['status'],'ABSTAIN')
 def test_report_year_total_requires_personnel_heading(self):
  for heading,status in [('Cán bộ, nhân viên','ACCEPTED'),('Máy móc','ABSTAIN'),('Kế hoạch cán bộ, nhân viên','ABSTAIN')]:
   s=self.snap('## Page 1\nBÁO CÁO THƯỜNG NIÊN 2031\n'+heading+'\n- Tổng số: 43\n');self.assertEqual(answer_fields('Năm 2031 công ty có bao nhiêu nhân viên?',s)['status'],status)
 def test_adjacent_money_uses_own_label_not_previous_dates(self):
  t='Vốn điều lệ: 123.456.000 đồng Vốn đầu tư của chủ sở hữu: 789.123.000 đồng'
  for token,label in [('123.456.000','Vốn điều lệ'),('789.123.000','Vốn đầu tư của chủ sở hữu')]:
   a=t.index(token);f=undated_monetary_field(t,a,a+len(token));self.assertEqual(f['row_label'],label);self.assertIsNone(f['fact_period'])
  t='Vốn điều lệ dự kiến năm 2031: 123.456.000 đồng';a=t.index('123.456');self.assertIsNone(undated_monetary_field(t,a,a+11))
 def test_tax_abbreviation_keeps_qualifiers(self):
  self.assertEqual(identity('Lợi nhuận sau thuế TNDN'),'profit_after_tax');self.assertNotEqual(identity('Lợi nhuận sau thuế TNDN của công ty mẹ'),'profit_after_tax');self.assertNotEqual(identity('Lợi nhuận trước thuế'),'profit_after_tax')

 def test_narrative_link_address_not_registered_address(self):
  s=self.snap('## Page 1\nĐịa chỉ: Trụ sở A\nThông tin đã công bố tại địa chỉ: https://new.test/report\n')
  self.assertEqual(answer_fields('Địa chỉ là gì?',s)['answer'],'Trụ sở A')

 def test_growth_prose_not_undated_multi_field(self):
  from hrm_finance_qa.program_data import _NUMBER_RE
  t='Doanh thu: 246,8 tỷ đồng, tăng 12,4% so với năm 2030, vượt 5% kế hoạch (235 tỷ đồng).'
  first=next(_NUMBER_RE.finditer(t));self.assertIsNone(undated_monetary_field(t,*first.span()))

 def test_two_money_fields_end_to_end_ignore_registration_dates(self):
  from test_post_hrm_semantic_transfer import FlatRanking
  from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
  self.snap('## Page 1\nBÁO CÁO THƯỜNG NIÊN 2031\nI. THÔNG TIN CHUNG\nNgày đăng ký lần đầu: 04/02/2000, thay đổi: 15/12/2020\nVốn điều lệ: 123.456.000 đồng Vốn đầu tư của chủ sở hữu: 789.123.000 đồng\n')
  engine=ContractDocumentEngine(FlatRanking())
  for question,expected in [('Vốn điều lệ là bao nhiêu?','123456000'),('Vốn đầu tư của chủ sở hữu là bao nhiêu?','789123000')]:
   r=engine.answer(question,self.p);self.assertEqual(r['status'],'ACCEPTED');self.assertEqual(r['canonical_value'],expected)
