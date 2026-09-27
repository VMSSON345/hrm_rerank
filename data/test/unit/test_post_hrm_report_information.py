from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from hrm_finance_qa.post_hrm_source import SourceSnapshot
from hrm_finance_qa.report_text_fields import answer_fields,source_fields,verify_text_fact

class ReportInformationTests(TestCase):
 def setUp(self):
  self.t=TemporaryDirectory();self.addCleanup(self.t.cleanup);self.p=Path(self.t.name)/'UNSEEN.md'
 def snapshot(self,text):self.p.write_text(text,encoding='utf8');return SourceSnapshot([self.p])
 def test_bilingual_labels_keep_identifiers_as_text(self):
  s=self.snapshot('## Page 1\n1. Tên tổ chức /Name of organization: New Company\n- Mã chứng khoán/ Stock code: XYZ\n- Điện thoại liên hệ/Tel.: 0012.3456\n')
  r=answer_fields('Cho tôi mã chứng khoán và số điện thoại',s);self.assertEqual([o['answer'] for o in r['outputs']],['XYZ','0012.3456']);self.assertIsNone(r['canonical_value']);self.assertFalse(r['hrm_used'])
 def test_inline_phone_fax_separate_values(self):
  s=self.snapshot('## Page 1\nĐiện thoại : 000 123 Fax: 000 456\n');r=answer_fields('Số điện thoại và fax là gì?',s);self.assertEqual([o['answer'] for o in r['outputs']],['000 123','000 456'])
 def test_branch_and_auditor_not_company(self):
  s=self.snapshot('## Page 1\nĐịa chỉ: Trụ sở A\nWebsite: company.test\nQuá trình hình thành và phát triển:\n2. Chi nhánh B\nĐịa chỉ: Chi nhánh B\nWebsite: branch.test\n');r=answer_fields('Địa chỉ công ty ở đâu?',s);self.assertEqual(r['answer'],'Trụ sở A')
 def test_unknown_qualifiers_not_silently_removed(self):
  s=self.snapshot('## Page 1\nĐịa chỉ: Company A\n')
  for q in ['Địa chỉ công ty con ở đâu?','Địa chỉ năm 2035 là gì?','Website và doanh thu năm 2032 là bao nhiêu?']:
   self.assertIsNone(answer_fields(q,s),q)
 def test_missing_text_not_filled_from_next_label(self):
  s=self.snapshot('## Page 1\nTên tiếng Anh:\nWebsite: company.test\n');r=answer_fields('Tên tiếng Anh của công ty là gì?',s);self.assertEqual(r['status'],'ABSTAIN')
 def test_partial_fields_keep_verified_information(self):
  s=self.snapshot('## Page 1\nWebsite: company.test\n');r=answer_fields('Website và email là gì?',s);self.assertEqual(r['coverage_state'],'PARTIAL');self.assertEqual(r['outputs'][0]['answer'],'company.test');self.assertEqual(r['outputs'][1]['status'],'ABSTAIN')
 def test_conflicting_fields_not_first_wins(self):
  s=self.snapshot('## Page 1\nWebsite: a.test\nWebsite: b.test\n');r=answer_fields('Website là gì?',s);self.assertEqual(r['status'],'ABSTAIN');self.assertIn('a.test',r['rendered']);self.assertIn('b.test',r['rendered'])
 def test_source_replay_detects_mutations(self):
  s=self.snapshot('## Page 1\nWebsite: unique.test\n');f=source_fields(s)[0];self.assertTrue(verify_text_fact(f,s))
  for changes in [{'text_value':'wrong.test'},{'metric':'field:email'},{'page':2},{'source_line_sha256':'0'*64},{'source_span':(0,5)},{'document_sha256':'0'*64}]:
   self.assertFalse(verify_text_fact(f.model_copy(update=changes),s),changes)
 def test_unrelated_numeric_question_not_intercepted(self):
  self.assertIsNone(answer_fields('Lợi nhuận năm 2032 tăng bao nhiêu phần trăm so với 2031?',self.snapshot('## Page 1\nWebsite: company.test\n')))

 def test_count_total_not_subgroup_or_salary(self):
  s=self.snapshot('## Page 1\nSố lượng cán bộ, nhân viên: Tính đến ngày 31/12/2032, toàn Công ty có 91 nhân sự, trong đó 17 nhân sự có chứng chỉ, lương 12.000.000 đồng/người/tháng\n')
  r=answer_fields('Năm 2032 công ty có bao nhiêu nhân viên?',s);self.assertEqual(r['answer'],'91 nhân sự')
  f=source_fields(s)[0];line=s.lines[self.p.name][1];a=line.index('17 nhân sự')
  self.assertFalse(verify_text_fact(f.model_copy(update={'source_span':(a,a+len('17 nhân sự')),'text_value':'17 nhân sự'}),s))
  self.assertEqual(answer_fields('Năm 2035 công ty có bao nhiêu nhân viên?',s)['status'],'ABSTAIN')
 def test_count_plan_not_actual(self):
  s=self.snapshot('## Page 1\nSố lượng lao động: Dự kiến năm 2032 có 123 người\n')
  self.assertEqual(answer_fields('Năm 2032 công ty có bao nhiêu nhân viên?',s)['status'],'ABSTAIN')

 def test_branch_context_ends_at_new_reporting_section(self):
  s=self.snapshot('## Page 1\nTên giao dịch: Company X\nĐịa chỉ: Root address\nQuá trình hình thành và phát triển:\n(2) Chi nhánh B\nNgành nghề kinh doanh: branch only\nĐịa chỉ: Branch address\n2. Tổ chức và nhân sự\nSố lượng cán bộ, nhân viên: Tính đến 31/12/2032, công ty có 92 người\n')
  self.assertEqual(answer_fields('Năm 2032 công ty có bao nhiêu nhân viên?',s)['answer'],'92 người')
  self.assertEqual(answer_fields('Ngành nghề kinh doanh là gì?',s)['status'],'ABSTAIN')
  self.assertEqual(answer_fields('Địa chỉ ở đâu?',s)['answer'],'Root address')

 def test_conflict_lists_values_without_computing(self):
  from test_post_hrm_semantic_transfer import FlatRanking
  from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
  text='## Page 1\nBÁO CÁO THƯỜNG NIÊN NĂM 2032\n1. Tình hình tài chính\nĐơn vị: đồng\nChỉ tiêu | TH 2032 | TH 2031\n--- | --- | ---\nLợi nhuận sau thuế | 800 | 600\n2. Kết quả kinh doanh\nĐơn vị: đồng\nChỉ tiêu | TH 2032 | TH 2031\n--- | --- | ---\nLợi nhuận sau thuế | 900 | 600\n'
  self.snapshot(text);r=ContractDocumentEngine(FlatRanking()).answer('Lợi nhuận sau thuế năm 2032 tăng bao nhiêu phần trăm so với 2031',self.p)
  self.assertEqual(r['status'],'ABSTAIN');self.assertIsNone(r['canonical_value']);self.assertIn('800 đồng',r['rendered']);self.assertIn('900 đồng',r['rendered'])
  self.assertFalse(r['disclosure_conflict']['computation_performed']);self.assertIsNone(r['proof']['execution_certificate'])
  self.assertEqual({x['canonical_value'] for x in r['disclosure_conflict']['disclosures']},{'800','900'})
