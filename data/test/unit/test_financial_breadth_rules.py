from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from hrm_finance_qa.post_hrm_engine import ContractDocumentEngine
from hrm_finance_qa.post_hrm_source import metric_key
from test_post_hrm_semantic_transfer import FlatRanking


class FinancialBreadthRulesTests(TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'OTHER_REPORT.md'
        self.engine=ContractDocumentEngine(FlatRanking())

    def write(self, text):
        self.path.write_text(text,encoding='utf-8')
        return self.path

    def test_annual_ratios_before_separator_and_formula_ratios(self):
        self.write('''## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
Phân tích tài chính (tiếp)
HIỆU QUẢ HOẠT ĐỘNG
Chỉ số | 2032 | 2033
ROA (LNST/Tổng tài sản) | 1,00% | 2,50%
ROE (LNST/Vốn chủ sở hữu) | 2,00% | 4,00%
--- | --- | ---
Biên lợi nhuận gộp | 10,0% | 12,0%
Chỉ số | 2032 | 2033
Thanh toán hiện hành (TSNH/Nợ NH) | 1,20 | 1,80
Thanh toán nhanh ((TSNH-HTK)/Nợ NH) | 0,90 | 1,40
''')
        expected=[
            ('ROA năm 2033 là bao nhiêu phần trăm?','2.50'),
            ('ROE năm 2033 là bao nhiêu phần trăm?','4.00'),
            ('Hệ số thanh toán hiện hành năm 2033 là bao nhiêu lần?','1.80'),
            ('Hệ số thanh toán nhanh năm 2033 là bao nhiêu lần?','1.40'),
        ]
        for question,value in expected:
            with self.subTest(question=question):
                result=self.engine.answer(question,self.path)
                self.assertEqual(result['status'],'ACCEPTED',result['rendered'])
                self.assertEqual(float(result['canonical_value']),float(value))
                self.assertEqual(result['citations'][0]['period'],'2033')

        for bare,metric in [('ROA','return_on_assets'),
                            ('Chỉ số ROE','return_on_equity')]:
            prompt=self.engine.answer(bare,self.path)
            self.assertEqual(prompt['status'],'ABSTAIN')
            self.assertEqual(prompt['named_ratio_year_choices']['metric'],metric)
            self.assertEqual(prompt['named_ratio_year_choices']['years'],
                             ['2032','2033'])
            self.assertIn('Bạn muốn dùng năm nào?',prompt['rendered'])
        margin=self.engine.answer('Biên lợi nhuận',self.path)
        self.assertIn('loại và năm',margin['rendered'])
        pair=self.engine.answer('ROA và ROE năm 2033 lần lượt là bao nhiêu phần trăm?',self.path)
        self.assertEqual(pair['status'],'ACCEPTED',pair['rendered'])
        self.assertEqual([float(x['canonical_value']) for x in pair['outputs']],[2.5,4.0])
        missing=self.engine.answer('ROA năm 2034 là bao nhiêu phần trăm?',self.path)
        self.assertEqual(missing['status'],'ABSTAIN')

    def test_plan_ratio_is_printed_and_money_scale_stays_unresolved(self):
        self.write('''## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
Trong năm 2033, Công ty hoàn thành 50% kế hoạch doanh thu:
Chỉ tiêu | Kế hoạch | Thực hiện | % hoàn thành kế hoạch
--- | --- | --- | ---
Tổng doanh thu hoạt động | 1000 | 500 | 50,00%
''')
        question='Tổng doanh thu hoạt động thực hiện năm 2033 đạt bao nhiêu phần trăm kế hoạch?'
        result=self.engine.answer(question,self.path)
        self.assertEqual(result['status'],'ACCEPTED',result['rendered'])
        self.assertEqual(float(result['canonical_value']),50.0)
        money=self.engine.answer('Tổng doanh thu hoạt động thực hiện năm 2033 là bao nhiêu đồng?',self.path)
        self.assertEqual(money['status'],'ABSTAIN',money['rendered'])
        self.write(self.path.read_text(encoding='utf-8').replace('50,00%','51,00%'))
        changed=self.engine.answer(question,self.path)
        self.assertEqual(changed['status'],'ACCEPTED',changed['rendered'])
        self.assertEqual(float(changed['canonical_value']),51.0)

    def test_margin_conflict_is_about_margin_and_requires_scope(self):
        self.write("""## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
Nhìn lại hoạt động năm 2033, công ty có thay đổi.
Doanh thu thuần hợp nhất tăng; biên lợi nhuận gộp phục hồi từ 10,0% lên 15,0%.
Phân tích tài chính năm 2033
Biên lợi nhuận gộp từ 10,0% lên 15,2%.
""")
        result=self.engine.answer('Biên lợi nhuận gộp năm 2033 là bao nhiêu phần trăm?',self.path)
        self.assertEqual(result['status'],'ABSTAIN')
        self.assertIn('biên lợi nhuận gộp',result['rendered'].lower())
        self.assertIn('15,0%',result['rendered'])
        self.assertIn('15,2%',result['rendered'])
        self.assertEqual(len(result['citations']),2)
        self.write('''## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
c) Hợp đồng hoặc giao dịch với người nội bộ và người có liên quan:
\\\\- Công ty thuê văn phòng của Công ty Liên quan:
\\\\+ Số tiền thuê: 5 triệu đồng/1 tháng
\\\\+ Quan hệ liên quan: người có liên quan của HĐQT.
''')
        unrelated=self.engine.answer('Biên lợi nhuận ròng năm 2033 là bao nhiêu phần trăm?',self.path)
        self.assertEqual(unrelated['status'],'ABSTAIN')
        self.assertNotIn('giao dịch',unrelated['rendered'].lower())

    def test_cash_flow_comparison_uses_same_year_sources(self):
        self.write("""## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
Báo cáo kết quả hoạt động kinh doanh
Đơn vị: VND
Chỉ tiêu | Năm 2033 | Năm 2032
--- | --- | ---
Lợi nhuận sau thuế | 150000 | 120000
Báo cáo lưu chuyển tiền tệ
Đơn vị: VND
Chỉ tiêu | Năm 2033 | Năm 2032
--- | --- | ---
Tiền thuần từ hoạt động kinh doanh | 100000 | 90000
""")
        result=self.engine.answer(
            'Dòng tiền thuần từ hoạt động kinh doanh năm 2033 có lớn hơn lợi nhuận sau thuế không?',
            self.path)
        self.assertEqual(result['status'],'ACCEPTED',result['rendered'])
        self.assertIn('Không.',result['rendered'])
        self.assertEqual(result['canonical_value'],'-50000')
        missing=self.engine.answer(
            'Dòng tiền thuần từ hoạt động kinh doanh năm 2034 có lớn hơn lợi nhuận sau thuế không?',
            self.path)
        self.assertEqual(missing['status'],'ABSTAIN')

    def test_coded_liabilities_and_cash_flow_names(self):
        self.assertEqual(metric_key('C. NỢ PHẢI TRẢ'),metric_key('Tổng nợ phải trả'))
        self.assertEqual(metric_key('Tiền thuần từ hoạt động kinh doanh'),
                         metric_key('Dòng tiền thuần từ hoạt động kinh doanh'))

    def test_local_results_year_and_unscaled_other_scope(self):
        self.write("""## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
1. Kết quả hoạt động kinh doanh năm 2033.
Lợi nhuận sau thuế : 500 đồng.
b. Kết quả sản xuất kinh doanh hợp nhất năm 2033.
Chỉ tiêu | Năm 2033 | Năm 2032
--- | --- | ---
Lợi nhuận sau thuế | 500 | 400
""")
        result=self.engine.answer('Lợi nhuận sau thuế năm 2033 là bao nhiêu đồng?',self.path)
        self.assertEqual(result['status'],'ACCEPTED',result['rendered'])
        self.assertEqual(result['canonical_value'],'500')
        self.assertIn(result['citations'][0]['period_binding'], {'local_results_heading','source_narrative_local'})
        wrong=self.engine.answer('Lợi nhuận sau thuế năm 2034 là bao nhiêu đồng?',self.path)
        self.assertEqual(wrong['status'],'ABSTAIN')

    def test_dated_plan_actual_ratio_column(self):
        self.write("""## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
Tình hình thực hiện so với kế hoạch.
STT | Nội dung | ĐVT | Kế hoạch 2033 | Thực hiện 2033 | Tỷ lệ (%)
--- | --- | --- | --- | --- | ---
STT: 01 | Tổng doanh thu | Triệu đồng | Kế hoạch 2033: 1.000 | Thực hiện 2033: 823 | 82,3
""")
        result=self.engine.answer(
            'Tổng doanh thu thực hiện năm 2033 đạt bao nhiêu phần trăm kế hoạch?',self.path)
        self.assertEqual(result['status'],'ACCEPTED',result['rendered'])
        self.assertEqual(float(result['canonical_value']),82.3)
        self.assertEqual(result['citations'][0]['period_binding'],'plan_actual_header_ratio')
        wrong=self.engine.answer(
            'Tổng doanh thu thực hiện năm 2032 đạt bao nhiêu phần trăm kế hoạch?',self.path)
        self.assertEqual(wrong['status'],'ABSTAIN')

    def test_audit_heading_and_opinion_across_page(self):
        self.write("""## Page 1
BÁO CÁO THƯỜNG NIÊN
Năm báo cáo : 2033
1. Đơn vị kiểm toán độc lập.
\\\\+ Công ty TNHH Kiểm toán Ví dụ
\\\\+ Ý kiến kiểm toán độc lập :
1
## Page 2
Theo ý kiến của chúng tôi, Báo cáo tài chính hợp nhất đã phản ánh trung thực và hợp lý tình hình tài chính tại ngày 31 tháng 12 năm 2033, phù hợp với chuẩn mực kế toán Việt Nam.
""")
        auditor=self.engine.answer('Đơn vị kiểm toán độc lập năm 2033 là công ty nào?',self.path)
        self.assertEqual(auditor['status'],'ACCEPTED',auditor['rendered'])
        self.assertIn('Công ty TNHH Kiểm toán Ví dụ',auditor['rendered'])
        opinion=self.engine.answer('Ý kiến kiểm toán độc lập năm 2033 là gì?',self.path)
        self.assertEqual(opinion['status'],'ACCEPTED',opinion['rendered'])
        self.assertEqual({c['page'] for c in opinion['citations']},{1,2})
        wrong=self.engine.answer('Đơn vị kiểm toán độc lập năm 2032 là công ty nào?',self.path)
        self.assertEqual(wrong['status'],'ABSTAIN')

    def test_ratio_labels_with_printed_formulas_and_unit_column(self):
        self.write("""## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
Tình hình tài chính
Chỉ tiêu | ĐVT | Năm 2032 | Năm 2033
1. Khả năng thanh toán |  |  |
+ Hệ số khả năng thanh toán ngắn hạn:TS ngắn hạn/Nợ ngắn hạn | Lần | 1.10 | 1.70
--- | --- | --- | ---
+ Hệ số thanh toán nhanh:TSLĐ - Hàng tồn khoNợ ngắn hạn | Lần | 0.80 | 1.30
2. Năng lực hoạt động |  |  |
+ Vòng quay hàng tồn khoGiá vốn bán hàng/Hàng tồn kho bình quân | Lần | 2.10 | 3.20
3. Khả năng sinh lời |  |  |
+ Hệ số lợi nhuận sau thuế/Tổng tài sản | % | 0.10 | Năm 2033: 0.05
""")
        for question,value in [
            ('Hệ số thanh toán hiện hành năm 2033 là bao nhiêu lần?',1.7),
            ('Hệ số thanh toán nhanh năm 2033 là bao nhiêu lần?',1.3),
            ('Vòng quay hàng tồn kho năm 2033 là bao nhiêu lần?',3.2),
            ('ROA năm 2033 là bao nhiêu phần trăm?',0.05),
        ]:
            with self.subTest(question=question):
                result=self.engine.answer(question,self.path)
                self.assertEqual(result['status'],'ACCEPTED',result['rendered'])
                self.assertEqual(float(result['canonical_value']),value)
        wrong=self.engine.answer('ROA năm 2034 là bao nhiêu phần trăm?',self.path)
        self.assertEqual(wrong['status'],'ABSTAIN')

    def test_qualified_opinion_and_internal_advance_are_source_bound(self):
        self.write("""## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
VI. Báo cáo tài chính
1. Ý kiến kiểm toán
Ý kiến ngoại trừ và vấn đề cần nhấn mạnh của kiểm toán:
Tại báo cáo năm 2033, Kiểm toán viên có đưa ra ý kiến ngoại trừ như sau:
Tại ngày 31/12/2033, Công ty có số dư tạm ứng cho Ông A – Kế toán trưởng Công ty với số tiền khoảng 2,5 tỷ VND.
""")
        opinion=self.engine.answer('Ý kiến kiểm toán báo cáo tài chính năm 2033 thuộc loại gì?',self.path)
        self.assertEqual(opinion['status'],'ACCEPTED',opinion['rendered'])
        self.assertIn('ngoại trừ',opinion['rendered'])
        advance=self.engine.answer(
            'Khoản tạm ứng cho Kế toán trưởng tại 31/12/2033 khoảng bao nhiêu?',self.path)
        self.assertEqual(advance['status'],'ACCEPTED',advance['rendered'])
        self.assertIn('2,5 tỷ VND',advance['rendered'])
        wrong=self.engine.answer(
            'Khoản tạm ứng cho Kế toán trưởng tại 31/12/2032 khoảng bao nhiêu?',self.path)
        self.assertEqual(wrong['status'],'ABSTAIN')

    def test_undated_stock_fact_does_not_inherit_nearby_date(self):
        self.write("""## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
2. Tình hình tài chính:
\\- Lỗ lũy kế đến thời điểm 31/12/2033 là: 100 đồng
\\- Tài sản ngắn hạn: 500 đồng
""")
        result=self.engine.answer(
            'Tài sản ngắn hạn tại ngày 31/12/2033 là bao nhiêu đồng?',self.path)
        self.assertEqual(result['status'],'ABSTAIN')
        self.assertIn('chưa gắn ngày chốt',result['rendered'])

    def test_rounded_summary_does_not_override_exact_other_scope(self):
        self.write("""## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
Tình hình tài chính
Đơn vị tính: Tỷ đồng
Chỉ tiêu | Năm 2033
--- | ---
Doanh thu thuần | Năm 2033: 344,09
## Page 2
Báo cáo kết quả hoạt động kinh doanh hợp nhất năm 2033
Đơn vị: VND
Chỉ tiêu | Thực hiện 2033 | Thực hiện 2032
--- | --- | ---
Doanh thu thuần bán hàng và cungcấp dịch vụ | Thực hiện 2033: 344.087.811.289 | Thực hiện 2032: 300.000.000.000
""")
        generic=self.engine.answer('Doanh thu thuần năm 2033 là bao nhiêu đồng?',self.path)
        self.assertEqual(generic['status'],'ACCEPTED',generic['rendered'])
        self.assertEqual(generic['canonical_value'],'344087811289')
        exact_source=self.path.read_text(encoding='utf-8')
        self.write(exact_source.split('## Page 2')[0])
        rounded_only=self.engine.answer('Doanh thu thuần năm 2033 là bao nhiêu đồng?',self.path)
        self.assertEqual(rounded_only['status'],'ACCEPTED',rounded_only['rendered'])
        self.assertEqual(rounded_only['canonical_value'],'344090000000')
        self.assertIn('khoảng',rounded_only['rendered'])
        self.assertIn('344,09 tỷ đồng',rounded_only['rendered'])
        self.write(exact_source)
        restored=self.engine.answer('Doanh thu thuần năm 2033 là bao nhiêu đồng?',self.path)
        self.assertEqual(restored['status'],'ACCEPTED',restored['rendered'])
        self.assertEqual(restored['canonical_value'],'344087811289')

    def test_requested_percent_requires_percent_source_unit(self):
        self.write("""## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
Phân tích tài chính
Chỉ số | Năm 2033
--- | ---
ROA (LNST/Tổng tài sản) | 0,05
""")
        result=self.engine.answer('ROA năm 2033 là bao nhiêu phần trăm?',self.path)
        self.assertEqual(result['status'],'ABSTAIN')
        self.assertIn('đơn vị phần trăm',result['rendered'])

    def test_different_scope_and_precision_on_another_mineru_report(self):
        source=Path(__file__).resolve().parents[3]/'data/test/suites/company_holdout_v1/sources/reserved/UDC.md'
        generic=self.engine.answer('Doanh thu thuần năm 2024 là bao nhiêu đồng?',source)
        self.assertEqual(generic['status'],'ABSTAIN',generic['rendered'])
        self.assertIn('344,09 tỷ đồng',generic['rendered'])
        self.assertIn('344.087.811.289 VND',generic['rendered'])
        explicit=self.engine.answer('Doanh thu thuần hợp nhất năm 2024 là bao nhiêu đồng?',source)
        self.assertEqual(explicit['status'],'ACCEPTED',explicit['rendered'])
        self.assertEqual(explicit['canonical_value'],'344087811289')
        roa=self.engine.answer('ROA năm 2024 là bao nhiêu phần trăm?',source)
        self.assertEqual(roa['status'],'ABSTAIN')
        self.assertIn('đơn vị phần trăm',roa['rendered'])

    def test_qualified_audit_and_going_concern_on_another_mineru_report(self):
        source=Path(__file__).resolve().parents[3]/'data/test/suites/company_holdout_v1/sources/reserved/UDC.md'
        opinion=self.engine.answer('Ý kiến kiểm toán báo cáo tài chính năm 2024 thuộc loại gì?',source)
        self.assertEqual(opinion['status'],'ACCEPTED',opinion['rendered'])
        self.assertIn('ngoại trừ',opinion['rendered'])
        concern=self.engine.answer(
            'Báo cáo kiểm toán có nghi ngờ đáng kể về khả năng hoạt động liên tục năm 2024 không?',
            source)
        self.assertEqual(concern['status'],'ACCEPTED',concern['rendered'])
        self.assertIn('Có.',concern['rendered'])
        composition=self.engine.answer(
            'Doanh thu chính năm 2024 đến từ các hoạt động nào và chiếm bao nhiêu phần trăm?',
            source)
        self.assertEqual(composition['status'],'ACCEPTED',composition['rendered'])
        self.assertIn('93%',composition['rendered'])

    def test_audit_firm_year_and_negative_related_disclosure(self):
        self.write('''## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
3.3. Hợp đồng hoặc giao dịch với người nội bộ và người có liên quan đến người nội bộ (Không phát sinh)
VI. Báo cáo tài chính
1. Ý kiến kiểm toán:
Theo ý kiến chúng tôi, Báo cáo tài chính đã phản ánh trung thực và hợp lý trên các khía cạnh trọng yếu tình hình tài chính năm 2033, phù hợp chuẩn mực kế toán Việt Nam.
2. Báo cáo tài chính được kiểm toán
\\- Báo cáo tài chính được kiểm toán bởi: Công ty TNHH Kiểm toán Mẫu
''')
        auditor=self.engine.answer('Đơn vị kiểm toán báo cáo tài chính năm 2033 là công ty nào?',self.path)
        self.assertEqual(auditor['status'],'ACCEPTED',auditor['rendered'])
        self.assertIn('Công ty TNHH Kiểm toán Mẫu',auditor['rendered'])
        wrong=self.engine.answer('Đơn vị kiểm toán báo cáo tài chính năm 2032 là công ty nào?',self.path)
        self.assertEqual(wrong['status'],'ABSTAIN')
        related=self.engine.answer('Theo mục 3.3, có phát sinh hợp đồng hoặc giao dịch với người nội bộ và người có liên quan không?',self.path)
        self.assertEqual(related['status'],'ACCEPTED',related['rendered'])
        self.assertIn('không phát sinh',related['rendered'])

    def test_parent_profit_row_before_repeated_header_is_not_total_profit(self):
        self.write('''## Page 1
BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH HỢP NHẤT
Năm 2033
Mã số | Chỉ tiêu | Thuyết minh | Năm 2033 | Năm 2032
Mã số | Chỉ tiêu | Thuyết minh | VND | VND
--- | --- | --- | --- | ---
60 | 16. Lợi nhuận sau thuế TNDN |  | Năm 2033: 1000 | Năm 2032: 900
61 | 17. Lợi nhuận sau thuế của công ty mẹ |  | Năm 2033: 800 | Năm 2032: 700
Mã số | Chỉ tiêu | Thuyết minh | Năm 2033 | Năm 2032
Mã số | Chỉ tiêu | Thuyết minh | VND | VND
--- | --- | --- | --- | ---
62 | 18. Lợi nhuận sau thuế của cổ đông không kiểm soát |  | Năm 2033: 200 | Năm 2032: 200
''')
        parent=self.engine.answer('Lợi nhuận sau thuế của công ty mẹ năm 2033 là bao nhiêu đồng?',self.path)
        self.assertEqual(parent['status'],'ACCEPTED',parent['rendered'])
        self.assertEqual(parent['canonical_value'],'800')
        self.assertEqual(parent['citations'][0]['row_label'],'17. Lợi nhuận sau thuế của công ty mẹ')
        total=self.engine.answer('Lợi nhuận sau thuế hợp nhất năm 2033 là bao nhiêu đồng?',self.path)
        self.assertEqual(total['status'],'ACCEPTED',total['rendered'])
        self.assertEqual(total['canonical_value'],'1000')


    def test_split_ratio_header_with_restated_year_and_unproven_percent(self):
        self.write('''## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
4. Kết quả tài chính
b. Các chỉ tiêu tài chính chủ yếu
Các chỉ tiêu | Năm 2032(trình bày lại) | Năm 2033
1. Chỉ tiêu về khả năng thanh toán |  |
+ Hệ số thanh toán ngắn hạn: |  |
--- | --- | ---
TSLĐ/Nợ ngắn hạn | 1,20 lần | 1,12 lần
+ Hệ số thanh toán nhanh: |  |
TSNH - Hàng tồn kho/Nợ ngắn hạn | 0,99 lần | 0,89 lần
2. Chỉ tiêu về năng lực hoạt động |  |
+ Vòng quay hàng tồn kho: |  |
Giá vốn hàng bán/Hàng tồn kho bình quân | Năm 2032(trình bày lại): 21,65 | Năm 2033: 21,47
3. Chỉ tiêu về khả năng sinh lời |  |
+ Hệ số Lợi nhuận sau thuế/Tổng tài sản | 6,89% | 8,37%
+ Hệ số Lợi nhuận sau thuế/Vốn chủ sở hữu | Năm 2032(trình bày lại): 12,12% | Năm 2033: 14,41
''')
        cases=[
            ('Hệ số thanh toán hiện hành năm 2033 là bao nhiêu lần?','1.12'),
            ('Hệ số thanh toán nhanh năm 2033 là bao nhiêu lần?','0.89'),
            ('ROA năm 2033 là bao nhiêu phần trăm?','8.37'),
            ('Vòng quay hàng tồn kho năm 2033 là bao nhiêu?','21.47'),
        ]
        for question,value in cases:
            with self.subTest(question=question):
                result=self.engine.answer(question,self.path)
                self.assertEqual(result['status'],'ACCEPTED',result['rendered'])
                self.assertEqual(float(result['canonical_value']),float(value))
        roe=self.engine.answer('ROE năm 2033 là bao nhiêu phần trăm?',self.path)
        self.assertEqual(roe['status'],'ABSTAIN')
        self.assertIn('đơn vị phần trăm',roe['rendered'])

    def test_same_sentence_plan_ratio_and_equal_precision_profit_conflict(self):
        self.write('''## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
4. Kết quả tài chính
a. Kết quả tài chính
Chỉ tiêu | Đơn vị tính | Năm 2032 | Năm 2033
--- | --- | --- | ---
Lợi nhuận sau thuế | triệu đồng | Năm 2032: 20756 | Năm 2033: 25327
Tổng lợi nhuận trước thuế năm 2032 là 26952 triệu đồng. Kế hoạch lợi nhuận năm 2033 là 27900 triệu đồng, thực hiện năm 2033 đạt 32958 triệu đồng (đạt 118,13% so với kế hoạch năm 2033).
Lợi nhuận sau thuế đạt 25326 triệu đồng, tăng 4570 triệu đồng so với thực hiện năm 2032 và tăng 3.306 triệu đồng so với kế hoạch năm 2033.
''')
        ratio=self.engine.answer('Lợi nhuận trước thuế thực hiện năm 2033 đạt bao nhiêu phần trăm kế hoạch?',self.path)
        self.assertEqual(ratio['status'],'ACCEPTED',ratio['rendered'])
        self.assertEqual(float(ratio['canonical_value']),118.13)
        conflict=self.engine.answer('Lợi nhuận sau thuế năm 2033 là bao nhiêu đồng?',self.path)
        self.assertEqual(conflict['status'],'ABSTAIN')
        self.assertIn('25.327.000.000',conflict['rendered'])
        self.assertIn('25.326.000.000',conflict['rendered'])
        self.assertEqual(len(conflict['citations']),2)

    def test_audit_reference_and_shareholder_transactions_other_layout(self):
        self.write('''## Page 1
BÁO CÁO THƯỜNG NIÊN NĂM 2033
Hợp đồng hoặc giao dịch với cổ đông nội bộ:
Năm 2033, Công ty thực hiện hợp đồng bán hàng với Công ty liên kết (cổ đông lớn nắm giữ 51% vốn) với số tiền 10 tỷ đồng. Ngoài ra, Công ty còn có các giao dịch với các bên liên quan với số tiền 2 triệu đồng.
## Page 2
1. Ý kiến kiểm toán viên
Ý kiến của kiểm toán viên tại văn bản số 001/2034 ngày 20/3/2034 của Công ty TNHH Kiểm toán và Tư vấn Mẫu
Theo ý kiến của chúng tôi, Báo cáo tài chính đã phản ánh trung thực và hợp lý trên các khía cạnh trọng yếu tình hình tài chính tại ngày 31/12/2033, phù hợp với chuẩn mực kế toán Việt Nam.
''')
        auditor=self.engine.answer('Đơn vị kiểm toán báo cáo tài chính năm 2033 là ai?',self.path)
        self.assertEqual(auditor['status'],'ACCEPTED',auditor['rendered'])
        self.assertIn('Công ty TNHH Kiểm toán và Tư vấn Mẫu',auditor['rendered'])
        opinion=self.engine.answer('Ý kiến kiểm toán báo cáo tài chính năm 2033 thuộc loại gì?',self.path)
        self.assertEqual(opinion['status'],'ACCEPTED',opinion['rendered'])
        self.assertIn('chấp nhận toàn phần',opinion['rendered'])
        related=self.engine.answer('Có giao dịch với cổ đông lớn hoặc bên liên quan năm 2033 không?',self.path)
        self.assertEqual(related['status'],'ACCEPTED',related['rendered'])
        self.assertIn('10 tỷ đồng',related['rendered'])
        self.assertIn('2 triệu đồng',related['rendered'])


