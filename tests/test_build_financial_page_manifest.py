import unittest

from scripts.build_financial_page_manifest import (
    MIN_FINANCIAL_SCORE,
    build_document_manifest,
    render_markdown,
    score_financial_page,
)


class FinancialPageScoringTests(unittest.TestCase):
    def test_structured_statement_page_exceeds_selection_threshold(self) -> None:
        result = score_financial_page(
            "BẢNG CÂN ĐỐI KẾ TOÁN\n"
            "Đơn vị tính: VND | Mã số | Thuyết minh | Số cuối năm | Số đầu năm\n"
            "TÀI SẢN NGẮN HẠN | 100 | 1.250.000.000 | 1.100.000.000\n"
            "TỔNG TÀI SẢN | 270 | 4.800.000.000 | 4.200.000.000\n"
            "NỢ PHẢI TRẢ | 300 | 2.100.000.000 | 1.900.000.000\n"
            "VỐN CHỦ SỞ HỮU | 400 | 2.700.000.000 | 2.300.000.000"
        )

        self.assertGreaterEqual(result["score"], MIN_FINANCIAL_SCORE)
        self.assertTrue(result["numeric_qa_ready"])
        self.assertIn("bang can doi ke toan", result["statement_hits"])
        self.assertIn("total_assets", result["metric_hits"])

    def test_narrative_with_isolated_metric_is_not_selected(self) -> None:
        result = score_financial_page(
            "Ban điều hành đánh giá tổng tài sản tăng tích cực trong năm. "
            "Chiến lược tiếp theo tập trung vào quản trị và phát triển thị trường."
        )

        self.assertLess(result["score"], MIN_FINANCIAL_SCORE)

    def test_table_of_contents_without_values_is_not_numeric_qa_ready(self) -> None:
        result = score_financial_page(
            "Báo cáo thường niên gồm Bảng cân đối kế toán, Báo cáo kết quả "
            "hoạt động kinh doanh, Báo cáo lưu chuyển tiền tệ và Thuyết minh "
            "báo cáo tài chính."
        )

        self.assertGreaterEqual(result["score"], MIN_FINANCIAL_SCORE)
        self.assertFalse(result["numeric_qa_ready"])

    def test_statement_names_with_two_governance_numbers_are_not_qa_ready(self) -> None:
        result = score_financial_page(
            "Báo cáo tài chính được kiểm toán gồm Bảng cân đối kế toán, "
            "Báo cáo kết quả hoạt động kinh doanh, Báo cáo lưu chuyển tiền tệ "
            "và Thuyết minh báo cáo tài chính. Cổ đông nhận 51,5 tỷ đồng và "
            "1,92 tỷ đồng cổ tức."
        )

        self.assertGreaterEqual(result["score"], MIN_FINANCIAL_SCORE)
        self.assertFalse(result["numeric_qa_ready"])

    def test_ratio_table_accepts_decimal_and_percent_values(self) -> None:
        result = score_financial_page(
            "Chỉ tiêu | Năm 2023 | Năm 2024\n"
            "--- | --- | ---\n"
            "Tài sản ngắn hạn/Nợ ngắn hạn | 1,16 | 1,17\n"
            "Lợi nhuận sau thuế/Tổng tài sản | -6,5% | 4.25%\n"
            "Lợi nhuận sau thuế/Vốn chủ sở hữu | 0,012 | 0,010"
        )

        self.assertGreaterEqual(result["numeric_values"], 6)
        self.assertEqual(result["page_class"], "financial_summary")
        self.assertTrue(result["numeric_qa_ready"])

    def test_single_metric_dense_table_is_a_financial_note(self) -> None:
        result = score_financial_page(
            "Tăng, giảm tài sản cố định vô hình\n"
            "Chỉ tiêu | Đầu năm | Tăng trong năm | Cuối năm\n"
            "--- | --- | --- | ---\n"
            "Nguyên giá | 146.129.900 | 20.000.000 | 166.129.900\n"
            "Khấu hao lũy kế | 40.000.000 | 12.000.000 | 52.000.000"
        )

        self.assertEqual(result["page_class"], "financial_note")
        self.assertTrue(result["financial_note_ready"])

    def test_governance_table_with_percentages_remains_non_financial(self) -> None:
        result = score_financial_page(
            "Họ và tên | Chức vụ | Số buổi tham dự | Tỷ lệ\n"
            "--- | --- | --- | ---\n"
            "Nguyễn Văn A | Chủ tịch HĐQT | 12 | 100%\n"
            "Trần Văn B | Thành viên HĐQT | 10 | 83,3%"
        )

        self.assertGreaterEqual(result["numeric_values"], 2)
        self.assertEqual(result["page_class"], "non_financial")
        self.assertFalse(result["numeric_qa_ready"])

    def test_financial_phrase_in_prose_does_not_relabel_governance_table(self) -> None:
        result = score_financial_page(
            "Ban kiểm soát đã xem xét tình hình tài chính và báo cáo tài chính.\n"
            "Họ và tên | Chức vụ | Số cổ phần | Tỷ lệ\n"
            "--- | --- | --- | ---\n"
            "Nguyễn Văn A | Chủ tịch HĐQT | 3.721.154 | 51,5%\n"
            "Trần Văn B | Thành viên HĐQT | 1.920.000 | 26,5%"
        )

        self.assertIn("tinh hinh tai chinh", result["summary_hits"])
        self.assertEqual(result["table_summary_hits"], [])
        self.assertEqual(result["page_class"], "non_financial")

    def test_share_par_value_page_is_selected_for_multiply_evidence(self) -> None:
        audit = {
            "document": "/data/report.pdf",
            "sha256": "abc",
            "pages": [{"page": 1, "native_usable": True, "reason": "usable"}],
        }
        texts = [
            "Vốn cổ phần\n"
            "Số cổ phiếu đang lưu hành | 2.000.000 | 1.800.000\n"
            "Mệnh giá cổ phiếu | 10.000 | 10.000"
        ]

        result = build_document_manifest(audit, texts)

        self.assertEqual(result["core_financial_pages"], 0)
        self.assertEqual(result["note_financial_pages"], 1)
        self.assertTrue(result["pages"][0]["share_par_value_ready"])
        self.assertEqual(result["pages"][0]["selection"], "note")

    def test_manifest_excludes_unusable_pages_and_adds_clean_neighbors(self) -> None:
        audit = {
            "document": "/data/report.pdf",
            "sha256": "abc",
            "pages": [
                {"page": 1, "native_usable": True, "reason": "usable"},
                {"page": 2, "native_usable": True, "reason": "usable"},
                {"page": 3, "native_usable": True, "reason": "usable"},
                {"page": 4, "native_usable": False, "reason": "empty"},
            ],
        }
        texts = [
            "Thông tin chung về doanh nghiệp và hoạt động quản trị.",
            "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH\n"
            "Mã số | Thuyết minh | Năm nay | Năm trước\n"
            "Doanh thu thuần | 10 | 900.000.000 | 800.000.000\n"
            "Lợi nhuận sau thuế | 60 | 120.000.000 | 100.000.000",
            "Thông tin bổ sung cho báo cáo và chữ ký người đại diện.",
            "BẢNG CÂN ĐỐI KẾ TOÁN | TỔNG TÀI SẢN | 4.000.000.000",
        ]

        result = build_document_manifest(audit, texts)
        by_page = {item["page"]: item for item in result["pages"]}

        self.assertEqual(result["native_usable_pages"], 3)
        self.assertEqual(result["core_financial_pages"], 1)
        self.assertEqual(result["selected_review_pages"], 3)
        self.assertEqual(by_page[2]["selection"], "core")
        self.assertEqual(by_page[1]["selection"], "neighbor")
        self.assertEqual(by_page[3]["selection"], "neighbor")
        self.assertNotIn(4, by_page)

    def test_markdown_reports_pipeline_failures_separately(self) -> None:
        report = {
            "metrics": {
                "documents_processed": 1,
                "documents_total": 2,
                "pipeline_failures": 1,
                "native_usable_pages": 1,
                "core_financial_pages": 0,
                "selected_review_pages": 0,
            },
            "documents": [
                {
                    "document": "/data/report.pdf",
                    "pages": [],
                }
            ],
        }

        markdown = render_markdown(report)

        self.assertIn("Documents processed: 1/2", markdown)
        self.assertIn("Pipeline failures: 1", markdown)


if __name__ == "__main__":
    unittest.main()
