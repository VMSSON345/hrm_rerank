from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from hrm_finance_qa.document_search import (
    VietnameseFinancialDocumentIndex,
    _is_heading,
)


class DocumentSearchHeadingTests(unittest.TestCase):
    def test_known_lowercase_headings_are_accent_insensitive(self) -> None:
        for heading in (
            "mục lục",
            "muc luc",
            "báo cáo tài chính",
            "bao cao tai chinh",
            "báo cáo tình hình tài chính hợp nhất",
            "bao cao tinh hinh tai chinh rieng",
            "báo cáo lưu chuyển tiền tệ riêng",
            "báo cáo tình hình biến động vốn chủ sở hữu",
        ):
            with self.subTest(heading=heading):
                self.assertTrue(_is_heading(heading))

    def test_table_of_contents_entries_are_not_sections(self) -> None:
        for entry in (
            "Mục lục ........ 2",
            "MUC LUC 2",
            "Bảng cân đối kế toán ........ 12",
            "BANG CAN DOI KE TOAN 12",
            "04. Báo cáo lưu chuyển tiền tệ hợp nhất",
            "Thuyết minh báo cáo tài chính hợp nhất",
        ):
            with self.subTest(entry=entry):
                self.assertFalse(_is_heading(entry))

    def test_financial_phrases_inside_narrative_are_not_headings(self) -> None:
        for line in (
            "Đánh giá kết quả hoạt động kinh doanh",
            "kết quả hoạt động kinh doanh của công ty",
            "tình hình tài chính luôn nằm trong tầm kiểm soát",
            "01. THÔNG TIN CHUNGTình hình tài chính4",
        ):
            with self.subTest(line=line):
                self.assertFalse(_is_heading(line))

    def test_missing_headings_fall_back_without_breaking_search(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "report.txt").write_text(
                "doanh thu thuần 100\n"
                "giá vốn hàng bán 80\n"
                "lợi nhuận gộp 20\n",
                encoding="utf-8",
            )

            index = VietnameseFinancialDocumentIndex.build_from_text_dir(source)

            outline = index.outline("report")
            self.assertEqual(len(outline.sections), 1)
            self.assertEqual(outline.sections[0].heading, "Document start")
            self.assertTrue(index.search("doanh thu thuần"))
            self.assertTrue(index.search("doanh thu thuan"))


if __name__ == "__main__":
    unittest.main()
