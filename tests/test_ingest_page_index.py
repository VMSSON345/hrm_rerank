import unittest

from hrm_finance_qa.document_page_index import PageRecord
from scripts.ingest_page_index import render_page_records_markdown


class PageIndexMarkdownExportTests(unittest.TestCase):
    def test_exports_page_provenance_and_extraction_method(self) -> None:
        records = [
            PageRecord(
                document_fingerprint="fingerprint",
                source_path="/tmp/report.pdf",
                source_name="report.pdf",
                source_size=100,
                source_mtime_ns=1,
                page_number=1,
                native_usable=True,
                native_score=1.0,
                native_reason="usable",
                native_text="Doanh thu thuần | 100 | 90",
                extraction_method="native",
                searchable_text="Doanh thu thuần | 100 | 90",
                ocr_state="not_needed",
            ),
            PageRecord(
                document_fingerprint="fingerprint",
                source_path="/tmp/report.pdf",
                source_name="report.pdf",
                source_size=100,
                source_mtime_ns=1,
                page_number=2,
                native_usable=False,
                native_score=0.0,
                native_reason="empty",
                native_text="",
                extraction_method="mineru",
                searchable_text="Tổng tài sản | 200 | 180",
                ocr_state="high_quality",
            ),
        ]

        output = render_page_records_markdown(records)

        self.assertIn('source_name: "report.pdf"', output)
        self.assertIn("## Page 1", output)
        self.assertIn("extraction_method=native", output)
        self.assertIn("Doanh thu thuần | 100 | 90", output)
        self.assertIn("## Page 2", output)
        self.assertIn("extraction_method=mineru", output)
        self.assertIn("Tổng tài sản | 200 | 180", output)


if __name__ == "__main__":
    unittest.main()
