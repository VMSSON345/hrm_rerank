from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ocr_financial_review_pages",
    ROOT / "scripts" / "ocr_financial_review_pages.py",
)
assert SPEC is not None and SPEC.loader is not None
OCR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OCR)


class OcrFinancialReviewPagesTests(unittest.TestCase):
    def test_accepts_bounded_discovery_selection(self) -> None:
        item = {
            "total_pages": 10,
            "pages": [
                {"page": 7, "selection": "discovery"},
                {"page": 2, "selection": "core"},
                {"page": 3, "selection": "not_selected"},
                {"page": 7, "selection": "discovery"},
            ],
        }
        self.assertEqual(OCR._selected_pages(item), [2, 7])

    def test_rejects_selected_page_outside_document(self) -> None:
        item = {
            "total_pages": 2,
            "pages": [{"page": 3, "selection": "discovery"}],
        }
        with self.assertRaisesRegex(ValueError, "outside"):
            OCR._selected_pages(item)

    def test_can_limit_ocr_to_table_structure_pages(self) -> None:
        item = {
            "total_pages": 10,
            "pages": [
                {
                    "page": 2,
                    "selection": "discovery",
                    "reasons": ["uniform_document_coverage"],
                },
                {
                    "page": 7,
                    "selection": "discovery",
                    "reasons": ["table_structure_score"],
                },
            ],
        }
        self.assertEqual(
            OCR._selected_pages(item, required_reason="table_structure_score"),
            [7],
        )


if __name__ == "__main__":
    unittest.main()
