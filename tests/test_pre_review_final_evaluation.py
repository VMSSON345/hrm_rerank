from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "pre_review_final_evaluation",
    ROOT / "scripts" / "pre_review_final_evaluation.py",
)
assert SPEC is not None and SPEC.loader is not None
PRE_REVIEW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PRE_REVIEW)


class FinalEvaluationPreReviewTests(unittest.TestCase):
    def _candidate_row(self, row_label: str = "Doanh thu"):
        candidate = {
            "value": "100",
            "raw": "100",
            "source_name": "AAA.pdf",
            "page": 1,
            "row_label": row_label,
            "column_label": "2025",
            "table_id": "table-1",
            "entity_id": "entity-aaa",
            "subtable_id": "table-1/whole-table",
            "unit": "VND",
        }
        return {
            "example_id": "example-1",
            "program": {"operation": "SELECT", "operands": [0]},
            "candidates": [candidate],
        }

    def test_layout_pass_still_requires_human_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "AAA.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((150, 80), "2025")
            page.insert_text((50, 120), "Doanh thu")
            page.insert_text((150, 120), "100")
            document.save(pdf_path)
            document.close()

            output, report = PRE_REVIEW.pre_review(
                [self._candidate_row()],
                [{"example_id": "example-1"}],
                {"AAA.pdf": pdf_path},
            )

        self.assertEqual(output[0]["review_decision"], "needs_human_pdf_review")
        self.assertEqual(report["decisions"], {"needs_human_pdf_review": 1})

    def test_corrupted_row_is_pre_rejected(self) -> None:
        candidate = {
            "row_label": "16,47%Tổng Nợ phải trả",
            "column_label": "2024",
            "raw": "100",
            "unit": "VND",
        }

        reasons = PRE_REVIEW._strong_text_reasons(candidate, 0)

        self.assertIn("operand_0:glued_percent_row", reasons)


if __name__ == "__main__":
    unittest.main()
