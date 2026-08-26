from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_final_evaluation_review",
    ROOT / "scripts" / "export_final_evaluation_review.py",
)
assert SPEC is not None and SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class FinalEvaluationReviewExportTests(unittest.TestCase):
    def test_exports_selected_operand_provenance(self) -> None:
        rows = [
            {
                "example_id": "external-test-1",
                "question": "Tong A va B?",
                "answer_value": "15",
                "program": {"operation": "ADD", "operands": [0, 1]},
                "candidates": [
                    {
                        "value": "10",
                        "source_name": "AAA_25CN_BCTN.pdf",
                        "page": 12,
                        "row_label": "A",
                        "column_label": "2025",
                        "unit": "VND",
                        "source_span": [1, 2],
                    },
                    {
                        "value": "5",
                        "source_name": "AAA_25CN_BCTN.pdf",
                        "page": 12,
                        "row_label": "B",
                        "column_label": "2025",
                        "unit": "VND",
                        "source_span": [3, 4],
                    },
                ],
                "supervision": {
                    "company_key": "final-company-aaa",
                    "review_status": "pending_independent_review",
                },
            }
        ]

        exported = EXPORTER.review_rows(rows)

        self.assertEqual(exported[0]["operation"], "ADD")
        self.assertEqual(exported[0]["operand_values"], "10 | 5")
        self.assertEqual(exported[0]["operand_rows"], "A | B")
        self.assertEqual(
            exported[0]["current_review_status"],
            "pending_independent_review",
        )
        self.assertEqual(exported[0]["review_decision"], "")
        self.assertEqual(exported[0]["second_review_decision"], "")
        self.assertEqual(exported[0]["second_reviewer"], "")
        self.assertEqual(exported[0]["adjudication_notes"], "")


if __name__ == "__main__":
    unittest.main()
