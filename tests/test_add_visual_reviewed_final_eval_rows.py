from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "add_visual_reviewed_final_eval_rows",
    ROOT / "scripts" / "add_visual_reviewed_final_eval_rows.py",
)
assert SPEC is not None and SPEC.loader is not None
VISUAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VISUAL)


def _row(example_id: str, company: str, source_name: str):
    candidate = {
        "source_name": source_name,
        "page": 1,
        "row_label": "Doanh thu",
        "column_label": "2025",
        "period": "2025",
        "unit": "VND",
        "value": "10",
        "context": "ROW: Doanh thu\nCOLUMN: 2025\nPERIOD: 2025",
    }
    return {
        "example_id": example_id,
        "question": "Doanh thu?",
        "answer_text": "10",
        "answer_value": "10",
        "program": {"operation": "SELECT", "operands": [0]},
        "candidates": [candidate],
        "supervision": {
            "company_key": company,
            "gold_operands": [
                {
                    "candidate_index": 0,
                    "source_name": source_name,
                    "page": 1,
                    "row": "Doanh thu",
                    "column": "2025",
                    "period": "2025",
                    "unit": "VND",
                }
            ],
        },
    }


class VisualReviewedFinalEvalTests(unittest.TestCase):
    def test_adds_new_company_without_changing_size_or_operation(self) -> None:
        base = [
            _row("base-1", "company-a", "A.pdf"),
            _row("base-2", "company-a", "A.pdf"),
        ]
        source = [_row("source-1", "company-b", "B.pdf")]
        corrections = [
            {
                "source_example_id": "source-1",
                "question": "Doanh thu năm 2024?",
                "answer_text": "10",
                "operand_patches": [
                    {
                        "column_label": "2024",
                        "period": "2024",
                        "unit": "VND",
                    }
                ],
                "review_notes": "Checked on PDF.",
            }
        ]

        output, report = VISUAL.add_corrections(
            base, source, corrections, seed=7
        )

        self.assertEqual(len(output), 2)
        self.assertEqual(report["companies"], 2)
        corrected = next(
            row for row in output if row["supervision"]["company_key"] == "company-b"
        )
        self.assertEqual(corrected["candidates"][0]["column_label"], "2024")
        self.assertEqual(
            corrected["supervision"]["review_status"],
            "pending_independent_review",
        )


if __name__ == "__main__":
    unittest.main()
