from __future__ import annotations

import importlib.util
import unittest
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_reviewed_final_evaluation",
    ROOT / "scripts" / "freeze_reviewed_final_evaluation.py",
)
assert SPEC is not None and SPEC.loader is not None
FREEZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FREEZER)


class FreezeReviewedFinalEvaluationTests(unittest.TestCase):
    def _row(self, operation: str, answer: Decimal):
        candidates = [
            {
                "value": "10",
                "source_name": "AAA.pdf",
                "page": 1,
                "row_label": "A",
                "column_label": "2025",
                "table_id": "table-1",
                "entity_id": "entity-aaa",
                "subtable_id": "table-1/whole-table",
            },
            {
                "value": "2",
                "source_name": "AAA.pdf",
                "page": 1,
                "row_label": "B",
                "column_label": "2025",
                "table_id": "table-1",
                "entity_id": "entity-aaa",
                "subtable_id": "table-1/whole-table",
            },
        ]
        pointers = [0] if operation == "SELECT" else [0, 1]
        gold_operands = [
            {
                "candidate_index": pointer,
                "source_name": candidates[pointer]["source_name"],
                "page": candidates[pointer]["page"],
                "row": candidates[pointer]["row_label"],
                "column": candidates[pointer]["column_label"],
            }
            for pointer in pointers
        ]
        return {
            "example_id": f"example-{operation.lower()}",
            "answer_value": str(answer),
            "program": {"operation": operation, "operands": pointers},
            "candidates": candidates,
            "supervision": {
                "company_key": "company-aaa",
                "gold_operands": gold_operands,
            },
        }

    def _rows(self):
        return [
            self._row("SELECT", Decimal("10")),
            self._row("ADD", Decimal("12")),
            self._row("SUBTRACT", Decimal("8")),
            self._row("DIVIDE", Decimal("5")),
            self._row("RATIO_PERCENT", Decimal("500")),
            self._row("PERCENT_CHANGE", Decimal("400")),
        ]

    def test_freezes_only_resolved_independent_reviews(self) -> None:
        rows = self._rows()
        reviews = [
            {
                "example_id": row["example_id"],
                "review_decision": "approved",
                "reviewer": "human-a",
                "review_notes": "checked against PDF",
            }
            for row in rows
        ]

        frozen, report = FREEZER.freeze_rows(
            rows,
            reviews,
            [],
            expected_examples=6,
            expected_per_operation=1,
            minimum_companies=1,
            minimum_second_review_fraction=0,
        )

        self.assertEqual(len(frozen), 6)
        self.assertEqual(report["companies"], 1)
        self.assertTrue(
            all(
                row["supervision"]["review_status"]
                == "independently_reviewed_approved"
                for row in frozen
            )
        )

    def test_refuses_pre_review_as_independent_gold(self) -> None:
        rows = self._rows()
        reviews = [
            {
                "example_id": row["example_id"],
                "review_decision": "needs_human_pdf_review",
                "reviewer": "codex_layout_pre_review",
            }
            for row in rows
        ]

        with self.assertRaisesRegex(ValueError, "unresolved review decision"):
            FREEZER.freeze_rows(
                rows,
                reviews,
                [],
                expected_examples=6,
                expected_per_operation=1,
                minimum_companies=1,
                minimum_second_review_fraction=0,
            )


if __name__ == "__main__":
    unittest.main()
