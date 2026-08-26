from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rebuild_layout_clean_final_evaluation",
    ROOT / "scripts" / "rebuild_layout_clean_final_evaluation.py",
)
assert SPEC is not None and SPEC.loader is not None
REBUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REBUILD)


def _row(company: str, value: str):
    return {
        "example_id": f"{company}-{value}",
        "program": {"operation": "SELECT", "operands": [0]},
        "candidates": [
            {
                "source_name": f"{company}.pdf",
                "page": 1,
                "raw": value,
                "value": value,
                "row_label": "Doanh thu",
                "column_label": "2025",
                "table_id": "table-1",
                "entity_id": company,
                "subtable_id": "table-1/whole-table",
            }
        ],
        "supervision": {"company_key": company},
    }


class LayoutCleanRebuildTests(unittest.TestCase):
    def test_round_robin_prefers_company_diversity(self) -> None:
        groups = {
            "company-a": [_row("company-a", "1"), _row("company-a", "2")],
            "company-b": [_row("company-b", "3")],
            "company-c": [_row("company-c", "4")],
        }

        selected = REBUILD._round_robin(groups, 3, seed=7)

        self.assertEqual(
            len({row["supervision"]["company_key"] for row in selected}),
            3,
        )

    def test_fingerprint_ignores_example_id(self) -> None:
        first = _row("company-a", "1")
        second = _row("company-a", "1")
        second["example_id"] = "renamed"

        self.assertEqual(REBUILD._fingerprint(first), REBUILD._fingerprint(second))

    def test_semantic_filter_rejects_truncated_financial_rows(self) -> None:
        row = _row("company-a", "1")
        row["candidates"][0]["row_label"] = "Tổng cộng tài"

        reasons = REBUILD._semantic_reasons(row)

        self.assertIn("operand_0:truncated_row", reasons)

    def test_semantic_filter_accepts_complete_financial_row(self) -> None:
        row = _row("company-a", "1")

        self.assertEqual(REBUILD._semantic_reasons(row), [])

    def test_semantic_filter_rejects_implausible_percent_value(self) -> None:
        row = _row("company-a", "476591")
        row["candidates"][0]["unit"] = "%"
        row["candidates"][0]["column_label"] = "%Thay đổi"

        reasons = REBUILD._semantic_reasons(row)

        self.assertIn("operand_0:implausible_percent_value", reasons)

if __name__ == "__main__":
    unittest.main()
