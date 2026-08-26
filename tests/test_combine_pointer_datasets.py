from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "combine_pointer_datasets",
    ROOT / "scripts" / "combine_pointer_datasets.py",
)
assert SPEC is not None and SPEC.loader is not None
COMBINER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMBINER)


def _row(source: str, company: str, value: str) -> dict[str, object]:
    return {
        "example_id": "duplicate-input-id",
        "question": "Giá trị là bao nhiêu?",
        "answer_value": value,
        "program": {"operation": "SELECT", "operands": [0]},
        "candidates": [{"source_name": source, "page": 3}],
        "supervision": {"company_key": company},
    }


class CombinePointerDatasetsTests(unittest.TestCase):
    def test_reassigns_unique_ids_and_deduplicates_provenance(self) -> None:
        first = _row("first.pdf", "company-1", "10")
        second = _row("second.pdf", "company-2", "20")

        combined, duplicates = COMBINER.combine_rows([first, first, second])

        self.assertEqual(len(combined), 2)
        self.assertEqual(duplicates, 1)
        self.assertEqual(
            [row["example_id"] for row in combined],
            [
                "synthetic-document-development-pointer-common-v1-000001",
                "synthetic-document-development-pointer-common-v1-000002",
            ],
        )
        self.assertEqual(
            {row["supervision"]["company_key"] for row in combined},
            {"company-1", "company-2"},
        )

    def test_rejects_missing_company_key(self) -> None:
        row = _row("first.pdf", "company-1", "10")
        row["supervision"].pop("company_key")

        with self.assertRaisesRegex(ValueError, "company_key"):
            COMBINER.combine_rows([row])


if __name__ == "__main__":
    unittest.main()
