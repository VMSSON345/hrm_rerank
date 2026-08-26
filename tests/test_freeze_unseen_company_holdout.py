from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "freeze_unseen_company_holdout",
    ROOT / "scripts" / "freeze_unseen_company_holdout.py",
)
assert SPEC is not None and SPEC.loader is not None
FREEZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FREEZER)


class FreezeUnseenCompanyHoldoutTests(unittest.TestCase):
    def test_builds_full_page_program_example_and_finds_gold_operands(self) -> None:
        row = {
            "example_id": "example-1",
            "classification": "development_only",
            "question_type": "current_ratio",
            "question": "Tính Current Ratio năm 2024.",
            "answer_value": "3",
            "source": {
                "document": "/workspace/report.pdf",
                "sha256": "a" * 64,
                "company_key": "company-a",
                "page": 7,
            },
            "evidence": [
                {
                    "metric": "current_assets",
                    "period": "2024",
                    "value": "600000",
                    "row_label": "Tài sản ngắn hạn",
                },
                {
                    "metric": "current_liabilities",
                    "period": "2024",
                    "value": "200000",
                    "row_label": "Nợ ngắn hạn",
                },
            ],
            "program": {
                "nodes": [
                    {
                        "id": "current_ratio",
                        "operation": "DIVIDE",
                        "inputs": ["current_assets", "current_liabilities"],
                    }
                ]
            },
        }
        context = """Đơn vị tính: đồng
Chỉ tiêu | Năm 2024 | Năm 2023
--- | --- | ---
Tài sản ngắn hạn | Năm 2024: 600.000 | Năm 2023: 550.000
Nợ ngắn hạn | Năm 2024: 200.000 | Năm 2023: 180.000
"""
        candidates = FREEZER.extract_numeric_candidates(
            row["question"], context, source_name="report.pdf", page=7
        )
        pointers, rejection = FREEZER._gold_pointers(row["evidence"], candidates)

        self.assertIsNone(rejection)
        self.assertIsNotNone(pointers)
        self.assertEqual(len(pointers), 2)
        self.assertNotEqual(pointers[0], pointers[1])

    def test_missing_operand_is_counted_as_candidate_recall_failure(self) -> None:
        evidence = [
            {"metric": "total_assets", "period": "2024", "value": "100", "row_label": "Tổng tài sản"}
        ]
        candidates = FREEZER.extract_numeric_candidates(
            "Tổng tài sản năm 2024 là bao nhiêu?",
            "Doanh thu | Năm 2024: 100",
        )

        pointers, rejection = FREEZER._gold_pointers(evidence, candidates)

        self.assertIsNone(pointers)
        self.assertEqual(rejection, "missing_operand:total_assets")


if __name__ == "__main__":
    unittest.main()
