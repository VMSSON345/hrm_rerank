from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_unseen_financial_qa_candidates",
    ROOT / "scripts" / "build_unseen_financial_qa_candidates.py",
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class UnseenFinancialQaCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {
            "document": "/workspace/report.pdf",
            "sha256": "a" * 64,
            "company_key": "company-a",
        }

    def test_generates_four_target_question_families_with_dag_provenance(self) -> None:
        text = """Đơn vị tính: đồng
Chỉ tiêu | Năm 2024 | Năm 2023
--- | --- | ---
Tài sản ngắn hạn | Năm 2024: 600.000 | Năm 2023: 550.000
Tổng tài sản | Năm 2024: 1.000.000 | Năm 2023: 900.000
Nợ ngắn hạn | Năm 2024: 200.000 | Năm 2023: 180.000
Nợ dài hạn | Năm 2024: 100.000 | Năm 2023: 90.000
Nợ phải trả | Năm 2024: 300.000 | Năm 2023: 270.000
Vay và nợ thuê tài chính ngắn hạn | Năm 2024: 80.000 | Năm 2023: 70.000
Vay và nợ thuê tài chính dài hạn | Năm 2024: 20.000 | Năm 2023: 15.000
Tổng nguồn vốn | Năm 2024: 1.000.000 | Năm 2023: 900.000
"""
        records, rejections = BUILDER.build_page_records(self.source, 7, text)
        by_type = {
            item["question_type"]: item for item in records if item["period"] == "2024"
        }

        self.assertEqual(
            set(by_type),
            {
                "total_liabilities_components",
                "accounting_equation_check",
                "current_ratio",
                "interest_bearing_debt_share",
            },
        )
        self.assertEqual(by_type["total_liabilities_components"]["answer_value"], "300000")
        self.assertEqual(by_type["accounting_equation_check"]["answer_value"], "0")
        self.assertEqual(by_type["current_ratio"]["answer_value"], "3")
        self.assertEqual(by_type["interest_bearing_debt_share"]["answer_value"], "10")
        self.assertEqual(
            len(by_type["interest_bearing_debt_share"]["program"]["nodes"]), 2
        )
        self.assertEqual(rejections, {})
        audit = BUILDER.audit_records(records)
        self.assertEqual(audit["records_passed"], len(records))
        self.assertEqual(audit["records_failed"], 0)

    def test_rejects_repeated_metric_scope_instead_of_guessing(self) -> None:
        text = """Đơn vị tính: đồng
Chỉ tiêu | Năm 2024
--- | ---
Tài sản ngắn hạn | Năm 2024: 600.000
Tài sản ngắn hạn | Năm 2024: 300.000
Nợ ngắn hạn | Năm 2024: 200.000
"""
        records, rejections = BUILDER.build_page_records(self.source, 8, text)

        self.assertEqual(records, [])
        self.assertGreater(rejections["ambiguous_metric_period"], 0)

    def test_formula_ratio_row_is_not_mistaken_for_raw_current_assets(self) -> None:
        text = """Chỉ tiêu | Năm 2024 | Năm 2023
--- | --- | ---
Tài sản ngắn hạn/Nợ ngắn hạn | Năm 2024: 1,2 | Năm 2023: 1,1
"""
        evidence = BUILDER.extract_page_evidence(
            text, source_name="report.pdf", page=9
        )

        self.assertEqual(evidence, [])

    def test_uses_year_header_for_value_cell_without_repeated_period_label(self) -> None:
        text = """Đơn vị tính: đồng
Chỉ tiêu | 2023 | Tỷ trọng | 2024 | Tỷ trọng | Thay đổi
--- | --- | --- | --- | --- | ---
Nợ ngắn hạn | 2023: 297.231.519 | 1,46% | 2024: 226.997.898 | 1,11% | -23,63%
Nợ dài hạn | 2023: 26.000.000 | 0,13% | 0 | 0,00% | -100,00%
"""
        records, _ = BUILDER.build_page_records(self.source, 13, text)
        latest = [
            item
            for item in records
            if item["period"] == "2024"
            and item["question_type"] == "total_liabilities_components"
        ]

        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["answer_value"], "226997898")
        self.assertEqual(latest[0]["evidence"][1]["value"], "0")


if __name__ == "__main__":
    unittest.main()
