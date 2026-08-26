from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.build_v12c_interest_bearing_debt_share_cohort import (
    _page_sections,
    _periods,
    _refuse_protected,
    _role_rows,
    discover,
)


class V12CInterestBearingDebtShareCohortTests(unittest.TestCase):
    def test_periods_ignore_regulation_year_and_preserve_opening_date(self) -> None:
        text = """## Page 4
Ban hành theo Thông tư số 202/2014/TT-BTC
Tại ngày 31 tháng 12 năm 2023 (tiếp theo)
| NGUỒN VỐN | Mã số | Thuyết minh | Số cuối năm | Số đầu năm |
"""
        self.assertEqual(
            _periods(_page_sections(text)[4]),
            ("31/12/2023", "01/01/2023"),
        )

    def test_protected_paths_include_validation(self) -> None:
        for name in (
            "validation.jsonl",
            "blind.jsonl",
            "final-data.json",
            "holdout.json",
            "test.jsonl",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                _refuse_protected(Path(name))

    def test_extracts_three_unique_same_period_roles(self) -> None:
        text = """## Page 12
| Mã | Chỉ tiêu | 2025 | 2024 |
Đơn vị: VND
| 320 | Vay ngắn hạn | 80.000 | 70.000 |
| 338 | Vay dài hạn | 20.000 | 30.000 |
| 440 | TỔNG NGUỒN VỐN | 1.000.000 | 900.000 |
"""
        lines = _page_sections(text)[12]
        roles, issues = _role_rows(lines, ("2025", "2024"))
        self.assertFalse(issues)
        self.assertEqual(set(roles), {
            "short_term_borrowings", "long_term_borrowings", "total_capital"
        })
        self.assertEqual(roles["short_term_borrowings"]["values"][0]["value"], "80000")

    def test_duplicate_role_is_ambiguous(self) -> None:
        text = """## Page 1
| Mã | Chỉ tiêu | 2025 |
Đơn vị: VND
| 320 | Vay ngắn hạn | 80.000 |
| 320 | Vay ngắn hạn khác | 10.000 |
| 338 | Vay dài hạn | 20.000 |
| 440 | Tổng nguồn vốn | 1.000.000 |
"""
        roles, issues = _role_rows(_page_sections(text)[1], ("2025",))
        self.assertNotIn("short_term_borrowings", roles)
        self.assertEqual(issues["short_term_borrowings:ambiguous"], 1)

    def test_note_references_are_not_amounts(self) -> None:
        text = """## Page 1
| Mã | Chỉ tiêu | 2024 | 2023 |
Đơn vị: VND
| 320 | Vay ngắn hạn | 19 | 528.523.819.180 | 564.844.530.548 |
| 338 | Vay dài hạn | 19 | 12.094.821.430 | 21.770.678.566 |
| 440 | Tổng nguồn vốn | 2.841.418.344.920 | 2.737.637.322.267 |
"""
        roles, issues = _role_rows(_page_sections(text)[1], ("2024", "2023"))
        self.assertFalse(issues)
        self.assertEqual(
            roles["short_term_borrowings"]["values"][0]["value"],
            "528523819180",
        )

    def test_adjacent_statement_does_not_move_role_amount_window(self) -> None:
        text = """## Page 1
| Chỉ tiêu | Mã số | 2023 | 2022 |
Đơn vị: VND
| 7. | Vay và nợ thuê tài chính ngắn hạn | 320 | 19 | 845.805.808.955 | 2.559.405.532 | - Trong đó: Chi phí lãi vay | 23 | 10.155.714.106 | 10.274.784.277 |
| 1. | Vay và nợ thuê tài chính dài hạn | 338 | 22 | 2.964.479.875 | 1.048.481.500 |
| Tổng cộng nguồn vốn | 440 | 15.237.966.427.299 | 14.166.860.720.016 |
"""
        roles, issues = _role_rows(_page_sections(text)[1], ("2023", "2022"))
        self.assertFalse(issues)
        self.assertEqual(
            [item["value"] for item in roles["short_term_borrowings"]["values"]],
            ["845805808955", "2559405532"],
        )

    def test_discovery_is_deterministic_and_pending_visual_qa(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = root / "DEV_25.pdf"
            pdf.write_bytes(b"locked-pdf")
            markdown = root / "markdown"
            markdown.mkdir()
            (markdown / "DEV_25.md").write_text(
                """## Page 2
| Mã | Chỉ tiêu | 2025 | 2024 |
Đơn vị: VND
| 320 | Vay ngắn hạn | 80.000 | 70.000 |
| 338 | Vay dài hạn | 20.000 | 30.000 |
| 440 | Tổng nguồn vốn | 1.000.000 | 900.000 |
""",
                encoding="utf-8",
            )
            import hashlib
            manifest = root / "development-pages.json"
            manifest.write_text(
                json.dumps({
                    "classification": "development_only",
                    "blind_test_used": False,
                    "documents": [{
                        "document": str(pdf),
                        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                        "company_key": "development-company-a",
                    }],
                }),
                encoding="utf-8",
            )
            first, audit = discover([(manifest, markdown)])
            second, _ = discover([(manifest, markdown)])
            self.assertEqual(first, second)
            self.assertEqual(len(first), 2)
            self.assertTrue(all(not row["eligible_for_positive"] for row in first))
            self.assertEqual(audit["summary"]["positive_records"], 0)
            self.assertFalse(audit["summary"]["ready_for_frozen_evaluation"])


if __name__ == "__main__":
    unittest.main()
