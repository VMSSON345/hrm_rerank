from __future__ import annotations

import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_subtract_development_cohort",
    ROOT / "scripts" / "build_subtract_development_cohort.py",
)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class SubtractDevelopmentCohortTests(unittest.TestCase):
    def _build(self, current: str, prior: str):
        manifest = {
            "classification": "development_only",
            "documents": [
                {
                    "document": "/workspace/NEW_24CN_BCTN.pdf",
                    "sha256": "a" * 64,
                    "company_key": "development-company-new",
                    "pages": [{"page": 1, "selection": "core"}],
                }
            ],
        }
        markdown = f"""## Page 1
Đơn vị tính: VND
| Chỉ tiêu | Năm 2024 | Năm 2023 |
|---|---:|---:|
| Doanh thu thuần | {current} | {prior} |
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "NEW_24CN_BCTN.md").write_text(markdown, encoding="utf-8")
            return BUILDER.build_cohort(manifest, root)

    def test_keeps_only_direction_proven_temporal_subtract(self) -> None:
        examples, report = self._build("1.200.000", "1.000.000")

        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example.program.operation, "SUBTRACT")
        self.assertEqual(example.answer_value, "200000")
        self.assertIn("Hãy tính 2024 trừ 2023", example.question)
        self.assertEqual(
            example.supervision["hard_negatives"][0]["type"],
            "reversed_operand_order",
        )
        self.assertEqual(report["metrics"]["direction_outcomes"], {"direction_supported": 1})
        self.assertEqual(report["metrics"]["integrity_issue_counts"], {})

    def test_rejects_equal_values_where_reversal_is_unobservable(self) -> None:
        examples, report = self._build("1.000.000", "1.000.000")

        self.assertEqual(examples, [])
        self.assertEqual(
            report["metrics"]["rejection_counts"][
                "equal_values_direction_unobservable"
            ],
            1,
        )

    def test_rejects_fraction_to_percent_scale_discontinuity(self) -> None:
        examples, _ = self._build("1.200.000", "1.000.000")
        example = examples[0]
        left, right = (example.candidates[index] for index in example.program.operands)

        self.assertTrue(
            BUILDER._cross_period_scale_discontinuity(
                replace(left, value="72.63", unit="x"),
                replace(right, value="0.71", unit="x"),
            )
        )
        self.assertFalse(
            BUILDER._cross_period_scale_discontinuity(
                replace(left, value="34.55", unit="vòng"),
                replace(right, value="82.45", unit="vòng"),
            )
        )

    def test_rejects_same_row_from_multiple_table_provenances(self) -> None:
        manifest = {
            "classification": "development_only",
            "documents": [
                {
                    "document": "/workspace/NEW_24CN_BCTN.pdf",
                    "sha256": "a" * 64,
                    "company_key": "development-company-new",
                    "pages": [{"page": 1, "selection": "core"}],
                }
            ],
        }
        markdown = """## Page 1
Đơn vị tính: VND
| Chỉ tiêu | Năm 2024 | Năm 2023 |
|---|---:|---:|
| Tổng tài sản | 1.200.000 | 1.000.000 |

| Chỉ tiêu công ty liên kết | Năm 2024 | Năm 2023 |
|---|---:|---:|
| Tổng tài sản | 900.000 | 800.000 |
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "NEW_24CN_BCTN.md").write_text(markdown, encoding="utf-8")
            examples, report = BUILDER.build_cohort(manifest, root)

        self.assertEqual(examples, [])
        self.assertGreater(
            report["metrics"]["rejection_counts"]["ambiguous_row_provenance"],
            0,
        )

    def test_rejects_value_that_precedes_its_assigned_row_in_merged_table(self) -> None:
        examples, _ = self._build("1.200.000", "1.000.000")
        candidate = examples[0].candidates[examples[0].program.operands[0]]
        merged = replace(
            candidate,
            row_label="Chi phí cho nhân viên",
            context=(
                "ROW: Chi phí cho nhân viên\n"
                "COLUMN: Năm 2024\nPERIOD: 2024\nTABLE: table-1\n"
                "SUBTABLE: table-1/whole-table\n"
                "HEADER: | Chỉ tiêu bảng trái | 2024 | 2023 | Chỉ tiêu bảng phải |\n"
                "CELL: | TỔNG CỘNG | <VALUE> 46.578.812 </VALUE> | "
                "27.601.196 | Chi phí cho nhân viên | (759.118) | (666.384) |"
            ),
        )

        self.assertFalse(BUILDER._cell_value_follows_row_label(merged))
        self.assertFalse(BUILDER._complete_table_candidate(merged))

    def test_rejects_amount_assigned_to_ratio_row(self) -> None:
        examples, _ = self._build("1.200.000", "1.000.000")
        candidate = examples[0].candidates[examples[0].program.operands[0]]
        ratio = replace(
            candidate,
            row_label="Tỷ suất sinh lợi trên vốn chủ sở hữu",
            value="7929950232",
        )

        self.assertFalse(BUILDER._ratio_value_scale_is_plausible(ratio))

    def test_display_period_keeps_only_unambiguous_year(self) -> None:
        examples, _ = self._build("1.200.000", "1.000.000")
        candidate = examples[0].candidates[examples[0].program.operands[0]]

        self.assertEqual(
            BUILDER._display_period(
                replace(candidate, period="Năm 2024 (triệu đồng)")
            ),
            "2024",
        )

    def test_rejects_shifted_disclosure_note_as_period_value(self) -> None:
        examples, _ = self._build("1.200.000", "1.000.000")
        candidate = examples[0].candidates[examples[0].program.operands[0]]
        shifted = replace(
            candidate,
            raw="34.3",
            value="34.3",
            context=(
                "ROW: Lãi cho vay và phí phải thu chưa thu được\n"
                "COLUMN: Năm 2024\nPERIOD: 2024\nTABLE: table-1\n"
                "SUBTABLE: table-1/whole-table\nHEADER: | Chỉ tiêu | 2024 | 2023 |\n"
                "CELL: | Lãi cho vay và phí phải thu chưa thu được | "
                "<VALUE> 34.3 </VALUE> | 930.391 | 740.310 |"
            ),
        )

        self.assertTrue(BUILDER._is_shifted_note_reference(shifted))
        self.assertFalse(BUILDER._complete_table_candidate(shifted))

    def test_rejects_amount_with_ratio_column_metadata(self) -> None:
        examples, _ = self._build("1.200.000", "1.000.000")
        candidate = examples[0].candidates[examples[0].program.operands[0]]
        shifted = replace(
            candidate,
            column_label="Tỷ trọng 2024",
            unit="VND",
            is_percent=False,
        )

        self.assertFalse(BUILDER._column_unit_contract_is_plausible(shifted))


if __name__ == "__main__":
    unittest.main()
