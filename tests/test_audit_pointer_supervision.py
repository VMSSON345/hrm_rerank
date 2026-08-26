from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from hrm_finance_qa.program_data import NumericCandidate, ReasoningProgram


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_document_derived_programs",
    ROOT / "scripts" / "audit_document_derived_programs.py",
)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


def _candidate(index: int, row: str, period: str, value: str) -> NumericCandidate:
    return NumericCandidate(
        index=index,
        value=value,
        raw=value,
        context=(
            f"ROW: {row}\nROW_TYPE: detail\nCOLUMN: {period}\nPERIOD: {period}\n"
            f"HEADER: test\nCELL: <VALUE>{value}</VALUE>"
        ),
        line_number=index + 1,
        char_start=0,
        char_end=len(value),
        unit="VND",
        is_percent=False,
        is_year_like=False,
        relevance=1.0,
        source_name="company-a.pdf",
        page=7,
        extraction_method="native",
        row_label=row,
        column_label=period,
        period=period,
        row_type="detail",
        source_span=(index * 10, index * 10 + len(value)),
    )


class PointerSupervisionAuditTests(unittest.TestCase):
    def _row(self) -> tuple[dict[str, object], tuple[NumericCandidate, ...], ReasoningProgram]:
        candidates = (
            _candidate(0, "Doanh thu thuần", "2024", "12000"),
            _candidate(1, "Lợi nhuận sau thuế", "2024", "3000"),
            _candidate(2, "Doanh thu thuần", "2023", "10000"),
            _candidate(3, "Lợi nhuận sau thuế", "2023", "2000"),
        )
        program = ReasoningProgram("RATIO_PERCENT", (0, 1))
        row = {
            "program": {"operation": program.operation, "operands": list(program.operands)},
            "supervision": {
                "schema": "structured-operand-supervision-v1",
                "company_key": "company-a",
                "gold_operands": [
                    {
                        "candidate_index": 0,
                        "role": "numerator",
                        "source_name": "company-a.pdf",
                        "page": 7,
                        "row": "Doanh thu thuần",
                        "column": "2024",
                        "period": "2024",
                        "unit": "VND",
                        "unit_status": "explicit",
                        "source_span": [0, 5],
                    },
                    {
                        "candidate_index": 1,
                        "role": "denominator",
                        "source_name": "company-a.pdf",
                        "page": 7,
                        "row": "Lợi nhuận sau thuế",
                        "column": "2024",
                        "period": "2024",
                        "unit": "VND",
                        "unit_status": "explicit",
                        "source_span": [10, 14],
                    },
                ],
                "hard_negatives": [
                    {"type": "same_row_wrong_period", "operand_slot": 0, "candidate_indices": [2]},
                    {"type": "same_period_wrong_row", "operand_slot": 0, "candidate_indices": [1]},
                    {"type": "same_row_wrong_period", "operand_slot": 1, "candidate_indices": [3]},
                    {"type": "same_period_wrong_row", "operand_slot": 1, "candidate_indices": [0]},
                    {"type": "reversed_operand_order", "operand_slot": None, "candidate_indices": [1, 0]},
                ],
            },
        }
        return row, candidates, program

    def test_accepts_complete_common_pointer_supervision(self) -> None:
        row, candidates, program = self._row()

        self.assertIsNone(
            AUDITOR._validate_pointer_common_supervision(row, candidates, program)
        )

    def test_rejects_missing_page_period_and_role(self) -> None:
        row, candidates, program = self._row()
        row["supervision"]["gold_operands"][0]["page"] = None
        self.assertEqual(
            AUDITOR._validate_pointer_common_supervision(row, candidates, program),
            "missing_gold_page",
        )

        row, candidates, program = self._row()
        row["supervision"]["gold_operands"][0]["period"] = ""
        self.assertEqual(
            AUDITOR._validate_pointer_common_supervision(row, candidates, program),
            "missing_gold_period",
        )

        row, candidates, program = self._row()
        row["supervision"]["gold_operands"][0]["role"] = "denominator"
        self.assertEqual(
            AUDITOR._validate_pointer_common_supervision(row, candidates, program),
            "gold_role_mismatch",
        )

    def test_rejects_missing_slot_specific_hard_negative(self) -> None:
        row, candidates, program = self._row()
        row["supervision"]["hard_negatives"] = [
            item
            for item in row["supervision"]["hard_negatives"]
            if not (
                item["type"] == "same_row_wrong_period"
                and item["operand_slot"] == 1
            )
        ]

        self.assertEqual(
            AUDITOR._validate_pointer_common_supervision(row, candidates, program),
            "missing_same_row_wrong_period",
        )

    def test_rejects_gold_source_span_that_does_not_match_candidate(self) -> None:
        row, candidates, program = self._row()
        row["supervision"]["gold_operands"][0]["source_span"] = [1, 6]

        self.assertEqual(
            AUDITOR._validate_pointer_common_supervision(row, candidates, program),
            "gold_candidate_source_span_mismatch",
        )

    def test_semantic_profile_rejects_comparison_column_as_period(self) -> None:
        row, candidates, program = self._row()
        for gold in row["supervision"]["gold_operands"]:
            gold["period"] = "2024 2023"
        self.assertEqual(
            AUDITOR._semantic_pointer_error(row),
            "semantic_comparison_column_not_period",
        )

    def test_semantic_profile_rejects_derived_comparison_column(self) -> None:
        row, _, _ = self._row()
        row["supervision"]["gold_operands"][0]["column"] = (
            "Thay doi so voi cung ky nam truoc"
        )
        self.assertEqual(
            AUDITOR._semantic_pointer_error(row),
            "semantic_derived_comparison_column",
        )

    def test_semantic_profile_rejects_narrative_and_non_metric_rows(self) -> None:
        row, _, _ = self._row()
        row["supervision"]["gold_operands"][0]["row"] = (
            "Trong nam Cong ty da thanh toan cac khoan no va tang doanh thu "
            "dang ke so voi cung ky"
        )
        self.assertEqual(
            AUDITOR._semantic_pointer_error(row),
            "semantic_narrative_row",
        )

        row, _, _ = self._row()
        row["supervision"]["gold_operands"][0]["row"] = "So 365 duong Tran"
        self.assertEqual(
            AUDITOR._semantic_pointer_error(row),
            "semantic_unrecognized_financial_row",
        )

    def test_semantic_profile_accepts_metric_rows_and_real_period(self) -> None:
        row, _, _ = self._row()
        row["supervision"]["gold_operands"][0]["row"] = "Tong tai san"
        row["supervision"]["gold_operands"][1]["row"] = "Tai san ngan han"
        self.assertIsNone(AUDITOR._semantic_pointer_error(row))


if __name__ == "__main__":
    unittest.main()
