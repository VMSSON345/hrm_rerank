from __future__ import annotations

import unittest
from decimal import Decimal

from hrm_finance_qa.direction_contract import assess_program_direction
from hrm_finance_qa.evidence_schema import EvidenceCell, EvidenceGraph
from hrm_finance_qa.program_data import ReasoningProgram
from hrm_finance_qa.question_planner import compile_question_plan
from hrm_finance_qa.symbolic_verifier import VerificationState, verify_program


def _cell(index: int, *, year: str, metric: str = "Doanh thu") -> EvidenceCell:
    return EvidenceCell(
        index=index,
        value=Decimal(index + 1),
        metric_label=metric,
        metric_key=metric.lower().replace(" ", ""),
        column=year,
        period=year,
        explicit_years=(year,),
        period_role="",
        entity="",
        section="",
        table_id="table-1",
        source_name="development.pdf",
        page=1,
        value_status="observed",
    )


class DirectionContractTests(unittest.TestCase):
    def test_new_minus_old_is_stable_under_candidate_permutation(self) -> None:
        plan = compile_question_plan(
            "Doanh thu thay đổi từ năm 2024 đến năm 2025 bao nhiêu?"
        )
        graph = EvidenceGraph(
            cells=(_cell(0, year="2024"), _cell(1, year="2025")),
            edges=(),
        )
        correct = assess_program_direction(
            plan, ReasoningProgram("SUBTRACT", (1, 0)), graph
        )
        reversed_result = assess_program_direction(
            plan, ReasoningProgram("SUBTRACT", (0, 1)), graph
        )

        self.assertEqual(correct.outcome, "direction_supported")
        self.assertEqual(reversed_result.outcome, "direction_reversed")

    def test_decrease_amount_requires_old_minus_new(self) -> None:
        plan = compile_question_plan(
            "Doanh thu năm 2025 thấp hơn năm 2024 bao nhiêu?"
        )
        graph = EvidenceGraph(
            cells=(_cell(0, year="2025"), _cell(1, year="2024")),
            edges=(),
        )

        result = assess_program_direction(
            plan, ReasoningProgram("SUBTRACT", (1, 0)), graph
        )

        self.assertEqual(result.outcome, "direction_supported")

    def test_left_minus_right_uses_metric_roles_not_values(self) -> None:
        plan = compile_question_plan(
            "Tính Tổng tài sản trừ Tổng nguồn vốn năm 2025 là bao nhiêu?"
        )
        graph = EvidenceGraph(
            cells=(
                _cell(0, year="2025", metric="Tổng nguồn vốn"),
                _cell(1, year="2025", metric="Tổng tài sản"),
            ),
            edges=(),
        )

        result = assess_program_direction(
            plan, ReasoningProgram("SUBTRACT", (1, 0)), graph
        )

        self.assertEqual(result.outcome, "direction_supported")

    def test_same_period_decrease_uses_reversed_named_roles(self) -> None:
        plan = compile_question_plan(
            "Chi phí tài chính thấp hơn Doanh thu tài chính bao nhiêu?"
        )
        graph = EvidenceGraph(
            cells=(
                _cell(0, year="2025", metric="Chi phí tài chính"),
                _cell(1, year="2025", metric="Doanh thu tài chính"),
            ),
            edges=(),
        )

        result = assess_program_direction(
            plan, ReasoningProgram("SUBTRACT", (1, 0)), graph
        )

        self.assertEqual(result.outcome, "direction_supported")

    def test_ambiguous_gap_is_not_turned_into_a_hard_negative(self) -> None:
        plan = compile_question_plan(
            "Chênh lệch giữa Doanh thu và Chi phí là bao nhiêu?"
        )
        graph = EvidenceGraph(
            cells=(
                _cell(0, year="2025", metric="Doanh thu"),
                _cell(1, year="2025", metric="Chi phí"),
            ),
            edges=(),
        )

        result = assess_program_direction(
            plan, ReasoningProgram("SUBTRACT", (0, 1)), graph
        )

        self.assertEqual(result.outcome, "ambiguous_wording")

    def test_verifier_rejects_ambiguous_signed_subtract(self) -> None:
        plan = compile_question_plan(
            "Chênh lệch giữa Doanh thu và Chi phí là bao nhiêu?"
        )
        graph = EvidenceGraph(
            cells=(
                _cell(0, year="2025", metric="Doanh thu"),
                _cell(1, year="2025", metric="Chi phí"),
            ),
            edges=(),
        )

        result = verify_program(plan, ReasoningProgram("SUBTRACT", (0, 1)), graph)

        self.assertFalse(result.accepted)
        self.assertIn("temporal_direction", result.reasons)

    def test_verifier_keeps_missing_period_as_unknown(self) -> None:
        plan = compile_question_plan(
            "Doanh thu cuối kỳ so với đầu kỳ chênh lệch bao nhiêu?"
        )
        graph = EvidenceGraph(
            cells=(_cell(0, year=""), _cell(1, year="")),
            edges=(),
        )

        result = verify_program(plan, ReasoningProgram("SUBTRACT", (0, 1)), graph)
        states = {item.name: item.state for item in result.constraints}

        self.assertEqual(states["temporal_direction"], VerificationState.UNKNOWN)
        self.assertTrue(result.accepted)

    def test_verifier_rejects_absolute_gap_as_plain_subtract(self) -> None:
        plan = compile_question_plan(
            "Độ lệch tuyệt đối giữa Doanh thu và Chi phí là bao nhiêu?"
        )
        graph = EvidenceGraph(
            cells=(
                _cell(0, year="2025", metric="Doanh thu"),
                _cell(1, year="2025", metric="Chi phí"),
            ),
            edges=(),
        )

        result = verify_program(plan, ReasoningProgram("SUBTRACT", (0, 1)), graph)

        self.assertFalse(result.accepted)


if __name__ == "__main__":
    unittest.main()
