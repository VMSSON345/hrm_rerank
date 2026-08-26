from __future__ import annotations

from decimal import Decimal
import unittest

from hrm_finance_qa.document_hrm import _grounding_gate_reasons
from hrm_finance_qa.evidence_schema import build_evidence_graph
from hrm_finance_qa.program_data import (
    NumericCandidate,
    ReasoningProgram,
    extract_numeric_candidates,
)
from hrm_finance_qa.program_synthesis import synthesize_typed_programs
from hrm_finance_qa.question_planner import (
    compile_question_plan,
    compile_question_proof_contract,
)
from hrm_finance_qa.symbolic_verifier import VerificationState, verify_program


def _flattened_plan_actual_candidates() -> tuple[NumericCandidate, ...]:
    context = "\n".join(
        (
            "STT Chỉ tiêu",
            "Thực hiện Kế hoạch Thực hiện % tăng giảm",
            "TH2023/",
            "KH 2023",
            "% tăng giảm",
            "TH2023/",
            "TH 2022 Năm 2022 Năm 2023 Năm 2023",
            (
                "2 Doanh thu hoạt động 363,638,445,420 "
                "472,723,394,169 304,912,639,828 -35.50% -16.15%"
            ),
        )
    )
    return extract_numeric_candidates(
        "Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?",
        context,
        extraction_method="native",
    )


def _candidate(
    *, column: str, header: str, extraction_method: str = "native"
) -> NumericCandidate:
    return NumericCandidate(
        index=0,
        value="100",
        raw="100",
        context="\n".join(
            (
                "ROW: Doanh thu hoạt động",
                f"COLUMN: {column}",
                "PERIOD: 2023",
                f"HEADER: | Chỉ tiêu | {header} |",
                "CELL: | Doanh thu hoạt động | <VALUE> 100 </VALUE> |",
            )
        ),
        line_number=1,
        char_start=0,
        char_end=3,
        unit="VND",
        is_percent=False,
        is_year_like=False,
        relevance=1.0,
        source_name="development.pdf",
        page=1,
        extraction_method=extraction_method,
    )


class ColumnRoleRuntimeQP2Tests(unittest.TestCase):
    def test_compiler_emits_explicit_select_column_roles(self) -> None:
        actual = compile_question_plan(
            "Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?"
        )
        plan = compile_question_plan(
            "Doanh thu hoạt động kế hoạch năm 2023 là bao nhiêu?"
        )
        revised = compile_question_plan(
            "Kế hoạch điều chỉnh doanh thu năm 2024 là bao nhiêu?"
        )

        self.assertEqual((actual.operation, actual.column_roles), ("SELECT", ("ACTUAL",)))
        self.assertEqual((plan.operation, plan.column_roles), ("SELECT", ("PLAN",)))
        self.assertEqual(
            (revised.operation, revised.column_roles),
            ("SELECT", ("REVISED_PLAN",)),
        )
        self.assertEqual(actual.metric_text, "doanh thu hoat dong")

    def test_compiler_emits_actual_over_plan_completion_ratio(self) -> None:
        plan = compile_question_plan(
            "Doanh thu hoạt động thực hiện năm 2023 hoàn thành bao nhiêu "
            "phần trăm kế hoạch năm 2023?"
        )

        self.assertEqual(plan.operation, "RATIO_PERCENT")
        self.assertEqual(plan.column_roles, ("ACTUAL", "PLAN"))
        self.assertEqual(plan.requested_years, ("2023",))

    def test_typed_select_only_materializes_the_requested_role(self) -> None:
        candidates = _flattened_plan_actual_candidates()
        graph = build_evidence_graph(candidates)
        values = {candidate.index: candidate.decimal for candidate in candidates}

        actual_plan = compile_question_plan(
            "Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?"
        )
        actual = synthesize_typed_programs(
            actual_plan,
            "SELECT",
            graph,
            focus_text=actual_plan.metric_text,
        )
        planned_plan = compile_question_plan(
            "Doanh thu hoạt động kế hoạch năm 2023 là bao nhiêu?"
        )
        planned = synthesize_typed_programs(
            planned_plan,
            "SELECT",
            graph,
            focus_text=planned_plan.metric_text,
        )

        self.assertTrue(actual.programs)
        self.assertTrue(planned.programs)
        self.assertEqual(
            {values[item.operands[0]] for item in actual.programs},
            {Decimal("304912639828")},
        )
        self.assertEqual(
            {values[item.operands[0]] for item in planned.programs},
            {Decimal("472723394169")},
        )

    def test_typed_completion_ratio_uses_actual_then_plan(self) -> None:
        candidates = _flattened_plan_actual_candidates()
        graph = build_evidence_graph(candidates)
        values = {candidate.index: candidate.decimal for candidate in candidates}
        plan = compile_question_plan(
            "Doanh thu hoạt động thực hiện năm 2023 hoàn thành bao nhiêu "
            "phần trăm kế hoạch năm 2023?"
        )
        result = synthesize_typed_programs(
            plan,
            "RATIO_PERCENT",
            graph,
            focus_text=plan.metric_text,
        )

        self.assertTrue(result.programs)
        for program in result.programs:
            self.assertEqual(
                tuple(values[index] for index in program.operands),
                (Decimal("304912639828"), Decimal("472723394169")),
            )

    def test_column_role_is_a_required_proof_obligation(self) -> None:
        plan = compile_question_plan(
            "Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?"
        )
        contract = compile_question_proof_contract(plan, "SELECT")

        self.assertIn("column_role_match", contract.required)

    def test_verifier_and_runtime_gate_distinguish_valid_unknown_and_invalid(self) -> None:
        question = "Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?"
        plan = compile_question_plan(question)
        cases = (
            (
                _candidate(column="Thực hiện 2023", header="Thực hiện 2023"),
                VerificationState.VALID,
                None,
            ),
            (
                _candidate(column="2023", header="2023"),
                VerificationState.UNKNOWN,
                "missing_requested_column_role",
            ),
            (
                _candidate(column="Kế hoạch 2023", header="Kế hoạch 2023"),
                VerificationState.INVALID,
                "column_role_mismatch",
            ),
            (
                _candidate(column="Kế hoạch 2023", header="Thực hiện 2023"),
                VerificationState.INVALID,
                "column_role_conflict",
            ),
        )
        for candidate, expected_state, expected_reason in cases:
            with self.subTest(column=candidate.column_label, reason=expected_reason):
                graph = build_evidence_graph((candidate,))
                verification = verify_program(
                    plan, ReasoningProgram("SELECT", (0,)), graph
                )
                obligations = {
                    item.name: item.state
                    for item in verification.proof_obligations
                }
                reasons = _grounding_gate_reasons(
                    question,
                    ReasoningProgram("SELECT", (0,)),
                    (candidate,),
                    verification,
                )
                self.assertEqual(
                    obligations["column_role_match"], expected_state
                )
                if expected_reason is None:
                    self.assertNotIn("column_role_mismatch", reasons)
                    self.assertNotIn("missing_requested_column_role", reasons)
                    self.assertNotIn("column_role_conflict", reasons)
                else:
                    self.assertIn(expected_reason, reasons)

    def test_qp3b_unknown_role_inventory_forces_safe_abstention(self) -> None:
        question = "Doanh thu hoáº¡t Ä‘á»™ng nÄƒm 2023 lÃ  bao nhiÃªu?"
        candidate = _candidate(
            column="Thá»±c hiá»‡n 2023",
            header="Thá»±c hiá»‡n 2023",
        )
        graph = build_evidence_graph(
            (candidate,), column_role_inventory_state="UNKNOWN"
        )
        plan = compile_question_plan(question)
        program = ReasoningProgram("SELECT", (0,))
        verification = verify_program(plan, program, graph)
        obligations = {
            item.name: (item.applicability, item.state)
            for item in verification.proof_obligations
        }
        reasons = _grounding_gate_reasons(
            question, program, (candidate,), verification
        )

        self.assertEqual(
            obligations["column_role_unique"],
            ("required", VerificationState.UNKNOWN),
        )
        self.assertIn("missing_column_role_inventory", reasons)


if __name__ == "__main__":
    unittest.main()
