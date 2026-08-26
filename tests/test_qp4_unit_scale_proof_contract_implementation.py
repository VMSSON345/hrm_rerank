from __future__ import annotations

from decimal import Decimal
import unittest

from hrm_finance_qa.evidence_schema import EvidenceCell
from hrm_finance_qa.program_data import ReasoningProgram
from hrm_finance_qa.question_planner import (
    compile_question_plan,
    compile_question_proof_contract,
)
from hrm_finance_qa.symbolic_verifier import VerificationState, verify_program


def _cell(
    index: int,
    *,
    metric: str,
    value: str,
    unit: str,
    unit_key: str,
) -> EvidenceCell:
    return EvidenceCell(
        index=index,
        value=Decimal(value),
        metric_label=metric,
        metric_key="".join(metric.lower().split()),
        column="",
        period="",
        explicit_years=(),
        period_role="",
        entity="reporting",
        section="",
        table_id="table-1",
        source_name="development-report.pdf",
        page=1,
        value_status="observed",
        unit=unit,
        unit_key=unit_key,
        row_id=f"row-{index}",
        column_id="value",
        source_span=(index * 10, index * 10 + 5),
        extraction_method="native",
    )


def _obligation(result, name: str):
    return next(item for item in result.proof_obligations if item.name == name)


class QP4UnitScaleProofContractImplementationTests(unittest.TestCase):
    def test_explicit_scale_adds_required_proof_obligation(self) -> None:
        plan = compile_question_plan(
            "Doanh thu theo đơn vị triệu đồng là bao nhiêu?"
        )
        contract = compile_question_proof_contract(plan, "SELECT")
        self.assertIn("unit_scale_match", contract.required)
        self.assertNotIn("unit_scale_match", contract.optional)
        self.assertNotIn("unit_scale_match", contract.not_applicable)

    def test_no_scale_keeps_obligation_not_applicable(self) -> None:
        plan = compile_question_plan("Doanh thu là bao nhiêu?")
        contract = compile_question_proof_contract(plan, "SELECT")
        self.assertNotIn("unit_scale_match", contract.required)
        self.assertNotIn("unit_scale_match", contract.optional)
        self.assertIn("unit_scale_match", contract.not_applicable)

    def test_matching_explicit_scale_is_valid(self) -> None:
        plan = compile_question_plan(
            "Doanh thu theo đơn vị triệu đồng là bao nhiêu?"
        )
        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            (_cell(0, metric="Doanh thu", value="125", unit="triệu đồng", unit_key="currency"),),
        )
        obligation = _obligation(result, "unit_scale_match")
        self.assertEqual(obligation.applicability, "required")
        self.assertIs(obligation.state, VerificationState.VALID)

    def test_missing_raw_evidence_scale_is_unknown(self) -> None:
        plan = compile_question_plan(
            "Doanh thu theo đơn vị triệu đồng là bao nhiêu?"
        )
        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            (_cell(0, metric="Doanh thu", value="125", unit="", unit_key="currency"),),
        )
        self.assertIs(
            _obligation(result, "unit_scale_match").state,
            VerificationState.UNKNOWN,
        )
        self.assertIs(result.proof_contract_state, VerificationState.UNKNOWN)

    def test_different_convertible_scale_without_conversion_is_unknown(self) -> None:
        plan = compile_question_plan(
            "Doanh thu theo đơn vị triệu đồng là bao nhiêu?"
        )
        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            (_cell(0, metric="Doanh thu", value="125000000", unit="VND", unit_key="currency"),),
        )
        self.assertIs(
            _obligation(result, "unit_scale_match").state,
            VerificationState.UNKNOWN,
        )
        self.assertIs(result.proof_contract_state, VerificationState.UNKNOWN)

    def test_mixed_explicit_operand_scales_are_invalid(self) -> None:
        plan = compile_question_plan(
            "Tổng nợ ngắn hạn và nợ dài hạn theo đơn vị triệu đồng là bao nhiêu?"
        )
        result = verify_program(
            plan,
            ReasoningProgram("ADD", (0, 1)),
            (
                _cell(0, metric="Nợ ngắn hạn", value="1000000", unit="VND", unit_key="currency"),
                _cell(1, metric="Nợ dài hạn", value="2", unit="triệu đồng", unit_key="currency"),
            ),
        )
        self.assertIs(
            _obligation(result, "unit_scale_match").state,
            VerificationState.INVALID,
        )
        self.assertIs(result.proof_contract_state, VerificationState.INVALID)

    def test_conflicting_requested_scales_are_invalid(self) -> None:
        plan = compile_question_plan(
            "Doanh thu theo nghìn đồng hoặc triệu đồng là bao nhiêu?"
        )
        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            (_cell(0, metric="Doanh thu", value="125", unit="triệu đồng", unit_key="currency"),),
        )
        self.assertEqual(plan.requested_scale, "conflict")
        self.assertIs(
            _obligation(result, "unit_scale_match").state,
            VerificationState.INVALID,
        )
        self.assertIs(result.proof_contract_state, VerificationState.INVALID)

    def test_monetary_scale_with_percent_output_is_invalid(self) -> None:
        plan = compile_question_plan(
            "Biên lợi nhuận là bao nhiêu phần trăm theo triệu đồng?"
        )
        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            (_cell(0, metric="Biên lợi nhuận", value="10", unit="%", unit_key="percent"),),
        )
        self.assertEqual(plan.requested_unit_family, "percent")
        self.assertIs(
            _obligation(result, "unit_scale_match").state,
            VerificationState.INVALID,
        )
        self.assertIs(result.proof_contract_state, VerificationState.INVALID)

    def test_base_vnd_request_matches_vnd_evidence(self) -> None:
        plan = compile_question_plan("Doanh thu là bao nhiêu đồng?")
        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            (_cell(0, metric="Doanh thu", value="125", unit="VND", unit_key="currency"),),
        )
        self.assertEqual(plan.requested_scale, "vnd")
        self.assertIs(
            _obligation(result, "unit_scale_match").state,
            VerificationState.VALID,
        )


if __name__ == "__main__":
    unittest.main()
