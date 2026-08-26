from __future__ import annotations

import unittest
from decimal import Decimal

from hrm_finance_qa.evidence_schema import (
    EvidenceCell,
    EvidenceRelation,
    build_evidence_graph,
)
from hrm_finance_qa.program_data import NumericCandidate, ReasoningProgram
from hrm_finance_qa.question_planner import (
    DirectionIntent,
    compile_question_plan,
    compile_question_proof_contract,
)
from hrm_finance_qa.symbolic_verifier import VerificationState, verify_program


def _cell(
    index: int,
    value: str,
    *,
    metric: str = "Doanh thu thuan",
    year: str = "",
    role: str = "",
    entity: str = "",
) -> EvidenceCell:
    return EvidenceCell(
        index=index,
        value=Decimal(value),
        metric_label=metric,
        metric_key="".join(metric.lower().split()),
        column=year,
        period=year,
        explicit_years=(year,) if year else (),
        period_role=role,
        entity=entity,
        section="",
        table_id="table-1",
        source_name="report.pdf",
        page=1,
        value_status="observed",
    )


class QuestionPlannerTests(unittest.TestCase):
    def test_absolute_change_is_not_misread_as_percent(self) -> None:
        plan = compile_question_plan(
            "Nợ dài hạn thay đổi bao nhiêu từ năm 2022 đến năm 2023?"
        )

        self.assertEqual(plan.operation, "SUBTRACT")
        self.assertEqual(plan.from_year, "2022")
        self.assertEqual(plan.to_year, "2023")
        self.assertEqual(plan.required_operand_years, ("2023", "2022"))
        self.assertEqual(plan.direction_intent, DirectionIntent.NEW_MINUS_OLD)

    def test_percent_wording_compiles_percent_change(self) -> None:
        plan = compile_question_plan(
            "Doanh thu thay đổi bao nhiêu phần trăm từ năm 2022 đến năm 2023?"
        )

        self.assertEqual(plan.operation, "PERCENT_CHANGE")
        self.assertTrue(plan.asks_percent)
        self.assertEqual(plan.direction_intent, DirectionIntent.NEW_MINUS_OLD)

    def test_implicit_two_year_change_orders_new_period_before_old(self) -> None:
        plan = compile_question_plan(
            "Chênh lệch doanh thu giữa năm 2022 và năm 2023 là bao nhiêu?"
        )

        self.assertEqual(plan.operation, "SUBTRACT")
        self.assertEqual(plan.required_operand_years, ("2023", "2022"))
        self.assertEqual(plan.direction_intent, DirectionIntent.NEW_MINUS_OLD)

    def test_from_to_with_parenthetical_units_keeps_explicit_direction(self) -> None:
        plan = compile_question_plan(
            "Doanh thu thay đổi từ Năm 2024 (VND) đến Năm 2025 (VND) bao nhiêu?"
        )

        self.assertEqual((plan.from_year, plan.to_year), ("2024", "2025"))
        self.assertEqual(plan.direction_intent, DirectionIntent.NEW_MINUS_OLD)

    def test_explicit_subtraction_compiles_left_minus_right_roles(self) -> None:
        plan = compile_question_plan(
            "Tính Tổng tài sản trừ Tổng nguồn vốn năm 2025 là bao nhiêu?"
        )

        self.assertEqual(plan.operation, "SUBTRACT")
        self.assertEqual(plan.operand_roles, ("tong tai san", "tong nguon von"))
        self.assertEqual(plan.direction_intent, DirectionIntent.LEFT_MINUS_RIGHT)

    def test_decrease_wording_has_positive_decrease_direction(self) -> None:
        plan = compile_question_plan(
            "Chi phí năm 2025 thấp hơn năm 2024 bao nhiêu?"
        )

        self.assertEqual(plan.operation, "SUBTRACT")
        self.assertEqual(plan.direction_intent, DirectionIntent.DECREASE_AMOUNT)
        self.assertEqual(plan.required_operand_years, ("2024", "2025"))

    def test_period_role_comparison_compiles_new_minus_old(self) -> None:
        plan = compile_question_plan(
            "Doanh thu cuối kỳ so với đầu kỳ chênh lệch bao nhiêu?"
        )

        self.assertEqual(plan.operation, "SUBTRACT")
        self.assertTrue(plan.temporal_change)
        self.assertEqual(plan.direction_intent, DirectionIntent.NEW_MINUS_OLD)

    def test_lower_than_compiles_positive_decrease_operand_roles(self) -> None:
        plan = compile_question_plan(
            "Chi phí tài chính thấp hơn Doanh thu tài chính bao nhiêu?"
        )

        self.assertEqual(plan.operation, "SUBTRACT")
        self.assertEqual(
            plan.operand_roles,
            ("doanh thu tai chinh", "chi phi tai chinh"),
        )
        self.assertEqual(plan.direction_intent, DirectionIntent.DECREASE_AMOUNT)

    def test_explicit_absolute_gap_is_not_plain_signed_subtract(self) -> None:
        plan = compile_question_plan(
            "Độ lệch tuyệt đối giữa Tổng tài sản và Tổng nguồn vốn là bao nhiêu?"
        )

        self.assertEqual(plan.direction_intent, DirectionIntent.ABSOLUTE_GAP)

    def test_unqualified_same_period_difference_is_ambiguous(self) -> None:
        plan = compile_question_plan(
            "Chênh lệch giữa Doanh thu và Chi phí là bao nhiêu?"
        )

        self.assertEqual(plan.direction_intent, DirectionIntent.AMBIGUOUS)

    def test_non_directional_lookup_has_no_direction_intent(self) -> None:
        plan = compile_question_plan("Doanh thu năm 2025 là bao nhiêu?")

        self.assertIsNone(plan.direction_intent)

    def test_named_company_is_separated_from_metric(self) -> None:
        plan = compile_question_plan(
            "Doanh thu năm 2023 của Công ty cổ phần ABC là bao nhiêu?"
        )

        self.assertEqual(plan.entity_mode, "named")
        self.assertIn("abc", plan.entity_terms)
        self.assertNotIn("cong ty", plan.metric_text)

    def test_query_ir_distinguishes_requested_output_unit(self) -> None:
        percent = compile_question_plan(
            "Biên lợi nhuận năm 2023 là bao nhiêu phần trăm?"
        )
        money = compile_question_plan(
            "Doanh thu thuần năm 2023 là bao nhiêu triệu đồng?"
        )

        self.assertEqual(percent.requested_unit_family, "percent")
        self.assertEqual(money.requested_unit_family, "currency")

    def test_binary_total_without_location_word_compiles_add_roles(self) -> None:
        plan = compile_question_plan(
            "Tổng nợ ngắn hạn và nợ dài hạn năm 2024 là bao nhiêu?"
        )

        self.assertEqual(plan.operation, "ADD")
        self.assertEqual(plan.operation_candidates, ("ADD",))
        self.assertEqual(plan.operand_roles, ("no ngan han", "no dai han"))
        self.assertFalse(plan.direct_metric_lookup)

    def test_identity_target_ignores_component_listing_followup(self) -> None:
        plan = compile_question_plan(
            "Tổng nợ phải trả năm 2026 là bao nhiêu? "
            "Hãy liệt kê nợ ngắn hạn và nợ dài hạn để kiểm tra."
        )

        self.assertIsNone(plan.operation)
        self.assertEqual(
            plan.financial_identity_ids, ("balance.total_liabilities",)
        )

    def test_same_period_balance_check_compiles_directional_subtract_roles(self) -> None:
        plan = compile_question_plan(
            "Tổng tài sản tính đến ngày 31/12/2026 có bằng Tổng nguồn vốn không? "
            "Nếu lệch thì chênh lệch là bao nhiêu?"
        )

        self.assertEqual(plan.operation, "SUBTRACT")
        self.assertEqual(plan.operation_candidates, ("SUBTRACT",))
        self.assertEqual(plan.operand_roles, ("tong tai san", "tong nguon von"))

    def test_current_ratio_identity_ignores_bilingual_and_qualitative_followup(self) -> None:
        plan = compile_question_plan(
            "Tính hệ số thanh toán hiện hành (Current Ratio) năm 2026 và cho biết "
            "doanh nghiệp có đủ khả năng trả nợ ngắn hạn không?"
        )

        self.assertEqual(plan.financial_identity_ids, ("ratio.current",))

    def test_identity_matching_handles_general_component_and_english_variants(self) -> None:
        cases = (
            (
                "Tong no phai tra nam 2026 bao gom no ngan han va no dai han la bao nhieu?",
                "balance.total_liabilities",
            ),
            (
                "Tong no phai tra nam 2026, trong do chi tiet no ngan han va no dai han.",
                "balance.total_liabilities",
            ),
            ("Tinh Current Ratio nam 2026 la bao nhieu?", "ratio.current"),
        )

        for question, identity_id in cases:
            with self.subTest(question=question):
                plan = compile_question_plan(question)
                self.assertIn(identity_id, plan.financial_identity_ids)

    def test_disclosed_coefficient_allows_lookup_and_computation(self) -> None:
        plan = compile_question_plan(
            "Hệ số nợ trên tổng tài sản năm 2024 là bao nhiêu?"
        )

        self.assertIsNone(plan.operation)
        self.assertEqual(plan.operation_candidates, ("SELECT", "DIVIDE"))
        self.assertEqual(plan.operand_roles, ("no", "tong tai san"))
        self.assertEqual(plan.requested_unit_family, "ratio")
        self.assertTrue(plan.direct_metric_lookup)

    def test_ratio_question_compiles_directional_operand_roles(self) -> None:
        plan = compile_question_plan(
            "Lợi nhuận sau thuế chiếm bao nhiêu phần trăm doanh thu thuần?"
        )

        self.assertEqual(plan.operation, "RATIO_PERCENT")
        self.assertEqual(
            plan.proof_operand_roles,
            ("loi nhuan sau thue", "doanh thu thuan"),
        )
        self.assertEqual(plan.operand_roles, ())

    def test_interest_bearing_debt_share_activates_multi_step_intent(self) -> None:
        plan = compile_question_plan(
            "Tong no vay tai chinh chiu lai gom vay ngan han va dai han "
            "chiem bao nhieu phan tram Tong nguon von nam 2025?"
        )

        self.assertEqual(plan.operation, "RATIO_PERCENT")
        self.assertTrue(plan.interest_bearing_debt_share)
        self.assertEqual(plan.requested_years, ("2025",))

    def test_proof_contract_excludes_non_applicable_constraints(self) -> None:
        plan = compile_question_plan(
            "Doanh thu thuần năm 2024 là bao nhiêu?"
        )
        contract = compile_question_proof_contract(plan, "SELECT")

        self.assertIn("query_metric", contract.required)
        self.assertIn("period_coverage", contract.required)
        self.assertIn("unit_algebra", contract.not_applicable)
        self.assertNotIn("unit_algebra", contract.optional)


class SymbolicVerifierTests(unittest.TestCase):
    def test_select_explicit_start_date_keeps_its_calendar_year(self) -> None:
        plan = compile_question_plan(
            "Đầu tư tài chính ngắn hạn tại 01/01/2025 là bao nhiêu?"
        )
        cell = _cell(
            0,
            "100",
            metric="Đầu tư tài chính ngắn hạn",
            year="2025",
            role="prior",
        )

        result = verify_program(
            plan, ReasoningProgram("SELECT", (0,)), (cell,)
        )

        obligation_states = {
            item.name: item.state for item in result.proof_obligations
        }
        self.assertEqual(
            obligation_states["period_coverage"], VerificationState.VALID
        )

    def test_temporal_prior_role_still_maps_to_requested_prior_year(self) -> None:
        plan = compile_question_plan(
            "Doanh thu thay đổi từ năm 2024 đến năm 2025?"
        )
        cell = _cell(0, "100", year="2025", role="prior")

        self.assertEqual(cell.effective_years(plan), {"2024"})

    def test_required_contract_ignores_optional_layout_gaps(self) -> None:
        plan = compile_question_plan(
            "Doanh thu thuần năm 2024 là bao nhiêu?"
        )
        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            (_cell(0, "100", metric="Doanh thu thuần", year="2024"),),
        )

        self.assertEqual(
            result.training_support_state, VerificationState.VALID
        )
        self.assertEqual(result.required_valid_fraction, 1.0)
        self.assertEqual(result.required_unknown_fraction, 0.0)
        self.assertTrue(
            any(
                item.applicability == "not_applicable"
                and item.state is None
                for item in result.proof_obligations
            )
        )

    def test_query_metric_contradiction_is_audit_only_in_phase_a(self) -> None:
        plan = compile_question_plan(
            "Doanh thu thuần năm 2024 là bao nhiêu?"
        )
        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            (_cell(0, "100", metric="Hàng tồn kho", year="2024"),),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(
            result.training_support_state, VerificationState.INVALID
        )
    def test_missing_period_metadata_is_unknown_not_invalid(self) -> None:
        plan = compile_question_plan(
            "Doanh thu thuan nam 2023 la bao nhieu?"
        )

        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            (_cell(0, "100", year=""),),
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.state, VerificationState.UNKNOWN)
        self.assertIn("period_coverage", result.unknown_reasons)

    def test_temporal_program_requires_one_metric_and_both_periods(self) -> None:
        plan = compile_question_plan(
            "Nợ dài hạn thay đổi bao nhiêu từ năm 2022 đến năm 2023?"
        )
        cells = (
            _cell(0, "1707170000", metric="No dai han", year="2023"),
            _cell(
                1,
                "941820740",
                metric="Tai san do dang dai han",
                year="2022",
            ),
        )

        result = verify_program(
            plan,
            ReasoningProgram("SUBTRACT", (0, 1)),
            cells,
        )

        self.assertFalse(result.accepted)
        self.assertIn("temporal_metric_mismatch", result.reasons)

    def test_implicit_two_year_change_rejects_reversed_direction(self) -> None:
        plan = compile_question_plan(
            "Chênh lệch doanh thu giữa năm 2022 và năm 2023 là bao nhiêu?"
        )
        cells = (
            _cell(0, "40", year="2023"),
            _cell(1, "49", year="2022"),
        )

        right = verify_program(
            plan, ReasoningProgram("SUBTRACT", (0, 1)), cells
        )
        reversed_program = verify_program(
            plan, ReasoningProgram("SUBTRACT", (1, 0)), cells
        )

        self.assertTrue(right.accepted)
        self.assertFalse(reversed_program.accepted)
        self.assertIn("temporal_direction", reversed_program.reasons)

    def test_reporting_company_wins_when_question_names_no_entity(self) -> None:
        plan = compile_question_plan(
            "Doanh thu thay đổi bao nhiêu từ năm 2022 đến năm 2023?"
        )
        cells = (
            _cell(0, "295", year="2023"),
            _cell(1, "313", year="2022"),
            _cell(2, "40", year="2023", entity="entity:cong ty lien ket"),
            _cell(3, "49", year="2022", entity="entity:cong ty lien ket"),
        )

        result = verify_program(
            plan,
            ReasoningProgram("SUBTRACT", (2, 3)),
            cells,
        )

        self.assertFalse(result.accepted)
        self.assertIn("reporting_entity_mismatch", result.reasons)

    def test_percent_change_from_zero_is_undefined(self) -> None:
        plan = compile_question_plan(
            "Nợ dài hạn thay đổi bao nhiêu phần trăm từ năm 2022 đến năm 2023?"
        )
        cells = (
            _cell(0, "1707170000", metric="No dai han", year="2023"),
            _cell(1, "0", metric="No dai han", year="2022"),
        )

        result = verify_program(
            plan,
            ReasoningProgram("PERCENT_CHANGE", (0, 1)),
            cells,
        )

        self.assertFalse(result.accepted)
        self.assertIn("zero_baseline_percent_change", result.reasons)

    def test_named_embedded_company_requires_matching_entity_cells(self) -> None:
        plan = compile_question_plan(
            "Doanh thu của Công ty cổ phần Khanh Hoa thay đổi bao nhiêu "
            "từ năm 2022 đến năm 2023?"
        )
        cells = (
            _cell(0, "295", year="2023"),
            _cell(1, "313", year="2022"),
            _cell(
                2,
                "40",
                year="2023",
                entity="entity:cong ty co phan khanh hoa",
            ),
            _cell(
                3,
                "49",
                year="2022",
                entity="entity:cong ty co phan khanh hoa",
            ),
        )

        wrong = verify_program(
            plan,
            ReasoningProgram("SUBTRACT", (0, 1)),
            cells,
        )
        right = verify_program(
            plan,
            ReasoningProgram("SUBTRACT", (2, 3)),
            cells,
        )

        self.assertIn("named_entity_mismatch", wrong.reasons)
        self.assertTrue(right.accepted)

    def test_multi_role_addition_rejects_unary_select(self) -> None:
        plan = compile_question_plan(
            "Tổng nợ ngắn hạn và nợ dài hạn năm 2024 là bao nhiêu?"
        )
        cells = (_cell(0, "100", metric="No ngan han", year="2024"),)

        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            cells,
        )

        self.assertFalse(result.accepted)
        self.assertIn("operand_arity_intent", result.reasons)

    def test_share_count_times_share_count_is_invalid_unit_algebra(self) -> None:
        plan = compile_question_plan(
            "Giá trị theo mệnh giá của số cổ phiếu đang lưu hành là bao nhiêu?"
        )
        cells = (
            _cell(0, "100", metric="So luong co phieu dang luu hanh"),
            _cell(1, "100", metric="So luong co phieu tu do"),
        )
        cells = tuple(
            EvidenceCell(**{**cell.__dict__, "unit": "shares", "unit_key": "share_count"})
            for cell in cells
        )

        result = verify_program(
            plan,
            ReasoningProgram("MULTIPLY", (0, 1)),
            cells,
        )

        self.assertFalse(result.accepted)
        self.assertIn("invalid_unit_algebra", result.reasons)


class EvidenceGraphTests(unittest.TestCase):
    @staticmethod
    def _candidate(index: int, value: str, year: str) -> NumericCandidate:
        return NumericCandidate(
            index=index,
            value=value,
            raw=value,
            context=(
                "ROW: Doanh thu thuan\n"
                "ROW_TYPE: detail\n"
                f"COLUMN: {year}\nPERIOD: {year}\n"
                "TABLE: page8-table1\n"
                "ENTITY: reporting-company"
            ),
            line_number=index + 1,
            char_start=0,
            char_end=len(value),
            unit="VND",
            is_percent=False,
            is_year_like=False,
            relevance=1.0,
            source_name="report.pdf",
        )

    def test_graph_encodes_typed_cell_relations(self) -> None:
        graph = build_evidence_graph(
            (
                self._candidate(0, "40", "2023"),
                self._candidate(1, "49", "2022"),
            )
        )

        relations = graph.relations(0, 1)
        self.assertIn(EvidenceRelation.SAME_METRIC, relations)
        self.assertIn(EvidenceRelation.SAME_ENTITY, relations)
        self.assertIn(EvidenceRelation.SAME_TABLE, relations)
        self.assertIn(EvidenceRelation.SAME_ROW, relations)
        self.assertIn(EvidenceRelation.SAME_UNIT, relations)
        self.assertIn(EvidenceRelation.PREVIOUS_PERIOD, relations)

    def test_share_row_overrides_inherited_currency_unit(self) -> None:
        candidate = NumericCandidate(
            index=0,
            value="10451182",
            raw="10.451.182",
            context=(
                "ROW: Số lượng cổ phần đang lưu hành\n"
                "CELL: Số lượng cổ phần đang lưu hành: "
                "<VALUE> 10.451.182 </VALUE> Cổ phần"
            ),
            line_number=1,
            char_start=0,
            char_end=10,
            unit="VND",
            is_percent=False,
            is_year_like=False,
            relevance=1.0,
        )

        graph = build_evidence_graph((candidate,))

        self.assertEqual(graph.cells[0].unit_key, "share_count")

    def test_par_value_row_is_currency_per_share_not_share_count(self) -> None:
        candidate = NumericCandidate(
            index=0,
            value="10000",
            raw="10.000",
            context="ROW: Mệnh giá cổ phiếu đang lưu hành",
            line_number=1,
            char_start=0,
            char_end=6,
            unit="VND",
            is_percent=False,
            is_year_like=False,
            relevance=1.0,
        )

        graph = build_evidence_graph((candidate,))

        self.assertEqual(graph.cells[0].unit_key, "currency_per_share")

    def test_par_value_unit_rejects_mismatched_tagged_table_cell(self) -> None:
        candidate = NumericCandidate(
            index=0,
            value="120",
            raw="120",
            context=(
                "ROW: Mệnh giá cổ phiếu: 10.000 VND/cổ phiếu\n"
                "CELL: | Mệnh giá cổ phiếu: 10.000 VND/cổ phiếu | "
                "Trong nước | <VALUE> 120 </VALUE> | 10.000.000 | 100% |"
            ),
            line_number=1,
            char_start=0,
            char_end=3,
            unit="VND",
            is_percent=False,
            is_year_like=False,
            relevance=1.0,
        )

        graph = build_evidence_graph((candidate,))

        self.assertEqual(graph.cells[0].unit_key, "currency")

    def test_par_value_unit_accepts_matching_tagged_value(self) -> None:
        candidate = NumericCandidate(
            index=0,
            value="10000",
            raw="10.000",
            context=(
                "ROW: Mệnh giá cổ phiếu: 10.000 VND/cổ phiếu\n"
                "CELL: Mệnh giá cổ phiếu: "
                "<VALUE> 10.000 </VALUE> VND/cổ phiếu"
            ),
            line_number=1,
            char_start=0,
            char_end=6,
            unit="VND",
            is_percent=False,
            is_year_like=False,
            relevance=1.0,
        )

        graph = build_evidence_graph((candidate,))

        self.assertEqual(graph.cells[0].unit_key, "currency_per_share")


if __name__ == "__main__":
    unittest.main()
