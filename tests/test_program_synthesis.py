from __future__ import annotations

import unittest

from hrm_finance_qa.evidence_schema import build_evidence_graph
from hrm_finance_qa.program_data import NumericCandidate, ReasoningProgram
from hrm_finance_qa.program_synthesis import (
    _pair_compatible,
    synthesize_typed_programs,
)
from hrm_finance_qa.question_planner import compile_question_plan


def _candidate(
    index: int,
    value: str,
    metric: str,
    year: str,
    *,
    entity: str = "",
    table: str = "table-1",
    unit: str = "VND",
) -> NumericCandidate:
    return NumericCandidate(
        index=index,
        value=value,
        raw=value,
        context=(
            f"ROW: {metric}\n"
            f"COLUMN: {year}\n"
            f"PERIOD: {year}\n"
            f"TABLE: {table}\n"
            f"SCOPE: {entity}"
        ),
        line_number=index + 1,
        char_start=index * 10,
        char_end=index * 10 + len(value),
        unit=unit,
        is_percent=False,
        is_year_like=False,
        relevance=1.0 - index * 0.05,
        source_name="unseen-layout-report.pdf",
        extraction_method="native",
    )


class TypedProgramSynthesisTests(unittest.TestCase):
    def test_lookup_uses_metric_period_and_reporting_scope(self) -> None:
        candidates = (
            _candidate(0, "49", "Doanh thu thuan", "2022"),
            _candidate(1, "40", "Doanh thu thuan", "2023"),
            _candidate(
                2,
                "295",
                "Doanh thu thuan",
                "2023",
                entity="entity:cong ty lien ket",
                table="embedded-table",
            ),
        )
        plan = compile_question_plan(
            "Doanh thu thuan nam 2023 la bao nhieu?"
        )

        result = synthesize_typed_programs(
            plan,
            "SELECT",
            build_evidence_graph(candidates),
            focus_text=plan.metric_text,
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.programs[0], ReasoningProgram("SELECT", (1,)))
        self.assertNotIn(ReasoningProgram("SELECT", (0,)), result.programs)
        self.assertNotIn(ReasoningProgram("SELECT", (2,)), result.programs)

    def test_temporal_template_crosses_layout_without_crossing_metric(self) -> None:
        candidates = (
            _candidate(
                0, "1707", "No dai han", "2025", table="balance-page-8"
            ),
            _candidate(
                1, "0", "No dai han", "2024", table="balance-page-9"
            ),
            _candidate(
                2,
                "941",
                "Tai san do dang dai han",
                "2024",
                table="assets-page-9",
            ),
        )
        plan = compile_question_plan(
            "No dai han thay doi bao nhieu tu nam 2024 den nam 2025?"
        )

        result = synthesize_typed_programs(
            plan,
            "SUBTRACT",
            build_evidence_graph(candidates),
            focus_text=plan.metric_text,
        )

        self.assertIn(
            ReasoningProgram("SUBTRACT", (0, 1)), result.programs
        )
        self.assertNotIn(
            ReasoningProgram("SUBTRACT", (0, 2)), result.programs
        )

    def test_decrease_template_synthesizes_old_minus_new(self) -> None:
        candidates = (
            _candidate(0, "90", "Doanh thu", "2025"),
            _candidate(1, "100", "Doanh thu", "2024"),
        )
        plan = compile_question_plan(
            "Doanh thu năm 2025 giảm bao nhiêu so với năm 2024?"
        )

        result = synthesize_typed_programs(
            plan,
            "SUBTRACT",
            build_evidence_graph(candidates),
            focus_text=plan.metric_text,
        )

        self.assertIn(ReasoningProgram("SUBTRACT", (1, 0)), result.programs)
        self.assertNotIn(ReasoningProgram("SUBTRACT", (0, 1)), result.programs)

    def test_ambiguous_difference_does_not_synthesize_signed_subtract(self) -> None:
        candidates = (
            _candidate(0, "100", "Doanh thu", "2025"),
            _candidate(1, "80", "Chi phi", "2025"),
        )
        plan = compile_question_plan(
            "Chênh lệch giữa Doanh thu và Chi phí là bao nhiêu?"
        )

        result = synthesize_typed_programs(
            plan,
            "SUBTRACT",
            build_evidence_graph(candidates),
            focus_text=plan.metric_text,
            operand_role_texts=("doanh thu", "chi phi"),
        )

        self.assertEqual(result.status, "direction_not_executable")
        self.assertEqual(result.programs, ())

    def test_explicit_same_period_difference_roles_override_temporal_layout(self) -> None:
        candidates = (
            _candidate(0, "1000", "Tong tai san", "2026"),
            _candidate(1, "1000", "Tong nguon von", "2026"),
            _candidate(2, "900", "Tong tai san", "2025"),
            _candidate(3, "900", "Tong nguon von", "2025"),
        )
        plan = compile_question_plan(
            "Tong tai san nam 2026 co bang Tong nguon von khong? "
            "Neu lech thi chenh lech la bao nhieu?"
        )

        result = synthesize_typed_programs(
            plan,
            "SUBTRACT",
            build_evidence_graph(candidates),
            focus_text=plan.metric_text,
            operand_role_texts=plan.operand_roles,
        )

        self.assertIn(ReasoningProgram("SUBTRACT", (0, 1)), result.programs)
        self.assertNotIn(ReasoningProgram("SUBTRACT", (2, 3)), result.programs)

    def test_role_synthesis_excludes_period_unknown_change_columns(self) -> None:
        candidates = (
            _candidate(0, "1000", "Tong tai san", "2026", unit="VND"),
            _candidate(1, "1000", "Tong nguon von", "2026", unit="VND"),
            _candidate(2, "-0.7", "Tong tai san", "", unit="%"),
            _candidate(3, "-0.7", "Tong nguon von", "", unit="%"),
        )
        question = (
            "Tong tai san nam 2026 co bang Tong nguon von khong? "
            "Neu lech thi chenh lech la bao nhieu?"
        )
        plan = compile_question_plan(question)

        result = synthesize_typed_programs(
            plan,
            "SUBTRACT",
            build_evidence_graph(candidates),
            focus_text=question,
            operand_role_texts=("tong tai san", "tong nguon von"),
        )

        self.assertIn(ReasoningProgram("SUBTRACT", (0, 1)), result.programs)
        self.assertNotIn(ReasoningProgram("SUBTRACT", (2, 3)), result.programs)

    def test_role_synthesis_keeps_unknown_period_when_it_is_only_evidence(self) -> None:
        candidates = (
            _candidate(0, "1000", "Tong tai san", "", unit="VND"),
            _candidate(1, "1000", "Tong nguon von", "", unit="VND"),
        )
        question = (
            "Tong tai san nam 2026 co bang Tong nguon von khong? "
            "Neu lech thi chenh lech la bao nhieu?"
        )
        plan = compile_question_plan(question)

        result = synthesize_typed_programs(
            plan,
            "SUBTRACT",
            build_evidence_graph(candidates),
            focus_text=question,
            operand_role_texts=("tong tai san", "tong nguon von"),
        )

        self.assertIn(ReasoningProgram("SUBTRACT", (0, 1)), result.programs)

    def test_unrelated_requested_year_does_not_suppress_unknown_role_evidence(
        self,
    ) -> None:
        candidates = (
            _candidate(0, "1000", "Tong tai san", "", unit="VND"),
            _candidate(1, "1000", "Tong nguon von", "", unit="VND"),
            _candidate(2, "50", "Doanh thu thuan", "2026", unit="VND"),
        )
        question = (
            "Tong tai san nam 2026 co bang Tong nguon von khong? "
            "Neu lech thi chenh lech la bao nhieu?"
        )
        plan = compile_question_plan(question)

        result = synthesize_typed_programs(
            plan,
            "SUBTRACT",
            build_evidence_graph(candidates),
            focus_text=question,
            operand_role_texts=("tong tai san", "tong nguon von"),
        )

        self.assertIn(ReasoningProgram("SUBTRACT", (0, 1)), result.programs)

    def test_named_entity_template_uses_matching_embedded_scope(self) -> None:
        candidates = (
            _candidate(0, "295", "Doanh thu thuan", "2023"),
            _candidate(
                1,
                "40",
                "Doanh thu thuan",
                "2023",
                entity="entity:cong ty co phan khanh hoa",
                table="associate-table",
            ),
        )
        plan = compile_question_plan(
            "Doanh thu thuan nam 2023 cua Cong ty co phan Khanh Hoa "
            "la bao nhieu?"
        )

        result = synthesize_typed_programs(
            plan,
            "SELECT",
            build_evidence_graph(candidates),
            focus_text=plan.metric_text,
        )

        self.assertEqual(result.programs, (ReasoningProgram("SELECT", (1,)),))

    def test_explicit_binary_roles_create_typed_add_assignments(self) -> None:
        candidates = (
            _candidate(0, "100", "No ngan han", "2024"),
            _candidate(1, "40", "Phai tra nguoi ban ngan han", "2024"),
            _candidate(2, "60", "No dai han", "2024"),
        )
        plan = compile_question_plan(
            "Tong No ngan han va Phai tra nguoi ban ngan han tai nam 2024"
        )

        result = synthesize_typed_programs(
            plan,
            "ADD",
            build_evidence_graph(candidates),
            focus_text=plan.metric_text,
            operand_role_texts=(
                "No ngan han",
                "Phai tra nguoi ban ngan han",
            ),
        )

        self.assertIn(ReasoningProgram("ADD", (0, 1)), result.programs)
        self.assertNotIn(ReasoningProgram("ADD", (0, 2)), result.programs)

    def test_add_recovers_full_metric_anchors_with_internal_conjunctions(self) -> None:
        candidates = (
            _candidate(0, "16", "Tien gui va vay Chinh phu NHNN", "2025"),
            _candidate(1, "148", "Tien gui va vay cac TCTD khac", "2025"),
            _candidate(2, "91", "Tien gui va vay khach hang", "2025"),
        )
        question = (
            "Tong Tien gui va vay Chinh phu NHNN va "
            "Tien gui va vay cac TCTD khac tai nam 2025 la bao nhieu?"
        )
        plan = compile_question_plan(question)

        result = synthesize_typed_programs(
            plan,
            "ADD",
            build_evidence_graph(candidates),
            focus_text=question,
            operand_role_texts=(
                "tien gui va vay chinh phu nhnn va tien gui",
                "vay cac tctd khac",
            ),
        )

        self.assertIn(ReasoningProgram("ADD", (0, 1)), result.programs)
        self.assertNotIn(ReasoningProgram("ADD", (0, 2)), result.programs)

    def test_explicit_currency_request_rejects_percent_cell(self) -> None:
        candidates = (
            _candidate(0, "40", "Doanh thu thuan", "2023", unit="VND"),
            _candidate(1, "82", "Doanh thu thuan", "2023", unit="%"),
        )
        plan = compile_question_plan(
            "Doanh thu thuan nam 2023 la bao nhieu dong?"
        )

        result = synthesize_typed_programs(
            plan,
            "SELECT",
            build_evidence_graph(candidates),
            focus_text=plan.metric_text,
        )

        self.assertIn(ReasoningProgram("SELECT", (0,)), result.programs)
        self.assertNotIn(ReasoningProgram("SELECT", (1,)), result.programs)

    def test_multiply_does_not_reward_same_unit_as_positive_proof(self) -> None:
        graph = build_evidence_graph(
            (
                _candidate(0, "10", "So luong", "2024", unit="VND"),
                _candidate(1, "20", "Menh gia", "2024", unit="VND"),
            )
        )

        compatible, _, proofs = _pair_compatible(
            graph,
            graph.cells[0],
            graph.cells[1],
            operation="MULTIPLY",
            require_same_metric=False,
        )

        self.assertTrue(compatible)
        self.assertNotIn("same_unit", proofs)

    def test_multiply_rejects_two_share_counts(self) -> None:
        graph = build_evidence_graph(
            (
                _candidate(
                    0,
                    "10451182",
                    "So luong co phan dang luu hanh",
                    "",
                    unit="shares",
                ),
                _candidate(
                    1,
                    "10451182",
                    "So luong co phan chuyen nhuong tu do",
                    "",
                    unit="shares",
                ),
            )
        )

        compatible, _, _ = _pair_compatible(
            graph,
            graph.cells[0],
            graph.cells[1],
            operation="MULTIPLY",
            require_same_metric=False,
        )

        self.assertFalse(compatible)

    def test_par_value_roles_override_noisy_share_unit(self) -> None:
        candidates = (
            _candidate(
                0,
                "101206352",
                "So luong co phieu dang luu hanh",
                "2025",
                unit="shares",
            ),
            _candidate(
                1,
                "10000",
                "Menh gia co phieu",
                "2025",
                unit="shares",
            ),
            _candidate(
                2,
                "20000",
                "So luong co phieu quy",
                "2025",
                unit="shares",
            ),
        )
        question = (
            "Gia tri theo menh gia cua so luong co phieu dang luu hanh "
            "tai nam 2025 la bao nhieu?"
        )
        plan = compile_question_plan(question)

        result = synthesize_typed_programs(
            plan,
            "MULTIPLY",
            build_evidence_graph(candidates),
            focus_text=question,
            operand_role_texts=("so luong co phieu dang luu hanh", "menh gia"),
        )

        self.assertIn(ReasoningProgram("MULTIPLY", (0, 1)), result.programs)
        self.assertNotIn(ReasoningProgram("MULTIPLY", (0, 2)), result.programs)

    def test_disclosed_ratio_lookup_prefers_exact_metric_row(self) -> None:
        candidates = (
            _candidate(
                0,
                "0.03",
                "He so No/Tong tai san",
                "2024",
                unit="x",
            ),
            _candidate(1, "3360", "Che bien Tan", "2024", unit="tan"),
            _candidate(
                2,
                "202353239327",
                "Tong gia tri tai san",
                "2024",
                unit="VND",
            ),
        )
        plan = compile_question_plan(
            "He so no tren tong tai san nam 2024 la bao nhieu?"
        )

        result = synthesize_typed_programs(
            plan,
            "SELECT",
            build_evidence_graph(candidates),
            focus_text=plan.metric_text,
        )

        self.assertEqual(result.programs[0], ReasoningProgram("SELECT", (0,)))


if __name__ == "__main__":
    unittest.main()
