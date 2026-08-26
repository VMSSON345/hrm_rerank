from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import unittest

from hrm_finance_qa.evidence_schema import EvidenceGraph, build_evidence_graph
from hrm_finance_qa.program_data import NumericCandidate, ReasoningProgram
from hrm_finance_qa.program_synthesis import synthesize_typed_programs
from hrm_finance_qa.question_planner import compile_question_plan
from hrm_finance_qa.symbolic_verifier import VerificationState, verify_program


def _candidate(
    index: int,
    metric: str,
    value: str,
    *,
    year: str = "2023",
    section: str = "",
    table: str = "table-main",
    unit: str = "VND",
    scope: str = "",
    source_span: tuple[int, int] | None = None,
) -> NumericCandidate:
    start, end = source_span or (100 + index * 20, 110 + index * 20)
    return NumericCandidate(
        index=index,
        value=value,
        raw=value,
        context=(
            f"ROW: {metric}\n"
            f"COLUMN: {year}\n"
            f"PERIOD: {year}\n"
            f"TABLE: {table}\n"
            f"ROW_ID: row:{index}:{metric}\n"
            f"COLUMN_ID: column:{year}\n"
            f"SECTION: {section}\n"
            f"SCOPE: {scope}"
        ),
        line_number=10 + index,
        char_start=start,
        char_end=end,
        unit=unit,
        is_percent=False,
        is_year_like=False,
        relevance=1.0,
        source_name="development-generic.pdf",
        page=1,
        extraction_method="native",
        source_span=(start, end),
    )


def _state(result, name: str) -> VerificationState:
    return next(item.state for item in result.proof_obligations if item.name == name)


class QP3CoreGeneralizationCoverageImplementationTests(unittest.TestCase):
    def test_structural_add_hypotheses_bind_internal_conjunction_metric(self) -> None:
        question = (
            "Tong tien va tuong duong tien va doanh thu thuan tai nam 2023 "
            "la bao nhieu?"
        )
        candidates = (
            _candidate(0, "Tien va tuong duong tien", "40"),
            _candidate(1, "Doanh thu thuan", "60"),
            _candidate(2, "No phai tra", "90"),
        )
        plan = compile_question_plan(question)
        self.assertEqual(plan.operation, "ADD")
        self.assertIn(
            ("tien va tuong duong tien", "doanh thu thuan"),
            plan.addition_role_hypotheses,
        )
        graph = build_evidence_graph(candidates)
        synthesis = synthesize_typed_programs(
            plan, "ADD", graph, focus_text=plan.metric_text
        )
        program = ReasoningProgram("ADD", (0, 1))
        self.assertIn(program, synthesis.programs)
        verification = verify_program(plan, program, graph)
        self.assertEqual(_state(verification, "operation_match"), VerificationState.VALID)
        self.assertEqual(_state(verification, "add_grounding"), VerificationState.VALID)

    def test_add_role_ignores_numeric_outline_marker_only(self) -> None:
        question = (
            "Tong I. Tai san ngan han va 1. Tien va cac khoan tuong duong "
            "tien tai dau nam la bao nhieu?"
        )
        candidates = (
            _candidate(0, "I. Tai san ngan han", "80", year="dau nam"),
            _candidate(
                1,
                "Tien va cac khoan tuong duong tien",
                "20",
                year="dau nam",
            ),
        )
        result = verify_program(
            compile_question_plan(question),
            ReasoningProgram("ADD", (0, 1)),
            build_evidence_graph(candidates),
        )

        self.assertEqual(_state(result, "add_grounding"), VerificationState.VALID)

    def test_add_role_outline_marker_does_not_license_narrow_row(self) -> None:
        question = (
            "Tong I. Tai san ngan han va 1. Tien va cac khoan tuong duong "
            "tien tai dau nam la bao nhieu?"
        )
        candidates = (
            _candidate(0, "I. Tai san ngan han", "80", year="dau nam"),
            _candidate(1, "Tien", "20", year="dau nam"),
        )
        result = verify_program(
            compile_question_plan(question),
            ReasoningProgram("ADD", (0, 1)),
            build_evidence_graph(candidates),
        )

        self.assertIsNot(
            _state(result, "add_grounding"), VerificationState.VALID
        )

    def test_compound_metric_conjunction_never_materializes_add_without_two_roles(self) -> None:
        question = (
            "Tong doanh thu ban hang va cung cap dich vu nam 2023 la bao nhieu?"
        )
        candidates = (
            _candidate(0, "Doanh thu ban hang va cung cap dich vu", "100"),
            _candidate(1, "No phai tra", "50"),
        )
        plan = compile_question_plan(question)
        graph = build_evidence_graph(candidates)
        synthesis = synthesize_typed_programs(
            plan, "ADD", graph, focus_text=plan.metric_text
        )
        accepted = [
            proposal.program
            for proposal in synthesis.proposals
            if verify_program(plan, proposal.program, graph).proof_contract_state
            is VerificationState.VALID
        ]
        self.assertEqual(accepted, [])

    def test_ambiguous_add_separators_remain_unanswered(self) -> None:
        question = (
            "Tong tien va tuong duong tien va doanh thu thuan va loi nhuan gop "
            "nam 2023 la bao nhieu?"
        )
        candidates = (
            _candidate(0, "Tien va tuong duong tien", "40"),
            _candidate(1, "Doanh thu thuan", "60"),
            _candidate(2, "Loi nhuan gop", "20"),
        )
        plan = compile_question_plan(question)
        graph = build_evidence_graph(candidates)
        synthesis = synthesize_typed_programs(
            plan, "ADD", graph, focus_text=plan.metric_text
        )
        self.assertGreater(len(plan.addition_role_hypotheses), 1)
        for proposal in synthesis.proposals:
            self.assertIsNot(
                verify_program(plan, proposal.program, graph).proof_contract_state,
                VerificationState.VALID,
            )

    def test_same_physical_cell_cannot_ground_two_add_roles(self) -> None:
        question = "Cong doanh thu thuan voi loi nhuan gop trong nam 2023."
        candidates = (
            _candidate(0, "Doanh thu thuan", "60", source_span=(100, 110)),
            _candidate(1, "Loi nhuan gop", "20", source_span=(100, 110)),
        )
        plan = compile_question_plan(question)
        result = verify_program(
            plan,
            ReasoningProgram("ADD", (0, 1)),
            build_evidence_graph(candidates),
        )
        self.assertEqual(_state(result, "add_grounding"), VerificationState.INVALID)

    def test_wrong_add_role_counterfactual_is_rejected(self) -> None:
        question = "Cong doanh thu thuan voi loi nhuan gop trong nam 2023."
        candidates = (
            _candidate(0, "Doanh thu thuan", "60"),
            _candidate(1, "Loi nhuan gop", "20"),
            _candidate(2, "No phai tra", "90"),
        )
        plan = compile_question_plan(question)
        result = verify_program(
            plan,
            ReasoningProgram("ADD", (0, 2)),
            build_evidence_graph(candidates),
        )
        self.assertEqual(_state(result, "add_grounding"), VerificationState.INVALID)

    def test_wrong_add_period_counterfactual_never_answers(self) -> None:
        question = "Cong doanh thu thuan voi loi nhuan gop trong nam 2023."
        candidates = (
            _candidate(0, "Doanh thu thuan", "60"),
            _candidate(1, "Loi nhuan gop", "20", year="2022"),
        )
        result = verify_program(
            compile_question_plan(question),
            ReasoningProgram("ADD", (0, 1)),
            build_evidence_graph(candidates),
        )
        self.assertIsNot(result.state, VerificationState.VALID)

    def test_add_current_period_requires_explicit_cell_period_proof(self) -> None:
        question = (
            "Cong chi phi ban hang voi chi phi quan ly doanh nghiep tai nam nay "
            "la bao nhieu?"
        )
        candidates = (
            _candidate(0, "Chi phi ban hang", "60", year="nam nay"),
            _candidate(1, "Chi phi quan ly doanh nghiep", "20", year="nam nay"),
        )
        plan = compile_question_plan(question)
        graph = build_evidence_graph(candidates)
        missing_period_graph = EvidenceGraph(
            cells=tuple(replace(cell, period="") for cell in graph.cells),
            edges=graph.edges,
        )

        valid = verify_program(
            plan,
            ReasoningProgram("ADD", (0, 1)),
            graph,
        )
        missing = verify_program(
            plan,
            ReasoningProgram("ADD", (0, 1)),
            missing_period_graph,
        )

        self.assertEqual(_state(valid, "add_grounding"), VerificationState.VALID)
        self.assertEqual(
            _state(missing, "add_grounding"), VerificationState.UNKNOWN
        )

    def test_add_current_period_rejects_explicit_prior_period_cells(self) -> None:
        question = (
            "Cong chi phi ban hang voi chi phi quan ly doanh nghiep tai nam nay "
            "la bao nhieu?"
        )
        candidates = (
            _candidate(0, "Chi phi ban hang", "60", year="nam truoc"),
            _candidate(
                1,
                "Chi phi quan ly doanh nghiep",
                "20",
                year="nam truoc",
            ),
        )
        result = verify_program(
            compile_question_plan(question),
            ReasoningProgram("ADD", (0, 1)),
            build_evidence_graph(candidates),
        )

        self.assertEqual(_state(result, "add_grounding"), VerificationState.INVALID)

    def test_add_bare_year_abstains_on_competing_same_year_columns(self) -> None:
        question = "Cong tien voi cac khoan phai thu tai 2023 la bao nhieu?"
        candidates = (
            _candidate(0, "Tien", "34", year="31/12/2023"),
            _candidate(1, "Tien", "52", year="01/01/2023"),
            _candidate(2, "Cac khoan phai thu", "69", year="31/12/2023"),
            _candidate(3, "Cac khoan phai thu", "71", year="01/01/2023"),
        )
        result = verify_program(
            compile_question_plan(question),
            ReasoningProgram("ADD", (0, 2)),
            build_evidence_graph(candidates),
        )

        self.assertEqual(
            _state(result, "period_coverage"), VerificationState.UNKNOWN
        )
        self.assertIsNot(result.proof_contract_state, VerificationState.VALID)

    def test_add_explicit_date_resolves_same_year_column_choice(self) -> None:
        question = "Cong tien voi cac khoan phai thu tai 31/12/2023 la bao nhieu?"
        candidates = (
            _candidate(0, "Tien", "34", year="31/12/2023"),
            _candidate(1, "Tien", "52", year="01/01/2023"),
            _candidate(2, "Cac khoan phai thu", "69", year="31/12/2023"),
            _candidate(3, "Cac khoan phai thu", "71", year="01/01/2023"),
        )
        result = verify_program(
            compile_question_plan(question),
            ReasoningProgram("ADD", (0, 2)),
            build_evidence_graph(candidates),
        )

        self.assertEqual(
            _state(result, "period_coverage"), VerificationState.VALID
        )

    def test_select_bare_year_keeps_existing_period_coverage(self) -> None:
        question = "Tien tai 2023 la bao nhieu?"
        candidates = (
            _candidate(0, "Tien", "34", year="31/12/2023"),
            _candidate(1, "Tien", "52", year="01/01/2023"),
        )
        result = verify_program(
            compile_question_plan(question),
            ReasoningProgram("SELECT", (0,)),
            build_evidence_graph(candidates),
        )

        self.assertEqual(
            _state(result, "period_coverage"), VerificationState.VALID
        )

    def test_wrong_add_unit_counterfactual_never_answers(self) -> None:
        question = "Cong doanh thu thuan voi loi nhuan gop trong nam 2023."
        candidates = (
            _candidate(0, "Doanh thu thuan", "60"),
            _candidate(1, "Loi nhuan gop", "20", unit="shares"),
        )
        result = verify_program(
            compile_question_plan(question),
            ReasoningProgram("ADD", (0, 1)),
            build_evidence_graph(candidates),
        )
        self.assertEqual(result.state, VerificationState.INVALID)
        self.assertEqual(
            next(item.state for item in result.constraints if item.name == "unit_match"),
            VerificationState.INVALID,
        )

    def test_add_rejects_ratio_cells_without_ratio_intent(self) -> None:
        question = "Cong tai san ngan han voi tai san dai han nam 2022."
        candidates = (
            _candidate(0, "Tai san ngan han", "85.62", unit="%"),
            _candidate(1, "Tai san dai han", "14.38", unit="%"),
        )
        result = verify_program(
            compile_question_plan(question),
            ReasoningProgram("ADD", (0, 1)),
            build_evidence_graph(candidates),
        )

        self.assertEqual(
            next(
                item.state
                for item in result.constraints
                if item.name == "unit_match"
            ),
            VerificationState.INVALID,
        )

    def test_add_allows_ratio_cells_with_explicit_ratio_intent(self) -> None:
        question = (
            "Cong ty trong tai san ngan han voi ty trong tai san dai han "
            "nam 2022 theo phan tram."
        )
        candidates = (
            _candidate(0, "Ty trong tai san ngan han", "85.62", unit="%"),
            _candidate(1, "Ty trong tai san dai han", "14.38", unit="%"),
        )
        result = verify_program(
            compile_question_plan(question),
            ReasoningProgram("ADD", (0, 1)),
            build_evidence_graph(candidates),
        )

        self.assertEqual(
            next(
                item.state
                for item in result.constraints
                if item.name == "unit_match"
            ),
            VerificationState.VALID,
        )

    def test_wrong_add_entity_counterfactual_never_answers(self) -> None:
        question = (
            "Cong doanh thu thuan cua cong ty me voi loi nhuan gop cua "
            "cong ty me trong nam 2023."
        )
        candidates = (
            _candidate(
                0, "Doanh thu thuan", "60", scope="parent_company"
            ),
            _candidate(
                1, "Loi nhuan gop", "20", scope="consolidated"
            ),
        )
        result = verify_program(
            compile_question_plan(question),
            ReasoningProgram("ADD", (0, 1)),
            build_evidence_graph(candidates),
        )
        self.assertIsNot(result.state, VerificationState.VALID)
        entity_states = {
            item.name: item.state
            for item in result.constraints
            if item.name in {"reporting_entity", "named_entity"}
        }
        self.assertTrue(entity_states)
        self.assertTrue(
            any(state is not VerificationState.VALID for state in entity_states.values())
        )

    def test_provenanced_composite_query_metric_is_valid(self) -> None:
        question = "Thu nhap lai thuan tu hoat dong dich vu nam 2023 la bao nhieu?"
        candidates = (
            _candidate(0, "Thu nhap hoat dong", "60", section="Lai thuan dich vu"),
        )
        plan = compile_question_plan(question)
        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            build_evidence_graph(candidates),
        )
        self.assertEqual(_state(result, "query_metric"), VerificationState.VALID)

    def test_unprovenanced_composite_query_metric_remains_unknown(self) -> None:
        question = "Thu nhap lai thuan tu hoat dong dich vu nam 2023 la bao nhieu?"
        candidates = (
            _candidate(0, "Thu nhap hoat dong", "60", section="Lai thuan dich vu"),
        )
        plan = compile_question_plan(question)
        graph = build_evidence_graph(candidates)
        cell = replace(graph.cells[0], header_path_provenance=())
        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            EvidenceGraph(cells=(cell,), edges=()),
        )
        self.assertEqual(_state(result, "query_metric"), VerificationState.UNKNOWN)

    def test_symmetric_metric_requires_provenanced_relative_period_corroboration(self) -> None:
        question = "Gia tri con lai cua tai san co dinh huu hinh vao cuoi?"
        candidate = _candidate(
            0,
            "1. Tai san co dinh huu hinh",
            "60",
            year="So cuoi nam",
        )
        plan = compile_question_plan(question)
        graph = build_evidence_graph((candidate,))
        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            graph,
        )
        unprovenanced = replace(graph.cells[0], header_path_provenance=())
        missing_proof = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            EvidenceGraph(cells=(unprovenanced,), edges=()),
        )
        self.assertEqual(_state(result, "query_metric"), VerificationState.VALID)
        self.assertEqual(
            _state(missing_proof, "query_metric"), VerificationState.UNKNOWN
        )

    def test_symmetric_metric_with_wrong_relative_period_remains_unknown(self) -> None:
        question = "Gia tri con lai cua tai san co dinh huu hinh vao cuoi?"
        candidate = _candidate(
            0,
            "1. Tai san co dinh huu hinh",
            "60",
            year="So dau nam",
        )
        result = verify_program(
            compile_question_plan(question),
            ReasoningProgram("SELECT", (0,)),
            build_evidence_graph((candidate,)),
        )
        self.assertEqual(_state(result, "query_metric"), VerificationState.UNKNOWN)

    def test_competing_composite_metric_cells_remain_unknown(self) -> None:
        question = "Thu nhap lai thuan tu hoat dong dich vu nam 2023 la bao nhieu?"
        candidates = (
            _candidate(
                0, "Thu nhap hoat dong", "60", section="Lai thuan dich vu", table="table-a"
            ),
            _candidate(
                1, "Thu nhap hoat dong", "70", section="Lai thuan dich vu", table="table-b"
            ),
        )
        plan = compile_question_plan(question)
        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            build_evidence_graph(candidates),
        )
        self.assertEqual(_state(result, "query_metric"), VerificationState.UNKNOWN)

    def test_shareholder_noun_is_not_removed_as_currency_unit(self) -> None:
        plan = compile_question_plan(
            "So luong co dong trong nuoc tai nam 2023 la bao nhieu?"
        )
        self.assertIn("co dong", plan.metric_text)

    def test_explicit_shareholder_metric_distinguishes_requested_row(self) -> None:
        question = "So luong co dong trong nuoc tai nam 2023 la bao nhieu?"
        candidates = (
            _candidate(
                0,
                "Co dong trong nuoc",
                "120",
                section="So luong co dong",
                table="table-shareholders",
            ),
            _candidate(
                1,
                "Co dong nuoc ngoai",
                "20",
                section="So luong co dong",
                table="table-shareholders",
            ),
        )
        plan = compile_question_plan(question)
        graph = build_evidence_graph(candidates)
        selected = verify_program(plan, ReasoningProgram("SELECT", (0,)), graph)
        wrong_entity = verify_program(plan, ReasoningProgram("SELECT", (1,)), graph)
        self.assertEqual(_state(selected, "query_metric"), VerificationState.VALID)
        self.assertIsNot(_state(wrong_entity, "query_metric"), VerificationState.VALID)

    def test_wrong_metric_counterfactual_is_invalid(self) -> None:
        question = "Thu nhap lai thuan tu hoat dong dich vu nam 2023 la bao nhieu?"
        candidates = (_candidate(0, "No phai tra", "60", section="Tai san"),)
        plan = compile_question_plan(question)
        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            build_evidence_graph(candidates),
        )
        self.assertEqual(_state(result, "query_metric"), VerificationState.INVALID)

    def test_candidate_permutation_preserves_add_and_query_decisions(self) -> None:
        add_question = "Cong doanh thu thuan voi loi nhuan gop trong nam 2023."
        add_candidates = (
            _candidate(0, "Doanh thu thuan", "60"),
            _candidate(1, "Loi nhuan gop", "20"),
            _candidate(2, "No phai tra", "90"),
        )
        add_plan = compile_question_plan(add_question)
        original = verify_program(
            add_plan,
            ReasoningProgram("ADD", (0, 1)),
            build_evidence_graph(add_candidates),
        )
        reversed_candidates = tuple(
            replace(candidate, index=index)
            for index, candidate in enumerate(reversed(add_candidates))
        )
        permuted = verify_program(
            add_plan,
            ReasoningProgram("ADD", (2, 1)),
            build_evidence_graph(reversed_candidates),
        )
        self.assertEqual(original.proof_contract_state, permuted.proof_contract_state)

    def test_missing_period_evidence_remains_unknown(self) -> None:
        candidate = _candidate(0, "Doanh thu", "60", year="")
        plan = compile_question_plan("Doanh thu nam 2023 la bao nhieu?")
        result = verify_program(
            plan,
            ReasoningProgram("SELECT", (0,)),
            build_evidence_graph((candidate,)),
        )
        self.assertEqual(_state(result, "period_coverage"), VerificationState.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
