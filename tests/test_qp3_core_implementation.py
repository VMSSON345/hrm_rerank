from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace
import unittest

from hrm_finance_qa.document_hrm import (
    _prediction_required_proof_rejection_reasons,
)
from hrm_finance_qa.evidence_schema import EvidenceGraph, build_evidence_graph
from hrm_finance_qa.program_data import (
    NumericCandidate,
    ProgramDAGInput,
    execute_program_dag,
)
from hrm_finance_qa.program_synthesis import synthesize_typed_programs
from hrm_finance_qa.question_planner import compile_question_plan
from hrm_finance_qa.symbolic_verifier import VerificationState, verify_program


def _candidate(
    index: int,
    value: str,
    metric: str,
    year: str,
    role: str,
    *,
    source_name: str = "unseen-company-layout.pdf",
    table_id: str = "table:core",
) -> NumericCandidate:
    column_id = f"column:{role.lower().replace(' ', '_')}:{year}"
    return NumericCandidate(
        index=index,
        value=value,
        raw=value,
        context=(
            f"ROW: {metric}\n"
            f"COLUMN: {role} {year}\n"
            f"PERIOD: {year}\n"
            f"TABLE: {table_id}\n"
            f"ROW_ID: row:{metric.lower().replace(' ', '_')}\n"
            f"COLUMN_ID: {column_id}"
        ),
        line_number=10 + index,
        char_start=100 + index * 10,
        char_end=105 + index * 10,
        unit="VND",
        is_percent=False,
        is_year_like=False,
        relevance=1.0,
        source_name=source_name,
        page=1,
        extraction_method="native",
    )


def _shortfall_fixture():
    question = (
        "Doanh thu hoạt động thực hiện năm 2023 còn thiếu bao nhiêu "
        "phần trăm để đạt kế hoạch năm 2023?"
    )
    candidates = (
        _candidate(0, "100", "Doanh thu hoạt động", "2023", "Kế hoạch"),
        _candidate(1, "80", "Doanh thu hoạt động", "2023", "Thực hiện"),
    )
    plan = compile_question_plan(question)
    graph = build_evidence_graph(candidates)
    synthesis = synthesize_typed_programs(
        plan, plan.operation or "", graph, focus_text=plan.metric_text
    )
    return question, candidates, plan, graph, synthesis.proposals[0]


class QP3CoreImplementationTests(unittest.TestCase):
    def test_typed_slots_lower_to_authoritative_multistep_dag(self) -> None:
        _, candidates, plan, graph, proposal = _shortfall_fixture()
        self.assertEqual(proposal.program_dag.nodes[0].operation, "SUBTRACT")
        self.assertEqual(proposal.program_dag.nodes[-1].operation, "RATIO_PERCENT")
        self.assertEqual(execute_program_dag(proposal.program_dag, candidates), Decimal("20.0"))

        verification = verify_program(
            plan,
            proposal.program,
            graph,
            program_dag=proposal.program_dag,
            slot_bindings=proposal.slot_bindings,
        )
        self.assertEqual(verification.proof_contract_state, VerificationState.VALID)
        required = {
            item.name: item.state
            for item in verification.proof_obligations
            if item.applicability == "required"
        }
        for obligation in (
            "reference_base_match",
            "logical_cell_unique",
            "source_provenance_complete",
            "all_leaf_nodes_valid",
            "all_intermediate_nodes_valid",
            "dag_execution_valid",
        ):
            self.assertEqual(required[obligation], VerificationState.VALID)

    def test_typed_column_roles_prove_subtract_direction_without_value_sign(self) -> None:
        question = (
            "Doanh thu hoạt động thực hiện năm 2023 trừ kế hoạch năm 2023 "
            "là bao nhiêu?"
        )
        candidates = (
            _candidate(0, "80", "Doanh thu hoạt động", "2023", "Thực hiện"),
            _candidate(1, "100", "Doanh thu hoạt động", "2023", "Kế hoạch"),
        )
        plan = compile_question_plan(question)
        graph = build_evidence_graph(candidates)
        proposal = synthesize_typed_programs(
            plan, plan.operation or "", graph, focus_text=plan.metric_text
        ).proposals[0]
        verification = verify_program(
            plan,
            proposal.program,
            graph,
            program_dag=proposal.program_dag,
            slot_bindings=proposal.slot_bindings,
        )
        states = {item.name: item.state for item in verification.proof_obligations}
        self.assertEqual(states["temporal_direction"], VerificationState.VALID)
        self.assertEqual(verification.proof_contract_state, VerificationState.VALID)

    def test_decrease_amount_uses_typed_roles_before_same_year_fallback(self) -> None:
        question = (
            "Doanh thu hoạt động thực hiện năm 2023 thấp hơn kế hoạch năm 2023 "
            "bao nhiêu đồng?"
        )
        candidates = (
            _candidate(0, "100", "Doanh thu hoạt động", "2023", "Kế hoạch"),
            _candidate(1, "80", "Doanh thu hoạt động", "2023", "Thực hiện"),
        )
        plan = compile_question_plan(question)
        graph = build_evidence_graph(candidates)
        proposal = synthesize_typed_programs(
            plan, plan.operation or "", graph, focus_text=plan.metric_text
        ).proposals[0]
        verification = verify_program(
            plan,
            proposal.program,
            graph,
            program_dag=proposal.program_dag,
            slot_bindings=proposal.slot_bindings,
        )
        states = {item.name: item.state for item in verification.proof_obligations}
        self.assertEqual(states["temporal_direction"], VerificationState.VALID)
        self.assertEqual(verification.proof_contract_state, VerificationState.VALID)

    def test_wrong_reference_base_is_invalid_even_when_dag_executes(self) -> None:
        _, _, plan, graph, proposal = _shortfall_fixture()
        final = proposal.program_dag.nodes[-1]
        actual_index = dict(proposal.slot_bindings)["actual"]
        wrong_final = replace(
            final,
            inputs=(final.inputs[0], ProgramDAGInput.candidate(actual_index)),
        )
        wrong_dag = replace(
            proposal.program_dag,
            nodes=(*proposal.program_dag.nodes[:-1], wrong_final),
        )
        verification = verify_program(
            plan,
            proposal.program,
            graph,
            program_dag=wrong_dag,
            slot_bindings=proposal.slot_bindings,
        )
        states = {item.name: item.state for item in verification.proof_obligations}
        self.assertEqual(states["reference_base_match"], VerificationState.INVALID)
        self.assertEqual(verification.proof_contract_state, VerificationState.INVALID)

    def test_missing_provenance_abstains_and_duplicate_logical_cell_rejects(self) -> None:
        question, candidates, plan, graph, proposal = _shortfall_fixture()
        missing_cells = (replace(graph.cells[0], source_span=None), *graph.cells[1:])
        missing_graph = EvidenceGraph(cells=missing_cells, edges=graph.edges)
        missing = verify_program(
            plan,
            proposal.program,
            missing_graph,
            program_dag=proposal.program_dag,
            slot_bindings=proposal.slot_bindings,
        )
        missing_states = {item.name: item.state for item in missing.proof_obligations}
        self.assertEqual(
            missing_states["source_provenance_complete"], VerificationState.UNKNOWN
        )
        self.assertEqual(missing.proof_contract_state, VerificationState.UNKNOWN)

        duplicate = replace(
            graph.cells[0],
            index=2,
            source_span=(900, 905),
        )
        duplicate_graph = EvidenceGraph(cells=(*graph.cells, duplicate), edges=graph.edges)
        duplicated = verify_program(
            plan,
            proposal.program,
            duplicate_graph,
            program_dag=proposal.program_dag,
            slot_bindings=proposal.slot_bindings,
        )
        duplicate_states = {item.name: item.state for item in duplicated.proof_obligations}
        self.assertEqual(
            duplicate_states["logical_cell_unique"], VerificationState.INVALID
        )
        self.assertEqual(duplicated.proof_contract_state, VerificationState.INVALID)

        prediction = SimpleNamespace(
            program=proposal.program,
            program_dag=proposal.program_dag,
            slot_bindings=proposal.slot_bindings,
            grounding_constraints_applied=(),
        )
        reasons = _prediction_required_proof_rejection_reasons(
            question, prediction, tuple(candidates)
        )
        self.assertEqual(reasons, ())

    def test_leaf_role_and_year_counterfactuals_are_invalid(self) -> None:
        _, _, plan, graph, proposal = _shortfall_fixture()
        role_swapped = replace(graph.cells[0], column_role="actual")
        role_graph = EvidenceGraph(cells=(role_swapped, graph.cells[1]), edges=graph.edges)
        role_result = verify_program(
            plan,
            proposal.program,
            role_graph,
            program_dag=proposal.program_dag,
            slot_bindings=proposal.slot_bindings,
        )
        self.assertEqual(role_result.proof_contract_state, VerificationState.INVALID)

        year_swapped = replace(
            graph.cells[0], explicit_years=("2022",), period="2022"
        )
        year_graph = EvidenceGraph(cells=(year_swapped, graph.cells[1]), edges=graph.edges)
        year_result = verify_program(
            plan,
            proposal.program,
            year_graph,
            program_dag=proposal.program_dag,
            slot_bindings=proposal.slot_bindings,
        )
        states = {item.name: item.state for item in year_result.proof_obligations}
        self.assertEqual(states["all_leaf_nodes_valid"], VerificationState.INVALID)

    def test_partial_multi_output_is_required_unknown(self) -> None:
        _, _, plan, graph, proposal = _shortfall_fixture()
        multi_plan = replace(
            plan,
            answer_mode="MULTI_OUTPUT_ALL_OR_ABSTAIN",
            requested_output_count=2,
        )
        verification = verify_program(
            multi_plan,
            proposal.program,
            graph,
            program_dag=proposal.program_dag,
            slot_bindings=proposal.slot_bindings,
        )
        states = {item.name: item.state for item in verification.proof_obligations}
        self.assertEqual(states["answer_mode_complete"], VerificationState.UNKNOWN)
        self.assertEqual(verification.proof_contract_state, VerificationState.UNKNOWN)

    def test_interest_bearing_debt_uses_three_leaves_and_two_nodes(self) -> None:
        question = (
            "Nợ vay ngắn hạn và nợ vay dài hạn chịu lãi năm 2023 chiếm bao "
            "nhiêu phần trăm tổng nguồn vốn năm 2023?"
        )
        candidates = (
            _candidate(0, "20", "Vay và nợ thuê tài chính ngắn hạn", "2023", "Thực hiện"),
            _candidate(1, "30", "Vay và nợ thuê tài chính dài hạn", "2023", "Thực hiện"),
            _candidate(2, "100", "Tổng nguồn vốn", "2023", "Thực hiện"),
        )
        plan = compile_question_plan(question)
        graph = build_evidence_graph(candidates)
        synthesis = synthesize_typed_programs(
            plan, plan.operation or "", graph, focus_text=plan.metric_text
        )
        self.assertEqual(synthesis.status, "ok")
        proposal = synthesis.proposals[0]
        self.assertEqual(proposal.program_dag.leaf_operands, (0, 1, 2))
        self.assertEqual(execute_program_dag(proposal.program_dag, candidates), Decimal("50.0"))
        verification = verify_program(
            plan,
            proposal.program,
            graph,
            program_dag=proposal.program_dag,
            slot_bindings=proposal.slot_bindings,
        )
        self.assertEqual(verification.proof_contract_state, VerificationState.VALID)

    def test_candidate_serialization_permutation_preserves_result(self) -> None:
        _, candidates, plan, _, proposal = _shortfall_fixture()
        reversed_candidates = (
            replace(candidates[1], index=0),
            replace(candidates[0], index=1),
        )
        reversed_graph = build_evidence_graph(reversed_candidates)
        reversed_result = synthesize_typed_programs(
            plan, plan.operation or "", reversed_graph, focus_text=plan.metric_text
        )
        reversed_proposal = reversed_result.proposals[0]
        self.assertEqual(
            execute_program_dag(proposal.program_dag, candidates),
            execute_program_dag(reversed_proposal.program_dag, reversed_candidates),
        )
        verification = verify_program(
            plan,
            reversed_proposal.program,
            reversed_graph,
            program_dag=reversed_proposal.program_dag,
            slot_bindings=reversed_proposal.slot_bindings,
        )
        self.assertEqual(verification.proof_contract_state, VerificationState.VALID)

    def test_question_side_ambiguity_is_consumed_by_final_proof_gate(self) -> None:
        question = "Doanh thu năm 2023 lớn hơn bao nhiêu phần trăm?"
        candidate = _candidate(0, "80", "Doanh thu", "2023", "Thực hiện")
        prediction = SimpleNamespace(
            program=SimpleNamespace(operation="SELECT", operands=(0,)),
            program_dag=None,
            slot_bindings=(),
            grounding_constraints_applied=(),
        )
        reasons = _prediction_required_proof_rejection_reasons(
            question, prediction, (candidate,)
        )
        self.assertEqual(reasons, ("question_abstention:missing_reference_base",))


if __name__ == "__main__":
    unittest.main()
