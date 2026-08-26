from __future__ import annotations

import json
import unittest

from hrm_finance_qa.evidence_schema import EvidenceGraph
from hrm_finance_qa.program_data import ReasoningProgram
from hrm_finance_qa.question_planner import compile_question_plan
from hrm_finance_qa.symbolic_verifier import (
    VerificationState,
    assess_column_role_proof,
    verify_program,
)
from scripts.evaluate_qp3b_column_role_proof import (
    DEFAULT_COHORT,
    _evidence_graph,
    build_report,
)


def _records() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in DEFAULT_COHORT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _scenario(name: str) -> dict[str, object]:
    return next(
        record
        for record in _records()
        if record["group_id"] == "PA01" and record["scenario_id"] == name
    )


class QP3BColumnRoleProofTests(unittest.TestCase):
    def test_frozen_cohort_is_exact(self) -> None:
        report = build_report()
        self.assertTrue(report["implementation_gate_passed"])
        self.assertEqual(report["exactness"]["state_exact"], 128)
        self.assertEqual(report["exactness"]["action_exact"], 128)
        self.assertEqual(report["failures"], [])

    def test_no_role_unique_complete_is_required_and_valid(self) -> None:
        record = _scenario("no_role_unique_complete")
        graph = _evidence_graph(record)
        plan = compile_question_plan(str(record["question"]))
        assessment = assess_column_role_proof(
            plan, graph, selected_indices=(0,), operation="SELECT"
        )
        self.assertEqual(assessment.state, VerificationState.VALID)
        result = verify_program(plan, ReasoningProgram("SELECT", (0,)), graph)
        obligation = next(
            item
            for item in result.proof_obligations
            if item.name == "column_role_unique"
        )
        self.assertEqual(obligation.applicability, "required")
        self.assertEqual(obligation.state, VerificationState.VALID)

    def test_no_role_multi_role_conflict_is_invalid_and_order_invariant(self) -> None:
        states = []
        for scenario in (
            "no_role_multi_role_conflict_complete",
            "no_role_conflict_permuted_complete",
        ):
            record = _scenario(scenario)
            graph = _evidence_graph(record)
            plan = compile_question_plan(str(record["question"]))
            states.append(
                assess_column_role_proof(
                    plan, graph, selected_indices=(0,), operation="SELECT"
                ).state
            )
        self.assertEqual(states, [VerificationState.INVALID] * 2)

    def test_same_value_does_not_collapse_distinct_role_cells(self) -> None:
        record = _scenario("no_role_same_value_multi_role_conflict_complete")
        graph = _evidence_graph(record)
        self.assertEqual(graph.cells[0].value, graph.cells[1].value)
        assessment = assess_column_role_proof(
            compile_question_plan(str(record["question"])),
            graph,
            selected_indices=(0,),
            operation="SELECT",
        )
        self.assertEqual(assessment.state, VerificationState.INVALID)
        self.assertEqual(assessment.action, "ABSTAIN")

    def test_unknown_inventory_is_required_unknown(self) -> None:
        record = _scenario("no_role_unique_inventory_unknown")
        graph = _evidence_graph(record)
        plan = compile_question_plan(str(record["question"]))
        assessment = assess_column_role_proof(
            plan, graph, selected_indices=(0,), operation="SELECT"
        )
        self.assertEqual(assessment.state, VerificationState.UNKNOWN)
        self.assertEqual(assessment.action, "ABSTAIN")
        result = verify_program(plan, ReasoningProgram("SELECT", (0,)), graph)
        obligation = next(
            item
            for item in result.proof_obligations
            if item.name == "column_role_unique"
        )
        self.assertEqual(obligation.applicability, "required")
        self.assertEqual(obligation.state, VerificationState.UNKNOWN)

    def test_complete_inventory_without_an_explicit_role_is_not_valid(self) -> None:
        record = _scenario("no_role_unique_complete")
        record["evidence_cells"][0]["column_role"] = ""
        graph = _evidence_graph(record)
        assessment = assess_column_role_proof(
            compile_question_plan(str(record["question"])),
            graph,
            selected_indices=(0,),
            operation="SELECT",
        )
        self.assertEqual(assessment.state, VerificationState.UNKNOWN)
        self.assertEqual(assessment.action, "ABSTAIN")

    def test_undeclared_runtime_inventory_preserves_legacy_behavior(self) -> None:
        record = _scenario("no_role_unique_complete")
        complete = _evidence_graph(record)
        graph = EvidenceGraph(cells=complete.cells, edges=complete.edges)
        plan = compile_question_plan(str(record["question"]))
        assessment = assess_column_role_proof(
            plan, graph, selected_indices=(0,), operation="SELECT"
        )
        self.assertFalse(assessment.applicable)
        result = verify_program(plan, ReasoningProgram("SELECT", (0,)), graph)
        obligation = next(
            item
            for item in result.proof_obligations
            if item.name == "column_role_unique"
        )
        self.assertEqual(obligation.applicability, "not_applicable")
        self.assertIsNone(obligation.state)

    def test_cross_extractor_conflict_is_invalid_without_complete_inventory(self) -> None:
        record = _scenario("no_role_cross_extractor_role_conflict_complete")
        complete = _evidence_graph(record)
        graph = EvidenceGraph(cells=complete.cells, edges=complete.edges)
        plan = compile_question_plan(str(record["question"]))
        result = verify_program(plan, ReasoningProgram("SELECT", (0,)), graph)
        self.assertIn("ambiguous_column_role_evidence", result.reasons)

    def test_qp3a_locked_artifacts_remain_unchanged(self) -> None:
        report = build_report()
        self.assertEqual(
            report["artifact_hashes"]["precommit"],
            "75cf5124b5107fce112c0e394cb7077c7c7e209b27fe461a23eb86af779d3c9f",
        )
        self.assertEqual(
            report["artifact_hashes"]["cohort"],
            "cfb7c5b38a55a721f81cc06cc4f00602e9df5f145c57590faf3bd5cb6634d1bf",
        )


if __name__ == "__main__":
    unittest.main()
