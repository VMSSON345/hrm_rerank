from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from hrm_finance_qa.question_planner import (
    compile_question_plan,
    compile_question_proof_contract,
)
from scripts.validate_qp4_unit_scale_source_residual_isolation_precommit import (
    build_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "finhrm_qp4_unit_scale_source_residual_isolation_precommit_v1.json"
TRACE = ROOT / "reports" / "finhrm_qp4_unit_scale_source_residual_trace_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp4_unit_scale_source_residual_isolation_precommit_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp4_unit_scale_source_residual_isolation_precommit_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4UnitScaleSourceResidualIsolationPrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.trace = json.loads(TRACE.read_text(encoding="utf-8"))

    def test_trace_is_company_source_and_layout_diverse(self) -> None:
        self.assertEqual(
            self.trace["coverage"],
            {"runs": 5, "companies": 5, "sources": 5, "real_layouts": 3},
        )
        self.assertEqual(self.trace["abw_records"], 0)
        self.assertEqual(self.trace["protected_records"], [])

    def test_all_four_answers_reproduce_the_same_proof_boundary_gap(self) -> None:
        summary = self.trace["summary"]
        self.assertEqual(summary["planner_typed"], 5)
        self.assertEqual(summary["answers"], 4)
        self.assertEqual(summary["old_winner_false_accepts"], 4)
        self.assertEqual(summary["answered_with_raw_unit_scale_preserved"], 4)
        self.assertEqual(summary["answered_with_unit_match_valid"], 4)
        self.assertEqual(summary["answered_with_unit_scale_obligation_declared"], 0)
        self.assertEqual(summary["answered_with_unit_scale_match_state"], 0)
        self.assertEqual(summary["answered_with_all_required_proofs_valid"], 4)

    def test_current_contract_reproduces_missing_scale_obligation(self) -> None:
        plan = compile_question_plan(
            "Tổng tài sản ngắn hạn và tài sản dài hạn theo đơn vị triệu đồng "
            "là bao nhiêu?"
        )
        contract = compile_question_proof_contract(plan, "ADD")
        self.assertEqual(plan.requested_scale, "million_vnd")
        self.assertNotIn("unit_scale_match", contract.required)
        self.assertNotIn("unit_scale_match", contract.optional)
        self.assertNotIn("unit_scale_match", contract.not_applicable)

    def test_attribution_stops_before_retrieval_binding_and_engine(self) -> None:
        attribution = self.config["layer_attribution"]
        self.assertTrue(attribution["planner_requested_scale_retained"])
        self.assertTrue(attribution["evidence_raw_unit_scale_retained"])
        self.assertFalse(attribution["retrieval_or_binding_winner_changed"])
        self.assertEqual(
            attribution["first_missing_semantic_boundary"],
            "question_proof_contract",
        )
        self.assertEqual(
            attribution["first_missing_runtime_assessment"], "symbolic_verifier"
        )
        self.assertTrue(
            attribution["engine_action_policy_already_fails_closed_on_required_unknown"]
        )

    def test_next_runtime_whitelist_and_symbols_are_exact(self) -> None:
        scope = self.config["next_implementation_scope"]
        self.assertEqual(
            scope["runtime_file_whitelist"],
            [
                "src/hrm_finance_qa/question_planner.py",
                "src/hrm_finance_qa/symbolic_verifier.py",
            ],
        )
        self.assertEqual(scope["new_proof_obligation"], "unit_scale_match")
        self.assertEqual(
            scope["exact_symbols"]["src/hrm_finance_qa/question_planner.py"],
            ["_PROOF_CONSTRAINTS", "compile_question_proof_contract"],
        )
        self.assertEqual(
            scope["exact_symbols"]["src/hrm_finance_qa/symbolic_verifier.py"],
            ["currency_unit_scale import", "_unit_scale_match_state", "verify_program"],
        )

    def test_future_contract_distinguishes_unknown_from_invalid(self) -> None:
        rules = " ".join(self.config["next_implementation_scope"]["semantic_contract"])
        self.assertIn("Missing raw evidence scale yields UNKNOWN", rules)
        self.assertIn("different but convertible explicit evidence scale", rules)
        self.assertIn("UNKNOWN and therefore ABSTAIN", rules)
        self.assertIn("Mixed explicit operand currency scales", rules)
        self.assertIn("INVALID and therefore REJECT", rules)
        self.assertIn("requested_scale=conflict", rules)

    def test_conversion_retrieval_binding_synthesis_engine_and_renderer_are_non_goals(self) -> None:
        non_goals = self.config["next_implementation_scope"]["non_goals"]
        for item in (
            "unit conversion execution",
            "retrieval changes",
            "candidate binding changes",
            "program synthesis changes",
            "evidence schema changes",
            "document reasoning engine changes",
            "answer renderer changes",
        ):
            self.assertIn(item, non_goals)

    def test_parent_runtime_and_checkpoint_hashes_are_exact(self) -> None:
        for parent in self.config["parents"].values():
            self.assertEqual(_digest(ROOT / parent["path"]), parent["sha256"])
        for group in (
            "authorized_runtime_hashes_before_next_implementation",
            "frozen_runtime_hashes",
        ):
            for relative, expected in self.config[group].items():
                self.assertEqual(_digest(ROOT / relative), expected)
        checkpoint = self.config["checkpoint"]
        self.assertEqual(_digest(ROOT / checkpoint["path"]), checkpoint["sha256"])

    def test_precommit_changes_no_runtime_and_keeps_all_later_gates_closed(self) -> None:
        self.assertFalse(self.config["runtime_changed_by_this_precommit"])
        self.assertFalse(
            self.config["downstream_runtime_change_authorized_in_same_step"]
        )
        for name, value in self.config["invariants"].items():
            self.assertFalse(value, name)


class QP4UnitScaleSourceResidualIsolationPrecommitReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (REPORT.exists() and LOCK.exists()):
            raise unittest.SkipTest("Generated residual-isolation precommit artifacts are absent")
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_exactly_reproducible(self) -> None:
        self.assertEqual(build_report(CONFIG), self.report)

    def test_report_and_lock_pass_without_runtime_change(self) -> None:
        self.assertTrue(self.report["precommit_gate_passed"])
        self.assertEqual(self.report["failed_gates"], [])
        self.assertFalse(self.report["runtime_changed"])
        self.assertFalse(
            self.report["downstream_runtime_change_authorized_in_same_step"]
        )
        self.assertEqual(self.lock["status"], "complete")
        self.assertTrue(self.lock["precommit_gate_passed"])
        self.assertFalse(self.lock["runtime_changed"])
        self.assertFalse(self.lock["downstream_runtime_change_authorized_in_same_step"])
        self.assertFalse(self.lock["conversion_retrieval_or_binding_edit_authorized"])
        self.assertFalse(self.lock["qp4_complete"])
        self.assertFalse(self.lock["qp5_authorized"])
        self.assertEqual(self.lock["phase_d"], "locked")

    def test_lock_artifact_hashes_are_exact(self) -> None:
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
