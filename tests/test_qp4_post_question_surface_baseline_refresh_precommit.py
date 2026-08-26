from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.validate_qp4_post_question_surface_baseline_refresh_precommit import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "finhrm_qp4_post_question_surface_baseline_refresh_precommit_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_refresh_precommit_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_refresh_precommit_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4PostQuestionSurfaceBaselineRefreshPrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_parent_chain_runtime_and_checkpoint_are_exact(self) -> None:
        for parent in self.config["parents"].values():
            self.assertEqual(_digest(ROOT / parent["path"]), parent["sha256"])
        for relative, expected in self.config["frozen_runtime_hashes"].items():
            self.assertEqual(_digest(ROOT / relative), expected)
        checkpoint = self.config["checkpoint"]
        self.assertEqual(_digest(ROOT / checkpoint["path"]), checkpoint["sha256"])

    def test_scope_is_measurement_only_and_has_no_runtime_whitelist(self) -> None:
        scope = self.config["refresh_scope"]
        self.assertEqual(scope["stage"], "measurement_only_no_runtime_change")
        self.assertEqual(scope["runtime_file_whitelist"], [])
        self.assertTrue(scope["historical_artifacts_are_immutable"])
        self.assertFalse(self.config["runtime_changed_by_this_precommit"])
        self.assertFalse(self.config["refresh_executed_by_this_precommit"])
        self.assertFalse(self.config["downstream_runtime_change_authorized_in_same_step"])

    def test_generalization_contract_forbids_cherry_picking(self) -> None:
        contract = self.config["generalization_contract"]
        self.assertTrue(contract["same_contract_for_every_frozen_eligible_record"])
        self.assertTrue(contract["evaluate_all_frozen_eligible_records"])
        self.assertTrue(contract["company_and_layout_group_metrics_required"])
        self.assertTrue(contract["macro_and_worst_group_metrics_required"])
        self.assertTrue(contract["errors_timeouts_and_abstentions_remain_in_denominator"])
        self.assertFalse(contract["report_specific_selection_or_exclusion"])
        self.assertFalse(contract["company_filename_page_year_value_example_or_question_literal_rules"])
        self.assertFalse(contract["synthetic_layout_counts_as_real_confirmation"])
        self.assertFalse(contract["unseen_or_universal_generalization_claim_authorized"])
        self.assertTrue(contract["fail_closed_outside_proven_support"])

    def test_full_compiler_and_question_surface_inventory_is_frozen(self) -> None:
        compiler = self.config["refresh_scope"]["compiler_tier"]
        self.assertEqual(
            compiler,
            {
                "core_answer_families": 13,
                "ambiguity_controls": 3,
                "frozen_surface_questions": 91,
                "corrected_hold_ahead_questions": 78,
                "semantic_minimal_pairs": 57,
            },
        )
        self.assertEqual(self.config["measurement_completion_gates"]["compiler_records_scored"], "242/242")

    def test_two_real_source_tiers_are_complete_and_diverse(self) -> None:
        scope = self.config["refresh_scope"]
        add = scope["add_source_tier"]
        actual = scope["actual_plan_source_tier"]
        self.assertEqual((add["anchors"], add["companies"], add["source_documents"], add["real_layout_families"]), (5, 5, 5, 3))
        self.assertEqual((add["surface_queries"], add["counterfactual_queries"], add["total_queries"]), (35, 15, 50))
        self.assertEqual(
            (actual["anchors"], actual["companies"], actual["source_documents"], actual["document_layout_families"], actual["table_layout_families"]),
            (4, 4, 4, 3, 4),
        )
        self.assertTrue(actual["company_disjoint_from_add_diagnosis"])
        self.assertEqual(actual["total_queries"], 24)
        self.assertEqual(scope["combined_source_queries"], 74)

    def test_all_residuals_and_execution_failures_remain_in_denominator(self) -> None:
        scope = self.config["refresh_scope"]
        self.assertEqual(
            scope["add_source_tier"]["remaining_surface_residuals"],
            {
                "SHORT_LONG_EQUIVALENT": 5,
                "DISCOURSE_PREFIX_SUFFIX": 5,
                "CLAUSE_PERMUTATION": 5,
            },
        )
        self.assertTrue(scope["every_query_must_be_attempted"])
        self.assertTrue(scope["execution_error_or_timeout_is_failed_observation"])
        self.assertTrue(scope["failed_observation_may_not_be_dropped"])
        self.assertFalse(scope["safe_abstention_counts_as_answer"])
        self.assertFalse(scope["mineru_auto_generate"])

    def test_refresh_completion_does_not_claim_qp4_pass(self) -> None:
        semantics = self.config["status_semantics"]
        self.assertTrue(semantics["refresh_complete_requires_every_frozen_record_attempted_and_scored"])
        self.assertTrue(semantics["refresh_complete_does_not_imply_qp4_gates_passed"])
        self.assertTrue(semantics["qp4_gates_passed_requires_every_semantic_gate"])
        self.assertTrue(semantics["residuals_must_be_reported_not_forced_to_answer"])
        self.assertEqual(semantics["missing_proof_action"], "ABSTAIN")
        self.assertEqual(semantics["contradictory_proof_action"], "REJECT")

    def test_semantic_safety_gates_remain_strict(self) -> None:
        gates = self.config["qp4_semantic_gates"]
        self.assertEqual(gates["compiler_surface_semantic_invariance"], "91/91")
        self.assertEqual(gates["compiler_hold_ahead_semantic_invariance"], "78/78")
        self.assertEqual(gates["compiler_counterfactual_detection"], "57/57")
        self.assertEqual(gates["add_source_prior_passes_retained"], "20/20")
        self.assertEqual(gates["add_source_surface_winner_invariance"], "35/35")
        self.assertTrue(all(value == 0 for value in gates["add_source_counterfactual_old_winner_false_accepts"].values()))
        self.assertTrue(all(value == "4/4" for value in gates["actual_plan_paired_gold_answers"].values()))
        self.assertTrue(all(value == 0 for value in gates["actual_plan_old_winner_false_accepts"].values()))
        self.assertEqual(gates["safe_actions"], {"ANSWER": 257, "ABSTAIN": 92, "REJECT": 14})
        self.assertEqual(gates["safe_execution"], "363/363")
        self.assertEqual(gates["safe_permutation"], "363/363")
        self.assertEqual(gates["safe_counterfactuals"], 141)
        self.assertEqual(gates["safe_counterfactual_false_accepts"], 0)

    def test_later_phases_remain_locked(self) -> None:
        for name, value in self.config["invariants"].items():
            self.assertFalse(value, name)
        self.assertTrue(self.config["authorized_next_step_after_lock"].startswith("QP4-POST-QUESTION-SURFACE-BASELINE-REFRESH"))


class QP4PostQuestionSurfaceBaselineRefreshPrecommitReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (REPORT.exists() and LOCK.exists()):
            raise unittest.SkipTest("Generated post-question-surface refresh precommit artifacts are absent")
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_exactly_reproducible(self) -> None:
        self.assertEqual(build_report(CONFIG), self.report)

    def test_report_passes_without_running_refresh_or_changing_runtime(self) -> None:
        self.assertTrue(self.report["precommit_gate_passed"])
        self.assertEqual(self.report["failed_gates"], [])
        self.assertFalse(self.report["refresh_executed"])
        self.assertFalse(self.report["runtime_changed"])
        self.assertFalse(self.report["qp4_complete"])
        self.assertFalse(self.report["qp5_authorized"])
        self.assertEqual(self.report["phase_d"], "locked")

    def test_lock_opens_only_the_exact_measurement_refresh(self) -> None:
        self.assertEqual(self.lock["status"], "complete")
        self.assertTrue(self.lock["precommit_gate_passed"])
        self.assertFalse(self.lock["refresh_executed"])
        self.assertFalse(self.lock["runtime_changed"])
        self.assertEqual(self.lock["runtime_file_whitelist_for_next_step"], [])
        self.assertTrue(self.lock["same_contract_for_every_frozen_eligible_record"])
        self.assertFalse(self.lock["report_specific_selection_or_exclusion"])
        self.assertTrue(self.lock["authorized_next_step"].startswith("QP4-POST-QUESTION-SURFACE-BASELINE-REFRESH"))
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
