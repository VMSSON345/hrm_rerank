from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.validate_qp4_post_unit_scale_development_baseline_refresh_precommit import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "finhrm_qp4_post_unit_scale_development_baseline_refresh_precommit_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp4_post_unit_scale_development_baseline_refresh_precommit_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp4_post_unit_scale_development_baseline_refresh_precommit_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4PostUnitScaleBaselineRefreshPrecommitTests(unittest.TestCase):
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

    def test_scope_is_measurement_only_with_no_runtime_whitelist(self) -> None:
        scope = self.config["refresh_scope"]
        self.assertEqual(scope["stage"], "measurement_only_no_runtime_change")
        self.assertEqual(scope["runtime_file_whitelist"], [])
        self.assertTrue(scope["evaluate_all_frozen_eligible_records"])
        self.assertTrue(scope["historical_artifacts_are_immutable"])

    def test_compiler_tier_is_frozen_at_full_qp4_inventory(self) -> None:
        compiler = self.config["refresh_scope"]["compiler_tier"]
        self.assertEqual(
            compiler,
            {
                "core_answer_families": 13,
                "ambiguity_controls": 3,
                "surface_variants": 91,
                "semantic_minimal_pairs": 57,
                "synthetic_layout_confirmation": False,
            },
        )

    def test_two_source_tiers_are_diverse_and_complete(self) -> None:
        scope = self.config["refresh_scope"]
        add = scope["add_source_tier"]
        actual = scope["actual_plan_source_tier"]
        self.assertEqual((add["anchors"], add["companies"], add["sources"], add["real_layouts"]), (5, 5, 5, 3))
        self.assertEqual(add["total_queries"], 50)
        self.assertEqual((actual["anchors"], actual["companies"], actual["sources"], actual["document_layouts"], actual["table_layouts"]), (4, 4, 4, 3, 4))
        self.assertTrue(actual["company_disjoint_from_add_diagnosis"])
        self.assertEqual(actual["total_queries"], 24)
        self.assertEqual(scope["combined_source_queries"], 74)

    def test_source_execution_boundary_and_fail_closed_policy_are_exact(self) -> None:
        scope = self.config["refresh_scope"]
        self.assertEqual(scope["source_run_boundary"], ["question", "source PDF path"])
        self.assertEqual(scope["per_query_timeout_seconds"], 15)
        self.assertTrue(scope["timeout_is_failure"])
        self.assertFalse(scope["mineru_auto_generate"])
        self.assertTrue(scope["existing_development_cache_only"])
        self.assertFalse(scope["gold_candidate_or_program_passed_to_engine"])
        self.assertFalse(scope["safe_abstention_counts_as_answer"])
        self.assertFalse(scope["candidate_rank_score_or_confidence_is_semantic_proof"])

    def test_refresh_gates_remain_strict(self) -> None:
        gates = self.config["refresh_gates"]
        self.assertEqual(gates["compiler_surface_semantic_invariance"], "91/91")
        self.assertEqual(gates["compiler_counterfactual_detection"], "57/57")
        self.assertEqual(gates["add_source_surface_winner_invariance"], "35/35")
        self.assertTrue(all(value == 0 for value in gates["add_source_counterfactual_old_winner_false_accepts"].values()))
        self.assertTrue(all(value == "4/4" for value in gates["actual_plan_paired_gold_answers"].values()))
        self.assertTrue(all(value == 0 for value in gates["actual_plan_old_winner_false_accepts"].values()))
        self.assertEqual(gates["safe_actions"], {"ANSWER": 257, "ABSTAIN": 92, "REJECT": 14})
        self.assertEqual(gates["safe_execution"], "363/363")
        self.assertEqual(gates["safe_permutation"], "363/363")
        self.assertEqual(gates["safe_counterfactuals"], 141)
        self.assertEqual(gates["safe_counterfactual_false_accepts"], 0)

    def test_precommit_does_not_execute_refresh_or_open_later_phases(self) -> None:
        self.assertFalse(self.config["runtime_changed_by_this_precommit"])
        self.assertFalse(self.config["refresh_executed_by_this_precommit"])
        self.assertFalse(self.config["downstream_runtime_change_authorized_in_same_step"])
        for name, value in self.config["invariants"].items():
            self.assertFalse(value, name)


class QP4PostUnitScaleBaselineRefreshPrecommitReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (REPORT.exists() and LOCK.exists()):
            raise unittest.SkipTest("Generated baseline-refresh precommit artifacts are absent")
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_exactly_reproducible(self) -> None:
        self.assertEqual(build_report(CONFIG), self.report)

    def test_report_passes_without_running_refresh_or_changing_runtime(self) -> None:
        self.assertTrue(self.report["precommit_gate_passed"])
        self.assertEqual(self.report["failed_gates"], [])
        self.assertTrue(self.report["unit_scale_gate_reclosed"])
        self.assertTrue(self.report["historical_baselines_preserved"])
        self.assertFalse(self.report["refresh_executed"])
        self.assertFalse(self.report["runtime_changed"])
        self.assertFalse(self.report["qp4_complete"])
        self.assertFalse(self.report["qp5_authorized"])
        self.assertEqual(self.report["phase_d"], "locked")

    def test_lock_is_exact_and_opens_only_frozen_refresh(self) -> None:
        self.assertEqual(self.lock["status"], "complete")
        self.assertTrue(self.lock["precommit_gate_passed"])
        self.assertFalse(self.lock["refresh_executed"])
        self.assertEqual(self.lock["runtime_file_whitelist_for_next_step"], [])
        self.assertTrue(self.lock["historical_baselines_preserved"])
        self.assertTrue(self.lock["authorized_next_step"].startswith("QP4-POST-UNIT-SCALE-DEVELOPMENT-BASELINE-REFRESH"))
        self.assertFalse(self.lock["qp4_complete"])
        self.assertFalse(self.lock["qp5_authorized"])
        self.assertEqual(self.lock["phase_d"], "locked")
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
