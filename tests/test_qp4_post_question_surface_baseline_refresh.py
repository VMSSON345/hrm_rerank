from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.evaluate_qp4_post_question_surface_baseline_refresh import (
    ACTUAL_PLAN_RAW,
    ADD_RAW,
    LOCK,
    MARKDOWN,
    PRECOMMIT,
    REPORT,
    ROOT,
    SAFE_REPLAY,
    TEST,
    build_report,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4PostQuestionSurfaceBaselineRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (ADD_RAW, ACTUAL_PLAN_RAW, SAFE_REPLAY, REPORT, LOCK)
        if not all(path.exists() for path in required):
            raise unittest.SkipTest("Generated post-question-surface refresh artifacts are absent")
        cls.config = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
        cls.add_raw = json.loads(ADD_RAW.read_text(encoding="utf-8"))
        cls.actual_raw = json.loads(ACTUAL_PLAN_RAW.read_text(encoding="utf-8"))
        cls.safe = json.loads(SAFE_REPLAY.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_every_frozen_source_query_is_attempted(self) -> None:
        self.assertEqual(len(self.add_raw["surface_results"]), 35)
        self.assertEqual(len(self.add_raw["counterfactual_results"]), 15)
        self.assertEqual(len(self.actual_raw["results"]), 12)
        self.assertEqual(len(self.add_raw["surface_results"]) + len(self.add_raw["counterfactual_results"]), 50)
        self.assertEqual(len(self.actual_raw["results"]) * 2, 24)
        self.assertEqual(self.config["refresh_scope"]["combined_source_queries"], 74)

    def test_real_source_diversity_is_exact_and_not_synthetic(self) -> None:
        add = self.report["add_source_backed"]
        actual = self.report["actual_plan_source_backed"]["coverage"]
        self.assertEqual((add["companies"], add["source_documents"], add["real_layout_families"]), (5, 5, 3))
        self.assertEqual((actual["companies"], actual["sources"], actual["document_layouts"], actual["table_layouts"]), (4, 4, 3, 4))
        self.assertFalse(self.report["generalization_boundary"]["synthetic_layout_counts_as_real_confirmation"])

    def test_full_compiler_inventory_is_measured(self) -> None:
        compiler = self.report["compiler"]
        self.assertEqual((compiler["canonical"]["exact"], compiler["canonical"]["total"]), (13, 13))
        self.assertEqual((compiler["ambiguity"]["exact"], compiler["ambiguity"]["total"]), (3, 3))
        self.assertEqual((compiler["frozen_surface"]["exact"], compiler["frozen_surface"]["total"]), (91, 91))
        self.assertEqual((compiler["corrected_hold_ahead"]["exact"], compiler["corrected_hold_ahead"]["total"]), (78, 78))
        self.assertEqual((compiler["minimal_pairs"]["detected"], compiler["minimal_pairs"]["total"]), (57, 57))

    def test_errors_timeouts_and_abstentions_are_not_removed(self) -> None:
        errors = self.report["source_errors"]
        self.assertTrue(errors["all_errors_retained_in_denominator"])
        self.assertEqual(
            errors["total_failed_observations"],
            errors["add_surface"]
            + errors["add_counterfactual"]
            + errors["actual_plan_base"]
            + errors["actual_plan_mutated"],
        )
        self.assertFalse(self.config["refresh_scope"]["safe_abstention_counts_as_answer"])

    def test_company_layout_macro_and_worst_group_metrics_are_present(self) -> None:
        groups = self.report["group_metrics"]
        for tier in groups.values():
            for summary in tier.values():
                self.assertGreater(summary["groups"], 0)
                self.assertIn("macro_pass_rate", summary)
                self.assertIn("worst_group_pass_rate", summary)
                self.assertEqual(summary["groups"], len(summary["by_group"]))

    def test_refresh_completion_is_separate_from_qp4_semantic_pass(self) -> None:
        measurement_failed = sorted(name for name, value in self.report["completion_checks"].items() if not value)
        semantic_failed = sorted(name for name, value in self.report["semantic_checks"].items() if not value)
        self.assertEqual(measurement_failed, self.report["failed_measurement_gates"])
        self.assertEqual(semantic_failed, self.report["failed_semantic_gates"])
        self.assertEqual(not measurement_failed, self.report["refresh_complete"])
        self.assertEqual(not measurement_failed and not semantic_failed, self.report["qp4_gates_passed"])

    def test_safe_replay_covers_execution_permutation_and_counterfactual(self) -> None:
        result = self.safe["results"]
        self.assertTrue(self.safe["safe_replay_gate_passed"])
        self.assertEqual(result["gold_execution_correct"], result["gold_total"])
        self.assertEqual((result["permutation_exact"], result["permutation_total"]), (363, 363))
        self.assertEqual(result["counterfactual_generated"], 141)
        self.assertEqual(result["counterfactual_false_accepts"], 0)

    def test_report_is_reproducible_from_raw_measurements(self) -> None:
        rebuilt = build_report(regression=self.report["verification"]["compatible_affected_regression"])
        self.assertEqual(rebuilt, self.report)

    def test_runtime_checkpoint_and_later_phases_remain_locked(self) -> None:
        for relative, expected in self.config["frozen_runtime_hashes"].items():
            self.assertEqual(_digest(ROOT / relative), expected)
        checkpoint = self.config["checkpoint"]
        self.assertEqual(_digest(ROOT / checkpoint["path"]), checkpoint["sha256"])
        self.assertFalse(self.report["runtime_changed"])
        self.assertFalse(self.report["qp4_complete"])
        self.assertFalse(self.report["qp5_authorized"])
        self.assertEqual(self.report["phase_d"], "locked")
        self.assertEqual(self.lock["runtime_file_whitelist_for_next_step"], [])

    def test_lock_hashes_every_material_artifact(self) -> None:
        artifacts = {row["path"]: row for row in self.lock["artifacts"]}
        for path in (PRECOMMIT, ADD_RAW, ACTUAL_PLAN_RAW, SAFE_REPLAY, REPORT, MARKDOWN, TEST):
            relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
            self.assertIn(relative, artifacts)
            self.assertEqual(_digest(path), artifacts[relative]["sha256"])
            self.assertEqual(path.stat().st_size, artifacts[relative]["size_bytes"])


if __name__ == "__main__":
    unittest.main()
