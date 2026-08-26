from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.evaluate_qp4_question_surface_implementation import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/finhrm_qp4_question_surface_implementation_v1.json"
REPORT = ROOT / "reports/finhrm_qp4_question_surface_implementation_v1.json"
LOCK = ROOT / "reports/finhrm_qp4_question_surface_implementation_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4QuestionSurfaceImplementationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_reproducible_without_source_or_model_rerun(self) -> None:
        self.assertEqual(build_report(self.config, self.report["regression"]), self.report)

    def test_compiler_surface_and_semantic_controls_all_pass(self) -> None:
        compiler = self.report["compiler"]
        self.assertEqual(compiler["canonical"]["exact"], 13)
        self.assertEqual(compiler["ambiguity"]["exact"], 3)
        self.assertEqual(compiler["isolated_residuals"], {"passed": 18, "total": 18})
        self.assertEqual(compiler["frozen_surface"]["exact"], 91)
        self.assertEqual(compiler["corrected_hold_ahead"]["exact"], 78)
        self.assertEqual(compiler["minimal_pairs"], {"detected": 57, "total": 57})

    def test_source_replay_retains_every_prior_pass_and_reports_residuals(self) -> None:
        source = self.report["source_backed"]
        self.assertEqual(source["companies"], 5)
        self.assertEqual(source["source_documents"], 5)
        self.assertEqual(source["real_layout_families"], 3)
        self.assertEqual(source["surface"]["passed"], 20)
        self.assertEqual(source["surface"]["total"], 35)
        self.assertEqual(source["previously_passing_retained"], 20)
        self.assertEqual(source["previously_passing_total"], 20)
        self.assertEqual(
            {row["category"]: row["failures"] for row in source["surface_residuals"]},
            {
                "CLAUSE_PERMUTATION": 5,
                "DISCOURSE_PREFIX_SUFFIX": 5,
                "SHORT_LONG_EQUIVALENT": 5,
            },
        )

    def test_regression_and_safe_replay_pass(self) -> None:
        self.assertTrue(self.report["regression"]["passed"])
        self.assertEqual(self.report["regression"]["observed"], 354)
        self.assertEqual(
            self.report["safe_replay"],
            {
                "actions": {"ABSTAIN": 92, "ANSWER": 257, "REJECT": 14},
                "execution": "363/363",
                "permutation": "363/363",
                "counterfactuals": 141,
                "counterfactual_false_accepts": 0,
            },
        )

    def test_only_question_planner_hash_changed(self) -> None:
        changed = [
            path
            for path, before in self.report["runtime_before"].items()
            if self.report["runtime_after"][path] != before
        ]
        self.assertEqual(changed, ["src/hrm_finance_qa/question_planner.py"])
        for relative, expected in self.report["runtime_after"].items():
            self.assertEqual(_digest(ROOT / relative), expected)
        self.assertEqual(
            self.report["implementation_scope"]["changed_symbols"],
            ["_strip_nonsemantic_metric_surface", "compile_question_plan"],
        )

    def test_lock_closes_only_this_implementation(self) -> None:
        self.assertEqual(self.lock["status"], "complete")
        self.assertTrue(self.lock["implementation_gate_passed"])
        self.assertEqual(self.lock["failed_gates"], [])
        self.assertFalse(self.lock["qp4_complete"])
        self.assertFalse(self.lock["qp5_authorized"])
        self.assertEqual(self.lock["phase_d"], "locked")
        self.assertIn("BASELINE-REFRESH-PRECOMMIT", self.lock["authorized_next_step"])
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
