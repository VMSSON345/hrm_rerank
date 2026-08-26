from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.evaluate_qp4_unit_scale_question_planner_implementation import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "finhrm_qp4_unit_scale_question_planner_implementation_v1.json"
RAW = ROOT / "reports" / "finhrm_qp4_unit_scale_question_planner_source_replay_v1.json"
SAFE = ROOT / "reports" / "finhrm_qp4_unit_scale_question_planner_safe_replay_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp4_unit_scale_question_planner_implementation_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp4_unit_scale_question_planner_implementation_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4UnitScaleQuestionPlannerImplementationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        required = (RAW, SAFE, REPORT, LOCK)
        if not all(path.exists() for path in required):
            raise unittest.SkipTest("Generated UNIT_SCALE implementation artifacts are absent")
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_exactly_reproducible_without_source_or_model_rerun(self) -> None:
        self.assertEqual(build_report(CONFIG, RAW, SAFE), self.report)

    def test_typed_intent_stage_passes_all_compiler_gates(self) -> None:
        self.assertTrue(self.report["typed_intent_stage_passed"])
        self.assertEqual(self.report["failed_gates"], [])
        self.assertEqual(
            self.report["compiler"]["unit_scale_detection"],
            {"detected": 13, "total": 13},
        )
        self.assertEqual(
            self.report["compiler"]["non_unit_semantics_retained"],
            {"passed": 13, "total": 13},
        )
        self.assertEqual(self.report["parser"]["explicit"]["passed"], 7)
        self.assertTrue(self.report["parser"]["missing_scale_is_none"])
        self.assertTrue(self.report["parser"]["conflict_fails_closed"])
        self.assertTrue(self.report["parser"]["incompatible_fails_closed"])

    def test_source_replay_is_diverse_and_residual_is_not_hidden(self) -> None:
        source = self.report["source_replay"]
        self.assertEqual(source["runs"], 5)
        self.assertEqual(source["companies"], 5)
        self.assertEqual(source["sources"], 5)
        self.assertEqual(source["real_layouts"], 3)
        self.assertEqual(source["typed_scale_detected"], 5)
        self.assertEqual(source["old_winner_false_accept_target"], 0)
        self.assertEqual(
            source["target_met"], source["old_winner_false_accepts"] == 0
        )
        self.assertTrue(self.report["source_residual_reported"])
        self.assertFalse(self.report["downstream_runtime_change_authorized"])

    def test_frozen_safe_replay_is_exact(self) -> None:
        self.assertEqual(
            self.report["safe_replay"]["actions"],
            {"ANSWER": 259, "ABSTAIN": 90, "REJECT": 14},
        )
        self.assertEqual(self.report["safe_replay"]["execution"], "363/363")
        self.assertEqual(self.report["safe_replay"]["permutation"], "363/363")
        self.assertEqual(self.report["safe_replay"]["counterfactuals"], 141)
        self.assertEqual(
            self.report["safe_replay"]["counterfactual_false_accepts"], 0
        )

    def test_only_question_planner_runtime_hash_changed(self) -> None:
        change = self.config["runtime_change"]
        self.assertEqual(change["whitelist"], ["src/hrm_finance_qa/question_planner.py"])
        self.assertNotEqual(change["before_sha256"], change["after_sha256"])
        self.assertEqual(
            _digest(ROOT / change["whitelist"][0]), change["after_sha256"]
        )
        for relative, expected in self.config["frozen_runtime_hashes"].items():
            self.assertEqual(_digest(ROOT / relative), expected)

    def test_lock_is_exact_and_keeps_later_runtime_and_phases_closed(self) -> None:
        self.assertEqual(self.lock["status"], "complete")
        self.assertTrue(self.lock["typed_intent_stage_passed"])
        self.assertFalse(self.lock["downstream_runtime_change_authorized"])
        self.assertFalse(self.lock["qp4_complete"])
        self.assertFalse(self.lock["qp5_authorized"])
        self.assertEqual(self.lock["phase_d"], "locked")
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
