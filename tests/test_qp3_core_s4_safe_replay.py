from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PRECOMMIT = ROOT / "config" / "finhrm_qp3_core_s4_safe_replay_precommit_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp3_core_s4_safe_replay_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp3_core_s4_safe_replay_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP3CoreS4SafeReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.precommit = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_source_is_hash_locked_and_contains_no_forbidden_marker(self) -> None:
        source = ROOT / self.precommit["source"]["path"]
        self.assertEqual(source.stat().st_size, self.precommit["source"]["size_bytes"])
        self.assertEqual(_digest(source), self.precommit["source"]["sha256"])
        lowered = source.read_bytes().lower()
        self.assertNotIn(b"abw", lowered)
        for marker in (b"validation", b"blind", b"holdout", b"sealed"):
            self.assertNotIn(marker, lowered)

    def test_safe_replay_covers_many_companies_and_layouts(self) -> None:
        result = self.report["results"]
        self.assertTrue(self.report["safe_replay_gate_passed"])
        self.assertEqual(result["gold_total"], 363)
        self.assertEqual(result["operations"], {"ADD": 123, "SELECT": 240})
        self.assertEqual(result["company"]["groups"], 121)
        self.assertEqual(result["layout"]["groups"], 6)

    def test_permutation_and_explicit_counterfactual_gates_pass(self) -> None:
        result = self.report["results"]
        self.assertEqual(result["permutation_exact"], result["permutation_total"])
        self.assertEqual(result["counterfactual_generated"], 123)
        self.assertEqual(result["counterfactual_false_accepts"], 0)
        self.assertEqual(
            result["counterfactual_mutations"],
            {"WRONG_ENTITY_SCOPE": 3, "WRONG_METRIC": 48, "WRONG_YEAR": 72},
        )

    def test_coverage_limit_is_reported_not_hidden(self) -> None:
        result = self.report["results"]
        self.assertEqual(result["gold_execution_correct"], 363)
        self.assertEqual(result["gold_answered"], 187)
        self.assertEqual(result["company"]["worst_gold_answer_rate"], 0.0)
        self.assertEqual(result["layout"]["worst_gold_answer_rate"], 0.0)
        self.assertIn("not establish unseen", self.report["interpretation"])

    def test_lock_and_runtime_hashes_are_exact(self) -> None:
        self.assertTrue(self.lock["safe_replay_gate_passed"])
        self.assertFalse(self.lock["qp4_authorized"])
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])
        for relative, expected in self.lock["runtime_hashes"].items():
            self.assertEqual(_digest(ROOT / relative), expected)

    def test_no_threshold_checkpoint_training_or_protected_access(self) -> None:
        invariants = self.report["invariants"]
        for key in (
            "abw_used",
            "protected_evaluation_loaded",
            "validation_loaded",
            "blind_loaded",
            "test_data_loaded",
            "candidate_rank_or_score_used_as_semantic_proof",
            "threshold_changed",
            "checkpoint_changed",
            "fine_tuning_performed",
            "runtime_changed_by_replay",
            "phase_d_unlocked",
            "safe_abstention_counted_as_answered",
            "unseen_confirmation_claim_made",
            "universal_coverage_claim_made",
        ):
            self.assertFalse(invariants[key], key)


if __name__ == "__main__":
    unittest.main()
