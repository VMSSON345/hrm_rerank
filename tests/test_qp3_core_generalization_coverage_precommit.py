from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PRECOMMIT = ROOT / "config" / "finhrm_qp3_core_generalization_coverage_precommit_v1.json"
SAFE_REPORT = ROOT / "reports" / "finhrm_qp3_core_s4_safe_replay_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp3_core_generalization_coverage_precommit_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP3CoreGeneralizationCoveragePrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.precommit = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
        cls.report = json.loads(SAFE_REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_all_parent_artifacts_and_runtime_are_hash_locked(self) -> None:
        for parent in self.precommit["parents"].values():
            self.assertEqual(_digest(ROOT / parent["path"]), parent["sha256"])
        source = self.precommit["allowed_evidence"]
        self.assertEqual(_digest(ROOT / source["source_path"]), source["source_sha256"])
        for relative, expected in self.precommit["runtime_hashes_at_precommit"].items():
            self.assertEqual(_digest(ROOT / relative), expected)

    def test_baseline_is_exact_and_does_not_count_abstention_as_answered(self) -> None:
        result = self.report["results"]
        baseline = self.precommit["baseline"]
        self.assertEqual(baseline["gold_answer"], result["gold_actions"]["ANSWER"])
        self.assertEqual(baseline["gold_abstain"], result["gold_actions"]["ABSTAIN"])
        self.assertEqual(baseline["gold_reject"], result["gold_actions"]["REJECT"])
        self.assertEqual(baseline["gold_execution_correct"], result["gold_execution_correct"])
        self.assertEqual(baseline["permutation_exact"], result["permutation_exact"])
        self.assertEqual(
            baseline["counterfactual_false_accepts"],
            result["counterfactual_false_accepts"],
        )
        self.assertEqual(
            baseline["gold_answer"] + baseline["gold_abstain"] + baseline["gold_reject"],
            result["gold_total"],
        )

    def test_target_taxonomy_is_derived_over_company_and_layout_groups(self) -> None:
        rows = [row for row in self.report["details"] if row["gold_action"] != "ANSWER"]

        def stats(obligations: set[str]) -> tuple[int, int, int]:
            selected = [
                row
                for row in rows
                if any(
                    f"required_proof_unknown:{obligation}" in row["gold_reasons"]
                    for obligation in obligations
                )
            ]
            return (
                len(selected),
                len({row["company_key"] for row in selected}),
                len({row["layout_family"] for row in selected}),
            )

        tracks = self.precommit["target_tracks"]
        expected = {
            "period_and_provenance": (74, 73, 2),
            "operation_and_add_grounding": (67, 47, 3),
            "query_metric": (33, 26, 6),
        }
        for name, values in expected.items():
            track = tracks[name]
            observed = stats(set(track["unknown_obligations"]))
            self.assertEqual(observed, values)
            self.assertEqual(track["unknown_examples"], values[0])
            self.assertEqual(track["distinct_companies"], values[1])
            self.assertEqual(track["distinct_layout_families"], values[2])

    def test_period_track_fails_closed_until_layout_diversity_exists(self) -> None:
        gate = self.precommit["diagnostic_matrix_gate"]
        tracks = self.precommit["target_tracks"]
        self.assertEqual(gate["minimum_distinct_layout_families_per_repair"], 3)
        self.assertTrue(tracks["period_and_provenance"]["admission"].startswith("BLOCKED_"))
        self.assertIn(
            "DIAGNOSIS_DIVERSITY_GATE_SATISFIED",
            tracks["operation_and_add_grounding"]["admission"],
        )
        self.assertIn(
            "DIAGNOSIS_DIVERSITY_GATE_SATISFIED",
            tracks["query_metric"]["admission"],
        )

    def test_precommit_contains_no_report_or_example_specific_rule(self) -> None:
        serialized = json.dumps(self.precommit, ensure_ascii=False).lower()
        self.assertNotIn("devco_", serialized)
        self.assertNotIn('"example_id":', serialized)
        self.assertNotIn('"company_key":', serialized)
        self.assertNotIn("candidate rank or score as operand-role proof" + " allowed", serialized)
        self.assertFalse(
            self.precommit["diagnostic_matrix_gate"][
                "individual_company_or_example_identifiers_may_be_encoded"
            ]
        )

    def test_safety_and_release_remain_locked(self) -> None:
        invariants = self.precommit["frozen_invariants"]
        for key, value in invariants.items():
            self.assertFalse(value, key)
        acceptance = self.precommit["postrepair_acceptance_gates"]
        self.assertEqual(acceptance["counterfactual_false_accepts"], 0)
        self.assertEqual(acceptance["candidate_permutation_exact_rate"], 1.0)
        self.assertTrue(acceptance["new_answer_requires_all_required_proofs_valid"])
        self.assertFalse(acceptance["safe_abstention_counted_as_answered"])
        self.assertFalse(acceptance["unseen_or_universal_generalization_claim_allowed"])

    def test_lock_artifacts_are_exact_and_next_step_is_diagnostic_only(self) -> None:
        self.assertFalse(self.lock["runtime_change_authorized"])
        self.assertFalse(self.lock["qp4_authorized"])
        self.assertIn("DIAGNOSTIC-MATRIX", self.lock["authorized_next_step"])
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
