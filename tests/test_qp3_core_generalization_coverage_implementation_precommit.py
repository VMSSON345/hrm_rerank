from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PRECOMMIT = (
    ROOT / "config" / "finhrm_qp3_core_generalization_coverage_implementation_precommit_v1.json"
)
SAFE_REPORT = ROOT / "reports" / "finhrm_qp3_core_s4_safe_replay_v1.json"
LOCK = (
    ROOT / "reports" / "finhrm_qp3_core_generalization_coverage_implementation_precommit_lock_v1.json"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP3CoreGeneralizationCoverageImplementationPrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.precommit = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
        cls.safe_report = json.loads(SAFE_REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_claude_parent_source_and_runtime_hashes_are_exact(self) -> None:
        claude = self.precommit["claude_md"]
        self.assertTrue(claude["read_before_scope_design"])
        self.assertEqual(_digest(ROOT / claude["path"]), claude["sha256"])
        for parent in self.precommit["parents"].values():
            self.assertEqual(_digest(ROOT / parent["path"]), parent["sha256"])
        evidence = self.precommit["allowed_evidence"]
        self.assertEqual(_digest(ROOT / evidence["source_path"]), evidence["source_sha256"])
        for relative, expected in self.precommit[
            "runtime_hashes_before_implementation"
        ].items():
            self.assertEqual(_digest(ROOT / relative), expected)

    def test_failure_taxonomy_matches_safe_replay(self) -> None:
        rows = self.safe_report["details"]

        def count(name: str) -> int:
            marker = f"required_proof_unknown:{name}"
            return sum(marker in row["gold_reasons"] for row in rows)

        diagnosis = self.precommit["aggregate_diagnosis"]
        self.assertEqual(diagnosis["period_and_provenance"]["unknown_examples"], 74)
        self.assertEqual(count("period_coverage"), 74)
        self.assertEqual(
            diagnosis["operation_and_add_grounding"]["unknown_examples"], 67
        )
        self.assertEqual(count("operation_match"), 67)
        self.assertEqual(count("add_grounding"), 67)
        self.assertEqual(diagnosis["query_metric"]["unknown_examples"], 33)
        self.assertEqual(count("query_metric"), 33)

    def test_period_track_is_fail_closed_not_repaired_from_absence(self) -> None:
        diagnosis = self.precommit["aggregate_diagnosis"]["period_and_provenance"]
        admission = self.precommit["track_admission"]["period_and_provenance"]
        self.assertEqual(diagnosis["selected_period_empty"], 74)
        self.assertEqual(diagnosis["selected_explicit_year_present"], 0)
        self.assertEqual(diagnosis["selected_period_role_present"], 0)
        self.assertEqual(diagnosis["header_contains_requested_year"], 0)
        self.assertFalse(admission["runtime_edit_admitted"])
        self.assertIn("KEEP_UNKNOWN", admission["policy"])
        self.assertFalse(
            self.precommit["postimplementation_replay_gates"][
                "period_unknown_may_decrease_without_explicit_lineage"
            ]
        )

    def test_add_and_query_metric_are_the_only_admitted_tracks(self) -> None:
        admission = self.precommit["track_admission"]
        admitted = {
            name for name, track in admission.items() if track["runtime_edit_admitted"]
        }
        self.assertEqual(
            admitted, {"operation_and_add_grounding", "query_metric"}
        )
        query = self.precommit["aggregate_diagnosis"]["query_metric"]
        self.assertEqual(query["provenanced_composite_alias_valid_candidates"], 8)
        self.assertEqual(query["provenanced_composite_candidate_companies"], 3)
        self.assertEqual(query["provenanced_composite_candidate_layouts"], 3)

    def test_exact_runtime_scope_is_surgical_and_frozen_files_are_disjoint(self) -> None:
        scope = self.precommit["exact_runtime_edit_scope"]
        self.assertEqual(
            set(scope),
            {
                "src/hrm_finance_qa/question_planner.py",
                "src/hrm_finance_qa/program_synthesis.py",
                "src/hrm_finance_qa/symbolic_verifier.py",
                "src/hrm_finance_qa/document_hrm.py",
            },
        )
        frozen = set(self.precommit["runtime_files_explicitly_frozen"])
        self.assertTrue(frozen.isdisjoint(scope))
        self.assertIn("src/hrm_finance_qa/evidence_schema.py", frozen)
        self.assertIn("src/hrm_finance_qa/financial_identities.py", frozen)
        for item in scope.values():
            self.assertTrue(item["symbols"])

    def test_replay_gates_preserve_safety_and_require_group_improvement(self) -> None:
        gates = self.precommit["postimplementation_replay_gates"]
        self.assertEqual(gates["retain_all_preexisting_answers"], "187/187")
        self.assertEqual(gates["safe_replay_gold_execution"], "363/363")
        self.assertEqual(gates["safe_replay_candidate_permutation"], "363/363")
        self.assertEqual(gates["safe_replay_counterfactual_false_accepts"], 0)
        self.assertEqual(gates["hard_negative_or_counterfactual_answer_count"], 0)
        for prefix in ("operation_add", "query_metric"):
            self.assertEqual(gates[f"{prefix}_improvement_minimum_companies"], 3)
            self.assertEqual(gates[f"{prefix}_improvement_minimum_layouts"], 3)
        self.assertTrue(gates["new_answer_requires_every_required_proof_valid"])
        self.assertFalse(gates["safe_abstention_counted_as_answered"])
        self.assertFalse(gates["unseen_or_universal_claim_allowed"])

    def test_no_runtime_threshold_checkpoint_training_or_release_change(self) -> None:
        for key, value in self.precommit["frozen_invariants"].items():
            self.assertFalse(value, key)
        serialized = json.dumps(self.precommit, ensure_ascii=False).lower()
        self.assertNotIn("devco_", serialized)
        self.assertNotIn('"example_id":', serialized)
        self.assertNotIn('"company_key":', serialized)

    def test_lock_artifacts_are_exact_and_only_scoped_implementation_is_next(self) -> None:
        self.assertTrue(self.lock["implementation_precommit_complete"])
        self.assertFalse(self.lock["runtime_changed"])
        self.assertTrue(self.lock["next_step_scoped_runtime_change_authorized"])
        self.assertFalse(self.lock["qp4_authorized"])
        self.assertIn("GENERALIZATION-COVERAGE-IMPLEMENTATION", self.lock["authorized_next_step"])
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
