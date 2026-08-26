from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_qp3_core_precommit_cohort import (
    DEFAULT_CORE,
    DEFAULT_OUTPUT,
    DEFAULT_PRECOMMIT,
    build_records,
)
from scripts.validate_qp3_core_precommit import (
    _refuse_protected,
    audit_record,
    build_report,
)


class QP3CorePrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.precommit = json.loads(DEFAULT_PRECOMMIT.read_text(encoding="utf-8"))
        cls.core = json.loads(DEFAULT_CORE.read_text(encoding="utf-8"))
        cls.families = {item["family_id"]: item for item in cls.core["families"]}
        cls.records = [
            json.loads(line)
            for line in DEFAULT_OUTPUT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _record(self, family_id: str, scenario_id: str) -> dict[str, object]:
        return deepcopy(
            next(
                item
                for item in self.records
                if item["family_id"] == family_id and item["scenario_id"] == scenario_id
            )
        )

    def test_builder_reproduces_frozen_cohort_exactly(self) -> None:
        self.assertEqual(build_records(self.precommit, self.core), self.records)
        self.assertEqual(len(self.records), 122)

    def test_precommit_gate_quotas_and_oracle_pass(self) -> None:
        report = build_report()
        self.assertTrue(report["precommit_gate_passed"], report["errors"])
        self.assertEqual(report["oracle"]["state_and_action_exact"], 122)
        self.assertEqual(report["oracle"]["false_accepts"], 0)
        self.assertEqual(report["oracle"]["gold_retained"], 26)
        self.assertEqual(report["oracle"]["gold_retention_total"], 26)
        self.assertTrue(all(report["quota_checks"].values()))

    def test_reference_base_is_required_and_wrong_base_rejects(self) -> None:
        record = self._record("PA08", "wrong_reference_base")
        observed = audit_record(record, self.precommit, self.families["PA08"])
        self.assertEqual(observed["proof_states"]["reference_base_match"], "INVALID")
        self.assertEqual(observed["action"], "REJECT")

    def test_logical_cell_and_provenance_fail_closed_independently(self) -> None:
        duplicate = self._record("PA01", "duplicate_logical_cell")
        duplicate_result = audit_record(duplicate, self.precommit, self.families["PA01"])
        self.assertEqual(duplicate_result["proof_states"]["logical_cell_unique"], "INVALID")
        self.assertEqual(duplicate_result["action"], "REJECT")

        missing = self._record("PA01", "missing_source_provenance")
        missing_result = audit_record(missing, self.precommit, self.families["PA01"])
        self.assertEqual(missing_result["proof_states"]["source_provenance_complete"], "UNKNOWN")
        self.assertEqual(missing_result["action"], "ABSTAIN")

    def test_incomplete_multi_output_cannot_return_a_partial_answer(self) -> None:
        record = self._record("TY01", "partial_multi_output")
        observed = audit_record(record, self.precommit, self.families["TY01"])
        self.assertEqual(observed["proof_states"]["answer_mode_complete"], "UNKNOWN")
        self.assertEqual(observed["action"], "ABSTAIN")

    def test_multistep_leaf_intermediate_and_execution_states_are_gating(self) -> None:
        cases = {
            "leaf_unknown": ("all_leaf_nodes_valid", "UNKNOWN", "ABSTAIN"),
            "leaf_invalid": ("all_leaf_nodes_valid", "INVALID", "REJECT"),
            "intermediate_unknown": ("all_intermediate_nodes_valid", "UNKNOWN", "ABSTAIN"),
            "intermediate_invalid": ("all_intermediate_nodes_valid", "INVALID", "REJECT"),
            "dag_execution_invalid": ("dag_execution_valid", "INVALID", "REJECT"),
        }
        for family_id in ("PA08", "PA09", "IBDS01"):
            for scenario_id, (obligation, state, action) in cases.items():
                with self.subTest(family_id=family_id, scenario_id=scenario_id):
                    record = self._record(family_id, scenario_id)
                    observed = audit_record(record, self.precommit, self.families[family_id])
                    self.assertEqual(observed["proof_states"][obligation], state)
                    self.assertEqual(observed["action"], action)

    def test_all_ambiguity_families_abstain_even_with_tempting_candidates(self) -> None:
        for family_id in ("C13", "C14", "C15"):
            record = self._record(family_id, "tempting_candidate")
            observed = audit_record(record, self.precommit, self.families[family_id])
            self.assertTrue(record["candidate_program_proposed"])
            self.assertFalse(record["candidate_program_materialized"])
            self.assertEqual(observed["action"], "ABSTAIN")

    def test_candidate_permutation_is_invariant_for_every_family(self) -> None:
        report = build_report()
        self.assertEqual(report["counterfactuals"]["permutation_exact"], 16)
        self.assertEqual(report["counterfactuals"]["permutation_total"], 16)

    def test_confirmation_is_company_and_layout_disjoint_and_group_reported(self) -> None:
        report = build_report()
        checks = report["generalization"]["checks"]
        self.assertTrue(all(checks.values()))
        self.assertEqual(report["generalization"]["company"]["groups"], 6)
        self.assertEqual(report["generalization"]["layout"]["groups"], 6)
        self.assertEqual(report["generalization"]["company"]["worst_group_exact_rate"], 1.0)
        self.assertEqual(report["generalization"]["layout"]["worst_group_exact_rate"], 1.0)
        serialized = "\n".join(json.dumps(item, ensure_ascii=False) for item in self.records).lower()
        self.assertNotIn("abw", serialized)

    def test_runtime_checkpoint_threshold_training_and_protected_sources_remain_locked(self) -> None:
        report = build_report()
        self.assertEqual(report["hash_audit"]["runtime_mismatches"], [])
        self.assertEqual(report["hash_audit"]["parent_mismatches"], [])
        self.assertEqual(report["hash_audit"]["dependency_mismatches"], [])
        for key in (
            "runtime_implementation_performed",
            "runtime_acceptance_policy_changed",
            "threshold_changed",
            "checkpoint_changed",
            "fine_tuning_performed",
            "global_validation_loaded",
            "sealed_validation_loaded",
            "blind_test_loaded",
            "phase_d_unlocked",
        ):
            self.assertFalse(report["invariants"][key])

    def test_validator_refuses_protected_evaluation_paths(self) -> None:
        protected = Path(tempfile.gettempdir()) / "sealed" / "blind_test.jsonl"
        with self.assertRaises(ValueError):
            _refuse_protected(protected)


if __name__ == "__main__":
    unittest.main()
