from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.audit_qp4_unit_scale_frozen_action_contract_review import build_audit
from scripts.validate_qp4_unit_scale_frozen_action_contract_review_precommit import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "finhrm_qp4_unit_scale_frozen_action_contract_review_precommit_v1.json"
AUDIT = ROOT / "reports" / "finhrm_qp4_unit_scale_frozen_action_contract_review_audit_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp4_unit_scale_frozen_action_contract_review_precommit_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp4_unit_scale_frozen_action_contract_review_precommit_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4UnitScaleFrozenActionContractReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.audit = json.loads(AUDIT.read_text(encoding="utf-8"))

    def test_full_audit_is_reproducible_and_passes(self) -> None:
        self.assertEqual(build_audit(), self.audit)
        self.assertTrue(self.audit["audit_passed"])
        self.assertEqual(self.audit["errors"], [])

    def test_exactly_two_generic_answer_to_abstain_deltas(self) -> None:
        self.assertEqual(self.audit["universe"], {"examples": 363, "unchanged_actions": 361})
        self.assertEqual(self.audit["action_transitions"], {"ANSWER->ABSTAIN": 2})
        self.assertEqual(len(self.audit["delta_rows"]), 2)
        for row in self.audit["delta_rows"]:
            self.assertTrue(row["generic_contract_eligible"])
            self.assertEqual(row["transition"], "ANSWER->ABSTAIN")
            self.assertEqual(row["requested_scale"], "vnd")
            self.assertEqual(row["current_reasons"], ["required_proof_unknown:unit_scale_match"])
            self.assertTrue(all(row["checks"].values()))

    def test_historical_contract_is_preserved_and_versioned(self) -> None:
        contract = self.config["versioned_action_contract"]
        self.assertTrue(contract["historical_contract_is_immutable"])
        self.assertEqual(
            contract["historical_pre_proof_actions"],
            {"ANSWER": 259, "ABSTAIN": 90, "REJECT": 14},
        )
        self.assertEqual(
            contract["reviewed_post_proof_actions"],
            {"ANSWER": 257, "ABSTAIN": 92, "REJECT": 14},
        )
        self.assertEqual(contract["allowed_transitions"]["ANSWER_TO_ABSTAIN"], 2)
        self.assertEqual(
            sum(value for key, value in contract["allowed_transitions"].items() if key != "ANSWER_TO_ABSTAIN"),
            0,
        )

    def test_ids_are_traceability_only_and_not_confirmation(self) -> None:
        contract = self.config["versioned_action_contract"]
        self.assertTrue(contract["example_ids_allowed_only_in_audit_artifacts"])
        self.assertTrue(contract["example_id_filename_company_page_year_value_or_question_literal_runtime_rule_forbidden"])
        self.assertTrue(contract["gold_answer_text_as_runtime_unit_proof_forbidden"])
        self.assertFalse(contract["contract_conflicts_counted_as_generalization_confirmation"])
        self.assertFalse(self.audit["generic_delta_definition"]["runtime_id_or_literal_rule_used"])

    def test_reclosure_gates_remain_strict(self) -> None:
        gate = self.config["gate_reclosure_contract"]
        self.assertEqual(gate["source_coverage"], {"runs": 5, "companies": 5, "sources": 5, "real_layouts": 3})
        self.assertEqual(gate["source_old_winner_false_accepts"], 0)
        self.assertEqual(gate["safe_actions"], {"ANSWER": 257, "ABSTAIN": 92, "REJECT": 14})
        self.assertEqual(gate["safe_execution"], "363/363")
        self.assertEqual(gate["safe_permutation"], "363/363")
        self.assertEqual(gate["safe_counterfactuals"], 141)
        self.assertEqual(gate["safe_counterfactual_false_accepts"], 0)

    def test_runtime_checkpoint_and_parents_are_frozen(self) -> None:
        for parent in self.config["parents"].values():
            self.assertEqual(_digest(ROOT / parent["path"]), parent["sha256"])
        for relative, expected in self.config["frozen_runtime_hashes"].items():
            self.assertEqual(_digest(ROOT / relative), expected)
        checkpoint = self.config["checkpoint"]
        self.assertEqual(_digest(ROOT / checkpoint["path"]), checkpoint["sha256"])

    def test_precommit_authorizes_no_runtime_or_historical_edit(self) -> None:
        self.assertFalse(self.config["runtime_changed_by_this_precommit"])
        self.assertFalse(self.config["downstream_runtime_change_authorized_in_same_step"])
        self.assertFalse(self.config["historical_artifact_overwrite_authorized"])
        for name, value in self.config["invariants"].items():
            self.assertFalse(value, name)


class QP4UnitScaleFrozenActionContractReviewReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (REPORT.exists() and LOCK.exists()):
            raise unittest.SkipTest("Generated action-contract review artifacts are absent")
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_reproducible(self) -> None:
        self.assertEqual(build_report(CONFIG), self.report)

    def test_report_and_lock_pass_without_runtime_change(self) -> None:
        self.assertTrue(self.report["precommit_gate_passed"])
        self.assertEqual(self.report["failed_gates"], [])
        self.assertFalse(self.report["runtime_changed"])
        self.assertFalse(self.report["historical_artifact_overwritten"])
        self.assertEqual(self.lock["status"], "complete")
        self.assertTrue(self.lock["precommit_gate_passed"])
        self.assertEqual(self.lock["runtime_file_whitelist_for_next_step"], [])
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
