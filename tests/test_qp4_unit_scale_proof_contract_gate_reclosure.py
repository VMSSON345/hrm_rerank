from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from hrm_finance_qa.question_planner import compile_question_plan, compile_question_proof_contract
from scripts.evaluate_qp4_unit_scale_proof_contract_gate_reclosure import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "finhrm_qp4_unit_scale_proof_contract_gate_reclosure_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_gate_reclosure_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_gate_reclosure_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4UnitScaleProofContractGateReclosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_parent_chain_runtime_and_checkpoint_are_hash_exact(self) -> None:
        for parent in self.config["parents"].values():
            self.assertEqual(_digest(ROOT / parent["path"]), parent["sha256"])
        for relative, expected in self.config["frozen_runtime_hashes"].items():
            self.assertEqual(_digest(ROOT / relative), expected)
        checkpoint = self.config["checkpoint"]
        self.assertEqual(_digest(ROOT / checkpoint["path"]), checkpoint["sha256"])

    def test_reviewed_contract_preserves_historical_and_post_proof_actions(self) -> None:
        contract = self.config["reviewed_gate_contract"]
        self.assertEqual(contract["historical_actions_retained"], {"ANSWER": 259, "ABSTAIN": 90, "REJECT": 14})
        self.assertEqual(contract["post_proof_actions"], {"ANSWER": 257, "ABSTAIN": 92, "REJECT": 14})
        self.assertEqual(contract["allowed_action_delta"], {"ANSWER_TO_ABSTAIN": 2})
        self.assertEqual(contract["unchanged_actions"], 361)

    def test_explicit_scale_requires_unit_scale_proof(self) -> None:
        plan = compile_question_plan("Doanh thu theo triệu đồng là bao nhiêu?")
        proof = compile_question_proof_contract(plan, "SELECT")
        self.assertEqual(plan.requested_scale, "million_vnd")
        self.assertIn("unit_scale_match", proof.required)

    def test_implicit_scale_marks_unit_scale_not_applicable(self) -> None:
        plan = compile_question_plan("Doanh thu là bao nhiêu?")
        proof = compile_question_proof_contract(plan, "SELECT")
        self.assertIsNone(plan.requested_scale)
        self.assertIn("unit_scale_match", proof.not_applicable)

    def test_missing_and_conflicting_scale_actions_fail_closed(self) -> None:
        contract = self.config["reviewed_gate_contract"]
        self.assertEqual((contract["missing_raw_scale_state"], contract["missing_raw_scale_action"]), ("UNKNOWN", "ABSTAIN"))
        self.assertEqual((contract["mixed_or_conflicting_scale_state"], contract["mixed_or_conflicting_scale_action"]), ("INVALID", "REJECT"))

    def test_reclosure_gates_are_exact(self) -> None:
        gates = self.config["reclosure_gates"]
        self.assertEqual(gates["source_coverage"], {"runs": 5, "companies": 5, "sources": 5, "real_layouts": 3})
        self.assertEqual(gates["source_old_winner_false_accepts"], 0)
        self.assertEqual(gates["safe_actions"], {"ANSWER": 257, "ABSTAIN": 92, "REJECT": 14})
        self.assertEqual(gates["safe_execution_correct"], 363)
        self.assertEqual(gates["safe_permutation_exact"], 363)
        self.assertEqual(gates["safe_counterfactuals"], 141)
        self.assertEqual(gates["safe_counterfactual_false_accepts"], 0)

    def test_reclosure_authorizes_no_runtime_or_historical_overwrite(self) -> None:
        self.assertFalse(self.config["runtime_changes_authorized"])
        self.assertFalse(self.config["historical_artifact_overwrite_authorized"])
        for name, value in self.config["invariants"].items():
            self.assertFalse(value, name)


class QP4UnitScaleProofContractGateReclosureReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (REPORT.exists() and LOCK.exists()):
            raise unittest.SkipTest("Generated gate-reclosure artifacts are absent")
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_exactly_reproducible(self) -> None:
        self.assertEqual(build_report(CONFIG), self.report)

    def test_gate_recloses_without_claiming_overall_qp4_complete(self) -> None:
        self.assertTrue(self.report["unit_scale_gate_reclosed"])
        self.assertEqual(self.report["failed_gates"], [])
        self.assertTrue(self.report["qp4_unit_scale_complete"])
        self.assertFalse(self.report["qp4_complete"])
        self.assertFalse(self.report["qp5_authorized"])
        self.assertEqual(self.report["phase_d"], "locked")
        self.assertFalse(self.report["runtime_changed"])
        self.assertTrue(self.report["historical_implementation_report_preserved"])

    def test_lock_is_exact_and_opens_only_baseline_refresh_precommit(self) -> None:
        self.assertEqual(self.lock["status"], "complete")
        self.assertTrue(self.lock["unit_scale_gate_reclosed"])
        self.assertEqual(self.lock["runtime_file_whitelist_for_next_step"], [])
        self.assertFalse(self.lock["historical_artifact_overwritten"])
        self.assertFalse(self.lock["qp4_complete"])
        self.assertFalse(self.lock["qp5_authorized"])
        self.assertEqual(self.lock["phase_d"], "locked")
        self.assertTrue(self.lock["authorized_next_step"].startswith("QP4-POST-UNIT-SCALE-DEVELOPMENT-BASELINE-REFRESH-PRECOMMIT"))
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
