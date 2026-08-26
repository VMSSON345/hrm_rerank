from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.evaluate_qp4_unit_scale_proof_contract_implementation import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "finhrm_qp4_unit_scale_proof_contract_implementation_v1.json"
SOURCE = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_source_replay_v1.json"
SAFE = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_safe_replay_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_implementation_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_implementation_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4UnitScaleProofContractImplementationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (REPORT.exists() and LOCK.exists()):
            raise unittest.SkipTest("Generated UNIT_SCALE proof implementation artifacts are absent")
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_exactly_reproducible_without_model_rerun(self) -> None:
        self.assertEqual(build_report(CONFIG, SOURCE, SAFE), self.report)

    def test_source_gate_closes_on_diverse_real_pdf_replay(self) -> None:
        source = self.report["source_replay"]
        self.assertTrue(source["gate_passed"])
        self.assertEqual(
            source["coverage"],
            {"runs": 5, "companies": 5, "sources": 5, "real_layouts": 3},
        )
        self.assertEqual(source["summary"]["old_winner_false_accepts"], 0)
        self.assertEqual(source["summary"]["prior_answer_to_abstain"], 4)
        self.assertEqual(source["summary"]["prior_answer_binding_retained"], 4)
        self.assertEqual(
            source["summary"]["prior_answer_scale_required_unknown"], 4
        )

    def test_frozen_execution_permutation_and_counterfactual_gates_still_pass(self) -> None:
        safe = self.report["safe_replay"]
        self.assertEqual(safe["execution"], "363/363")
        self.assertEqual(safe["permutation"], "363/363")
        self.assertEqual(safe["counterfactuals"], 141)
        self.assertEqual(safe["counterfactual_false_accepts"], 0)

    def test_implementation_gate_stays_open_on_exact_frozen_action_conflict(self) -> None:
        self.assertFalse(self.report["implementation_gate_passed"])
        self.assertEqual(self.report["failed_gates"], ["safe_actions_exact"])
        conflict = self.report["frozen_action_gate_conflict"]
        self.assertEqual(conflict["examples"], ["159", "621"])
        self.assertEqual(
            conflict["observed"],
            {"ANSWER": 257, "ABSTAIN": 92, "REJECT": 14},
        )
        self.assertEqual(
            conflict["expected"],
            {"ANSWER": 259, "ABSTAIN": 90, "REJECT": 14},
        )
        self.assertTrue(conflict["resolution_requires_new_precommit"])

    def test_two_runtime_files_changed_and_every_other_runtime_file_is_frozen(self) -> None:
        self.assertEqual(
            set(self.config["runtime_changes"]),
            {
                "src/hrm_finance_qa/question_planner.py",
                "src/hrm_finance_qa/symbolic_verifier.py",
            },
        )
        for relative, change in self.config["runtime_changes"].items():
            self.assertEqual(_digest(ROOT / relative), change["after_sha256"])
            self.assertNotEqual(change["before_sha256"], change["after_sha256"])
        for relative, expected in self.config["frozen_runtime_hashes"].items():
            self.assertEqual(_digest(ROOT / relative), expected)

    def test_lock_does_not_authorize_more_runtime_or_later_phases(self) -> None:
        self.assertEqual(self.lock["status"], "implementation_executed_gate_not_closed")
        self.assertFalse(self.lock["implementation_gate_passed"])
        self.assertTrue(self.lock["source_gate_passed"])
        self.assertFalse(self.lock["further_runtime_change_authorized"])
        self.assertFalse(self.lock["qp4_complete"])
        self.assertFalse(self.lock["qp5_authorized"])
        self.assertEqual(self.lock["phase_d"], "locked")
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
