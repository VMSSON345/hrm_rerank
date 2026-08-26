from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.evaluate_qp3_answer_presentation_implementation import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "finhrm_qp3_answer_presentation_development_cohort_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp3_answer_presentation_implementation_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp3_answer_presentation_implementation_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP3AnswerPresentationImplementationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_exactly_reproducible(self) -> None:
        self.assertEqual(build_report(CONFIG), self.report)

    def test_all_eligible_selection_is_generic_and_development_only(self) -> None:
        selection = self.report["selection_policy"]
        self.assertTrue(selection["evaluate_all_eligible_records"])
        self.assertFalse(selection["synthetic_fixture_counts_toward_coverage"])
        self.assertFalse(selection["abw_allowed"])
        self.assertFalse(selection["protected_evaluation_allowed"])
        self.assertIn("company identity", selection["forbidden_selection_inputs"])
        self.assertIn("question literal", selection["forbidden_selection_inputs"])
        self.assertEqual(self.report["results"]["eligible_source_backed_records"], 5)

    def test_each_mode_meets_company_layout_and_source_coverage(self) -> None:
        modes = self.report["results"]["mode_metrics"]
        self.assertEqual(set(modes), {"FACT", "COMPUTATION", "VERIFIED_SUMMARY"})
        self.assertEqual(modes["FACT"]["answers"], 10)
        self.assertEqual(modes["COMPUTATION"]["answers"], 5)
        self.assertEqual(modes["VERIFIED_SUMMARY"]["answers"], 5)
        for mode in modes.values():
            self.assertGreaterEqual(mode["companies"], 3)
            self.assertGreaterEqual(mode["real_layout_families"], 3)
            self.assertGreaterEqual(mode["source_documents"], 3)
            self.assertTrue(mode["coverage_gate_passed"])
            self.assertEqual(mode["company"]["worst_group_answer_rate"], 1.0)
            self.assertEqual(mode["layout"]["worst_group_answer_rate"], 1.0)

    def test_numeric_citation_derivation_and_claim_gates_pass(self) -> None:
        checks = self.report["results"]["presentation_checks"]
        self.assertTrue(all(checks.values()), checks)
        self.assertEqual(self.report["results"]["source_backed_scalar_envelopes"], 15)
        self.assertEqual(self.report["results"]["hard_negative_false_accepts"], 0)

    def test_all_permutations_and_hard_negatives_are_exact(self) -> None:
        result = self.report["results"]
        self.assertEqual(result["candidate_permutation_exact"], "5/5")
        self.assertEqual(result["citation_permutation_exact"], "15/15")
        self.assertEqual(result["summary_input_permutation_exact"], "5/5")
        self.assertEqual(result["presentation_and_report_tests"], "22/22")
        self.assertEqual(result["affected_regression_tests"], "293/293")
        self.assertNotIn("FALSE_ACCEPT", result["hard_negatives"].values())
        self.assertEqual(
            set(result["hard_negatives"]),
            {
                "PRES_WRONG_PAGE",
                "PRES_WRONG_STATEMENT",
                "PRES_WRONG_YEAR",
                "PRES_SCALE_X1000",
                "PRES_SWAPPED_OPERAND",
                "PRES_MISSING_SUMMARY_ROW",
                "PRES_UNCITED_QUALITATIVE_CAUSE",
                "PRES_PARENT_ACTION_PROMOTION",
            },
        )

    def test_frozen_core_replay_and_runtime_scope_are_preserved(self) -> None:
        result = self.report["results"]
        self.assertTrue(all(result["frozen_core_checks"].values()))
        self.assertEqual(result["frozen_core_actions"], {"ANSWER": 259, "ABSTAIN": 90, "REJECT": 14})
        self.assertEqual(result["frozen_core_execution"], "363/363")
        self.assertEqual(result["frozen_core_permutation"], "363/363")
        self.assertEqual(result["frozen_core_counterfactual_false_accepts"], 0)
        self.assertEqual(
            set(self.report["runtime_hashes"]),
            {
                "src/hrm_finance_qa/answer_presentation.py",
                "src/hrm_finance_qa/document_reasoning_engine.py",
            },
        )
        for relative, expected in self.report["frozen_runtime_hashes"].items():
            self.assertEqual(_digest(ROOT / relative), expected)

    def test_implementation_lock_closes_qp3_and_opens_only_qp4_precommit(self) -> None:
        self.assertTrue(self.report["implementation_gate_passed"])
        self.assertTrue(self.report["qp3_complete"])
        self.assertTrue(self.report["qp4_precommit_authorized"])
        self.assertFalse(self.report["qp4_runtime_change_authorized"])
        self.assertEqual(self.report["phase_d"], "locked")
        self.assertTrue(self.lock["implementation_gate_passed"])
        self.assertTrue(self.lock["qp3_complete"])
        self.assertTrue(self.lock["qp4_precommit_authorized"])
        self.assertFalse(self.lock["qp4_runtime_change_authorized"])
        self.assertIn("QP4-PRECOMMIT only", self.lock["authorized_next_step"])
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
