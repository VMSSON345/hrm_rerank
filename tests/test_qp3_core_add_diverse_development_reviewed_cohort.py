import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class Qp3CoreAddDiverseDevelopmentReviewedCohortTests(unittest.TestCase):
    def test_reviewed_cohort_and_safety_replay_are_exact(self) -> None:
        cohort = json.loads(
            (ROOT / "reports/finhrm_qp3_core_add_diverse_development_reviewed_cohort_v1.json").read_text(
                encoding="utf-8"
            )
        )
        lock = json.loads(
            (ROOT / "reports/finhrm_qp3_core_add_diverse_development_reviewed_cohort_lock_v1.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertTrue(cohort["reviewed_cohort_gate_passed"])
        self.assertEqual(
            cohort["metrics"]["selected_by_layout"],
            {"NATIVE_TEXT_TABLE": 4, "OCR_OR_HYBRID_TABLE": 4},
        )
        self.assertEqual(cohort["metrics"]["selected_examples"], 8)
        self.assertEqual(cohort["metrics"]["selected_companies"], 8)
        self.assertEqual(cohort["metrics"]["visual_rejections"], 1)
        self.assertTrue(
            all(
                row["review_status"] == "independently_reviewed_approved"
                and all(row["checks"].values())
                and row["counterfactual_action"] != "ANSWER"
                for row in cohort["details"]
            )
        )
        self.assertTrue(lock["reviewed_cohort_gate_passed"])
        self.assertFalse(lock["end_to_end_confirmation_passed"])
        self.assertEqual(lock["status"], "cohort_ready_end_to_end_pending")
        self.assertFalse(lock["pipeline_stage_status"]["chunk_retrieval"])
        self.assertFalse(
            lock["pipeline_stage_status"]["neural_operation_and_operand_proposal"]
        )

    def test_relative_period_false_accept_repair_is_generic(self) -> None:
        before = json.loads(
            (ROOT / "reports/finhrm_qp3_core_add_relative_period_counterfactual_audit_v1.json").read_text(
                encoding="utf-8"
            )
        )
        after = json.loads(
            (ROOT / "reports/finhrm_qp3_core_add_relative_period_counterfactual_post_fix_v1.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(before["metrics"]["false_accepts"], 4)
        self.assertGreaterEqual(before["metrics"]["false_accept_companies"], 3)
        self.assertGreaterEqual(before["metrics"]["false_accept_layout_families"], 2)
        self.assertEqual(after["metrics"]["counterfactuals"], before["metrics"]["counterfactuals"])
        self.assertEqual(after["metrics"]["false_accepts"], 0)


if __name__ == "__main__":
    unittest.main()
