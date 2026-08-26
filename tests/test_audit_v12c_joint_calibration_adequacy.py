from __future__ import annotations

from pathlib import Path
import unittest

from scripts.audit_v12c_joint_calibration_adequacy import audit


ROOT = Path(__file__).resolve().parents[1]


class V12CJointCalibrationAdequacyTests(unittest.TestCase):
    def test_locked_reports_show_confidence_overlap_and_no_authorization(self) -> None:
        report = audit(
            ROOT / "reports/v12c_interest_bearing_debt_share_independent_calibration_frozen_inference_v1.json",
            ROOT / "reports/v12c_interest_bearing_debt_share_negative_ambiguity_frozen_inference_v1.json",
        )
        self.assertEqual(report["positive"]["program_and_execution_correct"], 20)
        self.assertEqual(report["negative"]["controls"], 26)
        self.assertEqual(report["negative"]["false_accepts_at_frozen_threshold"], 0)
        self.assertEqual(report["negative"]["unblocked_controls_at_or_above_positive_confidence_floor"], 6)
        self.assertFalse(report["separation"]["confidence_only_separable"])
        self.assertFalse(report["gate"]["ready_for_threshold_calibration"])
        self.assertFalse(report["gate"]["threshold_change_authorized"])


if __name__ == "__main__":
    unittest.main()
