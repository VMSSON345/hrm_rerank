from __future__ import annotations

from pathlib import Path
import unittest

from scripts.validate_v12c_joint_calibration_protocol import validate


ROOT = Path(__file__).resolve().parents[1]


class V12CJointCalibrationProtocolTests(unittest.TestCase):
    def test_locked_protocol_matches_all_exclusion_cohorts(self) -> None:
        report = validate(ROOT / "config/v12c_joint_calibration_protocol_v1.json", ROOT)
        self.assertTrue(report["precommit_valid"])
        self.assertEqual(report["excluded_tickers"], 27)
        self.assertEqual(report["excluded_company_keys"], 27)
        self.assertFalse(report["ready_for_threshold_change"])


if __name__ == "__main__":
    unittest.main()
