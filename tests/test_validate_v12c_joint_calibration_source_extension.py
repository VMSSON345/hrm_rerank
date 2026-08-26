from __future__ import annotations

from pathlib import Path
import unittest

from scripts.validate_v12c_joint_calibration_source_extension import validate


ROOT = Path(__file__).resolve().parents[1]


class V12CJointCalibrationSourceExtensionTests(unittest.TestCase):
    def test_extension_preserves_base_rules_and_has_audited_source(self) -> None:
        report = validate(
            ROOT / "config/v12c_joint_calibration_source_extension_v1.json", ROOT
        )
        self.assertTrue(report["source_extension_valid"])
        self.assertFalse(report["frozen_rules_changed"])
        self.assertEqual(report["documents"], 399)
        self.assertEqual(report["company_groups"], 240)
        self.assertFalse(report["ready_for_threshold_change"])

    def test_remaining_source_block_is_cross_block_disjoint(self) -> None:
        report = validate(
            ROOT / "config/v12c_joint_calibration_source_extension_v2.json", ROOT
        )
        self.assertTrue(report["source_extension_valid"])
        self.assertFalse(report["frozen_rules_changed"])
        self.assertEqual(report["documents"], 391)
        self.assertEqual(report["company_groups"], 239)


if __name__ == "__main__":
    unittest.main()
