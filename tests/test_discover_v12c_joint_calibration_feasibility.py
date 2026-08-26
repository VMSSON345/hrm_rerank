from __future__ import annotations

import unittest

from scripts.discover_v12c_joint_calibration_feasibility import classify_signals, page_signals


class V12CJointCalibrationFeasibilityTests(unittest.TestCase):
    def test_complete_unique_triplet_is_positive_candidate(self) -> None:
        signals = page_signals(
            "Vay và nợ thuê tài chính ngắn hạn 80 Vay dài hạn 20 Tổng nguồn vốn 1000"
        )
        self.assertEqual(
            classify_signals(signals),
            ("positive_candidate", "complete_required_role_triplet"),
        )

    def test_two_required_roles_are_missing_role_negative(self) -> None:
        signals = page_signals("Vay ngắn hạn 80 Tổng cộng nguồn vốn 1000")
        self.assertEqual(
            classify_signals(signals),
            ("negative_candidate", "missing_required_raw_role"),
        )

    def test_broad_liabilities_are_semantic_negative_not_positive(self) -> None:
        signals = page_signals("Nợ ngắn hạn 80 Nợ dài hạn 20 Tổng nguồn vốn 1000")
        self.assertEqual(
            classify_signals(signals),
            ("negative_candidate", "invalid_borrowing_role_semantics"),
        )

    def test_duplicate_required_role_is_ambiguity_negative(self) -> None:
        signals = page_signals(
            "Vay ngắn hạn 80 Nợ vay ngắn hạn 81 Vay dài hạn 20 Tổng nguồn vốn 1000"
        )
        self.assertEqual(
            classify_signals(signals),
            ("negative_candidate", "ambiguous_duplicate_role"),
        )


if __name__ == "__main__":
    unittest.main()
