from __future__ import annotations

import unittest

from scripts.audit_qp2d_safe_abstention import build_report


class AuditQP2DSafeAbstentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()

    def test_qp2d_gate_passes(self) -> None:
        self.assertTrue(self.report["qp2d_gate_passed"])
        self.assertEqual(self.report["canonical_ambiguity_controls"]["exact"], 3)
        self.assertEqual(self.report["canonical_ambiguity_controls"]["total"], 3)

    def test_underspecification_and_repairs_are_symmetric(self) -> None:
        surfaces = self.report["underspecification_surfaces"]
        repairs = self.report["false_abstain_controls"]
        self.assertEqual(surfaces["missing_reference_base_exact"], 8)
        self.assertEqual(surfaces["missing_column_role_exact"], 6)
        self.assertEqual(repairs["explicit_reference_repair_exact"], 4)
        self.assertEqual(repairs["explicit_role_repair_exact"], 3)
        self.assertEqual(repairs["pa07_exact"], repairs["pa07_total"])

    def test_qp2a_qp2b_qp2c_answer_contract_is_retained(self) -> None:
        retention = self.report["answer_contract_retention"]
        self.assertEqual(retention["exact"], retention["total"])
        self.assertEqual(retention["total"], 108)

    def test_strict_delta_has_only_expected_legacy_improvements(self) -> None:
        self.assertEqual(self.report["strict_contract_delta"]["exact_before"], 14)
        self.assertEqual(self.report["strict_contract_delta"]["exact_after"], 16)
        self.assertEqual(
            self.report["legacy_collateral"]["improvements"],
            ["C14-canonical", "C15-canonical"],
        )
        self.assertEqual(self.report["legacy_collateral"]["regressions"], [])

    def test_new_reason_is_not_consumed_by_runtime(self) -> None:
        self.assertFalse(any(self.report["runtime_consumer_scan"].values()))
        invariants = self.report["invariants"]
        self.assertFalse(invariants["abstention_reason_consumed_by_runtime"])
        self.assertFalse(invariants["runtime_acceptance_policy_changed"])
        self.assertFalse(invariants["threshold_changed"])
        self.assertFalse(invariants["phase_d_unlocked"])


if __name__ == "__main__":
    unittest.main()
