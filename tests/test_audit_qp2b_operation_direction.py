from __future__ import annotations

import unittest

from scripts.audit_qp2b_operation_direction import build_report


class AuditQP2BOperationDirectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()

    def test_qp2b_gate_passes(self) -> None:
        self.assertTrue(self.report["qp2b_gate_passed"])

    def test_all_answer_and_target_variants_are_exact(self) -> None:
        self.assertEqual(self.report["all_answer_variants"]["exact"], 108)
        self.assertEqual(self.report["all_answer_variants"]["total"], 108)
        self.assertEqual(self.report["target_answer_variants"]["exact"], 54)
        self.assertEqual(self.report["target_answer_variants"]["total"], 54)

    def test_counterfactual_and_long_surfaces_are_exact(self) -> None:
        counterfactuals = self.report["counterfactuals"]
        self.assertEqual(counterfactuals["role_swap_exact"], 12)
        self.assertEqual(counterfactuals["role_swap_total"], 12)
        self.assertEqual(counterfactuals["year_shift_exact"], 12)
        self.assertEqual(counterfactuals["year_shift_total"], 12)
        self.assertEqual(self.report["surface_robustness"]["long_no_diacritic_reordered_exact"], 36)

    def test_qp2a_is_retained_without_collateral_regression(self) -> None:
        retention = self.report["qp2a_retention"]
        self.assertTrue(retention["qp2a_snapshot_gate_passed"])
        self.assertEqual(retention["current_canonical_slots"], 13)
        self.assertEqual(retention["current_frozen_variant_slots"], 108)
        self.assertEqual(self.report["legacy_collateral"]["regressions"], [])
        self.assertEqual(self.report["strict_contract_delta"]["operation_after"], 13)
        self.assertEqual(self.report["strict_contract_delta"]["direction_after"], 13)

    def test_runtime_boundary_is_explicit(self) -> None:
        invariants = self.report["invariants"]
        self.assertTrue(invariants["runtime_planning_behavior_changed"])
        self.assertFalse(invariants["operand_slots_consumed_by_runtime"])
        self.assertFalse(invariants["program_synthesis_source_changed"])
        self.assertFalse(invariants["symbolic_verifier_source_changed"])
        self.assertFalse(invariants["runtime_acceptance_policy_changed"])
        self.assertFalse(invariants["threshold_changed"])


if __name__ == "__main__":
    unittest.main()
