from __future__ import annotations

import unittest

from scripts.audit_qp2c_reference_dag import build_report


class AuditQP2CReferenceDAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()

    def test_qp2c_gate_passes(self) -> None:
        self.assertTrue(self.report["qp2c_gate_passed"])

    def test_canonical_and_frozen_programs_are_exact(self) -> None:
        self.assertEqual(self.report["canonical_answer_contract"]["exact"], 13)
        self.assertEqual(self.report["canonical_answer_contract"]["total"], 13)
        self.assertEqual(self.report["frozen_answer_variants"]["exact"], 108)
        self.assertEqual(self.report["frozen_answer_variants"]["total"], 108)

    def test_counterfactual_multi_step_and_abstain_controls_pass(self) -> None:
        self.assertEqual(self.report["counterfactuals"]["role_swap_exact"], 12)
        self.assertEqual(self.report["counterfactuals"]["year_shift_exact"], 12)
        self.assertEqual(self.report["multi_step_variants"]["pa08_pa09_exact"], 18)
        self.assertTrue(self.report["multi_step_variants"]["ibds01_canonical_exact"])
        self.assertEqual(self.report["abstain_no_dag_controls"]["pass"], 27)

    def test_qp2a_qp2b_and_legacy_are_retained(self) -> None:
        self.assertEqual(self.report["retention"]["qp2a_slot_exact"], 108)
        self.assertEqual(self.report["retention"]["qp2b_operation_direction_exact"], 108)
        self.assertEqual(self.report["legacy_collateral"]["regressions"], [])

    def test_dag_integrity_and_runtime_boundary_are_clean(self) -> None:
        self.assertEqual(self.report["dag_integrity"]["validation_error_examples"], 0)
        self.assertEqual(self.report["dag_integrity"]["provenance_violations"], [])
        invariants = self.report["invariants"]
        self.assertFalse(invariants["compiler_dag_consumed_by_runtime"])
        self.assertFalse(invariants["program_identity_ids_consumed_by_runtime"])
        self.assertFalse(invariants["legacy_financial_identity_ids_changed_for_ibds01"])
        self.assertFalse(invariants["runtime_acceptance_policy_changed"])
        self.assertFalse(invariants["threshold_changed"])


if __name__ == "__main__":
    unittest.main()
