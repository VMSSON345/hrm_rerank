from __future__ import annotations

import unittest

from scripts.audit_qp2a_operand_slots import build_report


class AuditQP2AOperandSlotsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()

    def test_qp2a_gate_passes(self) -> None:
        self.assertTrue(self.report["qp2a_gate_passed"])

    def test_all_canonical_and_frozen_answer_slots_are_exact(self) -> None:
        self.assertEqual(self.report["canonical_answer_slots"]["exact"], 13)
        self.assertEqual(self.report["canonical_answer_slots"]["total"], 13)
        self.assertEqual(self.report["frozen_answer_variant_slots"]["exact"], 108)
        self.assertEqual(self.report["frozen_answer_variant_slots"]["total"], 108)

    def test_missing_role_never_materializes_a_slot(self) -> None:
        self.assertEqual(self.report["missing_role_non_materialization"]["pass"], 12)
        self.assertEqual(self.report["missing_role_non_materialization"]["total"], 12)

    def test_no_provenance_or_collateral_violation(self) -> None:
        self.assertEqual(self.report["slot_integrity"]["provenance_violations"], [])
        self.assertEqual(self.report["slot_integrity"]["duplicate_id_families"], [])
        self.assertEqual(self.report["legacy_collateral"]["regressions"], [])

    def test_qp2a_does_not_claim_runtime_acceptance_change(self) -> None:
        invariants = self.report["invariants"]
        self.assertTrue(invariants["runtime_source_changed"])
        self.assertFalse(invariants["new_slot_fields_consumed_by_synthesis"])
        self.assertFalse(invariants["new_slot_fields_consumed_by_verifier"])
        self.assertFalse(invariants["runtime_acceptance_changed"])
        self.assertFalse(invariants["threshold_changed"])


if __name__ == "__main__":
    unittest.main()
