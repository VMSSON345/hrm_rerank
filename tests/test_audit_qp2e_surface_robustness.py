from __future__ import annotations

import unittest

from scripts.audit_qp2e_surface_robustness import build_report


class AuditQP2ESurfaceRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_report()

    def test_safe_gate_passes_on_identifiable_subset(self) -> None:
        self.assertTrue(self.report["qp2e_safe_gate_passed"])
        identifiable = self.report["compiler_identifiable_subset"]
        self.assertEqual(identifiable["exact"], identifiable["total"])
        self.assertEqual(identifiable["total"], 123)

    def test_raw_improvement_and_residual_partition_are_exact(self) -> None:
        raw = self.report["raw_catalog"]
        after = self.report["failure_partition_after"]
        self.assertEqual((raw["exact_before"], raw["exact_after"], raw["total"]), (92, 123, 135))
        self.assertEqual(len(raw["improvements"]), 31)
        self.assertEqual(raw["regressions"], [])
        self.assertEqual(after["total"], 12)
        self.assertEqual(after["by_variant"], {"hard_negative_missing_role": 12})
        self.assertEqual(after["unsafe_compiler_residuals"], [])

    def test_answer_ambiguity_and_role_use_sites_are_retained(self) -> None:
        retention = self.report["retention"]
        self.assertEqual(retention["answer_variants_exact"], retention["answer_variants_total"])
        self.assertEqual(retention["answer_variants_total"], 108)
        self.assertEqual(retention["ambiguity_controls_exact"], 12)
        self.assertEqual(retention["column_role_use_site_exact"], 135)
        self.assertEqual(retention["strict_canonical_exact"], 16)

    def test_catalog_collision_is_preserved_instead_of_overfit(self) -> None:
        value = self.report["catalog_non_identifiability"]
        self.assertTrue(value["behavior_collisions"])
        self.assertTrue(value["explicit_collision_preserved"])
        examples = {
            item["example_id"]
            for collision in value["behavior_collisions"]
            for item in collision["examples"]
        }
        self.assertIn("A05-canonical", examples)
        self.assertIn("TY02-hard_negative_missing_role", examples)

    def test_metric_year_counterfactuals_pass_without_runtime_consumers(self) -> None:
        generalization = self.report["metric_year_generalization"]
        self.assertEqual(generalization["exact"], generalization["total"])
        self.assertEqual(generalization["total"], 7)
        self.assertFalse(
            any(
                any(flags.values())
                for flags in self.report["runtime_consumer_scan"].values()
            )
        )
        invariants = self.report["invariants"]
        self.assertFalse(invariants["runtime_acceptance_policy_changed"])
        self.assertFalse(invariants["threshold_changed"])
        self.assertFalse(invariants["phase_d_unlocked"])


if __name__ == "__main__":
    unittest.main()
