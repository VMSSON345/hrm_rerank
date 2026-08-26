from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.validate_finhrm_thesis_core import DEFAULT_CONTRACT, validate_contract


ROOT = Path(__file__).resolve().parents[1]


class FinHRMThesisCorePrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(DEFAULT_CONTRACT.read_text(encoding="utf-8"))
        cls.families = {
            item["family_id"]: item for item in cls.contract["families"]
        }

    def test_contract_validator_passes(self) -> None:
        report = validate_contract()
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["counts"]["families"], 16)
        self.assertEqual(report["counts"]["answer_families"], 13)
        self.assertEqual(report["counts"]["abstain_families"], 3)

    def test_plan_actual_direction_and_reference_base_are_explicit(self) -> None:
        pa05 = self.families["PA05"]["expected"]
        pa08 = self.families["PA08"]["expected"]
        pa09 = self.families["PA09"]["expected"]

        self.assertEqual(pa05["dag"][0]["inputs"], ["plan", "actual"])
        self.assertEqual(pa08["direction_intent"], "PLAN_MINUS_ACTUAL")
        self.assertEqual(pa08["reference_base"], "plan")
        self.assertEqual(pa09["direction_intent"], "ACTUAL_MINUS_PLAN")
        self.assertEqual(pa09["reference_base"], "plan")

    def test_two_year_roles_are_bound_per_operand(self) -> None:
        ty04 = self.families["TY04"]["expected"]
        self.assertEqual(
            [(item["year"], item["column_role"]) for item in ty04["operands"]],
            [("2024", "PLAN"), ("2023", "ACTUAL")],
        )
        self.assertEqual(ty04["reference_base"], "actual_2023")

    def test_interest_bearing_debt_share_is_a_two_node_dag(self) -> None:
        identity = self.families["IBDS01"]["expected"]
        self.assertEqual(
            [node["operation"] for node in identity["dag"]],
            ["ADD", "RATIO_PERCENT"],
        )
        self.assertEqual(identity["reference_base"], "total_capital")
        self.assertNotIn(
            "total_liabilities",
            {item["id"] for item in identity["operands"]},
        )

    def test_ambiguity_controls_cannot_materialize_a_program(self) -> None:
        expected_reasons = {
            "C13": "ambiguous_signed_direction",
            "C14": "missing_reference_base",
            "C15": "missing_column_role",
        }
        for family_id, reason in expected_reasons.items():
            expected = self.families[family_id]["expected"]
            self.assertEqual(expected["behavior"], "ABSTAIN")
            self.assertIsNone(expected["root_operation"])
            self.assertEqual(expected["operands"], [])
            self.assertEqual(expected["dag"], [])
            self.assertEqual(expected["abstention_reason"], reason)

    def test_sealed_test_and_threshold_invariants_are_frozen(self) -> None:
        split = self.contract["split_protocol"]
        invariants = self.contract["invariants"]
        self.assertTrue(split["no_company_overlap"])
        self.assertFalse(split["sealed_test_threshold_selection_allowed"])
        self.assertFalse(invariants["protected_evaluation_sources_allowed"])
        self.assertFalse(invariants["threshold_change_allowed_during_precommit"])
        self.assertFalse(invariants["fine_tuning_allowed_during_precommit"])


if __name__ == "__main__":
    unittest.main()
