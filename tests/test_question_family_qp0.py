from __future__ import annotations

import unittest

from scripts.audit_question_plan_compiler import _checks
from scripts.build_question_family_catalog_plan_actual import (
    _base_specs,
    _expanded_records,
)


class QuestionFamilyCatalogQP0Tests(unittest.TestCase):
    def test_catalog_covers_all_precommitted_families_and_variants(self) -> None:
        records = _expanded_records(_base_specs())
        families = {str(row["family_id"]) for row in records}

        self.assertEqual(len(families), 59)
        self.assertEqual(len(records), 299)
        for prefix, count in (("PA", 12), ("TY", 12)):
            for index in range(1, count + 1):
                family = f"{prefix}{index:02d}"
                variants = {
                    str(row["variant"])
                    for row in records
                    if row["family_id"] == family
                }
                self.assertEqual(len(variants), 11)
                self.assertIn("minimal_pair_role_swap", variants)
                self.assertIn("minimal_pair_year_shift", variants)
                self.assertIn("hard_negative_missing_role", variants)
                self.assertIn("ambiguity_control", variants)

    def test_plan_actual_contract_carries_column_roles(self) -> None:
        records = _expanded_records(_base_specs())
        canonical = next(
            row for row in records if row["example_id"] == "PA03-canonical"
        )

        self.assertEqual(
            canonical["expected_ir"]["column_roles"], ["ACTUAL", "PLAN"]
        )
        self.assertEqual(
            canonical["expected_ir"]["root_operation"], "RATIO_PERCENT"
        )

    def test_two_year_percentage_point_contract_is_not_percent_change(self) -> None:
        records = _expanded_records(_base_specs())
        ty06 = next(row for row in records if row["example_id"] == "TY06-canonical")
        ty07 = next(row for row in records if row["example_id"] == "TY07-canonical")

        self.assertEqual(ty06["expected_ir"]["root_operation"], "SUBTRACT")
        self.assertEqual(
            ty06["expected_ir"]["requested_unit_family"], "percentage_point"
        )
        self.assertEqual(ty07["expected_ir"]["root_operation"], "PERCENT_CHANGE")
        self.assertEqual(ty07["expected_ir"]["requested_unit_family"], "percent")

    def test_compiler_audit_marks_missing_column_roles(self) -> None:
        expected = {
            "behavior": "ANSWER",
            "root_operation": "SELECT",
            "requested_years": ["2023"],
            "column_roles": ["ACTUAL"],
            "answer_mode": "SINGLE_SCALAR",
            "direction_intent": None,
            "requested_unit_family": "",
            "program_dag_operations": ["SELECT"],
            "entity_mode": "reporting",
        }
        actual = {
            "operation": None,
            "operation_candidates": [],
            "requested_years": ["2023"],
            "column_roles": [],
            "answer_mode": "SINGLE_SCALAR",
            "direction_intent": None,
            "requested_unit_family": "",
            "program_dag_operations": [],
            "entity_mode": "reporting",
            "abstention_reason": "",
        }

        checks = _checks(expected, actual)

        self.assertFalse(checks["operation"])
        self.assertFalse(checks["column_roles"])
        self.assertTrue(checks["requested_years"])


if __name__ == "__main__":
    unittest.main()
