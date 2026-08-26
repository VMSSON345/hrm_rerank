from __future__ import annotations

import json
from pathlib import Path
import unittest

from hrm_finance_qa.question_planner import compile_question_plan
from scripts.evaluate_qp4_post_unit_scale_baseline_diagnostic_review import (
    _actual_plan_passes,
)
from scripts.evaluate_qp4_development_baseline import _semantic_projection


ROOT = Path(__file__).resolve().parents[1]
ADD_BASELINE = ROOT / "reports/finhrm_qp4_post_question_surface_add_source_baseline_v1.json"
ACTUAL_COHORT = ROOT / "config/finhrm_qp4_actual_plan_ratio_direction_source_cohort_v1.json"


class QP4PostQuestionSurfaceResidualImplementationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.add = json.loads(ADD_BASELINE.read_text(encoding="utf-8"))
        cls.actual = json.loads(ACTUAL_COHORT.read_text(encoding="utf-8"))

    def test_all_fifteen_add_surface_residuals_retain_parent_projection(self) -> None:
        rows = [row for row in self.add["surface_results"] if not row["invariant"]]
        self.assertEqual(len(rows), 15)
        for row in rows:
            with self.subTest(category=row["category"], company=row["company_key"]):
                expected = _semantic_projection(compile_question_plan(row["parent_question"]))
                observed = _semantic_projection(compile_question_plan(row["question"]))
                self.assertEqual(observed, expected)

    def test_all_four_reference_base_mutations_compile_the_gold_program(self) -> None:
        anchors = [
            row
            for row in self.actual["anchors"]
            if "REFERENCE_BASE" in row["pairs"]
        ]
        self.assertEqual(len(anchors), 4)
        for anchor in anchors:
            pair = anchor["pairs"]["REFERENCE_BASE"]
            plan = compile_question_plan(pair["mutated_question"])
            with self.subTest(layout=anchor["document_layout_family"]):
                self.assertTrue(
                    _actual_plan_passes(
                        _semantic_projection(plan), pair["mutated_program"]
                    )
                )

    def test_explicit_percent_reference_uses_second_operand_as_denominator(self) -> None:
        plan = compile_question_plan(
            "Doanh thu thực hiện năm 2024 bằng bao nhiêu phần trăm thực hiện năm 2023?"
        )
        self.assertEqual(plan.operation, "RATIO_PERCENT")
        self.assertEqual(
            [(slot.column_role, slot.year) for slot in plan.operand_slots],
            [("ACTUAL", "2024"), ("ACTUAL", "2023")],
        )
        self.assertEqual(plan.reference_base, "actual_2023")
        self.assertEqual(plan.program_dag[-1].inputs, ("actual_2024", "actual_2023"))

    def test_role_bearing_prefix_is_preserved(self) -> None:
        plan = compile_question_plan(
            "Theo kế hoạch, doanh thu hoạt động năm 2024 là bao nhiêu?"
        )
        self.assertEqual(plan.column_roles, ("PLAN",))
        self.assertEqual(plan.operand_slots[0].column_role, "PLAN")

    def test_incomplete_add_does_not_become_add(self) -> None:
        plan = compile_question_plan("Tổng doanh thu tại năm 2024 là bao nhiêu?")
        self.assertNotEqual(plan.operation, "ADD")
        self.assertEqual(plan.addition_role_hypotheses, ())

    def test_percent_request_without_reference_base_stays_fail_closed(self) -> None:
        plan = compile_question_plan(
            "Doanh thu thực hiện năm 2024 đạt bao nhiêu phần trăm?"
        )
        self.assertEqual(plan.abstention_reason, "missing_reference_base")
        self.assertIsNone(plan.reference_base)


if __name__ == "__main__":
    unittest.main()
