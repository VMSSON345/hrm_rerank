from __future__ import annotations

import json
from pathlib import Path
import unittest

from hrm_finance_qa.question_planner import compile_question_plan
from scripts.evaluate_qp4_development_baseline import (
    _dimension_projection,
    _semantic_projection,
)
from scripts.validate_qp4_question_surface_implementation_precommit import (
    _surface_question,
)
from scripts.validate_qp4_question_surface_implementation_precommit_correction import (
    build_effective_precommit,
)


ROOT = Path(__file__).resolve().parents[1]
CORRECTION_CONFIG = ROOT / "config/finhrm_qp4_question_surface_implementation_precommit_correction_v1.json"
REFRESH = ROOT / "reports/finhrm_qp4_post_unit_scale_development_baseline_refresh_v1.json"
DIAGNOSTIC_RAW = ROOT / "reports/finhrm_qp4_post_unit_scale_baseline_diagnostic_review_raw_v1.json"


class QP4QuestionSurfaceImplementationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        correction = json.loads(CORRECTION_CONFIG.read_text(encoding="utf-8"))
        cls.config = build_effective_precommit(correction)
        cls.refresh = json.loads(REFRESH.read_text(encoding="utf-8"))
        cls.raw = json.loads(DIAGNOSTIC_RAW.read_text(encoding="utf-8"))
        cls.canonical = {
            row["family_id"]: row["question"]
            for row in cls.refresh["compiler"]["canonical"]["details"]
        }

    def test_all_18_isolated_residuals_now_match_frozen_projection(self) -> None:
        rows = self.raw["compiler_results"]
        self.assertEqual(len(rows), 18)
        for row in rows:
            with self.subTest(group=row["group"], family=row["family_id"]):
                question = next(
                    detail["question"]
                    for detail in self.refresh["compiler"]["surface"]["details"]
                    if detail["family_id"] == row["family_id"]
                    and detail["category"] == row["group"]
                )
                self.assertEqual(
                    _semantic_projection(compile_question_plan(question)),
                    row["semantic_plan_projection"]["expected"],
                )

    def test_full_frozen_91_question_surface_matrix_is_exact(self) -> None:
        details = self.refresh["compiler"]["surface"]["details"]
        self.assertEqual(len(details), 91)
        for row in details:
            with self.subTest(category=row["category"], family=row["family_id"]):
                expected = _semantic_projection(
                    compile_question_plan(self.canonical[row["family_id"]])
                )
                observed = _semantic_projection(compile_question_plan(row["question"]))
                self.assertEqual(observed, expected)

    def test_orthogonal_78_question_hold_ahead_is_exact(self) -> None:
        recipes = self.config["semantic_diversity_contract"][
            "orthogonal_hold_ahead"
        ]["categories"]
        checked = 0
        for family, canonical_question in self.canonical.items():
            expected = _semantic_projection(compile_question_plan(canonical_question))
            for category, category_recipes in recipes.items():
                for recipe in category_recipes:
                    question = _surface_question(canonical_question, recipe)
                    with self.subTest(category=category, family=family, recipe=recipe["text"]):
                        self.assertEqual(
                            _semantic_projection(compile_question_plan(question)),
                            expected,
                        )
                    checked += 1
        self.assertEqual(checked, 78)

    def test_all_57_semantic_minimal_pairs_remain_detected(self) -> None:
        details = self.refresh["compiler"]["counterfactual"]["details"]
        self.assertEqual(len(details), 57)
        for row in details:
            base = compile_question_plan(row["base_question"])
            mutated = compile_question_plan(row["mutated_question"])
            with self.subTest(dimension=row["dimension"], pair=row["pair_id"]):
                self.assertNotEqual(
                    _dimension_projection(base, row["dimension"]),
                    _dimension_projection(mutated, row["dimension"]),
                )

    def test_metric_bearing_parenthetical_is_not_erased(self) -> None:
        plan = compile_question_plan(
            "Doanh thu hoạt động (không bao gồm doanh thu tài chính) thực hiện năm 2023 là bao nhiêu?"
        )
        metrics = " ".join(slot.metric for slot in plan.operand_slots)
        self.assertIn("khong bao gom doanh thu tai chinh", metrics)

    def test_role_bearing_prefix_is_not_treated_as_report_discourse(self) -> None:
        plan = compile_question_plan(
            "Theo kế hoạch, doanh thu hoạt động năm 2023 là bao nhiêu?"
        )
        self.assertEqual(plan.column_roles, ("PLAN",))
        self.assertEqual(plan.operand_slots[0].column_role, "PLAN")

    def test_legitimate_metric_with_negative_words_is_not_truncated(self) -> None:
        plan = compile_question_plan(
            "Chi phí không cần thiết thực hiện năm 2023 là bao nhiêu?"
        )
        self.assertIn("chi phi khong can thiet", plan.operand_slots[0].metric)

    def test_explicit_scope_and_scale_remain_typed(self) -> None:
        plan = compile_question_plan(
            "Doanh thu hoạt động thực hiện năm 2023 của công ty mẹ theo đơn vị triệu đồng là bao nhiêu?"
        )
        self.assertEqual(plan.entity_mode, "named")
        self.assertEqual(plan.requested_scale, "million_vnd")
        self.assertEqual(plan.operand_slots[0].entity_scope, f"named:{plan.entity_text}")


if __name__ == "__main__":
    unittest.main()
