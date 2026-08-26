from __future__ import annotations

import json
from pathlib import Path
import unittest

from hrm_finance_qa.question_planner import compile_question_plan
from scripts.evaluate_qp4_development_baseline import (
    _compiler_baseline,
    _dimension_projection,
)


ROOT = Path(__file__).resolve().parents[1]
COHORT_CONFIG = ROOT / "config" / "finhrm_qp4_development_cohort_v1.json"


class QP4UnitScaleQuestionPlannerImplementationTests(unittest.TestCase):
    def test_explicit_monetary_scales_are_typed(self) -> None:
        cases = (
            ("Doanh thu năm 2024 là bao nhiêu VND?", "vnd"),
            ("Doanh thu năm 2024 là bao nhiêu đồng?", "vnd"),
            ("Doanh thu năm 2024 theo đơn vị đồng là bao nhiêu?", "vnd"),
            ("Doanh thu năm 2024 theo đồng là bao nhiêu?", "vnd"),
            ("Doanh thu năm 2024 là bao nhiêu nghìn đồng?", "thousand_vnd"),
            ("Doanh thu năm 2024 là bao nhiêu triệu đồng?", "million_vnd"),
            ("Doanh thu năm 2024 là bao nhiêu tỷ đồng?", "billion_vnd"),
        )
        for question, expected in cases:
            with self.subTest(question=question):
                plan = compile_question_plan(question)
                self.assertEqual(plan.requested_scale, expected)
                self.assertEqual(plan.requested_unit_family, "currency")
                self.assertEqual(plan.abstention_reason, "")

    def test_missing_scale_is_none_and_is_not_inferred(self) -> None:
        plan = compile_question_plan("Doanh thu năm 2024 là bao nhiêu?")
        self.assertIsNone(plan.requested_scale)
        self.assertEqual(plan.requested_unit_family, "")

    def test_scale_counterfactual_changes_only_unit_scale_projection(self) -> None:
        base = compile_question_plan("Doanh thu năm 2024 là bao nhiêu?")
        mutated = compile_question_plan(
            "Doanh thu năm 2024 theo đơn vị triệu đồng là bao nhiêu?"
        )
        self.assertNotEqual(
            _dimension_projection(base, "UNIT_SCALE"),
            _dimension_projection(mutated, "UNIT_SCALE"),
        )
        self.assertEqual(base.requested_years, mutated.requested_years)
        self.assertEqual(base.operation, mutated.operation)
        self.assertEqual(base.metric_text, mutated.metric_text)

    def test_conflicting_explicit_scales_fail_closed(self) -> None:
        plan = compile_question_plan(
            "Doanh thu năm 2024 theo nghìn đồng hoặc triệu đồng là bao nhiêu?"
        )
        self.assertEqual(plan.requested_scale, "conflict")
        self.assertEqual(plan.requested_unit_family, "currency")
        self.assertEqual(plan.abstention_reason, "conflicting_requested_scale")

    def test_percent_with_monetary_scale_fails_closed(self) -> None:
        plan = compile_question_plan(
            "Biên lợi nhuận năm 2024 là bao nhiêu phần trăm theo triệu đồng?"
        )
        self.assertEqual(plan.requested_scale, "million_vnd")
        self.assertEqual(plan.requested_unit_family, "percent")
        self.assertEqual(plan.abstention_reason, "incompatible_requested_scale")

    def test_ratio_with_monetary_scale_fails_closed(self) -> None:
        plan = compile_question_plan(
            "Hệ số nợ trên tổng tài sản năm 2024 theo tỷ đồng là bao nhiêu?"
        )
        self.assertEqual(plan.requested_scale, "billion_vnd")
        self.assertEqual(plan.requested_unit_family, "ratio")
        self.assertEqual(plan.abstention_reason, "incompatible_requested_scale")

    def test_existing_stricter_abstention_reason_is_retained(self) -> None:
        plan = compile_question_plan(
            "Chênh lệch giữa Doanh thu và Chi phí theo nghìn đồng hoặc "
            "triệu đồng là bao nhiêu?"
        )
        self.assertEqual(plan.requested_scale, "conflict")
        self.assertEqual(plan.abstention_reason, "ambiguous_signed_direction")

    def test_all_thirteen_compiler_families_detect_unit_scale(self) -> None:
        config = json.loads(COHORT_CONFIG.read_text(encoding="utf-8"))
        compiler = _compiler_baseline(config)
        unit = compiler["counterfactual"]["by_dimension"]["UNIT_SCALE"]
        self.assertEqual(unit["pairs"], 13)
        self.assertEqual(unit["families"], 13)
        self.assertEqual(unit["detected"], 13)
        self.assertEqual(unit["detection_rate"], 1.0)

    def test_canonical_ambiguity_and_surface_contracts_are_retained(self) -> None:
        config = json.loads(COHORT_CONFIG.read_text(encoding="utf-8"))
        compiler = _compiler_baseline(config)
        self.assertEqual(compiler["canonical"]["exact"], 13)
        self.assertEqual(compiler["canonical"]["total"], 13)
        self.assertEqual(compiler["ambiguity"]["exact"], 3)
        self.assertEqual(compiler["ambiguity"]["total"], 3)
        self.assertEqual(compiler["surface"]["exact"], 73)
        self.assertEqual(compiler["surface"]["total"], 91)


if __name__ == "__main__":
    unittest.main()
