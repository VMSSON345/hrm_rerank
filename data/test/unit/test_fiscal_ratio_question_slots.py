"""Question-only regressions for fiscal dates and two-operand financial ratios."""
from unittest import TestCase

from hrm_finance_qa.fiscal_period_evidence import fiscal_question_surface
from hrm_finance_qa.question_planner import (
    compile_question_plan, resolve_public_question,
)


class FiscalRatioQuestionSlotTests(TestCase):
    def test_parenthesized_fiscal_date_does_not_enter_metric(self):
        question = "Tổng tài sản cuối năm tài chính 2034 (31/03/2035) là bao nhiêu?"
        plan = compile_question_plan(resolve_public_question(question))
        self.assertEqual(plan.requested_years, ("2034",))
        self.assertEqual(plan.operation, "SELECT")
        self.assertEqual(
            [(slot.metric, slot.year) for slot in plan.operand_slots],
            [("tong tai san", "2034")],
        )

    def test_fiscal_date_conflict_or_impossible_date_fails_surface_rewrite(self):
        self.assertIsNone(fiscal_question_surface(
            "Tổng tài sản năm tài chính 2034 (30/09/2032) là bao nhiêu?"))
        self.assertIsNone(fiscal_question_surface(
            "Tổng tài sản năm tài chính 2034 (31/02/2034) là bao nhiêu?"))

    def test_ratio_default_actual_keeps_both_distinct_leaves(self):
        question = (
            "Nợ phải trả chiếm bao nhiêu phần trăm tổng tài sản "
            "cuối năm 2035?"
        )
        plan = compile_question_plan(resolve_public_question(question))
        self.assertEqual(plan.operation, "RATIO_PERCENT")
        self.assertEqual(
            [(slot.metric, slot.year, slot.column_role)
             for slot in plan.operand_slots],
            [("no phai tra", "2035", "ACTUAL"),
             ("tong tai san", "2035", "ACTUAL")],
        )

    def test_margin_default_actual_keeps_numerator_and_denominator(self):
        question = (
            "Tỷ lệ lợi nhuận sau thuế trên doanh thu thuần "
            "năm 2035 là bao nhiêu phần trăm?"
        )
        plan = compile_question_plan(resolve_public_question(question))
        self.assertEqual(plan.operation, "RATIO_PERCENT")
        self.assertEqual(
            [slot.metric for slot in plan.operand_slots],
            ["loi nhuan sau thue", "doanh thu thuan"],
        )

    def test_growth_direction_words_do_not_enter_revenue_metric(self):
        question = (
            "Doanh thu thuần năm tài chính 2034 (kết thúc 30/09/2034) "
            "tăng hay giảm bao nhiêu phần trăm so với năm 2033?"
        )
        plan = compile_question_plan(resolve_public_question(question))
        self.assertEqual(plan.operation, "PERCENT_CHANGE")
        self.assertEqual(
            [(slot.metric, slot.year) for slot in plan.operand_slots],
            [("doanh thu thuan", "2034"),
             ("doanh thu thuan", "2033")],
        )