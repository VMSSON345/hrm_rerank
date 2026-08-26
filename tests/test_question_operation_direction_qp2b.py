from __future__ import annotations

import unittest

from hrm_finance_qa.question_planner import DirectionIntent, compile_question_plan


class QuestionOperationDirectionQP2BTests(unittest.TestCase):
    def assert_operation_direction(
        self, question: str, operation: str, direction: DirectionIntent
    ) -> None:
        plan = compile_question_plan(question)
        self.assertEqual(plan.operation, operation)
        self.assertEqual(plan.direction_intent, direction)

    def test_explicit_subtraction_is_left_minus_right(self) -> None:
        self.assert_operation_direction(
            "Doanh thu hoạt động thực hiện năm 2023 trừ kế hoạch năm 2023 là bao nhiêu?",
            "SUBTRACT",
            DirectionIntent.LEFT_MINUS_RIGHT,
        )

    def test_excess_amount_is_subtraction(self) -> None:
        self.assert_operation_direction(
            "Doanh thu hoạt động thực hiện năm 2023 vượt kế hoạch năm 2023 bao nhiêu đồng?",
            "SUBTRACT",
            DirectionIntent.LEFT_MINUS_RIGHT,
        )

    def test_same_year_percent_change_uses_semantic_role_order(self) -> None:
        self.assert_operation_direction(
            "Doanh thu hoạt động thực hiện năm 2023 thay đổi bao nhiêu phần trăm "
            "so với kế hoạch năm 2023?",
            "PERCENT_CHANGE",
            DirectionIntent.LEFT_MINUS_RIGHT,
        )

    def test_shortfall_percent_has_explicit_plan_minus_actual_direction(self) -> None:
        self.assert_operation_direction(
            "Doanh thu hoạt động thực hiện năm 2023 còn thiếu bao nhiêu phần trăm "
            "để đạt kế hoạch năm 2023?",
            "RATIO_PERCENT",
            DirectionIntent.PLAN_MINUS_ACTUAL,
        )

    def test_excess_percent_has_explicit_actual_minus_plan_direction(self) -> None:
        self.assert_operation_direction(
            "Doanh thu hoạt động thực hiện năm 2023 vượt kế hoạch năm 2023 "
            "bao nhiêu phần trăm?",
            "RATIO_PERCENT",
            DirectionIntent.ACTUAL_MINUS_PLAN,
        )

    def test_cross_year_plan_actual_prefers_language_order_over_recency(self) -> None:
        self.assert_operation_direction(
            "Kế hoạch doanh thu năm 2024 cao hơn thực hiện năm 2023 bao nhiêu phần trăm?",
            "PERCENT_CHANGE",
            DirectionIntent.LEFT_MINUS_RIGHT,
        )

    def test_role_swaps_preserve_compositional_direction(self) -> None:
        cases = (
            (
                "Doanh thu hoạt động kế hoạch năm 2023 trừ thực hiện năm 2023 là bao nhiêu?",
                "SUBTRACT",
                DirectionIntent.LEFT_MINUS_RIGHT,
            ),
            (
                "Doanh thu hoạt động kế hoạch năm 2023 vượt thực hiện năm 2023 bao nhiêu đồng?",
                "SUBTRACT",
                DirectionIntent.LEFT_MINUS_RIGHT,
            ),
            (
                "Doanh thu hoạt động kế hoạch năm 2023 còn thiếu bao nhiêu phần trăm "
                "để đạt thực hiện năm 2023?",
                "RATIO_PERCENT",
                DirectionIntent.ACTUAL_MINUS_PLAN,
            ),
            (
                "Doanh thu hoạt động kế hoạch năm 2023 vượt thực hiện năm 2023 "
                "bao nhiêu phần trăm?",
                "RATIO_PERCENT",
                DirectionIntent.PLAN_MINUS_ACTUAL,
            ),
            (
                "Thực hiện doanh thu năm 2024 cao hơn kế hoạch năm 2023 bao nhiêu phần trăm?",
                "PERCENT_CHANGE",
                DirectionIntent.LEFT_MINUS_RIGHT,
            ),
        )
        for question, operation, direction in cases:
            with self.subTest(question=question):
                self.assert_operation_direction(question, operation, direction)

    def test_year_shift_does_not_change_role_based_direction(self) -> None:
        plan = compile_question_plan(
            "Kế hoạch doanh thu năm 2025 cao hơn thực hiện năm 2024 bao nhiêu phần trăm?"
        )

        self.assertEqual(plan.requested_years, ("2025", "2024"))
        self.assertEqual(plan.column_roles, ("PLAN", "ACTUAL"))
        self.assertEqual(plan.direction_intent, DirectionIntent.LEFT_MINUS_RIGHT)

    def test_long_and_no_diacritic_surfaces_compile_identically(self) -> None:
        questions = (
            "Can cu dung bang so lieu tai chinh trong bao cao va giu nguyen nam, "
            "don vi, pham vi cung vai tro cua tung cot, hay xac dinh: doanh thu "
            "hoat dong thuc hien nam 2023 vuot ke hoach nam 2023 bao nhieu dong?",
            "Doanh thu hoat dong thuc hien nam 2023 vuot ke hoach nam 2023 bao nhieu dong?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assert_operation_direction(
                    question, "SUBTRACT", DirectionIntent.LEFT_MINUS_RIGHT
                )

    def test_regular_two_year_actual_change_remains_new_minus_old(self) -> None:
        plan = compile_question_plan(
            "Doanh thu thực hiện năm 2023 tăng bao nhiêu phần trăm so với thực hiện năm 2022?"
        )

        self.assertEqual(plan.operation, "PERCENT_CHANGE")
        self.assertEqual(plan.direction_intent, DirectionIntent.NEW_MINUS_OLD)


if __name__ == "__main__":
    unittest.main()
