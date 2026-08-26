from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from hrm_finance_qa.question_planner import OperandSlot, compile_question_plan


def _signature(question: str) -> tuple[tuple[str, str, str, str, str, str], ...]:
    plan = compile_question_plan(question)
    return tuple(
        (
            slot.id,
            slot.metric,
            slot.year,
            slot.column_role,
            slot.entity_scope,
            slot.unit_family,
        )
        for slot in plan.operand_slots
    )


class QuestionOperandSlotsQP2ATests(unittest.TestCase):
    def test_operand_slot_is_immutable_and_versioned(self) -> None:
        plan = compile_question_plan(
            "Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?"
        )

        self.assertEqual(plan.operand_schema_version, "question-operand-slot-v1")
        self.assertIsInstance(plan.operand_slots[0], OperandSlot)
        with self.assertRaises(FrozenInstanceError):
            plan.operand_slots[0].year = "2024"  # type: ignore[misc]

    def test_actual_and_plan_select_have_distinct_typed_slots(self) -> None:
        self.assertEqual(
            _signature("Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?"),
            (("actual", "doanh thu hoat dong", "2023", "ACTUAL", "reporting", "currency_or_source"),),
        )
        self.assertEqual(
            _signature("Doanh thu hoạt động kế hoạch năm 2023 là bao nhiêu?"),
            (("plan", "doanh thu hoat dong", "2023", "PLAN", "reporting", "currency_or_source"),),
        )

    def test_actual_over_plan_preserves_operand_order(self) -> None:
        slots = _signature(
            "Doanh thu hoạt động thực hiện năm 2023 đạt bao nhiêu phần trăm "
            "kế hoạch năm 2023?"
        )

        self.assertEqual(
            slots,
            (
                ("actual", "doanh thu hoat dong", "2023", "ACTUAL", "reporting", "currency"),
                ("plan", "doanh thu hoat dong", "2023", "PLAN", "reporting", "currency"),
            ),
        )

    def test_completion_ratio_role_swap_is_preserved_in_slots_only(self) -> None:
        plan = compile_question_plan(
            "Doanh thu hoạt động kế hoạch năm 2023 đạt bao nhiêu phần trăm "
            "thực hiện năm 2023?"
        )

        # QP2A preserves legacy runtime fields until the later QP2/QP3 gates.
        self.assertEqual(plan.column_roles, ("ACTUAL", "PLAN"))
        self.assertEqual(
            [(slot.id, slot.column_role) for slot in plan.operand_slots],
            [("plan", "PLAN"), ("actual", "ACTUAL")],
        )

    def test_decrease_amount_reverses_plan_actual_operand_order(self) -> None:
        slots = _signature(
            "Doanh thu hoạt động thực hiện năm 2023 thấp hơn kế hoạch năm 2023 "
            "bao nhiêu đồng?"
        )

        self.assertEqual([slot[0] for slot in slots], ["plan", "actual"])
        self.assertTrue(all(slot[1] == "doanh thu hoat dong" for slot in slots))

        swapped = _signature(
            "Doanh thu hoạt động kế hoạch năm 2023 thấp hơn thực hiện năm 2023 "
            "bao nhiêu đồng?"
        )
        self.assertEqual([slot[0] for slot in swapped], ["actual", "plan"])

    def test_two_year_slots_bind_each_role_to_its_explicit_year(self) -> None:
        actual = _signature(
            "Doanh thu thực hiện năm 2023 tăng bao nhiêu so với thực hiện năm 2022?"
        )
        planned = _signature(
            "Doanh thu kế hoạch năm 2023 tăng bao nhiêu so với kế hoạch năm 2022?"
        )

        self.assertEqual(
            actual,
            (
                ("actual_2023", "doanh thu", "2023", "ACTUAL", "reporting", "currency"),
                ("actual_2022", "doanh thu", "2022", "ACTUAL", "reporting", "currency"),
            ),
        )
        self.assertEqual([slot[0] for slot in planned], ["plan_2023", "plan_2022"])

    def test_year_shift_changes_slot_identity_without_other_changes(self) -> None:
        shifted = _signature(
            "Doanh thu thực hiện năm 2024 tăng bao nhiêu phần trăm so với thực hiện năm 2023?"
        )

        self.assertEqual([slot[0] for slot in shifted], ["actual_2024", "actual_2023"])
        self.assertEqual([slot[2] for slot in shifted], ["2024", "2023"])

    def test_long_discourse_prefix_is_not_part_of_operand_metric(self) -> None:
        plan = compile_question_plan(
            "Căn cứ đúng bảng số liệu tài chính trong báo cáo và giữ nguyên năm, "
            "đơn vị, phạm vi cùng vai trò của từng cột, hãy xác định: doanh thu "
            "hoạt động thực hiện năm 2023 là bao nhiêu?"
        )

        self.assertEqual(plan.operand_slots[0].metric, "doanh thu hoat dong")

    def test_missing_column_role_does_not_materialize_operand_slot(self) -> None:
        plan = compile_question_plan("Doanh thu hoạt động năm 2023 là bao nhiêu?")

        self.assertEqual(plan.column_roles, ())
        self.assertEqual(plan.operand_slots, ())

    def test_intent_provenance_has_no_document_or_numeric_selection_signal(self) -> None:
        plan = compile_question_plan(
            "Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?"
        )
        sources = plan.operand_slots[0].intent_sources

        self.assertIn("metric:question_lexical", sources)
        self.assertIn("year:question_explicit", sources)
        self.assertIn("column_role:question_explicit", sources)
        rendered = " ".join(sources)
        for forbidden in ("value", "candidate", "page", "table", "left", "right"):
            self.assertNotIn(forbidden, rendered)

    def test_interest_bearing_debt_share_emits_three_identity_leaves_only(self) -> None:
        plan = compile_question_plan(
            "Nợ vay ngắn hạn và nợ vay dài hạn chịu lãi năm 2023 chiếm bao nhiêu "
            "phần trăm tổng nguồn vốn năm 2023?"
        )

        self.assertEqual(
            [slot.id for slot in plan.operand_slots],
            ["short_term_borrowings", "long_term_borrowings", "total_capital"],
        )
        self.assertEqual({slot.year for slot in plan.operand_slots}, {"2023"})
        self.assertEqual({slot.column_role for slot in plan.operand_slots}, {"ACTUAL"})
        self.assertTrue(
            all(
                "column_role:financial_statement_actual_default" in slot.intent_sources
                for slot in plan.operand_slots
            )
        )


if __name__ == "__main__":
    unittest.main()
