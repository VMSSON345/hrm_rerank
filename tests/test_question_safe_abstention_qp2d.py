from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import unittest

from hrm_finance_qa.question_planner import compile_question_plan


ROOT = Path(__file__).resolve().parents[1]


class QuestionSafeAbstentionQP2DTests(unittest.TestCase):
    def test_three_canonical_ambiguity_controls_emit_exact_reasons(self) -> None:
        cases = {
            "Chênh lệch giữa doanh thu thực hiện và kế hoạch năm 2023 là bao nhiêu?":
                "ambiguous_signed_direction",
            "Doanh thu năm 2023 lớn hơn bao nhiêu phần trăm?":
                "missing_reference_base",
            "Doanh thu hoạt động năm 2023 là bao nhiêu khi bảng có cả kế hoạch và thực hiện?":
                "missing_column_role",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                plan = compile_question_plan(question)
                self.assertEqual(plan.abstention_reason, expected)
                self.assertIsNone(plan.reference_base)
                self.assertEqual(plan.program_dag, ())

    def test_missing_reference_base_generalizes_across_surface_forms(self) -> None:
        questions = (
            "Doanh thu năm 2023 lớn hơn bao nhiêu phần trăm?",
            "Doanh thu năm 2023 cao hơn bao nhiêu phần trăm?",
            "Doanh thu năm 2023 thấp hơn bao nhiêu phần trăm?",
            "Doanh thu thực hiện năm 2023 tăng bao nhiêu phần trăm?",
            "Doanh thu thực hiện năm 2023 giảm bao nhiêu phần trăm?",
            "Doanh thu năm 2023 chiếm bao nhiêu phần trăm?",
            "Doanh thu năm 2023 đạt bao nhiêu phần trăm?",
            "Doanh thu nam 2023 lon hon bao nhieu phan tram?",
        )
        for question in questions:
            with self.subTest(question=question):
                plan = compile_question_plan(question)
                self.assertEqual(plan.abstention_reason, "missing_reference_base")
                self.assertIsNone(plan.reference_base)
                self.assertEqual(plan.program_dag, ())

    def test_explicit_reference_repairs_do_not_false_abstain(self) -> None:
        questions = (
            "Doanh thu hoạt động thực hiện năm 2023 thay đổi bao nhiêu phần trăm "
            "so với kế hoạch năm 2023?",
            "Doanh thu hoạt động thực hiện năm 2023 lớn hơn kế hoạch năm 2023 "
            "bao nhiêu phần trăm?",
            "Doanh thu hoạt động thực hiện năm 2023 chiếm bao nhiêu phần trăm "
            "kế hoạch năm 2023?",
            "Doanh thu thực hiện năm 2024 tăng bao nhiêu phần trăm so với "
            "thực hiện năm 2023?",
        )
        for question in questions:
            with self.subTest(question=question):
                plan = compile_question_plan(question)
                self.assertEqual(plan.abstention_reason, "")
                self.assertIsNotNone(plan.reference_base)
                self.assertTrue(plan.program_dag)

    def test_missing_column_role_generalizes_across_context_order_and_language(self) -> None:
        questions = (
            "Doanh thu hoạt động năm 2023 là bao nhiêu khi bảng có cả kế hoạch và thực hiện?",
            "Doanh thu hoat dong nam 2023 la bao nhieu khi bang co ca ke hoach va thuc hien?",
            "Doanh thu hoạt động năm 2023 là bao nhiêu nếu trong bảng có cả kế hoạch và thực hiện?",
            "Khi báo cáo có đồng thời actual và plan doanh thu hoạt động năm 2023 là bao nhiêu?",
            "Trong bảng có kế hoạch và thực hiện doanh thu hoạt động năm 2023 là bao nhiêu?",
            "Doanh thu hoạt động năm 2023 là bao nhiêu khi báo cáo có đồng thời actual và plan?",
        )
        for question in questions:
            with self.subTest(question=question):
                plan = compile_question_plan(question)
                self.assertEqual(plan.abstention_reason, "missing_column_role")
                self.assertEqual(plan.column_roles, ())
                self.assertEqual(plan.operand_slots, ())
                self.assertEqual(plan.program_dag, ())

    def test_role_context_does_not_hide_an_explicit_requested_role(self) -> None:
        cases = (
            (
                "Trong bảng có cả kế hoạch và thực hiện, doanh thu hoạt động "
                "thực hiện năm 2023 là bao nhiêu?",
                "ACTUAL",
                "actual",
            ),
            (
                "Khi bảng có cả actual và plan, doanh thu hoạt động plan năm 2023 là bao nhiêu?",
                "PLAN",
                "plan",
            ),
            (
                "Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu khi bảng "
                "có cả kế hoạch và thực hiện?",
                "ACTUAL",
                "actual",
            ),
        )
        for question, role, slot_id in cases:
            with self.subTest(question=question):
                plan = compile_question_plan(question)
                self.assertEqual(plan.abstention_reason, "")
                self.assertEqual(plan.operation, "SELECT")
                self.assertEqual(plan.column_roles, (role,))
                self.assertEqual(tuple(slot.id for slot in plan.operand_slots), (slot_id,))
                self.assertEqual(plan.program_dag_operations, ("SELECT",))

    def test_all_frozen_answer_variants_retain_answerability(self) -> None:
        rows = []
        with (ROOT / "config" / "question_family_catalog_plan_actual_v1.jsonl").open(
            "r", encoding="utf-8"
        ) as handle:
            for line in handle:
                row = json.loads(line)
                if row["expected_ir"]["behavior"] == "ANSWER" and row["family_id"] in {
                    "PA01", "PA02", "PA03", "PA04", "PA05", "PA06", "PA07", "PA08", "PA09",
                    "TY01", "TY02", "TY04",
                }:
                    rows.append(row)
        self.assertEqual(len(rows), 108)
        for row in rows:
            with self.subTest(example=row["example_id"]):
                self.assertEqual(compile_question_plan(row["question"]).abstention_reason, "")

    def test_abstention_reason_is_immutable_question_side_state(self) -> None:
        plan = compile_question_plan("Doanh thu năm 2023 lớn hơn bao nhiêu phần trăm?")
        with self.assertRaises(FrozenInstanceError):
            plan.abstention_reason = ""  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
