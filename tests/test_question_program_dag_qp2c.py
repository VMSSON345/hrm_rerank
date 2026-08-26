from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import unittest

from hrm_finance_qa.question_planner import (
    ProgramIntentNode,
    compile_question_plan,
)


ROOT = Path(__file__).resolve().parents[1]


def _dag_signature(plan: object) -> list[tuple[str, str, tuple[str, ...]]]:
    return [
        (node.id, node.operation, node.inputs)
        for node in plan.program_dag  # type: ignore[attr-defined]
    ]


class QuestionProgramDAGQP2CTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(
            (ROOT / "config" / "finhrm_thesis_core_v1.json").read_text(encoding="utf-8")
        )

    def test_all_answer_families_match_precommitted_reference_and_topology(self) -> None:
        checked = 0
        for family in self.contract["families"]:
            expected = family["expected"]
            if expected["behavior"] != "ANSWER":
                continue
            with self.subTest(family=family["family_id"]):
                plan = compile_question_plan(family["canonical_question"])
                expected_dag = [
                    (node["id"], node["operation"], tuple(node["inputs"]))
                    for node in expected["dag"]
                ]
                self.assertEqual(plan.reference_base, expected["reference_base"])
                self.assertEqual(_dag_signature(plan), expected_dag)
                self.assertEqual(plan.program_dag_validation_errors, ())
                checked += 1
        self.assertEqual(checked, 13)

    def test_program_nodes_are_immutable_and_versioned(self) -> None:
        plan = compile_question_plan(
            "Doanh thu hoạt động thực hiện năm 2023 đạt bao nhiêu phần trăm "
            "kế hoạch năm 2023?"
        )

        self.assertEqual(plan.program_schema_version, "question-program-intent-dag-v1")
        self.assertIsInstance(plan.program_dag[0], ProgramIntentNode)
        with self.assertRaises(FrozenInstanceError):
            plan.program_dag[0].operation = "SELECT"  # type: ignore[misc]

    def test_shortfall_and_excess_are_two_step_dags(self) -> None:
        shortfall = compile_question_plan(
            "Doanh thu hoạt động thực hiện năm 2023 còn thiếu bao nhiêu phần trăm "
            "để đạt kế hoạch năm 2023?"
        )
        excess = compile_question_plan(
            "Doanh thu hoạt động thực hiện năm 2023 vượt kế hoạch năm 2023 "
            "bao nhiêu phần trăm?"
        )

        self.assertEqual(shortfall.reference_base, "plan")
        self.assertEqual(
            _dag_signature(shortfall),
            [
                ("shortfall", "SUBTRACT", ("plan", "actual")),
                ("answer", "RATIO_PERCENT", ("shortfall", "plan")),
            ],
        )
        self.assertEqual(excess.reference_base, "plan")
        self.assertEqual(
            _dag_signature(excess),
            [
                ("excess", "SUBTRACT", ("actual", "plan")),
                ("answer", "RATIO_PERCENT", ("excess", "plan")),
            ],
        )
        self.assertEqual(shortfall.program_column_roles, ("PLAN", "ACTUAL", "PLAN"))
        self.assertEqual(excess.program_column_roles, ("ACTUAL", "PLAN", "PLAN"))

    def test_program_role_projection_preserves_role_swap(self) -> None:
        completion = compile_question_plan(
            "Doanh thu hoạt động kế hoạch năm 2023 đạt bao nhiêu phần trăm "
            "thực hiện năm 2023?"
        )

        self.assertEqual(completion.column_roles, ("ACTUAL", "PLAN"))
        self.assertEqual(completion.program_column_roles, ("PLAN", "ACTUAL"))

    def test_role_swap_changes_reference_base_by_semantics(self) -> None:
        completion = compile_question_plan(
            "Doanh thu hoạt động kế hoạch năm 2023 đạt bao nhiêu phần trăm "
            "thực hiện năm 2023?"
        )
        shortfall = compile_question_plan(
            "Doanh thu hoạt động kế hoạch năm 2023 còn thiếu bao nhiêu phần trăm "
            "để đạt thực hiện năm 2023?"
        )
        excess = compile_question_plan(
            "Doanh thu hoạt động kế hoạch năm 2023 vượt thực hiện năm 2023 "
            "bao nhiêu phần trăm?"
        )

        self.assertEqual(completion.reference_base, "actual")
        self.assertEqual(shortfall.reference_base, "actual")
        self.assertEqual(excess.reference_base, "actual")
        self.assertEqual(shortfall.program_dag[-1].inputs, ("shortfall", "actual"))
        self.assertEqual(excess.program_dag[-1].inputs, ("excess", "actual"))

    def test_two_year_reference_tracks_shifted_denominator(self) -> None:
        plan = compile_question_plan(
            "Doanh thu thực hiện năm 2024 tăng bao nhiêu phần trăm so với "
            "thực hiện năm 2023?"
        )

        self.assertEqual(plan.reference_base, "actual_2023")
        self.assertEqual(plan.program_dag[-1].inputs, ("actual_2024", "actual_2023"))

    def test_interest_bearing_debt_identity_is_explicit_but_not_executable(self) -> None:
        plan = compile_question_plan(
            "Nợ vay ngắn hạn và nợ vay dài hạn chịu lãi năm 2023 chiếm bao nhiêu "
            "phần trăm tổng nguồn vốn năm 2023?"
        )

        self.assertEqual(plan.financial_identity_ids, ())
        self.assertIn("balance.financial_debt", plan.program_identity_ids)
        self.assertTrue(plan.interest_bearing_debt_share)
        self.assertEqual(plan.reference_base, "total_capital")
        self.assertEqual(
            _dag_signature(plan),
            [
                (
                    "interest_bearing_debt",
                    "ADD",
                    ("short_term_borrowings", "long_term_borrowings"),
                ),
                (
                    "answer",
                    "RATIO_PERCENT",
                    ("interest_bearing_debt", "total_capital"),
                ),
            ],
        )

    def test_missing_role_and_ambiguity_do_not_materialize_dag(self) -> None:
        questions = (
            "Doanh thu hoạt động năm 2023 trừ năm 2023 là bao nhiêu?",
            "Doanh thu năm 2023 lớn hơn bao nhiêu phần trăm?",
            "Doanh thu hoạt động năm 2023 là bao nhiêu khi bảng có cả kế hoạch và thực hiện?",
        )
        for question in questions:
            with self.subTest(question=question):
                plan = compile_question_plan(question)
                self.assertIsNone(plan.reference_base)
                self.assertEqual(plan.program_dag, ())

    def test_validator_rejects_dangling_input_wrong_arity_denominator_and_unit(self) -> None:
        plan = compile_question_plan(
            "Doanh thu hoạt động thực hiện năm 2023 đạt bao nhiêu phần trăm "
            "kế hoạch năm 2023?"
        )
        broken = replace(
            plan,
            reference_base="actual",
            program_dag=(
                ProgramIntentNode(
                    id="answer",
                    operation="RATIO_PERCENT",
                    inputs=("missing",),
                    output_unit_family="currency",
                    intent_sources=("test:deliberate_invalid_dag",),
                ),
            ),
        )

        errors = broken.program_dag_validation_errors
        self.assertIn("wrong_arity:answer", errors)
        self.assertIn("unresolved_or_non_topological_input:answer", errors)
        self.assertIn("wrong_percent_output_unit:answer", errors)
        self.assertIn("reference_base_is_not_final_denominator", errors)


if __name__ == "__main__":
    unittest.main()
