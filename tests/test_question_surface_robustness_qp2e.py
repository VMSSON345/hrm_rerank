from __future__ import annotations

import json
from pathlib import Path
import unittest

from hrm_finance_qa.ontology import normalize_text
from hrm_finance_qa.question_planner import compile_question_plan


ROOT = Path(__file__).resolve().parents[1]


class QuestionSurfaceRobustnessQP2ETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = [
            json.loads(line)
            for line in (
                ROOT / "config" / "question_family_catalog_plan_actual_v1.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_program_role_projection_matches_all_frozen_use_sites(self) -> None:
        checked = 0
        for row in self.rows:
            if row["family_id"] not in {
                "PA01", "PA02", "PA03", "PA04", "PA05", "PA06", "PA07", "PA08", "PA09",
                "TY01", "TY02", "TY04",
            } or row["expected_ir"]["behavior"] != "ANSWER":
                continue
            with self.subTest(example=row["example_id"]):
                plan = compile_question_plan(row["question"])
                actual = plan.program_column_roles or plan.column_roles
                self.assertEqual(actual, tuple(row["expected_ir"]["column_roles"]))
                checked += 1
        self.assertEqual(checked, 108)

    def test_explicit_ambiguity_surfaces_have_exact_reason(self) -> None:
        rows = [
            row for row in self.rows
            if row["family_id"] in {
                "PA01", "PA02", "PA03", "PA04", "PA05", "PA06", "PA07", "PA08", "PA09",
                "TY01", "TY02", "TY04",
            } and row["variant"] == "ambiguity_control"
        ]
        self.assertEqual(len(rows), 12)
        for row in rows:
            with self.subTest(example=row["example_id"]):
                plan = compile_question_plan(row["question"])
                self.assertEqual(plan.abstention_reason, "ambiguous_metric_or_column_role")
                self.assertEqual(plan.program_dag, ())

    def test_distinct_years_are_a_lexical_reference_not_a_missing_base(self) -> None:
        questions = (
            "Doanh thu năm 2023 tăng bao nhiêu phần trăm so với năm 2022?",
            "Doanh thu năm 2024 cao hơn năm 2023 bao nhiêu phần trăm?",
            "Doanh thu cuối năm tăng bao nhiêu phần trăm so với đầu năm?",
        )
        for question in questions:
            with self.subTest(question=question):
                self.assertNotEqual(
                    compile_question_plan(question).abstention_reason,
                    "missing_reference_base",
                )

    def test_frozen_catalog_contains_an_unresolvable_behavior_collision(self) -> None:
        by_question: dict[str, list[dict[str, object]]] = {}
        for row in self.rows:
            by_question.setdefault(normalize_text(row["question"]), []).append(row)
        collision = by_question[
            normalize_text("Doanh thu năm 2023 tăng bao nhiêu phần trăm so với năm 2022?")
        ]
        behavior = {str(row["expected_ir"]["behavior"]) for row in collision}
        ids = {str(row["example_id"]) for row in collision}
        self.assertEqual(behavior, {"ANSWER", "ABSTAIN"})
        self.assertIn("A05-canonical", ids)
        self.assertIn("TY02-hard_negative_missing_role", ids)

    def test_legacy_runtime_column_roles_remain_unchanged(self) -> None:
        plan = compile_question_plan(
            "Doanh thu hoạt động thực hiện năm 2023 còn thiếu bao nhiêu phần trăm "
            "để đạt kế hoạch năm 2023?"
        )
        self.assertEqual(plan.column_roles, ("PLAN", "ACTUAL"))
        self.assertEqual(plan.program_column_roles, ("PLAN", "ACTUAL", "PLAN"))


if __name__ == "__main__":
    unittest.main()
