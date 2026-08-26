from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from hrm_finance_qa.question_planner import compile_question_plan
from scripts.evaluate_qp4_development_baseline import (
    _dimension_projection,
    _generated_counterfactual,
    _generated_surface,
    _semantic_projection,
    _source_counterfactual,
    _source_surface,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "finhrm_qp4_development_cohort_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4DevelopmentBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_parents_checkpoint_and_runtime_are_hash_exact(self) -> None:
        for parent in self.config["parents"].values():
            self.assertEqual(_digest(ROOT / parent["path"]), parent["sha256"])
        checkpoint = self.config["checkpoint"]
        self.assertEqual(_digest(ROOT / checkpoint["path"]), checkpoint["sha256"])
        for relative, expected in self.config["runtime_hashes"].items():
            self.assertEqual(_digest(ROOT / relative), expected)

    def test_cohort_freezes_all_core_and_ambiguity_families(self) -> None:
        compiler = self.config["compiler_cohort"]
        self.assertEqual(len(compiler["core_answer_families"]), 13)
        self.assertEqual(set(compiler["ambiguity_controls"]), {"C13", "C14", "C15"})
        self.assertEqual(
            set(compiler["generated_counterfactual_dimensions"]),
            {"OPERAND_YEAR", "ENTITY_SCOPE", "UNIT_SCALE"},
        )

    def test_semantic_projection_excludes_surface_and_scores(self) -> None:
        plan = compile_question_plan(
            "Doanh thu hoạt động thực hiện năm 2023 đạt bao nhiêu phần trăm kế hoạch năm 2023?"
        )
        projection = _semantic_projection(plan)
        self.assertNotIn("question", projection)
        self.assertNotIn("normalized", projection)
        self.assertNotIn("intent_sources", json.dumps(projection, ensure_ascii=False))
        self.assertNotIn("confidence", projection)
        self.assertNotIn("candidate_rank", projection)

    def test_generated_surface_variants_do_not_use_report_literals(self) -> None:
        question = "Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?"
        for category in self.config["compiler_cohort"][
            "generated_meaning_preserving_categories"
        ]:
            variant = _generated_surface(question, category)
            self.assertIn("2023", variant)
            self.assertNotEqual(question, variant)
            self.assertNotIn(".pdf", variant.lower())

    def test_year_counterfactual_changes_only_first_explicit_year(self) -> None:
        base = "Kế hoạch doanh thu năm 2024 cao hơn thực hiện năm 2023 bao nhiêu phần trăm?"
        mutated = _generated_counterfactual(base, "OPERAND_YEAR")
        self.assertIn("2025", mutated)
        self.assertIn("2023", mutated)
        self.assertNotIn("2024", mutated)

    def test_entity_and_unit_counterfactuals_are_explicit(self) -> None:
        base = "Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?"
        entity = _generated_counterfactual(base, "ENTITY_SCOPE")
        unit = _generated_counterfactual(base, "UNIT_SCALE")
        self.assertIn("công ty mẹ", entity)
        self.assertIn("triệu đồng", unit)
        self.assertIn("năm 2023", entity)
        self.assertIn("năm 2023", unit)

    def test_dimension_projection_is_specific(self) -> None:
        actual = compile_question_plan("Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?")
        plan = compile_question_plan("Doanh thu hoạt động kế hoạch năm 2023 là bao nhiêu?")
        self.assertNotEqual(
            _dimension_projection(actual, "COLUMN_ROLE"),
            _dimension_projection(plan, "COLUMN_ROLE"),
        )
        self.assertEqual(
            _dimension_projection(actual, "OPERAND_YEAR"),
            _dimension_projection(plan, "OPERAND_YEAR"),
        )

    def test_source_surface_clause_permutation_preserves_add_parts(self) -> None:
        base = "Tổng Tài sản ngắn hạn và Tài sản dài hạn tại 2022 là bao nhiêu?"
        variant = _source_surface(base, "CLAUSE_PERMUTATION")
        self.assertTrue(variant.startswith("Tại 2022"))
        self.assertIn("Tài sản ngắn hạn và Tài sản dài hạn", variant)

    def test_source_period_mutation_is_generic_for_numeric_and_relative_periods(self) -> None:
        numeric = _source_counterfactual(
            "Tổng A và B tại 2022 là bao nhiêu?", "OPERAND_YEAR"
        )
        relative = _source_counterfactual(
            "Tổng A và B tại cuoi ky là bao nhiêu?", "OPERAND_YEAR"
        )
        self.assertIn("2023", numeric)
        self.assertIn("dau ky", relative)

    def test_source_contract_does_not_overclaim_unavailable_add_dimensions(self) -> None:
        source = self.config["source_backed_cohort"]
        self.assertEqual(
            set(source["applicable_counterfactual_dimensions"]),
            {"OPERAND_YEAR", "ENTITY_SCOPE", "UNIT_SCALE"},
        )
        self.assertEqual(
            set(source["not_applicable_or_unconfirmed_dimensions"]),
            {"COLUMN_ROLE", "DIRECTION", "REFERENCE_BASE"},
        )
        self.assertFalse(source["safe_abstention_counts_as_answer"])
        self.assertFalse(source["synthetic_layout_counts_as_confirmation"])
        self.assertFalse(
            source["ocr_cache_policy"]["mineru_auto_generate_during_baseline"]
        )
        self.assertEqual(source["per_query_timeout_seconds"], 15)
        self.assertIn("failure", source["timeout_policy"])

    def test_baseline_itself_keeps_every_release_lock_closed(self) -> None:
        for name, value in self.config["invariants"].items():
            self.assertFalse(value, name)


if __name__ == "__main__":
    unittest.main()
