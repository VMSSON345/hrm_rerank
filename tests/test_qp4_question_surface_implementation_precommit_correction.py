from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.validate_qp4_question_surface_implementation_precommit_correction import (
    build_effective_precommit,
    build_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/finhrm_qp4_question_surface_implementation_precommit_correction_v1.json"
REPORT = ROOT / "reports/finhrm_qp4_question_surface_implementation_precommit_correction_v1.json"
LOCK = ROOT / "reports/finhrm_qp4_question_surface_implementation_precommit_correction_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4QuestionSurfacePrecommitCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_correction_replaces_only_the_typed_forecast_conflict(self) -> None:
        correction = self.config["correction"]
        self.assertEqual(correction["category"], "IRRELEVANT_DISTRACTOR_CLAUSE")
        self.assertEqual(correction["recipe_index_zero_based"], 1)
        self.assertIn("dự báo", correction["invalid_recipe"]["text"])
        self.assertNotIn("dự báo", correction["replacement_recipe"]["text"])
        self.assertIn("FORECAST column-role alias", correction["invalidity"])
        additional = self.config["additional_correction"]
        self.assertEqual(additional["category"], "BILINGUAL_PARENTHETICAL")
        self.assertEqual(additional["recipe_index_zero_based"], 1)
        self.assertEqual(additional["replacement_recipe"]["text"], "as reported / theo nguồn")

    def test_effective_parent_changes_exactly_one_recipe(self) -> None:
        parent_path = ROOT / self.config["parents"]["original_precommit_config"]["path"]
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        effective = build_effective_precommit(self.config)
        parent_recipes = parent["semantic_diversity_contract"]["orthogonal_hold_ahead"]["categories"]
        effective_recipes = effective["semantic_diversity_contract"]["orthogonal_hold_ahead"]["categories"]
        changed = []
        for category in parent_recipes:
            for index, recipe in enumerate(parent_recipes[category]):
                if recipe != effective_recipes[category][index]:
                    changed.append((category, index))
        self.assertEqual(
            changed,
            [
                ("BILINGUAL_PARENTHETICAL", 1),
                ("IRRELEVANT_DISTRACTOR_CLAUSE", 1),
            ],
        )

    def test_scope_remains_one_file_and_two_symbols(self) -> None:
        scope = self.config["implementation_scope_unchanged"]
        self.assertEqual(scope["runtime_file_whitelist"], ["src/hrm_finance_qa/question_planner.py"])
        self.assertEqual(
            scope["exact_symbols"],
            {
                "new_helper": ["_strip_nonsemantic_metric_surface"],
                "existing_function_edit": ["compile_question_plan"],
            },
        )
        self.assertFalse(scope["operation_intent_repair_authorized"])
        self.assertFalse(scope["retrieval_source_binding_verifier_edit_authorized"])

    def test_parent_and_runtime_hashes_are_exact(self) -> None:
        for parent in self.config["parents"].values():
            self.assertEqual(_digest(ROOT / parent["path"]), parent["sha256"])
        for relative, expected in self.config["runtime_hashes"].items():
            self.assertEqual(_digest(ROOT / relative), expected)

    def test_invariants_and_history_remain_closed(self) -> None:
        self.assertFalse(self.config["correction"]["historical_artifacts_modified"])
        self.assertFalse(self.config["correction"]["runtime_modified"])
        for name, value in self.config["invariants"].items():
            self.assertFalse(value, name)


class QP4QuestionSurfacePrecommitCorrectionReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (REPORT.exists() and LOCK.exists()):
            raise unittest.SkipTest("Generated correction artifacts are absent")
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_reproducible_and_passes_without_runtime_change(self) -> None:
        self.assertEqual(build_report(CONFIG), self.report)
        self.assertTrue(self.report["correction_gate_passed"])
        self.assertEqual(self.report["failed_gates"], [])
        self.assertFalse(self.report["runtime_changed"])
        self.assertEqual(self.report["effective_hold_ahead"]["questions"], 78)
        self.assertEqual(self.report["effective_hold_ahead"]["exact_before_implementation"], 42)

    def test_lock_hashes_are_exact(self) -> None:
        self.assertEqual(self.lock["status"], "complete")
        self.assertTrue(self.lock["correction_gate_passed"])
        self.assertFalse(self.lock["runtime_changed"])
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
