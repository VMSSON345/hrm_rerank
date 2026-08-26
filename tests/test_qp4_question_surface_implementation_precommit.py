from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.validate_qp4_question_surface_implementation_precommit import (
    build_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/finhrm_qp4_question_surface_implementation_precommit_v1.json"
REPORT = ROOT / "reports/finhrm_qp4_question_surface_implementation_precommit_v1.json"
LOCK = ROOT / "reports/finhrm_qp4_question_surface_implementation_precommit_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4QuestionSurfaceImplementationPrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_failure_is_generic_and_isolated_to_operand_metric(self) -> None:
        evidence = self.config["generic_failure_evidence"]
        self.assertEqual(evidence["first_failing_layer"], "QUESTION_PLAN")
        self.assertEqual(evidence["compiler_residuals"], 18)
        self.assertEqual(evidence["exact_changed_projection_fields"], ["operand_slots"])
        self.assertEqual(evidence["changed_operand_subfield"], "metric")
        self.assertEqual(evidence["groups"]["DISCOURSE_PREFIX_SUFFIX"]["families"], 12)
        self.assertEqual(evidence["groups"]["BILINGUAL_PARENTHETICAL"]["families"], 3)
        self.assertEqual(evidence["groups"]["IRRELEVANT_DISTRACTOR_CLAUSE"]["families"], 3)

    def test_question_diversity_covers_families_semantics_and_surfaces(self) -> None:
        diversity = self.config["semantic_diversity_contract"]
        self.assertEqual(diversity["core_answer_families"], 13)
        self.assertEqual(diversity["ambiguity_controls"], 3)
        self.assertEqual(
            set(diversity["root_operation_semantics"]),
            {"SELECT", "SUBTRACT", "RATIO_PERCENT", "PERCENT_CHANGE"},
        )
        self.assertEqual(diversity["frozen_surface_categories"], 7)
        self.assertEqual(diversity["frozen_surface_questions"], 91)
        self.assertEqual(diversity["frozen_minimal_pairs"], 57)
        self.assertEqual(len(diversity["minimal_pair_dimensions"]), 6)

    def test_orthogonal_hold_ahead_prevents_single_template_repairs(self) -> None:
        hold = self.config["semantic_diversity_contract"]["orthogonal_hold_ahead"]
        self.assertEqual(hold["families"], 13)
        self.assertEqual(hold["recipes_per_category"], 2)
        self.assertEqual(len(hold["categories"]), 3)
        self.assertEqual(hold["total_questions"], 78)
        self.assertEqual(hold["expected_postimplementation_semantic_projection_exact"], "78/78")
        self.assertFalse(hold["runtime_may_match_recipe_literals"])
        self.assertFalse(hold["counts_as_company_or_layout_confirmation"])

    def test_anti_erasure_controls_keep_all_six_semantic_dimensions(self) -> None:
        controls = self.config["semantic_diversity_contract"]["anti_erasure_controls"]
        self.assertTrue(controls["reuse_all_frozen_minimal_pairs"])
        self.assertEqual(controls["expected_postimplementation_detection"], "57/57")
        qualifiers = " ".join(controls["typed_qualifiers_that_must_never_be_discarded_as_surface"])
        for required in ("column role", "operand period", "entity scope", "unit family", "signed direction", "reference base"):
            self.assertIn(required, qualifiers)

    def test_next_runtime_scope_is_one_file_and_two_symbols(self) -> None:
        scope = self.config["implementation_scope"]
        self.assertEqual(scope["stage"], "operand_metric_surface_normalization_only")
        self.assertEqual(scope["runtime_file_whitelist"], ["src/hrm_finance_qa/question_planner.py"])
        self.assertEqual(scope["exact_symbols"]["new_helper"], ["_strip_nonsemantic_metric_surface"])
        self.assertEqual(scope["exact_symbols"]["existing_function_edit"], ["compile_question_plan"])
        self.assertEqual(
            scope["allowed_observable_change"],
            [
                "metric_text after removal of proven non-semantic surface framing",
                "operand_slots[*].metric derived from that metric_text",
            ],
        )

    def test_broader_parser_source_and_proof_edits_remain_locked(self) -> None:
        scope = self.config["implementation_scope"]
        for item in (
            "operation-intent repair",
            "ADD structural-frame repair",
            "SHORT_LONG_EQUIVALENT repair",
            "CLAUSE_PERMUTATION repair",
            "retrieval changes",
            "source loading or extraction changes",
            "candidate binding or program synthesis changes",
            "symbolic verifier or proof changes",
        ):
            self.assertIn(item, scope["non_goals"])
        self.assertEqual(len(self.config["frozen_runtime_files"]), 10)

    def test_source_replay_is_diverse_but_does_not_broaden_this_scope(self) -> None:
        source = self.config["source_backed_replay_contract"]
        self.assertEqual(source["anchors"], 5)
        self.assertEqual(source["companies"], 5)
        self.assertEqual(source["source_documents"], 5)
        self.assertEqual(source["real_layout_families"], 3)
        self.assertEqual(source["surface_queries"], 35)
        self.assertTrue(source["full_source_surface_replay_required"])
        self.assertFalse(source["source_residual_improvement_required_to_close_this_compiler_stage"])
        self.assertFalse(source["synthetic_question_counts_as_real_layout_confirmation"])

    def test_parent_runtime_and_checkpoint_hashes_are_exact(self) -> None:
        self.assertEqual(_digest(ROOT / self.config["claude_md"]["path"]), self.config["claude_md"]["sha256"])
        for parent in self.config["parents"].values():
            self.assertEqual(_digest(ROOT / parent["path"]), parent["sha256"])
        for relative, expected in self.config["runtime_hashes_before_implementation"].items():
            self.assertEqual(_digest(ROOT / relative), expected)
        checkpoint = self.config["checkpoint"]
        self.assertEqual(_digest(ROOT / checkpoint["path"]), checkpoint["sha256"])

    def test_precommit_and_later_phase_invariants_remain_closed(self) -> None:
        self.assertFalse(self.config["runtime_changed_by_this_precommit"])
        self.assertFalse(self.config["implementation_executed_by_this_precommit"])
        self.assertFalse(self.config["runtime_change_authorized_in_same_step"])
        for name, value in self.config["invariants"].items():
            self.assertFalse(value, name)


class QP4QuestionSurfaceImplementationPrecommitReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (REPORT.exists() and LOCK.exists()):
            raise unittest.SkipTest("Generated question-surface precommit artifacts are absent")
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_exactly_reproducible(self) -> None:
        self.assertEqual(build_report(CONFIG), self.report)

    def test_report_passes_without_runtime_change(self) -> None:
        self.assertTrue(self.report["precommit_gate_passed"])
        self.assertEqual(self.report["failed_gates"], [])
        self.assertFalse(self.report["runtime_changed"])
        self.assertFalse(self.report["runtime_change_authorized_in_same_step"])
        self.assertFalse(self.report["qp4_complete"])
        self.assertFalse(self.report["qp5_authorized"])
        self.assertEqual(self.report["phase_d"], "locked")

    def test_lock_hashes_and_next_step_scope_are_exact(self) -> None:
        self.assertEqual(self.lock["status"], "complete")
        self.assertTrue(self.lock["precommit_gate_passed"])
        self.assertFalse(self.lock["runtime_changed"])
        self.assertFalse(self.lock["runtime_change_authorized_in_same_step"])
        self.assertFalse(self.lock["retrieval_source_binding_or_verifier_edit_authorized"])
        self.assertEqual(
            self.lock["runtime_file_whitelist_for_next_step"],
            ["src/hrm_finance_qa/question_planner.py"],
        )
        self.assertEqual(
            self.lock["exact_symbol_whitelist_for_next_step"],
            {
                "new_helper": ["_strip_nonsemantic_metric_surface"],
                "existing_function_edit": ["compile_question_plan"],
            },
        )
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
