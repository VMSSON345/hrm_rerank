from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from hrm_finance_qa.question_planner import compile_question_plan
from scripts.evaluate_qp4_development_baseline import _dimension_projection
from scripts.validate_qp4_unit_scale_question_planner_precommit import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "config"
    / "finhrm_qp4_unit_scale_question_planner_implementation_precommit_v1.json"
)
REPORT = (
    ROOT
    / "reports"
    / "finhrm_qp4_unit_scale_question_planner_implementation_precommit_v1.json"
)
LOCK = (
    ROOT
    / "reports"
    / "finhrm_qp4_unit_scale_question_planner_implementation_precommit_lock_v1.json"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4UnitScaleQuestionPlannerPrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_generic_failure_evidence_spans_families_companies_and_layouts(self) -> None:
        evidence = self.config["generic_failure_evidence"]
        compiler = evidence["compiler"]
        source = evidence["source_backed"]
        self.assertEqual(compiler["pairs"], 13)
        self.assertEqual(compiler["families"], 13)
        self.assertEqual(compiler["detected"], 0)
        self.assertEqual(source["generic_failure_companies"], 4)
        self.assertEqual(source["generic_failure_sources"], 4)
        self.assertEqual(source["generic_failure_real_layouts"], 3)
        self.assertEqual(compiler["attributed_layer"], "question_planner")
        self.assertEqual(source["attributed_layer"], "question_planner")

    def test_current_runtime_collapses_explicit_monetary_scales(self) -> None:
        base = compile_question_plan("Doanh thu năm 2024 là bao nhiêu đồng?")
        mutated = compile_question_plan(
            "Doanh thu năm 2024 theo đơn vị triệu đồng là bao nhiêu?"
        )
        self.assertFalse(hasattr(base, "requested_scale"))
        self.assertFalse(hasattr(mutated, "requested_scale"))
        self.assertEqual(
            _dimension_projection(base, "UNIT_SCALE"),
            _dimension_projection(mutated, "UNIT_SCALE"),
        )
        self.assertEqual(base.requested_unit_family, "currency")
        self.assertEqual(mutated.requested_unit_family, "currency")

    def test_next_runtime_scope_is_one_file_and_exact_symbols(self) -> None:
        scope = self.config["implementation_scope"]
        self.assertEqual(
            scope["runtime_file_whitelist"],
            ["src/hrm_finance_qa/question_planner.py"],
        )
        self.assertEqual(scope["stage"], "typed_question_intent_only")
        self.assertEqual(scope["exact_symbols"]["QuestionPlan"], ["requested_scale"])
        self.assertEqual(
            scope["exact_symbols"]["new_helper"], ["_compile_requested_scale"]
        )
        self.assertEqual(
            scope["exact_symbols"]["existing_function_edit"],
            ["compile_question_plan"],
        )
        self.assertEqual(
            set(scope["typed_scale_values"]),
            {"vnd", "thousand_vnd", "million_vnd", "billion_vnd", "conflict"},
        )

    def test_contract_fails_closed_without_inference_or_conversion(self) -> None:
        scope = self.config["implementation_scope"]
        rules = " ".join(scope["semantic_rules"])
        self.assertIsNone(scope["missing_scale_value"])
        self.assertIn("Parse scale only from explicit question text.", rules)
        self.assertIn("requested_scale=None", rules)
        self.assertIn("conflicting_requested_scale", rules)
        self.assertIn("incompatible_requested_scale", rules)
        self.assertIn("unit conversion execution", scope["non_goals"])
        self.assertIn("candidate value rescaling", scope["non_goals"])

    def test_retrieval_binding_and_other_qp4_dimensions_remain_frozen(self) -> None:
        scope = self.config["implementation_scope"]
        for item in (
            "retrieval changes",
            "binding changes",
            "COLUMN_ROLE repair",
            "DIRECTION repair",
            "REFERENCE_BASE repair",
        ):
            self.assertIn(item, scope["non_goals"])
        self.assertNotIn(
            "src/hrm_finance_qa/question_planner.py", self.config["frozen_files"]
        )
        self.assertEqual(len(self.config["frozen_files"]), 7)

    def test_postimplementation_gates_freeze_safe_replay_and_staged_source_claim(self) -> None:
        gates = self.config["post_implementation_gates"]
        self.assertEqual(gates["compiler_unit_scale_detection"], "13/13")
        self.assertEqual(gates["source_backed_old_winner_false_accept_target"], 0)
        self.assertTrue(gates["source_backed_unit_scale_replay_required"])
        self.assertTrue(gates["source_target_is_not_required_to_close_typed_intent_stage"])
        self.assertTrue(
            gates[
                "source_residual_must_be_reported_before_any_downstream_preauthorization"
            ]
        )
        self.assertEqual(
            gates["frozen_safe_replay_actions"],
            {"ANSWER": 259, "ABSTAIN": 90, "REJECT": 14},
        )
        self.assertEqual(gates["frozen_safe_replay_execution"], "363/363")
        self.assertEqual(gates["frozen_safe_replay_permutation"], "363/363")
        self.assertEqual(gates["frozen_safe_replay_counterfactual_false_accepts"], 0)

    def test_parent_runtime_and_checkpoint_hashes_are_exact(self) -> None:
        for parent in self.config["parents"].values():
            self.assertEqual(_digest(ROOT / parent["path"]), parent["sha256"])
        for relative, expected in self.config[
            "runtime_hashes_before_implementation"
        ].items():
            self.assertEqual(_digest(ROOT / relative), expected)
        checkpoint = self.config["checkpoint"]
        self.assertEqual(_digest(ROOT / checkpoint["path"]), checkpoint["sha256"])

    def test_precommit_and_later_phase_invariants_remain_closed(self) -> None:
        for name, value in self.config["invariants"].items():
            self.assertFalse(value, name)


class QP4UnitScaleQuestionPlannerPrecommitReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (REPORT.exists() and LOCK.exists()):
            raise unittest.SkipTest("Generated UNIT_SCALE precommit artifacts are absent")
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_exactly_reproducible(self) -> None:
        self.assertEqual(build_report(CONFIG), self.report)

    def test_report_passes_without_runtime_change(self) -> None:
        self.assertTrue(self.report["precommit_gate_passed"])
        self.assertEqual(self.report["failed_gates"], [])
        self.assertFalse(self.report["runtime_changed"])
        self.assertFalse(self.report["qp4_runtime_change_authorized_in_same_step"])
        self.assertFalse(self.report["qp5_authorized"])
        self.assertEqual(self.report["phase_d"], "locked")

    def test_lock_hashes_are_exact_and_only_authorize_next_typed_stage(self) -> None:
        self.assertEqual(self.lock["status"], "complete")
        self.assertTrue(self.lock["precommit_gate_passed"])
        self.assertFalse(self.lock["runtime_changed"])
        self.assertFalse(self.lock["qp4_runtime_change_authorized_in_same_step"])
        self.assertFalse(self.lock["retrieval_or_binding_edit_authorized"])
        self.assertEqual(
            self.lock["runtime_file_whitelist_for_next_step"],
            ["src/hrm_finance_qa/question_planner.py"],
        )
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
