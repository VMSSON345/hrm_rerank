from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.validate_qp4_post_question_surface_baseline_diagnostic_review_precommit import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "finhrm_qp4_post_question_surface_baseline_diagnostic_review_precommit_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_diagnostic_review_precommit_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_diagnostic_review_precommit_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4PostQuestionSurfaceDiagnosticReviewPrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_parent_chain_runtime_and_checkpoint_are_exact(self) -> None:
        for parent in self.config["parents"].values():
            self.assertEqual(_digest(ROOT / parent["path"]), parent["sha256"])
        for relative, expected in self.config["frozen_runtime_hashes"].items():
            self.assertEqual(_digest(ROOT / relative), expected)
        checkpoint = self.config["checkpoint"]
        self.assertEqual(_digest(ROOT / checkpoint["path"]), checkpoint["sha256"])

    def test_compiler_is_only_a_full_no_regression_control(self) -> None:
        compiler = self.config["residual_inventory"]["compiler_controls"]
        self.assertEqual(
            compiler,
            {
                "canonical": "13/13",
                "ambiguity": "3/3",
                "frozen_surface": "91/91",
                "corrected_hold_ahead": "78/78",
                "minimal_pairs": "57/57",
                "residual_records": 0,
            },
        )
        contract = self.config["diagnostic_contract"]
        self.assertEqual(contract["compiler_residual_records"], 0)
        self.assertTrue(contract["compiler_inventory_is_no_regression_control"])

    def test_all_residual_and_source_error_observations_are_in_scope(self) -> None:
        inventory = self.config["residual_inventory"]
        self.assertEqual(inventory["source_observations_for_diagnosis"], 40)
        self.assertEqual(inventory["source_error_observations"], 12)
        self.assertEqual(inventory["non_error_observations"], 28)
        self.assertEqual(
            self.config["diagnostic_contract"]["source_observation_breakdown"],
            {
                "add_surface_residuals": 15,
                "add_counterfactual_source_errors": 1,
                "actual_plan_base_and_mutated": 24,
            },
        )

    def test_add_surface_residual_groups_are_diverse(self) -> None:
        add = self.config["residual_inventory"]["add_source_surface"]
        self.assertEqual((add["residual_queries"], add["companies"], add["sources"], add["real_layouts"]), (15, 5, 5, 3))
        self.assertEqual(set(add["groups"]), {"SHORT_LONG_EQUIVALENT", "DISCOURSE_PREFIX_SUFFIX", "CLAUSE_PERMUTATION"})
        for row in add["groups"].values():
            self.assertEqual((row["failures"], row["companies"], row["sources"], row["real_layouts"]), (5, 5, 5, 3))
            self.assertEqual(row["source_errors"], 1)
            self.assertTrue(row["compiler_projection_passed"])

    def test_single_counterfactual_error_cannot_nominate_runtime(self) -> None:
        row = self.config["residual_inventory"]["add_counterfactual_source_error"]
        self.assertEqual(row["failed_observations"], 1)
        self.assertEqual(row["dimension"], "OPERAND_YEAR")
        self.assertEqual((row["companies"], row["sources"], row["real_layouts"]), (1, 1, 1))
        self.assertEqual(row["false_accepts"], 0)
        self.assertFalse(row["generic_evidence_minimum_met"])
        self.assertTrue(self.config["diagnostic_contract"]["attribution_policy"]["singleton_or_non_diverse_stratum_cannot_nominate_runtime"])

    def test_actual_plan_residual_scope_is_complete(self) -> None:
        actual = self.config["residual_inventory"]["actual_plan_source"]
        self.assertEqual((actual["pair_rows"], actual["diagnostic_queries"]), (12, 24))
        self.assertEqual((actual["companies"], actual["sources"], actual["document_layouts"], actual["table_layouts"]), (4, 4, 3, 4))
        self.assertEqual({name: row["paired_gold_answers"] for name, row in actual["dimensions"].items()}, {"COLUMN_ROLE": 0, "DIRECTION": 0, "REFERENCE_BASE": 0})
        self.assertEqual(actual["old_winner_false_accepts"], 0)

    def test_diagnostic_scope_has_no_runtime_whitelist(self) -> None:
        contract = self.config["diagnostic_contract"]
        self.assertEqual(contract["stage"], "diagnostic_measurement_only_no_runtime_change")
        self.assertEqual(contract["runtime_file_whitelist"], [])
        self.assertTrue(contract["evaluate_all_records_in_each_residual_or_error_stratum"])
        self.assertEqual(contract["source_residual_observations"], 40)

    def test_first_failing_layer_order_and_trace_are_frozen(self) -> None:
        contract = self.config["diagnostic_contract"]
        self.assertEqual(
            contract["layer_order"],
            [
                "QUESTION_PLAN",
                "SOURCE_LOAD_OR_EXTRACTION",
                "CANDIDATE_POPULATION_OR_RETRIEVAL",
                "SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS",
                "SYMBOLIC_PROOF_OR_ACTION",
            ],
        )
        self.assertEqual(set(contract["first_failing_layer_rules"]), set(contract["layer_order"]))
        self.assertIn("candidate_support_signature", contract["trace_fields"])
        self.assertIn("selected_physical_evidence_signature", contract["trace_fields"])
        self.assertIn("proof_obligations", contract["trace_fields"])

    def test_selection_cannot_target_reports_examples_or_scores(self) -> None:
        forbidden = set(self.config["diagnostic_contract"]["selection_must_not_use"])
        self.assertTrue(
            {
                "company identity",
                "report filename",
                "page number",
                "year",
                "numeric value",
                "example or anchor id",
                "question literal",
                "candidate rank",
                "neural score",
                "confidence",
            }
            <= forbidden
        )

    def test_gold_is_post_hoc_and_source_errors_fail_closed(self) -> None:
        contract = self.config["diagnostic_contract"]
        self.assertFalse(contract["gold_candidate_or_program_passed_to_engine"])
        self.assertTrue(contract["gold_used_only_after_observation_for_diagnostic_comparison"])
        self.assertTrue(contract["timeout_or_document_source_error_is_failure"])
        self.assertTrue(contract["source_error_must_be_attributed_before_downstream_layers"])
        self.assertFalse(contract["safe_abstention_counts_as_answer"])
        self.assertFalse(contract["candidate_rank_score_or_confidence_is_semantic_proof"])
        self.assertFalse(contract["mineru_auto_generate"])

    def test_generic_evidence_does_not_auto_authorize_runtime(self) -> None:
        contract = self.config["diagnostic_contract"]
        self.assertEqual(
            contract["minimum_generic_evidence"],
            {"compiler_family_groups": 3, "source_company_groups": 3, "source_documents": 3, "real_layout_families": 3},
        )
        self.assertTrue(contract["attribution_policy"]["no_runtime_file_is_authorized_by_group_size_alone"])
        self.assertTrue(contract["attribution_policy"]["new_implementation_precommit_required_after_diagnosis"])
        self.assertTrue(contract["attribution_policy"]["source_extraction_or_retrieval_expansion_requires_separate_roadmap_authorization"])

    def test_precommit_does_not_execute_diagnostic_or_open_later_phases(self) -> None:
        self.assertFalse(self.config["runtime_changed_by_this_precommit"])
        self.assertFalse(self.config["diagnostic_executed_by_this_precommit"])
        self.assertFalse(self.config["downstream_runtime_change_authorized_in_same_step"])
        for name, value in self.config["invariants"].items():
            self.assertFalse(value, name)


class QP4PostQuestionSurfaceDiagnosticReviewPrecommitReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (REPORT.exists() and LOCK.exists()):
            raise unittest.SkipTest("Generated diagnostic-review precommit artifacts are absent")
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_exactly_reproducible(self) -> None:
        self.assertEqual(build_report(CONFIG), self.report)

    def test_report_closes_only_the_precommit(self) -> None:
        self.assertTrue(self.report["precommit_gate_passed"])
        self.assertEqual(self.report["failed_gates"], [])
        self.assertFalse(self.report["diagnostic_executed"])
        self.assertFalse(self.report["runtime_changed"])
        self.assertFalse(self.report["runtime_edit_authorized"])
        self.assertFalse(self.report["qp4_complete"])
        self.assertFalse(self.report["qp5_authorized"])
        self.assertEqual(self.report["phase_d"], "locked")

    def test_report_retains_all_source_failure_strata(self) -> None:
        evidence = self.report["derived_residual_evidence"]
        self.assertEqual(evidence["source_observations"], 40)
        self.assertEqual(evidence["source_error_observations"], 12)
        self.assertEqual(evidence["non_error_observations"], 28)

    def test_lock_hashes_artifacts_and_opens_only_diagnostics(self) -> None:
        self.assertEqual(self.lock["status"], "complete")
        self.assertTrue(self.lock["precommit_gate_passed"])
        self.assertFalse(self.lock["diagnostic_executed"])
        self.assertFalse(self.lock["runtime_edit_authorized"])
        self.assertEqual(self.lock["runtime_file_whitelist_for_next_step"], [])
        self.assertTrue(self.lock["authorized_next_step"].startswith("QP4-POST-QUESTION-SURFACE-BASELINE-DIAGNOSTIC-REVIEW"))
        self.assertFalse(self.lock["qp4_complete"])
        self.assertFalse(self.lock["qp5_authorized"])
        self.assertEqual(self.lock["phase_d"], "locked")
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
