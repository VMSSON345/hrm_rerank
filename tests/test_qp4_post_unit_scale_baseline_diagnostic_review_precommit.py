from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.validate_qp4_post_unit_scale_baseline_diagnostic_review_precommit import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/finhrm_qp4_post_unit_scale_baseline_diagnostic_review_precommit_v1.json"
REPORT = ROOT / "reports/finhrm_qp4_post_unit_scale_baseline_diagnostic_review_precommit_v1.json"
LOCK = ROOT / "reports/finhrm_qp4_post_unit_scale_baseline_diagnostic_review_precommit_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4PostUnitScaleDiagnosticReviewPrecommitTests(unittest.TestCase):
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

    def test_residual_inventory_is_generic_and_exact(self) -> None:
        inventory = self.config["residual_inventory"]
        self.assertEqual(inventory["compiler_surface"]["residual_records"], 18)
        self.assertEqual(
            {name: row["family_groups"] for name, row in inventory["compiler_surface"]["groups"].items()},
            {"DISCOURSE_PREFIX_SUFFIX": 12, "BILINGUAL_PARENTHETICAL": 3, "IRRELEVANT_DISTRACTOR_CLAUSE": 3},
        )
        add = inventory["add_source_surface"]
        self.assertEqual((add["residual_queries"], add["companies"], add["sources"], add["real_layouts"]), (15, 5, 5, 3))
        actual = inventory["actual_plan_source"]
        self.assertEqual((actual["diagnostic_queries"], actual["companies"], actual["sources"], actual["document_layouts"], actual["table_layouts"]), (24, 4, 4, 3, 4))

    def test_diagnostic_scope_has_no_runtime_whitelist(self) -> None:
        contract = self.config["diagnostic_contract"]
        self.assertEqual(contract["stage"], "diagnostic_measurement_only_no_runtime_change")
        self.assertEqual(contract["runtime_file_whitelist"], [])
        self.assertEqual(contract["compiler_residual_records"], 18)
        self.assertEqual(contract["source_residual_queries"], 39)
        self.assertEqual(contract["source_query_breakdown"], {"add_surface": 15, "actual_plan": 24})

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
        self.assertIn("selected_physical_evidence_signature", contract["trace_fields"])
        self.assertIn("proof_obligations", contract["trace_fields"])
        self.assertIn("stage_timings", contract["trace_fields"])

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

    def test_gold_is_post_hoc_and_fail_closed(self) -> None:
        contract = self.config["diagnostic_contract"]
        self.assertFalse(contract["gold_candidate_or_program_passed_to_engine"])
        self.assertTrue(contract["gold_used_only_after_observation_for_diagnostic_comparison"])
        self.assertTrue(contract["timeout_or_document_source_error_is_failure"])
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
        self.assertTrue(contract["attribution_policy"]["source_or_retrieval_layer_is_outside_current_qp4_runtime_whitelist"])

    def test_precommit_does_not_execute_diagnostic_or_open_later_phases(self) -> None:
        self.assertFalse(self.config["runtime_changed_by_this_precommit"])
        self.assertFalse(self.config["diagnostic_executed_by_this_precommit"])
        self.assertFalse(self.config["downstream_runtime_change_authorized_in_same_step"])
        for name, value in self.config["invariants"].items():
            self.assertFalse(value, name)


class QP4PostUnitScaleDiagnosticReviewPrecommitReportTests(unittest.TestCase):
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

    def test_report_derives_both_source_strata(self) -> None:
        evidence = self.report["derived_residual_evidence"]
        self.assertEqual(evidence["source_error_queries"], 11)
        self.assertEqual(evidence["non_error_source_queries"], 28)
        self.assertEqual(evidence["source_error_queries"] + evidence["non_error_source_queries"], 39)

    def test_lock_hashes_artifacts_and_opens_only_diagnostics(self) -> None:
        self.assertEqual(self.lock["status"], "complete")
        self.assertTrue(self.lock["precommit_gate_passed"])
        self.assertFalse(self.lock["diagnostic_executed"])
        self.assertFalse(self.lock["runtime_edit_authorized"])
        self.assertEqual(self.lock["runtime_file_whitelist_for_next_step"], [])
        self.assertTrue(self.lock["authorized_next_step"].startswith("QP4-POST-UNIT-SCALE-BASELINE-DIAGNOSTIC-REVIEW"))
        self.assertFalse(self.lock["qp4_complete"])
        self.assertFalse(self.lock["qp5_authorized"])
        self.assertEqual(self.lock["phase_d"], "locked")
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
