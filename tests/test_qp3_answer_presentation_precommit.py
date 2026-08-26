from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.validate_qp3_answer_presentation_precommit import evaluate


ROOT = Path(__file__).resolve().parents[1]
PRECOMMIT = ROOT / "config" / "finhrm_qp3_answer_presentation_precommit_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp3_answer_presentation_precommit_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp3_answer_presentation_precommit_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP3AnswerPresentationPrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.precommit = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_parent_core_lock_and_runtime_are_exact(self) -> None:
        claude = self.precommit["claude_md"]
        self.assertTrue(claude["read_before_scope_design"])
        self.assertEqual(_digest(ROOT / claude["path"]), claude["sha256"])
        for parent in self.precommit["parents"].values():
            self.assertEqual(_digest(ROOT / parent["path"]), parent["sha256"])
        core_lock = json.loads(
            (
                ROOT
                / self.precommit["parents"]["qp3_core_implementation_lock"]["path"]
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(core_lock["implementation_gate_passed"])
        self.assertEqual(core_lock["failed_gates"], [])
        self.assertFalse(core_lock["qp4_authorized"])
        self.assertEqual(core_lock["phase_d"], "locked")
        for relative, expected in self.precommit[
            "runtime_hashes_before_implementation"
        ].items():
            self.assertEqual(_digest(ROOT / relative), expected)

    def test_envelope_freezes_status_modes_and_required_fields(self) -> None:
        envelope = self.precommit["verified_answer_envelope_v1"]
        self.assertEqual(envelope["schema_version"], "finhrm-verified-answer-envelope-v1")
        self.assertEqual(envelope["status_values"], ["ANSWER", "ABSTAIN", "REJECT"])
        self.assertEqual(
            envelope["answer_mode_values"],
            ["FACT", "COMPUTATION", "VERIFIED_SUMMARY"],
        )
        self.assertEqual(
            set(envelope["top_level_required_fields"]),
            {
                "schema_version",
                "status",
                "answer_mode",
                "locale",
                "claim",
                "canonical_value",
                "display",
                "program",
                "citations",
                "proof",
                "summary_coverage",
                "warnings",
            },
        )

    def test_status_is_fail_closed_and_never_score_driven(self) -> None:
        proof = self.precommit["verified_answer_envelope_v1"]["proof"]
        self.assertIn("existing runtime action", proof["status_authority"])
        self.assertIn("never become ANSWER", proof["promotion_forbidden"])
        self.assertIn("required UNKNOWN", proof["abstain_rule"])
        self.assertIn("INVALID", proof["reject_rule"])
        self.assertFalse(
            proof["candidate_rank_neural_score_confidence_as_semantic_proof_allowed"]
        )
        self.assertFalse(proof["warning_text_parsing_as_status_authority_allowed"])

    def test_fact_and_citations_never_invent_provenance(self) -> None:
        fact = self.precommit["mode_contracts"]["FACT"]
        citations = self.precommit["verified_answer_envelope_v1"]["citations"]
        self.assertEqual(fact["eligible_program_roots"], ["SELECT"])
        self.assertEqual(citations["operand_citation_coverage_required"], 1.0)
        self.assertTrue(citations["source_span_must_resolve_to_selected_evidence"])
        self.assertIn("Omit", citations["nullable_field_policy"])
        self.assertIn("never emit page X", citations["page_render_policy"])

    def test_computation_formula_and_every_node_replay_exactly(self) -> None:
        contract = self.precommit["verified_answer_envelope_v1"]["program"]
        gates = self.precommit["implementation_replay_gates"]
        self.assertEqual(
            contract["formula_source"],
            "deterministically_rendered_from_executed_program_dag",
        )
        self.assertTrue(contract["each_leaf_must_reference_exactly_one_operand_and_citation"])
        self.assertTrue(contract["each_intermediate_must_replay_exactly"])
        self.assertTrue(contract["final_replay_must_equal_canonical_value"])
        self.assertFalse(contract["free_text_formula_as_authority_allowed"])
        self.assertEqual(gates["derivation_replay_exact_rate"], 1.0)
        self.assertEqual(gates["operand_citation_coverage"], 1.0)

    def test_summary_partial_abstains_and_conflict_rejects(self) -> None:
        summary = self.precommit["verified_answer_envelope_v1"]["summary_coverage"]
        self.assertIn("verified_count equals requested_count", summary["complete_answer_rule"])
        self.assertIn("overall status is ABSTAIN", summary["partial_rule"])
        self.assertIn("N/M", summary["partial_rule"])
        self.assertIn("REJECT", summary["conflict_rule"])
        self.assertTrue(summary["each_row_is_independent_verified_subanswer"])
        self.assertTrue(summary["qualitative_claim_requires_direct_cited_text_span"])
        self.assertFalse(summary["llm_generated_number_formula_citation_or_uncited_cause_allowed"])

    def test_renderer_is_deterministic_and_claim_bounded(self) -> None:
        renderer = self.precommit["deterministic_renderer_contract"]
        gates = self.precommit["implementation_replay_gates"]
        self.assertTrue(renderer["templates_only"])
        self.assertTrue(renderer["same_envelope_same_bytes"])
        self.assertTrue(renderer["technical_audit_output_must_remain_separate"])
        self.assertFalse(renderer["renderer_may_create_number_page_citation_formula_or_claim"])
        self.assertFalse(renderer["renderer_may_consume_confidence_or_rank"])
        self.assertEqual(gates["unsupported_claim_count"], 0)
        self.assertEqual(gates["summary_complete_false_claim_count"], 0)

    def test_development_cohort_requires_company_and_layout_diversity(self) -> None:
        cohort = self.precommit["development_generalization_cohort"]
        for field in (
            "minimum_verified_examples_per_mode",
            "minimum_distinct_companies_per_mode",
            "minimum_distinct_real_layout_families_per_mode",
            "minimum_distinct_source_documents_per_mode",
        ):
            self.assertGreaterEqual(cohort[field], 3)
        self.assertTrue(cohort["company_and_layout_disjoint_confirmation_required"])
        self.assertTrue(cohort["real_source_backed_required"])
        self.assertFalse(cohort["synthetic_fixture_counts_toward_coverage"])
        self.assertFalse(cohort["safe_abstention_counts_as_answer"])

    def test_hard_negatives_permutations_and_core_replay_are_frozen(self) -> None:
        gates = self.precommit["implementation_replay_gates"]
        hard_negatives = {
            item["id"] for item in self.precommit["hard_negative_taxonomy"]
        }
        self.assertEqual(
            hard_negatives,
            {
                "PRES_WRONG_PAGE",
                "PRES_WRONG_STATEMENT",
                "PRES_WRONG_YEAR",
                "PRES_SWAPPED_OPERAND",
                "PRES_SCALE_X1000",
                "PRES_MISSING_SUMMARY_ROW",
                "PRES_UNCITED_QUALITATIVE_CAUSE",
                "PRES_PARENT_ACTION_PROMOTION",
            },
        )
        self.assertEqual(gates["hard_negative_false_accept_count"], 0)
        self.assertEqual(gates["candidate_permutation_exact_rate"], 1.0)
        self.assertEqual(gates["citation_permutation_exact_rate"], 1.0)
        self.assertEqual(gates["summary_input_permutation_exact_rate"], 1.0)
        self.assertEqual(gates["frozen_safe_replay_gold_execution"], "363/363")
        self.assertEqual(gates["frozen_safe_replay_candidate_permutation"], "363/363")
        self.assertEqual(gates["frozen_safe_replay_counterfactual_false_accepts"], 0)
        self.assertEqual(
            gates["retain_core_actions"],
            {"ANSWER": 259, "ABSTAIN": 90, "REJECT": 14},
        )

    def test_runtime_whitelist_is_surgical_and_core_semantics_are_frozen(self) -> None:
        scope = self.precommit["exact_runtime_edit_scope_for_implementation"]
        self.assertEqual(
            set(scope),
            {
                "src/hrm_finance_qa/answer_presentation.py",
                "src/hrm_finance_qa/document_reasoning_engine.py",
            },
        )
        self.assertFalse((ROOT / "src/hrm_finance_qa/answer_presentation.py").exists())
        frozen = set(self.precommit["runtime_files_explicitly_frozen"])
        self.assertTrue(frozen.isdisjoint(scope))
        self.assertIn("src/hrm_finance_qa/symbolic_verifier.py", frozen)
        self.assertIn("src/hrm_finance_qa/question_planner.py", frozen)
        self.assertIn("src/hrm_finance_qa/program_synthesis.py", frozen)
        self.assertIn("src/hrm_finance_qa/document_hrm.py", frozen)

    def test_no_runtime_release_or_protected_data_change(self) -> None:
        allowed = self.precommit["allowed_evidence"]
        self.assertFalse(allowed["abw_allowed"])
        self.assertFalse(
            allowed[
                "protected_validation_blind_test_sealed_holdout_final_access_allowed"
            ]
        )
        self.assertFalse(allowed["synthetic_layout_allowed_as_generalization_confirmation"])
        for name, value in self.precommit["frozen_invariants"].items():
            self.assertFalse(value, name)
        self.assertIn(
            "QP3-ANSWER-PRESENTATION-IMPLEMENTATION only",
            self.precommit["authorized_next_step"],
        )
        self.assertIn("QP4 and Phase D remain locked", self.precommit["authorized_next_step"])

    def test_validator_report_and_lock_are_exact(self) -> None:
        evaluated = evaluate(PRECOMMIT)
        self.assertTrue(evaluated["precommit_gate_passed"], evaluated["failed_gates"])
        self.assertEqual(evaluated, self.report)
        self.assertEqual(self.report["status"], "PASS")
        self.assertTrue(self.lock["precommit_complete"])
        self.assertFalse(self.lock["runtime_changed"])
        self.assertTrue(self.lock["next_step_scoped_runtime_change_authorized"])
        self.assertFalse(self.lock["qp4_authorized"])
        self.assertEqual(self.lock["phase_d"], "locked")
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
