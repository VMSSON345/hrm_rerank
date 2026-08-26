from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.validate_qp4_precommit import evaluate


ROOT = Path(__file__).resolve().parents[1]
PRECOMMIT = ROOT / "config" / "finhrm_qp4_precommit_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp4_precommit_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp4_precommit_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4PrecommitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.precommit = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_entry_condition_is_exact_and_qp4_runtime_remains_locked(self) -> None:
        claude = self.precommit["claude_md"]
        self.assertTrue(claude["read_before_scope_design"])
        self.assertEqual(_digest(ROOT / claude["path"]), claude["sha256"])
        for parent in self.precommit["parents"].values():
            self.assertEqual(_digest(ROOT / parent["path"]), parent["sha256"])
        parent_lock = json.loads(
            (
                ROOT
                / self.precommit["parents"][
                    "qp3_answer_presentation_implementation_lock"
                ]["path"]
            ).read_text(encoding="utf-8")
        )
        self.assertTrue(parent_lock["qp3_complete"])
        self.assertTrue(parent_lock["qp4_precommit_authorized"])
        self.assertFalse(parent_lock["qp4_runtime_change_authorized"])
        self.assertEqual(parent_lock["phase_d"], "locked")

    def test_minimal_pairs_cover_exact_semantic_dimensions(self) -> None:
        contract = self.precommit["minimal_pair_counterfactual_contract"]
        expected = {
            "COLUMN_ROLE",
            "OPERAND_YEAR",
            "ENTITY_SCOPE",
            "UNIT_SCALE",
            "DIRECTION",
            "REFERENCE_BASE",
        }
        self.assertEqual(set(contract["required_dimensions"]), expected)
        self.assertEqual(
            {item["dimension"] for item in contract["mutation_taxonomy"]},
            expected,
        )
        self.assertIn("exactly one", contract["construction_rule"].lower())
        self.assertFalse(contract["score_only_rejection_allowed"])
        self.assertFalse(contract["same_numeric_result_counts_as_correct_old_program"])

    def test_counterfactual_oracle_is_tri_state_and_old_program_bounded(self) -> None:
        contract = self.precommit["minimal_pair_counterfactual_contract"]
        self.assertIn("INVALID", contract["old_program_rule"])
        self.assertIn("UNKNOWN", contract["old_program_rule"])
        self.assertEqual(contract["oracle_outcomes"]["explicit_contradiction"], "REJECT")
        self.assertEqual(
            contract["oracle_outcomes"]["missing_required_semantics_or_proof"],
            "ABSTAIN",
        )
        self.assertIn(
            "old program remains non-answerable",
            contract["oracle_outcomes"]["different_complete_typed_program"],
        )
        self.assertFalse(
            contract["candidate_rank_neural_score_confidence_as_semantic_proof_allowed"]
        )

    def test_short_long_equivalence_freezes_ir_eligible_set_and_winner(self) -> None:
        surface = self.precommit["long_question_and_surface_contract"]
        expected = {
            "SHORT_LONG_EQUIVALENT",
            "DISCOURSE_PREFIX_SUFFIX",
            "CLAUSE_PERMUTATION",
            "PUNCTUATION_VARIANT",
            "NO_DIACRITICS",
            "BILINGUAL_PARENTHETICAL",
            "IRRELEVANT_DISTRACTOR_CLAUSE",
        }
        self.assertEqual(set(surface["meaning_preserving_categories"]), expected)
        self.assertTrue(
            all(value == 1.0 for value in surface["required_invariance"].values())
        )
        projection = self.precommit["semantic_projection_v1"]
        self.assertIn("eligible_program_set", projection["equivalence_rule"])
        self.assertIn("winner signature", projection["equivalence_rule"])
        self.assertTrue(
            {"candidate_rank", "neural_score", "confidence"}.issubset(
                projection["forbidden_projection_fields"]
            )
        )

    def test_clause_permutation_and_distractor_do_not_erase_qualifiers(self) -> None:
        rules = self.precommit["long_question_and_surface_contract"]["category_rules"]
        self.assertIn("operand attachment remains explicit", rules["CLAUSE_PERMUTATION"])
        self.assertIn("no competing requested qualifier", rules["IRRELEVANT_DISTRACTOR_CLAUSE"])
        self.assertIn("preserve metric", rules["DISCOURSE_PREFIX_SUFFIX"])

    def test_repeated_metric_conflict_and_candidate_permutation_fail_closed(self) -> None:
        stress = self.precommit["evidence_stress_contract"]
        self.assertTrue(
            {"REPEATED_METRIC", "CONFLICTING_CELL", "CANDIDATE_PERMUTATION"}.issubset(
                stress["categories"]
            )
        )
        self.assertEqual(stress["missing_proof_action"], "ABSTAIN")
        self.assertEqual(stress["contradiction_action"], "REJECT")
        self.assertFalse(stress["rank_score_confidence_tiebreak_as_semantic_proof_allowed"])

    def test_multi_output_qualitative_anaphora_and_cross_document_are_separate(self) -> None:
        tracks = self.precommit["separate_audit_tracks"]
        self.assertEqual(
            set(tracks),
            {"MULTI_OUTPUT", "QUALITATIVE_FOLLOW_UP", "ANAPHORA", "CROSS_DOCUMENT"},
        )
        self.assertIn("All requested outputs", tracks["MULTI_OUTPUT"]["answered_definition"])
        self.assertIn("direct cited text span", tracks["QUALITATIVE_FOLLOW_UP"]["contract"])
        self.assertIn("conversation-state proof", tracks["ANAPHORA"]["contract"])
        self.assertIn(
            "No current QP4 coverage claim",
            tracks["CROSS_DOCUMENT"]["answered_definition"],
        )

    def test_source_backed_cohort_requires_company_and_layout_diversity(self) -> None:
        cohorts = self.precommit["development_cohort_contract"]
        compiler = cohorts["compiler_stress"]
        source = cohorts["source_backed_end_to_end"]
        self.assertEqual(compiler["minimum_answer_family_count"], 13)
        self.assertEqual(compiler["minimum_ambiguity_family_count"], 3)
        self.assertFalse(compiler["company_or_layout_coverage_claim_allowed"])
        for field in (
            "minimum_distinct_companies_per_applicable_counterfactual_dimension",
            "minimum_distinct_source_documents_per_applicable_counterfactual_dimension",
            "minimum_distinct_real_layout_families_per_applicable_counterfactual_dimension",
            "minimum_distinct_companies_for_long_surface_stress",
            "minimum_distinct_source_documents_for_long_surface_stress",
            "minimum_distinct_real_layout_families_for_long_surface_stress",
        ):
            self.assertGreaterEqual(source[field], 3)
        self.assertTrue(source["company_disjoint_from_diagnosis"])
        self.assertTrue(source["independent_review_required"])
        self.assertFalse(source["synthetic_fixture_counts_toward_coverage"])
        self.assertFalse(source["safe_abstention_counts_as_answer"])

    def test_replay_gates_retain_frozen_core_and_require_zero_false_accepts(self) -> None:
        gates = self.precommit["implementation_replay_gates"]
        self.assertEqual(gates["semantic_counterfactual_old_program_false_accept_count"], 0)
        self.assertEqual(gates["short_long_semantic_ir_exact_rate"], 1.0)
        self.assertEqual(gates["short_long_eligible_program_set_exact_rate"], 1.0)
        self.assertEqual(gates["short_long_winner_exact_rate"], 1.0)
        self.assertEqual(gates["candidate_permutation_exact_rate"], 1.0)
        self.assertTrue(gates["hash_seed_invariant"])
        self.assertEqual(gates["frozen_safe_replay_gold_execution"], "363/363")
        self.assertEqual(gates["frozen_safe_replay_candidate_permutation"], "363/363")
        self.assertEqual(gates["frozen_safe_replay_counterfactual_false_accepts"], 0)
        self.assertEqual(
            gates["retain_core_actions"],
            {"ANSWER": 259, "ABSTAIN": 90, "REJECT": 14},
        )

    def test_runtime_scope_is_conditional_surgical_and_currently_unchanged(self) -> None:
        whitelist = self.precommit["exact_runtime_edit_scope_for_qp4_implementation"]
        self.assertEqual(
            set(whitelist),
            {
                "src/hrm_finance_qa/question_planner.py",
                "src/hrm_finance_qa/program_synthesis.py",
                "src/hrm_finance_qa/symbolic_verifier.py",
                "src/hrm_finance_qa/direction_contract.py",
            },
        )
        self.assertTrue(
            set(self.precommit["runtime_files_explicitly_frozen"]).isdisjoint(whitelist)
        )
        admission = self.precommit["runtime_change_admission"]
        self.assertTrue(admission["baseline_diagnostic_required_before_any_runtime_edit"])
        self.assertIn("smallest", admission["conditional_scope_rule"])
        for relative, expected in self.precommit["runtime_hashes_at_precommit"].items():
            self.assertEqual(_digest(ROOT / relative), expected)

    def test_no_protected_data_runtime_release_or_later_phase_opened(self) -> None:
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
            "No QP4 runtime change is authorized or performed by this precommit",
            self.precommit["authorized_next_step"],
        )
        self.assertIn("QP5 and Phase D remain locked", self.precommit["authorized_next_step"])

    def test_validator_report_and_lock_are_exact(self) -> None:
        evaluated = evaluate(PRECOMMIT)
        self.assertTrue(evaluated["precommit_gate_passed"], evaluated["failed_gates"])
        self.assertEqual(evaluated, self.report)
        self.assertEqual(self.report["status"], "PASS")
        self.assertTrue(self.lock["precommit_complete"])
        self.assertTrue(self.lock["qp4_scoped_implementation_contract_ready"])
        self.assertFalse(self.lock["runtime_changed"])
        self.assertFalse(self.lock["qp4_runtime_change_authorized"])
        self.assertFalse(self.lock["qp5_authorized"])
        self.assertEqual(self.lock["phase_d"], "locked")
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
