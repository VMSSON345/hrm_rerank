import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "config" / "finhrm_report_generalization_invariant_v1.json"


class FinHRMReportGeneralizationInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    def test_parent_thesis_contract_is_unchanged(self) -> None:
        parent = self.contract["parent_contract"]
        payload = (ROOT / parent["path"]).read_bytes()
        self.assertEqual(hashlib.sha256(payload).hexdigest(), parent["sha256"])

    def test_report_specific_runtime_signals_are_forbidden(self) -> None:
        forbidden = set(self.contract["forbidden_runtime_signals"])
        self.assertTrue(
            {
                "company_identity",
                "document_filename_or_hash",
                "fixed_page_number",
                "fixed_year",
                "fixed_numeric_value_or_answer",
                "single_question_literal",
                "example_or_family_id_used_as_runtime_rule",
            }.issubset(forbidden)
        )

    def test_repair_requires_disjoint_multi_report_confirmation(self) -> None:
        rule = self.contract["repair_admission_rule"]
        self.assertTrue(rule["precommit_required"])
        self.assertTrue(rule["diagnosis_documents_excluded_from_confirmation"])
        self.assertTrue(rule["company_disjoint_from_diagnosis"])
        self.assertGreaterEqual(rule["minimum_distinct_companies"], 3)
        self.assertGreaterEqual(rule["minimum_distinct_layout_families"], 3)
        self.assertTrue(rule["positive_and_hard_negative_required"])
        self.assertTrue(rule["counterfactual_required"])
        self.assertTrue(rule["permutation_invariance_required"])
        self.assertTrue(rule["no_regression_on_all_frozen_development_cohorts"])

    def test_unseen_or_unsupported_reports_fail_closed(self) -> None:
        reporting = self.contract["release_reporting"]
        self.assertFalse(reporting["universal_coverage_claim_allowed"])
        self.assertTrue(reporting["report_unseen_company_results"])
        self.assertTrue(reporting["report_unseen_layout_results"])
        self.assertTrue(reporting["report_worst_group"])
        self.assertTrue(reporting["safe_abstention_is_not_counted_as_answered"])
        self.assertEqual(
            reporting["unsupported_layout_policy"],
            "ABSTAIN_WITH_EXPLICIT_PROOF_OR_PROVENANCE_REASON",
        )

    def test_protected_sources_threshold_and_phase_d_remain_locked(self) -> None:
        invariants = self.contract["invariants"]
        self.assertFalse(invariants["protected_evaluation_sources_allowed"])
        self.assertFalse(invariants["fine_tuning_allowed"])
        self.assertFalse(invariants["threshold_change_allowed"])
        self.assertFalse(invariants["phase_d_unlocked"])


if __name__ == "__main__":
    unittest.main()
