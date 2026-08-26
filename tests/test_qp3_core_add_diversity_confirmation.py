from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_qp3_core_add_diversity_confirmation.py"
REPORT = ROOT / "reports/finhrm_qp3_core_add_diversity_confirmation_v1.json"
LOCK = ROOT / "reports/finhrm_qp3_core_add_diversity_confirmation_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("qp3_add_diversity", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load diversity evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QP3CoreAddDiversityConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_exactly_reproducible(self) -> None:
        self.assertEqual(_load_evaluator().build_report(), self.report)

    def test_candidate_evidence_is_supporting_only(self) -> None:
        self.assertFalse(self.report["diversity_gate_passed"])
        self.assertEqual(
            self.report["current_evidence"]["role"],
            "SUPPORTING_ONLY_NOT_GATE_CLOSING",
        )
        self.assertEqual(self.report["current_evidence"]["positive_examples"], 10)

    def test_layout_coverage_exposes_the_actual_company_counts(self) -> None:
        layout = self.report["current_evidence"]["layout_coverage"]
        self.assertEqual(layout["DEVELOPMENT_OCR_OR_HYBRID_TABLE"]["distinct_companies"], 2)
        self.assertEqual(layout["DEVELOPMENT_NATIVE_MULTI_TABLE"]["distinct_companies"], 3)
        self.assertFalse(
            self.report["gate_checks"][
                "minimum_three_companies_in_every_additional_layout"
            ]
        )

    def test_pending_review_and_missing_end_to_end_stages_fail_closed(self) -> None:
        evidence = self.report["current_evidence"]
        self.assertEqual(evidence["independently_reviewed_positives"], 0)
        self.assertEqual(evidence["review_statuses"], {"pending_independent_review": 10})
        self.assertTrue(evidence["starts_from_pre_materialized_candidates"])
        self.assertFalse(
            self.report["gate_checks"]["all_required_end_to_end_stages_exercised"]
        )

    def test_safety_controls_still_pass(self) -> None:
        checks = self.report["gate_checks"]
        self.assertTrue(checks["candidate_level_confirmation_safety_clean"])
        self.assertTrue(checks["candidate_permutation_exact"])
        self.assertTrue(checks["counterfactual_false_accepts_zero"])
        for name, value in self.report["invariants"].items():
            self.assertFalse(value, name)

    def test_lock_matches_and_keeps_later_phases_closed(self) -> None:
        self.assertFalse(self.lock["diversity_gate_passed"])
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
