from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.evaluate_qp4_development_baseline import build_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "finhrm_qp4_development_cohort_v1.json"
RAW = ROOT / "reports" / "finhrm_qp4_source_backed_baseline_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp4_development_baseline_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp4_development_baseline_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4DevelopmentBaselineReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.raw = json.loads(RAW.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_exactly_reproducible_without_model_rerun(self) -> None:
        self.assertEqual(build_report(CONFIG, RAW), self.report)

    def test_source_boundary_is_real_pdf_and_company_layout_diverse(self) -> None:
        source = self.report["source_backed"]
        self.assertGreaterEqual(source["eligible_anchors"], 3)
        self.assertGreaterEqual(source["companies"], 3)
        self.assertGreaterEqual(source["source_documents"], 3)
        self.assertGreaterEqual(source["real_layout_families"], 3)
        for dimension in self.config["source_backed_cohort"][
            "applicable_counterfactual_dimensions"
        ]:
            values = source["counterfactual"]["by_dimension"][dimension]
            self.assertTrue(values["coverage_gate"], dimension)

    def test_all_source_runs_use_all_eligible_anchors(self) -> None:
        anchors = self.raw["eligible_anchors"]
        categories = len(
            self.config["source_backed_cohort"]["meaning_preserving_categories"]
        )
        dimensions = len(
            self.config["source_backed_cohort"][
                "applicable_counterfactual_dimensions"
            ]
        )
        self.assertEqual(len(self.raw["surface_results"]), anchors * categories)
        self.assertEqual(
            len(self.raw["counterfactual_results"]), anchors * dimensions
        )

    def test_unavailable_add_dimensions_are_not_claimed(self) -> None:
        self.assertEqual(
            set(self.report["source_dimensions_not_confirmed"]),
            {"COLUMN_ROLE", "DIRECTION", "REFERENCE_BASE"},
        )
        for dimension in self.report["source_dimensions_not_confirmed"]:
            self.assertEqual(
                self.report["source_backed"]["counterfactual"]["by_dimension"][
                    dimension
                ]["status"],
                "NOT_CONFIRMED",
            )

    def test_runtime_checkpoint_and_release_locks_are_unchanged(self) -> None:
        self.assertFalse(self.report["runtime_changed"])
        self.assertFalse(self.report["qp4_runtime_change_authorized"])
        self.assertFalse(self.report["qp5_authorized"])
        self.assertEqual(self.report["phase_d"], "locked")
        for relative, expected in self.report["runtime_hashes"].items():
            self.assertEqual(_digest(ROOT / relative), expected)
        checkpoint = self.config["checkpoint"]
        self.assertEqual(self.report["checkpoint_sha256"], checkpoint["sha256"])

    def test_frozen_safe_replay_contract_is_retained(self) -> None:
        replay = self.report["frozen_safe_replay"]
        self.assertEqual(replay["actions"], {"ANSWER": 259, "ABSTAIN": 90, "REJECT": 14})
        self.assertEqual(replay["execution"], "363/363")
        self.assertEqual(replay["permutation"], "363/363")
        self.assertEqual(replay["counterfactual_false_accepts"], 0)

    def test_generic_failures_are_reported_without_runtime_authorization(self) -> None:
        by_group = {row["failure_group"]: row for row in self.report["diagnosis"]}
        unit = by_group["source_counterfactual:UNIT_SCALE"]
        self.assertTrue(unit["generic_evidence"])
        self.assertEqual(unit["layer"], "question_planner")
        self.assertGreaterEqual(unit["company_groups"], 3)
        self.assertGreaterEqual(unit["real_layout_groups"], 3)
        self.assertFalse(self.report["qp4_runtime_change_authorized"])

    def test_lock_is_complete_but_does_not_authorize_runtime(self) -> None:
        self.assertTrue(self.lock["cohort_and_baseline_complete"])
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
