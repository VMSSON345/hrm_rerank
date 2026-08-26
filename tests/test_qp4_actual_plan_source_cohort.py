from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from hrm_finance_qa.question_planner import compile_question_plan
from scripts.evaluate_qp4_actual_plan_source_cohort import (
    _dimension_projection,
    _verify_config,
    build_report,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "finhrm_qp4_actual_plan_ratio_direction_source_cohort_v1.json"
RAW = ROOT / "reports" / "finhrm_qp4_actual_plan_source_baseline_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp4_actual_plan_source_cohort_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp4_actual_plan_source_cohort_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4ActualPlanSourceCohortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_config_is_hash_exact_and_development_safe(self) -> None:
        coverage = _verify_config(self.config)
        self.assertEqual(coverage["anchors"], 4)
        self.assertEqual(coverage["companies"], 4)
        self.assertEqual(coverage["sources"], 4)
        self.assertEqual(coverage["document_layouts"], 3)
        self.assertEqual(coverage["table_layouts"], 4)

    def test_cohort_is_company_disjoint_from_prior_diagnosis(self) -> None:
        diagnosis = set(self.config["selection_contract"]["diagnosis_company_keys"])
        cohort = {row["company_key"] for row in self.config["anchors"]}
        self.assertFalse(diagnosis & cohort)

    def test_every_anchor_has_explicit_three_cell_proof_and_visual_review(self) -> None:
        required = {"actual_2023", "plan_2023", "actual_2022"}
        for anchor in self.config["anchors"]:
            self.assertEqual(set(anchor["evidence"]), required)
            self.assertEqual(anchor["visual_review"]["decision"], "pass")
            self.assertIn("role_headers_explicit", anchor["visual_review"]["checks"])
            self.assertIn("values_exact", anchor["visual_review"]["checks"])

    def test_all_three_dimensions_are_compiler_detectable_on_every_source(self) -> None:
        for anchor in self.config["anchors"]:
            for dimension, pair in anchor["pairs"].items():
                base = compile_question_plan(pair["base_question"])
                mutated = compile_question_plan(pair["mutated_question"])
                self.assertNotEqual(
                    _dimension_projection(base, dimension),
                    _dimension_projection(mutated, dimension),
                    f"{anchor['anchor_id']}:{dimension}",
                )

    def test_questions_are_evaluation_fixtures_not_runtime_rules(self) -> None:
        script = (ROOT / "scripts" / "evaluate_qp4_actual_plan_source_cohort.py").read_text(
            encoding="utf-8"
        )
        for anchor in self.config["anchors"]:
            self.assertNotIn(anchor["source_name"], script)
            self.assertNotIn(anchor["anchor_id"], script)
            for cell in anchor["evidence"].values():
                self.assertNotIn(cell["value"], script)

    def test_release_locks_remain_closed(self) -> None:
        for name, value in self.config["invariants"].items():
            self.assertFalse(value, name)
        self.assertFalse(self.config["selection_contract"]["safe_abstention_counts_as_answer"])
        self.assertFalse(self.config["selection_contract"]["synthetic_layout_counts_as_confirmation"])
        self.assertFalse(self.config["selection_contract"]["mineru_auto_generate_during_baseline"])


class QP4ActualPlanSourceReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (RAW.exists() and REPORT.exists() and LOCK.exists()):
            raise unittest.SkipTest("Generated source baseline artifacts are not present yet")
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_exactly_reproducible_without_model_rerun(self) -> None:
        self.assertEqual(build_report(CONFIG, RAW), self.report)

    def test_source_evidence_closes_all_three_coverage_dimensions(self) -> None:
        self.assertTrue(self.report["source_evidence_coverage_complete"])
        self.assertEqual(
            set(self.report["dimensions"]),
            {"COLUMN_ROLE", "DIRECTION", "REFERENCE_BASE"},
        )
        for row in self.report["dimensions"].values():
            self.assertTrue(row["source_evidence_coverage_confirmed"])
            self.assertGreaterEqual(row["companies"], 3)
            self.assertGreaterEqual(row["sources"], 3)
            self.assertGreaterEqual(row["document_layouts"], 3)
            self.assertGreaterEqual(row["table_layouts"], 3)

    def test_runtime_and_later_phases_remain_locked(self) -> None:
        self.assertFalse(self.report["runtime_changed"])
        self.assertFalse(self.report["qp4_runtime_change_authorized"])
        self.assertFalse(self.report["qp5_authorized"])
        self.assertEqual(self.report["phase_d"], "locked")

    def test_lock_hashes_are_exact(self) -> None:
        self.assertTrue(self.lock["source_cohort_complete"])
        self.assertFalse(self.lock["runtime_changed"])
        self.assertFalse(self.lock["qp4_runtime_change_authorized"])
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
