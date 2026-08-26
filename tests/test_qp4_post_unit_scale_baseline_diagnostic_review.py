from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.evaluate_qp4_post_unit_scale_baseline_diagnostic_review import (
    LAYERS,
    LOCK,
    MARKDOWN,
    PRECOMMIT,
    RAW,
    REPORT,
    ROOT,
    TEST,
    build_report,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4PostUnitScaleBaselineDiagnosticReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
        cls.raw = json.loads(RAW.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_frozen_residual_inventory_was_run_exactly(self) -> None:
        self.assertEqual(len(self.raw["compiler_results"]), 18)
        self.assertEqual(len(self.raw["source_results"]), 39)
        tracks = {}
        for row in self.raw["source_results"]:
            tracks[row["track"]] = tracks.get(row["track"], 0) + 1
        self.assertEqual(tracks, {"ADD_SURFACE": 15, "ACTUAL_PLAN": 24})

    def test_engine_boundary_and_post_hoc_gold_are_preserved(self) -> None:
        inputs = self.raw["inputs"]
        self.assertEqual(inputs["source_run_boundary"], ["question", "source PDF path"])
        self.assertFalse(inputs["gold_candidate_or_program_passed_to_engine"])
        self.assertTrue(inputs["gold_used_only_after_observation"])
        self.assertFalse(inputs["mineru_auto_generate"])
        self.assertEqual(inputs["per_query_timeout_seconds"], 15)
        self.assertTrue(all(row["candidate_support"]["gold_used_post_hoc_only"] for row in self.raw["source_results"]))

    def test_every_residual_has_one_earliest_layer_or_pass(self) -> None:
        allowed = {*LAYERS, "PASS"}
        rows = [*self.raw["compiler_results"], *self.raw["source_results"]]
        self.assertTrue(all(row["first_failing_layer"] in allowed for row in rows))
        self.assertTrue(all(row["first_failing_layer"] == "QUESTION_PLAN" for row in self.raw["compiler_results"]))

    def test_source_error_never_becomes_abstention_pass(self) -> None:
        for row in self.raw["source_results"]:
            if row["trace"]["source_outcome"] != "ERROR":
                continue
            if row["semantic_plan_projection"]["passed"]:
                self.assertEqual(row["first_failing_layer"], "SOURCE_LOAD_OR_EXTRACTION")
            else:
                self.assertEqual(row["first_failing_layer"], "QUESTION_PLAN")

    def test_required_trace_fields_are_present(self) -> None:
        required = set(self.config["diagnostic_contract"]["trace_fields"])
        for row in self.raw["source_results"]:
            self.assertTrue(required <= set(row["trace"]))

    def test_report_is_reproducible_from_frozen_raw(self) -> None:
        self.assertEqual(build_report(), self.report)

    def test_report_aggregates_all_source_strata(self) -> None:
        self.assertEqual(self.report["source"]["queries"], 39)
        self.assertEqual(sum(self.report["source"]["by_first_failing_layer"].values()), 39)
        for row in self.report["source"]["groups"].values():
            self.assertEqual(sum(row["by_first_failing_layer"].values()), row["queries"])

    def test_diagnostic_does_not_authorize_runtime_or_later_phases(self) -> None:
        self.assertTrue(self.report["diagnostic_complete"])
        self.assertTrue(self.report["diagnostic_gate_passed"])
        self.assertFalse(self.report["runtime_changed"])
        self.assertFalse(self.report["runtime_edit_authorized"])
        self.assertFalse(self.report["implementation_precommit_candidate"]["runtime_edit_authorized"])
        self.assertTrue(self.report["implementation_precommit_candidate"]["new_precommit_required"])
        self.assertFalse(self.report["qp4_complete"])
        self.assertFalse(self.report["qp5_authorized"])
        self.assertEqual(self.report["phase_d"], "locked")

    def test_lock_hashes_every_material_artifact(self) -> None:
        artifacts = {row["path"]: row for row in self.lock["artifacts"]}
        for path in (PRECOMMIT, RAW, REPORT, MARKDOWN, TEST):
            relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
            self.assertIn(relative, artifacts)
            self.assertEqual(path.stat().st_size, artifacts[relative]["size_bytes"])
            self.assertEqual(_digest(path), artifacts[relative]["sha256"])
        self.assertFalse(self.lock["runtime_edit_authorized"])
        self.assertFalse(self.lock["qp4_complete"])
        self.assertFalse(self.lock["qp5_authorized"])
        self.assertEqual(self.lock["phase_d"], "locked")


if __name__ == "__main__":
    unittest.main()
