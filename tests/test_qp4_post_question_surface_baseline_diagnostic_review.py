from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.evaluate_qp4_post_question_surface_baseline_diagnostic_review import (
    LAYERS,
    LOCK,
    MARKDOWN,
    PRECOMMIT,
    PRECOMMIT_LOCK,
    PRECOMMIT_REPORT,
    RAW,
    REPORT,
    ROOT,
    TEST,
    build_report,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4PostQuestionSurfaceBaselineDiagnosticReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
        cls.raw = json.loads(RAW.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_frozen_source_inventory_was_run_exactly(self) -> None:
        self.assertEqual(self.raw["compiler_control"]["records"], 242)
        self.assertEqual(self.raw["compiler_control"]["exact"], 242)
        self.assertEqual(len(self.raw["source_results"]), 40)
        tracks: dict[str, int] = {}
        for row in self.raw["source_results"]:
            tracks[row["track"]] = tracks.get(row["track"], 0) + 1
        self.assertEqual(
            tracks,
            {
                "ADD_SURFACE": 15,
                "ADD_COUNTERFACTUAL_SOURCE_ERROR": 1,
                "ACTUAL_PLAN": 24,
            },
        )

    def test_engine_boundary_and_post_hoc_gold_are_preserved(self) -> None:
        inputs = self.raw["inputs"]
        self.assertEqual(inputs["source_run_boundary"], ["question", "source PDF path"])
        self.assertFalse(inputs["gold_candidate_or_program_passed_to_engine"])
        self.assertTrue(inputs["gold_used_only_after_observation"])
        self.assertFalse(inputs["mineru_auto_generate"])
        self.assertEqual(inputs["per_query_timeout_seconds"], 15)
        self.assertTrue(
            all(
                row["candidate_support"]["gold_used_post_hoc_only"]
                for row in self.raw["source_results"]
            )
        )

    def test_every_observation_has_one_earliest_layer_or_pass(self) -> None:
        allowed = {*LAYERS, "PASS"}
        self.assertTrue(
            all(
                row["first_failing_layer"] in allowed
                for row in self.raw["source_results"]
            )
        )

    def test_all_twelve_source_errors_stop_at_source_layer(self) -> None:
        errors = [
            row
            for row in self.raw["source_results"]
            if row["trace"]["source_outcome"] == "ERROR"
        ]
        self.assertEqual(len(errors), 12)
        self.assertTrue(
            all(
                row["first_failing_layer"] == "SOURCE_LOAD_OR_EXTRACTION"
                for row in errors
            )
        )

    def test_required_trace_fields_are_present(self) -> None:
        required = set(self.config["diagnostic_contract"]["trace_fields"])
        for row in self.raw["source_results"]:
            self.assertTrue(required <= set(row["trace"]))

    def test_report_is_reproducible_from_frozen_raw(self) -> None:
        self.assertEqual(build_report(), self.report)

    def test_report_aggregates_all_groups_and_strata(self) -> None:
        source = self.report["source"]
        self.assertEqual(source["queries"], 40)
        self.assertEqual(sum(source["by_first_failing_layer"].values()), 40)
        self.assertIn("macro_group_pass_rate", source)
        self.assertIn("worst_group_pass_rate", source)
        for key, group in source["groups"].items():
            self.assertEqual(
                sum(group["by_first_failing_layer"].values()), group["queries"]
            )
            track, name = key.split(":", 1)
            self.assertEqual(
                sum(
                    row["queries"]
                    for row in source["strata"]
                    if row["track"] == track and row["group"] == name
                ),
                group["queries"],
            )

    def test_runtime_candidates_require_diverse_generic_evidence(self) -> None:
        for row in self.report["implementation_precommit_candidates"]:
            self.assertGreaterEqual(row["companies"], 3)
            self.assertGreaterEqual(row["sources"], 3)
            self.assertGreaterEqual(row["real_layouts"], 3)
            self.assertTrue(row["generic_evidence"])
            self.assertFalse(row["runtime_edit_authorized"])
            self.assertTrue(row["new_precommit_required"])

    def test_singleton_add_counterfactual_error_cannot_nominate_runtime(self) -> None:
        singleton = [
            row
            for row in self.report["source"]["strata"]
            if row["track"] == "ADD_COUNTERFACTUAL_SOURCE_ERROR"
        ]
        self.assertEqual(len(singleton), 1)
        self.assertEqual(singleton[0]["queries"], 1)
        self.assertEqual(singleton[0]["companies"], 1)
        self.assertEqual(singleton[0]["sources"], 1)
        self.assertEqual(singleton[0]["real_layouts"], 1)
        self.assertFalse(singleton[0]["generic_evidence"])
        nominated = self.report["implementation_precommit_candidates"]
        self.assertFalse(
            any(row["track"] == "ADD_COUNTERFACTUAL_SOURCE_ERROR" for row in nominated)
        )

    def test_diagnostic_does_not_authorize_runtime_or_later_phases(self) -> None:
        self.assertTrue(self.report["diagnostic_complete"])
        self.assertTrue(self.report["diagnostic_gate_passed"])
        self.assertFalse(self.report["runtime_changed"])
        self.assertFalse(self.report["runtime_edit_authorized"])
        self.assertFalse(self.report["qp4_complete"])
        self.assertFalse(self.report["qp5_authorized"])
        self.assertEqual(self.report["phase_d"], "locked")
        self.assertFalse(self.lock["runtime_changed"])
        self.assertFalse(self.lock["runtime_edit_authorized"])
        self.assertFalse(self.lock["qp4_complete"])
        self.assertFalse(self.lock["qp5_authorized"])
        self.assertEqual(self.lock["phase_d"], "locked")

    def test_lock_hashes_every_material_artifact(self) -> None:
        artifacts = {row["path"]: row for row in self.lock["artifacts"]}
        evaluator = ROOT / "scripts" / "evaluate_qp4_post_question_surface_baseline_diagnostic_review.py"
        for path in (
            PRECOMMIT,
            PRECOMMIT_REPORT,
            PRECOMMIT_LOCK,
            evaluator,
            TEST,
            RAW,
            REPORT,
            MARKDOWN,
        ):
            relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
            self.assertIn(relative, artifacts)
            self.assertEqual(path.stat().st_size, artifacts[relative]["size_bytes"])
            self.assertEqual(_digest(path), artifacts[relative]["sha256"])


if __name__ == "__main__":
    unittest.main()
