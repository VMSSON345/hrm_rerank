from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from scripts.evaluate_qp4_post_unit_scale_development_baseline_refresh import (
    ACTUAL_PLAN_RAW,
    ADD_RAW,
    LOCK,
    MARKDOWN,
    PRECOMMIT,
    REPORT,
    ROOT,
    TEST,
    build_report,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class QP4PostUnitScaleDevelopmentBaselineRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))
        cls.add_raw = json.loads(ADD_RAW.read_text(encoding="utf-8"))
        cls.actual_raw = json.loads(ACTUAL_PLAN_RAW.read_text(encoding="utf-8"))

    def test_refresh_uses_exact_frozen_source_scope(self) -> None:
        scope = self.config["refresh_scope"]
        self.assertEqual(50, len(self.add_raw["surface_results"]) + len(self.add_raw["counterfactual_results"]))
        self.assertEqual(35, len(self.add_raw["surface_results"]))
        self.assertEqual(15, len(self.add_raw["counterfactual_results"]))
        self.assertEqual(12, len(self.actual_raw["results"]))
        self.assertEqual(74, scope["combined_source_queries"])

    def test_diverse_real_source_coverage_is_not_synthetic(self) -> None:
        add = self.report["add_source_backed"]
        actual = self.report["actual_plan_source_backed"]["coverage"]
        self.assertEqual((5, 5, 3), (add["companies"], add["source_documents"], add["real_layout_families"]))
        self.assertEqual((4, 4, 3, 4), (actual["companies"], actual["sources"], actual["document_layouts"], actual["table_layouts"]))
        self.assertFalse(self.report["invariants"]["synthetic_layout_counted_as_confirmation"])

    def test_compiler_inventory_is_complete(self) -> None:
        compiler = self.report["compiler"]
        self.assertEqual(13, compiler["canonical"]["total"])
        self.assertEqual(3, compiler["ambiguity"]["total"])
        self.assertEqual(91, compiler["surface"]["total"])
        self.assertEqual(57, compiler["counterfactual"]["total"])

    def test_report_is_reproducible_from_raw_measurements(self) -> None:
        rebuilt = build_report(
            compatible_regression=self.report["verification"]["compatible_affected_regression"]
        )
        self.assertEqual(self.report, rebuilt)

    def test_runtime_and_downstream_phases_remain_locked(self) -> None:
        self.assertFalse(self.report["runtime_changed"])
        self.assertFalse(self.report["qp4_complete"])
        self.assertFalse(self.report["qp5_authorized"])
        self.assertEqual("locked", self.report["phase_d"])
        self.assertEqual({}, {k: v for k, v in self.config["refresh_scope"].items() if k == "runtime_file_whitelist" and v})

    def test_gate_outcome_is_derived_not_forced(self) -> None:
        failed = sorted(name for name, passed in self.report["checks"].items() if not passed)
        self.assertEqual(failed, self.report["failed_gates"])
        self.assertEqual(not failed and self.report["refresh_complete"], self.report["baseline_gates_passed"])

    def test_lock_hashes_every_material_artifact(self) -> None:
        artifacts = {row["path"]: row for row in self.lock["artifacts"]}
        for path in (PRECOMMIT, ADD_RAW, ACTUAL_PLAN_RAW, REPORT, MARKDOWN, TEST):
            relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
            self.assertIn(relative, artifacts)
            self.assertEqual(digest(path), artifacts[relative]["sha256"])
            self.assertEqual(path.stat().st_size, artifacts[relative]["size_bytes"])


if __name__ == "__main__":
    unittest.main()
