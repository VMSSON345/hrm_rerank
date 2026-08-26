from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "finhrm_qp3_core_generalization_coverage_diagnostic_matrix_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp3_core_generalization_coverage_diagnostic_matrix_v1.json"
LOCK = ROOT / "reports" / "finhrm_qp3_core_generalization_coverage_diagnostic_matrix_lock_v1.json"
SCRIPT = ROOT / "scripts" / "build_qp3_core_generalization_coverage_diagnostic_matrix.py"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_builder():
    spec = importlib.util.spec_from_file_location("qp3_diagnostic_matrix", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load diagnostic matrix builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QP3CoreGeneralizationCoverageDiagnosticMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_parent_source_and_runtime_are_hash_locked(self) -> None:
        parent = self.config["parent_precommit"]
        source = self.config["source_report"]
        self.assertEqual(_digest(ROOT / parent["path"]), parent["sha256"])
        self.assertEqual(_digest(ROOT / source["path"]), source["sha256"])
        precommit = json.loads((ROOT / parent["path"]).read_text(encoding="utf-8"))
        for relative, expected in precommit["runtime_hashes_at_precommit"].items():
            self.assertEqual(_digest(ROOT / relative), expected)

    def test_report_is_exactly_reproducible(self) -> None:
        rebuilt = _load_builder().build_matrix(self.config)
        self.assertEqual(rebuilt, self.report)
        self.assertTrue(self.report["matrix_gate_passed"])
        self.assertEqual(self.report["errors"], [])

    def test_every_track_has_four_controls_and_group_diversity(self) -> None:
        minimum_companies = self.config["diversity_gate"][
            "minimum_distinct_companies_per_track_and_control"
        ]
        minimum_layouts = self.config["diversity_gate"][
            "minimum_distinct_layouts_per_track_and_control"
        ]
        expected_types = set(self.config["control_types"])
        for track in self.report["tracks"].values():
            self.assertEqual(set(track["controls"]), expected_types)
            for control in track["controls"].values():
                self.assertGreaterEqual(control["distinct_companies"], minimum_companies)
                self.assertGreaterEqual(control["distinct_layouts"], minimum_layouts)
                self.assertEqual(
                    len(control["matrix_cells"]), control["distinct_layouts"]
                )

    def test_track_counts_and_control_totals_are_frozen(self) -> None:
        expected = {
            "period_and_provenance": (74, 73, 2, 3, 3, 308),
            "operation_and_add_grounding": (67, 47, 3, 0, 3, 268),
            "query_metric": (33, 26, 6, 0, 6, 132),
        }
        for name, values in expected.items():
            track = self.report["tracks"][name]
            controls = sum(item["instances"] for item in track["controls"].values())
            observed = (
                track["source_examples"],
                track["source_companies"],
                track["source_layouts"],
                track["synthetic_layout_projection_instances"],
                track["matrix_layouts"],
                controls,
            )
            self.assertEqual(observed, values)
        self.assertEqual(self.report["totals"]["track_control_instances"], 708)
        self.assertEqual(self.report["totals"]["aggregate_matrix_cells"], 48)

    def test_period_projection_is_diagnostic_only(self) -> None:
        projection = self.config["tracks"]["period_and_provenance"][
            "layout_projection"
        ]
        self.assertTrue(projection["enabled"])
        self.assertEqual(projection["source_companies"], 3)
        self.assertFalse(projection["may_count_as_confirmation"])
        self.assertFalse(
            self.config["diversity_gate"][
                "synthetic_layout_counts_as_unseen_confirmation"
            ]
        )
        self.assertFalse(
            self.report["invariants"][
                "synthetic_layout_counted_as_unseen_confirmation"
            ]
        )

    def test_no_source_company_or_example_identifier_is_persisted(self) -> None:
        serialized = REPORT.read_text(encoding="utf-8").lower()
        self.assertNotIn("devco_", serialized)
        self.assertNotIn('"company_key":', serialized)
        self.assertNotIn('"example_id":', serialized)
        self.assertFalse(self.report["source"]["company_identifiers_persisted"])
        self.assertFalse(self.report["source"]["example_identifiers_persisted"])

    def test_safety_invariants_and_forbidden_actions_are_locked(self) -> None:
        for key, value in self.report["invariants"].items():
            self.assertFalse(value, key)
        for key, value in self.config["forbidden"].items():
            self.assertTrue(value, key)
        self.assertIn("IMPLEMENTATION-PRECOMMIT", self.report["authorized_next_step"])

    def test_lock_artifacts_are_exact_and_runtime_remains_locked(self) -> None:
        self.assertTrue(self.lock["matrix_gate_passed"])
        self.assertFalse(self.lock["runtime_change_authorized"])
        self.assertFalse(self.lock["qp4_authorized"])
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
