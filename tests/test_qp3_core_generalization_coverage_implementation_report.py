from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_qp3_core_generalization_coverage_implementation.py"
REPORT = ROOT / "reports/finhrm_qp3_core_generalization_coverage_implementation_v1.json"
LOCK = ROOT / "reports/finhrm_qp3_core_generalization_coverage_implementation_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("qp3_coverage_implementation", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load implementation evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QP3CoreGeneralizationCoverageImplementationReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_exactly_reproducible(self) -> None:
        self.assertEqual(_load_evaluator().build_report(), self.report)

    def test_safety_and_end_to_end_diversity_gate_pass(self) -> None:
        self.assertTrue(self.report["implementation_gate_passed"])
        self.assertEqual(self.report["failed_gates"], [])
        checks = self.report["gate_checks"]
        self.assertTrue(checks["add_real_layout_confirmation_passed"])
        self.assertTrue(checks["add_diverse_end_to_end_confirmation_passed"])
        self.assertTrue(checks["add_improvement_minimum_3_layouts"])

    def test_replay_counts_are_exact(self) -> None:
        result = self.report["results"]
        self.assertEqual(result["baseline_answers_retained"], "187/187")
        self.assertEqual(result["new_answers"], 72)
        self.assertEqual(result["gold_execution"], "363/363")
        self.assertEqual(result["permutation_exact"], "363/363")
        self.assertEqual(result["counterfactual_false_accepts"], 0)
        self.assertEqual(result["add_grounding"]["baseline_unknown"], 67)
        self.assertEqual(result["add_grounding"]["implementation_unknown"], 3)
        self.assertEqual(result["add_grounding"]["improvement_companies"], 44)
        self.assertEqual(result["add_grounding"]["safe_replay_improvement_layouts"], 1)
        self.assertEqual(result["add_grounding"]["confirmation_positive_answers"], 10)
        self.assertEqual(result["add_grounding"]["confirmation_companies"], 5)
        self.assertEqual(result["add_grounding"]["confirmation_layouts"], 2)
        self.assertEqual(result["add_grounding"]["improvement_layouts"], 1)
        self.assertEqual(result["add_grounding"]["candidate_supported_real_layouts"], 3)
        self.assertFalse(result["add_grounding"]["candidate_confirmation_gate_closing"])
        self.assertEqual(result["add_grounding"]["end_to_end_positive_answers"], 6)
        self.assertEqual(result["add_grounding"]["end_to_end_additional_layouts"], 2)
        self.assertEqual(result["add_grounding"]["confirmed_real_layouts"], 3)
        self.assertEqual(
            result["add_end_to_end_confirmation"]["layout_minimum_checks"],
            {"NATIVE_TEXT_TABLE": True, "OCR_OR_HYBRID_TABLE": True},
        )
        self.assertEqual(result["query_metric"]["baseline_unknown"], 33)
        self.assertEqual(result["query_metric"]["implementation_unknown"], 25)
        self.assertEqual(result["query_metric"]["improvement_companies"], 3)
        self.assertEqual(result["query_metric"]["improvement_layouts"], 3)
        self.assertEqual(result["period_coverage"]["baseline_unknown"], 74)
        self.assertEqual(result["period_coverage"]["implementation_unknown"], 74)

    def test_runtime_checkpoint_and_forbidden_actions_are_locked(self) -> None:
        for relative, expected in self.report["runtime_hashes"].items():
            self.assertEqual(_digest(ROOT / relative), expected)
        self.assertFalse(self.report["invariants"]["checkpoint_changed"])
        for name, value in self.report["invariants"].items():
            if name != "qp4_authorized":
                self.assertFalse(value, name)
        self.assertFalse(self.report["invariants"]["qp4_authorized"])

    def test_lock_is_exact_and_does_not_open_qp4(self) -> None:
        self.assertTrue(self.lock["implementation_gate_passed"])
        self.assertFalse(self.lock["qp4_authorized"])
        self.assertEqual(self.lock["phase_d"], "locked")
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
