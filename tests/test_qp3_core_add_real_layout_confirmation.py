from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_qp3_core_add_real_layout_confirmation.py"
PRECOMMIT = ROOT / "config/finhrm_qp3_core_add_real_layout_confirmation_precommit_v1.json"
REPORT = ROOT / "reports/finhrm_qp3_core_add_real_layout_confirmation_v1.json"
LOCK = ROOT / "reports/finhrm_qp3_core_add_real_layout_confirmation_lock_v1.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("qp3_add_real_layout", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load ADD real-layout evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QP3CoreAddRealLayoutConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.precommit = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))
        cls.lock = json.loads(LOCK.read_text(encoding="utf-8"))

    def test_report_is_exactly_reproducible(self) -> None:
        self.assertEqual(_load_evaluator().build_report(), self.report)

    def test_source_and_runtime_are_frozen(self) -> None:
        source = ROOT / self.precommit["source"]["path"]
        self.assertEqual(source.stat().st_size, self.precommit["source"]["size_bytes"])
        self.assertEqual(_digest(source), self.precommit["source"]["sha256"])
        for path, expected in self.precommit["runtime_hashes_frozen"].items():
            self.assertEqual(_digest(ROOT / path), expected)

    def test_confirmation_is_source_backed_and_proof_complete(self) -> None:
        self.assertTrue(self.report["confirmation_gate_passed"])
        self.assertEqual(self.report["failed_gates"], [])
        self.assertTrue(all(self.report["gate_checks"].values()))
        result = self.report["results"]
        self.assertEqual(result["gold_actions"], {"ANSWER": 10, "REJECT": 2})
        self.assertEqual(result["gold_execution_correct"], 12)
        self.assertEqual(result["positive_total"], 10)
        self.assertEqual(result["physical_collision_hard_negatives"], 2)
        self.assertEqual(result["permutation_exact"], 12)
        self.assertEqual(result["counterfactual_generated"], 12)
        self.assertEqual(result["counterfactual_false_accepts"], 0)

    def test_two_additional_real_layouts_and_six_companies(self) -> None:
        result = self.report["results"]
        self.assertEqual(result["companies"], 5)
        self.assertEqual(result["sources"], 5)
        self.assertEqual(result["all_selected_companies"], 6)
        self.assertEqual(result["all_selected_sources"], 6)
        self.assertEqual(
            result["additional_real_layout_families"],
            [
                "DEVELOPMENT_NATIVE_MULTI_TABLE",
                "DEVELOPMENT_OCR_OR_HYBRID_TABLE",
            ],
        )
        self.assertEqual(len(result["union_real_layout_families"]), 3)
        self.assertEqual(result["company"]["worst_answer_rate"], 1.0)
        self.assertEqual(result["layout"]["worst_answer_rate"], 1.0)

    def test_no_abw_protected_or_synthetic_layout_confirmation(self) -> None:
        source = self.report["source"]
        self.assertEqual(source["selected_diagnosis_records"], 0)
        self.assertEqual(source["selected_protected_records"], 0)
        for name, value in self.report["invariants"].items():
            self.assertFalse(value, name)

    def test_lock_is_exact_and_keeps_qp4_phase_d_closed(self) -> None:
        self.assertTrue(self.lock["confirmation_gate_passed"])
        self.assertFalse(self.lock["qp4_authorized"])
        self.assertEqual(self.lock["phase_d"], "locked")
        for artifact in self.lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertEqual(path.stat().st_size, artifact["size_bytes"])
            self.assertEqual(_digest(path), artifact["sha256"])


if __name__ == "__main__":
    unittest.main()
