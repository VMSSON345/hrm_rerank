from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compose_document_hrm_development_train.py"


def _row(
    example_id: str,
    source: str,
    *,
    operation: str = "SELECT",
    company: str | None = None,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "candidates": [{"source_name": source, "value": "1"}],
        "program": {"operation": operation, "operands": [0]},
        **({"supervision": {"company_key": company}} if company else {}),
    }


class ComposeDocumentHRMDevelopmentTrainTests(unittest.TestCase):
    def _write(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_keeps_calibration_and_eval_out_of_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.jsonl"
            train = root / "train.jsonl"
            calibration = root / "calibration.jsonl"
            evaluation = root / "eval.jsonl"
            self._write(base, [_row("real-1", "baseline.pdf")])
            self._write(train, [_row("synthetic-document-1", "train.pdf")])
            self._write(calibration, [_row("synthetic-document-2", "calibration.pdf")])
            self._write(evaluation, [_row("synthetic-document-3", "eval.pdf")])
            output = root / "training.jsonl"
            manifest = root / "manifest.json"
            completed = subprocess.run(
                [
                    sys.executable, str(SCRIPT), "--base", str(base),
                    "--document-train", str(train),
                    "--document-calibration", str(calibration),
                    "--document-eval", str(evaluation), "--output", str(output),
                    "--manifest", str(manifest),
                ],
                check=True, capture_output=True, text=True,
            )
            self.assertIn('"training_total": 2', completed.stdout)
            self.assertEqual(
                [json.loads(line)["example_id"] for line in output.read_text(encoding="utf-8").splitlines()],
                ["real-1", "synthetic-document-1"],
            )
            recorded = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(recorded["counts"]["document_eval_held_out"], 1)

    def test_rejects_overlapping_document_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {name: root / f"{name}.jsonl" for name in ("base", "train", "calibration", "eval")}
            self._write(paths["base"], [_row("real-1", "baseline.pdf")])
            self._write(paths["train"], [_row("synthetic-document-1", "same.pdf")])
            self._write(paths["calibration"], [_row("synthetic-document-2", "same.pdf")])
            self._write(paths["eval"], [_row("synthetic-document-3", "eval.pdf")])
            completed = subprocess.run(
                [
                    sys.executable, str(SCRIPT), "--base", str(paths["base"]),
                    "--document-train", str(paths["train"]),
                    "--document-calibration", str(paths["calibration"]),
                    "--document-eval", str(paths["eval"]), "--output", str(root / "out.jsonl"),
                    "--manifest", str(root / "manifest.json"),
                ], capture_output=True, text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("source PDFs overlap", completed.stderr)

    def test_balances_document_train_by_company_and_operation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.jsonl"
            train = root / "train.jsonl"
            calibration = root / "calibration.jsonl"
            evaluation = root / "eval.jsonl"
            self._write(base, [_row("real-1", "baseline.pdf")])
            self._write(train, [
                _row(f"select-{index}", "train.pdf", company="company-a")
                for index in range(3)
            ] + [
                _row(
                    f"subtract-{index}",
                    "train.pdf",
                    operation="SUBTRACT",
                    company="company-a",
                )
                for index in range(2)
            ])
            self._write(calibration, [_row("cal-1", "calibration.pdf")])
            self._write(evaluation, [_row("eval-1", "eval.pdf")])
            output = root / "training.jsonl"
            manifest = root / "manifest.json"

            subprocess.run(
                [
                    sys.executable, str(SCRIPT), "--base", str(base),
                    "--document-train", str(train),
                    "--document-calibration", str(calibration),
                    "--document-eval", str(evaluation), "--output", str(output),
                    "--manifest", str(manifest),
                    "--max-per-company-operation", "1",
                ],
                check=True, capture_output=True, text=True,
            )

            recorded = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(recorded["counts"]["document_train_before_balance"], 5)
            self.assertEqual(recorded["counts"]["document_train"], 2)
            operations = [
                json.loads(line)["program"]["operation"]
                for line in output.read_text(encoding="utf-8").splitlines()[1:]
            ]
            self.assertEqual(sorted(operations), ["SELECT", "SUBTRACT"])

    def test_filters_only_document_operations_and_keeps_base_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base.jsonl"
            train = root / "train.jsonl"
            calibration = root / "calibration.jsonl"
            evaluation = root / "eval.jsonl"
            self._write(base, [_row("base-sum4", "baseline.pdf", operation="SUM4")])
            self._write(train, [
                _row("train-add", "train.pdf", operation="ADD", company="company-a"),
                _row("train-sum4", "train.pdf", operation="SUM4", company="company-a"),
            ])
            self._write(calibration, [_row("cal-1", "calibration.pdf")])
            self._write(evaluation, [_row("eval-1", "eval.pdf")])
            output = root / "training.jsonl"
            manifest = root / "manifest.json"

            subprocess.run(
                [
                    sys.executable, str(SCRIPT), "--base", str(base),
                    "--document-train", str(train),
                    "--document-calibration", str(calibration),
                    "--document-eval", str(evaluation), "--output", str(output),
                    "--manifest", str(manifest),
                    "--document-operations", "ADD", "SUBTRACT",
                ],
                check=True, capture_output=True, text=True,
            )

            rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["example_id"] for row in rows], ["base-sum4", "train-add"])
            recorded = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(recorded["counts"]["document_train_before_operation_filter"], 2)
            self.assertEqual(recorded["counts"]["document_train_after_operation_filter"], 1)
            self.assertEqual(recorded["document_operations"], ["ADD", "SUBTRACT"])
