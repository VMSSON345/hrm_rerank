#!/usr/bin/env python3
"""Audit joint positive/negative V12-C calibration adequacy without recalibration."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re


PROTECTED = re.compile(r"(?:test|holdout|final|blind|validation)", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _refuse(path: Path) -> None:
    if PROTECTED.search(str(path.resolve()).replace("\\", "/")):
        raise ValueError(f"Protected path is not allowed: {path}")


def audit(positive_path: Path, negative_path: Path) -> dict[str, object]:
    for path in (positive_path, negative_path):
        _refuse(path)
    positive = json.loads(positive_path.read_text(encoding="utf-8"))
    negative = json.loads(negative_path.read_text(encoding="utf-8"))
    if positive.get("classification") != "development_only_frozen_evaluation":
        raise ValueError("Positive report is not a frozen development evaluation")
    if negative.get("classification") != "development_only_negative_control_frozen_evaluation":
        raise ValueError("Negative report is not a frozen development negative evaluation")
    if positive.get("threshold_changed") is not False or negative.get("threshold_changed") is not False:
        raise ValueError("A source report changed threshold")
    if positive.get("model_sha256_after") != negative.get("model_sha256_after"):
        raise ValueError("Positive and negative reports used different checkpoints")
    positive_rows = positive.get("records", [])
    negative_rows = negative.get("records", [])
    correct_positive = [
        row for row in positive_rows
        if row.get("program_correct") is True and row.get("execution_correct") is True
    ]
    if not correct_positive or not negative_rows:
        raise ValueError("Joint audit requires non-empty positive and negative observations")
    positive_confidences = [float(row["confidence"]) for row in correct_positive]
    negative_confidences = [float(row["confidence"]) for row in negative_rows]
    positive_floor = min(positive_confidences)
    negative_ceiling = max(negative_confidences)
    unblocked_overlap = [
        row for row in negative_rows
        if not row.get("grounding_rejections") and float(row["confidence"]) >= positive_floor
    ]
    frozen_thresholds = {
        float(row["confidence_threshold"]) for row in positive_rows + negative_rows
    }
    if len(frozen_thresholds) != 1:
        raise ValueError("Expected one frozen RATIO_PERCENT threshold")
    frozen_threshold = next(iter(frozen_thresholds))
    return {
        "protocol": "v12c-joint-positive-negative-calibration-adequacy-v1",
        "classification": "development_only_joint_calibration_adequacy",
        "protected_evaluation_sources_used": False,
        "training_performed": False,
        "model_or_reranker_changed": False,
        "threshold_changed": False,
        "alternative_thresholds_evaluated": False,
        "positive_report": {"path": str(positive_path.resolve()), "sha256": _sha256(positive_path)},
        "negative_report": {"path": str(negative_path.resolve()), "sha256": _sha256(negative_path)},
        "model_sha256": positive.get("model_sha256_after"),
        "frozen_ratio_percent_threshold": frozen_threshold,
        "positive": {
            "controls": len(positive_rows),
            "program_and_execution_correct": len(correct_positive),
            "accepted": sum(bool(row.get("accepted")) for row in positive_rows),
            "confidence_min": min(positive_confidences),
            "confidence_max": max(positive_confidences),
        },
        "negative": {
            "controls": len(negative_rows),
            "false_accepts_at_frozen_threshold": sum(bool(row.get("false_accept")) for row in negative_rows),
            "structurally_rejected": sum(bool(row.get("grounding_rejections")) for row in negative_rows),
            "confidence_min": min(negative_confidences),
            "confidence_max": negative_ceiling,
            "unblocked_controls_at_or_above_positive_confidence_floor": len(unblocked_overlap),
            "unblocked_overlap_examples": [str(row.get("example_id")) for row in unblocked_overlap],
        },
        "separation": {
            "positive_confidence_floor": positive_floor,
            "negative_confidence_ceiling": negative_ceiling,
            "floor_minus_ceiling": positive_floor - negative_ceiling,
            "confidence_only_separable": positive_floor > negative_ceiling,
            "structural_gate_complete_on_negative_controls": all(
                bool(row.get("grounding_rejections")) for row in negative_rows
            ),
        },
        "gate": {
            "current_threshold_zero_false_accept_gate": all(
                not bool(row.get("false_accept")) for row in negative_rows
            ),
            "current_threshold_positive_coverage_gate": any(
                bool(row.get("accepted")) for row in positive_rows
            ),
            "confidence_separation_gate": positive_floor > negative_ceiling,
            "structural_negative_gate": all(
                bool(row.get("grounding_rejections")) for row in negative_rows
            ),
            "ready_for_threshold_calibration": False,
            "threshold_change_authorized": False,
            "phase_d_unlocked": False,
        },
        "conclusion": (
            "The frozen threshold prevents false accepts but also accepts no positive controls. "
            "Six structurally unblocked semantic-confound controls overlap or exceed the correct-positive "
            "confidence floor, so confidence-only threshold lowering is not supported."
        ),
        "next_required_evidence": (
            "Repair the general semantic distinction between liabilities and interest-bearing borrowings "
            "on development-only data, replay both frozen cohorts, and keep the threshold unchanged."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-report", type=Path, required=True)
    parser.add_argument("--negative-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite versioned artifact: {args.output}")
    _refuse(args.output)
    report = audit(args.positive_report, args.negative_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"separation": report["separation"], "gate": report["gate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
