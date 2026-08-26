#!/usr/bin/env python3
"""Apply the precommitted V12-C operating-point gate to fresh frozen reports."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from math import comb
from pathlib import Path
import re


PROTECTED = re.compile(r"(?:test|holdout|final|blind|validation)", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cp_one_sided_lower(successes: int, trials: int, alpha: float = 0.05) -> float | None:
    if trials == 0:
        return None
    if successes == 0:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(100):
        value = (low + high) / 2
        tail = sum(
            comb(trials, index) * value**index * (1 - value) ** (trials - index)
            for index in range(successes, trials + 1)
        )
        if tail < alpha:
            low = value
        else:
            high = value
    return (low + high) / 2


def evaluate(protocol_path: Path, positive_path: Path, negative_path: Path) -> dict[str, object]:
    for path in (protocol_path, positive_path, negative_path):
        if PROTECTED.search(str(path.resolve()).replace("\\", "/")):
            raise ValueError(f"Protected path is not allowed: {path}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    positive = json.loads(positive_path.read_text(encoding="utf-8"))
    negative = json.loads(negative_path.read_text(encoding="utf-8"))
    if protocol.get("classification") != "development_only_precommitted_calibration_protocol":
        raise ValueError("Protocol is not the precommitted development-only protocol")
    if positive.get("classification") != "development_only_frozen_evaluation":
        raise ValueError("Positive report is not a frozen development evaluation")
    if negative.get("classification") != "development_only_negative_control_frozen_evaluation":
        raise ValueError("Negative report is not a frozen development negative evaluation")
    if positive.get("model_sha256_after") != negative.get("model_sha256_after"):
        raise ValueError("Reports used different checkpoint hashes")
    if positive.get("threshold_changed") is not False or negative.get("threshold_changed") is not False:
        raise ValueError("A source report changed threshold")

    positive_rows = positive["records"]
    negative_rows = negative["records"]
    all_thresholds = {float(row["confidence_threshold"]) for row in positive_rows + negative_rows}
    frozen = float(protocol["locked_inputs"]["frozen_ratio_percent_threshold"])
    if all_thresholds != {frozen}:
        raise ValueError(f"Observed thresholds differ from frozen protocol value: {all_thresholds}")
    positive_companies = {str(row["company_key"]) for row in positive_rows}
    negative_companies = {str(row["company_key"]) for row in negative_rows}
    if positive_companies & negative_companies:
        raise ValueError("Positive and negative companies overlap")

    accepted_positive = [row for row in positive_rows if row["accepted"]]
    correct_accepted_positive = [
        row for row in accepted_positive
        if row["program_correct"] and row["execution_correct"]
    ]
    false_accepted_negative = [row for row in negative_rows if row["false_accept"]]
    accepted_denominator = len(accepted_positive) + len(false_accepted_negative)
    accepted_numerator = len(correct_accepted_positive)
    precision = accepted_numerator / accepted_denominator if accepted_denominator else None
    lower = _cp_one_sided_lower(accepted_numerator, accepted_denominator)
    accepted_companies = {
        str(row["company_key"]) for row in accepted_positive + false_accepted_negative
    }
    positive_coverage = len(accepted_positive) / len(positive_rows)
    false_accept_rate = len(false_accepted_negative) / len(negative_rows)
    false_accepts_by_failure = Counter(
        str(row["proof_failure_type"]) for row in false_accepted_negative
    )
    observed_failures = {str(row["proof_failure_type"]) for row in negative_rows}
    gate = protocol["precommitted_operating_point_gate"]
    conditions = {
        "accepted_precision_point_estimate": (
            precision is not None and precision >= gate["minimum_accepted_precision_point_estimate"]
        ),
        "accepted_precision_one_sided_95_lower_bound": (
            lower is not None and lower >= gate["minimum_accepted_precision_one_sided_95_lower_bound"]
        ),
        "minimum_accepted_answers": accepted_denominator >= gate["minimum_accepted_answers"],
        "minimum_accepted_company_groups": len(accepted_companies) >= gate["minimum_accepted_company_groups"],
        "minimum_positive_accepted_coverage": positive_coverage >= gate["minimum_positive_accepted_coverage"],
        "maximum_negative_false_accept_rate": false_accept_rate <= gate["maximum_negative_false_accept_rate"],
        "zero_false_accept_in_every_negative_failure_stratum": all(
            false_accepts_by_failure[failure] == 0 for failure in observed_failures
        ),
        "checkpoint_hash_unchanged": (
            positive["model_sha256_before"] == positive["model_sha256_after"]
            == negative["model_sha256_before"] == negative["model_sha256_after"]
            == protocol["locked_inputs"]["checkpoint"]["sha256"]
        ),
        "runtime_threshold_unchanged": all_thresholds == {frozen},
    }
    all_conditions_pass = all(conditions.values())
    return {
        "protocol": "v12c-joint-calibration-fresh-operating-point-v1",
        "classification": "development_only_joint_calibration_frozen_operating_point",
        "protected_evaluation_sources_used": False,
        "training_performed": False,
        "model_or_reranker_changed": False,
        "threshold_changed": False,
        "alternative_thresholds_evaluated": False,
        "inputs": {
            "precommitted_protocol": {"path": str(protocol_path.resolve()), "sha256": _sha256(protocol_path)},
            "positive_report": {"path": str(positive_path.resolve()), "sha256": _sha256(positive_path)},
            "negative_report": {"path": str(negative_path.resolve()), "sha256": _sha256(negative_path)},
        },
        "frozen_ratio_percent_threshold": frozen,
        "cohort_adequacy": {
            "positive_controls": len(positive_rows),
            "positive_company_groups": len(positive_companies),
            "negative_controls": len(negative_rows),
            "negative_company_groups": len(negative_companies),
            "negative_failure_strata": sorted(observed_failures),
            "positive_negative_company_overlap": 0,
        },
        "measurement": {
            "positive_program_and_execution_correct": sum(
                bool(row["program_correct"] and row["execution_correct"]) for row in positive_rows
            ),
            "accepted_positive": len(accepted_positive),
            "positive_accepted_coverage": positive_coverage,
            "negative_false_accepts": len(false_accepted_negative),
            "negative_false_accept_rate": false_accept_rate,
            "negative_false_accepts_by_failure_stratum": {
                failure: false_accepts_by_failure[failure] for failure in sorted(observed_failures)
            },
            "accepted_answers": accepted_denominator,
            "accepted_company_groups": len(accepted_companies),
            "accepted_precision_numerator": accepted_numerator,
            "accepted_precision_denominator": accepted_denominator,
            "accepted_precision_point_estimate": precision,
            "accepted_precision_one_sided_95_lower_bound": lower,
        },
        "gate": {
            "conditions": conditions,
            "all_conditions_pass": all_conditions_pass,
            "zero_acceptance_is_not_zero_precision": accepted_denominator == 0,
            "threshold_change_authorized": False,
            "phase_d_unlocked": False,
        },
        "conclusion": (
            "All precommitted measurement conditions pass on this frozen replay. Because the runtime repair was "
            "designed after diagnosing these cohorts, this is repair evidence rather than an independent "
            "confirmation; it does not authorize a threshold change or unlock Phase D."
            if all_conditions_pass
            else
            "The fresh company-disjoint cohort is statistically adequate, and all positive programs execute "
            "correctly while all hard negatives abstain. The operating-point gate still fails at least one "
            "precommitted measurement condition."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--positive-report", type=Path, required=True)
    parser.add_argument("--negative-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {args.output}")
    report = evaluate(args.protocol, args.positive_report, args.negative_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"measurement": report["measurement"], "gate": report["gate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
