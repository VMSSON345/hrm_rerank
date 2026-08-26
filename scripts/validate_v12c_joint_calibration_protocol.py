#!/usr/bin/env python3
"""Validate the locked V12-C joint calibration protocol and exclusions."""

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


def _ticker(report_id: str) -> str:
    return Path(report_id).stem.split("_", 1)[0].upper()


def validate(protocol_path: Path, root: Path) -> dict[str, object]:
    _refuse(protocol_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "precommitted_before_new_source_discovery":
        raise ValueError("Protocol is not marked precommitted")
    if protocol.get("classification") != "development_only_precommitted_calibration_protocol":
        raise ValueError("Protocol classification is invalid")
    forbidden_true = (
        "protected_evaluation_sources_allowed",
        "training_or_fine_tuning_allowed",
        "model_or_reranker_change_allowed",
        "runtime_threshold_change_during_measurement_allowed",
        "alternative_threshold_search_allowed",
    )
    if any(protocol.get(key) is not False for key in forbidden_true):
        raise ValueError("Protocol permits a forbidden action")

    locked = protocol["locked_inputs"]
    checked_hashes: dict[str, str] = {}
    for name in ("development_catalog", "checkpoint", "runtime_source"):
        entry = locked[name]
        path = root / entry["path"]
        _refuse(path)
        actual = _sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"Locked hash mismatch: {name}")
        checked_hashes[str(entry["path"])] = actual

    observed_tickers: set[str] = set()
    observed_companies: set[str] = set()
    cohort_hashes: dict[str, str] = {}
    for entry in protocol["exclusion_cohorts"]:
        path = root / entry["path"]
        _refuse(path)
        actual = _sha256(path)
        if actual != entry["sha256"]:
            raise ValueError(f"Exclusion hash mismatch: {path}")
        cohort_hashes[str(entry["path"])] = actual
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not str(row.get("classification", "")).startswith("development_only"):
                    raise ValueError(f"Non-development row in exclusion: {path}")
                observed_tickers.add(_ticker(str(row.get("report_id", ""))))
                observed_companies.add(str(row.get("company_key", "")))
    observed_tickers.discard("")
    observed_companies.discard("")
    policy = protocol["exclusion_policy"]
    if observed_tickers != set(policy["excluded_tickers"]):
        raise ValueError("Precommitted ticker exclusion is incomplete")
    if observed_companies != set(policy["excluded_company_keys"]):
        raise ValueError("Precommitted company exclusion is incomplete")

    positive = protocol["cohort_contract"]["positive"]
    negative = protocol["cohort_contract"]["negative"]
    gate = protocol["precommitted_operating_point_gate"]
    if positive["minimum_controls"] < 30 or positive["minimum_company_groups"] < 15:
        raise ValueError("Positive stratum minimum is too small")
    if negative["minimum_controls"] < 30 or negative["minimum_company_groups"] < 15:
        raise ValueError("Negative stratum minimum is too small")
    if negative["minimum_failure_strata"] < 3:
        raise ValueError("Negative failure diversity is too small")
    if gate["minimum_accepted_precision_one_sided_95_lower_bound"] < 0.90:
        raise ValueError("Precision lower-bound gate is below 90%")
    if gate["maximum_negative_false_accept_rate"] != 0.0:
        raise ValueError("Negative false accepts must be forbidden")
    return {
        "protocol": protocol["protocol"],
        "protocol_sha256": _sha256(protocol_path),
        "checked_locked_hashes": checked_hashes,
        "checked_exclusion_hashes": cohort_hashes,
        "excluded_tickers": len(observed_tickers),
        "excluded_company_keys": len(observed_companies),
        "precommit_valid": True,
        "ready_for_development_only_discovery": True,
        "ready_for_threshold_change": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {args.output}")
    _refuse(args.output)
    report = validate(args.protocol, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
