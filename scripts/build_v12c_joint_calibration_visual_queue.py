#!/usr/bin/env python3
"""Build a deterministic, company-disjoint V12-C joint visual-QA queue."""

from __future__ import annotations

import argparse
from collections import Counter
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


def build(paths: list[Path], *, negative_controls: int = 30) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    inputs = []
    for path in paths:
        _refuse(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("classification") != "development_only_joint_calibration_discovery":
            raise ValueError(f"Not a joint development discovery: {path}")
        if payload.get("model_invoked") is not False:
            raise ValueError(f"Discovery invoked a model: {path}")
        rows.extend(payload.get("records", []))
        inputs.append({"path": str(path.resolve()), "sha256": _sha256(path)})

    positives = sorted(
        (row for row in rows if row.get("candidate_stratum") == "positive_candidate"),
        key=lambda row: (str(row["company_key"]), str(row["source_name"]), int(row["page"])),
    )
    positive_by_company: dict[str, dict[str, object]] = {}
    for row in positives:
        positive_by_company.setdefault(str(row["company_key"]), row)
    selected_positive = list(positive_by_company.values())
    positive_companies = set(positive_by_company)

    negatives = sorted(
        (
            row for row in rows
            if row.get("candidate_stratum") == "negative_candidate"
            and str(row["company_key"]) not in positive_companies
        ),
        key=lambda row: (
            str(row["proof_or_failure_type"]), str(row["company_key"]),
            str(row["source_name"]), int(row["page"]),
        ),
    )
    by_failure: dict[str, list[dict[str, object]]] = {}
    seen_company: set[str] = set()
    for row in negatives:
        company = str(row["company_key"])
        if company in seen_company:
            continue
        seen_company.add(company)
        by_failure.setdefault(str(row["proof_or_failure_type"]), []).append(row)
    if len(by_failure) < 3:
        raise ValueError("Fewer than three negative failure strata")

    selected_negative: list[dict[str, object]] = []
    for failure in sorted(by_failure):
        selected_negative.append(by_failure[failure].pop(0))
    remaining = sorted(
        (row for values in by_failure.values() for row in values),
        key=lambda row: (str(row["company_key"]), str(row["source_name"]), int(row["page"])),
    )
    selected_negative.extend(remaining[: max(0, negative_controls - len(selected_negative))])
    if len(selected_negative) != negative_controls:
        raise ValueError("Not enough company-disjoint negative controls")

    selected = []
    for stratum, values in (
        ("positive_candidate", selected_positive),
        ("negative_candidate", selected_negative),
    ):
        for row in values:
            selected.append({**row, "queue_stratum": stratum})
    selected.sort(key=lambda row: (
        str(row["queue_stratum"]), str(row["company_key"]),
        str(row["source_name"]), int(row["page"]),
    ))
    negative_counts = Counter(
        str(row["proof_or_failure_type"]) for row in selected_negative
    )
    return {
        "protocol": "v12c-joint-calibration-visual-queue-v1",
        "classification": "development_only_joint_calibration_pending_visual_qa",
        "selection_policy": {
            "maximum_pages_per_company_per_stratum": 1,
            "maximum_period_controls_per_page": 2,
            "positive_selection": "first source_name/page per company after deterministic sort",
            "negative_selection": "one per failure stratum, then deterministic company/source/page order",
            "model_outputs_used": False,
        },
        "inputs": inputs,
        "summary": {
            "positive_pages": len(selected_positive),
            "positive_company_groups": len(positive_companies),
            "positive_maximum_control_capacity": 2 * len(selected_positive),
            "negative_pages": len(selected_negative),
            "negative_company_groups": len({str(row["company_key"]) for row in selected_negative}),
            "negative_failure_type_counts": dict(sorted(negative_counts.items())),
            "cross_stratum_company_overlap": len(
                positive_companies & {str(row["company_key"]) for row in selected_negative}
            ),
            "ready_for_visual_qa": len(selected_positive) >= 15 and len(selected_negative) >= 30,
            "ready_for_inference": False,
        },
        "records": selected,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", action="append", type=Path, required=True)
    parser.add_argument("--negative-controls", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {args.output}")
    result = build(args.discovery, negative_controls=args.negative_controls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
