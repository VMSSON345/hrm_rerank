#!/usr/bin/env python3
"""Freeze an independent, visually reviewed development V12-C cohort."""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from hashlib import sha256
import json
from pathlib import Path
import re


PROTECTED = re.compile(r"(?:test|holdout|final|blind|validation)", re.IGNORECASE)
ROLES = (
    "short_term_borrowings",
    "long_term_borrowings",
    "total_capital",
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _refuse(path: Path) -> None:
    if PROTECTED.search(str(path.resolve()).replace("\\", "/")):
        raise ValueError(f"Protected path is not allowed: {path}")


def _ticker(source_name: str) -> str:
    return Path(source_name).stem.split("_", 1)[0].upper()


def _excluded_tickers(path: Path) -> set[str]:
    _refuse(path)
    output: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                output.add(_ticker(str(json.loads(line).get("report_id", ""))))
    return output


def build(
    discovery_path: Path,
    render_path: Path,
    decisions_path: Path,
    exclusion_cohort: Path,
    *,
    minimum_company_groups: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    for path in (discovery_path, render_path, decisions_path, exclusion_cohort):
        _refuse(path)
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    render = json.loads(render_path.read_text(encoding="utf-8"))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    allowed_discovery_classifications = {
        "development_only_discovery",
        "development_only_joint_calibration_pending_visual_qa",
    }
    allowed_render_classifications = {
        "development_only_pending_visual_qa",
        "development_only_joint_calibration_pending_visual_qa",
    }
    allowed_decision_classifications = {
        "development_only_visual_qa",
        "development_only_joint_calibration_visual_qa",
    }
    if discovery.get("classification") not in allowed_discovery_classifications:
        raise ValueError("Discovery is not development-only")
    if render.get("classification") not in allowed_render_classifications:
        raise ValueError("Render queue is not development-only")
    if decisions.get("classification") not in allowed_decision_classifications:
        raise ValueError("Visual decisions are not development-only")
    if decisions.get("discovery_sha256") != _sha256(discovery_path):
        raise ValueError("Visual decisions do not match discovery")
    if decisions.get("render_queue_sha256") != _sha256(render_path):
        raise ValueError("Visual decisions do not match render queue")

    joint_mode = discovery.get("classification") == "development_only_joint_calibration_pending_visual_qa"
    discovery_rows = {
        (str(row["source_name"]), int(row["page"])): row
        for row in discovery.get("records", [])
        if not joint_mode or row.get("queue_stratum") == "positive_candidate"
    }
    render_rows = {
        (str(row["source_name"]), int(row["page"])): row
        for row in render.get("records", [])
    }
    render_hashes = decisions.get("additional_render_hashes", {})
    all_decision_rows = decisions.get("records")
    if not isinstance(all_decision_rows, list):
        raise ValueError("Visual decision records are missing")
    decision_rows = [
        row for row in all_decision_rows
        if not joint_mode or row.get("queue_stratum") == "positive_candidate"
    ]
    if len(decision_rows) != len(discovery_rows) or set(discovery_rows) != {
        (str(row["source_name"]), int(row["page"])) for row in decision_rows
    }:
        raise ValueError("Every discovered page needs exactly one visual decision")
    excluded = _excluded_tickers(exclusion_cohort)
    records: list[dict[str, object]] = []
    approved_pages = rejected_pages = partial_pages = 0
    excluded_periods = 0
    for decision in decision_rows:
        source_name = str(decision["source_name"])
        page = int(decision["page"])
        key = (source_name, page)
        source = discovery_rows[key]
        render_row = render_rows.get(key)
        expected_render_hash = (
            render_row.get("render_sha256") if render_row is not None
            else render_hashes.get(f"{source_name}:{page}")
        )
        if decision.get("render_sha256") != expected_render_hash:
            raise ValueError(f"Render hash mismatch: {key}")
        if _ticker(source_name) in excluded:
            raise ValueError(f"Repair-cohort company overlap: {source_name}")
        if decision.get("company_key") != source.get("company_key"):
            raise ValueError(f"Company key mismatch: {key}")
        status = str(decision.get("decision", ""))
        if status == "rejected":
            rejected_pages += 1
            continue
        if status not in {"approved", "approved_partial"}:
            raise ValueError(f"Unsupported decision: {status}")
        approved_pages += 1
        partial_pages += int(status == "approved_partial")
        excluded_periods += len(decision.get("excluded_periods", {}))
        periods = decision.get("periods")
        if not isinstance(periods, dict) or not periods:
            raise ValueError(f"Approved page has no periods: {key}")
        unit = str(decision.get("unit", ""))
        scope = str(decision.get("scope", ""))
        if not unit or not scope:
            raise ValueError(f"Missing unit or scope: {key}")
        for period, values in sorted(periods.items()):
            if not isinstance(values, dict) or tuple(values) != ROLES:
                raise ValueError(f"Invalid role contract: {key} {period}")
            decimals = [Decimal(str(values[role])) for role in ROLES]
            if decimals[2] == 0:
                raise ValueError(f"Zero denominator: {key} {period}")
            with localcontext() as context:
                context.prec = 34
                answer = (decimals[0] + decimals[1]) / decimals[2] * Decimal(100)
            table_id = f"{Path(source_name).stem}:page-{page}:{scope}:balance-sheet"
            record_id = sha256(
                f"{source['source_sha256']}|{page}|{scope}|{period}".encode("utf-8")
            ).hexdigest()[:20]
            evidence = [
                {
                    "role": role,
                    "value": str(values[role]),
                    "raw_value": str(values[role]),
                    "period": period,
                    "unit": unit,
                    "source_name": source_name,
                    "source_sha256": source["source_sha256"],
                    "page": page,
                    "table_id": table_id,
                    "line_number": 0,
                    "source_line": "visual-QA locked PDF cell",
                    "extraction_method": "manual_visual_qa_exact_cell",
                    "scope": scope,
                }
                for role in ROLES
            ]
            records.append({
                "schema": "v12c-independent-calibration-cohort-v1",
                "classification": "development_only_independent_calibration_visual_qa_approved",
                "cohort_role": "independent_calibration_measurement",
                "example_id": f"v12c-cal-{record_id}",
                "company_key": source["company_key"],
                "report_id": source_name,
                "source_sha256": source["source_sha256"],
                "page": page,
                "table_id": table_id,
                "scope": scope,
                "period": period,
                "question": (
                    "Tổng nợ vay tài chính chịu lãi gồm vay ngắn hạn và dài hạn "
                    f"chiếm bao nhiêu phần trăm Tổng nguồn vốn tại {period}?"
                ),
                "answer_value": format(answer, "f"),
                "evidence": evidence,
                "review_status": "approved_visual_qa_independent_calibration",
                "eligible_for_positive": True,
            })
    records.sort(key=lambda row: (
        str(row["company_key"]), str(row["report_id"]), int(row["page"]), str(row["period"])
    ))
    companies = {str(row["company_key"]) for row in records}
    if len(companies) < minimum_company_groups:
        raise ValueError(
            f"Only {len(companies)} independent groups; need {minimum_company_groups}"
        )
    audit = {
        "protocol": "v12c-independent-calibration-cohort-freeze-v1",
        "classification": "development_only",
        "protected_evaluation_sources_used": False,
        "training_performed": False,
        "model_or_reranker_changed": False,
        "threshold_changed": False,
        "discovery_sha256": _sha256(discovery_path),
        "render_queue_sha256": _sha256(render_path),
        "visual_decisions_sha256": _sha256(decisions_path),
        "repair_exclusion_cohort_sha256": _sha256(exclusion_cohort),
        "repair_excluded_tickers": sorted(excluded),
        "summary": {
            "frozen_records": len(records),
            "independent_company_groups": len(companies),
            "minimum_company_groups": minimum_company_groups,
            "approved_pages": approved_pages,
            "approved_partial_pages": partial_pages,
            "rejected_pages": rejected_pages,
            "excluded_non_numeric_periods": excluded_periods,
            "company_overlap_with_repair_cohort": 0,
            "visual_qa_pass": True,
            "ready_for_frozen_inference": True,
            "ready_for_threshold_calibration": False,
        },
    }
    return records, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--render-queue", type=Path, required=True)
    parser.add_argument("--visual-decisions", type=Path, required=True)
    parser.add_argument("--exclude-cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--minimum-company-groups", type=int, default=8)
    args = parser.parse_args()
    for path in (args.output, args.audit):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite artifact: {path}")
        _refuse(path)
    records, audit = build(
        args.discovery,
        args.render_queue,
        args.visual_decisions,
        args.exclude_cohort,
        minimum_company_groups=args.minimum_company_groups,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    audit["cohort_path"] = str(args.output.resolve())
    audit["cohort_sha256"] = _sha256(args.output)
    audit["builder_sha256"] = _sha256(Path(__file__))
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
