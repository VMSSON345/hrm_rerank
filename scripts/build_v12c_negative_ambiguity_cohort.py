#!/usr/bin/env python3
"""Freeze a visually reviewed development-only V12-C negative-control cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re


PROTECTED = re.compile(r"(?:test|holdout|final|blind|validation)", re.IGNORECASE)
REQUIRED_ROLES = {
    "short_term_borrowings",
    "long_term_borrowings",
    "total_capital",
}
ALLOWED_SEMANTIC_ROLES = REQUIRED_ROLES | {
    "aggregate_borrowings_unknown_term",
    "current_liabilities",
    "long_term_liabilities",
    "total_liabilities",
    "total_assets",
    "owners_equity",
}


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


def _excluded_identity(paths: list[Path]) -> tuple[set[str], set[str]]:
    tickers: set[str] = set()
    companies: set[str] = set()
    for path in paths:
        _refuse(path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not str(row.get("classification", "")).startswith("development_only"):
                    raise ValueError(f"Exclusion cohort is not development-only: {path}")
                tickers.add(_ticker(str(row.get("report_id", ""))))
                companies.add(str(row.get("company_key", "")))
    tickers.discard("")
    companies.discard("")
    return tickers, companies


def build(
    discovery_path: Path,
    render_path: Path,
    decisions_path: Path,
    exclusion_cohorts: list[Path],
    *,
    minimum_company_groups: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    for path in (discovery_path, render_path, decisions_path, *exclusion_cohorts):
        _refuse(path)
    discovery = json.loads(discovery_path.read_text(encoding="utf-8"))
    render = json.loads(render_path.read_text(encoding="utf-8"))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    allowed_discovery_classifications = {
        "development_only_negative_control_discovery",
        "development_only_joint_calibration_pending_visual_qa",
    }
    allowed_render_classifications = {
        "development_only_negative_control_pending_visual_qa",
        "development_only_joint_calibration_pending_visual_qa",
    }
    allowed_decision_classifications = {
        "development_only_negative_control_visual_qa",
        "development_only_joint_calibration_visual_qa",
    }
    if discovery.get("classification") not in allowed_discovery_classifications:
        raise ValueError("Invalid discovery classification")
    if render.get("classification") not in allowed_render_classifications:
        raise ValueError("Invalid render classification")
    if decisions.get("classification") not in allowed_decision_classifications:
        raise ValueError("Invalid visual-decision classification")
    if decisions.get("discovery_sha256") != _sha256(discovery_path):
        raise ValueError("Visual decisions do not match discovery")
    if decisions.get("render_queue_sha256") != _sha256(render_path):
        raise ValueError("Visual decisions do not match render queue")

    joint_mode = discovery.get("classification") == "development_only_joint_calibration_pending_visual_qa"
    discovery_rows = {
        (str(row["source_name"]), int(row["page"])): row
        for row in discovery.get("records", [])
        if not joint_mode or row.get("queue_stratum") == "negative_candidate"
    }
    for row in decisions.get("additional_source_records", []):
        if row.get("candidate_stratum") != "negative_candidate":
            raise ValueError("Additional source record is not a negative candidate")
        discovery_rows[(str(row["source_name"]), int(row["page"]))] = row
    render_rows = {
        (str(row["source_name"]), int(row["page"])): row
        for row in render.get("records", [])
    }
    render_hashes = decisions.get("additional_render_hashes", {})
    excluded_tickers, excluded_companies = _excluded_identity(exclusion_cohorts)
    records: list[dict[str, object]] = []
    failure_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    selected_pages: set[tuple[str, int]] = set()
    for decision in decisions.get("records", []):
        if joint_mode and decision.get("queue_stratum") != "negative_candidate":
            continue
        if decision.get("decision") != "approved_negative":
            raise ValueError("Only approved negative decisions may enter the freeze")
        source_name = str(decision["source_name"])
        page = int(decision["page"])
        key = (source_name, page)
        if key in selected_pages:
            raise ValueError(f"Duplicate visual decision: {key}")
        selected_pages.add(key)
        source = discovery_rows.get(key)
        render_row = render_rows.get(key)
        if source is None:
            raise ValueError(f"Selected page is absent from discovery: {key}")
        expected_render_hash = (
            render_row.get("render_sha256") if render_row is not None
            else render_hashes.get(f"{source_name}:{page}")
        )
        if decision.get("render_sha256") != expected_render_hash:
            raise ValueError(f"Render hash mismatch: {key}")
        company_key = str(decision.get("company_key", ""))
        if company_key != source.get("company_key"):
            raise ValueError(f"Company mismatch: {key}")
        if _ticker(source_name) in excluded_tickers or company_key in excluded_companies:
            raise ValueError(f"Exclusion-cohort overlap: {key}")
        unit = str(decision.get("unit", ""))
        scope = str(decision.get("scope", ""))
        if not unit or not scope:
            raise ValueError(f"Missing unit/scope: {key}")
        failure_type = str(decision.get("proof_failure_type", ""))
        proof_status = str(decision.get("required_proof_status", ""))
        if proof_status not in {"INCOMPLETE_REQUIRED_ROLE_ABSENT", "INVALID_ROLE_SEMANTICS"}:
            raise ValueError(f"Unsupported proof status: {key}")
        periods = decision.get("periods")
        if not isinstance(periods, dict) or not periods:
            raise ValueError(f"No reviewed periods: {key}")
        for period, candidates in sorted(periods.items()):
            if not isinstance(candidates, list) or len(candidates) < 2:
                raise ValueError(f"Negative control needs at least two candidates: {key} {period}")
            seen_roles: set[str] = set()
            evidence: list[dict[str, object]] = []
            table_id = f"{Path(source_name).stem}:page-{page}:{scope}:visual-negative-control"
            for candidate in candidates:
                role = str(candidate.get("semantic_role", ""))
                if role not in ALLOWED_SEMANTIC_ROLES:
                    raise ValueError(f"Unsupported semantic role {role}: {key}")
                if role in seen_roles:
                    raise ValueError(f"Duplicate semantic role {role}: {key} {period}")
                seen_roles.add(role)
                value = str(candidate.get("value", ""))
                Decimal(value)
                evidence.append({
                    "semantic_role": role,
                    "row_label": str(candidate.get("row_label", "")),
                    "value": value,
                    "raw_value": value,
                    "period": period,
                    "unit": unit,
                    "source_name": source_name,
                    "source_sha256": source["source_sha256"],
                    "page": page,
                    "table_id": table_id,
                    "line_number": 0,
                    "source_line": "visual-QA locked PDF cell",
                    "extraction_method": "manual_visual_qa_negative_control",
                    "scope": scope,
                })
            actual_missing = REQUIRED_ROLES - seen_roles
            declared_missing = set(decision.get("missing_required_roles", []))
            if actual_missing != declared_missing or not actual_missing:
                raise ValueError(
                    f"Required-role failure is not exact: {key} {period}; "
                    f"actual={sorted(actual_missing)} declared={sorted(declared_missing)}"
                )
            record_id = sha256(
                f"{source['source_sha256']}|{page}|{scope}|{period}|{failure_type}".encode("utf-8")
            ).hexdigest()[:20]
            records.append({
                "schema": "v12c-negative-ambiguity-control-cohort-v1",
                "classification": "development_only_negative_ambiguity_visual_qa_approved",
                "cohort_role": "negative_false_accept_measurement",
                "example_id": f"v12c-neg-{record_id}",
                "company_key": company_key,
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
                "expected_outcome": "ABSTAIN",
                "proof_failure_type": failure_type,
                "required_proof_status": proof_status,
                "missing_required_roles": sorted(actual_missing),
                "evidence": evidence,
                "review_status": "approved_visual_qa_negative_control",
                "eligible_for_positive": False,
            })
            failure_counts[failure_type] += 1
            status_counts[proof_status] += 1

    records.sort(key=lambda row: (
        str(row["company_key"]), str(row["report_id"]), int(row["page"]), str(row["period"])
    ))
    companies = {str(row["company_key"]) for row in records}
    if len(companies) < minimum_company_groups:
        raise ValueError(f"Only {len(companies)} groups; need {minimum_company_groups}")
    audit = {
        "protocol": "v12c-negative-ambiguity-control-freeze-v1",
        "classification": "development_only_negative_control",
        "protected_evaluation_sources_used": False,
        "training_performed": False,
        "model_or_reranker_changed": False,
        "threshold_changed": False,
        "discovery_sha256": _sha256(discovery_path),
        "render_queue_sha256": _sha256(render_path),
        "visual_decisions_sha256": _sha256(decisions_path),
        "exclusion_cohorts": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in exclusion_cohorts
        ],
        "summary": {
            "frozen_negative_controls": len(records),
            "independent_company_groups": len(companies),
            "minimum_company_groups": minimum_company_groups,
            "approved_pages": len(selected_pages),
            "company_overlap_with_repair_or_positive_cohorts": 0,
            "proof_failure_type_counts": dict(sorted(failure_counts.items())),
            "required_proof_status_counts": dict(sorted(status_counts.items())),
            "visual_qa_pass": True,
            "ready_for_frozen_false_accept_measurement": True,
            "ready_for_threshold_change": False,
        },
    }
    return records, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--render-queue", type=Path, required=True)
    parser.add_argument("--visual-decisions", type=Path, required=True)
    parser.add_argument("--exclude-cohort", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--minimum-company-groups", type=int, default=10)
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
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
