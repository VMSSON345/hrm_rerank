#!/usr/bin/env python3
"""Freeze visually approved, development-only V12-C cohort records."""

from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from hashlib import sha256
import json
from pathlib import Path
import re


PROTECTED_NAME = re.compile(
    r"(?:test|holdout|final|blind|validation)", re.IGNORECASE
)
ROLE_ORDER = (
    "short_term_borrowings",
    "long_term_borrowings",
    "total_capital",
)
REQUIRED_VISUAL_CHECKS = (
    "balance_sheet_page",
    "roles_unique",
    "values_match",
    "period_headers_match",
    "unit_match",
    "entity_scope_match",
    "no_cross_panel_leakage",
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _refuse_protected(path: Path) -> None:
    if PROTECTED_NAME.search(path.name):
        raise ValueError(f"Protected evaluation path is not allowed: {path}")


def _load_json(path: Path) -> dict[str, object]:
    _refuse_protected(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    _refuse_protected(path)
    output: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            output.append(value)
    return output


def freeze(
    candidates_path: Path,
    decisions_path: Path,
    render_report_path: Path,
    *,
    minimum_company_groups: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    candidate_hash = _sha256(candidates_path)
    candidates = _load_jsonl(candidates_path)
    decisions = _load_json(decisions_path)
    render_report = _load_json(render_report_path)
    if decisions.get("candidate_sha256") != candidate_hash:
        raise ValueError("Visual decisions are not pinned to the candidate artifact")
    if render_report.get("candidate_sha256") != candidate_hash:
        raise ValueError("Render report is not pinned to the candidate artifact")
    if decisions.get("classification") != "development_only_visual_qa":
        raise ValueError("Visual decisions must be development-only")

    render_rows = render_report.get("records")
    decision_rows = decisions.get("records")
    if not isinstance(render_rows, list) or not isinstance(decision_rows, list):
        raise ValueError("Missing render or decision records")
    renders: dict[tuple[str, int], dict[str, object]] = {}
    for row in render_rows:
        key = (str(row["source_name"]), int(row["page"]))
        if key in renders:
            raise ValueError(f"Duplicate render key: {key}")
        _refuse_protected(Path(key[0]))
        renders[key] = row
    visual: dict[tuple[str, int], dict[str, object]] = {}
    for row in decision_rows:
        key = (str(row["source_name"]), int(row["page"]))
        if key in visual:
            raise ValueError(f"Duplicate visual decision key: {key}")
        _refuse_protected(Path(key[0]))
        if row.get("decision") != "approved":
            raise ValueError(f"Visual page is not approved: {key}")
        checks = row.get("checks")
        if not isinstance(checks, dict) or not all(
            checks.get(name) is True for name in REQUIRED_VISUAL_CHECKS
        ):
            raise ValueError(f"Incomplete visual checks: {key}")
        render = renders.get(key)
        if render is None or row.get("render_sha256") != render.get("render_sha256"):
            raise ValueError(f"Visual decision render mismatch: {key}")
        visual[key] = row

    frozen: list[dict[str, object]] = []
    example_ids: set[str] = set()
    signatures: set[tuple[object, ...]] = set()
    pages: set[tuple[str, int]] = set()
    companies: set[str] = set()
    for record in candidates:
        if record.get("classification") != "development_only_pending_visual_qa":
            raise ValueError("Candidate classification is not pending visual QA")
        if record.get("eligible_for_positive") is not False:
            raise ValueError("Candidate was positive before visual QA")
        source_name = str(record.get("report_id", ""))
        page = int(record.get("page", 0))
        key = (source_name, page)
        decision = visual.get(key)
        if decision is None:
            raise ValueError(f"Missing visual approval: {key}")
        pages.add(key)
        evidence = record.get("evidence")
        if not isinstance(evidence, list) or len(evidence) != 3:
            raise ValueError(f"Expected exactly three evidence leaves: {key}")
        roles = tuple(str(item.get("role", "")) for item in evidence)
        if roles != ROLE_ORDER or len(set(roles)) != 3:
            raise ValueError(f"Invalid role contract: {key}")
        shared_fields = ("period", "unit", "source_name", "source_sha256", "page", "table_id")
        for field in shared_fields:
            values = {str(item.get(field, "")) for item in evidence}
            if len(values) != 1:
                raise ValueError(f"Mixed evidence {field}: {key}")
        if source_name != evidence[0]["source_name"] or page != int(evidence[0]["page"]):
            raise ValueError(f"Record/evidence provenance mismatch: {key}")
        if record.get("source_sha256") != evidence[0]["source_sha256"]:
            raise ValueError(f"Source hash mismatch: {key}")
        period = str(record.get("period", ""))
        observed = decision.get("observed_values")
        if not isinstance(observed, dict) or period not in observed:
            raise ValueError(f"Missing visually observed period: {key} {period}")
        observed_values = observed[period]
        expected_values = {str(item["role"]): str(item["value"]) for item in evidence}
        if observed_values != expected_values:
            raise ValueError(f"Visual/value mismatch: {key} {period}")
        short, long, total = (Decimal(str(item["value"])) for item in evidence)
        if total == 0:
            raise ValueError(f"Zero denominator: {key} {period}")
        with localcontext() as context:
            context.prec = 34
            recomputed = (short + long) / total * Decimal(100)
        if Decimal(str(record.get("answer_value"))) != recomputed:
            raise ValueError(f"Answer arithmetic mismatch: {key} {period}")
        example_id = str(record.get("example_id", ""))
        if not example_id or example_id in example_ids:
            raise ValueError(f"Duplicate or missing example id: {example_id}")
        example_ids.add(example_id)
        company_key = str(record.get("company_key", ""))
        if not company_key:
            raise ValueError(f"Missing company key: {example_id}")
        companies.add(company_key)
        signature = (
            company_key,
            record.get("source_sha256"),
            page,
            record.get("table_id"),
            period,
        )
        if signature in signatures:
            raise ValueError(f"Duplicate semantic signature: {signature}")
        signatures.add(signature)
        approved = dict(record)
        approved["schema"] = "v12c-interest-bearing-debt-share-frozen-v1"
        approved["classification"] = "development_only_visual_qa_approved"
        approved["review_status"] = "approved_visual_qa"
        approved["eligible_for_positive"] = True
        approved["visual_review_id"] = decisions.get("review_id")
        frozen.append(approved)

    if pages != set(visual) or pages != set(renders):
        raise ValueError("Candidate, render, and visual page sets differ")
    if len(companies) < minimum_company_groups:
        raise ValueError(
            f"Only {len(companies)} company groups; need {minimum_company_groups}"
        )
    frozen.sort(
        key=lambda item: (
            str(item["company_key"]),
            str(item["report_id"]),
            int(item["page"]),
            str(item["period"]),
        )
    )
    audit = {
        "protocol": "v12c-interest-bearing-debt-share-cohort-freeze-v1",
        "classification": "development_only",
        "protected_evaluation_sources_used": False,
        "training_performed": False,
        "model_or_reranker_changed": False,
        "threshold_changed": False,
        "candidate_path": str(candidates_path.resolve()),
        "candidate_sha256": candidate_hash,
        "visual_decisions_path": str(decisions_path.resolve()),
        "visual_decisions_sha256": _sha256(decisions_path),
        "render_report_path": str(render_report_path.resolve()),
        "render_report_sha256": _sha256(render_report_path),
        "summary": {
            "frozen_records": len(frozen),
            "approved_pages": len(pages),
            "company_groups": len(companies),
            "minimum_company_groups": minimum_company_groups,
            "role_contract_pass": True,
            "period_unit_entity_lineage_pass": True,
            "arithmetic_reexecution_pass": True,
            "visual_qa_pass": True,
            "ready_for_frozen_evaluation": True,
            "ready_for_calibration": False,
        },
    }
    return frozen, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--visual-decisions", type=Path, required=True)
    parser.add_argument("--render-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--minimum-company-groups", type=int, default=4)
    args = parser.parse_args()
    for path in (args.output, args.audit):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite versioned artifact: {path}")
    frozen, audit = freeze(
        args.candidates,
        args.visual_decisions,
        args.render_report,
        minimum_company_groups=args.minimum_company_groups,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in frozen),
        encoding="utf-8",
    )
    audit["frozen_path"] = str(args.output.resolve())
    audit["frozen_sha256"] = _sha256(args.output)
    audit["freezer_sha256"] = _sha256(Path(__file__))
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
