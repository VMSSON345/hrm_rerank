#!/usr/bin/env python3
"""Discover development-only V12-C near-miss pages for negative controls."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

import fitz

from hrm_finance_qa.ontology import normalize_text


PROTECTED = re.compile(r"(?:test|holdout|final|blind|validation)", re.IGNORECASE)
ROLE_PHRASES = {
    "short_term_borrowings": (
        "vay ngan han",
        "vay va no thue tai chinh ngan han",
        "no vay ngan han",
    ),
    "long_term_borrowings": (
        "vay dai han",
        "vay va no thue tai chinh dai han",
        "no vay dai han",
    ),
    "total_capital": (
        "tong nguon von",
        "tong cong nguon von",
    ),
}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _refuse_protected(path: Path) -> None:
    if PROTECTED.search(str(path.resolve()).replace("\\", "/")):
        raise ValueError(f"Protected path is not allowed: {path}")


def _ticker(source_name: str) -> str:
    return Path(source_name).stem.split("_", 1)[0].upper()


def _cohort_exclusions(paths: list[Path]) -> tuple[set[str], set[str]]:
    tickers: set[str] = set()
    company_keys: set[str] = set()
    for path in paths:
        _refuse_protected(path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not str(row.get("classification", "")).startswith(
                    "development_only"
                ):
                    raise ValueError(f"Non-development exclusion row: {path}:{line_number}")
                ticker = _ticker(str(row.get("report_id", "")))
                company_key = str(row.get("company_key", ""))
                if ticker:
                    tickers.add(ticker)
                if company_key:
                    company_keys.add(company_key)
    if not tickers:
        raise ValueError("Exclusion cohorts are empty")
    return tickers, company_keys


def role_mentions(text: str) -> dict[str, int]:
    normalized = f" {normalize_text(text)} "
    mentions: dict[str, int] = {}
    for role, phrases in ROLE_PHRASES.items():
        matches: set[tuple[int, int]] = set()
        for phrase in phrases:
            for match in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized):
                matches.add(match.span())
        mentions[role] = len(matches)
    return mentions


def near_miss_type(mentions: dict[str, int]) -> str | None:
    present = {role for role, count in mentions.items() if count > 0}
    if len(present) < 2:
        return None
    if len(present) == 2:
        missing = next(role for role in ROLE_PHRASES if role not in present)
        return f"missing_{missing}"
    if any(count > 1 for count in mentions.values()):
        return "complete_lexical_with_duplicate_role_mentions"
    return "complete_lexical_requires_semantic_review"


def discover(
    catalog_path: Path,
    exclusion_cohorts: list[Path],
    pdf_root: Path,
) -> dict[str, object]:
    for path in (catalog_path, pdf_root, *exclusion_cohorts):
        _refuse_protected(path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("classification") != "development_only":
        raise ValueError("Catalog is not development-only")
    documents = catalog.get("documents")
    if not isinstance(documents, dict):
        raise ValueError("Catalog documents must be an object")
    excluded_tickers, excluded_company_keys = _cohort_exclusions(exclusion_cohorts)
    indexed_paths: dict[str, list[Path]] = {}
    for path in pdf_root.rglob("*.pdf"):
        _refuse_protected(path)
        indexed_paths.setdefault(path.name, []).append(path.resolve())

    records: list[dict[str, object]] = []
    counts = {
        "catalog_documents": len(documents),
        "excluded_documents": 0,
        "missing_or_ambiguous_pdf": 0,
        "hash_mismatch": 0,
        "documents_scanned": 0,
        "pages_scanned": 0,
    }
    for source_name, metadata in sorted(documents.items()):
        if not isinstance(metadata, dict):
            continue
        company_key = str(metadata.get("company_key", ""))
        if _ticker(source_name) in excluded_tickers or company_key in excluded_company_keys:
            counts["excluded_documents"] += 1
            continue
        paths = indexed_paths.get(source_name, [])
        if len(paths) != 1:
            counts["missing_or_ambiguous_pdf"] += 1
            continue
        source_path = paths[0]
        expected_hash = str(metadata.get("source_sha256", ""))
        if not expected_hash or _sha256(source_path) != expected_hash:
            counts["hash_mismatch"] += 1
            continue
        counts["documents_scanned"] += 1
        document = fitz.open(source_path)
        try:
            for page_index in range(document.page_count):
                counts["pages_scanned"] += 1
                text = document[page_index].get_text("text", sort=True)
                mentions = role_mentions(text)
                candidate_type = near_miss_type(mentions)
                if candidate_type is None:
                    continue
                normalized = normalize_text(text)
                records.append({
                    "source_name": source_name,
                    "source_path": str(source_path),
                    "source_sha256": expected_hash,
                    "company_key": company_key,
                    "page": page_index + 1,
                    "page_text_sha256": sha256(text.encode("utf-8")).hexdigest(),
                    "candidate_type": candidate_type,
                    "role_mentions": mentions,
                    "has_balance_sheet_marker": any(
                        marker in normalized
                        for marker in ("bang can doi ke toan", "bao cao tinh hinh tai chinh")
                    ),
                    "has_explicit_unit": any(
                        marker in normalized for marker in ("don vi", "vnd", "dong")
                    ),
                    "has_period_marker": any(
                        marker in normalized
                        for marker in ("31 12", "01 01", "so cuoi nam", "so dau nam")
                    ),
                    "excerpt": " ".join(text.split())[:1600],
                })
        finally:
            document.close()
    groups = {str(row["company_key"]) for row in records}
    type_counts: dict[str, int] = {}
    for row in records:
        key = str(row["candidate_type"])
        type_counts[key] = type_counts.get(key, 0) + 1
    return {
        "protocol": "v12c-negative-ambiguity-page-discovery-v1",
        "classification": "development_only_negative_control_discovery",
        "protected_evaluation_sources_used": False,
        "model_invoked": False,
        "threshold_changed": False,
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": _sha256(catalog_path),
        "exclusion_cohorts": [
            {"path": str(path.resolve()), "sha256": _sha256(path)}
            for path in exclusion_cohorts
        ],
        "excluded_tickers": sorted(excluded_tickers),
        "excluded_company_keys": sorted(excluded_company_keys),
        "counts": {
            **counts,
            "candidate_pages": len(records),
            "candidate_company_groups": len(groups),
            "candidate_type_counts": dict(sorted(type_counts.items())),
        },
        "records": records,
        "gate": {
            "independent_from_repair_and_positive_cohorts": True,
            "ready_for_visual_qa": bool(records),
            "ready_for_threshold_change": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--exclude-cohort", type=Path, action="append", required=True)
    parser.add_argument("--pdf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite versioned artifact: {args.output}")
    _refuse_protected(args.output)
    report = discover(args.catalog, args.exclude_cohort, args.pdf_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
