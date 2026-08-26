#!/usr/bin/env python3
"""Shortlist independent development PDF pages for V12-C calibration QA."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

import fitz

from hrm_finance_qa.ontology import normalize_text


ROOT = Path(__file__).resolve().parents[1]
PROTECTED = re.compile(r"(?:test|holdout|final|blind|validation)", re.IGNORECASE)


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


def excluded_tickers(cohort: Path) -> set[str]:
    _refuse_protected(cohort)
    output: set[str] = set()
    with cohort.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("classification") != "development_only_visual_qa_approved":
                raise ValueError("Exclusion cohort is not frozen development-only")
            output.add(_ticker(str(record.get("report_id", ""))))
    if not output:
        raise ValueError("Exclusion cohort is empty")
    return output


def _role_hits(text: str) -> dict[str, bool]:
    normalized = f" {normalize_text(text)} "
    return {
        "short_term_borrowings": any(
            phrase in normalized
            for phrase in (
                " vay ngan han ",
                " vay va no thue tai chinh ngan han ",
                " no vay ngan han ",
            )
        ),
        "long_term_borrowings": any(
            phrase in normalized
            for phrase in (
                " vay dai han ",
                " vay va no thue tai chinh dai han ",
                " no vay dai han ",
            )
        ),
        "total_capital": any(
            phrase in normalized
            for phrase in (
                " tong nguon von ",
                " tong cong nguon von ",
                " tong cong nguon von ",
            )
        ),
    }


def discover(
    catalog_path: Path,
    exclusion_cohort: Path,
    pdf_root: Path,
) -> dict[str, object]:
    for path in (catalog_path, exclusion_cohort, pdf_root):
        _refuse_protected(path)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("classification") != "development_only":
        raise ValueError("Catalog is not development-only")
    documents = catalog.get("documents")
    if not isinstance(documents, dict):
        raise ValueError("Catalog documents must be an object")
    excluded = excluded_tickers(exclusion_cohort)
    indexed_paths: dict[str, list[Path]] = {}
    for path in pdf_root.rglob("*.pdf"):
        _refuse_protected(path)
        indexed_paths.setdefault(path.name, []).append(path.resolve())

    records: list[dict[str, object]] = []
    counts = {
        "catalog_documents": len(documents),
        "excluded_same_company_documents": 0,
        "missing_or_ambiguous_pdf": 0,
        "hash_mismatch": 0,
        "documents_scanned": 0,
        "pages_scanned": 0,
    }
    for source_name, metadata in sorted(documents.items()):
        if _ticker(source_name) in excluded:
            counts["excluded_same_company_documents"] += 1
            continue
        if not isinstance(metadata, dict):
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
                page = document[page_index]
                text = page.get_text("text", sort=True)
                hits = _role_hits(text)
                if not all(hits.values()):
                    continue
                normalized = normalize_text(text)
                records.append(
                    {
                        "source_name": source_name,
                        "source_path": str(source_path),
                        "source_sha256": expected_hash,
                        "company_key": str(metadata.get("company_key", "")),
                        "page": page_index + 1,
                        "page_text_sha256": sha256(text.encode("utf-8")).hexdigest(),
                        "role_hits": hits,
                        "has_balance_sheet_marker": any(
                            marker in normalized
                            for marker in (
                                "bang can doi ke toan",
                                "bao cao tinh hinh tai chinh",
                            )
                        ),
                        "has_explicit_unit": any(
                            marker in normalized
                            for marker in ("don vi", "vnd", "dong")
                        ),
                        "has_period_marker": any(
                            marker in normalized
                            for marker in (
                                "31 12",
                                "so cuoi nam",
                                "so dau nam",
                                "nam nay",
                                "nam truoc",
                            )
                        ),
                        "excerpt": " ".join(text.split())[:1200],
                    }
                )
        finally:
            document.close()
    groups = {str(row["company_key"]) for row in records}
    return {
        "protocol": "v12c-independent-calibration-page-discovery-v1",
        "classification": "development_only_discovery",
        "protected_evaluation_sources_used": False,
        "model_invoked": False,
        "threshold_changed": False,
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": _sha256(catalog_path),
        "exclusion_cohort_path": str(exclusion_cohort.resolve()),
        "exclusion_cohort_sha256": _sha256(exclusion_cohort),
        "excluded_tickers": sorted(excluded),
        "counts": {
            **counts,
            "candidate_pages": len(records),
            "candidate_company_groups": len(groups),
        },
        "records": records,
        "gate": {
            "independent_company_groups_found": len(groups),
            "ready_for_visual_qa": bool(records),
            "ready_for_calibration": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--exclude-cohort", type=Path, required=True)
    parser.add_argument("--pdf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite versioned artifact: {args.output}")
    _refuse_protected(args.output)
    report = discover(args.catalog, args.exclude_cohort, args.pdf_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
