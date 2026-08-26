#!/usr/bin/env python3
"""Discover fresh development-only V12-C joint calibration candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

import fitz

from hrm_finance_qa.ontology import normalize_text
from scripts.validate_v12c_joint_calibration_protocol import validate
from scripts.validate_v12c_joint_calibration_source_extension import (
    validate as validate_source_extension,
)


PROTECTED = re.compile(r"(?:test|holdout|final|blind|validation)", re.IGNORECASE)
BORROWING_PHRASES = {
    "short_term_borrowings": (
        "vay va no thue tai chinh ngan han",
        "no vay ngan han",
        "vay ngan han",
    ),
    "long_term_borrowings": (
        "vay va no thue tai chinh dai han",
        "no vay dai han",
        "vay dai han",
    ),
    "total_capital": (
        "tong cong nguon von",
        "tong nguon von",
    ),
}
BROAD_LIABILITY_PHRASES = {
    "current_liabilities": ("no ngan han",),
    "long_term_liabilities": ("no dai han",),
    "total_liabilities": ("tong no phai tra", "tong cong no phai tra", "no phai tra"),
}
CONFOUND_TOTALS = {
    "total_assets": ("tong tai san", "tong cong tai san"),
    "total_liabilities": BROAD_LIABILITY_PHRASES["total_liabilities"],
    "owners_equity": ("von chu so huu",),
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


def _phrase_count(normalized: str, phrases: tuple[str, ...]) -> int:
    spans: set[tuple[int, int]] = set()
    padded = f" {normalized} "
    for phrase in phrases:
        for match in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", padded):
            spans.add(match.span())
    return len(spans)


def page_signals(text: str) -> dict[str, dict[str, int]]:
    normalized = normalize_text(text)
    return {
        "required_roles": {
            role: _phrase_count(normalized, phrases)
            for role, phrases in BORROWING_PHRASES.items()
        },
        "broad_liabilities": {
            role: _phrase_count(normalized, phrases)
            for role, phrases in BROAD_LIABILITY_PHRASES.items()
        },
        "confound_totals": {
            role: _phrase_count(normalized, phrases)
            for role, phrases in CONFOUND_TOTALS.items()
        },
    }


def classify_signals(signals: dict[str, dict[str, int]]) -> tuple[str, str] | None:
    required = signals["required_roles"]
    present = {role for role, count in required.items() if count > 0}
    if len(present) == 3:
        if any(count > 1 for count in required.values()):
            return "negative_candidate", "ambiguous_duplicate_role"
        return "positive_candidate", "complete_required_role_triplet"
    if len(present) == 2:
        return "negative_candidate", "missing_required_raw_role"

    broad = signals["broad_liabilities"]
    totals = signals["confound_totals"]
    broad_pair = broad["current_liabilities"] > 0 and broad["long_term_liabilities"] > 0
    denominator_or_confound = required["total_capital"] > 0 or any(totals.values())
    if broad_pair and denominator_or_confound:
        return "negative_candidate", "invalid_borrowing_role_semantics"
    return None


def discover(
    protocol_path: Path,
    root: Path,
    pdf_root: Path,
    source_extension_path: Path | None = None,
) -> dict[str, object]:
    for path in (protocol_path, root, pdf_root):
        _refuse(path)
    precommit = validate(protocol_path, root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source_extension = None
    if source_extension_path is None:
        catalog_entry = protocol["locked_inputs"]["development_catalog"]
    else:
        _refuse(source_extension_path)
        source_extension = validate_source_extension(source_extension_path, root)
        extension = json.loads(source_extension_path.read_text(encoding="utf-8"))
        if (root / extension["base_protocol"]["path"]).resolve() != protocol_path.resolve():
            raise ValueError("Source extension does not belong to this base protocol")
        catalog_entry = extension["source_catalog"]
    catalog_path = root / catalog_entry["path"]
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not str(catalog.get("classification", "")).startswith("development_only"):
        raise ValueError("Catalog is not development-only")
    documents = catalog.get("documents")
    if not isinstance(documents, dict):
        raise ValueError("Catalog documents must be an object")
    excluded_tickers = set(protocol["exclusion_policy"]["excluded_tickers"])
    excluded_companies = set(protocol["exclusion_policy"]["excluded_company_keys"])

    indexed_paths: dict[str, list[Path]] = {}
    for path in pdf_root.rglob("*.pdf"):
        _refuse(path)
        indexed_paths.setdefault(path.name, []).append(path.resolve())

    counts = {
        "catalog_documents": len(documents),
        "excluded_before_opening": 0,
        "missing_or_ambiguous_pdf": 0,
        "hash_mismatch": 0,
        "documents_scanned": 0,
        "pages_scanned": 0,
    }
    records: list[dict[str, object]] = []
    for source_name, metadata in sorted(documents.items()):
        if not isinstance(metadata, dict):
            continue
        company_key = str(metadata.get("company_key", ""))
        if _ticker(source_name) in excluded_tickers or company_key in excluded_companies:
            counts["excluded_before_opening"] += 1
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
                signals = page_signals(text)
                classification = classify_signals(signals)
                if classification is None:
                    continue
                stratum, failure_or_proof = classification
                records.append({
                    "source_name": source_name,
                    "source_path": str(source_path),
                    "source_sha256": expected_hash,
                    "company_key": company_key,
                    "page": page_index + 1,
                    "page_text_sha256": sha256(text.encode("utf-8")).hexdigest(),
                    "candidate_stratum": stratum,
                    "proof_or_failure_type": failure_or_proof,
                    "signals": signals,
                    "excerpt": " ".join(text.split())[:1600],
                })
        finally:
            document.close()

    positive = [row for row in records if row["candidate_stratum"] == "positive_candidate"]
    negative = [row for row in records if row["candidate_stratum"] == "negative_candidate"]
    positive_companies = {str(row["company_key"]) for row in positive}
    negative_companies = {str(row["company_key"]) for row in negative}
    failure_counts = Counter(str(row["proof_or_failure_type"]) for row in negative)
    contract = protocol["cohort_contract"]
    positive_feasible = (
        len(positive) >= contract["positive"]["minimum_controls"]
        and len(positive_companies) >= contract["positive"]["minimum_company_groups"]
    )
    negative_feasible = (
        len(negative) >= contract["negative"]["minimum_controls"]
        and len(negative_companies) >= contract["negative"]["minimum_company_groups"]
        and len(failure_counts) >= contract["negative"]["minimum_failure_strata"]
    )
    cross_stratum_disjoint_feasible = bool(
        positive_companies - negative_companies
    ) and bool(negative_companies - positive_companies)
    return {
        "protocol": "v12c-joint-calibration-feasibility-discovery-v1",
        "classification": "development_only_joint_calibration_discovery",
        "protected_evaluation_sources_used": False,
        "model_invoked": False,
        "threshold_changed": False,
        "alternative_thresholds_evaluated": False,
        "precommit": precommit,
        "source_extension": source_extension,
        "source_extension_path": (
            str(source_extension_path.resolve()) if source_extension_path else None
        ),
        "source_extension_sha256": (
            _sha256(source_extension_path) if source_extension_path else None
        ),
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": _sha256(protocol_path),
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": _sha256(catalog_path),
        "counts": {
            **counts,
            "positive_candidate_pages": len(positive),
            "positive_candidate_company_groups": len(positive_companies),
            "negative_candidate_pages": len(negative),
            "negative_candidate_company_groups": len(negative_companies),
            "negative_failure_type_counts": dict(sorted(failure_counts.items())),
        },
        "records": records,
        "feasibility": {
            "positive_minimum_met_before_visual_qa": positive_feasible,
            "negative_minimum_met_before_visual_qa": negative_feasible,
            "cross_stratum_company_disjoint_assignment_possible": cross_stratum_disjoint_feasible,
            "joint_protocol_feasible_before_visual_qa": (
                positive_feasible and negative_feasible and cross_stratum_disjoint_feasible
            ),
            "stop_rule_triggered": not (
                positive_feasible and negative_feasible and cross_stratum_disjoint_feasible
            ),
            "ready_for_visual_qa": (
                positive_feasible and negative_feasible and cross_stratum_disjoint_feasible
            ),
            "ready_for_inference": False,
            "ready_for_threshold_change": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--pdf-root", type=Path, required=True)
    parser.add_argument("--source-extension", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {args.output}")
    _refuse(args.output)
    report = discover(
        args.protocol,
        args.root,
        args.pdf_root,
        source_extension_path=args.source_extension,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "feasibility": report["feasibility"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
