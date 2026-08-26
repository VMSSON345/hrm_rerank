from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config/finhrm_qp3_core_add_diverse_development_cohort_precommit_v1.json"
DEFAULT_OUTPUT = ROOT / "reports/finhrm_qp3_core_add_diverse_development_source_audit_v1.json"
LAYOUT_AUDITOR = ROOT / "scripts/audit_unseen_pdf_layouts.py"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_layout_auditor():
    spec = importlib.util.spec_from_file_location("layout_auditor", LAYOUT_AUDITOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PDF layout auditor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_index(roots: list[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("*.pdf"):
            if path.name in result:
                raise ValueError(f"duplicate PDF filename: {path.name}")
            result[path.name] = path.resolve()
    return result


def build_report(precommit_path: Path = DEFAULT_PRECOMMIT) -> dict[str, Any]:
    precommit = json.loads(precommit_path.read_text(encoding="utf-8"))
    catalog_parent = precommit["parents"]["development_company_catalog"]
    catalog_path = ROOT / catalog_parent["path"]
    if _digest(catalog_path) != catalog_parent["sha256"]:
        raise RuntimeError("development catalog hash changed")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("classification") != precommit["source_universe"]["classification_required"]:
        raise RuntimeError("catalog is not development-only")
    documents = catalog.get("documents")
    if not isinstance(documents, dict):
        raise RuntimeError("catalog documents must be an object")
    source_index = _source_index([ROOT / "pdf/PDF_23", ROOT / "pdf/PDF_24"])
    excluded_markers = tuple(precommit["source_universe"]["excluded_markers"])
    auditor = _load_layout_auditor()
    details: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for source_name, metadata in sorted(documents.items()):
        lowered = source_name.lower()
        marker = next((item for item in excluded_markers if item in lowered), None)
        if marker is not None:
            excluded[f"marker:{marker}"] += 1
            continue
        if not isinstance(metadata, dict):
            excluded["invalid_catalog_metadata"] += 1
            continue
        source = source_index.get(source_name)
        if source is None:
            excluded["missing_pdf"] += 1
            continue
        expected_hash = str(metadata.get("source_sha256", ""))
        if _digest(source) != expected_hash:
            excluded["source_hash_mismatch"] += 1
            continue
        layout = auditor.audit_document(source)
        details.append(
            {
                "source_key": "DEVPDF_" + expected_hash[:16],
                "source_name": source_name,
                "source_path": source.relative_to(ROOT).as_posix(),
                "source_sha256": expected_hash,
                "company_key": str(metadata.get("company_key", "")),
                **layout,
            }
        )
    families = Counter(row["layout_family"] for row in details)
    checks = {
        "catalog_hash_exact": _digest(catalog_path) == catalog_parent["sha256"],
        "catalog_counts_exact": len(documents) == precommit["source_universe"]["catalog_documents"],
        "all_admitted_sources_hash_exact": all(
            _digest(ROOT / row["source_path"]) == row["source_sha256"] for row in details
        ),
        "zero_protected_or_abw_admitted": all(
            not any(marker in row["source_name"].lower() for marker in excluded_markers)
            for row in details
        ),
        "minimum_two_layout_families_available": len(families) >= 2,
    }
    return {
        "schema": "finhrm-qp3-core-add-diverse-development-source-audit-v1",
        "created_at": "2026-08-20",
        "classification": "development_only_hash_locked_pdf_layout_discovery",
        "source_catalog": {
            "path": catalog_path.relative_to(ROOT).as_posix(),
            "sha256": _digest(catalog_path),
            "documents": len(documents),
            "company_groups": len({item["company_key"] for item in documents.values()}),
        },
        "metrics": {
            "documents_admitted": len(details),
            "company_groups_admitted": len({row["company_key"] for row in details}),
            "layout_family_counts": dict(sorted(families.items())),
            "documents_with_statement_keyword_page": sum(
                bool(row["statement_keyword_pages"]) for row in details
            ),
            "documents_with_balance_term_page": sum(
                bool(row["balance_term_pages"]) for row in details
            ),
            "excluded": dict(sorted(excluded.items())),
        },
        "gate_checks": checks,
        "details": details,
        "invariants": {
            "protected_evaluation_loaded": False,
            "abw_admitted": False,
            "runtime_changed": False,
            "model_checkpoint_or_threshold_changed": False,
            "selection_used_filename_company_page_year_value_or_question_literal": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(args.precommit)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": report["metrics"], "checks": report["gate_checks"]}, ensure_ascii=False, sort_keys=True))
    return 0 if all(report["gate_checks"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
