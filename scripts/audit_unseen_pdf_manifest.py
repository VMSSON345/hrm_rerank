#!/usr/bin/env python3
"""Verify that a PDF intake manifest is disjoint from all protected company sets."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load_builder():
    path = Path(__file__).with_name("build_pdf_company_manifest.py")
    spec = importlib.util.spec_from_file_location("build_pdf_company_manifest", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import manifest builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_builder()


def _document_hashes(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = payload.get("documents", {})
    entries = documents.values() if isinstance(documents, dict) else documents
    hashes: set[str] = set()
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("source_sha256"), str):
            hashes.add(entry["source_sha256"].lower())
    return hashes


def _find_selected(names: set[str], roots: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("*.pdf"):
            if path.name not in names:
                continue
            if path.name in found:
                raise ValueError(f"Duplicate selected PDF filename: {path.name}")
            found[path.name] = path
    missing = names - set(found)
    if missing:
        raise FileNotFoundError(f"Selected PDFs not found: {sorted(missing)}")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--existing-manifest", type=Path, action="append", default=[])
    parser.add_argument(
        "--development-metadata-jsonl", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--protected-company-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--frozen-exclusion-manifest", type=Path, action="append", default=[]
    )
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    documents = payload.get("documents")
    if not isinstance(documents, dict):
        raise ValueError("Manifest documents must be an object")
    selected_identities = {BUILDER._company_identity(name) for name in documents}
    selected_hashes = {
        str(entry["source_sha256"]).lower() for entry in documents.values()
    }

    existing_identities = set(BUILDER._existing_company_keys(args.existing_manifest))
    metadata_by_path = {
        str(path.resolve()): BUILDER.development_metadata_identities([path])
        for path in args.development_metadata_jsonl
    }
    protected_identities = BUILDER.protected_company_identities(
        args.protected_company_manifest
    )
    frozen_excluded = BUILDER.frozen_excluded_identities(
        args.frozen_exclusion_manifest
    )
    protected_hashes = set()
    for path in args.existing_manifest + args.protected_company_manifest:
        protected_hashes.update(_document_hashes(path))

    sources = _find_selected(set(documents), args.source_root)
    hash_mismatches = []
    for source_name, source in sorted(sources.items()):
        actual = BUILDER._sha256(source)
        expected = str(documents[source_name]["source_sha256"]).lower()
        if actual != expected:
            hash_mismatches.append(
                {"source_name": source_name, "expected": expected, "actual": actual}
            )

    overlaps = {
        "existing_development_companies": sorted(selected_identities & existing_identities),
        "development_metadata_companies": {
            path: sorted(selected_identities & identities)
            for path, identities in metadata_by_path.items()
        },
        "protected_test_blind_companies": sorted(
            selected_identities & protected_identities
        ),
        "frozen_excluded_companies": sorted(selected_identities & frozen_excluded),
        "protected_document_hashes": sorted(selected_hashes & protected_hashes),
        "selected_hash_duplicates": len(selected_hashes) != len(documents),
        "selected_file_hash_mismatches": hash_mismatches,
    }
    overlap_count = (
        len(overlaps["existing_development_companies"])
        + sum(len(items) for items in overlaps["development_metadata_companies"].values())
        + len(overlaps["protected_test_blind_companies"])
        + len(overlaps["frozen_excluded_companies"])
        + len(overlaps["protected_document_hashes"])
        + int(overlaps["selected_hash_duplicates"])
        + len(hash_mismatches)
    )
    result = {
        "protocol": "v12b2-unseen-pdf-overlap-audit-v1",
        "classification": "development_only",
        "manifest": str(args.manifest.resolve()),
        "metadata_only_exclusion_policy": True,
        "questions_answers_or_model_outputs_used": False,
        "metrics": {
            "documents": len(documents),
            "company_groups": len(selected_identities),
            "overlap_violations": overlap_count,
            "hash_locked_documents_verified": len(documents) - len(hash_mismatches),
        },
        "overlaps": overlaps,
    }
    if overlap_count:
        raise RuntimeError(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["metrics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
