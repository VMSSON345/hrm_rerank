#!/usr/bin/env python3
"""Build a hash-locked, company-disjoint development PDF manifest.

Only filenames/company metadata are used to select a cohort.  Questions, answers,
financial values, and model outputs are never selection inputs.
"""

from __future__ import annotations

import argparse
import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Iterable


REPORT_TOKENS = {
    "annual",
    "audited",
    "bao",
    "bctc",
    "bctn",
    "cao",
    "chinh",
    "consolidated",
    "financial",
    "hop",
    "nhat",
    "report",
    "rieng",
    "separate",
    "statements",
    "tai",
}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _company_identity(source_name: str) -> str:
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", Path(source_name).stem.lower())
        if token
    ]
    retained = [
        token
        for token in tokens
        if token not in REPORT_TOKENS
        and not re.fullmatch(r"(?:19|20)\d{2}", token)
        and not re.fullmatch(r"\d{2}cn", token)
    ]
    if not retained:
        raise ValueError(f"Cannot derive company identity from filename: {source_name}")
    return "-".join(retained)


def _ticker_identity(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return _company_identity(f"{value.strip()}.pdf")


def _existing_company_keys(paths: list[Path]) -> dict[str, str]:
    company_keys: dict[str, str] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != "pdf-development-company-manifest-v1":
            raise ValueError(f"Unsupported company manifest protocol: {path}")
        documents = payload.get("documents")
        if not isinstance(documents, dict):
            raise ValueError(f"Company manifest documents must be an object: {path}")
        for source_name, entry in documents.items():
            if not isinstance(entry, dict):
                raise ValueError(f"Invalid company manifest entry: {source_name}")
            company_key = entry.get("company_key")
            if not isinstance(company_key, str) or not company_key.strip():
                raise ValueError(f"Missing company key: {source_name}")
            identity = _company_identity(str(source_name))
            previous = company_keys.get(identity)
            if previous is not None and previous != company_key.strip():
                raise ValueError(
                    f"Conflicting company keys for identity {identity}: "
                    f"{previous} != {company_key.strip()}"
                )
            company_keys[identity] = company_key.strip()
    return company_keys


def _identities_from_document_collection(documents: object) -> set[str]:
    identities: set[str] = set()
    if isinstance(documents, dict):
        entries: Iterable[tuple[object, object]] = documents.items()
    elif isinstance(documents, list):
        entries = ((None, entry) for entry in documents)
    else:
        raise ValueError("Protected manifest documents must be an object or list")

    for source_name, entry in entries:
        if isinstance(entry, dict):
            ticker_identity = _ticker_identity(entry.get("ticker"))
            candidate_name = entry.get("source_name") or entry.get("file") or source_name
        else:
            ticker_identity = None
            candidate_name = source_name or entry
        if ticker_identity is not None:
            identities.add(ticker_identity)
        elif isinstance(candidate_name, str) and candidate_name.strip():
            identities.add(_company_identity(candidate_name))
        else:
            raise ValueError(f"Invalid protected document entry: {entry}")
    return identities


def protected_company_identities(paths: list[Path]) -> set[str]:
    """Load identities from metadata-only split/test/blind manifests."""
    identities: set[str] = set()
    protected_roles = ("excluded", "calibration", "holdout_v2")
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") == "financial-pdf-corpus-split-v1":
            for role in protected_roles:
                entries = payload.get(role, [])
                if not isinstance(entries, list):
                    raise ValueError(f"Protected split role must be a list: {role}")
                for entry in entries:
                    if not isinstance(entry, dict) or not entry.get("file"):
                        raise ValueError(f"Invalid protected split entry in {role}: {entry}")
                    identities.add(_company_identity(str(entry["file"])))
            continue

        if "documents" not in payload:
            raise ValueError(f"Unsupported protected company manifest: {path}")
        identities.update(_identities_from_document_collection(payload["documents"]))
        excluded_tickers = payload.get("excluded_tickers", [])
        if not isinstance(excluded_tickers, list):
            raise ValueError(f"excluded_tickers must be a list: {path}")
        for ticker in excluded_tickers:
            identity = _ticker_identity(ticker)
            if identity is None:
                raise ValueError(f"Invalid excluded ticker in {path}: {ticker}")
            identities.add(identity)
    return identities


def development_metadata_identities(paths: list[Path]) -> set[str]:
    """Read only source-name/company grouping metadata from development JSONL."""
    identities: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                candidates = row.get("candidates", [])
                if not isinstance(candidates, list):
                    raise ValueError(f"candidates must be a list: {path}:{line_number}")
                source_names = {
                    candidate.get("source_name")
                    for candidate in candidates
                    if isinstance(candidate, dict) and candidate.get("source_name")
                }
                for source_name in source_names:
                    identities.add(_company_identity(str(source_name)))
    return identities


def frozen_excluded_identities(paths: list[Path]) -> set[str]:
    """Reuse an already audited exclusion set without reopening protected data."""

    identities: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != "pdf-development-company-manifest-v1":
            raise ValueError(f"Unsupported frozen exclusion manifest: {path}")
        values = payload.get("excluded_company_identities")
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value.strip() for value in values
        ):
            raise ValueError(f"Frozen exclusion identities are missing: {path}")
        identities.update(value.strip().lower() for value in values)
    return identities


def build_manifest(
    sources: list[Path],
    *,
    existing_manifests: list[Path],
    excluded_names: set[str],
    excluded_company_identities: set[str] | None = None,
    new_companies_only: bool = False,
    company_group_offset: int = 0,
    max_company_groups: int | None = None,
) -> dict[str, object]:
    if company_group_offset < 0:
        raise ValueError("company_group_offset must be non-negative")
    if max_company_groups is not None and max_company_groups < 1:
        raise ValueError("max_company_groups must be positive")
    excluded_identities = {
        identity.strip().lower()
        for identity in (excluded_company_identities or set())
        if identity.strip()
    }
    by_name: dict[str, Path] = {}
    for source in sources:
        resolved = source.expanduser().resolve()
        if resolved.suffix.lower() != ".pdf":
            continue
        if resolved.name in by_name:
            raise ValueError(f"Duplicate PDF filename: {resolved.name}")
        by_name[resolved.name] = resolved
    unknown_exclusions = excluded_names - set(by_name)
    if unknown_exclusions:
        raise ValueError(f"Excluded documents not found: {sorted(unknown_exclusions)}")

    existing_keys = _existing_company_keys(existing_manifests)
    by_identity: dict[str, list[tuple[str, Path]]] = {}
    documents_excluded_by_company = 0
    documents_excluded_as_existing = 0
    for source_name, source in sorted(by_name.items()):
        if source_name in excluded_names:
            continue
        identity = _company_identity(source_name)
        if identity in excluded_identities:
            documents_excluded_by_company += 1
            continue
        if new_companies_only and identity in existing_keys:
            documents_excluded_as_existing += 1
            continue
        by_identity.setdefault(identity, []).append((source_name, source))

    ordered_identities = sorted(
        by_identity,
        key=lambda identity: (sha256(identity.encode("utf-8")).hexdigest(), identity),
    )
    eligible_company_groups = len(ordered_identities)
    ordered_identities = ordered_identities[company_group_offset:]
    if max_company_groups is not None:
        ordered_identities = ordered_identities[:max_company_groups]
    selected_identities = set(ordered_identities)

    documents: dict[str, dict[str, object]] = {}
    reused_groups: set[str] = set()
    new_groups: set[str] = set()
    for identity in ordered_identities:
        company_key = existing_keys.get(identity)
        if company_key is None:
            company_key = f"development-company-{sha256(identity.encode('utf-8')).hexdigest()[:16]}"
            new_groups.add(company_key)
            identity_source = "deterministic_filename_group"
        else:
            reused_groups.add(company_key)
            identity_source = "existing_development_manifest"
        for source_name, source in by_identity[identity]:
            documents[source_name] = {
                "company_key": company_key,
                "source_sha256": _sha256(source),
                "source_bytes": source.stat().st_size,
                "identity_source": identity_source,
            }

    return {
        "protocol": "pdf-development-company-manifest-v1",
        "classification": "development_only_unseen_company_intake",
        "identity_policy": (
            "Cross-year filename identity is used only to construct opaque "
            "company-disjoint supervision groups and is never a model feature."
        ),
        "existing_manifests": [
            {"path": str(path), "sha256": _sha256(path)} for path in existing_manifests
        ],
        "excluded_documents": sorted(excluded_names),
        "excluded_company_identities": sorted(excluded_identities),
        "selection_policy": {
            "new_companies_only": new_companies_only,
            "company_group_offset": company_group_offset,
            "max_company_groups": max_company_groups,
            "ordering": "sha256(company_identity), then company_identity",
            "model_outputs_used": False,
            "financial_values_used": False,
            "questions_or_answers_used": False,
        },
        "metrics": {
            "documents_discovered": len(by_name),
            "documents_included": len(documents),
            "documents_excluded": (
                len(excluded_names)
                + documents_excluded_by_company
                + documents_excluded_as_existing
                + sum(
                    len(by_identity[identity])
                    for identity in set(by_identity) - selected_identities
                )
            ),
            "documents_excluded_by_company": documents_excluded_by_company,
            "documents_excluded_as_existing": documents_excluded_as_existing,
            "eligible_company_groups": eligible_company_groups,
            "company_groups": len({entry["company_key"] for entry in documents.values()}),
            "reused_company_groups": len(reused_groups),
            "new_company_groups": len(new_groups),
        },
        "documents": documents,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a hash-locked development PDF company manifest"
    )
    parser.add_argument("documents", nargs="+", type=Path)
    parser.add_argument("--existing-manifest", action="append", type=Path, default=[])
    parser.add_argument("--exclude-document", action="append", default=[])
    parser.add_argument(
        "--protected-split-manifest", action="append", type=Path, default=[]
    )
    parser.add_argument(
        "--protected-company-manifest", action="append", type=Path, default=[]
    )
    parser.add_argument(
        "--development-metadata-jsonl", action="append", type=Path, default=[]
    )
    parser.add_argument(
        "--frozen-exclusion-manifest", action="append", type=Path, default=[]
    )
    parser.add_argument("--exclude-company", action="append", default=[])
    parser.add_argument("--new-companies-only", action="store_true")
    parser.add_argument("--company-group-offset", type=int, default=0)
    parser.add_argument("--max-company-groups", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = sorted(
        {
            path
            for requested in args.documents
            for path in (requested.rglob("*.pdf") if requested.is_dir() else [requested])
        }
    )
    manifest_paths = [path.expanduser().resolve() for path in args.existing_manifest]
    protected_paths = [
        path.expanduser().resolve()
        for path in args.protected_split_manifest + args.protected_company_manifest
    ]
    metadata_paths = [
        path.expanduser().resolve() for path in args.development_metadata_jsonl
    ]
    excluded_identities = protected_company_identities(protected_paths)
    excluded_identities.update(development_metadata_identities(metadata_paths))
    frozen_exclusion_paths = [
        path.expanduser().resolve() for path in args.frozen_exclusion_manifest
    ]
    excluded_identities.update(frozen_excluded_identities(frozen_exclusion_paths))
    excluded_identities.update(
        _company_identity(f"{company}.pdf") for company in args.exclude_company
    )
    result = build_manifest(
        sources,
        existing_manifests=manifest_paths,
        excluded_names={Path(name).name for name in args.exclude_document},
        excluded_company_identities=excluded_identities,
        new_companies_only=args.new_companies_only,
        company_group_offset=args.company_group_offset,
        max_company_groups=args.max_company_groups,
    )
    result["protected_company_manifests"] = [
        {"path": str(path), "sha256": _sha256(path)} for path in protected_paths
    ]
    result["development_metadata_jsonl"] = [
        {
            "path": str(path),
            "sha256": _sha256(path),
            "fields_used": ["candidates.source_name"],
        }
        for path in metadata_paths
    ]
    result["frozen_exclusion_manifests"] = [
        {"path": str(path), "sha256": _sha256(path)}
        for path in frozen_exclusion_paths
    ]
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **result["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
