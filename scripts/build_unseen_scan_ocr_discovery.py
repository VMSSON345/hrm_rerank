#!/usr/bin/env python3
"""Build a bounded, structure-aware OCR queue for unseen scan PDFs."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import cv2
import fitz
import numpy as np


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _uniform_sample(values: list[int], count: int) -> list[int]:
    ordered = sorted(set(values))
    if count <= 0 or not ordered:
        return []
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indexes = {
        round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)
    }
    return [ordered[index] for index in sorted(indexes)]


def select_pages(
    total_pages: int,
    table_scores: dict[int, float],
    *,
    first_pages: int,
    uniform_pages: int,
    table_pages: int,
    max_pages: int,
) -> list[dict[str, object]]:
    if total_pages < 1 or min(first_pages, uniform_pages, table_pages) < 0:
        raise ValueError("Invalid page selection limits")
    if max_pages < 1:
        raise ValueError("max_pages must be positive")
    all_pages = list(range(1, total_pages + 1))
    reasons: dict[int, set[str]] = {}

    def add(pages: list[int], reason: str) -> None:
        for page in pages:
            reasons.setdefault(page, set()).add(reason)

    add(all_pages[:first_pages], "first_pages")
    add(_uniform_sample(all_pages, uniform_pages), "uniform_document_coverage")
    ranked_tables = sorted(
        table_scores,
        key=lambda page: (-table_scores[page], page),
    )[:table_pages]
    add(ranked_tables, "table_structure_score")
    ranked = sorted(
        reasons,
        key=lambda page: (
            "table_structure_score" not in reasons[page],
            -table_scores.get(page, 0.0),
            page,
        ),
    )[:max_pages]
    return [
        {
            "page": page,
            "selection": "discovery",
            "reasons": sorted(reasons[page]),
            "table_structure_score": round(table_scores.get(page, 0.0), 8),
        }
        for page in sorted(ranked)
    ]


def _table_structure_score(page: fitz.Page) -> float:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(0.65, 0.65), colorspace=fitz.csGRAY)
    image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width
    )
    binary = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(12, pixmap.width // 24), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(12, pixmap.height // 24))
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    line_pixels = np.count_nonzero(cv2.bitwise_or(horizontal, vertical))
    intersections = np.count_nonzero(cv2.bitwise_and(horizontal, vertical))
    area = max(pixmap.width * pixmap.height, 1)
    return line_pixels / area + 4.0 * intersections / area


def _find_sources(names: set[str], roots: list[Path]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("*.pdf"):
            if path.name not in names:
                continue
            if path.name in sources:
                raise ValueError(f"Duplicate selected PDF filename: {path.name}")
            sources[path.name] = path.resolve()
    missing = names - set(sources)
    if missing:
        raise FileNotFoundError(f"Selected PDFs not found: {sorted(missing)}")
    return sources


def build_manifest(
    company_manifest: dict[str, object],
    layout_audit: dict[str, object],
    sources: dict[str, Path],
    *,
    first_pages: int,
    uniform_pages: int,
    table_pages: int,
    max_pages: int,
) -> dict[str, object]:
    company_documents = company_manifest.get("documents")
    layout_documents = layout_audit.get("documents")
    if not isinstance(company_documents, dict) or not isinstance(layout_documents, dict):
        raise ValueError("Invalid company manifest or layout audit")
    scan_names = sorted(
        name
        for name, item in layout_documents.items()
        if isinstance(item, dict)
        and item.get("layout_family") == "scanned_image_dominant"
    )
    if set(scan_names) - set(company_documents):
        raise ValueError("Layout audit contains documents outside the company manifest")

    documents: list[dict[str, object]] = []
    for name in scan_names:
        metadata = company_documents[name]
        layout = layout_documents[name]
        if not isinstance(metadata, dict) or not isinstance(layout, dict):
            raise ValueError(f"Invalid document metadata: {name}")
        source = sources[name]
        expected = str(metadata.get("source_sha256", "")).lower()
        actual = _sha256(source)
        if not expected or expected != actual:
            raise RuntimeError(f"PDF checksum mismatch: {name}")
        document = fitz.open(source)
        expected_pages = int(layout.get("pages", 0))
        if expected_pages != len(document):
            raise RuntimeError(f"PDF page count changed: {name}")
        scores = {
            page_number: _table_structure_score(page)
            for page_number, page in enumerate(document, start=1)
        }
        pages = select_pages(
            len(document),
            scores,
            first_pages=first_pages,
            uniform_pages=uniform_pages,
            table_pages=table_pages,
            max_pages=max_pages,
        )
        documents.append(
            {
                "document": str(source),
                "source_name": name,
                "sha256": actual,
                "company_key": str(metadata.get("company_key", "")),
                "total_pages": len(document),
                "layout_family": "scanned_image_dominant",
                "pages": pages,
            }
        )
    return {
        "protocol": "v12b2-unseen-scan-ocr-discovery-v1",
        "classification": "development_only",
        "ocr_performed": False,
        "model_invoked": False,
        "selection_policy": {
            "first_pages": first_pages,
            "uniform_pages": uniform_pages,
            "table_structure_pages": table_pages,
            "maximum_pages_per_document": max_pages,
            "table_score_uses_financial_values": False,
            "test_or_blind_features_used": False,
        },
        "metrics": {
            "documents_queued": len(documents),
            "company_groups_queued": len(
                {str(item["company_key"]) for item in documents}
            ),
            "pages_queued": sum(len(item["pages"]) for item in documents),
        },
        "documents": documents,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company-manifest", type=Path, required=True)
    parser.add_argument("--layout-audit", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-pages", type=int, default=2)
    parser.add_argument("--uniform-pages", type=int, default=6)
    parser.add_argument("--table-pages", type=int, default=4)
    parser.add_argument("--max-pages", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    company_manifest = json.loads(args.company_manifest.read_text(encoding="utf-8"))
    layout_audit = json.loads(args.layout_audit.read_text(encoding="utf-8"))
    layout_documents = layout_audit.get("documents", {})
    scan_names = {
        name
        for name, item in layout_documents.items()
        if isinstance(item, dict)
        and item.get("layout_family") == "scanned_image_dominant"
    }
    sources = _find_sources(scan_names, args.source_root)
    result = build_manifest(
        company_manifest,
        layout_audit,
        sources,
        first_pages=args.first_pages,
        uniform_pages=args.uniform_pages,
        table_pages=args.table_pages,
        max_pages=args.max_pages,
    )
    result["inputs"] = {
        "company_manifest": str(args.company_manifest.resolve()),
        "company_manifest_sha256": _sha256(args.company_manifest),
        "layout_audit": str(args.layout_audit.resolve()),
        "layout_audit_sha256": _sha256(args.layout_audit),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["metrics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
