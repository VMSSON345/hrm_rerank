#!/usr/bin/env python3
"""Audit structural/layout diversity of a hash-locked PDF intake cohort."""

from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import fitz

fitz.TOOLS.mupdf_display_errors(False)


STATEMENT_TERMS = (
    "bao cao tinh hinh tai chinh",
    "bang can doi ke toan",
)
BALANCE_TERMS = ("tai san ngan han", "no phai tra")


def _plain(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return " ".join(
        "".join(char for char in normalized if unicodedata.category(char) != "Mn").split()
    )


def _find_sources(names: set[str], roots: list[Path]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in roots:
        for path in root.rglob("*.pdf"):
            if path.name not in names:
                continue
            if path.name in found:
                raise ValueError(f"Duplicate selected PDF filename: {path.name}")
            found[path.name] = path.resolve()
    missing = names - set(found)
    if missing:
        raise FileNotFoundError(f"Selected PDFs not found: {sorted(missing)}")
    return found


def _page_image_coverage(page: fitz.Page) -> float:
    page_area = max(page.rect.width * page.rect.height, 1.0)
    largest = 0.0
    try:
        images = page.get_image_info(hashes=False, xrefs=False)
    except Exception:
        return 0.0
    for image in images:
        try:
            rect = fitz.Rect(image["bbox"])
        except Exception:
            continue
        largest = max(largest, rect.width * rect.height)
    return min(largest / page_area, 1.0)


def _family(native_ratio: float, scanned_ratio: float, landscape_ratio: float) -> str:
    if scanned_ratio >= 0.60:
        return "scanned_image_dominant"
    if native_ratio >= 0.60 and landscape_ratio >= 0.20:
        return "native_text_mixed_orientation"
    if native_ratio >= 0.60:
        return "native_text_portrait"
    return "hybrid_scan_text"


def audit_document(path: Path) -> dict[str, object]:
    document = fitz.open(path)
    page_modes: Counter[str] = Counter()
    size_profiles: Counter[str] = Counter()
    landscape_pages = 0
    text_characters: list[int] = []
    statement_hits: list[int] = []
    balance_hits: list[int] = []
    best_text_page = (1, -1)

    for index, page in enumerate(document):
        text = page.get_text("text")
        plain = _plain(text)
        characters = len(text.strip())
        text_characters.append(characters)
        if characters > best_text_page[1]:
            best_text_page = (index + 1, characters)
        image_coverage = _page_image_coverage(page)
        if characters >= 100:
            page_modes["native_text"] += 1
        elif image_coverage >= 0.70:
            page_modes["scanned_image"] += 1
        else:
            page_modes["sparse_or_vector"] += 1
        if page.rect.width > page.rect.height:
            landscape_pages += 1
        size_profiles[f"{round(page.rect.width)}x{round(page.rect.height)}"] += 1
        if any(term in plain for term in STATEMENT_TERMS):
            statement_hits.append(index + 1)
        if all(term in plain for term in BALANCE_TERMS):
            balance_hits.append(index + 1)

    pages = len(document)
    native_ratio = page_modes["native_text"] / max(pages, 1)
    scanned_ratio = page_modes["scanned_image"] / max(pages, 1)
    landscape_ratio = landscape_pages / max(pages, 1)
    representative_page = (
        balance_hits[0]
        if balance_hits
        else statement_hits[0]
        if statement_hits
        else best_text_page[0]
    )
    return {
        "pages": pages,
        "layout_family": _family(native_ratio, scanned_ratio, landscape_ratio),
        "native_text_page_ratio": round(native_ratio, 6),
        "scanned_image_page_ratio": round(scanned_ratio, 6),
        "landscape_page_ratio": round(landscape_ratio, 6),
        "median_text_characters_per_page": round(
            statistics.median(text_characters) if text_characters else 0.0, 2
        ),
        "page_modes": dict(sorted(page_modes.items())),
        "page_size_profiles": dict(sorted(size_profiles.items())),
        "statement_keyword_pages": statement_hits,
        "balance_term_pages": balance_hits,
        "representative_page": representative_page,
        "representative_page_reason": (
            "balance_terms"
            if balance_hits
            else "statement_keyword"
            if statement_hits
            else "maximum_native_text"
        ),
    }


def _render(page_path: Path, page_number: int, output: Path) -> None:
    document = fitz.open(page_path)
    page = document[page_number - 1]
    pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    documents = manifest.get("documents")
    if not isinstance(documents, dict):
        raise ValueError("Manifest documents must be an object")
    paths = _find_sources(set(documents), args.source_root)

    audit: dict[str, dict[str, object]] = {}
    family_documents: defaultdict[str, list[str]] = defaultdict(list)
    for source_name in sorted(documents):
        entry = audit_document(paths[source_name])
        entry["source_path"] = str(paths[source_name])
        audit[source_name] = entry
        family_documents[str(entry["layout_family"])].append(source_name)

    rendered: list[dict[str, object]] = []
    for family, names in sorted(family_documents.items()):
        source_name = sorted(
            names,
            key=lambda name: (
                audit[name]["representative_page_reason"] != "balance_terms",
                audit[name]["representative_page_reason"] != "statement_keyword",
                name,
            ),
        )[0]
        page_number = int(audit[source_name]["representative_page"])
        output = args.render_dir / f"{family}__{Path(source_name).stem}__p{page_number}.png"
        _render(paths[source_name], page_number, output)
        rendered.append(
            {
                "layout_family": family,
                "source_name": source_name,
                "page": page_number,
                "reason": audit[source_name]["representative_page_reason"],
                "render": str(output.resolve()),
            }
        )

    family_counts = Counter(entry["layout_family"] for entry in audit.values())
    result = {
        "protocol": "v12b2-unseen-pdf-layout-audit-v1",
        "classification": "development_only",
        "manifest": str(args.manifest.resolve()),
        "selection_used_layout_or_financial_values": False,
        "metrics": {
            "documents": len(audit),
            "company_groups": len(
                {entry["company_key"] for entry in documents.values()}
            ),
            "layout_families": len(family_counts),
            "layout_family_counts": dict(sorted(family_counts.items())),
            "documents_with_statement_keyword_page": sum(
                bool(entry["statement_keyword_pages"]) for entry in audit.values()
            ),
            "documents_with_balance_term_page": sum(
                bool(entry["balance_term_pages"]) for entry in audit.values()
            ),
        },
        "rendered_representatives": rendered,
        "documents": audit,
    }
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
