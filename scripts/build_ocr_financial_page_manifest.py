#!/usr/bin/env python3
"""Score bounded MinerU discovery pages before supervision generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_financial_page_manifest import score_financial_page


PAGE_RE = re.compile(r"^## Page (\d+)\s*$")


def _read_pages(path: Path) -> dict[int, str]:
    pages: dict[int, list[str]] = {}
    current: int | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = PAGE_RE.match(line.strip())
        if match:
            current = int(match.group(1))
            if current in pages:
                raise ValueError(f"Duplicate page heading in {path}: {current}")
            pages[current] = []
        elif current is not None:
            pages[current].append(line)
    return {page: "\n".join(lines).strip() for page, lines in pages.items()}


def build_manifest(
    discovery_manifest: dict[str, object],
    markdown_dir: Path,
    *,
    document_names: set[str] | None = None,
    required_reason: str | None = None,
) -> dict[str, object]:
    raw_documents = discovery_manifest.get("documents")
    if not isinstance(raw_documents, list):
        raise ValueError("Discovery manifest documents must be a list")
    documents: list[dict[str, object]] = []
    for item in raw_documents:
        if not isinstance(item, dict):
            raise ValueError("Invalid discovery document")
        source = Path(str(item.get("document", "")))
        if document_names and source.name not in document_names:
            continue
        markdown_path = markdown_dir / f"{source.stem}.md"
        if not markdown_path.is_file():
            raise FileNotFoundError(f"Missing OCR Markdown: {markdown_path}")
        requested = {
            int(page["page"])
            for page in item.get("pages", [])
            if isinstance(page, dict)
            and page.get("selection") == "discovery"
            and (
                required_reason is None
                or (
                    isinstance(page.get("reasons"), list)
                    and required_reason in page["reasons"]
                )
            )
        }
        extracted = _read_pages(markdown_path)
        if set(extracted) != requested:
            raise ValueError(
                f"OCR page coverage mismatch for {source.name}: "
                f"requested={sorted(requested)}, extracted={sorted(extracted)}"
            )
        scored: dict[int, dict[str, object]] = {
            page: {
                "page": page,
                "extraction_method": "mineru",
                **score_financial_page(text),
            }
            for page, text in extracted.items()
        }
        core_pages = {
            page
            for page, result in scored.items()
            if result["page_class"] == "financial_summary"
        }
        note_pages = {
            page
            for page, result in scored.items()
            if result["page_class"] == "financial_note"
        }
        selected_pages = core_pages | note_pages
        for page in core_pages | note_pages:
            selected_pages.update(
                neighbor for neighbor in (page - 1, page + 1) if neighbor in scored
            )
        pages: list[dict[str, object]] = []
        for page in sorted(scored):
            result = scored[page]
            result["selection"] = (
                "core"
                if page in core_pages
                else "note"
                if page in note_pages
                else "neighbor"
                if page in selected_pages
                else "not_selected"
            )
            pages.append(result)
        documents.append(
            {
                "document": str(item["document"]),
                "sha256": str(item.get("sha256", "")),
                "company_key": str(item.get("company_key", "")),
                "total_pages": int(item["total_pages"]),
                "ocr_pages_scored": len(pages),
                "core_financial_pages": len(core_pages),
                "note_financial_pages": len(note_pages),
                "selected_review_pages": len(selected_pages),
                "pages": pages,
            }
        )
    return {
        "protocol": "ocr-financial-page-manifest-v2",
        "classification": "development_only",
        "scope": "Bounded MinerU discovery pages from unseen-company PDFs",
        "ocr_performed": True,
        "model_invoked": False,
        "selection_threshold": None,
        "selection_policy": {
            "scorer": "build_financial_page_manifest.score_financial_page",
            "page_classes": ["financial_summary", "financial_note", "non_financial"],
            "score_is_diagnostic_not_selection_gate": True,
            "clean_neighbor_window": 1,
            "no_ticker_or_protected_split_features": True,
            "required_discovery_reason": required_reason,
        },
        "documents": documents,
        "metrics": {
            "documents_processed": len(documents),
            "ocr_pages_scored": sum(int(item["ocr_pages_scored"]) for item in documents),
            "core_financial_pages": sum(int(item["core_financial_pages"]) for item in documents),
            "note_financial_pages": sum(int(item["note_financial_pages"]) for item in documents),
            "selected_review_pages": sum(int(item["selected_review_pages"]) for item in documents),
            "documents_with_financial_summary": sum(
                int(item["core_financial_pages"] > 0) for item in documents
            ),
            "documents_with_financial_note_only": sum(
                int(item["core_financial_pages"] == 0 and item["note_financial_pages"] > 0)
                for item in documents
            ),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render(report: dict[str, object]) -> str:
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    lines = [
        "# OCR financial-page development manifest",
        "",
        f"- Documents: {metrics['documents_processed']}",
        f"- OCR pages scored: {metrics['ocr_pages_scored']}",
        f"- Core financial pages: {metrics['core_financial_pages']}",
        f"- Financial-note pages: {metrics['note_financial_pages']}",
        f"- Selected pages including neighbors: {metrics['selected_review_pages']}",
        "",
    ]
    for document in report["documents"]:
        assert isinstance(document, dict)
        selected = [page for page in document["pages"] if page["selection"] != "not_selected"]
        if not selected:
            continue
        lines.extend([f"## {Path(str(document['document'])).name}", ""])
        for page in selected:
            lines.append(
                f"- Page {page['page']}: score={page['score']}, selection={page['selection']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery-manifest", type=Path, required=True)
    parser.add_argument("--markdown-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--document", action="append")
    parser.add_argument("--required-reason")
    args = parser.parse_args()
    for path in (args.output_json, args.output_markdown):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    discovery = json.loads(args.discovery_manifest.read_text(encoding="utf-8"))
    result = build_manifest(
        discovery,
        args.markdown_dir,
        document_names={Path(name).name for name in args.document} if args.document else None,
        required_reason=args.required_reason,
    )
    result["inputs"] = {
        "discovery_manifest": str(args.discovery_manifest.resolve()),
        "discovery_manifest_sha256": _sha256(args.discovery_manifest),
        "markdown_dir": str(args.markdown_dir.resolve()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_markdown.write_text(_render(result), encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
