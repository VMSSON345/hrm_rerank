#!/usr/bin/env python3
"""Build a development-only financial-page manifest from native PDF layout text."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import fitz
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_financial_page_manifest import score_financial_page  # noqa: E402
from hrm_finance_qa.document_sources import _clean_line, assess_native_text  # noqa: E402
from hrm_finance_qa.ontology import normalize_text  # noqa: E402
from hrm_finance_qa.program_data import parse_decimal  # noqa: E402


HEADER_TERMS = (
    "chi tieu",
    "khoan muc",
    "ma so",
    "thuyet minh",
    "so cuoi",
    "so dau",
    "nam nay",
    "nam truoc",
    "nam 20",
    "tai ngay",
)
PROTECTED_MARKERS = ("blind", "holdout", "final_eval", "test.jsonl")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _layout_segments(line: str) -> list[str]:
    return [item.strip() for item in re.split(r"(?:\t+| {2,})", line.strip()) if item.strip()]


def _is_financial_layout_row(segments: list[str]) -> bool:
    financial_values = 0
    for segment in segments:
        for match in re.finditer(r"[()+\-\d][\d., ()]*", segment):
            value = parse_decimal(match.group(0))
            if value is not None and abs(value) >= 1000:
                financial_values += 1
    return len(segments) >= 3 and financial_values >= 2


def _is_layout_header(segments: list[str]) -> bool:
    normalized = normalize_text(" ".join(segments))
    return len(segments) >= 2 and any(term in normalized for term in HEADER_TERMS)


def layout_to_table_context(text: str) -> str:
    lines: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        if "|" in raw:
            lines.append(_clean_line(raw))
            continue
        segments = _layout_segments(raw)
        if _is_financial_layout_row(segments) or _is_layout_header(segments):
            lines.append("| " + " | ".join(segments) + " |")
        else:
            lines.append(_clean_line(raw))
    return "\n".join(line for line in lines if line)


def _native_page_text(reader: PdfReader, fitz_document: fitz.Document, index: int) -> str:
    text = ""
    try:
        text = reader.pages[index].extract_text(extraction_mode="layout") or ""
    except Exception:
        text = ""
    if not assess_native_text(text).usable:
        try:
            text = fitz_document[index].get_text("text", sort=True) or ""
        except Exception:
            text = ""
    return layout_to_table_context(text)


def _selected_source_names(
    intake_names: set[str],
    excluded_names: set[str],
    layout_documents: dict[str, object],
    allowed_layout_families: set[str] | None,
) -> list[str]:
    names = sorted(intake_names - excluded_names)
    if not allowed_layout_families:
        return names
    return [
        name
        for name in names
        if isinstance(layout_documents.get(name), dict)
        and str(layout_documents[name].get("layout_family", ""))
        in allowed_layout_families
    ]


def build_manifest(
    intake: dict[str, object],
    layout_audit: dict[str, object],
    excluded_manifest: dict[str, object],
    *,
    allowed_layout_families: set[str] | None = None,
) -> tuple[dict[str, object], dict[str, dict[int, str]]]:
    if intake.get("classification") != "development_only_unseen_company_intake":
        raise ValueError("Expected locked unseen-company development intake")
    intake_documents = intake.get("documents")
    layout_documents = layout_audit.get("documents")
    excluded_documents = excluded_manifest.get("documents")
    if not isinstance(intake_documents, dict) or not isinstance(layout_documents, dict):
        raise ValueError("Input document maps are missing")
    if not isinstance(excluded_documents, list):
        raise ValueError("Excluded page manifest must contain a document list")
    excluded_names = {
        Path(str(item["document"])).name
        for item in excluded_documents
        if isinstance(item, dict)
    }
    selected_names = _selected_source_names(
        set(intake_documents),
        excluded_names,
        layout_documents,
        allowed_layout_families,
    )
    documents: list[dict[str, object]] = []
    markdown_pages: dict[str, dict[int, str]] = {}
    for source_name in selected_names:
        intake_entry = intake_documents[source_name]
        layout_entry = layout_documents.get(source_name)
        if not isinstance(intake_entry, dict) or not isinstance(layout_entry, dict):
            raise ValueError(f"Missing intake/layout entry: {source_name}")
        source = Path(str(layout_entry.get("source_path", ""))).resolve()
        if any(marker in str(source).lower() for marker in PROTECTED_MARKERS):
            raise ValueError(f"Refusing protected source: {source}")
        expected_hash = str(intake_entry.get("source_sha256", ""))
        if not source.is_file() or _sha256(source) != expected_hash:
            raise ValueError(f"Source hash mismatch: {source_name}")
        reader = PdfReader(str(source))
        fitz_document = fitz.open(source)
        pages: list[dict[str, object]] = []
        rendered_text: dict[int, str] = {}
        try:
            for index in range(len(reader.pages)):
                text = _native_page_text(reader, fitz_document, index)
                result = score_financial_page(text)
                selection = (
                    "core"
                    if result["page_class"] == "financial_summary"
                    else "note"
                    if result["page_class"] == "financial_note"
                    else "not_selected"
                )
                pages.append({"page": index + 1, "selection": selection, **result})
                if selection != "not_selected":
                    rendered_text[index + 1] = text
        finally:
            fitz_document.close()
        markdown_pages[source_name] = rendered_text
        documents.append(
            {
                "document": str(source),
                "sha256": expected_hash,
                "company_key": str(intake_entry["company_key"]),
                "total_pages": len(reader.pages),
                "native_pages_scored": len(reader.pages),
                "core_financial_pages": sum(item["selection"] == "core" for item in pages),
                "note_financial_pages": sum(item["selection"] == "note" for item in pages),
                "pages": pages,
            }
        )
    report = {
        "protocol": "native-subtract-financial-page-manifest-v1",
        "classification": "development_only",
        "scope": "Native/hybrid documents excluded from the existing OCR page manifest",
        "allowed_layout_families": sorted(allowed_layout_families or ()),
        "ocr_performed": False,
        "model_invoked": False,
        "blind_test_used": False,
        "documents": documents,
        "metrics": {
            "documents_processed": len(documents),
            "company_groups": len({item["company_key"] for item in documents}),
            "native_pages_scored": sum(item["native_pages_scored"] for item in documents),
            "core_financial_pages": sum(item["core_financial_pages"] for item in documents),
            "note_financial_pages": sum(item["note_financial_pages"] for item in documents),
            "documents_with_selected_pages": sum(
                bool(item["core_financial_pages"] or item["note_financial_pages"])
                for item in documents
            ),
        },
    }
    return report, markdown_pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intake", type=Path, required=True)
    parser.add_argument("--layout-audit", type=Path, required=True)
    parser.add_argument("--exclude-page-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-dir", type=Path, required=True)
    parser.add_argument("--layout-family", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists() or args.markdown_dir.exists():
        raise FileExistsError("Refusing to overwrite native page artifacts")
    intake = json.loads(args.intake.read_text(encoding="utf-8"))
    layout = json.loads(args.layout_audit.read_text(encoding="utf-8"))
    excluded = json.loads(args.exclude_page_manifest.read_text(encoding="utf-8"))
    report, markdown_pages = build_manifest(
        intake,
        layout,
        excluded,
        allowed_layout_families=set(args.layout_family),
    )
    report["inputs"] = {
        "intake": str(args.intake.resolve()),
        "intake_sha256": _sha256(args.intake),
        "layout_audit": str(args.layout_audit.resolve()),
        "layout_audit_sha256": _sha256(args.layout_audit),
        "exclude_page_manifest": str(args.exclude_page_manifest.resolve()),
        "exclude_page_manifest_sha256": _sha256(args.exclude_page_manifest),
    }
    args.markdown_dir.mkdir(parents=True, exist_ok=False)
    for source_name, pages in markdown_pages.items():
        body = "\n\n".join(
            f"## Page {page}\n{text}" for page, text in sorted(pages.items())
        )
        (args.markdown_dir / f"{Path(source_name).stem}.md").write_text(
            body.rstrip() + "\n" if body else "", encoding="utf-8"
        )
    report["markdown_dir"] = str(args.markdown_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
