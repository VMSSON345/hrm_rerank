#!/usr/bin/env python3
"""Generic structural scoring for financial-statement pages."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

from hrm_finance_qa.ontology import VARIABLES, normalize_text


MIN_FINANCIAL_SCORE = 8
STATEMENT_LABELS = (
    "bao cao tinh hinh tai chinh",
    "bang can doi ke toan",
    "bao cao ket qua hoat dong kinh doanh",
    "bao cao ket qua kinh doanh",
    "bao cao luu chuyen tien te",
    "thuyet minh bao cao tai chinh",
)
STRUCTURAL_LABELS = ("bao cao tai chinh", "ma so", "thuyet minh", "don vi tinh")
SUMMARY_LABELS = (
    "tinh hinh tai chinh",
    "cac chi tieu tai chinh chu yeu",
    "chi tieu tai chinh chu yeu",
    "ket qua hoat dong san xuat kinh doanh",
    "tong doanh thu",
    "tong chi phi",
    "loi nhuan truoc thue",
)
NOTE_LABELS = (
    "tang giam tai san co dinh",
    "phan phoi loi nhuan",
    "trich lap cac quy",
    "von co phan",
    "menh gia co phieu",
)
PERIOD_LABELS = (
    "nam nay",
    "nam truoc",
    "ky nay",
    "ky truoc",
    "so cuoi nam",
    "so dau nam",
    "tai ngay",
)
FINANCIAL_NUMBER_RE = re.compile(
    r"""
    (?<![\w/])
    \(?\s*[+-]?
    (?:
        \d+(?:[.,]\d+)?\s*%              # ratios and percentages
        |\d{1,3}(?:[. ]\d{3})+(?:,\d+)?  # 1.234.567 or 1 234 567,89
        |\d{1,3}(?:,\d{3})+(?:\.\d+)?   # 1,234,567.89
        |\d+[.,]\d+                       # decimal ratios without percent
        |\d{5,}                            # ungrouped financial amounts
    )
    \s*\)?
    (?![\w/])
    """,
    re.VERBOSE,
)


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {normalized_text} "


def _table_evidence(text: str) -> tuple[int, int]:
    """Count Markdown table rows without treating prose numbers as table density."""

    table_rows = 0
    numeric_table_rows = 0
    for line in text.splitlines():
        if line.count("|") < 2:
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if cells and all(not cell or set(cell) <= {"-", ":"} for cell in cells):
            continue
        table_rows += 1
        if FINANCIAL_NUMBER_RE.search(line):
            numeric_table_rows += 1
    return table_rows, numeric_table_rows


def _table_text(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if line.count("|") >= 2)


def score_financial_page(text: str) -> dict[str, object]:
    normalized = normalize_text(text)
    statement_hits = [
        label for label in STATEMENT_LABELS if _contains_phrase(normalized, label)
    ]
    structural_hits = [
        label for label in STRUCTURAL_LABELS if _contains_phrase(normalized, label)
    ]
    period_hits = [
        label for label in PERIOD_LABELS if _contains_phrase(normalized, label)
    ]
    summary_hits = [
        label for label in SUMMARY_LABELS if _contains_phrase(normalized, label)
    ]
    note_hits = [label for label in NOTE_LABELS if _contains_phrase(normalized, label)]
    metric_hits = sorted(
        {
            spec.key
            for spec in VARIABLES.values()
            if any(
                _contains_phrase(normalized, normalize_text(alias))
                for alias in spec.aliases
                if normalize_text(alias)
            )
        }
    )
    normalized_table = normalize_text(_table_text(text))
    table_summary_hits = [
        label for label in SUMMARY_LABELS if _contains_phrase(normalized_table, label)
    ]
    table_note_hits = [
        label for label in NOTE_LABELS if _contains_phrase(normalized_table, label)
    ]
    table_metric_hits = sorted(
        {
            spec.key
            for spec in VARIABLES.values()
            if any(
                _contains_phrase(normalized_table, normalize_text(alias))
                for alias in spec.aliases
                if normalize_text(alias)
            )
        }
    )
    numeric_values = len(FINANCIAL_NUMBER_RE.findall(text))
    table_rows, numeric_table_rows = _table_evidence(text)
    share_par_value_ready = (
        numeric_values >= 2
        and _contains_phrase(normalized, "menh gia")
        and (
            _contains_phrase(normalized, "co phieu")
            or _contains_phrase(normalized, "co phan")
        )
    )
    numeric_score = 3 if numeric_values >= 16 else 2 if numeric_values >= 6 else 1 if numeric_values >= 2 else 0
    score = (
        (8 if statement_hits else 0)
        + min(4, len(structural_hits))
        + min(2, len(period_hits))
        + min(6, len(metric_hits))
        + numeric_score
    )
    dense_numeric_table = table_rows >= 3 and numeric_table_rows >= 2
    metric_ready = (
        numeric_values >= 2
        and len(table_metric_hits) >= 2
        and dense_numeric_table
    )
    structured_table_ready = (
        numeric_values >= 2
        and dense_numeric_table
        and bool(statement_hits)
        and bool(table_metric_hits or table_summary_hits)
    )
    summary_ready = metric_ready or (
        dense_numeric_table
        and bool(table_summary_hits)
        and (numeric_values >= 4 or bool(table_metric_hits))
    ) or structured_table_ready
    note_ready = (
        not summary_ready
        and dense_numeric_table
        and numeric_values >= 2
        and (
            bool(table_note_hits)
            or len(table_metric_hits) == 1
            or (bool(note_hits) and bool(table_metric_hits))
            or share_par_value_ready
        )
    )
    page_class = (
        "financial_summary"
        if summary_ready
        else "financial_note"
        if note_ready or share_par_value_ready
        else "non_financial"
    )
    numeric_qa_ready = page_class != "non_financial"
    compact = re.sub(r"\s+", " ", text).strip()
    return {
        "score": score,
        "statement_hits": statement_hits,
        "structural_hits": structural_hits,
        "period_hits": period_hits,
        "summary_hits": summary_hits,
        "note_hits": note_hits,
        "metric_hits": metric_hits,
        "table_summary_hits": table_summary_hits,
        "table_note_hits": table_note_hits,
        "table_metric_hits": table_metric_hits,
        "numeric_values": numeric_values,
        "table_rows": table_rows,
        "numeric_table_rows": numeric_table_rows,
        "page_class": page_class,
        "financial_summary_ready": summary_ready,
        "financial_note_ready": page_class == "financial_note",
        "numeric_qa_ready": numeric_qa_ready,
        "share_par_value_ready": share_par_value_ready,
        "excerpt": compact[:280],
    }


def build_document_manifest(
    document_audit: dict[str, object], page_texts: Sequence[str]
) -> dict[str, object]:
    audited_pages = document_audit.get("pages", [])
    if not isinstance(audited_pages, list):
        raise ValueError("Audit document pages must be a list")
    if len(page_texts) != len(audited_pages):
        raise ValueError(f"PDF/audit page mismatch: {len(page_texts)} != {len(audited_pages)}")
    usable: dict[int, dict[str, object]] = {}
    for page_audit, text in zip(audited_pages, page_texts):
        if not isinstance(page_audit, dict) or not page_audit.get("native_usable"):
            continue
        page_number = int(page_audit["page"])
        usable[page_number] = {
            "page": page_number,
            "native_reason": str(page_audit.get("reason", "")),
            **score_financial_page(text),
        }
    core_pages = {
        page
        for page, item in usable.items()
        if item["page_class"] == "financial_summary"
    }
    note_pages = {
        page for page, item in usable.items() if item["page_class"] == "financial_note"
    }
    selected_pages = core_pages | note_pages
    for page in core_pages | note_pages:
        selected_pages.update(
            neighbor for neighbor in (page - 1, page + 1) if neighbor in usable
        )
    pages: list[dict[str, object]] = []
    for page_number in sorted(usable):
        item = usable[page_number]
        item["selection"] = (
            "core"
            if page_number in core_pages
            else "note"
            if page_number in note_pages
            else "neighbor"
            if page_number in selected_pages
            else "not_selected"
        )
        pages.append(item)
    return {
        "document": str(document_audit.get("document", "")),
        "sha256": str(document_audit.get("sha256", "")),
        "total_pages": len(audited_pages),
        "native_usable_pages": len(usable),
        "core_financial_pages": len(core_pages),
        "note_financial_pages": len(note_pages),
        "selected_review_pages": len(selected_pages),
        "pages": pages,
    }


def render_markdown(report: dict[str, object]) -> str:
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    lines = [
        "# Native financial-page development manifest",
        "",
        f"- Documents processed: {metrics['documents_processed']}/{metrics['documents_total']}",
        f"- Pipeline failures: {metrics['pipeline_failures']}",
        f"- Native usable pages scored: {metrics['native_usable_pages']}",
        f"- Core financial pages: {metrics['core_financial_pages']}",
        f"- Review pages including neighbors: {metrics['selected_review_pages']}",
        "",
    ]
    for document in report["documents"]:
        assert isinstance(document, dict)
        selected = [
            page for page in document["pages"] if page["selection"] != "not_selected"
        ]
        if not selected:
            continue
        lines.extend([f"## {Path(str(document['document'])).name}", ""])
        for page in selected:
            lines.append(
                f"- Page {page['page']}: score={page['score']}, selection={page['selection']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
