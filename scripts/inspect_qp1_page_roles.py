"""Read-only diagnosis of QP1 roles from an existing local page-index entry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.evidence_schema import evidence_cells  # noqa: E402
from hrm_finance_qa.ontology import normalize_text  # noqa: E402
from hrm_finance_qa.program_data import extract_numeric_candidates  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-contains", required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--metric-contains", default="")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--index", type=Path, default=ROOT / "cache" / "document_page_index.sqlite3"
    )
    args = parser.parse_args()

    uri = f"file:{args.index.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True, timeout=1.0) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT d.source_name, p.page_number, p.extraction_method,
                   p.searchable_text
            FROM pages p
            JOIN documents d USING(document_fingerprint)
            WHERE d.source_name LIKE ? AND p.page_number = ?
            ORDER BY p.updated_at DESC
            LIMIT 1
            """,
            (f"%{args.source_contains}%", args.page),
        ).fetchall()

    output: list[dict[str, object]] = []
    source_snippets: list[list[str]] = []
    metric = normalize_text(args.metric_contains)
    for row in rows:
        lines = row["searchable_text"].splitlines()
        for index, line in enumerate(lines):
            if metric and metric in normalize_text(line):
                source_snippets.append(
                    lines[max(0, index - 12) : min(len(lines), index + 4)]
                )
        candidates = extract_numeric_candidates(
            args.question,
            row["searchable_text"],
            source_name=row["source_name"],
            page=row["page_number"],
            extraction_method=row["extraction_method"],
        )
        for cell in evidence_cells(candidates):
            if metric and metric not in normalize_text(cell.metric_label):
                continue
            output.append(
                {
                    "value": str(cell.value),
                    "metric": cell.metric_label,
                    "column": cell.column,
                    "period": cell.period,
                    "header_path": list(cell.header_path),
                    "header_path_provenance": list(cell.header_path_provenance),
                    "column_role": cell.column_role,
                    "column_role_confidence": cell.column_role_confidence,
                    "column_role_conflict": cell.column_role_conflict,
                }
            )
    report = {
        "schema": "qp1-local-page-role-diagnosis-v1",
        "classification": "local_diagnosis_only",
        "confirmation_eligible": False,
        "strict_verifier_enabled": False,
        "threshold_changed": False,
        "source_contains": args.source_contains,
        "page": args.page,
        "question": args.question,
        "cached_pages": len(rows),
        "source_snippets": source_snippets[:3],
        "cells": output,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
