#!/usr/bin/env python3
"""Discover and freeze development-only three-role V12-C cohort candidates."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, localcontext
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable

import fitz

from hrm_finance_qa.ontology import normalize_text
from hrm_finance_qa.program_data import parse_decimal


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_NAME = re.compile(
    r"(?:test|holdout|final|blind|validation)", re.IGNORECASE
)
PAGE_MARKER = re.compile(r"^## Page (?P<page>\d+)\s*$")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
EXACT_DATE = re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])/(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}\b")
REPORT_DATE_YEAR = re.compile(
    r"(?:tai|vao)?\s*ngay\s+31\s+thang\s+12\s+nam\s+((?:19|20)\d{2})"
)
NOTE_REFERENCE = re.compile(r"^\(?\d{1,3}(?:\.\d{1,2})?(?:\([a-z]\))?\)?$", re.IGNORECASE)
ACCOUNT_CODES = {
    "short_term_borrowings": "320",
    "long_term_borrowings": "338",
    "total_capital": "440",
}
ROLE_ORDER = tuple(ACCOUNT_CODES)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _refuse_protected(path: Path) -> None:
    if PROTECTED_NAME.search(path.name):
        raise ValueError(f"Protected evaluation path is not allowed: {path}")


def _load_json(path: Path) -> dict[str, object]:
    _refuse_protected(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _page_sections(text: str) -> dict[int, list[tuple[int, str]]]:
    pages: dict[int, list[tuple[int, str]]] = {}
    current: int | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        marker = PAGE_MARKER.match(line)
        if marker:
            current = int(marker.group("page"))
            pages.setdefault(current, [])
            continue
        if current is not None:
            pages[current].append((line_number, line))
    return pages


def _role_from_line(line: str) -> str | None:
    normalized = f" {normalize_text(line)} "
    cells = [normalize_text(cell) for cell in line.split("|")]
    tokens = set(re.findall(r"\b\d{3}\b", normalized))
    if "320" in tokens and (" vay " in normalized or " no thue tai chinh " in normalized):
        return "short_term_borrowings"
    if "338" in tokens and (" vay " in normalized or " no thue tai chinh " in normalized):
        return "long_term_borrowings"
    if "440" in tokens and "nguon von" in normalized:
        return "total_capital"
    if any("tong nguon von" in cell for cell in cells):
        return "total_capital"
    if any(
        "vay" in cell and ("ngan han" in cell or "nghn han" in cell)
        for cell in cells
    ):
        return "short_term_borrowings"
    if any(
        "vay" in cell and ("dai han" in cell or "di han" in cell)
        for cell in cells
    ):
        return "long_term_borrowings"
    return None


def _periods(lines: list[tuple[int, str]]) -> tuple[str, ...]:
    header_lines = [line for _, line in lines[:12]]
    header = "\n".join(header_lines)
    dates = tuple(dict.fromkeys(EXACT_DATE.findall(header)))
    if len(dates) >= 2:
        return dates
    normalized_lines = [normalize_text(line) for line in header_lines]
    for line in header_lines:
        year_cells = [
            cell.strip()
            for cell in line.split("|")
            if YEAR.fullmatch(cell.strip())
        ]
        years = tuple(dict.fromkeys(year_cells))
        if len(years) >= 2:
            return tuple(f"31/12/{year}" for year in years)
    report_year: str | None = None
    for line in normalized_lines:
        match = REPORT_DATE_YEAR.search(line)
        if match:
            report_year = match.group(1)
            break
    normalized_header = "\n".join(normalized_lines)
    if report_year and "so cuoi nam" in normalized_header and "so dau nam" in normalized_header:
        return (f"31/12/{report_year}", f"01/01/{report_year}")
    if report_year:
        return (f"31/12/{report_year}",)
    return ()


def _unit(lines: list[tuple[int, str]]) -> str:
    normalized = normalize_text("\n".join(line for _, line in lines[:20]))
    if "ty dong" in normalized:
        return "billion_vnd"
    if "trieu dong" in normalized:
        return "million_vnd"
    if "nghin dong" in normalized:
        return "thousand_vnd"
    if "vnd" in normalized or "don vi dong" in normalized:
        return "VND"
    return "unspecified"


def _numeric_cells_after_role(line: str, role: str) -> list[tuple[str, Decimal]]:
    cells = [cell.strip() for cell in line.split("|") if cell.strip()]
    normalized = [normalize_text(cell) for cell in cells]
    account_code = ACCOUNT_CODES[role]
    code_positions = [index for index, cell in enumerate(normalized) if cell == account_code]
    label_positions = [
        index
        for index, cell in enumerate(normalized)
        if _role_from_line(f"| {cell} | {account_code} |") == role
        or role == "total_capital" and "tong nguon von" in cell
    ]
    if code_positions and label_positions:
        code_position = code_positions[0]
        # A PDF page can contain two side-by-side statements. Bind the role
        # label nearest to its account code so a later "chi phí lãi vay" cell
        # on the adjacent statement cannot move the amount window.
        label_position = min(
            label_positions,
            key=lambda index: (abs(index - code_position), index),
        )
        start = max(code_position, label_position) + 1
    else:
        start = max(code_positions + label_positions, default=-1) + 1
    output: list[tuple[str, Decimal]] = []
    for cell in cells[start:]:
        raw = cell.strip()
        if "%" in raw or re.search(r"[A-Za-zÀ-ỹ]", raw):
            continue
        # A bare small integer (19) or decimal-like token (21.2) is a
        # financial-statement note reference. Thousands-formatted amounts
        # such as 483.427 do not match this deliberately narrow pattern.
        if NOTE_REFERENCE.fullmatch(raw):
            continue
        value = parse_decimal(raw)
        if value is not None:
            output.append((raw, value))
    return output


def _role_rows(
    lines: list[tuple[int, str]], periods: tuple[str, ...]
) -> tuple[dict[str, dict[str, object]], Counter[str]]:
    found: dict[str, list[tuple[int, str]]] = {role: [] for role in ROLE_ORDER}
    for line_number, line in lines:
        role = _role_from_line(line)
        if role:
            found[role].append((line_number, line))
    issues: Counter[str] = Counter()
    output: dict[str, dict[str, object]] = {}
    for role in ROLE_ORDER:
        rows = found[role]
        if len(rows) != 1:
            issues[f"{role}:{'missing' if not rows else 'ambiguous'}"] += 1
            continue
        line_number, line = rows[0]
        values = _numeric_cells_after_role(line, role)
        if len(values) < len(periods):
            issues[f"{role}:incomplete_values"] += 1
            continue
        values = values[: len(periods)]
        output[role] = {
            "line_number": line_number,
            "source_line": line,
            "values": [
                {"period": period, "raw": raw, "value": format(value, "f")}
                for period, (raw, value) in zip(periods, values)
            ],
        }
    return output, issues


def _manifest_documents(
    manifest_path: Path, manifest: dict[str, object]
) -> dict[str, dict[str, object]]:
    if manifest.get("classification") != "development_only":
        raise ValueError(f"Manifest is not development_only: {manifest_path}")
    for key in ("blind_test_used", "test_jsonl_used", "sealed_validation_loaded"):
        if key in manifest and manifest.get(key) is not False:
            raise ValueError(f"Manifest has unsafe {key}: {manifest_path}")
    documents = manifest.get("documents")
    if not isinstance(documents, list):
        raise ValueError(f"Manifest document list is missing: {manifest_path}")
    output: dict[str, dict[str, object]] = {}
    for item in documents:
        if not isinstance(item, dict):
            continue
        name = Path(str(item.get("document", ""))).name
        if name:
            output[name] = item
    return output


def discover(
    sources: Iterable[tuple[Path, Path]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    documents_considered: list[dict[str, object]] = []
    exclusion_counts: Counter[str] = Counter()
    input_artifacts: list[dict[str, str]] = []
    for manifest_path, markdown_dir in sources:
        manifest_path = manifest_path.resolve()
        markdown_dir = markdown_dir.resolve()
        _refuse_protected(manifest_path)
        _refuse_protected(markdown_dir)
        manifest = _load_json(manifest_path)
        documents = _manifest_documents(manifest_path, manifest)
        input_artifacts.append(
            {"path": str(manifest_path), "sha256": _sha256(manifest_path)}
        )
        for markdown_path in sorted(markdown_dir.glob("*.md")):
            _refuse_protected(markdown_path)
            source_name = markdown_path.with_suffix(".pdf").name
            source = documents.get(source_name)
            if source is None:
                exclusion_counts["markdown_absent_from_locked_manifest"] += 1
                continue
            pdf_path = Path(str(source.get("document", ""))).resolve()
            _refuse_protected(pdf_path)
            expected_hash = str(source.get("sha256", ""))
            if not pdf_path.is_file() or _sha256(pdf_path) != expected_hash:
                raise ValueError(f"Locked source hash mismatch: {pdf_path}")
            company_key = str(source.get("company_key", ""))
            if not company_key:
                raise ValueError(f"Missing company key: {source_name}")
            pages = _page_sections(markdown_path.read_text(encoding="utf-8"))
            accepted_pages = 0
            document_reasons: Counter[str] = Counter()
            for page, lines in sorted(pages.items()):
                periods = _periods(lines)
                if not periods:
                    document_reasons["page_without_explicit_period"] += 1
                    continue
                roles, issues = _role_rows(lines, periods)
                if issues or set(roles) != set(ROLE_ORDER):
                    document_reasons.update(issues)
                    continue
                unit = _unit(lines)
                if unit == "unspecified":
                    document_reasons["page_without_explicit_unit"] += 1
                    continue
                accepted_pages += 1
                table_id = f"{Path(source_name).stem}:page-{page}:balance-sheet"
                by_role = {
                    role: {item["period"]: item for item in row["values"]}
                    for role, row in roles.items()
                }
                common_periods = [
                    period
                    for period in periods
                    if all(period in by_role[role] for role in ROLE_ORDER)
                ]
                for period in common_periods:
                    leaves = []
                    for role in ROLE_ORDER:
                        row = roles[role]
                        item = by_role[role][period]
                        leaves.append(
                            {
                                "role": role,
                                "value": item["value"],
                                "raw_value": item["raw"],
                                "period": period,
                                "unit": unit,
                                "source_name": source_name,
                                "source_sha256": expected_hash,
                                "page": page,
                                "table_id": table_id,
                                "line_number": row["line_number"],
                                "source_line": row["source_line"],
                                "extraction_method": "locked_native_markdown",
                            }
                        )
                    short, long, total = (Decimal(item["value"]) for item in leaves)
                    if total == 0:
                        document_reasons["zero_denominator"] += 1
                        continue
                    with localcontext() as context:
                        context.prec = 34
                        answer = (short + long) / total * Decimal(100)
                    record_id = sha256(
                        f"{expected_hash}|{page}|{period}".encode("utf-8")
                    ).hexdigest()[:20]
                    records.append(
                        {
                            "schema": "v12c-interest-bearing-debt-share-cohort-candidate-v1",
                            "classification": "development_only_pending_visual_qa",
                            "example_id": f"v12c-ibd-{record_id}",
                            "company_key": company_key,
                            "report_id": source_name,
                            "source_sha256": expected_hash,
                            "page": page,
                            "table_id": table_id,
                            "period": period,
                            "question": (
                                "Tổng nợ vay tài chính chịu lãi gồm vay ngắn hạn và dài hạn "
                                f"chiếm bao nhiêu phần trăm Tổng nguồn vốn tại {period}?"
                            ),
                            "program": {
                                "nodes": [
                                    {
                                        "id": "interest_bearing_debt",
                                        "operation": "ADD",
                                        "inputs": [
                                            "short_term_borrowings",
                                            "long_term_borrowings",
                                        ],
                                    },
                                    {
                                        "id": "interest_bearing_debt_share",
                                        "operation": "SHARE_OF_TOTAL",
                                        "inputs": [
                                            "interest_bearing_debt",
                                            "total_capital",
                                        ],
                                    },
                                ]
                            },
                            "answer_value": format(answer, "f"),
                            "evidence": leaves,
                            "review_status": "pending_visual_qa",
                            "eligible_for_positive": False,
                        }
                    )
            documents_considered.append(
                {
                    "source_name": source_name,
                    "company_key": company_key,
                    "source_sha256": expected_hash,
                    "markdown_path": str(markdown_path),
                    "markdown_sha256": _sha256(markdown_path),
                    "pages_considered": len(pages),
                    "candidate_pages": accepted_pages,
                    "exclusion_reasons": dict(document_reasons),
                }
            )
            exclusion_counts.update(document_reasons)
    records.sort(key=lambda item: (item["company_key"], item["report_id"], item["page"], item["period"]))
    companies = {str(item["company_key"]) for item in records}
    reports = {str(item["report_id"]) for item in records}
    audit = {
        "protocol": "v12c-interest-bearing-debt-share-cohort-discovery-v2",
        "classification": "development_only_discovery_pending_visual_qa",
        "protected_evaluation_sources_used": False,
        "model_output_used_for_selection": False,
        "training_performed": False,
        "threshold_changed": False,
        "inputs": input_artifacts,
        "summary": {
            "documents_considered": len(documents_considered),
            "candidate_records": len(records),
            "company_groups": len(companies),
            "report_groups": len(reports),
            "positive_records": 0,
            "ready_for_visual_qa": bool(records),
            "ready_for_frozen_evaluation": False,
            "ready_for_calibration": False,
        },
        "exclusion_counts": dict(exclusion_counts),
        "documents": documents_considered,
    }
    return records, audit


def render_records(records: list[dict[str, object]], render_dir: Path) -> list[dict[str, object]]:
    render_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for record in records:
        source_name = str(record["report_id"])
        page = int(record["page"])
        key = (source_name, page)
        if key in seen:
            continue
        seen.add(key)
        leaf = record["evidence"][0]
        source_hash = str(leaf["source_sha256"])
        source_paths = list(ROOT.glob(f"pdf/**/{source_name}"))
        if len(source_paths) != 1 or _sha256(source_paths[0]) != source_hash:
            raise ValueError(f"Cannot uniquely resolve locked PDF: {source_name}")
        output = render_dir / f"{Path(source_name).stem}__p{page}.png"
        document = fitz.open(source_paths[0])
        try:
            pixmap = document[page - 1].get_pixmap(
                matrix=fitz.Matrix(1.7, 1.7), alpha=False
            )
            pixmap.save(output)
        finally:
            document.close()
        rendered.append(
            {
                "source_name": source_name,
                "source_sha256": source_hash,
                "page": page,
                "render": str(output.resolve()),
                "render_sha256": _sha256(output),
            }
        )
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("PAGE_MANIFEST", "MARKDOWN_DIR"),
        required=True,
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--render-report", type=Path)
    args = parser.parse_args()
    for output in (args.candidates, args.audit, args.render_report):
        if output is not None and output.exists():
            raise FileExistsError(f"Refusing to overwrite versioned artifact: {output}")
    records, audit = discover(
        (Path(manifest), Path(markdown))
        for manifest, markdown in args.source
    )
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    args.candidates.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    audit["candidate_path"] = str(args.candidates.resolve())
    audit["candidate_sha256"] = _sha256(args.candidates)
    audit["builder_sha256"] = _sha256(Path(__file__))
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rendered: list[dict[str, object]] = []
    if args.render_dir is not None:
        if args.render_report is None:
            raise ValueError("--render-report is required with --render-dir")
        rendered = render_records(records, args.render_dir)
        args.render_report.parent.mkdir(parents=True, exist_ok=True)
        args.render_report.write_text(
            json.dumps(
                {
                    "protocol": "v12c-interest-bearing-debt-share-visual-queue-v1",
                    "classification": "development_only_pending_visual_qa",
                    "candidate_sha256": _sha256(args.candidates),
                    "pages_rendered": len(rendered),
                    "records": rendered,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "candidate_records": len(records),
                "company_groups": audit["summary"]["company_groups"],
                "pages_rendered": len(rendered),
                "ready_for_frozen_evaluation": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
