#!/usr/bin/env python3
"""Build provenance-locked financial QA/program candidates from reviewed OCR tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_ocr_financial_page_manifest import _read_pages
from hrm_finance_qa.ontology import normalize_text
from hrm_finance_qa.program_data import parse_decimal


TARGET_QUESTION_TYPES = (
    "total_liabilities_components",
    "accounting_equation_check",
    "current_ratio",
    "interest_bearing_debt_share",
)
PERIOD_VALUE_RE = re.compile(
    r"(?P<period>(?:19|20)\d{2})(?:\s*\([^)]{0,20}\))?\s*:\s*"
    r"(?P<value>\(?[-+]?(?:\d{1,3}(?:[., ]\d{3})+|\d+(?:[.,]\d+)?)\)?)"
)
YEAR_RE = re.compile(r"(?:19|20)\d{2}")
CELL_VALUE_RE = re.compile(
    r"^\s*(?P<value>\(?[-+]?(?:\d{1,3}(?:[., ]\d{3})+|\d+(?:[.,]\d+)?)\)?)\s*$"
)
ORDINAL_PREFIX_RE = re.compile(
    r"^(?:(?:[ivxlcdm]+)[.)-]\s*|(?:\d+(?:\.\d+)*)(?:[.)-]\s*|\s+))",
    re.I,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metric_key(row_label: str) -> str | None:
    normalized = ORDINAL_PREFIX_RE.sub("", normalize_text(row_label)).strip()
    if "/" in row_label or not normalized:
        return None
    if "vay va no thue tai chinh ngan han" in normalized:
        return "short_term_borrowings"
    if "vay va no thue tai chinh dai han" in normalized:
        return "long_term_borrowings"
    if "tong nguon von" in normalized:
        return "total_capital"
    if "tong gia tri tai san" in normalized or "tong tai san" in normalized:
        return "total_assets"
    if "no phai tra dai han" in normalized or "no dai han" in normalized:
        return "long_term_liabilities"
    if "no ngan han" in normalized:
        return "current_liabilities"
    if "no phai tra" in normalized:
        return "total_liabilities"
    if (
        "tai san ngan han" in normalized
        and "khac" not in normalized
        and "hang ton kho" not in normalized
    ):
        return "current_assets"
    return None


def _page_unit(text: str) -> str:
    normalized = normalize_text(text)
    if "ty dong" in normalized:
        return "billion_vnd"
    if "trieu dong" in normalized:
        return "million_vnd"
    if "nghin dong" in normalized:
        return "thousand_vnd"
    if "don vi tinh dong" in normalized or " dong " in f" {normalized} ":
        return "VND"
    return "unspecified"


def extract_page_evidence(
    text: str, *, source_name: str, page: int
) -> list[dict[str, object]]:
    unit = _page_unit(text)
    evidence: list[dict[str, object]] = []
    header_periods: dict[int, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.split("|")]
        row_label = cells[0]
        metric = _metric_key(row_label)
        if metric is None:
            candidate_header = {
                index: years[0]
                for index, cell in enumerate(cells)
                if len(years := YEAR_RE.findall(cell)) == 1
            }
            if candidate_header:
                header_periods = candidate_header
            continue
        found_periods: set[str] = set()
        for match in PERIOD_VALUE_RE.finditer(line):
            raw = match.group("value").strip()
            value = parse_decimal(raw)
            if value is None:
                continue
            found_periods.add(match.group("period"))
            evidence.append(
                {
                    "metric": metric,
                    "period": match.group("period"),
                    "value": format(value, "f"),
                    "raw_value": raw,
                    "row_label": row_label,
                    "unit": unit,
                    "source_name": source_name,
                    "page": page,
                    "page_line": line_number,
                    "source_line": line.strip(),
                    "extraction_method": "mineru",
                }
            )
        for column_index, period in header_periods.items():
            if period in found_periods or column_index >= len(cells):
                continue
            match = CELL_VALUE_RE.match(cells[column_index])
            if match is None:
                continue
            raw = match.group("value").strip()
            value = parse_decimal(raw)
            if value is None:
                continue
            evidence.append(
                {
                    "metric": metric,
                    "period": period,
                    "value": format(value, "f"),
                    "raw_value": raw,
                    "row_label": row_label,
                    "unit": unit,
                    "source_name": source_name,
                    "page": page,
                    "page_line": line_number,
                    "source_line": line.strip(),
                    "extraction_method": "mineru",
                    "period_source": "table_header_column",
                }
            )
    return evidence


def _unique_metric_map(
    evidence: list[dict[str, object]], period: str
) -> tuple[dict[str, dict[str, object]], set[str]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for item in evidence:
        if item["period"] == period:
            grouped.setdefault(str(item["metric"]), []).append(item)
    unique: dict[str, dict[str, object]] = {}
    ambiguous: set[str] = set()
    for metric, items in grouped.items():
        distinct = {(str(item["value"]), str(item["row_label"])) for item in items}
        if len(distinct) == 1:
            unique[metric] = items[0]
        else:
            ambiguous.add(metric)
    return unique, ambiguous


def _same_unit(items: list[dict[str, object]]) -> bool:
    units = {str(item["unit"]) for item in items}
    return len(units) == 1


def _decimal(item: dict[str, object]) -> Decimal:
    return Decimal(str(item["value"]))


def _number(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _display_number(value: Decimal, places: int = 4) -> str:
    if value == value.to_integral_value():
        return _number(value)
    quantum = Decimal(1).scaleb(-places)
    return _number(value.quantize(quantum))


def _record(
    *,
    source: dict[str, object],
    page: int,
    period: str,
    question_type: str,
    question: str,
    answer: str,
    answer_value: Decimal,
    evidence: list[dict[str, object]],
    nodes: list[dict[str, object]],
    checks: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    stem = Path(str(source["document"])).stem
    return {
        "schema": "v12b2-unseen-financial-qa-candidate-v1",
        "example_id": f"unseen-{stem}-p{page}-{period}-{question_type}",
        "classification": "development_only",
        "human_review_required": True,
        "question_type": question_type,
        "question": question,
        "answer": answer,
        "answer_value": _number(answer_value),
        "period": period,
        "source": {
            "document": str(source["document"]),
            "sha256": str(source.get("sha256", "")),
            "company_key": str(source.get("company_key", "")),
            "page": page,
            "page_class": "financial_summary",
            "extraction_method": "mineru",
        },
        "evidence": evidence,
        "program": {"type": "dag", "nodes": nodes},
        "checks": checks or [],
    }


def build_page_records(
    source: dict[str, object], page: int, text: str
) -> tuple[list[dict[str, object]], Counter[str]]:
    source_name = Path(str(source["document"])).name
    evidence = extract_page_evidence(text, source_name=source_name, page=page)
    periods = sorted({str(item["period"]) for item in evidence}, reverse=True)
    records: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()
    for period in periods:
        metrics, ambiguous = _unique_metric_map(evidence, period)
        if ambiguous:
            rejection_counts["ambiguous_metric_period"] += 1

        def operands(*keys: str) -> list[dict[str, object]] | None:
            if any(key in ambiguous or key not in metrics for key in keys):
                return None
            items = [metrics[key] for key in keys]
            if not _same_unit(items):
                rejection_counts["unit_mismatch"] += 1
                return None
            return items

        items = operands("current_liabilities", "long_term_liabilities")
        if items:
            total = _decimal(items[0]) + _decimal(items[1])
            checks: list[dict[str, object]] = []
            if "total_liabilities" in metrics and "total_liabilities" not in ambiguous:
                disclosed = metrics["total_liabilities"]
                if _same_unit(items + [disclosed]):
                    difference = total - _decimal(disclosed)
                    checks.append(
                        {
                            "type": "disclosed_total_liabilities",
                            "evidence": disclosed,
                            "difference": _number(difference),
                            "consistent": difference == 0,
                        }
                    )
            records.append(
                _record(
                    source=source,
                    page=page,
                    period=period,
                    question_type="total_liabilities_components",
                    question=(
                        f"Tổng nợ phải trả năm {period} là bao nhiêu? "
                        "Hãy liệt kê nợ ngắn hạn và nợ dài hạn để kiểm tra."
                    ),
                    answer=(
                        f"Nợ ngắn hạn: {_number(_decimal(items[0]))}; nợ dài hạn: "
                        f"{_number(_decimal(items[1]))}; tổng nợ phải trả: {_number(total)}."
                    ),
                    answer_value=total,
                    evidence=items,
                    nodes=[
                        {
                            "id": "total_liabilities",
                            "operation": "ADD",
                            "inputs": ["current_liabilities", "long_term_liabilities"],
                        }
                    ],
                    checks=checks,
                )
            )

        items = operands("total_assets", "total_capital")
        if items:
            difference = _decimal(items[0]) - _decimal(items[1])
            records.append(
                _record(
                    source=source,
                    page=page,
                    period=period,
                    question_type="accounting_equation_check",
                    question=(
                        f"Tổng tài sản năm {period} có bằng Tổng nguồn vốn không? "
                        "Nếu lệch thì chênh lệch là bao nhiêu?"
                    ),
                    answer=(
                        f"Tổng tài sản: {_number(_decimal(items[0]))}; Tổng nguồn vốn: "
                        f"{_number(_decimal(items[1]))}; chênh lệch: {_number(difference)}; "
                        f"{'cân bằng' if difference == 0 else 'không cân bằng'}."
                    ),
                    answer_value=difference,
                    evidence=items,
                    nodes=[
                        {
                            "id": "balance_difference",
                            "operation": "SUBTRACT",
                            "inputs": ["total_assets", "total_capital"],
                        }
                    ],
                )
            )

        items = operands("current_assets", "current_liabilities")
        if items and _decimal(items[1]) != 0:
            with localcontext() as context:
                context.prec = 34
                ratio = _decimal(items[0]) / _decimal(items[1])
            records.append(
                _record(
                    source=source,
                    page=page,
                    period=period,
                    question_type="current_ratio",
                    question=(
                        f"Tính hệ số thanh toán hiện hành (Current Ratio) năm {period} "
                        "và cho biết doanh nghiệp có đủ khả năng trả nợ ngắn hạn không?"
                    ),
                    answer=(
                        f"Current Ratio = {_display_number(ratio)} lần. Theo ngưỡng kiểm tra 1,0, "
                        f"doanh nghiệp {'có' if ratio >= 1 else 'không có'} đủ tài sản ngắn hạn "
                        "để bao phủ nợ ngắn hạn."
                    ),
                    answer_value=ratio,
                    evidence=items,
                    nodes=[
                        {
                            "id": "current_ratio",
                            "operation": "DIVIDE",
                            "inputs": ["current_assets", "current_liabilities"],
                        }
                    ],
                    checks=[
                        {
                            "type": "coverage_threshold",
                            "threshold": "1",
                            "passes": ratio >= 1,
                        }
                    ],
                )
            )

        items = operands(
            "short_term_borrowings", "long_term_borrowings", "total_capital"
        )
        if items and _decimal(items[2]) != 0:
            with localcontext() as context:
                context.prec = 34
                borrowings = _decimal(items[0]) + _decimal(items[1])
                share = borrowings / _decimal(items[2]) * Decimal(100)
            records.append(
                _record(
                    source=source,
                    page=page,
                    period=period,
                    question_type="interest_bearing_debt_share",
                    question=(
                        "Tổng nợ vay tài chính chịu lãi gồm vay ngắn hạn và dài hạn "
                        f"chiếm bao nhiêu phần trăm Tổng nguồn vốn năm {period}?"
                    ),
                    answer=(
                        f"Nợ vay chịu lãi: {_number(borrowings)}; tỷ trọng trên Tổng nguồn vốn: "
                        f"{_display_number(share)}%."
                    ),
                    answer_value=share,
                    evidence=items,
                    nodes=[
                        {
                            "id": "interest_bearing_debt",
                            "operation": "ADD",
                            "inputs": ["short_term_borrowings", "long_term_borrowings"],
                        },
                        {
                            "id": "interest_bearing_debt_share",
                            "operation": "RATIO_PERCENT",
                            "inputs": ["interest_bearing_debt", "total_capital"],
                        },
                    ],
                )
            )
    return records, rejection_counts


def audit_records(records: list[dict[str, object]]) -> dict[str, object]:
    failures: Counter[str] = Counter()
    for item in records:
        if item.get("schema") != "v12b2-unseen-financial-qa-candidate-v1":
            failures["invalid_schema"] += 1
            continue
        if item.get("classification") != "development_only":
            failures["invalid_classification"] += 1
        source = item.get("source")
        if not isinstance(source, dict) or source.get("page_class") != "financial_summary":
            failures["invalid_source_class"] += 1
            continue
        if len(str(source.get("sha256", ""))) != 64 or not source.get("company_key"):
            failures["incomplete_provenance"] += 1
        period = str(item.get("period", ""))
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            failures["missing_evidence"] += 1
            continue
        if any(
            not isinstance(row, dict)
            or str(row.get("period", "")) != period
            or int(row.get("page", -1)) != int(source.get("page", -2))
            for row in evidence
        ):
            failures["cross_period_or_page_evidence"] += 1
            continue
        values = {str(row["metric"]): Decimal(str(row["value"])) for row in evidence}
        if len(values) != len(evidence):
            failures["duplicate_evidence_metric"] += 1
            continue
        program = item.get("program")
        nodes = program.get("nodes") if isinstance(program, dict) else None
        if not isinstance(nodes, list) or not nodes:
            failures["missing_program"] += 1
            continue
        execution = dict(values)
        try:
            with localcontext() as context:
                context.prec = 34
                for node in nodes:
                    if not isinstance(node, dict):
                        raise ValueError("invalid node")
                    inputs = [execution[str(name)] for name in node.get("inputs", [])]
                    operation = str(node.get("operation", ""))
                    if operation == "ADD" and len(inputs) == 2:
                        result = inputs[0] + inputs[1]
                    elif operation == "SUBTRACT" and len(inputs) == 2:
                        result = inputs[0] - inputs[1]
                    elif operation == "DIVIDE" and len(inputs) == 2 and inputs[1] != 0:
                        result = inputs[0] / inputs[1]
                    elif operation == "RATIO_PERCENT" and len(inputs) == 2 and inputs[1] != 0:
                        result = inputs[0] / inputs[1] * Decimal(100)
                    else:
                        raise ValueError("invalid operation")
                    execution[str(node["id"])] = result
        except (KeyError, ValueError, ArithmeticError):
            failures["program_execution_failure"] += 1
            continue
        expected = Decimal(str(item.get("answer_value", "")))
        actual = execution[str(nodes[-1]["id"])]
        if actual != expected:
            failures["answer_program_mismatch"] += 1
        if period not in str(item.get("question", "")):
            failures["question_period_mismatch"] += 1
    failed = sum(failures.values())
    return {
        "records_checked": len(records),
        "records_passed": len(records) if failed == 0 else 0,
        "records_failed": 0 if failed == 0 else len(records),
        "failure_counts": dict(sorted(failures.items())),
    }


def build_dataset(
    page_manifest: dict[str, object], markdown_dir: Path
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if page_manifest.get("classification") != "development_only":
        raise ValueError("QA candidate generation requires development_only input")
    documents = page_manifest.get("documents")
    if not isinstance(documents, list):
        raise ValueError("Page manifest documents must be a list")
    records: list[dict[str, object]] = []
    rejections: Counter[str] = Counter()
    summary_pages = 0
    for source in documents:
        if not isinstance(source, dict):
            raise ValueError("Invalid page-manifest document")
        document = Path(str(source.get("document", "")))
        markdown_path = markdown_dir / f"{document.stem}.md"
        if not markdown_path.is_file():
            raise FileNotFoundError(f"Missing OCR Markdown: {markdown_path}")
        pages = _read_pages(markdown_path)
        for page_item in source.get("pages", []):
            if not isinstance(page_item, dict) or page_item.get("page_class") != "financial_summary":
                continue
            page = int(page_item["page"])
            if page not in pages:
                raise ValueError(f"Missing page {page} in {markdown_path}")
            summary_pages += 1
            page_records, page_rejections = build_page_records(source, page, pages[page])
            records.extend(page_records)
            rejections.update(page_rejections)
    records.sort(key=lambda item: str(item["example_id"]))
    ids = [str(item["example_id"]) for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate generated example_id")
    type_counts = Counter(str(item["question_type"]) for item in records)
    report = {
        "protocol": "v12b2-unseen-financial-qa-candidate-audit-v2",
        "classification": "development_only",
        "model_invoked": False,
        "blind_test_used": False,
        "human_review_required": True,
        "ready_for_phase_d_training": False,
        "metrics": {
            "documents_scanned": len(documents),
            "financial_summary_pages_scanned": summary_pages,
            "qa_candidates_generated": len(records),
            "company_groups_with_candidates": len(
                {str(item["source"]["company_key"]) for item in records}
            ),
            "question_type_counts": {
                question_type: type_counts[question_type]
                for question_type in TARGET_QUESTION_TYPES
            },
            "missing_question_types": [
                question_type
                for question_type in TARGET_QUESTION_TYPES
                if type_counts[question_type] == 0
            ],
            "rejection_counts": dict(sorted(rejections.items())),
        },
    }
    report["automatic_qa"] = audit_records(records)
    if report["automatic_qa"]["records_failed"]:
        raise ValueError(f"Generated QA candidates failed audit: {report['automatic_qa']}")
    return records, report


def _render_review(records: list[dict[str, object]], report: dict[str, object]) -> str:
    metrics = report["metrics"]
    lines = [
        "# V12-B.2 unseen financial QA candidate review",
        "",
        f"- Summary pages scanned: {metrics['financial_summary_pages_scanned']}",
        f"- QA candidates: {metrics['qa_candidates_generated']}",
        f"- Company groups represented: {metrics['company_groups_with_candidates']}",
        f"- Question types: `{json.dumps(metrics['question_type_counts'], ensure_ascii=False)}`",
        f"- Missing question types: `{json.dumps(metrics['missing_question_types'], ensure_ascii=False)}`",
        f"- Automatic QA: `{json.dumps(report['automatic_qa'], ensure_ascii=False)}`",
        "- Classification: development-only; human review required; not training-ready.",
        "",
    ]
    for item in records:
        source = item["source"]
        lines.extend(
            [
                f"## {item['example_id']}",
                "",
                f"- Source: {Path(str(source['document'])).name}, page {source['page']}",
                f"- Question: {item['question']}",
                f"- Candidate answer: {item['answer']}",
                f"- Program: `{json.dumps(item['program'], ensure_ascii=False)}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-manifest", type=Path, required=True)
    parser.add_argument("--markdown-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-review", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output_jsonl, args.output_report, args.output_review):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    page_manifest = json.loads(args.page_manifest.read_text(encoding="utf-8"))
    records, report = build_dataset(page_manifest, args.markdown_dir)
    report["inputs"] = {
        "page_manifest": str(args.page_manifest.resolve()),
        "page_manifest_sha256": _sha256(args.page_manifest),
        "markdown_dir": str(args.markdown_dir.resolve()),
    }
    for path in (args.output_jsonl, args.output_report, args.output_review):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_review.write_text(_render_review(records, report), encoding="utf-8")
    print(json.dumps(report["metrics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
