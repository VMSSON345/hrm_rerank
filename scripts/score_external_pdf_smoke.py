#!/usr/bin/env python3
"""Score an external PDF smoke report without model dependencies."""

from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


CASE_SEPARATOR = "=" * 80


def field(block: str, name: str) -> str:
    match = re.search(rf"(?m)^{re.escape(name)}:\s*(.+?)\s*$", block)
    return match.group(1).strip() if match else ""


def parse_decimal(value: str) -> Decimal | None:
    cleaned = value.replace(",", "").replace("%", "").strip()
    match = re.match(r"^[+-]?\d+(?:\.\d+)?", cleaned)
    if match is None:
        return None
    try:
        return Decimal(match.group(0))
    except InvalidOperation:
        return None


def parse_report(path: Path) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    cases: list[dict[str, object]] = []
    for raw_block in text.split(CASE_SEPARATOR):
        if "CASE:" not in raw_block:
            continue
        case_id = field(raw_block, "CASE")
        gold_operation = field(raw_block, "GOLD_OPERATION")
        gold_value_text = field(raw_block, "GOLD_VALUE")
        gold_pages_text = field(raw_block, "GOLD_PDF_PAGE")
        answer_text = field(raw_block, "Answer")
        status = field(raw_block, "Status")
        program = field(raw_block, "Program")
        command_exit = field(raw_block, "COMMAND_EXIT")
        operation_match = re.match(r"([A-Z0-9_]+)\(", program)
        actual_operation = operation_match.group(1) if operation_match else ""
        gold_value = parse_decimal(gold_value_text)
        actual_value = parse_decimal(answer_text)
        tolerance = (
            Decimal("0.0001")
            if gold_operation in {"RATIO_PERCENT", "PERCENT_CHANGE"}
            else Decimal("0.5")
        )
        value_correct = bool(
            gold_value is not None
            and actual_value is not None
            and abs(actual_value - gold_value) <= tolerance
        )
        expected_pages = {
            int(item) for item in re.findall(r"\d+", gold_pages_text)
        }
        evidence_section = raw_block.split("Retrieved source chunks:", 1)[0]
        evidence_pages = {
            int(item)
            for item in re.findall(r"\| [^|\n]+, page (\d+)\s*$", evidence_section, re.M)
        }
        page_exact = bool(expected_pages and expected_pages == evidence_pages)
        page_expected_covered = bool(
            expected_pages and expected_pages.issubset(evidence_pages)
        )
        page_any = bool(expected_pages & evidence_pages)
        operation_correct = actual_operation == gold_operation
        parsed_exit = int(command_exit) if command_exit.isdigit() else None
        if parsed_exit not in (None, 0):
            availability = "pipeline_failure"
        elif not actual_operation or actual_value is None:
            availability = "no_prediction"
        else:
            availability = "prediction"
        execution_correct = operation_correct and value_correct
        cases.append(
            {
                "case": case_id,
                "availability": availability,
                "gold_operation": gold_operation,
                "actual_operation": actual_operation,
                "gold_value": gold_value_text,
                "actual_answer": answer_text,
                "operation_correct": operation_correct,
                "value_correct": value_correct,
                "execution_correct": execution_correct,
                "gold_pages": sorted(expected_pages),
                "evidence_pages": sorted(evidence_pages),
                "page_exact": page_exact,
                "page_expected_covered": page_expected_covered,
                "page_any": page_any,
                "page_correct": page_exact,
                "grounded_correct": execution_correct and page_exact,
                "accepted": status == "ACCEPTED",
                "command_exit": parsed_exit,
            }
        )
    return cases


def render_markdown(source: Path, cases: list[dict[str, object]]) -> str:
    total = len(cases)
    predictions = [item for item in cases if item["availability"] == "prediction"]
    prediction_count = len(predictions)
    operation = sum(bool(item["operation_correct"]) for item in predictions)
    execution = sum(bool(item["execution_correct"]) for item in predictions)
    pages_exact = sum(bool(item["page_exact"]) for item in predictions)
    pages_covered = sum(
        bool(item["page_expected_covered"]) for item in predictions
    )
    pages_any = sum(bool(item["page_any"]) for item in predictions)
    pipeline_failures = sum(
        item["availability"] == "pipeline_failure" for item in cases
    )
    no_prediction = sum(item["availability"] == "no_prediction" for item in cases)
    accepted = [item for item in predictions if item["accepted"]]
    accepted_correct = sum(bool(item["execution_correct"]) for item in accepted)
    accepted_grounded = sum(bool(item["grounded_correct"]) for item in accepted)
    lines = [
        "# External PDF smoke score",
        "",
        f"- Source: `{source}`",
        f"- Evaluation cases: {total}",
        f"- Predictions: {prediction_count}/{total}",
        f"- Pipeline failures: {pipeline_failures}/{total}",
        f"- No prediction: {no_prediction}/{total}",
        f"- Operation accuracy: {operation}/{prediction_count}",
        f"- Execution accuracy: {execution}/{prediction_count}",
        f"- Exact page provenance: {pages_exact}/{prediction_count}",
        f"- Expected-page coverage: {pages_covered}/{prediction_count}",
        f"- Any-page provenance: {pages_any}/{prediction_count}",
        f"- Accepted coverage: {len(accepted)}/{total} ({len(accepted) / max(1, total):.1%})",
        f"- Accepted answer precision: {accepted_correct}/{len(accepted)}"
        if accepted
        else "- Accepted answer precision: n/a (no accepted cases)",
        f"- Accepted grounded precision: {accepted_grounded}/{len(accepted)}"
        if accepted
        else "- Accepted grounded precision: n/a (no accepted cases)",
        "",
        "| Case | Availability | Operation | Value | Exact pages | Status |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for item in cases:
        lines.append(
            f"| {item['case']} | "
            f"{item['availability']} | "
            f"{'PASS' if item['operation_correct'] else 'FAIL'} | "
            f"{'PASS' if item['value_correct'] else 'FAIL'} | "
            f"{'PASS' if item['page_exact'] else 'FAIL'} | "
            f"{'ACCEPTED' if item['accepted'] else 'ABSTAIN'} |"
        )
    failures = [item for item in cases if not item["execution_correct"]]
    lines.extend(["", "## Execution failures", ""])
    if not failures:
        lines.append("Không có.")
    for item in failures:
        lines.append(
            f"- `{item['case']}`: expected `{item['gold_operation']} = "
            f"{item['gold_value']}`, got `{item['actual_operation']} = "
            f"{item['actual_answer']}`."
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--md-output", type=Path)
    args = parser.parse_args()
    report = args.report.expanduser().resolve()
    cases = parse_report(report)
    if not cases:
        parser.error(f"No CASE blocks found in {report}")
    json_output = args.json_output or report.with_name(report.stem + "_score.json")
    md_output = args.md_output or report.with_name(report.stem + "_score.md")
    json_output.write_text(
        json.dumps({"source": str(report), "cases": cases}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown = render_markdown(report, cases)
    md_output.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved JSON: {json_output}")
    print(f"Saved Markdown: {md_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
