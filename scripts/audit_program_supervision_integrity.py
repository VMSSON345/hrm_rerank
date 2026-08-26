from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.evidence_schema import build_evidence_graph  # noqa: E402
from hrm_finance_qa.program_data import load_program_examples  # noqa: E402
from hrm_finance_qa.question_planner import compile_question_plan  # noqa: E402
from hrm_finance_qa.v12b2_supervision import (  # noqa: E402
    supervision_integrity_issues,
)


_PROTECTED_DATASET_MARKERS = (
    "test.jsonl",
    "final.jsonl",
    "final_eval",
    "blind",
    "holdout",
)
_PROTECTED_RECORD_MARKERS = ("validation", "test", "final", "blind", "holdout")
_MOJIBAKE_MARKERS = ("Ã", "Â", "Ä", "á»", "áº", "Æ", "â€")


def _refuse_protected_dataset(path: Path) -> None:
    normalized = path.name.lower()
    if any(marker in normalized for marker in _PROTECTED_DATASET_MARKERS):
        raise ValueError(f"Refusing protected test/final dataset: {path}")


def _has_visible_mojibake(value: str) -> bool:
    return any(marker in value for marker in _MOJIBAKE_MARKERS)


def _protected_record_reason(example_id: str, supervision: object) -> str:
    normalized_id = example_id.lower()
    for marker in _PROTECTED_RECORD_MARKERS:
        if marker in normalized_id:
            return f"example_id:{marker}"
    if not isinstance(supervision, dict):
        return ""
    for key in ("split", "partition", "subset", "source_split"):
        value = str(supervision.get(key, "")).lower()
        for marker in _PROTECTED_RECORD_MARKERS:
            if marker in value:
                return f"supervision.{key}:{marker}"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit typed gold-program integrity without reading blind/final data"
        )
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    total = excluded = visible_mojibake = 0
    operations: Counter[str] = Counter()
    excluded_operations: Counter[str] = Counter()
    protected_records: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    multiply_unit_profiles: Counter[str] = Counter()
    question_operation_mismatches: Counter[str] = Counter()
    excluded_examples: list[dict[str, object]] = []

    for source in args.input:
        _refuse_protected_dataset(source)
        for example in load_program_examples(source):
            protected_reason = _protected_record_reason(
                example.example_id, example.supervision
            )
            if protected_reason:
                protected_records[protected_reason] += 1
                continue
            total += 1
            operation = example.program.operation
            operations[operation] += 1
            visible_mojibake += int(
                _has_visible_mojibake(example.question)
                or _has_visible_mojibake(example.context)
            )
            graph = build_evidence_graph(example.candidates)
            plan = compile_question_plan(example.question)
            if (
                plan.operation is not None
                and plan.operation != example.program.operation
            ):
                question_operation_mismatches[
                    f"{example.program.operation}->{plan.operation}"
                ] += 1
            issues = supervision_integrity_issues(
                example, graph=graph, plan=plan
            )
            if operation == "MULTIPLY":
                units = sorted(
                    graph.cells[index].unit_key or "unknown"
                    for index in example.program.operands
                )
                multiply_unit_profiles[" x ".join(units)] += 1
            if not issues:
                continue
            excluded += 1
            excluded_operations[operation] += 1
            issue_counts.update(issues)
            if len(excluded_examples) < 100:
                excluded_examples.append(
                    {
                        "example_id": example.example_id,
                        "operation": operation,
                        "issues": list(issues),
                    }
                )

    report = {
        "schema": "program-supervision-integrity-audit-v2",
        "inputs": [str(path.resolve()) for path in args.input],
        "blind_test_used": False,
        "final_eval_used": False,
        "examples": total,
        "protected_records_detected_and_excluded": sum(
            protected_records.values()
        ),
        "protected_record_reasons": dict(protected_records),
        "integrity_eligible_examples": total - excluded,
        "integrity_excluded_examples": excluded,
        "integrity_coverage": (total - excluded) / total if total else 0.0,
        "operation_examples": dict(operations),
        "excluded_by_operation": dict(excluded_operations),
        "integrity_issue_counts": dict(issue_counts),
        "multiply_gold_unit_profiles": dict(multiply_unit_profiles),
        "question_operation_mismatches_diagnostic": dict(
            question_operation_mismatches
        ),
        "visible_mojibake_examples": visible_mojibake,
        "visible_mojibake_is_diagnostic_only": True,
        "excluded_examples_first_100": excluded_examples,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "examples",
        "protected_records_detected_and_excluded",
        "integrity_eligible_examples",
        "integrity_excluded_examples",
        "integrity_coverage",
        "excluded_by_operation",
        "integrity_issue_counts",
        "multiply_gold_unit_profiles",
        "question_operation_mismatches_diagnostic",
        "visible_mojibake_examples",
    )}, ensure_ascii=False, indent=2))
    print(f"Saved supervision audit: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
