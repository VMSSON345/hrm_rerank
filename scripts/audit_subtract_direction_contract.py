from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.direction_contract import assess_program_direction  # noqa: E402
from hrm_finance_qa.evidence_schema import build_evidence_graph  # noqa: E402
from hrm_finance_qa.program_data import load_program_examples  # noqa: E402
from hrm_finance_qa.question_planner import compile_question_plan  # noqa: E402
from hrm_finance_qa.symbolic_verifier import (  # noqa: E402
    VerificationState,
    verify_program,
)


_PROTECTED_MARKERS = ("test", "final", "blind", "holdout")
_PROTECTED_RECORD_MARKERS = (
    "validation",
    "test",
    "final",
    "blind",
    "holdout",
)
_MOJIBAKE_MARKERS = ("Ãƒ", "Ã‚", "Ã„", "Ã¡Â»", "Ã¡Âº", "Ã†", "Ã¢â‚¬")
_HEADER_ROW_TYPES = {"header", "section_header", "column_header"}


def refuse_protected_source(path: Path) -> None:
    normalized = str(path.resolve()).lower()
    if any(marker in normalized for marker in _PROTECTED_MARKERS):
        raise ValueError(f"Refusing protected test/final/blind/holdout source: {path}")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def audit_source(path: Path) -> dict[str, object]:
    refuse_protected_source(path)
    examples = load_program_examples(path)
    operation_counts: Counter[str] = Counter()
    intent_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    issue_counts: Counter[str] = Counter()
    verifier_state_counts: Counter[str] = Counter()
    verifier_direction_invalid_outcomes: Counter[str] = Counter()
    source_names: set[str] = set()
    company_keys: set[str] = set()
    review: list[dict[str, object]] = []
    protected_records: Counter[str] = Counter()
    protected_record_ids: list[str] = []

    for example in examples:
        protected_reason = _protected_record_reason(
            example.example_id, example.supervision
        )
        if protected_reason:
            protected_records[protected_reason] += 1
            if len(protected_record_ids) < 100:
                protected_record_ids.append(example.example_id)
            continue
        if example.program.operation not in {"SUBTRACT", "PERCENT_CHANGE"}:
            continue
        operation_counts[example.program.operation] += 1
        plan = compile_question_plan(example.question)
        graph = build_evidence_graph(example.candidates)
        assessment = assess_program_direction(plan, example.program, graph)
        intent = plan.direction_intent.value if plan.direction_intent else "NONE"
        intent_counts[intent] += 1
        outcome_counts[assessment.outcome] += 1

        cells_by_index = {cell.index: cell for cell in graph.cells}
        selected = [
            cells_by_index[index]
            for index in example.program.operands
            if index in cells_by_index
        ]
        source_names.update(cell.source_name for cell in selected if cell.source_name)
        supervision = example.supervision or {}
        company_key = str(supervision.get("company_key", ""))
        if company_key:
            company_keys.add(company_key)

        verification = verify_program(plan, example.program, graph)
        verifier_state_counts[verification.state.value] += 1
        states = {item.name: item.state for item in verification.constraints}
        if states.get("metric_match") is VerificationState.INVALID:
            issue_counts["wrong_metric"] += 1
        if states.get("period_coverage") is VerificationState.INVALID:
            issue_counts["missing_requested_period"] += 1
        if states.get("temporal_direction") is VerificationState.INVALID:
            issue_counts["verifier_direction_invalid"] += 1
            verifier_direction_invalid_outcomes[assessment.outcome] += 1
        if any(
            cell.row_type.lower() in _HEADER_ROW_TYPES for cell in selected
        ):
            issue_counts["header_endpoint"] += 1
        if any(not cell.metric_label for cell in selected):
            issue_counts["missing_metric_endpoint"] += 1
        if any(
            re.match(r"^\s*\(.*\)\s*$", example.candidates[index].raw)
            for index in example.program.operands
        ):
            issue_counts["parenthesized_negative_presentation"] += 1
        if any(cell.value < 0 for cell in selected):
            issue_counts["signed_negative_operand"] += 1
        if _has_visible_mojibake(example.question) or _has_visible_mojibake(
            example.context
        ):
            issue_counts["visible_mojibake_diagnostic"] += 1
        if plan.operation is not None and plan.operation != example.program.operation:
            issue_counts[f"operation_mismatch:{example.program.operation}->{plan.operation}"] += 1

        if assessment.outcome != "direction_supported" and len(review) < 100:
            review.append(
                {
                    "example_id": example.example_id,
                    "gold_operation": example.program.operation,
                    "planner_operation": plan.operation,
                    "direction_intent": intent,
                    "outcome": assessment.outcome,
                    "expected": list(assessment.expected),
                    "observed": list(assessment.observed),
                }
            )

    directional_total = sum(outcome_counts.values())
    supported = outcome_counts["direction_supported"]
    proved = supported + outcome_counts["direction_reversed"]
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "all_examples": len(examples),
        "development_eligible_examples": len(examples) - sum(protected_records.values()),
        "protected_records_detected_and_excluded": sum(protected_records.values()),
        "protected_record_reasons": dict(protected_records),
        "protected_record_ids_first_100": protected_record_ids,
        "directional_gold_examples": directional_total,
        "operation_counts": dict(operation_counts),
        "direction_intent_counts": dict(intent_counts),
        "direction_outcome_counts": dict(outcome_counts),
        "direction_supported_rate": supported / directional_total if directional_total else 0.0,
        "direction_provable_coverage": proved / directional_total if directional_total else 0.0,
        "direction_accuracy_when_provable": supported / proved if proved else 0.0,
        "issue_signal_counts": dict(issue_counts),
        "verifier_state_counts": dict(verifier_state_counts),
        "verifier_direction_invalid_outcomes": dict(
            verifier_direction_invalid_outcomes
        ),
        "unique_source_names": len(source_names),
        "unique_company_keys": len(company_keys),
        "non_supported_examples_first_100": review,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit SUBTRACT direction using development sources only"
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    sources = [audit_source(path) for path in args.input]
    aggregate_outcomes: Counter[str] = Counter()
    aggregate_intents: Counter[str] = Counter()
    aggregate_issues: Counter[str] = Counter()
    aggregate_operations: Counter[str] = Counter()
    aggregate_verifier_states: Counter[str] = Counter()
    aggregate_invalid_outcomes: Counter[str] = Counter()
    for source in sources:
        aggregate_outcomes.update(source["direction_outcome_counts"])
        aggregate_intents.update(source["direction_intent_counts"])
        aggregate_issues.update(source["issue_signal_counts"])
        aggregate_operations.update(source["operation_counts"])
        aggregate_verifier_states.update(source["verifier_state_counts"])
        aggregate_invalid_outcomes.update(
            source["verifier_direction_invalid_outcomes"]
        )
    directional_total = sum(aggregate_outcomes.values())
    supported = aggregate_outcomes["direction_supported"]
    proved = supported + aggregate_outcomes["direction_reversed"]
    protected_records = sum(
        int(source["protected_records_detected_and_excluded"])
        for source in sources
    )
    report = {
        "schema": "subtract-direction-contract-development-audit-v1",
        "classification": "development_only",
        "protected_evaluation_sources_used": False,
        "sources": sources,
        "summary": {
            "protected_records_detected_and_excluded": protected_records,
            "directional_gold_examples": directional_total,
            "operation_counts": dict(aggregate_operations),
            "direction_intent_counts": dict(aggregate_intents),
            "direction_outcome_counts": dict(aggregate_outcomes),
            "direction_supported_rate": supported / directional_total if directional_total else 0.0,
            "direction_provable_coverage": proved / directional_total if directional_total else 0.0,
            "direction_accuracy_when_provable": supported / proved if proved else 0.0,
            "issue_signal_counts": dict(aggregate_issues),
            "verifier_state_counts": dict(aggregate_verifier_states),
            "verifier_direction_invalid_outcomes": dict(
                aggregate_invalid_outcomes
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved direction audit: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
