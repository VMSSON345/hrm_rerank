from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
import hashlib
from itertools import permutations
import json
from pathlib import Path
import re
from typing import Any, Iterable

from hrm_finance_qa.document_hrm import DocumentHRMBundle
from hrm_finance_qa.document_reasoning_engine import (
    DocumentHRMReasoningEngine,
    DocumentReasoningAnswer,
)
from hrm_finance_qa.ontology import normalize_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = (
    ROOT
    / "config"
    / "finhrm_qp3_core_add_diverse_development_cohort_precommit_v1.json"
)
DEFAULT_COHORT = (
    ROOT
    / "data"
    / "finhrm_qp3_core_add_diverse_development_reviewed_cohort_v1.jsonl"
)
DEFAULT_COHORT_REPORT = (
    ROOT
    / "reports"
    / "finhrm_qp3_core_add_diverse_development_reviewed_cohort_v1.json"
)
DEFAULT_CHECKPOINT = ROOT / "models" / "document_hrm_V8_candidate.pt"
DEFAULT_OUTPUT = (
    ROOT
    / "reports"
    / "finhrm_qp3_core_add_diverse_end_to_end_v1.json"
)


@dataclass(frozen=True)
class InferenceCase:
    """The complete input boundary visible to the frozen inference engine."""

    example_id: str
    question: str
    source_path: Path
    source_sha256: str
    company_key: str
    layout_stratum: str
    layout_family: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _selected_gold_candidates(record: dict[str, Any]) -> list[dict[str, Any]]:
    by_index = {int(item["index"]): item for item in record["candidates"]}
    return [by_index[int(index)] for index in record["program"]["operands"]]


def _normalized_role_label(value: str) -> str:
    normalized = normalize_text(value)
    return re.sub(r"^(?:\d+|[ivxlcdm]+)\s+", "", normalized).strip()


def _labels_compatible(left: str, right: str) -> bool:
    normalized_left = _normalized_role_label(left)
    normalized_right = _normalized_role_label(right)
    if not normalized_left or not normalized_right:
        return normalized_left == normalized_right
    return (
        normalized_left == normalized_right
        or f" {normalized_left} " in f" {normalized_right} "
        or f" {normalized_right} " in f" {normalized_left} "
    )


def _physical_role_match(gold: dict[str, Any], observed: Any) -> bool:
    return (
        str(gold["value"]) == str(observed.value)
        and str(gold["source_name"]) == observed.source_name
        and int(gold["page"]) == int(observed.page or 0)
        and normalize_text(str(gold["period"])) == normalize_text(observed.period)
        and _labels_compatible(str(gold["row_label"]), observed.row_label)
        and _labels_compatible(str(gold["column_label"]), observed.column_label)
    )


def _physical_role_multiset_exact(
    gold: list[dict[str, Any]], observed: list[Any]
) -> bool:
    if len(gold) != len(observed):
        return False
    return any(
        all(_physical_role_match(expected, actual) for expected, actual in zip(gold, order))
        for order in permutations(observed)
    )


def _source_span(candidate: Any) -> tuple[int, int] | tuple[()]:
    value = (
        candidate.get("source_span", ())
        if isinstance(candidate, dict)
        else getattr(candidate, "source_span", ())
    )
    if not value or len(value) != 2:
        return ()
    return int(value[0]), int(value[1])


def _candidate_audit(candidate: Any) -> dict[str, Any]:
    return {
        "value": str(candidate.value),
        "source_name": candidate.source_name,
        "page": candidate.page,
        "row_label": candidate.row_label,
        "column_label": candidate.column_label,
        "period": candidate.period,
        "source_span": list(candidate.source_span),
        "extraction_method": candidate.extraction_method,
        "table_id": candidate.table_id,
        "subtable_id": candidate.subtable_id,
    }


def _run_frozen_inference(
    engine: DocumentHRMReasoningEngine,
    case: InferenceCase,
) -> DocumentReasoningAnswer:
    # Deliberately pass only the public PDF+question contract. All retrieval,
    # candidate, checkpoint and threshold settings remain frozen defaults.
    return engine.answer(case.question, case.source_path)


def _score_answer(
    case: InferenceCase,
    record: dict[str, Any],
    answer: DocumentReasoningAnswer,
) -> dict[str, Any]:
    gold = _selected_gold_candidates(record)
    observed = [item.candidate for item in answer.evidence]
    gold_values = Counter(str(item["value"]) for item in gold)
    observed_values = Counter(str(item.value) for item in observed)
    gold_spans = Counter(_source_span(item) for item in gold)
    observed_spans = Counter(_source_span(item) for item in observed)
    timings = dict(answer.stage_timings)
    retrieved_pages = sorted(
        {
            page
            for chunk in answer.retrieved_chunks
            if chunk.page_start is not None
            for page in range(
                int(chunk.page_start),
                int(chunk.page_end or chunk.page_start) + 1,
            )
        }
    )
    gold_pages = sorted({int(item["page"]) for item in gold})
    spans_present = bool(observed) and all(_source_span(item) for item in observed)
    provenance_complete = bool(observed) and all(
        item.source_name
        and item.page
        and item.row_label
        and item.column_label
        and item.period
        and item.extraction_method
        and _source_span(item)
        for item in observed
    )
    checks = {
        "document_discovery_ran": "document_discovery" in timings,
        "native_or_ocr_retrieval_ran": any(
            key.endswith("_retrieval") for key in timings
        ),
        "chunk_retrieval_nonempty": bool(answer.retrieved_chunks),
        "gold_page_retrieved": set(gold_pages).issubset(retrieved_pages),
        "candidate_materialization_nonempty": answer.candidate_count > 0,
        "neural_prediction_ran": any(key.endswith("_prediction") for key in timings),
        "operation_add": answer.program.operation == "ADD",
        "selected_arity_two": len(observed) == 2,
        "operand_values_exact": observed_values == gold_values,
        "operand_roles_and_physical_page_exact": _physical_role_multiset_exact(
            gold, observed
        ),
        "source_spans_present": spans_present,
        "source_spans_exact_diagnostic": observed_spans == gold_spans,
        "provenance_complete": provenance_complete,
        "gold_execution_exact": (
            answer.value is not None
            and answer.value == Decimal(str(record["answer_value"]))
        ),
        "accepted": not answer.abstained,
    }
    required_checks = tuple(
        key for key in checks if key != "source_spans_exact_diagnostic"
    )
    return {
        "example_id": case.example_id,
        "company_key": case.company_key,
        "layout_stratum": case.layout_stratum,
        "layout_family": case.layout_family,
        "source_path": str(case.source_path.relative_to(ROOT)),
        "source_sha256": case.source_sha256,
        "question": case.question,
        "checks": checks,
        "end_to_end_positive_passed": all(checks[key] for key in required_checks),
        "observed": {
            "abstained": answer.abstained,
            "value": None if answer.value is None else str(answer.value),
            "unit": answer.unit,
            "operation": answer.program.operation,
            "operands": list(answer.program.operands),
            "confidence": answer.prediction.confidence,
            "operation_confidence": answer.prediction.operation_confidence,
            "confidence_threshold": answer.confidence_threshold,
            "candidate_count": answer.candidate_count,
            "retrieved_pages": retrieved_pages,
            "evidence": [_candidate_audit(item) for item in observed],
            "grounding_constraints_applied": list(
                answer.prediction.grounding_constraints_applied
            ),
            "warnings": list(answer.warnings),
            "stage_timings": timings,
        },
        "gold_for_scoring_only": {
            "answer_value": str(record["answer_value"]),
            "operation": record["program"]["operation"],
            "physical_pages": gold_pages,
            "selected_candidates": [
                {
                    "value": str(item["value"]),
                    "source_name": item["source_name"],
                    "page": item["page"],
                    "row_label": item["row_label"],
                    "column_label": item["column_label"],
                    "period": item["period"],
                    "source_span": item["source_span"],
                }
                for item in gold
            ],
        },
    }


def _build_cases(
    cohort: Iterable[dict[str, Any]],
    cohort_report: dict[str, Any],
) -> tuple[list[InferenceCase], dict[str, dict[str, Any]]]:
    records = {str(item["example_id"]): item for item in cohort}
    details = {
        str(item["example_id"]): item for item in cohort_report["details"]
    }
    if set(records) != set(details):
        raise ValueError("Reviewed cohort JSONL and report example IDs differ")
    cases: list[InferenceCase] = []
    for example_id, record in records.items():
        detail = details[example_id]
        source_path = (ROOT / detail["source_path"]).resolve()
        cases.append(
            InferenceCase(
                example_id=example_id,
                question=str(record["question"]),
                source_path=source_path,
                source_sha256=str(detail["source_sha256"]),
                company_key=str(detail["company_key"]),
                layout_stratum=str(detail["layout_stratum"]),
                layout_family=str(detail["layout_family"]),
            )
        )
    return cases, records


def _coverage_gate_summary(
    results: list[dict[str, Any]],
    *,
    full_cohort: bool,
    required_layout_strata: tuple[str, ...],
    minimum_companies_per_layout: int,
) -> dict[str, Any]:
    passed = [item for item in results if item["end_to_end_positive_passed"]]
    passed_families = sorted({item["layout_family"] for item in passed})
    passed_companies_by_layout = {
        layout: sorted(
            {
                item["company_key"]
                for item in passed
                if item["layout_stratum"] == layout
            }
        )
        for layout in required_layout_strata
    }
    passed_sources_by_layout = {
        layout: sorted(
            {
                item["source_sha256"]
                for item in passed
                if item["layout_stratum"] == layout
            }
        )
        for layout in required_layout_strata
    }
    layout_minimum_checks = {
        layout: (
            len(passed_companies_by_layout[layout])
            >= minimum_companies_per_layout
            and len(passed_sources_by_layout[layout])
            >= minimum_companies_per_layout
        )
        for layout in required_layout_strata
    }
    errors = sum("error" in item for item in results)
    unsafe_false_accepts = sum(
        not item["end_to_end_positive_passed"]
        and "observed" in item
        and item["observed"]["abstained"] is False
        for item in results
    )
    gate_passed = (
        full_cohort
        and len(required_layout_strata) >= 2
        and all(layout_minimum_checks.values())
        and errors == 0
        and unsafe_false_accepts == 0
    )
    return {
        "gate_passed": gate_passed,
        "passed_layout_families": passed_families,
        "passed_layout_family_count": len(passed_families),
        "passed_companies_by_layout": passed_companies_by_layout,
        "passed_sources_by_layout": passed_sources_by_layout,
        "minimum_companies_per_layout": minimum_companies_per_layout,
        "layout_minimum_checks": layout_minimum_checks,
        "unsafe_false_accepts": unsafe_false_accepts,
        "errors": errors,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    precommit = json.loads(args.precommit.read_text(encoding="utf-8"))
    cohort_report = json.loads(args.cohort_report.read_text(encoding="utf-8"))
    cases, records = _build_cases(_load_jsonl(args.cohort), cohort_report)
    expected_checkpoint_hash = precommit["parents"]["frozen_checkpoint"]["sha256"]
    actual_checkpoint_hash = _sha256(args.checkpoint)
    if actual_checkpoint_hash != expected_checkpoint_hash:
        raise ValueError("Frozen checkpoint SHA-256 does not match the precommit")
    selected_ids = set(args.example_id or ())
    if selected_ids:
        unknown = selected_ids - {case.example_id for case in cases}
        if unknown:
            raise ValueError(f"Unknown example IDs: {sorted(unknown)}")
        cases = [case for case in cases if case.example_id in selected_ids]
    if args.limit > 0:
        cases = cases[: args.limit]
    for case in cases:
        if not case.source_path.is_file():
            raise FileNotFoundError(case.source_path)
        if _sha256(case.source_path) != case.source_sha256:
            raise ValueError(f"Source SHA-256 mismatch: {case.source_path}")

    bundle = DocumentHRMBundle.load(args.checkpoint, device="cpu")
    engine = DocumentHRMReasoningEngine(bundle)
    results: list[dict[str, Any]] = []
    for position, case in enumerate(cases, start=1):
        print(
            f"[{position}/{len(cases)}] {case.example_id} "
            f"{case.layout_stratum} {case.source_path.name}",
            flush=True,
        )
        try:
            answer = _run_frozen_inference(engine, case)
        except Exception as exc:  # keep the cohort diagnosis complete
            results.append(
                {
                    "example_id": case.example_id,
                    "company_key": case.company_key,
                    "layout_stratum": case.layout_stratum,
                    "layout_family": case.layout_family,
                    "source_path": str(case.source_path.relative_to(ROOT)),
                    "source_sha256": case.source_sha256,
                    "question": case.question,
                    "end_to_end_positive_passed": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            if args.fail_fast:
                raise
            continue
        results.append(_score_answer(case, records[case.example_id], answer))

    passed = sum(bool(item["end_to_end_positive_passed"]) for item in results)
    layout_counts = Counter(case.layout_stratum for case in cases)
    layout_passed = Counter(
        item["layout_stratum"]
        for item in results
        if item["end_to_end_positive_passed"]
    )
    full_cohort = len(cases) == len(records) and not selected_ids and args.limit <= 0
    required_layout_strata = tuple(
        precommit["deterministic_selection"]["layout_strata"]
    )
    minimum_companies_per_layout = int(
        precommit["deterministic_selection"]["minimum_companies_per_layout"]
    )
    coverage = _coverage_gate_summary(
        results,
        full_cohort=full_cohort,
        required_layout_strata=required_layout_strata,
        minimum_companies_per_layout=minimum_companies_per_layout,
    )
    report = {
        "schema": "finhrm-qp3-core-add-diverse-end-to-end-v1",
        "classification": "development_only_frozen_pdf_question_end_to_end_replay",
        "status": (
            "coverage_confirmed_with_safe_residual_abstention"
            if coverage["gate_passed"]
            else "diagnostic_incomplete_or_failed"
        ),
        "gate_passed": coverage["gate_passed"],
        "inputs": {
            "precommit": str(args.precommit.relative_to(ROOT)),
            "cohort": str(args.cohort.relative_to(ROOT)),
            "cohort_report": str(args.cohort_report.relative_to(ROOT)),
            "checkpoint": str(args.checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": actual_checkpoint_hash,
        },
        "inference_boundary": {
            "engine_inputs": ["question", "source_pdf_path"],
            "gold_candidates_or_program_passed_to_engine": False,
            "engine_defaults_overridden": False,
            "checkpoint_or_threshold_changed": False,
        },
        "metrics": {
            "full_cohort_run": full_cohort,
            "examples": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "examples_by_layout": dict(layout_counts),
            "passed_by_layout": dict(layout_passed),
            "passed_layout_families": coverage["passed_layout_families"],
            "passed_layout_family_count": coverage[
                "passed_layout_family_count"
            ],
            "passed_companies_by_layout": coverage[
                "passed_companies_by_layout"
            ],
            "passed_sources_by_layout": coverage["passed_sources_by_layout"],
            "minimum_companies_per_layout": coverage[
                "minimum_companies_per_layout"
            ],
            "layout_minimum_checks": coverage["layout_minimum_checks"],
            "unsafe_false_accepts": coverage["unsafe_false_accepts"],
            "errors": coverage["errors"],
        },
        "results": results,
        "invariants": {
            "protected_evaluation_loaded": False,
            "abw_loaded": False,
            "synthetic_layout_counted": False,
            "gold_used_for_inference": False,
            "candidate_rank_score_or_confidence_used_as_semantic_proof": False,
            "answer_presentation_unlocked": False,
            "qp4_or_phase_d_unlocked": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen PDF+question engine on the reviewed QP3 ADD "
            "development cohort without passing gold inference inputs."
        )
    )
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--cohort-report", type=Path, default=DEFAULT_COHORT_REPORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--example-id", action="append")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = evaluate(args)
    print(
        f"Saved {args.output}: {report['metrics']['passed']}/"
        f"{report['metrics']['examples']} passed; gate={report['gate_passed']}"
    )
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
