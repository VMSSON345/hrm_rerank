#!/usr/bin/env python3
"""Evaluate frozen V8/V12 runtime on the frozen development-only V12-C cohort."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Sequence

from hrm_finance_qa.document_hrm import (
    DocumentHRMBundle,
    _abstention_threshold_for_prediction,
    _grounding_gate_reasons,
    _proof_conditioned_acceptance_eligible,
)
from hrm_finance_qa.document_reasoning_engine import (
    _augment_financial_identity_candidates,
    _debt_share_role,
)
from hrm_finance_qa.program_data import (
    NumericCandidate,
    execute_program,
    execute_program_dag,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_V8_SHA256 = "3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de"
FROZEN_COHORT_SHA256 = "13e4f290f7301d159edda87ed267607a80963e72bd9853329352726f70cfd520"
ALLOWED_COHORT_CLASSIFICATIONS = {
    "development_only_visual_qa_approved",
    "development_only_independent_calibration_visual_qa_approved",
}
PROTECTED = re.compile(r"(?:test|holdout|final|blind|validation)", re.IGNORECASE)
ROLE_LABELS = {
    "short_term_borrowings": "Vay và nợ thuê tài chính ngắn hạn",
    "long_term_borrowings": "Vay và nợ thuê tài chính dài hạn",
    "total_capital": "Tổng cộng nguồn vốn",
}
UNIT_LABELS = {
    "VND": "VND",
    "million_vnd": "triệu VND",
    "thousand_vnd": "nghìn VND",
    "billion_vnd": "tỷ VND",
}
EXPECTED_DAG = ("ADD", "SHARE_OF_TOTAL")
EXPECTED_ROLES = tuple(ROLE_LABELS)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _refuse_protected(path: Path) -> None:
    normalized = str(path.resolve()).replace("\\", "/")
    if PROTECTED.search(normalized):
        raise ValueError(f"Protected evaluation path is not allowed: {path}")


def _load_frozen(path: Path) -> list[dict[str, object]]:
    _refuse_protected(path)
    output: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            if value.get("classification") not in ALLOWED_COHORT_CLASSIFICATIONS:
                raise ValueError(f"Unapproved cohort row at {path}:{line_number}")
            if value.get("eligible_for_positive") is not True:
                raise ValueError(f"Ineligible cohort row at {path}:{line_number}")
            output.append(value)
    if not output:
        raise ValueError("Frozen cohort is empty")
    return output


def build_runtime_candidates(record: dict[str, object]) -> tuple[NumericCandidate, ...]:
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 3:
        raise ValueError("A V12-C record must have exactly three evidence leaves")
    company_key = str(record.get("company_key", ""))
    candidates: list[NumericCandidate] = []
    for index, leaf in enumerate(evidence):
        if not isinstance(leaf, dict):
            raise ValueError("Evidence leaf must be an object")
        role = str(leaf.get("role", ""))
        if role != EXPECTED_ROLES[index]:
            raise ValueError(f"Unexpected evidence role order: {role}")
        raw_unit = str(leaf.get("unit", ""))
        unit = UNIT_LABELS.get(raw_unit)
        if unit is None:
            raise ValueError(f"Unsupported frozen unit: {raw_unit}")
        value = str(leaf.get("value", ""))
        period = str(leaf.get("period", ""))
        row_label = ROLE_LABELS[role]
        context = "\n".join(
            (
                f"ROW: {row_label}",
                "ROW_TYPE: disclosed_balance_sheet_line",
                f"COLUMN: {period}",
                f"PERIOD: {period}",
                f"TABLE: {leaf.get('table_id', '')}",
                f"SCOPE: {leaf.get('scope', record.get('scope', ''))}",
                f"ENTITY: {company_key}",
                f"CELL: <VALUE> {value} </VALUE> {unit}",
                f"SOURCE_LINE: {leaf.get('source_line', '')}",
            )
        )
        candidates.append(
            NumericCandidate(
                index=index,
                value=value,
                raw=str(leaf.get("raw_value", value)),
                context=context,
                line_number=int(leaf.get("line_number", 0)),
                char_start=0,
                char_end=len(value),
                unit=unit,
                is_percent=False,
                is_year_like=False,
                relevance=1.0,
                source_name=str(leaf.get("source_name", "")),
                page=int(leaf.get("page", 0)),
                extraction_method="frozen_visual_qa_leaf",
                row_label=row_label,
                column_label=period,
                period=period,
                row_type="disclosed_balance_sheet_line",
                table_id=str(leaf.get("table_id", "")),
                section=f"{leaf.get('scope', record.get('scope', ''))}_balance_sheet",
                scope=str(leaf.get("scope", record.get("scope", ""))),
                entity_id=company_key,
            )
        )
    return tuple(candidates)


def _program_dag_summary(prediction: object) -> tuple[tuple[str, ...], tuple[int, ...]]:
    dag = getattr(prediction, "program_dag", None)
    if dag is None:
        return (), ()
    return (
        tuple(node.operation for node in dag.nodes),
        tuple(int(index) for index in dag.leaf_operands),
    )


def _selected_generation_sources(prediction: object) -> tuple[str, ...]:
    trace = getattr(prediction, "program_search_trace", ())
    program = getattr(prediction, "program")
    selected = next((item for item in trace if item.program == program), None)
    return tuple(selected.generation_sources) if selected is not None else ()


def evaluate_record(
    bundle: DocumentHRMBundle,
    record: dict[str, object],
) -> dict[str, object]:
    raw = build_runtime_candidates(record)
    question = str(record.get("question", ""))
    candidates = _augment_financial_identity_candidates(
        question,
        raw,
        max_candidates=bundle.model.config.max_candidates,
    )
    derived = next(
        (
            item
            for item in candidates
            if item.derivation_id == "interest_bearing_debt"
        ),
        None,
    )
    prediction = bundle.predict(question, candidates)
    grounding_rejections = _grounding_gate_reasons(
        question,
        prediction.program,
        candidates,
        financial_identity_catalog_enabled=(
            "financial_identity_catalog"
            in prediction.grounding_constraints_applied
        ),
    )
    default_threshold = float(bundle.metadata.get("abstention_threshold", 0.75))
    threshold = _abstention_threshold_for_prediction(
        bundle,
        prediction,
        default_threshold,
    )
    proof_conditioned_acceptance = _proof_conditioned_acceptance_eligible(
        prediction
    )
    accepted = (
        prediction.confidence >= threshold or proof_conditioned_acceptance
    ) and not grounding_rejections
    dag_operations, leaf_indices = _program_dag_summary(prediction)
    leaf_roles = tuple(
        _debt_share_role(candidates[index])
        for index in leaf_indices
        if 0 <= index < len(candidates)
    )
    program_correct = (
        dag_operations == EXPECTED_DAG
        and leaf_roles == EXPECTED_ROLES
    )
    predicted_value: Decimal | None = None
    execution_error = ""
    try:
        dag = getattr(prediction, "program_dag", None)
        predicted_value = (
            execute_program_dag(dag, candidates)
            if dag is not None
            else execute_program(prediction.program, candidates)
        )
    except (InvalidOperation, IndexError, ValueError, ZeroDivisionError) as error:
        execution_error = type(error).__name__
    expected_value = Decimal(str(record.get("answer_value", "")))
    execution_correct = predicted_value == expected_value
    abstain_reasons: list[str] = []
    if prediction.confidence < threshold and not proof_conditioned_acceptance:
        abstain_reasons.append("below_frozen_threshold")
    abstain_reasons.extend(grounding_rejections)
    return {
        "example_id": record.get("example_id"),
        "company_key": record.get("company_key"),
        "report_id": record.get("report_id"),
        "page": record.get("page"),
        "period": record.get("period"),
        "question": question,
        "expected_answer_value": str(expected_value),
        "raw_candidate_count": len(raw),
        "runtime_candidate_count": len(candidates),
        "gold_derivation_materialized": derived is not None,
        "predicted_operation": prediction.program.operation,
        "predicted_operands": list(prediction.program.operands),
        "predicted_dag_operations": list(dag_operations),
        "predicted_leaf_indices": list(leaf_indices),
        "predicted_leaf_roles": list(leaf_roles),
        "predicted_value": str(predicted_value) if predicted_value is not None else None,
        "execution_error": execution_error,
        "program_correct": program_correct,
        "execution_correct": execution_correct,
        "confidence": prediction.confidence,
        "confidence_threshold": threshold,
        "accepted": accepted,
        "acceptance_basis": (
            "frozen_confidence_threshold"
            if prediction.confidence >= threshold
            else "complete_required_proof"
            if proof_conditioned_acceptance
            else None
        ),
        "abstain_reasons": abstain_reasons,
        "grounding_rejections": list(grounding_rejections),
        "grounding_constraints_applied": list(
            prediction.grounding_constraints_applied
        ),
        "generation_sources": list(_selected_generation_sources(prediction)),
        "program_candidates_considered": prediction.program_candidates_considered,
        "symbolic_program_rejections": prediction.symbolic_program_rejections,
        "symbolic_unknown_constraints": prediction.symbolic_unknown_constraints,
        "synthesized_program_candidates": prediction.synthesized_program_candidates,
    }


def _aggregate(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    accepted = sum(bool(row["accepted"]) for row in rows)
    program_correct = sum(bool(row["program_correct"]) for row in rows)
    execution_correct = sum(bool(row["execution_correct"]) for row in rows)
    accepted_program_correct = sum(
        bool(row["accepted"]) and bool(row["program_correct"]) for row in rows
    )
    accepted_execution_correct = sum(
        bool(row["accepted"]) and bool(row["execution_correct"]) for row in rows
    )
    derivations = sum(bool(row["gold_derivation_materialized"]) for row in rows)
    return {
        "examples": total,
        "gold_derivation_materialized": derivations,
        "gold_derivation_rate": derivations / total if total else 0.0,
        "program_correct": program_correct,
        "program_accuracy": program_correct / total if total else 0.0,
        "execution_correct": execution_correct,
        "execution_accuracy": execution_correct / total if total else 0.0,
        "accepted": accepted,
        "accepted_coverage": accepted / total if total else 0.0,
        "accepted_program_precision": (
            accepted_program_correct / accepted if accepted else None
        ),
        "accepted_execution_precision": (
            accepted_execution_correct / accepted if accepted else None
        ),
        "abstained": total - accepted,
    }


def summarize(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_company: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_period_semantics: dict[str, list[dict[str, object]]] = defaultdict(list)
    abstain_reasons: Counter[str] = Counter()
    for row in rows:
        by_company[str(row["company_key"])].append(row)
        period = str(row["period"])
        semantics = "opening_endpoint" if period.startswith("01/01/") else "closing_endpoint"
        by_period_semantics[semantics].append(row)
        abstain_reasons.update(str(reason) for reason in row["abstain_reasons"])
    return {
        "overall": _aggregate(rows),
        "company_groups": {
            company: _aggregate(group)
            for company, group in sorted(by_company.items())
        },
        "period_semantics": {
            semantics: _aggregate(group)
            for semantics, group in sorted(by_period_semantics.items())
        },
        "abstain_reason_counts": dict(sorted(abstain_reasons.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort",
        type=Path,
        default=ROOT / "data/v12c_interest_bearing_debt_share_development_frozen_v1.jsonl",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/document_hrm_V8_candidate.pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--expected-cohort-sha256",
        default=FROZEN_COHORT_SHA256,
        help="Required precommitted cohort hash; defaults to the repair cohort",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.cohort, args.model, args.output):
        _refuse_protected(path)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite versioned artifact: {args.output}")
    if _sha256(args.cohort) != args.expected_cohort_sha256:
        raise ValueError("Frozen cohort SHA-256 mismatch")
    model_hash_before = _sha256(args.model)
    if model_hash_before != FROZEN_V8_SHA256:
        raise ValueError("Frozen V8 SHA-256 mismatch")

    records = _load_frozen(args.cohort)
    bundle = DocumentHRMBundle.load(args.model, device=args.device)
    bundle.metadata = dict(bundle.metadata)
    bundle.metadata["typed_program_synthesis_enabled"] = True
    bundle.metadata["financial_identity_catalog_enabled"] = True
    bundle.metadata["candidate_recall_v12_4_enabled"] = True
    rows = [evaluate_record(bundle, record) for record in records]
    model_hash_after = _sha256(args.model)
    if model_hash_after != model_hash_before:
        raise RuntimeError("Frozen V8 changed during evaluation")
    summary = summarize(rows)
    closing = summary["period_semantics"].get("closing_endpoint", {})
    report = {
        "protocol": "v12c-interest-bearing-debt-share-frozen-inference-v1",
        "classification": "development_only_frozen_evaluation",
        "protected_evaluation_sources_used": False,
        "training_performed": False,
        "model_or_reranker_changed": False,
        "threshold_changed": False,
        "cohort_path": str(args.cohort.resolve()),
        "cohort_sha256": _sha256(args.cohort),
        "model_path": str(args.model.resolve()),
        "model_sha256_before": model_hash_before,
        "model_sha256_after": model_hash_after,
        "evaluator_sha256": _sha256(Path(__file__)),
        "runtime_source_sha256": {
            "src/hrm_finance_qa/document_hrm.py": _sha256(
                ROOT / "src/hrm_finance_qa/document_hrm.py"
            ),
            "src/hrm_finance_qa/document_reasoning_engine.py": _sha256(
                ROOT / "src/hrm_finance_qa/document_reasoning_engine.py"
            ),
        },
        "runtime_flags": {
            "typed_program_synthesis_enabled": True,
            "financial_identity_catalog_enabled": True,
            "candidate_recall_v12_4_enabled": True,
            "program_reranker_attached": False,
        },
        "scope": "evidence-conditioned; retrieval is outside this frozen cohort audit",
        "summary": summary,
        "records": rows,
        "gate": {
            "frozen_inference_completed": True,
            "overall_program_execution_gate": (
                summary["overall"]["program_accuracy"] == 1.0
                and summary["overall"]["execution_accuracy"] == 1.0
            ),
            "closing_endpoint_program_execution_gate": (
                closing.get("program_accuracy") == 1.0
                and closing.get("execution_accuracy") == 1.0
            ),
            "acceptance_observable": summary["overall"]["accepted"] > 0,
            "ready_for_calibration": False,
            "phase_d_unlocked": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"]["overall"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
