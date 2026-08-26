#!/usr/bin/env python3
"""Measure frozen-threshold false accepts on V12-C negative controls."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from hrm_finance_qa.document_reasoning_engine import _augment_financial_identity_candidates
from hrm_finance_qa.program_data import NumericCandidate


ROOT = Path(__file__).resolve().parents[1]
FROZEN_V8_SHA256 = "3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de"
PROTECTED = re.compile(r"(?:test|holdout|final|blind|validation)", re.IGNORECASE)
UNIT_LABELS = {
    "VND": "VND",
    "million_vnd": "triệu VND",
    "thousand_vnd": "nghìn VND",
    "billion_vnd": "tỷ VND",
}


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _refuse(path: Path) -> None:
    if PROTECTED.search(str(path.resolve()).replace("\\", "/")):
        raise ValueError(f"Protected path is not allowed: {path}")


def _load(path: Path) -> list[dict[str, object]]:
    _refuse(path)
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("classification") != "development_only_negative_ambiguity_visual_qa_approved":
                raise ValueError(f"Unapproved negative row: {path}:{line_number}")
            if row.get("expected_outcome") != "ABSTAIN" or row.get("eligible_for_positive") is not False:
                raise ValueError(f"Invalid negative contract: {path}:{line_number}")
            rows.append(row)
    if not rows:
        raise ValueError("Negative cohort is empty")
    return rows


def build_runtime_candidates(record: dict[str, object]) -> tuple[NumericCandidate, ...]:
    evidence = record.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < 2:
        raise ValueError("Negative control needs at least two evidence cells")
    company_key = str(record.get("company_key", ""))
    candidates: list[NumericCandidate] = []
    for index, leaf in enumerate(evidence):
        if not isinstance(leaf, dict):
            raise ValueError("Evidence cell must be an object")
        raw_unit = str(leaf.get("unit", ""))
        unit = UNIT_LABELS.get(raw_unit)
        if unit is None:
            raise ValueError(f"Unsupported unit: {raw_unit}")
        value = str(leaf.get("value", ""))
        period = str(leaf.get("period", ""))
        row_label = str(leaf.get("row_label", ""))
        scope = str(leaf.get("scope", ""))
        context = "\n".join((
            f"ROW: {row_label}",
            "ROW_TYPE: disclosed_financial_line",
            f"COLUMN: {period}",
            f"PERIOD: {period}",
            f"TABLE: {leaf.get('table_id', '')}",
            f"SCOPE: {scope}",
            f"ENTITY: {company_key}",
            f"CELL: <VALUE> {value} </VALUE> {unit}",
            f"SOURCE_LINE: {leaf.get('source_line', '')}",
        ))
        candidates.append(NumericCandidate(
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
            extraction_method="frozen_visual_qa_negative_control",
            row_label=row_label,
            column_label=period,
            period=period,
            row_type="disclosed_financial_line",
            table_id=str(leaf.get("table_id", "")),
            section=f"{scope}_financial_page",
            scope=scope,
            entity_id=company_key,
        ))
    return tuple(candidates)


def evaluate_record(bundle: DocumentHRMBundle, record: dict[str, object]) -> dict[str, object]:
    raw = build_runtime_candidates(record)
    question = str(record.get("question", ""))
    candidates = _augment_financial_identity_candidates(
        question, raw, max_candidates=bundle.model.config.max_candidates
    )
    prediction = bundle.predict(question, candidates)
    grounding_rejections = _grounding_gate_reasons(
        question,
        prediction.program,
        candidates,
        financial_identity_catalog_enabled=(
            "financial_identity_catalog" in prediction.grounding_constraints_applied
        ),
    )
    default_threshold = float(bundle.metadata.get("abstention_threshold", 0.75))
    threshold = _abstention_threshold_for_prediction(bundle, prediction, default_threshold)
    proof_conditioned_acceptance = _proof_conditioned_acceptance_eligible(
        prediction
    )
    accepted = (
        prediction.confidence >= threshold or proof_conditioned_acceptance
    ) and not grounding_rejections
    abstain_reasons: list[str] = []
    if prediction.confidence < threshold and not proof_conditioned_acceptance:
        abstain_reasons.append("below_frozen_threshold")
    abstain_reasons.extend(grounding_rejections)
    derivations = [
        item.derivation_id for item in candidates if getattr(item, "derivation_id", "")
    ]
    return {
        "example_id": record.get("example_id"),
        "company_key": record.get("company_key"),
        "report_id": record.get("report_id"),
        "page": record.get("page"),
        "period": record.get("period"),
        "proof_failure_type": record.get("proof_failure_type"),
        "required_proof_status": record.get("required_proof_status"),
        "missing_required_roles": record.get("missing_required_roles"),
        "expected_outcome": "ABSTAIN",
        "raw_candidate_count": len(raw),
        "runtime_candidate_count": len(candidates),
        "materialized_derivations": derivations,
        "predicted_operation": prediction.program.operation,
        "predicted_operands": list(prediction.program.operands),
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
        "false_accept": accepted,
        "abstain_reasons": abstain_reasons,
        "grounding_rejections": list(grounding_rejections),
        "grounding_constraints_applied": list(prediction.grounding_constraints_applied),
        "program_candidates_considered": prediction.program_candidates_considered,
        "symbolic_program_rejections": prediction.symbolic_program_rejections,
        "symbolic_unknown_constraints": prediction.symbolic_unknown_constraints,
        "synthesized_program_candidates": prediction.synthesized_program_candidates,
    }


def _aggregate(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    total = len(rows)
    false_accepts = sum(bool(row["false_accept"]) for row in rows)
    structurally_rejected = sum(bool(row["grounding_rejections"]) for row in rows)
    below_threshold = sum("below_frozen_threshold" in row["abstain_reasons"] for row in rows)
    confidences = [float(row["confidence"]) for row in rows]
    return {
        "controls": total,
        "false_accepts": false_accepts,
        "false_accept_rate": false_accepts / total if total else None,
        "safe_abstentions": total - false_accepts,
        "safe_abstention_rate": (total - false_accepts) / total if total else None,
        "structurally_rejected": structurally_rejected,
        "below_frozen_threshold": below_threshold,
        "confidence_min": min(confidences) if confidences else None,
        "confidence_max": max(confidences) if confidences else None,
    }


def summarize(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_failure: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_status: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_company: dict[str, list[dict[str, object]]] = defaultdict(list)
    abstain_reasons: Counter[str] = Counter()
    operations: Counter[str] = Counter()
    thresholds: Counter[str] = Counter()
    for row in rows:
        by_failure[str(row["proof_failure_type"])].append(row)
        by_status[str(row["required_proof_status"])].append(row)
        by_company[str(row["company_key"])].append(row)
        abstain_reasons.update(str(reason) for reason in row["abstain_reasons"])
        operations[str(row["predicted_operation"])] += 1
        thresholds[f"{float(row['confidence_threshold']):.15f}"] += 1
    return {
        "overall": _aggregate(rows),
        "proof_failure_types": {key: _aggregate(value) for key, value in sorted(by_failure.items())},
        "required_proof_status": {key: _aggregate(value) for key, value in sorted(by_status.items())},
        "company_groups": {key: _aggregate(value) for key, value in sorted(by_company.items())},
        "predicted_operation_counts": dict(sorted(operations.items())),
        "frozen_threshold_counts": dict(sorted(thresholds.items())),
        "abstain_reason_counts": dict(sorted(abstain_reasons.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--expected-cohort-sha256", required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/document_hrm_V8_candidate.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.cohort, args.model, args.output):
        _refuse(path)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite versioned artifact: {args.output}")
    if _sha256(args.cohort) != args.expected_cohort_sha256:
        raise ValueError("Frozen negative cohort SHA-256 mismatch")
    model_hash_before = _sha256(args.model)
    if model_hash_before != FROZEN_V8_SHA256:
        raise ValueError("Frozen V8 SHA-256 mismatch")
    records = _load(args.cohort)
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
    report = {
        "protocol": "v12c-negative-ambiguity-frozen-false-accept-v1",
        "classification": "development_only_negative_control_frozen_evaluation",
        "protected_evaluation_sources_used": False,
        "training_performed": False,
        "model_or_reranker_changed": False,
        "threshold_changed": False,
        "alternative_thresholds_evaluated": False,
        "cohort_path": str(args.cohort.resolve()),
        "cohort_sha256": _sha256(args.cohort),
        "model_path": str(args.model.resolve()),
        "model_sha256_before": model_hash_before,
        "model_sha256_after": model_hash_after,
        "evaluator_sha256": _sha256(Path(__file__)),
        "runtime_flags": {
            "typed_program_synthesis_enabled": True,
            "financial_identity_catalog_enabled": True,
            "candidate_recall_v12_4_enabled": True,
            "program_reranker_attached": False,
        },
        "scope": "evidence-conditioned negative controls; retrieval is outside this audit",
        "summary": summary,
        "records": rows,
        "gate": {
            "false_accept_measurement_completed": True,
            "zero_false_accepts_at_frozen_threshold": summary["overall"]["false_accepts"] == 0,
            "negative_control_coverage_sufficient_for_current_threshold_audit": True,
            "safe_lower_threshold_identified": False,
            "ready_for_threshold_change": False,
            "phase_d_unlocked": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"]["overall"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
