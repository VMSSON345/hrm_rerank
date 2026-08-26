#!/usr/bin/env python3
"""Audit proof-conditioned confidence on frozen development-only V12-C cohorts."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Callable, Sequence

from hrm_finance_qa.document_hrm import DocumentHRMBundle, _grounding_gate_reasons
from hrm_finance_qa.document_reasoning_engine import (
    _augment_financial_identity_candidates,
)

from evaluate_v12c_interest_bearing_debt_share_frozen import (
    build_runtime_candidates as build_positive_candidates,
)
from evaluate_v12c_negative_ambiguity_frozen import (
    build_runtime_candidates as build_negative_candidates,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_MODEL_SHA256 = (
    "3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de"
)
PROTECTED = re.compile(r"(?:test|holdout|final|blind|validation)", re.IGNORECASE)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _refuse_protected(path: Path) -> None:
    if PROTECTED.search(str(path.resolve()).replace("\\", "/")):
        raise ValueError(f"Protected evaluation path is not allowed: {path}")


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    _refuse_protected(path)
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            classification = str(value.get("classification", ""))
            if not classification.startswith("development_only_"):
                raise ValueError(f"Non-development row at {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise ValueError(f"Empty cohort: {path}")
    return rows


def _selected_trace(prediction: object) -> object | None:
    program = getattr(prediction, "program")
    return next(
        (
            item
            for item in getattr(prediction, "program_search_trace", ())
            if item.program == program
        ),
        None,
    )


def _audit_record(
    bundle: DocumentHRMBundle,
    record: dict[str, object],
    *,
    cohort_kind: str,
    builder: Callable,
) -> dict[str, object]:
    raw = builder(record)
    question = str(record.get("question", ""))
    candidates = _augment_financial_identity_candidates(
        question,
        raw,
        max_candidates=bundle.model.config.max_candidates,
    )
    prediction = bundle.predict(question, candidates)
    trace = _selected_trace(prediction)
    grounding_rejections = _grounding_gate_reasons(
        question,
        prediction.program,
        candidates,
        financial_identity_catalog_enabled=(
            "financial_identity_catalog"
            in prediction.grounding_constraints_applied
        ),
    )
    obligations = list(getattr(trace, "proof_obligations", ()))
    required = [item for item in obligations if item.startswith("required:")]
    required_valid = [item for item in required if item.endswith(":VALID")]
    return {
        "example_id": record.get("example_id"),
        "company_key": record.get("company_key"),
        "cohort_kind": cohort_kind,
        "proof_failure_type": record.get("proof_failure_type"),
        "confidence": prediction.confidence,
        "confidence_schema": prediction.confidence_schema,
        "operation_confidence": prediction.operation_confidence,
        "neural_operation_confidence": prediction.neural_operation_confidence,
        "pointer_confidences": list(prediction.pointer_confidences),
        "neural_pointer_confidences": list(
            prediction.neural_pointer_confidences
        ),
        "predicted_operation": prediction.program.operation,
        "grounding_rejections": list(grounding_rejections),
        "proof_contract_state": getattr(trace, "proof_contract_state", ""),
        "required_valid_fraction": getattr(
            trace, "required_valid_fraction", 0.0
        ),
        "required_unknown_fraction": getattr(
            trace, "required_unknown_fraction", 0.0
        ),
        "optional_support_fraction": getattr(
            trace, "optional_support_fraction", 0.0
        ),
        "verification_state": getattr(trace, "verification_state", ""),
        "proof_score": getattr(trace, "proof_score", 0.0),
        "unknown_reasons": list(getattr(trace, "unknown_reasons", ())),
        "rejection_reasons": list(getattr(trace, "rejection_reasons", ())),
        "generation_sources": list(getattr(trace, "generation_sources", ())),
        "proof_obligations": obligations,
        "required_obligation_count": len(required),
        "required_valid_count": len(required_valid),
        "required_proof_complete": bool(required)
        and len(required_valid) == len(required),
        "derived_interest_bearing_debt_materialized": any(
            getattr(item, "derivation_id", "") == "interest_bearing_debt"
            for item in candidates
        ),
    }


def _fingerprint(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["confidence_schema"],
        row["proof_contract_state"],
        row["required_valid_fraction"],
        row["required_unknown_fraction"],
        tuple(row["grounding_rejections"]),
        row["derived_interest_bearing_debt_materialized"],
    )


def _summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    kinds = sorted({str(row["cohort_kind"]) for row in rows})
    output: dict[str, object] = {}
    for kind in kinds:
        current = [row for row in rows if row["cohort_kind"] == kind]
        fingerprints = Counter(str(_fingerprint(row)) for row in current)
        output[kind] = {
            "controls": len(current),
            "companies": len({str(row["company_key"]) for row in current}),
            "proof_contract_state_counts": dict(
                sorted(Counter(str(row["proof_contract_state"]) for row in current).items())
            ),
            "required_proof_complete": sum(
                bool(row["required_proof_complete"]) for row in current
            ),
            "structurally_rejected": sum(
                bool(row["grounding_rejections"]) for row in current
            ),
            "derived_interest_bearing_debt_materialized": sum(
                bool(row["derived_interest_bearing_debt_materialized"])
                for row in current
            ),
            "confidence_min": min(float(row["confidence"]) for row in current),
            "confidence_max": max(float(row["confidence"]) for row in current),
            "confidence_schema_counts": dict(
                sorted(Counter(str(row["confidence_schema"]) for row in current).items())
            ),
            "fingerprints": dict(sorted(fingerprints.items())),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positive-cohort", type=Path, required=True)
    parser.add_argument("--positive-sha256", required=True)
    parser.add_argument("--negative-cohort", type=Path, required=True)
    parser.add_argument("--negative-sha256", required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models/document_hrm_V8_candidate.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (
        args.positive_cohort,
        args.negative_cohort,
        args.model,
        args.output,
    ):
        _refuse_protected(path)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite versioned artifact: {args.output}")
    if _sha256(args.positive_cohort) != args.positive_sha256:
        raise ValueError("Positive cohort hash mismatch")
    if _sha256(args.negative_cohort) != args.negative_sha256:
        raise ValueError("Negative cohort hash mismatch")
    model_hash_before = _sha256(args.model)
    if model_hash_before != FROZEN_MODEL_SHA256:
        raise ValueError("Frozen model hash mismatch")

    bundle = DocumentHRMBundle.load(args.model, device="cpu")
    bundle.metadata = dict(bundle.metadata)
    bundle.metadata["typed_program_synthesis_enabled"] = True
    bundle.metadata["financial_identity_catalog_enabled"] = True
    bundle.metadata["candidate_recall_v12_4_enabled"] = True
    rows = [
        _audit_record(
            bundle,
            row,
            cohort_kind="positive",
            builder=build_positive_candidates,
        )
        for row in _load_jsonl(args.positive_cohort)
    ]
    rows.extend(
        _audit_record(
            bundle,
            row,
            cohort_kind="negative",
            builder=build_negative_candidates,
        )
        for row in _load_jsonl(args.negative_cohort)
    )
    model_hash_after = _sha256(args.model)
    if model_hash_after != model_hash_before:
        raise RuntimeError("Frozen model changed during audit")
    report = {
        "protocol": "v12c-proof-conditioned-abstention-audit-v1",
        "classification": "development_only_frozen_diagnostic",
        "protected_evaluation_sources_used": False,
        "training_performed": False,
        "threshold_changed": False,
        "alternative_thresholds_evaluated": False,
        "positive_cohort_sha256": _sha256(args.positive_cohort),
        "negative_cohort_sha256": _sha256(args.negative_cohort),
        "model_sha256_before": model_hash_before,
        "model_sha256_after": model_hash_after,
        "summary": _summary(rows),
        "records": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
