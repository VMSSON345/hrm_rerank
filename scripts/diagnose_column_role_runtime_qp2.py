"""Read-only full-runtime diagnosis for the QP2 explicit column-role lane."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.document_hrm import DocumentHRMBundle  # noqa: E402
from hrm_finance_qa.document_reasoning_engine import (  # noqa: E402
    DocumentHRMReasoningEngine,
)
from hrm_finance_qa.evidence_schema import evidence_cells  # noqa: E402
from hrm_finance_qa.question_planner import compile_question_plan  # noqa: E402


PROTECTED_TOKENS = {"blind", "validation", "holdout", "final_eval", "test"}


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _guard(path: Path) -> None:
    if set(part.lower() for part in path.resolve().parts) & PROTECTED_TOKENS:
        raise ValueError(f"QP2 diagnosis refuses protected input: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document", type=Path)
    parser.add_argument("question")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "document_hrm_V8_candidate.pt",
    )
    args = parser.parse_args()
    document = args.document.resolve()
    _guard(document)

    bundle = DocumentHRMBundle.load(args.model.resolve(), device="cpu")
    bundle.metadata = dict(bundle.metadata)
    bundle.metadata.update(
        {
            "typed_program_synthesis_enabled": True,
            "typed_program_synthesis_version": "typed-slot-synthesis-v1",
            "financial_identity_catalog_enabled": True,
            "candidate_recall_v12_4_enabled": True,
        }
    )
    answer = DocumentHRMReasoningEngine(bundle).answer(
        args.question,
        document,
        ocr=False,
        use_page_index=True,
        page_index_path=ROOT / "cache" / "document_page_index.sqlite3",
    )
    selected_trace = next(
        (
            item
            for item in answer.prediction.program_search_trace
            if item.program == answer.program
        ),
        None,
    )
    selected_candidates = tuple(item.candidate for item in answer.evidence)
    selected_cells = evidence_cells(selected_candidates)
    plan = compile_question_plan(args.question)
    report = {
        "schema": "column-role-runtime-diagnosis-qp2-v1",
        "classification": "local_diagnosis_only",
        "confirmation_eligible": False,
        "document": str(document),
        "document_sha256": _digest(document),
        "question": args.question,
        "compiled_plan": {
            "operation": plan.operation,
            "column_roles": list(plan.column_roles),
            "requested_years": list(plan.requested_years),
            "metric_text": plan.metric_text,
        },
        "result": {
            "abstained": answer.abstained,
            "value": str(answer.value) if answer.value is not None else None,
            "unit": answer.unit,
            "program": {
                "operation": answer.program.operation,
                "operands": list(answer.program.operands),
            },
            "confidence": answer.prediction.confidence,
            "threshold": answer.confidence_threshold,
            "threshold_changed": False,
            "warnings": list(answer.warnings),
        },
        "selected_evidence": [
            {
                "value": str(cell.value),
                "metric": cell.metric_label,
                "column": cell.column,
                "period": cell.period,
                "column_role": cell.column_role,
                "column_role_confidence": cell.column_role_confidence,
                "column_role_conflict": cell.column_role_conflict,
                "source_name": cell.source_name,
                "page": cell.page,
            }
            for cell in selected_cells
        ],
        "selected_trace": (
            {
                "eligible": selected_trace.eligible,
                "verification_state": selected_trace.verification_state,
                "proof_contract_state": selected_trace.proof_contract_state,
                "required_valid_fraction": selected_trace.required_valid_fraction,
                "required_unknown_fraction": selected_trace.required_unknown_fraction,
                "rejection_reasons": list(selected_trace.rejection_reasons),
                "unknown_reasons": list(selected_trace.unknown_reasons),
                "generation_sources": list(selected_trace.generation_sources),
                "proof_obligations": list(selected_trace.proof_obligations),
            }
            if selected_trace is not None
            else None
        ),
        "invariants": {
            "fine_tuning_performed": False,
            "threshold_changed": False,
            "proof_conditioned_select_acceptance_enabled": False,
            "validation_loaded": False,
            "blind_test_loaded": False,
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
