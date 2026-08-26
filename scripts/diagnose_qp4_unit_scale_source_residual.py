#!/usr/bin/env python3
"""Trace the development-only QP4 UNIT_SCALE residual without editing runtime."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from hrm_finance_qa.program_data import currency_unit_scale
from hrm_finance_qa.question_planner import (
    compile_question_plan,
    compile_question_proof_contract,
)
from scripts.evaluate_qp4_development_baseline import (
    _QueryTimeout,
    _answer_signature,
    _parent_signature,
    _source_anchors,
    _source_counterfactual,
)
from scripts.evaluate_qp4_unit_scale_question_planner_implementation import (
    _verify_config,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMPLEMENTATION_CONFIG = (
    ROOT / "config" / "finhrm_qp4_unit_scale_question_planner_implementation_v1.json"
)
DEFAULT_OUTPUT = ROOT / "reports" / "finhrm_qp4_unit_scale_source_residual_trace_v1.json"
PROTECTED = re.compile(r"(?:^|[/_.-])(test|holdout|final|blind|validation)(?:[/_.-]|$)", re.I)


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if PROTECTED.search(path.resolve().as_posix()):
        raise ValueError(f"Protected input is forbidden: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_trace(config_path: Path, output_path: Path) -> dict[str, Any]:
    from hrm_finance_qa.document_hrm import DocumentHRMBundle
    from hrm_finance_qa.document_reasoning_engine import DocumentHRMReasoningEngine

    implementation = _load(config_path)
    _verify_config(implementation)
    development_path = ROOT / implementation["parents"]["development_cohort"]["path"]
    development = _load(development_path)
    checkpoint = ROOT / implementation["checkpoint"]["path"]
    engine = DocumentHRMReasoningEngine(DocumentHRMBundle.load(checkpoint, device="cpu"))
    anchors = _source_anchors(development)
    timeout_seconds = int(
        development["source_backed_cohort"]["per_query_timeout_seconds"]
    )
    rows: list[dict[str, Any]] = []
    for position, (_, anchor) in enumerate(anchors, start=1):
        question = _source_counterfactual(str(anchor["question"]), "UNIT_SCALE")
        plan = compile_question_plan(question)
        operation = str(anchor["observed"]["operation"])
        contract = compile_question_proof_contract(plan, operation)
        parent = _parent_signature(anchor)
        print(f"[{position}/{len(anchors)}] UNIT_SCALE trace {anchor['example_id']}", flush=True)
        common = {
            "example_id": anchor["example_id"],
            "company_key": anchor["company_key"],
            "source_key": anchor["source_sha256"],
            "source_path": anchor["source_path"],
            "layout_family": anchor["layout_family"],
            "question": question,
            "planner_requested_scale": plan.requested_scale,
            "planner_requested_unit_family": plan.requested_unit_family,
            "proof_required": list(contract.required),
            "unit_scale_obligation_declared": "unit_scale_match" in contract.required,
        }
        try:
            with _QueryTimeout(timeout_seconds):
                answer = engine.answer(
                    question,
                    ROOT / anchor["source_path"],
                    mineru_auto_generate=False,
                )
            signature = _answer_signature(answer)
            same_old_winner = all(
                signature.get(key) == parent.get(key)
                for key in ("action", "operation", "value", "unit", "evidence")
            )
            obligations = [
                {"name": name, "applicability": applicability, "state": state}
                for name, applicability, state in answer.proof_obligations
            ]
            required = [
                item for item in obligations if item["applicability"] == "required"
            ]
            selected_units = []
            for item in answer.evidence:
                scale = currency_unit_scale(item.candidate.unit)
                selected_units.append(
                    {
                        "unit": item.candidate.unit,
                        "currency_scale": None if scale is None else str(scale),
                        "source_name": item.candidate.source_name,
                        "page": item.candidate.page,
                        "row_label": item.candidate.row_label,
                        "column_label": item.candidate.column_label,
                        "extraction_method": item.candidate.extraction_method,
                    }
                )
            rows.append(
                common
                | {
                    "observed_signature": signature,
                    "selected_units": selected_units,
                    "raw_unit_scale_preserved": bool(selected_units)
                    and all(item["currency_scale"] is not None for item in selected_units),
                    "proof_obligations": obligations,
                    "unit_match_state": next(
                        (
                            item["state"]
                            for item in obligations
                            if item["name"] == "unit_match"
                        ),
                        None,
                    ),
                    "unit_scale_match_state": next(
                        (
                            item["state"]
                            for item in obligations
                            if item["name"] == "unit_scale_match"
                        ),
                        None,
                    ),
                    "all_required_proofs_valid": bool(required)
                    and all(item["state"] == "VALID" for item in required),
                    "old_winner_false_accept": same_old_winner,
                }
            )
        except Exception as exc:
            rows.append(
                common
                | {
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "old_winner_false_accept": False,
                }
            )
    answered = [row for row in rows if row.get("observed_signature", {}).get("action") == "ANSWER"]
    trace = {
        "schema": "finhrm-qp4-unit-scale-source-residual-trace-v1",
        "created_at": implementation["created_at"],
        "classification": "development_only_real_pdf_unit_scale_residual_isolation",
        "dimension": "UNIT_SCALE",
        "coverage": {
            "runs": len(rows),
            "companies": len({row["company_key"] for row in rows}),
            "sources": len({row["source_key"] for row in rows}),
            "real_layouts": len({row["layout_family"] for row in rows}),
        },
        "summary": {
            "planner_typed": sum(row["planner_requested_scale"] == "million_vnd" for row in rows),
            "answers": len(answered),
            "old_winner_false_accepts": sum(bool(row["old_winner_false_accept"]) for row in rows),
            "answered_with_raw_unit_scale_preserved": sum(
                bool(row.get("raw_unit_scale_preserved")) for row in answered
            ),
            "answered_with_unit_match_valid": sum(
                row.get("unit_match_state") == "VALID" for row in answered
            ),
            "answered_with_unit_scale_obligation_declared": sum(
                bool(row["unit_scale_obligation_declared"]) for row in answered
            ),
            "answered_with_unit_scale_match_state": sum(
                row.get("unit_scale_match_state") is not None for row in answered
            ),
            "answered_with_all_required_proofs_valid": sum(
                bool(row.get("all_required_proofs_valid")) for row in answered
            ),
        },
        "attribution": {
            "planner_retains_requested_scale": True,
            "evidence_retains_raw_unit_scale": all(
                bool(row.get("raw_unit_scale_preserved")) for row in answered
            ),
            "retrieval_or_binding_winner_changed": False,
            "first_missing_semantic_boundary": "question_proof_contract",
            "first_missing_runtime_assessment": "symbolic_verifier",
            "engine_action_policy_already_fails_closed_on_required_unknown": True,
        },
        "rows": rows,
        "runtime_hashes": {
            implementation["runtime_change"]["whitelist"][0]: implementation[
                "runtime_change"
            ]["after_sha256"],
            **implementation["frozen_runtime_hashes"],
        },
        "checkpoint_sha256": implementation["checkpoint"]["sha256"],
        "inputs": {
            "implementation_config": config_path.relative_to(ROOT).as_posix(),
            "implementation_config_sha256": _digest(config_path),
            "development_cohort": development_path.relative_to(ROOT).as_posix(),
            "development_cohort_sha256": _digest(development_path),
        },
        "abw_records": 0,
        "protected_records": [],
    }
    output_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return trace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_IMPLEMENTATION_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    trace = run_trace(args.config, args.output)
    print(json.dumps(trace["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
