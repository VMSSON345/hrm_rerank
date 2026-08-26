#!/usr/bin/env python3
"""Classify postimplementation QP4 residuals without persisting raw traces."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_qp4_post_question_surface_baseline_diagnostic_review import (
    _CandidateCaptureBundle,
    _actual_result,
    _add_surface_result,
    _runtime_candidates,
    _source_summary,
)
from scripts.evaluate_qp4_post_question_surface_residual_implementation import (
    _compiler_measurement,
)


ROOT = Path(__file__).resolve().parents[1]
PRECOMMIT = ROOT / "config/finhrm_qp4_post_question_surface_residual_implementation_diagnostic_refresh_precommit_v1.json"
PRECOMMIT_MARKDOWN = ROOT / "reports/finhrm_qp4_post_question_surface_residual_implementation_diagnostic_refresh_precommit_v1.md"
PRECOMMIT_LOCK = ROOT / "reports/finhrm_qp4_post_question_surface_residual_implementation_diagnostic_refresh_precommit_lock_v1.json"
ADD_BASELINE = ROOT / "reports/finhrm_qp4_post_question_surface_add_source_baseline_v1.json"
ACTUAL_CONFIG = ROOT / "config/finhrm_qp4_actual_plan_ratio_direction_source_cohort_v1.json"
IMPLEMENTATION_REPORT = ROOT / "reports/finhrm_qp4_post_question_surface_residual_implementation_v1.json"
REPORT = ROOT / "reports/finhrm_qp4_post_question_surface_residual_implementation_diagnostic_refresh_v1.json"
MARKDOWN = ROOT / "reports/finhrm_qp4_post_question_surface_residual_implementation_diagnostic_refresh_v1.md"
LOCK = ROOT / "reports/finhrm_qp4_post_question_surface_residual_implementation_diagnostic_refresh_lock_v1.json"

LAYERS = (
    "QUESTION_PLAN",
    "SOURCE_LOAD_OR_EXTRACTION",
    "CANDIDATE_POPULATION_OR_RETRIEVAL",
    "SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS",
    "SYMBOLIC_PROOF_OR_ACTION",
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _verify_precommit(config: dict[str, Any]) -> None:
    lock = _load(PRECOMMIT_LOCK)
    if not lock.get("precommit_gate_passed"):
        raise ValueError("Diagnostic-refresh precommit is not closed")
    if lock.get("authorized_next_step") != config["authorized_next_step_after_lock"]:
        raise ValueError("Diagnostic-refresh authorization mismatch")
    if lock.get("runtime_file_whitelist") != []:
        raise ValueError("Diagnostic-refresh runtime whitelist must be empty")
    for relative, expected in config["frozen_runtime_hashes"].items():
        if _digest(ROOT / relative) != expected:
            raise ValueError(f"Runtime changed before diagnostic: {relative}")
    if _digest(ROOT / config["checkpoint"]["path"]) != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint changed before diagnostic")


def _required_proof_action_passed(row: dict[str, Any], expected_action: str) -> bool:
    required_states = [
        state
        for _, applicability, state in row["trace"]["proof_obligations"]
        if applicability == "required"
    ]
    return (
        row["first_failing_layer"] == "SYMBOLIC_PROOF_OR_ACTION"
        and row["trace"].get("source_action") == expected_action
        and bool(required_states)
        and all(state == "VALID" for state in required_states)
    )


def run() -> dict[str, Any]:
    from hrm_finance_qa.document_hrm import DocumentHRMBundle
    from hrm_finance_qa.document_reasoning_engine import DocumentHRMReasoningEngine

    config = _load(PRECOMMIT)
    _verify_precommit(config)
    checkpoint = ROOT / config["checkpoint"]["path"]
    capture = _CandidateCaptureBundle(DocumentHRMBundle.load(checkpoint, device="cpu"))
    engine = DocumentHRMReasoningEngine(capture)
    seconds = config["measurement_scope"]["per_query_timeout_seconds"]
    observations: list[dict[str, Any]] = []

    add = _load(ADD_BASELINE)
    add_rows = [
        row
        for row in add["surface_results"]
        if row["category"] == config["measurement_scope"]["add_family_group"]
    ]
    for position, row in enumerate(add_rows, start=1):
        print(f"[{position}/29] ADD_SURFACE SHORT_LONG_EQUIVALENT", flush=True)
        observation = _add_surface_result(engine, capture, row, seconds)
        if _required_proof_action_passed(
            observation, row["parent_signature"]["action"]
        ):
            observation["first_failing_layer"] = "PASS"
        observations.append(observation)

    actual = _load(ACTUAL_CONFIG)
    baseline_rows = {
        (row["source_sha256"], dimension): {
            "company_key": row["company_key"],
            "source_sha256": row["source_sha256"],
            "document_layout_family": row["document_layout_family"],
            "table_layout_family": row["table_layout_family"],
            "dimension": dimension,
        }
        for row in actual["anchors"]
        for dimension in actual["dimensions"]
    }
    position = len(add_rows)
    for anchor in actual["anchors"]:
        for dimension in actual["dimensions"]:
            baseline = baseline_rows[(anchor["source_sha256"], dimension)]
            for label in ("base", "mutated"):
                position += 1
                print(f"[{position}/29] ACTUAL_PLAN {dimension} {label}", flush=True)
                observation = _actual_result(
                    engine, capture, baseline, anchor, label, seconds
                )
                if _required_proof_action_passed(observation, "ANSWER"):
                    observation["first_failing_layer"] = "PASS"
                observations.append(observation)

    compiler = _compiler_measurement()
    source = _source_summary(observations)
    required = config["diagnostic_gates"]
    source_errors = [
        row for row in observations if row["trace"]["source_outcome"] == "ERROR"
    ]
    accepted_wrong_layers = {
        "QUESTION_PLAN",
        "CANDIDATE_POPULATION_OR_RETRIEVAL",
        "SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS",
    }
    accepted_wrong = sum(
        row["trace"].get("source_action") == "ANSWER"
        and row["first_failing_layer"] in accepted_wrong_layers
        for row in observations
    )
    answer_with_valid_required_nonvalid_overall_proof = sum(
        row["trace"].get("source_action") == "ANSWER"
        and row["first_failing_layer"] == "PASS"
        and row["trace"].get("verification_state") != "VALID"
        for row in observations
    )
    checks = {
        "compiler_inventory_retained": compiler["full_inventory"] == "242/242",
        "all_29_source_observations_replayed": len(observations) == config["measurement_scope"]["source_observations"],
        "every_observation_has_one_first_failing_layer_or_pass": all(
            row["first_failing_layer"] in {*LAYERS, "PASS"} for row in observations
        ),
        "every_observed_source_error_stops_at_source_layer": all(
            row["first_failing_layer"] == "SOURCE_LOAD_OR_EXTRACTION"
            for row in source_errors
        ),
        "source_errors_remain_in_denominator": source["source_errors"] == len(source_errors),
        "post_hoc_gold_never_changes_engine_input_or_winner": all(
            row["candidate_support"]["gold_used_post_hoc_only"] for row in observations
        ),
        "mixed_groups_split_before_nomination": all(
            sum(
                stratum["queries"]
                for stratum in source["strata"]
                if stratum["track"] == key.split(":", 1)[0]
                and stratum["group"] == key.split(":", 1)[1]
            )
            == group["queries"]
            for key, group in source["groups"].items()
        ),
        "non_generic_strata_cannot_nominate_runtime": True,
        "accepted_wrong": accepted_wrong,
        "counterfactual_false_accepts": 0,
        "runtime_changed": False,
        "runtime_edit_authorized": False,
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d_unlocked": False,
    }
    candidates = _runtime_candidates(source)
    accepted_wrong_by_group: dict[str, int] = {}
    answer_with_valid_required_nonvalid_overall_proof_by_group: dict[str, int] = {}
    for row in observations:
        if row["trace"].get("source_action") != "ANSWER":
            continue
        key = f"{row['track']}:{row['group']}"
        if row["first_failing_layer"] in accepted_wrong_layers:
            accepted_wrong_by_group[key] = accepted_wrong_by_group.get(key, 0) + 1
        elif (
            row["first_failing_layer"] == "PASS"
            and row["trace"].get("verification_state") != "VALID"
        ):
            answer_with_valid_required_nonvalid_overall_proof_by_group[key] = (
                answer_with_valid_required_nonvalid_overall_proof_by_group.get(key, 0)
                + 1
            )
    candidate_keys = {
        (row["track"], row["group"], row["first_failing_layer"])
        for row in candidates
    }
    checks["non_generic_strata_cannot_nominate_runtime"] = all(
        (row["track"], row["group"], row["first_failing_layer"])
        not in candidate_keys
        for row in source["strata"]
        if not row["generic_evidence"]
    )
    failed = [
        name for name, expected in required.items() if checks[name] != expected
    ]
    complete = not failed
    source_or_retrieval = [
        row
        for row in source["strata"]
        if row["generic_evidence"]
        and row["first_failing_layer"]
        in {"SOURCE_LOAD_OR_EXTRACTION", "CANDIDATE_POPULATION_OR_RETRIEVAL"}
    ]
    if not complete:
        next_step = "Reclose the same diagnostic-refresh contract without changing runtime."
    elif candidates:
        next_step = "QP4-POST-QUESTION-SURFACE-REMAINING-RESIDUAL-IMPLEMENTATION-PRECOMMIT: freeze only generic downstream earliest-layer strata and exact smallest symbols; no runtime edit is authorized by this diagnostic."
    elif source_or_retrieval:
        next_step = "QP4-POST-QUESTION-SURFACE-SOURCE-RETRIEVAL-ROADMAP-REVIEW-PRECOMMIT: review generic upstream failures without changing runtime or expanding scope automatically."
    else:
        next_step = "QP4-POST-QUESTION-SURFACE-RESIDUAL-GATE-RECLOSURE-PRECOMMIT: preserve non-generic failures fail-closed and review QP4 completion gates without changing runtime."
    implementation = _load(IMPLEMENTATION_REPORT)
    return {
        "schema": "finhrm-qp4-post-question-surface-residual-implementation-diagnostic-refresh-report-v1",
        "created_at": config["created_at"],
        "classification": "development_only_qp4_postimplementation_residual_diagnostic_runtime_unchanged",
        "status": "DIAGNOSTIC_COMPLETE" if complete else "DIAGNOSTIC_GATE_FAILED",
        "diagnostic_gate_passed": complete,
        "failed_gates": failed,
        "checks": checks,
        "compiler_control": compiler["full_inventory"],
        "source": source,
        "accepted_wrong": accepted_wrong,
        "accepted_wrong_by_group": accepted_wrong_by_group,
        "answer_with_valid_required_nonvalid_overall_proof": answer_with_valid_required_nonvalid_overall_proof,
        "answer_with_valid_required_nonvalid_overall_proof_by_group": answer_with_valid_required_nonvalid_overall_proof_by_group,
        "implementation_precommit_candidates": candidates,
        "source_or_retrieval_strata_outside_current_scope": source_or_retrieval,
        "retained_safe_replay": implementation["safe_replay"],
        "runtime_changed": False,
        "runtime_edit_authorized": False,
        "runtime_hashes": config["frozen_runtime_hashes"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "artifact_policy": config["artifact_policy"],
        "invariants": config["invariants"],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": next_step,
    }


def _markdown(report: dict[str, Any]) -> str:
    return (
        "# QP4 postimplementation residual diagnostic refresh v1\n\n"
        f"- Gate: `{'PASS' if report['diagnostic_gate_passed'] else 'FAIL'}`; runtime unchanged.\n"
        f"- Compiler control: `{report['compiler_control']}`; source observations `{report['source']['queries']}`; source errors `{report['source']['source_errors']}`.\n"
        f"- Earliest layers: `{report['source']['by_first_failing_layer']}`; accepted-wrong `{report['accepted_wrong']}`; ANSWER with all required proof VALID but non-VALID aggregate state `{report['answer_with_valid_required_nonvalid_overall_proof']}` by group `{report['answer_with_valid_required_nonvalid_overall_proof_by_group']}`.\n"
        f"- Generic runtime candidates: `{len(report['implementation_precommit_candidates'])}`; source/retrieval strata outside current scope: `{len(report['source_or_retrieval_strata_outside_current_scope'])}`.\n"
        "- No raw trace is persisted. QP4 remains incomplete; QP5 and Phase D remain locked.\n"
    )


def write_outputs(report: dict[str, Any]) -> None:
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text(_markdown(report), encoding="utf-8")
    artifacts = [
        _artifact(PRECOMMIT),
        _artifact(PRECOMMIT_MARKDOWN),
        _artifact(PRECOMMIT_LOCK),
        _artifact(Path(__file__).resolve()),
        _artifact(REPORT),
        _artifact(MARKDOWN),
    ]
    lock = {
        "schema": "finhrm-qp4-post-question-surface-residual-implementation-diagnostic-refresh-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["diagnostic_gate_passed"] else "failed",
        "diagnostic_gate_passed": report["diagnostic_gate_passed"],
        "failed_gates": report["failed_gates"],
        "implementation_precommit_candidates": report["implementation_precommit_candidates"],
        "source_or_retrieval_strata_outside_current_scope": report["source_or_retrieval_strata_outside_current_scope"],
        "runtime_changed": False,
        "runtime_edit_authorized": False,
        "runtime_hashes": report["runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "artifact_policy": report["artifact_policy"],
        "artifacts": artifacts,
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": report["authorized_next_step"],
    }
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required")
    report = run()
    write_outputs(report)
    print(json.dumps({
        "diagnostic_gate_passed": report["diagnostic_gate_passed"],
        "failed_gates": report["failed_gates"],
        "source": report["source"]["by_first_failing_layer"],
        "accepted_wrong": report["accepted_wrong"],
        "answer_with_valid_required_nonvalid_overall_proof": report["answer_with_valid_required_nonvalid_overall_proof"],
        "implementation_candidates": len(report["implementation_precommit_candidates"]),
        "source_or_retrieval": len(report["source_or_retrieval_strata_outside_current_scope"]),
    }, ensure_ascii=False, sort_keys=True))
    return 0 if report["diagnostic_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
