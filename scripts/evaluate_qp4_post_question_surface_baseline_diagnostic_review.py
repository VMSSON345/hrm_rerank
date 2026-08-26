#!/usr/bin/env python3
"""Run the frozen no-runtime QP4 post-question-surface diagnostic review."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from scripts.evaluate_qp4_post_unit_scale_baseline_diagnostic_review import (
    _CandidateCaptureBundle,
    _actual_plan_passes,
    _actual_support,
    _add_support,
    _classify_actual,
    _classify_add,
    _run_question,
)
from scripts.evaluate_qp4_question_surface_implementation import compiler_summary


ROOT = Path(__file__).resolve().parents[1]
PRECOMMIT = ROOT / "config" / "finhrm_qp4_post_question_surface_baseline_diagnostic_review_precommit_v1.json"
PRECOMMIT_REPORT = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_diagnostic_review_precommit_v1.json"
PRECOMMIT_LOCK = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_diagnostic_review_precommit_lock_v1.json"
IMPLEMENTATION_CONFIG = ROOT / "config" / "finhrm_qp4_question_surface_implementation_v1.json"
ACTUAL_CONFIG = ROOT / "config" / "finhrm_qp4_actual_plan_ratio_direction_source_cohort_v1.json"
ADD_BASELINE_RAW = ROOT / "reports" / "finhrm_qp4_post_question_surface_add_source_baseline_v1.json"
ACTUAL_BASELINE_RAW = ROOT / "reports" / "finhrm_qp4_post_question_surface_actual_plan_source_baseline_v1.json"
SAFE_REPLAY = ROOT / "reports" / "finhrm_qp4_post_question_surface_safe_replay_v1.json"
RAW = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_diagnostic_review_raw_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_diagnostic_review_v1.json"
MARKDOWN = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_diagnostic_review_v1.md"
LOCK = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_diagnostic_review_lock_v1.json"
TEST = ROOT / "tests" / "test_qp4_post_question_surface_baseline_diagnostic_review.py"

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
    report = _load(PRECOMMIT_REPORT)
    lock = _load(PRECOMMIT_LOCK)
    if not report.get("precommit_gate_passed") or not lock.get("precommit_gate_passed"):
        raise ValueError("Diagnostic precommit is not closed")
    if lock.get("authorized_next_step") != config.get("authorized_next_step_after_lock"):
        raise ValueError("Diagnostic authorization mismatch")
    contract = config["diagnostic_contract"]
    if contract["runtime_file_whitelist"] or lock["runtime_file_whitelist_for_next_step"]:
        raise ValueError("Diagnostic runtime whitelist must be empty")
    if contract["allowed_new_evaluator"] != Path(__file__).resolve().relative_to(ROOT).as_posix():
        raise ValueError("This evaluator is not pre-authorized")
    for name, parent in config["parents"].items():
        path = ROOT / parent["path"]
        if not path.is_file() or _digest(path) != parent["sha256"]:
            raise ValueError(f"Frozen parent hash mismatch: {name}")
    checkpoint = ROOT / config["checkpoint"]["path"]
    if _digest(checkpoint) != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    for relative, expected in config["frozen_runtime_hashes"].items():
        if _digest(ROOT / relative) != expected:
            raise ValueError(f"Runtime hash mismatch: {relative}")


def _compiler_control() -> dict[str, Any]:
    compiler = compiler_summary(_load(IMPLEMENTATION_CONFIG))
    records = (
        compiler["canonical"]["total"]
        + compiler["ambiguity"]["total"]
        + compiler["frozen_surface"]["total"]
        + compiler["corrected_hold_ahead"]["total"]
        + compiler["minimal_pairs"]["total"]
    )
    exact = (
        compiler["canonical"]["exact"]
        + compiler["ambiguity"]["exact"]
        + compiler["frozen_surface"]["exact"]
        + compiler["corrected_hold_ahead"]["exact"]
        + compiler["minimal_pairs"]["detected"]
    )
    return {
        "records": records,
        "exact": exact,
        "canonical": f"{compiler['canonical']['exact']}/{compiler['canonical']['total']}",
        "ambiguity": f"{compiler['ambiguity']['exact']}/{compiler['ambiguity']['total']}",
        "frozen_surface": f"{compiler['frozen_surface']['exact']}/{compiler['frozen_surface']['total']}",
        "corrected_hold_ahead": f"{compiler['corrected_hold_ahead']['exact']}/{compiler['corrected_hold_ahead']['total']}",
        "minimal_pairs": f"{compiler['minimal_pairs']['detected']}/{compiler['minimal_pairs']['total']}",
    }


def _set_candidate_support(trace: dict[str, Any], support: dict[str, Any]) -> None:
    trace["candidate_support_signature"] = support


def _add_surface_result(
    engine: Any,
    capture: _CandidateCaptureBundle,
    row: dict[str, Any],
    seconds: int,
) -> dict[str, Any]:
    from hrm_finance_qa.question_planner import compile_question_plan
    from scripts.evaluate_qp4_development_baseline import _semantic_projection

    expected_plan = _semantic_projection(compile_question_plan(row["parent_question"]))
    observed_plan = _semantic_projection(compile_question_plan(row["question"]))
    plan_passed = observed_plan == expected_plan
    answer, candidates, trace = _run_question(
        engine,
        capture,
        row["question"],
        ROOT / row["source_path"],
        seconds,
    )
    support, matches = _add_support(candidates, row["parent_signature"]["evidence"])
    candidate_support = {
        "all_expected_physical_evidence_observed": support,
        "matched_expected_candidates": matches,
        "gold_used_post_hoc_only": True,
    }
    _set_candidate_support(trace, candidate_support)
    plan_trace = {"expected": expected_plan, "observed": observed_plan, "passed": plan_passed}
    trace["semantic_plan_projection"] = plan_trace
    layer = (
        "SOURCE_LOAD_OR_EXTRACTION"
        if trace["source_outcome"] == "ERROR"
        else _classify_add(
            plan_passed,
            answer,
            trace,
            support,
            row["parent_signature"],
        )
    )
    return {
        "track": "ADD_SURFACE",
        "group": row["category"],
        "company_key": row["company_key"],
        "source_key": row["source_key"],
        "layout_family": row["layout_family"],
        "selection_basis": "all records in a frozen failed ADD surface category",
        "semantic_plan_projection": plan_trace,
        "candidate_support": candidate_support,
        "trace": trace,
        "first_failing_layer": layer,
    }


def _add_counterfactual_error_result(
    engine: Any,
    capture: _CandidateCaptureBundle,
    row: dict[str, Any],
    seconds: int,
) -> dict[str, Any]:
    from hrm_finance_qa.question_planner import compile_question_plan
    from scripts.evaluate_qp4_development_baseline import _dimension_projection

    parent_projection = _dimension_projection(
        compile_question_plan(row["parent_question"]), row["dimension"]
    )
    mutated_projection = _dimension_projection(
        compile_question_plan(row["question"]), row["dimension"]
    )
    plan_passed = parent_projection != mutated_projection
    _, candidates, trace = _run_question(
        engine,
        capture,
        row["question"],
        ROOT / row["source_path"],
        seconds,
    )
    support, matches = _add_support(candidates, row["parent_signature"]["evidence"])
    candidate_support = {
        "all_parent_physical_evidence_observed": support,
        "matched_parent_candidates": matches,
        "gold_used_post_hoc_only": True,
    }
    _set_candidate_support(trace, candidate_support)
    plan_trace = {
        "dimension": row["dimension"],
        "parent": parent_projection,
        "mutated": mutated_projection,
        "passed": plan_passed,
    }
    trace["semantic_plan_projection"] = plan_trace
    if trace["source_outcome"] == "ERROR":
        layer = "SOURCE_LOAD_OR_EXTRACTION"
    elif not plan_passed:
        layer = "QUESTION_PLAN"
    else:
        layer = "PASS"
    return {
        "track": "ADD_COUNTERFACTUAL_SOURCE_ERROR",
        "group": row["dimension"],
        "company_key": row["company_key"],
        "source_key": row["source_key"],
        "layout_family": row["layout_family"],
        "selection_basis": "every source-error observation retained by the frozen ADD counterfactual baseline",
        "semantic_plan_projection": plan_trace,
        "candidate_support": candidate_support,
        "trace": trace,
        "first_failing_layer": layer,
    }


def _actual_result(
    engine: Any,
    capture: _CandidateCaptureBundle,
    baseline_row: dict[str, Any],
    anchor: dict[str, Any],
    label: str,
    seconds: int,
) -> dict[str, Any]:
    from hrm_finance_qa.question_planner import compile_question_plan
    from scripts.evaluate_qp4_actual_plan_source_cohort import _expected_value
    from scripts.evaluate_qp4_development_baseline import _semantic_projection

    pair = anchor["pairs"][baseline_row["dimension"]]
    question = pair[f"{label}_question"]
    program = pair[f"{label}_program"]
    projection = _semantic_projection(compile_question_plan(question))
    plan_passed = _actual_plan_passes(projection, program)
    answer, candidates, trace = _run_question(
        engine,
        capture,
        question,
        ROOT / anchor["source_path"],
        seconds,
    )
    support, matches = _actual_support(candidates, anchor, program["operands"])
    expected_value = _expected_value(anchor, program)
    candidate_support = {
        "all_expected_value_page_support_observed": support,
        "matched_expected_candidates": matches,
        "gold_used_post_hoc_only": True,
    }
    _set_candidate_support(trace, candidate_support)
    plan_trace = {
        "observed": projection,
        "expected_operation": program["operation"],
        "expected_operands": program["operands"],
        "passed": plan_passed,
    }
    trace["semantic_plan_projection"] = plan_trace
    layer = (
        "SOURCE_LOAD_OR_EXTRACTION"
        if trace["source_outcome"] == "ERROR"
        else _classify_actual(
            plan_passed,
            answer,
            trace,
            support,
            anchor,
            program,
            expected_value,
        )
    )
    return {
        "track": "ACTUAL_PLAN",
        "group": baseline_row["dimension"],
        "query_variant": label,
        "company_key": baseline_row["company_key"],
        "source_key": baseline_row["source_sha256"],
        "layout_family": baseline_row["document_layout_family"],
        "table_layout_family": baseline_row["table_layout_family"],
        "selection_basis": "all base and mutated queries in a frozen failed ACTUAL/PLAN dimension",
        "semantic_plan_projection": plan_trace,
        "candidate_support": candidate_support,
        "expected_execution": {
            "operation": program["operation"],
            "value": str(expected_value),
            "action": "ANSWER",
        },
        "trace": trace,
        "first_failing_layer": layer,
    }


def run_diagnostic(config: dict[str, Any], raw_path: Path = RAW) -> dict[str, Any]:
    from hrm_finance_qa.document_hrm import DocumentHRMBundle
    from hrm_finance_qa.document_reasoning_engine import DocumentHRMReasoningEngine

    _verify_precommit(config)
    checkpoint = ROOT / config["checkpoint"]["path"]
    capture = _CandidateCaptureBundle(DocumentHRMBundle.load(checkpoint, device="cpu"))
    engine = DocumentHRMReasoningEngine(capture)
    seconds = int(config["diagnostic_contract"]["per_query_timeout_seconds"])
    source_results: list[dict[str, Any]] = []
    position = 0

    add_raw = _load(ADD_BASELINE_RAW)
    for row in [item for item in add_raw["surface_results"] if not item["invariant"]]:
        position += 1
        print(f"[{position}/40] ADD_SURFACE {row['category']}", flush=True)
        source_results.append(_add_surface_result(engine, capture, row, seconds))
    for row in [item for item in add_raw["counterfactual_results"] if "error" in item]:
        position += 1
        print(f"[{position}/40] ADD_COUNTERFACTUAL_SOURCE_ERROR {row['dimension']}", flush=True)
        source_results.append(_add_counterfactual_error_result(engine, capture, row, seconds))

    actual_config = _load(ACTUAL_CONFIG)
    anchors = {row["source_sha256"]: row for row in actual_config["anchors"]}
    actual_raw = _load(ACTUAL_BASELINE_RAW)
    for baseline_row in actual_raw["results"]:
        anchor = anchors[baseline_row["source_sha256"]]
        for label in ("base", "mutated"):
            position += 1
            print(f"[{position}/40] ACTUAL_PLAN {baseline_row['dimension']} {label}", flush=True)
            source_results.append(
                _actual_result(engine, capture, baseline_row, anchor, label, seconds)
            )

    raw = {
        "schema": "finhrm-qp4-post-question-surface-baseline-diagnostic-review-raw-v1",
        "created_at": config["created_at"],
        "classification": "development_only_qp4_residual_diagnostic_runtime_unchanged",
        "inputs": {
            "precommit": PRECOMMIT.relative_to(ROOT).as_posix(),
            "precommit_sha256": _digest(PRECOMMIT),
            "checkpoint_sha256": _digest(checkpoint),
            "source_run_boundary": ["question", "source PDF path"],
            "gold_candidate_or_program_passed_to_engine": False,
            "gold_used_only_after_observation": True,
            "mineru_auto_generate": False,
            "per_query_timeout_seconds": seconds,
        },
        "compiler_control": _compiler_control(),
        "source_results": source_results,
        "runtime_hashes": {
            relative: _digest(ROOT / relative)
            for relative in config["frozen_runtime_hashes"]
        },
        "invariants": config["invariants"],
    }
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return raw


def _source_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["track"], row["group"])].append(row)
    groups: dict[str, Any] = {}
    strata: list[dict[str, Any]] = []
    pass_rates: list[float] = []
    for (track, group), values in sorted(grouped.items()):
        key = f"{track}:{group}"
        layers = Counter(row["first_failing_layer"] for row in values)
        pass_rate = layers.get("PASS", 0) / len(values)
        pass_rates.append(pass_rate)
        groups[key] = {
            "queries": len(values),
            "companies": len({row["company_key"] for row in values}),
            "sources": len({row["source_key"] for row in values}),
            "real_layouts": len({row["layout_family"] for row in values}),
            "pass_rate": pass_rate,
            "by_first_failing_layer": dict(sorted(layers.items())),
            "mixed_first_failure_group": len(layers) > 1,
        }
        for layer in sorted(layers):
            layer_rows = [row for row in values if row["first_failing_layer"] == layer]
            companies = len({row["company_key"] for row in layer_rows})
            sources = len({row["source_key"] for row in layer_rows})
            layouts = len({row["layout_family"] for row in layer_rows})
            strata.append(
                {
                    "track": track,
                    "group": group,
                    "first_failing_layer": layer,
                    "queries": len(layer_rows),
                    "companies": companies,
                    "sources": sources,
                    "real_layouts": layouts,
                    "generic_evidence": companies >= 3 and sources >= 3 and layouts >= 3,
                }
            )
    return {
        "queries": len(rows),
        "source_errors": sum(row["trace"]["source_outcome"] == "ERROR" for row in rows),
        "by_first_failing_layer": dict(sorted(Counter(row["first_failing_layer"] for row in rows).items())),
        "groups": groups,
        "strata": strata,
        "macro_group_pass_rate": sum(pass_rates) / len(pass_rates) if pass_rates else 0.0,
        "worst_group_pass_rate": min(pass_rates) if pass_rates else 0.0,
    }


def _runtime_candidates(source: dict[str, Any]) -> list[dict[str, Any]]:
    file_by_layer = {
        "QUESTION_PLAN": ["src/hrm_finance_qa/question_planner.py"],
        "SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS": ["src/hrm_finance_qa/program_synthesis.py"],
        "SYMBOLIC_PROOF_OR_ACTION": [
            "src/hrm_finance_qa/symbolic_verifier.py",
            "src/hrm_finance_qa/direction_contract.py",
        ],
    }
    output = []
    for row in source["strata"]:
        if not row["generic_evidence"] or row["first_failing_layer"] not in file_by_layer:
            continue
        output.append(
            row
            | {
                "eligible_existing_whitelist_files": file_by_layer[row["first_failing_layer"]],
                "runtime_edit_authorized": False,
                "new_precommit_required": True,
            }
        )
    return output


def build_report(config_path: Path = PRECOMMIT, raw_path: Path = RAW) -> dict[str, Any]:
    config = _load(config_path)
    _verify_precommit(config)
    raw = _load(raw_path)
    if raw.get("schema") != "finhrm-qp4-post-question-surface-baseline-diagnostic-review-raw-v1":
        raise ValueError("Unexpected diagnostic raw schema")
    if raw["inputs"]["precommit_sha256"] != _digest(config_path):
        raise ValueError("Diagnostic raw precommit hash mismatch")
    expected_runtime = {
        relative: _digest(ROOT / relative)
        for relative in config["frozen_runtime_hashes"]
    }
    if raw["runtime_hashes"] != expected_runtime:
        raise ValueError("Runtime changed during diagnostic")

    rows = raw["source_results"]
    source = _source_summary(rows)
    required_trace = set(config["diagnostic_contract"]["trace_fields"])
    checks = {
        "compiler_inventory_retained_242_of_242": raw["compiler_control"]["exact"] == raw["compiler_control"]["records"] == 242,
        "all_40_source_observations_replayed_and_classified": len(rows) == 40,
        "every_observation_has_exactly_one_first_failing_layer_or_pass": all(row["first_failing_layer"] in {*LAYERS, "PASS"} for row in rows),
        "all_12_source_errors_classified_as_source_load_or_extraction": source["source_errors"] == 12 and all(
            row["first_failing_layer"] == "SOURCE_LOAD_OR_EXTRACTION"
            for row in rows
            if row["trace"]["source_outcome"] == "ERROR"
        ),
        "source_errors_not_relabelled_as_abstention_pass": all(
            row["trace"]["source_outcome"] != "ERROR"
            or row["first_failing_layer"] == "SOURCE_LOAD_OR_EXTRACTION"
            for row in rows
        ),
        "post_hoc_gold_never_changes_engine_input_or_winner": (
            raw["inputs"]["gold_candidate_or_program_passed_to_engine"] is False
            and raw["inputs"]["gold_used_only_after_observation"] is True
            and all(row["candidate_support"]["gold_used_post_hoc_only"] for row in rows)
        ),
        "per_group_company_source_layout_counts_reported": all(
            {"companies", "sources", "real_layouts"} <= set(row)
            for row in source["groups"].values()
        ),
        "macro_and_worst_group_metrics_reported": (
            "macro_group_pass_rate" in source and "worst_group_pass_rate" in source
        ),
        "mixed_failure_groups_split_before_runtime_nomination": all(
            sum(item["queries"] for item in source["strata"] if item["track"] == key.split(":", 1)[0] and item["group"] == key.split(":", 1)[1]) == group["queries"]
            for key, group in source["groups"].items()
        ),
        "runtime_changed": False,
        "runtime_edit_authorized": False,
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d_unlocked": False,
    }
    if not all(required_trace <= set(row["trace"]) for row in rows):
        checks["every_observation_has_exactly_one_first_failing_layer_or_pass"] = False
    failed = [
        name
        for name, observed in checks.items()
        if observed != config["diagnostic_gates"][name]
    ]

    runtime_candidates = _runtime_candidates(source)
    source_or_retrieval = [
        row
        for row in source["strata"]
        if row["generic_evidence"]
        and row["first_failing_layer"]
        in {"SOURCE_LOAD_OR_EXTRACTION", "CANDIDATE_POPULATION_OR_RETRIEVAL"}
    ]
    if failed:
        next_step = "QP4-POST-QUESTION-SURFACE-DIAGNOSTIC-RECLOSURE-PRECOMMIT: resolve diagnostic contract failures without changing runtime."
    elif runtime_candidates:
        next_step = "QP4-POST-QUESTION-SURFACE-RESIDUAL-IMPLEMENTATION-PRECOMMIT: freeze only the generic earliest-layer strata and exact smallest runtime symbols; no runtime edit is authorized by this diagnostic."
    elif source_or_retrieval:
        next_step = "QP4-POST-QUESTION-SURFACE-SOURCE-RETRIEVAL-ROADMAP-REVIEW-PRECOMMIT: review generic upstream strata without expanding runtime scope or changing runtime."
    else:
        next_step = "QP4-POST-QUESTION-SURFACE-DIAGNOSTIC-RECLOSURE-PRECOMMIT: preserve fail-closed residuals because no generic runtime candidate was isolated."

    safe = _load(SAFE_REPLAY)["results"]
    diagnostic_complete = not failed
    return {
        "schema": "finhrm-qp4-post-question-surface-baseline-diagnostic-review-report-v1",
        "created_at": config["created_at"],
        "classification": "development_only_qp4_residual_diagnostic_runtime_unchanged",
        "status": "DIAGNOSTIC_COMPLETE_RUNTIME_UNCHANGED" if diagnostic_complete else "DIAGNOSTIC_GATE_FAILED_RUNTIME_UNCHANGED",
        "diagnostic_complete": diagnostic_complete,
        "diagnostic_gate_passed": diagnostic_complete,
        "failed_gates": failed,
        "checks": checks,
        "compiler_control": raw["compiler_control"],
        "source": source,
        "first_failing_layer_order": list(LAYERS),
        "implementation_precommit_candidates": runtime_candidates,
        "source_or_retrieval_strata_outside_current_scope": source_or_retrieval,
        "runtime_edit_authorized": False,
        "new_implementation_precommit_required": bool(runtime_candidates),
        "safe_replay_retained": {
            "actions": safe["gold_actions"],
            "execution": f"{safe['gold_execution_correct']}/{safe['gold_total']}",
            "permutation": f"{safe['permutation_exact']}/{safe['permutation_total']}",
            "counterfactuals": safe["counterfactual_generated"],
            "counterfactual_false_accepts": safe["counterfactual_false_accepts"],
        },
        "runtime_changed": False,
        "runtime_hashes": config["frozen_runtime_hashes"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "invariants": config["invariants"],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "raw_diagnostic": _artifact(raw_path),
        "authorized_next_step": next_step,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# QP4 post-question-surface baseline diagnostic review v1",
        "",
        f"- Diagnostic gate: `{'PASS' if report['diagnostic_gate_passed'] else 'FAIL'}`; runtime unchanged.",
        f"- Compiler control: `{report['compiler_control']['exact']}/{report['compiler_control']['records']}`.",
        f"- Source observations: `{report['source']['queries']}`; source errors `{report['source']['source_errors']}`.",
        f"- First failing layers: `{report['source']['by_first_failing_layer']}`.",
        f"- Macro/worst group pass rate: `{report['source']['macro_group_pass_rate']:.3f}/{report['source']['worst_group_pass_rate']:.3f}`.",
        "",
        "| Diagnostic group | Queries | Companies | PDFs | Real layouts | First-layer strata |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, row in report["source"]["groups"].items():
        lines.append(
            f"| {name} | {row['queries']} | {row['companies']} | {row['sources']} | {row['real_layouts']} | {row['by_first_failing_layer']} |"
        )
    lines.extend(
        [
            "",
            "Gold was used only after observation. Source errors are failures; safe abstention is not an answer. Rank, score and confidence are not proof.",
            "Runtime edit remains unauthorized. QP4 is incomplete; QP5 and Phase D remain locked.",
            "",
            "Next: " + report["authorized_next_step"],
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(report: dict[str, Any]) -> None:
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text(_markdown(report), encoding="utf-8")
    artifacts = (
        PRECOMMIT,
        PRECOMMIT_REPORT,
        PRECOMMIT_LOCK,
        Path(__file__).resolve(),
        TEST,
        RAW,
        REPORT,
        MARKDOWN,
    )
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    lock = {
        "schema": "finhrm-qp4-post-question-surface-baseline-diagnostic-review-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["diagnostic_complete"] else "failed",
        "diagnostic_complete": report["diagnostic_complete"],
        "diagnostic_gate_passed": report["diagnostic_gate_passed"],
        "failed_gates": report["failed_gates"],
        "implementation_precommit_candidates": report["implementation_precommit_candidates"],
        "runtime_changed": False,
        "runtime_edit_authorized": False,
        "runtime_hashes": report["runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "artifacts": [_artifact(path) for path in artifacts],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": report["authorized_next_step"],
    }
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-diagnostic", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    config = _load(PRECOMMIT)
    _verify_precommit(config)
    if args.run_diagnostic:
        run_diagnostic(config)
    report = build_report()
    if args.write:
        write_outputs(report)
    print(
        json.dumps(
            {
                "diagnostic_complete": report["diagnostic_complete"],
                "failed_gates": report["failed_gates"],
                "source_first_layers": report["source"]["by_first_failing_layer"],
                "implementation_precommit_candidates": len(report["implementation_precommit_candidates"]),
                "runtime_edit_authorized": report["runtime_edit_authorized"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["diagnostic_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
