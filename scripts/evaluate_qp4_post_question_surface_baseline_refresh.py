#!/usr/bin/env python3
"""Run the frozen QP4 post-question-surface development baseline refresh."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from scripts.evaluate_qp4_post_unit_scale_development_baseline_refresh import (
    _actual_plan_summary,
)
from scripts.evaluate_qp4_question_surface_implementation import (
    compiler_summary,
    run_regression,
    run_source_replay,
    source_summary,
)


ROOT = Path(__file__).resolve().parents[1]
PRECOMMIT = ROOT / "config" / "finhrm_qp4_post_question_surface_baseline_refresh_precommit_v1.json"
PRECOMMIT_REPORT = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_refresh_precommit_v1.json"
PRECOMMIT_LOCK = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_refresh_precommit_lock_v1.json"
IMPLEMENTATION_CONFIG = ROOT / "config" / "finhrm_qp4_question_surface_implementation_v1.json"
ACTUAL_PLAN_CONFIG = ROOT / "config" / "finhrm_qp4_actual_plan_ratio_direction_source_cohort_v1.json"
SAFE_PRECOMMIT = ROOT / "config" / "finhrm_qp3_core_s4_safe_replay_precommit_v1.json"
ADD_RAW = ROOT / "reports" / "finhrm_qp4_post_question_surface_add_source_baseline_v1.json"
ACTUAL_PLAN_RAW = ROOT / "reports" / "finhrm_qp4_post_question_surface_actual_plan_source_baseline_v1.json"
SAFE_REPLAY = ROOT / "reports" / "finhrm_qp4_post_question_surface_safe_replay_v1.json"
REPORT = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_refresh_v1.json"
MARKDOWN = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_refresh_v1.md"
LOCK = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_refresh_lock_v1.json"
TEST = ROOT / "tests" / "test_qp4_post_question_surface_baseline_refresh.py"
PROTECTED = re.compile(r"(?:^|[/_.-])(test|holdout|final|blind|validation|sealed)(?:[/_.-]|$)", re.I)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if PROTECTED.search(path.resolve().as_posix()):
        raise ValueError(f"Protected input is forbidden: {path}")
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
        raise ValueError("Post-question-surface refresh precommit is not closed")
    if lock.get("authorized_next_step") != config.get("authorized_next_step_after_lock"):
        raise ValueError("Refresh authorization mismatch")
    scope = config["refresh_scope"]
    if scope["runtime_file_whitelist"]:
        raise ValueError("Measurement refresh must have an empty runtime whitelist")
    if scope["allowed_new_evaluator"] != Path(__file__).resolve().relative_to(ROOT).as_posix():
        raise ValueError("This evaluator is not the pre-authorized evaluator")
    if not config["generalization_contract"]["evaluate_all_frozen_eligible_records"]:
        raise ValueError("Frozen eligible records may not be selected or subsampled")
    if config["generalization_contract"]["report_specific_selection_or_exclusion"]:
        raise ValueError("Report-specific selection is forbidden")
    for parent in config["parents"].values():
        path = ROOT / parent["path"]
        if not path.is_file() or _digest(path) != parent["sha256"]:
            raise ValueError(f"Frozen parent hash mismatch: {parent['path']}")
    checkpoint = ROOT / config["checkpoint"]["path"]
    if _digest(checkpoint) != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    for relative, expected in config["frozen_runtime_hashes"].items():
        if _digest(ROOT / relative) != expected:
            raise ValueError(f"Frozen runtime hash mismatch: {relative}")
    if lock.get("runtime_hashes") != config["frozen_runtime_hashes"]:
        raise ValueError("Precommit lock runtime snapshot mismatch")
    if lock.get("checkpoint_sha256") != config["checkpoint"]["sha256"]:
        raise ValueError("Precommit lock checkpoint mismatch")


def _derived_actual_plan_config(config: dict[str, Any]) -> dict[str, Any]:
    derived = deepcopy(_load(ACTUAL_PLAN_CONFIG))
    derived["runtime_hashes"] = {
        relative: config["frozen_runtime_hashes"][relative]
        for relative in derived["runtime_hashes"]
    }
    return derived


def run_measurements(config: dict[str, Any]) -> dict[str, Any]:
    from scripts.evaluate_qp3_core_s4_safe_replay import build_report as build_safe_replay
    from scripts.evaluate_qp4_actual_plan_source_cohort import run_source_baseline

    _verify_precommit(config)
    implementation_config = _load(IMPLEMENTATION_CONFIG)
    run_source_replay(implementation_config, ADD_RAW)
    run_source_baseline(_derived_actual_plan_config(config), ACTUAL_PLAN_RAW)
    safe = build_safe_replay(SAFE_PRECOMMIT)
    SAFE_REPLAY.write_text(json.dumps(safe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    regression = run_regression()
    _verify_precommit(config)
    return regression


def _boolean_group_metrics(
    rows: list[dict[str, Any]],
    group_key: str,
    passed: Any,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    by_group: dict[str, Any] = {}
    rates: list[float] = []
    for name, values in sorted(grouped.items()):
        count = len(values)
        passed_count = sum(bool(passed(row)) for row in values)
        rate = passed_count / count if count else 0.0
        rates.append(rate)
        by_group[name] = {"observations": count, "passed": passed_count, "pass_rate": rate}
    return {
        "groups": len(by_group),
        "by_group": by_group,
        "macro_pass_rate": sum(rates) / len(rates) if rates else 0.0,
        "worst_group_pass_rate": min(rates) if rates else 0.0,
    }


def _add_group_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    surface = raw["surface_results"]
    counterfactual = raw["counterfactual_results"]
    return {
        "surface_by_company": _boolean_group_metrics(surface, "company_key", lambda row: row["invariant"]),
        "surface_by_layout": _boolean_group_metrics(surface, "layout_family", lambda row: row["invariant"]),
        "counterfactual_by_company": _boolean_group_metrics(
            counterfactual,
            "company_key",
            lambda row: "error" not in row and not row["old_winner_false_accept"],
        ),
        "counterfactual_by_layout": _boolean_group_metrics(
            counterfactual,
            "layout_family",
            lambda row: "error" not in row and not row["old_winner_false_accept"],
        ),
    }


def _actual_group_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    rows = raw["results"]
    return {
        "paired_gold_by_company": _boolean_group_metrics(rows, "company_key", lambda row: row["paired_gold_answer"]),
        "paired_gold_by_document_layout": _boolean_group_metrics(
            rows,
            "document_layout_family",
            lambda row: row["paired_gold_answer"],
        ),
        "paired_gold_by_table_layout": _boolean_group_metrics(
            rows,
            "table_layout_family",
            lambda row: row["paired_gold_answer"],
        ),
        "counterfactual_safety_by_company": _boolean_group_metrics(
            rows,
            "company_key",
            lambda row: not row["old_winner_false_accept"],
        ),
    }


def _source_error_summary(add_raw: dict[str, Any], actual_raw: dict[str, Any]) -> dict[str, Any]:
    add_surface_errors = [row for row in add_raw["surface_results"] if "error" in row]
    add_counterfactual_errors = [row for row in add_raw["counterfactual_results"] if "error" in row]
    actual_base_errors = [row for row in actual_raw["results"] if "error" in row["base"]]
    actual_mutated_errors = [row for row in actual_raw["results"] if "error" in row["mutated"]]
    return {
        "add_surface": len(add_surface_errors),
        "add_counterfactual": len(add_counterfactual_errors),
        "actual_plan_base": len(actual_base_errors),
        "actual_plan_mutated": len(actual_mutated_errors),
        "total_failed_observations": (
            len(add_surface_errors)
            + len(add_counterfactual_errors)
            + len(actual_base_errors)
            + len(actual_mutated_errors)
        ),
        "all_errors_retained_in_denominator": all(not row["invariant"] for row in add_surface_errors)
        and all(not row["base"]["gold_answer_match"] for row in actual_base_errors)
        and all(not row["mutated"]["gold_answer_match"] for row in actual_mutated_errors),
    }


def _validate_raw_runtime(raw: dict[str, Any], config: dict[str, Any], paths: list[str]) -> None:
    expected = {relative: config["frozen_runtime_hashes"][relative] for relative in paths}
    if raw.get("runtime_hashes") != expected:
        raise ValueError("Raw measurement runtime snapshot mismatch")


def build_report(
    config_path: Path = PRECOMMIT,
    add_raw_path: Path = ADD_RAW,
    actual_raw_path: Path = ACTUAL_PLAN_RAW,
    safe_path: Path = SAFE_REPLAY,
    regression: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = _load(config_path)
    _verify_precommit(config)
    implementation_config = _load(IMPLEMENTATION_CONFIG)
    actual_config = _derived_actual_plan_config(config)
    add_raw = _load(add_raw_path)
    actual_raw = _load(actual_raw_path)
    safe = _load(safe_path)

    if add_raw.get("schema") != "finhrm-qp4-source-backed-baseline-v1":
        raise ValueError("Unexpected ADD raw schema")
    if actual_raw.get("schema") != "finhrm-qp4-actual-plan-source-baseline-v1":
        raise ValueError("Unexpected ACTUAL/PLAN raw schema")
    if safe.get("schema") != "finhrm-qp3-core-s4-safe-replay-v1":
        raise ValueError("Unexpected safe replay schema")
    if add_raw["inputs"]["cohort_config_sha256"] != _digest(ROOT / config["parents"]["add_source_cohort"]["path"]):
        raise ValueError("ADD raw does not use the frozen cohort")
    if actual_raw["inputs"]["config_sha256"] != _digest(ACTUAL_PLAN_CONFIG):
        raise ValueError("ACTUAL/PLAN raw does not use the frozen cohort")
    _validate_raw_runtime(add_raw, config, list(add_raw["runtime_hashes"]))
    _validate_raw_runtime(actual_raw, config, list(actual_config["runtime_hashes"]))
    _validate_raw_runtime(safe, config, list(safe["runtime_hashes"]))

    compiler = compiler_summary(implementation_config)
    add = source_summary(implementation_config, add_raw_path)
    actual = _actual_plan_summary(actual_config, actual_raw)
    add_groups = _add_group_metrics(add_raw)
    actual_groups = _actual_group_metrics(actual_raw)
    errors = _source_error_summary(add_raw, actual_raw)
    safe_results = safe["results"]
    scope = config["refresh_scope"]
    completion_targets = config["measurement_completion_gates"]
    semantic_targets = config["qp4_semantic_gates"]
    regression_result = regression or {
        "observed": 0,
        "exit_code": None,
        "passed": False,
        "status": "not_run",
    }

    compiler_records = (
        compiler["canonical"]["total"]
        + compiler["ambiguity"]["total"]
        + compiler["frozen_surface"]["total"]
        + compiler["corrected_hold_ahead"]["total"]
        + compiler["minimal_pairs"]["total"]
    )
    add_attempted = len(add_raw["surface_results"]) + len(add_raw["counterfactual_results"])
    actual_attempted = len(actual_raw["results"]) * scope["actual_plan_source_tier"]["queries_per_anchor_dimension"]
    completion_checks = {
        "compiler_records_scored": f"{compiler_records}/{compiler_records}" == completion_targets["compiler_records_scored"],
        "add_source_queries_attempted": f"{add_attempted}/{scope['add_source_tier']['total_queries']}" == completion_targets["add_source_queries_attempted"],
        "actual_plan_source_queries_attempted": f"{actual_attempted}/{scope['actual_plan_source_tier']['total_queries']}" == completion_targets["actual_plan_source_queries_attempted"],
        "combined_source_queries_attempted": f"{add_attempted + actual_attempted}/{scope['combined_source_queries']}" == completion_targets["combined_source_queries_attempted"],
        "all_errors_and_timeouts_recorded_as_failures": errors["all_errors_retained_in_denominator"],
        "per_company_per_layout_and_residual_breakdown_present": bool(add_groups and actual_groups and add["surface_residuals"]),
        "compatible_affected_regression": regression_result.get("passed") is True and regression_result.get("observed") == 354,
        "safe_replay_executed": safe["safe_replay_gate_passed"] is True,
    }

    semantic_checks = {
        "compiler_core_family_coverage": f"{compiler['canonical']['exact']}/{compiler['canonical']['total']}" == semantic_targets["compiler_core_family_coverage"],
        "compiler_ambiguity_controls": f"{compiler['ambiguity']['exact']}/{compiler['ambiguity']['total']}" == semantic_targets["compiler_ambiguity_controls"],
        "compiler_surface_semantic_invariance": f"{compiler['frozen_surface']['exact']}/{compiler['frozen_surface']['total']}" == semantic_targets["compiler_surface_semantic_invariance"],
        "compiler_hold_ahead_semantic_invariance": f"{compiler['corrected_hold_ahead']['exact']}/{compiler['corrected_hold_ahead']['total']}" == semantic_targets["compiler_hold_ahead_semantic_invariance"],
        "compiler_counterfactual_detection": f"{compiler['minimal_pairs']['detected']}/{compiler['minimal_pairs']['total']}" == semantic_targets["compiler_counterfactual_detection"],
        "add_source_prior_passes_retained": f"{add['previously_passing_retained']}/{add['previously_passing_total']}" == semantic_targets["add_source_prior_passes_retained"],
        "add_source_surface_winner_invariance": f"{add['surface']['passed']}/{add['surface']['total']}" == semantic_targets["add_source_surface_winner_invariance"],
        "safe_actions": safe_results["gold_actions"] == semantic_targets["safe_actions"],
        "safe_execution": f"{safe_results['gold_execution_correct']}/{safe_results['gold_total']}" == semantic_targets["safe_execution"],
        "safe_permutation": f"{safe_results['permutation_exact']}/{safe_results['permutation_total']}" == semantic_targets["safe_permutation"],
        "safe_counterfactuals": safe_results["counterfactual_generated"] == semantic_targets["safe_counterfactuals"],
        "safe_counterfactual_false_accepts": safe_results["counterfactual_false_accepts"] == semantic_targets["safe_counterfactual_false_accepts"],
    }
    for dimension, expected in semantic_targets["add_source_counterfactual_old_winner_false_accepts"].items():
        row = add["counterfactual"]["by_dimension"][dimension]
        semantic_checks[f"add_source_coverage:{dimension}"] = bool(row["coverage_gate"])
        semantic_checks[f"add_source_false_accepts:{dimension}"] = row["old_winner_false_accepts"] == expected
    for dimension, expected in semantic_targets["actual_plan_paired_gold_answers"].items():
        row = actual["dimensions"][dimension]
        semantic_checks[f"actual_plan_paired_gold:{dimension}"] = f"{row['paired_gold_answers']}/{row['anchors']}" == expected
        semantic_checks[f"actual_plan_false_accepts:{dimension}"] = row["old_winner_false_accepts"] == semantic_targets["actual_plan_old_winner_false_accepts"][dimension]

    failed_measurement = sorted(name for name, passed in completion_checks.items() if not passed)
    failed_semantic = sorted(name for name, passed in semantic_checks.items() if not passed)
    refresh_complete = not failed_measurement
    qp4_gates_passed = refresh_complete and not failed_semantic
    if not refresh_complete:
        next_step = "Resolve QP4-POST-QUESTION-SURFACE-BASELINE-REFRESH measurement failures without changing runtime."
    elif qp4_gates_passed:
        next_step = "QP4-CLOSURE-PRECOMMIT: verify every frozen QP4 gate before closing QP4; no runtime change is authorized."
    else:
        next_step = "QP4-POST-QUESTION-SURFACE-BASELINE-DIAGNOSTIC-REVIEW-PRECOMMIT: freeze a report-agnostic first-failing-layer review for the measured residual groups; no runtime change is authorized."

    return {
        "schema": "finhrm-qp4-post-question-surface-baseline-refresh-report-v1",
        "created_at": config["created_at"],
        "classification": "development_only_qp4_post_question_surface_measurement_runtime_unchanged",
        "status": "REFRESH_COMPLETE" if refresh_complete else "REFRESH_INCOMPLETE",
        "refresh_complete": refresh_complete,
        "qp4_gates_passed": qp4_gates_passed,
        "failed_measurement_gates": failed_measurement,
        "failed_semantic_gates": failed_semantic,
        "completion_checks": completion_checks,
        "semantic_checks": semantic_checks,
        "compiler": compiler,
        "add_source_backed": add,
        "actual_plan_source_backed": actual,
        "source_errors": errors,
        "group_metrics": {
            "add": add_groups,
            "actual_plan": actual_groups,
        },
        "verification": {
            "safe_replay": {
                "artifact": _artifact(safe_path),
                "actions": safe_results["gold_actions"],
                "execution": f"{safe_results['gold_execution_correct']}/{safe_results['gold_total']}",
                "permutation": f"{safe_results['permutation_exact']}/{safe_results['permutation_total']}",
                "counterfactuals": safe_results["counterfactual_generated"],
                "counterfactual_false_accepts": safe_results["counterfactual_false_accepts"],
                "company_macro_worst_answer_rate": [
                    safe_results["company"]["macro_gold_answer_rate"],
                    safe_results["company"]["worst_gold_answer_rate"],
                ],
                "layout_macro_worst_answer_rate": [
                    safe_results["layout"]["macro_gold_answer_rate"],
                    safe_results["layout"]["worst_gold_answer_rate"],
                ],
            },
            "compatible_affected_regression": regression_result,
        },
        "generalization_boundary": config["generalization_contract"],
        "runtime_changed": False,
        "runtime_hashes": config["frozen_runtime_hashes"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "invariants": config["invariants"],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": next_step,
    }


def _markdown(report: dict[str, Any]) -> str:
    compiler = report["compiler"]
    add = report["add_source_backed"]
    actual = report["actual_plan_source_backed"]
    safe = report["verification"]["safe_replay"]
    regression = report["verification"]["compatible_affected_regression"]
    lines = [
        "# QP4 post-question-surface baseline refresh v1",
        "",
        f"- Refresh: `{'COMPLETE' if report['refresh_complete'] else 'INCOMPLETE'}`; QP4 semantic gates: `{'PASS' if report['qp4_gates_passed'] else 'FAIL'}`.",
        f"- Compiler: canonical `{compiler['canonical']['exact']}/{compiler['canonical']['total']}`, ambiguity `{compiler['ambiguity']['exact']}/{compiler['ambiguity']['total']}`, frozen surface `{compiler['frozen_surface']['exact']}/{compiler['frozen_surface']['total']}`, corrected hold-ahead `{compiler['corrected_hold_ahead']['exact']}/{compiler['corrected_hold_ahead']['total']}`, minimal pairs `{compiler['minimal_pairs']['detected']}/{compiler['minimal_pairs']['total']}`.",
        f"- ADD source: `{add['surface']['passed']}/{add['surface']['total']}` surface invariance over `{add['companies']}` companies / `{add['source_documents']}` PDFs / `{add['real_layout_families']}` real layouts; prior passes retained `{add['previously_passing_retained']}/{add['previously_passing_total']}`.",
        f"- Source execution errors retained in denominator: `{report['source_errors']['total_failed_observations']}`.",
        f"- Safe replay: actions `{safe['actions']}`, execution `{safe['execution']}`, permutation `{safe['permutation']}`, counterfactual false accepts `{safe['counterfactual_false_accepts']}`.",
        f"- Compatible regression: `{regression.get('observed', 0)}/354`.",
        "",
        "| ACTUAL/PLAN dimension | Paired gold | Old-winner false accepts | Gate |",
        "|---|---:|---:|---|",
    ]
    for dimension, row in actual["dimensions"].items():
        lines.append(
            f"| {dimension} | {row['paired_gold_answers']}/{row['anchors']} | {row['old_winner_false_accepts']} | {'PASS' if row['runtime_gate_passed'] else 'FAIL'} |"
        )
    if report["failed_semantic_gates"]:
        lines.extend(["", "Failed semantic gates: " + ", ".join(f"`{name}`" for name in report["failed_semantic_gates"]) + "."])
    lines.extend(
        [
            "",
            "The same frozen contract was applied to every eligible development record. Errors, timeouts and abstentions were not removed or counted as answers. This is not unseen or universal-report confirmation.",
            "",
            "No runtime, threshold, checkpoint or model change was made. QP4 is incomplete; QP5 and Phase D remain locked.",
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
        ADD_RAW,
        ACTUAL_PLAN_RAW,
        SAFE_REPLAY,
        REPORT,
        MARKDOWN,
    )
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    lock = {
        "schema": "finhrm-qp4-post-question-surface-baseline-refresh-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["refresh_complete"] else "failed",
        "refresh_complete": report["refresh_complete"],
        "qp4_gates_passed": report["qp4_gates_passed"],
        "failed_measurement_gates": report["failed_measurement_gates"],
        "failed_semantic_gates": report["failed_semantic_gates"],
        "runtime_changed": False,
        "runtime_hashes": report["runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "runtime_file_whitelist_for_next_step": [],
        "same_contract_for_every_frozen_eligible_record": True,
        "artifacts": [_artifact(path) for path in artifacts],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": report["authorized_next_step"],
    }
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-measurements", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    config = _load(PRECOMMIT)
    _verify_precommit(config)
    regression = run_measurements(config) if args.run_measurements else None
    report = build_report(regression=regression)
    if args.write:
        write_outputs(report)
    print(
        json.dumps(
            {
                "refresh_complete": report["refresh_complete"],
                "qp4_gates_passed": report["qp4_gates_passed"],
                "failed_measurement_gates": report["failed_measurement_gates"],
                "failed_semantic_gates": report["failed_semantic_gates"],
                "add_source_surface": f"{report['add_source_backed']['surface']['passed']}/{report['add_source_backed']['surface']['total']}",
                "actual_plan_failed_dimensions": report["actual_plan_source_backed"]["failed_runtime_dimensions"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["refresh_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
