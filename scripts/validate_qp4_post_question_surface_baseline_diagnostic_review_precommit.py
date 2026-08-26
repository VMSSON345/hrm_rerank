#!/usr/bin/env python3
"""Validate the no-runtime post-question-surface diagnostic-review precommit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "finhrm_qp4_post_question_surface_baseline_diagnostic_review_precommit_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_diagnostic_review_precommit_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_diagnostic_review_precommit_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp4_post_question_surface_baseline_diagnostic_review_precommit_lock_v1.json"
TEST_PATH = ROOT / "tests" / "test_qp4_post_question_surface_baseline_diagnostic_review_precommit.py"
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


def _check(condition: bool, name: str, failed: list[str]) -> None:
    if not condition:
        failed.append(name)


def _compiler_controls(refresh: dict[str, Any]) -> dict[str, Any]:
    compiler = refresh["compiler"]
    return {
        "canonical": f"{compiler['canonical']['exact']}/{compiler['canonical']['total']}",
        "ambiguity": f"{compiler['ambiguity']['exact']}/{compiler['ambiguity']['total']}",
        "frozen_surface": f"{compiler['frozen_surface']['exact']}/{compiler['frozen_surface']['total']}",
        "corrected_hold_ahead": f"{compiler['corrected_hold_ahead']['exact']}/{compiler['corrected_hold_ahead']['total']}",
        "minimal_pairs": f"{compiler['minimal_pairs']['detected']}/{compiler['minimal_pairs']['total']}",
        "residual_records": (
            compiler["frozen_surface"]["total"] - compiler["frozen_surface"]["exact"]
            + compiler["corrected_hold_ahead"]["total"] - compiler["corrected_hold_ahead"]["exact"]
            + compiler["minimal_pairs"]["total"] - compiler["minimal_pairs"]["detected"]
        ),
    }


def _add_surface_inventory(raw: dict[str, Any], refresh: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in raw["surface_results"] if not row["invariant"]]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    compiler_categories = refresh["compiler"]["frozen_surface"]["by_category"]
    groups: dict[str, Any] = {}
    for name, values in sorted(grouped.items()):
        non_error = [row for row in values if "error" not in row]
        groups[name] = {
            "failures": len(values),
            "companies": len({row["company_key"] for row in values}),
            "sources": len({row["source_key"] for row in values}),
            "real_layouts": len({row["layout_family"] for row in values}),
            "source_errors": sum("error" in row for row in values),
            "non_error_outcomes": dict(sorted(Counter(row["observed_signature"]["action"] for row in non_error).items())),
            "compiler_projection_passed": compiler_categories[name]["pass_rate"] == 1.0,
        }
    return {
        "residual_queries": len(rows),
        "companies": len({row["company_key"] for row in rows}),
        "sources": len({row["source_key"] for row in rows}),
        "real_layouts": len({row["layout_family"] for row in rows}),
        "groups": groups,
    }


def _counterfactual_error_inventory(raw: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in raw["counterfactual_results"] if "error" in row]
    dimensions = {row["dimension"] for row in rows}
    error_types = {row["error"]["type"] for row in rows}
    return {
        "failed_observations": len(rows),
        "dimension": next(iter(dimensions)) if len(dimensions) == 1 else "MIXED",
        "companies": len({row["company_key"] for row in rows}),
        "sources": len({row["source_key"] for row in rows}),
        "real_layouts": len({row["layout_family"] for row in rows}),
        "error_type": next(iter(error_types)) if len(error_types) == 1 else "MIXED",
        "false_accepts": sum(bool(row["old_winner_false_accept"]) for row in rows),
        "generic_evidence_minimum_met": (
            len({row["company_key"] for row in rows}) >= 3
            and len({row["source_key"] for row in rows}) >= 3
            and len({row["layout_family"] for row in rows}) >= 3
        ),
    }


def _actual_inventory(raw: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw["results"]:
        grouped[row["dimension"]].append(row)
    dimensions: dict[str, Any] = {}
    for name, values in sorted(grouped.items()):
        outcomes: Counter[str] = Counter()
        source_errors = 0
        for row in values:
            for side in ("base", "mutated"):
                observation = row[side]
                if "error" in observation:
                    source_errors += 1
                else:
                    outcomes[observation["signature"]["action"]] += 1
        dimensions[name] = {
            "pair_rows": len(values),
            "paired_gold_answers": sum(bool(row["paired_gold_answer"]) for row in values),
            "source_errors": source_errors,
            "non_error_outcomes": dict(sorted(outcomes.items())),
        }
    rows = raw["results"]
    return {
        "pair_rows": len(rows),
        "diagnostic_queries": len(rows) * 2,
        "companies": len({row["company_key"] for row in rows}),
        "sources": len({row["source_sha256"] for row in rows}),
        "document_layouts": len({row["document_layout_family"] for row in rows}),
        "table_layouts": len({row["table_layout_family"] for row in rows}),
        "dimensions": dimensions,
        "old_winner_false_accepts": sum(bool(row["old_winner_false_accept"]) for row in rows),
    }


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load(config_path)
    failed: list[str] = []
    for name, parent in config["parents"].items():
        path = ROOT / parent["path"]
        _check(path.is_file() and _digest(path) == parent["sha256"], f"parent:{name}", failed)
    checkpoint = config["checkpoint"]
    _check(_digest(ROOT / checkpoint["path"]) == checkpoint["sha256"], "checkpoint", failed)
    for relative, expected in config["frozen_runtime_hashes"].items():
        _check(_digest(ROOT / relative) == expected, f"runtime:{relative}", failed)

    parents = config["parents"]
    refresh = _load(ROOT / parents["refresh_report"]["path"])
    refresh_lock = _load(ROOT / parents["refresh_lock"]["path"])
    add_raw = _load(ROOT / parents["add_source_raw"]["path"])
    actual_raw = _load(ROOT / parents["actual_plan_source_raw"]["path"])
    safe = _load(ROOT / parents["safe_replay"]["path"])
    inventory = config["residual_inventory"]

    _check(refresh["refresh_complete"] is True, "refresh_complete", failed)
    _check(refresh["qp4_gates_passed"] is False, "qp4_gate_open", failed)
    _check(refresh["failed_measurement_gates"] == [], "measurement_no_failures", failed)
    _check(refresh["failed_semantic_gates"] == inventory["failed_semantic_gates"], "failed_semantic_inventory", failed)
    _check(refresh_lock["status"] == "complete" and refresh_lock["refresh_complete"] is True, "refresh_lock_complete", failed)
    _check(refresh_lock["qp4_gates_passed"] is False, "refresh_lock_qp4_open", failed)
    _check(
        refresh_lock["authorized_next_step"].startswith("QP4-POST-QUESTION-SURFACE-BASELINE-DIAGNOSTIC-REVIEW-PRECOMMIT"),
        "diagnostic_precommit_authorized",
        failed,
    )

    compiler = _compiler_controls(refresh)
    add = _add_surface_inventory(add_raw, refresh)
    counterfactual_error = _counterfactual_error_inventory(add_raw)
    actual = _actual_inventory(actual_raw)
    _check(compiler == inventory["compiler_controls"], "compiler_inventory", failed)
    _check(add == inventory["add_source_surface"], "add_surface_inventory", failed)
    _check(counterfactual_error == inventory["add_counterfactual_source_error"], "counterfactual_error_inventory", failed)
    _check(actual == inventory["actual_plan_source"], "actual_plan_inventory", failed)
    _check(compiler["residual_records"] == 0, "compiler_zero_residuals", failed)
    _check((add["residual_queries"], add["companies"], add["sources"], add["real_layouts"]) == (15, 5, 5, 3), "add_scope", failed)
    _check((actual["diagnostic_queries"], actual["companies"], actual["sources"], actual["document_layouts"], actual["table_layouts"]) == (24, 4, 4, 3, 4), "actual_scope", failed)
    source_errors = (
        sum(row["source_errors"] for row in add["groups"].values())
        + counterfactual_error["failed_observations"]
        + sum(row["source_errors"] for row in actual["dimensions"].values())
    )
    source_observations = add["residual_queries"] + counterfactual_error["failed_observations"] + actual["diagnostic_queries"]
    _check(source_observations == inventory["source_observations_for_diagnosis"] == 40, "source_observation_count", failed)
    _check(source_errors == inventory["source_error_observations"] == 12, "source_error_count", failed)
    _check(source_observations - source_errors == inventory["non_error_observations"] == 28, "non_error_count", failed)

    contract = config["diagnostic_contract"]
    _check(contract["stage"] == "diagnostic_measurement_only_no_runtime_change", "diagnostic_only", failed)
    _check(contract["runtime_file_whitelist"] == [], "empty_runtime_whitelist", failed)
    _check(contract["compiler_residual_records"] == 0 and contract["compiler_inventory_is_no_regression_control"] is True, "compiler_control_only", failed)
    _check(contract["source_residual_observations"] == 40, "contract_source_count", failed)
    _check(contract["source_observation_breakdown"] == {"add_surface_residuals": 15, "add_counterfactual_source_errors": 1, "actual_plan_base_and_mutated": 24}, "contract_breakdown", failed)
    _check(contract["evaluate_all_records_in_each_residual_or_error_stratum"] is True, "evaluate_all", failed)
    _check(contract["source_run_boundary"] == ["question", "source PDF path"], "source_boundary", failed)
    _check(contract["gold_candidate_or_program_passed_to_engine"] is False and contract["gold_used_only_after_observation_for_diagnostic_comparison"] is True, "gold_boundary", failed)
    _check(contract["per_query_timeout_seconds"] == 15 and contract["timeout_or_document_source_error_is_failure"] is True, "timeout_policy", failed)
    _check(contract["source_error_must_be_attributed_before_downstream_layers"] is True, "source_error_attribution", failed)
    _check(contract["mineru_auto_generate"] is False and contract["existing_development_cache_only"] is True, "cache_policy", failed)
    _check(contract["safe_abstention_counts_as_answer"] is False, "abstention_not_answer", failed)
    _check(contract["candidate_rank_score_or_confidence_is_semantic_proof"] is False, "no_score_proof", failed)
    expected_layers = [
        "QUESTION_PLAN",
        "SOURCE_LOAD_OR_EXTRACTION",
        "CANDIDATE_POPULATION_OR_RETRIEVAL",
        "SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS",
        "SYMBOLIC_PROOF_OR_ACTION",
    ]
    _check(contract["layer_order"] == expected_layers, "layer_order", failed)
    _check(set(contract["first_failing_layer_rules"]) == set(expected_layers), "layer_rules", failed)
    _check(len(contract["trace_fields"]) == 13, "trace_schema", failed)
    forbidden = set(contract["selection_must_not_use"])
    _check(
        {
            "company identity",
            "report filename",
            "page number",
            "year",
            "numeric value",
            "example or anchor id",
            "question literal",
            "candidate rank",
            "neural score",
            "confidence",
        }
        <= forbidden,
        "forbidden_selection",
        failed,
    )
    _check(contract["minimum_generic_evidence"] == {"compiler_family_groups": 3, "source_company_groups": 3, "source_documents": 3, "real_layout_families": 3}, "generic_minimums", failed)
    _check(all(contract["attribution_policy"].values()), "attribution_policy", failed)

    safe_results = safe["results"]
    _check(safe["safe_replay_gate_passed"] is True and safe["errors"] == [], "safe_replay_passed", failed)
    _check(safe_results["gold_actions"] == {"ABSTAIN": 92, "ANSWER": 257, "REJECT": 14}, "safe_actions", failed)
    _check(safe_results["gold_execution_correct"] == safe_results["gold_total"] == 363, "safe_execution", failed)
    _check(safe_results["permutation_exact"] == safe_results["permutation_total"] == 363, "safe_permutation", failed)
    _check(safe_results["counterfactual_generated"] == 141 and safe_results["counterfactual_false_accepts"] == 0, "safe_counterfactual", failed)

    _check(config["runtime_changed_by_this_precommit"] is False, "runtime_unchanged", failed)
    _check(config["diagnostic_executed_by_this_precommit"] is False, "diagnostic_not_executed", failed)
    _check(config["downstream_runtime_change_authorized_in_same_step"] is False, "same_step_runtime_locked", failed)
    _check(all(value is False for value in config["invariants"].values()), "invariants", failed)
    locked_gate_names = {"runtime_changed", "runtime_edit_authorized", "qp4_complete", "qp5_authorized", "phase_d_unlocked"}
    _check(all(config["diagnostic_gates"][name] is False for name in locked_gate_names), "later_stages_locked", failed)

    passed = not failed
    return {
        "schema": "finhrm-qp4-post-question-surface-baseline-diagnostic-review-precommit-report-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "precommit_gate_passed": passed,
        "failed_gates": failed,
        "residual_inventory": inventory,
        "derived_residual_evidence": {
            "compiler_controls": compiler,
            "add_source_surface": add,
            "add_counterfactual_source_error": counterfactual_error,
            "actual_plan_source": actual,
            "source_observations": source_observations,
            "source_error_observations": source_errors,
            "non_error_observations": source_observations - source_errors,
        },
        "diagnostic_contract": contract,
        "diagnostic_gates": config["diagnostic_gates"],
        "runtime_changed": False,
        "diagnostic_executed": False,
        "runtime_edit_authorized": False,
        "runtime_hashes": config["frozen_runtime_hashes"],
        "checkpoint_sha256": checkpoint["sha256"],
        "safe_replay_retained": {
            "actions": safe_results["gold_actions"],
            "execution": f"{safe_results['gold_execution_correct']}/{safe_results['gold_total']}",
            "permutation": f"{safe_results['permutation_exact']}/{safe_results['permutation_total']}",
            "counterfactuals": safe_results["counterfactual_generated"],
            "counterfactual_false_accepts": safe_results["counterfactual_false_accepts"],
        },
        "future_diagnostic_outputs": config["future_diagnostic_outputs"],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": (
            config["authorized_next_step_after_lock"]
            if passed
            else "Resolve diagnostic-review precommit failures without running diagnostics or changing runtime."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    evidence = report["derived_residual_evidence"]
    return "\n".join(
        [
            "# QP4 post-question-surface baseline diagnostic-review precommit v1",
            "",
            f"- Gate: `{'PASS' if report['precommit_gate_passed'] else 'FAIL'}`.",
            "- Stage: diagnostic precommit only; no diagnostic replay or runtime edit occurred.",
            "- Compiler is a no-regression control: `242/242`; there are no compiler residual records to target.",
            f"- Diagnostic source scope: `{evidence['source_observations']}` observations = `15` ADD surface + `1` ADD counterfactual source error + `24` ACTUAL/PLAN.",
            f"- Source-error stratum: `{evidence['source_error_observations']}/40`; non-error stratum: `{evidence['non_error_observations']}/40`. Every error remains in the denominator.",
            "- Earliest-layer order: question plan -> source load/extraction -> candidate population/retrieval -> semantic binding/synthesis -> symbolic proof/action.",
            "- Gold is post-hoc diagnostic comparison only. Rank, score, confidence and safe abstention are not semantic proof.",
            "- Runtime whitelist is empty. Generic group size alone cannot authorize a file; singleton/non-diverse strata cannot nominate runtime.",
            "- QP4 is incomplete; QP5 and Phase D remain locked.",
            "",
            f"Next: {report['authorized_next_step']}",
            "",
        ]
    )


def write_outputs(
    report: dict[str, Any],
    report_path: Path = DEFAULT_REPORT,
    markdown_path: Path = DEFAULT_MARKDOWN,
    lock_path: Path = DEFAULT_LOCK,
) -> None:
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    artifacts = [DEFAULT_CONFIG, Path(__file__).resolve(), TEST_PATH, report_path, markdown_path]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    lock = {
        "schema": "finhrm-qp4-post-question-surface-baseline-diagnostic-review-precommit-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["precommit_gate_passed"] else "failed",
        "precommit_gate_passed": report["precommit_gate_passed"],
        "failed_gates": report["failed_gates"],
        "diagnostic_executed": False,
        "runtime_changed": False,
        "runtime_edit_authorized": False,
        "runtime_hashes": report["runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "runtime_file_whitelist_for_next_step": [],
        "future_diagnostic_outputs": report["future_diagnostic_outputs"],
        "artifacts": [_artifact(path) for path in artifacts],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": report["authorized_next_step"],
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    report = build_report(args.config)
    write_outputs(report, args.report, args.markdown, args.lock)
    print(
        json.dumps(
            {
                "precommit_gate_passed": report["precommit_gate_passed"],
                "failed_gates": report["failed_gates"],
                "source_observations": report["derived_residual_evidence"]["source_observations"],
                "source_errors": report["derived_residual_evidence"]["source_error_observations"],
                "runtime_edit_authorized": report["runtime_edit_authorized"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["precommit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
