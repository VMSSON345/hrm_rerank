#!/usr/bin/env python3
"""Validate and lock the no-runtime QP4 residual diagnostic-review contract."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/finhrm_qp4_post_unit_scale_baseline_diagnostic_review_precommit_v1.json"
DEFAULT_REPORT = ROOT / "reports/finhrm_qp4_post_unit_scale_baseline_diagnostic_review_precommit_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports/finhrm_qp4_post_unit_scale_baseline_diagnostic_review_precommit_v1.md"
DEFAULT_LOCK = ROOT / "reports/finhrm_qp4_post_unit_scale_baseline_diagnostic_review_precommit_lock_v1.json"
TEST_PATH = ROOT / "tests/test_qp4_post_unit_scale_baseline_diagnostic_review_precommit.py"
PROTECTED = re.compile(r"(?:^|[/_.-])(test|holdout|final|blind|validation|sealed)(?:[/_.-]|$)", re.I)


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _compiler_residuals(refresh: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in refresh["compiler"]["surface"]["details"]:
        if not row["semantic_invariant"]:
            grouped[row["category"]].append(row)
    return {
        "residual_records": sum(len(rows) for rows in grouped.values()),
        "groups": {
            name: {
                "failures": len(rows),
                "family_groups": len({row["family_id"] for row in rows}),
                "first_failing_layer": "QUESTION_PLAN",
            }
            for name, rows in sorted(grouped.items())
        },
    }


def _add_residuals(raw: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in raw["surface_results"] if not row["invariant"]]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    groups: dict[str, Any] = {}
    for name, values in sorted(grouped.items()):
        outcomes = Counter(
            row.get("observed_signature", {}).get("action") for row in values if "error" not in row
        )
        groups[name] = {
            "failures": len(values),
            "source_errors": sum("error" in row for row in values),
            "non_error_outcomes": dict(sorted(outcomes.items())),
            "compiler_projection_passed": name != "DISCOURSE_PREFIX_SUFFIX",
        }
    return {
        "residual_queries": len(rows),
        "companies": len({row["company_key"] for row in rows}),
        "sources": len({row["source_key"] for row in rows}),
        "real_layouts": len({row["layout_family"] for row in rows}),
        "groups": groups,
    }


def _actual_residuals(raw: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw["results"]:
        grouped[row["dimension"]].append(row)
    dimensions: dict[str, Any] = {}
    for name, rows in sorted(grouped.items()):
        observations = [row[label] for row in rows for label in ("base", "mutated")]
        outcomes = Counter(
            item.get("signature", {}).get("action") for item in observations if "error" not in item
        )
        dimensions[name] = {
            "pair_rows": len(rows),
            "paired_gold_answers": sum(bool(row["paired_gold_answer"]) for row in rows),
            "source_errors": sum("error" in item for item in observations),
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
    precommit_lock = _load(ROOT / parents["refresh_precommit_lock"]["path"])
    refresh = _load(ROOT / parents["refresh_report"]["path"])
    refresh_lock = _load(ROOT / parents["refresh_lock"]["path"])
    add_raw = _load(ROOT / parents["add_source_raw"]["path"])
    actual_raw = _load(ROOT / parents["actual_plan_source_raw"]["path"])
    safe = _load(ROOT / parents["safe_replay"]["path"])

    _check(precommit_lock["precommit_gate_passed"] is True, "refresh_precommit_passed", failed)
    _check(precommit_lock["runtime_file_whitelist_for_next_step"] == [], "refresh_runtime_whitelist_empty", failed)
    _check(refresh["refresh_complete"] is True and refresh["baseline_gates_passed"] is False, "refresh_failed_closed", failed)
    _check(refresh["runtime_changed"] is False and refresh_lock["runtime_changed"] is False, "refresh_runtime_unchanged", failed)
    _check(refresh_lock["status"] == "complete" and refresh_lock["qp4_complete"] is False, "refresh_lock_complete", failed)
    _check(
        refresh_lock["authorized_next_step"].startswith("QP4-POST-UNIT-SCALE-BASELINE-DIAGNOSTIC-REVIEW-PRECOMMIT"),
        "diagnostic_precommit_authorized",
        failed,
    )
    _check(refresh["failed_gates"] == config["residual_inventory"]["failed_refresh_gates"], "failed_gate_inventory", failed)

    compiler = _compiler_residuals(refresh)
    add = _add_residuals(add_raw)
    actual = _actual_residuals(actual_raw)
    inventory = config["residual_inventory"]
    _check(compiler == inventory["compiler_surface"], "compiler_residual_inventory", failed)
    _check(add == inventory["add_source_surface"], "add_residual_inventory", failed)
    _check(actual == inventory["actual_plan_source"], "actual_residual_inventory", failed)
    _check(compiler["residual_records"] == 18, "compiler_residual_count", failed)
    _check(add["residual_queries"] == 15 and (add["companies"], add["sources"], add["real_layouts"]) == (5, 5, 3), "add_residual_scope", failed)
    _check(actual["diagnostic_queries"] == 24 and (actual["companies"], actual["sources"], actual["document_layouts"], actual["table_layouts"]) == (4, 4, 3, 4), "actual_residual_scope", failed)
    _check(sum(row["source_errors"] for row in add["groups"].values()) == 3, "add_source_error_stratum", failed)
    _check(sum(row["source_errors"] for row in actual["dimensions"].values()) == 8, "actual_source_error_stratum", failed)

    contract = config["diagnostic_contract"]
    _check(contract["stage"] == "diagnostic_measurement_only_no_runtime_change", "diagnostic_only", failed)
    _check(contract["runtime_file_whitelist"] == [], "empty_runtime_whitelist", failed)
    _check(contract["compiler_residual_records"] == 18, "contract_compiler_count", failed)
    _check(contract["source_residual_queries"] == 39, "contract_source_count", failed)
    _check(contract["source_query_breakdown"] == {"add_surface": 15, "actual_plan": 24}, "contract_query_breakdown", failed)
    _check(contract["evaluate_all_records_in_each_generic_residual_group"] is True, "evaluate_all_residuals", failed)
    _check(contract["source_run_boundary"] == ["question", "source PDF path"], "source_boundary", failed)
    _check(contract["gold_candidate_or_program_passed_to_engine"] is False, "no_gold_engine_input", failed)
    _check(contract["gold_used_only_after_observation_for_diagnostic_comparison"] is True, "post_hoc_gold_only", failed)
    _check(contract["per_query_timeout_seconds"] == 15 and contract["timeout_or_document_source_error_is_failure"] is True, "failure_timeout", failed)
    _check(contract["mineru_auto_generate"] is False and contract["existing_development_cache_only"] is True, "cache_policy", failed)
    _check(contract["safe_abstention_counts_as_answer"] is False, "no_abstention_as_answer", failed)
    _check(contract["candidate_rank_score_or_confidence_is_semantic_proof"] is False, "no_score_proof", failed)
    _check(
        contract["layer_order"]
        == [
            "QUESTION_PLAN",
            "SOURCE_LOAD_OR_EXTRACTION",
            "CANDIDATE_POPULATION_OR_RETRIEVAL",
            "SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS",
            "SYMBOLIC_PROOF_OR_ACTION",
        ],
        "layer_order",
        failed,
    )
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
        "forbidden_selection_inputs",
        failed,
    )
    policy = contract["attribution_policy"]
    _check(all(policy.values()), "attribution_policy", failed)
    minimum = contract["minimum_generic_evidence"]
    _check(minimum == {"compiler_family_groups": 3, "source_company_groups": 3, "source_documents": 3, "real_layout_families": 3}, "generic_minimums", failed)
    _check(len(contract["trace_fields"]) == 11, "trace_schema", failed)

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
    _check(all(value is False for name, value in config["diagnostic_gates"].items() if name in {"runtime_changed", "runtime_edit_authorized", "qp4_complete", "qp5_authorized", "phase_d_unlocked"}), "diagnostic_locks", failed)

    passed = not failed
    return {
        "schema": "finhrm-qp4-post-unit-scale-baseline-diagnostic-review-precommit-report-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "precommit_gate_passed": passed,
        "failed_gates": failed,
        "residual_inventory": inventory,
        "derived_residual_evidence": {
            "compiler": compiler,
            "add_source": add,
            "actual_plan_source": actual,
            "source_error_queries": 11,
            "non_error_source_queries": 28,
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
            "# QP4 post-UNIT_SCALE baseline diagnostic-review precommit v1",
            "",
            f"- Gate: `{'PASS' if report['precommit_gate_passed'] else 'FAIL'}`.",
            "- Stage: diagnostic measurement precommit only; no diagnostic source replay or runtime edit occurred.",
            f"- Compiler residuals: `{evidence['compiler']['residual_records']}` across three generic surface groups.",
            f"- ADD surface residuals: `{evidence['add_source']['residual_queries']}` queries / `{evidence['add_source']['companies']}` companies / `{evidence['add_source']['sources']}` PDFs / `{evidence['add_source']['real_layouts']}` real layouts.",
            f"- ACTUAL/PLAN residuals: `{evidence['actual_plan_source']['diagnostic_queries']}` queries / `{evidence['actual_plan_source']['companies']}` companies / `{evidence['actual_plan_source']['sources']}` PDFs / `{evidence['actual_plan_source']['document_layouts']}` document layouts.",
            f"- Source-error stratum: `{evidence['source_error_queries']}/39`; non-error fail-closed stratum: `{evidence['non_error_source_queries']}/39`.",
            "- Earliest-layer order is frozen: question plan -> source load/extraction -> candidate population/retrieval -> semantic binding/synthesis -> symbolic proof/action.",
            "- Gold is post-hoc diagnostic evidence only and is never passed into the engine. Rank, score, confidence and safe abstention are not semantic proof.",
            "- Runtime whitelist is empty. Any later implementation requires a new precommit after generic multi-company/layout attribution.",
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
        "schema": "finhrm-qp4-post-unit-scale-baseline-diagnostic-review-precommit-lock-v1",
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
    print(json.dumps({
        "precommit_gate_passed": report["precommit_gate_passed"],
        "failed_gates": report["failed_gates"],
        "compiler_residuals": report["derived_residual_evidence"]["compiler"]["residual_records"],
        "source_residual_queries": report["diagnostic_contract"]["source_residual_queries"],
        "runtime_edit_authorized": report["runtime_edit_authorized"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if report["precommit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
