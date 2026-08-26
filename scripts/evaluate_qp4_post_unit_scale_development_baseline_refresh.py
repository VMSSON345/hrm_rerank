#!/usr/bin/env python3
"""Run the frozen post-UNIT_SCALE QP4 development baseline refresh."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

PRECOMMIT = ROOT / "config/finhrm_qp4_post_unit_scale_development_baseline_refresh_precommit_v1.json"
PRECOMMIT_REPORT = ROOT / "reports/finhrm_qp4_post_unit_scale_development_baseline_refresh_precommit_v1.json"
PRECOMMIT_LOCK = ROOT / "reports/finhrm_qp4_post_unit_scale_development_baseline_refresh_precommit_lock_v1.json"
ADD_CONFIG = ROOT / "config/finhrm_qp4_development_cohort_v1.json"
ACTUAL_PLAN_CONFIG = ROOT / "config/finhrm_qp4_actual_plan_ratio_direction_source_cohort_v1.json"
SAFE_REPLAY = ROOT / "reports/finhrm_qp4_unit_scale_proof_contract_safe_replay_v1.json"
ADD_RAW = ROOT / "reports/finhrm_qp4_post_unit_scale_source_backed_baseline_v1.json"
ACTUAL_PLAN_RAW = ROOT / "reports/finhrm_qp4_post_unit_scale_actual_plan_source_baseline_v1.json"
REPORT = ROOT / "reports/finhrm_qp4_post_unit_scale_development_baseline_refresh_v1.json"
MARKDOWN = ROOT / "reports/finhrm_qp4_post_unit_scale_development_baseline_refresh_v1.md"
LOCK = ROOT / "reports/finhrm_qp4_post_unit_scale_development_baseline_refresh_lock_v1.json"
TEST = ROOT / "tests/test_qp4_post_unit_scale_development_baseline_refresh.py"

REGRESSION_MODULES = (
    "tests.test_column_role_runtime_qp2",
    "tests.test_direction_contract",
    "tests.test_document_hrm",
    "tests.test_program_reranker",
    "tests.test_program_synthesis",
    "tests.test_qp3_answer_presentation_implementation",
    "tests.test_question_operand_slots_qp2a",
    "tests.test_question_operation_direction_qp2b",
    "tests.test_question_planner",
    "tests.test_question_program_dag_qp2c",
    "tests.test_question_safe_abstention_qp2d",
    "tests.test_question_surface_robustness_qp2e",
    "tests.test_qp4_unit_scale_proof_contract_implementation",
    "tests.test_qp4_unit_scale_proof_contract_implementation_report",
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise ValueError("Refresh precommit is not closed")
    if lock.get("authorized_next_step") != config.get("authorized_next_step_after_lock"):
        raise ValueError("Refresh authorization mismatch")
    if config["refresh_scope"]["runtime_file_whitelist"]:
        raise ValueError("Measurement step must have an empty runtime whitelist")
    if config["refresh_scope"]["allowed_new_evaluator"] != Path(__file__).resolve().relative_to(ROOT).as_posix():
        raise ValueError("This evaluator is not the pre-authorized evaluator")
    for item in config["parents"].values():
        path = ROOT / item["path"]
        if not path.is_file() or _digest(path) != item["sha256"]:
            raise ValueError(f"Frozen parent hash mismatch: {item['path']}")
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


def _derived_config(path: Path, frozen_runtime_hashes: dict[str, str]) -> dict[str, Any]:
    config = deepcopy(_load(path))
    config["runtime_hashes"] = {
        relative: frozen_runtime_hashes[relative]
        for relative in config["runtime_hashes"]
    }
    return config


def _validate_raw(raw: dict[str, Any], schema: str, runtime_hashes: dict[str, str]) -> None:
    if raw.get("schema") != schema:
        raise ValueError(f"Unexpected raw schema: {raw.get('schema')}")
    expected = {relative: _digest(ROOT / relative) for relative in runtime_hashes}
    if raw.get("runtime_hashes") != expected:
        raise ValueError("Runtime changed during source refresh")


def run_source_refresh(config: dict[str, Any]) -> None:
    from scripts.evaluate_qp4_actual_plan_source_cohort import run_source_baseline
    from scripts.evaluate_qp4_development_baseline import _verify_inputs, run_source_backed

    _verify_precommit(config)
    add_config = _derived_config(ADD_CONFIG, config["frozen_runtime_hashes"])
    actual_config = _derived_config(ACTUAL_PLAN_CONFIG, config["frozen_runtime_hashes"])
    _verify_inputs(add_config)
    run_source_backed(add_config, ADD_RAW)
    run_source_baseline(actual_config, ACTUAL_PLAN_RAW)


def _actual_plan_summary(config: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    from scripts.evaluate_qp4_actual_plan_source_cohort import _verify_config

    coverage = _verify_config(config)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw["results"]:
        grouped[str(row["dimension"])].append(row)
    dimensions: dict[str, Any] = {}
    for dimension in config["dimensions"]:
        rows = grouped[dimension]
        base_outcomes = Counter(
            row["base"].get("signature", {}).get("action")
            or f"ERROR:{row['base'].get('error', {}).get('type', 'UNKNOWN')}"
            for row in rows
        )
        mutated_outcomes = Counter(
            row["mutated"].get("signature", {}).get("action")
            or f"ERROR:{row['mutated'].get('error', {}).get('type', 'UNKNOWN')}"
            for row in rows
        )
        result = {
            "source_evidence_coverage_confirmed": len(rows) == len(config["anchors"]),
            "anchors": len(rows),
            "companies": len({row["company_key"] for row in rows}),
            "sources": len({row["source_sha256"] for row in rows}),
            "document_layouts": len({row["document_layout_family"] for row in rows}),
            "table_layouts": len({row["table_layout_family"] for row in rows}),
            "base_gold_answers": sum(bool(row["base"]["gold_answer_match"]) for row in rows),
            "mutated_gold_answers": sum(bool(row["mutated"]["gold_answer_match"]) for row in rows),
            "paired_gold_answers": sum(bool(row["paired_gold_answer"]) for row in rows),
            "same_observed_winners": sum(bool(row["same_observed_winner"]) for row in rows),
            "old_winner_false_accepts": sum(bool(row["old_winner_false_accept"]) for row in rows),
            "base_outcomes": dict(sorted(base_outcomes.items())),
            "mutated_outcomes": dict(sorted(mutated_outcomes.items())),
        }
        result["runtime_gate_passed"] = (
            result["paired_gold_answers"] == len(rows)
            and result["old_winner_false_accepts"] == 0
        )
        dimensions[dimension] = result
    return {
        "coverage": coverage,
        "source_evidence_coverage_complete": all(
            row["source_evidence_coverage_confirmed"] for row in dimensions.values()
        ),
        "dimensions": dimensions,
        "runtime_gates_passed": all(row["runtime_gate_passed"] for row in dimensions.values()),
        "failed_runtime_dimensions": sorted(
            name for name, row in dimensions.items() if not row["runtime_gate_passed"]
        ),
    }


def run_compatible_regression() -> dict[str, Any]:
    command = [sys.executable, "-m", "unittest", *REGRESSION_MODULES]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    match = re.search(r"Ran (\d+) tests?", completed.stdout)
    observed = int(match.group(1)) if match else 0
    return {
        "command": " ".join(command),
        "expected": 348,
        "observed": observed,
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0 and observed == 348,
        "output_tail": completed.stdout[-4000:],
    }


def build_report(
    config_path: Path = PRECOMMIT,
    add_raw_path: Path = ADD_RAW,
    actual_raw_path: Path = ACTUAL_PLAN_RAW,
    compatible_regression: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from scripts.evaluate_qp4_development_baseline import (
        _compiler_baseline,
        _diagnosis,
        _source_summary,
        _verify_inputs,
    )

    config = _load(config_path)
    _verify_precommit(config)
    add_config = _derived_config(ADD_CONFIG, config["frozen_runtime_hashes"])
    actual_config = _derived_config(ACTUAL_PLAN_CONFIG, config["frozen_runtime_hashes"])
    _verify_inputs(add_config)
    add_raw = _load(add_raw_path)
    actual_raw = _load(actual_raw_path)
    _validate_raw(add_raw, "finhrm-qp4-source-backed-baseline-v1", add_config["runtime_hashes"])
    _validate_raw(actual_raw, "finhrm-qp4-actual-plan-source-baseline-v1", actual_config["runtime_hashes"])
    if add_raw["inputs"]["cohort_config_sha256"] != _digest(ADD_CONFIG):
        raise ValueError("ADD source raw does not use the frozen cohort")
    if actual_raw["inputs"]["config_sha256"] != _digest(ACTUAL_PLAN_CONFIG):
        raise ValueError("ACTUAL/PLAN source raw does not use the frozen cohort")

    compiler = _compiler_baseline(add_config)
    add_source = _source_summary(add_config, add_raw)
    add_source["raw_surface_details"] = add_raw["surface_results"]
    add_source["raw_counterfactual_details"] = add_raw["counterfactual_results"]
    diagnosis = _diagnosis(compiler, add_source)
    add_source.pop("raw_surface_details")
    add_source.pop("raw_counterfactual_details")
    actual_plan = _actual_plan_summary(actual_config, actual_raw)

    gates = config["refresh_gates"]
    scope = config["refresh_scope"]
    failed: list[str] = []
    checks: dict[str, bool] = {}

    def check(name: str, value: bool) -> None:
        checks[name] = bool(value)
        if not value:
            failed.append(name)

    check("compiler_core_family_coverage", f"{compiler['canonical']['exact']}/{compiler['canonical']['total']}" == gates["compiler_core_family_coverage"])
    check("compiler_ambiguity_controls", f"{compiler['ambiguity']['exact']}/{compiler['ambiguity']['total']}" == gates["compiler_ambiguity_controls"])
    check("compiler_surface_semantic_invariance", f"{compiler['surface']['exact']}/{compiler['surface']['total']}" == gates["compiler_surface_semantic_invariance"])
    check("compiler_counterfactual_detection", f"{compiler['counterfactual']['detected']}/{compiler['counterfactual']['total']}" == gates["compiler_counterfactual_detection"])
    check("add_source_scope", (
        add_source["eligible_anchors"] == scope["add_source_tier"]["anchors"]
        and add_source["companies"] == scope["add_source_tier"]["companies"]
        and add_source["source_documents"] == scope["add_source_tier"]["sources"]
        and add_source["real_layout_families"] == scope["add_source_tier"]["real_layouts"]
        and len(add_raw["surface_results"]) + len(add_raw["counterfactual_results"]) == scope["add_source_tier"]["total_queries"]
    ))
    check("add_source_surface_winner_invariance", f"{add_source['surface']['passed']}/{add_source['surface']['total']}" == gates["add_source_surface_winner_invariance"])
    for dimension, expected in gates["add_source_counterfactual_old_winner_false_accepts"].items():
        row = add_source["counterfactual"]["by_dimension"][dimension]
        check(f"add_source_coverage:{dimension}", bool(row["coverage_gate"]))
        check(f"add_source_false_accepts:{dimension}", row["old_winner_false_accepts"] == expected)

    actual_scope = scope["actual_plan_source_tier"]
    check("actual_plan_source_scope", (
        actual_plan["coverage"]["anchors"] == actual_scope["anchors"]
        and actual_plan["coverage"]["companies"] == actual_scope["companies"]
        and actual_plan["coverage"]["sources"] == actual_scope["sources"]
        and actual_plan["coverage"]["document_layouts"] == actual_scope["document_layouts"]
        and actual_plan["coverage"]["table_layouts"] == actual_scope["table_layouts"]
        and len(actual_raw["results"]) * actual_scope["queries_per_anchor_dimension"] == actual_scope["total_queries"]
    ))
    for dimension, expected in gates["actual_plan_paired_gold_answers"].items():
        row = actual_plan["dimensions"][dimension]
        check(f"actual_plan_paired_gold:{dimension}", f"{row['paired_gold_answers']}/{row['anchors']}" == expected)
        check(f"actual_plan_false_accepts:{dimension}", row["old_winner_false_accepts"] == gates["actual_plan_old_winner_false_accepts"][dimension])

    safe = _load(SAFE_REPLAY)
    safe_results = safe["results"]
    safe_checks = {
        "actions": safe_results["gold_actions"] == gates["safe_actions"],
        "execution": f"{safe_results['gold_execution_correct']}/{safe_results['gold_total']}" == gates["safe_execution"],
        "permutation": f"{safe_results['permutation_exact']}/{safe_results['permutation_total']}" == gates["safe_permutation"],
        "permutation_rate": safe_results["permutation_exact"] / safe_results["permutation_total"] == gates["candidate_permutation_exact_rate"],
        "counterfactuals": safe_results["counterfactual_generated"] == gates["safe_counterfactuals"],
        "counterfactual_false_accepts": safe_results["counterfactual_false_accepts"] == gates["safe_counterfactual_false_accepts"],
    }
    for name, value in safe_checks.items():
        check(f"safe_replay:{name}", value)

    regression = compatible_regression or {
        "expected": 348,
        "observed": 0,
        "exit_code": None,
        "passed": False,
        "status": "not_run",
    }
    check("compatible_affected_regression", bool(regression.get("passed")))

    for dimension, row in actual_plan["dimensions"].items():
        if not row["runtime_gate_passed"]:
            diagnosis.append({
                "layer": "retrieval_or_binding_unisolated",
                "failure_group": f"actual_plan_source:{dimension}",
                "failures": row["anchors"] - row["paired_gold_answers"] + row["old_winner_false_accepts"],
                "company_groups": row["companies"],
                "source_groups": row["sources"],
                "real_layout_groups": row["document_layouts"],
                "generic_evidence": row["companies"] >= 3 and row["document_layouts"] >= 3,
                "automatic_whitelist_eligible": False,
                "runtime_edit_authorized": False,
            })

    measurement_complete = compatible_regression is not None
    gates_passed = measurement_complete and not failed
    next_step = (
        "QP4-CLOSURE-PRECOMMIT: verify all frozen QP4 gates before closing QP4; no runtime change is authorized in this refresh."
        if gates_passed
        else "QP4-POST-UNIT-SCALE-BASELINE-DIAGNOSTIC-REVIEW-PRECOMMIT: review only generic residual groups and isolate the first failing layer; no runtime change is authorized."
    )
    return {
        "schema": "finhrm-qp4-post-unit-scale-development-baseline-refresh-report-v1",
        "created_at": "2026-08-22",
        "classification": "development_only_qp4_post_unit_scale_baseline_refresh_runtime_unchanged",
        "status": "REFRESH_COMPLETE_RUNTIME_UNCHANGED" if measurement_complete else "REFRESH_SOURCE_COMPLETE_REGRESSION_PENDING",
        "refresh_complete": measurement_complete,
        "baseline_gates_passed": gates_passed,
        "failed_gates": sorted(set(failed)),
        "checks": checks,
        "compiler": compiler,
        "add_source_backed": add_source,
        "actual_plan_source_backed": actual_plan,
        "diagnosis": diagnosis,
        "verification": {
            "safe_replay": {
                "artifact": _artifact(SAFE_REPLAY),
                "checks": safe_checks,
                "actions": safe_results["gold_actions"],
                "execution": f"{safe_results['gold_execution_correct']}/{safe_results['gold_total']}",
                "permutation": f"{safe_results['permutation_exact']}/{safe_results['permutation_total']}",
                "counterfactuals": safe_results["counterfactual_generated"],
                "counterfactual_false_accepts": safe_results["counterfactual_false_accepts"],
            },
            "compatible_affected_regression": regression,
        },
        "runtime_changed": False,
        "runtime_hashes": config["frozen_runtime_hashes"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "invariants": config["invariants"] | {
            "runtime_changed": False,
            "safe_abstention_counted_as_answer": False,
            "source_queries": scope["combined_source_queries"],
        },
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": next_step,
    }


def _markdown(report: dict[str, Any]) -> str:
    compiler = report["compiler"]
    add = report["add_source_backed"]
    actual = report["actual_plan_source_backed"]
    lines = [
        "# QP4 post-UNIT_SCALE development baseline refresh v1",
        "",
        f"- Refresh: `{'COMPLETE' if report['refresh_complete'] else 'PENDING'}`; frozen gates: `{'PASS' if report['baseline_gates_passed'] else 'FAIL'}`.",
        f"- Compiler: canonical `{compiler['canonical']['exact']}/{compiler['canonical']['total']}`, ambiguity `{compiler['ambiguity']['exact']}/{compiler['ambiguity']['total']}`, surface `{compiler['surface']['exact']}/{compiler['surface']['total']}`, counterfactual `{compiler['counterfactual']['detected']}/{compiler['counterfactual']['total']}`.",
        f"- ADD source: {add['companies']} companies / {add['source_documents']} PDFs / {add['real_layout_families']} real layouts; surface `{add['surface']['passed']}/{add['surface']['total']}`.",
        f"- Safe replay: actions `{report['verification']['safe_replay']['actions']}`, execution `{report['verification']['safe_replay']['execution']}`, permutation `{report['verification']['safe_replay']['permutation']}`, counterfactual false accepts `{report['verification']['safe_replay']['counterfactual_false_accepts']}`.",
        f"- Compatible regression: `{report['verification']['compatible_affected_regression'].get('observed', 0)}/{report['verification']['compatible_affected_regression'].get('expected', 348)}`.",
        "",
        "| ACTUAL/PLAN dimension | Paired gold | Old-winner false accepts | Gate |",
        "|---|---:|---:|---|",
    ]
    for dimension, row in actual["dimensions"].items():
        lines.append(
            f"| {dimension} | {row['paired_gold_answers']}/{row['anchors']} | {row['old_winner_false_accepts']} | {'PASS' if row['runtime_gate_passed'] else 'FAIL'} |"
        )
    if report["failed_gates"]:
        lines.extend(["", "Failed gates: " + ", ".join(f"`{name}`" for name in report["failed_gates"]) + "."])
    lines.extend([
        "",
        "No runtime, threshold, checkpoint or model change was made. Safe abstention is not counted as an answer; synthetic layouts are not confirmation.",
        "",
        "Next: " + report["authorized_next_step"],
        "",
    ])
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
        REPORT,
        MARKDOWN,
    )
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    lock = {
        "schema": "finhrm-qp4-post-unit-scale-development-baseline-refresh-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["refresh_complete"] else "pending",
        "refresh_complete": report["refresh_complete"],
        "baseline_gates_passed": report["baseline_gates_passed"],
        "failed_gates": report["failed_gates"],
        "runtime_changed": False,
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
    parser.add_argument("--run-source-refresh", action="store_true")
    parser.add_argument("--run-compatible-regression", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    config = _load(PRECOMMIT)
    if args.run_source_refresh:
        run_source_refresh(config)
    regression = run_compatible_regression() if args.run_compatible_regression else None
    report = build_report(compatible_regression=regression)
    if args.write:
        write_outputs(report)
    print(json.dumps({
        "refresh_complete": report["refresh_complete"],
        "baseline_gates_passed": report["baseline_gates_passed"],
        "failed_gates": report["failed_gates"],
        "compiler": {
            "canonical": f"{report['compiler']['canonical']['exact']}/{report['compiler']['canonical']['total']}",
            "surface": f"{report['compiler']['surface']['exact']}/{report['compiler']['surface']['total']}",
            "counterfactual": f"{report['compiler']['counterfactual']['detected']}/{report['compiler']['counterfactual']['total']}",
        },
        "add_source_surface": f"{report['add_source_backed']['surface']['passed']}/{report['add_source_backed']['surface']['total']}",
        "actual_plan_failed_dimensions": report["actual_plan_source_backed"]["failed_runtime_dimensions"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if report["baseline_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
