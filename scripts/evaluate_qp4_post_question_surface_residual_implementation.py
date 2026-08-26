#!/usr/bin/env python3
"""Run the authorized QP4 question-plan residual implementation replay."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any
import unittest

from scripts.evaluate_qp4_development_baseline import _semantic_projection
from scripts.evaluate_qp4_post_unit_scale_baseline_diagnostic_review import (
    _actual_plan_passes,
)
from scripts.evaluate_qp4_question_surface_implementation import (
    compiler_summary,
    run_source_replay,
    source_summary,
)


ROOT = Path(__file__).resolve().parents[1]
PRECOMMIT = ROOT / "config/finhrm_qp4_post_question_surface_residual_implementation_precommit_v1.json"
PRECOMMIT_MARKDOWN = ROOT / "reports/finhrm_qp4_post_question_surface_residual_implementation_precommit_v1.md"
PRECOMMIT_LOCK = ROOT / "reports/finhrm_qp4_post_question_surface_residual_implementation_precommit_lock_v1.json"
IMPLEMENTATION_CONFIG = ROOT / "config/finhrm_qp4_question_surface_implementation_v1.json"
ACTUAL_CONFIG = ROOT / "config/finhrm_qp4_actual_plan_ratio_direction_source_cohort_v1.json"
SAFE_PRECOMMIT = ROOT / "config/finhrm_qp3_core_s4_safe_replay_precommit_v1.json"
REPORT = ROOT / "reports/finhrm_qp4_post_question_surface_residual_implementation_v1.json"
MARKDOWN = ROOT / "reports/finhrm_qp4_post_question_surface_residual_implementation_v1.md"
LOCK = ROOT / "reports/finhrm_qp4_post_question_surface_residual_implementation_lock_v1.json"
TEST = ROOT / "tests/test_qp4_post_question_surface_residual_implementation.py"

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
    "tests.test_qp4_unit_scale_proof_contract_implementation_report.QP4UnitScaleProofContractImplementationReportTests.test_source_gate_closes_on_diverse_real_pdf_replay",
    "tests.test_qp4_unit_scale_proof_contract_implementation_report.QP4UnitScaleProofContractImplementationReportTests.test_frozen_execution_permutation_and_counterfactual_gates_still_pass",
    "tests.test_qp4_unit_scale_proof_contract_implementation_report.QP4UnitScaleProofContractImplementationReportTests.test_implementation_gate_stays_open_on_exact_frozen_action_conflict",
    "tests.test_qp4_unit_scale_proof_contract_implementation_report.QP4UnitScaleProofContractImplementationReportTests.test_lock_does_not_authorize_more_runtime_or_later_phases",
    "tests.test_qp4_question_surface_implementation",
    "tests.test_qp4_post_question_surface_residual_implementation",
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


def _verify_authorization(config: dict[str, Any]) -> None:
    lock = _load(PRECOMMIT_LOCK)
    if not lock.get("precommit_gate_passed"):
        raise ValueError("Residual implementation precommit is not closed")
    if lock.get("authorized_next_step") != config["authorized_next_step_after_lock"]:
        raise ValueError("Implementation authorization mismatch")
    if lock.get("runtime_file_whitelist") != ["src/hrm_finance_qa/question_planner.py"]:
        raise ValueError("Unexpected runtime whitelist")
    if lock.get("exact_existing_symbols") != config["implementation_scope"]["exact_existing_symbols"]:
        raise ValueError("Unexpected symbol whitelist")
    for relative, expected in config["frozen_runtime_hashes"].items():
        if relative == "src/hrm_finance_qa/question_planner.py":
            continue
        if _digest(ROOT / relative) != expected:
            raise ValueError(f"Runtime outside whitelist changed: {relative}")
    planner = ROOT / "src/hrm_finance_qa/question_planner.py"
    if _digest(planner) == config["frozen_runtime_hashes"][planner.relative_to(ROOT).as_posix()]:
        raise ValueError("Authorized planner implementation was not applied")
    checkpoint = ROOT / config["checkpoint"]["path"]
    if _digest(checkpoint) != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint changed")


def _compiler_measurement() -> dict[str, Any]:
    from hrm_finance_qa.question_planner import compile_question_plan

    config = _load(IMPLEMENTATION_CONFIG)
    compiler = compiler_summary(config)
    exact = (
        compiler["canonical"]["exact"]
        + compiler["ambiguity"]["exact"]
        + compiler["frozen_surface"]["exact"]
        + compiler["corrected_hold_ahead"]["exact"]
        + compiler["minimal_pairs"]["detected"]
    )
    total = (
        compiler["canonical"]["total"]
        + compiler["ambiguity"]["total"]
        + compiler["frozen_surface"]["total"]
        + compiler["corrected_hold_ahead"]["total"]
        + compiler["minimal_pairs"]["total"]
    )
    add_raw = _load(ROOT / "reports/finhrm_qp4_post_question_surface_add_source_baseline_v1.json")
    add_rows = [row for row in add_raw["surface_results"] if not row["invariant"]]
    add_exact = sum(
        _semantic_projection(compile_question_plan(row["question"]))
        == _semantic_projection(compile_question_plan(row["parent_question"]))
        for row in add_rows
    )
    actual = _load(ACTUAL_CONFIG)
    reference_rows = [row["pairs"]["REFERENCE_BASE"] for row in actual["anchors"]]
    reference_exact = sum(
        _actual_plan_passes(
            _semantic_projection(compile_question_plan(row["mutated_question"])),
            row["mutated_program"],
        )
        for row in reference_rows
    )
    return {
        "full_inventory": f"{exact}/{total}",
        "details": compiler,
        "add_surface_projection": f"{add_exact}/{len(add_rows)}",
        "actual_plan_reference_base_projection": f"{reference_exact}/{len(reference_rows)}",
    }


def _actual_summary(raw: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in raw["results"]:
        grouped.setdefault(row["dimension"], []).append(row)
    dimensions: dict[str, Any] = {}
    accepted_wrong = 0
    source_errors = 0
    for name, rows in sorted(grouped.items()):
        base_gold = sum(row["base"]["gold_answer_match"] for row in rows)
        mutated_gold = sum(row["mutated"]["gold_answer_match"] for row in rows)
        paired = sum(row["paired_gold_answer"] for row in rows)
        false_accepts = sum(row["old_winner_false_accept"] for row in rows)
        for row in rows:
            for label in ("base", "mutated"):
                observation = row[label]
                source_errors += int("error" in observation)
                signature = observation.get("signature", {})
                accepted_wrong += int(
                    signature.get("action") == "ANSWER"
                    and not observation["gold_answer_match"]
                )
        dimensions[name] = {
            "pairs": len(rows),
            "companies": len({row["company_key"] for row in rows}),
            "sources": len({row["source_sha256"] for row in rows}),
            "document_layouts": len({row["document_layout_family"] for row in rows}),
            "table_layouts": len({row["table_layout_family"] for row in rows}),
            "base_gold_answers": base_gold,
            "mutated_gold_answers": mutated_gold,
            "paired_gold_answers": paired,
            "old_winner_false_accepts": false_accepts,
        }
    return {
        "queries": len(raw["results"]) * 2,
        "source_errors": source_errors,
        "accepted_wrong": accepted_wrong,
        "dimensions": dimensions,
    }


def _run_regression() -> dict[str, Any]:
    suite = unittest.defaultTestLoader.loadTestsFromNames(REGRESSION_MODULES)
    expected = suite.countTestCases()
    command = [sys.executable, "-m", "unittest", *REGRESSION_MODULES]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    match = re.search(r"Ran\s+(\d+)\s+tests?", output)
    observed = int(match.group(1)) if match else 0
    return {
        "expected": expected,
        "observed": observed,
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0 and observed == expected,
        "output_tail": output[-1200:],
    }


def run() -> dict[str, Any]:
    from scripts.evaluate_qp3_core_s4_safe_replay import build_report as build_safe
    from scripts.evaluate_qp4_actual_plan_source_cohort import run_source_baseline

    config = _load(PRECOMMIT)
    _verify_authorization(config)
    compiler = _compiler_measurement()
    implementation_config = _load(IMPLEMENTATION_CONFIG)
    actual_config = deepcopy(_load(ACTUAL_CONFIG))
    actual_config["runtime_hashes"] = {
        relative: _digest(ROOT / relative)
        for relative in actual_config["runtime_hashes"]
    }
    with TemporaryDirectory(prefix="finhrm_qp4_residual_") as directory:
        temporary = Path(directory)
        add_path = temporary / "add.json"
        actual_path = temporary / "actual.json"
        run_source_replay(implementation_config, add_path)
        add_raw = _load(add_path)
        add = source_summary(implementation_config, add_path)
        run_source_baseline(actual_config, actual_path)
        actual_raw = _load(actual_path)
        actual = _actual_summary(actual_raw)

    add_source_errors = sum("error" in row for row in add_raw["surface_results"])
    add_accepted_wrong = sum(
        row.get("signature", {}).get("action") == "ANSWER" and not row["invariant"]
        for row in add_raw["surface_results"]
    )
    add_false_accepts = sum(
        row["old_winner_false_accept"] for row in add_raw["counterfactual_results"]
    )
    add_measurement = {
        "queries": len(add_raw["surface_results"]) + len(add_raw["counterfactual_results"]),
        "companies": add_raw["companies"],
        "sources": add_raw["source_documents"],
        "real_layouts": add_raw["real_layout_families"],
        "surface": add["surface"],
        "previously_passing_retained": f"{add['previously_passing_retained']}/{add['previously_passing_total']}",
        "source_errors": add_source_errors,
        "accepted_wrong": add_accepted_wrong,
        "counterfactual_false_accepts": add_false_accepts,
    }
    safe = build_safe(SAFE_PRECOMMIT)
    safe_results = safe["results"]
    regression = _run_regression()
    controls = config["required_controls"]
    runtime_after = {
        relative: _digest(ROOT / relative)
        for relative in config["frozen_runtime_hashes"]
    }
    checks = {
        "full_compiler_inventory": compiler["full_inventory"] == controls["full_compiler_inventory_exact"],
        "add_surface_projection": compiler["add_surface_projection"] == controls["add_surface_projection_exact"],
        "actual_plan_reference_base_projection": compiler["actual_plan_reference_base_projection"] == controls["actual_plan_reference_base_projection_exact"],
        "all_source_queries_replayed": add_measurement["queries"] == 50 and actual["queries"] == 24,
        "source_errors_remain_failures": all(
            not row.get("invariant", False)
            for row in add_raw["surface_results"]
            if "error" in row
        ),
        "accepted_wrong_zero": add_measurement["accepted_wrong"] + actual["accepted_wrong"] == controls["accepted_wrong"],
        "old_winner_false_accepts_zero": add_measurement["counterfactual_false_accepts"] + sum(
            row["old_winner_false_accepts"] for row in actual["dimensions"].values()
        ) == controls["old_winner_counterfactual_false_accepts"],
        "regression": regression["passed"],
        "safe_actions": safe_results["gold_actions"] == controls["safe_replay_actions"],
        "safe_execution": f"{safe_results['gold_execution_correct']}/{safe_results['gold_total']}" == controls["safe_replay_execution"],
        "safe_permutation": f"{safe_results['permutation_exact']}/{safe_results['permutation_total']}" == controls["safe_replay_permutation"],
        "safe_counterfactuals": safe_results["counterfactual_generated"] == controls["safe_replay_counterfactuals"],
        "safe_false_accepts": safe_results["counterfactual_false_accepts"] == controls["safe_replay_counterfactual_false_accepts"],
        "runtime_outside_whitelist_unchanged": all(
            runtime_after[path] == before
            for path, before in config["frozen_runtime_hashes"].items()
            if path != "src/hrm_finance_qa/question_planner.py"
        ),
        "checkpoint_unchanged": _digest(ROOT / config["checkpoint"]["path"]) == config["checkpoint"]["sha256"],
        "no_persistent_temporary_artifacts": not any(ROOT.rglob("*finhrm_qp4_residual_*")),
    }
    failed = [name for name, passed in checks.items() if not passed]
    implementation_complete = not failed
    return {
        "schema": "finhrm-qp4-post-question-surface-residual-implementation-report-v1",
        "created_at": config["created_at"],
        "classification": "development_only_qp4_generic_question_plan_residual_implementation",
        "status": "IMPLEMENTATION_COMPLETE" if implementation_complete else "IMPLEMENTATION_GATE_FAILED",
        "implementation_gate_passed": implementation_complete,
        "failed_gates": failed,
        "checks": checks,
        "compiler": compiler,
        "add_source": add_measurement,
        "actual_plan_source": actual,
        "regression": regression,
        "safe_replay": {
            "gate": safe["safe_replay_gate_passed"],
            "actions": safe_results["gold_actions"],
            "execution": f"{safe_results['gold_execution_correct']}/{safe_results['gold_total']}",
            "permutation": f"{safe_results['permutation_exact']}/{safe_results['permutation_total']}",
            "counterfactuals": safe_results["counterfactual_generated"],
            "counterfactual_false_accepts": safe_results["counterfactual_false_accepts"],
        },
        "runtime_before": config["frozen_runtime_hashes"],
        "runtime_after": runtime_after,
        "changed_runtime_files": ["src/hrm_finance_qa/question_planner.py"],
        "changed_symbols": config["implementation_scope"]["exact_existing_symbols"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "artifact_policy": config["artifact_policy"],
        "invariants": config["invariants"] | {"runtime_changed": True},
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": (
            "QP4-POST-QUESTION-SURFACE-RESIDUAL-IMPLEMENTATION-DIAGNOSTIC-REFRESH-PRECOMMIT: freeze one measurement-only replay of remaining source-layer, retrieval, binding and proof residuals without changing runtime."
            if implementation_complete
            else "Resolve implementation gate failures inside the existing one-file/three-symbol whitelist."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    return (
        "# QP4 post-question-surface residual implementation v1\n\n"
        f"- Gate: `{'PASS' if report['implementation_gate_passed'] else 'FAIL'}`.\n"
        f"- Compiler: `{report['compiler']['full_inventory']}`; ADD surface `{report['compiler']['add_surface_projection']}`; reference base `{report['compiler']['actual_plan_reference_base_projection']}`.\n"
        f"- Source replay: ADD `{report['add_source']['queries']}` queries and ACTUAL/PLAN `{report['actual_plan_source']['queries']}` queries; accepted-wrong `{report['add_source']['accepted_wrong'] + report['actual_plan_source']['accepted_wrong']}`.\n"
        f"- Regression: `{report['regression']['observed']}/{report['regression']['expected']}`; safe execution/permutation `{report['safe_replay']['execution']}/{report['safe_replay']['permutation']}`; counterfactual false accepts `{report['safe_replay']['counterfactual_false_accepts']}`.\n"
        "- Only `question_planner.py` changed; checkpoint and every runtime file outside the whitelist are unchanged. QP4 remains incomplete; QP5 and Phase D remain locked.\n"
    )


def write_outputs(report: dict[str, Any]) -> None:
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    MARKDOWN.write_text(_markdown(report), encoding="utf-8")
    artifacts = [
        _artifact(PRECOMMIT),
        _artifact(PRECOMMIT_MARKDOWN),
        _artifact(PRECOMMIT_LOCK),
        _artifact(Path(__file__).resolve()),
        _artifact(TEST),
        _artifact(REPORT),
        _artifact(MARKDOWN),
    ]
    lock = {
        "schema": "finhrm-qp4-post-question-surface-residual-implementation-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["implementation_gate_passed"] else "failed",
        "implementation_gate_passed": report["implementation_gate_passed"],
        "failed_gates": report["failed_gates"],
        "runtime_before": report["runtime_before"],
        "runtime_after": report["runtime_after"],
        "changed_runtime_files": report["changed_runtime_files"],
        "changed_symbols": report["changed_symbols"],
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
        "implementation_gate_passed": report["implementation_gate_passed"],
        "failed_gates": report["failed_gates"],
        "compiler": report["compiler"]["full_inventory"],
        "add_surface": report["compiler"]["add_surface_projection"],
        "reference_base": report["compiler"]["actual_plan_reference_base_projection"],
        "regression": report["regression"]["observed"],
        "safe_replay": report["safe_replay"],
    }, ensure_ascii=False, sort_keys=True))
    return 0 if report["implementation_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
