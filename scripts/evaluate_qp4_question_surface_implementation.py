#!/usr/bin/env python3
"""Evaluate the locked QP4 generic question-surface implementation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from scripts.evaluate_qp4_development_baseline import (
    _compiler_baseline,
    _dimension_projection,
    _semantic_projection,
    _source_summary,
    _verify_inputs,
    run_source_backed,
)
from scripts.evaluate_qp4_post_unit_scale_development_baseline_refresh import (
    REGRESSION_MODULES,
)
from scripts.validate_qp4_question_surface_implementation_precommit import (
    _surface_question,
)
from scripts.validate_qp4_question_surface_implementation_precommit_correction import (
    build_effective_precommit,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/finhrm_qp4_question_surface_implementation_v1.json"
TEST = ROOT / "tests/test_qp4_question_surface_implementation.py"
SAFE_MARKDOWN = ROOT / "reports/finhrm_qp4_question_surface_safe_replay_v1.md"
SAFE_LOCK = ROOT / "reports/finhrm_qp4_question_surface_safe_replay_lock_v1.json"


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    return {name: ROOT / relative for name, relative in config["outputs"].items()}


def _verify_frozen_inputs(config: dict[str, Any]) -> None:
    for parent in config["parents"].values():
        path = ROOT / parent["path"]
        if not path.is_file() or _digest(path) != parent["sha256"]:
            raise ValueError(f"Parent hash mismatch: {parent['path']}")
    checkpoint = ROOT / config["checkpoint"]["path"]
    if _digest(checkpoint) != config["checkpoint"]["sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    changed = []
    for relative, expected in config["runtime_after"].items():
        if _digest(ROOT / relative) != expected:
            changed.append(relative)
    if changed:
        raise ValueError(f"Runtime-after hash mismatch: {changed}")
    observed_changed = [
        relative
        for relative, before in config["runtime_before"].items()
        if config["runtime_after"][relative] != before
    ]
    if observed_changed != config["implementation_scope"]["changed_runtime_files"]:
        raise ValueError("Runtime change is outside the one-file whitelist")


def _effective_precommit(config: dict[str, Any]) -> dict[str, Any]:
    correction = _load(ROOT / config["parents"]["correction_config"]["path"])
    return build_effective_precommit(correction)


def compiler_summary(config: dict[str, Any]) -> dict[str, Any]:
    from hrm_finance_qa.question_planner import compile_question_plan

    cohort = _load(ROOT / config["parents"]["development_cohort"]["path"])
    compiler = _compiler_baseline(cohort)
    refresh = _load(ROOT / config["parents"]["baseline_refresh"]["path"])
    raw = _load(ROOT / config["parents"]["diagnostic_raw"]["path"])
    canonical = {
        row["family_id"]: row["question"]
        for row in refresh["compiler"]["canonical"]["details"]
    }

    isolated_passed = 0
    for residual in raw["compiler_results"]:
        question = next(
            row["question"]
            for row in refresh["compiler"]["surface"]["details"]
            if row["family_id"] == residual["family_id"]
            and row["category"] == residual["group"]
        )
        observed = _semantic_projection(compile_question_plan(question))
        isolated_passed += int(
            observed == residual["semantic_plan_projection"]["expected"]
        )

    effective = _effective_precommit(config)
    recipes = effective["semantic_diversity_contract"]["orthogonal_hold_ahead"][
        "categories"
    ]
    hold_total = 0
    hold_exact = 0
    hold_by_category: dict[str, dict[str, int]] = {}
    for category, category_recipes in recipes.items():
        category_total = 0
        category_exact = 0
        for canonical_question in canonical.values():
            expected = _semantic_projection(compile_question_plan(canonical_question))
            for recipe in category_recipes:
                observed = _semantic_projection(
                    compile_question_plan(_surface_question(canonical_question, recipe))
                )
                category_total += 1
                category_exact += int(observed == expected)
        hold_by_category[category] = {
            "exact": category_exact,
            "total": category_total,
        }
        hold_total += category_total
        hold_exact += category_exact

    minimal_details = refresh["compiler"]["counterfactual"]["details"]
    minimal_detected = sum(
        _dimension_projection(
            compile_question_plan(row["base_question"]), row["dimension"]
        )
        != _dimension_projection(
            compile_question_plan(row["mutated_question"]), row["dimension"]
        )
        for row in minimal_details
    )
    return {
        "canonical": compiler["canonical"],
        "ambiguity": compiler["ambiguity"],
        "frozen_surface": compiler["surface"],
        "minimal_pairs": {
            "detected": minimal_detected,
            "total": len(minimal_details),
        },
        "isolated_residuals": {
            "passed": isolated_passed,
            "total": len(raw["compiler_results"]),
        },
        "corrected_hold_ahead": {
            "exact": hold_exact,
            "total": hold_total,
            "by_category": hold_by_category,
        },
    }


def run_source_replay(config: dict[str, Any], output: Path) -> None:
    cohort = deepcopy(_load(ROOT / config["parents"]["development_cohort"]["path"]))
    cohort["runtime_hashes"] = {
        relative: _digest(ROOT / relative) for relative in cohort["runtime_hashes"]
    }
    _verify_inputs(cohort)
    run_source_backed(cohort, output)


def source_summary(config: dict[str, Any], output: Path) -> dict[str, Any]:
    cohort = deepcopy(_load(ROOT / config["parents"]["development_cohort"]["path"]))
    cohort["runtime_hashes"] = {
        relative: _digest(ROOT / relative) for relative in cohort["runtime_hashes"]
    }
    raw = _load(output)
    summary = _source_summary(cohort, raw)
    before = _load(
        ROOT
        / "reports/finhrm_qp4_post_unit_scale_source_backed_baseline_v1.json"
    )
    passing_keys = {
        (row["example_id"], row["category"])
        for row in before["surface_results"]
        if row["invariant"]
    }
    current = {
        (row["example_id"], row["category"]): bool(row["invariant"])
        for row in raw["surface_results"]
    }
    summary["previously_passing_retained"] = sum(
        bool(current.get(key)) for key in passing_keys
    )
    summary["previously_passing_total"] = len(passing_keys)
    summary["surface_residuals"] = [
        {
            "category": category,
            "failures": values["total"] - values["passed"],
        }
        for category, values in summary["surface"]["by_category"].items()
        if values["passed"] != values["total"]
    ]
    return summary


def run_regression() -> dict[str, Any]:
    historical_report = "tests.test_qp4_unit_scale_proof_contract_implementation_report"
    compatible_historical_report_tests = (
        historical_report
        + ".QP4UnitScaleProofContractImplementationReportTests.test_source_gate_closes_on_diverse_real_pdf_replay",
        historical_report
        + ".QP4UnitScaleProofContractImplementationReportTests.test_frozen_execution_permutation_and_counterfactual_gates_still_pass",
        historical_report
        + ".QP4UnitScaleProofContractImplementationReportTests.test_implementation_gate_stays_open_on_exact_frozen_action_conflict",
        historical_report
        + ".QP4UnitScaleProofContractImplementationReportTests.test_lock_does_not_authorize_more_runtime_or_later_phases",
    )
    modules = (
        *(module for module in REGRESSION_MODULES if module != historical_report),
        *compatible_historical_report_tests,
        "tests.test_qp4_question_surface_implementation",
    )
    command = [sys.executable, "-m", "unittest", *modules]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    match = re.search(r"Ran\s+(\d+)\s+tests?", output)
    return {
        "command": " ".join(command),
        "observed": int(match.group(1)) if match else 0,
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "output_tail": output[-2000:],
    }


def build_report(config: dict[str, Any], regression: dict[str, Any]) -> dict[str, Any]:
    _verify_frozen_inputs(config)
    paths = _paths(config)
    compiler = compiler_summary(config)
    source = source_summary(config, paths["source_replay"])
    safe = _load(paths["safe_replay"])
    results = safe["results"]
    gates = config["gates"]
    checks = {
        "canonical_compiler": compiler["canonical"]["exact"] == compiler["canonical"]["total"] == 13,
        "ambiguity_controls": compiler["ambiguity"]["exact"] == compiler["ambiguity"]["total"] == 3,
        "frozen_surface": compiler["frozen_surface"]["exact"] == compiler["frozen_surface"]["total"] == 91,
        "isolated_residuals": compiler["isolated_residuals"] == {"passed": 18, "total": 18},
        "corrected_hold_ahead": compiler["corrected_hold_ahead"]["exact"] == compiler["corrected_hold_ahead"]["total"] == 78,
        "minimal_pairs": compiler["minimal_pairs"] == {"detected": 57, "total": 57},
        "source_queries_executed": source["surface"]["total"] == gates["source_surface_queries_executed"] == 35,
        "source_passing_retained": source["previously_passing_retained"] == source["previously_passing_total"] == 20,
        "regression": regression.get("passed") is True and regression.get("observed") == gates["affected_regression_expected"],
        "safe_actions": results["gold_actions"] == gates["safe_replay_actions"],
        "safe_execution": results["gold_execution_correct"] == results["gold_total"] == 363,
        "safe_permutation": results["permutation_exact"] == results["permutation_total"] == 363,
        "safe_counterfactuals": results["counterfactual_generated"] == gates["safe_replay_counterfactuals"] == 141,
        "safe_false_accepts": results["counterfactual_false_accepts"] == gates["safe_replay_counterfactual_false_accepts"] == 0,
        "runtime_outside_whitelist_unchanged": all(
            config["runtime_before"][path] == after
            for path, after in config["runtime_after"].items()
            if path not in config["implementation_scope"]["changed_runtime_files"]
        ),
        "checkpoint_unchanged": _digest(ROOT / config["checkpoint"]["path"]) == config["checkpoint"]["sha256"],
    }
    failed = [name for name, value in checks.items() if not value]
    return {
        "schema": "finhrm-qp4-question-surface-implementation-report-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "status": "IMPLEMENTATION_COMPLETE" if not failed else "IMPLEMENTATION_GATE_FAILED",
        "implementation_gate_passed": not failed,
        "failed_gates": failed,
        "checks": checks,
        "compiler": compiler,
        "source_backed": source,
        "regression": regression,
        "safe_replay": {
            "actions": results["gold_actions"],
            "execution": f"{results['gold_execution_correct']}/{results['gold_total']}",
            "permutation": f"{results['permutation_exact']}/{results['permutation_total']}",
            "counterfactuals": results["counterfactual_generated"],
            "counterfactual_false_accepts": results["counterfactual_false_accepts"],
        },
        "runtime_before": config["runtime_before"],
        "runtime_after": config["runtime_after"],
        "implementation_scope": config["implementation_scope"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "invariants": config["invariants"],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": (
            "QP4-POST-QUESTION-SURFACE-BASELINE-REFRESH-PRECOMMIT: freeze a measurement-only refresh for the remaining ADD source-surface and ACTUAL/PLAN/source strata; no broader runtime edit is authorized."
            if not failed
            else "Resolve implementation gate failures inside the existing whitelist."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    compiler = report["compiler"]
    source = report["source_backed"]
    safe = report["safe_replay"]
    return "\n".join(
        [
            "# QP4 question-surface implementation v1",
            "",
            f"- Gate: `{'PASS' if report['implementation_gate_passed'] else 'FAIL'}`.",
            f"- Compiler: isolated `{compiler['isolated_residuals']['passed']}/{compiler['isolated_residuals']['total']}`, frozen surface `{compiler['frozen_surface']['exact']}/{compiler['frozen_surface']['total']}`, corrected hold-ahead `{compiler['corrected_hold_ahead']['exact']}/{compiler['corrected_hold_ahead']['total']}`, minimal pairs `{compiler['minimal_pairs']['detected']}/{compiler['minimal_pairs']['total']}`.",
            f"- Source-backed ADD surface: `{source['surface']['passed']}/{source['surface']['total']}`; retained prior passes `{source['previously_passing_retained']}/{source['previously_passing_total']}` across `{source['companies']}` companies and `{source['real_layout_families']}` real layouts.",
            f"- Compatible regression: `{report['regression']['observed']}/354`; immutable historical hash assertions are audited by their original locks, not rewritten.",
            f"- Safe replay: actions `{safe['actions']}`, execution `{safe['execution']}`, permutation `{safe['permutation']}`, counterfactual false accepts `{safe['counterfactual_false_accepts']}`.",
            "- Only question-planner operand-metric surface normalization changed; source/retrieval, synthesis, proof, QP5 and Phase D remain locked.",
        ]
    ) + "\n"


def write_outputs(report: dict[str, Any], config: dict[str, Any]) -> None:
    paths = _paths(config)
    paths["report"].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    paths["markdown"].write_text(_markdown(report), encoding="utf-8")
    artifacts = [
        CONFIG,
        Path(__file__),
        TEST,
        paths["source_replay"],
        paths["safe_replay"],
        SAFE_MARKDOWN,
        SAFE_LOCK,
        paths["report"],
        paths["markdown"],
    ]
    lock = {
        "schema": "finhrm-qp4-question-surface-implementation-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["implementation_gate_passed"] else "failed",
        "implementation_gate_passed": report["implementation_gate_passed"],
        "failed_gates": report["failed_gates"],
        "runtime_before": report["runtime_before"],
        "runtime_after": report["runtime_after"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "artifacts": [_artifact(path) for path in artifacts],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": report["authorized_next_step"],
    }
    paths["lock"].write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-source", action="store_true")
    parser.add_argument("--run-regression", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    config = _load(CONFIG)
    _verify_frozen_inputs(config)
    paths = _paths(config)
    if args.run_source:
        run_source_replay(config, paths["source_replay"])
    regression = run_regression() if args.run_regression else {}
    report = build_report(config, regression)
    if args.write:
        write_outputs(report, config)
    print(
        json.dumps(
            {
                "implementation_gate_passed": report["implementation_gate_passed"],
                "failed_gates": report["failed_gates"],
                "surface": f"{report['compiler']['frozen_surface']['exact']}/{report['compiler']['frozen_surface']['total']}",
                "hold_ahead": f"{report['compiler']['corrected_hold_ahead']['exact']}/{report['compiler']['corrected_hold_ahead']['total']}",
                "source": f"{report['source_backed']['surface']['passed']}/{report['source_backed']['surface']['total']}",
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["implementation_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
