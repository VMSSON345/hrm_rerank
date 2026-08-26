#!/usr/bin/env python3
"""Validate and lock the minimal QP4 question-plan residual implementation scope."""

from __future__ import annotations

import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/finhrm_qp4_post_question_surface_residual_implementation_precommit_v1.json"
MARKDOWN = ROOT / "reports/finhrm_qp4_post_question_surface_residual_implementation_precommit_v1.md"
LOCK = ROOT / "reports/finhrm_qp4_post_question_surface_residual_implementation_precommit_lock_v1.json"


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


def validate(config_path: Path = CONFIG) -> dict[str, Any]:
    config = _load(config_path)
    failed: list[str] = []

    def check(value: bool, name: str) -> None:
        if not value:
            failed.append(name)

    for name, parent in config["parents"].items():
        path = ROOT / parent["path"]
        check(path.is_file() and _digest(path) == parent["sha256"], f"parent:{name}")
    diagnostic = _load(ROOT / config["parents"]["diagnostic_report"]["path"])
    diagnostic_lock = _load(ROOT / config["parents"]["diagnostic_lock"]["path"])
    check(diagnostic["diagnostic_complete"] is True, "diagnostic_complete")
    check(diagnostic_lock["diagnostic_gate_passed"] is True, "diagnostic_lock")
    check(diagnostic["compiler_control"]["exact"] == diagnostic["compiler_control"]["records"] == 242, "compiler_control")
    check(diagnostic["source"]["queries"] == 40, "source_observations")
    check(diagnostic["source"]["source_errors"] == 12, "source_errors")
    check(diagnostic["source"]["by_first_failing_layer"] == config["frozen_diagnostic"]["first_failing_layers"], "layer_inventory")

    expected = {
        (row["track"], row["group"]): row
        for row in config["frozen_diagnostic"]["generic_question_plan_strata"]
    }
    observed = {
        (row["track"], row["group"]): row
        for row in diagnostic["implementation_precommit_candidates"]
    }
    check(set(observed) == set(expected), "candidate_keys")
    for key, frozen in expected.items():
        row = observed.get(key, {})
        for field in ("queries", "companies", "sources", "real_layouts"):
            check(row.get(field) == frozen[field], f"candidate:{key}:{field}")
        check(row.get("first_failing_layer") == "QUESTION_PLAN", f"candidate:{key}:layer")
        check(row.get("generic_evidence") is True, f"candidate:{key}:generic")
        check(row.get("eligible_existing_whitelist_files") == ["src/hrm_finance_qa/question_planner.py"], f"candidate:{key}:file")

    scope = config["implementation_scope"]
    check(scope["runtime_file_whitelist"] == ["src/hrm_finance_qa/question_planner.py"], "one_file_whitelist")
    check(
        scope["exact_existing_symbols"]
        == ["_addition_role_hypotheses", "_ratio_roles", "compile_question_plan"],
        "exact_symbols",
    )
    check(scope["new_runtime_symbols_authorized"] == [], "no_new_runtime_symbols")
    check(
        not any(
            scope[name]
            for name in (
                "retrieval_edit_authorized",
                "source_or_extraction_edit_authorized",
                "program_synthesis_edit_authorized",
                "symbolic_verifier_edit_authorized",
                "direction_contract_edit_authorized",
                "threshold_checkpoint_or_model_change_authorized",
            )
        ),
        "downstream_locked",
    )
    planner_path = ROOT / scope["runtime_file_whitelist"][0]
    names = {
        node.name
        for node in ast.parse(planner_path.read_text(encoding="utf-8")).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    check(set(scope["exact_existing_symbols"]) <= names, "symbols_exist")

    for relative, expected_hash in config["frozen_runtime_hashes"].items():
        check(_digest(ROOT / relative) == expected_hash, f"runtime:{relative}")
    checkpoint = ROOT / config["checkpoint"]["path"]
    check(_digest(checkpoint) == config["checkpoint"]["sha256"], "checkpoint")
    check(all(value is False for value in config["invariants"].values()), "invariants")
    policy = config["artifact_policy"]
    check(policy == {
        "maximum_final_json": 1,
        "maximum_final_markdown": 1,
        "maximum_lock_or_manifest": 1,
        "raw_or_intermediate_artifacts_persisted": False,
        "temporary_artifacts_deleted_after_verification": True,
    }, "artifact_policy")
    return {
        "precommit_gate_passed": not failed,
        "failed_gates": failed,
        "runtime_changed": False,
        "runtime_file_whitelist": scope["runtime_file_whitelist"],
        "exact_existing_symbols": scope["exact_existing_symbols"],
        "generic_strata": list(expected.values()),
        "authorized_next_step": config["authorized_next_step_after_lock"] if not failed else "Resolve precommit failures without changing runtime.",
    }


def write_outputs(result: dict[str, Any], config_path: Path = CONFIG) -> None:
    config = _load(config_path)
    MARKDOWN.write_text(
        "# QP4 post-question-surface residual implementation precommit v1\n\n"
        f"- Gate: `{'PASS' if result['precommit_gate_passed'] else 'FAIL'}`; runtime unchanged.\n"
        "- Four generic `QUESTION_PLAN` strata are frozen: three ADD surface families and ACTUAL/PLAN reference base, each on 4 companies, 4 PDFs and 3 real layouts.\n"
        "- Runtime scope is one file and three existing symbols; source, retrieval, synthesis, verifier, direction, threshold, checkpoint and model remain locked.\n"
        "- Implementation must retain 242/242 compiler control, all source failures, zero accepted-wrong/false-accept, and 363/363 execution/permutation with 0/141 counterfactual false accepts.\n",
        encoding="utf-8",
    )
    artifacts = [_artifact(config_path), _artifact(Path(__file__).resolve()), _artifact(MARKDOWN)]
    lock = {
        "schema": "finhrm-qp4-post-question-surface-residual-implementation-precommit-lock-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "status": "complete" if result["precommit_gate_passed"] else "failed",
        **result,
        "runtime_hashes": config["frozen_runtime_hashes"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "artifact_policy": config["artifact_policy"],
        "artifacts": artifacts,
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
    }
    LOCK.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    result = validate(args.config)
    write_outputs(result, args.config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["precommit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
