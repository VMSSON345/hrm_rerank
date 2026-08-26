#!/usr/bin/env python3
"""Validate the artifact-minimal postimplementation diagnostic-refresh precommit."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/finhrm_qp4_post_question_surface_residual_implementation_diagnostic_refresh_precommit_v1.json"
MARKDOWN = ROOT / "reports/finhrm_qp4_post_question_surface_residual_implementation_diagnostic_refresh_precommit_v1.md"
LOCK = ROOT / "reports/finhrm_qp4_post_question_surface_residual_implementation_diagnostic_refresh_precommit_lock_v1.json"


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


def validate() -> dict[str, Any]:
    config = _load(CONFIG)
    failed: list[str] = []

    def check(value: bool, name: str) -> None:
        if not value:
            failed.append(name)

    for name, parent in config["parents"].items():
        path = ROOT / parent["path"]
        check(path.is_file() and _digest(path) == parent["sha256"], f"parent:{name}")
    report = _load(ROOT / config["parents"]["implementation_report"]["path"])
    lock = _load(ROOT / config["parents"]["implementation_lock"]["path"])
    check(report["implementation_gate_passed"] is True, "implementation_complete")
    check(lock["implementation_gate_passed"] is True, "implementation_lock")
    check(report["compiler"]["full_inventory"] == "242/242", "compiler_control")
    check(report["add_source"]["surface"]["passed"] == 33, "add_passed")
    check(report["add_source"]["surface"]["total"] == 35, "add_total")
    check(report["add_source"]["surface"]["by_category"]["SHORT_LONG_EQUIVALENT"] == {
        "total": 5,
        "passed": 3,
        "pass_rate": 0.6,
    }, "add_residual_group")
    check(report["actual_plan_source"]["queries"] == 24, "actual_queries")
    check(report["actual_plan_source"]["accepted_wrong"] == 0, "actual_accepted_wrong")
    scope = config["measurement_scope"]
    check(scope["runtime_file_whitelist"] == [], "empty_runtime_whitelist")
    check(scope["compiler_no_regression_records"] == 242, "compiler_control_target")
    check(scope["add_family_queries"] + scope["actual_plan_queries"] == scope["source_observations"] == 29, "observation_count")
    check(scope["persistent_raw_output_authorized"] is False, "no_raw_output")
    for relative, expected in config["frozen_runtime_hashes"].items():
        check(_digest(ROOT / relative) == expected, f"runtime:{relative}")
    check(_digest(ROOT / config["checkpoint"]["path"]) == config["checkpoint"]["sha256"], "checkpoint")
    check(all(value is False for value in config["invariants"].values()), "invariants")
    check(config["artifact_policy"]["raw_or_intermediate_artifacts_persisted"] is False, "artifact_policy")
    return {
        "precommit_gate_passed": not failed,
        "failed_gates": failed,
        "runtime_changed": False,
        "runtime_file_whitelist": [],
        "source_observations": scope["source_observations"],
        "authorized_next_step": config["authorized_next_step_after_lock"] if not failed else "Resolve diagnostic-refresh precommit failures without changing runtime.",
    }


def main() -> int:
    config = _load(CONFIG)
    result = validate()
    MARKDOWN.write_text(
        "# QP4 postimplementation residual diagnostic-refresh precommit v1\n\n"
        f"- Gate: `{'PASS' if result['precommit_gate_passed'] else 'FAIL'}`; runtime unchanged.\n"
        "- Scope: all 5 ADD short/long family records plus all 24 ACTUAL/PLAN queries; 242/242 compiler inventory remains a no-regression control.\n"
        "- Source errors stay in the denominator and stop at source; gold is post-hoc only. No raw/intermediate artifact may persist.\n"
        "- This step authorizes measurement only, not a runtime edit. QP4 remains incomplete; QP5 and Phase D remain locked.\n",
        encoding="utf-8",
    )
    artifacts = [_artifact(CONFIG), _artifact(Path(__file__).resolve()), _artifact(MARKDOWN)]
    lock = {
        "schema": "finhrm-qp4-post-question-surface-residual-implementation-diagnostic-refresh-precommit-lock-v1",
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
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["precommit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
