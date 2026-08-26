#!/usr/bin/env python3
"""Validate and lock the QP4 question-planner UNIT_SCALE implementation scope."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "finhrm_qp4_unit_scale_question_planner_implementation_precommit_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp4_unit_scale_question_planner_implementation_precommit_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp4_unit_scale_question_planner_implementation_precommit_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp4_unit_scale_question_planner_implementation_precommit_lock_v1.json"
TEST_PATH = ROOT / "tests" / "test_qp4_unit_scale_question_planner_precommit.py"
PROTECTED = re.compile(r"(?:^|[/_.-])(test|holdout|final|blind|validation)(?:[/_.-]|$)", re.I)


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if PROTECTED.search(path.resolve().as_posix()):
        raise ValueError(f"Protected input is forbidden: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _check(condition: bool, name: str, failed: list[str]) -> None:
    if not condition:
        failed.append(name)


def build_report(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load(config_path)
    failed: list[str] = []
    for name, parent in config["parents"].items():
        path = ROOT / parent["path"]
        _check(path.is_file() and _digest(path) == parent["sha256"], f"parent_hash:{name}", failed)
    checkpoint = config["checkpoint"]
    _check(
        _digest(ROOT / checkpoint["path"]) == checkpoint["sha256"],
        "checkpoint_hash",
        failed,
    )
    for relative, expected in config["runtime_hashes_before_implementation"].items():
        _check(_digest(ROOT / relative) == expected, f"runtime_hash:{relative}", failed)

    baseline = _load(ROOT / config["parents"]["qp4_development_baseline"]["path"])
    compiler = baseline["compiler"]["counterfactual"]["by_dimension"]["UNIT_SCALE"]
    source = baseline["source_backed"]["counterfactual"]["by_dimension"]["UNIT_SCALE"]
    diagnosis = {row["failure_group"]: row for row in baseline["diagnosis"]}
    expected = config["generic_failure_evidence"]
    _check(compiler["pairs"] == expected["compiler"]["pairs"] == 13, "compiler_pairs", failed)
    _check(compiler["families"] == expected["compiler"]["families"] == 13, "compiler_families", failed)
    _check(compiler["detected"] == expected["compiler"]["detected"] == 0, "compiler_detected", failed)
    _check(source["runs"] == expected["source_backed"]["runs"] == 5, "source_runs", failed)
    _check(source["companies"] == expected["source_backed"]["companies"] == 5, "source_companies", failed)
    _check(source["sources"] == expected["source_backed"]["sources"] == 5, "source_documents", failed)
    _check(source["real_layouts"] == expected["source_backed"]["real_layouts"] == 3, "source_layouts", failed)
    _check(
        source["old_winner_false_accepts"]
        == expected["source_backed"]["old_winner_false_accepts"]
        == 4,
        "source_false_accepts",
        failed,
    )
    for group in ("compiler_counterfactual:UNIT_SCALE", "source_counterfactual:UNIT_SCALE"):
        row = diagnosis.get(group, {})
        _check(row.get("generic_evidence") is True, f"generic_evidence:{group}", failed)
        _check(row.get("layer") == "question_planner", f"layer:{group}", failed)

    source_lock = _load(ROOT / config["parents"]["qp4_actual_plan_source_cohort_lock"]["path"])
    _check(source_lock["source_evidence_coverage_complete"] is True, "source_extension_complete", failed)
    _check(source_lock["retrieval_or_binding_edit_authorized"] is False, "retrieval_stays_locked", failed)
    _check(source_lock["qp4_runtime_change_authorized"] is False, "parent_runtime_stays_locked", failed)

    scope = config["implementation_scope"]
    _check(
        scope["runtime_file_whitelist"] == ["src/hrm_finance_qa/question_planner.py"],
        "one_file_whitelist",
        failed,
    )
    _check(scope["stage"] == "typed_question_intent_only", "typed_intent_stage", failed)
    _check(scope["exact_symbols"]["QuestionPlan"] == ["requested_scale"], "field_scope", failed)
    _check(scope["exact_symbols"]["new_helper"] == ["_compile_requested_scale"], "helper_scope", failed)
    _check(
        set(scope["typed_scale_values"])
        == {"vnd", "thousand_vnd", "million_vnd", "billion_vnd", "conflict"},
        "scale_domain",
        failed,
    )
    _check(scope["missing_scale_value"] is None, "missing_scale_none", failed)
    _check("unit conversion execution" in scope["non_goals"], "conversion_out_of_scope", failed)
    _check("retrieval changes" in scope["non_goals"], "retrieval_out_of_scope", failed)
    _check("binding changes" in scope["non_goals"], "binding_out_of_scope", failed)
    _check("COLUMN_ROLE repair" in scope["non_goals"], "column_role_out_of_scope", failed)
    _check("DIRECTION repair" in scope["non_goals"], "direction_out_of_scope", failed)
    _check("REFERENCE_BASE repair" in scope["non_goals"], "reference_out_of_scope", failed)
    _check(len(config["frozen_files"]) == 7, "frozen_file_count", failed)
    _check(
        not (set(config["frozen_files"]) & set(scope["runtime_file_whitelist"])),
        "whitelist_frozen_disjoint",
        failed,
    )

    gates = config["post_implementation_gates"]
    _check(gates["compiler_unit_scale_detection"] == "13/13", "future_compiler_gate", failed)
    _check(gates["source_backed_old_winner_false_accept_target"] == 0, "future_source_target", failed)
    _check(gates["source_target_is_not_required_to_close_typed_intent_stage"] is True, "staged_close", failed)
    _check(gates["source_residual_must_be_reported_before_any_downstream_preauthorization"] is True, "residual_gate", failed)
    _check(gates["frozen_safe_replay_actions"] == {"ANSWER": 259, "ABSTAIN": 90, "REJECT": 14}, "safe_actions", failed)
    _check(gates["frozen_safe_replay_execution"] == "363/363", "safe_execution", failed)
    _check(gates["frozen_safe_replay_permutation"] == "363/363", "safe_permutation", failed)
    _check(gates["frozen_safe_replay_counterfactual_false_accepts"] == 0, "safe_counterfactual", failed)
    _check(all(value is False for value in config["invariants"].values()), "precommit_invariants", failed)

    return {
        "schema": "finhrm-qp4-unit-scale-question-planner-implementation-precommit-report-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "precommit_gate_passed": not failed,
        "failed_gates": failed,
        "generic_failure_evidence": expected,
        "implementation_scope": scope,
        "frozen_files": config["frozen_files"],
        "post_implementation_gates": gates,
        "runtime_changed": False,
        "runtime_hashes": config["runtime_hashes_before_implementation"],
        "checkpoint_sha256": checkpoint["sha256"],
        "qp4_runtime_change_authorized_in_same_step": False,
        "authorized_next_step": config["authorized_next_step_after_lock"] if not failed else "Resolve precommit validation failures without changing runtime.",
        "qp5_authorized": False,
        "phase_d": "locked",
    }


def _markdown(report: dict[str, Any]) -> str:
    evidence = report["generic_failure_evidence"]
    source = evidence["source_backed"]
    lines = [
        "# QP4 UNIT_SCALE question-planner implementation precommit v1",
        "",
        f"- Gate: `{'PASS' if report['precommit_gate_passed'] else 'FAIL'}`",
        f"- Compiler evidence: `{evidence['compiler']['detected']}/{evidence['compiler']['pairs']}` UNIT_SCALE pairs detected before implementation.",
        f"- Source evidence: `{source['old_winner_false_accepts']}/{source['runs']}` old-winner false accepts; failures span `{source['generic_failure_companies']}` companies and `{source['generic_failure_real_layouts']}` real layouts.",
        "- Future runtime whitelist: `src/hrm_finance_qa/question_planner.py` only.",
        "- Typed scope: `QuestionPlan.requested_scale`, `_compile_requested_scale`, and wiring in `compile_question_plan`.",
        "- Conversion, retrieval, binding, verifier, renderer and other QP4 dimensions remain frozen.",
        "- This precommit does not change or authorize runtime in the same step; QP5 and Phase D remain locked.",
    ]
    if report["failed_gates"]:
        lines += ["", "Failed gates: " + ", ".join(report["failed_gates"])]
    return "\n".join(lines) + "\n"


def write_outputs(
    report: dict[str, Any],
    report_path: Path = DEFAULT_REPORT,
    markdown_path: Path = DEFAULT_MARKDOWN,
    lock_path: Path = DEFAULT_LOCK,
) -> None:
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    artifacts = [
        _artifact(DEFAULT_CONFIG),
        _artifact(Path(__file__)),
        _artifact(TEST_PATH),
        _artifact(report_path),
        _artifact(markdown_path),
    ]
    lock = {
        "schema": "finhrm-qp4-unit-scale-question-planner-implementation-precommit-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["precommit_gate_passed"] else "failed",
        "precommit_gate_passed": report["precommit_gate_passed"],
        "runtime_changed": False,
        "runtime_hashes": report["runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "runtime_file_whitelist_for_next_step": report["implementation_scope"]["runtime_file_whitelist"],
        "qp4_runtime_change_authorized_in_same_step": False,
        "retrieval_or_binding_edit_authorized": False,
        "artifacts": artifacts,
        "authorized_next_step": report["authorized_next_step"],
        "qp5_authorized": False,
        "phase_d": "locked",
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
        "runtime_changed": report["runtime_changed"],
        "whitelist": report["implementation_scope"]["runtime_file_whitelist"],
    }, ensure_ascii=False))
    return 0 if report["precommit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
