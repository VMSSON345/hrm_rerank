#!/usr/bin/env python3
"""Validate and lock the QP4 UNIT_SCALE source-residual proof scope."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "finhrm_qp4_unit_scale_source_residual_isolation_precommit_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp4_unit_scale_source_residual_isolation_precommit_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp4_unit_scale_source_residual_isolation_precommit_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp4_unit_scale_source_residual_isolation_precommit_lock_v1.json"
TEST_PATH = ROOT / "tests" / "test_qp4_unit_scale_source_residual_isolation_precommit.py"
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
        _check(path.is_file() and _digest(path) == parent["sha256"], f"parent:{name}", failed)
    checkpoint = config["checkpoint"]
    _check(_digest(ROOT / checkpoint["path"]) == checkpoint["sha256"], "checkpoint", failed)
    for relative, expected in config["authorized_runtime_hashes_before_next_implementation"].items():
        _check(_digest(ROOT / relative) == expected, f"authorized_runtime:{relative}", failed)
    for relative, expected in config["frozen_runtime_hashes"].items():
        _check(_digest(ROOT / relative) == expected, f"frozen_runtime:{relative}", failed)

    trace = _load(ROOT / config["parents"]["source_residual_trace"]["path"])
    coverage = trace["coverage"]
    summary = trace["summary"]
    expected = config["generic_residual_evidence"]
    for key in ("runs", "companies", "sources", "real_layouts"):
        _check(coverage[key] == expected[key], f"coverage:{key}", failed)
    for key in (
        "planner_typed",
        "answers",
        "old_winner_false_accepts",
        "answered_with_raw_unit_scale_preserved",
        "answered_with_unit_match_valid",
        "answered_with_unit_scale_obligation_declared",
        "answered_with_unit_scale_match_state",
        "answered_with_all_required_proofs_valid",
    ):
        _check(summary[key] == expected[key], f"summary:{key}", failed)
    _check(trace["abw_records"] == 0, "no_abw", failed)
    _check(trace["protected_records"] == [], "no_protected_records", failed)

    attribution = config["layer_attribution"]
    observed_attribution = trace["attribution"]
    attribution_keys = (
        ("planner_retains_requested_scale", "planner_requested_scale_retained"),
        ("evidence_retains_raw_unit_scale", "evidence_raw_unit_scale_retained"),
        ("retrieval_or_binding_winner_changed", "retrieval_or_binding_winner_changed"),
        ("first_missing_semantic_boundary", "first_missing_semantic_boundary"),
        ("first_missing_runtime_assessment", "first_missing_runtime_assessment"),
        (
            "engine_action_policy_already_fails_closed_on_required_unknown",
            "engine_action_policy_already_fails_closed_on_required_unknown",
        ),
    )
    for observed_key, config_key in attribution_keys:
        _check(
            observed_attribution[observed_key] == attribution[config_key],
            f"attribution:{config_key}",
            failed,
        )

    scope = config["next_implementation_scope"]
    _check(
        scope["runtime_file_whitelist"]
        == [
            "src/hrm_finance_qa/question_planner.py",
            "src/hrm_finance_qa/symbolic_verifier.py",
        ],
        "exact_two_file_whitelist",
        failed,
    )
    _check(scope["stage"] == "typed_unit_scale_proof_only", "proof_only_stage", failed)
    _check(scope["new_proof_obligation"] == "unit_scale_match", "proof_name", failed)
    _check(
        scope["exact_symbols"]["src/hrm_finance_qa/question_planner.py"]
        == ["_PROOF_CONSTRAINTS", "compile_question_proof_contract"],
        "planner_symbols",
        failed,
    )
    _check(
        scope["exact_symbols"]["src/hrm_finance_qa/symbolic_verifier.py"]
        == ["currency_unit_scale import", "_unit_scale_match_state", "verify_program"],
        "verifier_symbols",
        failed,
    )
    for non_goal in (
        "unit conversion execution",
        "retrieval changes",
        "candidate binding changes",
        "program synthesis changes",
        "evidence schema changes",
        "document reasoning engine changes",
        "answer renderer changes",
    ):
        _check(non_goal in scope["non_goals"], f"non_goal:{non_goal}", failed)

    planner_text = (ROOT / "src/hrm_finance_qa/question_planner.py").read_text(encoding="utf-8")
    verifier_text = (ROOT / "src/hrm_finance_qa/symbolic_verifier.py").read_text(encoding="utf-8")
    synthesis_text = (ROOT / "src/hrm_finance_qa/program_synthesis.py").read_text(encoding="utf-8")
    engine_text = (ROOT / "src/hrm_finance_qa/document_reasoning_engine.py").read_text(encoding="utf-8")
    _check("requested_scale" in planner_text, "planner_has_scale", failed)
    _check("unit_scale_match" not in planner_text, "proof_obligation_currently_absent", failed)
    _check("requested_scale" not in verifier_text, "verifier_assessment_currently_absent", failed)
    _check("requested_scale" not in synthesis_text, "synthesis_does_not_consume_scale", failed)
    _check(
        "any(state != VerificationState.VALID.value for state in required_states)" in engine_text,
        "engine_required_unknown_abstains",
        failed,
    )
    _check(
        "verification.state is VerificationState.INVALID" in engine_text,
        "engine_invalid_rejects",
        failed,
    )

    gates = config["post_implementation_gates"]
    _check(gates["source_old_winner_false_accept_target"] == 0, "source_target", failed)
    _check(gates["source_operation_value_evidence_binding_retained"] == "4/4", "binding_retention", failed)
    _check(gates["candidate_permutation_exact_rate"] == 1.0, "permutation_gate", failed)
    _check(gates["frozen_safe_replay_execution"] == "363/363", "execution_gate", failed)
    _check(gates["frozen_safe_replay_permutation"] == "363/363", "safe_permutation_gate", failed)
    _check(gates["frozen_safe_replay_counterfactuals"] == 141, "counterfactual_count", failed)
    _check(gates["frozen_safe_replay_counterfactual_false_accepts"] == 0, "counterfactual_target", failed)
    _check(config["runtime_changed_by_this_precommit"] is False, "runtime_unchanged", failed)
    _check(config["downstream_runtime_change_authorized_in_same_step"] is False, "same_step_locked", failed)
    _check(all(value is False for value in config["invariants"].values()), "all_invariants_closed", failed)

    current_hashes = {
        relative: _digest(ROOT / relative)
        for relative in (
            *config["authorized_runtime_hashes_before_next_implementation"],
            *config["frozen_runtime_hashes"],
        )
    }
    return {
        "schema": "finhrm-qp4-unit-scale-source-residual-isolation-precommit-report-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "precommit_gate_passed": not failed,
        "failed_gates": failed,
        "generic_residual_evidence": expected,
        "layer_attribution": attribution,
        "next_implementation_scope": scope,
        "post_implementation_gates": gates,
        "runtime_changed": False,
        "runtime_hashes": current_hashes,
        "checkpoint_sha256": checkpoint["sha256"],
        "downstream_runtime_change_authorized_in_same_step": False,
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": (
            config["authorized_next_step_after_lock"]
            if not failed
            else "Resolve residual-isolation precommit failures without changing runtime."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    evidence = report["generic_residual_evidence"]
    return "\n".join(
        [
            "# QP4 UNIT_SCALE source residual isolation precommit v1",
            "",
            f"- Gate: `{'PASS' if report['precommit_gate_passed'] else 'FAIL'}`.",
            f"- Coverage: `{evidence['runs']}` runs, `{evidence['companies']}` companies, `{evidence['sources']}` PDFs, `{evidence['real_layouts']}` real layouts.",
            f"- Old-winner false accepts: `{evidence['old_winner_false_accepts']}`.",
            f"- Raw scale preserved / unit_match VALID / scale obligation declared: `{evidence['answered_with_raw_unit_scale_preserved']}/{evidence['answered_with_unit_match_valid']}/{evidence['answered_with_unit_scale_obligation_declared']}` across the four answers.",
            "- First missing semantic boundary: `question_proof_contract`; first missing assessment: `symbolic_verifier`.",
            "- Future whitelist: `question_planner.py` and `symbolic_verifier.py` exact proof symbols only.",
            "- This step changes no runtime and authorizes no same-step implementation. Conversion, retrieval, binding, synthesis, engine and renderer remain frozen.",
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
    artifacts = [
        DEFAULT_CONFIG,
        ROOT / "scripts" / "diagnose_qp4_unit_scale_source_residual.py",
        ROOT / "reports" / "finhrm_qp4_unit_scale_source_residual_trace_v1.json",
        Path(__file__).resolve(),
        TEST_PATH,
        report_path,
        markdown_path,
    ]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    lock = {
        "schema": "finhrm-qp4-unit-scale-source-residual-isolation-precommit-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["precommit_gate_passed"] else "failed",
        "precommit_gate_passed": report["precommit_gate_passed"],
        "runtime_changed": False,
        "runtime_hashes": report["runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "runtime_file_whitelist_for_next_step": report["next_implementation_scope"][
            "runtime_file_whitelist"
        ],
        "downstream_runtime_change_authorized_in_same_step": False,
        "conversion_retrieval_or_binding_edit_authorized": False,
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
                "first_missing_boundary": report["layer_attribution"][
                    "first_missing_semantic_boundary"
                ],
                "whitelist": report["next_implementation_scope"][
                    "runtime_file_whitelist"
                ],
                "runtime_changed": report["runtime_changed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["precommit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
