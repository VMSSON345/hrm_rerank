#!/usr/bin/env python3
"""Reclose the QP4 UNIT_SCALE implementation gate under the reviewed action contract."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from hrm_finance_qa.question_planner import compile_question_plan, compile_question_proof_contract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "finhrm_qp4_unit_scale_proof_contract_gate_reclosure_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_gate_reclosure_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_gate_reclosure_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_gate_reclosure_lock_v1.json"
TEST_PATH = ROOT / "tests" / "test_qp4_unit_scale_proof_contract_gate_reclosure.py"
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
    for relative, expected in config["frozen_runtime_hashes"].items():
        _check(_digest(ROOT / relative) == expected, f"runtime:{relative}", failed)

    parents = config["parents"]
    review_report = _load(ROOT / parents["review_precommit_report"]["path"])
    review_lock = _load(ROOT / parents["review_precommit_lock"]["path"])
    audit = _load(ROOT / parents["action_delta_audit"]["path"])
    historical_implementation = _load(
        ROOT / parents["proof_implementation_historical_report"]["path"]
    )
    historical_lock = _load(
        ROOT / parents["proof_implementation_historical_lock"]["path"]
    )
    implementation_config = _load(ROOT / parents["proof_implementation_config"]["path"])
    source = _load(ROOT / parents["post_proof_source_replay"]["path"])
    safe = _load(ROOT / parents["post_proof_safe_replay"]["path"])
    contract = config["reviewed_gate_contract"]
    gates = config["reclosure_gates"]

    _check(review_report["precommit_gate_passed"] is True, "review_report_passed", failed)
    _check(review_report["failed_gates"] == [], "review_report_no_failures", failed)
    _check(review_lock["status"] == "complete" and review_lock["precommit_gate_passed"] is True, "review_lock_complete", failed)
    _check(review_lock["runtime_file_whitelist_for_next_step"] == [], "empty_runtime_whitelist", failed)
    _check(
        review_lock["authorized_next_step"].startswith("QP4-UNIT-SCALE-PROOF-CONTRACT-GATE-RECLOSURE"),
        "reclosure_authorized",
        failed,
    )
    _check(audit["audit_passed"] is True and audit["errors"] == [], "action_audit_passed", failed)
    _check(audit["universe"] == {"examples": 363, "unchanged_actions": 361}, "action_audit_universe", failed)
    _check(audit["action_transitions"] == {"ANSWER->ABSTAIN": 2}, "action_audit_delta", failed)
    _check(all(row["generic_contract_eligible"] for row in audit["delta_rows"]), "generic_deltas", failed)

    _check(historical_implementation["implementation_gate_passed"] is False, "historical_gate_remains_open", failed)
    _check(historical_implementation["failed_gates"] == ["safe_actions_exact"], "historical_failure_isolated", failed)
    _check(historical_implementation["further_runtime_change_authorized"] is False, "historical_runtime_locked", failed)
    _check(historical_lock["status"] == "implementation_executed_gate_not_closed", "historical_lock_preserved", failed)
    _check(
        implementation_config["implementation_contract"]["new_proof_obligation"]
        == contract["proof_obligation"] == "unit_scale_match",
        "proof_obligation",
        failed,
    )

    explicit_plan = compile_question_plan("Doanh thu theo triệu đồng là bao nhiêu?")
    implicit_plan = compile_question_plan("Doanh thu là bao nhiêu?")
    explicit_proof = compile_question_proof_contract(explicit_plan, "SELECT")
    implicit_proof = compile_question_proof_contract(implicit_plan, "SELECT")
    _check(explicit_plan.requested_scale == "million_vnd", "explicit_scale_typed", failed)
    _check("unit_scale_match" in explicit_proof.required, "explicit_scale_required", failed)
    _check("unit_scale_match" in implicit_proof.not_applicable, "implicit_scale_not_applicable", failed)

    _check(contract["historical_actions_retained"] == audit["historical_action_contract"], "historical_actions", failed)
    _check(contract["post_proof_actions"] == audit["reviewed_post_proof_action_contract"], "reviewed_actions", failed)
    _check(contract["allowed_action_delta"] == {"ANSWER_TO_ABSTAIN": 2}, "allowed_delta", failed)
    _check(contract["unchanged_actions"] == 361, "unchanged_actions", failed)
    _check(contract["audit_ids_are_not_runtime_rules"] is True, "ids_not_runtime", failed)
    _check(contract["gold_answer_text_is_not_runtime_proof"] is True, "gold_text_not_proof", failed)

    source_summary = source["summary"]
    _check(source["coverage"] == gates["source_coverage"], "source_coverage", failed)
    _check(source_summary["prior_answer_to_abstain"] == gates["source_prior_answer_to_abstain"] == 4, "source_abstain", failed)
    _check(source_summary["prior_answer_binding_retained"] == gates["source_binding_retained"] == 4, "source_binding", failed)
    _check(source_summary["prior_answer_scale_required_unknown"] == gates["source_required_unit_scale_unknown"] == 4, "source_unknown", failed)
    _check(source_summary["old_winner_false_accepts"] == gates["source_old_winner_false_accepts"] == 0, "source_false_accept", failed)
    _check(source_summary["errors"] == 0, "source_errors", failed)
    _check(source["abw_records"] == 0 and source["protected_records"] == [], "source_clean", failed)

    results = safe["results"]
    _check(results["gold_total"] == gates["safe_examples"] == 363, "safe_examples", failed)
    _check(results["gold_actions"] == gates["safe_actions"] == contract["post_proof_actions"], "safe_actions", failed)
    _check(results["gold_execution_correct"] == gates["safe_execution_correct"] == 363, "safe_execution", failed)
    _check(results["permutation_exact"] == results["permutation_total"] == gates["safe_permutation_exact"] == 363, "safe_permutation", failed)
    _check(results["counterfactual_generated"] == gates["safe_counterfactuals"] == 141, "safe_counterfactual_count", failed)
    _check(results["counterfactual_false_accepts"] == gates["safe_counterfactual_false_accepts"] == 0, "safe_counterfactual_false_accept", failed)
    _check(safe["safe_replay_gate_passed"] is True and safe["errors"] == [], "safe_replay_passed", failed)
    _check(safe["source"]["abw_records"] == 0 and safe["source"]["protected_records"] == [], "safe_source_clean", failed)

    _check(config["runtime_changes_authorized"] is False, "runtime_changes_locked", failed)
    _check(config["historical_artifact_overwrite_authorized"] is False, "historical_overwrite_locked", failed)
    _check(all(value is False for value in config["invariants"].values()), "invariants", failed)

    reclosed = not failed
    return {
        "schema": "finhrm-qp4-unit-scale-proof-contract-gate-reclosure-report-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "status": "complete" if reclosed else "gate_not_reclosed",
        "unit_scale_gate_reclosed": reclosed,
        "failed_gates": failed,
        "proof_contract": {
            "obligation": contract["proof_obligation"],
            "explicit_scale": contract["explicit_scale_obligation"],
            "missing_raw_scale": {
                "state": contract["missing_raw_scale_state"],
                "action": contract["missing_raw_scale_action"],
            },
            "mixed_or_conflicting_scale": {
                "state": contract["mixed_or_conflicting_scale_state"],
                "action": contract["mixed_or_conflicting_scale_action"],
            },
        },
        "source_replay": {
            "coverage": source["coverage"],
            "summary": source_summary,
            "gate_passed": all(
                name not in failed
                for name in ("source_coverage", "source_abstain", "source_binding", "source_unknown", "source_false_accept", "source_errors")
            ),
        },
        "safe_replay": {
            "actions": results["gold_actions"],
            "execution": f"{results['gold_execution_correct']}/{results['gold_total']}",
            "permutation": f"{results['permutation_exact']}/{results['permutation_total']}",
            "counterfactuals": results["counterfactual_generated"],
            "counterfactual_false_accepts": results["counterfactual_false_accepts"],
        },
        "action_contract_review": {
            "historical_actions": contract["historical_actions_retained"],
            "post_proof_actions": contract["post_proof_actions"],
            "unchanged_actions": contract["unchanged_actions"],
            "allowed_delta": contract["allowed_action_delta"],
            "generic_eligible_deltas": len(audit["delta_rows"]),
        },
        "compatible_affected_regression": gates["compatible_affected_regression"],
        "historical_implementation_report_preserved": True,
        "historical_artifact_overwritten": False,
        "runtime_changed": False,
        "runtime_hashes": config["frozen_runtime_hashes"],
        "checkpoint_sha256": checkpoint["sha256"],
        "qp4_unit_scale_complete": reclosed,
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": (
            config["authorized_next_step_after_reclosure"]
            if reclosed
            else "Resolve gate-reclosure evidence failures without runtime or historical-artifact changes."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    safe = report["safe_replay"]
    source = report["source_replay"]
    return "\n".join(
        [
            "# QP4 UNIT_SCALE proof-contract gate reclosure v1",
            "",
            f"- Gate: `{'PASS' if report['unit_scale_gate_reclosed'] else 'FAIL'}`.",
            f"- Source gate: `{'PASS' if source['gate_passed'] else 'FAIL'}` on `{source['coverage']['companies']}` companies / `{source['coverage']['sources']}` PDFs / `{source['coverage']['real_layouts']}` real layouts; false accepts `{source['summary']['old_winner_false_accepts']}`.",
            f"- Reviewed safe actions: `{safe['actions']}`; execution `{safe['execution']}`; permutation `{safe['permutation']}`.",
            f"- Counterfactuals: `{safe['counterfactuals']}`; false accepts `{safe['counterfactual_false_accepts']}`.",
            "- Historical `259/90/14` contract and the original failed implementation report remain immutable; the reviewed post-proof contract is versioned separately.",
            "- No runtime, checkpoint, threshold, model or historical artifact changed.",
            "- UNIT_SCALE is closed, but overall QP4 is incomplete; QP5 and Phase D remain locked.",
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
        "schema": "finhrm-qp4-unit-scale-proof-contract-gate-reclosure-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": report["status"],
        "unit_scale_gate_reclosed": report["unit_scale_gate_reclosed"],
        "failed_gates": report["failed_gates"],
        "runtime_changed": False,
        "runtime_hashes": report["runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "historical_artifact_overwritten": False,
        "runtime_file_whitelist_for_next_step": [],
        "artifacts": [_artifact(path) for path in artifacts],
        "qp4_unit_scale_complete": report["qp4_unit_scale_complete"],
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": report["authorized_next_step"],
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    report = build_report(args.config)
    write_outputs(report, args.report, args.markdown, args.lock)
    print(json.dumps({"unit_scale_gate_reclosed": report["unit_scale_gate_reclosed"], "failed_gates": report["failed_gates"]}, ensure_ascii=False))
    return 0 if report["unit_scale_gate_reclosed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
