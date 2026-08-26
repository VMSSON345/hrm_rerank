#!/usr/bin/env python3
"""Validate and lock the QP4 versioned frozen-action contract review."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from scripts.audit_qp4_unit_scale_frozen_action_contract_review import build_audit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "finhrm_qp4_unit_scale_frozen_action_contract_review_precommit_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp4_unit_scale_frozen_action_contract_review_precommit_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp4_unit_scale_frozen_action_contract_review_precommit_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp4_unit_scale_frozen_action_contract_review_precommit_lock_v1.json"
AUDIT_SCRIPT = ROOT / "scripts" / "audit_qp4_unit_scale_frozen_action_contract_review.py"
AUDIT_REPORT = ROOT / "reports" / "finhrm_qp4_unit_scale_frozen_action_contract_review_audit_v1.json"
TEST_PATH = ROOT / "tests" / "test_qp4_unit_scale_frozen_action_contract_review_precommit.py"
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

    audit = _load(AUDIT_REPORT)
    _check(audit == build_audit(), "audit_reproducible", failed)
    _check(audit["audit_passed"] is True and audit["errors"] == [], "audit_passed", failed)
    _check(audit["universe"] == {"examples": 363, "unchanged_actions": 361}, "audit_universe", failed)
    _check(audit["action_transitions"] == {"ANSWER->ABSTAIN": 2}, "audit_transition", failed)
    _check(all(audit["global_checks"].values()), "audit_global_checks", failed)
    _check(
        all(row["generic_contract_eligible"] for row in audit["delta_rows"]),
        "audit_generic_eligibility",
        failed,
    )
    _check(
        audit["generic_delta_definition"]["runtime_id_or_literal_rule_used"] is False,
        "no_id_literal_rule",
        failed,
    )

    historical = _load(ROOT / config["parents"]["historical_safe_replay"]["path"])
    current = _load(ROOT / config["parents"]["post_proof_safe_replay"]["path"])
    source = _load(ROOT / config["parents"]["post_proof_source_replay"]["path"])
    implementation = _load(ROOT / config["parents"]["proof_implementation_report"]["path"])
    contract = config["versioned_action_contract"]
    gate = config["gate_reclosure_contract"]

    _check(contract["historical_contract_is_immutable"] is True, "historical_immutable", failed)
    _check(
        historical["results"]["gold_actions"] == contract["historical_pre_proof_actions"],
        "historical_actions",
        failed,
    )
    _check(
        current["results"]["gold_actions"] == contract["reviewed_post_proof_actions"] == gate["safe_actions"],
        "reviewed_actions",
        failed,
    )
    _check(contract["allowed_transitions"]["ANSWER_TO_ABSTAIN"] == 2, "allowed_delta", failed)
    _check(
        sum(value for key, value in contract["allowed_transitions"].items() if key != "ANSWER_TO_ABSTAIN") == 0,
        "no_other_delta",
        failed,
    )
    _check(contract["unchanged_actions"] == 361, "unchanged_actions", failed)
    _check(contract["example_ids_allowed_only_in_audit_artifacts"] is True, "ids_audit_only", failed)
    _check(contract["gold_answer_text_as_runtime_unit_proof_forbidden"] is True, "gold_text_forbidden", failed)
    _check(contract["contract_conflicts_counted_as_generalization_confirmation"] is False, "no_confirmation", failed)

    source_summary = source["summary"]
    _check(source["coverage"] == gate["source_coverage"], "source_coverage", failed)
    _check(source_summary["prior_answer_to_abstain"] == gate["source_prior_answer_to_abstain"] == 4, "source_abstain", failed)
    _check(source_summary["prior_answer_binding_retained"] == 4, "source_binding", failed)
    _check(source_summary["prior_answer_scale_required_unknown"] == 4, "source_unknown", failed)
    _check(source_summary["old_winner_false_accepts"] == gate["source_old_winner_false_accepts"] == 0, "source_false_accept", failed)
    results = current["results"]
    _check(results["gold_total"] == gate["safe_examples"] == 363, "safe_examples", failed)
    _check(results["gold_execution_correct"] == 363 and gate["safe_execution"] == "363/363", "execution", failed)
    _check(results["permutation_exact"] == results["permutation_total"] == 363, "permutation", failed)
    _check(results["counterfactual_generated"] == gate["safe_counterfactuals"] == 141, "counterfactual_count", failed)
    _check(results["counterfactual_false_accepts"] == gate["safe_counterfactual_false_accepts"] == 0, "counterfactual_false_accept", failed)
    _check(
        implementation["failed_gates"] == ["safe_actions_exact"]
        and implementation["further_runtime_change_authorized"] is False,
        "implementation_conflict_isolated",
        failed,
    )
    _check(config["runtime_changed_by_this_precommit"] is False, "runtime_unchanged", failed)
    _check(config["downstream_runtime_change_authorized_in_same_step"] is False, "same_step_runtime_locked", failed)
    _check(config["historical_artifact_overwrite_authorized"] is False, "historical_overwrite_locked", failed)
    _check(all(value is False for value in config["invariants"].values()), "invariants", failed)

    return {
        "schema": "finhrm-qp4-unit-scale-frozen-action-contract-review-precommit-report-v1",
        "created_at": config["created_at"],
        "classification": config["classification"],
        "precommit_gate_passed": not failed,
        "failed_gates": failed,
        "versioned_action_contract": contract,
        "gate_reclosure_contract": gate,
        "audit_summary": {
            "examples": audit["universe"]["examples"],
            "unchanged_actions": audit["universe"]["unchanged_actions"],
            "action_transitions": audit["action_transitions"],
            "generic_eligible_deltas": sum(row["generic_contract_eligible"] for row in audit["delta_rows"]),
        },
        "runtime_changed": False,
        "runtime_hashes": config["frozen_runtime_hashes"],
        "checkpoint_sha256": checkpoint["sha256"],
        "historical_artifact_overwritten": False,
        "downstream_runtime_change_authorized_in_same_step": False,
        "qp4_complete": False,
        "qp5_authorized": False,
        "phase_d": "locked",
        "authorized_next_step": (
            config["authorized_next_step_after_lock"]
            if not failed
            else "Resolve action-contract review failures without changing runtime or historical artifacts."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    old = report["versioned_action_contract"]["historical_pre_proof_actions"]
    new = report["versioned_action_contract"]["reviewed_post_proof_actions"]
    return "\n".join(
        [
            "# QP4 UNIT_SCALE frozen action contract review precommit v1",
            "",
            f"- Gate: `{'PASS' if report['precommit_gate_passed'] else 'FAIL'}`.",
            f"- Historical contract retained: `{old['ANSWER']} ANSWER / {old['ABSTAIN']} ABSTAIN / {old['REJECT']} REJECT`.",
            f"- Reviewed post-proof contract: `{new['ANSWER']} ANSWER / {new['ABSTAIN']} ABSTAIN / {new['REJECT']} REJECT`.",
            "- Full audit: `363` examples; `361` unchanged; exactly `2 ANSWER → ABSTAIN`.",
            "- Both deltas satisfy one generic rule: explicit requested scale plus missing raw evidence unit makes required `unit_scale_match` UNKNOWN, therefore ABSTAIN.",
            "- IDs are audit traceability only. They are forbidden as runtime rules and are not generalization confirmation.",
            "- Execution `363/363`, permutation `363/363`, counterfactual `141` with `0` false accepts.",
            "- No runtime or historical artifact changed. QP4 remains incomplete; QP5 and Phase D remain locked.",
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
        AUDIT_SCRIPT,
        AUDIT_REPORT,
        Path(__file__).resolve(),
        TEST_PATH,
        report_path,
        markdown_path,
    ]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    lock = {
        "schema": "finhrm-qp4-unit-scale-frozen-action-contract-review-precommit-lock-v1",
        "created_at": report["created_at"],
        "classification": report["classification"],
        "status": "complete" if report["precommit_gate_passed"] else "failed",
        "precommit_gate_passed": report["precommit_gate_passed"],
        "runtime_changed": False,
        "runtime_hashes": report["runtime_hashes"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "historical_artifact_overwritten": False,
        "runtime_file_whitelist_for_next_step": [],
        "downstream_runtime_change_authorized_in_same_step": False,
        "artifacts": [_artifact(path) for path in artifacts],
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
    print(json.dumps({"precommit_gate_passed": report["precommit_gate_passed"], "failed_gates": report["failed_gates"]}, ensure_ascii=False))
    return 0 if report["precommit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
