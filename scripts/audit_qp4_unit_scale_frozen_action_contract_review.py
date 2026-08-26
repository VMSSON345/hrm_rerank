#!/usr/bin/env python3
"""Audit the frozen QP4 action delta introduced by UNIT_SCALE proof enforcement."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from hrm_finance_qa.question_planner import compile_question_plan


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEFORE = ROOT / "reports" / "finhrm_qp4_unit_scale_question_planner_safe_replay_v1.json"
DEFAULT_AFTER = ROOT / "reports" / "finhrm_qp4_unit_scale_proof_contract_safe_replay_v1.json"
DEFAULT_DATA = ROOT / "data" / "document_hrm_development_v1_train.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "finhrm_qp4_unit_scale_frozen_action_contract_review_audit_v1.json"
PROTECTED = re.compile(r"(?:^|[/_.-])(test|holdout|final|blind|validation|sealed)(?:[/_.-]|$)", re.I)


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _guard(path: Path) -> None:
    if PROTECTED.search(path.resolve().as_posix()):
        raise ValueError(f"Protected input is forbidden: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    _guard(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_records(path: Path, selected_ids: set[str]) -> dict[str, dict[str, Any]]:
    _guard(path)
    selected: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            example_id = str(record.get("example_id", ""))
            if example_id in selected_ids:
                selected[example_id] = record
    return selected


def _selected_candidate(record: dict[str, Any]) -> dict[str, Any]:
    operands = record.get("program", {}).get("operands", [])
    if len(operands) != 1:
        raise ValueError("Reviewed action delta must be unary SELECT")
    wanted = int(operands[0])
    candidates = record.get("candidates", [])
    for candidate in candidates:
        if int(candidate.get("index", -1)) == wanted:
            return candidate
    raise ValueError(f"Selected candidate {wanted} is absent")


def _gold_operand(record: dict[str, Any]) -> dict[str, Any]:
    operands = record.get("supervision", {}).get("gold_operands", [])
    if len(operands) != 1:
        raise ValueError("Reviewed action delta must have one gold operand")
    return operands[0]


def build_audit(
    before_path: Path = DEFAULT_BEFORE,
    after_path: Path = DEFAULT_AFTER,
    data_path: Path = DEFAULT_DATA,
) -> dict[str, Any]:
    before = _load_json(before_path)
    after = _load_json(after_path)
    before_details = {str(row["example_id"]): row for row in before["details"]}
    after_details = {str(row["example_id"]): row for row in after["details"]}
    errors: list[str] = []

    if set(before_details) != set(after_details):
        errors.append("example_universe_changed")
    common_ids = sorted(set(before_details) & set(after_details))
    changed_ids = [
        example_id
        for example_id in common_ids
        if before_details[example_id]["gold_action"] != after_details[example_id]["gold_action"]
    ]
    records = _load_records(data_path, set(changed_ids))
    if set(records) != set(changed_ids):
        errors.append("changed_development_records_missing")

    transitions = Counter()
    delta_rows: list[dict[str, Any]] = []
    generic_eligible = 0
    for example_id in changed_ids:
        prior = before_details[example_id]
        current = after_details[example_id]
        transitions[f"{prior['gold_action']}->{current['gold_action']}"] += 1
        record = records.get(example_id, {})
        try:
            plan = compile_question_plan(str(record["question"]))
            candidate = _selected_candidate(record)
            gold_operand = _gold_operand(record)
            checks = {
                "prior_action_answer": prior["gold_action"] == "ANSWER",
                "current_action_abstain": current["gold_action"] == "ABSTAIN",
                "prior_proof_valid": prior["gold_proof_contract_state"] == "VALID",
                "current_proof_unknown": current["gold_proof_contract_state"] == "UNKNOWN",
                "required_unit_scale_unknown_only": current["gold_reasons"]
                == ["required_proof_unknown:unit_scale_match"],
                "explicit_requested_scale": plan.requested_scale not in (None, "", "conflict"),
                "unary_select": record.get("program", {}).get("operation") == "SELECT"
                and len(record.get("program", {}).get("operands", [])) == 1,
                "selected_raw_unit_missing": str(candidate.get("unit", "")).strip() == "",
                "gold_unit_missing": str(gold_operand.get("unit", "")).strip() == ""
                and gold_operand.get("unit_status") == "missing",
                "operation_retained": prior["operation"] == current["operation"] == "SELECT",
                "execution_retained": prior["gold_execution_correct"] is True
                and current["gold_execution_correct"] is True,
                "permutation_retained": prior["permutation_exact"] is True
                and current["permutation_exact"] is True,
            }
            eligible = all(checks.values())
            generic_eligible += int(eligible)
            delta_rows.append(
                {
                    "example_id": example_id,
                    "company_key": current["company_key"],
                    "layout_family": current["layout_family"],
                    "transition": f"{prior['gold_action']}->{current['gold_action']}",
                    "requested_scale": plan.requested_scale,
                    "current_reasons": current["gold_reasons"],
                    "checks": checks,
                    "generic_contract_eligible": eligible,
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"delta_record:{example_id}:{exc}")

    before_results = before["results"]
    after_results = after["results"]
    global_checks = {
        "same_363_examples": len(common_ids) == len(before_details) == len(after_details) == 363,
        "exact_two_action_deltas": len(changed_ids) == 2,
        "only_answer_to_abstain": transitions == Counter({"ANSWER->ABSTAIN": 2}),
        "all_deltas_generic_eligible": generic_eligible == len(changed_ids) == 2,
        "historical_actions_retained": before_results["gold_actions"]
        == {"ABSTAIN": 90, "ANSWER": 259, "REJECT": 14},
        "reviewed_actions_exact": after_results["gold_actions"]
        == {"ABSTAIN": 92, "ANSWER": 257, "REJECT": 14},
        "no_answer_gained": not any(key.endswith("->ANSWER") for key in transitions),
        "no_reject_hidden": before_results["gold_actions"]["REJECT"]
        == after_results["gold_actions"]["REJECT"] == 14,
        "execution_363_of_363": before_results["gold_execution_correct"]
        == after_results["gold_execution_correct"] == 363,
        "permutation_363_of_363": before_results["permutation_exact"]
        == after_results["permutation_exact"] == 363,
        "counterfactual_141_zero_false_accept": before_results["counterfactual_generated"]
        == after_results["counterfactual_generated"] == 141
        and before_results["counterfactual_false_accepts"]
        == after_results["counterfactual_false_accepts"] == 0,
        "company_layout_universe_retained": before_results["company"]["groups"]
        == after_results["company"]["groups"] == 121
        and before_results["layout"]["groups"] == after_results["layout"]["groups"] == 6,
        "no_abw": before["source"]["abw_records"] == after["source"]["abw_records"] == 0,
        "no_protected_records": before["source"]["protected_records"]
        == after["source"]["protected_records"] == [],
    }
    if not all(global_checks.values()):
        errors.extend(name for name, passed in global_checks.items() if not passed)

    return {
        "schema": "finhrm-qp4-unit-scale-frozen-action-contract-review-audit-v1",
        "created_at": "2026-08-21",
        "classification": "development_train_only_contract_delta_audit",
        "audit_passed": not errors,
        "errors": errors,
        "inputs": {
            "historical_safe_replay": {"path": before_path.relative_to(ROOT).as_posix(), "sha256": _digest(before_path)},
            "post_proof_safe_replay": {"path": after_path.relative_to(ROOT).as_posix(), "sha256": _digest(after_path)},
            "development_train": {"path": data_path.relative_to(ROOT).as_posix(), "sha256": _digest(data_path)},
        },
        "universe": {"examples": len(common_ids), "unchanged_actions": len(common_ids) - len(changed_ids)},
        "historical_action_contract": before_results["gold_actions"],
        "reviewed_post_proof_action_contract": after_results["gold_actions"],
        "action_transitions": dict(sorted(transitions.items())),
        "generic_delta_definition": {
            "prior_action": "ANSWER",
            "post_proof_action": "ABSTAIN",
            "operation": "SELECT",
            "requested_scale": "explicit_non_conflicting",
            "selected_evidence_raw_unit": "missing",
            "required_proof": "unit_scale_match",
            "required_proof_state": "UNKNOWN",
            "policy": "UNKNOWN_TO_ABSTAIN",
            "runtime_id_or_literal_rule_used": False,
        },
        "delta_rows": delta_rows,
        "global_checks": global_checks,
        "interpretation": "The IDs are audit traceability only. Eligibility is semantic and report-agnostic; these cases are contract-conflict examples, not unseen-generalization confirmation.",
        "invariants": {
            "abw_used": False,
            "protected_evaluation_loaded": False,
            "runtime_changed": False,
            "threshold_changed": False,
            "checkpoint_changed": False,
            "fine_tuning_performed": False,
            "synthetic_layout_counted_as_confirmation": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = build_audit(args.before, args.after, args.data)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit_passed": audit["audit_passed"], "errors": audit["errors"]}, ensure_ascii=False))
    return 0 if audit["audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
