"""Audit-only strict column-role filter over frozen S4 accepted SELECT rows."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.evidence_schema import build_evidence_graph  # noqa: E402
from hrm_finance_qa.program_data import load_program_examples  # noqa: E402
from hrm_finance_qa.question_planner import compile_question_plan  # noqa: E402


PROTECTED_TOKENS = {"blind", "validation", "holdout", "test", "final_eval"}


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _guard(path: Path) -> None:
    if set(part.lower() for part in path.resolve().parts) & PROTECTED_TOKENS:
        raise ValueError(f"QP2 audit refuses protected input: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=ROOT / "reports" / "select_add_counterfactual_s4_development_v1.json",
    )
    parser.add_argument(
        "--development-data",
        type=Path,
        default=ROOT / "data" / "document_hrm_v12b2_development_plus_subtract_v3.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "reports" / "column_role_select_priority_qp2_development_v1.json",
    )
    args = parser.parse_args()
    for path in (args.baseline_report, args.development_data):
        _guard(path)

    baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    if baseline.get("classification") != "development_only_company_report_disjoint":
        raise ValueError("Expected the frozen development-only S4 report")
    examples = {
        example.example_id: example
        for example in load_program_examples(args.development_data)
    }
    rows: list[dict[str, object]] = []
    false_removed = 0
    true_removed = 0
    gold_role_retained = 0
    for detail in baseline.get("select_add_accepted_details", ()):
        if not isinstance(detail, dict) or detail.get("gold_operation") != "SELECT":
            continue
        plan = compile_question_plan(str(detail.get("question", "")))
        if len(plan.column_roles) != 1:
            continue
        example_id = str(detail["example_id"])
        example = examples[example_id]
        graph = build_evidence_graph(example.candidates)
        operation, raw_operands = str(detail["baseline_program"]).split(":", 1)
        if operation != "SELECT":
            continue
        operand = int(raw_operands)
        selected = graph.cells[operand]
        gold = graph.cells[example.program.operands[0]]
        required = plan.column_roles[0].lower()
        baseline_role_valid = (
            selected.column_role == required
            and not selected.column_role_conflict
        )
        gold_role_valid = (
            gold.column_role == required and not gold.column_role_conflict
        )
        baseline_correct = detail.get("baseline_state") == "correct"
        false_removed += int(not baseline_role_valid and not baseline_correct)
        true_removed += int(not baseline_role_valid and baseline_correct)
        gold_role_retained += int(gold_role_valid)
        rows.append(
            {
                "example_id": example_id,
                "question": detail["question"],
                "required_column_role": plan.column_roles[0],
                "baseline_state": detail["baseline_state"],
                "baseline_value": str(selected.value),
                "baseline_column": selected.column,
                "baseline_column_role": selected.column_role,
                "strict_role_filter_retains_baseline": baseline_role_valid,
                "gold_value": str(gold.value),
                "gold_column": gold.column,
                "gold_column_role": gold.column_role,
                "gold_role_retained": gold_role_valid,
            }
        )

    baseline_overall = baseline["policy_results"]["baseline"]["overall"]
    accepted = int(baseline_overall["accepted"]) - false_removed - true_removed
    correct = int(baseline_overall["correct"]) - true_removed
    false_accept = int(baseline_overall["false_accept"]) - false_removed
    examples_count = int(baseline_overall["examples"])
    nine_f_minus_t = 9 * false_removed - true_removed
    report = {
        "schema": "column-role-select-priority-qp2-development-v1",
        "classification": "development_only_company_report_disjoint",
        "evaluation_mode": "audit_only_filter_of_frozen_accepted_select_winners",
        "runtime_policy_authorized": False,
        "threshold_changed": False,
        "fine_tuning_performed": False,
        "validation_loaded": False,
        "blind_test_loaded": False,
        "explicit_role_accepted_select_examples": len(rows),
        "false_accept_removed_F": false_removed,
        "true_accept_removed_T": true_removed,
        "gold_role_retained": gold_role_retained,
        "gold_role_total": len(rows),
        "nine_F_minus_T": nine_f_minus_t,
        "conservative_overall_after_filter": {
            "examples": examples_count,
            "accepted": accepted,
            "correct": correct,
            "false_accept": false_accept,
            "coverage": accepted / examples_count,
            "accepted_precision": correct / accepted if accepted else None,
        },
        "release_gates": {
            "accepted_precision_90": correct / accepted >= 0.90 if accepted else False,
            "coverage_88_5": accepted / examples_count >= 0.885,
            "nine_F_minus_T_143": nine_f_minus_t >= 143,
            "new_false_accepts_zero": True,
            "sufficient_explicit_role_sample": False,
            "all_pass": False,
        },
        "decision": (
            "Keep proof-conditioned SELECT acceptance disabled. The frozen S4 "
            "accepted set contains only two explicit column-role SELECT rows; "
            "the strict filter removes one false accept and no true accept, but "
            "overall precision remains below 90% and 9F-T is below 143."
        ),
        "details": rows,
        "artifact_hashes": {
            "baseline_report": _digest(args.baseline_report),
            "development_data": _digest(args.development_data),
            "question_planner": _digest(ROOT / "src" / "hrm_finance_qa" / "question_planner.py"),
            "evidence_schema": _digest(ROOT / "src" / "hrm_finance_qa" / "evidence_schema.py"),
            "script": _digest(Path(__file__)),
            "checkpoint": _digest(ROOT / "models" / "document_hrm_V8_candidate.pt"),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
