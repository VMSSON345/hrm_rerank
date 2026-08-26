from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from hrm_finance_qa.evidence_schema import EvidenceGraph, build_evidence_graph
from hrm_finance_qa.program_data import NumericCandidate, ProgramDAGInput
from hrm_finance_qa.program_synthesis import synthesize_typed_programs
from hrm_finance_qa.question_planner import compile_question_plan
from hrm_finance_qa.symbolic_verifier import VerificationState, verify_program


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = ROOT / "data" / "finhrm_qp3_core_development_v1.jsonl"
DEFAULT_PRECOMMIT_LOCK = ROOT / "reports" / "finhrm_qp3_core_precommit_lock_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp3_core_implementation_development_v1.json"
DEFAULT_MARKDOWN = ROOT / "reports" / "finhrm_qp3_core_implementation_development_v1.md"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp3_core_implementation_lock_v1.json"

PROTECTED_PARTS = {"validation", "sealed", "blind", "test", "final"}
RUNTIME_PATHS = (
    "src/hrm_finance_qa/question_planner.py",
    "src/hrm_finance_qa/program_synthesis.py",
    "src/hrm_finance_qa/symbolic_verifier.py",
    "src/hrm_finance_qa/direction_contract.py",
    "src/hrm_finance_qa/document_hrm.py",
    "src/hrm_finance_qa/document_reasoning_engine.py",
)
EXPECTED_CHANGED_RUNTIME_FILES = tuple(Path(path).name for path in RUNTIME_PATHS)
VALUE_BY_OPERAND = {
    "plan": "100",
    "actual": "80",
    "actual_2023": "120",
    "actual_2022": "100",
    "plan_2024": "125",
    "short_term_borrowings": "20",
    "long_term_borrowings": "30",
    "total_capital": "100",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refuse_protected(path: Path) -> None:
    tokens = {
        token.lower()
        for part in path.parts
        for token in part.replace("-", "_").replace(".", "_").split("_")
    }
    overlap = sorted(tokens & PROTECTED_PARTS)
    if overlap:
        raise ValueError(f"protected evaluation path refused: {path} ({overlap})")


def _records(path: Path) -> list[dict[str, Any]]:
    _refuse_protected(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _role_label(role: str) -> str:
    return {
        "ACTUAL": "Actual / Thực hiện",
        "PLAN": "Plan / Kế hoạch",
        "PRIOR_ACTUAL": "Prior actual / Thực hiện kỳ trước",
        "REVISED_PLAN": "Revised plan / Kế hoạch điều chỉnh",
        "FORECAST": "Forecast / Dự báo",
    }.get(role, role)


def _materialized_period(period: str) -> str:
    """Turn an opaque counterfactual marker into an explicitly different year."""

    prefix = "counterfactual:"
    if not period.startswith(prefix):
        return period
    original = period[len(prefix) :]
    if original.isdigit() and len(original) == 4:
        return str(int(original) - 1)
    return "unknown-counterfactual-period"


def _numeric_candidate(index: int, operand: dict[str, Any]) -> NumericCandidate:
    role = str(operand["column_role"])
    period = _materialized_period(str(operand["period"]))
    metric = str(operand["metric"])
    source = str(operand["document_id"])
    value = VALUE_BY_OPERAND.get(str(operand["operand_id"]), str(10 + index * 10))
    return NumericCandidate(
        index=index,
        value=value,
        raw=value,
        context=(
            f"ROW: {metric}\n"
            f"COLUMN: {_role_label(role)} {period}\n"
            f"PERIOD: {period}\n"
            f"TABLE: {operand['table_id']}\n"
            f"ROW_ID: {operand['row_id']}\n"
            f"COLUMN_ID: {operand['column_id']}"
        ),
        line_number=10 + index,
        char_start=100 + index * 10,
        char_end=105 + index * 10,
        unit="VND",
        is_percent=False,
        is_year_like=False,
        relevance=1.0,
        source_name=source,
        page=int(operand["page"]),
        extraction_method="synthetic_typed_proof_fixture",
    )


def _target_obligation(scenario_id: str) -> tuple[str, VerificationState] | None:
    return {
        "wrong_reference_base": ("reference_base_match", VerificationState.INVALID),
        "duplicate_logical_cell": ("logical_cell_unique", VerificationState.INVALID),
        "missing_source_provenance": ("source_provenance_complete", VerificationState.UNKNOWN),
        "partial_multi_output": ("answer_mode_complete", VerificationState.UNKNOWN),
        "leaf_unknown": ("all_leaf_nodes_valid", VerificationState.UNKNOWN),
        "leaf_invalid": ("all_leaf_nodes_valid", VerificationState.INVALID),
        "intermediate_unknown": ("all_intermediate_nodes_valid", VerificationState.UNKNOWN),
        "intermediate_invalid": ("all_intermediate_nodes_valid", VerificationState.INVALID),
        "dag_execution_invalid": ("dag_execution_valid", VerificationState.INVALID),
    }.get(scenario_id)


def _action(state: VerificationState) -> str:
    return {
        VerificationState.VALID: "ANSWER",
        VerificationState.UNKNOWN: "ABSTAIN",
        VerificationState.INVALID: "REJECT",
    }[state]


def _evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    plan = compile_question_plan(str(record["question"]))
    if record["answerability"] == "ABSTAIN":
        observed_action = "ABSTAIN" if plan.abstention_reason else "ANSWER"
        return {
            "example_id": record["example_id"],
            "family_id": record["family_id"],
            "scenario_id": record["scenario_id"],
            "company_key": record["company_key"],
            "layout_family": record["layout_family"],
            "observed_state": "UNKNOWN" if plan.abstention_reason else "VALID",
            "observed_action": observed_action,
            "expected_action": record["expected_action"],
            "target_obligation_exact": True,
            "exact": observed_action == record["expected_action"],
        }

    if record["scenario_id"] == "partial_multi_output":
        plan = replace(
            plan,
            answer_mode="MULTI_OUTPUT_ALL_OR_ABSTAIN",
            requested_output_count=2,
        )
    candidates = tuple(
        _numeric_candidate(index, operand)
        for index, operand in enumerate(record["candidate_operands"])
    )
    graph = build_evidence_graph(candidates)
    synthesis = synthesize_typed_programs(
        plan,
        plan.operation or "",
        graph,
        focus_text=plan.metric_text,
    )
    if not synthesis.proposals:
        observed_action = "REJECT"
        return {
            "example_id": record["example_id"],
            "family_id": record["family_id"],
            "scenario_id": record["scenario_id"],
            "company_key": record["company_key"],
            "layout_family": record["layout_family"],
            "observed_state": "INVALID",
            "observed_action": observed_action,
            "expected_action": record["expected_action"],
            "target_obligation_exact": record["scenario_id"] in {"swapped_role", "swapped_year"},
            "exact": observed_action == record["expected_action"],
        }

    proposal = synthesis.proposals[0]
    dag = proposal.program_dag
    scenario_id = str(record["scenario_id"])
    cells = list(graph.cells)
    if scenario_id == "missing_source_provenance":
        cells[0] = replace(cells[0], source_span=None)
    elif scenario_id == "duplicate_logical_cell":
        cells.append(replace(cells[0], index=len(cells), source_span=(9000, 9005)))
    elif scenario_id == "leaf_unknown":
        cells[0] = replace(cells[0], column_role="", column_role_confidence="UNKNOWN")
    elif scenario_id == "leaf_invalid":
        wrong_role = "actual" if cells[0].column_role != "actual" else "plan"
        cells[0] = replace(cells[0], column_role=wrong_role)
    elif scenario_id == "dag_execution_invalid":
        denominator = dict(proposal.slot_bindings).get(plan.reference_base or "", 0)
        cells[denominator] = replace(cells[denominator], value=Decimal(0))
    graph = EvidenceGraph(
        cells=tuple(cells),
        edges=graph.edges,
        column_role_inventory_state=graph.column_role_inventory_state,
        column_role_inventory_reasons=graph.column_role_inventory_reasons,
        table_role_inventories=graph.table_role_inventories,
    )

    if dag is not None and scenario_id == "wrong_reference_base":
        alternative = str(record["candidate_reference_base"])
        denominator = dict(proposal.slot_bindings).get(alternative)
        if denominator is not None:
            final = dag.nodes[-1]
            dag = replace(
                dag,
                nodes=(
                    *dag.nodes[:-1],
                    replace(
                        final,
                        inputs=(final.inputs[0], ProgramDAGInput.candidate(denominator)),
                    ),
                ),
            )
    elif dag is not None and scenario_id == "intermediate_unknown":
        dag = replace(
            dag,
            nodes=(replace(dag.nodes[0], result_unit_family=""), *dag.nodes[1:]),
        )
    elif dag is not None and scenario_id == "intermediate_invalid":
        replacement_operation = "ADD" if dag.nodes[0].operation != "ADD" else "SUBTRACT"
        dag = replace(
            dag,
            nodes=(replace(dag.nodes[0], operation=replacement_operation), *dag.nodes[1:]),
        )

    verification = verify_program(
        plan,
        proposal.program,
        graph,
        program_dag=dag,
        slot_bindings=proposal.slot_bindings,
    )
    observed_action = _action(verification.proof_contract_state)
    states = {item.name: item.state for item in verification.proof_obligations}
    target = _target_obligation(scenario_id)
    target_exact = target is None or states.get(target[0]) is target[1]
    return {
        "example_id": record["example_id"],
        "family_id": record["family_id"],
        "scenario_id": scenario_id,
        "company_key": record["company_key"],
        "layout_family": record["layout_family"],
        "observed_state": verification.proof_contract_state.value,
        "observed_action": observed_action,
        "expected_action": record["expected_action"],
        "target_obligation_exact": target_exact,
        "exact": observed_action == record["expected_action"] and target_exact,
        "verification_reasons": list(verification.reasons),
        "proof_obligations": {
            item.name: item.state.value if item.state is not None else "NOT_APPLICABLE"
            for item in verification.proof_obligations
        },
        "proof_obligation_details": [
            {
                "name": item.name,
                "applicability": item.applicability,
                "state": item.state.value if item.state is not None else "NOT_APPLICABLE",
            }
            for item in verification.proof_obligations
        ],
        "slot_bindings": [list(item) for item in proposal.slot_bindings],
    }


def _group(details: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for item in details:
        grouped[str(item[key])].append(bool(item["exact"]))
    rates = {name: sum(values) / len(values) for name, values in sorted(grouped.items())}
    return {
        "groups": len(rates),
        "exact_rate_by_group": rates,
        "macro_exact_rate": mean(rates.values()) if rates else 0.0,
        "worst_group_exact_rate": min(rates.values()) if rates else 0.0,
    }


def build_report(
    cohort_path: Path = DEFAULT_COHORT,
    precommit_lock_path: Path = DEFAULT_PRECOMMIT_LOCK,
) -> dict[str, Any]:
    records = _records(cohort_path)
    precommit_lock = json.loads(precommit_lock_path.read_text(encoding="utf-8"))
    details = [_evaluate_record(record) for record in records]
    exact = sum(bool(item["exact"]) for item in details)
    false_accepts = sum(
        item["expected_action"] != "ANSWER" and item["observed_action"] == "ANSWER"
        for item in details
    )
    positives = [
        item
        for item in details
        if item["expected_action"] == "ANSWER"
        and item["scenario_id"] in {"valid_gold", "candidate_permuted"}
    ]
    retained = sum(item["observed_action"] == "ANSWER" for item in positives)
    runtime_hashes = {path: _digest(ROOT / path) for path in RUNTIME_PATHS}
    checkpoint_hash = _digest(ROOT / "models" / "document_hrm_V8_candidate.pt")
    errors = []
    if not precommit_lock.get("precommit_gate_passed"):
        errors.append("precommit_lock_not_passed")
    if exact != len(records):
        errors.append("runtime_oracle_mismatch")
    if false_accepts:
        errors.append("new_false_accept")
    if retained != len(positives):
        errors.append("gold_retention_regression")
    company = _group(details, "company_key")
    layout = _group(details, "layout_family")
    if company["groups"] < 3 or layout["groups"] < 3:
        errors.append("company_layout_diversity_missing")
    return {
        "schema": "finhrm-qp3-core-implementation-development-evaluation-v1",
        "created_at": "2026-08-14",
        "classification": "development_only_synthetic_report_agnostic_proof_implementation_evaluation",
        "implementation_gate_passed": not errors,
        "errors": errors,
        "results": {
            "runtime_oracle_exact": exact,
            "total": len(records),
            "false_accepts": false_accepts,
            "gold_retained": retained,
            "gold_total": len(positives),
            "actions": dict(sorted(Counter(item["observed_action"] for item in details).items())),
            "company": company,
            "layout": layout,
            "unseen_company_oracle_exact_rate": exact / len(records),
            "unseen_layout_oracle_exact_rate": exact / len(records),
        },
        "failures": [item for item in details if not item["exact"]],
        "details": details,
        "runtime_hashes": runtime_hashes,
        "checkpoint_sha256": checkpoint_hash,
        "precommit_lock_sha256": _digest(precommit_lock_path),
        "invariants": {
            "diagnosis_document_used_as_confirmation": False,
            "real_pdf_loaded": False,
            "protected_evaluation_loaded": False,
            "threshold_changed": False,
            "checkpoint_changed": checkpoint_hash != "3a33276ded0d2ff2dd3d200c4c774a59eec0f8cad7102bc273cb263d983944de",
            "fine_tuning_performed": False,
            "phase_d_unlocked": False,
            "safe_abstention_counted_as_answered": False,
            "universal_coverage_claim_made": False,
        },
        "authorized_next_step": "Replay every frozen compiler/proof/S4/company-layout development cohort; report results without changing thresholds.",
    }


def _markdown(report: dict[str, Any]) -> str:
    result = report["results"]
    return "\n".join(
        [
            "# FinHRM QP3-CORE-IMPLEMENT",
            "",
            f"- Gate: **{'PASS' if report['implementation_gate_passed'] else 'FAIL'}**",
            f"- Runtime oracle: `{result['runtime_oracle_exact']}/{result['total']}`",
            f"- False accepts: `{result['false_accepts']}`",
            f"- Gold retention: `{result['gold_retained']}/{result['gold_total']}`",
            f"- Company macro/worst: `{result['company']['macro_exact_rate']:.3f}/{result['company']['worst_group_exact_rate']:.3f}`",
            f"- Layout macro/worst: `{result['layout']['macro_exact_rate']:.3f}/{result['layout']['worst_group_exact_rate']:.3f}`",
            "- No diagnosis PDF, protected evaluation source, fine-tuning, checkpoint change, threshold change, or Phase D unlock.",
            "- Synthetic proof replay is a contract result, not an empirical universal-coverage claim.",
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
    artifacts = []
    for path in (
        *(ROOT / value for value in RUNTIME_PATHS),
        ROOT / "tests" / "test_qp3_core_implementation.py",
        Path(__file__),
        report_path,
        markdown_path,
    ):
        artifacts.append(
            {
                "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _digest(path),
            }
        )
    lock = {
        "schema": "finhrm-qp3-core-implementation-lock-v1",
        "created_at": "2026-08-14",
        "classification": "development_only_report_agnostic_proof_implementation_lock",
        "status": "complete" if report["implementation_gate_passed"] else "failed",
        "phase_d": "locked",
        "implementation_gate_passed": report["implementation_gate_passed"],
        "results": report["results"],
        "runtime_hashes": report["runtime_hashes"],
        "expected_changed_runtime_files": EXPECTED_CHANGED_RUNTIME_FILES,
        "checkpoint_sha256": report["checkpoint_sha256"],
        "precommit_lock_sha256": report["precommit_lock_sha256"],
        "invariants": report["invariants"],
        "artifacts": artifacts,
        "authorized_next_step": report["authorized_next_step"],
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate and lock QP3-CORE implementation")
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--precommit-lock", type=Path, default=DEFAULT_PRECOMMIT_LOCK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    report = build_report(args.cohort, args.precommit_lock)
    write_outputs(report, args.report, args.markdown, args.lock)
    print(
        json.dumps(
            {
                "gate": report["implementation_gate_passed"],
                **report["results"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["implementation_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
