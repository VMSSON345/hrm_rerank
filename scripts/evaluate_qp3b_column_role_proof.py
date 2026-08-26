from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from hrm_finance_qa.evidence_schema import EvidenceCell, EvidenceGraph
from hrm_finance_qa.ontology import normalize_text
from hrm_finance_qa.question_planner import compile_question_plan
from hrm_finance_qa.symbolic_verifier import assess_column_role_proof


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config" / "finhrm_qp3a_column_role_proof_precommit_v1.json"
DEFAULT_COHORT = ROOT / "data" / "finhrm_qp3a_column_role_proof_development_v1.jsonl"
DEFAULT_QP3A_LOCK = ROOT / "reports" / "finhrm_qp3a_column_role_proof_precommit_lock_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp3b_column_role_proof_development_v1.json"
PROTECTED_PATH_TOKENS = {"validation", "blind", "holdout", "test"}
RUNTIME_FILES = {
    "evidence_schema.py": ROOT / "src" / "hrm_finance_qa" / "evidence_schema.py",
    "question_planner.py": ROOT / "src" / "hrm_finance_qa" / "question_planner.py",
    "program_synthesis.py": ROOT / "src" / "hrm_finance_qa" / "program_synthesis.py",
    "symbolic_verifier.py": ROOT / "src" / "hrm_finance_qa" / "symbolic_verifier.py",
    "document_hrm.py": ROOT / "src" / "hrm_finance_qa" / "document_hrm.py",
    "document_reasoning_engine.py": ROOT / "src" / "hrm_finance_qa" / "document_reasoning_engine.py",
    "program_data.py": ROOT / "src" / "hrm_finance_qa" / "program_data.py",
}


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _refuse_protected(path: Path) -> None:
    overlap = {
        part.lower() for part in path.resolve().parts
    }.intersection(PROTECTED_PATH_TOKENS)
    if overlap:
        raise ValueError(f"Refusing protected input path {path}: {sorted(overlap)}")


def _read_json(path: Path) -> dict[str, Any]:
    _refuse_protected(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _refuse_protected(path)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Cohort must contain JSON objects")
    return records


def _evidence_graph(record: dict[str, Any]) -> EvidenceGraph:
    cells = record["evidence_cells"]
    token_values = {
        token: Decimal(index + 1)
        for index, token in enumerate(
            sorted({str(cell["extraction_value_token"]) for cell in cells})
        )
    }
    evidence_cells: list[EvidenceCell] = []
    for index, cell in enumerate(cells):
        period = str(cell["period"])
        role = str(cell["column_role"]).lower()
        evidence_cells.append(
            EvidenceCell(
                index=index,
                value=token_values[str(cell["extraction_value_token"])],
                metric_label=str(cell["metric"]),
                metric_key=normalize_text(str(cell["metric"])),
                column=role,
                period=period,
                explicit_years=(period,) if period else (),
                period_role="",
                entity=str(cell["entity_scope"]),
                section="",
                table_id=str(cell["table_id"]),
                source_name=str(cell["document_id"]),
                page=int(cell["page"]),
                value_status="observed",
                unit=str(cell["unit_family"]),
                unit_key=str(cell["unit_family"]),
                row_id=str(cell["row_id"]),
                column_id=str(cell["column_id"]),
                entity_id=str(cell["entity_scope"]),
                column_role=role,
                column_role_source="precommitted_synthetic_evidence",
                column_role_confidence=(
                    "CONFLICT" if cell["column_role_conflict"] else "EXPLICIT"
                ),
                column_role_evidence=(role,) if role else (),
                column_role_conflict=bool(cell["column_role_conflict"]),
                extraction_method=str(cell["extraction_method"]),
                source_span=tuple(int(value) for value in cell["source_span"]),
            )
        )
    return EvidenceGraph(
        cells=tuple(evidence_cells),
        edges=(),
        column_role_inventory_state=str(record["role_inventory_state"]),
    )


def build_report(
    precommit_path: Path = DEFAULT_PRECOMMIT,
    cohort_path: Path = DEFAULT_COHORT,
    qp3a_lock_path: Path = DEFAULT_QP3A_LOCK,
) -> dict[str, Any]:
    precommit = _read_json(precommit_path)
    records = _read_jsonl(cohort_path)
    qp3a_lock = _read_json(qp3a_lock_path)
    errors: list[str] = []

    locked_artifacts = {
        str(item.get("path", "")).replace("\\", "/"): item
        for item in qp3a_lock.get("artifacts", ())
    }
    for name, path in (("precommit", precommit_path), ("cohort", cohort_path)):
        relative_path = path.resolve().relative_to(ROOT.resolve()).as_posix()
        expected = locked_artifacts.get(relative_path, {}).get("sha256", "")
        if not expected or _digest(path) != expected:
            errors.append(f"qp3a_lock_hash_mismatch:{name}")
    locked_dependencies = {
        str(item.get("path", "")).replace("\\", "/"): item
        for item in qp3a_lock.get("frozen_dependencies", ())
    }
    checkpoint_path = ROOT / "models" / "document_hrm_V8_candidate.pt"
    checkpoint_relative = checkpoint_path.relative_to(ROOT).as_posix()
    expected_checkpoint_hash = locked_dependencies.get(
        checkpoint_relative, {}
    ).get("sha256", "")
    observed_checkpoint_hash = _digest(checkpoint_path)
    if (
        not expected_checkpoint_hash
        or observed_checkpoint_hash != expected_checkpoint_hash
    ):
        errors.append("qp3a_lock_hash_mismatch:checkpoint")

    expected_total = int(precommit["gate"]["exact_examples_required"])
    if len(records) != expected_total:
        errors.append("wrong_total_examples")

    details: list[dict[str, Any]] = []
    state_exact = 0
    action_exact = 0
    by_scenario = Counter()
    by_group = Counter()
    for record in records:
        graph = _evidence_graph(record)
        plan = compile_question_plan(str(record["question"]))
        assessment = assess_column_role_proof(
            plan,
            graph,
            selected_indices=tuple(cell.index for cell in graph.cells),
            operation=str(record["operation"]),
        )
        expected_state = str(record["expected_column_role_proof_state"])
        expected_action = str(record["expected_action"])
        state_ok = assessment.state.value == expected_state
        action_ok = assessment.action == expected_action
        state_exact += int(state_ok)
        action_exact += int(action_ok)
        by_scenario[str(record["scenario_id"])] += 1
        by_group[str(record["group_id"])] += 1
        details.append(
            {
                "example_id": record["example_id"],
                "observed_state": assessment.state.value,
                "expected_state": expected_state,
                "observed_action": assessment.action,
                "expected_action": expected_action,
                "reason": assessment.reason,
                "applicable": assessment.applicable,
                "exact": state_ok and action_ok,
            }
        )

    scenario_exact = Counter(
        record["scenario_id"]
        for record, detail in zip(records, details)
        if detail["exact"]
    )
    same_value_total = by_scenario[
        "no_role_same_value_multi_role_conflict_complete"
    ]
    same_value_exact = scenario_exact[
        "no_role_same_value_multi_role_conflict_complete"
    ]
    permutation_total = by_scenario["no_role_conflict_permuted_complete"]
    permutation_exact = scenario_exact["no_role_conflict_permuted_complete"]
    unknown_total = by_scenario["no_role_unique_inventory_unknown"]
    unknown_exact = scenario_exact["no_role_unique_inventory_unknown"]

    gate = precommit["gate"]
    implementation_gate_passed = (
        not errors
        and state_exact == action_exact == len(records) == expected_total
        and same_value_exact == same_value_total == gate["same_value_conflict_required"]
        and permutation_exact == permutation_total == gate["permutation_pairs_required"]
        and unknown_exact == unknown_total == gate["unknown_inventory_controls_required"]
    )
    return {
        "schema": "finhrm-qp3b-column-role-proof-development-evaluation-v1",
        "classification": "development_only_synthetic_evidence_evaluation",
        "implementation_gate_passed": implementation_gate_passed,
        "errors": errors,
        "cohort": {
            "examples": len(records),
            "groups": len(by_group),
            "by_scenario": dict(sorted(by_scenario.items())),
        },
        "exactness": {
            "state_exact": state_exact,
            "action_exact": action_exact,
            "total": len(records),
            "same_value_conflict_exact": same_value_exact,
            "same_value_conflict_total": same_value_total,
            "permutation_exact": permutation_exact,
            "permutation_total": permutation_total,
            "unknown_inventory_exact": unknown_exact,
            "unknown_inventory_total": unknown_total,
        },
        "failures": [detail for detail in details if not detail["exact"]],
        "details": details,
        "artifact_hashes": {
            "precommit": _digest(precommit_path),
            "cohort": _digest(cohort_path),
            "qp3a_lock": _digest(qp3a_lock_path),
            "checkpoint": observed_checkpoint_hash,
        },
        "runtime_hashes": {
            name: _digest(path) for name, path in RUNTIME_FILES.items()
        },
        "invariants": {
            "numeric_values_used": False,
            "numeric_answers_used": False,
            "candidate_scores_used": False,
            "candidate_order_used_as_signal": False,
            "threshold_changed": False,
            "checkpoint_changed": False,
            "fine_tuning_performed": False,
            "document_loaded": False,
            "pdf_loaded": False,
            "model_loaded": False,
            "global_validation_loaded": False,
            "sealed_validation_loaded": False,
            "blind_test_loaded": False,
            "phase_d_unlocked": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate QP3B against the frozen development-only QP3A cohort"
    )
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--qp3a-lock", type=Path, default=DEFAULT_QP3A_LOCK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = build_report(args.precommit, args.cohort, args.qp3a_lock)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["exactness"], ensure_ascii=False, sort_keys=True))
    print(f"implementation_gate_passed={report['implementation_gate_passed']}")
    return 0 if report["implementation_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
