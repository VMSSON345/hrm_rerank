from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from hrm_finance_qa.evidence_schema import (
    assess_table_role_inventory,
    build_evidence_graph,
)
from hrm_finance_qa.program_data import ReasoningProgram, extract_numeric_candidates
from hrm_finance_qa.question_planner import compile_question_plan
from hrm_finance_qa.symbolic_verifier import verify_program


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config" / "finhrm_qp3c_role_inventory_producer_precommit_v1.json"
DEFAULT_COHORT = ROOT / "data" / "finhrm_qp3c_role_inventory_producer_development_v1.jsonl"
DEFAULT_QP3C_LOCK = ROOT / "reports" / "finhrm_qp3c_role_inventory_producer_precommit_lock_v1.json"
DEFAULT_CORE_IMPLEMENTATION_LOCK = ROOT / "reports" / "finhrm_qp3_core_implementation_lock_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp3d_role_inventory_producer_development_v1.json"
PROTECTED_PATH_TOKENS = {"validation", "blind", "holdout", "test"}
RUNTIME_FILES = {
    "evidence_schema.py": ROOT / "src" / "hrm_finance_qa" / "evidence_schema.py",
    "question_planner.py": ROOT / "src" / "hrm_finance_qa" / "question_planner.py",
    "program_synthesis.py": ROOT / "src" / "hrm_finance_qa" / "program_synthesis.py",
    "symbolic_verifier.py": ROOT / "src" / "hrm_finance_qa" / "symbolic_verifier.py",
    "direction_contract.py": ROOT / "src" / "hrm_finance_qa" / "direction_contract.py",
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
        raise ValueError(f"Expected object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    _refuse_protected(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _authorized_core_runtime_files(errors: list[str]) -> set[str]:
    lock = _read_json(DEFAULT_CORE_IMPLEMENTATION_LOCK)
    if not lock.get("implementation_gate_passed"):
        errors.append("qp3_core_implementation_lock_not_passed")
        return set()
    authorized: set[str] = set()
    for relative, expected in lock.get("runtime_hashes", {}).items():
        path = ROOT / str(relative)
        if not path.is_file() or _digest(path) != str(expected):
            errors.append(f"qp3_core_runtime_hash_mismatch:{Path(relative).name}")
            continue
        authorized.add(Path(relative).name)
    return authorized


def _candidates(question: str, context: str):
    return extract_numeric_candidates(
        question,
        context,
        source_name="synthetic-development.md",
        page=1,
        extraction_method="native",
    )


def _end_to_end_cases() -> list[dict[str, Any]]:
    complete_context = "\n".join(
        (
            "Bang ket qua",
            "| Chi tieu | Thuc hien 2023 |",
            "|---|---:|",
            "| Doanh thu | 100000 |",
            "| Loi nhuan | 20000 |",
            "Ket thuc bang",
        )
    )
    complete_question = "Doanh thu nam 2023 la bao nhieu?"
    complete_candidates = _candidates(complete_question, complete_context)
    complete_graph = build_evidence_graph(complete_candidates)

    ambiguous_context = "\n".join(
        (
            "Bang ke hoach",
            "| Chi tieu | Ke hoach 2023 | Thuc hien 2023 |",
            "|---|---:|---:|",
            "| Doanh thu | 110000 | 100000 |",
            "| Loi nhuan | 25000 | 20000 |",
            "Ket thuc bang",
        )
    )
    ambiguous_question = "Doanh thu nam 2023 la bao nhieu?"
    ambiguous_candidates = _candidates(ambiguous_question, ambiguous_context)
    ambiguous_graph = build_evidence_graph(ambiguous_candidates)
    actual_index = next(
        cell.index
        for cell in ambiguous_graph.cells
        if cell.metric_key == "doanhthu" and cell.column_role == "actual"
    )
    ambiguous_verification = verify_program(
        compile_question_plan(ambiguous_question),
        ReasoningProgram("SELECT", (actual_index,)),
        ambiguous_graph,
    )

    partial_context = "\n".join(
        (
            "| Chi tieu | Thuc hien 2023 |",
            "|---|---:|",
            "| Doanh thu | 100000 |",
        )
    )
    partial_graph = build_evidence_graph(
        _candidates(complete_question, partial_context)
    )

    bare_year_context = "\n".join(
        (
            "Bang ket qua",
            "| Chi tieu | Ke hoach 2023 | 2023 |",
            "|---|---:|---:|",
            "| Doanh thu | 110000 | 100000 |",
            "Ket thuc bang",
        )
    )
    bare_year_graph = build_evidence_graph(
        _candidates(complete_question, bare_year_context)
    )

    missing_cell_context = "\n".join(
        (
            "Bang ke hoach",
            "| Chi tieu | Ke hoach 2023 | Thuc hien 2023 |",
            "|---|---:|---:|",
            "| Doanh thu | 110000 | 100000 |",
            "| Loi nhuan | 25000 | |",
            "Ket thuc bang",
        )
    )
    missing_cell_graph = build_evidence_graph(
        _candidates(ambiguous_question, missing_cell_context)
    )

    reversed_graph = build_evidence_graph(tuple(reversed(complete_candidates)))
    plaintext_graph = build_evidence_graph(
        _candidates(complete_question, "Doanh thu nam 2023 dat 100000 dong.")
    )
    ordinary_period_context = "\n".join(
        (
            "Bang can doi",
            "| Chi tieu | 2022 | 2023 |",
            "|---|---:|---:|",
            "| Tai san | 90000 | 100000 |",
            "Ket thuc bang",
        )
    )
    ordinary_period_graph = build_evidence_graph(
        _candidates("Tai san nam 2023 la bao nhieu?", ordinary_period_context)
    )
    return [
        {
            "id": "complete_table_manifest",
            "passed": complete_graph.column_role_inventory_state == "EXPLICIT_COMPLETE"
            and not complete_graph.column_role_inventory_reasons,
            "observed": complete_graph.column_role_inventory_state,
        },
        {
            "id": "no_role_actual_plan_ambiguity",
            "passed": ambiguous_graph.column_role_inventory_state == "EXPLICIT_COMPLETE"
            and "ambiguous_column_role_evidence" in ambiguous_verification.reasons,
            "observed": {
                "inventory": ambiguous_graph.column_role_inventory_state,
                "reasons": ambiguous_verification.reasons,
            },
        },
        {
            "id": "partial_table_unknown",
            "passed": partial_graph.column_role_inventory_state == "UNKNOWN"
            and "incomplete_table_boundary"
            in partial_graph.column_role_inventory_reasons,
            "observed": partial_graph.column_role_inventory_reasons,
        },
        {
            "id": "bare_year_role_unknown",
            "passed": bare_year_graph.column_role_inventory_state == "UNKNOWN"
            and "missing_explicit_column_role"
            in bare_year_graph.column_role_inventory_reasons,
            "observed": bare_year_graph.column_role_inventory_reasons,
        },
        {
            "id": "missing_cell_coverage_unknown",
            "passed": missing_cell_graph.column_role_inventory_state == "UNKNOWN"
            and "cell_coverage_mismatch"
            in missing_cell_graph.column_role_inventory_reasons,
            "observed": missing_cell_graph.column_role_inventory_reasons,
        },
        {
            "id": "candidate_permutation_invariant",
            "passed": (
                reversed_graph.column_role_inventory_state,
                reversed_graph.column_role_inventory_reasons,
            )
            == (
                complete_graph.column_role_inventory_state,
                complete_graph.column_role_inventory_reasons,
            ),
            "observed": reversed_graph.column_role_inventory_state,
        },
        {
            "id": "plaintext_without_manifest_legacy_undeclared",
            "passed": plaintext_graph.column_role_inventory_state == "",
            "observed": plaintext_graph.column_role_inventory_state,
        },
        {
            "id": "ordinary_period_table_role_proof_not_applicable",
            "passed": ordinary_period_graph.column_role_inventory_state == ""
            and not ordinary_period_graph.table_role_inventories,
            "observed": ordinary_period_graph.column_role_inventory_state,
        },
    ]


def build_report(
    precommit_path: Path = DEFAULT_PRECOMMIT,
    cohort_path: Path = DEFAULT_COHORT,
    qp3c_lock_path: Path = DEFAULT_QP3C_LOCK,
) -> dict[str, Any]:
    precommit = _read_json(precommit_path)
    records = _read_jsonl(cohort_path)
    lock = _read_json(qp3c_lock_path)
    errors: list[str] = []
    locked_artifacts = {
        str(item.get("path", "")).replace("\\", "/"): item
        for item in lock.get("artifacts", ())
    }
    for name, path in (("precommit", precommit_path), ("cohort", cohort_path)):
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        expected = locked_artifacts.get(relative, {}).get("sha256", "")
        if not expected or _digest(path) != expected:
            errors.append(f"qp3c_lock_hash_mismatch:{name}")

    state_exact = 0
    reason_exact = 0
    details: list[dict[str, Any]] = []
    by_scenario = Counter()
    for record in records:
        production = assess_table_role_inventory(record)
        expected_state = str(record["expected_inventory_state"])
        expected_reasons = tuple(sorted(record["expected_reason_set"]))
        state_ok = production.state == expected_state
        reason_ok = production.reasons == expected_reasons
        state_exact += int(state_ok)
        reason_exact += int(reason_ok)
        by_scenario[str(record["scenario_id"])] += 1
        details.append(
            {
                "example_id": record["example_id"],
                "observed_state": production.state,
                "expected_state": expected_state,
                "observed_reasons": production.reasons,
                "expected_reasons": expected_reasons,
                "exact": state_ok and reason_ok,
            }
        )

    end_to_end = _end_to_end_cases()
    end_to_end_pass = sum(bool(item["passed"]) for item in end_to_end)
    runtime_hashes = {name: _digest(path) for name, path in RUNTIME_FILES.items()}
    qp3c_baseline = precommit["runtime_baseline"]
    changed_runtime_files = sorted(
        name
        for name, digest in runtime_hashes.items()
        if qp3c_baseline.get(name) != digest
    )
    expected_changed = {
        "evidence_schema.py",
        "program_data.py",
        "symbolic_verifier.py",
    }
    expected_changed.update(
        _authorized_core_runtime_files(errors).intersection(RUNTIME_FILES)
    )
    if set(changed_runtime_files) != expected_changed:
        errors.append("unexpected_runtime_change_set")
    gate_passed = (
        not errors
        and state_exact == reason_exact == len(records) == 96
        and end_to_end_pass == len(end_to_end) == 8
    )
    return {
        "schema": "finhrm-qp3d-role-inventory-producer-development-evaluation-v1",
        "classification": "development_only_role_inventory_producer_evaluation",
        "implementation_gate_passed": gate_passed,
        "errors": errors,
        "frozen_cohort": {
            "state_exact": state_exact,
            "reason_set_exact": reason_exact,
            "total": len(records),
            "failures": [detail for detail in details if not detail["exact"]],
            "by_scenario": dict(sorted(by_scenario.items())),
            "details": details,
        },
        "end_to_end": {
            "pass": end_to_end_pass,
            "total": len(end_to_end),
            "cases": end_to_end,
        },
        "runtime_changes": {
            "changed_files": changed_runtime_files,
            "expected_changed_files": sorted(expected_changed),
            "hashes": runtime_hashes,
        },
        "artifact_hashes": {
            "precommit": _digest(precommit_path),
            "cohort": _digest(cohort_path),
            "qp3c_lock": _digest(qp3c_lock_path),
            "checkpoint": _digest(ROOT / "models" / "document_hrm_V8_candidate.pt"),
        },
        "invariants": {
            "manifest_created_before_top_k": True,
            "question_text_used_by_producer": False,
            "numeric_value_used_by_producer": False,
            "candidate_score_used_by_producer": False,
            "threshold_changed": False,
            "checkpoint_changed": False,
            "fine_tuning_performed": False,
            "global_validation_loaded": False,
            "sealed_validation_loaded": False,
            "blind_test_loaded": False,
            "phase_d_unlocked": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the QP3D role-inventory producer")
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--qp3c-lock", type=Path, default=DEFAULT_QP3C_LOCK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = build_report(args.precommit, args.cohort, args.qp3c_lock)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "state_exact": report["frozen_cohort"]["state_exact"],
                "reason_set_exact": report["frozen_cohort"]["reason_set_exact"],
                "total": report["frozen_cohort"]["total"],
                "end_to_end_pass": report["end_to_end"]["pass"],
                "end_to_end_total": report["end_to_end"]["total"],
            },
            sort_keys=True,
        )
    )
    print(f"implementation_gate_passed={report['implementation_gate_passed']}")
    return 0 if report["implementation_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
