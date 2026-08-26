from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from hrm_finance_qa.question_planner import compile_question_plan  # noqa: E402
from scripts.audit_qp2a_operand_slots import (  # noqa: E402
    _canonical_actual_slots,
    _canonical_expected_slots,
    _catalog_actual_slots,
    _catalog_expected_slots,
)


DEFAULT_CONTRACT = ROOT / "config" / "finhrm_thesis_core_v1.json"
DEFAULT_CATALOG = ROOT / "config" / "question_family_catalog_plan_actual_v1.jsonl"
DEFAULT_BEFORE = ROOT / "reports" / "finhrm_thesis_core_compiler_qp2a_v1.json"
DEFAULT_AFTER = ROOT / "reports" / "finhrm_thesis_core_compiler_qp2b_v1.json"
DEFAULT_QP2A_AUDIT = ROOT / "reports" / "finhrm_qp2a_operand_slots_development_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp2b_operation_direction_development_v1.json"
PROTECTED_PATH_TOKENS = {"validation", "blind", "holdout", "test"}
TARGET_FAMILIES = {"PA04", "PA06", "PA07", "PA08", "PA09", "TY04"}


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _refuse_protected(path: Path) -> None:
    overlap = {part.lower() for part in path.resolve().parts}.intersection(PROTECTED_PATH_TOKENS)
    if overlap:
        raise ValueError(f"Refusing protected path {path}: {sorted(overlap)}")


def _read_json(path: Path) -> dict[str, Any]:
    _refuse_protected(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def _read_catalog(path: Path) -> list[dict[str, Any]]:
    _refuse_protected(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("classification") != "development_only_synthetic_compiler_contract":
            raise ValueError("Catalog contains non-development data")
        if row.get("document_content_used") is not False or row.get("numeric_answer_used") is not False:
            raise ValueError("Catalog row uses document content or a numeric answer")
    return rows


def _direction_value(plan: Any) -> str | None:
    value = plan.direction_intent
    return value.value if value is not None else None


def _expected_direction(expected: dict[str, Any]) -> str | None:
    explicit = expected.get("direction_intent")
    if explicit is not None:
        return str(explicit)
    dag = list(expected.get("program_dag_operations", ()))
    roles = list(expected.get("column_roles", ()))
    if dag == ["SUBTRACT", "RATIO_PERCENT"] and len(roles) >= 2:
        first, second = roles[:2]
        if {first, second} == {"ACTUAL", "PLAN"}:
            return f"{first}_MINUS_{second}"
    return None


def _summary(details: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for detail in details:
        name = str(detail[key])
        counts[name]["total"] += 1
        counts[name]["exact"] += int(detail["exact"])
    return {
        name: {
            "exact": values["exact"],
            "total": values["total"],
            "exact_rate": values["exact"] / values["total"],
        }
        for name, values in sorted(counts.items())
    }


def build_report(
    contract_path: Path = DEFAULT_CONTRACT,
    catalog_path: Path = DEFAULT_CATALOG,
    before_path: Path = DEFAULT_BEFORE,
    after_path: Path = DEFAULT_AFTER,
    qp2a_audit_path: Path = DEFAULT_QP2A_AUDIT,
) -> dict[str, Any]:
    contract = _read_json(contract_path)
    catalog = _read_catalog(catalog_path)
    before = _read_json(before_path)
    after = _read_json(after_path)
    qp2a = _read_json(qp2a_audit_path)
    frozen_hash = contract["frozen_source_artifacts"]["question_family_catalog"]["sha256"]
    if frozen_hash != _digest(catalog_path):
        raise ValueError("Frozen catalog hash mismatch")
    core_ids = set(contract["scope"]["core_family_ids"])
    answer_ids = set(contract["scope"]["answer_family_ids"])

    qp2a_canonical_exact = 0
    for family in contract["families"]:
        if family["family_id"] not in answer_ids:
            continue
        plan = compile_question_plan(family["canonical_question"])
        qp2a_canonical_exact += (
            _canonical_expected_slots(family["expected"])
            == _canonical_actual_slots(plan)
        )

    details: list[dict[str, Any]] = []
    qp2a_variant_exact = 0
    for row in catalog:
        if row["family_id"] not in core_ids or row["expected_ir"]["behavior"] != "ANSWER":
            continue
        plan = compile_question_plan(row["question"])
        expected_operation = row["expected_ir"].get("root_operation")
        expected_direction = _expected_direction(row["expected_ir"])
        operation_exact = plan.operation == expected_operation
        direction_exact = _direction_value(plan) == expected_direction
        qp2a_variant_exact += (
            _catalog_expected_slots(row["expected_ir"])
            == _catalog_actual_slots(plan)
        )
        details.append(
            {
                "example_id": row["example_id"],
                "family_id": row["family_id"],
                "variant": row["variant"],
                "question": row["question"],
                "expected_operation": expected_operation,
                "actual_operation": plan.operation,
                "expected_direction": expected_direction,
                "actual_direction": _direction_value(plan),
                "operation_exact": operation_exact,
                "direction_exact": direction_exact,
                "exact": operation_exact and direction_exact,
            }
        )

    before_by_id = {
        item["example_id"]: bool(item["exact"])
        for item in before["legacy_catalog_robustness"]["details"]
    }
    after_by_id = {
        item["example_id"]: bool(item["exact"])
        for item in after["legacy_catalog_robustness"]["details"]
    }
    shared = sorted(set(before_by_id).intersection(after_by_id))
    improvements = [item for item in shared if not before_by_id[item] and after_by_id[item]]
    regressions = [item for item in shared if before_by_id[item] and not after_by_id[item]]
    target = [item for item in details if item["family_id"] in TARGET_FAMILIES]
    role_swaps = [item for item in details if item["variant"] == "minimal_pair_role_swap"]
    year_shifts = [item for item in details if item["variant"] == "minimal_pair_year_shift"]
    long_surfaces = [
        item for item in details if item["variant"] in {"long_clause", "no_diacritics", "clause_reordered"}
    ]
    exact = sum(item["exact"] for item in details)
    target_exact = sum(item["exact"] for item in target)
    role_swap_exact = sum(item["exact"] for item in role_swaps)
    year_shift_exact = sum(item["exact"] for item in year_shifts)
    long_exact = sum(item["exact"] for item in long_surfaces)
    qp2b_passed = (
        exact == len(details) == 108
        and target_exact == len(target) == 54
        and role_swap_exact == len(role_swaps) == 12
        and year_shift_exact == len(year_shifts) == 12
        and long_exact == len(long_surfaces) == 36
        and not regressions
        and qp2a["qp2a_gate_passed"] is True
        and qp2a_canonical_exact == 13
        and qp2a_variant_exact == 108
        and after["strict_canonical_contract"]["dimension_metrics"]["operation"]["pass"] == 13
        and after["strict_canonical_contract"]["dimension_metrics"]["direction_intent"]["pass"] == 13
    )
    return {
        "schema": "finhrm-qp2b-operation-direction-development-v1",
        "classification": "development_only_synthetic_compiler_audit",
        "evaluation_mode": "question_compiler_only_no_document_no_model_no_execution",
        "target_families": sorted(TARGET_FAMILIES),
        "qp2b_gate_passed": qp2b_passed,
        "all_answer_variants": {
            "exact": exact,
            "total": len(details),
            "exact_rate": exact / len(details),
        },
        "target_answer_variants": {
            "exact": target_exact,
            "total": len(target),
            "exact_rate": target_exact / len(target),
        },
        "counterfactuals": {
            "role_swap_exact": role_swap_exact,
            "role_swap_total": len(role_swaps),
            "year_shift_exact": year_shift_exact,
            "year_shift_total": len(year_shifts),
        },
        "surface_robustness": {
            "long_no_diacritic_reordered_exact": long_exact,
            "long_no_diacritic_reordered_total": len(long_surfaces),
        },
        "by_family": _summary(details, "family_id"),
        "by_variant": _summary(details, "variant"),
        "details": details,
        "strict_contract_delta": {
            "exact_before": before["strict_canonical_contract"]["exact"],
            "exact_after": after["strict_canonical_contract"]["exact"],
            "total": after["strict_canonical_contract"]["examples"],
            "operation_before": before["strict_canonical_contract"]["dimension_metrics"]["operation"]["pass"],
            "operation_after": after["strict_canonical_contract"]["dimension_metrics"]["operation"]["pass"],
            "direction_before": before["strict_canonical_contract"]["dimension_metrics"]["direction_intent"]["pass"],
            "direction_after": after["strict_canonical_contract"]["dimension_metrics"]["direction_intent"]["pass"],
            "dimension_total": 13,
        },
        "legacy_collateral": {
            "exact_before": before["legacy_catalog_robustness"]["exact"],
            "exact_after": after["legacy_catalog_robustness"]["exact"],
            "total": after["legacy_catalog_robustness"]["examples"],
            "improvements": improvements,
            "regressions": regressions,
        },
        "qp2a_retention": {
            "qp2a_snapshot_gate_passed": qp2a["qp2a_gate_passed"],
            "current_canonical_slots": qp2a_canonical_exact,
            "current_canonical_slots_total": 13,
            "current_frozen_variant_slots": qp2a_variant_exact,
            "current_frozen_variant_slots_total": 108,
        },
        "artifact_hashes": {
            "contract": _digest(contract_path),
            "catalog": _digest(catalog_path),
            "before_report": _digest(before_path),
            "after_report": _digest(after_path),
            "qp2a_audit": _digest(qp2a_audit_path),
            "qp2a_audit_script": _digest(ROOT / "scripts" / "audit_qp2a_operand_slots.py"),
            "question_planner": _digest(ROOT / "src" / "hrm_finance_qa" / "question_planner.py"),
            "program_synthesis": _digest(ROOT / "src" / "hrm_finance_qa" / "program_synthesis.py"),
            "symbolic_verifier": _digest(ROOT / "src" / "hrm_finance_qa" / "symbolic_verifier.py"),
            "audit_script": _digest(Path(__file__)),
        },
        "invariants": {
            "runtime_planning_behavior_changed": True,
            "operand_slots_consumed_by_runtime": False,
            "program_synthesis_source_changed": False,
            "symbolic_verifier_source_changed": False,
            "runtime_acceptance_policy_changed": False,
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
        "authorized_next_step": "QP2C reference-base and program-DAG compilation; keep runtime acceptance unchanged.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit QP2B operation and direction")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    parser.add_argument("--qp2a-audit", type=Path, default=DEFAULT_QP2A_AUDIT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = build_report(args.contract, args.catalog, args.before, args.after, args.qp2a_audit)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key not in {"details", "by_family", "by_variant"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
