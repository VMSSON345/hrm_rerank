from __future__ import annotations

import argparse
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
    _catalog_actual_slots,
    _catalog_expected_slots,
)
from scripts.audit_qp2b_operation_direction import _expected_direction  # noqa: E402
from scripts.audit_qp2c_reference_dag import (  # noqa: E402
    _actual_dag,
    _expected_catalog_program,
)


DEFAULT_CONTRACT = ROOT / "config" / "finhrm_thesis_core_v1.json"
DEFAULT_CATALOG = ROOT / "config" / "question_family_catalog_plan_actual_v1.jsonl"
DEFAULT_BEFORE = ROOT / "reports" / "finhrm_thesis_core_compiler_qp2c_v1.json"
DEFAULT_AFTER = ROOT / "reports" / "finhrm_thesis_core_compiler_qp2d_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp2d_safe_abstention_development_v1.json"
PROTECTED_PATH_TOKENS = {"validation", "blind", "holdout", "test"}
ANSWER_FAMILIES = {
    "PA01", "PA02", "PA03", "PA04", "PA05", "PA06", "PA07", "PA08", "PA09",
    "TY01", "TY02", "TY04",
}
RUNTIME_CONSUMERS = (
    ROOT / "src" / "hrm_finance_qa" / "program_synthesis.py",
    ROOT / "src" / "hrm_finance_qa" / "symbolic_verifier.py",
    ROOT / "src" / "hrm_finance_qa" / "document_hrm.py",
    ROOT / "src" / "hrm_finance_qa" / "document_reasoning_engine.py",
    ROOT / "src" / "hrm_finance_qa" / "program_data.py",
)


MISSING_REFERENCE_CASES = (
    "Doanh thu năm 2023 lớn hơn bao nhiêu phần trăm?",
    "Doanh thu năm 2023 cao hơn bao nhiêu phần trăm?",
    "Doanh thu năm 2023 thấp hơn bao nhiêu phần trăm?",
    "Doanh thu thực hiện năm 2023 tăng bao nhiêu phần trăm?",
    "Doanh thu thực hiện năm 2023 giảm bao nhiêu phần trăm?",
    "Doanh thu năm 2023 chiếm bao nhiêu phần trăm?",
    "Doanh thu năm 2023 đạt bao nhiêu phần trăm?",
    "Doanh thu nam 2023 lon hon bao nhieu phan tram?",
)
EXPLICIT_REFERENCE_REPAIRS = (
    "Doanh thu hoạt động thực hiện năm 2023 thay đổi bao nhiêu phần trăm so với kế hoạch năm 2023?",
    "Doanh thu hoạt động thực hiện năm 2023 lớn hơn kế hoạch năm 2023 bao nhiêu phần trăm?",
    "Doanh thu hoạt động thực hiện năm 2023 chiếm bao nhiêu phần trăm kế hoạch năm 2023?",
    "Doanh thu thực hiện năm 2024 tăng bao nhiêu phần trăm so với thực hiện năm 2023?",
)
MISSING_ROLE_CASES = (
    "Doanh thu hoạt động năm 2023 là bao nhiêu khi bảng có cả kế hoạch và thực hiện?",
    "Doanh thu hoat dong nam 2023 la bao nhieu khi bang co ca ke hoach va thuc hien?",
    "Doanh thu hoạt động năm 2023 là bao nhiêu nếu trong bảng có cả kế hoạch và thực hiện?",
    "Khi báo cáo có đồng thời actual và plan doanh thu hoạt động năm 2023 là bao nhiêu?",
    "Trong bảng có kế hoạch và thực hiện doanh thu hoạt động năm 2023 là bao nhiêu?",
    "Doanh thu hoạt động năm 2023 là bao nhiêu khi báo cáo có đồng thời actual và plan?",
)
EXPLICIT_ROLE_REPAIRS = (
    (
        "Trong bảng có cả kế hoạch và thực hiện, doanh thu hoạt động thực hiện năm 2023 là bao nhiêu?",
        "ACTUAL",
    ),
    (
        "Khi bảng có cả actual và plan, doanh thu hoạt động plan năm 2023 là bao nhiêu?",
        "PLAN",
    ),
    (
        "Doanh thu hoạt động thực hiện năm 2023 là bao nhiêu khi bảng có cả kế hoạch và thực hiện?",
        "ACTUAL",
    ),
)


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
            raise ValueError("Catalog row uses document content or numeric answers")
    return rows


def _reason_case(question: str, expected: str) -> dict[str, Any]:
    plan = compile_question_plan(question)
    return {
        "question": question,
        "expected_reason": expected,
        "actual_reason": plan.abstention_reason,
        "reference_base": plan.reference_base,
        "dag_node_count": len(plan.program_dag),
        "exact": (
            plan.abstention_reason == expected
            and plan.reference_base is None
            and not plan.program_dag
        ),
    }


def _reference_repair_case(question: str) -> dict[str, Any]:
    plan = compile_question_plan(question)
    return {
        "question": question,
        "actual_reason": plan.abstention_reason,
        "reference_base": plan.reference_base,
        "dag": _actual_dag(plan),
        "exact": not plan.abstention_reason and plan.reference_base is not None and bool(plan.program_dag),
    }


def _role_repair_case(question: str, expected_role: str) -> dict[str, Any]:
    plan = compile_question_plan(question)
    return {
        "question": question,
        "expected_role": expected_role,
        "actual_reason": plan.abstention_reason,
        "operation": plan.operation,
        "column_roles": list(plan.column_roles),
        "slot_ids": [slot.id for slot in plan.operand_slots],
        "exact": (
            not plan.abstention_reason
            and plan.operation == "SELECT"
            and plan.column_roles == (expected_role,)
            and len(plan.operand_slots) == 1
            and plan.program_dag_operations == ("SELECT",)
        ),
    }


def build_report(
    contract_path: Path = DEFAULT_CONTRACT,
    catalog_path: Path = DEFAULT_CATALOG,
    before_path: Path = DEFAULT_BEFORE,
    after_path: Path = DEFAULT_AFTER,
) -> dict[str, Any]:
    contract = _read_json(contract_path)
    catalog = _read_catalog(catalog_path)
    before = _read_json(before_path)
    after = _read_json(after_path)
    if contract["frozen_source_artifacts"]["question_family_catalog"]["sha256"] != _digest(catalog_path):
        raise ValueError("Frozen catalog hash mismatch")

    canonical_controls = []
    for family in contract["families"]:
        expected = family["expected"]
        if expected["behavior"] != "ABSTAIN":
            continue
        canonical_controls.append(
            {
                "family_id": family["family_id"],
                **_reason_case(family["canonical_question"], expected["abstention_reason"]),
            }
        )
    missing_reference = [_reason_case(question, "missing_reference_base") for question in MISSING_REFERENCE_CASES]
    missing_role = [_reason_case(question, "missing_column_role") for question in MISSING_ROLE_CASES]
    reference_repairs = [_reference_repair_case(question) for question in EXPLICIT_REFERENCE_REPAIRS]
    role_repairs = [_role_repair_case(question, role) for question, role in EXPLICIT_ROLE_REPAIRS]

    answer_retention: list[dict[str, Any]] = []
    for row in catalog:
        if row["family_id"] not in ANSWER_FAMILIES or row["expected_ir"]["behavior"] != "ANSWER":
            continue
        plan = compile_question_plan(row["question"])
        expected_ref, expected_dag = _expected_catalog_program(row)
        actual_direction = plan.direction_intent.value if plan.direction_intent is not None else None
        checks = {
            "no_false_abstain": not plan.abstention_reason,
            "qp2a_slots": _catalog_expected_slots(row["expected_ir"]) == _catalog_actual_slots(plan),
            "qp2b_operation_direction": (
                plan.operation == row["expected_ir"].get("root_operation")
                and actual_direction == _expected_direction(row["expected_ir"])
            ),
            "qp2c_reference_dag": (
                plan.reference_base == expected_ref
                and _actual_dag(plan) == expected_dag
                and not plan.program_dag_validation_errors
            ),
        }
        answer_retention.append(
            {
                "example_id": row["example_id"],
                "family_id": row["family_id"],
                "variant": row["variant"],
                "checks": checks,
                "exact": all(checks.values()),
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
    runtime_consumers = {
        str(path.relative_to(ROOT)): "abstention_reason" in path.read_text(encoding="utf-8")
        for path in RUNTIME_CONSUMERS
    }

    canonical_exact = sum(item["exact"] for item in canonical_controls)
    missing_ref_exact = sum(item["exact"] for item in missing_reference)
    missing_role_exact = sum(item["exact"] for item in missing_role)
    reference_repairs_exact = sum(item["exact"] for item in reference_repairs)
    role_repairs_exact = sum(item["exact"] for item in role_repairs)
    retention_exact = sum(item["exact"] for item in answer_retention)
    pa07 = [item for item in answer_retention if item["family_id"] == "PA07"]
    qp2d_gate_passed = (
        canonical_exact == len(canonical_controls) == 3
        and missing_ref_exact == len(missing_reference) == 8
        and missing_role_exact == len(missing_role) == 6
        and reference_repairs_exact == len(reference_repairs) == 4
        and role_repairs_exact == len(role_repairs) == 3
        and retention_exact == len(answer_retention) == 108
        and sum(item["exact"] for item in pa07) == len(pa07) == 9
        and not regressions
        and not any(runtime_consumers.values())
        and after["strict_canonical_contract"]["exact"] == 16
        and after["strict_canonical_contract"]["dimension_metrics"]["behavior"]["pass"] == 16
        and after["strict_canonical_contract"]["dimension_metrics"]["abstention_reason"]["pass"] == 16
    )
    return {
        "schema": "finhrm-qp2d-safe-abstention-development-v1",
        "classification": "development_only_synthetic_compiler_audit",
        "evaluation_mode": "question_compiler_only_no_document_no_model_no_execution",
        "qp2d_gate_passed": qp2d_gate_passed,
        "canonical_ambiguity_controls": {
            "exact": canonical_exact,
            "total": len(canonical_controls),
            "details": canonical_controls,
        },
        "underspecification_surfaces": {
            "missing_reference_base_exact": missing_ref_exact,
            "missing_reference_base_total": len(missing_reference),
            "missing_column_role_exact": missing_role_exact,
            "missing_column_role_total": len(missing_role),
            "missing_reference_details": missing_reference,
            "missing_role_details": missing_role,
        },
        "false_abstain_controls": {
            "explicit_reference_repair_exact": reference_repairs_exact,
            "explicit_reference_repair_total": len(reference_repairs),
            "explicit_role_repair_exact": role_repairs_exact,
            "explicit_role_repair_total": len(role_repairs),
            "pa07_exact": sum(item["exact"] for item in pa07),
            "pa07_total": len(pa07),
            "reference_repair_details": reference_repairs,
            "role_repair_details": role_repairs,
        },
        "answer_contract_retention": {
            "exact": retention_exact,
            "total": len(answer_retention),
            "details": answer_retention,
        },
        "strict_contract_delta": {
            "exact_before": before["strict_canonical_contract"]["exact"],
            "exact_after": after["strict_canonical_contract"]["exact"],
            "total": after["strict_canonical_contract"]["examples"],
            "behavior_before": before["strict_canonical_contract"]["dimension_metrics"]["behavior"]["pass"],
            "behavior_after": after["strict_canonical_contract"]["dimension_metrics"]["behavior"]["pass"],
            "abstention_reason_before": before["strict_canonical_contract"]["dimension_metrics"]["abstention_reason"]["pass"],
            "abstention_reason_after": after["strict_canonical_contract"]["dimension_metrics"]["abstention_reason"]["pass"],
        },
        "legacy_collateral": {
            "exact_before": before["legacy_catalog_robustness"]["exact"],
            "exact_after": after["legacy_catalog_robustness"]["exact"],
            "total": after["legacy_catalog_robustness"]["examples"],
            "improvements": improvements,
            "regressions": regressions,
        },
        "runtime_consumer_scan": runtime_consumers,
        "artifact_hashes": {
            "contract": _digest(contract_path),
            "catalog": _digest(catalog_path),
            "before_report": _digest(before_path),
            "after_report": _digest(after_path),
            "question_planner": _digest(ROOT / "src" / "hrm_finance_qa" / "question_planner.py"),
            "program_synthesis": _digest(ROOT / "src" / "hrm_finance_qa" / "program_synthesis.py"),
            "symbolic_verifier": _digest(ROOT / "src" / "hrm_finance_qa" / "symbolic_verifier.py"),
            "program_data": _digest(ROOT / "src" / "hrm_finance_qa" / "program_data.py"),
            "audit_script": _digest(Path(__file__)),
        },
        "invariants": {
            "runtime_planning_source_changed": True,
            "abstention_reason_consumed_by_runtime": False,
            "compiler_dag_consumed_by_runtime": False,
            "program_identity_ids_consumed_by_runtime": False,
            "program_synthesis_source_changed": False,
            "symbolic_verifier_source_changed": False,
            "executor_source_changed": False,
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
        "authorized_next_step": (
            "QP2E surface robustness on the frozen 135-row compiler catalog; keep abstention_reason, "
            "compiler DAG and program_identity_ids non-consuming."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit QP2D compiler-side safe abstention")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--before", type=Path, default=DEFAULT_BEFORE)
    parser.add_argument("--after", type=Path, default=DEFAULT_AFTER)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = build_report(args.contract, args.catalog, args.before, args.after)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        key: value
        for key, value in report.items()
        if key not in {
            "canonical_ambiguity_controls", "underspecification_surfaces",
            "false_abstain_controls", "answer_contract_retention",
        }
    }
    for key in (
        "canonical_ambiguity_controls", "underspecification_surfaces",
        "false_abstain_controls", "answer_contract_retention",
    ):
        summary[key] = {name: value for name, value in report[key].items() if not name.endswith("details")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
