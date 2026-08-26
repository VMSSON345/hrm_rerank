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

from hrm_finance_qa.ontology import normalize_text  # noqa: E402
from hrm_finance_qa.question_planner import compile_question_plan  # noqa: E402
from scripts.audit_qp2a_operand_slots import (  # noqa: E402
    _catalog_actual_slots,
    _catalog_expected_slots,
)
from scripts.audit_qp2b_operation_direction import _expected_direction  # noqa: E402


DEFAULT_CONTRACT = ROOT / "config" / "finhrm_thesis_core_v1.json"
DEFAULT_CATALOG = ROOT / "config" / "question_family_catalog_plan_actual_v1.jsonl"
DEFAULT_BEFORE = ROOT / "reports" / "finhrm_thesis_core_compiler_qp2b_v1.json"
DEFAULT_AFTER = ROOT / "reports" / "finhrm_thesis_core_compiler_qp2c_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp2c_reference_dag_development_v1.json"
PROTECTED_PATH_TOKENS = {"validation", "blind", "holdout", "test"}
FORBIDDEN_PROVENANCE_TOKENS = {"value", "candidate", "page", "table", "left", "right"}
TARGET_FAMILIES = {"PA03", "PA07", "PA08", "PA09", "TY02", "TY04", "IBDS01"}


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


def _actual_dag(plan: Any) -> list[tuple[str, str, tuple[str, ...]]]:
    return [(node.id, node.operation, node.inputs) for node in plan.program_dag]


def _catalog_slot_ids(expected: dict[str, Any]) -> list[str]:
    unique_slots = _catalog_expected_slots(expected)
    multi_year = len({slot[1] for slot in unique_slots}) > 1
    return [
        f"{slot[2].lower()}_{slot[1]}" if multi_year else slot[2].lower()
        for slot in unique_slots
    ]


def _expected_catalog_program(
    row: dict[str, Any],
) -> tuple[str | None, list[tuple[str, str, tuple[str, ...]]]]:
    expected = row["expected_ir"]
    operations = list(expected.get("program_dag_operations", ()))
    slot_ids = _catalog_slot_ids(expected)
    if not operations or not slot_ids:
        return None, []
    if len(operations) == 1:
        operation = operations[0]
        reference_base = (
            slot_ids[1]
            if operation in {"DIVIDE", "RATIO_PERCENT", "PERCENT_CHANGE"}
            else None
        )
        return reference_base, [("answer", operation, tuple(slot_ids))]
    if operations == ["SUBTRACT", "RATIO_PERCENT"] and len(slot_ids) == 2:
        normalized = normalize_text(row["question"])
        shortfall = any(marker in normalized for marker in ("con thieu", "chua dat"))
        intermediate = "shortfall" if shortfall else "excess"
        reference_base = slot_ids[0] if shortfall else slot_ids[1]
        return reference_base, [
            (intermediate, "SUBTRACT", tuple(slot_ids)),
            ("answer", "RATIO_PERCENT", (intermediate, reference_base)),
        ]
    raise ValueError(f"Unsupported expected catalog DAG: {row['example_id']}")


def _canonical_details(contract: dict[str, Any]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for family in contract["families"]:
        expected = family["expected"]
        if expected["behavior"] != "ANSWER":
            continue
        plan = compile_question_plan(family["canonical_question"])
        expected_dag = [
            (node["id"], node["operation"], tuple(node["inputs"]))
            for node in expected["dag"]
        ]
        actual_dag = _actual_dag(plan)
        identity_exact = True
        if family["family_id"] == "IBDS01":
            identity_exact = (
                plan.program_identity_ids == ("balance.financial_debt",)
                and plan.financial_identity_ids == ()
            )
        errors = list(plan.program_dag_validation_errors)
        details.append(
            {
                "family_id": family["family_id"],
                "expected_reference_base": expected["reference_base"],
                "actual_reference_base": plan.reference_base,
                "expected_dag": expected_dag,
                "actual_dag": actual_dag,
                "validation_errors": errors,
                "identity_exact": identity_exact,
                "exact": (
                    plan.reference_base == expected["reference_base"]
                    and actual_dag == expected_dag
                    and not errors
                    and identity_exact
                ),
            }
        )
    return details


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
    core_ids = set(contract["scope"]["core_family_ids"])
    canonical = _canonical_details(contract)
    canonical_exact = sum(item["exact"] for item in canonical)

    variants: list[dict[str, Any]] = []
    abstain_controls: list[dict[str, Any]] = []
    provenance_violations: list[dict[str, Any]] = []
    qp2a_slot_exact = 0
    qp2b_exact = 0
    for row in catalog:
        if row["family_id"] not in core_ids:
            continue
        plan = compile_question_plan(row["question"])
        if row["expected_ir"]["behavior"] == "ANSWER":
            expected_ref, expected_dag = _expected_catalog_program(row)
            actual_dag = _actual_dag(plan)
            errors = list(plan.program_dag_validation_errors)
            qp2a_slot_exact += (
                _catalog_expected_slots(row["expected_ir"])
                == _catalog_actual_slots(plan)
            )
            actual_direction = (
                plan.direction_intent.value if plan.direction_intent is not None else None
            )
            qp2b_exact += (
                plan.operation == row["expected_ir"].get("root_operation")
                and actual_direction == _expected_direction(row["expected_ir"])
            )
            for node in plan.program_dag:
                rendered = " ".join(node.intent_sources).lower()
                forbidden = sorted(
                    token for token in FORBIDDEN_PROVENANCE_TOKENS if token in rendered
                )
                if not node.intent_sources or forbidden:
                    provenance_violations.append(
                        {"example_id": row["example_id"], "node_id": node.id, "forbidden": forbidden}
                    )
            variants.append(
                {
                    "example_id": row["example_id"],
                    "family_id": row["family_id"],
                    "variant": row["variant"],
                    "expected_reference_base": expected_ref,
                    "actual_reference_base": plan.reference_base,
                    "expected_dag": expected_dag,
                    "actual_dag": actual_dag,
                    "validation_errors": errors,
                    "exact": plan.reference_base == expected_ref and actual_dag == expected_dag and not errors,
                }
            )
        else:
            abstain_controls.append(
                {
                    "example_id": row["example_id"],
                    "family_id": row["family_id"],
                    "variant": row["variant"],
                    "reference_base": plan.reference_base,
                    "dag_node_count": len(plan.program_dag),
                    "passed": plan.reference_base is None and not plan.program_dag,
                }
            )

    variant_exact = sum(item["exact"] for item in variants)
    abstain_pass = sum(item["passed"] for item in abstain_controls)
    role_swap = [item for item in variants if item["variant"] == "minimal_pair_role_swap"]
    year_shift = [item for item in variants if item["variant"] == "minimal_pair_year_shift"]
    multi_step = [item for item in variants if item["family_id"] in {"PA08", "PA09"}]
    before_by_id = {
        item["example_id"]: bool(item["exact"])
        for item in before["legacy_catalog_robustness"]["details"]
    }
    after_by_id = {
        item["example_id"]: bool(item["exact"])
        for item in after["legacy_catalog_robustness"]["details"]
    }
    shared = sorted(set(before_by_id).intersection(after_by_id))
    regressions = [item for item in shared if before_by_id[item] and not after_by_id[item]]
    improvements = [item for item in shared if not before_by_id[item] and after_by_id[item]]
    qp2c_passed = (
        canonical_exact == len(canonical) == 13
        and variant_exact == len(variants) == 108
        and abstain_pass == len(abstain_controls) == 27
        and sum(item["exact"] for item in role_swap) == len(role_swap) == 12
        and sum(item["exact"] for item in year_shift) == len(year_shift) == 12
        and sum(item["exact"] for item in multi_step) == len(multi_step) == 18
        and qp2a_slot_exact == 108
        and qp2b_exact == 108
        and not provenance_violations
        and not regressions
        and after["strict_canonical_contract"]["dimension_metrics"]["reference_base"]["pass"] == 13
        and after["strict_canonical_contract"]["dimension_metrics"]["program_dag"]["pass"] == 13
    )
    return {
        "schema": "finhrm-qp2c-reference-dag-development-v1",
        "classification": "development_only_synthetic_compiler_audit",
        "evaluation_mode": "question_compiler_only_no_document_no_model_no_execution",
        "target_families": sorted(TARGET_FAMILIES),
        "qp2c_gate_passed": qp2c_passed,
        "canonical_answer_contract": {
            "exact": canonical_exact,
            "total": len(canonical),
            "details": canonical,
        },
        "frozen_answer_variants": {
            "exact": variant_exact,
            "total": len(variants),
            "details": variants,
        },
        "abstain_no_dag_controls": {
            "pass": abstain_pass,
            "total": len(abstain_controls),
            "details": abstain_controls,
        },
        "counterfactuals": {
            "role_swap_exact": sum(item["exact"] for item in role_swap),
            "role_swap_total": len(role_swap),
            "year_shift_exact": sum(item["exact"] for item in year_shift),
            "year_shift_total": len(year_shift),
        },
        "multi_step_variants": {
            "pa08_pa09_exact": sum(item["exact"] for item in multi_step),
            "pa08_pa09_total": len(multi_step),
            "ibds01_canonical_exact": next(item["exact"] for item in canonical if item["family_id"] == "IBDS01"),
        },
        "retention": {
            "qp2a_slot_exact": qp2a_slot_exact,
            "qp2a_slot_total": 108,
            "qp2b_operation_direction_exact": qp2b_exact,
            "qp2b_operation_direction_total": 108,
        },
        "dag_integrity": {
            "validation_error_examples": sum(bool(item["validation_errors"]) for item in variants),
            "provenance_violations": provenance_violations,
        },
        "strict_contract_delta": {
            "exact_before": before["strict_canonical_contract"]["exact"],
            "exact_after": after["strict_canonical_contract"]["exact"],
            "total": after["strict_canonical_contract"]["examples"],
            "reference_base_before": before["strict_canonical_contract"]["dimension_metrics"]["reference_base"]["pass"],
            "reference_base_after": after["strict_canonical_contract"]["dimension_metrics"]["reference_base"]["pass"],
            "program_dag_before": before["strict_canonical_contract"]["dimension_metrics"]["program_dag"]["pass"],
            "program_dag_after": after["strict_canonical_contract"]["dimension_metrics"]["program_dag"]["pass"],
            "dimension_total": 13,
        },
        "legacy_collateral": {
            "exact_before": before["legacy_catalog_robustness"]["exact"],
            "exact_after": after["legacy_catalog_robustness"]["exact"],
            "total": after["legacy_catalog_robustness"]["examples"],
            "improvements": improvements,
            "regressions": regressions,
        },
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
            "compiler_dag_consumed_by_runtime": False,
            "program_identity_ids_consumed_by_runtime": False,
            "legacy_financial_identity_ids_changed_for_ibds01": False,
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
        "authorized_next_step": "QP2D safe abstention for C14 and C15; keep compiler DAG and program_identity_ids non-consuming.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit QP2C reference base and compiler DAG")
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
        if key not in {"canonical_answer_contract", "frozen_answer_variants", "abstain_no_dag_controls"}
    }
    for key in ("canonical_answer_contract", "frozen_answer_variants", "abstain_no_dag_controls"):
        summary[key] = {name: value for name, value in report[key].items() if name != "details"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
