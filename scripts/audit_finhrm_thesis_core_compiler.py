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

from hrm_finance_qa.ontology import normalize_text  # noqa: E402
from hrm_finance_qa.question_planner import (  # noqa: E402
    DirectionIntent,
    compile_question_plan,
    compile_question_proof_contract,
)
from scripts.audit_question_plan_compiler import (  # noqa: E402
    _actual_ir as legacy_actual_ir,
    _checks as legacy_checks,
)


DEFAULT_CONTRACT = ROOT / "config" / "finhrm_thesis_core_v1.json"
DEFAULT_CATALOG = ROOT / "config" / "question_family_catalog_plan_actual_v1.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_thesis_core_compiler_baseline_v1.json"
PROTECTED_PATH_TOKENS = {"validation", "blind", "holdout", "test"}


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _refuse_protected_input(path: Path) -> None:
    lowered = {part.lower() for part in path.resolve().parts}
    overlap = sorted(lowered.intersection(PROTECTED_PATH_TOKENS))
    if overlap:
        raise ValueError(f"Refusing protected input path {path}: {overlap}")


def _read_json(path: Path) -> dict[str, Any]:
    _refuse_protected_input(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_catalog(path: Path) -> list[dict[str, Any]]:
    _refuse_protected_input(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            if value.get("classification") != "development_only_synthetic_compiler_contract":
                raise ValueError(f"{path}:{line_number}: non-development row")
            if value.get("document_content_used") is not False:
                raise ValueError(f"{path}:{line_number}: document-conditioned row is forbidden")
            if value.get("numeric_answer_used") is not False:
                raise ValueError(f"{path}:{line_number}: numeric answer is forbidden")
            rows.append(value)
    return rows


def _verify_frozen_hashes(contract: dict[str, Any], contract_path: Path, catalog_path: Path) -> None:
    if contract.get("schema") != "finhrm-thesis-core-precommit-v1":
        raise ValueError("Unexpected thesis-core contract schema")
    frozen = contract.get("frozen_source_artifacts", {})
    expected_catalog = frozen.get("question_family_catalog", {}).get("sha256")
    if expected_catalog != _digest(catalog_path):
        raise ValueError("Frozen question-family catalog hash mismatch")
    expected_contract_status = "precommitted_before_qp2_qp3_runtime_changes"
    if contract.get("status") != expected_contract_status:
        raise ValueError("Thesis-core contract is not in the precommitted state")
    if not contract_path.is_file():
        raise ValueError("Missing thesis-core contract")


def _value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _slot_value(slot: Any, name: str, default: str = "") -> str:
    if isinstance(slot, dict):
        return str(slot.get(name, default))
    return str(getattr(slot, name, default))


def _explicit_operand_slots(plan: Any) -> list[dict[str, str]]:
    """Read only compiler-emitted slots; never derive slots inside the audit."""

    raw_slots = getattr(plan, "operand_slots", ())
    slots: list[dict[str, str]] = []
    for slot in raw_slots:
        slots.append(
            {
                "id": _slot_value(slot, "id"),
                "metric": _slot_value(slot, "metric"),
                "year": _slot_value(slot, "year"),
                "column_role": _slot_value(slot, "column_role").upper(),
                "entity_scope": _slot_value(slot, "entity_scope", _slot_value(slot, "scope")),
                "unit_family": _slot_value(slot, "unit_family"),
            }
        )
    return slots


def _explicit_program_dag(plan: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for node in getattr(plan, "program_dag", ()):
        nodes.append(
            {
                "id": _slot_value(node, "id"),
                "operation": _slot_value(node, "operation"),
                "inputs": list(getattr(node, "inputs", ())),
                "output_unit_family": _slot_value(node, "output_unit_family"),
                "intent_sources": list(getattr(node, "intent_sources", ())),
            }
        )
    return nodes


def _actual_contract_ir(plan: Any, root_operation: str | None) -> dict[str, Any]:
    direction = getattr(plan, "direction_intent", None)
    abstention_reason = str(getattr(plan, "abstention_reason", ""))
    if not abstention_reason and direction is DirectionIntent.AMBIGUOUS:
        abstention_reason = "ambiguous_signed_direction"
    actual_behavior = "ABSTAIN" if abstention_reason else "ANSWER"
    operation = getattr(plan, "operation", None)
    proof = None
    if root_operation:
        proof_value = compile_question_proof_contract(
            plan,
            root_operation,
            financial_identity_required=bool(getattr(plan, "financial_identity_ids", ())),
        )
        proof = {
            "required": list(proof_value.required),
            "optional": list(proof_value.optional),
            "not_applicable": list(proof_value.not_applicable),
        }
    operand_slots = _explicit_operand_slots(plan)
    typed_column_roles = [slot["column_role"] for slot in operand_slots]
    legacy_column_roles = list(getattr(plan, "column_roles", ()))
    return {
        "behavior": actual_behavior,
        "abstention_reason": abstention_reason,
        "operation": operation,
        "operation_candidates": list(getattr(plan, "operation_candidates", ())),
        "requested_years": list(getattr(plan, "requested_years", ())),
        "column_roles": typed_column_roles or legacy_column_roles,
        "legacy_column_roles": legacy_column_roles,
        "operand_slots": operand_slots,
        "answer_mode": str(getattr(plan, "answer_mode", "SINGLE_SCALAR")),
        "direction_intent": _value(direction),
        "reference_base": getattr(plan, "reference_base", None),
        "requested_unit_family": str(getattr(plan, "requested_unit_family", "")),
        "entity_mode": str(getattr(plan, "entity_mode", "reporting")),
        "metric_text": str(getattr(plan, "metric_text", "")),
        "program_dag": _explicit_program_dag(plan),
        "program_dag_operations": list(getattr(plan, "program_dag_operations", ())),
        "financial_identity_ids": list(getattr(plan, "financial_identity_ids", ())),
        "program_identity_ids": list(getattr(plan, "program_identity_ids", ())),
        "interest_bearing_debt_share": bool(getattr(plan, "interest_bearing_debt_share", False)),
        "proof_contract": proof,
    }


def _expected_years(expected: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(str(slot["year"]) for slot in expected.get("operands", ())))


def _expected_dag_operations(expected: dict[str, Any]) -> list[str]:
    return [str(node["operation"]) for node in expected.get("dag", ())]


def _operation_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    root = expected.get("root_operation")
    allowed = {str(value) for value in actual.get("operation_candidates", ()) if value}
    if actual.get("operation"):
        allowed.add(str(actual["operation"]))
    return bool(root) and str(root) in allowed


def _normalize_slot(slot: dict[str, Any]) -> dict[str, str]:
    return {
        "id": str(slot.get("id", "")),
        "metric": normalize_text(str(slot.get("metric", ""))),
        "year": str(slot.get("year", "")),
        "column_role": str(slot.get("column_role", "")).upper(),
        "entity_scope": str(slot.get("entity_scope", slot.get("scope", ""))),
        "unit_family": str(slot.get("unit_family", "")),
    }


def _strict_checks(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, bool]:
    behavior = str(expected.get("behavior", "ANSWER"))
    checks: dict[str, bool] = {
        "behavior": behavior == actual.get("behavior"),
        "abstention_reason": (
            str(expected.get("abstention_reason", "")) == str(actual.get("abstention_reason", ""))
            if behavior == "ABSTAIN"
            else True
        ),
        "answer_mode": str(expected.get("answer_mode", "SINGLE_SCALAR"))
        == str(actual.get("answer_mode", "SINGLE_SCALAR")),
        "requested_unit_family": str(expected.get("requested_unit_family", ""))
        == str(actual.get("requested_unit_family", "")),
    }
    if behavior == "ABSTAIN":
        return checks

    expected_slots = [_normalize_slot(slot) for slot in expected.get("operands", ())]
    actual_slots = [_normalize_slot(slot) for slot in actual.get("operand_slots", ())]
    expected_roles = [slot["column_role"] for slot in expected_slots]
    checks.update(
        {
            "operation": _operation_match(expected, actual),
            "requested_years": _expected_years(expected) == list(actual.get("requested_years", ())),
            "column_roles": expected_roles == list(actual.get("column_roles", ())),
            "operand_schema_present": bool(actual_slots),
            "operand_count": len(expected_slots) == len(actual_slots),
            "operand_id_binding": [slot["id"] for slot in expected_slots]
            == [slot["id"] for slot in actual_slots],
            "operand_metric_binding": [slot["metric"] for slot in expected_slots]
            == [slot["metric"] for slot in actual_slots],
            "operand_year_binding": [slot["year"] for slot in expected_slots]
            == [slot["year"] for slot in actual_slots],
            "operand_column_role_binding": expected_roles
            == [slot["column_role"] for slot in actual_slots],
            "operand_entity_binding": [slot["entity_scope"] for slot in expected_slots]
            == [slot["entity_scope"] for slot in actual_slots],
            "operand_unit_binding": [slot["unit_family"] for slot in expected_slots]
            == [slot["unit_family"] for slot in actual_slots],
            "direction_intent": expected.get("direction_intent") == actual.get("direction_intent"),
            "reference_base": expected.get("reference_base") == actual.get("reference_base"),
            "program_dag": _expected_dag_operations(expected)
            == list(actual.get("program_dag_operations", ())),
        }
    )
    if expected.get("reference_base") is None:
        checks["reference_base"] = actual.get("reference_base") is None
    if len(expected.get("dag", ())) <= 1 and not actual.get("program_dag_operations"):
        # A single-node DAG is completely specified by an exact root operation.
        checks["program_dag"] = checks["operation"]
    if any(node.get("operation") == "ADD" for node in expected.get("dag", ())):
        checks["financial_identity"] = (
            "balance.financial_debt" in actual.get("program_identity_ids", ())
            and bool(actual.get("interest_bearing_debt_share"))
        )
    return checks


def _dimension_summary(details: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for detail in details:
        for name, passed in detail["checks"].items():
            counts[name]["total"] += 1
            counts[name]["pass"] += int(passed)
    return {
        name: {
            "total": values["total"],
            "pass": values["pass"],
            "fail": values["total"] - values["pass"],
            "pass_rate": values["pass"] / values["total"] if values["total"] else 0.0,
        }
        for name, values in sorted(counts.items())
    }


def _bucket_summary(details: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for detail in details:
        bucket = str(detail[key])
        counts[bucket]["total"] += 1
        counts[bucket]["exact"] += int(detail["exact"])
    return {
        name: {
            "total": values["total"],
            "exact": values["exact"],
            "exact_rate": values["exact"] / values["total"] if values["total"] else 0.0,
        }
        for name, values in sorted(counts.items())
    }


def _canonical_details(contract: dict[str, Any]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for family in contract.get("families", ()):
        expected = family["expected"]
        question = str(family["canonical_question"])
        plan = compile_question_plan(question)
        actual = _actual_contract_ir(plan, expected.get("root_operation"))
        checks = _strict_checks(expected, actual)
        failed = sorted(name for name, passed in checks.items() if not passed)
        details.append(
            {
                "example_id": f"{family['family_id']}-thesis-core-canonical",
                "family_id": family["family_id"],
                "category": family["category"],
                "question": question,
                "expected_ir": expected,
                "actual_ir": actual,
                "checks": checks,
                "exact": not failed,
                "failed_dimensions": failed,
            }
        )
    return details


def _legacy_details(rows: list[dict[str, Any]], core_ids: set[str]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for row in rows:
        family_id = str(row.get("family_id", ""))
        if family_id not in core_ids:
            continue
        expected = row.get("expected_ir")
        if not isinstance(expected, dict):
            raise ValueError(f"Missing expected_ir: {row.get('example_id')}")
        plan = compile_question_plan(str(row["question"]))
        actual = legacy_actual_ir(plan)
        checks = legacy_checks(expected, actual)
        failed = sorted(name for name, passed in checks.items() if not passed)
        details.append(
            {
                "example_id": row["example_id"],
                "family_id": family_id,
                "category": row["category"],
                "difficulty_level": row["difficulty_level"],
                "variant": row["variant"],
                "question": row["question"],
                "expected_ir": expected,
                "actual_ir": actual,
                "checks": checks,
                "exact": not failed,
                "failed_dimensions": failed,
            }
        )
    return details


def _gap_taxonomy(
    canonical: list[dict[str, Any]], legacy: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    definitions = [
        (
            "G1_TYPED_OPERAND_SCHEMA",
            {
                "operand_schema_present",
                "operand_count",
                "operand_id_binding",
                "operand_metric_binding",
                "operand_year_binding",
                "operand_column_role_binding",
                "operand_entity_binding",
                "operand_unit_binding",
            },
            "QP2A",
        ),
        ("G2_OPERATION_AND_DIRECTION", {"operation", "direction_intent"}, "QP2B"),
        ("G3_REFERENCE_BASE_AND_DAG", {"reference_base", "program_dag", "financial_identity"}, "QP2C"),
        ("G4_SAFE_ABSTENTION", {"behavior", "abstention_reason"}, "QP2D"),
        ("G5_SURFACE_ROBUSTNESS", set(), "QP2E"),
    ]
    output: list[dict[str, Any]] = []
    for gap_id, dimensions, repair_stage in definitions:
        source = legacy if gap_id == "G5_SURFACE_ROBUSTNESS" else canonical
        affected: list[dict[str, Any]] = []
        for detail in source:
            failed = set(detail["failed_dimensions"])
            relevant = failed if gap_id == "G5_SURFACE_ROBUSTNESS" else failed.intersection(dimensions)
            if relevant:
                affected.append(
                    {
                        "example_id": detail["example_id"],
                        "family_id": detail["family_id"],
                        "failed_dimensions": sorted(relevant),
                    }
                )
        output.append(
            {
                "gap_id": gap_id,
                "repair_stage": repair_stage,
                "source": "legacy_catalog_variants" if source is legacy else "strict_canonical_contract",
                "affected_examples": len(affected),
                "affected_families": sorted({item["family_id"] for item in affected}),
                "details": affected,
            }
        )
    return output


def build_report(contract_path: Path = DEFAULT_CONTRACT, catalog_path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    contract = _read_json(contract_path)
    rows = _read_catalog(catalog_path)
    _verify_frozen_hashes(contract, contract_path, catalog_path)
    core_ids = {str(value) for value in contract["scope"]["core_family_ids"]}
    canonical = _canonical_details(contract)
    legacy = _legacy_details(rows, core_ids)
    catalog_ids = {str(detail["family_id"]) for detail in legacy}
    if core_ids - catalog_ids != {"IBDS01"}:
        raise ValueError(f"Unexpected core/catalog coverage gap: {sorted(core_ids - catalog_ids)}")
    if len(canonical) != len(core_ids):
        raise ValueError("Canonical contract does not cover every core family exactly once")

    canonical_exact = sum(int(detail["exact"]) for detail in canonical)
    legacy_exact = sum(int(detail["exact"]) for detail in legacy)
    return {
        "schema": "finhrm-thesis-core-compiler-baseline-v1",
        "classification": "development_only_synthetic_compiler_audit",
        "evaluation_mode": "compiler_only_no_document_no_model_no_execution",
        "precommit_contract": str(contract_path.relative_to(ROOT)),
        "strict_canonical_contract": {
            "examples": len(canonical),
            "family_count": len({detail["family_id"] for detail in canonical}),
            "exact": canonical_exact,
            "exact_rate": canonical_exact / len(canonical) if canonical else 0.0,
            "dimension_metrics": _dimension_summary(canonical),
            "by_family": _bucket_summary(canonical, "family_id"),
            "details": canonical,
        },
        "legacy_catalog_robustness": {
            "note": "Historical QP0/QP2-compatible scoring; IBDS01 has no post-hoc variants and is excluded from this section.",
            "examples": len(legacy),
            "family_count": len({detail["family_id"] for detail in legacy}),
            "exact": legacy_exact,
            "exact_rate": legacy_exact / len(legacy) if legacy else 0.0,
            "dimension_metrics": _dimension_summary(legacy),
            "by_family": _bucket_summary(legacy, "family_id"),
            "by_variant": _bucket_summary(legacy, "variant"),
            "details": legacy,
        },
        "gap_taxonomy": _gap_taxonomy(canonical, legacy),
        "repair_order": [
            "QP2A_TYPED_OPERAND_SCHEMA",
            "QP2B_OPERATION_DIRECTION",
            "QP2C_REFERENCE_BASE_PROGRAM_DAG",
            "QP2D_SAFE_ABSTENTION",
            "QP2E_SURFACE_ROBUSTNESS",
            "QP3_PROOF_INTEGRATION_AFTER_COUNTERFACTUAL_PASS",
        ],
        "release_gate": {
            "strict_core_exact_required": 1.0,
            "strict_core_exact_observed": canonical_exact / len(canonical) if canonical else 0.0,
            "passed": canonical_exact == len(canonical),
            "runtime_change_authorized": False,
        },
        "artifact_hashes": {
            "contract": _digest(contract_path),
            "catalog": _digest(catalog_path),
            "question_planner": _digest(ROOT / "src" / "hrm_finance_qa" / "question_planner.py"),
            "financial_identity_catalog": _digest(ROOT / "config" / "financial_identities.yaml"),
            "audit_script": _digest(Path(__file__)),
        },
        "invariants": {
            "document_loaded": False,
            "pdf_loaded": False,
            "model_loaded": False,
            "checkpoint_loaded": False,
            "global_validation_loaded": False,
            "sealed_validation_loaded": False,
            "blind_test_loaded": False,
            "fine_tuning_performed": False,
            "threshold_changed": False,
            "runtime_changed": False,
            "phase_d_unlocked": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the frozen FinHRM thesis-core compiler contract")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = build_report(args.contract, args.catalog)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema": report["schema"],
        "strict_canonical_contract": {
            key: value
            for key, value in report["strict_canonical_contract"].items()
            if key not in {"details", "by_family"}
        },
        "legacy_catalog_robustness": {
            key: value
            for key, value in report["legacy_catalog_robustness"].items()
            if key not in {"details", "by_family", "by_variant"}
        },
        "gap_taxonomy": [
            {key: value for key, value in gap.items() if key != "details"}
            for gap in report["gap_taxonomy"]
        ],
        "release_gate": report["release_gate"],
        "invariants": report["invariants"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved thesis-core compiler baseline: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
