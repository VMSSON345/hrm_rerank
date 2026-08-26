from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.ontology import normalize_text  # noqa: E402
from hrm_finance_qa.question_planner import compile_question_plan  # noqa: E402


DEFAULT_CONTRACT = ROOT / "config" / "finhrm_thesis_core_v1.json"
DEFAULT_CATALOG = ROOT / "config" / "question_family_catalog_plan_actual_v1.jsonl"
DEFAULT_BASELINE = ROOT / "reports" / "finhrm_thesis_core_compiler_baseline_v1.json"
DEFAULT_COMPILER_REPORT = ROOT / "reports" / "finhrm_thesis_core_compiler_qp2a_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp2a_operand_slots_development_v1.json"
PROTECTED_PATH_TOKENS = {"validation", "blind", "holdout", "test"}
FORBIDDEN_PROVENANCE_TOKENS = {"value", "candidate", "page", "table", "left", "right"}


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if {part.lower() for part in path.resolve().parts}.intersection(PROTECTED_PATH_TOKENS):
        raise ValueError(f"Refusing protected path: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def _read_catalog(path: Path) -> list[dict[str, Any]]:
    if {part.lower() for part in path.resolve().parts}.intersection(PROTECTED_PATH_TOKENS):
        raise ValueError(f"Refusing protected path: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("classification") != "development_only_synthetic_compiler_contract":
            raise ValueError("Catalog contains a non-development row")
        if row.get("document_content_used") is not False or row.get("numeric_answer_used") is not False:
            raise ValueError("QP2A catalog must not use documents or numeric answers")
    return rows


def _unique(items: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    output: list[tuple[str, ...]] = []
    for item in items:
        if item not in output:
            output.append(item)
    return output


def _catalog_expected_slots(expected: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    return _unique(
        [
            (
                normalize_text(str(slot.get("metric", ""))),
                str(slot.get("year", "")),
                str(slot.get("column_role", "")),
                str(slot.get("scope", "")),
            )
            for slot in expected.get("operand_slots", ())
        ]
    )


def _catalog_actual_slots(plan: Any) -> list[tuple[str, str, str, str]]:
    return [
        (normalize_text(slot.metric), slot.year, slot.column_role, slot.entity_scope)
        for slot in plan.operand_slots
    ]


def _canonical_expected_slots(expected: dict[str, Any]) -> list[tuple[str, str, str, str, str, str]]:
    return [
        (
            str(slot.get("id", "")),
            normalize_text(str(slot.get("metric", ""))),
            str(slot.get("year", "")),
            str(slot.get("column_role", "")),
            str(slot.get("entity_scope", "")),
            str(slot.get("unit_family", "")),
        )
        for slot in expected.get("operands", ())
    ]


def _canonical_actual_slots(plan: Any) -> list[tuple[str, str, str, str, str, str]]:
    return [
        (
            slot.id,
            normalize_text(slot.metric),
            slot.year,
            slot.column_role,
            slot.entity_scope,
            slot.unit_family,
        )
        for slot in plan.operand_slots
    ]


def build_report(
    contract_path: Path = DEFAULT_CONTRACT,
    catalog_path: Path = DEFAULT_CATALOG,
    baseline_path: Path = DEFAULT_BASELINE,
    compiler_report_path: Path = DEFAULT_COMPILER_REPORT,
) -> dict[str, Any]:
    contract = _read_json(contract_path)
    catalog = _read_catalog(catalog_path)
    baseline = _read_json(baseline_path)
    compiler = _read_json(compiler_report_path)
    frozen = contract["frozen_source_artifacts"]["question_family_catalog"]
    if frozen["sha256"] != _digest(catalog_path):
        raise ValueError("Frozen catalog hash mismatch")
    core_ids = set(contract["scope"]["core_family_ids"])
    answer_ids = set(contract["scope"]["answer_family_ids"])

    canonical_details: list[dict[str, Any]] = []
    provenance_violations: list[dict[str, Any]] = []
    duplicate_id_families: list[str] = []
    for family in contract["families"]:
        if family["family_id"] not in answer_ids:
            continue
        plan = compile_question_plan(family["canonical_question"])
        expected = _canonical_expected_slots(family["expected"])
        actual = _canonical_actual_slots(plan)
        ids = [slot.id for slot in plan.operand_slots]
        if len(ids) != len(set(ids)):
            duplicate_id_families.append(family["family_id"])
        for slot in plan.operand_slots:
            rendered = " ".join(slot.intent_sources).lower()
            forbidden = sorted(token for token in FORBIDDEN_PROVENANCE_TOKENS if token in rendered)
            if not slot.intent_sources or forbidden:
                provenance_violations.append(
                    {"family_id": family["family_id"], "slot_id": slot.id, "forbidden": forbidden}
                )
        canonical_details.append(
            {
                "family_id": family["family_id"],
                "expected_slots": expected,
                "actual_slots": actual,
                "exact": expected == actual,
            }
        )

    variant_details: list[dict[str, Any]] = []
    missing_role_details: list[dict[str, Any]] = []
    for row in catalog:
        if row["family_id"] not in core_ids:
            continue
        plan = compile_question_plan(row["question"])
        expected_ir = row["expected_ir"]
        if expected_ir["behavior"] == "ANSWER":
            expected = _catalog_expected_slots(expected_ir)
            actual = _catalog_actual_slots(plan)
            variant_details.append(
                {
                    "example_id": row["example_id"],
                    "family_id": row["family_id"],
                    "variant": row["variant"],
                    "expected_slots": expected,
                    "actual_slots": actual,
                    "exact": expected == actual,
                }
            )
        if row["variant"] == "hard_negative_missing_role":
            missing_role_details.append(
                {
                    "example_id": row["example_id"],
                    "family_id": row["family_id"],
                    "slot_count": len(plan.operand_slots),
                    "passed": not plan.operand_slots,
                }
            )

    baseline_by_id = {
        detail["example_id"]: bool(detail["exact"])
        for detail in baseline["legacy_catalog_robustness"]["details"]
    }
    current_by_id = {
        detail["example_id"]: bool(detail["exact"])
        for detail in compiler["legacy_catalog_robustness"]["details"]
    }
    shared = sorted(set(baseline_by_id).intersection(current_by_id))
    improvements = [key for key in shared if not baseline_by_id[key] and current_by_id[key]]
    regressions = [key for key in shared if baseline_by_id[key] and not current_by_id[key]]
    canonical_exact = sum(item["exact"] for item in canonical_details)
    variant_exact = sum(item["exact"] for item in variant_details)
    missing_role_pass = sum(item["passed"] for item in missing_role_details)
    qp2a_passed = (
        canonical_exact == len(canonical_details) == 13
        and variant_exact == len(variant_details) == 108
        and missing_role_pass == len(missing_role_details) == 12
        and not provenance_violations
        and not duplicate_id_families
        and not regressions
    )
    return {
        "schema": "finhrm-qp2a-operand-slots-development-v1",
        "classification": "development_only_synthetic_compiler_audit",
        "evaluation_mode": "question_compiler_only_no_document_no_model_no_execution",
        "qp2a_gate_passed": qp2a_passed,
        "canonical_answer_slots": {
            "exact": canonical_exact,
            "total": len(canonical_details),
            "exact_rate": canonical_exact / len(canonical_details),
            "details": canonical_details,
        },
        "frozen_answer_variant_slots": {
            "exact": variant_exact,
            "total": len(variant_details),
            "exact_rate": variant_exact / len(variant_details),
            "details": variant_details,
        },
        "missing_role_non_materialization": {
            "pass": missing_role_pass,
            "total": len(missing_role_details),
            "pass_rate": missing_role_pass / len(missing_role_details),
            "details": missing_role_details,
        },
        "slot_integrity": {
            "duplicate_id_families": duplicate_id_families,
            "provenance_violations": provenance_violations,
            "numeric_or_document_selection_signal_count": len(provenance_violations),
        },
        "strict_contract_delta": {
            "before_exact": baseline["strict_canonical_contract"]["exact"],
            "after_exact": compiler["strict_canonical_contract"]["exact"],
            "total": compiler["strict_canonical_contract"]["examples"],
            "typed_operand_before": baseline["strict_canonical_contract"]["dimension_metrics"]["operand_schema_present"]["pass"],
            "typed_operand_after": compiler["strict_canonical_contract"]["dimension_metrics"]["operand_schema_present"]["pass"],
            "typed_operand_total": compiler["strict_canonical_contract"]["dimension_metrics"]["operand_schema_present"]["total"],
        },
        "legacy_collateral": {
            "before_exact": baseline["legacy_catalog_robustness"]["exact"],
            "after_exact": compiler["legacy_catalog_robustness"]["exact"],
            "total": compiler["legacy_catalog_robustness"]["examples"],
            "improvements": improvements,
            "regressions": regressions,
        },
        "artifact_hashes": {
            "contract": _digest(contract_path),
            "catalog": _digest(catalog_path),
            "baseline": _digest(baseline_path),
            "compiler_report": _digest(compiler_report_path),
            "question_planner": _digest(ROOT / "src" / "hrm_finance_qa" / "question_planner.py"),
            "audit_script": _digest(Path(__file__)),
        },
        "invariants": {
            "runtime_source_changed": True,
            "new_slot_fields_consumed_by_synthesis": False,
            "new_slot_fields_consumed_by_verifier": False,
            "runtime_acceptance_changed": False,
            "threshold_changed": False,
            "fine_tuning_performed": False,
            "document_loaded": False,
            "pdf_loaded": False,
            "model_loaded": False,
            "global_validation_loaded": False,
            "sealed_validation_loaded": False,
            "blind_test_loaded": False,
            "phase_d_unlocked": False,
        },
        "authorized_next_step": "QP2B operation and direction intent repair; keep QP2A slots non-consuming until later gates pass.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit QP2A typed operand slots")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--compiler-report", type=Path, default=DEFAULT_COMPILER_REPORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = build_report(args.contract, args.catalog, args.baseline, args.compiler_report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key not in {
        "canonical_answer_slots", "frozen_answer_variant_slots", "missing_role_non_materialization"
    }}
    summary["canonical_answer_slots"] = {
        key: value for key, value in report["canonical_answer_slots"].items() if key != "details"
    }
    summary["frozen_answer_variant_slots"] = {
        key: value for key, value in report["frozen_answer_variant_slots"].items() if key != "details"
    }
    summary["missing_role_non_materialization"] = {
        key: value for key, value in report["missing_role_non_materialization"].items() if key != "details"
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
