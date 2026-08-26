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
from hrm_finance_qa.question_planner import compile_question_plan  # noqa: E402


DEFAULT_CONTRACT = ROOT / "config" / "finhrm_thesis_core_v1.json"
DEFAULT_CATALOG = ROOT / "config" / "question_family_catalog_plan_actual_v1.jsonl"
DEFAULT_BEFORE = ROOT / "reports" / "finhrm_thesis_core_compiler_qp2d_v1.json"
DEFAULT_AFTER = ROOT / "reports" / "finhrm_thesis_core_compiler_qp2e_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp2e_surface_robustness_development_v1.json"
PROTECTED_PATH_TOKENS = {"validation", "blind", "holdout", "test"}
CORE_ANSWER_FAMILIES = {
    "PA01", "PA02", "PA03", "PA04", "PA05", "PA06", "PA07", "PA08", "PA09",
    "TY01", "TY02", "TY04",
}
DEFERRED_VARIANT = "hard_negative_missing_role"
RUNTIME_CONSUMERS = (
    ROOT / "src" / "hrm_finance_qa" / "program_synthesis.py",
    ROOT / "src" / "hrm_finance_qa" / "symbolic_verifier.py",
    ROOT / "src" / "hrm_finance_qa" / "document_hrm.py",
    ROOT / "src" / "hrm_finance_qa" / "document_reasoning_engine.py",
    ROOT / "src" / "hrm_finance_qa" / "program_data.py",
)
GENERALIZATION_CASES = (
    {
        "question": "Lợi nhuận thực hiện năm 2028 còn thiếu bao nhiêu phần trăm để đạt kế hoạch năm 2028?",
        "reason": "", "years": ("2028",), "program_roles": ("PLAN", "ACTUAL", "PLAN"),
    },
    {
        "question": "Chi phí kế hoạch năm 2027 vượt thực hiện năm 2027 bao nhiêu phần trăm?",
        "reason": "", "years": ("2027",), "program_roles": ("PLAN", "ACTUAL", "ACTUAL"),
    },
    {
        "question": "Tài sản thực hiện năm 2029 tăng bao nhiêu phần trăm so với thực hiện năm 2027?",
        "reason": "", "years": ("2029", "2027"), "program_roles": ("ACTUAL", "ACTUAL"),
    },
    {
        "question": "Lợi nhuận năm 2028 là bao nhiêu khi bảng có cả kế hoạch và thực hiện?",
        "reason": "missing_column_role", "years": ("2028",), "program_roles": (),
    },
    {
        "question": "Chỉ tiêu tài chính năm 2028 là bao nhiêu khi bảng có nhiều cột cùng năm nhưng câu hỏi không xác định vai trò cột?",
        "reason": "ambiguous_metric_or_column_role", "years": ("2028",), "program_roles": (),
    },
    {
        "question": "Lợi nhuận năm 2028 cao hơn bao nhiêu phần trăm?",
        "reason": "missing_reference_base", "years": ("2028",), "program_roles": (),
    },
    {
        "question": "Chi phí năm 2028 tăng bao nhiêu phần trăm so với năm 2026?",
        "reason": "", "years": ("2028", "2026"), "program_roles": (),
    },
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


def _failed(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in details if not item["exact"]]


def _counter(values: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(item[key]) for item in values).items()))


def _dimension_counter(values: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for item in values:
        counter.update(str(value) for value in item["failed_dimensions"])
    return dict(sorted(counter.items()))


def _behavior_collisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[normalize_text(str(row["question"]))].append(row)
    collisions: list[dict[str, Any]] = []
    for normalized, items in sorted(grouped.items()):
        behaviors = {str(item["expected_ir"]["behavior"]) for item in items}
        if len(behaviors) < 2:
            continue
        collisions.append(
            {
                "normalized_question": normalized,
                "behaviors": sorted(behaviors),
                "examples": [
                    {
                        "example_id": item["example_id"],
                        "behavior": item["expected_ir"]["behavior"],
                        "abstention_reason": item["expected_ir"].get("abstention_reason", ""),
                    }
                    for item in items
                ],
            }
        )
    return collisions


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

    before_details = before["legacy_catalog_robustness"]["details"]
    after_details = after["legacy_catalog_robustness"]["details"]
    before_by_id = {item["example_id"]: item for item in before_details}
    after_by_id = {item["example_id"]: item for item in after_details}
    if set(before_by_id) != set(after_by_id):
        raise ValueError("Before/after reports do not cover the same frozen rows")
    before_failures = _failed(before_details)
    after_failures = _failed(after_details)
    improvements = sorted(
        example_id for example_id in before_by_id
        if not before_by_id[example_id]["exact"] and after_by_id[example_id]["exact"]
    )
    regressions = sorted(
        example_id for example_id in before_by_id
        if before_by_id[example_id]["exact"] and not after_by_id[example_id]["exact"]
    )

    deferred = [item for item in after_failures if item["variant"] == DEFERRED_VARIANT]
    unsafe_residual = [item for item in after_failures if item["variant"] != DEFERRED_VARIANT]
    identifiable = [item for item in after_details if item["variant"] != DEFERRED_VARIANT]
    identifiable_exact = sum(bool(item["exact"]) for item in identifiable)
    answer_rows = [
        item for item in after_details
        if item["family_id"] in CORE_ANSWER_FAMILIES
        and item["expected_ir"]["behavior"] == "ANSWER"
    ]
    ambiguity_rows = [item for item in after_details if item["variant"] == "ambiguity_control"]
    collisions = _behavior_collisions(catalog)
    collision_ids = {
        example["example_id"]
        for collision in collisions
        for example in collision["examples"]
    }
    explicit_collision_preserved = (
        "A05-canonical" in collision_ids
        and "TY02-hard_negative_missing_role" in collision_ids
        and not compile_question_plan(
            "Doanh thu năm 2023 tăng bao nhiêu phần trăm so với năm 2022?"
        ).abstention_reason
    )
    runtime_consumer_scan = {
        str(path.relative_to(ROOT)): {
            "program_column_roles": "program_column_roles" in path.read_text(encoding="utf-8"),
            "abstention_reason": "abstention_reason" in path.read_text(encoding="utf-8"),
        }
        for path in RUNTIME_CONSUMERS
    }
    new_field_consumed = any(
        any(flags.values()) for flags in runtime_consumer_scan.values()
    )
    generalization: list[dict[str, Any]] = []
    for expected in GENERALIZATION_CASES:
        plan = compile_question_plan(expected["question"])
        checks = {
            "reason": plan.abstention_reason == expected["reason"],
            "years": plan.requested_years == expected["years"],
            "program_roles": plan.program_column_roles == expected["program_roles"],
        }
        generalization.append(
            {
                "question": expected["question"],
                "expected": expected,
                "actual": {
                    "reason": plan.abstention_reason,
                    "years": plan.requested_years,
                    "program_roles": plan.program_column_roles,
                },
                "checks": checks,
                "exact": all(checks.values()),
            }
        )
    generalization_exact = sum(item["exact"] for item in generalization)

    safe_gate = (
        len(before_failures) == 43
        and len(improvements) == 31
        and len(after_failures) == len(deferred) == 12
        and not unsafe_residual
        and not regressions
        and identifiable_exact == len(identifiable) == 123
        and sum(bool(item["exact"]) for item in answer_rows) == len(answer_rows) == 108
        and sum(bool(item["exact"]) for item in ambiguity_rows) == len(ambiguity_rows) == 12
        and after["legacy_catalog_robustness"]["dimension_metrics"]["column_roles"]["pass"] == 135
        and after["strict_canonical_contract"]["exact"] == 16
        and bool(collisions)
        and explicit_collision_preserved
        and generalization_exact == len(generalization) == 7
        and not new_field_consumed
    )
    return {
        "schema": "finhrm-qp2e-surface-robustness-development-v1",
        "classification": "development_only_synthetic_compiler_audit",
        "evaluation_mode": "question_compiler_only_no_document_no_model_no_execution",
        "qp2e_safe_gate_passed": safe_gate,
        "raw_catalog": {
            "exact_before": before["legacy_catalog_robustness"]["exact"],
            "exact_after": after["legacy_catalog_robustness"]["exact"],
            "total": after["legacy_catalog_robustness"]["examples"],
            "exact_rate_after": after["legacy_catalog_robustness"]["exact_rate"],
            "improvements": improvements,
            "regressions": regressions,
        },
        "compiler_identifiable_subset": {
            "exact": identifiable_exact,
            "total": len(identifiable),
            "excluded_evidence_conditioned_rows": len(deferred),
        },
        "failure_partition_before": {
            "total": len(before_failures),
            "by_variant": _counter(before_failures, "variant"),
            "by_family": _counter(before_failures, "family_id"),
            "by_dimension": _dimension_counter(before_failures),
        },
        "failure_partition_after": {
            "total": len(after_failures),
            "by_variant": _counter(after_failures, "variant"),
            "by_family": _counter(after_failures, "family_id"),
            "by_dimension": _dimension_counter(after_failures),
            "unsafe_compiler_residuals": [item["example_id"] for item in unsafe_residual],
            "evidence_conditioned_deferred": [item["example_id"] for item in deferred],
        },
        "retention": {
            "answer_variants_exact": sum(bool(item["exact"]) for item in answer_rows),
            "answer_variants_total": len(answer_rows),
            "ambiguity_controls_exact": sum(bool(item["exact"]) for item in ambiguity_rows),
            "ambiguity_controls_total": len(ambiguity_rows),
            "strict_canonical_exact": after["strict_canonical_contract"]["exact"],
            "strict_canonical_total": after["strict_canonical_contract"]["examples"],
            "column_role_use_site_exact": after["legacy_catalog_robustness"]["dimension_metrics"]["column_roles"]["pass"],
            "column_role_use_site_total": after["legacy_catalog_robustness"]["dimension_metrics"]["column_roles"]["total"],
        },
        "metric_year_generalization": {
            "exact": generalization_exact,
            "total": len(generalization),
            "details": generalization,
        },
        "catalog_non_identifiability": {
            "behavior_collisions": collisions,
            "explicit_collision_preserved": explicit_collision_preserved,
            "decision": (
                "Do not force question-only abstention for hard_negative_missing_role. "
                "Require evidence showing multiple incompatible column roles in QP3."
            ),
        },
        "runtime_consumer_scan": runtime_consumer_scan,
        "artifact_hashes": {
            "contract": _digest(contract_path),
            "catalog": _digest(catalog_path),
            "before_report": _digest(before_path),
            "after_report": _digest(after_path),
            "question_planner": _digest(ROOT / "src" / "hrm_finance_qa" / "question_planner.py"),
            "question_plan_audit": _digest(ROOT / "scripts" / "audit_question_plan_compiler.py"),
            "program_synthesis": _digest(ROOT / "src" / "hrm_finance_qa" / "program_synthesis.py"),
            "symbolic_verifier": _digest(ROOT / "src" / "hrm_finance_qa" / "symbolic_verifier.py"),
            "program_data": _digest(ROOT / "src" / "hrm_finance_qa" / "program_data.py"),
            "audit_script": _digest(Path(__file__)),
        },
        "invariants": {
            "runtime_planning_source_changed": True,
            "program_column_roles_consumed_by_runtime": False,
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
            "QP3A precommit and development-only evidence-conditioned column-role proof integration "
            "for the 12 deferred hard negatives; do not change threshold."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit QP2E frozen surface robustness")
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
        key: value for key, value in report.items()
        if key not in {"raw_catalog", "failure_partition_after", "catalog_non_identifiability"}
    }
    summary["raw_catalog"] = {
        key: value for key, value in report["raw_catalog"].items()
        if key not in {"improvements", "regressions"}
    } | {
        "improvement_count": len(report["raw_catalog"]["improvements"]),
        "regression_count": len(report["raw_catalog"]["regressions"]),
    }
    summary["failure_partition_after"] = report["failure_partition_after"]
    summary["catalog_non_identifiability"] = {
        "behavior_collision_count": len(report["catalog_non_identifiability"]["behavior_collisions"]),
        "explicit_collision_preserved": report["catalog_non_identifiability"]["explicit_collision_preserved"],
        "decision": report["catalog_non_identifiability"]["decision"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
