from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.question_planner import compile_question_plan  # noqa: E402


PROTECTED_PATH_TOKENS = {"validation", "blind", "holdout", "test"}


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    lowered = {part.lower() for part in path.parts}
    if lowered.intersection(PROTECTED_PATH_TOKENS):
        raise ValueError(f"Refusing protected catalog path: {path}")
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            if row.get("classification") != "development_only_synthetic_compiler_contract":
                raise ValueError(f"{path}:{line_number}: non-development classification")
            rows.append(row)
    return rows


def _actual_ir(plan: object) -> dict[str, object]:
    direction = getattr(plan, "direction_intent", None)
    program_column_roles = tuple(getattr(plan, "program_column_roles", ()))
    return {
        "operation": getattr(plan, "operation", None),
        "operation_candidates": list(getattr(plan, "operation_candidates", ())),
        "requested_years": list(getattr(plan, "requested_years", ())),
        # Compare the frozen program contract at use-site granularity. A
        # multi-step DAG may reuse one immutable operand slot as denominator,
        # so its role appears more than once without duplicating slot identity.
        "column_roles": list(program_column_roles or getattr(plan, "column_roles", ())),
        "legacy_column_roles": list(getattr(plan, "column_roles", ())),
        "program_column_roles": list(program_column_roles),
        "operand_roles": list(getattr(plan, "operand_roles", ())),
        "proof_operand_roles": list(getattr(plan, "proof_operand_roles", ())),
        "answer_mode": getattr(plan, "answer_mode", "SINGLE_SCALAR"),
        "direction_intent": direction.value if direction is not None else None,
        "requested_unit_family": getattr(plan, "requested_unit_family", ""),
        "entity_mode": getattr(plan, "entity_mode", "reporting"),
        "program_dag_operations": list(getattr(plan, "program_dag_operations", ())),
        "abstention_reason": getattr(plan, "abstention_reason", ""),
        "metric_text": getattr(plan, "metric_text", ""),
    }


def _operation_match(expected: dict[str, object], actual: dict[str, object]) -> bool:
    root = expected.get("root_operation")
    dag = tuple(str(value) for value in expected.get("program_dag_operations", ()))
    allowed = {str(actual.get("operation"))} if actual.get("operation") else set()
    allowed.update(str(value) for value in actual.get("operation_candidates", ()))
    if root:
        return str(root) in allowed
    if dag == ("SELECT_OR_DIVIDE",):
        return {"SELECT", "DIVIDE"}.issubset(allowed)
    return not dag


def _abstention_signal(expected_reason: str, actual: dict[str, object]) -> bool:
    if str(actual.get("abstention_reason", "")) == expected_reason:
        return True
    if expected_reason == "ambiguous_signed_direction":
        return actual.get("direction_intent") == "AMBIGUOUS"
    return False


def _checks(expected: dict[str, object], actual: dict[str, object]) -> dict[str, bool]:
    behavior = str(expected.get("behavior", "ANSWER"))
    checks: dict[str, bool] = {
        "behavior": (
            _abstention_signal(str(expected.get("abstention_reason", "")), actual)
            if behavior == "ABSTAIN"
            else True
        ),
        "operation": True if behavior == "ABSTAIN" else _operation_match(expected, actual),
        "requested_years": list(expected.get("requested_years", ())) == list(actual.get("requested_years", ())),
        "column_roles": list(expected.get("column_roles", ())) == list(actual.get("column_roles", ())),
        "answer_mode": str(expected.get("answer_mode", "SINGLE_SCALAR")) == str(actual.get("answer_mode", "SINGLE_SCALAR")),
        "direction_intent": (
            expected.get("direction_intent") is None
            or expected.get("direction_intent") == actual.get("direction_intent")
        ),
        "requested_unit_family": (
            not expected.get("requested_unit_family")
            or expected.get("requested_unit_family") == actual.get("requested_unit_family")
        ),
        "entity_mode": expected.get("entity_mode", "reporting") == actual.get("entity_mode", "reporting"),
    }
    expected_dag = tuple(str(value) for value in expected.get("program_dag_operations", ()))
    if len(expected_dag) > 1 or any(value in {"COMPARE", "SELECT_OR_DIVIDE"} for value in expected_dag):
        checks["program_dag"] = expected_dag == tuple(str(value) for value in actual.get("program_dag_operations", ()))
    else:
        checks["program_dag"] = True
    return checks


def _metric_bucket() -> dict[str, Counter[str]]:
    return defaultdict(Counter)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the frozen QuestionPlan compiler against the QP0 catalog")
    parser.add_argument("--catalog", type=Path, default=ROOT / "config" / "question_family_catalog_plan_actual_v1.jsonl")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "question_plan_compiler_audit_development_v1.json")
    args = parser.parse_args()

    rows = _read_jsonl(args.catalog)
    dimensions = Counter()
    by_family = _metric_bucket()
    by_category = _metric_bucket()
    by_level = _metric_bucket()
    by_variant = _metric_bucket()
    details: list[dict[str, object]] = []

    for row in rows:
        expected = row.get("expected_ir")
        if not isinstance(expected, dict):
            raise ValueError(f"Missing expected_ir: {row.get('example_id')}")
        plan = compile_question_plan(str(row["question"]))
        actual = _actual_ir(plan)
        checks = _checks(expected, actual)
        exact = all(checks.values())
        failed = sorted(name for name, passed in checks.items() if not passed)
        for name, passed in checks.items():
            dimensions[f"{name}:total"] += 1
            dimensions[f"{name}:pass"] += int(passed)
        for bucket, key in (
            (by_family, str(row["family_id"])),
            (by_category, str(row["category"])),
            (by_level, str(row["difficulty_level"])),
            (by_variant, str(row["variant"])),
        ):
            bucket[key]["total"] += 1
            bucket[key]["exact"] += int(exact)
        details.append(
            {
                "example_id": row["example_id"],
                "family_id": row["family_id"],
                "category": row["category"],
                "difficulty_level": row["difficulty_level"],
                "variant": row["variant"],
                "question": row["question"],
                "expected_ir": expected,
                "actual_ir": actual,
                "checks": checks,
                "exact": exact,
                "failed_dimensions": failed,
            }
        )

    def summarize(values: dict[str, Counter[str]]) -> dict[str, dict[str, object]]:
        return {
            key: {
                "total": count["total"],
                "exact": count["exact"],
                "exact_rate": count["exact"] / count["total"] if count["total"] else 0.0,
            }
            for key, count in sorted(values.items())
        }

    dimension_summary = {}
    for name in sorted({key.split(":", 1)[0] for key in dimensions}):
        total = dimensions[f"{name}:total"]
        passed = dimensions[f"{name}:pass"]
        dimension_summary[name] = {"total": total, "pass": passed, "pass_rate": passed / total if total else 0.0}
    exact_total = sum(1 for row in details if row["exact"])
    report = {
        "schema": "question-plan-compiler-audit-development-v1",
        "classification": "development_only_synthetic_compiler_contract",
        "evaluation_mode": "compiler_only_no_document_no_model_inference",
        "catalog_examples": len(rows),
        "family_count": len({str(row["family_id"]) for row in rows}),
        "exact": exact_total,
        "exact_rate": exact_total / len(rows) if rows else 0.0,
        "dimension_metrics": dimension_summary,
        "by_family": summarize(by_family),
        "by_category": summarize(by_category),
        "by_level": summarize(by_level),
        "by_variant": summarize(by_variant),
        "global_validation_loaded": False,
        "sealed_validation_loaded": False,
        "blind_test_loaded": False,
        "fine_tuning_performed": False,
        "threshold_changed": False,
        "runtime_changed": False,
        "protected_read_hygiene": {
            "zero_protected_read_claim": False,
            "used_for_metrics_or_design": False,
            "note": "A prior broad interactive threshold search printed an old protected report record; it was not loaded by this script and was not used in any catalog expectation, policy, or metric.",
        },
        "artifact_hashes": {
            "catalog": _digest(args.catalog),
            "question_planner": _digest(ROOT / "src" / "hrm_finance_qa" / "question_planner.py"),
            "audit_script": _digest(Path(__file__)),
            "checkpoint": _digest(ROOT / "models" / "document_hrm_V8_candidate.pt"),
            "fold_manifest": _digest(ROOT / "data" / "document_hrm_v12b2_reranker_disjoint_folds_subtract_v4.json"),
            "phase_b_manifest": _digest(ROOT / "cache" / "v12b2_folds_subtract_v5" / "manifest.json"),
            "phase_c_manifest": _digest(ROOT / "cache" / "v12b2_phase_c_subtract_v5" / "manifest.json"),
        },
        "details": details,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"details", "by_family"}}, ensure_ascii=False, indent=2))
    print(f"Saved compiler-only audit: {args.report.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
