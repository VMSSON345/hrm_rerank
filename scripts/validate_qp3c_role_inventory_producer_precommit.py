from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config" / "finhrm_qp3c_role_inventory_producer_precommit_v1.json"
DEFAULT_COHORT = ROOT / "data" / "finhrm_qp3c_role_inventory_producer_development_v1.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp3c_role_inventory_producer_precommit_v1.json"
DEFAULT_CORE_IMPLEMENTATION_LOCK = ROOT / "reports" / "finhrm_qp3_core_implementation_lock_v1.json"
AUTHORIZED_IMPLEMENTATION_LOCKS = (
    ROOT / "reports" / "finhrm_qp3d_role_inventory_producer_lock_v1.json",
    ROOT / "reports" / "finhrm_qp3f_flattened_table_reconstruction_lock_v1.json",
    DEFAULT_CORE_IMPLEMENTATION_LOCK,
)
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
FORBIDDEN_SIGNAL_FIELDS = {
    "question",
    "question_text",
    "candidate_score",
    "candidate_rank",
    "numeric_value",
    "numeric_answer",
    "answer",
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
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("Cohort must contain JSON objects")
    return records


def _authorized_runtime_hashes(errors: list[str]) -> dict[str, str]:
    authorized: dict[str, str] = {}
    for lock_path in AUTHORIZED_IMPLEMENTATION_LOCKS:
        lock = _read_json(lock_path)
        if not lock.get("implementation_gate_passed"):
            errors.append(f"implementation_lock_not_passed:{lock_path.name}")
            return {}
        authorized.update(
            {
                Path(relative).name: str(digest)
                for relative, digest in lock.get("runtime_hashes", {}).items()
            }
        )
    for name, digest in authorized.items():
        path = ROOT / "src" / "hrm_finance_qa" / name
        if not path.is_file() or _digest(path) != digest:
            errors.append(f"qp3_core_runtime_hash_mismatch:{name}")
            return {}
    return authorized


def _physical_key(item: dict[str, Any]) -> tuple[str, ...]:
    return (
        str(item.get("document_id", "")),
        str(item.get("page", "")),
        json.dumps(item.get("source_span", []), separators=(",", ":")),
        str(item.get("cell_coordinate", "")),
    )


def audit_inventory(
    record: dict[str, Any], precommit: dict[str, Any]
) -> tuple[str, tuple[str, ...]]:
    reasons: set[str] = set()
    if record.get("table_boundary_state") != "COMPLETE":
        reasons.add("incomplete_table_boundary")
    if record.get("header_alignment_state") != "ALIGNED":
        reasons.add("header_alignment_unproven")

    expected_columns = tuple(str(value) for value in record.get("expected_data_column_ids", ()))
    headers = tuple(record.get("header_columns", ()))
    observed_columns = tuple(str(item.get("column_id", "")) for item in headers)
    if (
        not expected_columns
        or len(expected_columns) != len(set(expected_columns))
        or len(observed_columns) != len(set(observed_columns))
        or set(expected_columns) != set(observed_columns)
    ):
        reasons.add("header_column_coverage_mismatch")

    allowed_provenance = set(precommit["producer_contract"]["allowed_header_provenance"])
    allowed_confidence = set(precommit["producer_contract"]["allowed_role_confidence"])
    anchors = {
        str(item.get("anchor_id", "")): item
        for item in record.get("inheritance_anchors", ())
    }
    for header in headers:
        role = str(header.get("column_role", ""))
        confidence = str(header.get("column_role_confidence", ""))
        path = tuple(str(value) for value in header.get("header_path", ()))
        provenance = tuple(
            str(value) for value in header.get("header_path_provenance", ())
        )
        if header.get("column_role_conflict") or confidence == "CONFLICT":
            reasons.add("header_role_conflict")
        if not role or confidence not in allowed_confidence:
            reasons.add("missing_explicit_column_role")
        if (
            not path
            or len(path) != len(provenance)
            or any(value not in allowed_provenance for value in provenance)
        ):
            reasons.add("invalid_header_provenance")
        if confidence == "INHERITED":
            anchor = anchors.get(str(header.get("inheritance_anchor_id", "")))
            if (
                anchor is None
                or anchor.get("provenance") != "explicit_header"
                or str(anchor.get("column_role", "")) != role
                or not str(anchor.get("header_text", ""))
            ):
                reasons.add("missing_inheritance_anchor")

    expected_rows = tuple(str(value) for value in record.get("expected_row_ids", ()))
    declared_expected = {str(value) for value in record.get("expected_cell_coordinates", ())}
    cartesian_expected = {
        f"{row_id}|{column_id}"
        for row_id in expected_rows
        for column_id in expected_columns
    }
    observed = {str(value) for value in record.get("observed_cell_coordinates", ())}
    if not expected_rows or declared_expected != cartesian_expected or observed != cartesian_expected:
        reasons.add("cell_coverage_mismatch")

    physical = tuple(record.get("physical_observations", ()))
    physical_coordinates = {str(item.get("cell_coordinate", "")) for item in physical}
    if not observed.issubset(physical_coordinates):
        reasons.add("missing_physical_provenance")
    for item in physical:
        span = item.get("source_span", ())
        if (
            not str(item.get("document_id", ""))
            or not isinstance(item.get("page"), int)
            or not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(value, int) for value in span)
            or not str(item.get("extraction_method", ""))
        ):
            reasons.add("missing_physical_provenance")

    physical_roles: dict[tuple[str, ...], set[str]] = {}
    for item in physical:
        physical_roles.setdefault(_physical_key(item), set()).add(
            str(item.get("column_role", ""))
        )
    if any(len({role for role in roles if role}) > 1 for roles in physical_roles.values()):
        reasons.add("physical_role_conflict")
    if record.get("extractor_alignment_state") == "CONFLICT":
        reasons.add("cross_extractor_role_conflict")

    ordered = tuple(sorted(reasons))
    return ("EXPLICIT_COMPLETE" if not ordered else "UNKNOWN", ordered)


def _contains_forbidden_signal(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_SIGNAL_FIELDS:
                return True
            if _contains_forbidden_signal(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_signal(item) for item in value)
    return False


def _multiset(items: Any) -> Counter[str]:
    return Counter(
        json.dumps(item, ensure_ascii=False, sort_keys=True) for item in items
    )


def build_report(
    precommit_path: Path = DEFAULT_PRECOMMIT,
    cohort_path: Path = DEFAULT_COHORT,
) -> dict[str, Any]:
    precommit = _read_json(precommit_path)
    records = _read_jsonl(cohort_path)
    errors: list[str] = []
    if precommit.get("status") != "precommitted_before_qp3c_runtime_producer_changes":
        errors.append("precommit_status_invalid")
    for name, item in precommit["frozen_dependencies"].items():
        path = ROOT / item["path"]
        if _digest(path) != item["sha256"]:
            errors.append(f"frozen_hash_mismatch:{name}")

    runtime_hashes = {name: _digest(path) for name, path in RUNTIME_FILES.items()}
    authorized_runtime_hashes = _authorized_runtime_hashes(errors)
    runtime_mismatches = sorted(
        name
        for name, digest in runtime_hashes.items()
        if precommit["runtime_baseline"].get(name) != digest
        and authorized_runtime_hashes.get(name) != digest
    )
    errors.extend(f"runtime_changed:{name}" for name in runtime_mismatches)

    quota = precommit["quota"]
    if len(records) != quota["total_examples"]:
        errors.append("wrong_total_examples")
    example_ids = [str(record.get("example_id", "")) for record in records]
    if len(example_ids) != len(set(example_ids)):
        errors.append("duplicate_example_id")
    by_layout = Counter(str(record.get("layout_id", "")) for record in records)
    by_scenario = Counter(str(record.get("scenario_id", "")) for record in records)
    if len(by_layout) != quota["layout_groups"] or any(
        count != quota["scenarios_per_group"] for count in by_layout.values()
    ):
        errors.append("layout_quota_mismatch")
    if any(
        count != precommit["gate"]["scenario_balance_required"]
        for count in by_scenario.values()
    ):
        errors.append("scenario_quota_mismatch")

    state_exact = 0
    reason_exact = 0
    malformed: list[str] = []
    leakage: list[str] = []
    details: list[dict[str, Any]] = []
    roles: set[str] = set()
    layout_types: set[str] = set()
    for record in records:
        example_id = str(record.get("example_id", ""))
        required = {
            "schema", "classification", "example_id", "layout_id", "layout_type",
            "scenario_id", "document_id", "table_id", "table_boundary_state",
            "header_alignment_state", "extractor_alignment_state",
            "expected_data_column_ids", "expected_row_ids", "header_columns",
            "inheritance_anchors", "expected_cell_coordinates",
            "observed_cell_coordinates", "physical_observations",
            "expected_inventory_state", "expected_reason_set",
        }
        if (
            not required.issubset(record)
            or record.get("schema") != "finhrm-qp3c-role-inventory-producer-example-v1"
            or record.get("classification")
            != "development_only_synthetic_table_provenance_precommit"
        ):
            malformed.append(example_id)
        for flag in (
            "numeric_value_used", "numeric_answer_used", "candidate_score_used",
            "candidate_order_is_signal", "question_text_used", "document_content_used",
        ):
            if record.get(flag) is not False:
                leakage.append(example_id)
        if _contains_forbidden_signal(record):
            leakage.append(example_id)
        roles.update(
            str(item.get("column_role", ""))
            for item in record.get("header_columns", ())
            if item.get("column_role")
        )
        layout_types.add(str(record.get("layout_type", "")))
        state, reasons = audit_inventory(record, precommit)
        expected_state = str(record.get("expected_inventory_state", ""))
        expected_reasons = tuple(sorted(str(value) for value in record.get("expected_reason_set", ())))
        state_ok = state == expected_state
        reasons_ok = reasons == expected_reasons
        state_exact += int(state_ok)
        reason_exact += int(reasons_ok)
        details.append(
            {
                "example_id": example_id,
                "observed_state": state,
                "expected_state": expected_state,
                "observed_reasons": reasons,
                "expected_reasons": expected_reasons,
                "exact": state_ok and reasons_ok,
            }
        )
    if malformed:
        errors.append("malformed_examples")
    if leakage:
        errors.append("forbidden_signal_usage")
    if len(roles) < quota["minimum_distinct_roles"]:
        errors.append("insufficient_role_diversity")
    if len(layout_types) < quota["minimum_distinct_layout_types"]:
        errors.append("insufficient_layout_diversity")

    records_by_key = {
        (str(record["layout_id"]), str(record["scenario_id"])): record
        for record in records
    }
    permutation_exact = 0
    inherited_anchor_exact = 0
    conflict_exact = 0
    conflict_scenarios = {
        "conflicting_header_role",
        "cross_extractor_role_conflict",
        "duplicate_physical_role_conflict",
    }
    for layout_id in by_layout:
        base = records_by_key[(layout_id, "complete_explicit")]
        permuted = records_by_key[(layout_id, "complete_permuted")]
        same_evidence = (
            _multiset(base["header_columns"]) == _multiset(permuted["header_columns"])
            and _multiset(base["observed_cell_coordinates"])
            == _multiset(permuted["observed_cell_coordinates"])
            and _multiset(base["physical_observations"])
            == _multiset(permuted["physical_observations"])
        )
        base_audit = audit_inventory(base, precommit)
        permuted_audit = audit_inventory(permuted, precommit)
        permutation_exact += int(
            same_evidence
            and base_audit == permuted_audit == ("EXPLICIT_COMPLETE", ())
        )
        inherited = records_by_key[(layout_id, "complete_inherited_with_anchor")]
        inherited_state, inherited_reasons = audit_inventory(inherited, precommit)
        inherited_anchor_exact += int(
            inherited_state == "EXPLICIT_COMPLETE"
            and not inherited_reasons
            and any(
                item.get("column_role_confidence") == "INHERITED"
                for item in inherited["header_columns"]
            )
            and bool(inherited["inheritance_anchors"])
        )
    for record, detail in zip(records, details):
        if record["scenario_id"] in conflict_scenarios:
            conflict_exact += int(
                detail["exact"] and detail["observed_state"] == "UNKNOWN"
            )

    complete_count = sum(
        detail["observed_state"] == "EXPLICIT_COMPLETE" for detail in details
    )
    unknown_count = sum(detail["observed_state"] == "UNKNOWN" for detail in details)
    gate = precommit["gate"]
    precommit_gate_passed = (
        not errors
        and state_exact == gate["state_exact_required"] == len(records)
        and reason_exact == gate["reason_set_exact_required"] == len(records)
        and complete_count == quota["complete_examples"]
        and unknown_count == quota["unknown_examples"]
        and permutation_exact == gate["permutation_pairs_required"]
        and inherited_anchor_exact == gate["inherited_anchor_positives_required"]
        and conflict_exact == gate["conflict_negatives_required"]
    )
    return {
        "schema": "finhrm-qp3c-role-inventory-producer-precommit-audit-v1",
        "classification": "development_only_synthetic_table_provenance_precommit_audit",
        "precommit_gate_passed": precommit_gate_passed,
        "errors": errors,
        "cohort": {
            "examples": len(records),
            "layout_groups": len(by_layout),
            "layout_types": sorted(layout_types),
            "roles": sorted(roles),
            "by_scenario": dict(sorted(by_scenario.items())),
        },
        "oracle": {
            "state_exact": state_exact,
            "reason_set_exact": reason_exact,
            "total": len(records),
            "complete": complete_count,
            "unknown": unknown_count,
            "failures": [detail for detail in details if not detail["exact"]],
            "details": details,
        },
        "counterfactuals": {
            "permutation_exact": permutation_exact,
            "permutation_total": len(by_layout),
            "inherited_anchor_exact": inherited_anchor_exact,
            "inherited_anchor_total": len(by_layout),
            "conflict_negative_exact": conflict_exact,
            "conflict_negative_total": sum(
                count for scenario, count in by_scenario.items() if scenario in conflict_scenarios
            ),
        },
        "provenance": {
            "malformed_examples": malformed,
            "forbidden_signal_examples": sorted(set(leakage)),
            "numeric_values_used": False,
            "numeric_answers_used": False,
            "candidate_scores_used": False,
            "question_text_used": False,
            "document_content_used": False,
        },
        "runtime_baseline": {
            "observed_hashes": runtime_hashes,
            "mismatches": runtime_mismatches,
            "runtime_source_changed": bool(runtime_mismatches),
        },
        "artifact_hashes": {
            "precommit": _digest(precommit_path),
            "cohort": _digest(cohort_path),
            "validator": _digest(Path(__file__)),
        },
        "invariants": {
            "runtime_implementation_performed": False,
            "runtime_acceptance_policy_changed": False,
            "threshold_changed": False,
            "checkpoint_changed": False,
            "fine_tuning_performed": False,
            "pdf_loaded": False,
            "model_loaded": False,
            "global_validation_loaded": False,
            "sealed_validation_loaded": False,
            "blind_test_loaded": False,
            "phase_d_unlocked": False,
        },
        "authorized_next_step": precommit["authorized_next_step_after_precommit"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen QP3C producer precommit")
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = build_report(args.precommit, args.cohort)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "state_exact": report["oracle"]["state_exact"],
                "reason_set_exact": report["oracle"]["reason_set_exact"],
                "total": report["oracle"]["total"],
                **report["counterfactuals"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    print(f"precommit_gate_passed={report['precommit_gate_passed']}")
    return 0 if report["precommit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
