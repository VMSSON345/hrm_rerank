from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config" / "finhrm_qp3a_column_role_proof_precommit_v1.json"
DEFAULT_COHORT = ROOT / "data" / "finhrm_qp3a_column_role_proof_development_v1.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp3a_column_role_proof_precommit_v1.json"
DEFAULT_CORE_IMPLEMENTATION_LOCK = ROOT / "reports" / "finhrm_qp3_core_implementation_lock_v1.json"
AUTHORIZED_IMPLEMENTATION_LOCKS = (
    ROOT / "reports" / "finhrm_qp3b_column_role_proof_lock_v1.json",
    ROOT / "reports" / "finhrm_qp3d_role_inventory_producer_lock_v1.json",
    ROOT / "reports" / "finhrm_qp3f_flattened_table_reconstruction_lock_v1.json",
    DEFAULT_CORE_IMPLEMENTATION_LOCK,
)
PROTECTED_PATH_TOKENS = {"validation", "blind", "holdout", "test"}
ALLOWED_ROLES = {"ACTUAL", "PLAN", "REVISED_PLAN", "FORECAST", ""}
RUNTIME_FILES = {
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
    overlap = {part.lower() for part in path.resolve().parts}.intersection(PROTECTED_PATH_TOKENS)
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
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
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


def _logical_key(cell: dict[str, Any], fields: list[str]) -> tuple[str, ...]:
    return tuple(str(cell.get(field, "")) for field in fields)


def _physical_key(cell: dict[str, Any], fields: list[str]) -> tuple[str, ...]:
    values: list[str] = []
    for field in fields:
        value = cell.get(field, "")
        values.append(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return tuple(values)


def _oracle_state(record: dict[str, Any], precommit: dict[str, Any]) -> str:
    cells = record["evidence_cells"]
    logical_fields = precommit["evidence_contract"]["logical_role_group_key"]
    physical_fields = precommit["evidence_contract"]["physical_equivalence_key"]
    physical_roles: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for cell in cells:
        physical_roles[_physical_key(cell, physical_fields)].add(str(cell["column_role"]))
    if any(bool(cell.get("column_role_conflict")) for cell in cells):
        return "INVALID"
    if any(len(roles) > 1 for roles in physical_roles.values()):
        return "INVALID"

    if record["query_kind"] == "explicit_role":
        required = tuple(str(value) for value in record["query_column_roles"])
        observed = tuple(str(cell["column_role"]) for cell in cells)
        if not observed or any(not role for role in observed):
            return "UNKNOWN"
        return "VALID" if observed == required else "INVALID"

    if record["role_inventory_state"] != "EXPLICIT_COMPLETE":
        return "UNKNOWN"
    logical_roles: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for cell in cells:
        logical_roles[_logical_key(cell, logical_fields)].add(str(cell["column_role"]))
    if any(len({role for role in roles if role}) > 1 for roles in logical_roles.values()):
        return "INVALID"
    return "VALID" if logical_roles else "UNKNOWN"


def _expected_action(record: dict[str, Any], state: str) -> str:
    if state == "VALID":
        return "DEFER_TO_OTHER_PROOFS"
    if state == "UNKNOWN":
        return "ABSTAIN"
    return "REJECT" if record["query_kind"] == "explicit_role" else "ABSTAIN"


def _cell_multiset(record: dict[str, Any]) -> Counter[str]:
    return Counter(
        json.dumps(cell, ensure_ascii=False, sort_keys=True)
        for cell in record["evidence_cells"]
    )


def build_report(
    precommit_path: Path = DEFAULT_PRECOMMIT,
    cohort_path: Path = DEFAULT_COHORT,
) -> dict[str, Any]:
    precommit = _read_json(precommit_path)
    records = _read_jsonl(cohort_path)
    errors: list[str] = []
    if precommit.get("status") != "precommitted_before_qp3_runtime_changes":
        errors.append("precommit_status_invalid")
    for name, item in precommit["frozen_inputs"].items():
        path = ROOT / item["path"]
        if _digest(path) != item["sha256"]:
            errors.append(f"frozen_hash_mismatch:{name}")

    runtime_hashes = {name: _digest(path) for name, path in RUNTIME_FILES.items()}
    authorized_runtime_hashes = _authorized_runtime_hashes(errors)
    runtime_mismatches = sorted(
        name for name, digest in runtime_hashes.items()
        if precommit["runtime_baseline"].get(name) != digest
        and authorized_runtime_hashes.get(name) != digest
    )
    errors.extend(f"runtime_changed:{name}" for name in runtime_mismatches)

    example_ids = [str(record.get("example_id", "")) for record in records]
    if len(example_ids) != len(set(example_ids)):
        errors.append("duplicate_example_id")
    quota = precommit["quota"]
    if len(records) != quota["total_examples"]:
        errors.append("wrong_total_examples")

    scenarios = [item["id"] for item in precommit["scenario_matrix"]]
    by_scenario = Counter(str(record.get("scenario_id", "")) for record in records)
    by_group = Counter(str(record.get("group_id", "")) for record in records)
    expected_per_scenario = quota["deferred_family_groups"] + quota["generalization_groups"]
    for scenario in scenarios:
        if by_scenario[scenario] != expected_per_scenario:
            errors.append(f"scenario_quota:{scenario}")
    if any(count != quota["scenarios_per_group"] for count in by_group.values()):
        errors.append("group_scenario_imbalance")

    deferred_groups = {
        str(record["group_id"])
        for record in records
        if record.get("source_type") == "frozen_deferred_family"
    }
    generalization_groups = {
        str(record["group_id"])
        for record in records
        if record.get("source_type") == "precommitted_metric_year_generalization"
    }
    if len(deferred_groups) != quota["deferred_family_groups"]:
        errors.append("deferred_family_coverage")
    if len(generalization_groups) != quota["generalization_groups"]:
        errors.append("generalization_group_coverage")

    metrics: set[str] = set()
    years: set[str] = set()
    state_exact = 0
    action_exact = 0
    malformed_examples: list[str] = []
    leakage_examples: list[str] = []
    oracle_details: list[dict[str, Any]] = []
    for record in records:
        malformed = False
        if record.get("schema") != "finhrm-qp3a-column-role-proof-example-v1":
            malformed = True
        if record.get("classification") != "development_only_synthetic_evidence_precommit":
            malformed = True
        if not record.get("evidence_cells"):
            malformed = True
        for forbidden_flag in (
            "numeric_value_used", "numeric_answer_used", "document_content_used",
            "candidate_score_used", "candidate_order_is_signal",
        ):
            if record.get(forbidden_flag) is not False:
                leakage_examples.append(str(record["example_id"]))
        for cell in record.get("evidence_cells", ()):
            required = {
                "cell_id", "document_id", "page", "table_id", "row_id", "column_id",
                "metric", "period", "column_role", "entity_scope", "unit_family",
                "source_span", "extraction_method", "extraction_value_token",
                "column_role_conflict", "synthetic_provenance",
            }
            if not required.issubset(cell):
                malformed = True
            if str(cell.get("column_role", "")) not in ALLOWED_ROLES:
                malformed = True
            token = str(cell.get("extraction_value_token", ""))
            if not token.startswith("TOKEN_"):
                malformed = True
            metrics.add(str(cell.get("metric", "")))
            years.add(str(cell.get("period", "")))
        if malformed:
            malformed_examples.append(str(record["example_id"]))
        state = _oracle_state(record, precommit)
        action = _expected_action(record, state)
        state_ok = state == record.get("expected_column_role_proof_state")
        action_ok = action == record.get("expected_action")
        state_exact += int(state_ok)
        action_exact += int(action_ok)
        oracle_details.append(
            {
                "example_id": record["example_id"],
                "oracle_state": state,
                "expected_state": record.get("expected_column_role_proof_state"),
                "oracle_action": action,
                "expected_action": record.get("expected_action"),
                "exact": state_ok and action_ok,
            }
        )
    if malformed_examples:
        errors.append("malformed_examples")
    if leakage_examples:
        errors.append("forbidden_signal_usage")
    if len(metrics) < quota["minimum_distinct_metrics"]:
        errors.append("insufficient_metric_diversity")
    if len(years) < quota["minimum_distinct_years"]:
        errors.append("insufficient_year_diversity")

    same_value = [
        record for record in records
        if record["scenario_id"] == "no_role_same_value_multi_role_conflict_complete"
    ]
    same_value_exact = 0
    logical_fields = precommit["evidence_contract"]["logical_role_group_key"]
    for record in same_value:
        groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
        for cell in record["evidence_cells"]:
            groups[_logical_key(cell, logical_fields)].append(cell)
        exact = all(
            len({cell["column_role"] for cell in cells}) >= 2
            and len({cell["extraction_value_token"] for cell in cells}) == 1
            for cells in groups.values()
        )
        same_value_exact += int(exact)

    by_group_scenario = {
        (str(record["group_id"]), str(record["scenario_id"])): record
        for record in records
    }
    permutation_exact = 0
    for group in by_group:
        original = by_group_scenario[(group, "no_role_multi_role_conflict_complete")]
        permuted = by_group_scenario[(group, "no_role_conflict_permuted_complete")]
        same_evidence = _cell_multiset(original) == _cell_multiset(permuted)
        reverse_order = list(reversed(original["evidence_cells"])) == permuted["evidence_cells"]
        same_contract = (
            original["expected_column_role_proof_state"] == permuted["expected_column_role_proof_state"]
            and original["expected_action"] == permuted["expected_action"]
        )
        permutation_exact += int(same_evidence and reverse_order and same_contract)

    unknown_controls = [
        record for record in records
        if record["scenario_id"] == "no_role_unique_inventory_unknown"
    ]
    unknown_exact = sum(
        record["expected_column_role_proof_state"] == "UNKNOWN"
        and record["expected_action"] == "ABSTAIN"
        for record in unknown_controls
    )
    gate = precommit["gate"]
    gate_passed = (
        not errors
        and state_exact == action_exact == len(records) == gate["exact_examples_required"]
        and same_value_exact == len(same_value) == gate["same_value_conflict_required"]
        and permutation_exact == gate["permutation_pairs_required"]
        and unknown_exact == len(unknown_controls) == gate["unknown_inventory_controls_required"]
    )
    return {
        "schema": "finhrm-qp3a-column-role-proof-precommit-audit-v1",
        "classification": "development_only_synthetic_evidence_precommit_audit",
        "precommit_gate_passed": gate_passed,
        "errors": errors,
        "cohort": {
            "examples": len(records),
            "groups": len(by_group),
            "deferred_groups": len(deferred_groups),
            "generalization_groups": len(generalization_groups),
            "by_scenario": dict(sorted(by_scenario.items())),
            "distinct_metrics": sorted(metrics),
            "distinct_years": sorted(years),
        },
        "oracle": {
            "state_exact": state_exact,
            "action_exact": action_exact,
            "total": len(records),
            "details": oracle_details,
        },
        "counterfactuals": {
            "same_value_conflict_exact": same_value_exact,
            "same_value_conflict_total": len(same_value),
            "permutation_exact": permutation_exact,
            "permutation_total": len(by_group),
            "unknown_inventory_exact": unknown_exact,
            "unknown_inventory_total": len(unknown_controls),
        },
        "provenance": {
            "malformed_examples": malformed_examples,
            "forbidden_signal_examples": sorted(set(leakage_examples)),
            "numeric_values_used": False,
            "numeric_answers_used": False,
            "document_content_used": False,
            "candidate_scores_used": False,
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
            "document_loaded": False,
            "pdf_loaded": False,
            "model_loaded": False,
            "global_validation_loaded": False,
            "sealed_validation_loaded": False,
            "blind_test_loaded": False,
            "phase_d_unlocked": False
        },
        "authorized_next_step": precommit["authorized_next_step_after_precommit"]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the frozen QP3A column-role proof precommit")
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = build_report(args.precommit, args.cohort)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        key: value for key, value in report.items()
        if key not in {"oracle"}
    }
    summary["oracle"] = {
        key: value for key, value in report["oracle"].items() if key != "details"
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["precommit_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
