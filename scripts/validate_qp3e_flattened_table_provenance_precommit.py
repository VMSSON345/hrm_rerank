from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config" / "finhrm_qp3e_flattened_table_provenance_precommit_v1.json"
DEFAULT_COHORT = ROOT / "data" / "finhrm_qp3e_flattened_table_provenance_development_v1.jsonl"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp3e_flattened_table_provenance_precommit_v1.json"
DEFAULT_CORE_IMPLEMENTATION_LOCK = ROOT / "reports" / "finhrm_qp3_core_implementation_lock_v1.json"
AUTHORIZED_IMPLEMENTATION_LOCKS = (
    ROOT / "reports" / "finhrm_qp3f_flattened_table_reconstruction_lock_v1.json",
    DEFAULT_CORE_IMPLEMENTATION_LOCK,
)
PROTECTED_PATH_TOKENS = {"validation", "blind", "holdout", "test"}
RUNTIME_FILES = {
    "program_data.py": ROOT / "src" / "hrm_finance_qa" / "program_data.py",
    "evidence_schema.py": ROOT / "src" / "hrm_finance_qa" / "evidence_schema.py",
    "symbolic_verifier.py": ROOT / "src" / "hrm_finance_qa" / "symbolic_verifier.py",
    "question_planner.py": ROOT / "src" / "hrm_finance_qa" / "question_planner.py",
    "program_synthesis.py": ROOT / "src" / "hrm_finance_qa" / "program_synthesis.py",
    "document_hrm.py": ROOT / "src" / "hrm_finance_qa" / "document_hrm.py",
    "document_reasoning_engine.py": ROOT / "src" / "hrm_finance_qa" / "document_reasoning_engine.py",
}
FORBIDDEN_SIGNAL_FIELDS = {
    "question",
    "question_text",
    "candidate_score",
    "candidate_rank",
    "numeric_value",
    "numeric_answer",
    "answer",
    "top_k_candidates",
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


def _contains_forbidden_signal(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in FORBIDDEN_SIGNAL_FIELDS or _contains_forbidden_signal(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_signal(item) for item in value)
    return False


def _source_valid(source: Any, allowed_methods: set[str]) -> bool:
    if not isinstance(source, dict):
        return False
    span = source.get("source_span")
    bbox = source.get("bbox")
    return (
        bool(str(source.get("document_id", "")))
        and isinstance(source.get("page"), int)
        and source["page"] > 0
        and isinstance(span, list)
        and len(span) == 2
        and all(isinstance(value, int) for value in span)
        and span[0] < span[1]
        and isinstance(source.get("source_sequence"), int)
        and str(source.get("extraction_method", "")) in allowed_methods
        and isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) for value in bbox)
        and bbox[0] < bbox[2]
        and bbox[1] < bbox[3]
    )


def _source_key(observation: dict[str, Any]) -> tuple[str, ...]:
    source = observation.get("source", {})
    return (
        str(source.get("document_id", "")),
        str(source.get("page", "")),
        json.dumps(source.get("source_span", []), separators=(",", ":")),
    )


def audit_reconstruction(
    record: dict[str, Any], precommit: dict[str, Any]
) -> tuple[str, tuple[str, ...]]:
    reasons: set[str] = set()
    contract = precommit["reconstruction_contract"]
    allowed_methods = set(contract["allowed_extraction_methods"])
    allowed_confidence = set(contract["allowed_role_confidence"])
    allowed_header_provenance = set(contract["allowed_header_provenance"])

    if record.get("source_boundary_state") != "COMPLETE":
        reasons.add("incomplete_source_boundary")
    if record.get("table_continuation_state") != "CLOSED":
        reasons.add("incomplete_table_continuation")

    columns = tuple(str(value) for value in record.get("declared_data_column_ids", ()))
    rows = tuple(str(value) for value in record.get("declared_row_ids", ()))
    headers = tuple(record.get("header_observations", ()))
    row_observations = tuple(record.get("row_observations", ()))
    cells = tuple(record.get("cell_observations", ()))
    repeated_headers = tuple(record.get("repeated_header_observations", ()))
    declared_width = record.get("declared_data_width")

    header_columns = tuple(str(item.get("column_id", "")) for item in headers)
    if (
        not columns
        or len(columns) != len(set(columns))
        or len(headers) != len(columns)
        or len(header_columns) != len(set(header_columns))
        or set(header_columns) != set(columns)
    ):
        reasons.add("header_role_cardinality_mismatch")
    if declared_width != len(columns):
        reasons.add("header_data_width_mismatch")

    anchors = {
        str(item.get("anchor_id", "")): item
        for item in record.get("inheritance_anchors", ())
    }
    header_roles: dict[str, str] = {}
    header_bands: dict[str, str] = {}
    for header in headers:
        column_id = str(header.get("column_id", ""))
        role = str(header.get("canonical_role", ""))
        confidence = str(header.get("role_confidence", ""))
        provenance = str(header.get("header_provenance", ""))
        if not role or confidence not in allowed_confidence:
            reasons.add("missing_explicit_column_role")
        if provenance not in allowed_header_provenance:
            reasons.add("missing_physical_provenance")
        if confidence == "INHERITED":
            anchor = anchors.get(str(header.get("inheritance_anchor_id", "")))
            if (
                anchor is None
                or str(anchor.get("canonical_role", "")) != role
                or anchor.get("provenance") != "explicit_flattened_header"
                or not str(anchor.get("header_text", ""))
            ):
                reasons.add("missing_explicit_column_role")
        if not _source_valid(header.get("source"), allowed_methods):
            reasons.add("missing_physical_provenance")
        header_roles[column_id] = role
        header_bands[column_id] = str(header.get("column_band_id", ""))

    observed_row_ids = tuple(str(item.get("row_id", "")) for item in row_observations)
    if (
        not rows
        or len(rows) != len(set(rows))
        or len(observed_row_ids) != len(set(observed_row_ids))
        or set(observed_row_ids) != set(rows)
    ):
        reasons.add("row_membership_conflict")
    for row in row_observations:
        if not _source_valid(row.get("source"), allowed_methods):
            reasons.add("missing_physical_provenance")

    expected_coordinates = {
        f"{row_id}|{column_id}" for row_id in rows for column_id in columns
    }
    observed_coordinates = {
        str(item.get("cell_coordinate", "")) for item in cells
    }
    if observed_coordinates != expected_coordinates:
        reasons.add("cell_coverage_mismatch")

    physical_rows: dict[str, set[str]] = {}
    physical_spans: dict[tuple[str, ...], set[str]] = {}
    cell_sequences: list[int] = []
    for cell in cells:
        row_id = str(cell.get("row_id", ""))
        column_id = str(cell.get("column_id", ""))
        coordinate = str(cell.get("cell_coordinate", ""))
        if coordinate != f"{row_id}|{column_id}":
            reasons.add("row_membership_conflict")
        physical_id = str(cell.get("physical_cell_id", ""))
        physical_rows.setdefault(physical_id, set()).add(row_id)
        physical_spans.setdefault(_source_key(cell), set()).add(coordinate)
        if (
            column_id in header_bands
            and str(cell.get("column_band_id", "")) != header_bands[column_id]
        ):
            reasons.add("column_geometry_unaligned")
        source = cell.get("source")
        if not _source_valid(source, allowed_methods):
            reasons.add("missing_physical_provenance")
        elif isinstance(source, dict):
            cell_sequences.append(int(source["source_sequence"]))
    if any(len({row for row in values if row}) > 1 for values in physical_rows.values()):
        reasons.add("row_membership_conflict")
    if any(len({value for value in values if value}) > 1 for values in physical_spans.values()):
        reasons.add("physical_span_conflict")

    header_sequences = [
        int(header["source"]["source_sequence"])
        for header in headers
        if _source_valid(header.get("source"), allowed_methods)
    ]
    if (
        not header_sequences
        or not cell_sequences
        or max(header_sequences) >= min(cell_sequences)
    ):
        reasons.add("source_order_unproven")

    expected_role_map = {
        column_id: header_roles.get(column_id, "") for column_id in columns
    }
    for repeated in repeated_headers:
        repeated_map = {
            str(key): str(value)
            for key, value in dict(repeated.get("column_role_map", {})).items()
        }
        if repeated_map != expected_role_map:
            reasons.add("repeated_header_role_conflict")
        if not _source_valid(repeated.get("source"), allowed_methods):
            reasons.add("missing_physical_provenance")

    if record.get("extractor_alignment_state") != "CONSISTENT":
        reasons.add("cross_extractor_layout_conflict")

    ordered = tuple(sorted(reasons))
    return ("RECONSTRUCTABLE_COMPLETE" if not ordered else "UNKNOWN", ordered)


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
    if precommit.get("status") != "precommitted_before_qp3e_flattened_runtime_changes":
        errors.append("precommit_status_invalid")
    for name, item in precommit["frozen_dependencies"].items():
        if _digest(ROOT / item["path"]) != item["sha256"]:
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
    gate = precommit["gate"]
    if len(records) != quota["total_examples"]:
        errors.append("wrong_total_examples")
    ids = [str(record.get("example_id", "")) for record in records]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_example_id")
    by_layout = Counter(str(record.get("layout_id", "")) for record in records)
    by_scenario = Counter(str(record.get("scenario_id", "")) for record in records)
    if len(by_layout) != quota["layout_groups"] or any(
        count != quota["scenarios_per_group"] for count in by_layout.values()
    ):
        errors.append("layout_quota_mismatch")
    if any(
        count != gate["scenario_balance_required"] for count in by_scenario.values()
    ):
        errors.append("scenario_quota_mismatch")

    scenario_classes = {
        str(item["id"]): str(item["class"])
        for item in precommit["scenario_matrix"]
    }
    hard_negative_types = {
        scenario
        for scenario, scenario_class in scenario_classes.items()
        if scenario_class.endswith("hard_negative")
    }
    if len(hard_negative_types) < quota["minimum_hard_negative_types"]:
        errors.append("insufficient_hard_negative_types")

    malformed: list[str] = []
    leakage: list[str] = []
    details: list[dict[str, Any]] = []
    roles: set[str] = set()
    layout_types: set[str] = set()
    state_exact = 0
    reason_exact = 0
    required_fields = {
        "schema", "classification", "example_id", "layout_id", "layout_type",
        "scenario_id", "document_id", "table_id", "source_boundary_state",
        "table_continuation_state", "extractor_alignment_state",
        "declared_data_width", "declared_data_column_ids", "declared_row_ids",
        "header_observations", "inheritance_anchors", "row_observations",
        "cell_observations", "repeated_header_observations",
        "expected_reconstruction_state", "expected_reason_set",
    }
    for record in records:
        example_id = str(record.get("example_id", ""))
        if (
            not required_fields.issubset(record)
            or record.get("schema")
            != "finhrm-qp3e-flattened-table-provenance-example-v1"
            or record.get("classification")
            != "development_only_synthetic_flattened_table_provenance_precommit"
        ):
            malformed.append(example_id)
        for flag in (
            "numeric_value_used", "numeric_answer_used", "candidate_score_used",
            "candidate_rank_used", "candidate_order_is_signal", "question_text_used",
            "real_document_content_used", "top_k_used",
        ):
            if record.get(flag) is not False:
                leakage.append(example_id)
        if _contains_forbidden_signal(record):
            leakage.append(example_id)
        roles.update(
            str(item.get("canonical_role", ""))
            for item in record.get("header_observations", ())
            if item.get("canonical_role")
        )
        layout_types.add(str(record.get("layout_type", "")))
        state, reasons = audit_reconstruction(record, precommit)
        expected_state = str(record.get("expected_reconstruction_state", ""))
        expected_reasons = tuple(
            sorted(str(value) for value in record.get("expected_reason_set", ()))
        )
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
    repeated_header_exact = 0
    for layout_id in by_layout:
        base = records_by_key[(layout_id, "complete_explicit")]
        permuted = records_by_key[(layout_id, "complete_permuted")]
        same_evidence = (
            _multiset(base["header_observations"])
            == _multiset(permuted["header_observations"])
            and _multiset(base["row_observations"])
            == _multiset(permuted["row_observations"])
            and _multiset(base["cell_observations"])
            == _multiset(permuted["cell_observations"])
        )
        permutation_exact += int(
            same_evidence
            and audit_reconstruction(base, precommit)
            == audit_reconstruction(permuted, precommit)
            == ("RECONSTRUCTABLE_COMPLETE", ())
        )
        repeated = records_by_key[
            (layout_id, "complete_repeated_header_agreement")
        ]
        repeated_header_exact += int(
            bool(repeated["repeated_header_observations"])
            and audit_reconstruction(repeated, precommit)
            == ("RECONSTRUCTABLE_COMPLETE", ())
        )

    detail_by_id = {item["example_id"]: item for item in details}
    boundary_scenarios = {"partial_source_boundary", "open_table_continuation"}
    conflict_scenarios = {
        "overlapping_cell_span",
        "ambiguous_row_membership",
        "cross_extractor_layout_conflict",
        "repeated_header_role_conflict",
    }
    boundary_exact = sum(
        detail_by_id[str(record["example_id"])]["exact"]
        and detail_by_id[str(record["example_id"])]["observed_state"] == "UNKNOWN"
        for record in records
        if record["scenario_id"] in boundary_scenarios
    )
    conflict_exact = sum(
        detail_by_id[str(record["example_id"])]["exact"]
        and detail_by_id[str(record["example_id"])]["observed_state"] == "UNKNOWN"
        for record in records
        if record["scenario_id"] in conflict_scenarios
    )
    complete_count = sum(
        item["observed_state"] == "RECONSTRUCTABLE_COMPLETE" for item in details
    )
    unknown_count = sum(item["observed_state"] == "UNKNOWN" for item in details)

    precommit_gate_passed = (
        not errors
        and state_exact == gate["state_exact_required"] == len(records)
        and reason_exact == gate["reason_set_exact_required"] == len(records)
        and complete_count == quota["complete_examples"]
        and unknown_count == quota["unknown_examples"]
        and permutation_exact == gate["permutation_pairs_required"]
        and repeated_header_exact == gate["repeated_header_positives_required"]
        and boundary_exact == gate["boundary_hard_negatives_required"]
        and conflict_exact == gate["conflict_hard_negatives_required"]
    )
    return {
        "schema": "finhrm-qp3e-flattened-table-provenance-precommit-audit-v1",
        "classification": "development_only_synthetic_flattened_table_provenance_precommit_audit",
        "precommit_gate_passed": precommit_gate_passed,
        "errors": errors,
        "cohort": {
            "examples": len(records),
            "layout_groups": len(by_layout),
            "layout_types": sorted(layout_types),
            "roles": sorted(roles),
            "hard_negative_types": sorted(hard_negative_types),
            "by_scenario": dict(sorted(by_scenario.items())),
        },
        "oracle": {
            "state_exact": state_exact,
            "reason_set_exact": reason_exact,
            "total": len(records),
            "complete": complete_count,
            "unknown": unknown_count,
            "failures": [item for item in details if not item["exact"]],
            "details": details,
        },
        "counterfactuals": {
            "permutation_exact": permutation_exact,
            "permutation_total": len(by_layout),
            "repeated_header_positive_exact": repeated_header_exact,
            "repeated_header_positive_total": len(by_layout),
            "boundary_hard_negative_exact": boundary_exact,
            "boundary_hard_negative_total": sum(
                by_scenario[scenario] for scenario in boundary_scenarios
            ),
            "conflict_hard_negative_exact": conflict_exact,
            "conflict_hard_negative_total": sum(
                by_scenario[scenario] for scenario in conflict_scenarios
            ),
        },
        "provenance": {
            "malformed_examples": malformed,
            "forbidden_signal_examples": sorted(set(leakage)),
            "numeric_values_used": False,
            "numeric_answers_used": False,
            "candidate_scores_or_ranks_used": False,
            "candidate_order_used_as_signal": False,
            "question_text_used": False,
            "real_document_content_used": False,
            "top_k_used": False,
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
    parser = argparse.ArgumentParser(
        description="Validate the frozen QP3E flattened-table provenance precommit"
    )
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
