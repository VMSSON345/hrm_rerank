from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from hrm_finance_qa.evidence_schema import build_evidence_graph
from hrm_finance_qa.flattened_table_reconstruction import (
    RECONSTRUCTABLE_COMPLETE,
    assess_flattened_table_reconstruction,
    qp3d_manifest_from_flattened_record,
)
from hrm_finance_qa.program_data import extract_numeric_candidates, merge_candidates


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = ROOT / "data" / "finhrm_qp3e_flattened_table_provenance_development_v1.jsonl"
DEFAULT_LOCK = ROOT / "reports" / "finhrm_qp3e_flattened_table_provenance_precommit_lock_v1.json"
DEFAULT_CORE_IMPLEMENTATION_LOCK = ROOT / "reports" / "finhrm_qp3_core_implementation_lock_v1.json"
DEFAULT_REPLAY_AUTHORIZATION = ROOT / "config" / "finhrm_qp3_core_replay_authorization_v1.json"
DEFAULT_REPORT = ROOT / "reports" / "finhrm_qp3f_flattened_table_reconstruction_development_v1.json"
PROTECTED_PATH_TOKENS = {"validation", "blind", "holdout", "test"}
RUNTIME_FILES = {
    "program_data.py": ROOT / "src" / "hrm_finance_qa" / "program_data.py",
    "evidence_schema.py": ROOT / "src" / "hrm_finance_qa" / "evidence_schema.py",
    "symbolic_verifier.py": ROOT / "src" / "hrm_finance_qa" / "symbolic_verifier.py",
    "direction_contract.py": ROOT / "src" / "hrm_finance_qa" / "direction_contract.py",
    "question_planner.py": ROOT / "src" / "hrm_finance_qa" / "question_planner.py",
    "program_synthesis.py": ROOT / "src" / "hrm_finance_qa" / "program_synthesis.py",
    "document_hrm.py": ROOT / "src" / "hrm_finance_qa" / "document_hrm.py",
    "document_reasoning_engine.py": ROOT / "src" / "hrm_finance_qa" / "document_reasoning_engine.py",
    "document_page_index.py": ROOT / "src" / "hrm_finance_qa" / "document_page_index.py",
    "document_sources.py": ROOT / "src" / "hrm_finance_qa" / "document_sources.py",
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
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _authorized_core_runtime_files(errors: list[str]) -> set[str]:
    lock = _read_json(DEFAULT_CORE_IMPLEMENTATION_LOCK)
    if not lock.get("implementation_gate_passed"):
        errors.append("qp3_core_implementation_lock_not_passed")
        return set()
    authorized: set[str] = set()
    for relative, expected in lock.get("runtime_hashes", {}).items():
        path = ROOT / str(relative)
        if not path.is_file() or _digest(path) != str(expected):
            errors.append(f"qp3_core_runtime_hash_mismatch:{Path(relative).name}")
            continue
        authorized.add(Path(relative).name)
    return authorized


def _authorized_replay_harness_hashes(errors: list[str]) -> dict[str, str]:
    authorization = _read_json(DEFAULT_REPLAY_AUTHORIZATION)
    parent = authorization.get("parent_implementation_lock", {})
    parent_path = ROOT / str(parent.get("path", ""))
    if (
        not parent_path.is_file()
        or _digest(parent_path) != str(parent.get("sha256", ""))
    ):
        errors.append("replay_authorization_parent_lock_mismatch")
        return {}
    authorized: dict[str, str] = {}
    for item in authorization.get("authorized_harness_artifacts", ()):
        relative = str(item.get("path", "")).replace("\\", "/")
        path = ROOT / relative
        digest = str(item.get("sha256", ""))
        if (
            not path.is_file()
            or path.stat().st_size != int(item.get("size_bytes", -1))
            or _digest(path) != digest
        ):
            errors.append(f"replay_authorization_hash_mismatch:{relative}")
            return {}
        authorized[relative] = digest
    return authorized


def _placed_line(parts: list[tuple[int, str]], width: int = 205) -> str:
    output = [" "] * width
    for start, value in parts:
        for offset, character in enumerate(value):
            if start + offset < width:
                output[start + offset] = character
    return "".join(output).rstrip()


def flattened_context(
    *,
    rows: int = 2,
    bounded: bool = True,
    role_cardinality_mismatch: bool = False,
    geometry_conflict: bool = False,
) -> str:
    starts = [50, 75, 100, 125, 155]
    role_parts = [
        (starts[0], "Thực hiện"),
        (starts[1], "Kế hoạch"),
        (starts[2], "Thực hiện"),
        (starts[3], "TH 2023/KH2023"),
        (starts[4], "TH 2023/TH2022"),
    ]
    if role_cardinality_mismatch:
        role_parts.pop(1)
    lines = []
    if bounded:
        lines.extend(("Mở đầu trang", "STT Chỉ tiêu"))
    lines.append(_placed_line(role_parts))
    lines.append(
        _placed_line(
            [
                (starts[0], "Năm 2022"),
                (starts[1], "Năm 2023"),
                (starts[2], "Năm 2023"),
            ]
        )
    )
    for index in range(rows):
        current_starts = list(starts)
        if geometry_conflict and index == 0:
            current_starts[2] = 114
        values = [
            str(1000000000 + index),
            str(1100000000 + index),
            str(900000000 + index),
            "81.82%",
            "-10.00%",
        ]
        lines.append(
            _placed_line(
                [(0, f"{index + 1} Chỉ tiêu {index + 1}")]
                + list(zip(current_starts, values))
            )
        )
    if bounded:
        lines.extend(("Kết thúc bảng", "Thuyết minh tiếp theo"))
    return "\n".join(lines)


def _candidates(context: str, *, maximum: int = 64):
    return extract_numeric_candidates(
        "Chỉ tiêu 1 thực hiện năm 2023 là bao nhiêu?",
        context,
        max_candidates=maximum,
        source_name="synthetic-flattened-development.txt",
        page=1,
        source_line_start=100,
        extraction_method="native",
    )


def _end_to_end_cases() -> list[dict[str, Any]]:
    complete = _candidates(flattened_context())
    complete_graph = build_evidence_graph(complete)
    partial = _candidates(flattened_context(bounded=False))
    partial_graph = build_evidence_graph(partial)
    role_mismatch = _candidates(
        flattened_context(role_cardinality_mismatch=True)
    )
    role_mismatch_graph = build_evidence_graph(role_mismatch)
    geometry = _candidates(flattened_context(geometry_conflict=True))
    geometry_graph = build_evidence_graph(geometry)
    ordinary = extract_numeric_candidates(
        "Doanh thu năm 2023 là bao nhiêu?",
        "Mở đầu\nNăm 2022 Năm 2023\nDoanh thu 100000000 110000000\nKết thúc",
        source_name="ordinary-development.txt",
        page=1,
        extraction_method="native",
    )
    ordinary_graph = build_evidence_graph(ordinary)
    large = _candidates(flattened_context(rows=20), maximum=64)
    large_manifests = []
    for candidate in large:
        marker = "ROLE_INVENTORY_MANIFEST: "
        raw = next(
            (
                line[len(marker) :]
                for line in candidate.context.splitlines()
                if line.startswith(marker)
            ),
            "",
        )
        if raw:
            large_manifests.append(json.loads(raw))
    full_coordinates = max(
        (len(item.get("observed_cell_coordinates", ())) for item in large_manifests),
        default=0,
    )
    reversed_graph = build_evidence_graph(tuple(reversed(complete)))
    merged_lines = flattened_context().splitlines()
    overlapping_chunks = (
        SimpleNamespace(
            text="\n".join(merged_lines[:6]),
            source_name="synthetic-overlap-development.txt",
            page_start=1,
            page_end=1,
            line_start=100,
            score=1.0,
            extraction_methods=("native",),
        ),
        SimpleNamespace(
            text="\n".join(merged_lines[3:]),
            source_name="synthetic-overlap-development.txt",
            page_start=1,
            page_end=1,
            line_start=103,
            score=0.9,
            extraction_methods=("native",),
        ),
    )
    merged_candidates = merge_candidates(
        "Chỉ tiêu 1 thực hiện năm 2023 là bao nhiêu?",
        overlapping_chunks,
        max_candidates=64,
    )
    merged_graph = build_evidence_graph(merged_candidates)
    return [
        {
            "id": "complete_native_flattened_manifest",
            "passed": bool(complete)
            and complete_graph.column_role_inventory_state == "EXPLICIT_COMPLETE",
            "observed": complete_graph.column_role_inventory_state,
        },
        {
            "id": "complete_manifest_has_table_scoped_roles",
            "passed": bool(complete_graph.table_role_inventories)
            and all(
                item.state == "EXPLICIT_COMPLETE"
                for item in complete_graph.table_role_inventories
            ),
            "observed": [item.state for item in complete_graph.table_role_inventories],
        },
        {
            "id": "partial_flattened_does_not_emit_manifest",
            "passed": partial_graph.column_role_inventory_state == "",
            "observed": partial_graph.column_role_inventory_state,
        },
        {
            "id": "role_cardinality_mismatch_does_not_emit_manifest",
            "passed": role_mismatch_graph.column_role_inventory_state == "",
            "observed": role_mismatch_graph.column_role_inventory_state,
        },
        {
            "id": "geometry_conflict_does_not_emit_manifest",
            "passed": geometry_graph.column_role_inventory_state == "",
            "observed": geometry_graph.column_role_inventory_state,
        },
        {
            "id": "ordinary_period_plaintext_not_applicable",
            "passed": ordinary_graph.column_role_inventory_state == "",
            "observed": ordinary_graph.column_role_inventory_state,
        },
        {
            "id": "pre_top_k_full_flattened_structure",
            "passed": len(large) == 64 and full_coordinates == 100,
            "observed": {
                "retained_candidates": len(large),
                "manifest_coordinates": full_coordinates,
            },
        },
        {
            "id": "candidate_permutation_invariant",
            "passed": (
                complete_graph.column_role_inventory_state,
                complete_graph.column_role_inventory_reasons,
            )
            == (
                reversed_graph.column_role_inventory_state,
                reversed_graph.column_role_inventory_reasons,
            ),
            "observed": reversed_graph.column_role_inventory_state,
        },
        {
            "id": "overlapping_chunks_reconstruct_before_top_k",
            "passed": merged_graph.column_role_inventory_state
            == "EXPLICIT_COMPLETE",
            "observed": merged_graph.column_role_inventory_state,
        },
    ]


def build_report(
    cohort_path: Path = DEFAULT_COHORT,
    lock_path: Path = DEFAULT_LOCK,
) -> dict[str, Any]:
    cohort = _read_jsonl(cohort_path)
    lock = _read_json(lock_path)
    errors: list[str] = []
    replay_authorized = _authorized_replay_harness_hashes(errors)
    for entry in [*lock.get("artifacts", ()), *lock.get("frozen_dependencies", ())]:
        path = ROOT / str(entry.get("path", ""))
        relative = str(entry.get("path", "")).replace("\\", "/")
        observed_digest = _digest(path) if path.is_file() else ""
        if (
            not path.is_file()
            or path.stat().st_size != int(entry.get("size_bytes", -1))
            or observed_digest != str(entry.get("sha256", ""))
        ) and replay_authorized.get(relative) != observed_digest:
            errors.append(f"qp3e_lock_mismatch:{entry.get('path', '')}")

    state_exact = 0
    reason_exact = 0
    manifest_exact = 0
    details: list[dict[str, Any]] = []
    by_scenario: Counter[str] = Counter()
    for record in cohort:
        assessment = assess_flattened_table_reconstruction(record)
        expected_state = str(record["expected_reconstruction_state"])
        expected_reasons = tuple(sorted(record["expected_reason_set"]))
        state_ok = assessment.state == expected_state
        reason_ok = assessment.reasons == expected_reasons
        manifest = qp3d_manifest_from_flattened_record(record)
        manifest_ok = (manifest is not None) == (
            expected_state == RECONSTRUCTABLE_COMPLETE
        )
        state_exact += int(state_ok)
        reason_exact += int(reason_ok)
        manifest_exact += int(manifest_ok)
        by_scenario[str(record["scenario_id"])] += 1
        details.append(
            {
                "example_id": record["example_id"],
                "observed_state": assessment.state,
                "expected_state": expected_state,
                "observed_reasons": assessment.reasons,
                "expected_reasons": expected_reasons,
                "manifest_emitted": manifest is not None,
                "exact": state_ok and reason_ok and manifest_ok,
            }
        )

    end_to_end = _end_to_end_cases()
    current_hashes = {name: _digest(path) for name, path in RUNTIME_FILES.items()}
    baseline = lock["runtime_baseline"]
    changed = sorted(
        name for name, digest in current_hashes.items() if baseline.get(name) != digest
    )
    expected_runtime_changes = {
        "program_data.py",
        "document_page_index.py",
        "document_sources.py",
    }
    expected_runtime_changes.update(
        _authorized_core_runtime_files(errors).intersection(RUNTIME_FILES)
    )
    unexpected = [name for name in changed if name not in expected_runtime_changes]
    if unexpected:
        errors.extend(f"unexpected_runtime_change:{name}" for name in unexpected)
    if not expected_runtime_changes.issubset(changed):
        errors.append("flattened_runtime_not_changed")
    if state_exact != len(cohort):
        errors.append("state_mismatch")
    if reason_exact != len(cohort):
        errors.append("reason_mismatch")
    if manifest_exact != len(cohort):
        errors.append("manifest_policy_mismatch")
    if not all(item["passed"] for item in end_to_end):
        errors.append("end_to_end_control_failed")

    return {
        "schema": "finhrm-qp3f-flattened-table-reconstruction-evaluation-v1",
        "classification": "development_only_flattened_table_reconstruction_evaluation",
        "implementation_gate_passed": not errors,
        "errors": errors,
        "frozen_qp3e": {
            "state_exact": state_exact,
            "reason_set_exact": reason_exact,
            "manifest_policy_exact": manifest_exact,
            "total": len(cohort),
            "by_scenario": dict(sorted(by_scenario.items())),
            "failures": [item for item in details if not item["exact"]],
            "details": details,
        },
        "end_to_end": {
            "pass": sum(item["passed"] for item in end_to_end),
            "total": len(end_to_end),
            "cases": end_to_end,
        },
        "runtime": {
            "hashes": current_hashes,
            "changed_from_qp3e_precommit": changed,
            "unexpected_changes": unexpected,
            "new_runtime_file": "src/hrm_finance_qa/flattened_table_reconstruction.py",
        },
        "invariants": {
            "question_text_used_by_reconstruction": False,
            "numeric_value_or_answer_used_by_reconstruction": False,
            "candidate_score_rank_or_order_used_by_reconstruction": False,
            "top_k_used_as_completeness_proof": False,
            "threshold_changed": False,
            "checkpoint_changed": False,
            "fine_tuning_performed": False,
            "global_validation_loaded": False,
            "sealed_validation_loaded": False,
            "blind_test_loaded": False,
            "phase_d_unlocked": False,
        },
        "authorized_next_step": "QP3G analyze remaining development-only flattened/OCR coverage failures and real-PDF proposal stability before any acceptance change; threshold remains frozen.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate QP3F implementation")
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = build_report(args.cohort, args.lock)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "state_exact": report["frozen_qp3e"]["state_exact"],
                "reason_set_exact": report["frozen_qp3e"]["reason_set_exact"],
                "manifest_policy_exact": report["frozen_qp3e"]["manifest_policy_exact"],
                "total": report["frozen_qp3e"]["total"],
                "end_to_end_pass": report["end_to_end"]["pass"],
                "end_to_end_total": report["end_to_end"]["total"],
            },
            sort_keys=True,
        )
    )
    print(f"implementation_gate_passed={report['implementation_gate_passed']}")
    return 0 if report["implementation_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
