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


DEFAULT_PRECOMMIT = ROOT / "config" / "finhrm_qp3a_column_role_proof_precommit_v1.json"
DEFAULT_OUTPUT = ROOT / "data" / "finhrm_qp3a_column_role_proof_development_v1.jsonl"
PROTECTED_PATH_TOKENS = {"validation", "blind", "holdout", "test"}


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


def _read_catalog(path: Path) -> list[dict[str, Any]]:
    _refuse_protected(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row.get("classification") != "development_only_synthetic_compiler_contract":
            raise ValueError("Catalog contains non-development data")
        if row.get("document_content_used") is not False or row.get("numeric_answer_used") is not False:
            raise ValueError("Catalog row uses document content or numeric answers")
    return rows


def _verify_frozen_inputs(precommit: dict[str, Any]) -> None:
    for item in precommit["frozen_inputs"].values():
        path = ROOT / item["path"]
        _refuse_protected(path)
        if _digest(path) != item["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {path}")


def _role_swap(role: str) -> str:
    return {"ACTUAL": "PLAN", "PLAN": "ACTUAL"}.get(role, role)


def _logical_groups(operands: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for operand in operands:
        group = {
            "metric": normalize_text(str(operand["metric"])),
            "period": str(operand["year"]),
            "entity_scope": str(operand.get("entity_scope", "reporting")),
            "unit_family": str(operand.get("unit_family", "currency")),
        }
        key = tuple(group.values())
        if key in seen:
            continue
        seen.add(key)
        output.append(group)
    return output


def _cell(
    group_id: str,
    logical: dict[str, str],
    logical_index: int,
    role: str,
    ordinal: int,
    *,
    value_token: str,
    extraction_method: str = "synthetic_native",
    physical_slot: int | None = None,
    role_conflict: bool = False,
) -> dict[str, Any]:
    slot = ordinal if physical_slot is None else physical_slot
    return {
        "cell_id": f"{group_id}:g{logical_index}:c{ordinal}",
        "document_id": f"synthetic:{group_id}",
        "page": logical_index + 1,
        "table_id": f"table:{group_id}",
        "row_id": f"row:{normalize_text(logical['metric']).replace(' ', '_')}",
        "column_id": f"column:{role.lower()}:{ordinal}",
        "metric": logical["metric"],
        "period": logical["period"],
        "column_role": role,
        "entity_scope": logical["entity_scope"],
        "unit_family": logical["unit_family"],
        "source_span": [1000 * (logical_index + 1) + 10 * slot, 1000 * (logical_index + 1) + 10 * slot + 3],
        "extraction_method": extraction_method,
        "extraction_value_token": value_token,
        "column_role_conflict": role_conflict,
        "synthetic_provenance": "precommitted_question_independent_evidence_structure"
    }


def _scenario_cells(
    group_id: str,
    operands: list[dict[str, str]],
    scenario_id: str,
) -> list[dict[str, Any]]:
    groups = _logical_groups(operands)
    if scenario_id in {"explicit_role_positive_complete", "explicit_role_swap_negative_complete"}:
        output: list[dict[str, Any]] = []
        for index, operand in enumerate(operands):
            logical = {
                "metric": normalize_text(str(operand["metric"])),
                "period": str(operand["year"]),
                "entity_scope": str(operand.get("entity_scope", "reporting")),
                "unit_family": str(operand.get("unit_family", "currency")),
            }
            role = str(operand["column_role"])
            if scenario_id == "explicit_role_swap_negative_complete":
                role = _role_swap(role)
            output.append(
                _cell(group_id, logical, index, role, index, value_token=f"TOKEN_{index}_{role}")
            )
        return output

    if scenario_id in {"no_role_unique_complete", "no_role_unique_inventory_unknown"}:
        return [
            _cell(group_id, logical, index, "ACTUAL", index, value_token=f"TOKEN_{index}_ACTUAL")
            for index, logical in enumerate(groups)
        ]

    if scenario_id == "no_role_cross_extractor_role_conflict_complete":
        output = []
        for index, logical in enumerate(groups):
            token = f"TOKEN_{index}_SAME_PHYSICAL"
            output.extend(
                (
                    _cell(
                        group_id, logical, index, "ACTUAL", 2 * index,
                        value_token=token, extraction_method="synthetic_native",
                        physical_slot=index, role_conflict=True,
                    ),
                    _cell(
                        group_id, logical, index, "PLAN", 2 * index + 1,
                        value_token=token, extraction_method="synthetic_ocr",
                        physical_slot=index, role_conflict=True,
                    ),
                )
            )
        return output

    same_value = scenario_id == "no_role_same_value_multi_role_conflict_complete"
    output = []
    for index, logical in enumerate(groups):
        common = f"TOKEN_{index}_SAME_VALUE"
        output.extend(
            (
                _cell(
                    group_id, logical, index, "ACTUAL", 2 * index,
                    value_token=common if same_value else f"TOKEN_{index}_ACTUAL",
                ),
                _cell(
                    group_id, logical, index, "PLAN", 2 * index + 1,
                    value_token=common if same_value else f"TOKEN_{index}_PLAN",
                ),
            )
        )
    if scenario_id == "no_role_conflict_permuted_complete":
        output.reverse()
    return output


def _group_specs(
    precommit: dict[str, Any],
    contract: dict[str, Any],
    catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    families = {item["family_id"]: item for item in contract["families"]}
    catalog_by_id = {item["example_id"]: item for item in catalog}
    groups: list[dict[str, Any]] = []
    for example_id in precommit["scope"]["deferred_example_ids"]:
        row = catalog_by_id[example_id]
        family = families[row["family_id"]]
        groups.append(
            {
                "group_id": row["family_id"],
                "source_type": "frozen_deferred_family",
                "source_example_id": example_id,
                "explicit_question": family["canonical_question"],
                "no_role_question": row["question"],
                "operation": family["expected"]["root_operation"],
                "operands": family["expected"]["operands"],
            }
        )
    for item in precommit["scope"]["generalization_specs"]:
        groups.append(
            {
                "group_id": item["family_id"],
                "source_type": "precommitted_metric_year_generalization",
                "source_example_id": "",
                "explicit_question": item["explicit_question"],
                "no_role_question": item["no_role_question"],
                "operation": item["operation"],
                "operands": item["operands"],
            }
        )
    return groups


def build_records(precommit_path: Path = DEFAULT_PRECOMMIT) -> list[dict[str, Any]]:
    precommit = _read_json(precommit_path)
    if precommit.get("status") != "precommitted_before_qp3_runtime_changes":
        raise ValueError("QP3A contract is not precommitted")
    _verify_frozen_inputs(precommit)
    contract = _read_json(ROOT / precommit["frozen_inputs"]["thesis_core_contract"]["path"])
    catalog = _read_catalog(ROOT / precommit["frozen_inputs"]["question_family_catalog"]["path"])
    scenarios = precommit["scenario_matrix"]
    records: list[dict[str, Any]] = []
    for group in _group_specs(precommit, contract, catalog):
        for scenario in scenarios:
            scenario_id = scenario["id"]
            query_kind = scenario["query_kind"]
            cells = _scenario_cells(group["group_id"], group["operands"], scenario_id)
            records.append(
                {
                    "schema": "finhrm-qp3a-column-role-proof-example-v1",
                    "classification": "development_only_synthetic_evidence_precommit",
                    "example_id": f"{group['group_id']}:{scenario_id}",
                    "group_id": group["group_id"],
                    "source_type": group["source_type"],
                    "source_example_id": group["source_example_id"],
                    "scenario_id": scenario_id,
                    "question": group["explicit_question"] if query_kind == "explicit_role" else group["no_role_question"],
                    "query_kind": query_kind,
                    "operation": group["operation"],
                    "query_column_roles": (
                        [str(item["column_role"]) for item in group["operands"]]
                        if query_kind == "explicit_role" else []
                    ),
                    "role_inventory_state": scenario["inventory"],
                    "evidence_cells": cells,
                    "expected_column_role_proof_state": scenario["expected_state"],
                    "expected_action": scenario["expected_action"],
                    "numeric_value_used": False,
                    "numeric_answer_used": False,
                    "document_content_used": False,
                    "candidate_score_used": False,
                    "candidate_order_is_signal": False,
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen QP3A synthetic evidence cohort")
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    records = build_records(args.precommit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    print(json.dumps({"examples": len(records), "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
