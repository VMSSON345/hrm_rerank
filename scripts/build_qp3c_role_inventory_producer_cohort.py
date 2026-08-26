from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config" / "finhrm_qp3c_role_inventory_producer_precommit_v1.json"
DEFAULT_OUTPUT = ROOT / "data" / "finhrm_qp3c_role_inventory_producer_development_v1.jsonl"

ROLE_LABELS = {
    "ACTUAL": "Thuc hien / Actual",
    "PLAN": "Ke hoach / Plan",
    "REVISED_PLAN": "Ke hoach dieu chinh / Revised plan",
    "FORECAST": "Du bao / Forecast",
    "ACTUAL_TO_PLAN_RATIO": "TH/KH",
    "ACTUAL_TO_PRIOR_RATIO": "TH/Nam truoc",
}
SCENARIO_REASONS = {
    "complete_explicit": [],
    "complete_inherited_with_anchor": [],
    "complete_permuted": [],
    "missing_header_column": ["header_column_coverage_mismatch"],
    "unknown_role_label": ["missing_explicit_column_role"],
    "conflicting_header_role": [
        "header_role_conflict",
        "missing_explicit_column_role",
    ],
    "cross_extractor_role_conflict": [
        "cross_extractor_role_conflict",
        "physical_role_conflict",
    ],
    "incomplete_cell_coverage": ["cell_coverage_mismatch"],
    "partial_table_boundary": ["incomplete_table_boundary"],
    "unaligned_header_width": ["header_alignment_unproven"],
    "missing_physical_provenance": ["missing_physical_provenance"],
    "duplicate_physical_role_conflict": ["physical_role_conflict"],
}


def _base_record(layout: dict[str, Any]) -> dict[str, Any]:
    layout_id = str(layout["id"])
    document_id = f"synthetic:{layout_id}"
    table_id = f"table:{layout_id}"
    column_ids = [f"col:{index}:{role.lower()}" for index, role in enumerate(layout["roles"])]
    row_ids = [f"row:{index}" for index in range(int(layout["rows"]))]
    header_columns = []
    for column_id, role in zip(column_ids, layout["roles"]):
        header_columns.append(
            {
                "column_id": column_id,
                "header_path": [ROLE_LABELS[str(role)]],
                "header_path_provenance": ["explicit_header"],
                "column_role": str(role),
                "column_role_confidence": "EXPLICIT",
                "column_role_conflict": False,
                "inheritance_anchor_id": "",
            }
        )
    expected_coordinates = [
        f"{row_id}|{column_id}" for row_id in row_ids for column_id in column_ids
    ]
    physical_observations = []
    for index, coordinate in enumerate(expected_coordinates):
        column_id = coordinate.split("|", 1)[1]
        role = next(
            header["column_role"]
            for header in header_columns
            if header["column_id"] == column_id
        )
        physical_observations.append(
            {
                "cell_coordinate": coordinate,
                "document_id": document_id,
                "page": 1 + index // max(1, len(column_ids) * 2),
                "source_span": [1000 + index * 10, 1004 + index * 10],
                "extraction_method": "synthetic_native",
                "column_role": role,
            }
        )
    return {
        "schema": "finhrm-qp3c-role-inventory-producer-example-v1",
        "classification": "development_only_synthetic_table_provenance_precommit",
        "example_id": "",
        "layout_id": layout_id,
        "layout_type": str(layout["layout"]),
        "scenario_id": "",
        "document_id": document_id,
        "table_id": table_id,
        "table_boundary_state": "COMPLETE",
        "header_alignment_state": "ALIGNED",
        "extractor_alignment_state": "CONSISTENT",
        "expected_data_column_ids": column_ids,
        "expected_row_ids": row_ids,
        "header_columns": header_columns,
        "inheritance_anchors": [],
        "expected_cell_coordinates": expected_coordinates,
        "observed_cell_coordinates": list(expected_coordinates),
        "physical_observations": physical_observations,
        "expected_inventory_state": "",
        "expected_reason_set": [],
        "numeric_value_used": False,
        "numeric_answer_used": False,
        "candidate_score_used": False,
        "candidate_order_is_signal": False,
        "question_text_used": False,
        "document_content_used": False,
        "synthetic_provenance": "precommitted_question_independent_table_structure",
    }


def _different_role(role: str) -> str:
    return "PLAN" if role != "PLAN" else "ACTUAL"


def _apply_scenario(record: dict[str, Any], scenario_id: str) -> None:
    headers = record["header_columns"]
    physical = record["physical_observations"]
    if scenario_id == "complete_inherited_with_anchor":
        header = headers[0]
        anchor_id = f"anchor:{header['column_id']}"
        header["header_path_provenance"] = ["inherited_merged_header"]
        header["column_role_confidence"] = "INHERITED"
        header["inheritance_anchor_id"] = anchor_id
        record["inheritance_anchors"].append(
            {
                "anchor_id": anchor_id,
                "column_role": header["column_role"],
                "header_text": header["header_path"][0],
                "provenance": "explicit_header",
            }
        )
    elif scenario_id == "complete_permuted":
        headers.reverse()
        record["observed_cell_coordinates"].reverse()
        physical.reverse()
    elif scenario_id == "missing_header_column":
        headers.pop()
    elif scenario_id == "unknown_role_label":
        headers[0].update(
            {
                "header_path": ["Nam hien tai"],
                "column_role": "",
                "column_role_confidence": "UNKNOWN",
            }
        )
    elif scenario_id == "conflicting_header_role":
        headers[0].update(
            {
                "header_path": ["Ke hoach va Thuc hien"],
                "column_role": "",
                "column_role_confidence": "CONFLICT",
                "column_role_conflict": True,
            }
        )
    elif scenario_id == "cross_extractor_role_conflict":
        record["extractor_alignment_state"] = "CONFLICT"
        duplicate = deepcopy(physical[0])
        duplicate["extraction_method"] = "synthetic_ocr"
        duplicate["column_role"] = _different_role(duplicate["column_role"])
        physical.append(duplicate)
    elif scenario_id == "incomplete_cell_coverage":
        missing = record["observed_cell_coordinates"].pop()
        record["physical_observations"] = [
            item for item in physical if item["cell_coordinate"] != missing
        ]
    elif scenario_id == "partial_table_boundary":
        record["table_boundary_state"] = "PARTIAL"
    elif scenario_id == "unaligned_header_width":
        record["header_alignment_state"] = "UNALIGNED"
    elif scenario_id == "missing_physical_provenance":
        physical[0]["source_span"] = []
    elif scenario_id == "duplicate_physical_role_conflict":
        duplicate = deepcopy(physical[0])
        duplicate["column_role"] = _different_role(duplicate["column_role"])
        physical.append(duplicate)


def build_records(precommit: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    scenarios = {item["id"]: item for item in precommit["scenario_matrix"]}
    for layout in precommit["layout_specs"]:
        for scenario_id, scenario in scenarios.items():
            record = _base_record(layout)
            record["example_id"] = f"{layout['id']}:{scenario_id}"
            record["scenario_id"] = scenario_id
            _apply_scenario(record, scenario_id)
            record["expected_inventory_state"] = scenario["expected_state"]
            record["expected_reason_set"] = SCENARIO_REASONS[scenario_id]
            records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the frozen QP3C synthetic provenance cohort")
    parser.add_argument("--precommit", type=Path, default=DEFAULT_PRECOMMIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    precommit = json.loads(args.precommit.read_text(encoding="utf-8"))
    records = build_records(precommit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    print(f"wrote={len(records)} path={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
