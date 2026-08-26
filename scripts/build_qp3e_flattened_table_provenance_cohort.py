from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRECOMMIT = ROOT / "config" / "finhrm_qp3e_flattened_table_provenance_precommit_v1.json"
DEFAULT_OUTPUT = ROOT / "data" / "finhrm_qp3e_flattened_table_provenance_development_v1.jsonl"

ROLE_LABELS = {
    "ACTUAL": "Thuc hien / Actual",
    "PLAN": "Ke hoach / Plan",
    "PRIOR_ACTUAL": "Thuc hien ky truoc / Prior actual",
    "REVISED_PLAN": "Ke hoach dieu chinh / Revised plan",
    "FORECAST": "Du bao / Forecast",
    "ACTUAL_TO_PLAN_RATIO": "TH/KH / Actual to plan",
    "ACTUAL_TO_PRIOR_RATIO": "TH/Ky truoc / Actual to prior",
}
SCENARIO_REASONS = {
    "complete_explicit": [],
    "complete_repeated_header_agreement": [],
    "complete_permuted": [],
    "partial_source_boundary": ["incomplete_source_boundary"],
    "open_table_continuation": ["incomplete_table_continuation"],
    "header_role_cardinality_mismatch": ["header_role_cardinality_mismatch"],
    "header_data_width_mismatch": ["header_data_width_mismatch"],
    "unknown_header_role": ["missing_explicit_column_role"],
    "missing_row_cell": ["cell_coverage_mismatch"],
    "overlapping_cell_span": ["physical_span_conflict"],
    "ambiguous_row_membership": ["row_membership_conflict"],
    "source_order_conflict": ["source_order_unproven"],
    "cross_extractor_layout_conflict": ["cross_extractor_layout_conflict"],
    "repeated_header_role_conflict": ["repeated_header_role_conflict"],
    "missing_physical_provenance": ["missing_physical_provenance"],
    "column_geometry_unaligned": ["column_geometry_unaligned"],
}


def _source(
    *, document_id: str, method: str, sequence: int, span_start: int, band: int
) -> dict[str, Any]:
    left = 100 + band * 120
    return {
        "document_id": document_id,
        "page": 1,
        "source_span": [span_start, span_start + 7],
        "source_sequence": sequence,
        "extraction_method": method,
        "bbox": [left, 40 + sequence, left + 90, 52 + sequence],
    }


def _base_record(layout: dict[str, Any]) -> dict[str, Any]:
    layout_id = str(layout["id"])
    document_id = f"synthetic-flat:{layout_id}"
    table_id = f"flat-table:{layout_id}"
    method = str(layout["extractor"])
    column_ids = [
        f"column:{index}:{str(role).lower()}"
        for index, role in enumerate(layout["roles"])
    ]
    row_ids = [f"row:{index}" for index in range(int(layout["rows"]))]
    headers: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    for ordinal, (column_id, role) in enumerate(zip(column_ids, layout["roles"])):
        inherited = layout_id == "F05_MERGED_ROLE_HEADER" and ordinal == 0
        anchor_id = f"anchor:{column_id}" if inherited else ""
        headers.append(
            {
                "column_id": column_id,
                "column_ordinal": ordinal,
                "column_band_id": f"band:{ordinal}",
                "header_text": ROLE_LABELS[str(role)],
                "canonical_role": str(role),
                "role_confidence": "INHERITED" if inherited else "EXPLICIT",
                "header_provenance": (
                    "inherited_flattened_header"
                    if inherited
                    else "explicit_flattened_header"
                ),
                "inheritance_anchor_id": anchor_id,
                "source": _source(
                    document_id=document_id,
                    method=method,
                    sequence=10 + ordinal,
                    span_start=100 + ordinal * 10,
                    band=ordinal,
                ),
            }
        )
        if inherited:
            anchors.append(
                {
                    "anchor_id": anchor_id,
                    "canonical_role": str(role),
                    "header_text": ROLE_LABELS[str(role)],
                    "provenance": "explicit_flattened_header",
                }
            )

    row_observations: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    width = len(column_ids)
    for row_index, row_id in enumerate(row_ids):
        row_observations.append(
            {
                "row_id": row_id,
                "descriptor_id": f"descriptor:{row_index}",
                "source": _source(
                    document_id=document_id,
                    method=method,
                    sequence=100 + row_index * (width + 1),
                    span_start=1000 + row_index * 100,
                    band=-1,
                ),
            }
        )
        for column_index, column_id in enumerate(column_ids):
            coordinate = f"{row_id}|{column_id}"
            sequence = 101 + row_index * (width + 1) + column_index
            cells.append(
                {
                    "cell_coordinate": coordinate,
                    "row_id": row_id,
                    "column_id": column_id,
                    "column_ordinal": column_index,
                    "column_band_id": f"band:{column_index}",
                    "physical_cell_id": f"physical:{row_index}:{column_index}",
                    "source": _source(
                        document_id=document_id,
                        method=method,
                        sequence=sequence,
                        span_start=1100 + row_index * 100 + column_index * 10,
                        band=column_index,
                    ),
                }
            )

    return {
        "schema": "finhrm-qp3e-flattened-table-provenance-example-v1",
        "classification": "development_only_synthetic_flattened_table_provenance_precommit",
        "example_id": "",
        "layout_id": layout_id,
        "layout_type": str(layout["layout"]),
        "scenario_id": "",
        "document_id": document_id,
        "table_id": table_id,
        "source_boundary_state": "COMPLETE",
        "table_continuation_state": "CLOSED",
        "extractor_alignment_state": "CONSISTENT",
        "declared_data_width": width,
        "declared_data_column_ids": column_ids,
        "declared_row_ids": row_ids,
        "header_observations": headers,
        "inheritance_anchors": anchors,
        "row_observations": row_observations,
        "cell_observations": cells,
        "repeated_header_observations": [],
        "expected_reconstruction_state": "",
        "expected_reason_set": [],
        "numeric_value_used": False,
        "numeric_answer_used": False,
        "candidate_score_used": False,
        "candidate_rank_used": False,
        "candidate_order_is_signal": False,
        "question_text_used": False,
        "real_document_content_used": False,
        "top_k_used": False,
        "synthetic_provenance": "precommitted_question_independent_flattened_structure",
    }


def _different_role(role: str) -> str:
    return "PLAN" if role != "PLAN" else "ACTUAL"


def _repeated_header(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "segment_id": "segment:repeat:1",
        "column_role_map": {
            str(header["column_id"]): str(header["canonical_role"])
            for header in record["header_observations"]
        },
        "source": _source(
            document_id=str(record["document_id"]),
            method=str(record["header_observations"][0]["source"]["extraction_method"]),
            sequence=50,
            span_start=500,
            band=0,
        ),
    }


def _apply_scenario(record: dict[str, Any], scenario_id: str) -> None:
    headers = record["header_observations"]
    rows = record["row_observations"]
    cells = record["cell_observations"]
    if scenario_id == "complete_repeated_header_agreement":
        record["repeated_header_observations"].append(_repeated_header(record))
    elif scenario_id == "complete_permuted":
        headers.reverse()
        rows.reverse()
        cells.reverse()
    elif scenario_id == "partial_source_boundary":
        record["source_boundary_state"] = "PARTIAL"
    elif scenario_id == "open_table_continuation":
        record["table_continuation_state"] = "OPEN"
    elif scenario_id == "header_role_cardinality_mismatch":
        headers.pop()
    elif scenario_id == "header_data_width_mismatch":
        record["declared_data_width"] += 1
    elif scenario_id == "unknown_header_role":
        headers[0].update(
            {
                "canonical_role": "",
                "role_confidence": "UNKNOWN",
                "header_text": "Current period",
            }
        )
    elif scenario_id == "missing_row_cell":
        cells.pop()
    elif scenario_id == "overlapping_cell_span":
        duplicate = deepcopy(cells[1])
        duplicate["source"] = deepcopy(cells[0]["source"])
        cells.append(duplicate)
    elif scenario_id == "ambiguous_row_membership":
        duplicate = deepcopy(cells[0])
        second_row = str(record["declared_row_ids"][1])
        duplicate["row_id"] = second_row
        duplicate["cell_coordinate"] = f"{second_row}|{duplicate['column_id']}"
        duplicate["source"]["source_span"] = [9000, 9007]
        duplicate["source"]["source_sequence"] = 900
        cells.append(duplicate)
    elif scenario_id == "source_order_conflict":
        headers[0]["source"]["source_sequence"] = 9999
    elif scenario_id == "cross_extractor_layout_conflict":
        record["extractor_alignment_state"] = "CONFLICT"
    elif scenario_id == "repeated_header_role_conflict":
        repeated = _repeated_header(record)
        first_column = str(record["declared_data_column_ids"][0])
        repeated["column_role_map"][first_column] = _different_role(
            repeated["column_role_map"][first_column]
        )
        record["repeated_header_observations"].append(repeated)
    elif scenario_id == "missing_physical_provenance":
        cells[0]["source"]["source_span"] = []
    elif scenario_id == "column_geometry_unaligned":
        cells[0]["column_band_id"] = "band:unresolved"


def build_records(precommit: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios = {item["id"]: item for item in precommit["scenario_matrix"]}
    records: list[dict[str, Any]] = []
    for layout in precommit["layout_specs"]:
        for scenario_id, scenario in scenarios.items():
            record = _base_record(layout)
            record["example_id"] = f"{layout['id']}:{scenario_id}"
            record["scenario_id"] = scenario_id
            _apply_scenario(record, scenario_id)
            record["expected_reconstruction_state"] = scenario["expected_state"]
            record["expected_reason_set"] = SCENARIO_REASONS[scenario_id]
            records.append(record)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the frozen QP3E flattened-table provenance cohort"
    )
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
