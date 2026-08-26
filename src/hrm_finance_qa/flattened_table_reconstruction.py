from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping


RECONSTRUCTABLE_COMPLETE = "RECONSTRUCTABLE_COMPLETE"
UNKNOWN = "UNKNOWN"
_ALLOWED_METHODS = {
    "synthetic_native_line",
    "synthetic_ocr_bbox",
    "synthetic_native_ocr_agreement",
    "native",
    "text",
    "ocr",
    "mineru",
}
_ALLOWED_CONFIDENCE = {"EXPLICIT", "INHERITED"}
_ALLOWED_HEADER_PROVENANCE = {
    "explicit_flattened_header",
    "inherited_flattened_header",
}


@dataclass(frozen=True)
class FlattenedReconstructionAssessment:
    state: str
    reasons: tuple[str, ...]


def _source_valid(source: Any) -> bool:
    if not isinstance(source, Mapping):
        return False
    span = source.get("source_span")
    bbox = source.get("bbox")
    return (
        bool(str(source.get("document_id", "")))
        and isinstance(source.get("page"), int)
        and int(source["page"]) > 0
        and isinstance(span, list)
        and len(span) == 2
        and all(isinstance(value, int) for value in span)
        and span[0] < span[1]
        and isinstance(source.get("source_sequence"), int)
        and str(source.get("extraction_method", "")) in _ALLOWED_METHODS
        and isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(value, (int, float)) for value in bbox)
        and bbox[0] < bbox[2]
        and bbox[1] < bbox[3]
    )


def _source_key(observation: Mapping[str, Any]) -> tuple[str, ...]:
    source = observation.get("source", {})
    if not isinstance(source, Mapping):
        return ("", "", "[]")
    return (
        str(source.get("document_id", "")),
        str(source.get("page", "")),
        json.dumps(source.get("source_span", []), separators=(",", ":")),
    )


def assess_flattened_table_reconstruction(
    record: Mapping[str, Any],
) -> FlattenedReconstructionAssessment:
    """Apply the frozen QP3E provenance contract without answer signals."""

    reasons: set[str] = set()
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
    header_columns = tuple(str(item.get("column_id", "")) for item in headers)
    if (
        not columns
        or len(columns) != len(set(columns))
        or len(headers) != len(columns)
        or len(header_columns) != len(set(header_columns))
        or set(header_columns) != set(columns)
    ):
        reasons.add("header_role_cardinality_mismatch")
    if record.get("declared_data_width") != len(columns):
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
        if not role or confidence not in _ALLOWED_CONFIDENCE:
            reasons.add("missing_explicit_column_role")
        if provenance not in _ALLOWED_HEADER_PROVENANCE:
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
        if not _source_valid(header.get("source")):
            reasons.add("missing_physical_provenance")
        header_roles[column_id] = role
        header_bands[column_id] = str(header.get("column_band_id", ""))

    observed_rows = tuple(str(item.get("row_id", "")) for item in row_observations)
    if (
        not rows
        or len(rows) != len(set(rows))
        or len(observed_rows) != len(set(observed_rows))
        or set(observed_rows) != set(rows)
    ):
        reasons.add("row_membership_conflict")
    for row in row_observations:
        if not _source_valid(row.get("source")):
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
        physical_rows.setdefault(str(cell.get("physical_cell_id", "")), set()).add(
            row_id
        )
        physical_spans.setdefault(_source_key(cell), set()).add(coordinate)
        if (
            column_id in header_bands
            and str(cell.get("column_band_id", "")) != header_bands[column_id]
        ):
            reasons.add("column_geometry_unaligned")
        source = cell.get("source")
        if not _source_valid(source):
            reasons.add("missing_physical_provenance")
        elif isinstance(source, Mapping):
            cell_sequences.append(int(source["source_sequence"]))
    if any(len({row for row in values if row}) > 1 for values in physical_rows.values()):
        reasons.add("row_membership_conflict")
    if any(len({value for value in values if value}) > 1 for values in physical_spans.values()):
        reasons.add("physical_span_conflict")

    header_sequences = [
        int(header["source"]["source_sequence"])
        for header in headers
        if _source_valid(header.get("source"))
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
        if not _source_valid(repeated.get("source")):
            reasons.add("missing_physical_provenance")
    if record.get("extractor_alignment_state") != "CONSISTENT":
        reasons.add("cross_extractor_layout_conflict")

    ordered = tuple(sorted(reasons))
    return FlattenedReconstructionAssessment(
        RECONSTRUCTABLE_COMPLETE if not ordered else UNKNOWN,
        ordered,
    )


def qp3d_manifest_from_flattened_record(
    record: Mapping[str, Any],
) -> dict[str, object] | None:
    """Materialize a QP3D manifest only after the QP3E proof succeeds."""

    assessment = assess_flattened_table_reconstruction(record)
    if assessment.state != RECONSTRUCTABLE_COMPLETE:
        return None
    headers = tuple(record.get("header_observations", ()))
    raw_headers = [
        {
            "column_id": str(header.get("column_id", "")),
            "header_path": [str(header.get("header_text", ""))],
            "header_path_provenance": [
                "inherited_merged_header"
                if header.get("role_confidence") == "INHERITED"
                else "explicit_header"
            ],
        }
        for header in headers
    ]
    cells = tuple(record.get("cell_observations", ()))
    physical = []
    for cell in cells:
        source = cell.get("source", {})
        physical.append(
            {
                "cell_coordinate": str(cell.get("cell_coordinate", "")),
                "document_id": str(source.get("document_id", "")),
                "page": int(source.get("page", 0)),
                "source_span": list(source.get("source_span", ())),
                "extraction_method": str(source.get("extraction_method", "")),
                "column_id": str(cell.get("column_id", "")),
            }
        )
    return {
        "schema": "finhrm-table-role-inventory-manifest-v1",
        "document_id": str(record.get("document_id", "")),
        "table_id": str(record.get("table_id", "")),
        "table_boundary_state": "COMPLETE",
        "header_alignment_state": "ALIGNED",
        "extractor_alignment_state": "CONSISTENT",
        "raw_header_columns": raw_headers,
        "declared_data_column_ids": list(
            str(value) for value in record.get("declared_data_column_ids", ())
        ),
        "expected_row_ids": list(
            str(value) for value in record.get("declared_row_ids", ())
        ),
        "observed_cell_coordinates": [
            str(cell.get("cell_coordinate", "")) for cell in cells
        ],
        "physical_observations": physical,
        "flattened_reconstruction_state": assessment.state,
        "flattened_reconstruction_reasons": list(assessment.reasons),
    }
