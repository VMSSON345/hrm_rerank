from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
import re
from typing import Any, Mapping, Sequence

from .ontology import normalize_text
from .program_data import NumericCandidate
from .question_planner import QuestionPlan


EVIDENCE_SCHEMA_VERSION = "evidence-cell-header-role-v2"
_EVIDENCE_SCHEMA_CONTRACT = {
    "version": EVIDENCE_SCHEMA_VERSION,
    "column_roles": (
        "actual",
        "plan",
        "revised_plan",
        "forecast",
        "actual_to_plan_ratio",
        "actual_to_prior_ratio",
    ),
    "confidence_states": ("EXPLICIT", "INHERITED", "CONFLICT", "UNKNOWN"),
    "conflict_policy": "same-logical-cell-cross-extractor-role-disagreement",
    "unknown_policy": "do-not-infer-from-column-order",
}
EVIDENCE_SCHEMA_FINGERPRINT = sha256(
    json.dumps(
        _EVIDENCE_SCHEMA_CONTRACT,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _context_field(context: str, name: str) -> str:
    match = re.search(
        rf"(?im)(?:^|\n){re.escape(name)}:\s*(.*?)(?=\n[A-Z_]+:|$)",
        context,
    )
    return match.group(1).strip() if match else ""


def _markdown_cells(line: str) -> list[str]:
    stripped = line.strip()
    if "|" not in stripped:
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _marked_cell_index(context: str) -> tuple[int | None, int]:
    """Return the tagged cell index and data-row width from ``CELL`` context."""

    row = _context_field(context, "CELL")
    cells = _markdown_cells(row)
    if not cells:
        return None, 0
    for index, cell in enumerate(cells):
        if "<VALUE>" in cell.upper():
            return index, len(cells)
    return None, len(cells)


def _aligned_header_cells(raw: str, data_width: int) -> list[str]:
    cells = _markdown_cells(raw)
    if not cells or data_width <= 0:
        return cells
    if len(cells) < data_width:
        return [""] * (data_width - len(cells)) + cells
    if len(cells) > data_width:
        return cells[len(cells) - data_width :]
    return cells


def _header_path(
    candidate: NumericCandidate,
    column: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Reconstruct the raw hierarchical header path for a numeric cell.

    Blank upper-level cells are treated as a Markdown/PDF merged-cell
    continuation and filled from the nearest explicit cell on the left.  This
    is provenance, not role inference: the eventual role still has to be
    stated by a header token.  Alignment follows the extractor's existing
    right-alignment rule for rows where OCR dropped an empty descriptor cell.
    """

    tagged_index, data_width = _marked_cell_index(candidate.context)
    raw_header = _context_field(candidate.context, "HEADER")
    header_lines = [
        item.strip() for item in raw_header.split(" || ") if item.strip()
    ]
    path: list[str] = []
    provenance: list[str] = []
    if tagged_index is not None:
        for raw_line in header_lines:
            cells = _aligned_header_cells(raw_line, data_width)
            if tagged_index >= len(cells):
                continue
            part = cells[tagged_index].strip()
            source = "explicit_header"
            if not part:
                for previous in range(tagged_index - 1, -1, -1):
                    if cells[previous].strip():
                        part = cells[previous].strip()
                        source = "inherited_merged_header"
                        break
            normalized = normalize_text(part)
            if not part or normalized in {
                "chi tieu",
                "noi dung",
                "item",
                "indicator",
                "column name",
            }:
                continue
            if normalize_text(part) not in {normalize_text(value) for value in path}:
                path.append(part)
                provenance.append(source)

    # Legacy/native contexts may carry only COLUMN.  Preserve that raw label
    # rather than fabricating hierarchy.  Slash-separated labels are extractor
    # output, so each component remains explicit provenance.
    if not path and column:
        for part in (item.strip() for item in column.split("/") if item.strip()):
            if normalize_text(part) not in {normalize_text(value) for value in path}:
                path.append(part)
                provenance.append("explicit_column_label")
    return tuple(path), tuple(provenance)


def _contains_phrase(value: str, phrases: tuple[str, ...]) -> bool:
    padded = f" {normalize_text(value)} "
    return any(f" {phrase} " in padded for phrase in phrases)


def _column_role_from_header(value: str) -> str:
    """Map explicit header vocabulary to a canonical column role.

    The function intentionally does not inspect row labels, question text,
    numeric values, or left/right order.  Ambiguous mixtures return an empty
    role and are handled as conflicts by the caller.
    """

    normalized = normalize_text(value)
    if not normalized:
        return ""
    padded = f" {normalized} "
    tokens = set(normalized.split())
    has_actual = _contains_phrase(
        value, ("thuc hien", "thuc te", "actual", "realized", "realised")
    ) or "th" in tokens
    has_plan = _contains_phrase(
        value, ("ke hoach", "plan", "target", "muc tieu")
    ) or "kh" in tokens
    has_revised = _contains_phrase(
        value,
        (
            "ke hoach dieu chinh",
            "ke hoach sua doi",
            "revised plan",
            "adjusted plan",
            "kh dieu chinh",
        ),
    )
    has_forecast = _contains_phrase(
        value, ("du bao", "uoc tinh", "forecast", "estimate", "estimated")
    )
    ratio_marker = bool(
        "%" in value
        or "/" in value
        or "ty le" in normalized
        or "so voi" in normalized
        or "hoan thanh" in normalized
        or "comparison" in normalized
    )
    prior_marker = _contains_phrase(
        value,
        (
            "nam truoc",
            "ky truoc",
            "prior",
            "previous",
            "actual prior",
            "th nam truoc",
        ),
    )
    years = re.findall(r"\b(?:19|20)\d{2}\b", normalized)

    if ratio_marker and has_actual and has_plan:
        return "actual_to_plan_ratio"
    if ratio_marker and has_actual and (
        prior_marker
        or len(set(years)) >= 2
        or ("%" in value and "/" in value and bool(years))
    ):
        return "actual_to_prior_ratio"
    if has_revised:
        return "revised_plan"
    if has_forecast:
        return "forecast"
    if has_actual and has_plan:
        return ""
    if has_actual:
        return "actual"
    if has_plan:
        return "plan"
    # A bare year, percentage, or position contains no column-role proof.
    if padded:
        return ""
    return ""


def _resolve_column_role(
    column: str,
    header_path: tuple[str, ...],
    header_provenance: tuple[str, ...],
) -> tuple[str, str, str, tuple[str, ...], bool]:
    observations: list[tuple[str, str, str]] = []
    if column:
        role = _column_role_from_header(column)
        if role:
            observations.append((role, "column_label", column))
    for index, part in enumerate(header_path):
        role = _column_role_from_header(part)
        if role:
            source = (
                header_provenance[index]
                if index < len(header_provenance)
                else "explicit_header"
            )
            observations.append((role, source, part))

    if len(header_path) > 1:
        composite = " / ".join(header_path)
        role = _column_role_from_header(composite)
        if role:
            observations.append((role, "header_path_composite", composite))

    roles = {role for role, _, _ in observations}
    evidence = tuple(dict.fromkeys(raw for _, _, raw in observations))
    resolved_specialization = ""
    if "actual_to_plan_ratio" in roles and roles <= {
        "actual_to_plan_ratio",
        "actual",
        "plan",
    }:
        resolved_specialization = "actual_to_plan_ratio"
    elif "actual_to_prior_ratio" in roles and roles <= {
        "actual_to_prior_ratio",
        "actual",
    }:
        resolved_specialization = "actual_to_prior_ratio"
    elif "revised_plan" in roles and roles <= {"revised_plan", "plan"}:
        resolved_specialization = "revised_plan"
    if resolved_specialization:
        roles = {resolved_specialization}
    if len(roles) > 1:
        return "", "conflicting_header_evidence", "CONFLICT", evidence, True
    if not roles:
        return "", "", "UNKNOWN", (), False
    role = next(iter(roles))
    sources = tuple(dict.fromkeys(source for _, source, _ in observations))
    confidence = (
        "INHERITED"
        if sources and all(source == "inherited_merged_header" for source in sources)
        else "EXPLICIT"
    )
    return role, "+".join(sources), confidence, evidence, False


def _metric_key(label: str) -> str:
    normalized = normalize_text(label)
    normalized = re.sub(r"^(?:[ivxlcdm]+|\d+)[.)\s-]+", "", normalized)
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalize_text(value)).strip("_")


def _numeric_signatures(value: str) -> set[str]:
    signatures: set[str] = set()
    for token in re.findall(r"(?<!\w)[+-]?\d[\d.,\s]*", value):
        digits = re.sub(r"\D", "", token)
        if not digits or (len(digits) == 4 and digits.startswith(("19", "20"))):
            continue
        signatures.add(digits.lstrip("0") or "0")
    return signatures


def _tagged_value_signature(context: str) -> str:
    match = re.search(r"(?is)<VALUE>\s*(.*?)\s*</VALUE>", context)
    if match is None:
        return ""
    signatures = _numeric_signatures(match.group(1))
    return next(iter(signatures), "")


def _par_value_provenance_matches(metric_label: str, context: str) -> bool:
    """Reject a table-neighbour value mislabeled as the disclosed par value.

    PDF table extraction can attach the full row label (including a disclosed
    par value) to every numeric cell in that row.  When both the label and the
    tagged value contain numbers, the tagged value must be the number stated
    by that label.  Sparse/legacy evidence without a tagged value remains
    usable instead of being rejected solely for missing provenance metadata.
    """

    tagged = _tagged_value_signature(context)
    if not tagged:
        return True
    disclosed = _numeric_signatures(metric_label)
    return not disclosed or tagged in disclosed


def _unit_key(
    unit: str,
    *,
    is_percent: bool,
    metric_label: str = "",
    context: str = "",
) -> str:
    # ``normalize_text`` deliberately removes punctuation, including ``%``.
    # Preserve an explicit percent marker before normalization so it remains
    # typed evidence rather than being downgraded to UNKNOWN.
    metric = normalize_text(metric_label)
    normalized = normalize_text(unit)
    compact_unit = re.sub(r"[^a-z0-9]+", "", normalized)
    metric_declares_par_value = (
        "menh gia" in metric
        and any(marker in metric for marker in ("co phieu", "co phan"))
        and _par_value_provenance_matches(metric_label, context)
    )
    if metric_declares_par_value or (
        (
            "menh gia" in metric
            or "pershare" in compact_unit
            or "cophieu" in compact_unit
            or "cophan" in compact_unit
        )
        and any(marker in normalized for marker in ("vnd", "dong"))
        and (
            "menh gia" not in metric
            or _par_value_provenance_matches(metric_label, context)
        )
    ):
        return "currency_per_share"
    if any(
        phrase in metric
        for phrase in (
            "so luong co phieu",
            "so luong co phan",
            "co phieu dang luu hanh",
            "co phan dang luu hanh",
            "co phieu da phat hanh",
            "co phan da phat hanh",
        )
    ):
        return "share_count"
    if normalized in {"shares", "share", "co phieu", "co phan"}:
        return "share_count"
    if is_percent or "%" in unit or "％" in unit:
        return "percent"
    if normalized in {"phan tram", "percent", "pct"}:
        return "percent"
    if "he so" in metric or normalized in {"x", "lan", "ratio"}:
        return "ratio"
    if any(marker in normalized for marker in ("vnd", "dong", "trieu", "ty")):
        return "currency"
    return _identifier(normalized)


class EvidenceRelation(str, Enum):
    SAME_METRIC = "SAME_METRIC"
    SAME_ENTITY = "SAME_ENTITY"
    SAME_TABLE = "SAME_TABLE"
    SAME_ROW = "SAME_ROW"
    SAME_UNIT = "SAME_UNIT"
    PREVIOUS_PERIOD = "PREVIOUS_PERIOD"
    NEXT_PERIOD = "NEXT_PERIOD"
    PART_OF_TOTAL = "PART_OF_TOTAL"


@dataclass(frozen=True)
class EvidenceEdge:
    source: int
    target: int
    relation: EvidenceRelation


@dataclass(frozen=True)
class TableRoleInventory:
    source_name: str
    table_id: str
    state: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceGraph:
    cells: tuple["EvidenceCell", ...]
    edges: tuple[EvidenceEdge, ...]
    column_role_inventory_state: str = ""
    column_role_inventory_reasons: tuple[str, ...] = ()
    table_role_inventories: tuple[TableRoleInventory, ...] = ()

    def relations(self, source: int, target: int) -> frozenset[EvidenceRelation]:
        return frozenset(
            edge.relation
            for edge in self.edges
            if edge.source == source and edge.target == target
        )

    def has_relation(
        self,
        source: int,
        target: int,
        relation: EvidenceRelation,
    ) -> bool:
        return relation in self.relations(source, target)


@dataclass(frozen=True)
class EvidenceCell:
    index: int
    value: Decimal
    metric_label: str
    metric_key: str
    column: str
    period: str
    explicit_years: tuple[str, ...]
    period_role: str
    entity: str
    section: str
    table_id: str
    source_name: str
    page: int | None
    value_status: str
    unit: str = ""
    unit_key: str = ""
    row_id: str = ""
    column_id: str = ""
    entity_id: str = ""
    subtable_id: str = ""
    row_type: str = ""
    is_percent: bool = False
    is_year_like: bool = False
    relevance: float = 0.0
    column_role: str = ""
    column_role_source: str = ""
    column_role_confidence: str = "UNKNOWN"
    column_role_evidence: tuple[str, ...] = ()
    column_role_conflict: bool = False
    header_path: tuple[str, ...] = ()
    header_path_provenance: tuple[str, ...] = ()
    raw_header: tuple[str, ...] = ()
    extraction_method: str = ""
    source_span: tuple[int, int] | None = None

    @property
    def embedded_entity(self) -> bool:
        return normalize_text(self.entity_id or self.entity).startswith("entity ")

    def effective_years(self, plan: QuestionPlan) -> set[str]:
        explicit = set(self.explicit_years)
        requested = sorted(plan.requested_years, key=int)
        if explicit and requested and not (
            plan.temporal_change and len(requested) >= 2
        ):
            return explicit
        if self.period_role == "prior":
            if requested:
                return {
                    requested[-2]
                    if len(requested) >= 2
                    else str(int(requested[-1]) - 1)
                }
            if explicit:
                return {str(max(int(item) for item in explicit) - 1)}
        if self.period_role == "current" and not explicit and requested:
            return {requested[-1]}
        return explicit


@dataclass(frozen=True)
class RoleInventoryProduction:
    state: str
    reasons: tuple[str, ...]
    tables_audited: int = 0


def _role_inventory_physical_key(item: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(item.get("document_id", "")),
        str(item.get("page", "")),
        json.dumps(item.get("source_span", []), separators=(",", ":")),
        str(item.get("cell_coordinate", "")),
    )


def assess_table_role_inventory(
    record: Mapping[str, Any],
) -> RoleInventoryProduction:
    """Apply the frozen QP3C table-level completeness contract."""

    reasons: set[str] = set()
    if record.get("table_boundary_state") != "COMPLETE":
        reasons.add("incomplete_table_boundary")
    if record.get("header_alignment_state") != "ALIGNED":
        reasons.add("header_alignment_unproven")

    expected_columns = tuple(
        str(value) for value in record.get("expected_data_column_ids", ())
    )
    headers = tuple(record.get("header_columns", ()))
    observed_columns = tuple(
        str(item.get("column_id", "")) for item in headers
    )
    if (
        not expected_columns
        or len(expected_columns) != len(set(expected_columns))
        or len(observed_columns) != len(set(observed_columns))
        or set(expected_columns) != set(observed_columns)
    ):
        reasons.add("header_column_coverage_mismatch")

    allowed_provenance = {
        "explicit_header",
        "inherited_merged_header",
        "explicit_column_label",
    }
    anchors = {
        str(item.get("anchor_id", "")): item
        for item in record.get("inheritance_anchors", ())
    }
    for header in headers:
        role = str(header.get("column_role", ""))
        confidence = str(header.get("column_role_confidence", ""))
        path = tuple(str(value) for value in header.get("header_path", ()))
        provenance = tuple(
            str(value)
            for value in header.get("header_path_provenance", ())
        )
        if header.get("column_role_conflict") or confidence == "CONFLICT":
            reasons.add("header_role_conflict")
        if not role or confidence not in {"EXPLICIT", "INHERITED"}:
            reasons.add("missing_explicit_column_role")
        if (
            not path
            or len(path) != len(provenance)
            or any(value not in allowed_provenance for value in provenance)
        ):
            reasons.add("invalid_header_provenance")
        if confidence == "INHERITED":
            anchor = anchors.get(
                str(header.get("inheritance_anchor_id", ""))
            )
            if (
                anchor is None
                or anchor.get("provenance") != "explicit_header"
                or str(anchor.get("column_role", "")) != role
                or not str(anchor.get("header_text", ""))
            ):
                reasons.add("missing_inheritance_anchor")

    expected_rows = tuple(
        str(value) for value in record.get("expected_row_ids", ())
    )
    declared_expected = {
        str(value) for value in record.get("expected_cell_coordinates", ())
    }
    cartesian_expected = {
        f"{row_id}|{column_id}"
        for row_id in expected_rows
        for column_id in expected_columns
    }
    observed = {
        str(value) for value in record.get("observed_cell_coordinates", ())
    }
    if (
        not expected_rows
        or declared_expected != cartesian_expected
        or observed != cartesian_expected
    ):
        reasons.add("cell_coverage_mismatch")

    physical = tuple(record.get("physical_observations", ()))
    physical_coordinates = {
        str(item.get("cell_coordinate", "")) for item in physical
    }
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
        physical_roles.setdefault(
            _role_inventory_physical_key(item), set()
        ).add(str(item.get("column_role", "")))
    if any(
        len({role for role in roles if role}) > 1
        for roles in physical_roles.values()
    ):
        reasons.add("physical_role_conflict")
    if record.get("extractor_alignment_state") == "CONFLICT":
        reasons.add("cross_extractor_role_conflict")

    ordered = tuple(sorted(reasons))
    return RoleInventoryProduction(
        "EXPLICIT_COMPLETE" if not ordered else "UNKNOWN",
        ordered,
        1,
    )


def _materialize_runtime_role_inventory_record(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    raw_headers = tuple(manifest.get("raw_header_columns", ()))
    derived_headers: list[dict[str, Any]] = []
    for raw in raw_headers:
        path = tuple(str(value) for value in raw.get("header_path", ()))
        provenance = tuple(
            str(value)
            for value in raw.get("header_path_provenance", ())
        )
        role, _, confidence, _, conflict = _resolve_column_role(
            "", path, provenance
        )
        derived_headers.append(
            {
                "column_id": str(raw.get("column_id", "")),
                "header_path": list(path),
                "header_path_provenance": list(provenance),
                "column_role": role,
                "column_role_confidence": confidence,
                "column_role_conflict": conflict,
                "inheritance_anchor_id": "",
            }
        )

    expected_columns = tuple(
        sorted(
            {
                str(value)
                for value in manifest.get("declared_data_column_ids", ())
                if str(value)
            }
        )
    )
    observed = tuple(
        str(value)
        for value in manifest.get("observed_cell_coordinates", ())
        if "|" in str(value)
        and str(value).split("|", 1)[1] in expected_columns
    )
    headers = [
        header
        for header in derived_headers
        if str(header["column_id"]) in expected_columns
    ]

    anchors: list[dict[str, str]] = []
    for header in headers:
        if header["column_role_confidence"] != "INHERITED":
            continue
        anchor = next(
            (
                candidate
                for candidate in headers
                if candidate is not header
                and candidate["column_role"] == header["column_role"]
                and "explicit_header"
                in candidate["header_path_provenance"]
            ),
            None,
        )
        if anchor is None:
            continue
        anchor_id = f"anchor:{anchor['column_id']}:{header['column_id']}"
        header["inheritance_anchor_id"] = anchor_id
        anchors.append(
            {
                "anchor_id": anchor_id,
                "column_role": str(header["column_role"]),
                "header_text": " / ".join(anchor["header_path"]),
                "provenance": "explicit_header",
            }
        )

    roles_by_column = {
        str(header["column_id"]): str(header["column_role"])
        for header in headers
    }
    physical = []
    for item in manifest.get("physical_observations", ()):
        if str(item.get("column_id", "")) not in expected_columns:
            continue
        current = dict(item)
        current["column_role"] = roles_by_column.get(
            str(current.get("column_id", "")), ""
        )
        physical.append(current)
    rows = tuple(str(value) for value in manifest.get("expected_row_ids", ()))
    expected_coordinates = [
        f"{row_id}|{column_id}"
        for row_id in rows
        for column_id in expected_columns
    ]
    return {
        "table_boundary_state": manifest.get("table_boundary_state", "PARTIAL"),
        "header_alignment_state": manifest.get("header_alignment_state", "UNALIGNED"),
        "extractor_alignment_state": manifest.get("extractor_alignment_state", "CONFLICT"),
        "expected_data_column_ids": list(expected_columns),
        "expected_row_ids": list(rows),
        "header_columns": headers,
        "inheritance_anchors": anchors,
        "expected_cell_coordinates": expected_coordinates,
        "observed_cell_coordinates": list(observed),
        "physical_observations": physical,
    }


def produce_table_role_inventories(
    candidates: Sequence[NumericCandidate],
    cells: Sequence[EvidenceCell] | None = None,
) -> tuple[TableRoleInventory, ...]:
    """Produce table-scoped states only from pre-Top-K table manifests."""

    manifests: dict[
        tuple[str, str], dict[str, Mapping[str, Any]]
    ] = {}
    table_candidates = [candidate for candidate in candidates if candidate.table_id]
    malformed_keys: set[tuple[str, str]] = set()
    missing_keys: set[tuple[str, str]] = set()
    for candidate in table_candidates:
        table_key = (candidate.source_name, candidate.table_id)
        raw = _context_field(candidate.context, "ROLE_INVENTORY_MANIFEST")
        if not raw:
            missing_keys.add(table_key)
            continue
        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError:
            malformed_keys.add(table_key)
            continue
        if not isinstance(manifest, dict):
            malformed_keys.add(table_key)
            continue
        manifest_key = json.dumps(manifest, ensure_ascii=True, sort_keys=True)
        manifests.setdefault(table_key, {})[manifest_key] = manifest
    if not manifests:
        return ()

    observed_cells = tuple(cells) if cells is not None else evidence_cells(candidates)
    output: list[TableRoleInventory] = []
    for (source_name, table_id), table_manifests in manifests.items():
        materialized = [
            _materialize_runtime_role_inventory_record(manifest)
            for manifest in table_manifests.values()
        ]
        # A table without any explicit role dimension is outside this proof.
        # Current/prior financial statement columns therefore retain legacy
        # behavior instead of being mislabeled as incomplete plan/actual data.
        role_bearing = any(
            header.get("column_role") or header.get("column_role_conflict")
            for record in materialized
            for header in record.get("header_columns", ())
        )
        if not role_bearing:
            continue
        reasons: set[str] = set()
        if (source_name, table_id) in malformed_keys:
            reasons.add("malformed_role_inventory_manifest")
        if (source_name, table_id) in missing_keys:
            reasons.add("missing_table_role_inventory_manifest")
        for record in materialized:
            reasons.update(assess_table_role_inventory(record).reasons)
        if any(
            cell.source_name == source_name
            and cell.table_id == table_id
            and cell.column_role_conflict
            for cell in observed_cells
        ):
            reasons.add("cross_extractor_role_conflict")
        ordered = tuple(sorted(reasons))
        output.append(
            TableRoleInventory(
                source_name=source_name,
                table_id=table_id,
                state="EXPLICIT_COMPLETE" if not ordered else "UNKNOWN",
                reasons=ordered,
            )
        )
    return tuple(
        sorted(output, key=lambda item: (item.source_name, item.table_id))
    )


def produce_column_role_inventory(
    candidates: Sequence[NumericCandidate],
    cells: Sequence[EvidenceCell] | None = None,
) -> RoleInventoryProduction:
    """Aggregate table-scoped producer states for legacy graph consumers."""

    inventories = produce_table_role_inventories(candidates, cells)
    if not inventories:
        return RoleInventoryProduction("", (), 0)
    reasons = tuple(
        sorted({reason for item in inventories for reason in item.reasons})
    )
    return RoleInventoryProduction(
        "EXPLICIT_COMPLETE"
        if all(item.state == "EXPLICIT_COMPLETE" for item in inventories)
        else "UNKNOWN",
        reasons,
        len(inventories),
    )


def evidence_cells(
    candidates: Sequence[NumericCandidate],
) -> tuple[EvidenceCell, ...]:
    output: list[EvidenceCell] = []
    for candidate in candidates:
        row = _context_field(candidate.context, "ROW") or candidate.row_label
        column = (
            _context_field(candidate.context, "COLUMN")
            or candidate.column_label
        )
        period = _context_field(candidate.context, "PERIOD") or candidate.period
        header_path, header_path_provenance = _header_path(candidate, column)
        raw_header = tuple(
            item.strip()
            for item in (
                _context_field(candidate.context, "RAW_HEADER")
                or _context_field(candidate.context, "HEADER")
            ).split(" || ")
            if item.strip()
        )
        (
            column_role,
            column_role_source,
            column_role_confidence,
            column_role_evidence,
            column_role_conflict,
        ) = _resolve_column_role(column, header_path, header_path_provenance)
        descriptor = normalize_text(f"{period} {column}")
        if any(
            marker in f" {descriptor} "
            for marker in (" nam truoc ", " dau nam ", " dau ky ", " 01 01 ")
        ):
            period_role = "prior"
        elif any(
            marker in f" {descriptor} "
            for marker in (" nam nay ", " cuoi nam ", " cuoi ky ", " 31 12 ")
        ):
            period_role = "current"
        else:
            period_role = ""
        origin = _context_field(candidate.context, "VALUE_ORIGIN")
        if origin == "IMPLIED_BLANK_ZERO_FROM_TABLE_AND_NARRATIVE":
            value_status = "implied_blank_zero"
        elif origin:
            value_status = "derived"
        else:
            value_status = "observed"
        scope = _context_field(candidate.context, "SCOPE") or candidate.scope
        output.append(
            EvidenceCell(
                index=candidate.index,
                value=candidate.decimal,
                metric_label=row,
                metric_key=_metric_key(row),
                column=column,
                period=period,
                explicit_years=tuple(
                    dict.fromkeys(
                        re.findall(
                            r"\b(?:19|20)\d{2}\b",
                            normalize_text(f"{period} {column}"),
                        )
                    )
                ),
                period_role=period_role,
                entity=scope,
                section=(
                    _context_field(candidate.context, "SECTION")
                    or candidate.section
                ),
                table_id=(
                    _context_field(candidate.context, "TABLE")
                    or candidate.table_id
                ),
                source_name=candidate.source_name,
                page=candidate.page,
                value_status=value_status,
                unit=candidate.unit,
                unit_key=_unit_key(
                    candidate.unit,
                    is_percent=candidate.is_percent,
                    metric_label=row,
                    context=candidate.context,
                ),
                row_id=(
                    _context_field(candidate.context, "ROW_ID")
                    or _identifier(
                        "|".join(
                            part
                            for part in (
                                candidate.table_id,
                                candidate.subtable_id,
                                row,
                            )
                            if part
                        )
                    )
                ),
                column_id=(
                    _context_field(candidate.context, "COLUMN_ID")
                    or _identifier(column)
                ),
                entity_id=(
                    _context_field(candidate.context, "ENTITY")
                    or candidate.entity_id
                ),
                subtable_id=(
                    _context_field(candidate.context, "SUBTABLE")
                    or candidate.subtable_id
                ),
                row_type=(
                    _context_field(candidate.context, "ROW_TYPE")
                    or candidate.row_type
                ),
                is_percent=candidate.is_percent,
                is_year_like=candidate.is_year_like,
                relevance=float(candidate.relevance),
                column_role=column_role,
                column_role_source=column_role_source,
                column_role_confidence=column_role_confidence,
                column_role_evidence=column_role_evidence,
                column_role_conflict=column_role_conflict,
                header_path=header_path,
                header_path_provenance=header_path_provenance,
                raw_header=raw_header,
                extraction_method=candidate.extraction_method,
                source_span=candidate.source_span,
            )
        )

    # If independent extractors assign different explicit roles to the same
    # logical source cell, retain the evidence but mark every rendition as a
    # conflict.  No score is allowed to choose one extractor's role here.
    logical_groups: dict[tuple[object, ...], list[int]] = {}
    for position, cell in enumerate(output):
        if cell.source_span is not None:
            span_key: tuple[object, ...] = (
                "span",
                cell.source_name,
                cell.page,
                cell.source_span,
                str(cell.value),
            )
            logical_groups.setdefault(span_key, []).append(position)
        semantic_key: tuple[object, ...] = (
            "semantic",
            cell.source_name,
            cell.page,
            cell.metric_key,
            tuple(cell.explicit_years),
            _identifier(cell.entity_id or cell.entity),
            str(cell.value),
        )
        logical_groups.setdefault(semantic_key, []).append(position)

    conflicting_positions: dict[int, set[str]] = {}
    for positions in logical_groups.values():
        methods = {
            output[position].extraction_method
            for position in positions
            if output[position].extraction_method
        }
        roles = {
            output[position].column_role
            for position in positions
            if output[position].column_role
        }
        if len(methods) < 2 or len(roles) < 2:
            continue
        for position in positions:
            conflicting_positions.setdefault(position, set()).update(
                evidence
                for member in positions
                for evidence in output[member].column_role_evidence
            )
    for position, evidence in conflicting_positions.items():
        combined_evidence = tuple(sorted(evidence, key=normalize_text))
        output[position] = replace(
            output[position],
            column_role="",
            column_role_source="cross_extractor_conflict",
            column_role_confidence="CONFLICT",
            column_role_evidence=combined_evidence,
            column_role_conflict=True,
        )
    return tuple(output)


def build_evidence_graph(
    candidates: Sequence[NumericCandidate],
    *,
    column_role_inventory_state: str | None = None,
) -> EvidenceGraph:
    """Build report-independent structural relations between numeric cells.

    Only relations supported by explicit extraction metadata are emitted.
    Missing metadata remains absent so the verifier can represent it as
    UNKNOWN instead of inventing a contradiction.
    """

    if column_role_inventory_state not in {
        None,
        "",
        "EXPLICIT_COMPLETE",
        "UNKNOWN",
    }:
        raise ValueError(
            "column_role_inventory_state must be None, empty, EXPLICIT_COMPLETE, or UNKNOWN"
        )
    cells = evidence_cells(candidates)
    if column_role_inventory_state is None:
        table_inventories = produce_table_role_inventories(candidates, cells)
        inventory_reasons = tuple(
            sorted(
                {
                    reason
                    for item in table_inventories
                    for reason in item.reasons
                }
            )
        )
        resolved_inventory_state = (
            ""
            if not table_inventories
            else "EXPLICIT_COMPLETE"
            if all(
                item.state == "EXPLICIT_COMPLETE"
                for item in table_inventories
            )
            else "UNKNOWN"
        )
    else:
        table_inventories = ()
        resolved_inventory_state = column_role_inventory_state
        inventory_reasons = ()
    edges: set[EvidenceEdge] = set()

    def symmetric(
        left: EvidenceCell,
        right: EvidenceCell,
        relation: EvidenceRelation,
    ) -> None:
        edges.add(EvidenceEdge(left.index, right.index, relation))
        edges.add(EvidenceEdge(right.index, left.index, relation))

    for position, left in enumerate(cells):
        for right in cells[position + 1 :]:
            if left.metric_key and left.metric_key == right.metric_key:
                symmetric(left, right, EvidenceRelation.SAME_METRIC)
            left_entity = _identifier(left.entity_id or left.entity)
            right_entity = _identifier(right.entity_id or right.entity)
            if left_entity and left_entity == right_entity:
                symmetric(left, right, EvidenceRelation.SAME_ENTITY)
            if left.table_id and left.table_id == right.table_id:
                symmetric(left, right, EvidenceRelation.SAME_TABLE)
            if left.row_id and left.row_id == right.row_id:
                symmetric(left, right, EvidenceRelation.SAME_ROW)
            if left.unit_key and left.unit_key == right.unit_key:
                symmetric(left, right, EvidenceRelation.SAME_UNIT)

            left_years = {int(year) for year in left.explicit_years}
            right_years = {int(year) for year in right.explicit_years}
            if len(left_years) == 1 and len(right_years) == 1:
                left_year = next(iter(left_years))
                right_year = next(iter(right_years))
                if left_year + 1 == right_year:
                    edges.add(
                        EvidenceEdge(
                            left.index,
                            right.index,
                            EvidenceRelation.NEXT_PERIOD,
                        )
                    )
                    edges.add(
                        EvidenceEdge(
                            right.index,
                            left.index,
                            EvidenceRelation.PREVIOUS_PERIOD,
                        )
                    )
                elif right_year + 1 == left_year:
                    edges.add(
                        EvidenceEdge(
                            right.index,
                            left.index,
                            EvidenceRelation.NEXT_PERIOD,
                        )
                    )
                    edges.add(
                        EvidenceEdge(
                            left.index,
                            right.index,
                            EvidenceRelation.PREVIOUS_PERIOD,
                        )
                    )

            left_type = normalize_text(left.row_type)
            right_type = normalize_text(right.row_type)
            same_table = bool(left.table_id and left.table_id == right.table_id)
            if same_table and left_type == "detail" and right_type in {
                "total",
                "subtotal",
            }:
                edges.add(
                    EvidenceEdge(
                        left.index,
                        right.index,
                        EvidenceRelation.PART_OF_TOTAL,
                    )
                )
            if same_table and right_type == "detail" and left_type in {
                "total",
                "subtotal",
            }:
                edges.add(
                    EvidenceEdge(
                        right.index,
                        left.index,
                        EvidenceRelation.PART_OF_TOTAL,
                    )
                )

    return EvidenceGraph(
        cells=cells,
        edges=tuple(
            sorted(
                edges,
                key=lambda edge: (
                    edge.source,
                    edge.target,
                    edge.relation.value,
                ),
            )
        ),
        column_role_inventory_state=resolved_inventory_state,
        column_role_inventory_reasons=inventory_reasons,
        table_role_inventories=table_inventories,
    )
