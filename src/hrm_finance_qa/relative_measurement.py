"""Source-scoped measurement bases for dimensionless, same-row arithmetic.

This never supplies a currency or an absolute scale. An opaque row unit can
cancel only between distinct cells of one physically identified source row.
"""
from hashlib import sha256
import json
import re
from typing import Sequence

from .ontology import normalize_text
from .physical_coordinates import LINE_SPAN, physical_coordinate_key

RELATIVE_OPERATIONS = frozenset({"RATIO_PERCENT", "PERCENT_CHANGE", "DIVIDE"})


def same_row_measurement_basis(operation: str, items: Sequence[object]) -> dict | None:
    if operation not in RELATIVE_OPERATIONS or len(items) != 2:
        return None
    if any(getattr(item, "unit", "").strip() or getattr(item, "scale", "") or getattr(item, "is_percent", False) for item in items):
        return None
    keys = [physical_coordinate_key(item) for item in items]
    if any(key is None or key[0] != LINE_SPAN for key in keys):
        return None
    left, right = keys
    if left[:-1] != right[:-1] or not (left[-1][1] <= right[-1][0] or right[-1][1] <= left[-1][0]):
        return None
    if any(key[-1][0] == key[-1][1] for key in keys):
        return None
    tables = [getattr(item, "table_id", "") for item in items]
    rows = [getattr(item, "row_label", None) or getattr(item, "metric_label", "") for item in items]
    if not all(tables) or tables[0] != tables[1] or not all(rows) or rows[0] != rows[1]:
        return None
    entities = [getattr(item, "entity_scope", "") or getattr(item, "entity", "") or getattr(item, "entity_id", "") for item in items]
    if not all(entities) or entities[0] != entities[1]:
        return None
    columns = [getattr(item, "column_label", None) or getattr(item, "column", "") for item in items]
    if not all(columns) or columns[0] == columns[1]:
        return None
    for column in columns:
        # Unit, scale, scope or other qualifiers require explicit interpretation.
        # Blank parsed units alone do not prove that a measurement basis cancels.
        text = normalize_text(column)
        if not re.search(r"\b(?:19|20)\d{2}\b", text):
            return None
        text = re.sub(r"\b(?:19|20)\d{2}\b", " ", text)
        for phrase in ("ke hoach dieu chinh", "thuc hien", "thuc te", "ke hoach", "muc tieu", "nam", "year", "actual", "plan", "target", "th", "kh"):
            text = re.sub(r"\b" + phrase + r"\b", " ", text)
        if re.sub(r"[\s|/()\-]+", "", text):
            return None
    payload = {
        "schema": "same-physical-row-measurement-basis-v1",
        "source_name": left[1], "page": left[2], "line_number": left[3],
        "source_line_sha256": left[4], "table_id": tables[0], "row_label": rows[0],
        "columns": sorted(columns), "entity_scope": entities[0], "absolute_unit_known": False,
    }
    fingerprint = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf8")).hexdigest()
    return {**payload, "unit_symbol": "rowbasis" + fingerprint}
