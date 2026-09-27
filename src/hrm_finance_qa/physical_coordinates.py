"""Versioned physical coordinates shared by extraction, search and proofs.

legacy_page_v1 preserves historical artifacts. New text producers use line_v2:
(source, page, source line number, line SHA-256, half-open character span). Table-cell producers
declare table_cell_v1. Unresolved or mixed coordinate systems never prove equality.
"""
LEGACY_SPAN = "legacy_page_v1"
LINE_SPAN = "line_v2"
TABLE_CELL = "table_cell_v1"
UNRESOLVED_SPAN = "unresolved_v1"
SPAN_SCOPES = frozenset((LEGACY_SPAN, LINE_SPAN, TABLE_CELL, UNRESOLVED_SPAN, "line_v1"))

def line_digest(text):
    from hashlib import sha256
    return sha256(text.encode("utf-8")).hexdigest()



def physical_coordinate_key(item):
    scope = getattr(item, "source_span_scope", LEGACY_SPAN)
    if scope not in SPAN_SCOPES or scope in {UNRESOLVED_SPAN, "line_v1"}:
        return None
    source, page = getattr(item, "source_name", ""), getattr(item, "page", None)
    if not source or page is None:
        return None
    if scope == TABLE_CELL:
        parts = tuple(getattr(item, name, "") for name in
                      ("table_id", "row_id", "column_id"))
        return (scope, source, page, *parts) if all(parts) else None
    span = getattr(item, "source_span", None)
    if span is None or len(span) != 2:
        return None
    if scope == LEGACY_SPAN:
        return source, page, tuple(span)
    if (any(isinstance(v, bool) or not isinstance(v, int) for v in span)
            or span[0] < 0 or span[1] < span[0]):
        return None
    line = getattr(item, "line_number", None)
    if isinstance(line, bool) or not isinstance(line, int) or line < 1:
        return None
    digest = getattr(item, "source_line_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        return None
    return scope, source, page, line, digest, tuple(span)


def is_versioned_coordinate(item):
    return getattr(item, "source_span_scope", LEGACY_SPAN) != LEGACY_SPAN
