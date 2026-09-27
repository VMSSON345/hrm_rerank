"""Compare source disclosures without choosing or changing an operand.

Numeric precision compatibility is assessed only after identity and source-format
resolution, against one finest observed anchor. Unknown conventions remain unknown.
"""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
import re

from .ontology import normalize_text
from .physical_coordinates import physical_coordinate_key, is_versioned_coordinate, line_digest


def metric_identity(label):
    from .metric_semantics import identity
    return identity(label)


def _witnesses_valid(items):
    return bool(items) and all(
        isinstance(item, dict) and isinstance(item.get("text"), str)
        and item.get("sha256") == line_digest(item["text"])
        and isinstance(item.get("line_number"), int) and item["line_number"] > 0
        and isinstance(item.get("page"), int) and item["page"] > 0
        for item in items
    )


def _scope_key(cell):
    semantic = cell.table_semantics or {}
    if semantic.get("scope_conflict") or semantic.get("entity_conflict"):
        return None
    from .source_assertions import source_assertion_matches_cell
    if source_assertion_matches_cell(cell):
        scope = cell.source_assertion["binding"].get("accounting_scope")
        if scope:
            return ("accounting_scope", scope)
        # Initial legal capital belongs to the reporting legal entity, not
        # to the incidental section in which its formation is described.
        if (cell.source_assertion['binding'].get('event_kind')=='initial'
                and metric_identity(cell.metric_label)=='charter_capital'
                and cell.entity_id.startswith('entity:source:') and not cell.entity
                and not semantic.get('entity_id')):
            return ('issuer_initial_capital', cell.source_name, cell.entity_id)
    semantic = cell.table_semantics or {}
    if semantic.get("scope_conflict") or semantic.get("entity_conflict"):
        return None
    if semantic.get("accounting_scope") and _witnesses_valid(semantic.get("scope_evidence")):
        return ("accounting_scope", semantic["accounting_scope"])
    from .column_context import statement_reporting_company
    if (cell.entity_id.startswith("entity:source:") and not cell.entity
            and not semantic.get("entity_id")
            and (_witnesses_valid(semantic.get("reporting_company_operations_evidence"))
                 or statement_reporting_company(cell))):
        return ("reporting_company_operations", cell.source_name)
    return (tuple(cell.source_section_path), normalize_text(cell.section))


def _quantum(cell, scale):
    # raw preserves trailing zeros. Resolve comma/dot ambiguity by checking
    # the parsed value, never by guessing a locale from a company identifier.
    raw = getattr(cell, "source_numeric_text", "").strip().replace(" ", "")
    raw = raw.strip("()%").replace("−", "-")
    if not raw or not re.fullmatch(r"[+-]?[\d.,]+", raw):
        return None
    possibilities = set()
    for decimal_mark, grouping_mark in ((",", "."), (".", ",")):
        value = raw.replace(grouping_mark, "")
        if value.count(decimal_mark) > 1:
            continue
        fraction = len(value.split(decimal_mark)[1]) if decimal_mark in value else 0
        try:
            if abs(Decimal(value.replace(decimal_mark, "."))) == abs(cell.value):
                possibilities.add(Decimal(10) ** -fraction * scale)
        except InvalidOperation:
            continue
    return next(iter(possibilities)) if len(possibilities) == 1 else None



def _compare_report_fields(cells):
    from .report_context import report_field_matches
    from .metric_semantics import report_field_identity
    from .program_data import currency_unit_scale
    from .number_formats import resolution_state
    identities=set();physical={};numbers=set()
    for cell in cells:
        if not report_field_matches(cell) or resolution_state(cell) != "VALID":
            return {"equivalent":False,"basis":"","reasons":("disclosure_identity_incomplete",)}
        semantic=cell.table_semantics or {}
        if semantic.get("scope_conflict") or semantic.get("entity_conflict"):
            return {"equivalent":False,"basis":"","reasons":("disclosure_scope_unresolved",)}
        scale=currency_unit_scale(cell.unit)
        if scale is None or not cell.entity_id or cell.value_status != "observed":
            return {"equivalent":False,"basis":"","reasons":("disclosure_identity_incomplete",)}
        # Identity comes before values: matching numbers cannot collapse a
        # different field, entity, reporting year, or accounting perimeter.
        identity=(cell.source_name,cell.entity_id,cell.entity,
                  cell.report_field["report_year"],report_field_identity(cell.metric_label),
                  semantic.get("accounting_scope",""),cell.unit_key)
        identities.add(identity)
        with localcontext() as arithmetic:
            arithmetic.prec=max(80,len(cell.value.as_tuple().digits)+len(str(scale))+4)
            number=cell.value*scale
        key=physical_coordinate_key(cell)
        signature=(identity,number)
        if key in physical and physical[key]!=signature:
            return {"equivalent":False,"basis":"","reasons":("disclosure_coordinate_conflict",)}
        physical[key]=signature;numbers.add(number)
    if len(physical)<2:return {"equivalent":False,"basis":"","reasons":()}
    if len(identities)!=1:return {"equivalent":False,"basis":"","reasons":("disclosure_metric_or_identity_unresolved",)}
    if len(numbers)!=1:return {"equivalent":False,"basis":"","reasons":("disclosure_values_incompatible",)}
    return {"equivalent":True,"basis":"same_report_field_exact_unit_conversion","reasons":()}

def compare_disclosures(cells):
    from .program_data import currency_unit_scale
    cells = tuple(cells)
    if cells and any(cell.report_field for cell in cells):
        return _compare_report_fields(cells)
    from .source_assertions import source_assertion_matches_cell
    initial_event=bool(cells) and all(source_assertion_matches_cell(c)
        and c.source_assertion['binding'].get('event_kind')=='initial'
        and metric_identity(c.metric_label)=='charter_capital' for c in cells)
    explicit_dates={tuple(map(int,m.groups())) for c in cells
        if source_assertion_matches_cell(c)
        for m in re.finditer(r"(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})",c.column)}
    if initial_event and len(explicit_dates)>1:
        return {'equivalent':False,'basis':'','reasons':('disclosure_fact_date_conflict',)}
    physical, tables, identities, scopes, numbers = {}, {}, set(), set(), []
    reasons = set()
    for cell in cells:
        from .number_formats import resolution_state
        if resolution_state(cell) != "VALID":
            return {"equivalent": False, "basis": "", "reasons": ("disclosure_number_format_unresolved",)}
        from .source_assertions import source_assertion_matches_cell
        assertion = source_assertion_matches_cell(cell)
        coordinate = physical_coordinate_key(cell)
        scale = currency_unit_scale(cell.unit)
        if (coordinate is None or not is_versioned_coordinate(cell)
                or not cell.source_section_path or not cell.entity_id or (not cell.table_id and not assertion)
                or not cell.metric_label or cell.value_status != "observed" or scale is None
                or cell.column_role not in {"actual", "plan", "revised_plan", "forecast"}
                or cell.column_role_conflict or not cell.column_role_evidence
                or cell.column_role_confidence not in {"EXPLICIT", "INHERITED"}
                or len(cell.explicit_years) != 1
                or normalize_text(cell.period) != cell.explicit_years[0]):
            return {"equivalent": False, "basis": "", "reasons": ("disclosure_identity_incomplete",)}
        from .column_context import resolved_relative_year
        column_surface=normalize_text(cell.column)
        column_surface=re.sub(r"\s+(?:(?:dvt|don vi(?: tinh)?)\s+)?(?:(?:nghin|trieu|ty)\s+)?(?:dong|vnd|usd|eur|jpy|gbp)$",'',column_surface)
        column_surface=re.sub(r"\s+(?:trinh bay lai|restated)$", "", column_surface)
        remainder = "" if resolved_relative_year(cell) or assertion else re.sub(r"\b(?:19|20)\d{2}\b", "", column_surface)
        # A validated exact date remains part of an ordinary fact identity.
        # An initial-capital event may also have a year-only rendition, but
        # two explicitly different dates above never merge.
        fact_date=() if initial_event or not assertion else tuple(re.findall(r"\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b",cell.column))
        if set(remainder.split()) - {
            "nam", "th", "thuc", "hien", "te", "actual", "kh", "ke", "hoach",
            "plan", "revised", "adjusted", "dieu", "chinh", "forecast", "du", "bao",
        }:
            reasons.add("disclosure_period_basis_unresolved")
        scope = _scope_key(cell)
        if scope is None:
            reasons.add("disclosure_scope_unresolved")
        scopes.add(scope)
        # Resolve the relative label only when its source witness proves the year.
        # This does not establish accounting scope or change any numeric value.
        period_role = "" if resolved_relative_year(cell) else cell.period_role
        identity = (cell.source_name, normalize_text(cell.entity_id), normalize_text(cell.entity),
                    metric_identity(cell.metric_label), tuple(cell.explicit_years), period_role,
                    cell.column_role, cell.unit_key,
                    cell.row_type if metric_identity(cell.metric_label).startswith("literal:") or metric_identity(cell.metric_label)=="revenue_unspecified" else "", fact_date)
        identities.add(identity)
        signature = (identity, scope, cell.value, scale)
        if coordinate in physical and physical[coordinate] != signature:
            reasons.add("disclosure_coordinate_conflict")
        physical[coordinate] = signature
        if not assertion:
            tables.setdefault((cell.source_name, cell.page, cell.table_id), set()).add(coordinate)
        # Decimal's ambient context may have only 28 digits. Source unit
        # conversion must not round a long coefficient before comparison.
        with localcontext() as arithmetic:
            arithmetic.prec = max(80, len(cell.value.as_tuple().digits) + len(str(scale)) + 4)
            canonical = cell.value * scale
        numbers.append((cell, canonical, _quantum(cell, scale)))
    if len(physical) < 2:
        return {"equivalent": False, "basis": "", "reasons": ()}
    if len(identities) != 1:
        reasons.add("disclosure_metric_or_identity_unresolved")
    if len(scopes) != 1:
        reasons.add("disclosure_scope_unresolved")
    if any(len(group) != 1 for group in tables.values()):
        reasons.add("disclosure_intra_table_duplicate")
    if reasons:
        return {"equivalent": False, "basis": "", "reasons": tuple(sorted(reasons))}
    if len({number for _, number, _ in numbers}) == 1:
        return {"equivalent": True, "basis": "exact_unit_conversion", "reasons": ()}
    if any(quantum is None for _, _, quantum in numbers):
        return {"equivalent": False, "basis": "", "reasons": ("disclosure_precision_unresolved",)}
    # Every rendition must agree with ONE most precise observed anchor.
    # Pairwise overlaps cannot form transitive chains of apparent agreement.
    minimum = min(q for _, _, q in numbers)
    anchors = {n for _, n, q in numbers if q == minimum}
    if len(anchors) != 1:
        return {"equivalent": False, "basis": "", "reasons": ("disclosure_values_incompatible",)}
    anchor = next(iter(anchors))
    declared = True
    with localcontext() as context:
        context.prec = max(80, *(len(n.as_tuple().digits) + abs(n.as_tuple().exponent) + 8 for _, n, _ in numbers))
        for cell, number, quantum in numbers:
            if number == anchor:
                continue
            evidence = (cell.table_semantics or {}).get("nearest_rounding_evidence", ())
            # Do not reinterpret an explicit thousand/unit rounding rule as
            # decimal-place rounding. This version supports a declaration
            # tying nearest rounding to the displayed decimal precision only.
            if not evidence:
                # Compatibility under the reported resolution, not a claim
                # that the source explicitly declares nearest rounding.
                declared = False
            elif not _witnesses_valid(evidence) or not any(re.search(
                    r"do chinh xac hien thi|chu so thap phan hien thi|displayed (?:decimal )?precision",
                    normalize_text(item["text"])) for item in evidence):
                return {"equivalent": False, "basis": "", "reasons": ("disclosure_rounding_convention_unresolved",)}
            if (anchor / quantum).quantize(Decimal(1), rounding=ROUND_HALF_UP) * quantum != number:
                return {"equivalent": False, "basis": "", "reasons": ("disclosure_values_incompatible",)}
    return {"equivalent": True, "basis": "source_declared_nearest_rounding" if declared else "compatible_display_precision", "reasons": ()}
