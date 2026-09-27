"""User convention for an absent monetary unit; never source unit evidence."""
from dataclasses import replace
import re

from .ontology import metric_unit_family, normalize_text

DEFAULT_VND_POLICY = "missing_monetary_unit_vnd_v1"
DEFAULT_VND_NOTE = "Đơn vị mặc định là đồng vì bằng chứng được dùng không ghi đơn vị."


def monetary_label(label):
    """Classify the source metric, independently of query, amount or company."""
    from .metric_semantics import normalized_label
    text = normalized_label(re.sub(r"^\s*\d+\s*[-.)]?\s*", "", label))
    if re.search(r"\b(?:ty le|ty suat|he so|bien loi nhuan|tang truong|phan tram|"
                 r"lai suat|interest rate|yield|turnover|so luong|san luong|dien tich|lao dong|nhan vien|nguoi|"
                 r"co phieu|co phan|tren moi|binh quan|vong quay|eps|roe|roa|"
                 r"margin|ratio|percent|per|shares|employees)\b", text) or "%" in label or "/" in label:
        return False
    from .metric_semantics import identity, ALIASES
    if metric_unit_family(text) == "currency" or identity(text) in ALIASES:
        return True
    return bool(re.match(r"(?:tong )?(?:doanh thu|thu nhap|loi nhuan|lai|lo|chi phi|gia von|"
                         r"tai san|no phai tra|no ngan han|no dai han|von|tien|"
                         r"phai thu|phai tra|hang ton kho|quy|khoan ky quy|"
                         r"revenue|profit|income|expenses|assets|liabilities|capital|cash)\b", text))


def _field(context, name):
    match = re.search(rf"(?m)^{name}:\s*(.*)$", context)
    return match[1].strip() if match else ""


def apply_default_unit(candidate):
    """Run after explicit source bindings. Conflicting/other units are not absent."""
    if candidate.unit or candidate.is_percent or candidate.is_year_like:
        return candidate
    if not monetary_label(candidate.row_label):
        return candidate
    context = candidate.context
    column = normalize_text(candidate.column_label)
    if ("UNIT_SOURCE_CONFLICT: true" in context
            or "%" in candidate.column_label
            or re.search(r"\b(?:ty le|ty suat|phan tram|so voi|he so|ratio|percent|"
                         r"usd|eur|jpy|cny|m2|m3|kg|tan|ton|kwh)\b", column)):
        return candidate
    from .program_data import _markdown_cells, _explicit_currency_units
    row = re.sub(r"</?VALUE>", "", _field(context, "CELL"))
    # An explicitly populated unit column can contain an unsupported currency
    # or measurement. Preserve that uncertainty rather than relabel it VND.
    cells = _markdown_cells(row)
    for header in _field(context, "HEADER").split(" || "):
        headers = _markdown_cells(header)
        if len(headers) == len(cells):
            for heading, cell in zip(headers, cells):
                if normalize_text(heading) in {"dvt", "don vi", "don vi tinh", "unit", "units"} and normalize_text(cell):
                    return candidate
    local = " ".join((candidate.row_label, candidate.column_label, row))
    if (any(symbol in local for symbol in "$€£¥")
            or re.search(r"(?:\bx\s*\d+|[×*]\s*\d+|10\s*\^\s*\d+)", local, re.I)
            or _explicit_currency_units(local)
            or re.search(r"\b(?:usd|eur|jpy|cny|gbp|dollar|dollars|m2|m3|kg|tan|ton|kwh)\b", normalize_text(local))):
        return candidate
    return replace(candidate, unit="VND", unit_assumption=DEFAULT_VND_POLICY)


def valid_assumption(item):
    policy = getattr(item, "unit_assumption", "")
    if not policy:
        return True
    label = getattr(item, "row_label", "") or getattr(item, "metric_label", "")
    return (policy == DEFAULT_VND_POLICY and item.unit == "VND"
            and monetary_label(label) and not getattr(item, "is_percent", False))


def effective_source_binding(binding, item):
    """Keep source parsing literal; apply the declared convention when verifying."""
    if binding and not binding.get("unit") and valid_assumption(item) and getattr(item, "unit_assumption", ""):
        return {**binding, "unit": "VND"}
    return binding
