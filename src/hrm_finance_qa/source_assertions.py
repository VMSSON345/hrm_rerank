"""Bind a dated monetary statement to its own text, independently of a query."""
from datetime import date
import json
import re

from .ontology import normalize_text
from .physical_coordinates import LINE_SPAN, line_digest, physical_coordinate_key


def _exact_dated_amount_assertion(line: str, start: int, end: int) -> dict | None:
    from .program_data import _NUMBER_RE, _is_answer_metadata_number, _number_local_unit, _decimal_string, parse_decimal
    if "|" in line:
        return None
    matches = []
    for token in _NUMBER_RE.finditer(line):
        if (token.start() and line[token.start() - 1] == "/") or (token.end() < len(line) and line[token.end()] == "/"):
            continue
        value = parse_decimal(token.group(0))
        if value is not None and not _is_answer_metadata_number(line, token, value):
            matches.append(token)
    if len(matches) != 1 or (matches[0].start(), matches[0].end()) != (start, end):
        return None
    match = re.search(
        r"(?:^|[,;:])\s*(?P<label>[^,;:]{2,180}?)\s+"
        r"(?P<period_text>tại(?:\s+ngày)?\s+(?P<day>\d{1,2})/(?P<month>\d{1,2})/(?P<year>(?:19|20)\d{2}))"
        r"\s+(?:là|đạt|có\s+giá\s+trị)\s*$", line[:start], re.IGNORECASE,
    )
    if not match:
        return None
    try:
        date(int(match["year"]), int(match["month"]), int(match["day"]))
    except ValueError:
        return None
    label = match["label"].strip(" -*•.")
    normalized = normalize_text(label)
    if re.search(r"\b(?:khong|chua|neu|gia su|co the|cong ty|don vi|subsidiary)\b", normalized):
        return None
    if re.search(r"\b(?:du kien|du bao|uoc tinh|se)\b", normalized):
        role = "forecast"
    elif re.search(r"\b(?:ke hoach|muc tieu)\b", normalized):
        role = "plan"
    else:
        role = "actual"
    return {"row_label": label, "column_label": match["period_text"],
            "period": match["year"], "column_role": role,
            "unit": _number_local_unit(line, end),
            "value": _decimal_string(parse_decimal(line[start:end]))}


_BINDING_FIELDS = ("row_label", "column_label", "period", "column_role", "unit", "value")


def _context_year(records):
    years=[]
    for record in records or ():
        text=record.get("text", "")
        if line_digest(text)!=record.get("sha256"):
            return ""
        match=re.fullmatch(r"nam ((?:19|20)\d{2})", normalize_text(text))
        if not match:
            return ""
        years.append(match[1])
    return years[0] if years and len(set(years))==1 else ""


def dated_amount_assertion(line: str, start: int, end: int, period_context=None) -> dict | None:
    exact=_exact_dated_amount_assertion(line,start,end)
    if exact:
        return exact
    if "|" in line:
        return None
    from .program_data import _number_local_unit, parse_decimal, _decimal_string
    from .unit_defaults import monetary_label
    from .metric_semantics import normalize_metric_text
    prefix=line[:start]
    movement=re.search(r"(?P<label>vốn điều lệ|von dieu le)\s+từ\s+[\d.,]+\s+(?:đồng|VND|tỷ đồng|triệu đồng)\s+lên\s*$",prefix,re.I)
    if movement:
        if re.search(r"\b(?:neu|gia su|khong|chua|co the|cong ty con|don vi|subsidiary)\b",normalize_text(prefix)):
            return None
        dates=list(re.finditer(r"(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})",prefix[:movement.start()]))
        if dates:
            d=dates[-1]
            try:dt=date(int(d[3]),int(d[2]),int(d[1]))
            except ValueError:return None
            value=parse_decimal(line[start:end])
            if value is not None:
                return {"row_label":movement['label'],"column_label":f"tại ngày {dt.day}/{dt.month}/{dt.year}","period":str(dt.year),"column_role":"actual","unit":_number_local_unit(line,end),"value":_decimal_string(value),"event_kind":"capital_change"}
    match=re.search(
        r"(?P<label>(?:(?:Tổng|Tong)\s+)?(?:doanh thu|lợi nhuận|loi nhuan|LNST|LNTT|LN trước thuế|LN sau thuế|chi phí|chi phi|nợ phải trả|no phai tra|lỗ lũy kế|lo luy ke|vốn điều lệ|von dieu le|vốn đầu tư của chủ sở hữu)[^:;.!?\n]{0,140}?)"
        r"\s*(?P<verb>:|là|la|đạt|dat|dự kiến|du kien)\s*$", prefix, re.I)
    if not match:
        return None
    preceding_sentences=re.split(r"[.;!?]\s+", prefix[:match.start()])
    lead=preceding_sentences[-1]
    lead_plain=normalize_text(lead)
    if re.search(r"\b(?:neu|gia su|khong|chua|co the)\b", lead_plain):
        return None
    label=match["label"].strip();plain=normalize_text(label)
    if re.search(r"\b(?:khong|chua|neu|gia su|co the|cong ty con|don vi|subsidiary)\b", plain):
        return None
    # A named actor in the assertion cannot silently become the report issuer.
    if "cong ty" in re.sub(r"\bcua cong ty\b", "",plain):
        return None
    explicit_date=re.search(r"(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})",label)
    date_in_label=explicit_date is not None
    if explicit_date is None:
        nearby=list(re.finditer(r"(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})",lead[-160:]))
        if nearby:explicit_date=nearby[-1]
    # An immediately preceding formation sentence can date its initial
    # capital assertion. It must refer to the generic reporting company,
    # contain one consistent date, and not describe a conditional/other actor.
    if explicit_date is None and not lead.strip() and len(preceding_sentences)>1 and re.search(r"\bban dau\b",plain):
        previous=preceding_sentences[-2]
        prior_plain=normalize_text(previous)
        dates=list(re.finditer(r"(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})",previous))
        if (re.search(r"\bcong ty duoc thanh lap\b",prior_plain)
                and not re.search(r"\b(?:neu|gia su|khong|chua|cong ty con|don vi|subsidiary)\b",prior_plain)
                and len({tuple(map(int,d.groups())) for d in dates})==1):
            explicit_date=dates[0]
    formation=re.fullmatch(r"cong ty(?: duoc)? thanh lap(?: tu)? nam ((?:19|20)\d{2}) voi(?: so)?",lead_plain)
    formation_year=formation[1] if formation and normalize_metric_text(label).startswith('von dieu le') else ''
    years=re.findall(r"\b(?:19|20)\d{2}\b",label)
    if len(set(years))>1:
        return None
    leading=re.match(r"\s*(?:\\?[-*•>]\s*)*(?:Năm|Nam)\s+((?:19|20)\d{2})\b",line,re.I)
    year=years[0] if years else explicit_date[3] if explicit_date else formation_year if formation_year else leading[1] if leading else _context_year(period_context)
    if not year:
        return None
    column="Năm " + year
    if explicit_date:
        try:
            day=date(int(explicit_date[3]),int(explicit_date[2]),int(explicit_date[1]))
        except ValueError:
            return None
        column=f"tại ngày {day.day}/{day.month}/{day.year}"
        if date_in_label:
            label=label[:explicit_date.start()] + label[explicit_date.end():]
    governing=re.search(r"\b(?:du kien|du bao|uoc tinh|ke hoach|muc tieu)(?:\s+nam)?(?:\s+(?:19|20)\d{2})?(?:\s+cua cong ty)?\s*$",lead_plain)
    role_text=plain+" "+normalize_text(match["verb"])+" "+(governing[0] if governing else "")
    role="forecast" if re.search(r"\b(?:du kien|du bao|uoc tinh)\b",role_text) else "plan" if re.search(r"\b(?:ke hoach|muc tieu)\b",role_text) else "actual"
    scope="consolidated" if "hop nhat" in plain else "separate" if re.search(r"\brieng(?: le| cong ty me)?\b",plain) else ""
    label=re.sub(r"\b(?:năm|nam)?\s*(?:19|20)\d{2}\b", " ", label,flags=re.I)
    label=re.sub(r"\b(?:tại ngày|tai ngay|đến ngày|den ngay|tại|tai|đến|den|hợp nhất|hop nhat|riêng lẻ|rieng le|ban đầu|ban dau|của công ty|cua cong ty|dự kiến|du kien|kế hoạch|ke hoach)\b", " ",label,flags=re.I)
    label=re.sub(r"\s+", " ", label).strip(" ,:-")
    if not monetary_label(normalize_metric_text(label)):
        return None
    value=parse_decimal(line[start:end])
    if value is None:
        return None
    binding={"row_label":label,"column_label":column,"period":year,"column_role":role,
             "unit":_number_local_unit(line,end),"value":_decimal_string(value)}
    if formation_year or re.search(r"\b(?:ban dau|lan dau|chuyen doi loai hinh|chuyen thanh cong ty co phan)\b",lead_plain+" "+plain):
        binding["event_kind"]="initial"
    if scope:
        binding["accounting_scope"]=scope
    return binding


def _period_context_matches(source):
    for record in source.get("period_context", ()):
        if (record.get("source_name")!=source.get("source_name") or record.get("page")!=source.get("page")
                or not 0 < source["line_number"]-record.get("line_number",-100) <= 3):
            return False
    return not source.get("period_context") or bool(_context_year(source["period_context"]))


def validated_source_assertion(candidate: object) -> dict | None:
    """Recompute a producer's assertion from its hashed physical source line."""
    if getattr(candidate, "source_span_scope", "") != LINE_SPAN or physical_coordinate_key(candidate) is None:
        return None
    marker = re.search(r"(?m)^SOURCE_ASSERTION: (.+)$", candidate.context)
    if not marker:
        return None
    try:
        source = json.loads(marker[1])
        line = source["text"]
        if not _period_context_matches(source):
            return None
        if (source["source_name"] != candidate.source_name or source["page"] != candidate.page
                or source["line_number"] != candidate.line_number
                or line_digest(line) != candidate.source_line_sha256):
            return None
        from .unit_defaults import effective_source_binding
        binding = effective_source_binding(dated_amount_assertion(line, *candidate.source_span, source.get("period_context")), candidate)
        if not binding or any(str(getattr(candidate, name)) != binding[name] for name in _BINDING_FIELDS):
            return None
        return {**source, "sha256": candidate.source_line_sha256, "binding": binding}
    except (KeyError, TypeError, ValueError):
        return None


def source_assertion_matches_cell(cell: object) -> bool:
    """Do not let a copied witness authenticate modified EvidenceCell fields."""
    source = getattr(cell, "source_assertion", None)
    if not isinstance(source, dict) or getattr(cell, "source_span_scope", "") != LINE_SPAN or physical_coordinate_key(cell) is None:
        return False
    try:
        if not _period_context_matches(source):
            return False
        if (source["source_name"] != cell.source_name or source["page"] != cell.page
                or source["line_number"] != cell.line_number
                or line_digest(source["text"]) != cell.source_line_sha256):
            return False
        from .unit_defaults import effective_source_binding
        binding = effective_source_binding(dated_amount_assertion(source["text"], *cell.source_span, source.get("period_context")), cell)
        expected = {"row_label": cell.metric_label, "column_label": cell.column,
                    "period": cell.period, "column_role": cell.column_role,
                    "unit": cell.unit, "value": str(cell.value)}
        return binding is not None and binding == source["binding"] and {k:binding[k] for k in _BINDING_FIELDS} == expected
    except (KeyError, TypeError, ValueError):
        return False
