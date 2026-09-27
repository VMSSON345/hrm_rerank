"""Explicit report-year references and source-bound, undated report fields."""
from pathlib import Path
import json
import re

from .ontology import normalize_text
from .physical_coordinates import line_digest, physical_coordinate_key, LINE_SPAN


def report_reference(question):
    pattern = (r"\s*(?:(?:được\s+)?(?:nêu|ghi|trình\s+bày)\s+)?(?:trong|theo)\s+"
               r"báo\s+cáo(?:\s+thường\s+niên)?\s+(?:năm\s+)?((?:19|20)\d{2})")
    matches = list(re.finditer(pattern, question, re.I))
    if len(matches) != 1:
        return None
    match = matches[0]
    remaining = (question[:match.start()] + question[match.end():]).strip()
    if re.search(r"\b(?:19|20)\d{2}\b", remaining):
        return None
    return match[1], remaining


def title_year(text):
    plain = normalize_text(text)
    explicit = re.findall(r"\bnam bao cao\s+((?:19|20)\d{2})\b", plain)
    if not explicit:
        explicit = re.findall(r"\b(?:bao cao thuong nien|annual report)\s+(?:nam\s+)?((?:19|20)\d{2})\b", plain)
    if not explicit:
        # A company name may occur between the report heading and its year.
        # The witness stays on this exact source line; no filename/query year.
        explicit = re.findall(r"\b(?:bao cao thuong nien|annual report)\b[^\n]{0,180}?\bnam\s+((?:19|20)\d{2})\b", plain)
    return explicit[0] if len(set(explicit)) == 1 else ""


def annotate_report_context(lines, annotations):
    _annotate_field_number_formats(lines, annotations)
    by_source = {}
    for line in lines:
        year = title_year(line.text)
        if year:
            by_source.setdefault(line.source_path, []).append({"source_name": Path(line.source_path).name, "year": year, "text": line.text,
                "page": line.page, "line_number": line.line_number, "sha256": line_digest(line.text)})
    for i, line in enumerate(lines):
        titles = by_source.get(line.source_path, [])
        if titles and len({t["year"] for t in titles}) == 1:
            annotations[i]["report_context"] = {"report_year": titles[0]["year"], "title_evidence": titles}


def _annotate_field_number_formats(lines, annotations):
    """Reuse witnessed numeric conventions within a bounded profile section.

    Fields already contribute format witnesses for tables. Apply the same
    mechanism to the fields themselves; neither query nor candidate score is
    read here, and mixed conventions remain unresolved.
    """
    from .program_data import _NUMBER_RE, currency_unit_scale
    from .number_formats import alternatives, _profile, resolve
    groups = {}
    records = []
    for i, line in enumerate(lines):
        if "|" in line.text:
            continue
        for token in _NUMBER_RE.finditer(line.text):
            field = undated_monetary_field(line.text, *token.span())
            if not field:
                continue
            from .unit_defaults import monetary_label
            if currency_unit_scale(field["unit"]) is None and (field["unit"] or not monetary_label(field["row_label"])):
                continue
            key = (line.source_path, line.page, tuple(annotations[i].get("source_section_path", ())))
            raw = token.group(0).strip()
            witness = {"source_name": Path(line.source_path).name, "page": line.page,
                       "line_number": line.line_number, "text": line.text,
                       "sha256": line_digest(line.text), "span": list(token.span()), "raw": raw}
            record = {"alternatives": alternatives(raw), "witness": witness}
            groups.setdefault(key, []).append(record)
            records.append((i, key, raw, list(token.span())))
    for i, key, raw, span in records:
        profile = _profile(groups[key], "field_section_monetary_convention")
        resolution = resolve(raw, profile)
        annotations[i].setdefault("number_formats", []).append({**resolution, "span": span})


def undated_monetary_field(line, start, end):
    from .program_data import _NUMBER_RE
    tokens = list(_NUMBER_RE.finditer(line))
    # Multi-number prose (growth, dates, targets) belongs to the dated
    # assertion parser. Only a sequence of complete labelled fields is a
    # multi-field profile row.
    if len(tokens) > 1 and not all(_undated_monetary_segment(line, *t.span()) for t in tokens):
        return None
    return _undated_monetary_segment(line, start, end)


def _undated_monetary_segment(line, start, end):
    from .program_data import _NUMBER_RE, _number_local_unit, parse_decimal, _decimal_string
    if "|" in line:
        return None
    tokens = list(_NUMBER_RE.finditer(line))
    if not any(start == t.start() and end in {t.end(), t.start()+len(t[0].rstrip())} for t in tokens):
        return None
    # Each explicit field owns its number; preceding registration dates and
    # adjacent monetary fields are not column headers.
    segment_start = 0
    previous = [t for t in tokens if t.end() <= start]
    if previous:
        tail = line[previous[-1].end():start]
        boundary = re.match(r"\s*(?:(?:triệu|tỷ|nghìn|ngàn)\s+)?(?:đồng|VNĐ|VND)\b[\s;,]*", tail, re.I)
        if not boundary:
            return None
        segment_start = previous[-1].end() + boundary.end()
    prefix = re.fullmatch(r"\s*(?:\\?[-*•]\s*)?(?P<label>[^:;\d]{2,150}?)(?:\s*:\s*|\s+(?:là|đạt|có giá trị)\s+)", line[segment_start:start], re.I)
    unit = _number_local_unit(line, end)
    if not prefix:
        return None
    label = prefix["label"].strip()
    from .unit_defaults import monetary_label
    if not unit and not monetary_label(label):
        return None
    if re.search(r"\b(?:cong ty con|don vi|du kien|du bao|ke hoach|ngay|nam)\b", normalize_text(label)):
        return None
    value = parse_decimal(line[start:end])
    if value is None:
        return None
    return {"row_label": label, "value": _decimal_string(value), "unit": unit, "fact_period": None}


def _valid_title_witness(context, source_name):
    titles = context.get("title_evidence", [])
    year = context.get("report_year", "")
    if not titles:
        return not year
    return bool(year) and all(
        t.get("source_name") == source_name
        and title_year(t.get("text", "")) == year
        and line_digest(t.get("text", "")) == t.get("sha256")
        for t in titles
    )


def validated_report_field(candidate):
    context = getattr(candidate, "report_context", None)
    if not isinstance(context, dict) or getattr(candidate, "source_span_scope", "") != LINE_SPAN or physical_coordinate_key(candidate) is None:
        return None
    from .entity_ownership import validate_owner
    if context.get("entity_owner") and validate_owner(candidate, context["entity_owner"]) != "VALID":
        return None
    source_match = re.search(r"(?m)^REPORT_FIELD_SOURCE: (.+)$", candidate.context)
    if not source_match:
        return None
    try:
        source = json.loads(source_match[1])
        if not _valid_title_witness(context, candidate.source_name):
            return None
        if (source["source_name"] != candidate.source_name or source["page"] != candidate.page
                or source["line_number"] != candidate.line_number or line_digest(source["text"]) != candidate.source_line_sha256):
            return None
        from .unit_defaults import effective_source_binding
        field = effective_source_binding(undated_monetary_field(source["text"], *candidate.source_span), candidate)
        if not field or any(field[name] != str(getattr(candidate, name)) for name in ("row_label", "value", "unit")) or candidate.period:
            return None
        return {**context, "source": source, "source_line_sha256": candidate.source_line_sha256, "field": field}
    except (KeyError, TypeError, ValueError):
        return None


def report_field_matches(item):
    """Recheck an EvidenceCell/citation witness; report year is not fact period."""
    witness = getattr(item, "report_field", None)
    if not witness or physical_coordinate_key(item) is None:
        return False
    from .entity_ownership import validate_owner
    if witness.get("entity_owner") and validate_owner(item, witness["entity_owner"]) != "VALID":
        return False
    try:
        source = witness["source"]
        if (source["source_name"] != item.source_name or source["page"] != item.page or source["line_number"] != item.line_number
                or line_digest(source["text"]) != item.source_line_sha256 or item.period):
            return False
        if not _valid_title_witness(witness, item.source_name):
            return False
        from .unit_defaults import effective_source_binding
        field = effective_source_binding(undated_monetary_field(source["text"], *item.source_span), item)
        label = getattr(item, "row_label", None) or getattr(item, "metric_label", "")
        return field == witness["field"] and field == {"row_label": label, "value": str(item.value), "unit": item.unit, "fact_period": None}
    except (KeyError, TypeError, ValueError):
        return False
