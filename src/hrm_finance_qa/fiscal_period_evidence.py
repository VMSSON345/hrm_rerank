"""Witnessed fiscal labels separate from calendar dates in source facts."""
from __future__ import annotations

from datetime import date
from hashlib import sha256
import re

from .ontology import normalize_text


_BALANCE_METRICS = {
    "total_assets", "literal:tai san ngan han", "current_assets",
    "total_liabilities", "current_liabilities", "owners_equity",
}
_INCOME_METRICS = {
    "net_revenue", "gross_revenue", "profit_after_tax",
    "profit_after_tax_parent", "profit_before_tax", "gross_profit",
    "operating_profit",
}


def _witness(snapshot, name, number):
    text = snapshot.lines[name][number - 1]
    return {"source_name": name, "line_number": number,
            "page": snapshot.pages[name][number], "text": text,
            "sha256": sha256(text.encode()).hexdigest()}


def _date(text):
    found = re.search(r"(?<!\d)(\d{1,2})[/.](\d{1,2})[/.]((?:19|20)\d{2})(?!\d)", text)
    if not found:
        return ""
    try:
        return date(int(found[3]), int(found[2]), int(found[1])).isoformat()
    except ValueError:
        return ""


def fiscal_binding(snapshot, candidate, cell, metric, period, as_of_date, role):
    """Bind FY only when report title and local statement date agree.

    A date alone never defines the fiscal label. A new conflicting statement
    heading prevents inheritance across the page boundary.
    """
    if role != "actual" or metric not in _BALANCE_METRICS | _INCOME_METRICS:
        return None
    name = candidate.source_name
    lines = snapshot.lines[name]
    titles = []
    for number, raw in enumerate(lines[:min(len(lines), 160)], 1):
        found = re.search(r"\bbao cao thuong nien nam tai chinh ((?:19|20)\d{2})\b",
                          normalize_text(raw))
        if found:
            titles.append((number, found[1]))
    if len({year for _, year in titles}) != 1:
        return None
    report_year = next(iter({year for _, year in titles}))
    family = "balance" if metric in _BALANCE_METRICS else "income"
    headings = []
    for number in range(max(1, candidate.line_number - 120), candidate.line_number):
        raw = lines[number - 1]
        if snapshot.pages[name][candidate.line_number] - snapshot.pages[name][number] > 2:
            continue
        normal = normalize_text(raw)
        if family == "balance":
            compatible = "bang can doi ke toan" in normal
        else:
            compatible = ("bao cao ket qua" in normal and
                          "kinh doanh" in normal)
        if not compatible:
            continue
        end = _date(raw)
        if end:
            headings.append((number, end, normal))
    if not headings:
        return None
    # Any closer heading of this statement family is authoritative, including
    # a contradictory year. It must not be bypassed to find an older match.
    number, end, normal = headings[-1]
    if int(end[:4]) not in {int(report_year), int(report_year) + 1}:
        return None
    if family == "balance":
        if as_of_date != end or period != end[:4]:
            return None
    elif period and period not in {end[:4], str(int(end[:4]) - 1)}:
        return None
    scopes = set()
    if "hop nhat" in normal:
        scopes.add("consolidated")
    if "rieng le" in normal or re.search(r"\bbao cao .*\b rieng\b", normal):
        scopes.add("separate")
    if len(scopes) != 1:
        return None
    scope = next(iter(scopes))
    existing = (cell.table_semantics or {}).get("accounting_scope", "")
    if existing and existing != scope:
        return None
    column = normalize_text(cell.column)
    prior = bool(re.search(r"\b(?:nam|ky) truoc\b", column))
    fiscal_year = str(int(report_year) - 1) if prior else report_year
    witnesses = (_witness(snapshot, name, titles[0][0]),
                 _witness(snapshot, name, number))
    if not all(snapshot.witness_valid(item, name) for item in witnesses):
        return None
    return {"fiscal_year": fiscal_year,
            "fiscal_end_date": "" if prior else end,
            "scope": scope, "evidence": witnesses}


def fiscal_question_surface(question):
    """Return a planner surface only for explicit, non-conflicting FY wording."""
    text = normalize_text(question)
    if "nam tai chinh" not in text:
        return None
    scope = ("consolidated" if "hop nhat" in text else
             "separate" if "rieng le" in text else "")
    if "hop nhat" in text and "rieng le" in text:
        return None
    explicit = re.search(r"\bnam tai chinh(?: thuc hien)? ((?:19|20)\d{2})\b", text)
    ended = re.search(
        r"\b(?:cua )?nam tai chinh ket thuc ngay "
        r"(\d{1,2}) (\d{1,2}) ((?:19|20)\d{2})\b", text)
    if explicit:
        fy = explicit[1]
        # Normalization removes slashes and parentheses. A bare date such as
        # "(30/09/2024)" must be removed as a whole before the row metric is
        # compiled; otherwise its day/month survive after the year is stripped.
        full_dates = list(re.finditer(
            r"\b(\d{1,2}) (\d{1,2}) ((?:19|20)\d{2})\b", text))
        for match in full_dates:
            try:
                date(int(match[3]), int(match[2]), int(match[1]))
            except ValueError:
                return None
        if any(int(match[3]) not in {int(fy), int(fy) + 1}
               for match in full_dates):
            return None
        cleaned = re.sub(
            r"\b(?:(?:(?:tai|vao)\s+ngay|ket thuc(?:\s+ngay)?)\s+)?"
            r"\d{1,2}\s+\d{1,2}\s+(?:19|20)\d{2}\b",
            " ", text)
        cleaned = re.sub(r"\bnam tai chinh thuc hien " + fy + r"\b",
                         "nam " + fy, cleaned)
        cleaned = re.sub(r"\bnam tai chinh " + fy + r"\b",
                         "nam " + fy, cleaned)
    elif ended:
        year = ended[3]
        cleaned = text[:ended.start()] + " nam " + year + text[ended.end():]
    else:
        return None
    cleaned = re.sub(r"\b(?:hop nhat|rieng le)\b", " ", cleaned)
    # Keep the change operation while preventing its wording from becoming
    # part of a fiscal-year metric label in the recursive planner pass.
    cleaned = re.sub(r"\b(?:tang hay giam|giam hay tang)\b",
                     "thay doi", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, scope
