"""Recover percentage facts from structurally aligned, glued Markdown rows.

This only binds a token already extracted with its exact line span. It does not
infer an absolute currency unit or repair arbitrary OCR text.
"""
from __future__ import annotations

from decimal import Decimal
from hashlib import sha256
import re

from .metric_semantics import identity
from .ontology import normalize_text
from .number_formats import alternatives
from .post_hrm_contract import SourceFact


_PERCENT = re.compile(r'[+-]?\d+(?:[.,]\d+)?%')
_YEAR_CELL = re.compile(r'\s*Năm\s+((?:19|20)\d{2})\s*:\s*(.*?)\s*', re.I)
_LABEL_START = re.compile(r'\s*\+\s*(?=(?:Hệ số|Tỷ suất|Biên)\b)', re.I)


def _edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for index, char in enumerate(left, 1):
        current = [index]
        for other, target in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[other] + 1,
                               previous[other - 1] + (char != target)))
        previous = current
    return previous[-1]


def _margin_metric(label, *, facts, source_name, year, line_number, page,
                   entity, scope, snapshot):
    if '/' not in label:
        return '', None
    numerator, denominator = label.split('/', 1)
    numerator = re.sub(r'^(?:he so|ty suat|bien)\s+', '', normalize_text(numerator))
    profit = identity(numerator)
    if profit not in {'profit_after_tax', 'gross_profit'}:
        return '', None
    support = None
    if identity(denominator) != 'net_revenue':
        normalized = normalize_text(denominator)
        if (_edit_distance(normalized, 'doanh thu thuan') > 2
                or _edit_distance(normalized, 'doanh thu hoat dong') <= 2):
            return '', None
        # A bounded OCR alias needs a separately labelled actual source row.
        nearby = [f for f in facts if f.source_verified
                  and f.metric == 'net_revenue' and f.role == 'actual'
                  and f.source_name == source_name and f.period == year
                  and f.entity == entity and f.scope == scope
                  and abs(f.line_number - line_number) <= 40
                  and snapshot.pages[source_name][f.line_number] == page]
        if not nearby:
            return '', None
        source_fact = min(nearby, key=lambda f: (
            abs(f.line_number - line_number), f.line_number))
        raw_line = snapshot.lines[source_name][source_fact.line_number - 1]
        versions = (raw_line, re.sub(r'\s+', ' ', raw_line).strip())
        text = next((line for line in versions
                     if sha256(line.encode()).hexdigest()
                     == source_fact.source_line_sha256), None)
        if text is None:
            return '', None
        support = {'source_name': source_name,
                   'line_number': source_fact.line_number,
                   'page': source_fact.page, 'text': text,
                   'source_line_sha256': source_fact.source_line_sha256,
                   'basis': 'bounded_ocr_metric_alias',
                   'metric': source_fact.metric}
    return {'profit_after_tax': 'net_margin',
            'gross_profit': 'gross_margin'}[profit], support


def recover_compound_ratio_facts(candidates, snapshot, existing_facts):
    """Create typed ratio facts only for complete one-to-one row alignment."""
    by_line = {}
    for candidate in candidates:
        if candidate.is_percent and candidate.source_name in snapshot.lines:
            by_line.setdefault((candidate.source_name, candidate.line_number), []).append(candidate)
    recovered = []
    seen = set()
    for (source_name, line_number), line_candidates in by_line.items():
        try:
            line, _ = snapshot.bound_line(line_candidates[0])
        except ValueError:
            continue
        if '+' not in line or '|' not in line:
            continue
        first = line.split('|', 1)[0]
        pieces = _LABEL_START.split(first)
        labels = [piece.strip() for piece in pieces[1:]]
        if len(labels) < 2 or any(not label or '/' not in label for label in labels):
            continue
        parts = []
        offset = 0
        for part in line.split('|'):
            parts.append((part, offset))
            offset += len(part) + 1
        years = {}
        valid = True
        for cell, start in parts[1:]:
            if not cell.strip():
                continue
            match = _YEAR_CELL.fullmatch(cell)
            if not match or match[1] in years:
                valid = False
                break
            tokens = list(_PERCENT.finditer(match[2]))
            if (len(tokens) != len(labels)
                    or _PERCENT.sub('', match[2]).strip()):
                valid = False
                break
            years[match[1]] = [(token[0], (start + match.start(2) + token.start(),
                                            start + match.start(2) + token.end()))
                               for token in tokens]
        if not valid or not years:
            continue
        candidate_by_span = {tuple(c.source_span): c for c in line_candidates}
        for year, tokens in years.items():
            for label, (raw, span) in zip(labels, tokens):
                metric, metric_support = _margin_metric(
                    label, facts=existing_facts, source_name=source_name,
                    year=year, line_number=line_number,
                    page=snapshot.pages[source_name][line_number],
                    entity=line_candidates[0].entity_id or 'entity:source:' + source_name,
                    scope=line_candidates[0].scope, snapshot=snapshot)
                if not metric:
                    continue
                candidate = candidate_by_span.get(span)
                if (candidate is None or candidate.raw != raw
                        or candidate.source_line_sha256 != sha256(line.encode()).hexdigest()
                        or (source_name, line_number, span) in seen):
                    continue
                try:
                    bound, fragment = snapshot.bound_line(candidate)
                except ValueError:
                    continue
                if bound != line or fragment != raw:
                    continue
                interpretations = {
                    (Decimal(option['value']), Decimal(option['quantum']))
                    for option in alternatives(raw).values()}
                if len(interpretations) != 1:
                    continue
                value, precision = next(iter(interpretations))
                if value != Decimal(candidate.value):
                    continue
                seen.add((source_name, line_number, span))
                witness = {'source_name': source_name, 'line_number': line_number,
                           'page': candidate.page, 'text': line,
                           'source_line_sha256': sha256(line.encode()).hexdigest()}
                recovered.append(SourceFact(
                    fact_id=f'{source_name}:{line_number}:{span[0]}',
                    metric=metric, metric_label=label, kind='ratio',
                    value=value, raw_value=raw, unit='%', currency='',
                    unit_resolved=True, printed_quantum=precision,
                    scale=Decimal('.01'), quantum=precision * Decimal('.01'),
                    period=year, period_binding='aligned_compound_ratio_column',
                    role='actual', entity=candidate.entity_id or 'entity:source:' + source_name,
                    scope=candidate.scope, source_name=source_name,
                    document_sha256=snapshot.hashes[source_name],
                    source_line_sha256=candidate.source_line_sha256,
                    line_number=line_number, source_span=span,
                    page=candidate.page, source_context=(witness,) + ((metric_support,) if metric_support else ()),
                    column_label=f'Năm {year}',
                    table_key=f'{source_name}|{candidate.page}|{candidate.table_id}|aligned_ratio',
                    section=candidate.section, source_kind='table',
                    reporting_entity=(not candidate.entity_id or candidate.entity_id.startswith('entity:source:')),
                    number_format_basis='intrinsic_token',
                    source_verified=True, candidate_index=candidate.index))
    return recovered


def recover_glued_liquidity_ratio_facts(candidates, snapshot):
    """Bind two printed ratio labels to two year columns only with 2x2 spans."""
    from .metric_semantics import normalized_label
    numeric = re.compile(r'(?<!\d)\d+(?:[.,]\d+)*(?!\d)')
    by_line = {}
    for candidate in candidates:
        if candidate.source_name in snapshot.lines and not candidate.is_percent:
            by_line.setdefault((candidate.source_name, candidate.line_number), []).append(candidate)
    facts = []
    for (name, number), row_candidates in by_line.items():
        try:
            line, _ = snapshot.bound_line(row_candidates[0])
        except ValueError:
            continue
        if line.count('|') < 2:
            continue
        first = line.split('|', 1)[0]
        labels = [part.strip() for part in re.split(r'\s+-\s+', first)]
        if len(labels) != 3:
            continue
        labels = labels[1:]
        normalized = [normalized_label(label) for label in labels]
        if not all('thanh toan' in label for label in normalized):
            continue
        def proposed_metric(label):
            if re.search(r'\b(?:ngan han|ngn hn)\b', label) and 'nhanh' not in label:
                return 'current_ratio'
            if re.search(r'\bnhanh\b', label) and not re.search(r'\b(?:ngan han|ngn hn)\b', label):
                return 'quick_ratio'
            return ''
        metrics = [proposed_metric(label) for label in normalized]
        if set(metrics) != {'current_ratio', 'quick_ratio'}:
            continue
        prior = snapshot.lines[name][max(0, number-5):number-1]
        headers = [(number-len(prior)+i, text) for i, text in enumerate(prior)
                   if '|' in text and 'nam ' in normalize_text(text)
                   and 'chi tieu' in normalize_text(text)]
        if not headers:
            continue
        header_n, header = headers[-1]
        parts, offset = [], 0
        for part in line.split('|'):
            parts.append((part, offset))
            offset += len(part) + 1
        years = {}
        valid = True
        for part, start in parts[1:]:
            if not part.strip():
                continue
            match = re.fullmatch(r'\s*Năm\s+((?:19|20)\d{2})\s*:\s*(.*?)\s*', part, re.I)
            if not match or match[1] in years:
                valid = False
                break
            tokens = list(numeric.finditer(match[2]))
            if len(tokens) != 2 or numeric.sub('', match[2]).strip():
                valid = False
                break
            years[match[1]] = [(token[0],
                (start + match.start(2) + token.start(),
                 start + match.start(2) + token.end())) for token in tokens]
        header_years = set(re.findall(r'(?:19|20)\d{2}', header))
        if not valid or len(years) != 2 or set(years) != header_years:
            continue
        def token_span(candidate):
            left,right=candidate.source_span
            while left<right and line[left].isspace():left+=1
            while right>left and line[right-1].isspace():right-=1
            return left,right
        candidate_by_span = {token_span(c): c for c in row_candidates}
        staged = []
        for year, tokens in years.items():
            for label, metric, (raw, span) in zip(labels, metrics, tokens):
                candidate = candidate_by_span.get(span)
                if not candidate or candidate.raw != raw:
                    valid = False
                    break
                try:
                    bound, fragment = snapshot.bound_line(candidate)
                except ValueError:
                    valid = False
                    break
                interpretations = {(Decimal(option['value']), Decimal(option['quantum']))
                                   for option in alternatives(raw).values()}
                if bound != line or fragment != raw or len(interpretations) != 1:
                    valid = False
                    break
                value, precision = next(iter(interpretations))
                if value != Decimal(candidate.value):
                    valid = False
                    break
                staged.append((candidate, year, label, metric, raw, tuple(candidate.source_span), value, precision))
            if not valid:
                break
        if not valid or len(staged) != 4:
            continue
        witnesses = tuple({'source_name': name, 'line_number': n,
                           'page': snapshot.pages[name][n], 'text': text,
                           'source_line_sha256': sha256(text.encode()).hexdigest()}
                          for n, text in ((header_n, header), (number, line)))
        for candidate, year, label, metric, raw, span, value, precision in staged:
            facts.append(SourceFact(
                fact_id=f'{name}:{number}:{span[0]}', metric=metric,
                metric_label=label, kind='ratio', value=value, raw_value=raw,
                unit='x', currency='', unit_resolved=True,
                printed_quantum=precision, scale=Decimal(1), quantum=precision,
                period=year, period_binding='aligned_glued_ratio_column',
                role='actual', entity=candidate.entity_id or 'entity:source:' + name,
                scope=candidate.scope, source_name=name,
                document_sha256=snapshot.hashes[name],
                source_line_sha256=candidate.source_line_sha256,
                line_number=number, source_span=span, page=candidate.page,
                source_context=witnesses, column_label=f'Năm {year}',
                table_key=f'{name}|{candidate.page}|{candidate.table_id}|aligned_ratio',
                section=candidate.section, source_kind='table',
                reporting_entity=(not candidate.entity_id or
                                  candidate.entity_id.startswith('entity:source:')),
                number_format_basis='intrinsic_token', source_verified=True,
                candidate_index=candidate.index))
    return facts
