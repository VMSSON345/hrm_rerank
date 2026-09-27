"""Resolve a bare revenue request to one source-backed metric before selection.

This interprets the user's wording only. Source values and the ordinary
verifier/certificate path remain unchanged.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import re

from .metric_semantics import identity
from .ontology import normalize_text
from .question_planner import compile_question_plan, resolve_public_question
from .question_request_planner import compile_question_request, SINGLE_OUTPUT

_REVENUE_LABELS = {
    'revenue_unspecified': 'Doanh thu',
    'net_revenue': 'Doanh thu thuần',
    'total_revenue': 'Tổng doanh thu',
    'literal:doanh thu ban hang va cung cap dich vu': 'Doanh thu bán hàng và cung cấp dịch vụ',
    'literal:doanh thu tai chinh': 'Doanh thu tài chính',
    'literal:doanh thu hoat dong tai chinh': 'Doanh thu hoạt động tài chính',
}
_DOCUMENT_REVENUE = set(_REVENUE_LABELS) - {
    'literal:doanh thu tai chinh', 'literal:doanh thu hoat dong tai chinh',
}
_BARE = re.compile(r'\b(?:doanh\s+thu|revenue|sales)\b', re.I)


@dataclass(frozen=True)
class RevenueResolution:
    original_question: str
    resolved_question: str
    metric: str = ''
    basis: str = ''
    document_sha256: str = ''
    previous_question: str = ''
    witnesses: tuple = ()
    alternatives: tuple = ()
    bare_metric: str = "doanh thu"

    def as_dict(self):
        return asdict(self)

    @property
    def clarification(self):
        if not self.alternatives:
            return ''
        return ('Bạn muốn so sánh ' if normalize_text(self.original_question).startswith('so sanh ') else 'Bạn muốn hỏi ') + ' hay '.join(self.alternatives) + '?'

    @property
    def note(self):
        if not self.basis or self.alternatives:
            return ''
        basis = 'theo câu hỏi trước trên cùng tài liệu' if self.basis == 'conversation_metric' else 'theo hàng và kỳ trong tài liệu'
        return f'Đang hiểu “{self.bare_metric}” là {self.metric.lower()}, {basis}.'


def _request_metrics(question):
    request = compile_question_request(question)
    return request, {identity(slot.metric) for o in request.outputs for slot in o.plan.operand_slots}


_PROFIT_LABELS = {
    'literal:loi nhuan': 'Lợi nhuận',
    'profit_after_tax': 'Lợi nhuận sau thuế',
    'profit_before_tax': 'Lợi nhuận trước thuế',
    'gross_profit': 'Lợi nhuận gộp',
    'literal:loi nhuan tu hoat dong kinh doanh': 'Lợi nhuận từ hoạt động kinh doanh',
    'literal:loi nhuan thuan tu hoat dong kinh doanh': 'Lợi nhuận thuần từ hoạt động kinh doanh',
    'literal:loi nhuan khac': 'Lợi nhuận khác',
}


def resolve_revenue_question(question, document, *, previous_question=''):
    return _resolve_metric_question(question, document, previous_question=previous_question)


def resolve_metric_question(question, document, *, previous_question=''):
    _, metrics = _request_metrics(question)
    if metrics == {'literal:loi nhuan'}:
        return _resolve_metric_question(question, document, previous_question=previous_question,
            labels=_PROFIT_LABELS, document_metrics=set(_PROFIT_LABELS),
            bare_key='literal:loi nhuan', bare=re.compile(r'\b(?:lợi\s+nhuận|loi\s+nhuan)\b',re.I),
            search_terms=('loi nhuan',), bare_metric='lợi nhuận')
    return resolve_revenue_question(question, document, previous_question=previous_question)


def _resolve_metric_question(question, document, *, previous_question='',
        labels=_REVENUE_LABELS, document_metrics=_DOCUMENT_REVENUE,
        bare_key='revenue_unspecified', bare=_BARE,
        search_terms=('doanh thu','revenue','sales'), bare_metric='doanh thu'):
    unchanged = RevenueResolution(question, question)
    request, metrics = _request_metrics(question)
    # Do not rewrite explicit metrics, mixed-metric requests or unsupported splits.
    if metrics != {bare_key} or not (request.mode == SINGLE_OUTPUT or request.split_strategy == 'COMPARISON_OVERVIEW'):
        return unchanged
    path = Path(document)
    if not path.is_file() or path.suffix.lower() not in {'.md', '.txt'}:
        return unchanged
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if previous_question:
        _, previous_metrics = _request_metrics(previous_question)
        if len(previous_metrics) == 1:
            key = next(iter(previous_metrics))
            # Only the metric is inherited: never years, role, scope or values.
            if key in labels and key != bare_key:
                label = labels[key]
                return RevenueResolution(question, bare.sub(label, question), label,
                    'conversation_metric', digest, previous_question, bare_metric=bare_metric)

    from .document_sources import read_document, chunk_lines
    from .program_data import merge_candidates, currency_unit_scale
    from .evidence_schema import build_evidence_graph
    from .symbolic_verifier import _slot_period_state, _query_source_frame_state, VerificationState
    from .disclosure_equivalence import _scope_key
    chunks = [c for c in chunk_lines(read_document(path))
              if any(term in normalize_text(c.text) for term in search_terms)]
    candidates = merge_candidates(question, chunks, max_candidates=10000)
    candidates = tuple(c for c in candidates if identity(c.row_label) in document_metrics
                       and not c.is_percent and not c.is_year_like)
    if not candidates:
        return unchanged
    cells = build_evidence_graph(candidates).cells
    plans = [o.plan for o in request.outputs if o.plan.operation == 'SELECT']
    if not plans:
        plans = [compile_question_plan(resolve_public_question(question))]
    requirements = [(p, slot) for p in plans for slot in p.operand_slots]
    groups = {}
    signatures = {}
    for cell in cells:
        if cell.embedded_entity:
            continue
        key = (identity(cell.metric_label), cell.entity_id or cell.entity, _scope_key(cell))
        coverage, witnesses = groups.setdefault(key, (set(), []))
        for i, (plan, slot) in enumerate(requirements):
            if plan.requested_source_kind and _query_source_frame_state(plan, cell) is not VerificationState.VALID:
                continue
            if _slot_period_state(plan, slot, cell) is not VerificationState.VALID:
                continue
            role = normalize_text(cell.column_role)
            if cell.column_role_conflict or (role and slot.column_role and role != slot.column_role.lower()):
                continue
            coverage.add(i)
            scale = currency_unit_scale(cell.unit)
            outcome = (key[1], key[2], cell.value * scale if scale is not None else None)
            signatures.setdefault(key[0], {}).setdefault(i, set()).add(outcome)
            witnesses.append({'source_name': cell.source_name, 'page': cell.page,
                'line_number': cell.line_number, 'line_sha256': cell.source_line_sha256,
                'row_label': cell.metric_label, 'column_label': cell.column,
                'period': cell.period, 'column_role': cell.column_role})
    complete = {key[0] for key, (coverage, _) in groups.items()
                if requirements and len(coverage) == len(requirements)}
    available = {key[0] for key, (coverage, _) in groups.items() if coverage}
    choices = complete or available
    basis = 'document_metric_periods'
    if bare_key=='revenue_unspecified' and len(choices) > 1 and 'net_revenue' in complete:
        # A disclosed interpretation of a generic request, not a synonym rule.
        # Equal numbers alone do not establish matching scope/years/roles.
        baseline = signatures['net_revenue']
        comparable = all(len(values) == 1 and next(iter(values))[1] is not None
                         and next(iter(values))[2] is not None for values in baseline.values())
        if comparable and all(signatures[k] == baseline for k in choices):
            choices = {'net_revenue'}
            basis = 'document_net_revenue_interpretation'
    if len(choices) > 1:
        return RevenueResolution(question, question, basis='ambiguous_source_metrics',
            document_sha256=digest, alternatives=tuple(labels[k] for k in sorted(choices)), bare_metric=bare_metric)
    if len(choices) != 1:
        return unchanged
    key = next(iter(choices))
    label = labels[key]
    if key == bare_key:
        return unchanged
    witnesses = tuple(w for group, (_, ws) in groups.items()
                      if group[0] == key or (basis == 'document_net_revenue_interpretation' and group[0] in complete)
                      for w in ws)
    return RevenueResolution(question, bare.sub(label, question), label,
        basis, digest, witnesses=witnesses, bare_metric=bare_metric)
