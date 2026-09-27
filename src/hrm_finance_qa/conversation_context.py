"""Resolve explicit metric-only follow-ups before compiling question intent."""
import re
from datetime import datetime
from .ontology import normalize_text
from .metric_semantics import identity, ALIASES, normalized_label
from .question_planner import compile_question_plan, resolve_public_question


def _metric_followup(question, previous_question):
    match = re.fullmatch(r'\s*(?:thế\s+còn|vậy\s+còn|còn|the\s+con|vay\s+con|con)\s+(.+?)\s*[?.!]*\s*', question, re.I)
    if not match or not previous_question:
        return None
    replacement = match[1].strip(' ?.!.')
    # An explicit new date/scope/operation is a new instruction, not a metric
    # substitution. Unknown qualifiers must not be silently discarded.
    if identity(replacement) not in ALIASES:
        return None
    prior = compile_question_plan(resolve_public_question(previous_question))
    if not prior.operation or not prior.requested_years:
        return None
    if prior.abstention_reason not in {'', 'ambiguous_comparison_measure'}:
        return None
    expected = normalized_label(prior.metric_text)
    words = list(re.finditer(r'\w+', previous_question))
    spans = []
    for i in range(len(words)):
        for j in range(i, min(i + 14, len(words))):
            a, b = words[i].start(), words[j].end()
            if normalized_label(previous_question[a:b]) == expected:
                spans.append((a, b))
    if not spans:
        return None
    a, b = max(spans, key=lambda span: span[1] - span[0])
    # More than one disjoint metric occurrence needs full request resolution.
    if any(z <= a or x >= b for x, z in spans):
        return None
    canonical = previous_question[:a] + replacement + previous_question[b:]
    after = compile_question_plan(resolve_public_question(canonical))
    if (after.operation, after.requested_years, after.column_roles, after.direction_intent) != (prior.operation, prior.requested_years, prior.column_roles, prior.direction_intent):
        return None
    return {'original_question': question, 'resolved_question': canonical,
            'previous_question': previous_question, 'basis': 'explicit_metric_followup',
            'inherited': {'years': list(prior.requested_years), 'operation': prior.operation,
                          'column_roles': list(prior.column_roles), 'scope': prior.requested_accounting_scope}}


def _normal(question):
    return normalize_text(question.replace('%',' phần trăm '))

_SCOPE_WORDS={'consolidated':'hợp nhất','separate':'riêng lẻ'}
_ENDPOINT=re.compile(r'^(?:the con|vay con|con) (dau|cuoi) (nam|ky)(?: ((?:19|20)\d{2}))?(?: (hop nhat|rieng le))?$')
_COMPARE=re.compile(r'^(?:so voi(?: nam)? ((?:19|20)\d{2}) (?:tang|giam|tang hay giam|giam hay tang) bao nhieu phan tram|co (?:tang|giam) so voi(?: nam)? ((?:19|20)\d{2}) khong (?:tang|giam) bao nhieu phan tram)(?: (hop nhat|rieng le))?$')


def parse_period_choice(reply):
    """Parse only a bare period choice; keep day and year precision distinct."""
    value=_normal(reply).strip(' \t\r\n.?!')
    value=re.sub(r'^(?:toi chon|cho toi|lay|dung)\s+', '', value)
    match=re.fullmatch(
        r'(?:(dau|cuoi)(?:\s+(?:ky|nam))?)?(?:\s*ngay)?\s*'
        r'(?:(\d{1,2}\s+\d{1,2}\s+(?:19|20)\d{2})|((?:nam\s+)?(?:19|20)\d{2}))?',
        value)
    if not match or not any(match.groups()):return None
    role={'dau':'opening','cuoi':'closing'}.get(match[1], '')
    if match[2]:
        try:period=datetime.strptime(match[2],'%d %m %Y').date().isoformat()
        except ValueError:return None
    else:
        period=(match[3] or '').removeprefix('nam ').strip()
    return role,period


def period_question(question, role, period, *, replace=False):
    """Carry one SELECT metric/scope forward using an evidenced period."""
    if role not in {'','opening','closing'}:return None
    if re.fullmatch(r'(?:19|20)\d{2}',period):
        year=period
        qualifier=(('đầu ' if role=='opening' else 'cuối ') if role else '')+'năm '+year
    elif role and re.fullmatch(r'(?:19|20)\d{2}-\d{2}-\d{2}',period):
        try:date=datetime.fromisoformat(period)
        except ValueError:return None
        year=period[:4]
        qualifier=('đầu' if role=='opening' else 'cuối')+' kỳ ngày '+date.strftime('%d/%m/%Y')
    else:return None
    prior=compile_question_plan(resolve_public_question(question))
    if prior.operation!='SELECT' or prior.abstention_reason or len(prior.operand_slots)!=1 or not prior.metric_text:return None
    if prior.requested_report_year or prior.requested_source_years or prior.requested_source_year_count:return None
    if not replace and prior.requested_years and prior.requested_years!=(year,):return None
    if replace:
        old=re.compile(
            r'\b(?:(?:đầu|cuối)\s+(?:kỳ|năm)(?:\s+ngày\s+\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}'
            r'|\s+(?:19|20)\d{2})|năm\s+(?:19|20)\d{2})',re.I)
        if len(old.findall(question))!=1:return None
        rewritten=old.sub(qualifier,question,count=1)
    else:
        if prior.requested_period_roles:return None
        words=list(re.finditer(r'\w+',question))
        spans=[]
        for i in range(len(words)):
            for j in range(i,min(i+16,len(words))):
                a,b=words[i].start(),words[j].end()
                if normalize_text(question[a:b])==normalize_text(prior.metric_text):spans.append((a,b))
        if len(spans)!=1:return None
        _,end=spans[0]
        rewritten=question[:end]+' '+qualifier+question[end:]
    after=compile_question_plan(resolve_public_question(rewritten))
    if (after.abstention_reason or after.operation!='SELECT' or len(after.operand_slots)!=1
            or identity(after.metric_text)!=identity(prior.metric_text)
            or after.entity_mode!=prior.entity_mode
            or after.requested_accounting_scope!=prior.requested_accounting_scope
            or after.requested_period_roles!=((role,) if role else ())
            or after.requested_years!=(year,)):
        return None
    return rewritten


def public_intent_message(reason):
    return {
        'ambiguous_signed_direction':'Bạn muốn lấy chỉ tiêu nào trừ chỉ tiêu nào, hay tính độ chênh lệch tuyệt đối?',
        'ambiguous_comparison_measure':'Bạn muốn so sánh chênh lệch số tiền hay tỷ lệ phần trăm?',
        'missing_column_role':'Bạn muốn dùng số thực hiện hay số kế hoạch?',
        'source_scope_ambiguous':'Bạn muốn dùng báo cáo hợp nhất hay riêng lẻ? Nguồn còn công bố chưa rõ phạm vi.',
        'missing_reference_base':'Bạn muốn so sánh chỉ tiêu nào, của năm nào với năm nào?',
        'followup_context_missing':'Bạn muốn hỏi chỉ tiêu nào và năm nào? Vui lòng nêu rõ để tra lại trong tài liệu đã chọn.',
    }.get(reason,'Chưa xác định đủ yêu cầu để trả lời. Bạn vui lòng nêu rõ chỉ tiêu, kỳ và phép tính cần dùng.' if re.fullmatch(r'[a-z][a-z0-9_]*',reason) else reason)

def followup_clarification(question):
    n=_normal(question)
    if re.fullmatch(r'(?:the con|vay con|con) nam (?:19|20)\d{2}',n):
        return 'Bạn muốn xem riêng số liệu của năm này hay đổi năm trong phép so sánh? Vui lòng nêu rõ chỉ tiêu và cách so sánh.'
    if re.fullmatch(r'(?:nam )?(?:19|20)\d{2}',n):
        return public_intent_message('followup_context_missing')
    if re.fullmatch(r'(?:vay )?so voi(?: nam)? (?:19|20)\d{2} thi sao',n):
        return public_intent_message('followup_context_missing')
    if re.match(r'^(?:the con|vay con|con) (?:dau|cuoi) (?:nam|ky)\b',n):
        return public_intent_message('followup_context_missing')
    if re.match(r'^(?:so voi|co (?:tang|giam) so voi)\b',n) and 'phan tram' in n:
        return public_intent_message('followup_context_missing')
    return ''

def resolve_followup(question, previous_question):
    """Complete intent only. No previous answer/value is accepted or reused."""
    metric_result=_metric_followup(question,previous_question)
    if metric_result is not None:return metric_result
    if not previous_question:return None
    # A short currency follow-up changes only the requested unit. Recompile
    # the previous intent from its question; never carry a raw source value.
    normalized=_normal(question).strip(' .?!')
    if (re.fullmatch(r'(?:(?:vay|the) )?(?:quy ra|quy doi sang|bang|la) dong (?:la )?bao nhieu',normalized)
            or re.fullmatch(r'(?:(?:vay|the) )?bao nhieu dong',normalized)):
        prior=compile_question_plan(resolve_public_question(previous_question))
        if (prior.operation=='SELECT' and not prior.abstention_reason
                and len(prior.operand_slots)==1 and len(prior.requested_years)==1
                and not re.search(r'\b(?:dong|vnd|nghin|trieu|ty|usd|eur)\b',_normal(previous_question))
                and re.search(r'bao nhiêu\s*[?.!]*\s*$',previous_question,re.I)):
            rewritten=re.sub(r'bao nhiêu(\s*[?.!]*\s*)$',r'bao nhiêu đồng\1',previous_question,flags=re.I)
            after=compile_question_plan(resolve_public_question(rewritten))
            if (not after.abstention_reason and after.operation=='SELECT'
                    and len(after.operand_slots)==1 and after.requested_years==prior.requested_years
                    and identity(after.metric_text)==identity(prior.metric_text)
                    and after.requested_accounting_scope==prior.requested_accounting_scope
                    and after.requested_period_roles==prior.requested_period_roles):
                return {'original_question':question,'resolved_question':rewritten,
                        'previous_question':previous_question,'basis':'explicit_currency_followup',
                        'inherited':{'metric':prior.metric_text,'years':list(prior.requested_years),
                                     'scope':prior.requested_accounting_scope,'operation':'SELECT'}}
    choice=parse_period_choice(question)
    if choice is not None and choice[0] and choice[1]:
        prior=compile_question_plan(resolve_public_question(previous_question))
        if prior.operation=='SELECT' and len(prior.requested_years)==1:
            resolved=period_question(previous_question,*choice,replace=True)
            if resolved is not None:
                return {'original_question':question,'resolved_question':resolved,
                        'previous_question':previous_question,'basis':'dated_endpoint_followup',
                        'inherited':{'metric':prior.metric_text,'scope':prior.requested_accounting_scope,
                                     'operation':prior.operation}}
    n=_normal(question)
    year_only=re.fullmatch(r'(?:the con|vay con|con) nam ((?:19|20)\d{2})',n)
    if year_only:
        prior=compile_question_plan(resolve_public_question(previous_question))
        if (prior.operation!='SELECT' or prior.abstention_reason or len(prior.operand_slots)!=1
                or len(prior.requested_years)!=1 or not prior.metric_text
                or prior.requested_report_year or prior.requested_source_years):return None
        old=prior.requested_years[0]
        if set(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)',previous_question))!={old}:return None
        resolved=re.sub(r'(?<!\d)'+re.escape(old)+r'(?!\d)',year_only[1],previous_question)
        after=compile_question_plan(resolve_public_question(resolved))
        if after.abstention_reason or after.operation!='SELECT' or after.requested_years!=(year_only[1],):return None
        return {'original_question':question,'resolved_question':resolved,'previous_question':previous_question,
                'basis':'explicit_year_followup','inherited':{'metric':prior.metric_text,'scope':prior.requested_accounting_scope,
                'operation':prior.operation,'column_roles':list(prior.column_roles)}}
    elliptical=re.fullmatch(r'(?:vay )?so voi(?: nam)? ((?:19|20)\d{2}) thi sao',n)
    if elliptical:
        prior_operation=compile_question_plan(resolve_public_question(previous_question)).operation
        if prior_operation!='PERCENT_CHANGE':return None
        n=f'so voi nam {elliptical[1]} tang hay giam bao nhieu phan tram'
    endpoint=_ENDPOINT.fullmatch(n);comparison=_COMPARE.fullmatch(n)
    if not endpoint and not comparison:return None
    prior=compile_question_plan(resolve_public_question(previous_question))
    # Only a single, grounded lookup can supply the omitted metric/year.
    # Complex event/table/date constraints must not silently disappear.
    if prior.abstention_reason:return None
    if prior.operation=='SELECT' and len(prior.requested_years)==1:
        current=prior.requested_years[0]
    elif comparison and prior.operation=='PERCENT_CHANGE' and len(prior.operand_slots)==2:
        bases=[s for s in prior.operand_slots if s.id==prior.reference_base]
        targets=[s for s in prior.operand_slots if s.id!=prior.reference_base]
        if len(bases)!=1 or len(targets)!=1 or not targets[0].year or targets[0].year==bases[0].year:return None
        if len({(s.column_role,s.entity_scope,s.period_role) for s in prior.operand_slots})!=1:return None
        current=targets[0].year
    else:return None
    if not prior.operand_slots or len({identity(s.metric) for s in prior.operand_slots})!=1:return None
    if prior.entity_mode!='reporting' or prior.requested_event or prior.requested_source_kind or prior.requested_source_heading:return None
    if prior.requested_value_column or prior.requested_section or prior.requested_report_year or prior.requested_source_years or prior.requested_source_year_count:return None
    if re.search(r'\bquy\b|ban nien|\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}',previous_question.lower()):return None
    metric=prior.operand_slots[0].metric
    if normalize_text(metric) in {'con','voi tang','co giam so voi khong giam',''}:return None
    explicit_scope=endpoint[4] if endpoint else comparison[3]
    scope={'hop nhat':'hợp nhất','rieng le':'riêng lẻ'}.get(explicit_scope) or _SCOPE_WORDS.get(prior.requested_accounting_scope,'')
    role=prior.operand_slots[0].column_role
    if role not in {'','ACTUAL','PLAN'}:return None
    role_text='kế hoạch' if role=='PLAN' else ''
    if endpoint:
        year=endpoint[3] or current
        when=('đầu' if endpoint[1]=='dau' else 'cuối')+' '+('năm' if endpoint[2]=='nam' else 'kỳ')
        canonical=f'{metric} {role_text} {scope} {when} năm {year} là bao nhiêu?'
        canonical=canonical.replace('năm năm','năm')
        basis='endpoint_followup'
    else:
        if prior.requested_period_roles:return None
        reference=comparison[1] or comparison[2]
        if reference==current:return None
        canonical=f'{metric} {role_text} {scope} năm {current} tăng bao nhiêu phần trăm so với năm {reference}?'
        basis='comparison_followup'
    canonical=re.sub(r'\s+',' ',canonical).strip()
    after=compile_question_plan(resolve_public_question(canonical))
    if after.abstention_reason or not after.operand_slots:return None
    if any(identity(slot.metric)!=identity(metric) for slot in after.operand_slots):return None
    return {'original_question':question,'resolved_question':canonical,'previous_question':previous_question,
            'basis':basis,'inherited':{'metric':metric,'years':[current] if comparison or not endpoint[3] else [],
                                     'scope':scope if not explicit_scope else '', 'column_role':role},
            'explicit':{'year':(endpoint[3] or '') if endpoint else reference,'scope':scope if explicit_scope else ''}}


def complete_followup_year(reply, question, previous_question):
    """Fill only the target year of a pending comparison with known intent."""
    year=re.fullmatch(r'(?:nam )?((?:19|20)\d{2})',_normal(reply))
    comparison=_COMPARE.fullmatch(_normal(question))
    if not year or not comparison or not previous_question:return None
    # Use the existing resolver's complete constraint checks, including scope.
    probe=resolve_followup(question,previous_question)
    if probe is None:
        # Same-year comparisons are deliberately blocked; change only the known
        # lookup year, then apply the normal resolver and all its checks.
        prior=compile_question_plan(resolve_public_question(previous_question))
        if prior.operation!='SELECT' or len(prior.requested_years)!=1:return None
        seed=re.sub(r'(?<!\d)'+re.escape(prior.requested_years[0])+r'(?!\d)',year[1],previous_question)
        probe=resolve_followup(question,seed)
        return probe['resolved_question'] if probe else None
    plan=compile_question_plan(resolve_public_question(probe['resolved_question']))
    target=next(s.year for s in plan.operand_slots if s.id!=plan.reference_base)
    reference=next(s.year for s in plan.operand_slots if s.id==plan.reference_base)
    if year[1]==reference:return None
    return re.sub(r'(?<!\d)'+re.escape(target)+r'(?!\d)',year[1],probe['resolved_question'])
