"""Complete an offered metric choice within the pending document question."""
import json
import re
from hrm_finance_qa.metric_semantics import identity
from hrm_finance_qa.ontology import normalize_text

_BARE = {
    'loi nhuan': re.compile(r'\b(?:lợi\s+nhuận|loi\s+nhuan)\b', re.I),
    'doanh thu': re.compile(r'\b(?:doanh\s+thu|revenue|sales)\b', re.I),
}

def pending_metric_choice(result, question, document_sha256):
    resolution = result.get('question_resolution') or {}
    if (result.get('status') != 'ABSTAIN'
            or resolution.get('basis') not in {'ambiguous_source_metrics','unmapped_source_metrics'}
            or resolution.get('document_sha256') != document_sha256
            or resolution.get('original_question') != question.strip()):
        return None
    options = resolution.get('alternatives')
    bare = normalize_text(resolution.get('bare_metric', ''))
    if (bare not in _BARE or not isinstance(options, (list, tuple))
            or not 2 <= len(options) <= 16
            or any(not isinstance(x, str) or not x.strip() for x in options)):
        return None
    return json.dumps({'kind':'metric_choice','question': question.strip(), 'alternatives': options,
        'bare_metric': bare, 'assistant_prompt':result.get('rendered',''),
        'document_sha256': document_sha256}, ensure_ascii=False)

def complete_metric_choice(reply, pending, document_sha256):
    """Keep the pending operation/years; a fresh dated/qualified query is not a choice."""
    if not pending:
        return None
    try:
        context = json.loads(pending)
    except (ValueError, TypeError):
        return None
    if not isinstance(context, dict) or context.get('document_sha256') != document_sha256:
        return None
    bare = context.get('bare_metric')
    question = context.get('question')
    options = context.get('alternatives')
    if (bare not in _BARE or not isinstance(question, str)
            or not isinstance(options, list) or any(not isinstance(x, str) for x in options)):
        return None
    choice = normalize_text(reply).strip(' .?!:')
    choice = re.sub(r'^(?:toi muon hoi|toi chon|cho toi|chon|la)\s+', '', choice)
    choice = re.sub(r'\s+(?:nhe|di|a)$', '', choice)
    if not choice or len(choice) > 160 or re.search(r'\d', choice):
        return None
    matches = []
    for label in options:
        normalized = normalize_text(label)
        short = normalized[len(bare):].strip() if normalized.startswith(bare + ' ') else normalized
        if choice in {normalized, short} or identity(choice) == identity(label):
            matches.append(label)
    if len(matches) != 1:
        return None
    rewritten, count = _BARE[bare].subn(matches[0], question)
    if not count:return None
    # Replacing a known metric with a literal source label must not lose the
    # original question's default actual basis. This changes intent, not facts.
    from hrm_finance_qa.question_planner import compile_question_plan,resolve_public_question
    original=compile_question_plan(resolve_public_question(question))
    completed=compile_question_plan(resolve_public_question(rewritten))
    if original.column_roles and set(original.column_roles)=={'ACTUAL'} and not completed.column_roles:
        rewritten=_BARE[bare].sub(matches[0]+' thực hiện',question)
    return rewritten



def pending_named_ratio_year(result, question, document_sha256):
    choice = result.get('named_ratio_year_choices') or {}
    years = choice.get('years')
    if (result.get('status') != 'ABSTAIN'
            or choice.get('question') != question
            or choice.get('metric') not in {'return_on_assets','return_on_equity'}
            or not isinstance(years, list) or not 1 <= len(years) <= 8
            or any(not isinstance(y, str) or not re.fullmatch(r'(?:19|20)\d{2}', y)
                   for y in years)
            or any(c.get('document_sha256') != document_sha256
                   for c in result.get('citations') or [])):
        return None
    return json.dumps({'kind':'named_ratio_year','question':question,
        'metric':choice['metric'],'years':years,
        'assistant_prompt':result.get('rendered',''),
        'document_sha256':document_sha256},ensure_ascii=False)


def complete_pending_named_ratio_year(reply, pending, document_sha256):
    from hrm_finance_qa.conversation_context import parse_period_choice
    try: context=json.loads(pending or 'null')
    except (ValueError,TypeError): return None
    if (not isinstance(context,dict) or context.get('kind')!='named_ratio_year'
            or context.get('document_sha256')!=document_sha256): return None
    parsed=parse_period_choice(reply)
    if parsed is None or parsed[0] or parsed[1] not in context.get('years',[]): return None
    metric=context.get('metric')
    if metric not in {'return_on_assets','return_on_equity'}: return None
    name='ROA' if metric=='return_on_assets' else 'ROE'
    return f'{name} năm {parsed[1]} là bao nhiêu?'


def pending_followup_year(result, question, previous_question, document_sha256):
    from hrm_finance_qa.conversation_context import _COMPARE, _normal
    if not result.get('intent_clarification_required') or not _COMPARE.fullmatch(_normal(question)):
        return None
    return json.dumps({'kind':'followup_year','question':question,
        'previous_question':previous_question or '', 'assistant_prompt':result.get('rendered',''),
        'document_sha256':document_sha256},ensure_ascii=False)


def complete_pending_year(reply, pending, document_sha256):
    from hrm_finance_qa.conversation_context import complete_followup_year
    try:context=json.loads(pending or 'null')
    except (ValueError,TypeError):return None
    if not isinstance(context,dict) or context.get('kind')!='followup_year' or context.get('document_sha256')!=document_sha256:return None
    return complete_followup_year(reply,context['question'],context['previous_question'])


def pending_period_choice(result, question, document_sha256):
    """Keep only period options proved by the ambiguity's source facts."""
    from hrm_finance_qa.question_planner import compile_question_plan, resolve_public_question

    if result.get('status')!='ABSTAIN':return None
    checks=(result.get('validation') or {}).get('checks') or []
    period_checks=[c for c in checks if c.get('code')=='source_period_ambiguous'
                   and c.get('status')=='INSUFFICIENT_EVIDENCE' and c.get('blocking')]
    if len(period_checks)!=1:return None
    plan=compile_question_plan(resolve_public_question(question))
    if plan.operation!='SELECT' or plan.abstention_reason or len(plan.operand_slots)!=1 or plan.requested_period_roles:return None
    fact_ids=set(period_checks[0].get('fact_ids') or [])
    options=set()
    scopes=set()
    for citation in result.get('citations') or []:
        if citation.get('operand_ref') not in fact_ids:continue
        if citation.get('document_sha256')!=document_sha256:continue
        if identity(citation.get('row_label',''))!=identity(plan.metric_text):continue
        role=citation.get('period_role')
        if role not in {'','opening','closing'}:continue
        date=citation.get('as_of_date')
        year=citation.get('period')
        header=normalize_text(citation.get('column_label',''))
        binding=citation.get('period_binding')
        if role and isinstance(date,str) and re.fullmatch(r'(?:19|20)\d{2}-\d{2}-\d{2}',date):
            period=date
        elif isinstance(year,str) and re.fullmatch(r'(?:19|20)\d{2}',year):
            if role:
                marker='dau' if role=='opening' else 'cuoi'
                if (not re.search(r'\b'+marker+r'\b',header)
                        or not (binding=='audited_statement_period'
                                or (binding=='source_column' and year in header))):
                    continue
            elif binding!='source_column' or year not in header:
                continue
            period=year
        else:continue
        options.add((role,period))
        scopes.add(citation.get('entity_scope',''))
    if not 2<=len(options)<=8 or len(scopes)!=1:return None
    return json.dumps({'kind':'period_choice','question':question.strip(),
                       'assistant_prompt':result.get('rendered',''),
                       'options':[{'role':r,'period':d} for r,d in sorted(options)],
                       'document_sha256':document_sha256},ensure_ascii=False)


def complete_pending_period(reply, pending, document_sha256):
    from hrm_finance_qa.conversation_context import parse_period_choice, period_question

    try:context=json.loads(pending or 'null')
    except (ValueError,TypeError):return None
    if not isinstance(context,dict) or context.get('kind')!='period_choice' or context.get('document_sha256')!=document_sha256:return None
    parsed=parse_period_choice(reply)
    if parsed is None:return None
    role,period=parsed
    options=context.get('options')
    if not isinstance(options,list):return None
    matches=[x for x in options if isinstance(x,dict) and (not role or x.get('role')==role)
             and (not period or x.get('period')==period)]
    if len(matches)!=1:return None
    return period_question(context.get('question',''),matches[0]['role'],matches[0]['period'])


def unmatched_period_reply(reply, pending, document_sha256):
    """Keep the original clarification alive when a short reply selects no source option."""
    from hrm_finance_qa.conversation_context import parse_period_choice

    try:context=json.loads(pending or 'null')
    except (ValueError,TypeError):return None
    if not isinstance(context,dict) or context.get('kind')!='period_choice' or context.get('document_sha256')!=document_sha256:return None
    if parse_period_choice(reply) is None or complete_pending_period(reply,pending,document_sha256) is not None:return None
    prompt=context.get('assistant_prompt','')
    if not isinstance(prompt,str) or not prompt.strip():return None
    return 'Chưa xác nhận được thời điểm bạn chọn từ nguồn. '+prompt.strip()


def _scope_choice(reply):
    choice=normalize_text(reply).strip(' .?!:')
    choice=re.sub(r'^(?:toi chon|cho toi|lay|dung|chon)\s+', '', choice)
    if choice in {'hop nhat','bao cao hop nhat'}:return 'consolidated'
    if choice in {'rieng le','bao cao rieng','bao cao rieng le'}:return 'separate'
    return None


def _citation_accounting_scope(citation):
    value=str(citation.get('entity_scope',''))
    if value in {'consolidated','separate'}:return value
    for scope in ('consolidated','separate'):
        if value.endswith(':scope:'+scope):return scope
    return ''


def pending_scope_choice(result, question, document_sha256):
    """Offer only perimeters proved by the assistant's source-bound citations."""
    from hrm_finance_qa.question_planner import compile_question_plan,resolve_public_question
    if result.get('status')!='ABSTAIN':return None
    checks=(result.get('validation') or {}).get('checks') or []
    scope_checks=[check for check in checks if check.get('code')=='source_scope_ambiguous'
                  and check.get('status')=='INSUFFICIENT_EVIDENCE' and check.get('blocking')]
    if len(scope_checks)!=1:return None
    plan=compile_question_plan(resolve_public_question(question))
    if plan.operation!='SELECT' or plan.abstention_reason or not plan.metric_text or plan.requested_accounting_scope:return None
    if len(plan.requested_years)>1 or plan.requested_event or plan.requested_source_kind:return None
    fact_ids=set(scope_checks[0].get('fact_ids') or [])
    offered=[]
    for citation in result.get('citations') or []:
        if citation.get('operand_ref') not in fact_ids or citation.get('document_sha256')!=document_sha256:continue
        scope=_citation_accounting_scope(citation)
        if scope:
            offered.append({'scope':scope,'operand_ref':citation['operand_ref'],
                            'row_label':citation.get('row_label',''),'period':citation.get('period',''),
                            'page':citation.get('page')})
    options=sorted({item['scope'] for item in offered})
    if not options:return None
    return json.dumps({'kind':'scope_choice','question':question.strip(),
        'assistant_prompt':result.get('rendered',''),'options':options,
        'offered_disclosures':offered,'document_sha256':document_sha256},ensure_ascii=False)


def complete_pending_scope(reply, pending, document_sha256):
    """Carry the previous metric/period, apply the selected scope, then requery."""
    from hrm_finance_qa.question_planner import compile_question_plan,resolve_public_question
    try:context=json.loads(pending or 'null')
    except (ValueError,TypeError):return None
    if not isinstance(context,dict) or context.get('kind')!='scope_choice' or context.get('document_sha256')!=document_sha256:return None
    scope=_scope_choice(reply)
    if scope is None or scope not in context.get('options',[]):return None
    question=context.get('question','')
    if not isinstance(question,str) or not question.strip():return None
    original=compile_question_plan(resolve_public_question(question))
    wording='hợp nhất' if scope=='consolidated' else 'riêng lẻ'
    rewritten=question.rstrip().rstrip('?.! ').rstrip()+' '+wording+'?'
    completed=compile_question_plan(resolve_public_question(rewritten))
    if (original.operation!='SELECT' or completed.operation!='SELECT'
            or completed.abstention_reason or completed.requested_accounting_scope!=scope
            or identity(completed.metric_text)!=identity(original.metric_text)
            or completed.requested_years!=original.requested_years
            or completed.requested_period_roles!=original.requested_period_roles
            or completed.requested_unit_family!=original.requested_unit_family
            or completed.entity_mode!=original.entity_mode):return None
    return rewritten

def begin_clarification(result, question, previous_question, document_sha256):
    """Store the original request, the assistant's question, and source choices."""
    return (pending_metric_choice(result,question,document_sha256)
            or pending_period_choice(result,question,document_sha256)
            or pending_named_ratio_year(result,question,document_sha256)
            or pending_scope_choice(result,question,document_sha256)
            or pending_followup_year(result,question,previous_question,document_sha256))


def resume_clarification(reply, pending, document_sha256):
    """Apply a short reply to the pending request, or leave a new question alone."""
    from hrm_finance_qa.conversation_context import parse_period_choice

    try:context=json.loads(pending or 'null')
    except (ValueError,TypeError):return None
    if not isinstance(context,dict) or context.get('document_sha256')!=document_sha256:return None
    kind=context.get('kind') or ('metric_choice' if context.get('bare_metric') else '')
    if kind=='metric_choice':
        completed=complete_metric_choice(reply,pending,document_sha256)
    elif kind=='period_choice':
        completed=complete_pending_period(reply,pending,document_sha256)
    elif kind=='named_ratio_year':
        completed=complete_pending_named_ratio_year(reply,pending,document_sha256)
    elif kind=='followup_year':
        completed=complete_pending_year(reply,pending,document_sha256)
    elif kind=='scope_choice':
        completed=complete_pending_scope(reply,pending,document_sha256)
    else:return None
    if completed:return {'question':completed}
    if kind=='period_choice':
        message=unmatched_period_reply(reply,pending,document_sha256)
    elif kind=='scope_choice' and _scope_choice(reply):
        selected=_scope_choice(reply)
        if selected not in context.get('options',[]):
            label='hợp nhất' if selected=='consolidated' else 'riêng lẻ'
            available=', '.join('hợp nhất' if x=='consolidated' else 'riêng lẻ' for x in context.get('options',[]))
            message=f'Nguồn chưa chứng minh số liệu {label} cho chỉ tiêu và kỳ đang hỏi. Phạm vi có chứng cứ: {available}. Bạn có thể chọn phạm vi đó hoặc đặt câu hỏi đầy đủ khác.'
        else:
            message='Chưa ghép được phạm vi đã chọn với câu hỏi trước; vui lòng nêu đầy đủ chỉ tiêu, kỳ và phạm vi.'
    else:
        choice=normalize_text(reply).strip(' .?!:')
        if kind in {'followup_year','named_ratio_year'}:
            fragment=bool(re.fullmatch(r'(?:nam )?(?:19|20)\d{2}',choice))
        elif kind=='scope_choice':
            fragment=_scope_choice(reply) is not None
        else:
            fragment=bool(choice and len(choice)<=160 and not re.search(
                r'\d|\b(?:bao nhieu|tinh|tru|so voi|tai sao)\b',choice))
        prompt=context.get('assistant_prompt','')
        message=('Lựa chọn này chưa đủ để xác nhận. '+prompt.strip()
                 if fragment and isinstance(prompt,str) and prompt.strip() else None)
    if message:return {'message':message,'question':context.get('question',''),'pending':pending}
    return None


