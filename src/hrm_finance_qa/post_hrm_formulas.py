"""Question contracts for explicitly requested EPS, P/E and profit margin.

Reading a printed EPS is SELECT. Computing EPS needs earnings attributable to
common shareholders and weighted-average common shares; no end-of-year proxy.
"""
from dataclasses import replace
from datetime import date
import re
from .ontology import normalize_text
from .question_planner import compile_question_plan,resolve_public_question,OperandSlot,ProgramIntentNode,DirectionIntent


def margin_operand_plan(question):
    """Expose named margin operands without selecting a neural operation.

    A printed margin remains a separate fact candidate. These slots enumerate
    sourced calculation candidates for the experimental numeric path.
    """
    n=normalize_text(question)
    if ('bien loi nhuan' not in n or not ('phan tram' in n or '%' in question)
            or not ('bao nhieu' in n or re.search(r'\btinh\b',n))
            or ';' in question or '\n' in question):
        return None
    years=set(re.findall(r'\b(?:19|20)\d{2}\b',n))
    if len(years)!=1:return None
    variants=(('gop','Lợi nhuận gộp'),('truoc thue','Lợi nhuận trước thuế'),
              ('rong','Lợi nhuận sau thuế'),('sau thue','Lợi nhuận sau thuế'))
    selected={label for token,label in variants if re.search(r'\b'+token+r'\b',n)}
    if len(selected)!=1:return None
    base=compile_question_plan(resolve_public_question(question))
    year=next(iter(years))
    slots=tuple(OperandSlot(id=key,metric=metric,year=year,
        column_role='ACTUAL',entity_scope=base.entity_mode,
        unit_family='currency',intent_sources=('named_margin_operands',))
        for key,metric in (('profit',next(iter(selected))),('revenue','Doanh thu thuần')))
    return replace(base,operand_slots=slots)


def financial_calculation_plan(question):
    n=normalize_text(question);base=compile_question_plan(resolve_public_question(question))

    # Compute the named ratio only when the question states its two operands.
    if ("he so thanh toan hien hanh" in n
            and "tai san ngan han" in n and "no ngan han" in n
            and re.search(r"\bchia\b", n) and ";" not in question
            and "\n" not in question):
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", n))
        if len(years) != 1:
            return replace(base, abstention_reason="Cần nêu một năm cho hai đầu vào."), "", ""
        endpoint = "closing" if "cuoi nam" in n else "opening" if "dau nam" in n else ""
        year = next(iter(years))
        slots = tuple(OperandSlot(
            id=key, metric=metric, year=year, column_role="ACTUAL",
            entity_scope=base.entity_mode, unit_family="currency",
            intent_sources=("explicit_current_ratio_operands",),
            period_role=endpoint)
            for key, metric in (("current_assets", "Tài sản ngắn hạn"),
                                ("current_liabilities", "Nợ ngắn hạn")))
        node = ProgramIntentNode("answer", "DIVIDE", tuple(s.id for s in slots),
                                 "ratio", ("explicit_current_ratio_operands",))
        return replace(base, operation="DIVIDE",
                       metric_text="Hệ số thanh toán hiện hành",
                       operand_slots=slots, program_dag=(node,),
                       abstention_reason=""), "", ""

    profit_gap=re.fullmatch(
        r'(?:tinh )?(?:chenh lech|do lech)(?: giua)? '
        r'(loi nhuan (?:truoc|sau) thue) va (loi nhuan (?:truoc|sau) thue) '
        r'nam ((?:19|20)\d{2})(?: la bao nhieu(?: dong| vnd)?)?',
        n.strip(' ?.!'),
    )
    if profit_gap and profit_gap[1] != profit_gap[2]:
        left,right,year=profit_gap.groups()
        slots=tuple(OperandSlot(id=key,metric=metric,year=year,
            column_role='ACTUAL',entity_scope=base.entity_mode,
            unit_family='currency',intent_sources=('explicit_same_year_profit_gap',))
            for key,metric in (('gap_left',left),('gap_right',right)))
        node=ProgramIntentNode('answer','SUBTRACT',tuple(slot.id for slot in slots),
            'currency',('explicit_absolute_gap',))
        return replace(base,operation='SUBTRACT',operand_slots=slots,
            operand_roles=(left,right),direction_intent=DirectionIntent.ABSOLUTE_GAP,
            program_dag=(node,),abstention_reason=''),'', ''
    eps=bool(re.search(r'\beps\b|lai co ban tren co phieu',n))
    pe=bool(re.search(r'\bp e\b',n))
    margin='bien loi nhuan' in n and bool(re.search(r'\btinh\b',n))
    greater=re.fullmatch(
        r'(.+?) nam ((?:19|20)\d{2}) co lon hon (.+?) khong',
        n.strip(' ?.!'),
    )
    if greater and not any((eps,pe,margin)):
        left,compare_year,right=greater.groups()
        from .metric_semantics import identity
        known={'operating_cash_flow','profit_after_tax','profit_before_tax',
               'net_revenue','total_revenue','total_assets','total_liabilities',
               'owners_equity'}
        if identity(left) in known and identity(right) in known:
            slots=tuple(OperandSlot(id=key,metric=metric,year=compare_year,
                column_role='ACTUAL',entity_scope=base.entity_mode,
                unit_family='currency',intent_sources=('explicit_greater_than',))
                for key,metric in (('compare_left',left),('compare_right',right)))
            node=ProgramIntentNode('answer','SUBTRACT',tuple(slot.id for slot in slots),
                'currency',('explicit_greater_than',))
            return replace(base,operation='SUBTRACT',operand_slots=slots,
                program_dag=(node,),abstention_reason=''),'', ''
    share=re.fullmatch(
        r'(.+?) chiem bao nhieu phan tram (.+?) (?:((?:cuoi|dau)) nam|nam) ((?:19|20)\d{2})(?: la)?',
        n.strip(' ?.!'),
    )
    if share and not any((eps,pe,margin)):
        numerator,denominator,endpoint,share_year=share.groups()
        # Each operand retains the explicit financial row name and endpoint.
        # The source verifier must still find both in one scope and unit.
        if re.search(r'\b(?:ke hoach|du kien|du bao)\b',denominator):
            return None
        roles='closing' if endpoint=='cuoi' else 'opening' if endpoint=='dau' else ''
        slots=tuple(OperandSlot(id=key,metric=metric,year=share_year,column_role='ACTUAL',
            entity_scope=base.entity_mode,unit_family='currency',
            intent_sources=('explicit_metric_share',),period_role=roles)
            for key,metric in (('share_numerator',numerator),('share_denominator',denominator)))
        node=ProgramIntentNode('answer','RATIO_PERCENT',tuple(slot.id for slot in slots),
            'percent',('explicit_metric_share',))
        return replace(base,operation='RATIO_PERCENT',operand_slots=slots,
            program_dag=(node,),abstention_reason=''),'', ''
    attainment=re.fullmatch(
        r'(.+?) thuc hien nam ((?:19|20)\d{2}) dat bao nhieu phan tram ke hoach',
        n.strip(' ?.!'),
    )
    if attainment and not any((eps,pe,margin)):
        metric,attainment_year=attainment.groups()
        slot=OperandSlot(id='attainment',metric=metric,year=attainment_year,
            column_role='ACTUAL_TO_PLAN_RATIO',entity_scope=base.entity_mode,
            unit_family='percent',intent_sources=('explicit_actual_to_plan',))
        node=ProgramIntentNode('answer','SELECT',(slot.id,),
            'percent',('explicit_actual_to_plan',))
        return replace(base,operation='SELECT',operand_slots=(slot,),
            program_dag=(node,),abstention_reason=''),'', ''
    if not any((eps,pe,margin)):return None
    # This bounded route does not partition unrelated multi-part questions.
    if ';' in question or '\n' in question:return None
    calculate=bool(re.search(r'\btinh\b',n)) or pe or margin
    fiscal=re.search(r'\bnam\s+((?:19|20)\d{2})\b',n)
    year=fiscal[1] if fiscal else ''
    role='PLAN' if re.search(r'\bke hoach\b|\bkh\b',n) else 'ACTUAL'
    price_date='';error='';name='';spec=[];operation='SELECT'
    if eps and not pe:
        if calculate:
            name='eps';operation='DIVIDE'
            spec=[('earnings_attributable_to_common','Lợi nhuận phân bổ cho cổ đông phổ thông',year,'currency'),('weighted_average_common_shares','Số cổ phiếu phổ thông bình quân gia quyền',year,'shares')]
        else:spec=[('reported_eps','Lãi cơ bản trên cổ phiếu',year,'currency_per_share')]
    elif pe:
        name='pe';operation='DIVIDE'
        match=re.search(r'(?<!\d)(\d{1,2})[/.](\d{1,2})[/.]((?:19|20)\d{2})(?!\d)',question)
        if match:
            try:price_date=date(int(match[3]),int(match[2]),int(match[1])).isoformat()
            except ValueError:error='Ngày giá cổ phiếu không hợp lệ.'
        if not price_date:error='Cần ngày giá cổ phiếu trong nguồn để tính P/E.'
        spec=[('price_per_share','Giá cổ phiếu',price_date,'currency_per_share'),('eps','Lãi cơ bản trên cổ phiếu',year,'currency_per_share')]
    else:
        name='margin';operation='RATIO_PERCENT'
        profit='Lợi nhuận sau thuế' if 'sau thue' in n or 'rong' in n else 'Lợi nhuận trước thuế' if 'truoc thue' in n else 'Lợi nhuận gộp' if 'gop' in n else ''
        if not profit:error='Cần xác định biên lợi nhuận gộp, trước thuế hay sau thuế.'
        spec=[('profit',profit or 'Lợi nhuận',year,'currency'),('revenue','Doanh thu thuần',year,'currency')]
    if calculate and not year:error=error or 'Cần năm của đầu vào để tính chỉ tiêu này.'
    slots=tuple(OperandSlot(id=id,metric=metric,year=period,column_role=role,entity_scope=base.entity_mode,unit_family=unit,intent_sources=('explicit_financial_calculation',)) for id,metric,period,unit in spec)
    nodes=(ProgramIntentNode('answer',operation,tuple(s.id for s in slots),'percent' if operation=='RATIO_PERCENT' else 'currency_per_share' if eps and not pe else 'ratio',('explicit_financial_calculation',)),)
    return replace(base,operation=operation,operand_slots=slots,program_dag=nodes,abstention_reason=error),name,price_date
