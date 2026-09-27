"""Read a uniquely published financial ratio from its physical source row."""
from __future__ import annotations
from decimal import Decimal, ROUND_HALF_UP
import re
from .ontology import normalize_text
from .post_hrm_source import metric_key


def _shown(value):
    if not value:return '0'
    shown=format(value,'f')
    return (shown.rstrip('0').rstrip('.') if '.' in shown else shown).replace('.',',')


def _response(question, fact, companions, rendered, *, warning=''):
    citations=list({c['operand_ref']:c for f in (fact,*companions) for c in [f.citation()]}.values())
    if warning:rendered+='\n'+warning
    return {'status':'ACCEPTED','final_action':'ANSWER','answer':str(fact.value),
            'canonical_value':str(fact.value),'unit':fact.unit,'display_value':str(fact.value),
            'display_unit':fact.unit,'answer_type':'published_ratio','numeric_answer':True,
            'rendered':rendered,'citations':citations,
            'validation_engine':'post_hrm_contract_v1','execution_route':'published_ratio_lookup',
            'proof':{'verification_engine':'post_hrm_contract_v1','source_action':'ANSWER',
                     'source_certificate':{'citations':citations},'execution_certificate':None},
            'question':question}


def _same_physical_row(left,right):
    return (left.source_name==right.source_name and left.line_number==right.line_number
            and left.table_key.rsplit('|',1)[0]==right.table_key.rsplit('|',1)[0])


def answer_published_ratio(question, facts, *, fact_options_only=False):
    q=normalize_text(question)
    years=tuple(dict.fromkeys(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)',q)))
    key='';mode='';expected_unit=''
    if ('doanh thu thuan' in q and 'phan tram' in q and 'so voi' in q
            and len(years)==2 and ('tang' in q or 'giam' in q)):
        key='net_revenue';mode='year_change';expected_unit='%'
    elif ('phai thu ngan han' in q and 'bien dong' in q and 'so voi' in q
          and len(years)==2):
        key=metric_key('Các khoản phải thu ngắn hạn');mode='year_change';expected_unit='%'
    elif ('thuc hien nam' in q and 'ke hoach' in q and 'phan tram' in q
          and len(years)==1):
        metric=q.split('thuc hien nam',1)[0].strip()
        key=metric_key(metric);mode='plan_attainment';expected_unit='%'
    elif len(years)==1 and 'bao nhieu' in q:
        matched=[(metric,unit) for phrase,metric,unit in (
            ('he so no tren tong tai san','debt_to_assets_ratio','x'),
            ('he so no tren von chu so huu','debt_to_equity_ratio','x'),
            ('roe','return_on_equity','%'),
            ('roa','return_on_assets','%'),
            ('vong quay tong tai san','total_asset_turnover','ratio'),
        ) if re.search(r'\b'+re.escape(phrase)+r'\b',q)]
        if len(matched)==1:
            key,expected_unit=matched[0];mode='single'
    if not key:return None
    if mode=='year_change' and (years[0]!=max(years) or years[1]!=min(years)):
        return None
    candidates=[]
    for fact in facts:
        if not(fact.source_verified and fact.source_kind=='table' and fact.metric==key
               and fact.unit==expected_unit and fact.reporting_entity):continue
        if mode=='single':
            if fact.period!=years[0] or fact.role!='actual':continue
            related=[]
        elif mode=='plan_attainment':
            if fact.period!=years[0] or fact.role!='actual_to_plan_ratio':continue
            related=[other for other in facts if other.source_verified and other.metric==key
                     and _same_physical_row(other,fact) and other.period==years[0]
                     and other.role in {'actual','plan'} and other.scope==fact.scope]
            if {other.role for other in related}!={'actual','plan'}:continue
        else:
            if fact.role!='actual' or fact.period or not re.search(r'(?:tang.?giam|\+/-|thay doi)',fact.column_label+' '+normalize_text(fact.column_label),re.I):
                continue
            related=[other for other in facts if other.source_verified and other.metric==key
                     and _same_physical_row(other,fact) and other.period in years
                     and other.role=='actual' and other.scope==fact.scope]
            if {other.period for other in related}!=set(years):continue
            current=next(other for other in related if other.period==max(years))
            base=next(other for other in related if other.period==min(years))
            if (not current.unit_resolved or not base.unit_resolved
                    or current.kind!=base.kind or current.currency!=base.currency
                    or not base.canonical_value):continue
            recomputed=(current.canonical_value/base.canonical_value-Decimal(1))*Decimal(100)
            tolerance=max(fact.printed_quantum/Decimal(2),Decimal('0.005'))
            if abs(recomputed-fact.value)>tolerance:continue
        candidates.append((fact,related))
    if fact_options_only:return candidates
    if not candidates:return None
    values={(fact.value,fact.scope) for fact,_ in candidates}
    if len(values)!=1:return None
    fact,related=min(candidates,key=lambda pair:(pair[0].page,pair[0].line_number))
    source=f'{fact.source_name.rsplit(".",1)[0]}, trang {fact.page}'
    warning=''
    if mode=='single':
        label={'return_on_equity':'ROE','return_on_assets':'ROA',
               'debt_to_assets_ratio':'Hệ số Nợ/Tổng tài sản',
               'debt_to_equity_ratio':'Hệ số Nợ/Vốn chủ sở hữu',
               'total_asset_turnover':'Vòng quay tổng tài sản'}[key]
        suffix='%' if expected_unit=='%' else ' lần' if expected_unit=='ratio' else ''
        rendered=f'{label} năm {years[0]} là {_shown(fact.value)}{suffix}. Nguồn: {source}.'
        related=[]
    elif mode=='year_change':
        later=max(years);earlier=min(years)
        direction='giảm' if fact.value<0 else 'tăng' if fact.value>0 else 'không đổi'
        label='Doanh thu thuần' if key=='net_revenue' else 'Các khoản phải thu ngắn hạn'
        rendered=f'{label} năm {later} {direction} {_shown(abs(fact.value))}% so với năm {earlier}, theo tỷ lệ in trong bảng. Nguồn: {source}.'
    else:
        label='Doanh thu thuần' if key=='net_revenue' else 'Lợi nhuận sau thuế' if key=='profit_after_tax' else fact.metric_label
        rendered=f'{label} thực hiện năm {years[0]} đạt {fact.raw_value} kế hoạch, theo tỷ lệ in trong bảng. Nguồn: {source}.'
        actual=next(other for other in related if other.role=='actual')
        plan=next(other for other in related if other.role=='plan')
        if (actual.unit_resolved and plan.unit_resolved and plan.canonical_value
                and actual.currency==plan.currency and actual.kind==plan.kind=='money'):
            recomputed=(actual.canonical_value/plan.canonical_value*Decimal(100)).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)
            printed=fact.value.quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)
            if recomputed!=printed:
                warning=(f'Tính lại từ hai số cùng hàng cho {_shown(recomputed)}%; bảng in {fact.raw_value}. '
                         'Nguồn không giải thích quy tắc làm tròn; giữ tỷ lệ công bố và nêu rõ chênh lệch.')
    return _response(question,fact,related,rendered,warning=warning)
