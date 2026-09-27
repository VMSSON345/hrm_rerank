"""Staged post-HRM runtime: source contracts and exact computation, no old gates."""
from __future__ import annotations
from dataclasses import replace
from decimal import Decimal,ROUND_HALF_UP
from pathlib import Path
from itertools import combinations
import re,time
from collections import OrderedDict, Counter
from .post_hrm_contract import ValidationReport,same_amount,same_row_unresolved_percent,accounting_checks,cross_statement_checks,compute_records,compute_named
from .post_hrm_source import SourceSnapshot,build_source_facts,metric_key,period_basis
from .document_sources import build_chunks,rank_chunks,expand_neighbor_chunks,OCRConfig,discover_documents
from .program_data import merge_candidates,OPERATIONS
from .question_planner import compile_question_plan,resolve_public_question
from .question_request_planner import compile_question_request
from .metric_semantics import isolated_metric_labels,set_source_metric_labels,display_label
from .ontology import normalize_text
from .revenue_question_context import RevenueResolution,_PROFIT_LABELS,_REVENUE_LABELS

from contextvars import ContextVar
from contextlib import contextmanager
from functools import wraps

_STAGE_TIMES=ContextVar("post_hrm_stage_times",default=None)
_TIMER_STACK=ContextVar("post_hrm_timer_stack",default=())
_PROGRAM_GATE_TRACE=ContextVar("post_hrm_program_gate_trace",default=None)

@contextmanager
def _measure(stage):
    parent=_TIMER_STACK.get();token=_TIMER_STACK.set((*parent,stage));started=time.perf_counter()
    try:yield
    finally:
        elapsed=time.perf_counter()-started;_TIMER_STACK.reset(token)
        record=_STAGE_TIMES.get()
        if record is not None:
            record[stage]+=elapsed
            if parent:record[parent[-1]]-=elapsed

def _timed_answer(function):
    @wraps(function)
    def run(*args,**kwargs):
        timings={key:0.0 for key in ('understanding','source_catalog','retrieval','hrm','source_binding')}
        token=_STAGE_TIMES.set(timings);started=time.perf_counter()
        gate={"stage":"question_planning"} if getattr(args[0],"numeric_program_mode","off") != "off" else None
        gate_token=_PROGRAM_GATE_TRACE.set(gate)
        try:
            result=function(*args,**kwargs)
            total=time.perf_counter()-started
            if gate is not None:result.setdefault("program_gate_trace",gate)
            result['timings_seconds']={**timings,'validation_and_rendering':max(0.0,total-sum(timings.values())),'total':total}
            return result
        finally:
            _PROGRAM_GATE_TRACE.reset(gate_token)
            _STAGE_TIMES.reset(token)
    return run

def _rank_with_timing(*args,**kwargs):
    with _measure('hrm'):return raw_hrm_ranking(*args,**kwargs)

def _bind_with_timing(*args,**kwargs):
    with _measure('source_binding'):return bind_catalog_candidates(*args,**kwargs)

ENGINE_ID='post_hrm_contract_v1'

def _number(value,rounded=False):
    value=Decimal(str(value))
    if rounded:value=value.quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
    text=format(value,',f')
    if '.' in text:text=text.rstrip('0').rstrip('.')
    return text.replace(',','@').replace('.',',').replace('@','.')

def _source_display_name(name):
    return re.sub(r'^[0-9a-fA-F]{10}-', '', Path(name).stem)

def _logical_key(f):
    endpoint=f.period_role if f.period_role in {'opening','closing'} else ''
    point=f.as_of_date or (f.period_start if endpoint=='opening' else f.period_end if endpoint=='closing' else '')
    dates=('point',point) if point and not (f.period_start and f.period_end and f.period_start!=f.period_end) else ('bounds',f.period_start,f.period_end)
    relative=f.comparison_role if not f.period and not endpoint and not any((f.as_of_date,f.period_start,f.period_end)) else ''
    return f.metric,f.period,f.period_basis,endpoint,dates,relative,f.role,f.entity,f.scope,f.currency

def _fact_time_text(f):
    endpoint={'opening':' đầu kỳ','closing':' cuối kỳ'}.get(f.period_role,'')
    point=f.as_of_date or (f.period_start if f.period_role=='opening' else f.period_end if f.period_role=='closing' else '')
    if point:return endpoint+' ngày '+ '/'.join(reversed(point.split('-')))
    return endpoint+(f' năm {f.period}' if f.period else '')


def raw_hrm_ranking(bundle,question,candidates):
    if hasattr(bundle,'rank_raw'):
        return bundle.rank_raw(question,candidates)
    config=bundle.model.config
    final,segments=bundle._predict_single_output(question,candidates,config.max_segments,max(1,config.minimum_halt_segments))
    pointers=final.pointer_logits[0].detach().cpu().tolist()
    selected=final.select_pointer_logits[0].detach().cpu().tolist() if final.select_pointer_logits is not None else pointers[0]
    pairs={operation:final.pair_logits[0,i].detach().cpu().tolist() for i,operation in enumerate(OPERATIONS)} if final.pair_logits is not None else {}
    return {'pointers':pointers,'select':selected,'pairs':pairs,'segments':segments,
        'neural_operation':OPERATIONS[int(final.operation_logits[0].argmax())]}

def bind_catalog_candidates(candidates,source_facts):
    """Map neural pointers to once-normalized physical source facts.

    Ranking may reorder or truncate candidates; it cannot change the source
    value, unit or period. Partial/tampered spans have no catalog entry.
    """
    catalog={}
    for f in source_facts:
        key=(f.source_name,f.line_number,f.source_span,f.source_line_sha256)
        catalog.setdefault(key,[]).append(f)
    bound=[];missing=[]
    for c in candidates:
        key=(c.source_name,c.line_number,tuple(c.source_span or ()),c.source_line_sha256)
        entries=[f for f in catalog.get(key,[]) if f.metric==metric_key(c.row_label) or f.period_binding in {'numbered_balance_continuation','source_plan_sentence','same_sentence_comparison','layout_scalar_date','layout_scalar','source_narrative_local','profile_report_title','layout_date_columns','aligned_compound_ratio_column'}]
        if not entries:missing.append({'candidate':c.index,'reason':'source_catalog_binding_missing'})
        bound.extend(f.model_copy(update={'candidate_index':c.index}) for f in entries)
    return bound,missing

class ContractDocumentEngine:
    def __init__(self,bundle,publication_policy='post_hrm_contract_v1',*,retrieval_mode='adaptive_graph',numeric_program_mode='off'):
        self.bundle=bundle;self.publication_policy=publication_policy;self._catalog_cache=OrderedDict()
        if retrieval_mode not in {'legacy','hybrid','hybrid_graph','adaptive_graph'}:raise ValueError('Unknown retrieval mode')
        from .hybrid_retrieval import HybridEvidenceRetriever
        self.retrieval_mode=retrieval_mode
        if numeric_program_mode not in {'off','hrm','baseline'}:raise ValueError('Unknown numeric program mode')
        self.numeric_program_mode=numeric_program_mode
        self.retriever=HybridEvidenceRetriever(getattr(bundle,'semantic_encoder',None))

    def _matches(self,f,slot,plan,*,check_metric=True):
        parent_profit=(plan.operation=='SELECT' and plan.entity_text=='parent_company'
                       and metric_key(slot.metric)=='profit_after_tax')
        required_metric='profit_after_tax_parent' if parent_profit else metric_key(slot.metric)
        if check_metric and f.metric!=required_metric:return False
        if plan.operation=='SELECT' and plan.requested_unit_family=='percent' and f.unit!='%':
            return False
        if slot.unit_family.lower() in {'currency','money'} and f.kind!='money':
            if not(plan.operation=='SELECT' and plan.requested_value_column):return False
        if slot.year and f.period!=slot.year:
            fiscal_query='nam tai chinh' in normalize_text(plan.question)
            witnessed=bool(f.fiscal_evidence) and all(
                self.snapshot.witness_valid(w,f.source_name)
                for w in f.fiscal_evidence)
            fiscal_label=('nam tai chinh '+slot.year in normalize_text(plan.question)
                          and f.fiscal_year==slot.year)
            fiscal_end=('nam tai chinh ket thuc ngay' in normalize_text(plan.question)
                        and f.fiscal_end_date.startswith(slot.year+'-'))
            if not(fiscal_query and witnessed and (fiscal_label or fiscal_end)):
                return False
        requested_basis=period_basis(plan.question)
        if (requested_basis or f.period_basis) and requested_basis!=f.period_basis:return False
        requested_date=re.search(r'(?<!\d)(\d{1,2})[/.](\d{1,2})[/.]((?:19|20)\d{2})(?!\d)',plan.question)
        if requested_date and slot.id!='eps':
            from datetime import date
            try:target=date(int(requested_date[3]),int(requested_date[2]),int(requested_date[1])).isoformat()
            except ValueError:return False
            fiscal_end=f.fiscal_end_date if 'nam tai chinh ket thuc ngay' in normalize_text(plan.question) else ''
            if target not in {f.as_of_date,f.period_start,f.period_end,fiscal_end}:return False
        if slot.column_role and f.role!=slot.column_role.lower() and not(plan.operation=='SELECT' and plan.requested_value_column):
            if not(f.is_profile and slot.column_role.upper()=='ACTUAL' and (not slot.year or (f.period_binding=='profile_report_title' and f.period==slot.year)) and plan.operation=='SELECT' and not plan.requested_period_roles):return False
        if slot.period_role and (f.comparison_role if slot.period_role.lower() in {'current','prior'} else f.period_role)!=slot.period_role.lower():return False
        if plan.requested_accounting_scope and f.scope!=plan.requested_accounting_scope:return False
        if plan.entity_mode=='reporting' and not f.reporting_entity:return False
        if plan.entity_mode=='named' and not parent_profit:
            hay=normalize_text(f.entity+' '+f.source_name)
            if not all(x in hay.split() for x in plan.entity_terms):return False
        if plan.requested_value_column and not set(normalize_text(plan.requested_value_column).split())<=set(normalize_text(f.column_label).split()):return False
        if plan.requested_section and plan.requested_section not in f.section:return False
        if plan.requested_source_kind=='profile' and not f.is_profile:return False
        if plan.requested_source_kind in {'table','narrative'} and f.source_kind!=plan.requested_source_kind:return False
        if plan.requested_source_heading and plan.requested_source_heading not in f.section:return False
        if plan.requested_report_year and f.report_year!=plan.requested_report_year:return False
        if plan.requested_source_years and not set(plan.requested_source_years)<=set(f.source_years):return False
        if plan.requested_source_year_count and len(f.source_years)!=plan.requested_source_year_count:return False
        return True

    def _metric_resolution(self,question,previous,document,candidates,facts):
        plan=compile_question_plan(resolve_public_question(question));key=metric_key(plan.metric_text)
        labels=_PROFIT_LABELS if key=='literal:loi nhuan' else _REVENUE_LABELS if key=='revenue_unspecified' else None
        if labels is not None:labels={metric_key(label):label for label in labels.values()}
        if labels is None:return RevenueResolution(question,question)
        bare='lợi nhuận' if key=='literal:loi nhuan' else 'doanh thu'
        pattern=r'\b(?:lợi\s+nhuận|loi\s+nhuan)\b' if bare=='lợi nhuận' else r'\b(?:doanh\s+thu|revenue|sales)\b'
        digest=self.snapshot.hashes.get(document.name,'')
        if previous:
            prior=compile_question_plan(resolve_public_question(previous));prior_key=metric_key(prior.metric_text)
            if prior_key in labels and prior_key!=key:
                label=labels[prior_key];return RevenueResolution(question,re.sub(pattern,label,question,flags=re.I),label,'conversation_metric',digest,previous,bare_metric=bare)
        # Disclosure discovery is independent of successful numeric binding.
        # Incomplete format/role/period metadata must not erase another meaning.
        from .evidence_schema import evidence_cells
        options=set();unmapped=set()
        years=set(plan.requested_years)
        requested_roles={slot.column_role.lower() for slot in plan.operand_slots if slot.column_role}
        for candidate,cell in zip(candidates,evidence_cells(candidates)):
            choice=metric_key(cell.metric_label)
            if cell.embedded_entity or cell.is_percent or cell.is_year_like:continue
            if choice==key and not plan.requested_source_kind:continue
            if years and cell.period and re.fullmatch(r'(?:19|20)\d{2}',cell.period) and cell.period not in years:continue
            if requested_roles and cell.column_role and cell.column_role not in requested_roles:continue
            scope=(cell.table_semantics or {}).get('accounting_scope','')
            if plan.requested_accounting_scope and scope and scope!=plan.requested_accounting_scope:continue
            if plan.requested_section and plan.requested_section not in ' '.join(cell.source_section_path):continue
            kind='profile' if cell.report_field else 'narrative' if cell.source_assertion else 'table'
            if plan.requested_source_kind and kind!=plan.requested_source_kind:continue
            section=' '.join(cell.source_section_path)
            if plan.requested_source_heading and plan.requested_source_heading not in section:continue
            if plan.requested_report_year:
                report_year=str((cell.report_field or {}).get('report_year','') or (candidate.report_context or {}).get('report_year',''))
                if report_year!=plan.requested_report_year:continue
            source_years=set(re.findall(r'(?:19|20)\d{2}',' '.join(w.get('text','') for w in (cell.table_semantics or {}).get('header_evidence',[]))))
            if plan.requested_source_years and not set(plan.requested_source_years)<=source_years:continue
            if plan.requested_source_year_count and len(source_years)!=plan.requested_source_year_count:continue
            if choice in labels:options.add(choice)
            elif bare=='lợi nhuận' and re.search(r'\b(?:profit|earnings|loi nhuan|net income)\b',normalize_text(cell.metric_label)):
                # Preserve unfamiliar source meanings; do not silently treat them
                # as synonyms or erase them because only another label is known.
                unmapped.add(cell.metric_label)
        if len(options)==1 and unmapped:
            return RevenueResolution(question,question,basis='unmapped_source_metrics',document_sha256=digest,alternatives=tuple([labels[next(iter(options))],*sorted(unmapped)]),bare_metric=bare)
        if bare=='doanh thu'  and len(options)>1 and 'net_revenue' in options:
            years=set(plan.requested_years)
            signatures={}
            for choice in options:
                signatures[choice]={(f.period,f.role,f.entity,f.scope,f.canonical_value) for f in facts
                    if f.metric==choice and f.kind=='money' and f.role=='actual' and f.reporting_entity and (not years or f.period in years)}
            net=signatures['net_revenue']
            if net and (not years or {x[0] for x in net}==years) and all(signatures[k]==net for k in options):
                label=labels['net_revenue']
                return RevenueResolution(question,re.sub(pattern,label,question,flags=re.I),label,'document_net_revenue_interpretation',digest,bare_metric=bare)
        if options=={key}:return RevenueResolution(question,question)
        if len(options)==1:
            label=labels[next(iter(options))];return RevenueResolution(question,re.sub(pattern,label,question,flags=re.I),label,'document_metric_periods',digest,bare_metric=bare)
        if len(options)>1:return RevenueResolution(question,question,basis='ambiguous_source_metrics',document_sha256=digest,alternatives=tuple(labels[k] for k in sorted(options)),bare_metric=bare)
        return RevenueResolution(question,question)

    def _empty(self,question,message,report=None,citations=()):
        report=report or ValidationReport()
        from .conversation_context import public_intent_message
        message=public_intent_message(message)
        return {'answer':None,'canonical_value':None,'unit':None,'status':'ABSTAIN','final_action':'ABSTAIN',
            'rendered':message,'citations':list(citations),'validation_engine':ENGINE_ID,'validation':report.model_dump(mode='json'),
            'proof':{'verification_engine':ENGINE_ID,'source_action':'ABSTAIN','execution_certificate':None},'question':question}

    def _raw_query_allowed(self,plan):
        if plan.operation!='SELECT' or len(plan.operand_slots)!=1 or plan.requested_scale:
            return False
        query=normalize_text(plan.question)
        return not re.search(r'\b(?:dong|vnd|vnd|usd|eur|nghin|trieu|ty|so tien|quy doi|chinh xac)\b',query)

    def _name_collision_peers(self,fact,all_facts):
        """Show related disclosures; equal printed values never transfer units.

        Metric identity and unit provenance are independent. The source peer
        must have an explicit/inherited unit, the same temporal/entity/scope
        identity, and a bounded RELATED label relation. Unrelated equal values
        are not presented as competing declarations of the requested metric.
        """
        if fact.kind!='money' or fact.unit_resolved:
            return []
        from .metric_semantics import relation
        identity=_logical_key(fact)[1:]
        peers=[other for other in all_facts if other.source_verified
               and other.kind=='money' and other.unit_resolved
               and other.source_name==fact.source_name
               and other.document_sha256==fact.document_sha256
               and other.metric!=fact.metric and other.value==fact.value
               and _logical_key(other)[1:]==identity
               and relation(fact.metric_label,other.metric_label)=='RELATED']
        peers.sort(key=lambda other:(abs((other.page or 0)-(fact.page or 0)),
                                     other.quantum,other.line_number,other.fact_id))
        result=[];seen=set()
        for peer in peers:
            if peer.metric not in seen:
                result.append(peer);seen.add(peer.metric)
            if len(result)==2:break
        return result

    def _unit_response(self,question,plan,report,fact,all_facts,companions):
        if not plan.requested_accounting_scope:
            variants=[f for f in all_facts if f.source_verified and f.kind=='money'
                      and f.source_name==fact.source_name and f.metric==fact.metric
                      and f.period==fact.period and f.period_role==fact.period_role
                      and f.role==fact.role and f.entity==fact.entity]
            if len({f.scope for f in variants})>1:
                variants=list({(f.scope,f.raw_value):f for f in variants}.values())
                report.add('data_contract','source_scope_ambiguous','INSUFFICIENT_EVIDENCE',
                           'Cùng chỉ tiêu/kỳ có công bố ở nhiều phạm vi.',blocking=True,facts=tuple(variants))
                details=['Nguồn có nhiều phạm vi cho '+display_label(fact.metric_label)+
                         _fact_time_text(fact)+'; cần chọn phạm vi trước khi đọc một số gốc.']
                for f in variants:
                    scope={'consolidated':'hợp nhất','separate':'riêng lẻ','standalone':'riêng lẻ'}.get(f.scope,'chưa rõ phạm vi')
                    details.append(f'- {scope}: `{f.raw_value}` '+(f.unit or '(chưa rõ đơn vị)')+
                                   f'; nguồn: {_source_display_name(f.source_name)}, trang {f.page}.')
                result=self._empty(question,'\n'.join(details),report,[f.citation() for f in variants])
                result.update(answer_type='explained_halt',metric_link_state='SCOPE_AMBIGUOUS',
                              unit_state=fact.unit_state,raw_disclosures=[
                                  {'metric':f.metric,'metric_label':f.metric_label,'raw_value':f.raw_value,
                                   'unit':f.unit if f.unit_resolved else None,'unit_state':f.unit_state,
                                   'scope':f.scope,'citations':[f.citation()]} for f in variants])
                return result
        peers=self._name_collision_peers(fact,all_facts)
        metric_link='NAME_COLLISION_AMBIGUOUS' if peers else 'EXACT_METRIC'
        citations=[fact.citation(),*[p.citation() for p in peers]]
        label=display_label(fact.metric_label)
        source=f'{_source_display_name(fact.source_name)}, trang {fact.page}'
        title=f'{label}{_fact_time_text(fact)}'
        detail=f'{title} ghi số gốc `{fact.raw_value}` trong bảng ({source}), nhưng bảng chưa chứng minh đơn vị tiền/thang đo.'
        extra=[]
        disclosures=[{'metric':fact.metric,'metric_label':fact.metric_label,'raw_value':fact.raw_value,
                      'unit':None,'unit_state':fact.unit_state,'citations':[fact.citation()]}]
        if plan.operation=='SUBTRACT' and len(plan.operand_slots)==2:
            others=[f for f in all_facts if f.source_verified and not f.unit_resolved
                    and f.fact_id!=fact.fact_id and f.source_name==fact.source_name
                    and f.table_key==fact.table_key and f.line_number==fact.line_number
                    and f.metric==fact.metric and f.entity==fact.entity and f.scope==fact.scope
                    and f.period==fact.period and f.period_role!=fact.period_role]
            for other in list({f.fact_id:f for f in others}.values())[:2]:
                extra.append(f'Cùng hàng, {display_label(other.metric_label)}{_fact_time_text(other)} '
                             f'ghi số gốc `{other.raw_value}` (chưa rõ đơn vị).')
                citations.append(other.citation())
                disclosures.append({'metric':other.metric,'metric_label':other.metric_label,
                                    'raw_value':other.raw_value,'unit':None,
                                    'unit_state':other.unit_state,'citations':[other.citation()]})
        for peer in peers:
            peer_unit=' '+peer.unit if peer.unit_resolved else ' (chưa rõ đơn vị)'
            extra.append(f'Công bố riêng: {display_label(peer.metric_label)}{_fact_time_text(peer)} ghi `{peer.raw_value}`{peer_unit} '
                         f'({_source_display_name(peer.source_name)}, trang {peer.page}).')
            disclosures.append({'metric':peer.metric,'metric_label':peer.metric_label,'raw_value':peer.raw_value,
                                'unit':peer.unit if peer.unit_resolved else None,'unit_state':peer.unit_state,
                                'citations':[peer.citation()]})
        if peers:
            extra.append('Hai dòng có trị số được đọc giống nhau nhưng khác tên chỉ tiêu; chưa có chứng cứ để đồng nhất chỉ tiêu hoặc chuyển đơn vị giữa chúng.')
        if plan.operation=='SELECT' and re.search(r'\b(?:dong|vnd|vnd|usd|eur|nghin|trieu|ty)\b',normalize_text(plan.question)):
            summaries=[f for f in all_facts if f.source_verified and f.source_name==fact.source_name
                       and f.metric==fact.metric and f.page==fact.page and f.unit_resolved
                       and f.period!=fact.period and not f.period and f.fact_id!=fact.fact_id]
            if summaries:
                summary=min(summaries,key=lambda f:abs(f.line_number-fact.line_number))
                extra.append(f'Cùng trang còn công bố làm tròn `{summary.raw_value} {summary.unit}`; '
                             'chưa chứng minh số trong bảng có cùng thang đo và ngày chốt.')
                citations.append(summary.citation())
                disclosures.append({'metric':summary.metric,'metric_label':summary.metric_label,
                                    'raw_value':summary.raw_value,'unit':summary.unit,
                                    'unit_state':summary.unit_state,'citations':[summary.citation()]})
        report.add('data_contract','source_unit_scale_missing','INSUFFICIENT_EVIDENCE',
                   'Chưa có chứng cứ đơn vị tiền cho chỉ tiêu được hỏi.',blocking=True,facts=tuple(companions))
        if self._raw_query_allowed(plan) and fact.source_kind=='table' and fact.unit_state=='UNIT_UNRESOLVED_RAW':
            rendered=('Số hiển thị nguyên văn cho '+title+f': `{fact.raw_value}`. '
                      'Nguồn chưa ghi đơn vị tiền/thang đo cho ô này; đây không phải giá trị đồng đã xác nhận. '
                      f'Nguồn: {source}.')
            if extra:rendered+='\n'+'\n'.join(extra)
            return {'answer':None,'canonical_value':None,'unit':None,'raw_value':fact.raw_value,
                    'status':'RAW_DISCLOSURE','final_action':'DISCLOSE_RAW','answer_type':'raw_disclosure',
                    'numeric_answer':False,'unit_state':fact.unit_state,'metric_link_state':metric_link,
                    'raw_disclosures':disclosures,'rendered':rendered,'citations':citations,
                    'validation_engine':ENGINE_ID,'validation':report.model_dump(mode='json'),
                    'proof':{'verification_engine':ENGINE_ID,'source_action':'DISCLOSE_RAW',
                             'source_certificate':{'citations':citations},'execution_certificate':None},
                    'question':question}
        message='Không thể xác định số tiền bằng đơn vị được hỏi: '+detail
        if extra:message+='\n'+'\n'.join(extra)
        result=self._empty(question,message,report,citations)
        result.update(answer_type='explained_halt',unit_state=fact.unit_state,
                      metric_link_state=metric_link,raw_disclosures=disclosures)
        return result

    def _conflict(self,question,report,facts):
        facts=list({f.fact_id:f for f in facts}.values())
        groups={}
        for f in facts:
            key=(f.metric,f.period,f.role,f.unit,f.value,f.scale)
            groups.setdefault(key,[]).append(f)
        lines=['Các công bố trong nguồn chưa thống nhất; chưa thể chọn một số duy nhất để trả lời hoặc tính toán.']
        disclosures=[]
        for fs in groups.values():
            f=fs[0];pages=sorted({x.page for x in fs if x.page is not None})
            unit='đồng' if f.unit=='VND' else f.unit
            role={'actual':'thực hiện','plan':'kế hoạch','forecast':'dự báo'}.get(f.role,f.role)
            if f.canonical_value is None:
                shown=f.raw_value+' (chưa xác định đơn vị/thang đo)'
            else:
                value=f.canonical_value if f.currency=='VND' else f.value
                unit='đồng' if f.currency=='VND' else unit
                shown=_number(value)+' '+unit
            lines.append(f'- {display_label(f.metric_label)} {role}{_fact_time_text(f)}: {shown}; nguồn: {_source_display_name(f.source_name)}, trang '+', '.join(map(str,pages))+'.')
            disclosures.append({'metric':f.metric,'period':f.period,'role':f.role,'raw_value':f.raw_value,'unit':f.unit,'scale':str(f.scale) if f.unit_resolved else None,'canonical_value':str(f.canonical_value) if f.canonical_value is not None else None,'citations':[x.citation() for x in fs]})
        if any(not f.unit_resolved for f in facts):lines.append('Công bố chưa rõ đơn vị được giữ nguyên số gốc; không quy đổi hoặc dùng để chứng minh mâu thuẫn.')
        result=self._empty(question,'\n'.join(lines),report,[f.citation() for f in {f.fact_id:f for f in facts}.values()])
        result['disclosure_conflict']={'status':'UNRESOLVED','disclosures':disclosures,'computation_performed':False}
        return result,{}

    def _scalar(self,question,all_chunks,all_candidates,all_facts,*,known_facts=None,catalog_errors=()):
        from .post_hrm_formulas import financial_calculation_plan
        with _measure('understanding'):
            special=financial_calculation_plan(question)
            plan=special[0] if special else compile_question_plan(resolve_public_question(question));report=ValidationReport()
            if not plan.operand_slots and plan.operation=='SELECT' and not plan.abstention_reason and len(plan.requested_years)<=1 and (any(f.metric==metric_key(plan.metric_text) for f in all_facts)
                or metric_key(plan.metric_text) in {'average_collection_period','operating_cash_flow','current_ratio',
                    'quick_ratio','return_on_assets','return_on_equity','gross_margin','net_margin'}):
                from .question_planner import OperandSlot,ProgramIntentNode
                slot=OperandSlot('lookup',plan.metric_text,plan.requested_years[0] if plan.requested_years else '',plan.column_roles[0] if plan.column_roles else 'ACTUAL',plan.entity_mode,plan.requested_unit_family or 'currency_or_source',('explicit_source_metric','actual_default'))
                plan=replace(plan,operand_slots=(slot,),program_dag=(ProgramIntentNode('answer','SELECT',('lookup',),slot.unit_family,('single_fact_lookup',)),))
            if plan.abstention_reason or not plan.operation or not plan.operand_slots:
                return self._empty(question,plan.abstention_reason or 'Chưa xác định được phép tính hoặc chỉ tiêu cần trả lời.',report),{}
            # A catalog-wide absence of this published efficiency measure is
            # conclusive for a direct lookup. Avoid expensive neural ranking
            # over unrelated cells; never synthesize missing credit sales.
            if (plan.operation=='SELECT' and len(plan.operand_slots)==1
                    and metric_key(plan.operand_slots[0].metric) in {'average_collection_period','operating_cash_flow'}
                    and not any(f.metric==metric_key(plan.operand_slots[0].metric) for f in all_facts)):
                key=metric_key(plan.operand_slots[0].metric)
                message=('Chưa xác nhận được Kỳ thu tiền bình quân trong nguồn; cần công nợ bình quân và doanh thu bán chịu cùng kỳ để tính.'
                         if key=='average_collection_period' else
                         'Chưa xác nhận được dòng tiền thuần từ hoạt động kinh doanh cùng kỳ trong Markdown đã chọn.')
                report.add('data_contract','required_fact_missing','INSUFFICIENT_EVIDENCE',message,blocking=True)
                return self._empty(question,message,report),{}
        with _measure('retrieval'):
            limit=self.bundle.model.config.max_candidates
            retrieval_trace={'mode':'reused_verified_facts'}
            if known_facts is not None:
                candidates=[]
            elif self.retrieval_mode in {'legacy','adaptive_graph'}:
                ranked=rank_chunks(question,all_chunks,top_k=8);ranked=expand_neighbor_chunks(ranked,all_chunks,neighbor_window=1,max_chunks=12)
                candidates=merge_candidates(resolve_public_question(question),ranked,max_candidates=limit)
                retrieval_trace={'mode':'legacy','selected_candidates':len(candidates),'budget':limit}
                if self.retrieval_mode=='adaptive_graph':
                    initial_facts,_=_bind_with_timing(candidates,all_facts)
                    complete=all(any(self._matches(f,slot,plan) for f in initial_facts) for slot in plan.operand_slots)
                    if complete:
                        retrieval_trace=self.retriever.trace_existing(candidates,all_candidates,all_facts,plan.operand_slots,lambda f,slot:self._matches(f,slot,plan),limit)
                    else:
                        candidates,retrieval_trace=self.retriever.retrieve(resolve_public_question(question),all_chunks,all_candidates,all_facts,plan.operand_slots,lambda f,slot:self._matches(f,slot,plan),limit,use_graph=True)
                        retrieval_trace.update(mode='adaptive_graph',strategy='expand_missing_operand_pool')
            else:
                candidates,retrieval_trace=self.retriever.retrieve(resolve_public_question(question),all_chunks,all_candidates,all_facts,plan.operand_slots,lambda f,slot:self._matches(f,slot,plan),limit,use_graph=self.retrieval_mode=='hybrid_graph')
        if known_facts is None:
            ranking=_rank_with_timing(self.bundle,resolve_public_question(question),candidates) if candidates else {'pointers':[],'select':[],'pairs':{},'segments':0}
            facts,source_errors=_bind_with_timing(candidates,all_facts)
        else:
            ranking={'pointers':[],'select':[],'pairs':{},'segments':0};facts=list(known_facts);source_errors=[]
        pools=[];pool_rankings=[]
        for slot in plan.operand_slots:
            slot_ranking=ranking
            pool=[f for f in facts if self._matches(f,slot,plan)]
            if not slot.year and plan.operation=='SELECT' and not plan.requested_event:
                profiles=[f for f in pool if f.is_profile]
                if profiles:pool=profiles
            if not pool and known_facts is None:
                available=[f for f in all_facts if self._matches(f,slot,plan)]
                indices={f.candidate_index for f in available}
                recovery=[c for c in all_candidates if c.index in indices][:limit]
                if recovery:
                    recovery=[replace(c,index=i) for i,c in enumerate(recovery)]
                    reranking=_rank_with_timing(self.bundle,resolve_public_question(question),recovery)
                    rebound,_=_bind_with_timing(recovery,all_facts)
                    pool=[f for f in rebound if self._matches(f,slot,plan)]
                    # Original scores are incomparable to the new pool. Store the
                    # actual re-ranking on these facts for this slot only.
                    if pool:
                        slot_ranking=reranking
                        report.add('data_contract','bounded_source_recovery','PASS','Đã xếp hạng lại các ứng viên nguồn đúng điều kiện bị thiếu ở lượt truy hồi đầu.',facts=tuple(pool))
                        pool=sorted(pool,key=lambda f:reranking['select'][f.candidate_index] if plan.operation=='SELECT' else reranking['pointers'][min(len(pools),len(reranking['pointers'])-1)][f.candidate_index],reverse=True)
            if not pool:
                if plan.operation=='SELECT' and plan.requested_unit_family=='percent':
                    wrong_unit=[f for f in all_facts if f.metric==metric_key(slot.metric)
                                and (not slot.year or f.period==slot.year)
                                and f.unit!='%']
                    if wrong_unit:
                        message=('Nguồn có '+display_label(slot.metric)+
                                 ' nhưng chưa chứng minh ô số có đơn vị phần trăm; không đổi tỷ số sang %.')
                        report.add('data_contract','requested_percent_unit_unproven',
                                   'INSUFFICIENT_EVIDENCE',message,blocking=True,
                                   facts=tuple(wrong_unit[:3]))
                        return self._empty(question,message,report,
                                           [f.citation() for f in wrong_unit[:3]]),{}
                date_match=re.search(r'(?<!\d)\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}(?!\d)',plan.question)
                same_metric=[f for f in all_facts if f.metric==metric_key(slot.metric)]
                undated=[f for f in same_metric if f.unit_resolved and not f.as_of_date
                         and not f.period_start and not f.period_end and not f.period]
                if (date_match and undated and not any(f.as_of_date or f.period_start or f.period_end
                                                         for f in same_metric)):
                    message=('Đã thấy công bố '+display_label(slot.metric)+
                             ' nhưng đoạn nguồn chưa gắn ngày chốt '+date_match[0]+
                             '; chưa thể dùng số này cho thời điểm được hỏi.')
                    report.add('data_contract','source_date_unbound','INSUFFICIENT_EVIDENCE',
                               message,blocking=True,facts=tuple(undated[:3]))
                    return self._empty(question,message,report,[f.citation() for f in undated[:3]]),{}
                if slot.period_role in {'closing','opening'} and undated:
                    endpoint='cuối' if slot.period_role=='closing' else 'đầu'
                    period=f' năm {slot.year}' if slot.year else ''
                    message=('Đã thấy công bố '+display_label(slot.metric)+
                             f' nhưng đoạn nguồn chưa gắn thời điểm {endpoint} kỳ{period};'+
                             ' chưa thể dùng số này cho phép tính.')
                    report.add('data_contract','source_endpoint_unbound','INSUFFICIENT_EVIDENCE',
                               message,blocking=True,facts=tuple(undated[:3]))
                    return self._empty(question,message,report,[f.citation() for f in undated[:3]]),{}
                detail=f' cho năm/kỳ {slot.year}' if slot.year else ''
                related_ids={c.index for c in all_candidates if metric_key(c.row_label)==metric_key(slot.metric) and (not slot.year or c.period==slot.year)}
                related_errors=[e for e in catalog_errors if e.get('candidate') in related_ids]
                uncertain_number=any(e.get('reason','').startswith('source_number') for e in related_errors)
                message=('Đã tìm thấy số liệu '+display_label(slot.metric)+detail+', nhưng chưa xác nhận được cách đọc dấu phân cách số trong nguồn; hệ thống chưa công bố giá trị.' if uncertain_number else 'Chưa xác nhận được số liệu '+display_label(slot.metric)+detail+' trong nguồn đã chọn.')
                report.add('data_contract','number_interpretation_unresolved' if uncertain_number else 'required_fact_missing','INSUFFICIENT_EVIDENCE',message,blocking=True)
                result=self._empty(question,message,report);result['source_contract_errors']=[*source_errors,*related_errors];result['retrieval_trace']=retrieval_trace;return result,{}
            pools.append(pool);pool_rankings.append(slot_ranking)
        def slot_score(i,f):
            rank=pool_rankings[i]
            values=rank['select'] if plan.operation=='SELECT' else rank['pointers'][min(i,len(rank['pointers'])-1)] if rank['pointers'] else []
            return values[f.candidate_index] if 0<=f.candidate_index<len(values) else 0
        if len(pools)>1:
            common=set.intersection(*[{(f.entity,f.scope) for f in pool} for pool in pools])
            if not common:
                report.add('data_contract','incompatible_operand_scope','FAIL','Các đầu vào không thuộc cùng doanh nghiệp và phạm vi báo cáo.',blocking=True)
                return self._empty(question,'Chưa có đủ đầu vào cùng doanh nghiệp và phạm vi để tính.',report),{}
            scope=max(sorted(common),key=lambda key:sum(max(slot_score(i,f) for f in pool if (f.entity,f.scope)==key) for i,pool in enumerate(pools)))
            pools=[[f for f in pool if (f.entity,f.scope)==scope] for pool in pools]
        bindings={};selected=[]
        for i,(slot,pool) in enumerate(zip(plan.operand_slots,pools)):
            def score(f):return slot_score(i,f)
            # An annual stock lookup uses the explicitly labelled closing
            # balance when the same pool also has opening balances. Explicit
            # endpoint/date queries keep their own matching constraints.
            if (plan.operation=='SELECT' and slot.year and not slot.period_role
                    and not re.search(r'\d{1,2}[/ .]\d{1,2}[/ .](?:19|20)\d{2}',plan.question)
                    and any(f.period_role=='closing' for f in pool)
                    and any(f.period_role=='opening' for f in pool)):
                pool=[f for f in pool if f.period_role!='opening']
            # Unspecified scope is not permission for rank to choose between
            # incompatible disclosures, including an unknown scope.
            if not plan.requested_accounting_scope:
                if plan.operation=='SELECT' and any(f.unit_resolved for f in pool):
                    pool=[f for f in pool if f.unit_resolved]
                variants=[f for f in all_facts if f.source_verified and self._matches(f,slot,plan)
                          and any(_logical_key(f)[:7]+(f.source_name,)+_logical_key(f)[9:]
                                  ==_logical_key(p)[:7]+(p.source_name,)+_logical_key(p)[9:]
                                  for p in pool)]
                by_time={}
                for f in variants:
                    key=_logical_key(f);by_time.setdefault(key[:7]+(f.source_name,)+key[9:],[]).append(f)
                ambiguous=next((fs for fs in by_time.values() if len({f.scope for f in fs})>1
                    and any(a.unit_resolved and b.unit_resolved
                            and a.canonical_value!=b.canonical_value
                            for a,b in combinations(fs,2))),None)
                if ambiguous:
                    report.add('data_contract','source_scope_ambiguous','INSUFFICIENT_EVIDENCE',
                        'Các công bố khác số có phạm vi khác nhau hoặc chưa rõ.',blocking=True,facts=tuple(ambiguous))
                    message='Nguồn có nhiều số với phạm vi khác nhau hoặc chưa rõ. Bạn muốn dùng báo cáo hợp nhất hay riêng lẻ? Cần xác nhận phạm vi của công bố chưa ghi rõ trước khi chọn số.'
                    details=list(dict.fromkeys(
                        f'- {display_label(f.metric_label)} năm {f.period}: {f.raw_value} '+(f.unit or 'chưa rõ đơn vị')+
                        (f' (quy đổi {_number(f.canonical_value)} đồng)' if f.canonical_value is not None and f.currency=='VND' and f.scale!=1 else '')+
                        '; '+{'consolidated':'hợp nhất','separate':'riêng lẻ','standalone':'riêng lẻ'}.get(f.scope,'chưa rõ phạm vi')+
                        f'; trang {f.page}.' for f in ambiguous))
                    message+='\n'+'\n'.join(details)
                    return self._empty(question,message,report,[f.citation() for f in ambiguous]),{}
            # Resolve time from the full source catalog, never from neural rank
            # or the subset that happened to fit the retrieval budget.
            temporal=[f for f in all_facts if self._matches(f,slot,plan)
                      and any((f.metric,f.entity,f.scope,f.role,f.currency)==(p.metric,p.entity,p.scope,p.role,p.currency) for p in pool)]
            if any(f.period_role=='closing' for f in pool) and not any(f.period_role=='opening' for f in pool):
                temporal=[f for f in temporal if f.period_role!='opening']
            def point(f):
                return f.as_of_date or (f.period_start if f.period_role=='opening' else f.period_end if f.period_role=='closing' else '')
            dated=[f for f in temporal if point(f)]
            # A compatible annual restatement with no day is not evidence of
            # another closing date. Distinct explicit dates remain distinct,
            # even when they report equal amounts.
            temporal_identities=[f for f in temporal if point(f) or not any(
                f.period==d.period and f.period_basis==d.period_basis
                and (not f.period_role or f.period_role==d.period_role)
                and not f.period_start and not f.period_end and same_amount(f,d)
                for d in dated)]
            identities={(_logical_key(f)[1:6]) for f in temporal_identities}
            if len(identities)>1:
                report.add('data_contract','source_period_ambiguous','INSUFFICIENT_EVIDENCE',
                           'Có nhiều kỳ hoặc ngày chốt phù hợp; cần làm rõ thời điểm.',blocking=True,facts=tuple(temporal))
                options=sorted({_fact_time_text(f).strip() or 'chưa rõ kỳ' for f in temporal})
                prompt=('Bạn muốn dùng năm nào?' if all(re.fullmatch(r'năm (?:19|20)\d{2}',o) for o in options)
                        else 'Bạn muốn dùng kỳ hoặc ngày chốt nào?')
                return self._empty(question,'Nguồn có nhiều thời điểm: '+ '; '.join(options)+'. '+prompt,report,[f.citation() for f in temporal]),{}
            groups={}
            for f in pool:groups.setdefault(_logical_key(f),[]).append(f)
            choices=sorted(groups.values(),key=lambda fs:max(score(f) for f in fs),reverse=True)
            group=choices[0]
            reference=min(group,key=lambda f:(bool(f.unit_assumption),f.quantum,-score(f)))
            # Inspect other retrieved declarations of the selected identity;
            # retain genuine conflicts rather than prune them before checking.
            companions=[f for f in all_facts if _logical_key(f)==_logical_key(reference) and self._matches(f,slot,plan)]
            explicit=[f for f in companions if f.unit_resolved]
            if explicit and reference.unit_assumption:
                reference=min(explicit,key=lambda f:(f.quantum,-score(f)))
            unresolved_units=[f for f in companions if f.unit_assumption and explicit]
            if unresolved_units:

                def scale_compatible(f):
                    return any(
                        abs(reference.canonical_value - f.value * factor)
                        <= max(reference.quantum, f.printed_quantum * factor) / 2
                        for factor in (Decimal(1), Decimal(1000),
                                       Decimal(1000000), Decimal(1000000000)))
                divergent = [f for f in unresolved_units if not scale_compatible(f)]
                if divergent:
                    message = ('Các công bố cùng chỉ tiêu/kỳ chưa thống nhất; một công bố '
                               'chưa xác định đơn vị/thang đo. Cần đối chiếu nguồn trước khi chọn.')
                    report.add('data_contract', 'unresolved_competing_disclosure',
                               'INSUFFICIENT_EVIDENCE', message, blocking=True,
                               facts=(reference, *divergent))
                    return self._empty(question, message, report,
                                       [f.citation() for f in (reference, *divergent)]), {}

                report.add('data_contract','optional_disclosure_unit_missing','NOT_APPLICABLE','Công bố phụ thiếu đơn vị; dùng công bố có đơn vị nguồn rõ ràng.',facts=tuple(unresolved_units))
            comparable=explicit
            if not explicit:
                unresolved = (plan.operation == 'PERCENT_CHANGE' and companions
                    and all(not f.unit_resolved and f.unit_assumption == 'source_unit_unresolved'
                            for f in companions))
                same_value = unresolved and len({f.value for f in companions}) == 1
                peers = [s for s in plan.operand_slots if s.id != slot.id]
                pairable = [f for f in companions if same_value and any(
                    same_row_unresolved_percent([f, other])
                    for peer in peers for other in all_facts
                    if self._matches(other, peer, plan))]
                if selected:
                    pairable = [f for f in pairable if same_row_unresolved_percent([f, selected[0]])]
                if pairable:
                    reference = max(pairable, key=score)
                else:
                    values={f.value for f in companions if not f.unit_resolved}
                    if len(values)>1 and len({(f.period,f.role,f.entity,f.scope) for f in companions})==1:
                        report.add('data_contract','source_values_conflict','FAIL',
                                   'Các số gốc cùng chỉ tiêu/kỳ khác nhau; chưa thể chọn một số.',blocking=True,facts=tuple(companions))
                        return self._conflict(question,report,companions)
                    return self._unit_response(question,plan,report,reference,all_facts,companions),{}
            if explicit and any(not same_amount(reference,f) for f in comparable):
                report.add('data_contract','source_values_conflict','FAIL','Các công bố cùng chỉ tiêu/kỳ/phạm vi chưa thống nhất.',blocking=True,facts=(reference,*companions))
                return self._conflict(question,report,[reference,*companions])
            # Prefer the finest compatible disclosure after checking source conflicts.
            compatible=[f for f in companions if explicit and same_amount(reference,f)]
            if compatible:reference=min([reference,*compatible],key=lambda f:(f.quantum,bool(f.unit_assumption),-score(f)))
            if reference.fact_id in {f.fact_id for f in selected}:
                return self._empty(question,'Chưa xác nhận được các đầu vào riêng biệt cần cho phép tính.',report),{}
            bindings[slot.id]=reference;selected.append(reference)
        margin_crosscheck=()
        if (plan.operation=='SELECT' and len(selected)==1
                and selected[0].metric in {'net_margin','gross_margin'}):
            from .post_hrm_formulas import margin_operand_plan
            from .post_hrm_numeric_program import _printed_margin_issue
            margin_plan=margin_operand_plan(question)
            issue,margin_crosscheck=(
                _printed_margin_issue(self,margin_plan,selected[0],all_facts)
                if margin_plan else ('',()))
            if issue:
                message=('Tỷ lệ in và hai số nguồn cùng kỳ chưa thống nhất; chưa công bố.'
                         if issue=='calculation_disagrees_with_printed_ratio' else
                         'Tỷ lệ khôi phục từ bảng dính ô chưa có hai toán hạng độc lập đủ chứng cứ.')
                report.add('data_contract',issue,'INSUFFICIENT_EVIDENCE',
                           message,blocking=True,facts=tuple(selected))
                return self._empty(question,message,report,
                                   [selected[0].citation()]),{}
            if selected[0].period_binding=='aligned_compound_ratio_column':
                report.add('data_contract','aligned_compound_ratio','PASS',
                           'Tỷ lệ có span riêng, gắn nhãn theo thứ tự một-một của bảng và khớp hai toán hạng độc lập.',
                           facts=tuple(selected)+margin_crosscheck)
        unscaled=[f for f in selected if not f.unit_resolved]
        same_row_ratio = plan.operation == 'PERCENT_CHANGE' and same_row_unresolved_percent(selected)
        if unscaled and not same_row_ratio:
            report.add('data_contract','source_unit_scale_missing','INSUFFICIENT_EVIDENCE','Bảng chưa nêu đơn vị/thang đo tiền và chưa có bằng chứng đủ để suy ra.',blocking=True,facts=tuple(selected))
            return self._empty(question,'Bảng chưa ghi rõ đơn vị tiền (đồng, nghìn, triệu hay tỷ đồng). Bạn vui lòng xác nhận đơn vị trước khi công bố số tiền.',report,[f.citation() for f in selected]),{}
        if same_row_ratio:
            report.add('data_contract','same_row_scale_cancellation','PASS',
                       'Hai số cùng hàng và bảng; thang đo chung triệt tiêu trong phép tính phần trăm.',facts=tuple(selected))
        report.add('data_contract','source_and_intent','PASS','Đã đối chiếu các đầu vào bắt buộc với nguồn và ý hỏi.',facts=tuple(selected))
        records=[(n.id,n.operation,n.inputs) for n in plan.program_dag] or [('answer',plan.operation,tuple(bindings))]
        absolute_gap=(plan.operation=='SUBTRACT' and
                      getattr(plan.direction_intent,'value',plan.direction_intent)=='ABSOLUTE_GAP')
        if absolute_gap and (len(selected)!=2 or any(f.kind!='money' or f.currency!='VND' for f in selected)
                             or (selected[0].period,selected[0].role,selected[0].entity,selected[0].scope)
                             != (selected[1].period,selected[1].role,selected[1].entity,selected[1].scope)):
            report.add('data_contract','absolute_gap_inputs_incompatible','FAIL',
                       'Hai chỉ tiêu phải cùng kỳ, phạm vi và đơn vị tiền.',blocking=True,facts=tuple(selected))
            return self._empty(question,'Chưa xác nhận hai số tiền cùng kỳ, phạm vi và đơn vị để tính chênh lệch.',report),{}
        if plan.operation=='SUBTRACT' and tuple(bindings)==('subtract_left','subtract_right'):
            if len(records)!=1 or records[0][1]!='SUBTRACT' or tuple(records[0][2])!=('subtract_left','subtract_right'):
                report.add('data_contract','explicit_subtract_order','FAIL','Thứ tự chương trình khác thứ tự hai vế trong câu hỏi.',blocking=True)
                return self._empty(question,'Chưa xác nhận được đúng thứ tự hai đầu vào của phép trừ.',report),{}
            report.add('data_contract','explicit_subtract_order','PASS','Đã kiểm vế trái trừ vế phải theo câu hỏi.',blocking=True,facts=tuple(selected))
        if plan.operation=='SUBTRACT' and plan.comparison_target and plan.comparison_base:
            decreasing=getattr(plan.direction_intent,'value',plan.direction_intent)=='DECREASE_AMOUNT'
            expected=(plan.comparison_base,plan.comparison_target) if decreasing else (plan.comparison_target,plan.comparison_base)
            if len(records)!=1 or tuple(records[0][2])!=expected:
                report.add('data_contract','comparison_direction_consistent','FAIL','Chiều phép tính không khớp đối tượng và mốc so sánh trong câu hỏi.',blocking=True)
                return self._empty(question,'Chưa xác nhận được chiều phép tính theo câu hỏi.',report),{}
            report.add('data_contract','comparison_direction_consistent','PASS','Đã đối chiếu thứ tự toán hạng với ý hỏi tăng/giảm.',blocking=True)
        output_unit='%' if plan.operation in {'PERCENT_CHANGE','RATIO_PERCENT'} else 'VND' if all(f.kind=='money' and f.currency=='VND' and '/' not in f.unit for f in selected) and plan.operation in {'SELECT','ADD','SUBTRACT','AVERAGE','SUM3','SUM4','SUM_N'} else None
        if plan.operation=='SELECT' and selected[0].unit=='%':output_unit='%'
        if plan.operation=='DIVIDE' and not special:output_unit='x'
        try:
            result=compute_named(special[1],bindings,price_date=special[2]) if special and special[1] else compute_records(records,bindings,output_unit=output_unit)
            if absolute_gap and result.decimal_value < 0:
                left,right=records[0][2]
                records=[('answer','SUBTRACT',(right,left))]
                result=compute_records(records,bindings,output_unit=output_unit)
        except (ValueError,ArithmeticError) as e:
            report.add('computation','execution','FAIL',str(e),blocking=True,facts=tuple(selected));return self._empty(question,'Chưa thể tính từ các số liệu đã xác nhận. Vui lòng nêu rõ hai chỉ tiêu, kỳ và đơn vị cần dùng.',report),{}
        report.add('computation','independent_recompute','PASS','Đã tính và kiểm tra lại chương trình bằng số chính xác.',facts=tuple(selected))
        related=[f for f in all_facts if any((f.entity,f.scope,f.period,f.role)==(s.entity,s.scope,s.period,s.role) for s in selected)]
        report.checks.extend(accounting_checks(related));report.checks.extend(cross_statement_checks(all_facts))
        if not report.can_publish:return self._empty(question,'Chưa xác nhận đủ các điều kiện bắt buộc của đáp án.',report),{}
        value=result.decimal_value;unit=result.output.unit.symbol;citations=[f.citation(ref) for ref,f in bindings.items()]
        citations.extend(f.citation(ref) for ref,f in zip(
            ('crosscheck_profit','crosscheck_revenue'),margin_crosscheck))
        display_scale={'million_vnd':Decimal(1000000),'billion_vnd':Decimal(1000000000),'thousand_vnd':Decimal(1000)}.get(plan.requested_scale,Decimal(1)) if unit=='VND' else Decimal(1)
        display_unit={Decimal(1000000):'triệu đồng',Decimal(1000000000):'tỷ đồng',Decimal(1000):'nghìn đồng'}.get(display_scale,unit)
        rendered=self._render(plan,value/display_scale,display_unit,selected)
        if special and special[1]:
            label={'eps':'EPS','pe':'P/E','margin':'Biên lợi nhuận'}[special[1]]
            rendered=f'{label}: {_number(value,True)} '+('đồng/cổ phiếu.' if unit=='VND/share' else '%.' if unit=='%' else 'lần.')
        selected_ids={f.fact_id for f in selected}
        warnings=[c.message for c in report.checks if c.status=='WARNING' and selected_ids.intersection(c.fact_ids)]
        if warnings:rendered+='\nLưu ý: '+' '.join(dict.fromkeys(warnings))
        unit_note='Có công bố khác chưa xác định được đơn vị/thang đo; chưa thể đối chiếu với số có đơn vị rõ được dùng ở đây.' if any(c.code=='optional_disclosure_unit_missing' for c in report.checks) else ''
        source_refs=sorted({f'{_source_display_name(f.source_name)}, trang {f.page}' for f in selected});rendered+=' Nguồn: '+'; '.join(source_refs)+'.'
        if unit_note:rendered+='\n'+unit_note
        if plan.operation=='PERCENT_CHANGE' and len(selected)==2:
            current,base=selected
            current_value=current.canonical_value if current.unit_resolved else current.value
            base_value=base.canonical_value if base.unit_resolved else base.value
            rendered+=f'\nCông thức: ({_number(current_value)} − {_number(base_value)}) / {_number(base_value)} × 100 = {_number(value,True)}%.'
        obligations=[(c.code,'required' if c.blocking or c.code in {'source_and_intent','independent_recompute'} else 'optional','VALID' if c.status=='PASS' else 'UNKNOWN' if c.status in {'NOT_APPLICABLE','INSUFFICIENT_EVIDENCE'} else 'INVALID') for c in report.checks]
        payload={'display_value':str(value/display_scale),'display_unit':display_unit,'answer':str(value),'canonical_value':str(value),'unit':unit,'status':'ACCEPTED','final_action':'ANSWER','rendered':rendered,'citations':citations,
            'validation_engine':ENGINE_ID,'validation':report.model_dump(mode='json'),'program':{'operation':plan.operation,'operands':[f.candidate_index for f in selected]},
            'verified_program':[(n[0],n[1],list(n[2])) for n in records],'evidence_resolution':'UNIQUE','candidate_count':len(candidates),
            'proof':{'verification_engine':ENGINE_ID,'source_action':'ANSWER','verification_state':'VALID','obligations':obligations,'execution_certificate':result.certificate.to_dict()},
            'retrieval_trace':retrieval_trace,
            'hrm':{'segments':ranking['segments'],'neural_operation':ranking.get('neural_operation'),'legacy_symbolic_used':False}}
        return payload,bindings

    def _render(self,plan,value,unit,facts):
        label=display_label(facts[0].metric_label);kind=plan.operation
        if kind=='DIVIDE' and metric_key(plan.metric_text)=='current_ratio':
            label=display_label(plan.metric_text)
        if kind=='SELECT' and facts[0].period_binding=='aligned_compound_ratio_column':
            label=display_label(plan.metric_text)
        if (kind=='SELECT' and metric_key(facts[0].metric_label) in
                {'current_ratio','quick_ratio','inventory_turnover','return_on_assets','return_on_equity'}
                and re.search(r'[:/]|^\s*[+\-]',facts[0].metric_label)):
            label=display_label(plan.metric_text)
        suffix='đồng' if unit=='VND' else '' if unit in {'ratio','1'} else unit
        role={'plan':' kế hoạch','forecast':' dự báo','revised_plan':' kế hoạch điều chỉnh'}.get(facts[0].role,'')
        if kind=='SUBTRACT' and getattr(plan.direction_intent,'value',plan.direction_intent)=='ABSOLUTE_GAP':
            left,right=facts
            return (f'Chênh lệch giữa {display_label(left.metric_label)} và '
                    f'{display_label(right.metric_label)} năm {left.period} là {_number(value)} {suffix}.')
        if kind=='SUBTRACT' and tuple(s.id for s in plan.operand_slots)==('compare_left','compare_right'):
            left,right=facts
            verdict='Có' if value>0 else 'Không'
            return (f'{verdict}. {display_label(left.metric_label)}'+_fact_time_text(left)+
                    f' là {_number(left.canonical_value)} đồng; {display_label(right.metric_label)}'+
                    _fact_time_text(right)+f' là {_number(right.canonical_value)} đồng.')
        if kind=='SUBTRACT' and tuple(s.id for s in plan.operand_slots)==('subtract_left','subtract_right'):
            descriptions=[display_label(f.metric_label)+{'plan':' kế hoạch','forecast':' dự báo'}.get(f.role,'')+_fact_time_text(f) for f in facts]
            return descriptions[0]+' trừ '+descriptions[1]+f' bằng {_number(value)} {suffix}.'
        if kind=='SELECT':
            precision='khoảng ' if (facts[0].kind=='money' and unit=='VND'
                and facts[0].quantum>Decimal(1) and facts[0].role!='plan') else ''
            stated_unit=' lần' if unit in {'ratio','1','x'} and re.search(r'\bbao\s+nhieu\s+lan\b',normalize_text(plan.question)) else ('%' if suffix=='%' else ' '+suffix if suffix else '')
            polarity=' ('+('dương' if value>0 else 'âm' if value<0 else 'bằng không')+')' if re.search(r'\bduong\s+hay\s+am\b',normalize_text(plan.question)) else ''
            source_precision=(f' (nguồn ghi {facts[0].raw_value} {facts[0].unit})'
                              if precision else '')
            return f'{label}{role}'+_fact_time_text(facts[0])+f' là {precision}{_number(value)}{stated_unit}{polarity}{source_precision}.'
        if kind=='PERCENT_CHANGE':
            base_value=facts[1].canonical_value if facts[1].unit_resolved else facts[1].value
            if base_value<0:return f'{label}: tỷ lệ thay đổi theo công thức (mới − cũ) / cũ là {_number(value,True)}%; kỳ gốc âm nên cần đọc cùng hai số đầu vào.'
            time0=_fact_time_text(facts[0]) if facts[0].period_role in {'opening','closing'} else f' năm {facts[0].period}'
            time1=_fact_time_text(facts[1]) if facts[1].period_role in {'opening','closing'} else f' năm {facts[1].period}'
            return f'{label}{time0} '+('tăng' if value>0 else 'giảm' if value<0 else 'không đổi')+(f' khoảng {_number(abs(value),True)}%' if value else '')+f' so với{time1}.'
        if kind=='SUBTRACT' and plan.comparison_target and len({s.column_role for s in plan.operand_slots})>1:
            by_id=dict(zip((s.id for s in plan.operand_slots),facts));target=by_id[plan.comparison_target];base=by_id[plan.comparison_base]
            delta=target.canonical_value-base.canonical_value
            direction='tăng' if delta>0 else 'giảm' if delta<0 else 'không đổi'
            role_words={'plan':' kế hoạch','actual':' thực hiện','forecast':' dự báo'}
            return f'{label}{role_words.get(target.role,"")}{_fact_time_text(target)} {direction} {_number(abs(value))} {suffix} so với{role_words.get(base.role,"")}{_fact_time_text(base)}.'
        if kind=='SUBTRACT' and len(facts)==2 and {f.period_role for f in facts}=={'opening','closing'}:
            opening=next(f for f in facts if f.period_role=='opening')
            closing=next(f for f in facts if f.period_role=='closing')
            direction='tăng' if closing.canonical_value>opening.canonical_value else 'giảm' if closing.canonical_value<opening.canonical_value else 'không đổi'
            return f'{label}{_fact_time_text(closing)} {direction} {_number(abs(value))} {suffix} so với{_fact_time_text(opening)}.'
        if kind in {'DIVIDE','RATIO_PERCENT'}:return f'{label}: {_number(value,True)}'+('%' if kind=='RATIO_PERCENT' else ' lần' if unit in {'ratio','1','x'} else ' '+suffix)+'.'
        return f'{label}: {_number(value)} {suffix}.'

    def _comparison_render(self,outputs,bindings):
        if len(outputs)!=3 or any(o['status']!='ACCEPTED' for o in outputs):return None
        facts=[list(b.values()) for b in bindings]
        selected_ids={f.fact_id for fs in facts for f in fs}
        if any(c['status']=='WARNING' and selected_ids.intersection(c.get('fact_ids',[])) for o in outputs for c in o.get('validation',{}).get('checks',[])):return None
        if any(not fs for fs in facts) or len(facts[2])!=2:return None
        first,second=facts[0][0],facts[1][0]
        if first.metric!=second.metric or outputs[0]['display_unit']!=outputs[1]['display_unit']:return None
        unit=outputs[0]['display_unit'];unit='đồng' if unit=='VND' else unit
        note='Đơn vị mặc định là đồng vì bằng chứng được dùng không ghi đơn vị.'
        label=display_label(first.metric_label)
        def source(fs):
            by_name={}
            for f in fs:by_name.setdefault(f.source_name,set()).add(f.page)
            return '; '.join(_source_display_name(name)+', trang '+', '.join(str(p) for p in sorted(pages) if p is not None) for name,pages in sorted(by_name.items()))+'.'
        sections=[]
        same_year=first.period==second.period
        plan_actual=same_year and {first.role,second.role}=={'actual','plan'}
        for o,f in zip(outputs[:2],(first,second)):
            role=' kế hoạch' if same_year and f.role=='plan' else ' thực hiện' if same_year else ''
            text=f'{label}{role}{_fact_time_text(f)} là {_number(o["display_value"])} {unit}. Nguồn: '+source([f])
            if f.unit_assumption:text+='\n'+note
            sections.append(text)
        change=Decimal(outputs[2]['display_value']);current,base=facts[2]
        direction='tăng' if change>0 else 'giảm' if change<0 else 'không đổi'
        text=f'{label}{_fact_time_text(current)} {direction}'+(f' {_number(abs(change))} {unit}' if change else '')+f' so với '+('kế hoạch cùng năm.' if plan_actual else _fact_time_text(base).strip()+'.')
        scale={'triệu đồng':Decimal(1000000),'tỷ đồng':Decimal(1000000000),'nghìn đồng':Decimal(1000)}.get(outputs[2]['display_unit'],Decimal(1))
        text+=f'\nCông thức: ({_number(current.canonical_value/scale)} − {_number(base.canonical_value/scale)}) = {_number(change)} {unit}.'
        text+='\nNguồn: '+source(facts[2])
        if any(f.unit_assumption for f in facts[2]):text+='\n'+note
        prefix='Đối chiếu từng kỳ/loại số liệu và chênh lệch.'
        if first.role==second.role=='actual':prefix+='\nĐang dùng số thực hiện theo cách hiểu mặc định.'
        return prefix+'\n\nSố liệu thứ nhất:\n'+sections[0]+'\n\nSố liệu thứ hai:\n'+sections[1]+'\n\nChênh lệch:\n'+text

    @isolated_metric_labels
    @_timed_answer
    def answer(self,question,document,*,previous_question='',**options):
        from .conversation_context import resolve_followup
        with _measure('understanding'):
            context_resolution=resolve_followup(question,previous_question)
            if context_resolution:question=context_resolution['resolved_question']
            question=re.sub(r'\bLNST\b','Lợi nhuận sau thuế',question,flags=re.I)
            question=re.sub(r'\bLNTT\b','Lợi nhuận trước thuế',question,flags=re.I)
        from .conversation_context import followup_clarification, public_intent_message
        with _measure('understanding'):
            wording=normalize_text(question)
            literal_composition=('doanh thu chinh' in wording and 'chiem' in wording
                                 and 'bao nhieu phan tram' in wording)
            message=None if literal_composition else followup_clarification(question)
            preliminary=compile_question_plan(resolve_public_question(question))
            gate=_PROGRAM_GATE_TRACE.get()
            if gate is not None:gate.update({"planner_operation":preliminary.operation,
                "planner_slot_count":len(preliminary.operand_slots),
                "planner_abstention_reason":preliminary.abstention_reason,
                "requested_years":list(preliminary.requested_years)})
            from .question_planner import _explicit_subtract_slots
            subtract_surface=preliminary.operation=='SUBTRACT' or bool(re.search(r'\btrừ\b',question,re.I))
            incomplete_subtract=subtract_surface and _explicit_subtract_slots(normalize_text(question),preliminary.requested_years)==()
            if incomplete_subtract:
                if gate is not None:gate.update({"stage":"question_planning","stop_reason":"subtract_operands_incomplete"})
                report=ValidationReport()
                report.add('data_contract','subtract_operands_incomplete','INSUFFICIENT_EVIDENCE','Phép trừ chưa xác định đủ hai đầu vào.',blocking=True)
                result=self._empty(question,'Bạn muốn lấy chỉ tiêu nào trừ chỉ tiêu nào, ở năm hoặc kỳ nào? Vui lòng nêu đủ hai đầu vào.',report)
                result['intent_clarification_required']=True
                return result
            if message or (preliminary.abstention_reason=='missing_reference_base' and not literal_composition):
                if gate is not None:gate.update({"stage":"question_planning","stop_reason":"followup_context_missing" if message else "missing_reference_base"})
                code='followup_context_missing' if message else 'missing_reference_base'
                message=message or public_intent_message(code)
                report=ValidationReport()
                report.add('data_contract',code,'INSUFFICIENT_EVIDENCE',message,blocking=True)
                result=self._empty(question,message,report)
                result['intent_clarification_required']=True
                return result
        with _measure('source_catalog'):
            start=time.perf_counter();documents=discover_documents(document);self.snapshot=SourceSnapshot(documents)
            from .report_text_fields import answer_fields
            text_result=answer_fields(question,self.snapshot)
            if text_result is not None:return text_result
            cache_key=(tuple(sorted(self.snapshot.hashes.items())),
                       tuple(sorted((name,digest) for name,(_,digest) in self.snapshot.pdf_sources.items())))
            cached=self._catalog_cache.get(cache_key)
            cache_was_present=cached is not None
            if cached is None:
                chunks=build_chunks(documents,ocr_config=OCRConfig(enabled=False))
                all_candidates=merge_candidates('',chunks,max_candidates=10000)
                all_facts,errors=build_source_facts(all_candidates,self.snapshot)
                cached=(chunks,all_candidates,all_facts,errors)
                self._catalog_cache[cache_key]=cached
                while len(self._catalog_cache)>3:self._catalog_cache.popitem(last=False)
            self._catalog_cache.move_to_end(cache_key)
            chunks,all_candidates,all_facts,errors=cached
            gate=_PROGRAM_GATE_TRACE.get()
            if gate is not None:gate.update({"stage":"source_catalog",
                "numeric_candidate_count":len(all_candidates),
                "typed_fact_count":len(all_facts),
                "fact_error_reasons":dict(Counter(item.get("reason","unknown") for item in errors)),
                "source_period_conflicts":[item for name in documents for item in
                    self.snapshot.period_conflicts.get(Path(name).name,())]})
            set_source_metric_labels(c.row_label for c in all_candidates)

        bare_query=normalize_text(question).strip(' ?!.')
        ratio_metric=metric_key(preliminary.metric_text)
        if (not preliminary.requested_years and not preliminary.abstention_reason
            and ratio_metric in {'return_on_assets','return_on_equity'}
            and re.fullmatch(r'(?:chi so )?(?:roa|roe)(?: la bao nhieu(?: phan tram)?)?',bare_query)):
            ratio_facts=[f for f in all_facts if f.source_verified and f.metric==ratio_metric
                         and f.role=='actual' and f.reporting_entity
                         and re.fullmatch(r'(?:19|20)\d{2}',f.period)]
            years=sorted({f.period for f in ratio_facts})
            if 1<=len(years)<=8:
                label='ROA' if ratio_metric=='return_on_assets' else 'ROE'
                message=('Nguồn có '+label+' năm '+', '.join(years)+
                         '. Bạn muốn dùng năm nào?')
                report=ValidationReport()
                report.add('data_contract','named_ratio_year_missing',
                           'INSUFFICIENT_EVIDENCE',message,blocking=True,
                           facts=tuple(ratio_facts))
                witnesses=[next(f for f in ratio_facts if f.period==year)
                           for year in years]
                result=self._empty(question,message,report,
                                   [f.citation() for f in witnesses])
                result['named_ratio_year_choices']={
                    'metric':ratio_metric,'years':years,'question':question}
                return result
        if bare_query in {'bien loi nhuan','ty suat loi nhuan'}:
            message=('Bạn muốn biên lợi nhuận gộp, trước thuế hay sau thuế '
                     'của năm nào? Vui lòng nêu cả loại và năm trong một câu.')
            report=ValidationReport()
            report.add('data_contract','margin_definition_and_year_missing',
                       'INSUFFICIENT_EVIDENCE',message,blocking=True)
            return self._empty(question,message,report)
        if self.numeric_program_mode != 'off':
            from .post_hrm_numeric_program import evaluate_numeric_program
            programmed=evaluate_numeric_program(self,question,chunks,all_candidates,all_facts,
                                                selector=self.numeric_program_mode)
            if programmed is not None:
                from .post_hrm_proposals import _valid_citations
                if not self.snapshot.unchanged() or (programmed.get('citations') and
                        not _valid_citations(self.snapshot,programmed['citations'])):
                    report=ValidationReport()
                    report.add('data_contract','source_changed_or_citation_invalid',
                               'FAIL','Nguồn hoặc citation thay đổi trong lúc xử lý.',blocking=True)
                    return self._empty(question,'Tài liệu hoặc citation đã thay đổi; cần đọc lại nguồn.',report)
                programmed['source_catalog_cache_hit']=cache_was_present
                programmed['document_count']=len(documents)
                programmed['legacy_symbolic_used']=False
                return programmed
        from .post_hrm_raw_source import answer_unparsed_raw
        from .post_hrm_proposals import select_source_proposal,verify_source_proposal
        from .report_disclosures import answer_disclosures
        from .post_hrm_published import answer_published_ratio
        proposers=(
            lambda: answer_disclosures(question,self.snapshot),
            lambda: answer_unparsed_raw(question,self.snapshot,all_candidates,all_facts),
            lambda: answer_published_ratio(question,all_facts),
        )
        for propose in proposers:
            proposal=propose()
            if proposal is None:continue
            if proposal['status']=='ABSTAIN' and not proposal.get('citations'):
                return proposal
            source_only=proposal.get('answer_type') in {'text','raw_disclosure','explained_halt'}
            if source_only:
                # The current checkpoint accepts numeric tensors only. Text and
                # unresolved printed tokens keep their source contract, without
                # a fabricated zero-valued HRM candidate.
                chosen=proposal
                trace={'mode':'source_contract_only','proposed_citations':len(proposal.get('citations') or [])}
            else:
                chosen,trace=select_source_proposal(self.bundle,question,proposal,self.snapshot,
                                                    all_candidates,all_facts,chunks)
            if chosen is not None and not verify_source_proposal(chosen,propose(),self.snapshot):
                chosen=None
                trace['reason']='post_hrm_source_verification_failed'
            if chosen is None:
                report=ValidationReport()
                message='HRM chưa chọn được chứng cứ nguồn phù hợp để công bố câu trả lời.'
                report.add('data_contract','hrm_evidence_selection_unresolved',
                           'INSUFFICIENT_EVIDENCE',message,blocking=True)
                result=self._empty(question,message,report,proposal.get('citations',()))
                result['retrieval_trace']=trace
                return result
            chosen['proposal_route']=chosen.get('execution_route')
            chosen['execution_route']='source_contract_only' if source_only else 'hrm_source_proposal'
            chosen['retrieval_trace']=trace
            chosen['source_catalog_cache_hit']=cached is not None and cache_was_present
            chosen['document_count']=len(documents)
            chosen['legacy_symbolic_used']=False
            return chosen
        with _measure('understanding'):
            resolution=self._metric_resolution(question,previous_question,Path(document),all_candidates,all_facts)
            if resolution.clarification:
                result=self._empty(question,resolution.clarification);result['question_resolution']=resolution.as_dict();result['warnings']=['profit_metric_ambiguous' if resolution.bare_metric=='lợi nhuận' else 'revenue_metric_ambiguous'];return result
            request=compile_question_request(resolution.resolved_question)
            from .post_hrm_formulas import financial_calculation_plan
            if financial_calculation_plan(resolution.resolved_question):
                from .question_request_planner import QuestionRequestPlan,RequestedOutput
                text=resolution.resolved_question;plan=financial_calculation_plan(text)[0]
                request=QuestionRequestPlan(text,'SINGLE_OUTPUT',(RequestedOutput('output_01',1,(0,len(text)),text,text,plan),),'SINGLE_OUTPUT','')
        if request.fail_closed:return self._empty(question,'Chưa xác định được các phần cần trả lời: '+request.abstention_reason)
        outputs=[];known=[]
        for output in request.outputs:
            reuse=list({f.fact_id:f for bindings in known for f in bindings.values()}.values()) if request.split_strategy=='COMPARISON_OVERVIEW' and len(outputs)==2 else None
            result,bindings=self._scalar(output.question,chunks,all_candidates,all_facts,known_facts=reuse,catalog_errors=errors);result['output_id']=output.output_id;outputs.append(result);known.append(bindings)
        if not self.snapshot.unchanged():return self._empty(question,'Tài liệu đã thay đổi trong lúc xử lý; cần đọc lại nguồn.')
        if len(outputs)==1:result=outputs[0]
        else:
            accepted=[o for o in outputs if o['status']=='ACCEPTED'];result={'status':'ACCEPTED' if len(accepted)==len(outputs) else 'ABSTAIN','final_action':'ANSWER' if len(accepted)==len(outputs) else 'ABSTAIN','answer':None,'canonical_value':None,'unit':None,'outputs':outputs,'output_count':len(outputs),'coverage_state':'FULL' if len(accepted)==len(outputs) else 'PARTIAL' if accepted else 'NONE','rendered':'\n\n'.join(dict.fromkeys(o['rendered'] for o in outputs)),'validation_engine':ENGINE_ID,'citations':[]}
        if request.split_strategy=='COMPARISON_OVERVIEW':
            compact=self._comparison_render(outputs,known)
            if compact:result['rendered']=compact
        if resolution.basis:result['question_resolution']=resolution.as_dict();result['rendered']=resolution.note+'\n'+result['rendered']
        if context_resolution:
            result['context_resolution']=context_resolution
            if not result.get('question_resolution'):result['question_resolution']=context_resolution
        result['source_catalog_cache_hit']=cached is not None and cache_was_present
        result['service_answer_seconds']=time.perf_counter()-start;result['document_count']=len(documents);result['legacy_symbolic_used']=False
        return result


