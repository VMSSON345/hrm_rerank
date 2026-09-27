"""Explain unresolved printed numbers without parsing or converting them to money."""
from __future__ import annotations
import re
from pathlib import Path
from .evidence_schema import evidence_cells
from .ontology import normalize_text


def _citation(snapshot,candidate,raw):
    return {'operand_ref':f'{candidate.source_name}:{candidate.line_number}:{candidate.source_span[0]}',
            'source_name':candidate.source_name,'page':candidate.page,
            'line_number':candidate.line_number,'source_span':list(candidate.source_span),
            'source_span_scope':'line_v2','source_line_sha256':candidate.source_line_sha256,
            'document_sha256':snapshot.hashes[candidate.source_name],
            'raw_value':raw,'unit_resolved':False,'unit_state':'UNIT_UNRESOLVED_RAW',
            'unit':'','scale':None,'number_format_basis':'unresolved_printed_token',
            'extraction_method':'mineru_markdown'}


def answer_unparsed_raw(question,snapshot,candidates,facts):
    q=normalize_text(question)
    years=set(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)',q))
    if len(years)!=1:return None
    year=next(iter(years))
    total_debt='tong no phai tra' in q and 'bao nhieu' in q
    loan_pair='vay ngan han' in q and 'vay dai han' in q and 'bao nhieu' in q
    if not (total_debt or loan_pair):return None
    found={}
    for candidate,cell in zip(candidates,evidence_cells(candidates)):
        if cell.period!=year or cell.column_role!='actual' or cell.is_percent:continue
        header=normalize_text(' '.join(cell.raw_header))
        if total_debt:
            role='total' if normalize_text(cell.metric_label)=='tong cong' and 'no phai tra' in header else ''
        else:
            label=normalize_text(cell.metric_label)
            role='short' if ('vay' in label and 'ngan han' in label) else 'long' if ('vay' in label and 'dai han' in label) else ''
            if role and 'no phai tra' not in header:role=''
        if not role:continue
        try:line,raw=snapshot.bound_line(candidate)
        except ValueError:continue
        if not raw or raw in {'-','—'}:continue
        if any(f.source_verified and f.source_name==candidate.source_name
               and f.line_number==candidate.line_number and f.period==year
               and f.raw_value==raw and f.kind=='money' and f.unit_resolved for f in facts):
            return None
        found.setdefault(role,[]).append((candidate,cell,raw))
    needed=('total',) if total_debt else ('short','long')
    if any(role not in found for role in needed):return None
    selected=[]
    for role in needed:
        values={raw for _,_,raw in found[role]}
        if len(values)!=1:return None
        selected.append(found[role][0])
    citations=[_citation(snapshot,c,raw) for c,_,raw in selected]
    source=Path(selected[0][0].source_name).stem
    if total_debt:
        raw=selected[0][2]
        rendered=(f'Tổng nợ phải trả năm {year} ghi số gốc `{raw}` trong bảng ({source}, trang {selected[0][0].page}). '
                  'Nguồn chưa chứng minh đơn vị tiền và quy ước dấu phân cách của ô này; '
                  'chưa thể công bố số tiền chính xác bằng đồng.')
        status='ABSTAIN';action='ABSTAIN';answer_type='explained_halt'
    else:
        short,long=selected
        rendered=(f'Theo bảng, Vay và nợ thuê tài chính ngắn hạn năm {year} ghi `{short[2]}`; '
                  f'Vay và nợ thuê tài chính dài hạn ghi `{long[2]}`. '
                  'Đây là các chuỗi số gốc; nguồn chưa chứng minh đơn vị và quy ước dấu phân cách, '
                  'nên chưa quy đổi ra đồng hoặc cộng hai khoản. '
                  f'Nguồn: {source}, trang {short[0].page}.')
        status='RAW_DISCLOSURE';action='DISCLOSE_RAW';answer_type='raw_disclosure'
    return {'status':status,'final_action':action,'answer_type':answer_type,
            'answer':None,'canonical_value':None,'unit':None,'numeric_answer':False,
            'unit_state':'UNIT_UNRESOLVED_RAW','raw_disclosures':[
                {'metric_label':cell.metric_label,'raw_value':raw,'unit':None,
                 'citations':[cite]} for (_,cell,raw),cite in zip(selected,citations)],
            'rendered':rendered,'citations':citations,'question':question,
            'validation_engine':'post_hrm_contract_v1','execution_route':'unparsed_raw_source',
            'proof':{'verification_engine':'post_hrm_contract_v1','source_action':action,
                     'source_certificate':{'citations':citations},'execution_certificate':None}}
