"""Source-owned legal-entity sections, independent of values and QA labels.

This is separate from consolidated/separate accounting scope. A heading about
related companies is not authority to assign its fields to the report issuer.
"""
from pathlib import Path
from functools import lru_cache
import re
from .ontology import normalize_text
from .physical_coordinates import line_digest
from .source_bindings import section_heading


def _surface(text):
    text=text.strip().lstrip('\\-*•✓√+> ').strip().strip('*_')
    return re.sub(r'^(?:#{1,6}\s+|(?:\d+\.)*\d+[.)]?\s+|[IVXLCDM]+[.)]\s+)','',text).strip()


def _name(text):
    text=_surface(text).strip(' :')
    plain=normalize_text(text)
    if len(text)>160 or '|' in text or ':' in text:return ''
    if not re.match(r'^(?:(?:tong\s+)?cong ty|tap doan|ngan hang|company|corporation)\s+',plain):return ''
    if re.match(r'^(?:cac )?cong ty (?:me|con|lien ket|lien doanh)\b',plain):return ''
    if re.search(r'\b(?:khong co|khong phat sinh|bao cao|doanh thu|von dieu le|loi nhuan)\b',plain):return ''
    return text


def _event(text):
    raw=_surface(text);plain=normalize_text(raw)
    if '|' in text:return None
    roles={'me':'parent_company','con':'subsidiary','lien ket':'associate','lien doanh':'joint_venture'}
    hits=list(re.finditer(r'\bcong ty (me|con|lien ket|lien doanh)\b',plain))
    if not hits:return None
    if re.match(r'^(?:danh sach|thong tin (?:ve|cac)|cac cong ty)\b',plain) or len(hits)>1:
        if not re.match(r'^(?:danh sach|thong tin|cac cong ty|cong ty)\b',plain):return None
        return {'relation':roles[hits[0][1]] if len(hits)==1 else 'related','name':'','state':'UNKNOWN','group':True}
    m=re.match(r'^cong ty (me|con|lien ket|lien doanh)(?:\s|$)',plain)
    if not m:return None
    suffix=raw.split(':',1)[1].strip() if ':' in raw else ''
    if not suffix:
        named=re.match(r'^công\s+ty\s+(?:mẹ|con|liên\s+kết|liên\s+doanh)\s+(.*)$',raw,re.I)
        suffix=named[1].strip() if named else ''
    name=_name(suffix)
    # A relationship heading can itself introduce a short proper name, e.g.
    # "Công ty con ALPHA". Preserve the established identity for that form.
    if (not name and suffix and suffix[0].isupper() and len(suffix)<=120
            and not any(x in suffix for x in ':|;')
            and not re.search(r'\b(?:khong|chua|danh sach|nam|ket qua|doanh thu|loi nhuan)\b',normalize_text(suffix))
            and not re.search(r'\d',suffix)):
        name=suffix
    return {'relation':roles[m[1]],'name':name,'state':'VALID' if name else 'UNKNOWN','group':False}


def _records(lines):
    return [{'source_name':Path(x.source_path).name,'source_path':str(x.source_path),'page':x.page,'line_number':x.line_number,
             'text':x.text,'sha256':line_digest(x.text)} for x in lines]


def page_heading_indices(lines, *, records=False):
    """Unnumbered chapter starts, shared by ownership and table ancestry."""
    from .source_bindings import declared_unit
    source_records=lines if records else _records(lines)
    previous=None;front=True;breaks=set()
    for i,rec in enumerate(source_records):
        key=(rec.get('source_path',rec['source_name']),rec['page'])
        if key!=previous:front=True
        text=rec['text'].strip();plain=normalize_text(text)
        marker=bool(re.fullmatch(r'## Page \d+',text))
        running=bool(re.match(r'^(?:bao cao thuong nien|annual report)\b',plain)) or text.isdigit()
        company=_name(text)
        if (front and section_heading(text) is None and not _event(text) and not running and not company and not declared_unit(text)
                and '|' not in text and len(text)<=160 and sum(c.isalpha() for c in text)>=6 and text==text.upper()):
            breaks.add(i)
        if not (marker or running or company):front=False
        previous=key
    return breaks


def _walk(records, *, with_segments=True):
    """Yield owner plus the full bounded source segment that establishes it."""
    source=None;active=None;segment=[];section_level=100;boundary=100;previous_page=None;section_record=None;page_front=True
    page_breaks=page_heading_indices(records, records=True)
    for position,rec in enumerate(records):
        if rec.get('source_path',rec['source_name'])!=source:
            source=rec.get('source_path',rec['source_name']);active=None;segment=[];section_level=100;previous_page=None;section_record=None;page_front=True
        heading=section_heading(rec['text']);event=_event(rec['text'])
        if position in page_breaks:
            active=None;segment=[];section_record=None;section_level=100
        if heading:
            section_level=heading[0];section_record=rec
            if active and heading[0]<=boundary:active=None;segment=[]
        elif re.match(r'^#{1,6}\s+',rec['text']) and not re.match(r'^## Page \d+\s*$',rec['text']):
            active=None;segment=[]
        if active and previous_page is not None and rec['page'] is not None and rec['page'] not in (previous_page,previous_page+1):
            active={**active,'name':'','state':'UNKNOWN'}
        if event:
            active=event;boundary=section_level;segment=([section_record] if section_record is not None and section_record is not rec else [])+[rec]
        elif active:
            segment.append(rec)
            name=_name(rec['text'])
            # Source metadata is evaluated before chunk retrieval. Running
            # report headers do not start a different company record.
            if name and not (previous_page!=rec['page'] and active['name']):
                active={**active,'name':name,'state':'VALID'}
        if active:
            yield {'state':active['state'],'relation':active['relation'],'name':active['name'],
                   'entity_id':'entity:context:'+normalize_text(active['name']) if active['name'] else 'entity:unresolved:'+rec['source_name'],
                   'source_name':rec['source_name'],'page':rec['page'],'line_number':rec['line_number'],
                   'source_line_sha256':rec['sha256'],'segment':list(segment) if with_segments else []}
        else:yield None
        previous_page=rec['page']


def annotate(lines, annotations):
    for i,owner in enumerate(_walk(_records(lines))):
        if owner:annotations[i]['entity_owner']=owner


def owner_of(item):
    context=(getattr(item,'table_semantics',None) or {}).get('entity_owner')
    if context is None:
        context=(getattr(item,'report_context',None) or getattr(item,'report_field',None) or {}).get('entity_owner')
    return context


@lru_cache(maxsize=1024)
def _verified_segment(records):
    # Cache keys contain every immutable source field. An in-place change to
    # a witness therefore changes its key; a digest supplied by metadata alone
    # is never authority. Cached values are immutable semantic summaries.
    if not records:
        return None
    source=records[0][0:2]
    if any(r[0:2]!=source or line_digest(r[4])!=r[5] for r in records):
        return None
    if any(a[3]>=b[3] for a,b in zip(records,records[1:])):
        return None
    materialized=[dict(zip(('source_name','source_path','page','line_number','text','sha256'),r)) for r in records]
    reconstructed=None
    for reconstructed in _walk(materialized, with_segments=False):
        pass
    if reconstructed is None:
        return None
    return tuple(reconstructed[k] for k in ('state','relation','name','entity_id'))


def validate_owner(item, owner=None):
    owner=owner if owner is not None else owner_of(item)
    if owner is None:return None
    try:
        segment=owner['segment']
        if not segment:return 'INVALID'
        last=segment[-1]
        if any(last[k]!=getattr(item,k) for k in ('source_name','page','line_number')) or last['sha256']!=item.source_line_sha256:return 'INVALID'
        records=tuple((r['source_name'],r.get('source_path',r['source_name']),r['page'],r['line_number'],r['text'],r['sha256']) for r in segment)
        reconstructed=_verified_segment(records)
        if reconstructed is None or reconstructed!=tuple(owner[k] for k in ('state','relation','name','entity_id')):return 'INVALID'
        identity=getattr(item,'entity_id','') or getattr(item,'entity_scope','')
        if identity!=owner['entity_id']:return 'INVALID'
        return owner['state']
    except (KeyError,TypeError,AttributeError,ValueError):return 'INVALID'


def query_owner_state(plan, item):
    """None delegates legacy cells; explicit related sections never do so."""
    owner=owner_of(item)
    if owner is None:return None
    state=validate_owner(item,owner)
    if state!='VALID':return state
    if plan.entity_mode=='reporting':return 'INVALID'
    # Separate parent financial statements describe accounting scope. They do
    # not request the legal parent named in a related-company directory.
    if plan.requested_accounting_scope:return 'INVALID'
    if plan.entity_text=='parent_company':return 'VALID' if owner['relation']=='parent_company' else 'INVALID'
    expected=set(normalize_text(plan.entity_text).split())-{'cong','ty','co','phan','tnhh','tong','tap','doan','me','con','lien','ket','lien','doanh'}
    observed=set(normalize_text(owner['name']).split())
    return 'VALID' if expected and expected<=observed else 'INVALID'
