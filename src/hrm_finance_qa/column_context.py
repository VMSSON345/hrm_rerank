"""Source-bound column meaning and financial reporting periods, independent of queries."""
from __future__ import annotations
import re
from .ontology import normalize_text
from .physical_coordinates import line_digest


def normalize_period_text(text: str) -> str:
    """Separate a year fused to a currency label; leave source text untouched."""
    text = re.sub(r"\b((?:19|20)\d{2})(?=(?:VND|USD|EUR|GBP|JPY|đồng|dong)\b)",
                  r"\1 ", text, flags=re.IGNORECASE)
    return normalize_text(text)


def column_kind(column: str) -> str:
    text = normalize_text(column)
    if re.search(r"\b(?:bq|binh quan|trung binh|average|mean)\s+(?:\d+|nhieu|cac)\s+(?:nam|ky|years?|periods?)\b", text):
        return "multi_period_average"
    if re.search(r"\b(?:so sanh|thay doi|chenh lech|tang giam|tang truong|ss|difference|change|versus)\b", text):
        return "comparison_ratio" if "%" in column or re.search(r"\b(?:ty le|phan tram|percent)\b", text) else "comparison"
    if re.fullmatch(r"(?:nam )?(?:nay|truoc)|current year|prior year|previous year", text):
        return "relative_year"
    return "period_value"


def is_annual_operand_column(column: str, metric: str = "") -> bool:
    kind = column_kind(column)
    if kind == "multi_period_average":
        return column_kind(metric) == kind
    if kind in {"comparison", "comparison_ratio"}:
        return False
    return True


def financial_period(text: str) -> str | None:
    """Read a financial period, never a signing date or the annual-report title."""
    text = normalize_text(text)
    year = r"((?:19|20)\d{2})"
    date = r"(?:ngay\s+)?31\s+(?:thang\s+)?12\s+(?:nam\s+)?" + year
    matches = re.findall(r"\b(?:cho )?nam tai chinh ket thuc\s+" + date, text)
    if "cho nam tai chinh ket thuc cung ngay" in text:
        matches += re.findall(r"\b(?:bang can doi ke toan|tinh hinh tai chinh)(?: cua cong ty)? tai\s+" + date, text)
    if re.search(r"\b(?:bao cao tai chinh|bao cao ket qua hoat dong kinh doanh|bao cao luu chuyen tien te)\b", text):
        matches += re.findall(r"\bcho nam(?: tai chinh)?\s+" + year + r"\b", text)
    if ("bao cao tai chinh" in text and "cho nam ket thuc cung ngay" in text
            and "ket qua hoat dong kinh doanh" in text and "luu chuyen tien te" in text):
        matches += re.findall(r"\btinh hinh tai chinh(?: cua cong ty .+?)? tai\s+" + date, text)
    values = set(matches)
    return next(iter(values)) if len(values) == 1 else None


def _statement_title(text: str) -> bool:
    text = normalize_text(text)
    return len(text) < 180 and bool(re.search(
        r"\b(?:bao cao tai chinh|bang can doi ke toan|bao cao ket qua hoat dong kinh doanh|"
        r"bao cao luu chuyen tien te|thuyet minh bao cao tai chinh)\b", text))


def _company_name(text):
    text=re.sub(r"^\s*\\?[-*•]?\s*(?:Tên giao dịch|Ten giao dich)\s*:\s*", "", text, flags=re.I)
    from .entity_ownership import _name
    if re.search(r"\b(?:sau day|la cong ty|co tru so|duoc thanh lap)\b", normalize_text(text)):return ''
    name = normalize_text(_name(text))
    if not re.match(r"^cong ty (?:cp|co phan|tnhh|trach nhiem huu han)\b", name):return ''
    return re.sub(r"^cong ty cp\b", "cong ty co phan", name)


def _statement_company(text):
    text = normalize_text(text)
    match = re.search(r"\bbao cao tai chinh(?: kem theo)? cua (cong ty .+?)(?= sau day| duoc lap| bao gom)", text)
    if not match:
        match = re.search(r'\btinh hinh tai chinh cua (cong ty .+?) tai ngay 31 thang 12 nam (?:19|20)\d{2}\b',text)
    return _company_name(match[1]) if match else ''


def _notes_chapter(text):
    # Roman numbering restarts inside notes. Only recognizable note sections
    # stay in a notes block that the fiscal declaration explicitly includes.
    text = re.sub(r"^[ivxlcdm]+ ", "", normalize_text(text))
    return bool(re.match(r"^(?:dac diem hoat dong|nam tai chinh|chuan muc va che do ke toan|"
                         r"cac chinh sach ke toan|thong tin bo sung cho cac khoan muc|"
                         r"(?:nhung )?thong tin khac)\b", text))


def annotate_statement_periods(lines, blocks, annotations):
    """Bound fiscal anchors to financial sections; closed sections cannot inherit."""
    from .source_bindings import section_heading
    source = None; anchor = None; active = False; notes = False; contexts = {}; issuer = None; first_page = None
    for i,line in enumerate(lines):
        if line.source_path != source:
            source=line.source_path; anchor=None; active=False; notes=False; issuer=None; first_page=line.page
        text=line.text; normalized=normalize_text(text); heading=section_heading(text)
        if first_page is None and line.page is not None:first_page=line.page
        if issuer is None and line.page is not None and line.page == first_page and _company_name(text):
            issuer={"text":text,"sha256":line_digest(text),"page":line.page,"line_number":line.line_number}
        if "|" not in text:
            if re.fullmatch(r"bao cao kiem toan doc lap", normalized):
                anchor=None; active=False; notes=False
            title=_statement_title(text)
            notes_title = bool(re.search(r"\b(?:ban )?thuyet minh bao cao tai chinh\b", normalized)) and title
            if notes_title:
                notes = bool(financial_period(text) or anchor and "thuyet minh bao cao tai chinh" in normalize_text(anchor["text"]))
            # A note's internal chapter is not a new annual-report chapter.
            # An unknown chapter or a new statement block closes that context.
            if heading and heading[0] == 0 and not (notes and _notes_chapter(text)) and (
                    not title or notes and not notes_title):
                anchor=None; active=False; notes=False
            name = _company_name(text)
            if active and issuer and name and name != _company_name(issuer["text"]):
                anchor=None; active=False; notes=False
            if heading and re.search(r"\b(?:ke hoach|du bao|du kien|forecast|plan)\b", normalized):
                anchor=None; active=False; notes=False
            year=financial_period(text)
            if year:
                anchor={"year":year,"entity_id":(annotations[i].get("entity_owner") or {}).get("entity_id", ""),"source_name":str(source).replace('\\','/').split('/')[-1],
                        "page":line.page,"line_number":line.line_number,"text":text,"sha256":line_digest(text),
                        "issuer":issuer if issuer and _statement_company(text)==_company_name(issuer["text"]) else None}
            if title:active=True
        contexts[i]=(anchor if active else None)
    for position, block in enumerate(blocks):
        start,end=block['start'],block['end'];anchor=contexts[start]
        if not anchor:continue
        owner=(annotations[start].get('entity_owner') or {}).get('entity_id','')
        if owner and owner!=anchor.get('entity_id', ''):continue
        header=annotations[start].get('table_semantics',{}).get('header_evidence',[])
        continuation=None
        if not _relative_statement_header(header) and position:
            continuation=_statement_continuation(lines,blocks[position-1],block,annotations,anchor)
            if continuation:header=continuation['headers']
        if not header:continue
        for i in range(start,end):
            semantic=dict(annotations[i].get('table_semantics') or {})
            semantic['statement_period']={**anchor,"headers":header}
            if continuation:
                semantic['statement_continuation']=continuation
                semantic['header_evidence']=header
                left_binding=annotations[blocks[position-1]['start']]
                if not annotations[i]['unit'] and not annotations[i]['unit_conflict'] and left_binding['unit'] and not left_binding['unit_conflict']:
                    annotations[i]['unit']=left_binding['unit']
                    annotations[i]['unit_evidence']=left_binding['unit_evidence']
            annotations[i]['table_semantics']=semantic


def _relative_statement_header(headers):
    from .program_data import _markdown_cells
    for h in headers:
        cells=_markdown_cells(h.get('text',''))
        if (len(cells)==4 and normalize_text(cells[0]) in {'ma so','maso','stt'}
                and normalize_text(cells[1]) in {'chi tieu','noi dung'}
                and {normalize_text(x) for x in cells[2:]}=={'nam nay','nam truoc'}):return tuple(cells)
    return ()


def _statement_row_code(text):
    from .program_data import _markdown_cells
    cells=_markdown_cells(text)
    if len(cells)!=4:return None
    match=re.fullmatch(r"(?:(?:ma so|maso|stt) )?(\d{1,3})",normalize_text(cells[0]))
    return int(match[1]) if match and re.search(r"[a-z]",normalize_text(cells[1])) else None


def _statement_furniture(text,page):
    plain=normalize_text(text)
    if re.fullmatch(r"(?:page|trang) \d+",plain) or plain==str(page):return True
    if len(text)<=32 and text.startswith('$') and text.endswith('$') and not re.search(r'\d',text):return True
    return (0<len(plain)<=12 and text==text.upper() and not re.search(r'\d|cong ty|vnd|dong|usd|trieu|ty|nam|kh|th',plain))


def _statement_continuation(lines,left,right,annotations,anchor):
    a,b=lines[left['start']],lines[right['start']]
    sem=annotations[left['start']].get('table_semantics') or {}
    headers=sem.get('header_evidence',[]);columns=_relative_statement_header(headers)
    previous=sem.get('statement_period') or {}
    path=annotations[right['start']]['source_section_path']
    if (not columns or previous.get('sha256')!=anchor.get('sha256')
            or a.source_path!=b.source_path or b.page!=a.page+1
            or path!=annotations[left['start']]['source_section_path']
            or not any('bao cao ket qua hoat dong kinh doanh' in p for p in path)):return None
    left_rows=[x for x in lines[left['start']:left['end']] if _statement_row_code(x.text) is not None]
    right_rows=[x for x in lines[right['start']:right['end']] if not set(x.text)<=set('-|: ')]
    if not left_rows or not right_rows or any(_statement_row_code(x.text) is None for x in right_rows):return None
    codes=[_statement_row_code(left_rows[-1].text)]+[_statement_row_code(x.text) for x in right_rows]
    if any(x>=y for x,y in zip(codes,codes[1:])):return None
    gap=lines[left['end']:right['start']]
    if any(not _statement_furniture(x.text,a.page) for x in gap):return None
    def witness(x):return {'text':x.text,'sha256':line_digest(x.text),'page':x.page,'line_number':x.line_number}
    return {'source_name':a.source_name,'headers':headers,'columns':columns,
            'last_previous_row':witness(left_rows[-1]),'rows':[witness(x) for x in right_rows],
            'gap':[witness(x) for x in gap],'source_section_path':path}


def continued_statement_column(candidate):
    """Replay header/row alignment across one proven statement continuation."""
    from .program_data import _markdown_cells
    sem=getattr(candidate,'table_semantics',None) or {};proof=sem.get('statement_continuation') or {}
    headers=proof.get('headers',[]);columns=_relative_statement_header(headers)
    rows=proof.get('rows',[]);prior=proof.get('last_previous_row') or {};gap=proof.get('gap',[])
    if not columns or not rows or proof.get('source_name')!=candidate.source_name:return None
    if tuple(proof.get('source_section_path',()))!=tuple(candidate.source_section_path):return None
    witnesses=[*headers,prior,*rows,*gap]
    if any(w.get('sha256')!=line_digest(w.get('text','')) for w in witnesses):return None
    if any(h.get('page')!=candidate.page-1 for h in headers) or prior.get('page')!=candidate.page-1:return None
    if any(w.get('page')!=candidate.page for w in rows):return None
    if any(not _statement_furniture(w['text'],candidate.page-1) for w in gap):return None
    codes=[_statement_row_code(prior['text'])]+[_statement_row_code(w['text']) for w in rows]
    if any(x is None for x in codes) or any(x>=y for x,y in zip(codes,codes[1:])):return None
    line=next((w for w in rows if w['line_number']==candidate.line_number and w['sha256']==candidate.source_line_sha256),None)
    span=candidate.source_span
    if not line or not span or not 0<=span[0]<span[1]<=len(line['text']):return None
    if '|' in line['text'][span[0]:span[1]]:return None
    index=line['text'][:span[0]].count('|')-int(line['text'].lstrip().startswith('|'))
    if index not in (2,3):return None
    return columns[index]


def _statement_witness(candidate):
    column=getattr(candidate,'column_label',getattr(candidate,'column',''))
    evidence=(getattr(candidate,'table_semantics',None) or {}).get('statement_period') or {}
    if (evidence.get('source_name')!=candidate.source_name
            or evidence.get('sha256')!=line_digest(evidence.get('text',''))
            or evidence.get('year')!=financial_period(evidence.get('text',''))):return None
    if evidence.get('entity_id') and evidence['entity_id'] != getattr(candidate,'entity_id',''):return None
    headers=evidence.get('headers',[])
    if not headers or any(h.get('sha256')!=line_digest(h.get('text','')) for h in headers):return None
    if not any(normalize_text(column) in normalize_text(h['text']) for h in headers):return None
    if any(h.get('page')!=candidate.page or h.get('line_number',0)>candidate.line_number for h in headers):
        if continued_statement_column(candidate)!=column:return None
    if not isinstance(candidate.page, int) or candidate.page <= 0:return None
    if evidence.get('line_number', 0) > candidate.line_number or evidence.get('page', 0) > candidate.page:return None
    return evidence


def statement_reporting_company(candidate):
    """Explicit financial statement attribution to the named report issuer."""
    from .table_semantics import accounting_scope
    evidence = _statement_witness(candidate)
    if not evidence or accounting_scope(evidence['text']):return False
    issuer = evidence.get('issuer') or {}
    return bool(issuer and issuer.get('sha256') == line_digest(issuer.get('text', ''))
                and 0 < issuer.get('line_number', 0) <= evidence['line_number']
                and 0 < issuer.get('page', 0) <= evidence['page']
                and _company_name(issuer.get('text', ''))
                and _company_name(issuer['text']) == _statement_company(evidence['text']))


def resolved_relative_year(candidate):
    """Return a validated year only for an explicitly relative source column."""
    column=getattr(candidate,'column_label',getattr(candidate,'column',''))
    if column_kind(column) != 'relative_year':return None
    evidence = _statement_witness(candidate)
    if not evidence:return None
    year=int(evidence['year'])
    if normalize_text(column) in {'nam truoc','prior year','previous year'}:year-=1
    period = getattr(candidate, "period", "")
    if re.fullmatch(r"(?:19|20)\d{2}", period) and period != str(year):return None
    return str(year)
