"""Source-only numbered sections and explicitly bounded Markdown table units."""
from __future__ import annotations
import re
from .ontology import normalize_text
from .physical_coordinates import line_digest

_HEADING = re.compile(r'^\s*(?:#{1,6}\s*)?((?:\d+\.)*\d+|[IVXLCDM]+|[a-z])[.)]\s+([^|]+)$')

def section_heading(text):
    m=_HEADING.match(text.strip().strip('*'))
    if not m:
        plain=normalize_text(text.strip().strip('*'))
        if re.fullmatch(r"ket qua (?:hoat dong )?(?:san xuat )?kinh doanh(?: nam (?:19|20)\d{2})?", plain):
            return 50,plain
        return None
    if len(m[2])>120 or not re.match(r'[^\W\d_]',m[2],re.UNICODE):return None
    number,title=m.groups()
    level=0 if re.fullmatch('[IVXLCDM]+',number) else 1+number.count('.') if number[0].isdigit() else 10
    return level,normalize_text(number+' '+title).strip()

def requested_section(question):
    match=re.search(r'(?:theo|trong|tại|tai)\s+(?:mục|muc|phần|phan|section)\s+((?:\d+\.)*\d+|[IVXLCDM]+)\s*[:.\-]?\s+(.+?)(?=\s+(?:là bao nhiêu|la bao nhieu|chênh lệch|chenh lech|bao nhiêu|bao nhieu)|[,?;]|$)',question,re.I)
    return normalize_text(match[1]+' '+match[2]).strip() if match else ''

def question_source_frame(question):
    """Separate a named table/paragraph locator from the fact being requested."""
    text = normalize_text(question)
    table = re.search(r"\b(?:trong|theo|o|cua) bang (giai doan ((?:19|20)\d{2}) ((?:19|20)\d{2})|nam nam|tinh hinh tai chinh|ket qua(?: kinh doanh)?(?: hop nhat)?)\b", text)
    narrative = re.search(r"\b(?:(?:duoc neu )?(?:trong|theo|o) (?:phan|doan)|(?:phan|doan)) danh gia(?: (ket qua|cua ban tong giam doc))?(?: neu)?\b|\bhoat dong kinh doanh neu\b", text)
    profile = re.search(r"\b(?:duoc ghi )?(?:o|trong) thong tin khai quat\b", text)
    match = table or narrative or profile
    if not match:
        return question, None
    kind = "table" if table else "narrative" if narrative else "profile"
    heading = ""
    years = ()
    count = 0
    if table:
        phrase = table[1]
        if table[2] and table[3]:
            first,last=int(table[2]),int(table[3])
            if last<first or last-first>20:
                return question, None
            years=tuple(str(y) for y in range(first,last+1))
        elif phrase == "nam nam":
            count=5
        else:
            heading="tinh hinh tai chinh" if phrase=="tinh hinh tai chinh" else "ket qua kinh doanh" if "kinh doanh" in phrase else "ket qua"
    elif narrative:
        heading="danh gia cua ban tong giam doc" if narrative[1]=="cua ban tong giam doc" else "ket qua" if narrative[1]=="ket qua" else ""
    else:
        heading="thong tin"
    core=(text[:match.start()] + " " + text[match.end():]).strip()
    core=re.sub(r"\s+", " ", core)
    if kind == "narrative":
        core=re.sub(r"^((?:thuc hien )?(?:nam )?(?:19|20)\d{2}) neu (.+)$", r"\2 \1", core)
    return core, {"kind":kind,"heading":heading,"years":years,"year_count":count}


def declared_unit(text):
    from .program_data import _explicit_currency_units
    plain=text.strip().lstrip('>').strip().strip('|').strip().strip('*_')
    normalized=normalize_text(plain)
    if not re.match(r'^(?:don vi(?: tinh)?|dvt|units?)\b',normalized) or len(normalized.split())>9:return set()
    return _explicit_currency_units(plain)


def _numbered_page_continuations(lines, blocks, annotations):
    """Link a headerless fragment only with ordinal, section and page proof."""
    from .program_data import _markdown_cells, _is_markdown_separator, _is_table_header_or_separator
    links=[]
    def code(text):
        first=_markdown_cells(text)[0] if _markdown_cells(text) else ''
        return tuple(map(int,first.split('.'))) if re.fullmatch(r"[0-9]+(?:\.[0-9]+)*",first) else None
    def width(text):
        # A leading pipe may be an empty row-code cell in MinerU output.
        return len(_markdown_cells(text)) + int(text.lstrip().startswith('|')) + int(text.rstrip().endswith('|'))
    def witness(x):
        return {'source_name':str(x.source_path).replace('\\','/').split('/')[-1], 'page':x.page,'line_number':x.line_number,'text':x.text,'sha256':line_digest(x.text)}
    for left,right in zip(blocks,blocks[1:]):
        a,b=lines[left['start']],lines[right['start']]
        if a.source_path!=b.source_path or a.page is None or b.page!=a.page+1:continue
        if annotations[left['start']]['source_section_path']!=annotations[right['start']]['source_section_path']:continue
        gap=lines[left['end']:right['start']]
        if any(normalize_text(x.text) and not re.fullmatch(r'(?:page|trang) [0-9]+|scanned with camscanner|tiep theo|continued',normalize_text(x.text)) for x in gap):continue
        header_end=next((j for j in range(left['start'],left['end']) if _is_markdown_separator(lines[j].text)),None)
        if header_end is None:continue
        headers=lines[left['start']:header_end]
        if not headers or not any(re.search(r'chi tieu|noi dung',normalize_text(h.text)) for h in headers):continue
        if len(set(re.findall(r'(?:19|20)\d{2}',' '.join(h.text for h in headers))))<2:
            # Relative endpoints may inherit a year from their own adjacent
            # balance title, but still require the ordinal continuation proof.
            before=[x for x in lines[max(0,left['start']-3):left['start']] if not declared_unit(x.text)]
            title=before[-1] if before else None
            normal=normalize_text(' '.join(h.text for h in headers))
            if (title is None or title.page!=a.page
                    or not re.fullmatch(r'(?:[a-z] )?bang can doi ke toan(?: hop nhat| rieng(?: le)?)? nam (?:19|20)\d{2}',normalize_text(title.text))
                    or not re.search(r'\bdau (?:nam|ky)\b',normal)
                    or not re.search(r'\bcuoi (?:nam|ky)\b',normal)):continue
        body=[x for x in lines[right['start']:right['end']] if not _is_markdown_separator(x.text)]
        if not body or not all(not _is_table_header_or_separator([x.text],0) for x in body):continue
        if any(normalize_text(cell) in {'muc','noi dung','chi tieu','ma so','thuyet minh','item','indicator'} for x in body for cell in _markdown_cells(x.text)):continue
        oldcodes=[code(x.text) for x in lines[header_end+1:left['end']] if code(x.text)]
        newcodes=[code(x.text) for x in body if code(x.text)]
        if not oldcodes or not newcodes:continue
        last,first=oldcodes[-1],newcodes[0]
        if len(last)!=len(first) or last[:-1]!=first[:-1] or first[-1]!=last[-1]+1:continue
        target_width=len(_markdown_cells(headers[-1].text))
        if any(width(x.text)!=target_width for x in body):continue
        if any(declared_unit(x.text) or section_heading(x.text) for x in gap):continue
        if right['units'] and right['units']!=left['units']:continue
        proof={'headers':[witness(x) for x in headers],'gap':[witness(x) for x in gap],
               'previous_code':last,'next_code':first,'width':target_width}
        right['units']=left['units'];left['number_group']=left['start'];right['number_group']=left['start']
        for j in range(right['start'],right['end']):annotations[j]['numbered_continuation']=proof
        links.append((left,right))
    return links

def source_annotations(lines):
    """Do not rewrite source text, coordinates, retrieval scores or numbers."""
    from .program_data import _markdown_cells
    from .entity_ownership import page_heading_indices
    page_breaks=page_heading_indices(lines)
    result={};stack=[];prior_source=None
    for i,line in enumerate(lines):
        if line.source_path!=prior_source:stack=[];prior_source=line.source_path
        if i in page_breaks:stack=[(-1,normalize_text(line.text))]
        heading=section_heading(line.text)
        if heading:
            level,title=heading
            while stack and stack[-1][0]>=level:stack.pop()
            stack.append((level,title))
        result[i]={'line_number':line.line_number,'line_sha256':line_digest(line.text),'source_section_path':tuple(x[1] for x in stack),'unit':'','unit_conflict':False,'unit_evidence':[], 'section_reset':i in page_breaks}
    blocks=[];i=0
    while i<len(lines):
        if '|' not in lines[i].text or declared_unit(lines[i].text):i+=1;continue
        start=i;key=(lines[i].source_path,lines[i].page)
        while i<len(lines) and (lines[i].source_path,lines[i].page)==key and '|' in lines[i].text and not declared_unit(lines[i].text) and (i==start or lines[i].line_number==lines[i-1].line_number+1):i+=1
        end=i
        if end-start<2:continue
        header=tuple(normalize_text(x) for x in _markdown_cells(lines[start].text))
        # A continuation needs a repeated descriptor/year header, not just equal width.
        explicit_header=bool(header and re.search(r'chi tieu|noi dung|indicator|item',header[0]))
        units=[]
        for j in [start-1,end]:
            if not 0<=j<len(lines) or lines[j].source_path!=key[0] or lines[j].page!=key[1]:continue
            declared=declared_unit(lines[j].text)
            if not declared:continue
            owned_footer = lines[j].text.lstrip().startswith('>')
            if owned_footer and j < start:continue
            # A bare declaration between adjacent tables cannot prove which one owns it.
            other=j-1 if j<start else j+1
            if not owned_footer and 0<=other<len(lines) and '|' in lines[other].text and not declared_unit(lines[other].text):continue
            units.append((declared,j))
        blocks.append({'start':start,'end':end,'header':header,'explicit_header':explicit_header,'units':units})
    repeated_running = {}
    for line in lines:
        if line.text == line.text.upper() and len(line.text) <= 100 and '|' not in line.text:
            repeated_running.setdefault((line.source_path, line.text), set()).add(line.page)
    # Resolve bounded continuations before applying a footer backwards.
    for position, (left, right) in enumerate(zip(blocks, blocks[1:]), 1):
        a,b=lines[left['end']-1],lines[right['start']]
        if a.source_path!=b.source_path or a.page is None or b.page!=a.page+1:continue
        if not left['explicit_header'] or left['header']!=right['header']:continue
        path=result[left['start']]['source_section_path']
        if not path or path!=result[right['start']]['source_section_path']:continue
        gap=lines[left['end']:right['start']]
        # Only recognizable running matter may separate the two fragments.
        if any(not re.match(r'^(?:page\b|trang\b|tiep theo\b|continued\b)', normalize_text(x.text)) and len(repeated_running.get((x.source_path, x.text), ())) < 2 for x in gap):continue
        old_labels={normalize_text(_markdown_cells(x.text)[0]) for x in lines[left['start']+1:left['end']] if _markdown_cells(x.text) and not set(x.text)<=set('-|: ')}
        new_labels={normalize_text(_markdown_cells(x.text)[0]) for x in lines[right['start']+1:right['end']] if _markdown_cells(x.text) and not set(x.text)<=set('-|: ')}
        if old_labels & new_labels:continue
        first = left.get('group_start', position - 1)
        merged = list({j: (unit, j) for block in blocks[first:position + 1] for unit, j in block['units']}.values())
        for block in blocks[first:position + 1]:
            block['units'] = merged
            block['group_start'] = first
    numbered_links = _numbered_page_continuations(lines, blocks, result)
    for block in blocks:
        units=set().union(*(u for u,_ in block['units'])) if block['units'] else set()
        evidence=[{'page':lines[j].page,'line_number':lines[j].line_number,'text':lines[j].text,'sha256':line_digest(lines[j].text)} for _,j in block['units']]
        for j in range(block['start'],block['end']):
            result[j].update(unit=next(iter(units)) if len(units)==1 else '',unit_conflict=len(units)>1,unit_evidence=evidence)
    from .number_formats import annotate as annotate_numbers
    annotate_numbers(lines, blocks, result)
    from .table_semantics import annotate_tables
    annotate_tables(lines, blocks, result)
    for left,right in numbered_links:
        prior=result[left['start']].get('table_semantics',{})
        for j in range(right['start'],right['end']):
            result[j]['table_semantics']={**prior,'header_evidence':result[j]['numbered_continuation']['headers'],
                'numbered_continuation':result[j]['numbered_continuation']}

    from .report_context import annotate_report_context
    annotate_report_context(lines, result)
    from .entity_ownership import annotate as annotate_ownership
    annotate_ownership(lines, result)
    # Named sections of the issuer's own operating review establish legal
    # reporting context, not a fabricated separate/consolidated perimeter.
    # Related-company segments and explicit entity/scope conflicts never inherit it.
    headings = {}
    for i, line in enumerate(lines):
        heading = section_heading(line.text)
        if heading:
            headings[(line.source_path, heading[1])] = line
        semantic = result[i].get("table_semantics") or {}
        if (semantic.get("reporting_company_operations_evidence")
                or result[i].get("entity_owner") or semantic.get("entity_id")
                or semantic.get("entity_conflict") or semantic.get("scope_conflict")):
            continue
        witnesses = []
        for title in result[i]["source_section_path"]:
            if not re.search(r"\b(?:tinh hinh hoat dong trong nam|bao cao va danh gia cua ban (?:tong )?giam doc|danh gia cua hoi dong quan tri ve hoat dong cua cong ty|danh gia ket qua (?:hoat dong )?(?:san xuat )?kinh doanh)\b", title):
                continue
            if re.search(r"\b(?:cong ty con|cong ty lien ket|cong ty me)\b", title):
                continue
            context = headings.get((line.source_path, title))
            if context is not None:
                witnesses.append({"page":context.page,"line_number":context.line_number,
                                  "text":context.text,"sha256":line_digest(context.text)})
        if witnesses:
            qualifiers = r"\b(?:cong ty con|don vi|rieng|hop nhat|bo phan|linh vuc|khu vuc|phan khuc|segment|division|subsidiary)\b"
            ancestor_text = " / ".join(result[i]["source_section_path"])
            header_start = min((w["line_number"] for w in semantic.get("header_evidence", ())), default=line.line_number)
            owner_start = max(w["line_number"] for w in witnesses)
            local_text = " ".join(candidate.text for candidate in lines
                if candidate.source_path == line.source_path and owner_start <= candidate.line_number < header_start
                and "|" not in candidate.text and not declared_unit(candidate.text))
            # Segment words in general narrative (e.g. regulation of a
            # business sector) do not label this table's accounting scope.
            # Headings and explicit data/segment introductions still do.
            local_scope = r"\b(?:cong ty con|don vi|rieng|hop nhat|subsidiary)\b"
            segment_intro = r"^(?:(?:ket qua|so lieu|doanh thu|loi nhuan)(?: hoat dong)?(?: san xuat)?(?: kinh doanh)?(?: cua|tu|cho)? )?(?:bo phan|linh vuc|khu vuc|phan khuc|segment|division)\b"
            segment_lines = [candidate.text for candidate in lines
                if candidate.source_path == line.source_path and owner_start <= candidate.line_number < header_start
                and "|" not in candidate.text and not declared_unit(candidate.text)
                and re.search(segment_intro, normalize_text(candidate.text.lstrip("\\- *•>")))]
            if (not re.search(qualifiers, ancestor_text)
                    and not re.search(local_scope, normalize_text(local_text)) and not segment_lines):
                result[i]["table_semantics"] = {**semantic, "reporting_company_operations_evidence": witnesses}
    from .column_context import annotate_statement_periods
    annotate_statement_periods(lines, blocks, result)
    return result