"""Source-bound table roles and accounting scopes; never inspect a question/value."""
from __future__ import annotations
import re
from .ontology import normalize_text
from .physical_coordinates import line_digest



def _scope_title(text):
    value = normalize_text(text)
    value = re.sub(r"^(?:(?:bang|table) [0-9]+|(?:[ivxlcdm]+|[0-9]+)(?: [0-9]+)*) ", "", value)
    return re.sub(r" nam (?:19|20)\d{2}$", "", value)


def accounting_scope(text):
    value=normalize_text(text)
    consolidated=bool(re.search(r"\b(?:hop nhat|toan he thong|consolidated)\b",value))
    separate=bool(re.search(r"\b(?:rieng cong ty me|rieng le|tai rieng cong ty|tai chinh rieng|bao cao rieng|bctc rieng|separate financial|standalone)\b",value))
    if consolidated and separate:return "conflict"
    return "consolidated" if consolidated else "separate" if separate else ""


def _title_role(text):
    value=normalize_text(text)
    actual=bool(re.search(r"\b(?:thuc hien|thuc te|ket qua (?:hoat dong )?(?:san xuat )?kinh doanh|tinh hinh tai chinh|actual results|financial position|financial performance|business performance|operating results|income statement|statement of profit or loss)\b",value))
    plan=bool(re.search(r"\b(?:ke hoach|plan|budget|targets?)\b",value))
    forecast=bool(re.search(r"\b(?:du bao|du kien|uoc tinh|forecast|estimated?)\b",value))
    if forecast:return "forecast"
    if (re.match(r"^nguon\b",value) and re.search(r"\b(?:bctc|bao cao tai chinh|financial statements)\b",value) and not plan):
        return "actual"
    if plan and actual:return "mixed"
    if plan:return "revised_plan" if re.search(r"dieu chinh|sua doi|revised|adjusted",value) else "plan"
    return "actual" if actual else ""


def annual_data_column(column, period):
    """Only bare annual data headers inherit a role, never ratios/dates/order."""
    year=normalize_text(period)
    if not re.fullmatch(r"(?:19|20)\d{2}",year):return False
    # Currency captions decorate the annual header; they do not change its
    # year or role. Keep all other qualifiers (KH, comparison, scope) intact.
    label=normalize_text(column)
    label=re.sub(r"\s+(?:(?:dvt|don vi(?: tinh)?)\s+)?(?:(?:nghin|trieu|ty)\s+)?(?:dong|vnd|usd|eur|jpy|gbp)$", "", label)
    # Restatement marks the reported version, not a different year/role.
    # Keep the original column in evidence; incompatible versions still fail
    # the disclosure-value/identity checks.
    label=re.sub(r"\s+(?:trinh bay lai|restated)$", "", label)
    return bool(re.fullmatch(r"(?:(?:nam|year)\s*)?"+year,label))


def annotate_tables(lines, blocks, annotations):
    from .source_bindings import section_heading, declared_unit
    running_pages={}
    for line in lines:
        if line.text==line.text.upper() and len(line.text)<=100 and "|" not in line.text:
            running_pages.setdefault((line.source_path,line.text),set()).add(line.page)
    stack=[];caption=[];source=None;contexts={};caption_consumed=False
    for i,line in enumerate(lines):
        if source!=line.source_path:stack=[];caption=[];source=line.source_path;caption_consumed=False
        if annotations[i].get("section_reset"):
            stack=[(-1,i)];caption=[];caption_consumed=False
        text=line.text.strip();heading=section_heading(text)
        if heading:
            while stack and stack[-1][0]>=heading[0]:stack.pop()
            stack.append((heading[0],i));caption=[]
        elif "|" not in text and not text.startswith(">") and not declared_unit(text):
            plain=normalize_text(text.lstrip("\\- *•"))
            title=bool(len(text)<=220 and re.match(r"^(?:(?:bang|table) \d+ )?(?:ket qua|tinh hinh tai chinh|tinh hinh thuc hien so voi ke hoach|bao cao tai chinh|so lieu|nguon|financial performance|financial position|business performance|operating results|income statement|statement of profit or loss)\b",plain))
            # Short prospective captions also constrain the table when they
            # are unnumbered. Do not treat a general sentence as a heading.
            title = title or bool(re.fullmatch(
                r"(?:ke hoach(?: dieu chinh)?|du kien|du bao|(?:business |financial )?(?:plan|forecast|budget|targets?))(?: (?:kinh doanh|tai chinh))?(?: (?:nam |year )?(?:19|20)\d{2})?", plain))
            if title:
                # Adjacent captions may separately declare role and scope.
                # Compose before their table, on the same page. A new caption
                # after a table starts a new local context.
                adjacent = (caption and not caption_consumed
                            and lines[caption[-1]].page == line.page
                            and not re.match(r"^\\?[-*•]", text))
                caption = caption + [i] if adjacent or plain.startswith("nguon ") else [i]
                caption_consumed = False
            elif re.match(r"^\\?[-*•]",text):caption=[]
            elif caption and not (
                re.fullmatch(r"\d+",plain)
                or re.match(r"^(?:page|trang|tiep theo|continued)\b",plain)
                or text.startswith("## Page ")
                or len(running_pages.get((line.source_path,line.text),()))>=2
            ):
                # Ordinary prose or an unnumbered unrelated heading closes a
                # local caption. Numbered ancestors keep their own scope.
                caption=[]
        if "|" in text:caption_consumed=True
        contexts[i]=[j for _,j in stack]+caption
    def evidence(indices):
        return [{"page":lines[j].page,"line_number":lines[j].line_number,"text":lines[j].text,"sha256":line_digest(lines[j].text)} for j in dict.fromkeys(indices)]
    for pos,block in enumerate(blocks):
        start=block["start"];end=block["end"]
        context=list(contexts[start]);scopes=[];roles=[];entities=[]
        for j in context:
            role=_title_role(lines[j].text)
            if role:roles.append((role,j))
            scope=accounting_scope(lines[j].text)
            if scope:scopes.append((scope,j))
            value=normalize_text(lines[j].text)
            match=re.search(r"\b(?:cong ty|don vi) (?:con|lien ket|lien doanh) (.+)$",value)
            if match and "cac cong ty" not in value and not re.search(r"\b(?:khong co|khong phat sinh)\b",value):
                entities.append(("entity:context:"+match[1].strip(),j))
        # A source attribution immediately after the table is owned by it.
        for j in range(end,min(end+3,len(lines))):
            if lines[j].source_path!=lines[start].source_path or lines[j].page!=lines[start].page:break
            text=lines[j].text.strip()
            if declared_unit(text):continue
            plain=normalize_text(text.lstrip("> "))
            if re.match(r"^(?:nguon|xem (?:tai )?(?:bao cao|bctc))\b",plain):
                scope=accounting_scope(text)
                if scope:scopes.append((scope,j))
                # A table attributed to financial statements reports actual
                # figures. Its explicit plan/forecast/ratio columns still win.
                if (re.search(r"\b(?:bao cao tai chinh|bctc|financial statements)\b", plain)
                        and not re.search(r"\b(?:ke hoach|du bao|du kien|forecast|plan|target)\b", plain)):
                    roles.append(("actual",j))
            break
        # Specific source-column roles will override this table-wide default.
        role=roles[-1][0] if roles else ""
        if role=="actual":
            prospective=[r for r in roles if r[0] in {"plan","revised_plan","forecast","mixed"}]
            if prospective:role=prospective[-1][0]
        # Adjacent unnumbered titles have no hierarchy resolving incompatible
        # roles. Preserve the conflict instead of choosing the last title.
        local_roles={r for r,j in roles if section_heading(lines[j].text) is None}
        if len(local_roles)>1:role="mixed"
        values={s for s,_ in scopes}
        conflict="conflict" in values or len(values)>1
        scope=next(iter(values)) if len(values)==1 and not conflict else ""
        names={name for name,_ in entities}
        role_years=sorted({year for r,j in roles if r==role for year in re.findall(r"\b(?:19|20)\d{2}\b",normalize_text(lines[j].text))}) if role in {"plan","revised_plan","forecast"} else []
        from .program_data import _markdown_cells, _resolved_table_header_role
        separator=next((j for j in range(start,min(end,start+6)) if set(lines[j].text)<=set("-|: ")),start+1)
        # MinerU can emit observed ratio rows before a separator. A row with
        # two scalar cells is data, even when its numbers are below 1,000.
        # Never promote those rows into header evidence for earlier cells.
        header_end=separator
        for j in range(start+1,separator):
            cells=_markdown_cells(lines[j].text)
            if len(cells)<3:continue
            values=[cell.rsplit(":",1)[-1].strip() for cell in cells[1:]]
            if sum(bool(re.fullmatch(r"[()+\-−]?\d[\d., ]*[%)]?", value)) and not bool(re.fullmatch(r"(?:19|20)\d{2}",value)) for value in values)>=2:
                header_end=j
                break
        # An explicitly titled actual-vs-plan table identifies the bare-year
        # lane as actual when a separate header explicitly identifies PLAN.
        # Neither the question nor numeric magnitudes supply this role.
        comparison_titles=[j for j in context if re.fullmatch(
            r"(?:tinh hinh )?thuc hien so voi ke hoach",
            normalize_text(lines[j].text.lstrip("\\- *•")))]
        header_cells=[cell for j in range(start,header_end) for cell in _markdown_cells(lines[j].text)]
        explicit_header_roles={_resolved_table_header_role((cell,)) for cell in header_cells}
        # A financial-statement attribution owned by this table also proves
        # its bare-year actual columns. A comparison caption alone does not
        # contradict that proof; an explicit prospective title still does.
        actual_statement_footer=any(r=="actual" and j>=end for r,j in roles)
        actual_plan_comparison=(role=="mixed" and comparison_titles
            and ("plan" in explicit_header_roles or actual_statement_footer)
            and not any(r in {"plan","revised_plan","forecast"} for r,j in roles))
        if actual_plan_comparison:
            role="actual"
        # Sections describing the reporting company's operations can repeat
        # an annual fact. Keep the owning headings as evidence; specific
        # scope/entity/segment qualifiers prevent this generic scope claim.
        operation_context = [j for j in context if re.fullmatch(
            r"(?:tinh hinh|danh gia ket qua|ket qua) (?:hoat dong )?(?:san xuat )?kinh doanh",
            _scope_title(lines[j].text))]
        # A financial/plan subsection inherits the reporting-company owner
        # only through its explicit annual-report ancestor. Section titles
        # alone, arbitrary narrative, or a matching number cannot establish it.
        title_text = lambda j: re.sub(r"^(?:[ivxlcdm]+|[0-9]+)(?: [0-9]+)* ", "", normalize_text(lines[j].text)).rstrip(":")
        owners = [j for j in context if re.fullmatch(
            r"tinh hinh hoat dong trong nam(?: (?:19|20)\d{2})?|bao cao va danh gia cua ban (?:tong )?giam doc|danh gia cua hoi dong quan tri ve hoat dong cua cong ty",
            title_text(j))]
        financial_context = [j for j in context if re.fullmatch(
            r"tinh hinh tai chinh|ke hoach (?:san xuat )?kinh doanh(?: nam (?:19|20)\d{2})?",
            title_text(j))]
        if owners and financial_context:
            operation_context = list(dict.fromkeys(operation_context + owners + financial_context))
        qualified_context = any(re.search(
            r"\b(?:cong ty con|don vi|rieng|hop nhat|bo phan|linh vuc|khu vuc|segment|division|subsidiary)\b",
            normalize_text(lines[j].text)) for j in context)
        operation_evidence = evidence(operation_context) if operation_context and not qualified_context and not names else []
        rounding = [j for j in context if re.search(
            r"\b(?:lam tron.*gan nhat|rounded.*nearest)\b", normalize_text(lines[j].text))]
        # Only an immediately attached note can add a rounding convention.
        if end < len(lines) and lines[end].page == lines[start].page and lines[end].source_path == lines[start].source_path:
            if lines[end].text.lstrip().startswith(">") and re.search(r"\b(?:lam tron.*gan nhat|rounded.*nearest)\b", normalize_text(lines[end].text)):
                rounding.append(end)
        # Only adjacent prose before this table may supply a local year or
        # explicitly identify the actual year used as a planning baseline.
        local=[]
        for j in range(start-1,max(-1,start-4),-1):
            if lines[j].source_path!=lines[start].source_path or lines[j].page!=lines[start].page or "|" in lines[j].text or section_heading(lines[j].text):break
            if declared_unit(lines[j].text):continue
            local.append(j)
        local_period=[j for j in local if annual_statement_year(lines[j].text)]
        actual_years=[j for j in local if actual_result_years(lines[j].text)]
        semantic={"context_source_name":str(lines[start].source_path).replace('\\','/').split('/')[-1],
                  "local_period_evidence":evidence(local_period),
                  "actual_year_evidence":evidence(actual_years),
                  "nearest_rounding_evidence":evidence(rounding),
                  "reporting_company_operations_evidence":operation_evidence,
                  "header_evidence":evidence(range(start,header_end)),
                  "role":role if role!="mixed" else "", "role_years":role_years,
                  "entity_id":next(iter(names)) if len(names)==1 else "", "entity_conflict":len(names)>1,
                  "entity_evidence":evidence([j for _,j in entities]), "role_ambiguous":role=="mixed",
                  "role_evidence":evidence([j for _,j in roles]+(list(range(start,header_end)) if actual_plan_comparison else [])), "accounting_scope":scope,
                  "scope_conflict":conflict,"scope_evidence":evidence([j for _,j in scopes])}
        for j in range(start,end):annotations[j]["table_semantics"]=semantic


def annual_statement_year(text):
    plain=normalize_text(text)
    match=re.match(r"trong nam ((?:19|20)\d{2})\b",plain)
    years=set(re.findall(r"\b(?:19|20)\d{2}\b",plain))
    if (not match or years!={match[1]} or not re.search(r"\bcong ty\b",plain)
            or not re.search(r"\b(?:doanh thu|loi nhuan|ket qua|thuc hien|ke hoach)\b",plain)
            or re.search(r"\b(?:neu|gia su|quy|thang)\b",plain)):
        return ""
    return match[1]


def actual_result_years(text):
    years=set()
    for sentence in re.split(r"[.;!?]\s+",text):
        plain=normalize_text(sentence)
        if re.search(r"\b(?:neu|gia su|du kien|du bao|uoc tinh)\b",plain):continue
        years.update(re.findall(r"\bket qua (?:hoat dong )?(?:san xuat )?kinh doanh (?:cua )?nam ((?:19|20)\d{2})\b",plain))
    return years


def bound_column_context(candidate):
    """Validate source-only context; never infer a year from a question."""
    semantic=getattr(candidate,'table_semantics',None) or {}
    column=getattr(candidate,'column_label',getattr(candidate,'column',''))
    period=getattr(candidate,'period','')
    headers=semantic.get('header_evidence',[])
    if semantic.get('context_source_name')!=candidate.source_name or not headers:return None,None
    if any(w.get('sha256')!=line_digest(w.get('text','')) or w.get('page')!=candidate.page or w.get('line_number',0)>candidate.line_number for w in headers):return None,None
    if not any(normalize_text(column) in normalize_text(w['text']) for w in headers):return None,None
    first=min(w['line_number'] for w in headers)
    def valid(w):
        return w.get('sha256')==line_digest(w.get('text','')) and w.get('page')==candidate.page and 0<first-w.get('line_number',0)<=3
    year=None;role=None
    label=normalize_text(column)
    label=re.sub(r"\s+(?:(?:dvt|don vi(?: tinh)?)\s+)?(?:(?:nghin|trieu|ty)\s+)?(?:dong|vnd|usd|eur|jpy|gbp)$",'',label)
    if re.fullmatch(r"(?:ke hoach|kh|thuc hien|th|actual|plan|du kien|du bao|forecast|ke hoach dieu chinh)",label):
        records=semantic.get('local_period_evidence',[])
        values={annual_statement_year(w['text']) for w in records if valid(w)}
        if records and all(valid(w) for w in records) and len(values)==1 and '' not in values:
            observed=next(iter(values))
            if period in ['',observed]:year=observed
    if annual_data_column(column,period):
        records=semantic.get('actual_year_evidence',[])
        if records and all(valid(w) for w in records) and any(period in actual_result_years(w['text']) for w in records):role='actual'
    return year,role
