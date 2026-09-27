"""Source-only numeric conventions with explicit ambiguity and witnesses."""
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from .ontology import normalize_text
from .physical_coordinates import line_digest

FORMATS = {"decimal_comma": (",", "."), "decimal_dot": (".", ",")}
CURRENCY = re.compile(r"(?:\b(?:ty|trieu|nghin|tr)\s*\.?\s*(?:dong|d)\b|\b(?:vnd|vndong|dong)\b)")


def alternatives(raw):
    text=raw.strip().replace("\u00a0", " ").replace("−", "-")
    negative=text.startswith("(") and ")" in text
    text=re.sub(r"(?i)phần\s*trăm|%", "", text).strip(" ()+")
    text=text.replace(" ", "").replace("'", "").replace("’", "")
    output={}
    for name,(decimal,group) in FORMATS.items():
        if text.count(decimal)>1:continue
        whole,sep,fraction=text.partition(decimal)
        sign="-" if whole.startswith("-") else ""
        integer=whole.lstrip("+-")
        if group in integer:
            if not re.fullmatch(r"[1-9]\d{0,2}(?:"+re.escape(group)+r"\d{3})+",integer):continue
        elif not integer.isdigit():continue
        if sep and (not fraction.isdigit() or group in fraction):continue
        try:
            value=Decimal(sign+integer.replace(group,"")+("."+fraction if sep else ""))
            if negative:value=-value
        except InvalidOperation:continue
        output[name]={"value":str(value),"quantum":str(Decimal(10)**(-len(fraction) if sep else 0))}
    # Multiple grouped triplets followed by a different separator and another
    # triplet may be OCR-mixed grouping, not an intrinsically certain decimal.
    # Preserve both hypotheses. A source convention/anchor must resolve them.
    if re.fullmatch(r"[+-]?[1-9]\d{0,2}(?:,\d{3}){2,}\.\d{3}|[+-]?[1-9]\d{0,2}(?:\.\d{3}){2,},\d{3}", text):
        grouped = Decimal(text.replace(',', '').replace('.', ''))
        if negative: grouped = -grouped
        output['mixed_grouping'] = {'value': str(grouped), 'quantum': '1'}
    return output


def resolve(raw, profile=None):
    options=alternatives(raw)
    values={Decimal(x["value"]) for x in options.values()}
    base={"raw":raw,"alternatives":options,"evidence":[],"basis":"intrinsic_token"}
    if len(values)==1:
        item=next(iter(options.values()))
        return {**base,**item,"state":"VALID"}
    if not options:return {**base,"state":"INVALID","basis":"malformed_number"}
    profile=profile or {}
    name=profile.get("format")
    if name in options and profile.get("evidence") and not profile.get("conflict"):
        return {**base,**options[name],"state":"VALID","basis":profile["basis"],
                "format":name,"evidence":profile["evidence"]}
    return {**base,"state":"UNKNOWN","basis":"conflicting_conventions" if profile.get("conflict") else "ambiguous_separator"}


def _profile(records,basis):
    evidence=[r for r in records if len(r["alternatives"])==1]
    names={next(iter(r["alternatives"])) for r in evidence}
    return {"format":next(iter(names)) if len(names)==1 else "", "conflict":len(names)>1,
            "basis":basis,"evidence":[r["witness"] for r in evidence[:2]]}


def annotate(lines, blocks, annotations):
    """Attach per-token profiles before retrieval; never read question/gold."""
    from .program_data import _NUMBER_RE, _iter_number_matches
    records=[];by_block={};by_source={}
    for bi,block in enumerate(blocks):
        bi = block.get('number_group', ('block', bi))
        start,end=block["start"],block["end"]
        table_unit=annotations[start].get("unit","")
        if not table_unit:
            # A unit in a numbered caption is still an explicit unit.
            for j in range(max(0,start-2),start):
                if lines[j].page==lines[start].page and re.match(r"^(?:don vi|dvt)",normalize_text(lines[j].text)):
                    table_unit=lines[j].text
        for j in range(start,end):
            line=lines[j];bounds=[-1]+[m.start() for m in re.finditer(r"\|",line.text)]+[len(line.text)]
            parts=[line.text[a+1:b].strip() for a,b in zip(bounds,bounds[1:])]
            amount=bool(CURRENCY.search(normalize_text(" ".join(parts)+" "+table_unit)))
            ratio=any(normalize_text(x) in {"lan","times"} or x=="%" for x in parts)
            header_end=next((h for h in range(start,end) if re.fullmatch(r"[| :\-]+",lines[h].text)),start+1)
            headers=[lines[h].text.split("|") for h in range(start,header_end)]
            for match in _iter_number_matches(line.text):
                col=next((i for i,(a,b) in enumerate(zip(bounds,bounds[1:])) if a<match.start()<b),None)
                if col is None:continue
                prefix=line.text[bounds[col]+1:match.start()]
                suffix=line.text[match.end():bounds[col+1]].strip()
                if prefix.rsplit(":",1)[-1].strip() or suffix:continue
                raw=match.group(0).strip();options=alternatives(raw)
                header=" ".join(h[col] for h in headers if col<len(h))
                ratio_column="%" in header or bool(re.search(r"ty le|ty suat|muc do hoan thanh|tang giam|tang truong",normalize_text(header)))
                count_column=bool(re.search(r"\b(?:so luong|so co phan|so co phieu|number of|shares|headcount|count|stt|so thu tu)\b",normalize_text(header)))
                family="count" if count_column else "ratio" if ratio or "%" in raw or ratio_column else "currency" if amount else "other"
                witness={"source_name":str(line.source_path).replace("\\","/").split("/")[-1],"page":line.page,
                    "line_number":line.line_number,"text":line.text,"sha256":line_digest(line.text),
                    "span":[match.start(),match.end()],"raw":raw,"column":col}
                record={"block":bi,"line":j,"column":col,"family":family,"raw":raw,"alternatives":options,"witness":witness}
                records.append(record);by_block.setdefault((bi,family),[]).append(record)
                if family=="currency":by_source.setdefault(str(line.source_path),[]).append(record)
    # Source fallback only from unambiguous monetary tokens, not arbitrary
    # percentages, dates or text language. Mixed source conventions do not vote.
    for li,line in enumerate(lines):
        if "|" in line.text:continue
        for match in _NUMBER_RE.finditer(line.text):
            if not CURRENCY.match(normalize_text(line.text[match.end():match.end()+35])):continue
            raw=match.group(0).strip();options=alternatives(raw)
            witness={"source_name":str(line.source_path).replace("\\","/").split("/")[-1],"page":line.page,
                "line_number":line.line_number,"text":line.text,"sha256":line_digest(line.text),"span":[match.start(),match.end()],"raw":raw}
            if len(options)==1:
                by_source.setdefault(str(line.source_path),[]).append({"alternatives":options,"witness":witness})
            from .report_context import undated_monetary_field
            if undated_monetary_field(line.text,*match.span()):continue
            bi=("narrative",str(line.source_path),line.page,tuple(annotations[li].get("source_section_path",())))
            record={"block":bi,"line":li,"column":0,"family":"currency","raw":raw,"alternatives":options,"witness":witness}
            records.append(record);by_block.setdefault((bi,"currency"),[]).append(record)
    for r in records:
        group=by_block[(r["block"],r["family"])]
        col=_profile([x for x in group if x["column"]==r["column"]],"column_convention")
        table=_profile(group,"table_convention")
        source=_profile(by_source.get(str(lines[r["line"]].source_path),[]),"source_monetary_convention") if r["family"]=="currency" else {}
        profile=col if col["evidence"] else table if table["evidence"] else source
        # Both separator styles among ambiguous amounts in the same table
        # cannot be resolved merely by a convention elsewhere in the report.
        ambiguous_marks={mark for x in group if len({v['value'] for v in x['alternatives'].values()})>1 for mark in ",." if mark in x['raw']}
        if not col["evidence"] and not table["evidence"] and len(ambiguous_marks)>1:
            profile={"conflict":True}
        # A short decimal such as 19.1 does not establish whether 2.439
        # is a three-place decimal or a grouped integer in a mixed-style table.
        mark=re.fullmatch(r"[()+-]*[1-9]\d*[.,]\d{3}[)%]*",r["raw"])
        decimal=FORMATS.get(profile.get("format"),(None,None))[0]
        if mark and decimal and decimal in r["raw"]:
            strong=any(len(set(w["raw"]) & set(",."))>1 or w["raw"].count(",")>1 or w["raw"].count(".")>1
                       or re.search(r"[.,]\d{3}(?:[%)]|$)",w["raw"]) for w in profile.get("evidence",[]))
            if not strong:profile={}
        result=resolve(r["raw"],profile)
        result["span"]=r["witness"]["span"]
        annotations[r["line"]].setdefault("number_formats",[]).append(result)


def for_candidate(raw, start, records):
    hits=[r for r in records if r["raw"]==raw and r["span"][0]==start]
    return hits[0] if len(hits)==1 else resolve(raw)


def local_records(text,source_name,page,line_start):
    from types import SimpleNamespace
    lines=[SimpleNamespace(text=t,source_path=source_name,page=page,line_number=line_start+i) for i,t in enumerate(text.splitlines())]
    annotations={i:{"unit":""} for i in range(len(lines))};blocks=[];i=0
    while i<len(lines):
        if "|" not in lines[i].text:i+=1;continue
        start=i
        while i<len(lines) and "|" in lines[i].text:i+=1
        blocks.append({"start":start,"end":i})
    annotate(lines,blocks,annotations)
    return annotations

def printed_quantum(cell, scale):
    raw=getattr(cell,"source_numeric_text","") or getattr(cell,"raw","")
    values={Fraction(v["quantum"])*Fraction(scale) for v in alternatives(raw).values()
            if Decimal(v["value"])==Decimal(str(cell.value))}
    return next(iter(values)) if len(values)==1 else None


def nearest_printed(value, quantum):
    quotient=Fraction(value)/Fraction(quantum)
    n,d=abs(quotient.numerator),quotient.denominator
    return (-1 if quotient<0 else 1)*((2*n+d)//(2*d))*Fraction(quantum)


def resolution_state(cell):
    raw=getattr(cell,"source_numeric_text","") or getattr(cell,"raw","")
    if not raw:return "VALID"  # legacy synthetic cells have no format claim
    record=getattr(cell,"number_format",None) or resolve(raw)
    if record.get("state")!="VALID":return record.get("state","UNKNOWN")
    if record.get("raw")!=raw:return "INVALID"
    options=alternatives(raw)
    try:
        actual=Decimal(str(cell.value))
        if Decimal(record["value"])!=actual:return "INVALID"
    except (KeyError,InvalidOperation):return "INVALID"
    if len({Decimal(v["value"]) for v in options.values()})==1:
        return "VALID" if actual==Decimal(next(iter(options.values()))["value"]) else "INVALID"
    name=record.get("format")
    if record.get("basis") in {"same_fact_exact_rendering", "same_fact_display_precision", "same_fact_coarse_reference"}:
        from types import SimpleNamespace
        from .program_data import currency_unit_scale
        support=record.get("support",{})
        other=SimpleNamespace(**support)
        if support.get("number_format",{}).get("basis") in {"same_fact_exact_rendering", "same_fact_display_precision", "same_fact_coarse_reference"}:return "INVALID"
        if not fact_key(cell) or resolution_state(other)!="VALID" or fact_key(other)!=fact_key(cell):return "INVALID"
        if (getattr(cell,"page",None),getattr(cell,"line_number",None))==(other.page,other.line_number):return "INVALID"
        left,right=currency_unit_scale(cell.unit),currency_unit_scale(other.unit)
        if left is None or right is None:return "INVALID"
        observed=Fraction(actual)*Fraction(left);anchor=Fraction(Decimal(str(other.value)))*Fraction(right)
        if record["basis"]=="same_fact_exact_rendering":
            if observed!=anchor:return "INVALID"
        elif record["basis"]=="same_fact_coarse_reference":
            fine=printed_quantum(cell,left);coarse=printed_quantum(other,right)
            if fine is None or coarse is None or not fine<coarse or nearest_printed(observed,coarse)!=anchor:return "INVALID"
            matches=[item for item in options.values() if Fraction(item['quantum'])*Fraction(left)<coarse
                     and nearest_printed(Fraction(item['value'])*Fraction(left),coarse)==anchor]
            if len(matches)!=1:return "INVALID"
        else:
            coarse=printed_quantum(cell,left);fine=printed_quantum(other,right)
            if coarse is None or fine is None or not fine<coarse or nearest_printed(anchor,coarse)!=observed:return "INVALID"
            matches=[item for item in options.values() if fine<Fraction(item['quantum'])*Fraction(left)
                     and nearest_printed(anchor,Fraction(item['quantum'])*Fraction(left))==Fraction(item['value'])*Fraction(left)]
            if len(matches)!=1:return "INVALID"
        return "VALID" if name in options and Decimal(options[name]["value"])==actual else "INVALID"
    if name not in options or Decimal(options[name]["value"])!=actual:return "INVALID"
    witnesses=record.get("evidence",[])
    if not witnesses:return "UNKNOWN"
    for w in witnesses:
        if w.get("sha256")!=line_digest(w.get("text","")) or w.get("source_name")!=cell.source_name:return "INVALID"
        span=w.get("span",[])
        if len(span)!=2 or w["text"][span[0]:span[1]].strip()!=w.get("raw"):return "INVALID"
        if set(alternatives(w["raw"]))!={name}:return "INVALID"
    return "VALID"

def fact_key(cell):
    from .report_context import report_field_matches
    if report_field_matches(cell):
        from .metric_semantics import report_field_identity
        semantic=getattr(cell,"table_semantics",None) or {}
        if semantic.get("scope_conflict") or semantic.get("entity_conflict"):return None
        return (cell.source_name,"report_field",report_field_identity(getattr(cell,"metric_label",getattr(cell,"row_label",""))),
                cell.report_field["report_year"],cell.entity_id,getattr(cell,"entity",""),semantic.get("accounting_scope",""))
    from .metric_semantics import identity
    from .disclosure_equivalence import _scope_key
    if not getattr(cell,"source_section_path",()) or _scope_key(cell) is None:return None
    role=getattr(cell,"column_role","")
    period=getattr(cell,"period","")
    if role not in {"actual","plan","revised_plan","forecast"} or not re.fullmatch(r"(?:19|20)\d{2}",period):return None
    return (cell.source_name,identity(getattr(cell,"metric_label",getattr(cell,"row_label",""))),period,role,cell.entity_id,_scope_key(cell))


def reconcile_exact_repetitions(candidates):
    """An independently identified finer fact can resolve one numeric interpretation.

    One hop, exact/printed-quantum compatibility, no candidate answer. All independently resolved
    witnesses in the group must agree. The selected operand is not changed by
    the verifier; this runs while source candidates are being constructed.
    """
    from dataclasses import replace
    from .evidence_schema import evidence_cells
    from .program_data import currency_unit_scale
    cells=evidence_cells(candidates);known={}
    for c in cells:
        if fact_key(c) and resolution_state(c)=="VALID" and currency_unit_scale(c.unit) is not None:
            known.setdefault(fact_key(c),[]).append(c)
    output=[]
    for original,c in zip(candidates,cells):
        witnesses=known.get(fact_key(c),[])
        scale=currency_unit_scale(c.unit)
        if resolution_state(c)!="UNKNOWN" or scale is None or not witnesses:
            output.append(original);continue
        values={Fraction(w.value)*Fraction(currency_unit_scale(w.unit)) for w in witnesses}
        if len(values)!=1:output.append(original);continue
        goal=next(iter(values));opts=alternatives(c.source_numeric_text)
        matched=[(name,item) for name,item in opts.items() if Fraction(Decimal(item['value']))*Fraction(scale)==goal]
        basis='same_fact_exact_rendering'
        if not matched:
            # Identity was established without numeric agreement. Permit only
            # one interpretation compatible with a strictly finer source fact.
            fine_quantums=[printed_quantum(w,currency_unit_scale(w.unit)) for w in witnesses]
            if any(q is None for q in fine_quantums):output.append(original);continue
            fine=max(fine_quantums)
            matched=[(name,item) for name,item in opts.items()
                     if fine<Fraction(item['quantum'])*Fraction(scale)
                     and nearest_printed(goal,Fraction(item['quantum'])*Fraction(scale))==Fraction(item['value'])*Fraction(scale)]
            basis='same_fact_display_precision'
            if not matched:
                # A source-bound coarser rendering may distinguish exactly one
                # finer interpretation. Identity is established BEFORE values;
                # neither the question nor an answer reference participates.
                coarse=min(fine_quantums)
                matched=[(name,item) for name,item in opts.items()
                         if Fraction(item['quantum'])*Fraction(scale)<coarse
                         and nearest_printed(Fraction(item['value'])*Fraction(scale),coarse)==goal]
                basis='same_fact_coarse_reference'
        if len(matched)!=1:output.append(original);continue
        source=next((w for w in witnesses if (w.page,w.line_number)!=(c.page,c.line_number)),None)
        if source is None:output.append(original);continue
        name,item=matched[0]
        fields=('source_name','source_numeric_text','value','number_format','unit','metric_label','period','column_role','entity_id','entity','section','source_section_path','table_semantics','page','line_number','report_field','source_span','source_span_scope','source_line_sha256','unit_assumption')
        support={k:getattr(source,k,None) for k in fields};support['value']=str(support['value'])
        fmt={**resolve(c.source_numeric_text),**item,'state':'VALID','format':name,'basis':basis,'support':support}
        output.append(replace(original,value=str(Decimal(item['value'])),number_format=fmt))
    return output
