"""Recover an omitted table unit from another MinerU rendering of the same page.

The caller must bind both renderings to the same PDF hash and physical page.
Only unique, exact whole-table matches qualify; values and existing units never
change. There is no question, answer, company identifier or locale guess here.
"""
from collections import Counter
from hashlib import sha256
from .source_bindings import declared_unit


def _blocks(lines):
    result=[];i=0
    while i<len(lines):
        if '|' not in lines[i] or declared_unit(lines[i]):i+=1;continue
        start=i
        while i<len(lines) and '|' in lines[i] and not declared_unit(lines[i]):i+=1
        signature=tuple(x.strip() for x in lines[start:i])
        if len(signature)>=2:result.append((start,i,signature))
    return result


def recover_table_units(original, refreshed):
    """Return original lines plus witnessed unit captions, and an audit trail."""
    old=list(original);new=list(refreshed);before=_blocks(old);after=_blocks(new)
    old_counts=Counter(x[2] for x in before);new_counts=Counter(x[2] for x in after)
    matches={sig:(start,end) for start,end,sig in after};insertions={};audit=[]
    for start,end,sig in before:
        if old_counts[sig]!=1 or new_counts[sig]!=1:continue
        ns,ne=matches[sig]
        captions=[]
        if ns>0 and declared_unit(new[ns-1]) and not new[ns-1].lstrip().startswith('>'):
            captions.append((ns-1,new[ns-1]))
        if ne<len(new) and new[ne].lstrip().startswith('>') and declared_unit(new[ne]):
            captions.append((ne,new[ne]))
        units=set().union(*(declared_unit(text) for _,text in captions)) if captions else set()
        if len(units)!=1:continue
        existing=set()
        if start>0 and not old[start-1].lstrip().startswith('>'):existing|=declared_unit(old[start-1])
        if end<len(old) and old[end].lstrip().startswith('>'):existing|=declared_unit(old[end])
        record={'old_table_start':start,'refreshed_table_start':ns,'table_sha256':sha256('\n'.join(sig).encode()).hexdigest(),
                'unit':next(iter(units)),'refreshed_unit_line':captions[0][0],'refreshed_unit_text':captions[0][1]}
        if existing:
            record['status']='ALREADY_PRESENT' if existing==units else 'EXISTING_UNIT_CONFLICT'
        else:
            record['status']='RECOVERED_FROM_EXACT_TABLE';insertions[start]=captions[0][1].lstrip('> ').strip()
        audit.append(record)
    output=[]
    for i,line in enumerate(old):
        if i in insertions:output.append(insertions[i])
        output.append(line)
    return output,audit
