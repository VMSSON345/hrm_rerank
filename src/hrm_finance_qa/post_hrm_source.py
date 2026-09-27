"""Source-bound fact construction without the legacy symbolic graph/verifier."""
from collections import defaultdict
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
import json,re
from .post_hrm_contract import SourceFact
from .evidence_schema import evidence_cells
from .metric_semantics import identity,normalized_label
from .ontology import normalize_text
from .number_formats import alternatives
from .program_data import currency_unit_scale,_explicit_currency_units
from .column_context import resolved_relative_year
from .unit_defaults import monetary_label

_EXTRA={
 'literal:tai san ngan han':('tai san ngan han','current assets'),
 'total_assets':('tong cong tai san','tong tai san'),
 'owners_equity':('von chu so huu',),
 'total_liabilities':('no phai tra','tong no phai tra','total liabilities'),
 'gross_revenue':('doanh thu ban hang va cung cap dich vu','doanh thu gop','gross revenue'),
 'revenue_deductions':('cac khoan giam tru doanh thu','khoan giam tru doanh thu'),
 'current_income_tax':('chi phi thue tndn hien hanh','chi phi thue thu nhap doanh nghiep hien hanh'),
 'deferred_income_tax':('chi phi thue tndn hoan lai','chi phi thue thu nhap doanh nghiep hoan lai'),
 'weighted_average_common_shares':('so co phieu pho thong binh quan gia quyen','weighted average common shares'),
 'earnings_attributable_to_common':('loi nhuan phan bo cho co dong pho thong','earnings attributable to common'),
 'price_per_share':('gia co phieu','gia thi truong moi co phieu','price per share'),
 'earnings_per_share':('eps','lai co ban tren co phieu','earnings per share'),
}
def _ocr_distance(left, right):
 previous=list(range(len(right)+1))
 for i,char in enumerate(left,1):
  current=[i]
  for j,target in enumerate(right,1):
   current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(char!=target)))
  previous=current
 return previous[-1]


def _ocr_metric_cell(snapshot,c,cell,line):
 """A fuzzy label is only a proposal; require an aligned, witnessed table row."""
 from dataclasses import replace
 from .metric_semantics import ALIASES
 from .program_data import _markdown_cells
 if not metric_key(cell.metric_label).startswith('literal:') or '|' not in line:
  return cell
 semantic=cell.table_semantics or {}
 if semantic.get('scope_conflict') or semantic.get('role_ambiguous'):
  return cell
 headers=semantic.get('header_evidence',[])
 if not headers or not all(snapshot.witness_valid(w,c.source_name) and
                            w['line_number']<c.line_number for w in headers):
  return cell
 fields=_markdown_cells(line)
 index=line[:c.source_span[0]].count('|')-int(line.lstrip().startswith('|'))
 if index<1 or index>=len(fields) or not cell.column:
  return cell
 dated_column=bool(re.search(
     r'(?:19|20)\d{2}|nam nay|nam truoc|cuoi nam|dau nam|cui nam|du nam',
     normalize_text(cell.column)))
 observed=normalized_label(cell.metric_label)
 ranked=[]
 for key,aliases in ALIASES.items():
  if key in {'revenue_unspecified','total_revenue'}:
   continue
  distance,alias=min((_ocr_distance(observed,alias),alias) for alias in aliases)
  ranked.append((distance,key,alias))
 ranked.sort()
 if not ranked:
  return cell
 best,key,alias=ranked[0]
 if best==0 or best>min(6,max(2,len(alias)//4)) or (
     len(ranked)>1 and ranked[1][0]-best<2):
  return cell
 # A printed accounting code, when present, has priority over lexical
 # similarity. Glued multi-code rows cannot identify a unique fact.
 codes={'current_assets':'100','total_assets':'270',
        'total_liabilities':'300','current_liabilities':'310',
        'owners_equity':'400','net_revenue':'10',
        'profit_after_tax':'60','profit_before_tax':'50'}
 # A printed code in the first table cell must agree with the OCR label.
 leading_code=re.fullmatch(r'(?:ma so\s*:?\s*)?(\d{2,3})',
                            normalize_text(fields[0])) if fields else None
 if leading_code and leading_code[1].lstrip('0')!=codes.get(key,'').lstrip('0'):
  return cell
 if not dated_column:
  # A unique printed code immediately after the OCR label proves only the
  # metric. It does not grant a period, unit, or scope.
  found_codes=[(j,m[1]) for j,part in enumerate(fields)
               if (m:=re.fullmatch(r'(?:ma so )?(\d{2,3})',
                                    normalize_text(part)))]
  if (len(found_codes)!=1 or found_codes[0][1]!=codes.get(key)
      or found_codes[0][0]<1 or found_codes[0][0]>=index
      or normalized_label(fields[found_codes[0][0]-1])!=observed):
   return cell
 if len(fields)>2 and re.search(r'\bma\s*(?:so|s[oó])\b',
                               normalize_text(fields[1])):
  found=set(re.findall(r'(?<!\d)\d{2,3}(?!\d)',fields[1]))
  expected=codes.get(key)
  if len(found)!=1 or not expected or next(iter(found)).lstrip('0')!=expected.lstrip('0'):
   return cell
 witness={'source_name':c.source_name,'line_number':c.line_number,'page':c.page,
          'text':line,'sha256':sha256(line.encode()).hexdigest()}
 if not snapshot.witness_valid(witness,c.source_name):
  return cell
 return replace(cell,metric_label=alias,table_semantics={
     **semantic,'ocr_metric_evidence':[witness,*headers],
     'ocr_metric_proposals':[{'metric':item[1],'distance':item[0]}
                             for item in ranked[:3]]})


def metric_key(label):
 raw=re.sub(r'^\s*[+\-•]\s*','',label)
 prefix=re.split(r'[:(]',raw,1)[0].strip()
 prefix_identity=identity(prefix)
 if prefix_identity in {'current_ratio','quick_ratio','return_on_assets','return_on_equity'} and (':' in raw or '(' in raw):
  return prefix_identity
 normalized=normalized_label(label)
 acronym=re.search(r"\b(?:ROE|ROA)\b",raw,re.I)
 if acronym and re.search(r"\b(?:hệ số|tỷ suất|lợi nhuận)\b",raw,re.I):
  return {'roe':'return_on_equity','roa':'return_on_assets'}[acronym[0].lower()]
 if normalized.startswith('vong quay tong tai san') and '/' in raw:
  return 'total_asset_turnover'
 # MinerU can glue a printed formula to a turnover label. Keep the
 # numerator/denominator words as evidence before removing that suffix.
 if (normalized.startswith('vong quay hang ton kho') and '/' in raw
     and 'gia von' in normalized and 'hang ton kho binh quan' in normalized):
  return 'inventory_turnover'
 if (normalized.startswith('he so thanh toan ngan han') and '/' in raw
     and 'tai san ngan han' in normalized and 'no ngan han' in normalized):
  return 'current_ratio'
 return next((key for key,labels in _EXTRA.items() if normalized in labels),identity(label))

class SourceSnapshot:
 def __init__(self,paths):
  self.files={};self.lines={};self.pages={};self.hashes={};self.pdf_sources={};self._pdf_word_cache={};self.period_conflicts={}
  for raw in paths:
   p=Path(raw)
   if p.suffix.lower() not in {'.md','.txt'}:raise ValueError('Prepare source with MinerU before validation')
   if p.name in self.files:raise ValueError('Ambiguous source file name')
   data=p.read_bytes();self.files[p.name]=p;self.hashes[p.name]=sha256(data).hexdigest();self.lines[p.name]=data.decode('utf-8-sig').splitlines();page=1;pages={}
   from .native_pdf_unit import verified_companion_pdf
   companion=verified_companion_pdf(p,self.hashes[p.name])
   if companion:self.pdf_sources[p.name]=companion
   for n,line in enumerate(self.lines[p.name],1):
    m=re.match(r'^## Page (\d+)',line)
    if m:page=int(m[1])
    pages[n]=page
   self.pages[p.name]=pages
   prior=None;conflicts=[]
   for n,line in enumerate(self.lines[p.name],1):
    normal=normalize_text(line)
    if '|' in line or 'bang can doi ke toan' not in normal:continue
    date_match=re.search(r'(?<!\d)\d{1,2}[/.]\d{1,2}[/.]((?:19|20)\d{2})(?!\d)',line)
    if not date_match:continue
    current={'source_name':p.name,'line_number':n,'page':pages[n],
             'text':line,'sha256':sha256(line.encode()).hexdigest(),
             'year':date_match[1]}
    if (prior and re.search(r'\b(?:tiep|tip) theo\b',normal)
        and pages[n]-prior['page']<=1 and current['year']!=prior['year']):
     conflicts.append({'previous':prior,'continuation':current,
                       'reason':'continuation_title_year_conflict'})
    prior=current
   self.period_conflicts[p.name]=tuple(conflicts)
 def unchanged(self):return (all(sha256(p.read_bytes()).hexdigest()==self.hashes[n] for n,p in self.files.items())
  and all(sha256(path.read_bytes()).hexdigest()==digest for path,digest in self.pdf_sources.values()))
 def bound_line(self,c):
  if c.source_name not in self.lines or c.source_span_scope!='line_v2':raise ValueError('source_not_bound')
  lines=self.lines[c.source_name]
  if not isinstance(c.line_number,int) or not 1<=c.line_number<=len(lines):raise ValueError('source_line_missing')
  raw=lines[c.line_number-1];variants=(raw,re.sub(r'\s+',' ',raw).strip());line=next((s for s in variants if sha256(s.encode()).hexdigest()==c.source_line_sha256),None)
  if line is None:raise ValueError('source_hash_mismatch')
  span=c.source_span
  if not span or not 0<=span[0]<span[1]<=len(line):raise ValueError('source_span_invalid')
  if c.page is not None and self.pages[c.source_name][c.line_number]!=c.page:raise ValueError('source_page_mismatch')
  fragment=line[span[0]:span[1]];raw=fragment.strip()
  from .program_data import _iter_number_matches
  def trimmed(a,b):
   chunk=line[a:b];return a+len(chunk)-len(chunk.lstrip()),b-len(chunk)+len(chunk.rstrip())
  if not any(trimmed(*m.span())==trimmed(*span) for m in _iter_number_matches(line)):
   raise ValueError('source_span_not_whole_number')
  return line,raw
 def witness_valid(self,w,name):
  if not isinstance(w,dict) or w.get('source_name',name)!=name:return False
  n=w.get('line_number');text=w.get('text')
  if not isinstance(n,int) or not isinstance(text,str) or not 1<=n<=len(self.lines[name]):return False
  raw=self.lines[name][n-1]
  if text not in (raw,re.sub(r'\s+',' ',raw).strip()):return False
  if w.get('sha256') and sha256(text.encode()).hexdigest()!=w['sha256']:return False
  return w.get('page',self.pages[name][n])==self.pages[name][n]

def period_basis(text):
 normal=normalize_text(text)
 quarter=re.search(r'\bquy\s*(iv|iii|ii|i|[1-4])\b',normal)
 if quarter:return 'quarter_'+{'i':'1','ii':'2','iii':'3','iv':'4'}.get(quarter[1],quarter[1])
 if re.search(r'\b6 thang|ban nien',normal):return 'six_months'
 if re.search(r'\b9 thang',normal):return 'nine_months'
 return ''

def _narrative_endpoint_witness(snapshot,c,metric,raw,period):
 """Corroborate a table endpoint by an independent same-value disclosure."""
 if not period or not re.fullmatch(r'(?:19|20)\d{2}',period):
  return None
 from .metric_semantics import ALIASES
 if metric not in ALIASES:return None
 witnesses=[]
 for n,line in enumerate(snapshot.lines[c.source_name],1):
  if n==c.line_number or '|' in line or not line.strip():continue
  normal=normalize_text(line)
  endpoint=re.search(r'\b(?:cuoi|cui) nam '+re.escape(period)+r'\b',normal)
  if not endpoint or re.search(r'\b(?:ke hoach|du kien|du bao)\b',normal):
   continue
  matches=list(re.finditer(r'(?<![\d.,])'+re.escape(raw)+r'(?![\d.,])',line))
  if len(matches)!=1:continue
  prefix=re.sub(r'\b(?:cua|ca) cong ty$', '', normal[:endpoint.start()].strip()).strip()
  if not prefix:continue
  ranked=[]
  for key,aliases in ALIASES.items():
   distance,alias=min((_ocr_distance(prefix,alias),alias) for alias in aliases)
   ranked.append((distance,key,alias))
  ranked.sort()
  best,key,alias=ranked[0]
  if (key!=metric or best>min(6,max(2,len(alias)//4))
      or (len(ranked)>1 and ranked[1][0]-best<2)):
   continue
  witness={'source_name':c.source_name,'line_number':n,
           'page':snapshot.pages[c.source_name][n],'text':line,
           'sha256':sha256(line.encode()).hexdigest(),
           'basis':'equivalent_narrative_endpoint','metric':metric,
           'year':period,'raw_value':raw}
  if snapshot.witness_valid(witness,c.source_name):
   witnesses.append(witness)
 return witnesses[0] if len(witnesses)==1 else None


def _table_key(cell):return '|'.join((cell.source_name,str(cell.page),cell.table_id,sha256('||'.join(cell.raw_header).encode()).hexdigest()[:12]))
def _unit(text):
 text=re.sub(r'(?<=\d)(?=(?:VND|VNĐ|đồng)\b)',' ',text,flags=re.I)
 units=_explicit_currency_units(text);scales={currency_unit_scale(u):u for u in units if currency_unit_scale(u) is not None}
 if len(scales)>1:raise ValueError('source_unit_conflict')
 return next(iter(scales.values())) if scales else ''
def _table_unit_witnesses(snapshot,c,cell=None):
 """Only explicit declarations governing the contiguous table, not value matches."""
 lines=snapshot.lines[c.source_name];at=c.line_number-1
 if '|' not in lines[at]:return []
 start=end=at
 while start>0 and '|' in lines[start-1]:start-=1
 while end+1<len(lines) and '|' in lines[end+1]:end+=1
 found=[]
 if cell:
  for h in (cell.table_semantics or {}).get('header_evidence',[]):
   if snapshot.witness_valid(h,c.source_name) and h['line_number']<c.line_number:
    for i in range(h['line_number']-2,max(-1,h['line_number']-6),-1):
     t=lines[i].strip()
     if not t:continue
     if t.startswith('## Page') or '|' in t:break
     # A standalone denomination immediately before this witnessed header
     # belongs to this table only, not to the entire report.
     if re.fullmatch(r'(?:(?:ty|ti|trieu|nghin|ngan)\s+)?(?:dong|vnd)',
                     normalize_text(t)) and _unit(t) and all(
                         not lines[j].strip() for j in range(i+1,h['line_number']-1)):
      found.append({'source_name':c.source_name,'line_number':i+1,
                    'page':snapshot.pages[c.source_name][i+1],'text':lines[i]});break
     if re.match(r'^(?:don v(?:i)?(?: tinh)?|dvt|units?)\b',normalize_text(t)) and _unit(t):
      found.append({'source_name':c.source_name,'line_number':i+1,'page':snapshot.pages[c.source_name][i+1],'text':lines[i]});break
 for positions in (range(start-1,max(-1,start-7),-1),range(end+1,min(len(lines),end+5))):
  for i in positions:
   text=lines[i].strip();normal=normalize_text(text)
   if not text:continue
   if text.startswith('## Page') or '|' in text:break
   declaration=re.search(r'\b(?:don v(?:i)?(?: tinh)?|dvt|unit(?:s)?)\b',normal)
   if declaration and _unit(text):
    found.append({'source_name':c.source_name,'line_number':i+1,'page':snapshot.pages[c.source_name][i+1],'text':lines[i]})
   # A following prose passage or a new section is not a table denomination.
   if i>end and not declaration:break
 return found

def _source_unit(snapshot,c,cell,line):
 if cell.is_percent or '%' in cell.column:return '%','',Decimal('.01'),''
 if '|' in line:
  from .program_data import _markdown_cells,_aligned_table_header_cells
  row=_markdown_cells(line)
  value_index=line[:c.source_span[0]].count('|')-int(line.lstrip().startswith('|'))
  for witness in (cell.table_semantics or {}).get('header_evidence',[]):
   if not snapshot.witness_valid(witness,c.source_name):
    raise ValueError('source_unit_header_witness_mismatch')
   headings=_aligned_table_header_cells(_markdown_cells(witness['text']),len(row))
   for unit_index,heading in enumerate(headings):
    if (unit_index<value_index and 0<=unit_index<len(row)
        and normalize_text(heading) in {'dvt','don vi','don vi tinh','unit','units'}
        and row[unit_index].strip()=='%'):
     return '%','',Decimal('.01'),''
 if metric_key(cell.metric_label)=='weighted_average_common_shares':return 'shares','',Decimal(1),''
 if metric_key(cell.metric_label) in {'earnings_per_share','price_per_share'}:
  explicit=_unit(line)
  return ('VND/share','VND',Decimal(1),'') if explicit and currency_unit_scale(explicit)==1 else ('','VND',Decimal(1),'source_unit_unresolved')
 start,end=c.source_span;segment=line[line.rfind('|',0,start)+1:].split('|',1)[0]
 suffix=line[end:].split('|',1)[0]
 token=re.match(r'\s*(?:(?:triệu|tỷ|tỉ|nghìn|ngàn)\s+)?(?:đồng|VND|VNĐ)\b',suffix,re.I)
 explicit=_unit(token[0]) if token else ''
 if explicit:return explicit,'VND',currency_unit_scale(explicit),''
 label_unit=_unit(cell.metric_label)
 if label_unit:return label_unit,'VND',currency_unit_scale(label_unit),''
 # Specific row/column declarations precede a table-wide denomination.
 # Read the column from physical header witnesses, not from another column.
 from .program_data import _markdown_cells,_aligned_table_header_cells
 row_cells=_markdown_cells(line);column_index=line[:c.source_span[0]].count('|')-int(line.lstrip().startswith('|'))
 row_units=[_unit(part.strip()) for part in row_cells if re.fullmatch(r'(?:(?:ty|ti|trieu|nghin|ngan|tr)\s*)?(?:dong|vnd|d)',normalize_text(part.strip()))]
 row_units=[u for u in row_units if u]
 column_units=[];column_texts=[]
 for w in (cell.table_semantics or {}).get('header_evidence',[]):
  if not snapshot.witness_valid(w,c.source_name):raise ValueError('source_unit_header_witness_mismatch')
  if w['line_number']>=c.line_number:continue
  headings=_aligned_table_header_cells(_markdown_cells(w['text']),len(row_cells))
  if 0<=column_index<len(headings):
   column_texts.append(headings[column_index]);u=_unit(headings[column_index])
   if u:column_units.append(u)
 foreign_texts=[part for part in row_cells if re.fullmatch(r'(?:(?:trieu|ty|nghin|million|billion|thousand) )?(?:usd|eur|jpy|gbp|cny)',normalize_text(part))]+column_texts
 foreign={currency.upper() for text in foreign_texts for currency in re.findall(r'\b(?:usd|eur|jpy|gbp|cny)\b',normalize_text(text))}
 if foreign:
  if len(foreign)>1 or row_units or column_units:raise ValueError('source_specific_currency_conflict')
  if any(re.search(r'\b(?:trieu|ty|nghin|million|billion|thousand)\b',normalize_text(t)) for t in foreign_texts):raise ValueError('source_foreign_scale_unresolved')
  currency=next(iter(foreign));return currency,currency,Decimal(1),''
 # A bare VND column identifies currency; an explicit scale elsewhere remains
 # authoritative, matching the existing table-scale convention.
 specific=[*row_units,*(u for u in column_units if currency_unit_scale(u)!=Decimal(1))]
 if len({currency_unit_scale(u) for u in specific})>1:raise ValueError('source_specific_unit_conflict')
 if specific:return specific[0],'VND',currency_unit_scale(specific[0]),''
 marker=re.search(r'(?m)^UNIT_SOURCE: (.+)$',c.context);witnesses=json.loads(marker[1]) if marker else [];witnessed=[]
 for w in [*witnesses,*_table_unit_witnesses(snapshot,c,cell)]:
  if snapshot.witness_valid(w,c.source_name) :
   u=_unit(w['text'])
   if u:witnessed.append(u)
 lines=snapshot.lines[c.source_name];at=c.line_number-1
 while at>0 and '|' in lines[at-1] and not lines[at-1].startswith('## Page'):at-=1
 for pos in range(at-1,-1,-1):
  text=lines[pos].strip()
  if not text:continue
  if re.match(r'^(?:don v(?:i)?(?: tinh)?|dvt|units?)\b',normalize_text(text)):
   u=_unit(text)
   if u:witnessed.append(u)
   break
  if '|' in text or text.startswith('## Page'):break

 if len({currency_unit_scale(u) for u in witnessed})>1:raise ValueError('source_unit_conflict')
 if witnessed:return witnessed[0],'VND',currency_unit_scale(witnessed[0]),''
 if column_units:return column_units[0],'VND',currency_unit_scale(column_units[0]),''
 if monetary_label(cell.metric_label) or metric_key(cell.metric_label) in {'total_assets','owners_equity'}:
  from .native_pdf_unit import native_column_unit
  witness=native_column_unit(snapshot,c.source_name,c.page,cell.column,
                             line[c.source_span[0]:c.source_span[1]].strip())
  if witness:
   return witness['unit'],'VND',currency_unit_scale(witness['unit']),''
  foreign=re.search(r'\b(?:USD|EUR|JPY|GBP|CNY)\b',segment+' '+cell.column)
  if foreign:return foreign[0],foreign[0],Decimal(1),''
  return '','VND',Decimal(1),'source_unit_unresolved'
 return c.unit or 'ratio','',Decimal(1),''


def _bound_period(snapshot,c,cell,line):
 """Check year and actual/plan against the physical column; never the query."""
 semantic=cell.table_semantics or {}
 if semantic.get('context_source_name',c.source_name)!=c.source_name:
  raise ValueError('source_context_document_mismatch')
 for key in ('header_evidence','role_evidence','scope_evidence','actual_year_evidence','local_period_evidence','layout_evidence'):
  if any(not snapshot.witness_valid(w,c.source_name) for w in semantic.get(key,[])):
   raise ValueError('source_context_witness_mismatch')
 if semantic.get('layout_basis')=='source_narrative_local':
  from .source_layout_bindings import narrative_binding
  bound=narrative_binding(snapshot,c)
  if not bound or bound['year']!=cell.period or metric_key(bound['label'])!=metric_key(cell.metric_label):raise ValueError('source_narrative_replay_mismatch')
  return bound['year']
 headers=semantic.get('header_evidence',[])
 column_parts=[]
 if '|' in line:
  from .program_data import _markdown_cells
  column_index=line[:c.source_span[0]].count('|')-int(line.lstrip().startswith('|'))
  for witness in headers:
   if witness['line_number']>c.line_number:raise ValueError('source_header_after_value')
   if witness['line_number']==c.line_number:continue
   values=_markdown_cells(witness['text'])
   if semantic.get('numbered_continuation'):
    from .program_data import _aligned_table_header_cells
    values=_aligned_table_header_cells(values,len(_markdown_cells(line)))
   if 0<=column_index<len(values):column_parts.append(values[column_index])
  inline=line[:c.source_span[0]].rsplit('|',1)[-1]
  if ':' in inline:column_parts.append(inline)
 else:column_parts.append(line[:c.source_span[0]])
 source_column=normalize_text(' '.join(column_parts))
 years=set(re.findall(r'(?:19|20)\d{2}',source_column))
 period=resolved_relative_year(cell) or cell.period
 anchor=semantic.get('statement_period') or {}
 if anchor.get('basis') in {'adjacent_balance_title_date','adjacent_balance_title_year','numbered_balance_continuation'} and snapshot.witness_valid(anchor,c.source_name):
  if re.fullmatch(r'(?:so )?(?:cuoi|dau) (?:nam|ky)',normalize_text(cell.column)):period=anchor['year']
 if anchor.get('basis') in {'layout_scalar_date','source_narrative_local'} and snapshot.witness_valid(anchor,c.source_name):period=anchor['year']
 if period and not re.fullmatch(r'(?:19|20)\d{2}',period):
  parsed=set(re.findall(r'(?:19|20)\d{2}',period));period=next(iter(parsed)) if len(parsed)==1 else ''
 if len(years)==1:
  witnessed=next(iter(years))
  if period and period!=witnessed:raise ValueError('source_column_year_mismatch')
  period=witnessed
 if not period and re.fullmatch(r'(?:nam )?(?:truoc|nay)',normalize_text(cell.column)):
  local_years=set()
  for w in semantic.get('role_evidence',[]):
   if snapshot.witness_valid(w,c.source_name):
    local_years.update(re.findall(r'(?:19|20)\d{2}',w['text']))
  if len(local_years)==1:
   current=int(next(iter(local_years)));period=str(current-int('truoc' in normalize_text(cell.column)))
 if (headers and period and not years and anchor.get('basis')!='numbered_balance_continuation'
     and not re.search(r'truoc|nay|dau|cuoi',source_column)):
  # A merged header may carry its year in a parent cell. The parser must retain
  # a physically checked header path; a free-standing candidate year is insufficient.
  header_years=set(re.findall(r'(?:19|20)\d{2}',' '.join(w['text'] for w in headers)))
  local=semantic.get('local_period_evidence',[]) if semantic.get('period_origin')=='local_table_context' else []
  local_years=set(re.findall(r'(?:19|20)\d{2}',' '.join(w['text'] for w in local
      if w.get('line_number',0)<min(h['line_number'] for h in headers)
      and w.get('page')==c.page and snapshot.witness_valid(w,c.source_name))))
  if period not in header_years and (len(local_years)!=1 or period not in local_years):
   raise ValueError('source_column_year_unbound')
 plan=bool(re.search(r'ke hoach|\b(?:kh|plan|budget|target)\b',source_column));actual=bool(re.search(r'thuc hien|\b(?:th|actual)\b',source_column))
 if not cell.is_percent:
  if plan and not actual and cell.column_role=='actual':raise ValueError('source_actual_plan_mismatch')
  if actual and not plan and cell.column_role=='plan':raise ValueError('source_actual_plan_mismatch')
 if metric_key(cell.metric_label)=='price_per_share':
  match=re.search(r'(?<!\d)(\d{1,2})[/.](\d{1,2})[/.]((?:19|20)\d{2})(?!\d)',' '.join(column_parts))
  if match:
   from datetime import date
   period=date(int(match[3]),int(match[2]),int(match[1])).isoformat()
 return period


def _unit_evidence(snapshot,c,cell,line,unit):
 """Return only physical source lines that justify this cell's denomination."""
 if not unit or unit in {'%','shares'}:return ()
 from .program_data import _markdown_cells,_aligned_table_header_cells
 target=currency_unit_scale(unit)
 def matches(text):
  if target is not None:
   normalized=re.sub(r'(?<=\d)(?=(?:VND|VNĐ|đồng)\b)',' ',text,flags=re.I)
   try:return any(currency_unit_scale(u)==target for u in _explicit_currency_units(normalized))
   except ValueError:return False
  return bool(re.search(r'\b'+re.escape(unit)+r'\b',text,re.I))
 def record(n,basis):
  raw=snapshot.lines[c.source_name][n-1]
  return {'source_name':c.source_name,'line_number':n,'page':snapshot.pages[c.source_name][n],
   'text':raw,'source_line_sha256':sha256(raw.encode()).hexdigest(),'basis':basis}
 found={}
 parts=_markdown_cells(line) if '|' in line else []
 index=line[:c.source_span[0]].count('|')-int(line.lstrip().startswith('|')) if parts else -1
 selected=parts[index] if 0<=index<len(parts) else line
 direct_suffix=re.match(r'\s*(?:(?:triệu|tỷ|tỉ|nghìn|ngàn)\s+)?(?:đồng|VND|VNĐ)\b',line[c.source_span[1]:],re.I)
 if matches(cell.metric_label) or matches(selected) or (direct_suffix and matches(direct_suffix[0])) or any(
   matches(part) and re.fullmatch(r'(?:(?:ty|ti|trieu|nghin|ngan|tr)\s*)?(?:dong|vnd|d|usd|eur|jpy|gbp|cny)',normalize_text(part))
   for part in parts):
  found[c.line_number]=record(c.line_number,'row_or_cell')
 if parts:
  for w in (cell.table_semantics or {}).get('header_evidence',[]):
   if not snapshot.witness_valid(w,c.source_name) or w['line_number']>=c.line_number:continue
   headings=_aligned_table_header_cells(_markdown_cells(w['text']),len(parts))
   if 0<=index<len(headings) and matches(headings[index]):
    found[w['line_number']]=record(w['line_number'],'column_header')
 for w in _table_unit_witnesses(snapshot,c,cell):
  if snapshot.witness_valid(w,c.source_name) and matches(w['text']):
   found[w['line_number']]=record(w['line_number'],'table_declaration')
 marker=re.search(r'(?m)^UNIT_SOURCE: (.+)$',c.context)
 for w in json.loads(marker[1]) if marker else []:
  if snapshot.witness_valid(w,c.source_name) and matches(w['text']):
   found[w['line_number']]=record(w['line_number'],'source_declaration')
 from .native_pdf_unit import native_column_unit
 native=native_column_unit(snapshot,c.source_name,c.page,cell.column,
                           line[c.source_span[0]:c.source_span[1]].strip())
 if native and native['unit']==unit:
  return tuple(found[n] for n in sorted(found))+(native,)
 return tuple(found[n] for n in sorted(found))


def _local_actual_table_witness(snapshot,c,cell):
 """Use an adjacent financial-results heading to identify actual columns, not dates."""
 semantic=cell.table_semantics or {}
 headers=semantic.get('header_evidence',[])
 if not headers or semantic.get('role_ambiguous') or semantic.get('scope_conflict'):return None
 if not all(snapshot.witness_valid(w,c.source_name) for w in headers):return None
 first=min(w['line_number'] for w in headers)
 if first>=c.line_number:return None
 lines=snapshot.lines[c.source_name]
 near=[]
 for pos in range(first-2,max(-1,first-6),-1):
  raw=lines[pos]
  if raw.startswith('## Page') or '|' in raw or snapshot.pages[c.source_name][pos+1]!=c.page:break
  if raw.strip():near.append((pos+1,raw))
 if not near:near=[]
 # A formula-labelled ratio can belong to a financial-analysis section that
 # spans several adjacent tables. The explicit section is still required.
 # Do not grant that wider ownership to generic margin or amount rows.
 if "/" in cell.metric_label and "(" in cell.metric_label:
  for pos in range(first-2,max(-1,first-22),-1):
   raw=lines[pos]
   if raw.startswith('## Page') or snapshot.pages[c.source_name][pos+1]!=c.page:break
   normal=normalize_text(raw)
   if '|' not in raw and re.search(r'\b(?:ke hoach|du kien|du bao|forecast|budget|plan)\b',normal):break
   if '|' not in raw and re.fullmatch(r'(?:phan tich tai chinh(?: tiep)?|financial analysis(?: continued)?)',normal):
    near.append((pos+1,raw));break
 for n,raw in near:
  normal=normalize_text(raw)
  if re.search(r'\b(?:ke hoach|du kien|du bao|forecast|budget|plan)\b',normal):return None
  if re.fullmatch(r'(?:(?:[0-9]+|[ivxlcdm]+)[/. ]+)?(?:a[)/. ]+)?(?:tinh hinh tai chinh|tinh hinh tai san|phan tich tai chinh(?: tiep)?|financial analysis(?: continued)?|bao cao ket qua hoat dong kinh doanh|bao cao luu chuyen tien te|cash flow statement|income statement|financial performance|financial position|financial results)[:.]?',normal):
   witness={'source_name':c.source_name,'line_number':n,'page':snapshot.pages[c.source_name][n],
    'text':raw,'sha256':sha256(raw.encode()).hexdigest()}
   return witness if snapshot.witness_valid(witness,c.source_name) else None
 return None


def _physical_column_role(snapshot,c,cell,line):
 """Prefer a witnessed column result or adjacent audited-table note over a remote plan section."""
 if '|' not in line:return cell
 semantic=cell.table_semantics or {}
 headers=semantic.get('header_evidence',[])
 if not headers:return cell
 if cell.column_role=='actual_to_plan_ratio' and cell.is_percent:
  if any(not snapshot.witness_valid(w,c.source_name) for w in headers):
   raise ValueError('source_role_header_witness_mismatch')
  return cell
 if any(not snapshot.witness_valid(w,c.source_name) for w in headers):
  raise ValueError('source_role_header_witness_mismatch')
 from .program_data import _markdown_cells,_aligned_table_header_cells
 from dataclasses import replace
 row=_markdown_cells(line)
 index=line[:c.source_span[0]].count('|')-int(line.lstrip().startswith('|'))
 declarations=[]
 for w in headers:
  if w['line_number']>=c.line_number:continue
  aligned=_aligned_table_header_cells(_markdown_cells(w['text']),len(row))
  if not 0<=index<len(aligned):continue
  normal=normalize_text(aligned[index])
  if re.search(r'\b(?:ke hoach|plan|budget|du kien|du bao|forecast)\b',normal):
   declarations.append(('plan',w))
  elif re.search(r'\b(?:chi tieu|muc tieu)\b.*\bthong qua\b',normal) and any(
    j!=index and re.search(r'\b(?:th|thuc hien|actual|ket qua|result)\b',normalize_text(heading))
    for j,heading in enumerate(aligned)):
   # An approved target paired with an explicit actual-result column is a plan.
   declarations.append(('plan',w))
  if re.search(r'\b(?:ket qua|thuc hien|actual|realized|realised|result)\b',normal):
   declarations.append(('actual',w))
 roles={role for role,_ in declarations}
 if len(roles)>1:raise ValueError('source_specific_role_conflict')
 if declarations:
  role,witness=declarations[-1]
 else:
  first=min(w['line_number'] for w in headers)
  witness=None;role=''
  for pos in range(first-2,max(-1,first-6),-1):
   raw=snapshot.lines[c.source_name][pos]
   if raw.startswith('## Page') or '|' in raw or snapshot.pages[c.source_name][pos+1]!=c.page:break
   normal=normalize_text(raw)
   if not normal:continue
   if re.match(r'^(?:dvt|don v(?:i)?(?: tinh)?|units?)\b',normal):continue
   if re.search(r'\b(?:nguon|source)\b',normal) and (
     re.search(r'bao cao tai chinh.*kiem toan',normal)
     or re.search(r'audited financial (?:statement|report)',normal)):
    role='actual'
    witness={'source_name':c.source_name,'line_number':pos+1,
     'page':snapshot.pages[c.source_name][pos+1],'text':raw,'sha256':sha256(raw.encode()).hexdigest()}
   break
 if not role or not witness or not snapshot.witness_valid(witness,c.source_name):return cell
 return replace(cell,column_role=role,table_semantics={
  **semantic,'role':role,'role_evidence':[*semantic.get('role_evidence',[]),witness]})

def _source_context(snapshot,c,cell,line,unit):
 semantic=cell.table_semantics or {};witnesses=[{'source_name':c.source_name,'line_number':c.line_number,'page':c.page,'text':line}]
 for key in ('header_evidence','role_evidence','scope_evidence','actual_year_evidence','local_period_evidence','layout_evidence','ocr_metric_evidence'):
  witnesses.extend(semantic.get(key,[]))
 anchor=semantic.get('statement_period') or {}
 if anchor:witnesses.extend([anchor,*anchor.get('headers',[])])
 witnesses.extend((c.report_context or {}).get('title_evidence',[]))
 witnesses.extend((cell.number_format or {}).get('evidence',[]))
 marker=re.search(r'(?m)^UNIT_SOURCE: (.+)$',c.context)
 if marker:witnesses.extend(json.loads(marker[1]))
 witnesses.extend(_table_unit_witnesses(snapshot,c,cell))
 lines=snapshot.lines[c.source_name];at=c.line_number-1
 while at>0 and '|' in lines[at-1]:at-=1
 for pos in range(at-1,-1,-1):
  text=lines[pos].strip()
  if '|' in text or text.startswith('## Page'):break
  if re.match(r'^(?:don v(?:i)?(?: tinh)?|dvt|units?)\b',normalize_text(text)):
   if _unit(text)==unit:witnesses.append({'text':lines[pos],'line_number':pos+1,'page':snapshot.pages[c.source_name][pos+1]})
   break
 result={}
 for w in [*witnesses,*_table_unit_witnesses(snapshot,c,cell)]:
  if snapshot.witness_valid(w,c.source_name):
   result[w['line_number']]={'source_name':c.source_name,'line_number':w['line_number'],'page':snapshot.pages[c.source_name][w['line_number']],
      'text':w['text'],'source_line_sha256':sha256(w['text'].encode()).hexdigest()}
 return tuple(result[n] for n in sorted(result))

def _local_balance_context(snapshot, candidate, cell):
 """Bind a split balance-sheet title/date immediately above its own header."""
 from dataclasses import replace
 semantic=cell.table_semantics or {};headers=semantic.get('header_evidence',[])
 from .column_context import _statement_witness
 statement=_statement_witness(cell)
 # A dated balance-sheet title immediately above this header governs its
 # columns. A remote prior-year audit paragraph cannot override that title.
 if (headers and not semantic.get('scope_conflict')
     and re.fullmatch(r'(?:so )?(?:cuoi|dau) (?:nam|ky)',normalize_text(cell.column))
     and all(snapshot.witness_valid(w,candidate.source_name) for w in headers)):
  first=min(w['line_number'] for w in headers)
  lines=snapshot.lines[candidate.source_name]
  for n in range(first-1,max(0,first-6),-1):
   if snapshot.pages[candidate.source_name][n]!=candidate.page:break
   raw=lines[n-1]
   if '|' in raw:continue
   match=re.fullmatch(
    r'bang can doi ke toan(?: hop nhat| rieng le)?(?: tiep theo)? (?:vao|tai) ngay '
    r'(\d{1,2}) (?:thang )?(\d{1,2}) nam ((?:19|20)\d{2})',
    normalize_text(raw))
   if not match:continue
   from datetime import date as calendar_date
   from .table_semantics import accounting_scope
   try:closing=calendar_date(int(match[3]),int(match[2]),int(match[1])).isoformat()
   except ValueError:return cell
   year=match[3];column=normalize_text(cell.column)

   if (cell.period and cell.period!=year and
       not re.fullmatch(r'(?:so )?(?:cuoi|dau) (?:nam|ky)',normalize_text(cell.period))):
    return cell
   if (statement and statement.get('year')!=year
       and 'bang can doi ke toan' in normalize_text(statement.get('text',''))):
    return cell
   scope=accounting_scope(raw)
   if semantic.get('accounting_scope') and scope and scope!=semantic['accounting_scope']:
    return cell
   title={'source_name':candidate.source_name,'line_number':n,'page':candidate.page,
          'text':raw,'sha256':sha256(raw.encode()).hexdigest()}
   if not snapshot.witness_valid(title,candidate.source_name):return cell
   anchor={**title,'year':year,'as_of_date':closing,
           'headers':headers,'basis':'adjacent_balance_title_date'}
   return replace(cell,period=year,column_role='actual',
       table_semantics={**semantic,'statement_period':anchor,'role':'actual',
           'role_evidence':[*semantic.get('role_evidence',[]),title],
           'accounting_scope':scope or semantic.get('accounting_scope',''),
           'scope_evidence':[*semantic.get('scope_evidence',[]),title]})
 if (statement and snapshot.witness_valid(statement,candidate.source_name)
     and all(snapshot.witness_valid(w,candidate.source_name) for w in statement.get('headers',[]))
     and re.match(r'^(?:so )?(?:dau|cuoi) (?:nam|ky)(?: trinh bay lai)?$',normalize_text(cell.column))
     and not semantic.get('scope_conflict') and not semantic.get('role_ambiguous')):
  return replace(cell,period=statement['year'],column_role='actual',table_semantics={**semantic,'role':'actual','role_evidence':[*semantic.get('role_evidence',[]),statement],'statement_period':{**statement,'basis':'audited_statement_period'}})
 if not headers or semantic.get('statement_period') or semantic.get('scope_conflict') or semantic.get('role_ambiguous'):return cell
 if not all(snapshot.witness_valid(w,candidate.source_name) for w in headers):return cell
 first=min(w['line_number'] for w in headers);lines=snapshot.lines[candidate.source_name]
 prior=[(j+1,lines[j]) for j in range(max(0,first-5),first-1) if lines[j].strip()]
 prior=[(n,t) for n,t in prior if not re.match(r'^(?:don v(?:i)?(?: tinh)?|dvt|units?)\b',normalize_text(t))]
 # A year in the immediately governing balance-sheet title is a period,
 # not proof of a calendar closing date. Trusted continuation headers may
 # point to the previous page; scope remains anchored to that header.
 if prior:
  tn,title=prior[-1];normal=normalize_text(title)
  if re.fullmatch(r'bang can doi ke toan(?: hop nhat| rieng(?: le)?)?',normal) and re.search(r'\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}',cell.column) and snapshot.pages[candidate.source_name][tn]==headers[0].get('page'):
   from .table_semantics import accounting_scope
   scope=accounting_scope(title)
   if semantic.get('accounting_scope') and scope and scope!=semantic['accounting_scope']:return cell
   w={'source_name':candidate.source_name,'line_number':tn,'page':snapshot.pages[candidate.source_name][tn],'text':title,'sha256':sha256(title.encode()).hexdigest()}
   return replace(cell,table_semantics={**semantic,'role':semantic.get('role') or 'actual','role_evidence':[*semantic.get('role_evidence',[]),w],'accounting_scope':scope or semantic.get('accounting_scope',''),'scope_evidence':[*semantic.get('scope_evidence',[]),w]})
  match=re.fullmatch(r'(?:[a-z] )?bang can doi ke toan(?: hop nhat| rieng(?: le)?)? nam ((?:19|20)\d{2})',normal)
  if match and snapshot.pages[candidate.source_name][tn]==headers[0].get('page') and re.search(r'\b(?:dau|cuoi) (?:nam|ky)\b',normalize_text(cell.column)):
   from .table_semantics import accounting_scope
   scope=accounting_scope(title)
   if semantic.get('accounting_scope') and scope and scope!=semantic['accounting_scope']:return cell
   w={'source_name':candidate.source_name,'line_number':tn,'page':snapshot.pages[candidate.source_name][tn],'text':title,'sha256':sha256(title.encode()).hexdigest()}
   return replace(cell,table_semantics={**semantic,'statement_period':{**w,'year':match[1],'headers':headers,'basis':'adjacent_balance_title_year'},'role':semantic.get('role') or 'actual','role_evidence':[*semantic.get('role_evidence',[]),w],'accounting_scope':scope or semantic.get('accounting_scope',''),'scope_evidence':[*semantic.get('scope_evidence',[]),w]})
 if len(prior)<2:return cell
 (tn,title),(dn,date)=prior[-2:]
 if snapshot.pages[candidate.source_name][tn]!=headers[0].get('page') or snapshot.pages[candidate.source_name][dn]!=headers[0].get('page'):return cell
 normal=normalize_text(title)
 if not re.fullmatch(r'bang can doi ke toan(?: hop nhat| rieng(?: le)?)?',normal):return cell
 match=re.fullmatch(r'(?:(?:vao|tai) )?(?:ngay )?(\d{1,2}) (?:thang )?(\d{1,2}) (?:nam )?((?:19|20)\d{2})',normalize_text(date))
 if not match:return cell
 from datetime import date as calendar_date
 try:closing_date=calendar_date(int(match[3]),int(match[2]),int(match[1])).isoformat()
 except ValueError:return cell
 if closing_date[5:]!='12-31' and any('cuoi nam' in normalize_text(w['text']) for w in headers):return cell
 column_time=normalize_text(cell.column)
 endpoint_label=re.fullmatch(r'(?:so )?(?:cuoi|dau) (?:nam|ky)',column_time)
 dated_current_year=(closing_date.endswith('-12-31') and column_time==f'nam {match[3]}')
 if not endpoint_label and not dated_current_year:return cell
 from .table_semantics import accounting_scope
 scope=accounting_scope(title)
 if normal.endswith((' rieng',' rieng le')):scope='separate'
 if semantic.get('accounting_scope') and scope!=semantic['accounting_scope']:return cell
 def witness(n,t):return {'source_name':candidate.source_name,'line_number':n,'page':snapshot.pages[candidate.source_name][n],'text':t,'sha256':sha256(t.encode()).hexdigest()}
 tw,dw=witness(tn,title),witness(dn,date)
 anchor={**dw,'year':match[3],'as_of_date':closing_date,'headers':headers,'basis':'adjacent_balance_title_date'}
 semantic={**semantic,'statement_period':anchor,'role':semantic.get('role') or 'actual',
           'role_evidence':[*semantic.get('role_evidence',[]),tw,dw],
           'accounting_scope':scope or semantic.get('accounting_scope',''),
           'scope_evidence':[*semantic.get('scope_evidence',[]),tw]}
 return replace(cell,table_semantics=semantic)

def _continued_balance_cell(snapshot,c,cell):
 """Rebind coded balance rows to their witnessed five-column table header."""
 from dataclasses import replace
 from .program_data import _markdown_cells
 if not hasattr(snapshot,'balance_rows'):
  snapshot.balance_rows={}
  for name,lines in snapshot.lines.items():
   pending=None;active=None;last=0;last_page=0
   def witness(n):
    return {'source_name':name,'line_number':n,'page':snapshot.pages[name][n],'text':lines[n-1],'sha256':sha256(lines[n-1].encode()).hexdigest()}
   for n,line in enumerate(lines,1):
    norm=normalize_text(line)
    title=re.fullmatch(r'(?:[a-z] )?bang can doi ke toan(?: (hop nhat|rieng le))? nam ((?:19|20)\d{2})',norm)
    dated=re.fullmatch(r'bang can doi ke toan(?: (hop nhat|rieng le))? tai ngay \d{1,2} \d{1,2} ((?:19|20)\d{2})',norm)
    if dated:title=dated
    if 'bang can doi ke toan' in norm and '|' not in line:
     active=None;pending=(n,title) if title else None;last=0
    fields=_markdown_cells(line) if '|' in line else []
    if (pending and n-pending[0]<=4 and len(fields)==5
        and (('chi tieu' in normalize_text(fields[0])) or normalize_text(fields[0]) in {'tai san','taisan','nguon von'})
        and 'ma' in normalize_text(fields[1])
        and re.search(r'cuoi|cui',normalize_text(fields[3]))
        and re.search(r'dau|du nam',normalize_text(fields[4]))):
     tn,t=pending;active=(witness(tn),witness(n),fields,t[2],{'hop nhat':'consolidated','rieng le':'separate'}.get(t[1],''));last_page=snapshot.pages[name][n];pending=None;continue
    if not active or len(fields)!=5:continue
    code=re.fullmatch(r'(?:ma so )?([1-4]\d{2})(?:[a-z])?',normalize_text(fields[1]),re.I)
    if not code:continue
    number=int(code[1]);page=snapshot.pages[name][n]
    if number<last or page-last_page>2:continue
    if not re.search(r'\d',fields[3]+fields[4]):continue
    snapshot.balance_rows[(name,n)]=active;last=number;last_page=page
    if number==440:active=None
 record=snapshot.balance_rows.get((c.source_name,c.line_number))
 if not record:return cell
 line=snapshot.lines[c.source_name][c.line_number-1];fields=_markdown_cells(line)
 column=line[:c.source_span[0]].count('|')-int(line.lstrip().startswith('|'))
 if column not in {3,4}:return cell
 title,header,labels,year,scope=record
 # Explicit contradictory cell years/scopes must not be overwritten.
 if (cell.period and cell.period!=year
     and not re.fullmatch(r'(?:so )?(?:cuoi|dau) (?:nam|ky)',normalize_text(cell.period))):
  return cell
 existing_scope=(cell.table_semantics or {}).get('accounting_scope','')
 if existing_scope and scope and existing_scope!=scope:return cell
 semantic={'header_evidence':[header],'role':'actual','role_evidence':[title],
           'accounting_scope':scope,'scope_evidence':[title],
           'statement_period':{**title,'year':year,'headers':[header],'basis':'numbered_balance_continuation'}}
 return replace(cell,metric_label=fields[0],column=labels[column],period=year,
     period_role='closing' if column==3 else 'opening',column_role='actual',column_role_conflict=False,
     source_section_path=(title['text'],),table_semantics=semantic)

def _plan_sentence(snapshot,c):
 if c.source_name not in snapshot.lines or not 1<=c.line_number<=len(snapshot.lines[c.source_name]):return None
 line=snapshot.lines[c.source_name][c.line_number-1]
 if '|' in line:return None
 prefix=line[:c.source_span[0]]
 match=re.fullmatch(r'\s*\\?[-*•]?\s*Kế hoạch (?P<metric>.+?)(?: của Toàn công ty)? năm (?P<year>(?:19|20)\d{2})\s*:\s*',prefix,re.I)
 return match if match and metric_key(match['metric']) in {'profit_before_tax','profit_after_tax','net_revenue','total_revenue'} else None


def _plan_sentence_cell(snapshot,c,cell):
 from dataclasses import replace
 m=_plan_sentence(snapshot,c)
 if not m:return cell
 line=snapshot.lines[c.source_name][c.line_number-1]
 w={'source_name':c.source_name,'line_number':c.line_number,'page':c.page,'text':line,'sha256':sha256(line.encode()).hexdigest()}
 semantic={'role':'plan','role_evidence':[w],'statement_period':{**w,'year':m['year'],'headers':[],'basis':'source_plan_sentence'}}
 return replace(cell,metric_label=m['metric'],column='',period=m['year'],column_role='plan',column_role_conflict=False,table_semantics=semantic)

def _narrative_financial_cell(snapshot,c,cell):
 from dataclasses import replace
 if c.source_name not in snapshot.lines or not 1<=c.line_number<=len(snapshot.lines[c.source_name]):return cell
 line=snapshot.lines[c.source_name][c.line_number-1]
 if '|' in line:return cell
 prefix=line[:c.source_span[0]];suffix=line[c.source_span[1]:]
 metric=re.match(r'^\s*(?P<label>(?:Tổng\s+)?lợi nhuận (?:trước|sau) thuế)\s+(?:đạt|là)\s*$',prefix,re.I)
 role='';year='';column=''
 if metric:
  prior=re.search(r'so với thực hiện năm\s+((?:19|20)\d{2})\b',suffix,re.I)
  plan=re.search(r'so với kế hoạch năm\s+((?:19|20)\d{2})\b',suffix,re.I)
  if prior and plan and int(plan[1])==int(prior[1])+1:
   role='actual';year=plan[1];column='Năm '+year
 else:
  metric=re.match(r'^\s*(?P<label>(?:Tổng\s+)?lợi nhuận (?:trước|sau) thuế)\b',line,re.I)
  actual=re.search(r'thực hiện năm\s+((?:19|20)\d{2})\s+đạt\s+[\d.,]+\s*(?:nghìn|triệu|tỷ)?\s*(?:đồng|VND|VNĐ)?\s*\(đạt\s*$',prefix,re.I)
  target=re.match(r'\s*so với kế hoạch năm\s+((?:19|20)\d{2})\)\s*[.]?\s*$',suffix,re.I)
  if metric and actual and target and actual[1]==target[1] and c.raw.endswith('%'):
   role='actual_to_plan_ratio';year=actual[1];column='% hoàn thành kế hoạch'
 if not role:return cell
 witness={'source_name':c.source_name,'line_number':c.line_number,'page':c.page,
          'text':line,'sha256':sha256(line.encode()).hexdigest()}
 if not snapshot.witness_valid(witness,c.source_name):return cell
 semantic={**(cell.table_semantics or {}),'role':role,'role_evidence':[witness],
           'layout_basis':'same_sentence_financial_claim',
           'statement_period':{**witness,'year':year,'headers':[],'basis':'same_sentence_comparison'}}
 return replace(cell,metric_label=metric['label'],column=column,period=year,
                column_role=role,column_role_conflict=False,table_semantics=semantic)
def _layout_cell(snapshot,c,cell):
 from dataclasses import replace
 from .source_layout_bindings import binding,date_text
 b=binding(snapshot,c)
 if not b:return cell
 if cell.period and b['year'] and cell.period!=b['year'] and b['basis']!='source_narrative_local':return cell
 semantic={**(cell.table_semantics or {}),'header_evidence':[b['header']] if b['header'] else [],
           'role':b['role'],'role_evidence':b['witnesses'],'layout_evidence':b['witnesses'],'layout_basis':b['basis']}
 if b['basis']=='source_narrative_local':
  semantic['accounting_scope']=b.get('scope','')
 if not b['table']:
  anchor=next((w for w in b['witnesses'] if w['line_number']!=c.line_number and b['date'] and date_text(w['text'])==b['date']),b['witnesses'][0])
  semantic['statement_period']={**anchor,'year':b['year'],'as_of_date':b['date'],'headers':[],'basis':b['basis']}
 else:semantic.pop('statement_period',None)
 return replace(cell,metric_label=b['label'],column=b['column'],period=b['year'],period_role=b['period_role'],column_role=b['role'],column_role_conflict=False,table_semantics=semantic,**({'report_field':None,'source_assertion':None,'is_percent':False} if b['basis']=='source_narrative_local' else {}))

def build_source_facts(candidates,snapshot):
 cells=tuple(_layout_cell(snapshot,c,_narrative_financial_cell(snapshot,c,_plan_sentence_cell(snapshot,c,_continued_balance_cell(snapshot,c,cell)))) for c,cell in zip(candidates,evidence_cells(candidates)));prepared=[];profiles=defaultdict(set);doc_profiles=defaultdict(set);errors=[]
 for candidate,cell in zip(candidates,cells):
  try:
   if not cell.metric_label or (cell.is_year_like and not monetary_label(cell.metric_label)):continue
   line,raw=snapshot.bound_line(candidate);options=alternatives(raw)
   if not options:raise ValueError('source_number_malformed')
   from .source_layout_bindings import binding as layout_binding
   lb=layout_binding(snapshot,candidate)
   if not cell.source_assertion and not cell.report_field and not (lb and metric_key(lb['label'])==metric_key(cell.metric_label)):
    labels=line[:candidate.source_span[0]].split('|')[:-1] if '|' in line else [line.split(':',1)[0]]
    if not any(metric_key(label.strip())==metric_key(cell.metric_label) for label in labels) and not (_plan_sentence(snapshot,candidate) and metric_key(_plan_sentence(snapshot,candidate)['metric'])==metric_key(cell.metric_label)) and not ((cell.table_semantics or {}).get('layout_basis')=='same_sentence_financial_claim' and any(snapshot.witness_valid(w,candidate.source_name) for w in (cell.table_semantics or {}).get('role_evidence',[]))):raise ValueError('source_metric_not_in_row')
   cell=_ocr_metric_cell(snapshot,candidate,cell,line)
   # A plan/actual table may call the printed ratio "Tỷ lệ (%)" instead
   # of "% hoàn thành kế hoạch". Bind it only when both dated columns name
   # the same year and the adjacent title explicitly compares performance
   # with plan.
   if cell.is_percent and normalize_text(cell.column) in {'ty le','ty le phan tram'}:
    semantic=cell.table_semantics or {}
    headers=semantic.get('header_evidence',[])
    titles=semantic.get('role_evidence',[])
    if (headers and all(snapshot.witness_valid(w,candidate.source_name) for w in headers)
        and any(snapshot.witness_valid(w,candidate.source_name)
                and 'thuc hien so voi ke hoach' in normalize_text(w['text']) for w in titles)):
     header_text=' '.join(w['text'] for w in headers)
     plan_years=set(re.findall(r'ke hoach\s*((?:19|20)\d{2})',normalize_text(header_text)))
     actual_years=set(re.findall(r'thuc hien\s*((?:19|20)\d{2})',normalize_text(header_text)))
     if len(plan_years)==len(actual_years)==1 and plan_years==actual_years:
      from dataclasses import replace
      cell=replace(cell,period=next(iter(plan_years)),
          column_role='actual_to_plan_ratio',column_role_conflict=False,
          table_semantics={**semantic,'period_origin':'plan_actual_header_ratio'})
   if cell.column_role_conflict and cell.is_percent and re.fullmatch(
       r'(?:% )?hoan thanh ke hoach',normalize_text(cell.column)):
    semantic=cell.table_semantics or {}
    headers=semantic.get('header_evidence',[])
    local=semantic.get('local_period_evidence',[])
    if (headers and all(snapshot.witness_valid(w,candidate.source_name) for w in headers)
        and any('ke hoach' in normalize_text(w['text']) and 'thuc hien' in normalize_text(w['text']) for w in headers)):
     local_years=set(re.findall(r'(?:19|20)\d{2}',' '.join(
         w['text'] for w in local if w.get('page')==candidate.page
         and w.get('line_number',0)<min(h['line_number'] for h in headers)
         and snapshot.witness_valid(w,candidate.source_name))))
     if len(local_years)==1:
      from dataclasses import replace
      cell=replace(cell,period=next(iter(local_years)),
                   column_role='actual_to_plan_ratio',column_role_conflict=False,
                   table_semantics={**semantic,'period_origin':'local_table_context'})
   if cell.column_role_conflict or (cell.table_semantics or {}).get('scope_conflict'):raise ValueError('source_header_conflict')
   table=_table_key(cell)+'|'+('ratio' if cell.is_percent else 'amount')
   if len(options)==1:profiles[table].update(options);doc_profiles[(cell.source_name,cell.is_percent)].update(options)
   prepared.append((candidate,cell,line,raw,options,table))
  except (ValueError,TypeError,KeyError) as e:errors.append({'candidate':candidate.index,'reason':str(e)})
 facts=[]
 for c,cell,line,raw,options,table in prepared:
  try:
   distinct={Decimal(v['value']) for v in options.values()}
   convention=next(iter(profiles[table])) if len(profiles[table])==1 else next(iter(doc_profiles[(cell.source_name,cell.is_percent)])) if not profiles[table] and len(doc_profiles[(cell.source_name,cell.is_percent)])==1 else None
   basis='intrinsic_token' if len(distinct)==1 else 'table_numeric_cells' if len(profiles[table])==1 else 'document_numeric_cells'
   number=cell.number_format or {};format_evidence=number.get('evidence',[])
   if not format_evidence and number.get('basis')=='same_fact_exact_rendering':
    support=number.get('support',{})
    if support.get('source_name')==c.source_name and metric_key(support.get('metric_label',''))==metric_key(cell.metric_label) and support.get('period')==cell.period:
     format_evidence=(support.get('number_format') or {}).get('evidence',[])
   checked_formats=set()
   for w in format_evidence:
    if not snapshot.witness_valid(w,c.source_name):raise ValueError('source_number_format_witness_mismatch')
    span=w.get('span',[]);text=w['text']
    if len(span)!=2 or not 0<=span[0]<span[1]<=len(text):raise ValueError('source_number_format_span_invalid')
    token=text[span[0]:span[1]].strip()
    if token!=w.get('raw','').strip():raise ValueError('source_number_format_token_mismatch')
    formats=alternatives(token)
    if len(formats)==1:checked_formats.update(formats)
   if len(checked_formats)==1 and number.get('format') in checked_formats and number['format'] in options:
    convention=number['format'];basis='source_bound_'+number.get('basis','format')
   if len(distinct)==1:parsed=next(iter(options.values()))
   elif convention in options:parsed=options[convention]
   else:raise ValueError('source_number_convention_missing_or_conflicting')
   cell=_physical_column_role(snapshot,c,_local_balance_context(snapshot,c,cell),line)
   unit,currency,scale,assumption=_source_unit(snapshot,c,cell,line);kind='ratio' if unit=='%' else 'money' if currency else 'quantity'
   semantic=cell.table_semantics or {};period=_bound_period(snapshot,c,cell,line)
   anchor=semantic.get('statement_period')
   if anchor and (not snapshot.witness_valid(anchor,c.source_name) or any(not snapshot.witness_valid(w,c.source_name) for w in anchor.get('headers',[]))):raise ValueError('source_statement_period_witness_mismatch')
   if metric_key(cell.metric_label)!='price_per_share' and period and not re.fullmatch(r'(?:19|20)\d{2}',period):
    years=set(re.findall(r'(?:19|20)\d{2}',period));period=next(iter(years)) if len(years)==1 else ''
   role=cell.column_role
   if not role:
    from .table_semantics import annual_data_column
    heading=normalize_text(' '.join(cell.source_section_path))
    declared=semantic.get('role','')
    if declared in {'actual','plan','forecast','revised_plan'}:role=declared
    elif not semantic.get('role_ambiguous') and annual_data_column(cell.column,period):
     local=_local_actual_table_witness(snapshot,c,cell)
     if local:
      role='actual'
      semantic={**semantic,'role':'actual','role_evidence':[*semantic.get('role_evidence',[]),local]}
      from dataclasses import replace
      cell=replace(cell,table_semantics=semantic)
     elif re.search(r'tinh hinh|ket qua|bao cao tai chinh',heading):
      role='forecast' if re.search(r'du bao|du kien|trien vong|dinh huong',heading) else 'plan' if 'ke hoach' in heading and 'thuc hien' not in heading else 'actual'
   prefix=line[:c.source_span[0]]
   is_profile=semantic.get('layout_basis')=='profile_report_title' or bool(cell.report_field) or (not cell.source_assertion and not cell.period and '|' not in line and ':' in prefix and not re.search(r'(?:19|20)\d{2}',prefix) and metric_key(prefix.split(':',1)[0])==metric_key(cell.metric_label))
   if not is_profile and not cell.period and not role and '|' in line:
    from .program_data import _markdown_cells
    fields=_markdown_cells(line)
    is_profile=(len(fields)==2 and metric_key(fields[0])==metric_key(cell.metric_label)
                and not re.search(r'\d',fields[0]))
   if is_profile:role='profile'
   elif not role and cell.source_assertion:role=cell.source_assertion.get('binding',{}).get('column_role','')
   if not role and '|' in line and re.search(r'(?<!\d)\d{1,2}[/.]\d{1,2}[/.](?:19|20)\d{2}(?!\d)',cell.column):
    preceding=[(pos+1,raw) for pos,raw in enumerate(snapshot.lines[c.source_name][:c.line_number-1])
               if snapshot.pages[c.source_name][pos+1]==c.page and '|' not in raw]
    titles=[(n,raw) for n,raw in preceding if 'bang can doi ke toan' in normalize_text(raw)]
    if titles:
     title_n,title=titles[-1]
     newer_statement=any(n>title_n and re.search(r'bao cao (?:ket qua|luu chuyen)',
                         normalize_text(raw)) for n,raw in preceding)
     if not newer_statement and not re.search(r'ke hoach|du kien|du bao',normalize_text(title)):
      proof={'source_name':c.source_name,'line_number':title_n,'page':c.page,
             'text':title,'sha256':sha256(title.encode()).hexdigest()}
      if not snapshot.witness_valid(proof,c.source_name):raise ValueError('source_balance_role_witness_mismatch')
      role='actual'
      from dataclasses import replace
      cell=replace(cell,table_semantics={**semantic,'role':'actual',
          'role_evidence':[*semantic.get('role_evidence',[]),proof]})
      semantic=cell.table_semantics
   period_binding='';local_results_witness=None
   if '|' not in line and role in {'','profile'} and not period:
    local_lines=snapshot.lines[c.source_name]
    for pos in range(c.line_number-2,max(-1,c.line_number-7),-1):
     heading=local_lines[pos]
     if snapshot.pages[c.source_name][pos+1]!=c.page or heading.startswith('## Page') or '|' in heading:break
     title=normalize_text(heading)
     found=re.fullmatch(r'(?:\d+(?: \d+)* )?ket qua (?:hoat dong )?(?:san xuat )?kinh doanh nam ((?:19|20)\d{2})',title)
     if found:
      period=found[1];role='actual';is_profile=False;period_binding='local_results_heading'
      local_results_witness={'source_name':c.source_name,'line_number':pos+1,
          'page':c.page,'text':heading,'sha256':sha256(heading.encode()).hexdigest()}
      if not snapshot.witness_valid(local_results_witness,c.source_name):
       raise ValueError('source_local_results_witness_mismatch')
      break
     if re.match(r'\d+ (?:tinh hinh|ke hoach|nhung thay doi)',title):break
   section=' '.join(cell.source_section_path);normal=normalize_text(section);statement='cash_flow' if 'luu chuyen tien' in normal else 'income_statement' if 'ket qua' in normal and 'kinh doanh' in normal else 'balance_sheet' if 'can doi ke toan' in normal else ''
   column_normal=normalize_text(cell.column)
   # Relative position must be explicit; an opening balance is not 'prior'
   # merely because the old parser used that tag for its endpoint.
   comparison_role=('prior' if re.search(r'\b(?:(?:nam|ky) truoc|prior|previous)\b',column_normal) else
                    'current' if re.search(r'\b(?:(?:nam|ky) (?:nay|hien tai)|current)\b',column_normal) else '')
   period_role=cell.period_role if cell.period_role not in {'current','prior'} else ''
   if re.search(r'\b(?:dau (?:nam|ky)|so du dau)\b',column_normal):period_role='opening'
   elif re.search(r'\b(?:cuoi (?:nam|ky)|so du cuoi)\b',column_normal):period_role='closing'
   if not period_role and '|' not in line:
    narrative_prefix=normalize_text(line[:c.source_span[0]])
    if re.search(r'\b(?:tai )?(?:thoi diem )?cuoi (?:nam|ky)\b',narrative_prefix):period_role='closing'
    elif re.search(r'\b(?:tai )?(?:thoi diem )?dau (?:nam|ky)\b',narrative_prefix):period_role='opening'
   period_binding=period_binding if period_binding=='local_results_heading' else anchor.get('basis','') if anchor else 'local_table_context' if period and semantic.get('period_origin')=='local_table_context' else 'plan_actual_header_ratio' if period and semantic.get('period_origin')=='plan_actual_header_ratio' else 'source_column' if period else '' 
   period_start=period_end=as_of_date='';method='';cashflow_start=False
   normalized_column=normalize_text(cell.column)
   date_match=re.search(r'(?<!\d)(\d{1,2})[/.](\d{1,2})[/.]((?:19|20)\d{2})(?!\d)',cell.column)
   if date_match:
    from datetime import date
    day,month,year=map(int,date_match.groups());date_text=date(year,month,day).isoformat();as_of_date=date_text
    if period_role=='opening':period_start=date_text
    elif period_role=='closing':period_end=date_text
   elif anchor and anchor.get('basis')=='adjacent_balance_title_date' and anchor.get('year') and re.search(r'dau nam|dau ky|cuoi nam|cuoi ky',normalized_column):
    year=int(anchor['year'])
    if 'dau' in normalized_column:
     period_role='opening'
     if 'dau nam' in normalized_column:period_start=f'{year}-01-01'
    else:period_role='closing';period_end=anchor.get('as_of_date','')
   elif (anchor and anchor.get('basis')=='adjacent_balance_title_date'
         and anchor.get('as_of_date','').endswith('-12-31')
         and normalized_column==f"nam {anchor.get('year')}"):
    period_role='closing';period_end=anchor['as_of_date']
   if anchor and anchor.get('basis')=='layout_scalar_date':
    as_of_date=anchor.get('as_of_date','')
   if not as_of_date and period_role in {'opening','closing'}:
    as_of_date=period_start if period_role=='opening' else period_end
   if not as_of_date and '|' not in line:
    stated=re.search(r'(?<!\d)(\d{1,2})[/.](\d{1,2})[/.]((?:19|20)\d{2})(?!\d)',line[:c.source_span[0]])
    if stated:
     from datetime import date
     as_of_date=date(int(stated[3]),int(stated[2]),int(stated[1])).isoformat()
   before=snapshot.lines[c.source_name][max(0,c.line_number-80):c.line_number-1]
   titles=[normalize_text(t) for t in before if '|' not in t and re.search(r'bao cao (?:luu chuyen tien te|ket qua)',normalize_text(t))]
   if titles:
    title=titles[-1]
    if 'luu chuyen tien te' in title:
     statement='cash_flow'
     method='indirect' if 'gian tiep' in title else 'direct' if 'truc tiep' in title else ''
    elif 'ket qua' in title:statement='income_statement'
   if statement=='cash_flow' and method=='indirect' and metric_key(cell.metric_label) in {'profit_before_tax','profit_after_tax'}:
    descriptor=normalize_text(line[:c.source_span[0]])
    cashflow_start=bool(re.match(r'^(?:0?1 )?(?:loi nhuan|tong loi nhuan)',descriptor) or re.match(r'^0?1 ',descriptor))
   context=c.report_context or {};report_year=str(context.get('report_year',''))
   if is_profile and not report_year and re.search(r'thong tin (?:chung|khai quat)',normal):
    from .report_context import title_year
    local_titles=[{'source_name':c.source_name,'line_number':i+1,'page':c.page,'text':t,'year':title_year(t)} for i,t in enumerate(snapshot.lines[c.source_name][:c.line_number-1]) if snapshot.pages[c.source_name][i+1]==c.page and title_year(t)]
    if local_titles and len({w['year'] for w in local_titles})==1:
     report_year=local_titles[-1]['year'];context={'report_year':report_year,'title_evidence':local_titles}
     from dataclasses import replace
     cell=replace(cell,table_semantics={**semantic,'local_period_evidence':[*semantic.get('local_period_evidence',[]),*local_titles]})

   if report_year and (not context.get('title_evidence') or any(not snapshot.witness_valid(w,c.source_name) for w in context['title_evidence'])):report_year=''
   # Local plans and effective/change dates override the report's publication year.
   if is_profile or ('|' not in line and ':' in prefix and re.search(r'thong tin (?:chung|khai quat)',normal)):
    local_lines=snapshot.lines[c.source_name]
    at=c.line_number-1
    nearby=[line]
    local_modifier=r'^(?:(?:[0-9]+|[ivxlcdm]+) )?(?:ke hoach|du kien|du bao|dinh huong|thay doi|dieu chinh|tang von|giam von|(?:co )?hieu luc|tu ngay|ke tu|ngay|tai ngay|vao ngay)\b'
    for pos in range(at-1,max(-1,at-4),-1):
     text=local_lines[pos]
     if text.startswith('## Page') or ('|' in text) or ':' in text:break
     if text.strip() and not re.search(local_modifier,normalize_text(text)):break
     nearby.append(text)
    for pos in range(at+1,min(len(local_lines),at+3)):
     text=local_lines[pos]
     if text.startswith('#') or ':' in text or '|' in text:break
     if text.strip() and not re.search(local_modifier,normalize_text(text)):break
     nearby.append(text)
    local=normalize_text(' '.join(nearby))
    uncertain_profile=bool(re.search(r'\b(?:ke hoach|du kien|du bao|dinh huong|thay doi|dieu chinh|tang von|giam von|hieu luc|tu ngay|ke tu)\b',normal+' '+local))
    local_years=set(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)',local))
    if uncertain_profile or (local_years and local_years!={report_year}):
     period='';period_binding='';as_of_date='';report_year='';role='';is_profile=False
   # Report profile year is allowed only under a validated same-page
   # annual-report title and profile section. It is not an as-of date.
   if is_profile and not period and not as_of_date and report_year and re.search(r'thong tin (?:chung|khai quat)',normal) and not cell.embedded_entity and not cell.source_assertion:
    titles=[w for w in context.get('title_evidence',[]) if w.get('page')==c.page and w.get('line_number',0)<c.line_number and snapshot.witness_valid(w,c.source_name)]
    if titles:
     period=report_year;period_binding='profile_report_title'
   unit_proof=_unit_evidence(snapshot,c,cell,line,unit)
   if kind=='money' and not assumption and not unit_proof:
    raise ValueError('source_unit_witness_missing')
   from .fiscal_period_evidence import fiscal_binding
   metric=metric_key(cell.metric_label)
   fiscal_role=role
   if not fiscal_role and normalize_text(cell.column) in {'nam nay','nam truoc'}:
    fiscal_role='actual'
   fiscal=fiscal_binding(snapshot,c,cell,metric,period,as_of_date,fiscal_role)
   if fiscal and not role:role='actual'
   endpoint_witness=None
   if not period_role and role=='actual' and metric in {
       'total_assets','current_assets','total_liabilities','current_liabilities',
       'owners_equity','net_revenue','profit_after_tax','profit_before_tax'}:
    endpoint_witness=_narrative_endpoint_witness(snapshot,c,metric,raw,period)
    if endpoint_witness:period_role='closing'
   if (fiscal and not period_role and as_of_date==fiscal['fiscal_end_date']
       and metric in {'total_assets','literal:tai san ngan han','current_assets',
                      'total_liabilities','current_liabilities','owners_equity'}):
    period_role='closing';period_end=as_of_date
   scope=(cell.source_assertion or {}).get('binding',{}).get('accounting_scope','') or semantic.get('accounting_scope','')
   if fiscal and not scope:scope=fiscal['scope']
   context=tuple(_source_context(snapshot,c,cell,line,unit))+((local_results_witness,) if local_results_witness else ())
   if fiscal:context+=fiscal['evidence']
   if endpoint_witness:context+=(endpoint_witness,)
   facts.append(SourceFact(fact_id=f'{c.source_name}:{c.line_number}:{c.source_span[0]}',metric=metric,metric_label=cell.metric_label,kind=kind,value=Decimal(parsed['value']),raw_value=raw,unit=unit,currency=currency,unit_resolved=not bool(assumption),printed_quantum=Decimal(parsed['quantum']),scale=scale,quantum=Decimal(parsed['quantum'])*scale,period=period,fiscal_year=fiscal['fiscal_year'] if fiscal else '',fiscal_end_date=fiscal['fiscal_end_date'] if fiscal else '',fiscal_evidence=fiscal['evidence'] if fiscal else (),period_role=period_role,comparison_role=comparison_role,period_binding=period_binding,role=role,entity=cell.entity_id or 'entity:source:'+c.source_name,scope=scope,source_name=c.source_name,document_sha256=snapshot.hashes[c.source_name],source_line_sha256=c.source_line_sha256,line_number=c.line_number,source_span=tuple(c.source_span),page=c.page,source_context=context,unit_evidence=unit_proof,column_label=cell.column,table_key=table,section=section,statement=statement,period_basis=period_basis(cell.column),as_of_date=as_of_date,period_start=period_start,period_end=period_end,cashflow_method=method,cashflow_start=cashflow_start,restated='trinh bay lai' in normalize_text(cell.column),source_kind='profile' if is_profile else 'table' if '|' in line else 'narrative',is_profile=is_profile,is_historical=bool(cell.source_assertion and cell.source_assertion.get('binding',{}).get('event')),reporting_entity=not cell.embedded_entity,source_years=tuple(sorted(set(re.findall(r'(?:19|20)\d{2}',' '.join(h.get('text','') for h in semantic.get('header_evidence',[])))))),report_year=report_year or str((cell.report_field or {}).get('report_year','')),unit_assumption=assumption,number_format_basis=basis,source_verified=True,candidate_index=c.index))
  except (ValueError,TypeError,KeyError) as e:errors.append({'candidate':c.index,'reason':str(e)})
 from .compound_ratio_facts import recover_compound_ratio_facts,recover_glued_liquidity_ratio_facts
 facts.extend(recover_compound_ratio_facts(candidates,snapshot,facts))
 facts.extend(recover_glued_liquidity_ratio_facts(candidates,snapshot))
 return facts,errors




