"""Conservative, source-only label/value and dated-column bindings."""
from decimal import Decimal
from datetime import date
from hashlib import sha256
import re
from .ontology import normalize_text
from .number_formats import alternatives
from .unit_defaults import monetary_label

DATE=re.compile(r'(?<!\d)(\d{1,2})[/.](\d{1,2})[/.]((?:19|20)\d{2})(?!\d)')
MONEY=re.compile(r'^\s*([0-9][0-9.,]*)\s*(tỷ đồng|triệu đồng|nghìn đồng|đồng|VND|VNĐ)\s*$',re.I)
SCALES={'ty dong':Decimal(10)**9,'trieu dong':Decimal(10)**6,'nghin dong':Decimal(1000),'dong':Decimal(1),'vnd':Decimal(1)}

def witness(snapshot,name,n):
 text=snapshot.lines[name][n-1]
 return {'source_name':name,'line_number':n,'page':snapshot.pages[name][n],'text':text,'sha256':sha256(text.encode()).hexdigest()}

def date_text(text):
 m=DATE.search(text)
 if not m:return ''
 try:return date(int(m[3]),int(m[2]),int(m[1])).isoformat()
 except ValueError:return ''

def amount(raw,convention=None):
 options=alternatives(raw);values={Decimal(x['value']) for x in options.values()}
 if len(values)==1:return next(iter(options.values()))
 return options.get(convention)

def layout(snapshot):
 if hasattr(snapshot,'label_layout'):return snapshot.label_layout
 from .post_hrm_source import metric_key
 data={'cards':{},'dates':{},'tables':{}}
 for name,lines in snapshot.lines.items():
  page_lines={}
  for n,line in enumerate(lines,1):page_lines.setdefault(snapshot.pages[name][n],[]).append(n)
  for page,indices in page_lines.items():
   footers=[n for n in indices if re.search(r'so lieu tinh den|so lieu tai ngay',normalize_text(lines[n-1])) and date_text(lines[n-1])]
   dates={date_text(lines[n-1]) for n in footers}
   asof=next(iter(dates)) if len(dates)==1 else ''
   if asof:data['dates'][(name,page)]=(asof,footers[0])
   for n in indices:
    m=MONEY.fullmatch(lines[n-1])
    if not m or n<2 or snapshot.pages[name].get(n-1)!=page:continue
    label=lines[n-2].strip()
    if not monetary_label(label) and metric_key(label) not in {'total_assets','owners_equity'}:continue
    if len(label)>90 or re.search(r'\d|[|:]|lich su|ke hoach|du kien|cong ty con',normalize_text(label)):continue
    record={'label':label,'metric':metric_key(label),'value_line':n,'value_raw':m[1],'unit':m[2],'scale':SCALES[normalize_text(m[2])],'label_line':n-1,'date':asof,'date_line':footers[0] if asof else None,'page':page}
    data['cards'][(name,n)]=record
    if asof:data['dates'][(name,page)]=(asof,footers[0])
  # Flattened profile columns: bind only one monetary field to one money-only
  # value inside the same bounded profile block, with a witnessed report year.
  from .report_context import title_year
  for page,indices in page_lines.items():
   starts=[n for n in indices if normalize_text(lines[n-1]) in {'general information','thong tin chung','thong tin khai quat'}]
   if not starts:continue
   start=starts[0]
   stops=[n for n in indices if n>start and re.search(r'company history|qua trinh hinh thanh|lich su phat trien',normalize_text(lines[n-1]))]
   end=min(stops) if stops else indices[-1]+1
   labels=[n for n in indices if start<n<end and metric_key(lines[n-1].strip())=='charter_capital']
   values=[n for n in indices if start<n<end and MONEY.fullmatch(lines[n-1].strip())]
   titles=[n for n in indices if title_year(lines[n-1])]
   if len(labels)!=1 or len(values)!=1 or not titles or len({title_year(lines[n-1]) for n in titles})!=1:continue
   ln,vn=labels[0],values[0];m=MONEY.fullmatch(lines[vn-1].strip())
   data['cards'][(name,vn)]={'label':lines[ln-1].strip(),'metric':'charter_capital','value_line':vn,'value_raw':m[1],'unit':m[2],'scale':SCALES[normalize_text(m[2])],'label_line':ln,'date':'','date_line':None,'page':page,'profile_year':title_year(lines[titles[-1]-1]),'year_line':titles[-1]}
  from .program_data import _markdown_cells
  current=None
  for n,line in enumerate(lines,1):
   if '|' not in line:
    if line.strip():current=None
    continue
   fields=_markdown_cells(line)
   dated=[i for i,f in enumerate(fields) if re.fullmatch(DATE.pattern+r'\s*(?:VND|VNĐ|đồng)?',f.strip(),re.I) and date_text(f)]
   if len(dated)>=2 and all(i in dated or normalize_text(f) in {'','code','note','ma so','maso','thuyet minh','thuyet','tai san','assets','chi tieu','indicators'} for i,f in enumerate(fields)):
    current={'header_line':n,'labels':fields,'date_columns':dated,'rows':[],'page':snapshot.pages[name][n]}
    data['tables'][(name,n)]=current;continue
   if current and len(fields)==len(current['labels']) and snapshot.pages[name][n]==current['page']:
    if re.fullmatch(r'[-: ]+',fields[0]):continue
    if not monetary_label(fields[0]) and metric_key(fields[0]) not in {'total_assets','owners_equity','literal:tai san ngan han'} and not ('100' in fields and normalize_text(fields[0]).startswith('tai san ')):continue
    current['rows'].append((n,fields))
 snapshot.label_layout=data;return data

def binding(snapshot,c):
 if c.source_name not in snapshot.lines or not 1<=c.line_number<=len(snapshot.lines[c.source_name]):return None
 data=layout(snapshot);name=c.source_name;n=c.line_number;line=snapshot.lines[name][n-1]
 narrative=narrative_binding(snapshot,c)
 if narrative and not data['cards'].get((name,n),{}).get('date'):return narrative
 card=data['cards'].get((name,n))
 if card and card.get('profile_year'):
  return {'label':card['label'],'column':'','year':card['profile_year'],'role':'profile','period_role':'','date':'','basis':'profile_report_title','witnesses':[witness(snapshot,name,card['label_line']),witness(snapshot,name,n),witness(snapshot,name,card['year_line'])],'header':None,'table':None}
 if card:
  return {'label':card['label'],'column':'','year':card['date'][:4],'role':'actual' if card['date'] else '', 'period_role':'closing' if card['date'].endswith('-12-31') else 'opening' if card['date'].endswith('-01-01') else '', 'date':card['date'],'basis':'layout_scalar_date' if card['date'] else 'layout_scalar', 'witnesses':[witness(snapshot,name,card['label_line']),witness(snapshot,name,n)]+([witness(snapshot,name,card['date_line'])] if card['date_line'] else []),'header':None,'table':None}
 column=line[:c.source_span[0]].count('|')-int(line.lstrip().startswith('|'))
 for (source,header),table in data['tables'].items():
  if source!=name or column not in table['date_columns']:continue
  rows=[row for rn,row in table['rows'] if rn==n]
  if not rows:continue
  d=date_text(table['labels'][column]);label=rows[0][0]
  if '100' in rows[0] and normalize_text(label).startswith('tai san '):label='Tài sản ngắn hạn'
  near=' '.join(snapshot.lines[name][max(0,header-5):header-1])
  if re.search(r'ke hoach|du kien|du bao',normalize_text(near)):return None
  return {'label':label,'column':table['labels'][column],'year':d[:4],'role':'actual','period_role':'opening' if d.endswith('-01-01') else 'closing' if d.endswith('-12-31') else '', 'date':d,'basis':'layout_date_columns','witnesses':[witness(snapshot,name,header)],'header':witness(snapshot,name,header),'table':table}
 return None


# Scalar statements with explicit money units. Boundaries are physical source
# spans, never query text, report filenames or values specific to a report.
METRIC = r'(?:doanh thu thuần|tổng doanh thu|lợi nhuận trước thuế|lợi nhuận sau thuế|lợi nhuận gộp|vốn điều lệ|vốn chủ sở hữu|tổng tài sản|tài sản ngắn hạn|hàng tồn kho)'
QUAL = r'(?:\s+(?:hợp nhất|riêng lẻ|thực hiện|kế hoạch|dự kiến|năm\s+(?:19|20)\d{2}))*'
LABEL_END = re.compile(r'(?P<label>'+METRIC+r')(?P<qual>'+QUAL+r')\s*(?:đạt|là|:)?\s*$',re.I)

def local_period(text):
 normalized=normalize_text(text)
 topic=re.match(r'^(?:nhin lai|danh gia|tong ket) (?:hoat dong|ket qua kinh doanh|tinh hinh tai chinh) nam ((?:19|20)\d{2})\b',normalized)
 if topic and not re.search(r'ke hoach|du kien|du bao|cong ty con',normalized):return topic[1],'actual'
 years=set(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)',normalized))
 if len(years)!=1 or DATE.search(text):return None
 if re.search(r'bao cao thuong nien|so voi|nam truoc|lich su|thanh lap|giay phep|dang ky|cap ngay|thay doi|hieu luc|cong ty con|cong ty lien ket',normalized):return None
 if not re.search(r'^(?:trong )?nam |(?:phan tich|tinh hinh|ket qua|diem nhan|diem nhan|hoat dong).*?(?:tai chinh|kinh doanh|nam )',normalized):return None
 role='forecast' if re.search(r'du kien|du bao',normalized) else 'plan' if 'ke hoach' in normalized else 'actual'
 return next(iter(years)),role

def narrative_binding(snapshot,c):
 name=c.source_name;n=c.line_number;line=snapshot.lines[name][n-1]
 if '|' in line:return None
 start,end=c.source_span
 if not re.match(r'\s*(?:(?:tỷ|tỉ|triệu|nghìn|ngàn)\s+)?(?:đồng|VND|VNĐ)\b',line[end:],re.I):return None
 prefix=line[:start];match=LABEL_END.search(prefix);w=[]
 if re.search(r'\b(?:quy|thang|luy ke|dau nam|cuoi nam|tai ngay)\b',normalize_text(prefix)):return None
 if match:
  label=match['label'];qual=match['qual'];local=prefix[:match.start()]
  # Exclude qualified submetrics, comparisons and other enterprises.
  if re.search(r'công ty con|công ty liên kết|của [A-ZĐ]|tỷ lệ|biên ',local):return None
  # A previous sentence cannot donate its year across a change of topic.
  local=re.split(r'(?<=[.;!?])\s+',local)[-1]
  years=set(re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)',local+' '+qual))
  if len(years)>1:return None
  year=next(iter(years)) if years else ''
  role='forecast' if re.search(r'dự kiến|dự báo',local+' '+qual,re.I) else 'plan' if re.search(r'kế hoạch',local+' '+qual,re.I) else 'actual'
  scope='consolidated' if 'hợp nhất' in qual.lower() else 'standalone' if 'riêng lẻ' in qual.lower() else ''
  w=[witness(snapshot,name,n)]
 elif MONEY.fullmatch(line.strip()) and n>1:
  label=snapshot.lines[name][n-2].strip();qual='';year='';role='';scope=''
  if not re.fullmatch(METRIC,label,re.I) or snapshot.pages[name][n-1]!=c.page:return None
  w=[witness(snapshot,name,n-1),witness(snapshot,name,n)]
 else:return None
 if not year:
  # Only the adjacent substantive heading/introduction may supply a year.
  pos=n-2 if match else n-3
  while pos>=0 and not snapshot.lines[name][pos].strip():pos-=1
  if pos<0 or snapshot.pages[name][pos+1]!=c.page:return None
  context=local_period(snapshot.lines[name][pos])
  if not context:return None
  year,inherited_role=context
  if role not in {'plan','forecast'}:role=inherited_role
  w.append(witness(snapshot,name,pos+1))
 return {'label':label,'column':'','year':year,'role':role,'period_role':'','date':'','basis':'source_narrative_local','witnesses':w,'header':None,'table':None,'scope':scope}
