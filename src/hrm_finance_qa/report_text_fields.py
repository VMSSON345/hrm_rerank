"""Literal report fields: typed text and replayable source spans, no numeric HRM proxy."""
from pathlib import Path
from hashlib import sha256
import re
from .ontology import normalize_text
from .post_hrm_contract import SourceFact

QUANTITY_FIELDS={'shares_outstanding':'shares','treasury_shares':'shares','preferred_shares':'shares','total_shares':'shares','foreign_ownership':'%','planned_dividend':'%'}
FIELDS={
 'shares_outstanding':('Cổ phiếu đang lưu hành',('so co phieu dang luu hanh','co phieu dang luu hanh')),
 'treasury_shares':('Cổ phiếu quỹ',('so co phieu quy','co phieu quy','cp quy')),
 'preferred_shares':('Cổ phiếu ưu đãi',('co phieu uu dai',)),
 'total_shares':('Tổng số cổ phiếu',('tong so co phieu',)),
 'foreign_ownership':('Tỷ lệ sở hữu cổ đông nước ngoài',('ty le so huu cua co dong nuoc ngoai',)),
 'planned_dividend':('Cổ tức dự kiến',('co tuc du kien','co tuc ke hoach')),
 'company_name':('Tên giao dịch',('ten giao dich','ten to chuc','ten cong ty','name of organization','company name','trading name')),
 'english_name':('Tên tiếng Anh',('ten tieng anh','english name')),
 'short_name':('Tên viết tắt',('ten viet tat','abbreviated name','short name')),
 'address':('Địa chỉ',('dia chi','dia chi tru so','tru so chinh','dia chi tru so chinh','address','head office','registered address')),
 'phone':('Điện thoại',('so dien thoai','dien thoai','dien thoai lien he','so dien thoai lien he','telephone','phone','tel')),
 'fax':('Fax',('fax','so fax')),
 'website':('Website',('website','web','trang web','trang thong tin dien tu')),
 'email':('Email',('email','e mail','email lien he','thu dien tu')),
 'stock_code':('Mã chứng khoán',('ma chung khoan','ma co phieu','stock code','ticker')),
 'business':('Ngành nghề kinh doanh',('nganh nghe kinh doanh','nganh nghe','linh vuc kinh doanh','business activities')),
 'average_salary':('Tiền lương bình quân',('tien luong binh quan','muc luong binh quan','luong binh quan')),
 'weekly_hours':('Giờ làm việc mỗi tuần',('gio lam viec moi tuan',)),
 'employee_count':('Số lượng nhân viên',('so luong nhan vien','so luong can bo nhan vien','so luong lao dong','nhan vien','nhan su','nguoi lao dong')),
 'disclosure_contact':('Người thực hiện công bố thông tin',('nguoi thuc hien cong bo thong tin','nguoi cong bo thong tin')),
}
ALIASES={a:k for k,(_,aliases) in FIELDS.items() for a in aliases}

def key(label):
 text=normalize_text(label).strip(' .:')
 text=re.sub(r'^(?:[0-9]+|[ivxlcdm]+)\s+','',text)
 if re.fullmatch(r'(?:so luong (?:can bo nhan vien|lao dong)(?: muc luong trung binh doi voi nguoi lao dong)?|nhan su toan cong ty)(?: (?:vao thoi diem ngay|tinh den) [0-9 ]+(?: la)?)?',text):return 'employee_count'
 parts=[normalize_text(p).strip(' .:') for p in re.split(r'[/]',label)]
 keys={ALIASES[p] for p in parts if p in ALIASES}
 return next(iter(keys)) if len(keys)==1 and all(p in ALIASES for p in parts) else ALIASES.get(text) or next((k for a,k in ALIASES.items() if a.replace(' ','')==text.replace(' ','')),None)

def requested_fields(question):
 """Accept only field lookup wording; residual dates/entities/operations survive."""
 text=normalize_text(question);hits=[]
 if re.fullmatch(r'(?:nhan vien|nguoi lao dong) lam viec bao nhieu gio moi tuan',text):return ('weekly_hours',)
 pattern=r'(?<!\w)(?:'+'|'.join(re.escape(a) for a in sorted(ALIASES,key=len,reverse=True))+r')(?!\w)'
 def take(m):hits.append(ALIASES[m[0]]);return ' '
 rest=re.sub(pattern,take,text)
 allowed={'cho','toi','biet','xin','hay','vui','long','cua','cong','ty','doanh','nghiep','la','gi','nao','o','dau','va','voi','bao','cao','trong','nay','duoc','ghi','nhu','the','nao','lien','he','qua','co','ai','thong','tin','ve','mot','so','tai'}
 if hits and set(hits)=={'employee_count'}:
  rest=re.sub(r'\bnam\s+(?:19|20)\d{2}\b',' ',rest)
  rest=re.sub(r'\b(?:ngay )?\d{1,2} \d{1,2} (?:19|20)\d{2}\b',' ',rest);allowed.update({'bao','nhieu','ngay'})
 if hits and set(hits)=={'average_salary'}:
  rest=re.sub(r'\bnam (?:19|20)\d{2}\b',' ',rest);allowed.update({'bao','nhieu','dong','moi','nguoi','thang'})
 if hits and all(k in QUANTITY_FIELDS for k in hits):
  rest=re.sub(r'\b(?:nam )?(?:19|20)\d{2}\b|\b(?:ngay )?\d{1,2} \d{1,2} (?:19|20)\d{2}\b',' ',rest)
  rest=re.sub(r'\b(?:ngay )?\d{1,2} \d{1,2}\b',' ',rest);allowed.update({'bao','nhieu','phan','tram','cuoi','dau','ngay','so','luong'})
 if not hits or any(w not in allowed for w in re.findall(r'\w+',rest)):return ()
 return tuple(dict.fromkeys(hits))

def _witness(snapshot,name,n,kind):
 line=snapshot.lines[name][n-1]
 return {'kind':kind,'line_number':n,'text':line,'sha256':sha256(line.encode()).hexdigest(),'page':snapshot.pages[name][n]}


def _segments(line):
 """Known colon labels, with original offsets; URLs are never labels."""
 starts=[]
 for colon in re.finditer(':',line):
  end=colon.start()
  for start in [0]+[m.end() for m in re.finditer(r'\s+',line[:end])]:
   label=re.sub(r'^\s*(?:\\?[-*•]\s*)?(?:(?:\d+|[a-z])[.)]\s*)?','',line[start:end]).strip()
   field=key(label) if start==0 or not re.match(r'\d',label) else None
   if field and (start==0 or starts and field in {'phone','fax','email','website'}):
    starts.append((start,colon.end(),field,label));break
 for i,(start,value_start,field,label) in enumerate(starts):
  end=starts[i+1][0] if i+1<len(starts) else len(line)
  raw=line[value_start:end];a=value_start+len(raw)-len(raw.lstrip());b=value_start+len(raw.rstrip())
  if a<b:yield field,label,a,b
 # A true two-cell row can directly bind a label to a textual value.
 if line.count('|')==1 and not starts:
  left,right=line.split('|');field=key(left.strip().rstrip(':'))
  if field and right.strip():
   a=len(left)+1+len(right)-len(right.lstrip());yield field,left.strip(),a,len(line.rstrip())


def _descriptors(snapshot,name):
 from .report_context import title_year
 lines=snapshot.lines[name];result=[];profile_closed=False;foreign_section=False;business_heading=None
 titles=[n for n,line in enumerate(lines,1) if title_year(line)]
 years={title_year(lines[n-1]) for n in titles}
 def emit(n,field,label,a,b,period='',context=()):
  if a<b and lines[n-1][a:b] not in {'-','—','N/A'}:result.append((n,field,label,a,b,period,tuple(context)))
 for n,line in enumerate(lines,1):
  plain=normalize_text(line);bare=re.sub(r'^(?:[0-9]+(?: [0-9]+)*|[ivxlcdm]+) ','',plain)
  if bare in {'thong tin chung','thong tin khai quat','gioi thieu chung','thong tin doanh nghiep','thong tin lien he','lien he','contact','general information','company profile'} and not foreign_section:profile_closed=False
  if re.match(r'^(?:(?:[0-9]+|[ivxlcdm]+) )?(?:qua trinh|lich su|history|nganh nghe|business activities|co cau|to chuc nhan su|bao cao kiem toan|audit report)',plain) or re.search(r'^(?:\([0-9]+\)\s*)?.*\b(?:chi nhanh|branch|subsidiary)\b',plain):profile_closed=True
  if re.match(r'^\s*(?:[IVX]+[.]|[0-9]+[.](?![0-9])|[0-9]+[.][0-9]+[.]?\s)',line) and ':' not in line:foreign_section=False
  heading=re.sub(r'^\s*(?:\([0-9]+\)|[0-9]+[.])\s*','',line)
  if ':' not in heading and len(heading)<130 and re.search(r'\b(?:chi nhanh|branch|subsidiary|bao cao kiem toan|audit report)\b',normalize_text(heading)):foreign_section=True
  if foreign_section:business_heading=None;continue
  if not line.strip() or line.lstrip().startswith(('<!--','## Page','|---')):continue
  # Lists belong to an explicit business heading until the next section or
  # narrative paragraph. Each item retains its own physical source span.
  if re.fullmatch(r'nganh nghe kinh doanh(?: chinh)?(?: cua [a-z0-9 ]+)?(?: gom)?',bare):
   business_heading=n;continue
  bullet=re.match(r'\s*\\?[-*•]\s+(\S.*)',line)
  if business_heading and bullet:
   emit(n,'business',FIELDS['business'][0],bullet.start(1),len(line.rstrip()),context=[_witness(snapshot,name,business_heading,'business_heading')]);continue
  if business_heading and result and result[-1][1]=='business' and line[0:1].islower() and not lines[result[-1][0]-1].rstrip().endswith((';','.',')')):
   previous=result[-1];result[-1]=previous[:6]+(previous[6]+(_witness(snapshot,name,n,'business_continuation'),),);continue
  if business_heading and (re.match(r'^\s*\d+[.]',line) or len(line)>40 and not bullet):business_heading=None
  # Label and value may occupy separate lines within one source page.
  standalone=key(line.strip())
  if n<len(lines) and snapshot.pages[name][n+1]==snapshot.pages[name][n] and not profile_closed:
   value=lines[n].strip();field=standalone
   if field in {'company_name','english_name','short_name','stock_code','address','phone','fax','website','email'} and value and not key(value) and not list(_segments(value)) and '|' not in value and len(value)<300:
    valid=not re.fullmatch(r'[0-9.,]+',value)
    if field=='stock_code':valid=bool(re.match(r'^[A-Z]{2,6}(?:\s|$|[·—-])',value))
    if field=='email':valid=bool(re.fullmatch(r'[^ @]+@[^ @]+\.[^ @]+',value))
    if field=='website':valid=bool(re.match(r'(?:https?://|www\.)',value))
    if valid:
     a=lines[n].find(value);emit(n+1,field,line.strip(),a,a+len(value),context=[_witness(snapshot,name,n,'field_label')])
   if normalize_text(line) in {'email website','website email'}:
    for field,pattern in [('email',r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}'),('website',r'(?:https?://|www\.)[^\s·]+')]:
     m=re.search(pattern,lines[n])
     if m:emit(n+1,field,FIELDS[field][0],*m.span(),context=[_witness(snapshot,name,n,'field_label')])
  for field,label,a,b in _segments(line):
   if profile_closed and field not in {'business','employee_count'}:continue
   period='';context=[]
   if field=='employee_count':
    raw=line[a:b];count=re.search(r'(?<![\d.,])\d+(?:[.,]\d{3})*\s+(?:cán bộ nhân viên|nhân sự|nhân viên|người)\b',raw,re.I)
    if not count or re.search(r'\b(?:ke hoach|du kien|du bao|forecast|plan)\b',normalize_text(raw[:count.start()])):continue
    b=a+count.end();a+=count.start();local=set(re.findall(r'(?:19|20)\d{2}',line));period=next(iter(local)) if len(local)==1 else ''
   if field=='address' and n<len(lines):
    following=lines[n].strip()
    # Only an address ending mid administrative phrase can continue unlabelled.
    if re.search(r'(?:thị xã|thành phố|huyện|tỉnh|quận|phường|xã)\s*$',line[a:b],re.I) and following and not re.search(r'[:|]|^#|^\d+[.]',following):
     context.append(_witness(snapshot,name,n+1,'address_continuation'))
   emit(n,field,label,a,b,period,context)
  # Narrative totals have explicit personnel ownership and an explicit year.
  count=re.search(r'(?:tổng số (?:lao động|nhân viên)(?: trong [^.]{0,90}?)?|số lượng (?:CBNV|cán bộ nhân viên|nhân viên|lao động)(?: tại [^.]{0,90}?)?)\s+(?:là|có|đạt)\s+(?P<count>\d+(?:[.,]\d{3})*\s+(?:người|nhân sự|nhân viên))\b',line,re.I)
  if count and not re.search(r'\b(?:ke hoach|du kien|du bao|forecast|plan|cong ty con|chi nhanh)\b',plain):
   local=set(re.findall(r'(?:19|20)\d{2}',line))
   if len(local)==1:emit(n,'employee_count',FIELDS['employee_count'][0],*count.span('count'),next(iter(local)))
  if not re.search(r'\b(?:ke hoach|du kien|du bao|forecast|plan|cong ty con|chi nhanh)\b',plain):
   salary=re.search(r'tiền lương bình quân(?: trong)? năm ((?:19|20)\d{2})\s*:\s*(?P<value>\d+(?:[.,]\d{3})*\s*(?:đồng|đ)\s*/\s*người\s*/\s*tháng)',line,re.I)
   if salary:emit(n,'average_salary',FIELDS['average_salary'][0],*salary.span('value'),salary[1])
   hours=re.search(r'(?:chế độ )?làm việc\s+(?P<value>\d+(?:[.,]\d+)?\s*giờ\s*/\s*tuần)',line,re.I)
   if hours:emit(n,'weekly_hours',FIELDS['weekly_hours'][0],*hours.span('value'))
  total=re.fullmatch(r'\s*\\?[-*•]?\s*Tổng số\s*:\s*(?P<count>\d+)\s*',line,re.I)
  if total and len(years)==1:
   nearby=[j for j in range(max(1,n-6),n) if re.search(r'\b(?:can bo nhan vien|thong ke can bo)\b',normalize_text(lines[j-1]))]
   context_text=' '.join(normalize_text(x) for x in lines[max(0,n-7):n])
   if nearby and not re.search(r'\b(?:ke hoach|du kien|du bao|chi nhanh|cong ty con)\b',context_text):
    emit(n,'employee_count',FIELDS['employee_count'][0],*total.span('count'),next(iter(years)),[_witness(snapshot,name,nearby[-1],'personnel_heading'),_witness(snapshot,name,titles[0],'report_year')])
 result.extend(_quantity_descriptors(snapshot,name))
 return result


def source_fields(snapshot):
 facts=[]
 for name in snapshot.lines:
  for n,field,label,a,b,period,context in _descriptors(snapshot,name):
   line=snapshot.lines[name][n-1];value=line[a:b]
   facts.append(SourceFact(fact_id=f'{name}:{n}:{a}',metric='field:'+field,metric_label=label,kind='text',text_value=value,raw_value=value,value=None,entity='entity:source:'+name,source_name=name,document_sha256=snapshot.hashes[name],source_line_sha256=sha256(line.encode()).hexdigest(),line_number=n,source_span=(a,b),page=snapshot.pages[name][n],period=period,source_context=context,source_kind='profile',is_profile=True,role='profile',source_verified=True))
 return facts


def verify_text_fact(fact,snapshot):
 if fact.kind!='text' or fact.source_name not in snapshot.lines:return False
 name=fact.source_name;n=fact.line_number
 if fact.document_sha256!=snapshot.hashes[name] or not 1<=n<=len(snapshot.lines[name]):return False
 line=snapshot.lines[name][n-1];a,b=fact.source_span
 if sha256(line.encode()).hexdigest()!=fact.source_line_sha256 or snapshot.pages[name][n]!=fact.page:return False
 if not 0<=a<b<=len(line) or line[a:b]!=fact.text_value:return False
 return (n,fact.metric.removeprefix('field:'),fact.metric_label,a,b,fact.period,fact.source_context) in _descriptors(snapshot,name)


def _field_value(fact):
 return ' '.join([fact.text_value]+[w['text'].strip() for w in fact.source_context if w.get('kind') in {'address_continuation','business_continuation'}])


def _equivalent_value(field,value):
 if field in {'phone','fax'} and re.fullmatch(r'[+0-9(). \-]+',value):return re.sub(r'[^+0-9]','',value)
 if field=='website':
  from urllib.parse import urlsplit
  parsed=urlsplit(value if '://' in value else 'https://'+value)
  return (parsed.hostname.removeprefix('www.') if parsed.hostname else value,parsed.port,parsed.path.rstrip('/'),parsed.query,parsed.fragment)
 if field=='address':
  # Punctuation/case and a complete, conventional city abbreviation only.
  text=normalize_text(value);return re.sub(r'\btp hcm\b','tp ho chi minh',text)
 return value


def answer_fields(question,snapshot):
 requested=requested_fields(question)
 if not requested:return None
 facts=source_fields(snapshot);outputs=[]
 for field in requested:
  years=set(re.findall(r'(?:19|20)\d{2}',question)) if field in {'employee_count','average_salary','weekly_hours'} or field in QUANTITY_FIELDS else set()
  found=[f for f in facts if f.metric=='field:'+field and (not years or f.period in years)]
  date=re.search(r'(?<!\d)(\d{1,2})[/.](\d{1,2})[/.]((?:19|20)\d{2})(?!\d)',question)
  if date:
   target=tuple(map(int,date.groups()))
   found=[f for f in found if any(tuple(map(int,m.groups()))==target for m in re.finditer(r'(?<!\d)(\d{1,2})[/.](\d{1,2})[/.]((?:19|20)\d{2})(?!\d)',snapshot.lines[f.source_name][f.line_number-1]+(' '+' '.join(w['text'] for w in f.source_context if w.get('kind')=='as_of_footer') if field in QUANTITY_FIELDS else '')))]
  values={(re.sub(r'[^0-9]','',f.text_value),f.period) if field=='employee_count' else (_quantity_number(field,f.text_value),f.period) if field in QUANTITY_FIELDS else _equivalent_value(field,_field_value(f)) for f in found};title=FIELDS[field][0]
  if field=='employee_count' and len(years)==1:title+=' năm '+next(iter(years))
  if not found:
   outputs.append({'status':'ABSTAIN','final_action':'ABSTAIN','answer':None,'canonical_value':None,'answer_type':'text','rendered':f'Chưa xác nhận được {title.lower()} từ trường thông tin có liên kết rõ trong nguồn đã chọn.','citations':[]});continue
  if len(values)>1 and field!='business':
   outputs.append({'status':'ABSTAIN','final_action':'ABSTAIN','answer':None,'canonical_value':None,'answer_type':'text','rendered':title+' có nhiều công bố chưa thống nhất:\n'+'\n'.join(f'- {f.text_value} (trang {f.page}).' for f in found),'citations':[f.citation() for f in found]});continue
  selected=found[0];published='\n'.join('- '+v for v in dict.fromkeys(_field_value(f) for f in found)) if field=='business' and len(found)>1 else _field_value(selected);cit=selected.citation();cit['text_value']=selected.text_value;cit['answer_type']='text';name=re.sub(r'^[a-f0-9]{10}-','',Path(selected.source_name).stem)
  outputs.append({'status':'ACCEPTED','final_action':'ANSWER','answer_type':'text','answer':published,'canonical_value':None,'unit':None,'rendered':f'{title}: {published} Nguồn: {name}, trang {selected.page}.','citations':[dict(f.citation(),text_value=f.text_value,answer_type='text') for f in found] if field=='business' else [cit],
   'proof':{'verification_engine':'post_hrm_contract_v1','source_action':'ANSWER','source_certificate':{'schema':'source-text-span-v1','facts':[f.model_dump(mode='json') for f in found] if field=='business' else [selected.model_dump(mode='json')]}},'validation':{'checks':[{'layer':'data_contract','code':'literal_text_source_replay','status':'PASS','blocking':True,'message':'Đã đối chiếu nguyên văn trường, nhãn và tọa độ nguồn.'}]}})
  if field in QUANTITY_FIELDS:
   number=_quantity_number(field,selected.text_value)
   outputs[-1].update(canonical_value=str(number),answer=str(number),unit=QUANTITY_FIELDS[field],answer_type='quantity')
   if QUANTITY_FIELDS[field]=='shares' and not re.search(r'cổ phiếu|\bCP\b',published,re.I):outputs[-1]['rendered']=f'{title}: {published} cổ phiếu. Nguồn: {name}, trang {selected.page}.'
 if not snapshot.unchanged():return {'status':'ABSTAIN','answer':None,'canonical_value':None,'rendered':'Tài liệu đã thay đổi trong lúc xử lý.','citations':[]}
 if len(outputs)==1:result=outputs[0]
 else:
  count=sum(o['status']=='ACCEPTED' for o in outputs);result={'status':'ACCEPTED' if count==len(outputs) else 'ABSTAIN','final_action':'ANSWER' if count==len(outputs) else 'ABSTAIN','answer':None,'canonical_value':None,'outputs':outputs,'output_count':len(outputs),'coverage_state':'FULL' if count==len(outputs) else 'PARTIAL' if count else 'NONE','rendered':'\n\n'.join(o['rendered'] for o in outputs),'citations':[]}
 result.update(validation_engine='post_hrm_contract_v1',legacy_symbolic_used=False,execution_route='source_text_lookup',hrm_used=False,document_count=len(snapshot.files))
 return result


def _quantity_number(field,value):
 from decimal import Decimal
 token=re.search(r'\d[\d.,]*',value)[0]
 return Decimal(token.replace(',','.')) if QUANTITY_FIELDS[field]=='%' else Decimal(token.replace('.','').replace(',',''))


def _quantity_descriptors(snapshot,name):
 from .source_layout_bindings import layout
 from .program_data import _markdown_cells
 lines=snapshot.lines[name];data=layout(snapshot);out=[]
 count=re.compile(r'\s*\d+(?:[.,]\d{3})*\s*')
 labels={a:k for a,k in ALIASES.items() if k in QUANTITY_FIELDS and QUANTITY_FIELDS[k]=='shares'}
 def emit(n,field,label,a,b,period='',context=()):out.append((n,field,label,a,b,period,tuple(context)))
 def context(n):
  record=data['dates'].get((name,snapshot.pages[name][n]))
  return (record[0][:4],[_witness(snapshot,name,record[1],'as_of_footer')]) if record else ('',[])
 # Repeated numeric/label runs establish orientation; do not pick an arbitrary neighbour.
 i=0
 while i<len(lines):
  if not (count.fullmatch(lines[i]) or normalize_text(lines[i]) in labels):i+=1;continue
  start=i;page=snapshot.pages[name][i+1]
  while i<len(lines) and snapshot.pages[name][i+1]==page and (count.fullmatch(lines[i]) or normalize_text(lines[i]) in labels):i+=1
  run=list(range(start,i));reverse=bool(count.fullmatch(lines[start]))
  for j in range(0,len(run)-1,2):
   v,l=(run[j],run[j+1]) if reverse else (run[j+1],run[j])
   field=labels.get(normalize_text(lines[l]))
   if not field or not count.fullmatch(lines[v]):continue
   year,w=context(v+1);a=len(lines[v])-len(lines[v].lstrip());b=len(lines[v].rstrip())
   emit(v+1,field,lines[l].strip(),a,b,year,[*w,_witness(snapshot,name,l+1,'field_label')])
 table=None
 for n,line in enumerate(lines,1):
  m=re.search(r'Tương đương\s+(?P<value>\d+(?:[.,]\d{3})*\s+cổ phiếu)',line,re.I)
  if m and any(source==name and 0<n-ln<=3 and r['metric']=='charter_capital' for (source,ln),r in data['cards'].items()):
   year,w=context(n);emit(n,'total_shares',FIELDS['total_shares'][0],*m.span('value'),year,w)
  fields=_markdown_cells(line) if '|' in line else []
  if not fields:
   if line.strip():table=None
   continue
  normalized=[normalize_text(x) for x in fields]
  if any('so luong cp' in x for x in normalized) and any('ty le' in x for x in normalized):table=('shares',n,normalized);continue
  if any('ke hoach' in x and re.search(r'(?:19|20)\d{2}',x) for x in normalized) and any('thuc hien' in x for x in normalized):table=('dividend',n,normalized);continue
  if table and len(fields)==len(table[2])-1 and line.lstrip().startswith('|') and normalized[0]=='tong cong':
   fields=['',*fields];normalized=['',*normalized]
  if not table or len(fields)!=len(table[2]):continue
  kind,hn,headers=table;year,w=context(n)
  if kind=='shares':
   for col,header in enumerate(headers):
    field='foreign_ownership' if 'co dong nuoc ngoai' in normalize_text(' '.join(fields[:col])) and 'ty le' in header else 'total_shares' if any(x=='tong cong' for x in normalized[:col]) and 'so luong cp' in header else None
    if not field:continue
    raw=fields[col];a=line.find(raw);emit(n,field,FIELDS[field][0],a,a+len(raw),year,[*w,_witness(snapshot,name,hn,'field_header')])
  elif 'chia co tuc' in normalize_text(fields[0]):
   for col,header in enumerate(headers):
    yr=re.search(r'(?:19|20)\d{2}',header)
    if yr and 'ke hoach' in header and re.fullmatch(r'\d+(?:[.,]\d+)?%',fields[col].strip()):
     raw=fields[col];a=line.find(raw,line.find('|')+1)
     # Equal actual and plan cells still need the plan column's own span.
     parts=list(re.finditer(r'[^|]+',line));a=parts[col].start()+len(parts[col][0])-len(parts[col][0].lstrip())
     emit(n,'planned_dividend',FIELDS['planned_dividend'][0],a,a+len(raw),yr[0],[_witness(snapshot,name,hn,'field_header')])
 return out
