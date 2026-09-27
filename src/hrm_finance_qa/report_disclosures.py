"""Source-bound answers for bounded audit and related-party disclosures."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import re
from .ontology import normalize_text


def _citation(snapshot, name, number, start=0, end=None):
    line=snapshot.lines[name][number-1]
    end=len(line) if end is None else end
    return {
        'source_name':name,'page':snapshot.pages[name][number],
        'line_number':number,'source_span':[start,end],
        'source_span_scope':'line_v2','source_line_sha256':sha256(line.encode()).hexdigest(),
        'document_sha256':snapshot.hashes[name],'text_value':line[start:end],
        'answer_type':'text','extraction_method':'mineru_markdown',
    }


def _result(question,snapshot,name,rendered,citations,accepted=False):
    if not snapshot.unchanged():
        rendered='Tài liệu đã thay đổi trong lúc xử lý; cần đọc lại nguồn.'
        citations=[];accepted=False
    return {
        'status':'ACCEPTED' if accepted else 'ABSTAIN',
        'final_action':'ANSWER' if accepted else 'ABSTAIN',
        'answer_type':'text','answer':rendered if accepted else None,
        'canonical_value':None,'unit':None,'rendered':rendered,
        'citations':citations,'validation_engine':'post_hrm_contract_v1',
        'execution_route':'source_disclosure_lookup','legacy_symbolic_used':False,
        'document_count':len(snapshot.files),
        'proof':{'verification_engine':'post_hrm_contract_v1',
                 'source_action':'ANSWER' if accepted else 'ABSTAIN',
                 'source_certificate':{'schema':'source-text-span-v1','citations':citations}},
    }


def answer_disclosures(question,snapshot):
    q=normalize_text(question)
    margin_conflict=bool(re.search(r'\bbien loi nhuan (?:gop|rong)\b',q)) and not re.search(r'\btinh\b',q)
    audit_firm=bool(re.search(r'\b(?:don vi|cong ty) kiem toan\b',q))
    audit_opinion='y kien kiem toan' in q or 'y kien ngoai tru' in q or (audit_firm and 'dua ra y kien gi' in q)
    emphasis='doan nhan manh' in q and 'hoat dong lien tuc' in q
    going_concern='hoat dong lien tuc' in q and ('nghi ngo' in q or 'khong chac chan' in q)
    related=bool(re.search(r'giao dich.*(?:nguoi (?:co )?lien quan|nguoi noi bo|ben lien quan)',q)
                 or re.search(r'hop dong.*(?:nguoi noi bo|nguoi (?:co )?lien quan|ben lien quan)',q))
    composition='doanh thu' in q and ('chu yeu' in q or 'doanh thu chinh' in q) and ('chiem' in q or 'bao nhieu phan tram' in q)
    composition_text='doanh thu' in q and ('chu yeu' in q or 'mang kinh doanh' in q) and not ('chiem' in q or 'bao nhieu phan tram' in q)
    liquidity_risk=('nguy co mat thanh khoan' in q or ('mat thanh khoan' in q and ('ket luan' in q or 'co the' in q))
                    or ('mat kha nang thanh toan' in q and ('ket luan' in q or 'co the' in q)))
    internal_advance='tam ung' in q and 'ke toan truong' in q
    if not any((margin_conflict,audit_firm,audit_opinion,emphasis,going_concern,related,composition,composition_text,liquidity_risk,internal_advance)):
        return None
    if len(snapshot.files)!=1:
        return _result(question,snapshot,'','Cần chọn một báo cáo để kiểm chứng nội dung này.',[])
    name=next(iter(snapshot.files))
    lines=snapshot.lines[name]
    source=re.sub(r'^[a-f0-9]{10}-','',Path(name).stem)
    year_match=re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)',q)
    if margin_conflict and len(set(year_match))==2:
        kind='gộp' if 'bien loi nhuan gop' in q else 'ròng'
        pattern=re.compile(r'biên lợi nhuận '+kind+
            r'\b[^.]{0,90}?từ mức\s*(\d+(?:[,.]\d+)?)%\s*năm\s*((?:19|20)\d{2})'
            r'\s*lên mức\s*(\d+(?:[,.]\d+)?)%\s*năm\s*((?:19|20)\d{2})',re.I)
        claims=[(n,m) for n,line in enumerate(lines,1) if (m:=pattern.search(line))
                and {m[2],m[4]}==set(year_match)]
        if len(claims)==1:
            n,m=claims[0]
            return _result(question,snapshot,name,
                f'Biên lợi nhuận {kind} tăng từ {m[1]}% năm {m[2]} lên {m[3]}% năm {m[4]}. '
                f'Nguồn: {source}, trang {snapshot.pages[name][n]}.',
                [_citation(snapshot,name,n,*m.span())],True)
        if len(claims)>1:
            return _result(question,snapshot,name,
                'Nguồn có nhiều công bố biên lợi nhuận khác nhau; chưa chọn được một cặp kỳ duy nhất.',
                [_citation(snapshot,name,n,*m.span()) for n,m in claims])
    if len(set(year_match))>1:
        return _result(question,snapshot,name,'Cần nêu một năm báo cáo cho câu hỏi này.',[])
    year=year_match[0] if year_match else ''
    from .report_context import title_year
    title_years={title_year(line) for line in lines[:100] if title_year(line)}
    title_years.update(match[1] for line in lines[:100]
        if (match:=re.fullmatch(r'nam bao cao ((?:19|20)\d{2})',normalize_text(line))))
    if year and len(title_years)==1 and year not in title_years:
        return _result(question,snapshot,name,
            f'Báo cáo được chọn là năm {next(iter(title_years))}; chưa có căn cứ cho năm {year}.',[])
    if margin_conflict and year:
        kind='gộp' if 'bien loi nhuan gop' in q else 'ròng'
        pattern=re.compile(r'biên lợi nhuận '+kind+r'\b[^.]{0,150}?\blên\s*(\d+(?:[,.]\d+)?)%',re.I)
        claims=[]
        for n,line in enumerate(lines,1):
            match=pattern.search(line)
            if not match:continue
            local=line+' '+(lines[n-2] if n>1 and snapshot.pages[name][n-1]==snapshot.pages[name][n] else '')
            if year not in local:continue
            claims.append((n,match[1],bool(re.search(r'\bhợp nhất\b',line,re.I))))
        values={value.replace(',','.') for _,value,_ in claims}
        if len(values)>1:
            details='; '.join(f'{value}% '+('hợp nhất' if consolidated else 'chưa rõ phạm vi')+
                f' (trang {snapshot.pages[name][n]})' for n,value,consolidated in claims)
            return _result(question,snapshot,name,
                f'Nguồn có các công bố biên lợi nhuận {kind} năm {year} khác nhau: {details}. Cần xác định phạm vi trước khi chọn số.',
                [_citation(snapshot,name,n) for n,_,_ in claims])
    if margin_conflict:
        return None
    if going_concern:
        found=[n for n,line in enumerate(lines,1)
               if 'hoat dong lien tuc' in normalize_text(line)
               and 'nghi ngo dang ke' in normalize_text(line)
               and (not year or year in line)]
        if len(found)==1:
            n=found[0]
            return _result(question,snapshot,name,
                f'Có. Báo cáo nêu yếu tố không chắc chắn trọng yếu dẫn đến nghi ngờ đáng kể về khả năng hoạt động liên tục. Nguồn: {source}, trang {snapshot.pages[name][n]}.',
                [_citation(snapshot,name,n)],True)
        return _result(question,snapshot,name,
            'Chưa xác nhận được một đoạn kiểm toán gắn với khả năng hoạt động liên tục cùng kỳ trong Markdown.',[])
    if liquidity_risk:
        return _result(question,snapshot,name,
            'Chưa đủ căn cứ để kết luận nguy cơ mất thanh khoản chỉ từ một chỉ số. Cần xem dòng tiền, lịch đáo hạn nợ và thuyết minh khả năng thanh toán.',[])
    if composition_text:
        pattern=re.compile(r'doanh thu (?:mảng|mang)\s+(.{3,90}?)\s+vốn chiếm tỷ trọng lớn',re.I)
        claims=[(n,m) for n,line in enumerate(lines,1) if (m:=pattern.search(line))
                and (not year or year in line)]
        if len(claims)==1:
            n,m=claims[0]
            return _result(question,snapshot,name,
                f'Báo cáo mô tả doanh thu mảng {m[1].strip()} chiếm tỷ trọng lớn'
                +(f' trong năm {year}' if year else '')
                +f'; nguồn không nêu tỷ trọng chính xác. Nguồn: {source}, trang {snapshot.pages[name][n]}.',
                [_citation(snapshot,name,n,*m.span())],True)
        if len(claims)>1:
            return _result(question,snapshot,name,'Có nhiều mảng được mô tả chiếm tỷ trọng lớn; chưa xác định một mảng duy nhất.',
                           [_citation(snapshot,name,n,*m.span()) for n,m in claims])
    if composition:
        found=[]
        for n,line in enumerate(lines,1):
            original=(re.search(r'chủ yếu (?:là|đến từ) (.+?),\s*chiếm\s*(\d+(?:[,.]\d+)?)%\s*tổng doanh thu',line,re.I)
                      or re.search(r'Doanh thu chính[^.]{0,100}?đến từ (.+?)\s+chiếm\s*(\d+(?:[,.]\d+)?)%\s*tổng doanh thu',line,re.I))
            if original and (not year or year in line):
                found.append((n,original))
        if len(found)!=1:
            return _result(question,snapshot,name,'Chưa xác nhận được một khoản mục doanh thu chủ yếu và tỷ trọng cùng kỳ trong nguồn.',[])
        n,match=found[0]
        item=match[1].strip();share=match[2]
        return _result(question,snapshot,name,
            f'Doanh thu chủ yếu từ {item}, chiếm {share}% tổng doanh thu'+(f' năm {year}' if year else '')+
            f'. Nguồn: {source}, trang {snapshot.pages[name][n]}.',
            [_citation(snapshot,name,n,*match.span())],True)
    if internal_advance:
        found=[]
        for n,line in enumerate(lines,1):
            plain=normalize_text(line)
            if 'tam ung' not in plain or 'ke toan truong' not in plain:continue
            if year and year not in line:continue
            date=re.search(r'(?<!\d)(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})(?!\d)',question)
            if date and date[0] not in line:continue
            match=re.search(r'(?:khoảng|khoang)\s*(\d+(?:[,.]\d+)?\s*(?:tỷ|triệu)\s*(?:VND|VNĐ|đồng))',line,re.I)
            if match:found.append((n,match))
        if len(found)!=1:
            return _result(question,snapshot,name,'Chưa xác nhận được một khoản tạm ứng cho người nội bộ cùng ngày/kỳ trong nguồn.',[])
        n,match=found[0]
        amount=match[1].strip()
        return _result(question,snapshot,name,
            f'Báo cáo nêu khoản tạm ứng cho Kế toán trưởng khoảng {amount}. Nguồn: {source}, trang {snapshot.pages[name][n]}.',
            [_citation(snapshot,name,n,*match.span(1))],True)
    if audit_firm:
        found=[]
        for n,line in enumerate(lines,1):
            plain=normalize_text(line)
            if re.search(r'bao cao tai chinh[^|]{0,80}?duoc kiem toan boi\b',plain):
                match=re.search(r'kiểm toán bởi\s*:?\s*(Công ty[^|.]{3,160}?)(?=\s+được đính kèm|\s+cho năm|\s*\.|$)',line,re.I)
                if match and re.search(r'\b(?:cong ty|firm|co\.)\b',normalize_text(match[1])):
                    opinion_context=' '.join(lines[max(0,n-5):n])
                    if not year or year in opinion_context or title_years=={year}:
                        found.append((n,match[1].strip(),*match.span(1)))
            if re.match(r'^y kien cua kiem toan vien tai van ban\b',plain):
                match=re.search(r'(Công ty\s+[^.]{0,80}?Kiểm toán[^.]{0,100})\s*$',line,re.I)
                if match and (not year or year in line or title_years=={year}):
                    found.append((n,match[1].strip(),*match.span(1)))
            if re.fullmatch(r'(?:\d+ )?don vi kiem toan doc lap',plain):
                for following in range(n+1,min(len(lines)+1,n+5)):
                    if snapshot.pages[name][following]!=snapshot.pages[name][n]:break
                    candidate=lines[following-1]
                    match=re.search(r'(Công ty[^|]{4,160}?Kiểm toán[^|]{0,100})\s*$',candidate,re.I)
                    if match:
                        found.append((following,match[1].strip(),*match.span(1)))
                        break
                    if re.match(r'\d+\s+[a-z]',normalize_text(candidate)):break
        values={normalize_text(value) for _,value,_,_ in found}
        if len(values)!=1:
            return _result(question,snapshot,name,'Chưa xác nhận được một đơn vị kiểm toán duy nhất từ nguồn.',[])
        n,value,a,b=found[0]
        if audit_opinion:
            opinions=[]
            for other in range(max(1,n-5),min(len(lines),n+5)+1):
                if snapshot.pages[name][other]!=snapshot.pages[name][n]:continue
                match=re.search(r'Ý kiến kiểm toán\s*:\s*(Chấp nhận toàn phần|Ngoại trừ|Từ chối đưa ra ý kiến)',lines[other-1],re.I)
                if match:opinions.append((other,match))
            if len(opinions)!=1:
                return _result(question,snapshot,name,
                    'Đã xác nhận đơn vị kiểm toán nhưng chưa xác nhận một ý kiến cùng kỳ duy nhất.',
                    [_citation(snapshot,name,n,a,b)])
            other,opinion=opinions[0]
            return _result(question,snapshot,name,
                f'Đơn vị kiểm toán: {value}; ý kiến: {opinion[1]}. Nguồn: {source}, trang {snapshot.pages[name][n]}.',
                [_citation(snapshot,name,n,a,b),_citation(snapshot,name,other,*opinion.span(1))],True)
        return _result(question,snapshot,name,f'Đơn vị kiểm toán: {value}. Nguồn: {source}, trang {snapshot.pages[name][n]}.',
                       [_citation(snapshot,name,n,a,b)],True)
    if audit_opinion:
        qualified=[]
        for n,line in enumerate(lines,1):
            plain=normalize_text(line)
            if 'y kien ngoai tru' not in plain or 'kiem toan' not in plain:continue
            if re.search(r'khong (?:co|dua ra) y kien ngoai tru',plain):continue
            for following in range(n+1,min(len(lines)+1,n+5)):
                body=lines[following-1]
                normalized=normalize_text(body)
                if ('kiem toan vien' in normalized and 'dua ra y kien ngoai tru' in normalized
                    and (not year or year in body)):
                    qualified.append((n,following))
                    break
        if len(qualified)==1:
            heading,body=qualified[0]
            prefix='Có. ' if 'y kien ngoai tru' in q and ('khong' in q or 'co ' in q) else ''
            return _result(question,snapshot,name,
                f'{prefix}Ý kiến kiểm toán là ý kiến ngoại trừ; nguồn cũng nêu vấn đề cần nhấn mạnh. Nguồn: {source}, trang {snapshot.pages[name][body]}.',
                [_citation(snapshot,name,heading),_citation(snapshot,name,body)],True)
        if len(qualified)>1:
            return _result(question,snapshot,name,'Có nhiều đoạn ý kiến ngoại trừ chưa phân biệt được kỳ trong nguồn.',[])
        paragraph_qualified=[]
        for n,line in enumerate(lines,1):
            plain=normalize_text(line)
            if ('theo y kien' not in plain or 'ngoai tru' not in plain
                or 'bao cao tai chinh' not in plain or year and year not in line):
                continue
            before=' '.join(normalize_text(item) for item in lines[max(0,n-5):n-1])
            if 'y kien kiem toan' in before or 'y kien cua kiem toan vien' in before:
                paragraph_qualified.append(n)
        if len(paragraph_qualified)==1:
            n=paragraph_qualified[0]
            return _result(question,snapshot,name,
                f'Ý kiến kiểm toán là ý kiến ngoại trừ. Nguồn: {source}, trang {snapshot.pages[name][n]}.',
                [_citation(snapshot,name,n)],True)
        if len(paragraph_qualified)>1:
            return _result(question,snapshot,name,'Có nhiều đoạn ý kiến ngoại trừ chưa phân biệt được kỳ trong nguồn.',[])
        candidates=[]
        for n,line in enumerate(lines,1):
            plain=normalize_text(line)
            if not re.fullmatch(r'(?:(?:\d+|[ivxlcdm]+) )?y kien kiem toan(?: vien| doc lap)?',plain):
                continue
            for following in range(n+1,min(len(lines)+1,n+10)):
                paragraph=lines[following-1]
                content=normalize_text(paragraph)
                if not content or paragraph.startswith('## Page') or re.fullmatch(r'\d+',content):
                    continue
                if re.match(r'^y kien cua kiem toan vien tai van ban\b',content):continue
                if re.match(r'\d+ (?![0-9])',content) and 'theo y kien' not in content:
                    break
                if ('phan anh trung thuc va hop ly' in content and 'phu hop' in content
                    and not re.search(r'ngoai tru|tu choi|khong the dua ra|khong chap nhan',content)
                    and (not year or year in paragraph)):
                    candidates.append((n,following))
                break
        if len(candidates)!=1:
            return _result(question,snapshot,name,'Chưa xác định chắc chắn loại ý kiến kiểm toán từ đoạn ý kiến trong nguồn.',[])
        heading,body=candidates[0]
        return _result(question,snapshot,name,
            f'Ý kiến kiểm toán nêu báo cáo tài chính phản ánh trung thực, hợp lý và phù hợp quy định áp dụng; đây là ý kiến chấp nhận toàn phần. Nguồn: {source}, trang {snapshot.pages[name][body]}.',
            [_citation(snapshot,name,heading),_citation(snapshot,name,body)],True)
    if emphasis:
        hits=[n for n,line in enumerate(lines,1) if 'hoat dong lien tuc' in normalize_text(line)
              and ('nhan manh' in normalize_text(line) or
                   any('nhan manh' in normalize_text(lines[j]) for j in range(max(0,n-3),n-1)))]
        if not hits:
            return _result(question,snapshot,name,
                'Chưa tìm thấy đoạn nhấn mạnh về khả năng hoạt động liên tục trong Markdown; chưa đủ căn cứ để khẳng định báo cáo không có đoạn này.',[])
        return _result(question,snapshot,name,
            'Nguồn có nhắc đến đoạn nhấn mạnh về khả năng hoạt động liên tục; cần đọc nguyên văn để xác định nội dung.',
            [_citation(snapshot,name,hits[0])])
    # A report can disclose a transaction without proving it abnormal or unlawful.
    if re.search(r'bat thuong|vi pham|co hop ly|co rui ro',q):
        return _result(question,snapshot,name,
            'Các giao dịch được công bố chưa đủ để kết luận bất thường hoặc vi phạm; cần điều khoản, giá thị trường và bằng chứng tuân thủ để đánh giá.',[])
    negative=[]
    for n,line in enumerate(lines,1):
        plain=normalize_text(line)
        if (re.search(r'\b3 3\b',plain) and 'hop dong hoac giao dich' in plain
            and 'nguoi noi bo' in plain and 'khong phat sinh' in plain):
            negative.append(n)
    if negative and ('muc 3 3' in q or 'khong phat sinh' in q):
        if len(negative)!=1:
            return _result(question,snapshot,name,'Có nhiều mục công bố chưa phân biệt được trong nguồn.',[])
        n=negative[0]
        return _result(question,snapshot,name,
            f'Theo mục 3.3, không phát sinh hợp đồng hoặc giao dịch với người nội bộ và người có liên quan đến người nội bộ. Nguồn: {source}, trang {snapshot.pages[name][n]}.',
            [_citation(snapshot,name,n)],True)
    shareholder=[]
    for heading,line in enumerate(lines,1):
        if 'hop dong hoac giao dich voi co dong noi bo' not in normalize_text(line):continue
        for n in range(heading+1,min(len(lines)+1,heading+6)):
            body=lines[n-1];plain=normalize_text(body)
            if snapshot.pages[name][n]!=snapshot.pages[name][heading]:break
            if year and year not in body:continue
            if 'co dong lon' not in plain or 'ben lien quan' not in plain:continue
            amounts=list(re.finditer(r'\d[\d.,]*\s*(?:tỷ|triệu)\s*đồng',body,re.I))
            if len(amounts)>=2:shareholder.append((heading,n,amounts))
    if len(shareholder)==1:
        heading,n,amounts=shareholder[0]
        values=[match[0].strip() for match in amounts[:2]]
        cites=[_citation(snapshot,name,heading),_citation(snapshot,name,n)]
        return _result(question,snapshot,name,
            f'Có công bố giao dịch với cổ đông lớn và bên liên quan: {values[0]}; {values[1]}. Nguồn: {source}, trang {snapshot.pages[name][n]}.',
            cites,True)
    if len(shareholder)>1:
        return _result(question,snapshot,name,'Có nhiều công bố giao dịch chưa phân biệt được đối tượng/kỳ.',[])
    headings=[n for n,line in enumerate(lines,1) if 'hop dong hoac giao dich voi nguoi noi bo va nguoi co lien quan' in normalize_text(line)]
    examples=[]
    for heading in headings:
        for n in range(heading+1,min(len(lines)+1,heading+25)):
            line=lines[n-1]
            plain=normalize_text(line)
            if re.match(r'^\s*[d-g][).]',plain) or line.startswith('## Page') and n>heading+20:break
            if 'thue van phong cua' in plain or 'thue mat bang cua' in plain or 'ky hop dong cung cap dich vu voi' in plain:
                for j in range(n+1,min(len(lines)+1,n+4)):
                    amount=re.search(r'(?P<amount>\d[\d.,]*\s*(?:triệu đồng|VND|VNĐ|đồng)(?:\s*/\s*1?\s*tháng)?)',lines[j-1],re.I)
                    if amount:
                        relation=next((k for k in range(j+1,min(len(lines)+1,j+4))
                            if 'quan he lien quan' in normalize_text(lines[k-1])
                            and snapshot.pages[name][k]==snapshot.pages[name][j]),None)
                        if relation:examples.append((n,j,amount,relation))
                        break
    if examples:
        selected=examples[:2]
        parts=[];cites=[]
        for n,j,amount,relation in selected:
            action=lines[n-1].lstrip('\\- ').strip().rstrip(':')
            value=amount['amount'].strip()
            parts.append(f'{action}; giá trị công bố {value}')
            cites.extend([_citation(snapshot,name,headings[0]),_citation(snapshot,name,n),
                          _citation(snapshot,name,j,*amount.span('amount')),_citation(snapshot,name,relation)])
        return _result(question,snapshot,name,
            'Có công bố giao dịch với người có liên quan: '+'; '.join(parts)+
            f'. Nguồn: {source}, trang '+', '.join(map(str,sorted({c['page'] for c in cites})))+'.',
            cites,True)
    return _result(question,snapshot,name,
        'Chưa xác nhận được công bố giao dịch cùng đúng đối tượng và kỳ trong Markdown đã chọn.',[])


