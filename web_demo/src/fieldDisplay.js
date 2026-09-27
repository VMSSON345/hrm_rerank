// Compact only complete literal-field answers. Values remain exact source text.
const LABEL = /^(?:Tên giao dịch|Tên tiếng Anh|Tên viết tắt|Địa chỉ|Điện thoại|Fax|Website|Email|Mã chứng khoán|Ngành nghề kinh doanh|Người thực hiện công bố thông tin|Số lượng nhân viên(?: năm \d{4})?)$/;
export function fieldDisplay(raw) {
 const paragraphs=(raw || '').split('\n\n');
 if(paragraphs.length<2)return null;
 const fields=paragraphs.map(p=>p.match(/^([^:\n]+): ([\s\S]+?) Nguồn: ([^\n;]+), trang (\d+)\.$/));
 if(fields.some(x=>!x || !LABEL.test(x[1])))return null;
 const sources=new Map();
 for(const f of fields){if(!sources.has(f[3]))sources.set(f[3],new Set());sources.get(f[3]).add(f[4]);}
 return {text:fields.map(f=>`${f[1]}: ${f[2]}`).join('\n\n'),sourceLabel:[...sources].map(([name,pages])=>`${name} · trang ${[...pages].join(', ')}`).join('; '),details:''};
}
