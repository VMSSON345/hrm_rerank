"""Shared bounded metric identity; related is never synonymous by score alone."""
import re
from contextvars import ContextVar
from functools import wraps
_SOURCE_METRIC_LABELS=ContextVar("source_metric_labels",default=())

def isolated_metric_labels(function):
    @wraps(function)
    def call(*args,**kwargs):
        token=_SOURCE_METRIC_LABELS.set(())
        try:return function(*args,**kwargs)
        finally:_SOURCE_METRIC_LABELS.reset(token)
    return call

def set_source_metric_labels(labels):
    _SOURCE_METRIC_LABELS.set(tuple(dict.fromkeys(x for x in labels if x)))
from .ontology import normalize_text

ALIASES={
    "accumulated_loss":("lo luy ke", "khoan lo luy ke"),
    "overdue_liabilities":("no phai tra qua han",),
    "salary_cost":("chi phi tien luong", "chi phi luong", "tien luong"),
    "profit_loss":("lai lo", "lai (lo)"),
    "gross_profit":("loi nhuan gop", "loi nhuan gop ve ban hang va cung cap dich vu"),
    "charter_capital":("von dieu le", "charter capital", "von dieu le charter capital"),
    "owner_invested_capital":("von dau tu cua chu so huu",),
    "owners_equity":("von chu so huu", "owners equity", "shareholders equity"),
    "net_revenue":("doanh thu thuan","doanh thu thuan ban hang dich vu","doanh thu thuan ban hang va dich vu",
                   "doanh thu thuan ve ban hang va cung cap dich vu","doanh thu thuan ban hang va cung cap dich vu","net revenue","net sales"),
    "current_assets":("tai san ngan han", "tai san luu dong", "current assets"),
    "current_liabilities":("no ngan han", "current liabilities"),
    "total_assets":("tong tai san","tong gia tri tai san","total assets"),
    "total_liabilities":("no phai tra", "tong no phai tra", "no phai tra thoi diem cuoi", "total liabilities"),
    "operating_cash_flow":("tien thuan tu hoat dong kinh doanh", "dong tien thuan tu hoat dong kinh doanh", "luu chuyen tien thuan tu hoat dong kinh doanh", "net cash from operating activities", "net operating cash flow"),
    "current_ratio":("he so thanh toan hien hanh", "kha nang thanh toan hien hanh", "he so kha nang thanh toan ngan han", "he so thanh toan ngan han", "tsld no ngan han", "thanh toan hien hanh", "thanh toan hien hanh tsnh no nh", "current ratio"),
    "quick_ratio":("he so thanh toan nhanh", "kha nang thanh toan nhanh", "tsnh hang ton kho no ngan han", "thanh toan nhanh", "thanh toan nhanh tsnh htk no nh", "quick ratio"),
    "debt_to_assets_ratio":("he so no tren tong tai san", "he so no tong tai san", "debt to assets ratio"),
    "debt_to_equity_ratio":("he so no tren von chu so huu", "he so no von chu so huu", "debt to equity ratio", "d e"),
    "total_asset_turnover":("vong quay tong tai san", "total asset turnover"),
    "return_on_assets":("roa", "chi so roa", "roa loi nhuan sau thue tong tai san", "he so loi nhuan sau thue tong tai san", "ty suat loi nhuan sau thue tong tai san", "ty suat loi nhuan sau thue tren tong tai san", "return on assets"),
    "return_on_equity":("roe", "chi so roe", "roe loi nhuan sau thue von chu so huu", "ty suat loi nhuan sau thue tren von chu so huu", "he so loi nhuan sau thue von chu so huu", "return on equity"),
    "gross_margin":("bien loi nhuan gop", "gross margin"),
    "net_margin":("bien loi nhuan rong", "net margin"),
    "average_collection_period":("ky thu tien binh quan", "so ngay thu tien binh quan", "days sales outstanding"),
    "inventory_turnover":("vong quay hang ton kho", "gia von hang ban hang ton kho binh quan", "inventory turnover"),
    "revenue_unspecified":("doanh thu","revenue","sales"),
    "total_revenue":("tong doanh thu","total revenue"),
    "profit_after_tax_parent":("loi nhuan sau thue cua cong ty me", "loi nhuan sau thue thuoc ve co dong cong ty me", "loi nhuan sau thue thuoc ve co dong cua cong ty me"),
    "profit_after_tax":("loi nhuan sau thue","loi nhuan sau thue thu nhap doanh nghiep","profit after tax","profit after income tax","profit after corporate income tax","net profit after tax","earnings after tax","lnst"),
    "profit_before_tax":("loi nhuan truoc thue","loi nhuan ke toan truoc thue","tong loi nhuan truoc thue","tong loi nhuan ke toan truoc thue","profit before tax","profit before income tax","profit before corporate income tax","earnings before tax","lntt"),
}
LOOKUP={v:k for k,values in ALIASES.items() for v in values}

# Expand explicit financial phrases only; a bare "LN" remains ambiguous.
# Preserve owner, segment and tax qualifiers, and never rewrite source witnesses.
_METRIC_SHORT_FORMS = {
    "tndn": "thu nhap doanh nghiep",
    "ln truoc thue": "loi nhuan truoc thue",
    "ln sau thue": "loi nhuan sau thue",
    "lntt": "loi nhuan truoc thue",
    "lnst": "loi nhuan sau thue",
    "ln gop": "loi nhuan gop",
}
_SHORT_FORM_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(x) for x in _METRIC_SHORT_FORMS) + r")\b"
)


def normalize_metric_text(text):
    """Shared vocabulary for query/source matching, not source evidence."""
    normalized = normalize_text(text)
    # MinerU occasionally joins a common two-word financial collocation.
    # Repair the label only; source text and numeric witnesses stay untouched.
    normalized = re.sub(r"\bcungcap\b", "cung cap", normalized)
    return _SHORT_FORM_PATTERN.sub(
        lambda match: _METRIC_SHORT_FORMS[match[0]], normalized
    )


def row_code_equation_span(label):
    """Only numeric row-code equations, never financial values or text notes."""
    equation = re.search(r"\s*[({]\s*\d{1,3}\s*=\s*[\d+*/().\s-]+[)}]\s*$", label)
    return equation.span() if equation and all(len(token) <= 3 for token in re.findall(r"\d+", equation[0])) else None


def normalized_label(label):
    label=re.sub(r"^\s*(?:\d+|[IVXLCDM]+|[A-Z])\s*[-.)]\s*(?=[^\W\d_])","",label)
    def unit(match):
        return " " if re.fullmatch(r"(?:(?:nghin|trieu|ty) )?(?:dong|vnd)",normalize_text(match[1])) else match[0]
    label=re.sub(r"\(([^()]*)\)",unit,label)
    equation=row_code_equation_span(label)
    if equation:label=label[:equation[0]]
    return normalize_metric_text(label)

def identity(label):
    text=normalized_label(label)
    return LOOKUP.get(text,"literal:"+text)

def relation(expected,observed):
    left,right=identity(expected),identity(observed)
    if left==right:return "SAME"
    if left in ALIASES and right in ALIASES:
        if left.startswith("profit_") and right.startswith("profit_"):return "DIFFERENT"
        if {left,right}<={"net_revenue","revenue_unspecified","total_revenue"}:return "RELATED"
        return "DIFFERENT"
    # A qualified row cannot inherit an unqualified concept identity merely
    # because it contains the words (e.g. net revenue for export or a segment).
    l,r=normalized_label(expected),normalized_label(observed)
    # "Other" names a residual category, not a spelling variant of its total.
    # Compare both directions; lexical overlap must not erase this qualifier.
    if bool(re.search(r"\b(?:khac|other)\b", l)) != bool(re.search(r"\b(?:khac|other)\b", r)):
        return "DIFFERENT"
    if (left in ALIASES or right in ALIASES) and (l in r or r in l):return "RELATED"
    return "UNCLASSIFIED"


def report_field_identity(label):
    """A referential field label can explicitly name the reporting company.

    Only the complete 'equivalent to X of the company' construction is
    unwrapped; 'cash equivalents', subsidiaries and named owners stay distinct.
    """
    text = normalized_label(label)
    match = re.fullmatch(r"tuong duong (.+?) (?:cua )?cong ty", text)
    if match and match[1] not in {"tien", "tien va cac khoan"}:
        text = match[1]
    return identity(text)


# A metric is a positive lexical span. Unknown qualifiers around it remain in
# the result; they cannot be erased just to make a known alias match.
_QUERY_PREFIX = re.compile(r"^(?:(?:cho toi(?: biet| so)?|cho biet|hay|xin|vui long|cong ty(?: da)?(?: co| ghi| ghi nhan| chi| bat dau voi)?|muc|so|khoan|voi muc|bao nhieu|duoc neu|trong dinh huong|tai|den)\s*)*$")
_QUERY_SUFFIX = re.compile(r"^(?:(?:o muc nao|muc nao|dat muc nao|bao nhieu|thi(?: duoc)?|la|dat|duoc neu|duoc ghi|duoc ghi nhan|cua|cong ty|tinh|nam sau tru nam truoc|nam truoc tru nam sau|nao)\s*)*$")

def metric_span(text, source_labels=()):
    """Return a source/vocabulary span only when all outside text is framing."""
    plain=normalize_metric_text(text).strip(" ,?.")
    labels=set(LOOKUP) | {normalized_label(x) for x in (*_SOURCE_METRIC_LABELS.get(),*source_labels) if x}
    matches=[]
    for label in labels:
        for hit in re.finditer(r"(?<!\w)"+re.escape(label)+r"(?!\w)",plain):
            matches.append((len(label),hit.start(),hit.end(),label))
    if not matches:return plain
    _,start,end,label=max(matches)
    before=plain[:start].strip(" ,:;?");after=plain[end:].strip(" ,:;?")
    if _QUERY_PREFIX.fullmatch(before) and _QUERY_SUFFIX.fullmatch(after):
        return label
    # Preserve residual qualifiers such as khac, xuat khau, dieu chinh and owners.
    return plain


# Display vocabulary is separate from physical labels and source witnesses.
DISPLAY_LABELS = {
    "profit_after_tax_parent": "Lợi nhuận sau thuế của công ty mẹ",
    "profit_after_tax": "Lợi nhuận sau thuế",
    "profit_before_tax": "Lợi nhuận trước thuế",
    "net_revenue": "Doanh thu thuần",
    "total_revenue": "Tổng doanh thu",
    "revenue_unspecified": "Doanh thu",
    "gross_profit": "Lợi nhuận gộp",
    "charter_capital": "Vốn điều lệ",
    "total_assets": "Tổng tài sản",
    "current_assets": "Tài sản ngắn hạn",
    "owners_equity": "Vốn chủ sở hữu",
    "total_liabilities": "Nợ phải trả",
    "operating_cash_flow": "Dòng tiền thuần từ hoạt động kinh doanh",
    "current_ratio": "Hệ số thanh toán hiện hành",
    "quick_ratio": "Hệ số thanh toán nhanh",
    "return_on_assets": "ROA",
    "return_on_equity": "ROE",
    "gross_margin": "Biên lợi nhuận gộp",
    "net_margin": "Biên lợi nhuận ròng",
    "average_collection_period": "Kỳ thu tiền bình quân",
    "inventory_turnover": "Vòng quay hàng tồn kho",
}

def display_label(label):
    return DISPLAY_LABELS.get(identity(label), label)



