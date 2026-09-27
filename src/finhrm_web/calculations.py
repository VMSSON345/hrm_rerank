"""Reference arithmetic, exclusively for explicitly entered values (million VND)."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
LABEL_VI = {
    "tien_va_tuong_duong_tien": "Tiền và tương đương tiền",
    "tai_san_ngan_han": "Tài sản ngắn hạn",
    "phai_thu_ngan_han": "Các khoản phải thu ngắn hạn",
    "phai_thu_khach_hang": "Phải thu khách hàng",
    "hang_ton_kho": "Hàng tồn kho",
    "tai_san_co_dinh": "Tài sản cố định",
    "chi_phi_xay_dung_co_ban_do_dang": "Chi phí xây dựng cơ bản dở dang",
    "chi_phi_tra_truoc_dai_han": "Chi phí trả trước dài hạn",
    "tong_tai_san": "Tổng tài sản",
    "no_ngan_han": "Nợ ngắn hạn",
    "phai_tra_nguoi_ban": "Phải trả người bán",
    "vay_ngan_han": "Vay và nợ thuê tài chính ngắn hạn",
    "vay_dai_han": "Vay và nợ thuê tài chính dài hạn",
    "no_phai_tra": "Nợ phải trả",
    "von_chu_so_huu": "Vốn chủ sở hữu",
    "doanh_thu_thuan": "Doanh thu thuần",
    "doanh_thu_chua_thuc_hien": "Doanh thu chưa thực hiện",
    "loi_nhuan_gop": "Lợi nhuận gộp",
    "chi_phi_lai_vay": "Chi phí lãi vay",
    "loi_nhuan_truoc_thue": "Lợi nhuận trước thuế",
    "loi_nhuan_sau_thue": "Lợi nhuận sau thuế",
    "luu_chuyen_tien_thuan_tu_hoat_dong_kinh_doanh": "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
    "luu_chuyen_tien_thuan_tu_hoat_dong_dau_tu": "Lưu chuyển tiền thuần từ hoạt động đầu tư",
    "luu_chuyen_tien_thuan_tu_hoat_dong_tai_chinh": "Lưu chuyển tiền thuần từ hoạt động tài chính",
    "luu_chuyen_tien_thuan_trong_ky": "Lưu chuyển tiền thuần trong kỳ",
}


@dataclass
class LineItem:
    key: str
    label: str
    value: Optional[float]
    unit: str = "triệu đồng"
    confidence: str = "auto"  # "auto" | "manual" (sau khi người dùng sửa tay)
    source_page: Optional[int] = None


@dataclass
class FinancialExtractionResult:
    line_items: Dict[str, LineItem] = field(default_factory=dict)

    def get(self, key: str) -> Optional[float]:
        item = self.line_items.get(key)
        return item.value if item else None


@dataclass
class Ratio:
    key: str
    label: str
    value: Optional[float]
    unit: str
    benchmark_note: str
    flag: Optional[str] = None  # "warning" | "danger" | None


def _safe_div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def compute_ratios(fe: FinancialExtractionResult) -> List[Ratio]:
    g = fe.get
    tsnh = g("tai_san_ngan_han")
    hang_ton_kho = g("hang_ton_kho")
    no_ngan_han = g("no_ngan_han")
    tong_ts = g("tong_tai_san")
    no_pt = g("no_phai_tra")
    vcsh = g("von_chu_so_huu")
    dtt = g("doanh_thu_thuan")
    lnst = g("loi_nhuan_sau_thue")
    lntt = g("loi_nhuan_truoc_thue")
    cp_lai_vay = g("chi_phi_lai_vay")
    tien = g("tien_va_tuong_duong_tien")
    phai_thu = g("phai_thu_khach_hang") or g("phai_thu_ngan_han")
    cfo = g("luu_chuyen_tien_thuan_tu_hoat_dong_kinh_doanh")
    vay_ngan_han = g("vay_ngan_han")
    vay_dai_han = g("vay_dai_han")

    ratios: List[Ratio] = []

    current_ratio = _safe_div(tsnh, no_ngan_han)
    ratios.append(Ratio(
        "current_ratio", "Hệ số thanh toán hiện hành", current_ratio, "lần",
        "Thường >1 được coi là an toàn ở mức cơ bản; <1 cảnh báo áp lực thanh khoản ngắn hạn.",
        flag="danger" if current_ratio is not None and current_ratio < 1 else
             ("warning" if current_ratio is not None and current_ratio < 1.2 else None),
    ))

    quick_ratio = _safe_div(
        (tsnh - hang_ton_kho) if (tsnh is not None and hang_ton_kho is not None) else None,
        no_ngan_han,
    )
    ratios.append(Ratio(
        "quick_ratio", "Hệ số thanh toán nhanh", quick_ratio, "lần",
        "Loại trừ hàng tồn kho — phản ánh khả năng trả nợ ngắn hạn thực chất hơn.",
        flag="warning" if quick_ratio is not None and quick_ratio < 0.8 else None,
    ))

    debt_to_equity = _safe_div(no_pt, vcsh)
    ratios.append(Ratio(
        "debt_to_equity", "Nợ phải trả / Vốn chủ sở hữu", debt_to_equity, "lần",
        "Trên 2-3 lần tuỳ ngành cần xem xét kỹ mức độ phụ thuộc vốn vay.",
        flag="danger" if debt_to_equity is not None and debt_to_equity > 3 else
             ("warning" if debt_to_equity is not None and debt_to_equity > 2 else None),
    ))

    net_margin = _safe_div(lnst, dtt)
    net_margin_pct = net_margin * 100 if net_margin is not None else None
    ratios.append(Ratio(
        "net_margin", "Biên lợi nhuận ròng", net_margin_pct, "%",
        "Tỷ lệ lợi nhuận sau thuế trên doanh thu thuần.",
        flag="warning" if net_margin_pct is not None and net_margin_pct < 0 else None,
    ))

    roa = _safe_div(lnst, tong_ts)
    roa_pct = roa * 100 if roa is not None else None
    ratios.append(Ratio(
        "roa", "ROA (Lợi nhuận / Tổng tài sản)", roa_pct, "%",
        "Hiệu quả sử dụng tổng tài sản để sinh lời.",
    ))

    roe = _safe_div(lnst, vcsh)
    roe_pct = roe * 100 if roe is not None else None
    ratios.append(Ratio(
        "roe", "ROE (Lợi nhuận / Vốn chủ sở hữu)", roe_pct, "%",
        "Hiệu quả sinh lời trên vốn chủ sở hữu.",
    ))

    interest_coverage = _safe_div(lntt, cp_lai_vay)
    ratios.append(Ratio(
        "interest_coverage", "Hệ số khả năng trả lãi vay (EBIT/Chi phí lãi vay, xấp xỉ)",
        interest_coverage, "lần",
        "Dưới 1.5-2 lần cho thấy khả năng trả lãi vay yếu.",
        flag="danger" if interest_coverage is not None and interest_coverage < 1.5 else
             ("warning" if interest_coverage is not None and interest_coverage < 2.5 else None),
    ))

    cash_ratio = _safe_div(tien, no_ngan_han)
    ratios.append(Ratio(
        "cash_ratio", "Hệ số thanh toán bằng tiền", cash_ratio, "lần",
        "Khả năng trang trải nợ ngắn hạn chỉ bằng tiền và tương đương tiền; cần đọc cùng dòng tiền kinh doanh.",
        flag="warning" if cash_ratio is not None and cash_ratio < 0.05 else None,
    ))

    working_capital = (tsnh - no_ngan_han) if tsnh is not None and no_ngan_han is not None else None
    ratios.append(Ratio(
        "net_working_capital", "Vốn lưu động ròng", working_capital, "triệu đồng",
        "Tài sản ngắn hạn trừ nợ ngắn hạn; số âm cho thấy nguồn vốn ngắn hạn đang thiếu hụt.",
        flag="danger" if working_capital is not None and working_capital < 0 else None,
    ))

    receivable_days = _safe_div(phai_thu * 365 if phai_thu is not None else None, dtt)
    ratios.append(Ratio(
        "receivable_days", "Số ngày phải thu (ước tính)", receivable_days, "ngày",
        "Ước tính từ số dư cuối kỳ/doanh thu; cần so sánh nhiều kỳ và đối chiếu tuổi nợ chi tiết.",
        flag="warning" if receivable_days is not None and receivable_days > 90 else None,
    ))

    inventory_days = _safe_div(hang_ton_kho * 365 if hang_ton_kho is not None else None, dtt)
    ratios.append(Ratio(
        "inventory_days", "Số ngày tồn kho (ước tính)", inventory_days, "ngày",
        "Ước tính theo số dư cuối kỳ/doanh thu; cần điều chỉnh theo ngành và giá vốn khi có dữ liệu.",
        flag="warning" if inventory_days is not None and inventory_days > 120 else None,
    ))

    operating_cash_to_profit = _safe_div(cfo, lnst)
    ratios.append(Ratio(
        "operating_cash_to_profit", "Dòng tiền kinh doanh / LNST", operating_cash_to_profit, "lần",
        "Đo chất lượng lợi nhuận. Dưới 1 hoặc âm cần kiểm tra doanh thu, công nợ và vốn lưu động.",
        flag="danger" if operating_cash_to_profit is not None and operating_cash_to_profit < 0 else
             ("warning" if operating_cash_to_profit is not None and operating_cash_to_profit < 0.8 else None),
    ))

    debt_service_capacity = _safe_div(cfo, vay_ngan_han or no_ngan_han)
    ratios.append(Ratio(
        "debt_service_capacity", "Dòng tiền kinh doanh / Nợ ngắn hạn", debt_service_capacity, "lần",
        "Chỉ báo sơ bộ về khả năng tự tạo tiền để đáp ứng nghĩa vụ ngắn hạn; không thay thế lịch trả nợ thực tế.",
        flag="danger" if debt_service_capacity is not None and debt_service_capacity < 0 else
             ("warning" if debt_service_capacity is not None and debt_service_capacity < 0.3 else None),
    ))

    total_borrowings = None
    if vay_ngan_han is not None or vay_dai_han is not None:
        total_borrowings = (vay_ngan_han or 0) + (vay_dai_han or 0)
    borrowings_to_assets = _safe_div(total_borrowings, tong_ts)
    borrowings_to_assets_pct = borrowings_to_assets * 100 if borrowings_to_assets is not None else None
    ratios.append(Ratio(
        "borrowings_to_assets", "Dư nợ vay / Tổng tài sản", borrowings_to_assets_pct, "%",
        "Phản ánh mức độ sử dụng vốn vay khi PDF có tách riêng vay ngắn hạn/dài hạn.",
        flag="warning" if borrowings_to_assets_pct is not None and borrowings_to_assets_pct > 50 else None,
    ))

    return ratios


def simulate_stress_test(
    fe: FinancialExtractionResult,
    *,
    interest_rate_shock_pct: float = 0,
    exchange_rate_shock_pct: float = 0,
    cashflow_decline_pct: float = 0,
    cycle_extension_days: float = 0,
    foreign_currency_debt: float = 0,
) -> Dict:
    """Stress test đơn giản, minh bạch giả định và không thay thế mô hình PD/LGD.

    - Lãi suất tác động lên chi phí lãi vay hiện hữu.
    - Tỷ giá tác động lên dư nợ ngoại tệ do cán bộ nhập (vì BCTC không luôn tách rõ).
    - Chu kỳ kinh doanh kéo dài làm chậm phần tiền thu về từ khoản phải thu cuối kỳ.
    """
    g = fe.get
    cfo = g("luu_chuyen_tien_thuan_tu_hoat_dong_kinh_doanh")
    lntt = g("loi_nhuan_truoc_thue")
    interest_cost = g("chi_phi_lai_vay")
    receivables = g("phai_thu_khach_hang") or g("phai_thu_ngan_han")
    short_debt = g("vay_ngan_han") or g("no_ngan_han")

    additional_interest = (interest_cost or 0) * interest_rate_shock_pct / 100
    fx_loss = max(foreign_currency_debt, 0) * exchange_rate_shock_pct / 100
    collection_delay_impact = max(receivables or 0, 0) * max(cycle_extension_days, 0) / 365
    stressed_cfo = None if cfo is None else cfo * (1 - cashflow_decline_pct / 100) - collection_delay_impact
    stressed_pbt = None if lntt is None else lntt - additional_interest - fx_loss
    stressed_interest = (interest_cost or 0) + additional_interest
    stressed_interest_coverage = _safe_div(stressed_pbt, stressed_interest)
    stressed_debt_service = _safe_div(stressed_cfo, short_debt)

    missing = []
    if cfo is None:
        missing.append("Chưa có LCTT từ hoạt động kinh doanh")
    if lntt is None:
        missing.append("Chưa có lợi nhuận trước thuế")
    if interest_cost is None:
        missing.append("Chưa có chi phí lãi vay")
    if short_debt is None:
        missing.append("Chưa có nợ ngắn hạn/dư nợ vay ngắn hạn")

    risk = "Tích cực"
    if stressed_cfo is not None and stressed_cfo < 0 or stressed_interest_coverage is not None and stressed_interest_coverage < 1.5:
        risk = "Rủi ro cao"
    elif stressed_debt_service is not None and stressed_debt_service < 0.3 or stressed_interest_coverage is not None and stressed_interest_coverage < 2.5:
        risk = "Cần theo dõi"
    elif missing:
        risk = "Chưa đủ dữ liệu"

    conclusions = []
    if stressed_cfo is not None:
        conclusions.append(f"Dòng tiền HĐKD sau stress: {stressed_cfo:,.0f} triệu đồng.")
    if stressed_interest_coverage is not None:
        conclusions.append(f"Khả năng trả lãi sau stress: {stressed_interest_coverage:.2f} lần.")
    if stressed_debt_service is not None:
        conclusions.append(f"Dòng tiền HĐKD/nợ ngắn hạn sau stress: {stressed_debt_service:.2f} lần.")
    if not conclusions:
        conclusions.append("Không đủ dữ liệu số để tính kịch bản; bổ sung BCLCTT, lịch trả nợ và chi phí lãi vay.")

    return {
        "risk_level": risk,
        "assumptions": {
            "interest_rate_shock_pct": interest_rate_shock_pct,
            "exchange_rate_shock_pct": exchange_rate_shock_pct,
            "cashflow_decline_pct": cashflow_decline_pct,
            "cycle_extension_days": cycle_extension_days,
            "foreign_currency_debt": foreign_currency_debt,
        },
        "baseline": {
            "operating_cashflow": cfo,
            "profit_before_tax": lntt,
            "interest_cost": interest_cost,
            "short_term_debt": short_debt,
        },
        "stressed": {
            "operating_cashflow": stressed_cfo,
            "profit_before_tax": stressed_pbt,
            "interest_cost": stressed_interest if interest_cost is not None else None,
            "additional_interest": additional_interest if interest_cost is not None else None,
            "fx_loss": fx_loss if foreign_currency_debt else 0,
            "collection_delay_impact": collection_delay_impact if receivables is not None else None,
            "interest_coverage": stressed_interest_coverage,
            "debt_service_capacity": stressed_debt_service,
        },
        "missing_inputs": missing,
        "conclusions": conclusions,
        "disclaimer": "Kịch bản nhạy cảm chỉ phục vụ thẩm định sơ bộ; cần thay bằng dự báo dòng tiền, lịch trả nợ và giả định đã được phê duyệt khi trình tín dụng.",
    }
