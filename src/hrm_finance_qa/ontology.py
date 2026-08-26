from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class VariableSpec:
    key: str
    label: str
    item_ids: tuple[str, ...]
    statement: str | None
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class FormulaSpec:
    key: str
    label: str
    aliases: tuple[str, ...]
    variables: tuple[str, ...]
    expression: str
    unit: str
    note: str = ""


_PDF_UNICODE_GLYPH_RE = re.compile(r"/uni([0-9a-fA-F]{4})")


def decode_pdf_unicode_glyph_names(value: str) -> str:
    """Decode glyph names leaked by broken PDF ToUnicode maps.

    Some otherwise usable Vietnamese text layers contain fragments such as
    ``T/uni1ED4NG`` instead of ``TỔNG``.  Decode only the unambiguous Adobe
    ``/uniXXXX`` form; malformed names are deliberately left untouched.
    """

    return _PDF_UNICODE_GLYPH_RE.sub(
        lambda match: chr(int(match.group(1), 16)), value
    )


def normalize_text(value: str) -> str:
    value = decode_pdf_unicode_glyph_names(value)
    value = value.replace("đ", "d").replace("Đ", "D")
    value = "".join(
        char
        for char in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


VARIABLES: dict[str, VariableSpec] = {
    "net_sales": VariableSpec(
        "net_sales",
        "Doanh thu thuan",
        ("net_sales", "net_sales_from_insurance_business", "sales"),
        "income_statement",
        ("doanh thu thuan", "net sales", "revenue"),
    ),
    "sales": VariableSpec(
        "sales",
        "Doanh thu ban hang",
        ("sales",),
        "income_statement",
        ("doanh thu ban hang", "gross sales"),
    ),
    "sales_deductions": VariableSpec(
        "sales_deductions",
        "Cac khoan giam tru doanh thu",
        ("sales_deductions",),
        "income_statement",
        ("giam tru doanh thu", "sales deductions"),
    ),
    "cost_of_sales": VariableSpec(
        "cost_of_sales",
        "Gia von hang ban",
        ("cost_of_sales",),
        "income_statement",
        ("gia von", "cost of sales", "cogs"),
    ),
    "gross_profit": VariableSpec(
        "gross_profit",
        "Loi nhuan gop",
        ("gross_profit",),
        "income_statement",
        ("loi nhuan gop", "gross profit"),
    ),
    "net_profit": VariableSpec(
        "net_profit",
        "Loi nhuan sau thue",
        (
            "net_profit_loss_after_tax",
            "profit_after_tax",
            "attributable_to_parent_company",
        ),
        "income_statement",
        ("loi nhuan sau thue", "lai rong", "net profit", "net income"),
    ),
    "operating_profit": VariableSpec(
        "operating_profit",
        "Loi nhuan tu hoat dong kinh doanh",
        ("operating_profit_loss", "net_operating_profit_from_insurance_operation"),
        "income_statement",
        ("loi nhuan hoat dong", "operating profit", "ebit"),
    ),
    "interest_expense": VariableSpec(
        "interest_expense",
        "Chi phi lai vay",
        ("interest_expenses", "interests_expenses"),
        "income_statement",
        ("chi phi lai vay", "interest expense"),
    ),
    "depreciation": VariableSpec(
        "depreciation",
        "Khau hao",
        ("depreciation_and_amortization", "depreciation_of_fixed_assets"),
        "cash_flow",
        ("khau hao", "depreciation", "amortization"),
    ),
    "total_assets": VariableSpec(
        "total_assets",
        "Tong tai san",
        ("total_assets",),
        "balance_sheet",
        ("tong tai san", "total assets"),
    ),
    "current_assets": VariableSpec(
        "current_assets",
        "Tai san ngan han",
        ("current_assets",),
        "balance_sheet",
        ("tai san ngan han", "current assets"),
    ),
    "liabilities": VariableSpec(
        "liabilities",
        "No phai tra",
        ("liabilities", "total_liabilities"),
        "balance_sheet",
        ("no phai tra", "total liabilities", "liabilities"),
    ),
    "current_liabilities": VariableSpec(
        "current_liabilities",
        "No ngan han",
        ("current_liabilities",),
        "balance_sheet",
        ("no ngan han", "current liabilities"),
    ),
    "owners_equity": VariableSpec(
        "owners_equity",
        "Von chu so huu",
        ("owners_equity",),
        "balance_sheet",
        ("von chu so huu", "equity", "owners equity"),
    ),
    "debt_repayment": VariableSpec(
        "debt_repayment",
        "Tien tra no goc vay",
        ("repayment_of_loans", "repayment_of_borrowings"),
        "cash_flow",
        ("tra no goc vay", "debt repayment", "repayment of loans"),
    ),
}


FORMULAS: dict[str, FormulaSpec] = {
    "gross_margin": FormulaSpec(
        "gross_margin",
        "Biên lợi nhuận gộp",
        ("biên lợi nhuận gộp", "tỷ suất lợi nhuận gộp", "gross margin"),
        ("net_sales", "gross_profit", "cost_of_sales"),
        "gross_profit / net_sales",
        "percent",
        "Nếu thiếu gross_profit, hệ thống suy ra net_sales + cost_of_sales theo giá trị có dấu trong CSV.",
    ),
    "net_margin": FormulaSpec(
        "net_margin",
        "Biên lợi nhuận ròng",
        ("biên lợi nhuận ròng", "biên lãi ròng", "biên lợi nhuận sau thuế", "net margin"),
        ("net_profit", "net_sales"),
        "net_profit / net_sales",
        "percent",
    ),
    "roa": FormulaSpec(
        "roa",
        "ROA",
        (
            "roa",
            "return on assets",
            "tỷ suất sinh lời trên tài sản",
            "tỷ suất lợi nhuận trên tổng tài sản",
            "lợi nhuận trên tài sản",
        ),
        ("net_profit", "total_assets"),
        "net_profit / total_assets",
        "percent",
        "MVP dùng tổng tài sản cuối kỳ, không phải tài sản bình quân.",
    ),
    "roe": FormulaSpec(
        "roe",
        "ROE",
        (
            "roe",
            "return on equity",
            "tỷ suất sinh lời trên vốn chủ",
            "tỷ suất lợi nhuận trên vốn chủ sở hữu",
            "lợi nhuận trên vốn chủ sở hữu",
        ),
        ("net_profit", "owners_equity"),
        "net_profit / owners_equity",
        "percent",
        "MVP dùng vốn chủ sở hữu cuối kỳ, không phải vốn bình quân.",
    ),
    "current_ratio": FormulaSpec(
        "current_ratio",
        "Hệ số thanh toán hiện hành",
        (
            "hệ số thanh toán hiện hành",
            "khả năng thanh toán hiện hành",
            "khả năng thanh toán ngắn hạn",
            "hệ số thanh toán ngắn hạn",
            "thanh khoản hiện hành",
            "current ratio",
        ),
        ("current_assets", "current_liabilities"),
        "current_assets / current_liabilities",
        "ratio",
    ),
    "debt_to_equity": FormulaSpec(
        "debt_to_equity",
        "Nợ trên vốn chủ sở hữu",
        (
            "nợ trên vốn chủ",
            "hệ số nợ trên vốn chủ sở hữu",
            "tỷ số nợ trên vốn chủ sở hữu",
            "debt to equity",
            "d e",
        ),
        ("liabilities", "owners_equity"),
        "liabilities / owners_equity",
        "ratio",
    ),
    "debt_ratio": FormulaSpec(
        "debt_ratio",
        "Hệ số nợ trên tài sản",
        ("hệ số nợ", "tỷ lệ nợ", "tỷ số nợ", "debt ratio", "nợ trên tài sản", "d a"),
        ("liabilities", "total_assets"),
        "liabilities / total_assets",
        "percent",
    ),
    "ebitda": FormulaSpec(
        "ebitda",
        "EBITDA ước tính",
        ("ebitda",),
        ("operating_profit", "depreciation"),
        "operating_profit + abs(depreciation)",
        "currency",
        "Ước tính từ lợi nhuận hoạt động và khấu hao trong bộ dữ liệu.",
    ),
    "dscr": FormulaSpec(
        "dscr",
        "DSCR ước tính",
        ("dscr", "debt service coverage ratio", "he so kha nang tra no"),
        ("operating_profit", "depreciation", "interest_expense", "debt_repayment"),
        "(operating_profit + abs(depreciation)) / (abs(interest_expense) + abs(debt_repayment))",
        "ratio",
        "Biến thể gần đúng do CSV không có debt service theo hợp đồng tín dụng.",
    ),
}


def resolve_metric(value: str) -> FormulaSpec | None:
    normalized = normalize_text(value)
    if normalized in FORMULAS:
        return FORMULAS[normalized]
    candidates = sorted(
        (
            (normalize_text(alias), formula)
            for formula in FORMULAS.values()
            for alias in (formula.key, *formula.aliases)
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    padded = f" {normalized} "
    for alias, formula in candidates:
        if f" {alias} " in padded:
            return formula
    return None


def resolve_variable(value: str) -> VariableSpec | None:
    normalized = normalize_text(value)
    if normalized in VARIABLES:
        return VARIABLES[normalized]
    for variable in VARIABLES.values():
        aliases = (variable.key, *variable.item_ids, *variable.aliases)
        if any(normalized == normalize_text(alias) for alias in aliases):
            return variable
    return None
