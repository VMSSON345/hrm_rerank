from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation

from .ontology import FORMULAS, VARIABLES, FormulaSpec, resolve_metric, resolve_variable
from .repository import Fact, FinancialRepository


class QAError(ValueError):
    pass


@dataclass(frozen=True)
class ConstraintCheck:
    name: str
    status: str
    actual: Decimal | None
    expected: Decimal | None
    relative_error: Decimal | None
    evidence: tuple[Fact, ...]


@dataclass(frozen=True)
class Answer:
    ticker: str
    year: int
    metric: FormulaSpec
    value: Decimal
    facts: dict[str, Fact]
    trace: tuple[str, ...]
    checks: tuple[ConstraintCheck, ...]
    adjustments: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def render(self) -> str:
        if self.metric.unit == "percent":
            formatted = f"{self.value * 100:,.2f}%"
        elif self.metric.unit == "ratio":
            formatted = f"{self.value:,.4f} lần"
        else:
            formatted = f"{self.value:,.0f}"
        lines = [
            f"Kết quả: {self.metric.label} của {self.ticker} năm {self.year} = {formatted}",
            f"Công thức: {self.metric.expression}",
        ]
        if self.metric.note:
            lines.append(f"Giả định: {self.metric.note}")
        if self.adjustments:
            lines.append("What-if: " + "; ".join(self.adjustments))
        if self.warnings:
            lines.append("Cảnh báo:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        lines.append("Bằng chứng:")
        for key, fact in self.facts.items():
            lines.append(f"- {key} = {fact.value:,.4f} | {fact.citation}")
        lines.append("Dấu vết HRM:")
        lines.extend(f"- {step}" for step in self.trace)
        completed = [check for check in self.checks if check.status != "skipped"]
        if completed:
            lines.append("Đánh giá ràng buộc:")
            for check in completed:
                error = "n/a" if check.relative_error is None else f"{check.relative_error * 100:.3f}%"
                lines.append(f"- {check.name}: {check.status}, sai lệch={error}")
        return "\n".join(lines)


class FinancialQAEngine:
    def __init__(self, repository: FinancialRepository):
        self.repository = repository

    def answer(
        self,
        question: str = "",
        *,
        ticker: str | None = None,
        year: int | None = None,
        metric: str | None = None,
        adjustments: tuple[str, ...] = (),
    ) -> Answer:
        formula = resolve_metric(metric or question)
        if formula is None:
            supported = ", ".join(FORMULAS)
            raise QAError(f"Không nhận diện được chỉ số. Hỗ trợ: {supported}")
        ticker, ticker_inferred = self._resolve_ticker(question, ticker)
        year, year_inferred = self._resolve_year(question, ticker, year)

        trace = [
            f"Tầng 1 - document: chọn báo cáo của {ticker}.",
            "Tầng 2 - section: chọn income_statement, balance_sheet hoặc cash_flow theo ontology.",
            f"Expansion Agent: {formula.expression}; biến cần tìm: {', '.join(formula.variables)}.",
        ]
        if ticker_inferred:
            trace.append(f"Ticker được nhận diện từ câu hỏi: {ticker}.")
        if year_inferred:
            trace.append(f"Câu hỏi không có năm; dùng năm mới nhất trong dữ liệu: {year}.")

        facts: dict[str, Fact] = {}
        for variable_key in formula.variables:
            fact = self.repository.find_fact(ticker, year, VARIABLES[variable_key])
            if fact is not None:
                facts[variable_key] = fact
                trace.append(
                    f"Tầng 3-4 - table/cell: {variable_key} -> {fact.statement}/{fact.item_id}, năm {year}."
                )

        values = {key: fact.value for key, fact in facts.items()}
        applied = self._apply_adjustments(values, adjustments)
        value = self._calculate(formula.key, values)
        checks = self.validate(ticker, year)
        warnings = self._plausibility_warnings(formula.key, value, values)
        trace.append("Solving Agent: tính bằng Decimal, không dùng số chấm động nhị phân.")
        trace.append("Evaluation Agent: đối chiếu các phương trình kế toán có đủ bằng chứng.")
        return Answer(
            ticker,
            year,
            formula,
            value,
            facts,
            tuple(trace),
            checks,
            adjustments=applied,
            warnings=warnings,
        )

    def validate(self, ticker: str, year: int, tolerance: Decimal = Decimal("0.01")) -> tuple[ConstraintCheck, ...]:
        def get(key: str) -> Fact | None:
            return self.repository.find_fact(ticker, year, VARIABLES[key])

        checks: list[ConstraintCheck] = []
        constraints = (
            (
                "Tổng tài sản = Nợ phải trả + Vốn chủ sở hữu",
                get("total_assets"),
                (get("liabilities"), get("owners_equity")),
                lambda parts: parts[0].value + parts[1].value,
            ),
            (
                "Lợi nhuận gộp = Doanh thu thuần + Giá vốn có dấu",
                get("gross_profit"),
                (get("net_sales"), get("cost_of_sales")),
                lambda parts: parts[0].value + parts[1].value,
            ),
            (
                "Doanh thu thuần = Doanh thu + Giảm trừ doanh thu",
                get("net_sales"),
                (get("sales"), get("sales_deductions")),
                lambda parts: parts[0].value + parts[1].value,
            ),
        )
        for name, actual_fact, part_facts, calculator in constraints:
            evidence = tuple(fact for fact in (actual_fact, *part_facts) if fact is not None)
            if actual_fact is None or any(fact is None for fact in part_facts):
                checks.append(ConstraintCheck(name, "skipped", None, None, None, evidence))
                continue
            expected = calculator(part_facts)  # type: ignore[arg-type]
            denominator = max(abs(actual_fact.value), abs(expected), Decimal(1))
            relative_error = abs(actual_fact.value - expected) / denominator
            status = "pass" if relative_error <= tolerance else "fail"
            checks.append(
                ConstraintCheck(name, status, actual_fact.value, expected, relative_error, evidence)
            )
        return tuple(checks)

    def _resolve_ticker(self, question: str, ticker: str | None) -> tuple[str, bool]:
        if ticker:
            value = ticker.upper()
            if value not in self.repository.tickers():
                raise QAError(f"Không tìm thấy mã {value} trong dữ liệu")
            return value, False
        available = set(self.repository.tickers())
        token_pattern = r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]{1,4}(?![A-Za-z0-9])"
        uppercase_tokens = re.findall(token_pattern, question)
        uppercase_matches = [
            token for token in uppercase_tokens if token == token.upper() and token in available
        ]
        matches = uppercase_matches
        if not matches:
            tokens = re.findall(token_pattern, question)
            matches = [token.upper() for token in tokens if token.upper() in available]
        if not matches:
            raise QAError("Không nhận diện được mã cổ phiếu trong dữ liệu, ví dụ: HPG, FPT, AAA")
        unique_matches = tuple(dict.fromkeys(matches))
        if len(unique_matches) > 1:
            raise QAError(
                "Câu hỏi chứa nhiều mã cổ phiếu; hãy hỏi từng mã hoặc truyền --ticker rõ ràng: "
                + ", ".join(unique_matches)
            )
        return unique_matches[0], True

    def _resolve_year(self, question: str, ticker: str, year: int | None) -> tuple[int, bool]:
        years = self.repository.years(ticker)
        if not years:
            raise QAError(f"Mã {ticker} không có năm dữ liệu hợp lệ")
        if year is None:
            match = re.search(r"\b(20\d{2})\b", question)
            if match:
                year = int(match.group(1))
        if year is None:
            return years[-1], True
        if year not in years:
            raise QAError(f"{ticker} không có dữ liệu năm {year}; các năm có sẵn: {', '.join(map(str, years))}")
        return year, False

    @staticmethod
    def _calculate(metric: str, values: dict[str, Decimal]) -> Decimal:
        required: dict[str, tuple[str, ...]] = {
            "net_margin": ("net_profit", "net_sales"),
            "roa": ("net_profit", "total_assets"),
            "roe": ("net_profit", "owners_equity"),
            "current_ratio": ("current_assets", "current_liabilities"),
            "debt_to_equity": ("liabilities", "owners_equity"),
            "debt_ratio": ("liabilities", "total_assets"),
            "ebitda": ("operating_profit", "depreciation"),
            "dscr": ("operating_profit", "depreciation", "interest_expense", "debt_repayment"),
        }
        if metric == "gross_margin":
            missing = [key for key in ("net_sales",) if key not in values]
            if "gross_profit" not in values and "cost_of_sales" not in values:
                missing.append("gross_profit hoac cost_of_sales")
            if missing:
                raise QAError("Thiếu dữ liệu để tính: " + ", ".join(missing))
            if "gross_profit" in values:
                numerator = values["gross_profit"]
            else:
                numerator = values["net_sales"] + values["cost_of_sales"]
            return FinancialQAEngine._divide(numerator, values["net_sales"])

        missing = [key for key in required[metric] if key not in values]
        if missing:
            raise QAError("Thiếu dữ liệu để tính: " + ", ".join(missing))
        if metric == "net_margin":
            return FinancialQAEngine._divide(values["net_profit"], values["net_sales"])
        if metric == "roa":
            return FinancialQAEngine._divide(values["net_profit"], values["total_assets"])
        if metric == "roe":
            return FinancialQAEngine._divide(values["net_profit"], values["owners_equity"])
        if metric == "current_ratio":
            return FinancialQAEngine._divide(values["current_assets"], values["current_liabilities"])
        if metric == "debt_to_equity":
            return FinancialQAEngine._divide(values["liabilities"], values["owners_equity"])
        if metric == "debt_ratio":
            return FinancialQAEngine._divide(values["liabilities"], values["total_assets"])
        if metric == "ebitda":
            return values["operating_profit"] + abs(values["depreciation"])
        if metric == "dscr":
            cash_proxy = values["operating_profit"] + abs(values["depreciation"])
            debt_service = abs(values["interest_expense"]) + abs(values["debt_repayment"])
            return FinancialQAEngine._divide(cash_proxy, debt_service)
        raise QAError(f"Chua cai dat cong thuc {metric}")

    @staticmethod
    def _divide(numerator: Decimal, denominator: Decimal) -> Decimal:
        try:
            return numerator / denominator
        except (DivisionByZero, InvalidOperation):
            raise QAError("Mẫu số bằng 0 hoặc không hợp lệ") from None

    @staticmethod
    def _apply_adjustments(values: dict[str, Decimal], adjustments: tuple[str, ...]) -> tuple[str, ...]:
        applied: list[str] = []
        for raw in adjustments:
            if "=" not in raw:
                raise QAError(f"What-if khong hop le: {raw}; dung variable=+10% hoac variable=123")
            name, raw_value = (part.strip() for part in raw.split("=", 1))
            variable = resolve_variable(name)
            if variable is None or variable.key not in values:
                raise QAError(f"Bien what-if '{name}' khong thuoc cong thuc hoac khong co du lieu")
            try:
                if raw_value.endswith("%"):
                    change = Decimal(raw_value[:-1]) / Decimal(100)
                    old_value = values[variable.key]
                    values[variable.key] = old_value * (Decimal(1) + change)
                else:
                    old_value = values[variable.key]
                    values[variable.key] = Decimal(raw_value.replace(",", ""))
            except InvalidOperation:
                raise QAError(f"Gia tri what-if khong hop le: {raw_value}") from None
            applied.append(f"{variable.key}: {old_value} -> {values[variable.key]}")
        return tuple(applied)

    @staticmethod
    def _plausibility_warnings(
        metric: str, value: Decimal, values: dict[str, Decimal]
    ) -> tuple[str, ...]:
        warnings: list[str] = []
        denominator_keys = {
            "gross_margin": "net_sales",
            "net_margin": "net_sales",
            "roa": "total_assets",
            "roe": "owners_equity",
            "current_ratio": "current_liabilities",
            "debt_to_equity": "owners_equity",
            "debt_ratio": "total_assets",
        }
        denominator_key = denominator_keys.get(metric)
        if denominator_key and values.get(denominator_key, Decimal(1)) <= 0:
            warnings.append(
                f"{denominator_key} không dương; tỷ lệ cần được diễn giải thận trọng."
            )

        unusual = False
        if metric == "gross_margin":
            unusual = value < 0 or value > 1
        elif metric in {"net_margin", "roa"}:
            unusual = abs(value) > 1
        elif metric == "roe":
            unusual = abs(value) > 5
        elif metric == "current_ratio":
            unusual = value < 0 or value > 20
        elif metric in {"debt_to_equity", "debt_ratio", "dscr"}:
            unusual = value < 0 or value > 20
        if unusual:
            warnings.append(
                "Kết quả nằm ngoài khoảng thường gặp; nên kiểm tra quy ước dấu, đơn vị và báo cáo gốc."
            )
        return tuple(warnings)
