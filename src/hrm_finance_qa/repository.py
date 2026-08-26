from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .ontology import VariableSpec


FINANCIAL_SUFFIX = "_financial_statements_by_year.csv"
MARKET_SUFFIX = "_market_data_daily.csv"


def _is_aggregate(path: Path, suffix: str) -> bool:
    return path.name.lower() == f"raw{suffix}"


@dataclass(frozen=True)
class Fact:
    ticker: str
    year: int
    statement: str
    source: str
    line_order: int
    item: str
    item_en: str
    item_id: str
    value: Decimal

    @property
    def citation(self) -> str:
        label = self.item or self.item_en or self.item_id
        return (
            f"{self.ticker} {self.year} | {self.statement} | {label} "
            f"[{self.item_id}] | source={self.source}, line={self.line_order}"
        )


def _decimal(raw: str | None) -> Decimal | None:
    if raw is None or not raw.strip() or raw.strip().lower() in {"nan", "none", "null"}:
        return None
    try:
        return Decimal(raw.strip())
    except InvalidOperation:
        return None


class FinancialRepository:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir).expanduser().resolve()
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Khong tim thay thu muc du lieu: {self.data_dir}")
        self._tickers: tuple[str, ...] | None = None
        self._fact_cache: dict[str, tuple[Fact, ...]] = {}

    def tickers(self) -> tuple[str, ...]:
        if self._tickers is None:
            tickers = [
                path.name[: -len(FINANCIAL_SUFFIX)].upper()
                for path in self.data_dir.glob(f"*{FINANCIAL_SUFFIX}")
                if not _is_aggregate(path, FINANCIAL_SUFFIX)
            ]
            self._tickers = tuple(sorted(tickers))
        return self._tickers

    def financial_path(self, ticker: str) -> Path:
        if ticker.upper() == "RAW":
            raise FileNotFoundError("RAW là file tổng hợp, không phải mã cổ phiếu")
        path = self.data_dir / f"{ticker.upper()}{FINANCIAL_SUFFIX}"
        if not path.is_file():
            raise FileNotFoundError(f"Khong co bao cao tai chinh cho ma {ticker.upper()}")
        return path

    def facts(self, ticker: str, year: int | None = None) -> tuple[Fact, ...]:
        ticker = ticker.upper()
        if ticker not in self._fact_cache:
            loaded: list[Fact] = []
            with self.financial_path(ticker).open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("ticker", "").upper() != ticker:
                        continue
                    value = _decimal(row.get("value"))
                    raw_year = row.get("year", "")
                    if value is None or not raw_year.isdigit():
                        continue
                    try:
                        line_order = int(float(row.get("line_order") or 0))
                    except ValueError:
                        line_order = 0
                    loaded.append(
                        Fact(
                            ticker=ticker,
                            year=int(raw_year),
                            statement=row.get("statement", ""),
                            source=row.get("source", ""),
                            line_order=line_order,
                            item=row.get("item", ""),
                            item_en=row.get("item_en", ""),
                            item_id=row.get("item_id", ""),
                            value=value,
                        )
                    )
            self._fact_cache[ticker] = tuple(loaded)
        facts = self._fact_cache[ticker]
        return facts if year is None else tuple(fact for fact in facts if fact.year == year)

    def years(self, ticker: str) -> tuple[int, ...]:
        return tuple(sorted({fact.year for fact in self.facts(ticker)}))

    def find_fact(self, ticker: str, year: int, variable: VariableSpec) -> Fact | None:
        facts = self.facts(ticker, year)
        for item_id in variable.item_ids:
            matches = [fact for fact in facts if fact.item_id == item_id]
            if variable.statement:
                preferred = [fact for fact in matches if fact.statement == variable.statement]
                matches = preferred or matches
            if matches:
                return min(matches, key=lambda fact: fact.line_order)
        return None

    def profile(self) -> dict[str, object]:
        financial_files = [
            path
            for path in self.data_dir.glob(f"*{FINANCIAL_SUFFIX}")
            if not _is_aggregate(path, FINANCIAL_SUFFIX)
        ]
        market_files = [
            path
            for path in self.data_dir.glob(f"*{MARKET_SUFFIX}")
            if not _is_aggregate(path, MARKET_SUFFIX)
        ]
        years: set[int] = set()
        statements: Counter[str] = Counter()
        item_ids: Counter[str] = Counter()
        rows = 0
        for path in financial_files:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows += 1
                    if row.get("year", "").isdigit():
                        years.add(int(row["year"]))
                    statements[row.get("statement", "unknown")] += 1
                    if row.get("item_id"):
                        item_ids[row["item_id"]] += 1
        return {
            "financial_files": len(financial_files),
            "market_files": len(market_files),
            "tickers": len(self.tickers()),
            "financial_rows": rows,
            "years": sorted(years),
            "statements": statements,
            "top_item_ids": item_ids.most_common(15),
        }
