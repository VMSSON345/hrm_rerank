from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re

from hrm_finance_qa.document_hrm import _candidate_period_years
from hrm_finance_qa.document_reasoning_engine import (
    _augment_interest_bearing_debt_share,
    _debt_share_role,
)
from hrm_finance_qa.program_data import NumericCandidate


PROTECTED_NAME = re.compile(
    r"(?:test|holdout|final|blind|validation)", re.IGNORECASE
)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_source(path: Path) -> dict[str, object]:
    if PROTECTED_NAME.search(path.name):
        raise ValueError(f"Protected evaluation source is not allowed: {path}")
    source_rows = 0
    occurrences: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source_rows += 1
            payload = json.loads(line)
            candidates = tuple(
                NumericCandidate(**item) for item in payload.get("candidates", ())
            )
            years = sorted(
                {
                    year
                    for candidate in candidates
                    for year in _candidate_period_years(candidate)
                },
                key=int,
            )
            supervision = payload.get("supervision")
            supervision = supervision if isinstance(supervision, dict) else {}
            company_key = str(supervision.get("company_key", ""))
            for year in years:
                question = (
                    "Tong no vay tai chinh chiu lai gom vay ngan han va dai han "
                    "chiem bao nhieu phan tram Tong nguon von nam "
                    f"{year}?"
                )
                augmented = _augment_interest_bearing_debt_share(
                    question,
                    candidates,
                    max_candidates=max(1, len(candidates) + 1),
                )
                derived = next(
                    (
                        item
                        for item in augmented
                        if item.derivation_id == "interest_bearing_debt"
                    ),
                    None,
                )
                if derived is None:
                    continue
                raw = [augmented[index] for index in derived.derived_operands]
                totals = [
                    item
                    for item in augmented
                    if _debt_share_role(item) == "total_capital"
                    and year in _candidate_period_years(item)
                    and item.source_name == derived.source_name
                    and item.table_id == derived.table_id
                ]
                if len(totals) != 1:
                    continue
                total = totals[0]
                signature = (
                    company_key,
                    derived.source_name,
                    derived.table_id,
                    year,
                    *(item.value for item in raw),
                    total.value,
                )
                if signature in seen:
                    continue
                seen.add(signature)
                occurrences.append(
                    {
                        "example_id": payload.get("example_id", ""),
                        "company_key": company_key,
                        "source_name": derived.source_name,
                        "table_id": derived.table_id,
                        "year": year,
                        "short_term_borrowings": raw[0].value,
                        "long_term_borrowings": raw[1].value,
                        "total_capital": total.value,
                        "review_status": supervision.get("review_status", ""),
                    }
                )
    companies = sorted(
        {
            str(item["company_key"])
            for item in occurrences
            if item["company_key"]
        }
    )
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "source_rows": source_rows,
        "unique_structural_cases": len(occurrences),
        "company_keys": companies,
        "cases": occurrences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sources = [audit_source(Path(item).resolve()) for item in args.input]
    companies = sorted(
        {
            company
            for source in sources
            for company in source["company_keys"]
        }
    )
    cases = sum(int(source["unique_structural_cases"]) for source in sources)
    report = {
        "protocol": "v12c-interest-bearing-debt-share-cohort-discovery-v1",
        "classification": "development_only",
        "protected_evaluation_sources_used": False,
        "sources": sources,
        "summary": {
            "unique_structural_cases": cases,
            "unique_company_keys": len(companies),
            "company_keys": companies,
            "company_disjoint_calibration_ready": cases >= 10 and len(companies) >= 4,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
