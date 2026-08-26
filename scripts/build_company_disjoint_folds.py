from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]


def _company_key(row: dict[str, object]) -> str:
    supervision = row.get("supervision")
    if not isinstance(supervision, dict):
        raise ValueError("Every fold row needs supervision.company_key")
    company = str(supervision.get("company_key", "")).strip().lower()
    if not company:
        raise ValueError("Every fold row needs supervision.company_key")
    return company


def _operation(row: dict[str, object]) -> str:
    program = row.get("program")
    return str(program.get("operation", "")) if isinstance(program, dict) else ""


def _candidate_dicts(row: dict[str, object]) -> tuple[dict[str, object], ...]:
    raw = row.get("candidates")
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, dict))


def _report_keys(row: dict[str, object]) -> tuple[str, ...]:
    reports = {
        str(candidate.get("source_name", "")).strip().lower()
        for candidate in _candidate_dicts(row)
        if str(candidate.get("source_name", "")).strip().lower()
        not in {"", "context"}
    }
    return tuple(sorted(reports))


def _source_key(row: dict[str, object]) -> str:
    methods = {
        str(candidate.get("extraction_method", "")).strip().lower()
        for candidate in _candidate_dicts(row)
        if str(candidate.get("extraction_method", "")).strip()
    }
    return "|".join(sorted(methods)) or "unknown"


def _layout_family(row: dict[str, object]) -> str:
    candidates = _candidate_dicts(row)
    structured = any(
        candidate.get("row_label")
        and (candidate.get("column_label") or candidate.get("period"))
        for candidate in candidates
    )
    row_only = any(
        candidate.get("row_label")
        and not (candidate.get("column_label") or candidate.get("period"))
        for candidate in candidates
    )
    structure = (
        "mixed"
        if structured and row_only
        else "structured_table"
        if structured
        else "row_only"
        if row_only
        else "prose"
    )

    period_text = " ".join(
        str(candidate.get("period") or candidate.get("column_label") or "")
        for candidate in candidates
    ).lower()
    has_year = bool(re.search(r"\b(?:19|20)\d{2}\b", period_text))
    has_relative = any(
        marker in period_text
        for marker in ("dau nam", "cuoi nam", "nam nay", "nam truoc", "dau ky", "cuoi ky")
    )
    period_style = (
        "mixed_period"
        if has_year and has_relative
        else "explicit_period"
        if has_year
        else "relative_period"
        if has_relative
        else "sparse_period"
    )

    scopes = {
        str(candidate.get("scope", "")).strip().lower()
        for candidate in candidates
        if str(candidate.get("scope", "")).strip()
    }
    scope_style = next(iter(scopes)) if len(scopes) == 1 else "mixed" if scopes else "unscoped"
    return f"{structure}:{period_style}:{scope_style}"


def _period_key(row: dict[str, object]) -> str:
    question = str(row.get("question", ""))
    years = tuple(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", question)))
    if years:
        return "|".join(years)
    program = row.get("program")
    operands = program.get("operands", []) if isinstance(program, dict) else []
    candidates = _candidate_dicts(row)
    periods: list[str] = []
    for raw_index in operands if isinstance(operands, list) else []:
        if not isinstance(raw_index, int) or not 0 <= raw_index < len(candidates):
            continue
        value = str(candidates[raw_index].get("period", "")).strip().lower()
        if value:
            periods.append(value)
    return "|".join(dict.fromkeys(periods)) or "unknown"


def _group_metadata(row: dict[str, object]) -> dict[str, str]:
    reports = _report_keys(row)
    metadata = {
        "company": _company_key(row),
        "report": "|".join(reports) or "unknown",
        "source": _source_key(row),
        "layout_family": _layout_family(row),
        "period": _period_key(row),
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    metadata["group_key"] = sha256(encoded.encode("utf-8")).hexdigest()
    return metadata


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _company_report_components(
    rows: Sequence[dict[str, object]],
) -> list[tuple[tuple[str, ...], list[dict[str, object]]]]:
    companies = sorted({_company_key(row) for row in rows})
    union = _UnionFind(companies)
    report_companies: dict[str, set[str]] = {}
    for row in rows:
        company = _company_key(row)
        for report in _report_keys(row):
            report_companies.setdefault(report, set()).add(company)
    for linked in report_companies.values():
        ordered = sorted(linked)
        for company in ordered[1:]:
            union.union(ordered[0], company)

    component_companies: dict[str, set[str]] = {}
    for company in companies:
        component_companies.setdefault(union.find(company), set()).add(company)
    component_rows: dict[str, list[dict[str, object]]] = {
        root: [] for root in component_companies
    }
    for row in rows:
        component_rows[union.find(_company_key(row))].append(row)
    return [
        (
            tuple(sorted(component_companies[root])),
            sorted(component_rows[root], key=lambda item: str(item.get("example_id", ""))),
        )
        for root in sorted(component_companies)
    ]


def _build_folds(
    rows: Sequence[dict[str, object]],
    *,
    fold_count: int = 5,
    minimum_companies: int = 10,
) -> list[dict[str, object]]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    companies = {_company_key(row) for row in rows}
    if len(companies) < minimum_companies:
        raise ValueError(f"Need at least {minimum_companies} companies")
    components = _company_report_components(rows)
    if len(components) < fold_count:
        raise ValueError("Shared reports leave too few disjoint fold components")

    assignments: list[list[tuple[tuple[str, ...], list[dict[str, object]]]]] = [
        [] for _ in range(fold_count)
    ]
    fold_company_counts = [0] * fold_count
    fold_row_counts = [0] * fold_count
    fold_operations = [Counter() for _ in range(fold_count)]
    ordered_components = sorted(
        components,
        key=lambda item: (-len(item[0]), -len(item[1]), item[0]),
    )
    for component_companies, component_rows in ordered_components:
        operations = Counter(_operation(row) for row in component_rows)
        minimum_company_count = min(fold_company_counts)
        eligible = [
            index
            for index, count in enumerate(fold_company_counts)
            if count == minimum_company_count
        ]
        target = min(
            eligible,
            key=lambda index: (
                fold_row_counts[index],
                sum((fold_operations[index] + operations).values()),
                index,
            ),
        )
        assignments[target].append((component_companies, component_rows))
        fold_company_counts[target] += len(component_companies)
        fold_row_counts[target] += len(component_rows)
        fold_operations[target].update(operations)

    output: list[dict[str, object]] = []
    all_rows = list(rows)
    for fold_index, assigned in enumerate(assignments):
        validation_companies = {
            company for companies_, _ in assigned for company in companies_
        }
        validation = [
            row for _, component_rows in assigned for row in component_rows
        ]
        train = [
            row for row in all_rows if _company_key(row) not in validation_companies
        ]
        train_companies = {_company_key(row) for row in train}
        train_reports = {report for row in train for report in _report_keys(row)}
        validation_reports = {
            report for row in validation for report in _report_keys(row)
        }
        train_groups = {_group_metadata(row)["group_key"] for row in train}
        validation_groups = {
            _group_metadata(row)["group_key"] for row in validation
        }
        if train_companies & validation_companies:
            raise ValueError("Company overlap within fold")
        if train_reports & validation_reports:
            raise ValueError("Report overlap within fold")
        if train_groups & validation_groups:
            raise ValueError("Composite group overlap within fold")
        output.append(
            {
                "fold": fold_index,
                "train": train,
                "validation": validation,
                "train_companies": tuple(sorted(train_companies)),
                "validation_companies": tuple(sorted(validation_companies)),
                "company_overlap": 0,
                "report_overlap": 0,
                "group_overlap": 0,
            }
        )
    return output


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            rows.append(value)
    return rows


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _reject_protected(path: Path) -> None:
    normalized = str(path).lower().replace("\\", "/")
    if any(marker in normalized for marker in ("blind", "test.jsonl", "final_eval")):
        raise ValueError(f"Protected evaluation input is not allowed: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build five company/report-disjoint reranker folds for V12-B.2 Phase B"
    )
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "train.jsonl")
    parser.add_argument(
        "--sealed-validation",
        type=Path,
        default=ROOT / "data" / "validation.jsonl",
        help="Used only to exclude IDs and entire company/report groups",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--minimum-companies", type=int, default=10)
    args = parser.parse_args()

    _reject_protected(args.input)
    rows = _read_jsonl(args.input)
    sealed = _read_jsonl(args.sealed_validation)
    sealed_ids = {str(row.get("example_id", "")) for row in sealed}
    sealed_companies = {_company_key(row) for row in sealed}
    sealed_reports = {report for row in sealed for report in _report_keys(row)}
    development = [
        row
        for row in rows
        if str(row.get("example_id", "")) not in sealed_ids
        and _company_key(row) not in sealed_companies
        and not (set(_report_keys(row)) & sealed_reports)
    ]
    folds = _build_folds(
        development,
        fold_count=args.fold_count,
        minimum_companies=args.minimum_companies,
    )
    group_rows = {
        str(row.get("example_id", "")): _group_metadata(row)
        for row in development
    }
    payload = {
        "schema": "v12b2-reranker-disjoint-folds-v1",
        "disjoint_claim": "reranker-disjoint-not-end-to-end-unseen-generator",
        "input": str(args.input.resolve()),
        "input_sha256": _digest(args.input),
        "sealed_validation": str(args.sealed_validation.resolve()),
        "sealed_validation_sha256": _digest(args.sealed_validation),
        "blind_test_used": False,
        "fold_count": len(folds),
        "development_examples": len(development),
        "company_count": len({_company_key(row) for row in development}),
        "group_schema": ["company", "report", "source", "layout_family", "period"],
        "examples": group_rows,
        "example_operations": {
            str(row.get("example_id", "")): _operation(row) for row in development
        },
        "folds": [
            {
                "fold": fold["fold"],
                "train_example_ids": [str(row.get("example_id", "")) for row in fold["train"]],
                "validation_example_ids": [
                    str(row.get("example_id", "")) for row in fold["validation"]
                ],
                "train_companies": fold["train_companies"],
                "validation_companies": fold["validation_companies"],
                "company_overlap": fold["company_overlap"],
                "report_overlap": fold["report_overlap"],
                "group_overlap": fold["group_overlap"],
                "train_operation_counts": dict(Counter(_operation(row) for row in fold["train"])),
                "validation_operation_counts": dict(
                    Counter(_operation(row) for row in fold["validation"])
                ),
            }
            for fold in folds
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in ("schema", "fold_count", "development_examples", "company_count")}, ensure_ascii=False, indent=2))
    print(f"Saved fold manifest: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
