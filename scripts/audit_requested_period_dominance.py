from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from hrm_finance_qa.document_hrm import (
    _candidate_period_years,
    _prefer_requested_period_indices,
    _requested_years,
)
from hrm_finance_qa.program_data import NumericCandidate


PROTECTED_NAME = re.compile(r"(?:test|holdout|final|blind)", re.IGNORECASE)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_unknown_percent(candidate: NumericCandidate) -> bool:
    if _candidate_period_years(candidate):
        return False
    return (
        candidate.is_percent
        or candidate.unit.strip().lower() in {"%", "percent", "percentage"}
        or "%" in candidate.column_label
    )


def audit_source(path: Path) -> dict[str, object]:
    if PROTECTED_NAME.search(path.name):
        raise ValueError(f"Protected evaluation source is not allowed: {path}")

    source_rows = 0
    rows_with_requested_year = 0
    structural_examples = 0
    structural_groups = 0
    passed_groups = 0
    failures: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            source_rows += 1
            payload = json.loads(line)
            question = str(payload.get("question", ""))
            requested = set(_requested_years(question))
            if not requested:
                continue
            rows_with_requested_year += 1
            candidates = tuple(
                NumericCandidate(**candidate) for candidate in payload["candidates"]
            )
            row_groups: dict[str, set[int]] = defaultdict(set)
            for index, candidate in enumerate(candidates):
                row_label = " ".join(candidate.row_label.lower().split())
                if row_label:
                    row_groups[row_label].add(index)

            example_has_pattern = False
            for row_label, allowed in row_groups.items():
                requested_matches = {
                    index
                    for index in allowed
                    if _candidate_period_years(candidates[index], question) & requested
                }
                unknown_percent = {
                    index
                    for index in allowed
                    if _is_unknown_percent(candidates[index])
                }
                if not requested_matches or not unknown_percent:
                    continue

                example_has_pattern = True
                structural_groups += 1
                selected = _prefer_requested_period_indices(
                    question, candidates, allowed
                )
                passed = selected == requested_matches
                if passed:
                    passed_groups += 1
                elif len(failures) < 20:
                    failures.append(
                        {
                            "line": line_number,
                            "example_id": payload.get("example_id", ""),
                            "row_label": row_label,
                            "requested_years": sorted(requested),
                            "expected_indices": sorted(requested_matches),
                            "selected_indices": sorted(selected),
                        }
                    )
                if len(samples) < 10:
                    samples.append(
                        {
                            "example_id": payload.get("example_id", ""),
                            "requested_years": sorted(requested),
                            "row_label": row_label,
                            "absolute_columns": [
                                candidates[index].column_label
                                for index in sorted(requested_matches)
                            ],
                            "unknown_percent_columns": [
                                candidates[index].column_label
                                for index in sorted(unknown_percent)
                            ],
                        }
                    )
            if example_has_pattern:
                structural_examples += 1

    return {
        "path": path.as_posix(),
        "sha256": _file_sha256(path),
        "source_rows": source_rows,
        "rows_with_requested_year": rows_with_requested_year,
        "structural_examples": structural_examples,
        "structural_groups": structural_groups,
        "passed_groups": passed_groups,
        "failed_groups": structural_groups - passed_groups,
        "samples": samples,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit requested-period dominance over same-row unknown percentage "
            "columns in development-only JSONL sources."
        )
    )
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sources = [audit_source(Path(item).resolve()) for item in args.input]
    total_groups = sum(int(source["structural_groups"]) for source in sources)
    failed_groups = sum(int(source["failed_groups"]) for source in sources)
    report = {
        "protocol": "requested-period-dominance-development-audit-v1",
        "protected_evaluation_sources_used": False,
        "policy": (
            "Within a role-equivalent row, explicit evidence matching a requested "
            "period dominates period-unknown percentage/change evidence; unknown "
            "period remains eligible when no explicit match exists."
        ),
        "sources": sources,
        "summary": {
            "structural_groups": total_groups,
            "passed_groups": total_groups - failed_groups,
            "failed_groups": failed_groups,
            "gate_passed": total_groups > 0 and failed_groups == 0,
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if total_groups == 0 or failed_groups:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
