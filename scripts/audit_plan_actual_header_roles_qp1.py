from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.cache_schema import (  # noqa: E402
    candidate_cache_schema_identity,
)
from hrm_finance_qa.evidence_schema import evidence_cells  # noqa: E402
from hrm_finance_qa.ontology import normalize_text  # noqa: E402
from hrm_finance_qa.program_data import load_program_examples  # noqa: E402


PROTECTED_TOKENS = {"blind", "test", "validation", "holdout", "final_eval"}


def _digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _guard_development_path(path: Path) -> None:
    lowered = {part.lower() for part in path.resolve().parts}
    if lowered.intersection(PROTECTED_TOKENS):
        raise ValueError(f"QP1 refuses protected input: {path}")
    if "development" not in path.name.lower():
        raise ValueError(f"QP1 input must be explicitly development-only: {path}")


def _requested_role(question: str) -> str:
    normalized = f" {normalize_text(question)} "
    actual = any(
        phrase in normalized
        for phrase in (" thuc hien ", " thuc te ", " ket qua dat duoc ")
    )
    plan = any(
        phrase in normalized
        for phrase in (" ke hoach ", " muc tieu ", " du kien ")
    )
    if actual == plan:
        return ""
    return "actual" if actual else "plan"


def _swap_role_text(value: str) -> str:
    """Create an audit-only actual/plan minimal pair without changing values."""

    actual_patterns = (
        r"thực\s*hiện",
        r"thuc\s*hien",
        r"thực\s*tế",
        r"thuc\s*te",
        r"\bactual\b",
        r"\bTH\b",
    )
    plan_patterns = (
        r"kế\s*hoạch",
        r"ke\s*hoach",
        r"\bplan(?:ned)?\b",
        r"\bKH\b",
    )
    output = value
    for pattern in actual_patterns:
        output = re.sub(pattern, "__QP1_PLAN__", output, flags=re.IGNORECASE)
    for pattern in plan_patterns:
        output = re.sub(pattern, "__QP1_ACTUAL__", output, flags=re.IGNORECASE)
    return output.replace("__QP1_PLAN__", "Kế hoạch").replace(
        "__QP1_ACTUAL__", "Thực hiện"
    )


def _counterfactual_candidate(candidate: object) -> object:
    return replace(
        candidate,
        context=_swap_role_text(candidate.context),
        column_label=_swap_role_text(candidate.column_label),
        period=_swap_role_text(candidate.period),
    )


def audit(input_path: Path) -> dict[str, object]:
    _guard_development_path(input_path)
    examples = load_program_examples(input_path)
    counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    extraction_counts: Counter[str] = Counter()
    gold_requested_counts: Counter[str] = Counter()
    unknown_samples: list[dict[str, object]] = []
    conflict_samples: list[dict[str, object]] = []
    role_examples: set[str] = set()
    conflict_examples: set[str] = set()

    for example in examples:
        cells = evidence_cells(example.candidates)
        by_index = {cell.index: cell for cell in cells}
        counts["examples"] += 1
        counts["cells"] += len(cells)
        for candidate, cell in zip(example.candidates, cells):
            counts["header_context_cells"] += int("\nHEADER:" in candidate.context)
            counts["header_path_cells"] += int(bool(cell.header_path))
            counts["source_span_cells"] += int(cell.source_span is not None)
            counts["known_role_cells"] += int(bool(cell.column_role))
            counts["unknown_role_cells"] += int(not cell.column_role)
            counts["conflict_cells"] += int(cell.column_role_conflict)
            role_counts[cell.column_role or "unknown"] += 1
            confidence_counts[cell.column_role_confidence] += 1
            source_counts[cell.column_role_source or "none"] += 1
            extraction_counts[cell.extraction_method or "unspecified"] += 1
            if cell.column_role:
                role_examples.add(example.example_id)
            if cell.column_role_conflict:
                conflict_examples.add(example.example_id)
                if len(conflict_samples) < 20:
                    conflict_samples.append(
                        {
                            "example_id": example.example_id,
                            "candidate_index": cell.index,
                            "column": cell.column,
                            "header_path": list(cell.header_path),
                            "role_evidence": list(cell.column_role_evidence),
                            "role_source": cell.column_role_source,
                            "page": cell.page,
                            "source_name": cell.source_name,
                        }
                    )
            if (
                not cell.column_role
                and "\nHEADER:" in candidate.context
                and len(unknown_samples) < 20
            ):
                unknown_samples.append(
                    {
                        "example_id": example.example_id,
                        "candidate_index": cell.index,
                        "column": cell.column,
                        "header_path": list(cell.header_path),
                        "page": cell.page,
                        "source_name": cell.source_name,
                    }
                )

            if cell.column_role not in {"actual", "plan"}:
                continue
            counts["counterfactual_cells"] += 1
            swapped_candidate = _counterfactual_candidate(candidate)
            swapped = evidence_cells((swapped_candidate,))[0]
            expected = "plan" if cell.column_role == "actual" else "actual"
            if (
                swapped.column_role == expected
                and not swapped.column_role_conflict
            ):
                counts["counterfactual_flip_pass"] += 1
            else:
                counts["counterfactual_flip_fail"] += 1

        requested = _requested_role(example.question)
        if requested:
            gold_requested_counts["examples"] += 1
            gold_requested_counts[f"requested_{requested}"] += 1
            selected = [
                by_index[operand]
                for operand in example.program.operands
                if operand in by_index
            ]
            if len(selected) == len(example.program.operands):
                gold_requested_counts["gold_operands_resolved"] += 1
            selected_roles = {cell.column_role for cell in selected}
            if selected and selected_roles == {requested}:
                gold_requested_counts["gold_role_exact"] += 1
            elif any(cell.column_role_conflict for cell in selected):
                gold_requested_counts["gold_role_conflict"] += 1
            elif any(not cell.column_role for cell in selected):
                gold_requested_counts["gold_role_unknown"] += 1
            else:
                gold_requested_counts["gold_role_mismatch"] += 1

    counterfactual_total = counts["counterfactual_cells"]
    report: dict[str, object] = {
        "schema": "plan-actual-header-role-audit-development-v1",
        "classification": "development_only_audit",
        "scope": "QP1 evidence representation and extraction only",
        "runtime_acceptance_changed": False,
        "strict_verifier_enabled": False,
        "threshold_changed": False,
        "fine_tuning_performed": False,
        "global_validation_loaded": False,
        "sealed_validation_loaded": False,
        "blind_test_loaded": False,
        "input": str(input_path.resolve()),
        "input_sha256": _digest(input_path),
        "cache_identity_contract": candidate_cache_schema_identity(),
        "counts": dict(sorted(counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "role_source_counts": dict(sorted(source_counts.items())),
        "extraction_method_counts": dict(sorted(extraction_counts.items())),
        "examples_with_known_role": len(role_examples),
        "examples_with_conflict": len(conflict_examples),
        "gold_requested_role_audit": dict(sorted(gold_requested_counts.items())),
        "counterfactual_role_flip_rate": (
            counts["counterfactual_flip_pass"] / counterfactual_total
            if counterfactual_total
            else 0.0
        ),
        "conflict_samples": conflict_samples,
        "unknown_header_samples": unknown_samples,
        "decision": (
            "QP1 is representation-only. Do not enable strict acceptance; "
            "use unknown/conflict inventory to guide QP2/QP3."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            ROOT
            / "data"
            / "document_hrm_v12b2_development_plus_subtract_v3.jsonl"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=(
            ROOT / "reports" / "plan_actual_header_role_audit_development_v1.json"
        ),
    )
    args = parser.parse_args()
    report = audit(args.input.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
