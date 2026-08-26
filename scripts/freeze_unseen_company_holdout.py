#!/usr/bin/env python3
"""Freeze reviewed unseen-company QA into full-page ProgramExample holdout rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_ocr_financial_page_manifest import _read_pages
from hrm_finance_qa.ontology import normalize_text
from hrm_finance_qa.program_data import (
    NumericCandidate,
    ProgramExample,
    ReasoningProgram,
    extract_numeric_candidates,
    program_example_to_dict,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_tokens(text: str) -> set[str]:
    ignored = {"i", "ii", "iii", "iv", "v", "c", "d", "1", "2", "3"}
    return {
        token
        for token in normalize_text(text).split()
        if token not in ignored and len(token) > 1
    }


def _candidate_match_score(
    evidence: dict[str, object], candidate: NumericCandidate
) -> float | None:
    if Decimal(str(evidence["value"])) != candidate.decimal:
        return None
    period = str(evidence["period"])
    normalized_context = normalize_text(candidate.context)
    candidate_period = normalize_text(candidate.period or candidate.column_label)
    if period not in candidate_period and period not in normalized_context:
        return None
    expected_tokens = _row_tokens(str(evidence["row_label"]))
    candidate_tokens = _row_tokens(candidate.row_label or candidate.context)
    overlap = len(expected_tokens & candidate_tokens)
    if expected_tokens and overlap / len(expected_tokens) < 0.6:
        return None
    score = float(overlap)
    score += 4.0 if period in candidate_period else 1.0
    expected_row = normalize_text(str(evidence["row_label"]))
    candidate_row = normalize_text(candidate.row_label)
    if expected_row and expected_row == candidate_row:
        score += 8.0
    elif expected_row and expected_row in normalized_context:
        score += 4.0
    return score


def _gold_pointers(
    evidence: list[dict[str, object]], candidates: tuple[NumericCandidate, ...]
) -> tuple[tuple[int, ...] | None, str | None]:
    pointers: list[int] = []
    for item in evidence:
        ranked = sorted(
            (
                (score, candidate.index)
                for candidate in candidates
                if (score := _candidate_match_score(item, candidate)) is not None
            ),
            reverse=True,
        )
        if not ranked:
            return None, f"missing_operand:{item['metric']}"
        best_score = ranked[0][0]
        best = [index for score, index in ranked if score == best_score]
        if len(best) != 1:
            return None, f"ambiguous_operand:{item['metric']}"
        pointers.append(best[0])
    if len(set(pointers)) != len(pointers):
        return None, "reused_operand_pointer"
    return tuple(pointers), None


def build_holdout(
    reviewed_rows: list[dict[str, object]], markdown_dir: Path
) -> tuple[list[ProgramExample], list[dict[str, object]], dict[str, int]]:
    page_cache: dict[str, dict[int, str]] = {}
    examples: list[ProgramExample] = []
    cases: list[dict[str, object]] = []
    rejection_counts: Counter[str] = Counter()
    for row in reviewed_rows:
        source = row.get("source")
        if not isinstance(source, dict):
            raise ValueError("Reviewed row has no source provenance")
        if row.get("classification") != "development_only":
            raise ValueError("Holdout source must be development_only")
        document = Path(str(source["document"]))
        markdown_path = markdown_dir / f"{document.stem}.md"
        if not markdown_path.is_file():
            raise FileNotFoundError(f"Missing OCR Markdown: {markdown_path}")
        if document.stem not in page_cache:
            page_cache[document.stem] = _read_pages(markdown_path)
        page = int(source["page"])
        context = page_cache[document.stem].get(page)
        if context is None:
            raise ValueError(f"Missing page {page} in {markdown_path}")
        question = str(row["question"])
        candidates = extract_numeric_candidates(
            question,
            context,
            max_candidates=64,
            source_name=document.name,
            page=page,
            extraction_method="mineru",
            financial_identity_catalog_enabled=True,
        )
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or len(evidence) != 2:
            raise ValueError("Current single-step holdout requires exactly two operands")
        pointers, rejection = _gold_pointers(evidence, candidates)
        case = {
            "example_id": str(row["example_id"]),
            "source_document": str(source["document"]),
            "source_sha256": str(source["sha256"]),
            "company_key": str(source["company_key"]),
            "page": page,
            "question_type": str(row["question_type"]),
            "candidate_count": len(candidates),
            "gold_operands_found": pointers is not None,
            "rejection_reason": rejection,
        }
        cases.append(case)
        if pointers is None:
            rejection_counts[rejection or "unknown"] += 1
            continue
        nodes = row.get("program", {}).get("nodes", [])
        if not isinstance(nodes, list) or len(nodes) != 1:
            raise ValueError("Current holdout supports one-node programs only")
        operation = str(nodes[0]["operation"])
        program = ReasoningProgram(operation, pointers)
        program.validate(len(candidates))
        examples.append(
            ProgramExample(
                example_id=str(row["example_id"]),
                question=question,
                context=context,
                answer_text=str(row["answer_value"]),
                candidates=candidates,
                program=program,
                answer_value=str(row["answer_value"]),
                task="QA",
                difficulty="unseen_company_development_holdout",
                supervision={
                    "classification": "development_holdout_evaluation_only",
                    "source": source,
                    "question_type": str(row["question_type"]),
                    "reviewed_candidate_id": str(row["example_id"]),
                },
            )
        )
    return examples, cases, dict(sorted(rejection_counts.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-candidates", type=Path, required=True)
    parser.add_argument("--markdown-dir", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output_dataset, args.output_manifest):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
    rows = [
        json.loads(line)
        for line in args.reviewed_candidates.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    examples, cases, rejections = build_holdout(rows, args.markdown_dir)
    args.output_dataset.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_dataset.write_text(
        "".join(
            json.dumps(program_example_to_dict(example), ensure_ascii=False) + "\n"
            for example in examples
        ),
        encoding="utf-8",
    )
    manifest = {
        "protocol": "v12b2-unseen-company-development-holdout-v1",
        "classification": "development_holdout_evaluation_only",
        "frozen": True,
        "freeze_date": "2026-08-09",
        "model_invoked": False,
        "blind_test_used": False,
        "training_use_forbidden": True,
        "inputs": {
            "reviewed_candidates": str(args.reviewed_candidates.resolve()),
            "reviewed_candidates_sha256": _sha256(args.reviewed_candidates),
            "markdown_dir": str(args.markdown_dir.resolve()),
        },
        "output": {
            "dataset": str(args.output_dataset.resolve()),
            "dataset_sha256": _sha256(args.output_dataset),
        },
        "metrics": {
            "reviewed_cases": len(rows),
            "program_examples": len(examples),
            "candidate_recall": len(examples) / len(rows) if rows else 0.0,
            "company_groups": len({case["company_key"] for case in cases}),
            "rejection_counts": rejections,
        },
        "cases": cases,
    }
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["metrics"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
