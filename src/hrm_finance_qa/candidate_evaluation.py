from __future__ import annotations

import re
from decimal import Decimal
from typing import Mapping, Sequence

from .program_data import parse_decimal


_RAW_NUMBER_RE = re.compile(
    r"(?<![\w/])\(?[-+]?(?:"
    r"\d{1,3}(?:\.\d{3})+(?:,\d+)?"
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|\d{1,3}(?:[ \u00a0]\d{3})+"
    r"|\d+(?:[.,]\d+)?"
    r")\)?"
)


def score_candidate_availability(
    case: Mapping[str, object],
    *,
    retrieved_pages: Sequence[int],
    prequota_candidates: Sequence[object],
    final_candidates: Sequence[object],
    candidates_at_32: Sequence[object] | None = None,
    candidates_at_64: Sequence[object] | None = None,
    page_chunks: Sequence[object] | None = None,
    retrieved_chunks: Sequence[object] | None = None,
    candidate_chunks: Sequence[object] | None = None,
    error: object = None,
) -> dict[str, object]:
    expected_operands = [
        item for item in case.get("operands", []) if isinstance(item, Mapping)
    ] if isinstance(case.get("operands", []), Sequence) else []
    ranked_pages = list(dict.fromkeys(int(page) for page in retrieved_pages))
    retrieved_page_set = set(ranked_pages)
    selected_page_set = (
        _chunk_pages(page_chunks)
        if page_chunks is not None
        else set(retrieved_page_set)
    )
    operands: list[dict[str, object]] = []
    for expected in expected_operands:
        value = str(expected.get("value", ""))
        page = int(expected["pdf_page"])
        prequota_exact = _matches(prequota_candidates, value=value, page=page)
        final_exact = _matches(final_candidates, value=value, page=page)
        prequota_any_page = _matches(prequota_candidates, value=value, page=None)
        final_any_page = _matches(final_candidates, value=value, page=None)
        cutoff_32_exact = _matches(
            candidates_at_32 if candidates_at_32 is not None else final_candidates,
            value=value,
            page=page,
        )
        cutoff_64_exact = _matches(
            candidates_at_64 if candidates_at_64 is not None else final_candidates,
            value=value,
            page=page,
        )
        raw_on_selected_page = (
            _raw_matches(page_chunks, value=value, page=page)
            if page_chunks is not None
            else page in selected_page_set
        )
        raw_in_retrieved_chunks = (
            _raw_matches(retrieved_chunks, value=value, page=page)
            if retrieved_chunks is not None
            else page in retrieved_page_set
        )
        extraction_chunks = (
            candidate_chunks
            if candidate_chunks is not None
            else retrieved_chunks
        )
        raw_in_candidate_chunks = (
            _raw_matches(extraction_chunks, value=value, page=page)
            if extraction_chunks is not None
            else page in retrieved_page_set
        )
        if page not in selected_page_set:
            loss_stage = "page_retrieval"
        elif not raw_on_selected_page:
            loss_stage = "source_representation"
        elif not raw_in_candidate_chunks:
            loss_stage = "chunk_retrieval"
        elif not prequota_exact:
            loss_stage = "provenance" if prequota_any_page else "extraction"
        elif not final_exact:
            loss_stage = "quota"
        else:
            loss_stage = "available"
        operands.append(
            {
                "role": str(expected.get("role", "")),
                "value": value,
                "expected_page": page,
                "page_selected": page in selected_page_set,
                "retrieved_page": page in retrieved_page_set,
                "gold_page_at_5": page in set(ranked_pages[:5]),
                "gold_page_at_10": page in set(ranked_pages[:10]),
                "raw_value_on_selected_page": raw_on_selected_page,
                "raw_value_in_retrieved_chunks": raw_in_retrieved_chunks,
                "raw_value_in_candidate_chunks": raw_in_candidate_chunks,
                "prequota_present": bool(prequota_exact),
                "prequota_present_any_page": bool(prequota_any_page),
                "final_present": bool(final_exact),
                "final_present_any_page": bool(final_any_page),
                "present_at_32": bool(cutoff_32_exact),
                "present_at_64": bool(cutoff_64_exact),
                "loss_stage": loss_stage,
                "prequota_matches": prequota_exact,
                "final_matches": final_exact,
            }
        )
    return {
        "case_id": str(case.get("case_id", case.get("case", ""))),
        "document": str(case.get("document", "")),
        "question": str(case.get("question", "")),
        "operation": str(case.get("operation", "")),
        "availability": "pipeline_failure" if error else "evaluated",
        "error": str(error) if error else None,
        "selected_pages": sorted(selected_page_set),
        "retrieved_pages": ranked_pages,
        "prequota_candidate_count": len(prequota_candidates),
        "final_candidate_count": len(final_candidates),
        "operands": operands,
        "all_operands_prequota": bool(operands)
        and all(bool(item["prequota_present"]) for item in operands),
        "all_operands_final": bool(operands)
        and all(bool(item["final_present"]) for item in operands),
        "all_gold_pages_at_5": bool(operands)
        and all(bool(item["gold_page_at_5"]) for item in operands),
        "all_gold_pages_at_10": bool(operands)
        and all(bool(item["gold_page_at_10"]) for item in operands),
        "all_operands_at_32": bool(operands)
        and all(bool(item["present_at_32"]) for item in operands),
        "all_operands_at_64": bool(operands)
        and all(bool(item["present_at_64"]) for item in operands),
    }


def aggregate_candidate_availability(
    cases: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    evaluated = [item for item in cases if item.get("availability") == "evaluated"]
    operands = [
        operand
        for item in evaluated
        for operand in item.get("operands", [])
        if isinstance(operand, Mapping)
    ]
    pair_cases = [
        item
        for item in evaluated
        if len(
            [
                operand
                for operand in item.get("operands", [])
                if isinstance(operand, Mapping)
            ]
        )
        >= 2
    ]
    cases_count = len(evaluated)
    operands_count = len(operands)
    page_at_5 = sum(
        item.get("all_gold_pages_at_5") is True for item in evaluated
    )
    page_at_10 = sum(
        item.get("all_gold_pages_at_10") is True for item in evaluated
    )
    operand_at_32 = sum(
        operand.get("present_at_32") is True for operand in operands
    )
    operand_at_64 = sum(
        operand.get("present_at_64") is True for operand in operands
    )
    pair_at_64 = sum(
        item.get("all_operands_at_64") is True for item in pair_cases
    )
    return {
        "cases": len(cases),
        "evaluated_cases": len(evaluated),
        "pipeline_failures": len(cases) - len(evaluated),
        "cases_with_all_operands_prequota": sum(
            item.get("all_operands_prequota") is True for item in evaluated
        ),
        "cases_with_all_operands_final": sum(
            item.get("all_operands_final") is True for item in evaluated
        ),
        "operands": len(operands),
        "operands_prequota": sum(
            operand.get("prequota_present") is True for operand in operands
        ),
        "operands_final": sum(
            operand.get("final_present") is True for operand in operands
        ),
        "gold_page_recall_at_5": {
            "numerator": page_at_5,
            "denominator": cases_count,
            "recall": page_at_5 / cases_count if cases_count else 0.0,
        },
        "gold_page_recall_at_10": {
            "numerator": page_at_10,
            "denominator": cases_count,
            "recall": page_at_10 / cases_count if cases_count else 0.0,
        },
        "operand_recall_at_32": {
            "numerator": operand_at_32,
            "denominator": operands_count,
            "recall": operand_at_32 / operands_count if operands_count else 0.0,
        },
        "operand_recall_at_64": {
            "numerator": operand_at_64,
            "denominator": operands_count,
            "recall": operand_at_64 / operands_count if operands_count else 0.0,
        },
        "pair_recall_at_64": {
            "numerator": pair_at_64,
            "denominator": len(pair_cases),
            "recall": pair_at_64 / len(pair_cases) if pair_cases else 0.0,
        },
        "loss_stages": {
            stage: sum(operand.get("loss_stage") == stage for operand in operands)
            for stage in (
                "available",
                "page_retrieval",
                "source_representation",
                "chunk_retrieval",
                "extraction",
                "provenance",
                "quota",
            )
        },
    }


def _matches(
    candidates: Sequence[object], *, value: str, page: int | None
) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_value = str(_field(candidate, "value", ""))
        candidate_page = _field(candidate, "page", None)
        if candidate_value != value:
            continue
        if page is not None and candidate_page != page:
            continue
        matches.append(
            {
                "index": _field(candidate, "index", None),
                "page": candidate_page,
                "relevance": _field(candidate, "relevance", None),
                "context": str(_field(candidate, "context", "")),
            }
        )
    return matches


def _chunk_pages(chunks: Sequence[object]) -> set[int]:
    pages: set[int] = set()
    for chunk in chunks:
        start = _field(chunk, "page_start", None)
        end = _field(chunk, "page_end", start)
        if isinstance(start, int):
            final = end if isinstance(end, int) and end >= start else start
            pages.update(range(start, final + 1))
    return pages


def _raw_matches(
    chunks: Sequence[object] | None, *, value: str, page: int
) -> bool:
    if chunks is None:
        return False
    expected = parse_decimal(value)
    if expected is None:
        return False
    for chunk in chunks:
        start = _field(chunk, "page_start", None)
        end = _field(chunk, "page_end", start)
        if not isinstance(start, int):
            continue
        final = end if isinstance(end, int) and end >= start else start
        if not start <= page <= final:
            continue
        for match in _RAW_NUMBER_RE.finditer(str(_field(chunk, "text", ""))):
            observed = parse_decimal(match.group(0))
            if isinstance(observed, Decimal) and observed == expected:
                return True
    return False


def _field(item: object, name: str, default: object) -> object:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)
