from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Mapping, Sequence


def decimal_or_none(value: object) -> Decimal | None:
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None


def close_enough(actual: Decimal, expected: Decimal) -> bool:
    difference = abs(actual - expected)
    scale = max(abs(expected), Decimal(1))
    tolerance = (
        Decimal("0.035")
        if scale < Decimal(1000)
        else max(Decimal("0.5"), scale * Decimal("0.00000000001"))
    )
    return difference <= tolerance


def score_document_response(
    case: Mapping[str, object],
    response: Mapping[str, object],
    *,
    error: object = None,
) -> dict[str, object]:
    expected_operation = str(case.get("operation", ""))
    response_program = response.get("program")
    actual_operation = (
        str(response_program.get("operation", ""))
        if isinstance(response_program, Mapping)
        else ""
    )
    expected_answer = decimal_or_none(case.get("answer_decimal"))
    actual_answer = decimal_or_none(response.get("answer"))
    status = str(response.get("status", "")).upper()

    error_text = str(error).lower() if error else ""
    if error and any(
        marker in error_text for marker in ('"status": "not_ready"', "not_ready")
    ):
        availability = "not_ready"
    elif error:
        availability = "pipeline_failure"
    elif status in {"NOT_READY", "NOT READY"}:
        availability = "not_ready"
    elif not actual_operation or actual_answer is None:
        availability = "no_prediction"
    else:
        availability = "prediction"

    is_prediction = availability == "prediction"
    operation_correct = (
        actual_operation == expected_operation if is_prediction else None
    )
    execution_correct = (
        bool(
            expected_answer is not None
            and actual_answer is not None
            and close_enough(actual_answer, expected_answer)
        )
        if is_prediction
        else None
    )

    expected_operands = [
        item for item in case.get("operands", []) if isinstance(item, Mapping)
    ] if isinstance(case.get("operands", []), Sequence) else []
    expected_pages = sorted(
        {
            int(item["pdf_page"])
            for item in expected_operands
            if item.get("pdf_page") is not None
        }
    )
    evidence = [
        item for item in response.get("evidence", []) if isinstance(item, Mapping)
    ] if isinstance(response.get("evidence", []), Sequence) else []
    actual_pages = sorted(
        {
            int(item["page"])
            for item in evidence
            if item.get("page") is not None
        }
    )
    expected_page_set = set(expected_pages)
    actual_page_set = set(actual_pages)
    page_provenance_exact = (
        bool(expected_page_set) and expected_page_set == actual_page_set
        if is_prediction
        else None
    )
    page_provenance_expected_covered = (
        bool(expected_page_set) and expected_page_set.issubset(actual_page_set)
        if is_prediction
        else None
    )
    page_provenance_any = (
        bool(expected_page_set & actual_page_set) if is_prediction else None
    )

    operand_provenance = _score_operand_provenance(expected_operands, evidence)
    operand_provenance_all = (
        bool(operand_provenance)
        and all(bool(item["exact"]) for item in operand_provenance)
        if is_prediction
        else None
    )
    accepted = status == "ACCEPTED" and is_prediction
    grounded_correct = bool(
        operation_correct and execution_correct and operand_provenance_all
    ) if is_prediction else None

    return {
        "case_id": str(case.get("case_id", case.get("case", ""))),
        "availability": availability,
        "error": str(error) if error else None,
        "expected_operation": expected_operation,
        "actual_operation": actual_operation,
        "operation_correct": operation_correct,
        "expected_answer": (
            str(expected_answer) if expected_answer is not None else None
        ),
        "actual_answer": str(actual_answer) if actual_answer is not None else None,
        "execution_correct": execution_correct,
        "expected_pages": expected_pages,
        "actual_pages": actual_pages,
        "page_provenance_exact": page_provenance_exact,
        "page_provenance_expected_covered": page_provenance_expected_covered,
        "page_provenance_any": page_provenance_any,
        "operand_provenance": operand_provenance,
        "operand_provenance_all": operand_provenance_all,
        "grounded_correct": grounded_correct,
        "status": response.get("status"),
        "accepted": accepted,
        "confidence": response.get("confidence"),
        "unit": response.get("unit"),
    }


def _score_operand_provenance(
    expected_operands: Sequence[Mapping[str, object]],
    evidence: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    used_evidence: set[int] = set()
    for index, expected in enumerate(expected_operands, start=1):
        expected_value = decimal_or_none(expected.get("value"))
        expected_page = (
            int(expected["pdf_page"])
            if expected.get("pdf_page") is not None
            else None
        )
        numbered = [
            (evidence_index, item)
            for evidence_index, item in enumerate(evidence)
            if evidence_index not in used_evidence
            and item.get("operand_number") == index
        ]
        pool = numbered or [
            (evidence_index, item)
            for evidence_index, item in enumerate(evidence)
            if evidence_index not in used_evidence
        ]
        ranked: list[tuple[int, int, Mapping[str, object]]] = []
        for evidence_index, item in pool:
            actual_value = decimal_or_none(item.get("value"))
            actual_page = (
                int(item["page"]) if item.get("page") is not None else None
            )
            value_correct = bool(
                expected_value is not None
                and actual_value is not None
                and actual_value == expected_value
            )
            page_correct = expected_page is not None and actual_page == expected_page
            ranked.append(
                (
                    int(value_correct) + int(page_correct),
                    evidence_index,
                    item,
                )
            )
        selected = max(ranked, default=None, key=lambda item: item[0])
        actual: Mapping[str, object] = selected[2] if selected else {}
        if selected:
            used_evidence.add(selected[1])
        actual_value = decimal_or_none(actual.get("value"))
        actual_page = (
            int(actual["page"]) if actual.get("page") is not None else None
        )
        value_correct = bool(
            expected_value is not None
            and actual_value is not None
            and actual_value == expected_value
        )
        page_correct = expected_page is not None and actual_page == expected_page
        output.append(
            {
                "operand_number": index,
                "role": str(expected.get("role", "")),
                "expected_value": (
                    str(expected_value) if expected_value is not None else None
                ),
                "actual_value": (
                    str(actual_value) if actual_value is not None else None
                ),
                "expected_page": expected_page,
                "actual_page": actual_page,
                "value_correct": value_correct,
                "page_correct": page_correct,
                "exact": value_correct and page_correct,
            }
        )
    return output


def aggregate_document_scores(
    scored: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    total = len(scored)
    predictions = [item for item in scored if item.get("availability") == "prediction"]
    accepted = [item for item in predictions if item.get("accepted")]
    accepted_answer_correct = [
        item
        for item in accepted
        if item.get("operation_correct") and item.get("execution_correct")
    ]
    accepted_grounded_correct = [
        item for item in accepted if item.get("grounded_correct")
    ]
    accepted_count = len(accepted)
    return {
        "cases": total,
        "predictions": len(predictions),
        "accepted": accepted_count,
        "abstained": sum(
            str(item.get("status", "")).upper() == "ABSTAIN"
            for item in predictions
        ),
        "not_ready": sum(
            item.get("availability") == "not_ready" for item in scored
        ),
        "pipeline_failures": sum(
            item.get("availability") == "pipeline_failure" for item in scored
        ),
        "no_prediction": sum(
            item.get("availability") == "no_prediction" for item in scored
        ),
        "operation_correct": sum(
            item.get("operation_correct") is True for item in predictions
        ),
        "execution_correct": sum(
            item.get("execution_correct") is True for item in predictions
        ),
        "page_provenance_exact": sum(
            item.get("page_provenance_exact") is True for item in predictions
        ),
        "page_provenance_expected_covered": sum(
            item.get("page_provenance_expected_covered") is True
            for item in predictions
        ),
        "page_provenance_any": sum(
            item.get("page_provenance_any") is True for item in predictions
        ),
        "operand_provenance_all": sum(
            item.get("operand_provenance_all") is True for item in predictions
        ),
        "accepted_answer_correct": len(accepted_answer_correct),
        "accepted_grounded_correct": len(accepted_grounded_correct),
        "accepted_answer_precision": _ratio_text(
            len(accepted_answer_correct), accepted_count
        ),
        "accepted_grounded_precision": _ratio_text(
            len(accepted_grounded_correct), accepted_count
        ),
    }


def render_document_score_markdown(
    score: Mapping[str, object], *, title: str
) -> str:
    metrics = score["metrics"]
    cases = score["cases"]
    total = int(metrics["cases"])
    predictions = int(metrics["predictions"])
    accepted = int(metrics["accepted"])
    answer_correct = int(metrics["accepted_answer_correct"])
    grounded_correct = int(metrics["accepted_grounded_correct"])
    rows = [
        f"# {title}",
        "",
        f"- Evaluation cases: {total}",
        f"- Predictions: {predictions}/{total}",
        f"- Accepted: {accepted}/{total}",
        f"- ABSTAIN: {metrics['abstained']}/{total}",
        f"- NOT_READY: {metrics['not_ready']}/{total}",
        f"- Pipeline failures: {metrics['pipeline_failures']}/{total}",
        f"- No prediction: {metrics['no_prediction']}/{total}",
        f"- Operation accuracy: {metrics['operation_correct']}/{predictions}",
        f"- Execution/value accuracy: {metrics['execution_correct']}/{predictions}",
        f"- Exact page provenance: {metrics['page_provenance_exact']}/{predictions}",
        "- Expected-page coverage: "
        f"{metrics['page_provenance_expected_covered']}/{predictions}",
        f"- Any-page provenance: {metrics['page_provenance_any']}/{predictions}",
        f"- Operand provenance: {metrics['operand_provenance_all']}/{predictions}",
        f"- Accepted coverage: {accepted}/{total}",
        f"- Accepted answer precision: {answer_correct}/{accepted}"
        if accepted
        else "- Accepted answer precision: n/a (0 accepted)",
        f"- Accepted grounded precision: {grounded_correct}/{accepted}"
        if accepted
        else "- Accepted grounded precision: n/a (0 accepted)",
        "",
        "| Case | Availability | Operation | Value | Exact pages | Operands | Status |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for item in cases:
        rows.append(
            "| {case_id} | {availability} | {operation} | {value} | "
            "{pages} | {operands} | {status} |".format(
                case_id=item["case_id"],
                availability=item["availability"],
                operation=_pass_fail_na(item.get("operation_correct")),
                value=_pass_fail_na(item.get("execution_correct")),
                pages=_pass_fail_na(item.get("page_provenance_exact")),
                operands=_pass_fail_na(item.get("operand_provenance_all")),
                status=item.get("status") or "n/a",
            )
        )
    return "\n".join(rows) + "\n"


def _ratio_text(numerator: int, denominator: int) -> str | None:
    if not denominator:
        return None
    return str(Decimal(numerator) / Decimal(denominator))


def _pass_fail_na(value: object) -> str:
    if value is True:
        return "PASS"
    if value is False:
        return "FAIL"
    return "n/a"
