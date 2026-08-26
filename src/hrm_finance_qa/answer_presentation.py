from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, localcontext
from enum import Enum
import hashlib
import json
from typing import Sequence

from .program_data import COMMUTATIVE, currency_unit_scale
from .question_planner import compile_question_plan


_SCHEMA_VERSION = "finhrm-verified-answer-envelope-v1"
_LOCALE = "vi-VN"
_ALLOWED_SCALES = {"1", "1000", "1000000", "1000000000"}
_OPERATIONS = {
    "SELECT",
    "ADD",
    "SUBTRACT",
    "MULTIPLY",
    "DIVIDE",
    "RATIO_PERCENT",
    "PERCENT_CHANGE",
    "AVERAGE",
    "SUM3",
    "SUM4",
    "SHARE_OF_TOTAL",
}


class AnswerStatus(str, Enum):
    ANSWER = "ANSWER"
    ABSTAIN = "ABSTAIN"
    REJECT = "REJECT"


class AnswerMode(str, Enum):
    FACT = "FACT"
    COMPUTATION = "COMPUTATION"
    VERIFIED_SUMMARY = "VERIFIED_SUMMARY"


@dataclass(frozen=True)
class VerifiedCitation:
    operand_ref: str
    value: str
    unit: str
    scale: str
    period: str
    entity_scope: str
    source_name: str
    source_span: tuple[int, int] | None
    logical_cell_id: str
    extraction_method: str
    document_title: str | None = None
    statement_or_note: str | None = None
    page: int | None = None
    table_id: str | None = None
    row_label: str | None = None
    column_label: str | None = None


@dataclass(frozen=True)
class VerifiedClaim:
    metric: str | None
    period: str | None
    entity_scope: str | None
    value: str | None
    unit: str | None
    scale: str | None
    qualitative_highlight: str | None = None
    qualitative_citation: VerifiedCitation | None = None


@dataclass(frozen=True)
class VerifiedProof:
    source_action: str
    verification_state: str
    obligations: tuple[tuple[str, str, str | None], ...]
    acceptance_basis: tuple[str, ...]


@dataclass(frozen=True)
class SummaryCoverage:
    requested_count: int
    verified_count: int
    missing_fields: tuple[str, ...]
    conflicting_fields: tuple[str, ...]
    coverage_state: str
    requested_fields: tuple[str, ...]
    rows: tuple[tuple[str, "VerifiedAnswerEnvelope"], ...]


@dataclass(frozen=True)
class VerifiedAnswerEnvelope:
    schema_version: str
    status: AnswerStatus
    answer_mode: AnswerMode
    locale: str
    claim: VerifiedClaim | None
    canonical_value: str | None
    display: tuple[str, str, str, str] | None
    program: tuple[tuple[str, str, tuple[str, ...], str], ...]
    program_output: str | None
    citations: tuple[VerifiedCitation, ...]
    proof: VerifiedProof
    summary_coverage: SummaryCoverage | None
    warnings: tuple[str, ...]


def _decimal_text(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


def _execute(operation: str, values: Sequence[Decimal]) -> Decimal:
    with localcontext() as context:
        context.prec = 34
        if operation == "SELECT":
            return values[0]
        if operation == "ADD":
            return values[0] + values[1]
        if operation == "SUBTRACT":
            return values[0] - values[1]
        if operation == "MULTIPLY":
            return values[0] * values[1]
        if operation == "DIVIDE":
            if values[1] == 0:
                raise ZeroDivisionError
            return values[0] / values[1]
        if operation in {"RATIO_PERCENT", "SHARE_OF_TOTAL"}:
            if values[1] == 0:
                raise ZeroDivisionError
            return values[0] / values[1] * Decimal(100)
        if operation == "PERCENT_CHANGE":
            if values[1] == 0:
                raise ZeroDivisionError
            return (values[0] - values[1]) / values[1] * Decimal(100)
        if operation == "AVERAGE":
            return (values[0] + values[1]) / Decimal(2)
        if operation in {"SUM3", "SUM4"}:
            return sum(values, Decimal(0))
    raise ValueError(f"unsupported presentation operation: {operation}")


def _citation_payload(citation: VerifiedCitation) -> dict[str, object]:
    return {
        "value": citation.value,
        "unit": citation.unit,
        "scale": citation.scale,
        "period": citation.period,
        "entity_scope": citation.entity_scope,
        "source_name": citation.source_name,
        "source_span": citation.source_span,
        "extraction_method": citation.extraction_method,
        "document_title": citation.document_title,
        "statement_or_note": citation.statement_or_note,
        "page": citation.page,
        "table_id": citation.table_id,
        "row_label": citation.row_label,
        "column_label": citation.column_label,
    }


def _citation_id(citation: VerifiedCitation) -> str:
    payload = json.dumps(
        _citation_payload(citation),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "cell-" + hashlib.sha256(payload).hexdigest()[:24]


def _candidate_scale(unit: str) -> str:
    scale = currency_unit_scale(unit)
    return _decimal_text(scale) if scale is not None else "1"


def _candidate_key(candidate: object) -> tuple[object, ...]:
    span = tuple(getattr(candidate, "source_span", ()) or ())
    return (
        str(getattr(candidate, "source_name", "")),
        int(getattr(candidate, "page", 0) or 0),
        str(getattr(candidate, "table_id", "")),
        span,
        str(getattr(candidate, "row_label", "")),
        str(getattr(candidate, "column_label", "")),
        str(getattr(candidate, "period", "")),
        str(getattr(candidate, "value", "")),
    )


def _make_citation(operand_ref: str, candidate: object) -> VerifiedCitation:
    unit = str(getattr(candidate, "unit", "")).strip()
    scope = str(
        getattr(candidate, "scope", "") or getattr(candidate, "entity_id", "")
    ).strip()
    source_span = getattr(candidate, "source_span", None)
    if source_span is not None:
        source_span = tuple(int(value) for value in source_span)
    citation = VerifiedCitation(
        operand_ref=operand_ref,
        value=str(getattr(candidate, "value", "")).strip(),
        unit=unit,
        scale=_candidate_scale(unit),
        period=str(getattr(candidate, "period", "")).strip(),
        entity_scope=scope,
        source_name=str(getattr(candidate, "source_name", "")).strip(),
        source_span=source_span,
        logical_cell_id="",
        extraction_method=str(getattr(candidate, "extraction_method", "")).strip(),
        page=getattr(candidate, "page", None),
        table_id=str(getattr(candidate, "table_id", "")).strip() or None,
        row_label=str(getattr(candidate, "row_label", "")).strip() or None,
        column_label=str(getattr(candidate, "column_label", "")).strip() or None,
    )
    return replace(citation, logical_cell_id=_citation_id(citation))


def _required_state(
    obligations: Sequence[tuple[str, str, str | None]], names: set[str]
) -> str | None:
    states = [
        state
        for name, applicability, state in obligations
        if applicability == "required" and name in names
    ]
    if "INVALID" in states:
        return "INVALID"
    if "UNKNOWN" in states or None in states:
        return "UNKNOWN"
    if "VALID" in states:
        return "VALID"
    return None


def _program_records(
    answer: object,
    citation_by_index: dict[int, VerifiedCitation],
) -> tuple[
    tuple[tuple[str, str, tuple[str, ...], str], ...],
    str | None,
    str | None,
]:
    values = {
        citation.operand_ref: Decimal(citation.value) * Decimal(citation.scale)
        for citation in citation_by_index.values()
    }
    results: dict[str, Decimal] = {}
    records: list[tuple[str, str, tuple[str, ...], str]] = []
    dag = getattr(answer, "program_dag", None) or getattr(
        getattr(answer, "prediction", None), "program_dag", None
    )
    if dag is not None:
        raw_nodes = tuple(dag.nodes)
        output_node = str(dag.output_node)
    else:
        program = getattr(getattr(answer, "prediction", None), "program", None)
        if program is None:
            return (), None, None
        raw_nodes = (
            _SimpleProgramNode(
                "answer",
                str(program.operation),
                tuple(("candidate", int(index)) for index in program.operands),
            ),
        )
        output_node = "answer"

    for raw_node in raw_nodes:
        operation = str(raw_node.operation).upper()
        inputs: list[str] = []
        for raw_input in raw_node.inputs:
            kind = str(getattr(raw_input, "kind", raw_input[0] if isinstance(raw_input, tuple) else ""))
            reference = getattr(raw_input, "reference", raw_input[1] if isinstance(raw_input, tuple) else None)
            if kind == "candidate" and isinstance(reference, int):
                citation = citation_by_index.get(reference)
                if citation is None:
                    return (), None, None
                inputs.append(citation.operand_ref)
            elif kind == "node" and isinstance(reference, str):
                inputs.append(reference)
            else:
                return (), None, None
        if operation in COMMUTATIVE or operation in {"SUM3", "SUM4"}:
            inputs.sort()
        try:
            input_values = [values[item] if item in values else results[item] for item in inputs]
            result = _execute(operation, input_values)
        except (KeyError, InvalidOperation, ZeroDivisionError, ValueError):
            return (), None, None
        node_id = str(raw_node.node_id)
        results[node_id] = result
        records.append((node_id, operation, tuple(inputs), _decimal_text(result)))
    output = results.get(output_node)
    return tuple(records), output_node, _decimal_text(output) if output is not None else None


@dataclass(frozen=True)
class _SimpleProgramNode:
    node_id: str
    operation: str
    inputs: tuple[tuple[str, int | str], ...]


def _nonanswer(
    envelope: VerifiedAnswerEnvelope, status: AnswerStatus
) -> VerifiedAnswerEnvelope:
    claim = replace(envelope.claim, value=None) if envelope.claim is not None else None
    return replace(
        envelope,
        status=status,
        claim=claim,
        canonical_value=None,
        display=None,
    )


def _build_summary(
    requested_fields: Sequence[str],
    summary_rows: Sequence[tuple[str, VerifiedAnswerEnvelope]],
    conflicting_fields: Sequence[str],
) -> VerifiedAnswerEnvelope:
    requested = tuple(dict.fromkeys(str(item) for item in requested_fields if str(item)))
    conflicts = tuple(
        item for item in requested if item in set(str(value) for value in conflicting_fields)
    )
    row_map: dict[str, VerifiedAnswerEnvelope] = {}
    for row_id, envelope in summary_rows:
        if row_id not in requested or row_id in row_map:
            continue
        if envelope.answer_mode is AnswerMode.VERIFIED_SUMMARY:
            continue
        if envelope.status is not AnswerStatus.ANSWER:
            continue
        if validate_verified_answer_envelope(envelope):
            continue
        row_map[row_id] = envelope
    rows = tuple((row_id, row_map[row_id]) for row_id in requested if row_id in row_map)
    missing = tuple(row_id for row_id in requested if row_id not in row_map)
    verified_count = len(rows)
    if conflicts:
        status = AnswerStatus.REJECT
        coverage_state = "CONFLICT"
        source_action = "REJECT"
        complete_state = "INVALID"
    elif requested and not missing:
        status = AnswerStatus.ANSWER
        coverage_state = "COMPLETE"
        source_action = "ANSWER"
        complete_state = "VALID"
    else:
        status = AnswerStatus.ABSTAIN
        coverage_state = "VERIFIED_PARTIAL"
        source_action = "ABSTAIN"
        complete_state = "UNKNOWN"
    return VerifiedAnswerEnvelope(
        schema_version=_SCHEMA_VERSION,
        status=status,
        answer_mode=AnswerMode.VERIFIED_SUMMARY,
        locale=_LOCALE,
        claim=None,
        canonical_value=None,
        display=None,
        program=(),
        program_output=None,
        citations=tuple(
            citation
            for _, row in rows
            for citation in sorted(row.citations, key=lambda item: item.operand_ref)
        ),
        proof=VerifiedProof(
            source_action=source_action,
            verification_state=complete_state,
            obligations=(("answer_mode_complete", "required", complete_state),),
            acceptance_basis=("independent_verified_subanswers",),
        ),
        summary_coverage=SummaryCoverage(
            requested_count=len(requested),
            verified_count=verified_count,
            missing_fields=missing,
            conflicting_fields=conflicts,
            coverage_state=coverage_state,
            requested_fields=requested,
            rows=rows,
        ),
        warnings=("incomplete",) if status is AnswerStatus.ABSTAIN else (),
    )


def build_verified_answer_envelope(
    answer: object | None = None,
    *,
    answer_mode: AnswerMode | str | None = None,
    requested_fields: Sequence[str] = (),
    summary_rows: Sequence[tuple[str, VerifiedAnswerEnvelope]] = (),
    conflicting_fields: Sequence[str] = (),
) -> VerifiedAnswerEnvelope:
    """Build presentation data only from a frozen runtime result and its proof."""

    if answer_mode is not None:
        answer_mode = AnswerMode(answer_mode)
    if answer_mode is AnswerMode.VERIFIED_SUMMARY:
        return _build_summary(requested_fields, summary_rows, conflicting_fields)
    if answer is None:
        raise ValueError("a scalar reasoning answer is required")

    operation = str(answer.prediction.program.operation).upper()
    mode = answer_mode or (
        AnswerMode.FACT if operation == "SELECT" else AnswerMode.COMPUTATION
    )
    obligations = tuple(
        (str(name), str(applicability), None if state is None else str(state))
        for name, applicability, state in getattr(answer, "proof_obligations", ())
    )
    verification_state = str(getattr(answer, "verification_state", "UNKNOWN"))
    source_action = str(getattr(answer, "source_action", "ABSTAIN"))
    required_states = [
        state for _, applicability, state in obligations if applicability == "required"
    ]
    if verification_state == "INVALID" or "INVALID" in required_states:
        status = AnswerStatus.REJECT
    elif (
        source_action != "ANSWER"
        or not required_states
        or any(state != "VALID" for state in required_states)
        or bool(getattr(answer, "abstained", True))
    ):
        status = AnswerStatus.ABSTAIN
    else:
        status = AnswerStatus.ANSWER

    evidence = tuple(getattr(answer, "evidence", ()))
    ordered_candidates = sorted(
        (item.candidate for item in evidence), key=_candidate_key
    )
    citation_by_index: dict[int, VerifiedCitation] = {}
    for position, candidate in enumerate(ordered_candidates, start=1):
        citation_by_index[int(candidate.index)] = _make_citation(
            f"O{position}", candidate
        )
    citations = tuple(
        sorted(citation_by_index.values(), key=lambda item: item.operand_ref)
    )
    program, output_node, replay_value = _program_records(answer, citation_by_index)

    plan = compile_question_plan(str(getattr(answer, "question", "")))
    metric_state = _required_state(obligations, {"query_metric", "metric_match"})
    period_state = _required_state(obligations, {"period_coverage"})
    entity_state = _required_state(
        obligations, {"reporting_entity", "named_entity", "entity_match"}
    )
    unit_state = _required_state(obligations, {"unit_match", "unit_compatibility"})
    metric = plan.metric_text.strip() if metric_state == "VALID" else None
    periods = tuple(sorted({item.period for item in citations if item.period}))
    period = " / ".join(periods) if periods and period_state == "VALID" else None
    entities = {item.entity_scope for item in citations if item.entity_scope}
    entity_scope = (
        next(iter(entities))
        if len(entities) == 1 and entity_state == "VALID"
        else None
    )
    answer_unit = str(getattr(answer, "unit", "")).strip()
    display_scale = _candidate_scale(answer_unit)
    base_unit = "VND" if currency_unit_scale(answer_unit) is not None else answer_unit
    if unit_state != "VALID":
        base_unit = ""

    canonical_value = replay_value if status is AnswerStatus.ANSWER else None
    display = None
    if status is AnswerStatus.ANSWER and canonical_value is not None:
        display = (
            _decimal_text(Decimal(str(answer.value))),
            answer_unit,
            display_scale,
            _LOCALE,
        )
    claim = VerifiedClaim(
        metric=metric,
        period=period,
        entity_scope=entity_scope,
        value=canonical_value,
        unit=base_unit or None,
        scale="1" if base_unit else None,
    )
    envelope = VerifiedAnswerEnvelope(
        schema_version=_SCHEMA_VERSION,
        status=status,
        answer_mode=mode,
        locale=_LOCALE,
        claim=claim,
        canonical_value=canonical_value,
        display=display,
        program=program,
        program_output=output_node,
        citations=citations,
        proof=VerifiedProof(
            source_action=source_action,
            verification_state=verification_state,
            obligations=obligations,
            acceptance_basis=tuple(getattr(answer, "acceptance_basis", ())),
        ),
        summary_coverage=None,
        warnings=tuple(str(item) for item in getattr(answer, "verification_reasons", ())),
    )
    if mode is AnswerMode.FACT and operation != "SELECT":
        return _nonanswer(envelope, AnswerStatus.REJECT)
    if mode is AnswerMode.COMPUTATION and operation == "SELECT":
        return _nonanswer(envelope, AnswerStatus.REJECT)
    errors = validate_verified_answer_envelope(envelope)
    if status is AnswerStatus.ANSWER and errors:
        contradiction_markers = {
            "program_replay_mismatch",
            "display_value_mismatch",
            "citation_identity_mismatch",
            "wrong_mode_for_program",
        }
        fail_status = (
            AnswerStatus.REJECT
            if any(error in contradiction_markers for error in errors)
            else AnswerStatus.ABSTAIN
        )
        return _nonanswer(envelope, fail_status)
    return envelope


def _validate_program(envelope: VerifiedAnswerEnvelope) -> tuple[str, ...]:
    errors: list[str] = []
    citations = {item.operand_ref: item for item in envelope.citations}
    if len(citations) != len(envelope.citations):
        errors.append("duplicate_operand_citation")
        return tuple(errors)
    values: dict[str, Decimal] = {}
    try:
        values = {
            ref: Decimal(item.value) * Decimal(item.scale)
            for ref, item in citations.items()
        }
    except InvalidOperation:
        errors.append("invalid_operand_decimal")
        return tuple(errors)
    results: dict[str, Decimal] = {}
    for node_id, operation, inputs, recorded in envelope.program:
        if operation not in _OPERATIONS or node_id in results:
            errors.append("invalid_program_node")
            continue
        try:
            input_values = [values[item] if item in values else results[item] for item in inputs]
            result = _execute(operation, input_values)
            if result != Decimal(recorded):
                errors.append("program_replay_mismatch")
            results[node_id] = result
        except (KeyError, InvalidOperation, ZeroDivisionError, ValueError):
            errors.append("invalid_program_input")
    if envelope.program_output not in results:
        errors.append("missing_program_output")
    elif envelope.canonical_value is not None:
        try:
            if results[envelope.program_output] != Decimal(envelope.canonical_value):
                errors.append("program_replay_mismatch")
        except InvalidOperation:
            errors.append("invalid_canonical_decimal")
    return tuple(dict.fromkeys(errors))


def validate_verified_answer_envelope(
    envelope: VerifiedAnswerEnvelope,
) -> tuple[str, ...]:
    errors: list[str] = []
    if envelope.schema_version != _SCHEMA_VERSION:
        errors.append("wrong_schema_version")
    if envelope.locale != _LOCALE:
        errors.append("unsupported_locale")
    required = [
        state
        for _, applicability, state in envelope.proof.obligations
        if applicability == "required"
    ]
    if envelope.proof.verification_state == "INVALID" and envelope.status is not AnswerStatus.REJECT:
        errors.append("invalid_proof_not_rejected")
    if envelope.status is AnswerStatus.ANSWER:
        if envelope.proof.source_action != "ANSWER":
            errors.append("parent_action_promoted")
        if not required or any(state != "VALID" for state in required):
            errors.append("required_proof_not_valid")

    if envelope.answer_mode is AnswerMode.VERIFIED_SUMMARY:
        coverage = envelope.summary_coverage
        if coverage is None:
            return tuple((*errors, "missing_summary_coverage"))
        if coverage.requested_count != len(coverage.requested_fields):
            errors.append("summary_requested_count_mismatch")
        if coverage.verified_count != len(coverage.rows):
            errors.append("summary_verified_count_mismatch")
        row_ids = tuple(row_id for row_id, _ in coverage.rows)
        if len(set(row_ids)) != len(row_ids):
            errors.append("duplicate_summary_row")
        if any(row_id not in coverage.requested_fields for row_id in row_ids):
            errors.append("summary_row_not_requested")
        for _, row in coverage.rows:
            if row.status is not AnswerStatus.ANSWER or validate_verified_answer_envelope(row):
                errors.append("unverified_summary_row")
            if (
                row.claim is not None
                and row.claim.qualitative_highlight
                and row.claim.qualitative_citation is None
            ):
                errors.append("uncited_qualitative_claim")
        if coverage.coverage_state == "COMPLETE":
            if coverage.missing_fields or coverage.conflicting_fields:
                errors.append("false_complete_summary")
            if coverage.verified_count != coverage.requested_count:
                errors.append("false_complete_summary")
            if envelope.status is not AnswerStatus.ANSWER:
                errors.append("complete_summary_not_answer")
        elif coverage.coverage_state == "VERIFIED_PARTIAL":
            if envelope.status is not AnswerStatus.ABSTAIN:
                errors.append("partial_summary_not_abstain")
        elif coverage.coverage_state == "CONFLICT":
            if envelope.status is not AnswerStatus.REJECT or not coverage.conflicting_fields:
                errors.append("summary_conflict_not_rejected")
        else:
            errors.append("invalid_summary_coverage_state")
        return tuple(dict.fromkeys(errors))

    if envelope.summary_coverage is not None:
        errors.append("unexpected_summary_coverage")
    if envelope.status is not AnswerStatus.ANSWER:
        if envelope.canonical_value is not None or envelope.display is not None:
            errors.append("nonanswer_exposes_accepted_value")
        if envelope.claim is not None and envelope.claim.value is not None:
            errors.append("nonanswer_claim_has_value")
        return tuple(dict.fromkeys(errors))

    if envelope.claim is None:
        errors.append("missing_claim")
    else:
        for field in (
            envelope.claim.metric,
            envelope.claim.period,
            envelope.claim.entity_scope,
            envelope.claim.value,
            envelope.claim.unit,
            envelope.claim.scale,
        ):
            if field is None or not str(field).strip():
                errors.append("incomplete_answer_claim")
                break
        if envelope.claim.value != envelope.canonical_value:
            errors.append("claim_value_mismatch")
        if envelope.claim.qualitative_highlight and envelope.claim.qualitative_citation is None:
            errors.append("uncited_qualitative_claim")
    if envelope.canonical_value is None or envelope.display is None:
        errors.append("missing_answer_value")
    else:
        display_value, display_unit, display_scale, display_locale = envelope.display
        if display_scale not in _ALLOWED_SCALES or display_locale != _LOCALE or not display_unit:
            errors.append("invalid_display_contract")
        try:
            if Decimal(display_value) * Decimal(display_scale) != Decimal(envelope.canonical_value):
                errors.append("display_value_mismatch")
        except InvalidOperation:
            errors.append("invalid_display_decimal")
    if not envelope.citations:
        errors.append("missing_citation")
    for citation in envelope.citations:
        if (
            not citation.source_name
            or citation.source_span is None
            or len(citation.source_span) != 2
            or not citation.logical_cell_id
            or not citation.extraction_method
            or not citation.value
            or not citation.unit
            or citation.scale not in _ALLOWED_SCALES
            or not citation.period
            or not citation.entity_scope
        ):
            errors.append("incomplete_citation")
        if citation.logical_cell_id != _citation_id(citation):
            errors.append("citation_identity_mismatch")
    errors.extend(_validate_program(envelope))
    root_operation = envelope.program[-1][1] if envelope.program else ""
    if envelope.answer_mode is AnswerMode.FACT:
        if root_operation != "SELECT" or len(envelope.citations) != 1:
            errors.append("wrong_mode_for_program")
    elif envelope.answer_mode is AnswerMode.COMPUTATION:
        if root_operation == "SELECT" or not root_operation:
            errors.append("wrong_mode_for_program")
    return tuple(dict.fromkeys(errors))


def _format_vi(value: str) -> str:
    decimal = Decimal(value)
    sign = "-" if decimal < 0 else ""
    absolute = _decimal_text(abs(decimal))
    integer, dot, fraction = absolute.partition(".")
    grouped = ".".join(
        [integer[max(0, len(integer) - offset - 3) : len(integer) - offset] for offset in range(0, len(integer), 3)][::-1]
    )
    return sign + grouped + (("," + fraction) if dot else "")


def _render_citation(citation: VerifiedCitation) -> str:
    parts = [citation.document_title or citation.source_name]
    if citation.statement_or_note:
        parts.append(citation.statement_or_note)
    if citation.page is not None:
        parts.append(f"trang {citation.page}")
    if citation.table_id:
        parts.append(f"bảng {citation.table_id}")
    if citation.row_label:
        parts.append(f"dòng {citation.row_label}")
    if citation.column_label:
        parts.append(f"cột {citation.column_label}")
    if citation.source_span is not None:
        parts.append(
            f"vùng ký tự {citation.source_span[0]}–{citation.source_span[1]}"
        )
    return ", ".join(parts)


def _formula(envelope: VerifiedAnswerEnvelope) -> str:
    expressions = {citation.operand_ref: citation.operand_ref for citation in envelope.citations}
    for node_id, operation, inputs, _ in envelope.program:
        values = [expressions[item] for item in inputs]
        if operation == "SELECT":
            expression = values[0]
        elif operation == "ADD":
            expression = "(" + " + ".join(values) + ")"
        elif operation == "SUBTRACT":
            expression = f"({values[0]} - {values[1]})"
        elif operation == "MULTIPLY":
            expression = f"({values[0]} × {values[1]})"
        elif operation == "DIVIDE":
            expression = f"({values[0]} / {values[1]})"
        elif operation in {"RATIO_PERCENT", "SHARE_OF_TOTAL"}:
            expression = f"({values[0]} / {values[1]} × 100)"
        elif operation == "PERCENT_CHANGE":
            expression = f"(({values[0]} - {values[1]}) / {values[1]} × 100)"
        elif operation == "AVERAGE":
            expression = f"(({values[0]} + {values[1]}) / 2)"
        else:
            expression = "(" + " + ".join(values) + ")"
        expressions[node_id] = expression
    return expressions[envelope.program_output]


def _summary_source(row: VerifiedAnswerEnvelope) -> str:
    return "; ".join(
        _render_citation(citation)
        for citation in sorted(row.citations, key=lambda item: item.operand_ref)
    )


def render_verified_answer(envelope: VerifiedAnswerEnvelope) -> str:
    """Render only validated envelope fields using fixed Vietnamese templates."""

    errors = validate_verified_answer_envelope(envelope)
    if errors:
        raise ValueError("invalid verified answer envelope: " + ", ".join(errors))
    if envelope.answer_mode is AnswerMode.VERIFIED_SUMMARY:
        coverage = envelope.summary_coverage
        assert coverage is not None
        if envelope.status is AnswerStatus.REJECT:
            return (
                "Bằng chứng mâu thuẫn; không thể chấp nhận bản tóm tắt. "
                "Mục mâu thuẫn: " + ", ".join(coverage.conflicting_fields) + "."
            )
        prefix = (
            "Tóm tắt đã xác minh đầy đủ."
            if envelope.status is AnswerStatus.ANSWER
            else f"Tóm tắt đã xác minh một phần: {coverage.verified_count}/{coverage.requested_count} mục; chưa đủ bằng chứng cho bản tóm tắt đầy đủ."
        )
        lines = [
            prefix,
            "Nội dung | Số liệu/so sánh | Điểm nổi bật | Nguồn",
            "--- | --- | --- | ---",
        ]
        row_map = dict(coverage.rows)
        for row_id in coverage.requested_fields:
            row = row_map.get(row_id)
            if row is None or row.claim is None or row.display is None:
                continue
            highlight = row.claim.qualitative_highlight or ""
            source = _summary_source(row)
            cells = (
                row.claim.metric or row_id,
                f"{_format_vi(row.display[0])} {row.display[1]}".strip(),
                highlight,
                source,
            )
            lines.append(" | ".join(cell.replace("|", "\\|") for cell in cells))
        if coverage.missing_fields:
            lines.append("Chưa xác minh: " + ", ".join(coverage.missing_fields) + ".")
        return "\n".join(lines)

    if envelope.status is AnswerStatus.REJECT:
        return "Bằng chứng mâu thuẫn; không thể chấp nhận câu trả lời."
    if envelope.status is AnswerStatus.ABSTAIN:
        return "Không đủ bằng chứng đã xác minh để trả lời."
    assert envelope.claim is not None and envelope.display is not None
    value = f"{_format_vi(envelope.display[0])} {envelope.display[1]}".strip()
    if envelope.answer_mode is AnswerMode.FACT:
        return (
            f"{envelope.claim.metric} ({envelope.claim.period}) là {value}. "
            f"Nguồn: {_render_citation(envelope.citations[0])}."
        )
    lines = [
        f"Kết quả: {envelope.claim.metric} ({envelope.claim.period}) là {value}.",
        f"Công thức: {_formula(envelope)} = {_format_vi(envelope.display[0])}.",
        "Các toán hạng đã xác minh:",
    ]
    for citation in sorted(envelope.citations, key=lambda item: item.operand_ref):
        lines.append(
            f"- Toán hạng {citation.operand_ref}: {_format_vi(citation.value)} {citation.unit}. "
            f"Nguồn: {_render_citation(citation)}."
        )
    lines.append(
        f"Kết luận: {envelope.claim.metric} ({envelope.claim.period}) là {value}."
    )
    return "\n".join(lines)
