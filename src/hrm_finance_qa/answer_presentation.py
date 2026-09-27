from __future__ import annotations

from .physical_coordinates import LEGACY_SPAN, physical_coordinate_key, is_versioned_coordinate

from .source_action_policy import AnswerDecision, finish_answer_decision

from dataclasses import asdict, dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from enum import Enum
import hashlib
import json
import re
from types import SimpleNamespace
from typing import Sequence

from .ontology import normalize_text
from .exact_dag_executor import (
    execute_program_records,
    export_lean4_dag,
    verify_dag_proof_certificate,
)
from .exact_executor import (
    ExactExecutionError,
    ExactOperand,
    UnsupportedOperationError,
    execute_decimal_operation,
)
from .program_data import (
    COMMUTATIVE,
    PROGRAM_OPERATIONS,
    ProgramDataError,
    currency_unit_scale,
    dag_operation_arity_bounds,
    is_sum_operation,
)
from .question_planner import compile_question_plan
from .relative_measurement import same_row_measurement_basis
from .report_context import validated_report_field, report_field_matches


_SCHEMA_VERSION = "finhrm-verified-answer-envelope-v1"
_LOCALE = "vi-VN"
_ALLOWED_SCALES = {"1", "1000", "1000000", "1000000000"}
_OPERATIONS = {*PROGRAM_OPERATIONS, "SHARE_OF_TOTAL"}


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
    source_span_scope: str = LEGACY_SPAN
    source_line_sha256: str = ""
    line_number: int | None = None
    row_id: str = ""
    column_id: str = ""
    report_field: dict | None = None
    unit_assumption: str = ""


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
    runtime_source_action: str = ""
    evidence_resolution: str = "UNKNOWN"
    evidence_resolution_reasons: tuple[str, ...] = ()
    arithmetic_verifier: str = ""
    execution_certificate: dict[str, object] | None = None
    action_decision: AnswerDecision | None = None
    supporting_disclosures: tuple[dict, ...] = ()
    applicability_details: tuple[dict, ...] = ()
    lookup_recovery: dict | None = None


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
    # ``Decimal.normalize()`` applies the ambient context and can silently
    # round a 34-digit verified division back to the process default of 28
    # digits. Format the already-computed Decimal directly, then remove only
    # insignificant trailing zeros.
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _execute(operation: str, values: Sequence[Decimal]) -> Decimal:
    try:
        return execute_decimal_operation(operation, values)
    except UnsupportedOperationError as exc:
        raise ValueError(f"unsupported presentation operation: {operation}") from exc


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
        **({"report_field": citation.report_field} if citation.report_field else {}),
        **({"unit_assumption": citation.unit_assumption} if citation.unit_assumption else {}),
        **({"source_span_scope": citation.source_span_scope, "line_number": citation.line_number,
            "row_id": citation.row_id, "column_id": citation.column_id,
            "source_line_sha256": citation.source_line_sha256}
           if citation.source_span_scope != LEGACY_SPAN else {}),
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
    """Return a scale only when the source unit states one explicitly."""

    scale = currency_unit_scale(unit)
    if scale is not None:
        return _decimal_text(scale)
    if "%" in unit:
        return "1"
    normalized = normalize_text(unit).strip()
    if not normalized:
        return ""
    # A named non-currency unit is explicitly on a 1:1 display basis. Currency
    # wording without a parseable denomination is different: its scale is
    # missing and must never silently become one.
    currency_markers = {
        "currency",
        "dong",
        "vnd",
        "nghin",
        "ngan",
        "trieu",
        "ty",
        "thousand",
        "million",
        "billion",
    }
    if currency_markers.intersection(normalized.split()):
        return ""
    return "1"


def _candidate_key(candidate: object) -> tuple[object, ...]:
    span = tuple(getattr(candidate, "source_span", ()) or ())
    return (
        str(getattr(candidate, "source_name", "")),
        int(getattr(candidate, "page", 0) or 0),
        str(getattr(candidate, "table_id", "")),
        span,
        (str(getattr(candidate, "source_span_scope", LEGACY_SPAN)),
         getattr(candidate, "line_number", 0) or 0) if is_versioned_coordinate(candidate) else (),
        str(getattr(candidate, "row_label", "")),
        str(getattr(candidate, "column_label", "")),
        str(getattr(candidate, "period", "")),
        str(getattr(candidate, "value", "")),
    )


def _make_citation(operand_ref: str, candidate: object) -> VerifiedCitation:
    from .program_data import _column_period
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
        unit_assumption=getattr(candidate, "unit_assumption", ""),
        scale=_candidate_scale(unit),
        period=str(getattr(candidate, "period", "")).strip() or _column_period(str(getattr(candidate, "column_label", ""))),
        entity_scope=scope,
        source_name=str(getattr(candidate, "source_name", "")).strip(),
        source_span=source_span,
        source_span_scope=getattr(candidate, "source_span_scope", LEGACY_SPAN),
        source_line_sha256=getattr(candidate, "source_line_sha256", ""),
        line_number=getattr(candidate, "line_number", None),
        row_id=getattr(candidate, "row_id", ""), column_id=getattr(candidate, "column_id", ""),
        logical_cell_id="",
        report_field=validated_report_field(candidate),
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
    required_states = [
        state
        for name, applicability, state in obligations
        if applicability == "required" and name in names
    ]
    states = required_states or [
        state
        for name, applicability, state in obligations
        if applicability == "optional" and name in names
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
    *,
    relative_basis: dict | None = None,
) -> tuple[
    tuple[tuple[str, str, tuple[str, ...], str], ...],
    str | None,
    str | None,
]:
    try:
        with localcontext() as context:
            context.prec = 80
            values = {
                citation.operand_ref: Decimal(citation.value)
                * Decimal("1" if relative_basis else citation.scale)
                for citation in citation_by_index.values()
            }
    except InvalidOperation:
        return (), None, None
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
        if operation in COMMUTATIVE:
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


def _certificate_evidence_ref(citation: VerifiedCitation) -> dict[str, object]:
    """Return the exact physical source identity embedded in a certificate."""

    return {
        "operand_ref": citation.operand_ref,
        "logical_cell_id": citation.logical_cell_id,
        **_citation_payload(citation),
    }


def _build_exact_execution_certificate(
    program: Sequence[tuple[str, str, tuple[str, ...], str]],
    output_node: str | None,
    citations: Sequence[VerifiedCitation],
    *,
    output_unit: str,
) -> tuple[dict[str, object], Decimal]:
    """Replay a verified program with exact rationals outside the HRM."""

    if not program or not output_node or not citations or not output_unit:
        raise ExactExecutionError("Incomplete exact-execution inputs")
    relative_basis = _program_relative_basis(program, output_node, citations)
    operands = {
        citation.operand_ref: ExactOperand(
            value=citation.value,
            unit=relative_basis["unit_symbol"] if relative_basis else citation.unit,
            evidence_ref={**_certificate_evidence_ref(citation),
                          **({"relative_measurement_basis": json.dumps(relative_basis, ensure_ascii=False, sort_keys=True)} if relative_basis else {})},
            operand_id=citation.operand_ref,
        )
        for citation in citations
    }
    # Recompute every node from cited operands; recorded Decimal results are
    # outputs to audit, never inputs trusted by the exact executor.
    exact = execute_program_records(
        program,
        operands,
        output_node=output_node,
        output_unit=output_unit,
        decimal_precision=34,
    )
    certificate = exact.certificate.to_dict()
    if not verify_dag_proof_certificate(certificate):
        raise ExactExecutionError("Exact execution certificate failed replay")
    return certificate, exact.decimal_value


def _program_relative_basis(program, output_node, citations) -> dict | None:
    """Only one direct dimensionless operation can consume an opaque row unit."""
    if len(program) != 1:
        return None
    node_id, operation, inputs, _ = program[0]
    if node_id != output_node or len(inputs) != 2 or len(set(inputs)) != 2 or set(inputs) != {c.operand_ref for c in citations}:
        return None
    return same_row_measurement_basis(operation, citations)


def export_verified_answer_lean4(
    envelope: "VerifiedAnswerEnvelope",
    *,
    theorem_name: str = "finhrm_exact_answer",
) -> str:
    """Export an accepted exact certificate; Lean remains an optional audit."""

    if envelope.status is not AnswerStatus.ANSWER:
        raise ValueError("Only an accepted answer can be exported to Lean")
    if validate_verified_answer_envelope(envelope):
        raise ValueError("Cannot export an invalid answer envelope to Lean")
    certificate = envelope.proof.execution_certificate
    if certificate is None or not verify_dag_proof_certificate(certificate):
        raise ValueError("The answer has no valid exact execution certificate")
    return export_lean4_dag(certificate, theorem_name=theorem_name)


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
        program=(),
        program_output=None,
        proof=replace(
            envelope.proof,
            arithmetic_verifier="",
            execution_certificate=None,
            action_decision=(finish_answer_decision(envelope.proof.action_decision, status.value)
                             if envelope.proof.action_decision else None),
        ),
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


def _build_verified_answer_envelope(
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
    runtime_source_action = str(getattr(answer, "source_action", "ABSTAIN"))
    has_explicit_resolution = hasattr(answer, "evidence_resolution")
    evidence_resolution = str(
        getattr(answer, "evidence_resolution", "UNKNOWN")
    )
    evidence_resolution_reasons = tuple(
        str(item)
        for item in getattr(answer, "evidence_resolution_reasons", ())
    )
    required_states = [
        state for _, applicability, state in obligations if applicability == "required"
    ]
    missing_periods = tuple(
        str(period)
        for period in getattr(answer, "missing_periods", ())
        if str(period)
    )
    explicit_unknown_resolution = bool(
        has_explicit_resolution
        and evidence_resolution == "UNKNOWN"
        and evidence_resolution_reasons
    )
    if missing_periods:
        status = AnswerStatus.ABSTAIN
    elif has_explicit_resolution and evidence_resolution == "CONFLICT":
        status = AnswerStatus.REJECT
    elif verification_state == "INVALID" or "INVALID" in required_states:
        status = AnswerStatus.REJECT
    elif (
        has_explicit_resolution
        and evidence_resolution in {"MISSING", "AMBIGUOUS"}
    ) or explicit_unknown_resolution:
        status = AnswerStatus.ABSTAIN
    elif (
        runtime_source_action != "ANSWER"
        or not required_states
        or any(state != "VALID" for state in required_states)
        or bool(getattr(answer, "abstained", True))
    ):
        status = AnswerStatus.ABSTAIN
    else:
        status = AnswerStatus.ANSWER
    source_action = status.value

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
    relative_basis = same_row_measurement_basis(operation, citations)
    dag = getattr(answer, "program_dag", None) or getattr(answer.prediction, "program_dag", None)
    if dag is not None and (len(dag.nodes) != 1 or dag.nodes[0].operation != operation):
        relative_basis = None
    if _required_state(obligations, {"unit_compatibility"}) != "VALID":
        relative_basis = None
    program, output_node, replay_value = _program_records(
        answer, citation_by_index, relative_basis=relative_basis
    )
    if relative_basis and _program_relative_basis(program, output_node, citations) != relative_basis:
        relative_basis = None
        program, output_node, replay_value = _program_records(answer, citation_by_index)

    plan = compile_question_plan(str(getattr(answer, "question", "")))
    metric_state = _required_state(obligations, {"query_metric", "metric_match"})
    period_state = _required_state(obligations, {"period_coverage"})
    entity_state = _required_state(
        obligations, {"reporting_entity", "named_entity", "entity_match"}
    )
    unit_state = _required_state(
        obligations,
        {"unit_match", "unit_compatibility", "unit_scale_match"},
    )
    metric = plan.metric_text.strip() if metric_state == "VALID" else None
    periods = tuple(sorted({item.period for item in citations if item.period}))
    period = " / ".join(periods) if periods and period_state == "VALID" else None
    if (plan.requested_report_year and _required_state(obligations, {"report_year_match"}) == "VALID"
            and citations and all(report_field_matches(c) and c.report_field["report_year"] == plan.requested_report_year for c in citations)):
        period = "báo cáo " + plan.requested_report_year
    entities = {item.entity_scope for item in citations if item.entity_scope}
    entity_scope = (
        next(iter(entities))
        if len(entities) == 1 and entity_state == "VALID"
        else None
    )
    answer_unit = str(getattr(answer, "unit", "")).strip()
    display_scale = _candidate_scale(answer_unit)
    base_unit = "VND" if currency_unit_scale(answer_unit) is not None else answer_unit
    required_complete = bool(required_states) and all(
        state == "VALID" for state in required_states
    )
    unit_scale_valid = (
        _required_state(obligations, {"unit_scale_match"}) == "VALID"
    )
    explicit_citation_scales = bool(citations) and all(
        citation.scale in _ALLOWED_SCALES for citation in citations
    )
    requested_display = {
        "vnd": ("VND", "1"),
        "thousand_vnd": ("nghìn đồng", "1000"),
        "million_vnd": ("triệu đồng", "1000000"),
        "billion_vnd": ("tỷ đồng", "1000000000"),
    }.get(plan.requested_scale)
    use_verified_requested_scale = bool(
        requested_display
        and evidence_resolution == "UNIQUE"
        and required_complete
        and unit_scale_valid
        and explicit_citation_scales
    )
    if use_verified_requested_scale:
        assert requested_display is not None
        answer_unit, display_scale = requested_display
        base_unit = "VND"
    if unit_state != "VALID":
        base_unit = ""

    canonical_value = replay_value if status is AnswerStatus.ANSWER else None
    display = None
    if status is AnswerStatus.ANSWER and canonical_value is not None and answer.value is not None:
        display_value = _decimal_text(Decimal(str(answer.value)))
        if use_verified_requested_scale:
            with localcontext() as context:
                context.prec = 80
                display_value = _decimal_text(
                    Decimal(canonical_value) / Decimal(display_scale)
                )
        display = (
            display_value,
            answer_unit,
            display_scale,
            _LOCALE,
        )
    execution_certificate: dict[str, object] | None = None
    exact_execution_error = False
    if status is AnswerStatus.ANSWER and display is not None:
        try:
            execution_certificate, exact_display_value = (
                _build_exact_execution_certificate(
                    program,
                    output_node,
                    citations,
                    output_unit=answer_unit,
                )
            )
            if exact_display_value != Decimal(display[0]):
                raise ExactExecutionError(
                    "Exact executor disagrees with the displayed result"
                )
        except (ExactExecutionError, InvalidOperation, ZeroDivisionError):
            exact_execution_error = True
    # SELECT may have no requested unit obligation. Source-declared currency or percent
    # unit is still publishable after the exact certificate checked that same
    # single citation and output unit. Do not turn missing/invalid unit proof
    # into VALID, or infer a unit for multi-operand arithmetic here.
    source_select_unit_verified = bool(
        operation == "SELECT" and len(citations) == 1
        and required_complete and evidence_resolution == "UNIQUE"
        and unit_state is None
        and all(applicability == "not_applicable" for name, applicability, _ in obligations
                if name in {"unit_match", "unit_compatibility", "unit_scale_match"})
        and (currency_unit_scale(citations[0].unit) is not None or citations[0].unit == "%")
        and citations[0].unit == answer_unit
        and execution_certificate is not None and not exact_execution_error
    )
    if source_select_unit_verified:
        base_unit = "VND" if currency_unit_scale(citations[0].unit) is not None else citations[0].unit
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
            acceptance_basis=tuple(
                dict.fromkeys(
                    (
                        *tuple(getattr(answer, "acceptance_basis", ())),
                        *(c.unit_assumption for c in citations if c.unit_assumption),
                        *(("exact_fraction_certificate",) if execution_certificate else ()),
                        *(("source_currency_select_unit" if base_unit == "VND" else "source_percent_select_unit",) if source_select_unit_verified else ()),
                        *(("same_physical_row_unit_cancellation",) if relative_basis else ()),
                    )
                )
            ),
            runtime_source_action=runtime_source_action,
            evidence_resolution=evidence_resolution,
            evidence_resolution_reasons=evidence_resolution_reasons,
            arithmetic_verifier="exact-fraction-dag-v1",
            execution_certificate=execution_certificate,
            supporting_disclosures=tuple(asdict(mapping)
                for assessment in getattr(answer, "operand_mapping_assessments", ())
                if assessment.equivalent_disclosures
                for mapping in assessment.valid_physical_mappings),
        ),
        summary_coverage=None,
        warnings=tuple(
            dict.fromkeys(
                (
                    *(str(item) for item in getattr(answer, "verification_reasons", ())),
                    *(item for item in getattr(answer, "warnings", ()) if item == "public_default_actual"),
                    *(f"missing_requested_period:{period}" for period in missing_periods),
                    *(f"unverified_requested_period:{year}" for year in plan.requested_years
                      if period_state in {"UNKNOWN", "INVALID"}),
                    *(("exact_execution_failed",) if exact_execution_error else ()),
                )
            )
        ),
    )
    if status is not AnswerStatus.ANSWER:
        return _nonanswer(envelope, status)
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
            "missing_exact_execution_certificate",
            "invalid_exact_execution_certificate",
            "exact_execution_result_mismatch",
            "wrong_mode_for_program",
        }
        fail_status = (
            AnswerStatus.REJECT
            if any(error in contradiction_markers for error in errors)
            else AnswerStatus.ABSTAIN
        )
        envelope = replace(envelope, warnings=tuple(dict.fromkeys(
            (*envelope.warnings, *("envelope_validation:" + error for error in errors))
        )))
        return _nonanswer(envelope, fail_status)
    return envelope


def _validate_program(envelope: VerifiedAnswerEnvelope) -> tuple[str, ...]:
    errors: list[str] = []
    citations = {item.operand_ref: item for item in envelope.citations}
    if len(citations) != len(envelope.citations):
        errors.append("duplicate_operand_citation")
        return tuple(errors)
    values: dict[str, Decimal] = {}
    relative_basis = _program_relative_basis(envelope.program, envelope.program_output, envelope.citations)
    try:
        with localcontext() as context:
            context.prec = 80
            values = {
                ref: Decimal(item.value) * Decimal("1" if relative_basis else item.scale)
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
            minimum, maximum = dag_operation_arity_bounds(operation)
        except ProgramDataError:
            errors.append("invalid_program_node")
            continue
        if not minimum <= len(inputs) <= maximum:
            errors.append("invalid_program_arity")
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
    from .unit_defaults import valid_assumption
    if any(not valid_assumption(c) for c in envelope.citations):
        errors.append("invalid_unit_assumption")
    if envelope.schema_version != _SCHEMA_VERSION:
        errors.append("wrong_schema_version")
    if envelope.locale != _LOCALE:
        errors.append("unsupported_locale")
    required = [
        state
        for _, applicability, state in envelope.proof.obligations
        if applicability == "required"
    ]
    if (
        envelope.proof.verification_state == "INVALID"
        and envelope.status is not AnswerStatus.REJECT
        and envelope.proof.evidence_resolution != "AMBIGUOUS"
    ):
        errors.append("invalid_proof_not_rejected")
    if envelope.status is AnswerStatus.ANSWER:
        decision = envelope.proof.action_decision
        if decision is not None and (
            not decision.policy_accepted or decision.final_action != "ANSWER"
            or any(state != "VALID" for state in (decision.execution_status,
                   decision.certificate_status, decision.presentation_status))
        ):
            errors.append("action_decision_not_accepted")
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

    from .column_context import column_kind
    column_lookup = (
        envelope.answer_mode is AnswerMode.FACT and len(envelope.citations) == 1
        and column_kind(envelope.citations[0].column_label or "") in {"comparison", "comparison_ratio"}
        and _required_state(envelope.proof.obligations, {"column_role_match"}) == "VALID"
        and _required_state(envelope.proof.obligations, {"period_coverage"}) is None
    )
    undated_field_lookup = (
        envelope.answer_mode is AnswerMode.FACT and len(envelope.citations) == 1
        and report_field_matches(envelope.citations[0])
        and _required_state(envelope.proof.obligations, {"period_coverage", "report_year_match"}) is None
        and _required_state(envelope.proof.obligations, {"source_provenance_complete"}) == "VALID"
        and _required_state(envelope.proof.obligations, {"logical_cell_unique"}) == "VALID"
    )
    if envelope.claim is None:
        errors.append("missing_claim")
    else:
        for field in (
            envelope.claim.metric,
            *((envelope.claim.period,) if not (column_lookup or undated_field_lookup) else ()),
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
            with localcontext() as context:
                context.prec = 80
                if Decimal(display_value) * Decimal(display_scale) != Decimal(
                    envelope.canonical_value
                ):
                    errors.append("display_value_mismatch")
        except InvalidOperation:
            errors.append("invalid_display_decimal")
    if not envelope.citations:
        errors.append("missing_citation")
    relative_basis = _program_relative_basis(envelope.program, envelope.program_output, envelope.citations)
    if relative_basis and _required_state(envelope.proof.obligations, {"unit_compatibility"}) != "VALID":
        errors.append("relative_unit_compatibility_not_verified")
    for citation in envelope.citations:
        if (
            not citation.source_name
            or citation.source_span is None
            or len(citation.source_span) != 2
            or not citation.logical_cell_id
            or not citation.extraction_method
            or not citation.value
            or (not citation.unit and not relative_basis)
            or (citation.scale not in _ALLOWED_SCALES and not relative_basis)
            or (not citation.period and not report_field_matches(citation) and not column_lookup)
            or not citation.entity_scope
        ):
            errors.append("incomplete_citation")
        if is_versioned_coordinate(citation) and physical_coordinate_key(citation) is None:
            errors.append("unresolved_citation_coordinates")
        if citation.logical_cell_id != _citation_id(citation):
            errors.append("citation_identity_mismatch")
    if envelope.proof.supporting_disclosures:
        if "equivalent_disclosures_same_fact" not in envelope.proof.evidence_resolution_reasons or envelope.proof.evidence_resolution != "UNIQUE":
            errors.append("unverified_supporting_disclosures")
        for location in envelope.proof.supporting_disclosures:
            if location.get("source_name") not in {c.source_name for c in envelope.citations} or physical_coordinate_key(SimpleNamespace(**location)) is None:
                errors.append("invalid_supporting_disclosure_coordinates")
    errors.extend(_validate_execution_certificate(envelope))
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
    absolute = _decimal_text(decimal.copy_abs())
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
        elif is_sum_operation(operation):
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


def _rendered_number(envelope: VerifiedAnswerEnvelope) -> tuple[str, bool]:
    """Round long computed percentages for reading; keep exact API/certificate."""
    assert envelope.display is not None
    value = Decimal(envelope.display[0])
    relative_division = bool(envelope.program and envelope.program[-1][1] == "DIVIDE" and envelope.display[1] in {"", "ratio", "1"})
    if envelope.answer_mode is AnswerMode.COMPUTATION and (envelope.display[1] == "%" or relative_division):
        with localcontext() as context:
            context.prec = max(80, len(value.as_tuple().digits) + 4, value.adjusted() + 4)
            rounded = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if rounded != value:
            return _format_vi(_decimal_text(rounded)), True
    return _format_vi(envelope.display[0]), False


def _substitution(envelope: VerifiedAnswerEnvelope) -> str:
    assignments = "; ".join(
        f"{citation.operand_ref} = {_format_vi(citation.value)} {citation.unit}".strip()
        for citation in sorted(envelope.citations, key=lambda item: item.operand_ref)
    )
    assert envelope.display is not None
    number, rounded = _rendered_number(envelope)
    result = f"{number} {envelope.display[1]}".strip()
    relation = "≈" if rounded else "="
    return f"{assignments}; {_formula(envelope)} {relation} {result}"


def _operand_reason(citation: VerifiedCitation) -> str:
    identity: list[str] = []
    if citation.row_label:
        identity.append(f"dòng “{citation.row_label}”")
    if citation.column_label:
        identity.append(f"cột “{citation.column_label}”")
    if citation.period:
        identity.append(f"kỳ {citation.period}")
    location = ", ".join(identity) or "ô dữ liệu đã xác minh"
    return (
        f"Vì chọn {citation.operand_ref}: giá trị {_format_vi(citation.value)} "
        f"{citation.unit} thuộc {location}."
    )


def _summary_source(row: VerifiedAnswerEnvelope) -> str:
    return "; ".join(
        _render_citation(citation)
        for citation in sorted(row.citations, key=lambda item: item.operand_ref)
    )


def render_verified_answer(envelope: VerifiedAnswerEnvelope) -> str:
    note = "Đang hiểu câu hỏi là số thực hiện. " if "public_default_actual" in envelope.warnings else ""
    rendered = note + _render_verified_answer(envelope)
    if envelope.status is AnswerStatus.ANSWER and any(c.unit_assumption for c in envelope.citations):
        from .unit_defaults import DEFAULT_VND_NOTE
        rendered += "\n" + DEFAULT_VND_NOTE
    return rendered


def _compact_sources(envelope: VerifiedAnswerEnvelope) -> str:
    locations = {}
    for citation in envelope.citations:
        locations.setdefault(citation.source_name, set()).add(citation.page)
    for location in envelope.proof.supporting_disclosures:
        locations.setdefault(location["source_name"], set()).add(location["page"])
    parts = []
    for source, pages in sorted(locations.items()):
        name = re.sub(r"--[0-9a-f]{16}(?=\.)", "", source)
        known = sorted(p for p in pages if p is not None)
        if known:
            parts.append(name + ", trang " + ", ".join(str(p) for p in known))
        else:
            parts.append(_render_citation(next(c for c in envelope.citations if c.source_name == source)))
    return "Nguồn: " + "; ".join(parts) + "."


def _render_verified_answer(envelope: VerifiedAnswerEnvelope) -> str:
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

    if (
        envelope.status in {AnswerStatus.REJECT, AnswerStatus.ABSTAIN}
        and "question_column_role_unspecified" in envelope.proof.evidence_resolution_reasons
    ):
        return (
            "Cần làm rõ: báo cáo có cả số kế hoạch và thực hiện cho chỉ tiêu/kỳ này. "
            "Bạn muốn số thực hiện hay số kế hoạch?"
        )
    if envelope.status in {AnswerStatus.ABSTAIN, AnswerStatus.REJECT}:
        required = {name: state for name, applicability, state in envelope.proof.obligations if applicability == "required"}
        periods = tuple(item.split(":", 1)[1] for item in envelope.warnings if item.startswith("missing_requested_period:"))
        unverified = tuple(item.split(":", 1)[1] for item in envelope.warnings
                           if item.startswith("unverified_requested_period:"))
        period_note = (
            " Chưa xác nhận được số liệu cho năm/kỳ được hỏi: " + ", ".join(dict.fromkeys(unverified)) + "."
            if unverified and required.get("period_coverage") in {"UNKNOWN", "INVALID"} else ""
        )
        # A missing-period diagnosis must not conceal a known scope mismatch.
        # These messages describe the selected evidence, never invent a source claim.
        if required.get("query_entity_context") in {"UNKNOWN", "INVALID"}:
            return "Chưa xác nhận được số liệu thuộc đúng doanh nghiệp được hỏi; hệ thống chưa công bố đáp án." + period_note
        if required.get("query_accounting_scope") in {"UNKNOWN", "INVALID"}:
            return "Chưa xác nhận được bằng chứng thuộc đúng phạm vi báo cáo được hỏi; hệ thống chưa công bố số liệu." + period_note
        if required.get("named_entity") == "INVALID" or required.get("operand_entity_match") == "INVALID":
            return "Hệ thống chưa chọn được số liệu của đúng doanh nghiệp hoặc phạm vi được hỏi."
        if required.get("report_year_match") in {"UNKNOWN", "INVALID"}:
            return "Chưa xác nhận được tài liệu đang dùng thuộc đúng năm báo cáo được hỏi."
        if (periods and envelope.proof.evidence_resolution == "MISSING"
                and required.get("period_coverage") in {"UNKNOWN", "INVALID"}):
            return "Chưa tìm thấy thông tin phù hợp với chỉ tiêu và năm/kỳ được hỏi trong phần tài liệu đã truy xuất: " + ", ".join(dict.fromkeys(periods)) + "."
        if required.get("query_metric") in {"UNKNOWN", "INVALID"}:
            return "Chưa xác nhận được bằng chứng phù hợp với chỉ tiêu được hỏi; hệ thống chưa công bố đáp án."
        if period_note:
            return period_note.strip()
        if required.get("unit_scale_match") in {"UNKNOWN", "INVALID"}:
            return "Chưa xác nhận được đơn vị và thang đo của số liệu từ nguồn; chưa thể quy đổi để trả lời."
    if envelope.status in {AnswerStatus.ABSTAIN, AnswerStatus.REJECT} and "source_number_format_unresolved" in envelope.proof.evidence_resolution_reasons:
        return "Chưa phân biệt được dấu thập phân và dấu hàng nghìn của số liệu nguồn; hệ thống chưa công bố đáp án."
    if envelope.status in {AnswerStatus.ABSTAIN, AnswerStatus.REJECT}:
        reasons = set(envelope.proof.evidence_resolution_reasons)
        if 'comparison_input_unverified' in reasons:
            return 'Chưa xác minh đủ hai số liệu đầu vào để tính chênh lệch; các phần đã xác minh được giữ lại.'
        if 'comparison_inputs_not_comparable' in reasons:
            return 'Chưa xác nhận hai số liệu có cùng chỉ tiêu, đơn vị và phạm vi để so sánh; chưa tính chênh lệch.'
    decision = envelope.proof.action_decision
    if (envelope.status is AnswerStatus.ABSTAIN and decision is not None
            and decision.proof_eligible and not decision.policy_accepted):
        if decision.calibration_status == "CALIBRATED":
            return "Đã tìm được bằng chứng phù hợp, nhưng kết quả chưa đạt ngưỡng tin cậy để công bố đáp án."
        return "Đã tìm được bằng chứng phù hợp, nhưng hệ thống chưa hoàn tất hiệu chỉnh độ tin cậy nên chưa công bố đáp án."
    missing_periods = tuple(item.split(":", 1)[1] for item in envelope.warnings
                            if item.startswith("missing_requested_period:"))
    if envelope.status is AnswerStatus.ABSTAIN and missing_periods:
        return ("Chưa tìm thấy thông tin cho chỉ tiêu và năm/kỳ được hỏi trong tài liệu đang dùng: "
                + ", ".join(dict.fromkeys(missing_periods)) + ".")
    mismatch_names = {"column_role_match", "query_section", "query_metric", "period_coverage", "query_accounting_scope", "operand_entity_match"}
    if (envelope.status in {AnswerStatus.REJECT, AnswerStatus.ABSTAIN}
            and any(name in mismatch_names and state == "INVALID"
                    for name, applicability, state in envelope.proof.obligations
                    if applicability == "required")):
        return "Hệ thống chưa chọn được bằng chứng phù hợp với chỉ tiêu, kỳ hoặc phạm vi được hỏi; chưa thể trả lời."
    if envelope.status is AnswerStatus.REJECT:
        return "Bằng chứng được chọn chưa vượt qua kiểm chứng; chưa thể chấp nhận câu trả lời."
    if envelope.status is AnswerStatus.ABSTAIN:
        if envelope.proof.evidence_resolution == "AMBIGUOUS":
            reasons = set(envelope.proof.evidence_resolution_reasons)
            detail = []
            if reasons & {"disclosure_metric_or_identity_unresolved", "related_metric_identity_unresolved"}:
                detail.append("chưa đủ căn cứ xác nhận các nhãn chỉ tiêu/công bố có cùng ý nghĩa")
            if "disclosure_number_format_unresolved" in reasons:
                detail.append("chưa phân biệt được dấu thập phân và dấu hàng nghìn của số liệu nguồn")
            if "disclosure_scope_unresolved" in reasons:
                detail.append("chưa xác nhận các công bố thuộc cùng phạm vi báo cáo")
            if reasons & {"disclosure_precision_unresolved", "disclosure_rounding_convention_unresolved"}:
                detail.append("chưa xác định được độ chính xác hoặc quy ước làm tròn giữa các công bố")
            if "disclosure_values_incompatible" in reasons:
                detail.append("các số cùng chỉ tiêu và phạm vi chưa thống nhất theo độ chính xác nguồn")
            if detail:
                return "Chưa công bố đáp án vì " + "; ".join(detail) + "."
            return (
                "Chưa xác định được công bố nào trong tài liệu phù hợp đầy đủ với câu hỏi; "
                "hệ thống chưa công bố số liệu."
            )
        missing_periods = tuple(
            item.split(":", 1)[1]
            for item in envelope.warnings
            if item.startswith("missing_requested_period:") and ":" in item
        )
        if missing_periods:
            return (
                "Báo cáo không có thông tin cho năm/kỳ được yêu cầu: "
                + ", ".join(dict.fromkeys(missing_periods))
                + "; hệ thống không thực hiện phép tính."
            )
        return "Không đủ bằng chứng đã xác minh để trả lời."
    assert envelope.claim is not None and envelope.display is not None
    number, rounded = _rendered_number(envelope)
    unit = "đồng" if envelope.display[1] == "VND" else envelope.display[1]
    value = f"{'khoảng ' if rounded else ''}{number} {unit}".strip()
    metric = envelope.claim.metric or ""
    labels = {c.row_label for c in envelope.citations if c.row_label}
    if len(labels) == 1 and _required_state(envelope.proof.obligations, {"query_metric", "metric_match"}) == "VALID":
        metric = next(iter(labels))
    metric = re.sub(r"^\d+\s*[-.)]\s*", "", metric)
    if envelope.citations[0].report_field:
        metric = metric.split("/", 1)[0].strip()
        owner = envelope.citations[0].report_field.get("entity_owner")
        if owner:
            metric += " của " + owner["name"]
    metric = metric[:1].upper() + metric[1:]
    period = envelope.claim.period or ""
    period_text = "năm " + period if re.fullmatch(r"(?:19|20)\d{2}", period) else period
    introduction = f"{metric} {period_text} là {value}."
    if not period and report_field_matches(envelope.citations[0]):
        introduction = f"Theo thông tin được nêu trong báo cáo, {metric[:1].lower() + metric[1:]} là {value}."
    if period.startswith("báo cáo "):
        introduction = f"Theo báo cáo năm {period.removeprefix('báo cáo ')}, {metric[:1].lower() + metric[1:]} là {value}."
    citations = {c.operand_ref: c for c in envelope.citations}
    if envelope.answer_mode is AnswerMode.FACT:
        c = envelope.citations[0]
        from .evidence_schema import _column_role_from_header
        role = _column_role_from_header(c.column_label or "")
        role_text = {"actual": "thực hiện", "plan": "kế hoạch", "revised_plan": "kế hoạch điều chỉnh", "forecast": "dự báo"}.get(role)
        if role_text and re.fullmatch(r"(?:19|20)\d{2}", period):
            introduction = f"{metric} {role_text} {period_text} là {value}."
        if c.column_label and re.fullmatch(r"tại(?: ngày)? \d{1,2}/\d{1,2}/(?:19|20)\d{2}", c.column_label, re.I):
            introduction = f"{metric} {c.column_label} là {value}."
        from .column_context import column_kind
        if c.column_label and column_kind(c.column_label) in {"comparison", "comparison_ratio"}:
            introduction = f"{metric} tại cột “{c.column_label}” là {value}."
        return introduction + " " + _compact_sources(envelope)
    if len(envelope.program) == 1:
        _, operation, inputs, _ = envelope.program[0]
        if len(inputs) == 2 and all(ref in citations for ref in inputs):
            left, right = (citations[ref] for ref in inputs)
            if operation == "PERCENT_CHANGE" and all(re.fullmatch(r"(?:19|20)\d{2}", c.period) for c in (left, right)):
                amount = Decimal(envelope.display[0])
                if Decimal(right.value) > 0:
                    if amount:
                        change = "tăng" if amount > 0 else "giảm"
                        magnitude = ("khoảng " if rounded else "") + number.removeprefix("-")
                        introduction = f"{metric} năm {left.period} {change} {magnitude}% so với năm {right.period}."
                    else:
                        introduction = f"{metric} năm {left.period} không đổi so với năm {right.period} (0%)."
                else:
                    introduction = f"Tỷ lệ thay đổi {metric[:1].lower()+metric[1:]} năm {left.period} so với năm {right.period}, tính trên giá trị năm gốc âm, là {value}."
            if operation in {"RATIO_PERCENT", "DIVIDE"} and left.period != right.period and all(re.fullmatch(r"(?:19|20)\d{2}", c.period) for c in (left, right)):
                magnitude = ("khoảng " if rounded else "") + number
                comparison = magnitude + ("%" if operation == "RATIO_PERCENT" else " lần")
                introduction = f"{metric} năm {left.period} bằng {comparison} năm {right.period}."
            if operation == "SUBTRACT" and all(re.fullmatch(r"(?:19|20)\d{2}", c.period) for c in (left, right)) and int(left.period) > int(right.period):
                amount = Decimal(envelope.display[0])
                change = "tăng" if amount > 0 else "giảm" if amount < 0 else "không đổi"
                if amount:
                    introduction = f"{metric} năm {left.period} {change} {_format_vi(str(abs(amount)))} {unit} so với năm {right.period}."
                else:
                    introduction = f"{metric} năm {left.period} không đổi so với năm {right.period}."
            if operation == "SUBTRACT" and left.period == right.period:
                from .evidence_schema import _column_role_from_header
                roles = tuple(_column_role_from_header(c.column_label or "") for c in (left, right))
                if set(roles) == {"actual", "plan"}:
                    names = {"actual": "thực hiện", "plan": "kế hoạch"}
                    amount = Decimal(envelope.display[0])
                    if amount:
                        direction = "cao hơn" if amount > 0 else "thấp hơn"
                        gap = ("khoảng " if rounded else "") + number.removeprefix("-")
                        introduction = f"{metric} {names[roles[0]]} năm {left.period} {direction} {names[roles[1]]} {gap} {unit}."
                    else:
                        introduction = f"{metric} {names[roles[0]]} năm {left.period} bằng {names[roles[1]]}; chênh lệch là 0 {unit}."
            if operation == "RATIO_PERCENT":
                from .evidence_schema import _column_role_from_header
                if _column_role_from_header(left.column_label or "") == "actual" and _column_role_from_header(right.column_label or "") == "plan":
                    introduction = f"{metric} thực hiện năm {left.period} đạt {value} kế hoạch năm {right.period}."
    substitutions = {ref: f"{_format_vi(c.value)} {'đồng' if c.unit == 'VND' else c.unit}".strip() for ref, c in citations.items()}
    expression = re.sub(r"\bO\d+\b", lambda m: substitutions[m[0]], _formula(envelope))
    formula = f"Công thức: {expression} {'≈' if rounded else '='} {number} {unit}."
    precision_note = (
        "Kết quả tính từ số liệu trình bày trong bảng; các công bố đối chiếu có độ làm tròn khác nhau."
        if "disclosures_compatible_at_source_precision" in envelope.proof.evidence_resolution_reasons else ""
    )
    if precision_note:
        introduction = re.sub(r" (là|tăng|giảm) (?!khoảng\b)", r" \1 khoảng ", introduction)
    return "\n".join(x for x in (introduction, formula, precision_note, _compact_sources(envelope)) if x)



def _validate_execution_certificate(
    envelope: VerifiedAnswerEnvelope,
) -> tuple[str, ...]:
    """Bind the external arithmetic proof to this exact envelope."""

    verifier = envelope.proof.arithmetic_verifier
    certificate = envelope.proof.execution_certificate
    if not verifier and certificate is None:
        return ()
    if verifier != "exact-fraction-dag-v1":
        return ("unsupported_arithmetic_verifier",)
    if certificate is None:
        return ("missing_exact_execution_certificate",)
    if not verify_dag_proof_certificate(certificate):
        return ("invalid_exact_execution_certificate",)

    # A self-consistent arithmetic proof does not authenticate its evidence.
    # Rebuild from the actual citations, including units and physical sources.
    try:
        if envelope.display is None:
            return ("exact_execution_result_mismatch",)
        rebuilt, _ = _build_exact_execution_certificate(
            envelope.program,
            envelope.program_output,
            envelope.citations,
            output_unit=envelope.display[1],
        )
        if rebuilt != certificate:
            return ("exact_execution_evidence_mismatch",)
    except (ExactExecutionError, InvalidOperation, ZeroDivisionError):
        return ("invalid_exact_execution_certificate",)


    raw_nodes = certificate.get("nodes")
    if not isinstance(raw_nodes, list) or len(raw_nodes) != len(envelope.program):
        return ("exact_execution_program_mismatch",)
    errors: list[str] = []
    for expected, observed in zip(envelope.program, raw_nodes):
        node_id, operation, inputs, _ = expected
        if not isinstance(observed, dict) or not isinstance(observed.get("inputs"), list):
            errors.append("exact_execution_program_mismatch")
            continue
        observed_inputs = observed["inputs"]
        references = tuple(
            str(item.get("reference"))
            for item in observed_inputs
            if isinstance(item, dict)
        )
        if (
            str(observed.get("node_id")) != node_id
            or str(observed.get("operation")) != operation
            or references != inputs
            or len(references) != len(observed_inputs)
        ):
            errors.append("exact_execution_program_mismatch")
    if str(certificate.get("output_node")) != str(envelope.program_output):
        errors.append("exact_execution_program_mismatch")

    raw_refs = certificate.get("leaf_evidence_refs")
    if not isinstance(raw_refs, list):
        errors.append("exact_execution_evidence_mismatch")
    else:
        observed_ids = {
            str(item.get("logical_cell_id"))
            for item in raw_refs
            if isinstance(item, dict)
        }
        if observed_ids != {item.logical_cell_id for item in envelope.citations}:
            errors.append("exact_execution_evidence_mismatch")

    if envelope.display is not None:
        try:
            output_record = next(
                item
                for item in raw_nodes
                if isinstance(item, dict)
                and str(item.get("node_id")) == str(envelope.program_output)
            )
            value = output_record["proof"]["result"]["value"]["decimal"]
            if Decimal(str(value)) != Decimal(envelope.display[0]):
                errors.append("exact_execution_result_mismatch")
        except (KeyError, StopIteration, TypeError, InvalidOperation):
            errors.append("invalid_exact_execution_certificate")
    return tuple(dict.fromkeys(errors))


def build_verified_answer_envelope(answer=None, **kwargs):
    envelope = _build_verified_answer_envelope(answer, **kwargs)
    envelope = replace(envelope, proof=replace(envelope.proof,
        applicability_details=getattr(answer, "applicability_details", ()),
        lookup_recovery=getattr(answer, "lookup_recovery", None)))
    decision = getattr(answer, "action_decision", None)
    if decision is None:
        return envelope
    if envelope.status is AnswerStatus.ANSWER and not decision.policy_accepted:
        envelope = _nonanswer(envelope, AnswerStatus.ABSTAIN)
    final = finish_answer_decision(
        decision, envelope.status.value,
        execution_failed="exact_execution_failed" in envelope.warnings,
        reason_codes=tuple(w for w in envelope.warnings if w.startswith("envelope_validation:")))
    return replace(envelope, proof=replace(envelope.proof, action_decision=final))
