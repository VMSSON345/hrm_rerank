from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from .answer_presentation import (
    AnswerStatus,
    VerifiedAnswerEnvelope,
    render_verified_answer,
    validate_verified_answer_envelope,
)
from .source_action_policy import aggregate_output_actions
from .question_request_planner import (
    ABSTAIN,
    MULTI_OUTPUT,
    QuestionRequestPlan,
    RequestedOutput,
)


REQUEST_ANSWER_SCHEMA_VERSION = "verified-request-answer-v1"


@dataclass(frozen=True)
class RequestedOutputAnswer:
    request: RequestedOutput
    envelope: VerifiedAnswerEnvelope
    evidence_resolution: str
    evidence_resolution_reasons: tuple[str, ...]
    classification: str

    @property
    def output_id(self) -> str:
        return self.request.output_id

    @property
    def proof(self) -> object:
        return self.envelope.proof

    @property
    def citations(self) -> tuple[object, ...]:
        return tuple(self.envelope.citations)


@dataclass(frozen=True)
class VerifiedRequestAnswer:
    schema_version: str
    request_plan: QuestionRequestPlan
    status: AnswerStatus
    coverage_state: str
    outputs: tuple[RequestedOutputAnswer, ...]
    verified_output_ids: tuple[str, ...]
    missing_output_ids: tuple[str, ...]
    ambiguous_output_ids: tuple[str, ...]
    conflicting_output_ids: tuple[str, ...]
    unknown_output_ids: tuple[str, ...]
    abstention_reason: str = ""

    @property
    def final_action(self) -> str:
        return aggregate_output_actions(tuple(
            "ANSWER" if item.classification == "VERIFIED" else
            "REJECT" if item.classification == "CONFLICT" else "ABSTAIN"
            for item in self.outputs))

    def render_verified(self) -> str:
        return render_verified_request_answer(self)


def _classify_output(
    answer: object,
    envelope: VerifiedAnswerEnvelope,
) -> str:
    errors = validate_verified_answer_envelope(envelope)
    resolution = str(getattr(answer, "evidence_resolution", "UNKNOWN"))
    if envelope.status is AnswerStatus.REJECT or resolution == "CONFLICT":
        return "CONFLICT"
    if resolution == "MISSING":
        return "MISSING"
    if resolution == "AMBIGUOUS":
        return "AMBIGUOUS"
    if (
        resolution == "UNIQUE"
        and not errors
        and envelope.status is AnswerStatus.ANSWER
    ):
        return "VERIFIED"
    return "UNKNOWN"


def _synchronize_output_envelope(
    answer: object,
    envelope: VerifiedAnswerEnvelope,
    resolution: str,
    reasons: tuple[str, ...],
) -> VerifiedAnswerEnvelope:
    """Attach the scalar runtime evidence snapshot without changing its claim."""

    runtime_action = str(getattr(answer, "source_action", "")).strip() or (
        envelope.proof.runtime_source_action or envelope.proof.source_action
    )
    proof = replace(
        envelope.proof,
        runtime_source_action=runtime_action,
        evidence_resolution=resolution,
        evidence_resolution_reasons=reasons,
    )
    return replace(envelope, proof=proof)


def build_verified_request_answer(
    request_plan: QuestionRequestPlan,
    scalar_answers: Sequence[object] = (),
) -> VerifiedRequestAnswer:
    if request_plan.mode == ABSTAIN:
        if scalar_answers:
            raise ValueError("a fail-closed request plan cannot have scalar answers")
        return VerifiedRequestAnswer(
            schema_version=REQUEST_ANSWER_SCHEMA_VERSION,
            request_plan=request_plan,
            status=AnswerStatus.ABSTAIN,
            coverage_state="UNSPLITTABLE",
            outputs=(),
            verified_output_ids=(),
            missing_output_ids=(),
            ambiguous_output_ids=(),
            conflicting_output_ids=(),
            unknown_output_ids=(),
            abstention_reason=request_plan.abstention_reason,
        )
    if len(scalar_answers) != len(request_plan.outputs):
        raise ValueError("scalar answer count does not match requested outputs")

    outputs: list[RequestedOutputAnswer] = []
    for requested, answer in zip(request_plan.outputs, scalar_answers):
        resolution = str(getattr(answer, "evidence_resolution", "UNKNOWN"))
        reasons = tuple(
            str(item)
            for item in getattr(answer, "evidence_resolution_reasons", ())
        )
        envelope = _synchronize_output_envelope(
            answer,
            answer.verified_envelope(),
            resolution,
            reasons,
        )
        classification = _classify_output(answer, envelope)
        outputs.append(
            RequestedOutputAnswer(
                request=requested,
                envelope=envelope,
                evidence_resolution=resolution,
                evidence_resolution_reasons=reasons,
                classification=classification,
            )
        )

    def ids(classification: str) -> tuple[str, ...]:
        return tuple(
            item.request.output_id
            for item in outputs
            if item.classification == classification
        )

    verified = ids("VERIFIED")
    missing = ids("MISSING")
    ambiguous = ids("AMBIGUOUS")
    conflicting = ids("CONFLICT")
    unknown = ids("UNKNOWN")
    if conflicting:
        status = AnswerStatus.REJECT
        coverage_state = "CONFLICT"
    elif len(verified) == len(outputs):
        status = AnswerStatus.ANSWER
        coverage_state = "COMPLETE"
    else:
        status = AnswerStatus.ABSTAIN
        coverage_state = "PARTIAL" if verified else "UNVERIFIED"
    return VerifiedRequestAnswer(
        schema_version=REQUEST_ANSWER_SCHEMA_VERSION,
        request_plan=request_plan,
        status=status,
        coverage_state=coverage_state,
        outputs=tuple(outputs),
        verified_output_ids=verified,
        missing_output_ids=missing,
        ambiguous_output_ids=ambiguous,
        conflicting_output_ids=conflicting,
        unknown_output_ids=unknown,
    )


def validate_verified_request_answer(
    answer: VerifiedRequestAnswer,
) -> tuple[str, ...]:
    from .question_request_planner import validate_question_request_plan
    errors: list[str] = list(validate_question_request_plan(answer.request_plan))
    if answer.schema_version != REQUEST_ANSWER_SCHEMA_VERSION:
        errors.append("wrong_schema_version")
    if answer.request_plan.mode == ABSTAIN:
        if answer.outputs or answer.status is not AnswerStatus.ABSTAIN:
            errors.append("invalid_unsplittable_response")
        if not answer.abstention_reason:
            errors.append("missing_request_abstention_reason")
        return tuple(errors)
    if len(answer.outputs) != len(answer.request_plan.outputs):
        errors.append("output_count_mismatch")
    output_ids = tuple(item.output_id for item in answer.outputs)
    if len(output_ids) != len(set(output_ids)):
        errors.append("duplicate_output_id")
    for expected, observed in zip(answer.request_plan.outputs, answer.outputs):
        if observed.request != expected:
            errors.append("output_order_or_identity_mismatch")
        envelope_errors = validate_verified_answer_envelope(observed.envelope)
        if envelope_errors:
            errors.append("invalid_subanswer_envelope")
        if (
            observed.proof.evidence_resolution != observed.evidence_resolution
            or observed.proof.evidence_resolution_reasons
            != observed.evidence_resolution_reasons
        ):
            errors.append("subanswer_proof_snapshot_mismatch")
        if observed.classification != _classify_output(
            observed,
            observed.envelope,
        ):
            errors.append("subanswer_classification_mismatch")
        if observed.classification == "VERIFIED":
            if observed.envelope.status is not AnswerStatus.ANSWER:
                errors.append("nonanswer_promoted")
    classified = {
        "VERIFIED": answer.verified_output_ids,
        "MISSING": answer.missing_output_ids,
        "AMBIGUOUS": answer.ambiguous_output_ids,
        "CONFLICT": answer.conflicting_output_ids,
        "UNKNOWN": answer.unknown_output_ids,
    }
    observed_ids = tuple(item.request.output_id for item in answer.outputs)
    flattened = tuple(
        output_id for values in classified.values() for output_id in values
    )
    if len(flattened) != len(set(flattened)) or set(flattened) != set(observed_ids):
        errors.append("classification_partition_mismatch")
    if answer.coverage_state == "COMPLETE":
        if answer.status is not AnswerStatus.ANSWER:
            errors.append("complete_not_answer")
        if len(answer.verified_output_ids) != len(answer.outputs):
            errors.append("false_complete")
    elif answer.coverage_state == "CONFLICT":
        if (
            answer.status is not AnswerStatus.REJECT
            or not answer.conflicting_output_ids
        ):
            errors.append("conflict_not_rejected")
    elif answer.coverage_state in {"PARTIAL", "VERIFIED_PARTIAL", "UNVERIFIED"}:
        if answer.status is not AnswerStatus.ABSTAIN:
            errors.append("incomplete_not_abstained")
        if answer.coverage_state in {"PARTIAL", "VERIFIED_PARTIAL"} and not (
            answer.verified_output_ids
            and len(answer.verified_output_ids) < len(answer.outputs)
        ):
            errors.append("false_partial")
    else:
        errors.append("invalid_coverage_state")
    return tuple(dict.fromkeys(errors))


def _nonanswer_text(item: RequestedOutputAnswer) -> str:
    return render_verified_answer(item.envelope)


def _intent_note(item: RequestedOutputAnswer) -> str:
    return ("Đang hiểu câu hỏi là số thực hiện. "
            if "column_role:default_actual" in item.request.assumptions else "")

def render_verified_request_answer(answer: VerifiedRequestAnswer) -> str:
    errors = validate_verified_request_answer(answer)
    if errors:
        raise ValueError("invalid verified request answer: " + ", ".join(errors))
    if answer.request_plan.mode == ABSTAIN:
        return (
            "Không thể tách câu hỏi thành các đầu ra độc lập đã được nêu rõ; "
            "hệ thống không suy diễn hoặc bỏ sót một phần yêu cầu. Lý do: "
            f"{answer.abstention_reason}."
        )
    from .comparison_overview import STRATEGY, render_comparison
    if answer.request_plan.split_strategy == STRATEGY:
        return render_comparison(answer)
    if answer.request_plan.mode != MULTI_OUTPUT:
        return _intent_note(answer.outputs[0]) + render_verified_answer(answer.outputs[0].envelope)

    lines = [
        (
            "Kết quả nhiều phần đã xác minh đầy đủ."
            if answer.status is AnswerStatus.ANSWER
            else (
                "Một số mục chưa vượt qua kiểm chứng; hệ thống chưa chấp nhận toàn bộ kết quả."
                if answer.status is AnswerStatus.REJECT
                else (
                    "PARTIAL - kết quả nhiều phần đã xác minh một phần: "
                    f"{len(answer.verified_output_ids)}/{len(answer.outputs)} mục; "
                    "trạng thái tổng thể là ABSTAIN."
                )
            )
        )
    ]
    for item in answer.outputs:
        lines.extend(
            (
                "",
                (
                    f"## {item.output_id} / Mục {item.request.ordinal}: "
                    f"{item.request.source_text}"
                ),
                f"Trạng thái mục: {item.classification}.",
                (
                    _intent_note(item) + (render_verified_answer(item.envelope)
                    if item.classification == "VERIFIED"
                    else _nonanswer_text(item))
                ),
            )
        )
    return "\n".join(lines)


__all__ = (
    "REQUEST_ANSWER_SCHEMA_VERSION",
    "RequestedOutputAnswer",
    "VerifiedRequestAnswer",
    "build_verified_request_answer",
    "render_verified_request_answer",
    "validate_verified_request_answer",
)
