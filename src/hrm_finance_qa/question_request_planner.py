from __future__ import annotations

from dataclasses import dataclass
import re

from .ontology import normalize_text
from .question_planner import QuestionPlan, compile_question_plan, resolve_public_question, public_question_assumptions


QUESTION_REQUEST_SCHEMA_VERSION = "question-request-plan-v1"
SINGLE_OUTPUT = "SINGLE_OUTPUT"
MULTI_OUTPUT = "MULTI_OUTPUT"
ABSTAIN = "ABSTAIN"
VERIFIED_PARTIAL_FAIL_CLOSED = "VERIFIED_PARTIAL_FAIL_CLOSED"


@dataclass(frozen=True)
class RequestedOutput:
    """One explicitly requested output in original question order.

    ``source_span`` is a half-open character interval over the original request.
    ``source_text`` is therefore always recoverable without normalization.
    ``question`` is a standalone question compiled independently into ``plan``;
    it may add an explicit shared predicate, year, or unit from the same
    request when a two-metric respectively construction leaves them outside
    one metric's source span. It never invents a metric, period, entity, unit,
    or comparison direction. The public ACTUAL default is recorded separately
    in ``assumptions``.
    """

    output_id: str
    ordinal: int
    source_span: tuple[int, int]
    source_text: str
    question: str
    plan: QuestionPlan
    assumptions: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuestionRequestPlan:
    """Pure question-side partition; it never reads evidence or model output."""

    question: str
    mode: str
    outputs: tuple[RequestedOutput, ...]
    split_strategy: str
    abstention_reason: str
    response_policy: str = VERIFIED_PARTIAL_FAIL_CLOSED
    schema_version: str = QUESTION_REQUEST_SCHEMA_VERSION

    @property
    def requested_output_count(self) -> int:
        return len(self.outputs)

    @property
    def is_multi_output(self) -> bool:
        return self.mode == MULTI_OUTPUT

    @property
    def fail_closed(self) -> bool:
        return self.mode == ABSTAIN


@dataclass(frozen=True)
class _Partition:
    spans: tuple[tuple[int, int], ...]
    strategy: str


_NUMBERED_MARKER = re.compile(
    r"(?P<lead>^|[\n\r;:]|\?\s+|\.\s+)\s*"
    r"(?P<marker>\(\d{1,2}\)|\d{1,2}[.)])\s+",
    re.MULTILINE,
)
_AND_SEPARATOR = re.compile(r"\s+(?:va|và|and)\s+", re.IGNORECASE)
_RESPECTIVELY = re.compile(
    r"\s+(?:lan\s+luot|lần\s+lượt)\s+"
    r"(?:(?:la|là)\s+)?bao\s+(?:nhieu|nhiêu)\b",
    re.IGNORECASE,
)
_OUTPUT_PREDICATE = re.compile(
    r"\bbao\s+nhieu\b|\bla\s+gi\b|\bwhat\s+is\b|\bhow\s+much\b",
)
_IMPERATIVE_PREFIXES = (
    "cho biet ",
    "hay cho biet ",
    "xac dinh ",
    "hay xac dinh ",
    "tinh ",
    "hay tinh ",
    "neu gia tri ",
    "liet ke ",
)
_MULTI_INTENT_MARKERS = (
    "hai ket qua",
    "ba ket qua",
    "bon ket qua",
    "hai gia tri",
    "ba gia tri",
    "bon gia tri",
    "tung ket qua",
    "tung gia tri",
    "moi ket qua",
    "moi gia tri",
    "lan luot",
    "tra loi rieng",
    "liet ke cac gia tri",
)
_INSTRUCTION_TOKENS = {
    "2",
    "3",
    "4",
    "ba",
    "biet",
    "bon",
    "cac",
    "cho",
    "day",
    "gia",
    "hai",
    "hay",
    "ke",
    "ket",
    "liet",
    "loi",
    "long",
    "moi",
    "muc",
    "nhung",
    "qua",
    "rieng",
    "sau",
    "tinh",
    "tra",
    "tri",
    "tung",
    "vui",
    "xac",
    "dinh",
}
_SUBJECT_STOPWORDS = {
    "actual",
    "bao",
    "bang",
    "biet",
    "cho",
    "dong",
    "gia",
    "hay",
    "hien",
    "ke",
    "ket",
    "la",
    "liet",
    "nam",
    "nhieu",
    "phan",
    "plan",
    "qua",
    "thuc",
    "tinh",
    "tram",
    "tri",
    "vnd",
    "xac",
    "dinh",
}
_CROSS_OUTPUT_REFERENCES = (
    "cac gia tri nay",
    "chung",
    "gia tri do",
    "gia tri nay",
    "gia tri tren",
    "hai gia tri",
    "hai ket qua",
    "ket qua do",
    "ket qua nay",
    "ket qua tren",
    "so lieu do",
    "so lieu nay",
    "so lieu tren",
    "tong hai ket qua",
)


def _trim_span(question: str, start: int, end: int) -> tuple[int, int]:
    while start < end and question[start].isspace():
        start += 1
    while end > start and question[end - 1].isspace():
        end -= 1
    return start, end


def _safe_instruction_prefix(text: str) -> bool:
    normalized = normalize_text(text).strip()
    if re.fullmatch(
        r"(?:(?:hay|vui long)\s+)?tra loi"
        r"(?:\s+cac muc sau)?\s+theo dung thu tu",
        normalized,
    ):
        return True
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    return bool(tokens) and tokens.issubset(_INSTRUCTION_TOKENS)


def _strip_leading_instruction(
    question: str, span: tuple[int, int]
) -> tuple[int, int] | None:
    start, end = _trim_span(question, *span)
    colon = question.find(":", start, end)
    if colon < 0:
        return start, end
    prefix = question[start:colon]
    if _safe_instruction_prefix(prefix):
        return _trim_span(question, colon + 1, end)
    if any(marker in normalize_text(prefix) for marker in _MULTI_INTENT_MARKERS):
        return None
    return start, end


def _predicate_count(text: str) -> int:
    return len(_OUTPUT_PREDICATE.findall(normalize_text(text)))


def _has_output_request(text: str) -> bool:
    normalized = normalize_text(text).strip()
    return _predicate_count(text) > 0 or normalized.startswith(_IMPERATIVE_PREFIXES)


def _has_independent_subject(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9]+", normalize_text(text))
    meaningful = [
        token
        for token in tokens
        if token not in _SUBJECT_STOPWORDS
        and token not in _INSTRUCTION_TOKENS
        and not re.fullmatch(r"(?:19|20)\d{2}", token)
    ]
    return bool(meaningful)


def _has_cross_output_reference(text: str) -> bool:
    normalized = f" {normalize_text(text)} "
    return any(f" {phrase} " in normalized for phrase in _CROSS_OUTPUT_REFERENCES)


def _standalone_question(source_text: str) -> str:
    text = source_text.strip()
    if _has_output_request(text):
        return text
    return text.rstrip(" .;,:?") + " là bao nhiêu?"


def _declared_output_count(question: str) -> int | None:
    normalized = normalize_text(question)
    words = {"hai": 2, "ba": 3, "bon": 4}
    counts = {
        words.get(match.group(1), int(match.group(1)) if match.group(1).isdigit() else 0)
        for match in re.finditer(
            r"\b(hai|ba|bon|[234])\s+(?:ket qua|gia tri|muc)\b", normalized
        )
    }
    counts.discard(0)
    if not counts:
        return None
    if len(counts) > 1:
        return -1
    return next(iter(counts))


def _numbered_partition(question: str) -> _Partition | None:
    markers = list(_NUMBERED_MARKER.finditer(question))
    if len(markers) < 2:
        return None
    shared_prefix = question[: markers[0].start("marker")]
    if shared_prefix.strip(" \t\r\n:;.-") and not _safe_instruction_prefix(shared_prefix):
        return _Partition((), "UNSAFE_SHARED_CONTEXT")
    spans = []
    for position, marker in enumerate(markers):
        end = (
            markers[position + 1].start("marker")
            if position + 1 < len(markers)
            else len(question)
        )
        spans.append(_trim_span(question, marker.end("marker"), end))
    return _Partition(tuple(spans), "NUMBERED")


def _question_sentence_partition(question: str) -> _Partition | None:
    ends = [match.end() for match in re.finditer(r"\?", question)]
    if len(ends) < 2:
        return None
    spans: list[tuple[int, int]] = []
    start = 0
    for end in ends:
        spans.append(_trim_span(question, start, end))
        start = end
    if question[start:].strip(" \t\r\n.;"):
        return _Partition((), "INVALID_TRAILING_TEXT")
    return _Partition(tuple(spans), "QUESTION_SENTENCES")


def _semicolon_partition(question: str) -> _Partition | None:
    separators = [match.start() for match in re.finditer(r";", question)]
    if not separators:
        return None
    boundaries = (0, *(position + 1 for position in separators), len(question))
    ends = (*separators, len(question))
    spans = tuple(
        _trim_span(question, boundaries[index], ends[index])
        for index in range(len(ends))
    )
    return _Partition(spans, "SEMICOLON")


def _respectively_partition(question: str) -> _Partition | None:
    marker = _RESPECTIVELY.search(question)
    if marker is None:
        return None
    tail = question[marker.end() :].strip(" \t\r\n.?!")
    if tail:
        return _Partition((), "INVALID_RESPECTIVELY_TAIL")
    prefix_start, prefix_end = _trim_span(question, 0, marker.start())
    stripped = _strip_leading_instruction(question, (prefix_start, prefix_end))
    if stripped is None:
        return _Partition((), "UNSAFE_SHARED_CONTEXT")
    prefix_start, prefix_end = stripped
    separators = list(_AND_SEPARATOR.finditer(question, prefix_start, prefix_end))
    if len(separators) != 1:
        return _Partition((), "AMBIGUOUS_RESPECTIVELY_LIST")
    separator = separators[0]
    spans = (
        _trim_span(question, prefix_start, separator.start()),
        _trim_span(question, separator.end(), prefix_end),
    )
    return _Partition(spans, "RESPECTIVELY")


def _shared_ratio_respectively(question: str) -> QuestionRequestPlan | None:
    """Partition two named ratios with one explicitly shared year and unit."""
    match = re.fullmatch(
        r"\s*(?P<first>.+?)\s+(?:và|and)\s+(?P<second>.+?)"
        r"\s+năm\s+(?P<year>(?:19|20)\d{2})\s+"
        r"lần\s+lượt\s+(?:là\s+)?bao\s+nhiêu\s+"
        r"(?P<unit>phần\s+trăm|lần|%)\s*[?.!]*",
        question, re.IGNORECASE,
    )
    if match is None:
        return None
    from .metric_semantics import identity
    ratios={"return_on_assets","return_on_equity","current_ratio",
            "quick_ratio","gross_margin","net_margin"}
    metrics=(match.group('first').strip(),match.group('second').strip())
    if any(identity(metric) not in ratios for metric in metrics):
        return None
    year=match.group('year');unit=match.group('unit')
    outputs=[]
    for ordinal,(group,metric) in enumerate(zip(('first','second'),metrics),1):
        start,end=match.span(group)
        standalone=f"{metric} năm {year} là bao nhiêu {unit}?"
        plan=compile_question_plan(standalone)
        outputs.append(RequestedOutput(
            output_id=f"output_{ordinal:02d}",ordinal=ordinal,
            source_span=(start,end),source_text=question[start:end],
            question=standalone,plan=plan,
            assumptions=("shared_explicit_year","shared_explicit_unit"),
        ))
    result=QuestionRequestPlan(question,MULTI_OUTPUT,tuple(outputs),
        "SHARED_RATIO_RESPECTIVELY","")
    return result if not validate_question_request_plan(result) else _abstain(
        question,"invalid_shared_ratio_request","SHARED_RATIO_RESPECTIVELY")


def _repeated_predicate_partition(question: str) -> _Partition | None:
    separators = list(_AND_SEPARATOR.finditer(question))
    if not separators or _predicate_count(question) < 2:
        return None
    if len(separators) > 8:
        return _Partition((), "AMBIGUOUS_CONJUNCTIONS")

    partitions: set[tuple[tuple[int, int], ...]] = set()
    # Enumerate conjunction boundaries. A segment is an output only when it
    # contains exactly one explicit answer predicate; conjunctions inside a
    # calculation therefore remain inside that segment.
    for mask in range(1, 1 << len(separators)):
        spans: list[tuple[int, int]] = []
        start = 0
        for index, separator in enumerate(separators):
            if mask & (1 << index):
                spans.append(_trim_span(question, start, separator.start()))
                start = separator.end()
        spans.append(_trim_span(question, start, len(question)))
        prepared: list[tuple[int, int]] = []
        valid = True
        for index, span in enumerate(spans):
            current = _strip_leading_instruction(question, span) if index == 0 else span
            if current is None:
                valid = False
                break
            text = question[current[0] : current[1]]
            if _predicate_count(text) != 1 or not _has_independent_subject(text):
                valid = False
                break
            prepared.append(current)
        if valid:
            partitions.add(tuple(prepared))
    if not partitions:
        return _Partition((), "UNSPLITTABLE_REPEATED_PREDICATE")
    if len(partitions) > 1:
        return _Partition((), "AMBIGUOUS_REPEATED_PREDICATE")
    return _Partition(next(iter(partitions)), "REPEATED_PREDICATE")


def _explicit_multi_intent(question: str) -> bool:
    normalized = normalize_text(question)
    return (
        any(marker in normalized for marker in _MULTI_INTENT_MARKERS)
        or _predicate_count(question) >= 2
        or len(re.findall(r"\?", question)) >= 2
        or _numbered_partition(question) is not None
    )


def _abstain(question: str, reason: str, strategy: str = "NONE") -> QuestionRequestPlan:
    return QuestionRequestPlan(
        question=question,
        mode=ABSTAIN,
        outputs=(),
        split_strategy=strategy,
        abstention_reason=reason,
    )


def _build_partition_plan(
    question: str,
    partition: _Partition,
    *,
    inventory_fragments_allowed: bool,
) -> QuestionRequestPlan:
    if not partition.spans:
        reason = re.sub(r"\s+", "_", normalize_text(partition.strategy).lower())
        return _abstain(question, reason, partition.strategy)

    prepared: list[tuple[int, int]] = []
    for index, span in enumerate(partition.spans):
        current = _strip_leading_instruction(question, span) if index == 0 else _trim_span(question, *span)
        if current is None:
            return _abstain(question, "unsafe_shared_context", partition.strategy)
        start, end = current
        if start >= end:
            return _abstain(question, "empty_requested_output", partition.strategy)
        source_text = question[start:end]
        if not _has_independent_subject(source_text):
            return _abstain(question, "output_lacks_independent_subject", partition.strategy)
        if not inventory_fragments_allowed and not _has_output_request(source_text):
            return _abstain(question, "output_lacks_explicit_predicate", partition.strategy)
        if index > 0 and _has_cross_output_reference(source_text):
            return _abstain(question, "cross_output_reference", partition.strategy)
        prepared.append((start, end))

    declared = _declared_output_count(question)
    if declared is not None and declared != len(prepared):
        return _abstain(question, "declared_output_count_mismatch", partition.strategy)

    outputs: list[RequestedOutput] = []
    for ordinal, (start, end) in enumerate(prepared, start=1):
        source_text = question[start:end]
        original_standalone = _standalone_question(source_text)
        standalone = resolve_public_question(original_standalone)
        scalar_plan = compile_question_plan(standalone)
        if (
            scalar_plan.requested_output_count != 1
            or scalar_plan.answer_mode != "SINGLE_SCALAR"
            or scalar_plan.abstention_reason == "unsupported_multi_output_request"
        ):
            return _abstain(question, "subquestion_not_scalar", partition.strategy)
        outputs.append(
            RequestedOutput(
                output_id=f"output_{ordinal:02d}",
                ordinal=ordinal,
                source_span=(start, end),
                source_text=source_text,
                question=standalone,
                plan=scalar_plan,
                assumptions=public_question_assumptions(original_standalone),
            )
        )

    if len(outputs) < 2:
        return _abstain(question, "multi_output_partition_too_small", partition.strategy)
    result = QuestionRequestPlan(
        question=question,
        mode=MULTI_OUTPUT,
        outputs=tuple(outputs),
        split_strategy=partition.strategy,
        abstention_reason="",
    )
    errors = validate_question_request_plan(result)
    return result if not errors else _abstain(
        question, "invalid_request_plan:" + ",".join(errors), partition.strategy
    )


def _single_output_plan(question: str) -> QuestionRequestPlan:
    start, end = _trim_span(question, 0, len(question))
    if start >= end:
        return _abstain(question, "empty_question")
    source_text = question[start:end]
    resolved = resolve_public_question(source_text)
    scalar_plan = compile_question_plan(resolved)
    if scalar_plan.requested_output_count != 1 or scalar_plan.answer_mode != "SINGLE_SCALAR":
        return _abstain(question, "ambiguous_or_unsplittable_multi_output")
    output = RequestedOutput(
        output_id="output_01",
        ordinal=1,
        source_span=(start, end),
        source_text=source_text,
        question=resolved,
        plan=scalar_plan,
        assumptions=public_question_assumptions(source_text),
    )
    return QuestionRequestPlan(
        question=question,
        mode=SINGLE_OUTPUT,
        outputs=(output,),
        split_strategy="NONE",
        abstention_reason="",
        response_policy="SINGLE_OUTPUT",
    )


def compile_question_request(question: str) -> QuestionRequestPlan:
    """Compile one scalar or explicitly separable multi-output request.

    Structural boundaries are intentionally ordered from strongest to weakest.
    If a strong multi-output signal cannot produce independent scalar clauses,
    this function returns an empty fail-closed plan instead of falling back to
    the legacy scalar planner.
    """

    if not isinstance(question, str) or not question.strip():
        return _abstain(question if isinstance(question, str) else "", "empty_question")

    from .comparison_overview import build_comparison_request
    overview = build_comparison_request(question)
    if overview is not None:
        return overview
    normalized = normalize_text(question)
    inventory_fragments_allowed = any(
        marker in normalized
        for marker in (
            "hai ket qua",
            "ba ket qua",
            "bon ket qua",
            "hai gia tri",
            "ba gia tri",
            "bon gia tri",
            "tung ket qua",
            "tung gia tri",
            "tra loi rieng",
            "liet ke cac gia tri",
        )
    )

    numbered = _numbered_partition(question)
    if numbered is not None:
        return _build_partition_plan(
            question,
            numbered,
            inventory_fragments_allowed=inventory_fragments_allowed,
        )

    question_sentences = _question_sentence_partition(question)
    if question_sentences is not None:
        return _build_partition_plan(
            question,
            question_sentences,
            inventory_fragments_allowed=False,
        )

    semicolon = _semicolon_partition(question)
    if semicolon is not None:
        semicolon_texts = [question[start:end] for start, end in semicolon.spans]
        if inventory_fragments_allowed or all(
            _has_output_request(text) for text in semicolon_texts
        ):
            return _build_partition_plan(
                question,
                semicolon,
                inventory_fragments_allowed=inventory_fragments_allowed,
            )

    shared_ratios = _shared_ratio_respectively(question)
    if shared_ratios is not None:
        return shared_ratios

    respectively = _respectively_partition(question)
    if respectively is not None:
        return _build_partition_plan(
            question,
            respectively,
            inventory_fragments_allowed=True,
        )

    repeated = _repeated_predicate_partition(question)
    if repeated is not None:
        return _build_partition_plan(
            question,
            repeated,
            inventory_fragments_allowed=False,
        )

    if _explicit_multi_intent(question):
        return _abstain(question, "ambiguous_or_unsplittable_multi_output")
    return _single_output_plan(question)


def validate_question_request_plan(plan: QuestionRequestPlan) -> tuple[str, ...]:
    """Return deterministic structural errors without reading document evidence."""

    from .comparison_overview import STRATEGY, build_comparison_request
    if plan.split_strategy == STRATEGY:
        return () if plan == build_comparison_request(plan.question) else ("invalid_comparison_overview_plan",)
    errors: list[str] = []
    if plan.schema_version != QUESTION_REQUEST_SCHEMA_VERSION:
        errors.append("wrong_schema_version")
    if plan.mode not in {SINGLE_OUTPUT, MULTI_OUTPUT, ABSTAIN}:
        errors.append("invalid_mode")
    if plan.mode == ABSTAIN:
        if plan.outputs:
            errors.append("abstain_has_outputs")
        if not plan.abstention_reason:
            errors.append("abstain_missing_reason")
        return tuple(errors)
    if plan.abstention_reason:
        errors.append("nonabstain_has_reason")
    if plan.mode == SINGLE_OUTPUT and len(plan.outputs) != 1:
        errors.append("single_output_count_mismatch")
    if plan.mode == MULTI_OUTPUT and len(plan.outputs) < 2:
        errors.append("multi_output_count_mismatch")

    previous_end = -1
    seen_ids: set[str] = set()
    for ordinal, output in enumerate(plan.outputs, start=1):
        if output.ordinal != ordinal:
            errors.append("non_deterministic_ordinal")
        if output.output_id != f"output_{ordinal:02d}" or output.output_id in seen_ids:
            errors.append("invalid_output_id")
        seen_ids.add(output.output_id)
        start, end = output.source_span
        if not (0 <= start < end <= len(plan.question)):
            errors.append("invalid_source_span")
            continue
        if start < previous_end:
            errors.append("overlapping_or_reordered_span")
        previous_end = end
        if plan.question[start:end] != output.source_text:
            errors.append("source_span_text_mismatch")
        if output.plan.question != output.question:
            errors.append("subplan_question_mismatch")
        if output.plan != compile_question_plan(output.question):
            errors.append("subplan_not_deterministic")
        if output.plan.requested_output_count != 1:
            errors.append("subplan_not_scalar")
    return tuple(dict.fromkeys(errors))


__all__ = (
    "ABSTAIN",
    "MULTI_OUTPUT",
    "QUESTION_REQUEST_SCHEMA_VERSION",
    "SINGLE_OUTPUT",
    "VERIFIED_PARTIAL_FAIL_CLOSED",
    "QuestionRequestPlan",
    "RequestedOutput",
    "compile_question_request",
    "validate_question_request_plan",
)
