from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import re

from .financial_identities import match_financial_identities
from .ontology import metric_unit_family, normalize_text
from .metric_semantics import normalize_metric_text


_ENTITY_STOPWORDS = {
    "bao",
    "cao",
    "co",
    "cong",
    "cua",
    "nam",
    "phan",
    "tap",
    "tong",
    "ty",
}


class DirectionIntent(str, Enum):
    """Question-level subtraction direction, independent of candidate order."""

    NEW_MINUS_OLD = "NEW_MINUS_OLD"
    LEFT_MINUS_RIGHT = "LEFT_MINUS_RIGHT"
    DECREASE_AMOUNT = "DECREASE_AMOUNT"
    ABSOLUTE_GAP = "ABSOLUTE_GAP"
    PLAN_MINUS_ACTUAL = "PLAN_MINUS_ACTUAL"
    ACTUAL_MINUS_PLAN = "ACTUAL_MINUS_PLAN"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True)
class OperandSlot:
    """One immutable question-side operand identity.

    Slots contain intent compiled from the question and ontology only.  They
    never contain a numeric value, candidate index, page, table coordinate or
    any other document-conditioned selection signal.
    """

    id: str
    metric: str
    year: str
    column_role: str
    entity_scope: str
    unit_family: str
    intent_sources: tuple[str, ...]
    # year carries a calendar year when the question names one. Financial
    # statement endpoints such as cuoi nam/dau nam are orthogonal to the
    # calendar year, so keep their source-proven role explicitly instead of
    # fabricating a year token. Empty preserves the pre-V14 schema.
    period_role: str = ""


@dataclass(frozen=True)
class ProgramIntentNode:
    """One topologically ordered, non-executable compiler-side DAG node."""

    id: str
    operation: str
    inputs: tuple[str, ...]
    output_unit_family: str
    intent_sources: tuple[str, ...]


@dataclass(frozen=True)
class QuestionPlan:
    """Typed intent shared by retrieval, HRM decoding and verification."""

    question: str
    normalized: str
    operation: str | None
    metric_text: str
    requested_years: tuple[str, ...]
    from_year: str | None
    to_year: str | None
    entity_mode: str
    entity_text: str
    entity_terms: tuple[str, ...]
    asks_percent: bool
    requested_unit_family: str
    requested_scale: str | None
    temporal_change: bool
    operation_candidates: tuple[str, ...]
    operand_roles: tuple[str, ...]
    direct_metric_lookup: bool
    financial_identity_ids: tuple[str, ...]
    proof_operand_roles: tuple[str, ...]
    direction_intent: DirectionIntent | None
    requested_period_roles: tuple[str, ...] = ()
    column_roles: tuple[str, ...] = ()
    addition_role_hypotheses: tuple[tuple[str, str], ...] = ()
    operand_slots: tuple[OperandSlot, ...] = ()
    operand_schema_version: str = "question-operand-slot-v1"
    reference_base: str | None = None
    comparison_target: str | None = None
    comparison_base: str | None = None
    program_dag: tuple[ProgramIntentNode, ...] = ()
    program_schema_version: str = "question-program-intent-dag-v1"
    program_identity_ids: tuple[str, ...] = ()
    abstention_reason: str = ""
    answer_mode: str = "SINGLE_SCALAR"
    requested_output_count: int = 1
    requested_report_year: str = ""
    requested_value_column: str = ""
    requested_event: str = ""
    requested_source_kind: str = ""
    requested_source_heading: str = ""
    requested_source_years: tuple[str, ...] = ()
    requested_source_year_count: int = 0

    @property
    def requested_accounting_scope(self) -> str:
        from .table_semantics import accounting_scope
        return accounting_scope(self.question)

    @property
    def requested_section(self) -> str:
        from .source_bindings import requested_section
        return requested_section(self.question)

    @property
    def program_dag_operations(self) -> tuple[str, ...]:
        return tuple(node.operation for node in self.program_dag)

    @property
    def program_column_roles(self) -> tuple[str, ...]:
        """Expand leaf roles in executable DAG order without duplicating slots.

        ``operand_slots`` is an identity set, while a multi-step program may
        reuse one identity as a denominator. This projection preserves that
        use-site order for compiler audits without changing the legacy runtime
        ``column_roles`` field or materializing duplicate operand identities.
        """

        slot_roles = {slot.id: slot.column_role for slot in self.operand_slots}
        expanded: dict[str, tuple[str, ...]] = {
            slot.id: (slot.column_role,) for slot in self.operand_slots
        }
        for node in self.program_dag:
            roles: list[str] = []
            for input_id in node.inputs:
                if input_id in slot_roles:
                    roles.append(slot_roles[input_id])
                else:
                    roles.extend(expanded.get(input_id, ()))
            expanded[node.id] = tuple(roles)
        return expanded.get("answer", ())

    @property
    def program_dag_validation_errors(self) -> tuple[str, ...]:
        return _program_intent_dag_errors(
            self.operand_slots,
            self.program_dag,
            reference_base=self.reference_base,
            root_operation=self.operation,
        )

    @property
    def required_operand_years(self) -> tuple[str, ...]:
        if self.from_year and self.to_year:
            if self.direction_intent is DirectionIntent.DECREASE_AMOUNT:
                return self.from_year, self.to_year
            return self.to_year, self.from_year
        if self.temporal_change and len(self.requested_years) >= 2:
            ordered = sorted(set(self.requested_years), key=int, reverse=True)
            if self.direction_intent is DirectionIntent.DECREASE_AMOUNT:
                ordered.reverse()
            return tuple(ordered)
        return self.requested_years

    @property
    def required_operand_period_roles(self) -> tuple[str, ...]:
        """Return relative endpoints in executable temporal order.

        Relative endpoints are typed separately from calendar years. This is
        important when both cuoi nam 2025 and dau nam 2025 carry the same
        printed year but identify different physical columns.
        """

        roles = self.requested_period_roles
        if self.temporal_change and {"closing", "opening"}.issubset(set(roles)):
            return ("opening", "closing") if self.direction_intent is DirectionIntent.DECREASE_AMOUNT else ("closing", "opening")
        if (
            self.temporal_change
            and {"current", "prior"}.issubset(set(roles))
        ):
            if self.direction_intent is DirectionIntent.DECREASE_AMOUNT:
                return "prior", "current"
            return "current", "prior"
        return roles

    @property
    def interest_bearing_debt_share(self) -> bool:
        slot_ids = {slot.id for slot in self.operand_slots}
        if {
            "short_term_borrowings",
            "long_term_borrowings",
            "total_capital",
        }.issubset(slot_ids):
            return True
        normalized = f" {self.normalized} "
        asks_share = (
            " chiem bao nhieu phan tram " in normalized
            or " ty trong " in normalized
            and " tren " in normalized
        )
        names_financial_debt = any(
            phrase in normalized
            for phrase in (
                " no vay chiu lai ",
                " no vay tai chinh chiu lai ",
                " tong no vay chiu lai ",
                " vay va no thue tai chinh ",
            )
        )
        return (
            asks_share
            and names_financial_debt
            and " ngan han " in normalized
            and " dai han " in normalized
            and " tong nguon von " in normalized
        )


_PROOF_CONSTRAINTS = (
    "execution_valid",
    "pointer_uniqueness",
    "operation_match",
    "operand_arity_intent",
    "set_completeness",
    "operand_period_match",
    "operand_entity_match",
    "query_metric",
    "column_role_match",
    "column_role_unique",
    "logical_cell_unique",
    "source_provenance_complete",
    "answer_mode_complete",
    "reference_base_match",
    "all_leaf_nodes_valid",
    "all_intermediate_nodes_valid",
    "dag_execution_valid",
    "operand_roles",
    "add_grounding",
    "financial_identity",
    "metric_match",
    "entity_match",
    "unit_match",
    "unit_scale_match",
    "unit_compatibility",
    "unit_algebra",
    "same_table",
    "same_row",
    "period_coverage",
    "temporal_direction",
    "denominator_safe",
    "reporting_entity",
    "named_entity",
    "graph_same_metric",
    "graph_same_entity",
    "graph_same_table",
    "graph_same_row",
    "graph_same_unit",
    "graph_temporal_order",
)


@dataclass(frozen=True)
class QuestionProofContract:
    """Question-conditioned verifier obligations.

    Optional evidence may improve ranking but cannot make proof completeness
    UNKNOWN. Constraints that do not apply are kept explicit for audit and are
    excluded from every completeness denominator.
    """

    required: tuple[str, ...]
    optional: tuple[str, ...]
    not_applicable: tuple[str, ...]


def compile_question_proof_contract(
    plan: QuestionPlan,
    operation: str,
    *,
    financial_identity_required: bool = False,
) -> QuestionProofContract:
    """Compile the Phase-A proof contract for one operation hypothesis."""

    required = {
        "execution_valid",
        "pointer_uniqueness",
        "operation_match",
        "operand_arity_intent",
    }
    applicable = set(required)
    arity_two_or_more = operation != "SELECT"

    if operation == "SELECT":
        required.add("query_metric")
        applicable.add("query_metric")
    else:
        applicable.update(
            {
                "metric_match",
                "entity_match",
                "unit_match",
                "same_table",
                "same_row",
            }
        )

    if plan.requested_value_column or plan.column_roles or any(slot.column_role for slot in plan.operand_slots):
        required.add("column_role_match")
        applicable.add("column_role_match")

    # QP3-CORE applies the provenance contract only to questions for which the
    # compiler produced immutable typed slots.  This keeps non-core legacy
    # questions outside the declared support lane instead of pretending that
    # sparse metadata is complete.
    if plan.operand_slots or plan.requested_report_year:
        required.update({"logical_cell_unique", "source_provenance_complete"})
        applicable.update({"logical_cell_unique", "source_provenance_complete"})

    if plan.requested_years or plan.requested_period_roles:
        applicable.add("period_coverage")
        if operation == "SELECT" or plan.temporal_change or operation == "AVERAGE":
            required.add("period_coverage")

    if (
        plan.direction_intent is not None
        and operation in {"SUBTRACT", "PERCENT_CHANGE"}
    ):
        required.add("temporal_direction")
        applicable.add("temporal_direction")
        if plan.temporal_change:
            required.add("metric_match")
    elif operation == "AVERAGE":
        required.add("metric_match")

    if operation in {"DIVIDE", "RATIO_PERCENT", "PERCENT_CHANGE"}:
        required.update(
            {"reference_base_match", "denominator_safe", "unit_compatibility"}
        )
        applicable.update(
            {"reference_base_match", "denominator_safe", "unit_compatibility"}
        )
    if operation in {"DIVIDE", "RATIO_PERCENT"}:
        required.add("operand_roles")
        applicable.add("operand_roles")
    if operation == "MULTIPLY":
        required.update({"operand_roles", "unit_algebra"})
        applicable.update({"operand_roles", "unit_algebra"})
    if operation == "ADD":
        required.add("add_grounding")
        applicable.update({"operand_roles", "add_grounding"})
    if operation == "SUM_N":
        required.update(
            {
                "set_completeness",
                "operand_roles",
                "add_grounding",
                "operand_period_match",
                "operand_entity_match",
                "unit_compatibility",
            }
        )
        applicable.update(
            {
                "set_completeness",
                "operand_roles",
                "add_grounding",
                "operand_period_match",
                "operand_entity_match",
                "unit_compatibility",
            }
        )

    if financial_identity_required:
        required.add("financial_identity")
        applicable.add("financial_identity")

    if plan.requested_scale is not None:
        required.add("unit_scale_match")
        applicable.add("unit_scale_match")

    if plan.requested_output_count > 1:
        required.add("answer_mode_complete")
        applicable.add("answer_mode_complete")

    if len(plan.program_dag) > 1:
        required.update(
            {
                "all_leaf_nodes_valid",
                "all_intermediate_nodes_valid",
                "dag_execution_valid",
            }
        )
        applicable.update(
            {
                "all_leaf_nodes_valid",
                "all_intermediate_nodes_valid",
                "dag_execution_valid",
            }
        )

    if plan.entity_mode == "named":
        required.add("named_entity")
        applicable.add("named_entity")
    else:
        applicable.add("reporting_entity")

    if arity_two_or_more:
        applicable.update(
            {
                "graph_same_metric",
                "graph_same_entity",
                "graph_same_table",
                "graph_same_row",
                "graph_same_unit",
            }
        )
        if plan.temporal_change:
            applicable.add("graph_temporal_order")

    optional = applicable - required
    not_applicable = set(_PROOF_CONSTRAINTS) - applicable
    ordered = {name: index for index, name in enumerate(_PROOF_CONSTRAINTS)}
    sort_key = lambda name: ordered[name]
    return QuestionProofContract(
        required=tuple(sorted(required, key=sort_key)),
        optional=tuple(sorted(optional, key=sort_key)),
        not_applicable=tuple(sorted(not_applicable, key=sort_key)),
    )


_OUTPUT_UNIT_DIRECTIVE = re.compile(
    r"\b(?:tinh bang|quy doi sang|quy doi ra|bieu thi bang|"
    r"tinh theo don vi|theo don vi|don vi la|don vi)\s+"
    r"(?P<unit>(?:(?:nghin|ngan|trieu|ty)\s+)?dong|vnd)\b"
)
_OUTPUT_SCALES = {"dong": "vnd", "vnd": "vnd", "nghin dong": "thousand_vnd",
                  "ngan dong": "thousand_vnd", "trieu dong": "million_vnd", "ty dong": "billion_vnd"}


def _strip_output_unit_directives(text: str) -> str:
    # Remove only a directive WITH an explicit supported unit. The original
    # question still controls output scale, conflict detection and arithmetic.
    return _OUTPUT_UNIT_DIRECTIVE.sub(" ", text)


def _compile_requested_scale(normalized: str) -> str | None:
    """Compile an explicit monetary output scale from question text only."""

    padded = f" {normalized} "
    scales: set[str] = {_OUTPUT_SCALES[re.sub(r"\s+", " ", m["unit"])]
                        for m in _OUTPUT_UNIT_DIRECTIVE.finditer(normalized)}
    patterns = (
        ("thousand_vnd", r"\bnghin\s+dong\b"),
        ("million_vnd", r"\btrieu\s+dong\b"),
        ("billion_vnd", r"\bty\s+dong\b"),
    )
    for scale, pattern in patterns:
        if re.search(pattern, normalized):
            scales.add(scale)
    if re.search(r"\bvnd\b", normalized) or any(
        phrase in padded
        for phrase in (
            " bao nhieu dong ",
            " don vi dong ",
            " theo dong ",
        )
    ):
        scales.add("vnd")
    if len(scales) > 1:
        return "conflict"
    return next(iter(scales), None)


_RELATIVE_PERIOD_TOKEN = re.compile(
    r"\b(?:"
    r"cuoi\s+(?:nam|ky)|"
    r"dau\s+(?:nam|ky)|"
    r"nam\s+(?:nay|hien\s+tai|truoc)"
    r")\b"
)


def _compile_requested_period_roles(normalized: str) -> tuple[str, ...]:
    """Compile source-independent financial endpoints in lexical order."""

    roles: list[str] = []
    for match in _RELATIVE_PERIOD_TOKEN.finditer(normalized):
        token = match.group(0)
        roles.append(
            "closing" if token.startswith("cuoi ") else
            "opening" if token.startswith("dau ") else
            "current" if token in {"nam nay", "nam hien tai"} else "prior"
        )
    return tuple(roles)


def _strip_relative_period_surface(text: str) -> str:
    """Remove typed endpoint wording from a row-metric surface."""

    return re.sub(
        r"\b(?:(?:tai|vao|trong|tinh\s+den)\s+)?"
        r"(?:so\s+)?(?:cuoi\s+(?:nam|ky)|dau\s+(?:nam|ky)|"
        r"nam\s+(?:nay|hien\s+tai|truoc))\b",
        " ",
        text,
    )


def _strip_calendar_period_surface(text: str) -> str:
    """Remove a year introducer only when followed by an explicit calendar year.

    This changes the row-label surface, not the year/period obligation compiled
    from the full question. Other uses of the same word remain in the metric.
    """
    return re.sub(
        r"\b(?:vao|trong|tai|o|cho|cua|theo)\s+nam\s+(?:19|20)\d{2}\b",
        " ",
        text,
    )


def _ordered_operand_period_roles(
    requested: tuple[str, ...],
    *,
    temporal_change: bool,
    direction_intent: DirectionIntent | None,
) -> tuple[str, ...]:
    if temporal_change and {"closing", "opening"}.issubset(set(requested)):
        return ("opening", "closing") if direction_intent is DirectionIntent.DECREASE_AMOUNT else ("closing", "opening")
    if temporal_change and {"current", "prior"}.issubset(set(requested)):
        if direction_intent is DirectionIntent.DECREASE_AMOUNT:
            return "prior", "current"
        return "current", "prior"
    return requested


def _normalize_period_measurement(question: str) -> str:
    """Bind two explicitly named years to target/base without using source values."""
    q = re.sub(r"^\s*\d{1,2}[.)]\s+(?=[^\W\d_])", "", question).strip()
    year = r"((?:19|20)\d{2})"
    suffix = r"\s*[?.!]*$"
    prefix_growth = re.fullmatch(
        r"(?:tốc độ tăng trưởng|toc do tang truong|tỷ lệ tăng trưởng|ty le tang truong|mức tăng trưởng|muc tang truong|tỷ lệ tăng|ty le tang|tỷ lệ giảm|ty le giam)\s+(.+?)\s+(?:năm\s+|nam\s+)?" + year
        + r"\s+(?:so với|so voi)\s+(?:năm\s+|nam\s+)?" + year
        + r"(?:\s+(?:là|la))?(?:\s+bao nhiêu|\s+bao nhieu)?(?:\s+(?:phần trăm|phan tram|%))?" + suffix, q, re.I)
    growth = re.fullmatch(r"(.+?)\s+(?:năm\s+|nam\s+)?" + year
        + r"\s+(?:tăng|tang|giảm|giam|thay đổi|thay doi|biến động|bien dong)\s+(?:bao nhiêu|bao nhieu)\s*(?:phần trăm|phan tram|%)\s+(?:so với|so voi)\s+(?:năm\s+|nam\s+)?" + year + suffix, q, re.I)
    match = prefix_growth or growth
    if match and _has_financial_role_anchor(normalize_text(match[1])):
        metric, target, base = match.groups()
        if target != base:
            # A signed change uses the named comparison year as denominator.
            # The renderer describes the actual sign even if the question says tăng.
            return f"{metric} thay đổi từ năm {base} đến năm {target} bao nhiêu phần trăm?"
    ratio = re.fullmatch(r"(.+?)\s+(?:năm\s+|nam\s+)?" + year
        + r"\s+(?:bằng|bang|gấp|gap|chiếm|chiem)\s+(?:bao nhiêu|bao nhieu)\s*(phần trăm|phan tram|%|lần|lan)\s+(?:(?:so với|so voi|của|cua)\s+)?(?:năm\s+|nam\s+)?" + year + suffix, q, re.I)
    if ratio and _has_financial_role_anchor(normalize_text(ratio[1])):
        metric, target, unit, base = ratio.groups()
        if target != base:
            unit = "lần" if normalize_text(unit)=="lan" else "phần trăm"
            return f"{metric} năm {target} chia cho năm {base} là bao nhiêu {unit}?"
    return q


def _same_metric_period_ratio(normalized: str):
    match = re.fullmatch(r"(.+?) (?:nam )?((?:19|20)\d{2}) chia cho (?:nam )?((?:19|20)\d{2}) (?:la )?bao nhieu(?: phan tram| lan)?", normalized)
    return match if match and match[2]!=match[3] and _has_financial_role_anchor(match[1]) else None


def normalize_question_framing(question: str) -> str:
    """Remove request politeness, preserving metrics and every semantic qualifier."""
    if re.fullmatch(r'\s*Kế hoạch .+ năm (?:19|20)\d{2} của toàn công ty là bao nhiêu[?.! ]*',question,re.I):
        question=re.sub(r'\s+của toàn công ty\b','',question,flags=re.I)
    question = _normalize_period_measurement(question)
    subtraction = re.match(r"^\s*(?:(?:lấy|lay)\s+)?(.+?)\s+(?:(?:năm|nam)\s+)?((?:19|20)\d{2})\s+(?:trừ|tru)\s+(.*?)((?:19|20)\d{2})(.*)$", question, re.I)
    if subtraction:
        metric, left_year, repeated_metric, right_year, tail = subtraction.groups()
        from .unit_defaults import monetary_label
        repeated_metric = re.sub(r"\b(?:năm|nam)\s*$", "", repeated_metric, flags=re.I).strip()
        # A fully stated right-hand metric keeps the literal signed A - B intent.
        # Only an elided year operand needs the temporal-difference rewrite.
        if monetary_label(normalize_metric_text(metric)) and not repeated_metric:
            question = f"Chênh lệch {metric} từ năm {right_year} đến năm {left_year}{tail}"
    # Relative ordering here describes arithmetic, not source current/prior columns.
    ordering = re.search(r"[,;]\s*(?:tính|tinh)\s+(?:năm|nam)\s+(sau|trước|truoc)\s+(?:trừ|tru)\s+(?:năm|nam)\s+(trước|truoc|sau)\s*$", question, re.I)
    explicit_years = sorted(set(re.findall(r"\b(?:19|20)\d{2}\b", question)))
    if ordering and len(explicit_years)==2 and normalize_text(ordering[1])!=normalize_text(ordering[2]):
        body=question[:ordering.start()]
        first,last=explicit_years if normalize_text(ordering[1])=="sau" else explicit_years[::-1]
        body=re.sub(r"(?:giữa|giua)\s+(?:19|20)\d{2}\s+(?:và|va)\s+(?:19|20)\d{2}", f"từ năm {first} đến năm {last}", body,flags=re.I)
        question=body
    pattern = r"^\s*(?:(?:vui lòng|vui long|xin)\s+)?(?:(?:hãy|hay)\s+)?(?:cho biết|cho biet|cung cấp|cung cap)\s+|^\s*(?:please\s+)?tell me\s+"
    if normalize_text(question).startswith("theo ") and "," in question:
        prefix, body = question.split(",", 1)
        stripped = re.sub(pattern, "", body, count=1, flags=re.IGNORECASE)
        return prefix + ", " + stripped.strip() if stripped != body else question
    stripped = re.sub(pattern, "", question, count=1, flags=re.IGNORECASE)
    return stripped[:1].upper() + stripped[1:] if stripped != question else question


def public_question_assumptions(question: str) -> tuple[str, ...]:
    framed = normalize_question_framing(question)
    assumptions = ("question_framing:normalized",) if framed != question else ()
    if resolve_public_question(framed) != framed:
        assumptions += ("column_role:default_actual",)
    return assumptions


def _reported_column_request(normalized: str) -> tuple[str, str]:
    """Recognize an explicit lookup location, not an arithmetic change request."""
    from .column_context import column_kind
    match = re.fullmatch(
        r"(.+)\s+(?:tai|o|trong|theo)\s+(?:cot\s+)?(.+?)\s+(?:la |bang )?bao nhieu",
        normalized,
    )
    if match:
        request, column = match.groups()
        if (column_kind(column) in {"comparison", "comparison_ratio"}
                and _has_financial_role_anchor(request)
                and not re.search(r"\b(?:tinh|so sanh|chenh lech|thay doi|tru|chia|nhan)\b", request)):
            return request + " la bao nhieu", column
    return normalized, ""


def compile_question_plan(question: str) -> QuestionPlan:
    from .fiscal_period_evidence import fiscal_question_surface
    fiscal = fiscal_question_surface(question)
    if fiscal and fiscal[0] != normalize_text(question):
        surface, scope = fiscal
        inner = compile_question_plan(resolve_public_question(surface))
        if scope and inner.requested_accounting_scope and inner.requested_accounting_scope != scope:
            return inner
        return replace(inner, question=question, normalized=normalize_text(question))
    framed=re.fullmatch(r'\s*Theo bảng (kết quả (?:sản xuất )?kinh doanh (hợp nhất|riêng lẻ) năm ((?:19|20)\d{2})),\s*(.+)',question,re.I)
    if framed:
        heading,scope,year,body=framed.groups()
        if re.search(r'\bnăm trước\b',body,re.I) and not re.search(r'(?:19|20)\d{2}',body):
            body=re.sub(r'\bnăm trước\b','năm '+str(int(year)-1),body,flags=re.I)
        if not re.search(r'hợp nhất|riêng lẻ',body,re.I):body=body.rstrip('?. ')+f' {scope}?'
        inner=compile_question_plan(resolve_public_question(body))
        return replace(inner,question=question,requested_source_kind='table',requested_source_heading=normalize_text(heading))
    from .source_bindings import question_source_frame
    core, source_frame = question_source_frame(question)
    if source_frame:
        inner = compile_question_plan(core)
        return replace(inner, question=question,
            requested_source_kind=source_frame["kind"],
            requested_source_heading=source_frame["heading"],
            requested_source_years=source_frame["years"],
            requested_source_year_count=source_frame["year_count"])
    from .report_context import report_reference
    reference = report_reference(question)
    if reference:
        year, content_question = reference
        plan = compile_question_plan(content_question)
        return replace(plan, question=question, normalized=normalize_text(question), requested_report_year=year,
                       operation=plan.operation or "SELECT", operation_candidates=plan.operation_candidates or ("SELECT",))
    from .column_context import normalize_period_text
    normalized = normalize_period_text(normalize_question_framing(question))
    normalized, requested_value_column = _reported_column_request(normalized)
    padded = f" {normalized} "
    requested_scale = _compile_requested_scale(normalized)
    years = tuple(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", normalized)))
    requested_period_roles = _compile_requested_period_roles(normalized)
    asks_percent = (
        "%" in question
        or "phan tram" in normalized
        or "ty trong" in normalized
        or "ty le tang" in normalized
        or "ty le giam" in normalized
        or "toc do tang truong" in normalized
    )
    requested_unit_family = ""
    if asks_percent:
        requested_unit_family = "percent"
    elif "he so" in normalized or " bao nhieu lan " in padded:
        requested_unit_family = "ratio"
    elif (
        "gia tri theo menh gia" in normalized
        and any(term in normalized for term in ("co phieu", "co phan"))
    ):
        requested_unit_family = "currency"
    elif requested_scale is not None:
        requested_unit_family = "currency"
    change_wording = any(
        phrase in normalized
        for phrase in (
            "thay doi",
            "chenh lech",
            "tang bao nhieu",
            "giam bao nhieu",
            "muc tang",
            "muc giam",
            "cao hon",
            "thap hon",
            "lon hon",
            "nho hon",
            "it hon",
            "nhieu hon",
        )
    ) or bool(
        # A source noun such as "bang so sanh" describes the report layout;
        # it is not a request to compute a change. The arithmetic phrase
        # remains recognized everywhere else.
        re.search(r"(?<!\bbang )\bso sanh\b", normalized)
    )
    explicit_period_comparison = any(
        phrase in normalized
        for phrase in (
            "cuoi nam so voi dau nam",
            "cuoi ky so voi dau ky",
            "nam nay so voi nam truoc",
        )
    )
    temporal_change = explicit_period_comparison or change_wording and (
        len(years) >= 2
        or ("cuoi nam" in normalized and "dau nam" in normalized)
        or ("cuoi ky" in normalized and "dau ky" in normalized)
        or ("nam nay" in normalized and "nam truoc" in normalized)
    )

    sum_operand_roles = _explicit_sum_operand_roles(question)
    addition_role_hypotheses = _addition_role_hypotheses(normalized)
    addition_roles = _explicit_addition_roles(normalized)
    symmetric_difference_roles = _symmetric_difference_roles(normalized)
    difference_roles = (
        _comparison_difference_roles(normalized)
        or _explicit_subtraction_roles(normalized)
        or _relative_difference_roles(normalized)
        or symmetric_difference_roles
    )
    coefficient_roles = _coefficient_roles(normalized)
    par_value_roles = _par_value_roles(normalized)
    ratio_roles = _ratio_roles(normalized)
    multiplication_roles = _multiplication_roles(normalized)
    role_request_text, role_context_text = _partition_column_role_context(normalized)
    lexical_column_roles = _compile_column_roles(role_request_text)
    contextual_column_roles = _compile_column_roles(role_context_text)
    column_roles = lexical_column_roles
    direct_metric_lookup = coefficient_roles is not None and not requested_value_column
    explicit_plan_actual_pair = (
        len(lexical_column_roles) == 2
        and set(lexical_column_roles) == {"ACTUAL", "PLAN"}
    )
    shortfall_percent = (
        explicit_plan_actual_pair
        and asks_percent
        and any(marker in normalized for marker in ("con thieu", "chua dat"))
    )
    excess_percent = (
        explicit_plan_actual_pair and asks_percent and "vuot" in normalized
    )
    explicit_role_subtract = (
        explicit_plan_actual_pair and " tru " in padded
    )
    excess_amount = (
        explicit_plan_actual_pair
        and not asks_percent
        and "vuot" in normalized
    )

    operation: str | None = None
    if requested_value_column:
        operation = "SELECT"
    elif len(sum_operand_roles) >= 3 and not asks_percent:
        operation = "SUM_N"
    elif shortfall_percent or excess_percent:
        # These are multi-step ratios: (target-observed)/base or
        # (observed-target)/base. QP2B identifies the root operation and
        # direction; QP2C will materialize the SUBTRACT->RATIO_PERCENT DAG.
        operation = "RATIO_PERCENT"
    elif explicit_role_subtract or excess_amount:
        operation = "SUBTRACT"
    elif difference_roles is not None:
        operation = "SUBTRACT"
    elif (
        addition_role_hypotheses
        and not asks_percent
        and (
            normalized.startswith("cong ")
            or normalized.startswith("tinh cong ")
            or " co tong gia tri " in f" {normalized} "
            or len(addition_role_hypotheses) > 1
            or any(
                _has_financial_role_anchor(left)
                and _has_financial_role_anchor(right)
                for left, right in addition_role_hypotheses
            )
        )
    ):
        operation = "ADD"
    elif asks_percent and change_wording:
        operation = "PERCENT_CHANGE"
    elif temporal_change:
        # ``bình quân`` can legitimately occur inside a metric name, for
        # example ``Doanh thu/Tổng tài sản bình quân``.  The leading
        # arithmetic request is still a signed difference, so the explicit
        # temporal-change cue must take precedence over embedded metric wording.
        operation = "SUBTRACT"
    elif any(
        phrase in normalized
        for phrase in (
            "cao hon",
            "thap hon",
            "lon hon",
            "nho hon",
            "it hon",
            "nhieu hon",
        )
    ):
        operation = "SUBTRACT"
    elif (
        ratio_roles is not None
        and " bang bao nhieu lan " in padded
    ):
        operation = "DIVIDE"
    elif ratio_roles is not None and asks_percent:
        operation = "RATIO_PERCENT"
    elif (
        "ty le" in normalized
        and " tren " in padded
        or " chia cho " in padded
        and asks_percent
    ):
        operation = "RATIO_PERCENT"
    elif " chia cho " in padded:
        operation = "DIVIDE"
    elif any(term in normalized for term in ("binh quan", "trung binh")) and (
        " giua " in padded
        or "cuoi nam" in normalized
        and "dau nam" in normalized
        or "cuoi ky" in normalized
        and "dau ky" in normalized
    ):
        operation = "AVERAGE"
    elif any(
        phrase in padded
        for phrase in (
            " nhan voi ",
            " nhan cho ",
            " tich cua ",
            " tich giua ",
            " tinh tich ",
        )
    ):
        operation = "MULTIPLY"
    elif par_value_roles is not None:
        operation = "MULTIPLY"
    elif (
        asks_percent
        and len(column_roles) == 2
        and len(set(column_roles)) == 1
        and len(years) == 2
        and len(set(years)) == 2
    ):
        operation = 'RATIO_PERCENT'
    elif column_roles and len(set(column_roles)) == 1:
        # An explicit plan/actual/forecast lookup is a typed SELECT request,
        # not an unconstrained neural operation hypothesis.
        operation = "SELECT"

    if (
        len(column_roles) == 2
        and set(column_roles) == {"ACTUAL", "PLAN"}
        and (
            "hoan thanh" in normalized
            or "dat bao nhieu phan tram" in normalized
            or "ty le thuc hien" in normalized
            or "thuc hien so voi ke hoach" in normalized
        )
        and asks_percent
    ):
        operation = "RATIO_PERCENT"
        column_roles = ("ACTUAL", "PLAN")

    column_roles = _align_column_roles(
        column_roles,
        operation=operation,
        requested_years=years,
        operand_count=len(sum_operand_roles) or None,
    )

    if _same_metric_period_ratio(normalized) and len(column_roles)==1:
        # One requested role governs both explicitly named periods.
        column_roles = column_roles * 2

    if direct_metric_lookup:
        # A disclosed financial coefficient can be read directly from a row or
        # recomputed from its numerator and denominator. Keep both hypotheses
        # so the verifier can prefer explicit evidence without forbidding a
        # valid calculation on a different report layout.
        operation = None
        operation_candidates = ("SELECT", "DIVIDE")
        operand_roles = coefficient_roles or ()
    else:
        operation_candidates = (operation,) if operation else ()
        if operation == "SUBTRACT":
            operand_roles = difference_roles or addition_roles or ()
        elif operation == "ADD":
            operand_roles = addition_roles or difference_roles or ()
        elif operation == "SUM_N":
            operand_roles = sum_operand_roles
        elif operation == "DIVIDE":
            operand_roles = ratio_roles or coefficient_roles or ()
        else:
            operand_roles = (
                addition_roles
                or difference_roles
                or par_value_roles
                or ()
            )
    # A public ACTUAL default applies to both operands of an explicit
    # financial ratio. Preserve their two lexical metric identities.
    if (operation in {"DIVIDE", "RATIO_PERCENT"}
            and ratio_roles is not None and not operand_roles):
        operand_roles = ratio_roles
    proof_operand_roles = (
        operand_roles or ratio_roles or multiplication_roles or ()
    )

    from_year: str | None = None
    to_year: str | None = None
    direction = re.search(
        r"\btu(?:\s+nam)?\s+((?:19|20)\d{2})\s+"
        r"(?:(?:vnd|dong|nghin\s+dong|trieu\s+dong|ty\s+dong)\s+)?"
        r"(?:sang|den)(?:\s+nam)?\s+((?:19|20)\d{2})\b",
        normalized,
    )
    if direction:
        from_year, to_year = direction.groups()
    direction_intent = _compile_direction_intent(
        normalized,
        operation=operation,
        temporal_change=temporal_change,
        from_year=from_year,
        to_year=to_year,
        operand_roles=operand_roles,
        column_roles=column_roles,
    )

    entity_text = ""
    entity_match = re.search(
        r"\bcua\s+((?:(?:tong\s+)?cong ty|tap doan|don vi)\b.*?)"
        r"(?=\s+(?:nam|tu nam|giua nam|tai nam|tai ngay|vao ngay|"
        r"tai thoi diem|vao thoi diem|thay doi|tang|giam|"
        r"chenh lech|chiem|la bao nhieu)\b|[?.!,]|$)",
        normalized,
    )
    if entity_match:
        entity_text = entity_match.group(1).strip()
    parent_scope_match = re.search(
        r"\b(?:cua\s+)?(?:cong ty me|"
        r"bao cao(?: tai chinh)? rieng(?: cong ty)?|rieng cong ty)\b",
        normalized,
    )
    from .table_semantics import accounting_scope
    # Separate accounts describe an accounting perimeter of the issuer.
    # An explicit request for its legal parent remains a named-entity query.
    if parent_scope_match and accounting_scope(normalized) not in {"separate", "conflict"}:
        entity_text = "parent_company"
    elif accounting_scope(normalized) in {"separate", "conflict"} and entity_text in {"cong ty me", "cong ty", "don vi"}:
        entity_text = ""
    entity_terms = tuple(
        token
        for token in re.findall(r"[a-z0-9]+", entity_text)
        if len(token) >= 3 and token not in _ENTITY_STOPWORDS
    )

    event_initial=bool(re.search(r"\b(?:lan dau|ban dau|bat dau|chuyen doi loai hinh|chuyen thanh cong ty co phan)\b",normalized))
    event_change=bool(re.search(r"\b(?:phat hanh|tang len|von dieu le moi)\b",normalized))
    requested_event="initial" if event_initial else "capital_change" if event_change and "von dieu le" in normalized else ""
    period_ratio = _same_metric_period_ratio(normalized)
    metric_input=period_ratio[1] if period_ratio else normalized
    for date_token in re.findall(r"\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b",question):
        metric_input=re.sub(r"(?:(?:tai|den|sau)\s+)?(?:(?:ngay|thoi diem)\s+)?"+re.escape(normalize_text(date_token)), " ",metric_input)
    metric_text = re.sub(
        r"\b(?:hay\s+)?(?:tinh\s+)?(?:thuc hien\s+)?(?:nam\s+)?(?:19|20)\d{2}\s+(?:tru|minus)\s+(?:nam\s+)?(?:19|20)\d{2}\b",
        " ", metric_input,
    )
    # Scope remains in normalized question and its independent obligation.
    metric_text = re.sub(r"\b(?:(?:(?:theo|trong|tu)\s+)?(?:bao cao(?: tai chinh)? rieng(?: cong ty me)?|bctc rieng)|(?:cua\s+)?toan he thong|hop nhat|rieng le|rieng cong ty me)\b", " ", metric_text)
    if role_context_text:
        metric_text = metric_text.replace(role_context_text, " ")
    if entity_match:
        metric_text = (
            metric_text.replace(entity_match.group(0), " ", 1)
        )
    if parent_scope_match:
        metric_text = (
            metric_text.replace(parent_scope_match.group(0), " ", 1)
        )
    metric_text = normalize_metric_text(metric_text)
    metric_text = _strip_nonsemantic_metric_surface(metric_text)
    metric_text = _strip_calendar_period_surface(_strip_relative_period_surface(metric_text))
    metric_text = re.sub(
        r"\b(?:tai|vao)\s+ngay\s+\d{1,2}\s+thang\s+\d{1,2}\s+"
        r"nam\s+(?:19|20)\d{2}\b",
        " ",
        metric_text,
    )
    metric_text = re.sub(r"\b(?:19|20)\d{2}\b", " ", metric_text)
    metric_text = re.sub(
        r"\b(?:nghin\s+dong|trieu\s+dong|ty\s+dong|vnd)\b|"
        r"(?<!hoat )(?<!co )\bdong\b",
        " ",
        metric_text,
    )
    metric_text = re.sub(r"\b(?:cua\s+)?toan\s+cong\s+ty\b", " ", metric_text)
    metric_text = re.sub(r"\b(?:theo\s+)?don\s+vi\b", " ", metric_text)
    metric_text = re.sub(
        r"\b(?:la bao nhieu|bao nhieu|phan tram|tu nam|den nam|sang nam|"
        r"giua nam|thay doi|chenh lech|muc tang|muc giam|so sanh)\b",
        " ",
        metric_text,
    )
    metric_text = re.sub(r"\bnam\b", " ", metric_text)
    if column_roles:
        metric_text = re.sub(
            r"\b(?:theo\s+)?(?:ke hoach dieu chinh|ke hoach sua doi|revised plan|"
            r"adjusted plan|thuc hien|thuc te|actual|ke hoach|plan|"
            r"du bao|du kien|uoc tinh|forecast|kh|th)\b",
            " ",
            metric_text,
        )
        metric_text = re.sub(
            r"\b(?:hoan thanh|dat duoc|dat|ket qua)\b", " ", metric_text
        )
    # Output units and a requested polarity description are not metric
    # qualifiers. Preserve the source-independent year and unit obligations.
    metric_text = re.sub(r"\bduong hay am\b|\blan\b", " ", metric_text)
    metric_text = re.sub(r"\s+ngay\s*$", " ", metric_text)
    metric_text = re.sub(r"\s+", " ", metric_text).strip(" ?.,")
    if temporal_change and len(set(years)) >= 2:
        metric_text = re.sub(
            r"\b(?:so\s+voi|va|voi|giua)\b",
            " ",
            metric_text,
        )
        metric_text = re.sub(r"\s+", " ", metric_text).strip(" ?.,")
        tokens = metric_text.split()
        if (
            len(tokens) >= 2
            and len(tokens) % 2 == 0
            and tokens[: len(tokens) // 2] == tokens[len(tokens) // 2 :]
        ):
            metric_text = " ".join(tokens[: len(tokens) // 2])
    # Section framing is a separate query obligation, not part of the row name.
    if re.match(r"^(?:theo|dua tren|can cu|doi voi|trong pham vi)\b", metric_text):
        metric_text = _clean_operand_metric(metric_text, operation=operation)
    metric_text = re.sub(r"\s+(?:giua|va)(?:\s+(?:giua|va))*$", "", metric_text).strip()
    metric_text = re.sub(r"^(?:(?:cho toi(?: biet| so)?|muc|so)\s+)+", "", metric_text)
    metric_text = re.sub(r"\s+(?:duoc (?:cong ty )?ghi nhan|duoc (?:cong ty )?ghi|duoc neu|la|dat)$", "", metric_text).strip()
    if temporal_change:
        metric_text = re.sub(r"\s+(?:tang|giam)$", "", metric_text)
    if requested_event:
        metric_text=re.sub(r"\b(?:khi|luc|sau|den|tai)(?:\s+ngay)?\b", " ",metric_text)
        metric_text=re.sub(r"\b(?:chuyen thanh cong ty co phan|chuyen doi loai hinh|cong ty dang ky lan dau|dot phat hanh tra co tuc|ban dau|bat dau voi muc|tang len|moi|ngay)\b", " ",metric_text)
        metric_text=re.sub(r"\s+", " ",metric_text).strip()
    else:
        metric_text=re.sub(r"\b(?:tai thoi diem|tai ngay|den ngay|tai|den)\s*$", " ",metric_text).strip()
    from .metric_semantics import metric_span
    metric_text = metric_span(metric_text)
    if period_ratio:
        # A repeated period references the same metric, not a row named "năm".
        operand_roles = (metric_text, metric_text)
    identity_metric_text = metric_text
    for parenthetical in re.findall(r"\(([^)]*)\)", question):
        normalized_parenthetical = normalize_text(parenthetical).strip()
        if normalized_parenthetical:
            identity_metric_text = identity_metric_text.replace(
                normalized_parenthetical, " "
            )
    identity_metric_text = re.split(
        r"\b(?:hay\s+liet\s+ke|va\s+cho\s+biet|neu\s+|de\s+kiem\s+tra|"
        r"bao\s+gom|gom|trong\s+do|chi\s+tiet|duoc\s+cau\s+thanh)\b",
        identity_metric_text,
        maxsplit=1,
    )[0]
    identity_metric_text = re.sub(r"\s+", " ", identity_metric_text).strip()
    financial_identity_ids = tuple(
        identity.identity_id
        for identity in match_financial_identities(identity_metric_text)
    )

    # Conversational subject framing is not part of a financial row label.
    # Only remove the complete generic subject construction; a named company
    # remains represented by entity_mode/entity_text and its own obligation.
    from .unit_defaults import monetary_label
    generic_subject = re.fullmatch(r"cong ty co (.+)", metric_text)
    if generic_subject and monetary_label(generic_subject[1]):
        metric_text = generic_subject[1]

    # A single factual metric and one explicit year have an unambiguous scalar
    # operation even when no ACTUAL/PLAN role is named. QP2A slots deliberately
    # remain empty; the verifier's role-agnostic mapping checks source identity.
    if requested_event=="capital_change" and operation is None and len(set(years))==1 and not asks_percent:
        operation="SELECT"
        operation_candidates=("SELECT",)
    residual_answer_mode, residual_output_count = _compile_answer_mode(normalized)
    residual_scalar_fact = (
        operation is None
        and residual_answer_mode == "SINGLE_SCALAR"
        and residual_output_count == 1
        and question.count("?") <= 1
        and bool(metric_text)
        and (
            len(set(years)) == 1
            and len(set(requested_period_roles)) <= 1
            or not years
            and len(set(requested_period_roles)) == 1
            or not years
            and not requested_period_roles
            and monetary_label(metric_text)
            and not re.search(r"\b(?:tai sao|vi sao|phan tich|giai thich|du doan|nen|co nen)\b", normalized)
        )
        and not asks_percent
        and not change_wording
        and not temporal_change
        and not direct_metric_lookup
        and not (
            financial_identity_ids
            and requests_financial_identity_computation(normalized)
        )
        and difference_roles is None
        and not addition_role_hypotheses
        and ratio_roles is None
        and multiplication_roles is None
        and par_value_roles is None
        and not _has_compound_financial_metric(metric_text)
    )
    if residual_scalar_fact:
        operation = "SELECT"
        operation_candidates = ("SELECT",)
    # A named financial ratio can be read directly from a source table. An
    # explicit percentage request alone does not imply a two-input computation.
    from .metric_semantics import identity as metric_identity
    if (operation is None and metric_identity(metric_text) in {
            "return_on_assets", "return_on_equity", "current_ratio",
            "quick_ratio", "gross_margin", "net_margin",
        } and len(set(years)) == 1 and not temporal_change
        and not change_wording and not ratio_roles and not difference_roles):
        operation = "SELECT"
        operation_candidates = ("SELECT",)

    entity_mode = "named" if entity_terms else "reporting"
    operand_period_roles = _ordered_operand_period_roles(
        requested_period_roles,
        temporal_change=temporal_change,
        direction_intent=direction_intent,
    )
    if (operation=='PERCENT_CHANGE' and temporal_change
            and len(years)==1 and set(requested_period_roles)=={'opening','closing'}):
        operand_period_roles=('closing','opening')
    operand_years=((from_year, to_year) if direction_intent is DirectionIntent.DECREASE_AMOUNT else (to_year, from_year))         if from_year and to_year and operation in {"SUBTRACT", "PERCENT_CHANGE"} else years
    # Reordering opening/closing operands must move their explicitly attached
    # calendar years too. Never zip reordered endpoints with lexical years.
    if (len(years)==2 and len(requested_period_roles)==2
            and set(requested_period_roles)=={'opening','closing'}
            and operation in {'SUBTRACT','PERCENT_CHANGE'}):
        named_endpoints=re.findall(r'\b(dau|cuoi) (?:nam|ky) ((?:19|20)\d{2})\b',normalized)
        if len(named_endpoints)==2 and len({name for name,year in named_endpoints})==2:
            endpoint_years={('opening' if name=='dau' else 'closing'):year for name,year in named_endpoints}
            if operation=='PERCENT_CHANGE':operand_period_roles=('closing','opening')
            operand_years=tuple(endpoint_years[role] for role in operand_period_roles)
    operand_slots = _compile_operand_slots(
        normalized,
        metric_text=metric_text,
        operation=operation,
        requested_years=operand_years,
        requested_period_roles=operand_period_roles,
        # Typed slots preserve distinct roles in lexical operand order.  When
        # one explicit role governs multiple periods, consume the compiler's
        # aligned repetition so every year still receives a typed slot.
        column_roles=(
            lexical_column_roles
            if len(lexical_column_roles) > 1
            else column_roles
        ),
        operand_roles=(
            operand_roles
            if column_roles
            else operand_roles or ratio_roles or multiplication_roles or ()
        ),
        requested_unit_family=requested_unit_family,
        entity_mode=entity_mode,
        entity_text=entity_text,
    )
    if requested_value_column:
        operand_slots = tuple(replace(slot, unit_family=requested_unit_family or "currency_or_source") for slot in operand_slots)
    operand_ids = {slot.id for slot in operand_slots}
    program_identity_ids = (
        ("balance.financial_debt",)
        if {
        "short_term_borrowings",
        "long_term_borrowings",
        "total_capital",
        }.issubset(operand_ids)
        else ()
    )
    reference_base, program_dag = _compile_program_intent_dag(
        normalized,
        operation=operation,
        direction_intent=direction_intent,
        operand_slots=operand_slots,
        requested_unit_family=requested_unit_family,
    )
    if _program_intent_dag_errors(
        operand_slots,
        program_dag,
        reference_base=reference_base,
        root_operation=operation,
    ):
        # Fail closed: an invalid compiler DAG must never become a runtime
        # proposal. The flat legacy fields remain available for audit.
        reference_base, program_dag = None, ()
    answer_mode, requested_output_count = _compile_answer_mode(normalized)
    abstention_reason = _compile_abstention_reason(
        normalized,
        asks_percent=asks_percent,
        operation=operation,
        direction_intent=direction_intent,
        reference_base=reference_base,
        reference_base_named=bool(ratio_roles or coefficient_roles),
        requested_column_roles=lexical_column_roles,
        contextual_column_roles=contextual_column_roles,
    )
    if not abstention_reason and requested_scale == "conflict":
        abstention_reason = "conflicting_requested_scale"
    elif (
        not abstention_reason
        and requested_scale is not None
        and requested_unit_family in {"percent", "ratio"}
    ):
        abstention_reason = "incompatible_requested_scale"
    if not abstention_reason and requested_output_count > 1:
        abstention_reason = "unsupported_multi_output_request"

    return QuestionPlan(
        question=question,
        normalized=normalized,
        requested_event=requested_event,
        operation=operation,
        metric_text=metric_text,
        requested_years=years,
        from_year=from_year,
        to_year=to_year,
        entity_mode=entity_mode,
        entity_text=entity_text,
        entity_terms=entity_terms,
        asks_percent=asks_percent,
        requested_unit_family=requested_unit_family,
        requested_scale=requested_scale,
        temporal_change=temporal_change,
        operation_candidates=operation_candidates,
        operand_roles=operand_roles,
        direct_metric_lookup=direct_metric_lookup,
        financial_identity_ids=financial_identity_ids,
        proof_operand_roles=proof_operand_roles,
        direction_intent=direction_intent,
        requested_period_roles=requested_period_roles,
        column_roles=column_roles,
        addition_role_hypotheses=addition_role_hypotheses,
        operand_slots=operand_slots,
        comparison_target=(comparison_target_base(normalized,operand_slots) or (None,None))[0],
        comparison_base=(comparison_target_base(normalized,operand_slots) or (None,None))[1],
        reference_base=reference_base,
        program_dag=program_dag,
        program_identity_ids=program_identity_ids,
        abstention_reason=abstention_reason,
        answer_mode=answer_mode,
        requested_output_count=requested_output_count,
        requested_value_column=requested_value_column,
    )


_PROGRAM_INTENT_ARITY = {
    "SELECT": 1,
    "ADD": 2,
    "SUBTRACT": 2,
    "MULTIPLY": 2,
    "DIVIDE": 2,
    "RATIO_PERCENT": 2,
    "PERCENT_CHANGE": 2,
    "AVERAGE": 2,
    "SUM3": 3,
    "SUM4": 4,
}


def _program_intent_arity_valid(operation: str, operand_count: int) -> bool:
    if operation == "SUM_N":
        return 2 <= operand_count <= 8
    arity = _PROGRAM_INTENT_ARITY.get(operation)
    return arity is not None and operand_count == arity


def _program_node(
    node_id: str,
    operation: str,
    inputs: tuple[str, ...],
    output_unit_family: str,
    *sources: str,
) -> ProgramIntentNode:
    return ProgramIntentNode(
        id=node_id,
        operation=operation,
        inputs=inputs,
        output_unit_family=output_unit_family,
        intent_sources=("operation:question_compiler", "inputs:operand_slot_ids") + sources,
    )


def comparison_target_base(normalized, slots):
    """Question-side identities; never infer target/base from values or rank."""
    if len(slots)!=2 or len({s.metric for s in slots})!=1 or normalized.count('so voi')!=1:return None
    def select(part):
        years=set(re.findall(r'\b(?:19|20)\d{2}\b',part))
        roles=set()
        if 'ke hoach' in part:roles.add('PLAN')
        if 'thuc hien' in part or 'thuc te' in part:roles.add('ACTUAL')
        periods=set(_compile_requested_period_roles(part))
        if len(years)>1 or len(roles)>1 or len(periods)>1 or not (years or roles or periods):return None
        found=[s.id for s in slots if (not years or s.year in years) and (not roles or s.column_role in roles) and (not periods or s.period_role in periods)]
        return found[0] if len(found)==1 else None
    left,right=normalized.split('so voi');target,base=select(left),select(right)
    return (target,base) if target and base and target!=base else None


def _compile_program_intent_dag(
    normalized: str,
    *,
    operation: str | None,
    direction_intent: DirectionIntent | None,
    operand_slots: tuple[OperandSlot, ...],
    requested_unit_family: str,
) -> tuple[str | None, tuple[ProgramIntentNode, ...]]:
    """Compile reference base and topology without binding evidence candidates."""

    if operation is None or not operand_slots:
        return None, ()
    slot_ids = tuple(slot.id for slot in operand_slots)
    slot_id_set = set(slot_ids)
    debt_ids = {
        "short_term_borrowings",
        "long_term_borrowings",
        "total_capital",
    }
    if operation == "RATIO_PERCENT" and debt_ids.issubset(slot_id_set):
        nodes = (
            _program_node(
                "interest_bearing_debt",
                "ADD",
                ("short_term_borrowings", "long_term_borrowings"),
                "currency",
                "identity:balance.financial_debt",
            ),
            _program_node(
                "answer",
                "RATIO_PERCENT",
                ("interest_bearing_debt", "total_capital"),
                "percent",
                "reference_base:total_capital",
            ),
        )
        return "total_capital", nodes

    if (
        operation == "RATIO_PERCENT"
        and len(slot_ids) == 2
        and direction_intent in {
            DirectionIntent.PLAN_MINUS_ACTUAL,
            DirectionIntent.ACTUAL_MINUS_PLAN,
        }
    ):
        is_shortfall = any(marker in normalized for marker in ("con thieu", "chua dat"))
        intermediate_id = "shortfall" if is_shortfall else "excess"
        reference_base = slot_ids[0] if is_shortfall else slot_ids[1]
        nodes = (
            _program_node(
                intermediate_id,
                "SUBTRACT",
                (slot_ids[0], slot_ids[1]),
                "currency",
                f"direction:{direction_intent.value}",
            ),
            _program_node(
                "answer",
                "RATIO_PERCENT",
                (intermediate_id, reference_base),
                "percent",
                f"reference_base:{reference_base}",
            ),
        )
        return reference_base, nodes

    if not _program_intent_arity_valid(operation, len(slot_ids)):
        return None, ()
    comparison=comparison_target_base(normalized,operand_slots)
    if comparison and operation=='SUBTRACT' and direction_intent is DirectionIntent.DECREASE_AMOUNT:
        slot_ids=(comparison[1],comparison[0])
    reference_base = (
        slot_ids[1]
        if operation in {"DIVIDE", "RATIO_PERCENT", "PERCENT_CHANGE"}
        else None
    )
    if operation == "SELECT":
        output_unit = operand_slots[0].unit_family
    elif operation in {"RATIO_PERCENT", "PERCENT_CHANGE"}:
        output_unit = "percent"
    elif operation == "DIVIDE":
        output_unit = requested_unit_family or "ratio"
    elif operation == "MULTIPLY" and {
        slot.unit_family for slot in operand_slots
    } == {"share_count", "currency_per_share"}:
        output_unit = "currency"
    else:
        output_unit = requested_unit_family or operand_slots[0].unit_family
    sources = (f"reference_base:{reference_base}",) if reference_base else ()
    return reference_base, (
        _program_node("answer", operation, slot_ids, output_unit, *sources),
    )


def _program_intent_dag_errors(
    operand_slots: tuple[OperandSlot, ...],
    nodes: tuple[ProgramIntentNode, ...],
    *,
    reference_base: str | None,
    root_operation: str | None,
) -> tuple[str, ...]:
    """Validate topology, arity, reference base and basic unit algebra."""

    if not nodes:
        return ()
    errors: list[str] = []
    operand_ids = [slot.id for slot in operand_slots]
    if len(operand_ids) != len(set(operand_ids)):
        errors.append("duplicate_operand_id")
    available = set(operand_ids)
    units = {slot.id: slot.unit_family for slot in operand_slots}
    node_ids: list[str] = []
    for node in nodes:
        if not node.id or node.id in available:
            errors.append(f"duplicate_node_id:{node.id}")
        arity = _PROGRAM_INTENT_ARITY.get(node.operation)
        if node.operation == "SUM_N":
            if not 2 <= len(node.inputs) <= 8:
                errors.append(f"wrong_arity:{node.id}")
        elif arity is None:
            errors.append(f"unsupported_operation:{node.operation}")
        elif len(node.inputs) != arity:
            errors.append(f"wrong_arity:{node.id}")
        if any(item not in available for item in node.inputs):
            errors.append(f"unresolved_or_non_topological_input:{node.id}")
        input_units = [units.get(item, "") for item in node.inputs]
        if node.operation in {"ADD", "SUBTRACT", "AVERAGE", "SUM_N"}:
            known = {unit for unit in input_units if unit}
            if len(known) > 1 and known != {"currency", "currency_or_source"}:
                errors.append(f"incompatible_additive_units:{node.id}")
            if node.output_unit_family not in {"currency", "currency_or_source"}:
                errors.append(f"wrong_additive_output_unit:{node.id}")
        if node.operation in {"RATIO_PERCENT", "PERCENT_CHANGE"}:
            if node.output_unit_family != "percent":
                errors.append(f"wrong_percent_output_unit:{node.id}")
            if len(input_units) == 2 and not all(input_units):
                errors.append(f"unknown_ratio_input_unit:{node.id}")
        available.add(node.id)
        units[node.id] = node.output_unit_family
        node_ids.append(node.id)
    if node_ids[-1:] != ["answer"]:
        errors.append("final_node_is_not_answer")
    if nodes[-1].operation != root_operation:
        errors.append("root_operation_mismatch")
    if reference_base is not None:
        if reference_base not in operand_ids:
            errors.append("reference_base_is_not_operand")
        if len(nodes[-1].inputs) < 2 or nodes[-1].inputs[1] != reference_base:
            errors.append("reference_base_is_not_final_denominator")
    elif nodes[-1].operation in {"DIVIDE", "RATIO_PERCENT", "PERCENT_CHANGE"}:
        errors.append("missing_reference_base")
    return tuple(dict.fromkeys(errors))


def _clean_operand_metric(
    metric_text: str, *, operation: str | None = None
) -> str:
    """Remove arithmetic wording while retaining the requested row metric."""

    cleaned = _strip_calendar_period_surface(_strip_relative_period_surface(
        _strip_output_unit_directives(normalize_metric_text(metric_text))
    ))
    if " hay xac dinh " in f" {cleaned} ":
        cleaned = cleaned.rsplit("hay xac dinh", 1)[-1].strip()
    cleaned = re.sub(
        r"^(?:hay cho biet|theo bao cao|dua tren so lieu duoc cong bo|"
        r"doi voi cung chi tieu va pham vi bao cao)\s+",
        "",
        cleaned,
    )
    anchor_starts = [
        match.start()
        for anchor in _FINANCIAL_ROLE_ANCHORS
        for match in re.finditer(rf"\b{re.escape(anchor)}\b", cleaned)
    ]
    if anchor_starts:
        core_start = min(anchor_starts)
        total_prefix = re.search(
            r"\b(?:tong|tong cong)\s+$",
            cleaned[:core_start],
        )
        if total_prefix is not None:
            core_start = total_prefix.start()
        prefix = cleaned[:core_start].strip()
        if any(
            prefix.startswith(marker)
            for marker in (
                "theo ",
                "dua tren ",
                "can cu ",
                "doi voi ",
                "trong pham vi ",
                "so voi ",
                "thi ",
            )
        ):
            cleaned = cleaned[core_start:]
    cleaned = re.split(
        r"\s+(?:tru|thap hon|cao hon|nho hon|it hon|nhieu hon|vuot|"
        r"con thieu|chua dat|thay doi|tang|giam|so voi)\b",
        cleaned,
        maxsplit=1,
    )[0]
    if operation in {"DIVIDE", "RATIO_PERCENT", "PERCENT_CHANGE"}:
        cleaned = re.sub(
            r"\s+bang(?:\s+muc)?(?:\s+cung)?\s*$",
            "",
            cleaned,
        )
    if operation == "AVERAGE":
        cleaned = re.sub(r"\b(?:binh\s+quan|trung\s+binh|giua)\b", " ", cleaned)
        cleaned = re.sub(r"(?:\s+va\s*)+$", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip(" ?.,")


def _slot_unit_family(operation: str | None, requested_unit_family: str, metric: str = "") -> str:
    # The requested answer may be a percent/ratio while its leaf operands are
    # monetary values. A direct SELECT can preserve an explicitly requested
    # unit or defer to the source cell.
    if operation == "SELECT":
        return requested_unit_family or metric_unit_family(metric) or "currency_or_source"
    return "currency"


def _slot_id(
    column_role: str,
    year: str,
    period_role: str,
    *,
    multi_period: bool,
) -> str:
    base = column_role.lower()
    period_key = "_".join(item for item in (year, period_role) if item)
    return f"{base}_{period_key}" if multi_period and period_key else base


def _align_slot_periods(
    requested_years: tuple[str, ...],
    requested_period_roles: tuple[str, ...],
    *,
    arity: int,
) -> tuple[tuple[str, str], ...]:
    """Align explicit calendar years and relative endpoints to leaf slots."""

    if arity < 1:
        return ()
    if requested_period_roles:
        if len(requested_period_roles) == 1:
            period_roles = requested_period_roles * arity
        elif len(requested_period_roles) == arity:
            period_roles = requested_period_roles
        else:
            return ()
    else:
        period_roles = ("",) * arity

    if requested_years:
        if len(requested_years) == 1:
            years = requested_years * arity
        elif len(requested_years) == arity:
            years = requested_years
        else:
            return ()
    elif requested_period_roles:
        years = ("",) * arity
    else:
        return ()
    return tuple(zip(years, period_roles))


def _compile_interest_bearing_debt_slots(
    normalized: str,
    *,
    requested_years: tuple[str, ...],
    requested_period_roles: tuple[str, ...],
    entity_scope: str,
) -> tuple[OperandSlot, ...]:
    """Compile the three leaves of the precommitted debt-share identity.

    This only materializes question intent. Identity activation and its
    ADD->RATIO_PERCENT DAG remain QP2C responsibilities.
    """

    padded = f" {normalized} "
    asks_share = " chiem bao nhieu phan tram " in padded
    has_short = " no vay ngan han " in padded or " vay ngan han " in padded
    has_long = " no vay dai han " in padded or " vay dai han " in padded
    has_capital = " tong nguon von " in padded
    period_roles = tuple(dict.fromkeys(requested_period_roles))
    if not (
        asks_share
        and has_short
        and has_long
        and has_capital
        and (requested_years or period_roles)
        and len(period_roles) <= 1
    ):
        return ()
    year = requested_years[0] if requested_years else ""
    period_role = period_roles[0] if period_roles else ""
    common_sources = (
        (
            "year:question_explicit"
            if year
            else "year:not_applicable_relative_period"
        ),
        (
            "period_role:question_explicit"
            if period_role
            else "period_role:unspecified"
        ),
        "column_role:financial_statement_actual_default",
        "entity:question_scope",
        "unit:financial_identity_leaf",
    )
    return (
        OperandSlot(
            id="short_term_borrowings",
            metric="vay va no thue tai chinh ngan han",
            year=year,
            column_role="ACTUAL",
            entity_scope=entity_scope,
            unit_family="currency",
            intent_sources=("metric:short_term_borrowing_lexical",) + common_sources,
            period_role=period_role,
        ),
        OperandSlot(
            id="long_term_borrowings",
            metric="vay va no thue tai chinh dai han",
            year=year,
            column_role="ACTUAL",
            entity_scope=entity_scope,
            unit_family="currency",
            intent_sources=("metric:long_term_borrowing_lexical",) + common_sources,
            period_role=period_role,
        ),
        OperandSlot(
            id="total_capital",
            metric="tong nguon von",
            year=year,
            column_role="ACTUAL",
            entity_scope=entity_scope,
            unit_family="currency",
            intent_sources=("metric:total_capital_lexical",) + common_sources,
            period_role=period_role,
        ),
    )


def _explicit_subtract_slots(normalized: str, requested_years: tuple[str, ...]) -> tuple[OperandSlot, ...] | None:
    """Compile each side of literal A minus B as an independent SELECT.

    None leaves other arithmetic grammars unchanged. Empty means an explicit
    subtraction was recognized but cannot be bound safely from the question.
    """
    parts = re.split(r"(?<!giam )(?<!khau )(?<!loai )\btru\b(?! so\b)", normalized)
    if len(parts) != 2:
        return None
    parts[0] = re.sub(r"^(?:(?:hay |vui long )?lay|tinh|hay tinh)\s+", "", parts[0]).strip()
    parts[1] = re.sub(r"\s+(?:(?:thi |se )?(?:duoc |la |bang )?)?bao nhieu.*$", "", parts[1]).strip(" ?.,")
    if any(re.search(r"\b(?:cong|chia|nhan|so voi)\b", part) for part in parts):
        return None
    if not parts[0] or not parts[1]:
        return ()
    # A minus a year/endpoint is the existing same-metric temporal grammar.
    if re.match(r"^(?:(?:dau|cuoi) (?:nam|ky)\s*)?(?:nam\s+)?(?:19|20)\d{2}(?:\s+(?:theo|trong|don vi)\b.*)?$", parts[1]) or re.fullmatch(r"(?:dau|cuoi) (?:nam|ky)",parts[1]):
        return None
    slots = []
    for index, part in enumerate(parts):
        if not part:
            return ()
        years = re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", part)
        if not years and len(set(requested_years)) == 1:
            part += " nam " + requested_years[0]
        lookup = compile_question_plan(resolve_public_question(part + " la bao nhieu?"))
        if lookup.operation != "SELECT" or lookup.abstention_reason or len(lookup.operand_slots) != 1:
            return None
        slot = lookup.operand_slots[0]
        if not slot.metric or not (slot.year or slot.period_role):
            return ()
        slots.append(replace(slot, id=("subtract_left" if index == 0 else "subtract_right"),
                             intent_sources=slot.intent_sources + ("operand:explicit_subtract_order",)))
    # Fully stated operands retain their own years and subtraction order.
    # Elided year/endpoint operands are delegated above to the temporal grammar.
    return tuple(slots)


def _compile_operand_slots(
    normalized: str,
    *,
    metric_text: str,
    operation: str | None,
    requested_years: tuple[str, ...],
    requested_period_roles: tuple[str, ...],
    column_roles: tuple[str, ...],
    operand_roles: tuple[str, ...],
    requested_unit_family: str,
    entity_mode: str,
    entity_text: str,
) -> tuple[OperandSlot, ...]:
    """Compile document-independent operand identities for QP2A.

    An explicit role is mandatory for plan/actual slots. A missing role stays
    missing instead of being inferred from year, word order or a future table
    position. The sole default-role exception is a recognized financial-
    statement identity whose leaves are actual disclosed balances.
    """

    if operation == "SUBTRACT":
        explicit = _explicit_subtract_slots(normalized, requested_years)
        if explicit is not None:
            return explicit
    entity_scope = "reporting" if entity_mode == "reporting" else f"named:{entity_text}"
    identity_slots = _compile_interest_bearing_debt_slots(
        normalized,
        requested_years=requested_years,
        requested_period_roles=requested_period_roles,
        entity_scope=entity_scope,
    )
    if identity_slots:
        return identity_slots
    if not requested_years and not requested_period_roles:
        from .unit_defaults import monetary_label
        metric = _clean_operand_metric(metric_text, operation=operation)
        if (operation == "SELECT" and monetary_label(metric)
                and len(set(column_roles)) <= 1
                and not _has_compound_financial_metric(metric)):
            return (OperandSlot(
                id="lookup:" + re.sub(r"[^a-z0-9]+", "_", metric).strip("_"),
                metric=metric, year="", period_role="",
                column_role=column_roles[0] if column_roles else "ACTUAL",
                entity_scope=entity_scope,
                unit_family=requested_unit_family or "currency",
                intent_sources=("metric:question_lexical", "operation:single_fact_lookup",
                                "entity:question_scope", "time:not_requested"),
            ),)
        return ()
    if not column_roles:
        # Explicit X/Y division and enumerated sums still provide typed metric
        # identities when the source column role is intentionally unspecified.
        # Do not invent ACTUAL/PLAN: every non-conflicting physical lane stays
        # eligible, so a genuine role ambiguity remains AMBIGUOUS.
        existing_roleless_arity_valid = (
            operation in {"DIVIDE", "RATIO_PERCENT"}
            and len(operand_roles) == 2
            or operation == "MULTIPLY"
            and _par_value_roles(normalized) is not None
            and len(operand_roles) == 2
            or operation in {"ADD", "SUM3", "SUM4", "SUM_N"}
            and 2 <= len(operand_roles) <= 8
        )
        explicit_relative_binary = (
            bool(requested_period_roles)
            and operation
            in {
                "SUBTRACT",
                "MULTIPLY",
                "RATIO_PERCENT",
                "PERCENT_CHANGE",
                "AVERAGE",
            }
            and (
                len(operand_roles) == 2
                or operation in {"SUBTRACT", "PERCENT_CHANGE", "AVERAGE"}
                and len(set(requested_period_roles)) == 2
            )
        )
        explicit_relative_select = (
            operation == "SELECT"
            and len(set(requested_period_roles)) == 1
            and bool(metric_text)
        )
        if not (
            existing_roleless_arity_valid
            or explicit_relative_binary
            or explicit_relative_select
        ):
            return ()
        operand_metrics = tuple(
            _clean_operand_metric(value, operation=operation)
            for value in operand_roles
        )
        if explicit_relative_select:
            metrics = (_clean_operand_metric(metric_text, operation=operation),)
        elif len(operand_metrics) == 2 or existing_roleless_arity_valid:
            metrics = operand_metrics
        else:
            metric = _clean_operand_metric(metric_text, operation=operation)
            metrics = (metric, metric)
        if not all(metrics):
            return ()
        periods = _align_slot_periods(
            requested_years,
            requested_period_roles,
            arity=len(metrics),
        )
        if not periods:
            return ()
        # Par-value intent gives two different input dimensions. The currency
        # answer must not make the share-count operand a currency slot.
        par_value_units = (
            ("share_count", "currency_per_share")
            if operation == "MULTIPLY" and _par_value_roles(normalized) is not None
            else ()
        )
        return tuple(
            OperandSlot(
                id=(
                    "metric_"
                    + str(index + 1)
                    + ":"
                    + re.sub(r"[^a-z0-9]+", "_", metric).strip("_")
                    + ":"
                    + "_".join(item for item in (year, period_role) if item)
                ),
                metric=metric,
                year=year,
                column_role="",
                entity_scope=entity_scope,
                unit_family=par_value_units[index] if par_value_units else _slot_unit_family(operation, requested_unit_family, metric),
                intent_sources=(
                    "metric:question_operand_role",
                    (
                        "year:question_explicit"
                        if year
                        else "year:not_applicable_relative_period"
                    ),
                    (
                        "period_role:question_explicit"
                        if period_role
                        else "period_role:unspecified"
                    ),
                    "column_role:unspecified",
                    "entity:question_scope",
                    "unit:operation_contract",
                ),
                period_role=period_role,
            )
            for index, (metric, (year, period_role)) in enumerate(
                zip(metrics, periods)
            )
        )

    roles = column_roles
    if (len(roles) == 1 and len(requested_years) == 1
            and len(operand_roles) == 2
            and operation in {"DIVIDE", "RATIO_PERCENT"}
            and len(set(requested_period_roles)) <= 1):
        # One actual lane can supply both named metric leaves of a ratio.
        # Keep distinct metrics; never duplicate an evidence fact.
        roles = roles * 2
    # One actual/plan lane can provide both endpoints of the same named year.
    # Keep the endpoints as distinct operands, not two copies of one fact.
    if (len(roles)==1 and len(requested_years)==1
            and len(requested_period_roles)==2
            and set(requested_period_roles)=={'opening','closing'}
            and operation in {'SUBTRACT','PERCENT_CHANGE','AVERAGE'}
            and not operand_roles):
        roles=roles*2
    if requested_period_roles:
        periods = _align_slot_periods(
            requested_years,
            requested_period_roles,
            arity=len(roles),
        )
        if not periods:
            return ()
    else:
        years: tuple[str, ...]
        if len(requested_years) == 1:
            years = (requested_years[0],) * len(roles)
        elif len(roles) == len(requested_years):
            if (
                operation == 'RATIO_PERCENT'
                and len(roles) == 2
                and len(set(roles)) == 1
                and len(set(requested_years)) == 2
                and not _same_metric_period_ratio(normalized)
            ):
                years = tuple(sorted(requested_years, key=int, reverse=True))
            else:
                years = requested_years
        elif len(roles) == 1:
            # Do not duplicate a role unless the existing column-role compiler
            # has explicitly aligned it to every requested period.
            return ()
        else:
            return ()
        periods = tuple((year, "") for year in years)

    metric = _clean_operand_metric(metric_text, operation=operation)
    operand_metrics = tuple(
        _clean_operand_metric(value, operation=operation)
        for value in operand_roles
    )
    metrics = (
        operand_metrics
        if len(operand_metrics) == len(roles) and all(operand_metrics)
        else (metric,) * len(roles)
    )
    if not all(metrics):
        return ()
    multi_metric = len(set(metrics)) > 1
    multi_period = len(set(periods)) > 1
    role_source = "column_role:question_explicit"
    year_source = "year:question_explicit"
    slots = tuple(
        OperandSlot(
            id=(
                "metric_"
                + str(index + 1)
                + ":"
                + re.sub(r"[^a-z0-9]+", "_", slot_metric).strip("_")
                + ":"
                + role.lower()
                + ":"
                + "_".join(item for item in (year, period_role) if item)
                if multi_metric
                else _slot_id(
                    role,
                    year,
                    period_role,
                    multi_period=multi_period,
                )
            ),
            metric=slot_metric,
            year=year,
            column_role=role,
            entity_scope=entity_scope,
            unit_family=_slot_unit_family(operation, requested_unit_family, slot_metric),
            intent_sources=(
                (
                    "metric:question_operand_role"
                    if multi_metric
                    else "metric:question_lexical"
                ),
                year_source,
                (
                    "period_role:question_explicit"
                    if period_role
                    else "period_role:unspecified"
                ),
                role_source,
                "entity:question_scope",
                "unit:operation_contract",
            ),
            period_role=period_role,
        )
        for index, (role, (year, period_role), slot_metric) in enumerate(
            zip(roles, periods, metrics)
        )
    )
    # The slot list is an operand identity set. A multi-step DAG may reuse a
    # slot as a denominator later; QP2A must not duplicate that identity.
    unique: list[OperandSlot] = []
    seen: set[str] = set()
    for slot in slots:
        if slot.id in seen:
            continue
        seen.add(slot.id)
        unique.append(slot)
    return tuple(unique)


_COLUMN_ROLE_CONTEXT_START = re.compile(
    r"\b(?:(?:khi|neu|trong truong hop)\s+)?"
    r"(?:(?:trong|tai)\s+)?(?:bang|bao cao)\s+"
    r"(?:(?:co|gom|bao gom|chua|the hien)(?:\s+(?:ca|dong thoi))?|"
    r"dong thoi\s+co)\b"
)
_COLUMN_ROLE_CONTEXT_TOKEN = re.compile(
    r"\b(?:ke hoach dieu chinh|ke hoach sua doi|revised plan|adjusted plan|"
    r"du bao|uoc tinh|forecast|estimated?|thuc hien|thuc te|actual|realized|"
    r"realised|ket qua dat duoc|ke hoach|plan|muc tieu|target)\b"
)


def _partition_column_role_context(normalized: str) -> tuple[str, str]:
    """Separate table-role context from the role requested by the question.

    A phrase such as ``khi bảng có cả kế hoạch và thực hiện`` describes an
    ambiguity in the source layout; it does not request both values.  Keeping
    that context out of the requested-role parser prevents a missing role from
    being silently converted into a two-operand program.  The partition is
    entirely question-side and never inspects a document or candidate.
    """

    marker = _COLUMN_ROLE_CONTEXT_START.search(normalized)
    if marker is None:
        return normalized, ""
    role_matches = list(_COLUMN_ROLE_CONTEXT_TOKEN.finditer(normalized, marker.end()))
    distinct: list[re.Match[str]] = []
    distinct_roles: set[str] = set()
    for match in role_matches:
        role = _compile_column_roles(match.group(0))
        if not role or role[0] in distinct_roles:
            continue
        distinct_roles.add(role[0])
        distinct.append(match)
        if len(distinct_roles) == 2:
            break
    if len(distinct_roles) < 2:
        return normalized, ""

    prefix = normalized[: marker.start()].strip()
    if _has_financial_role_anchor(prefix):
        # The ordinary request precedes a trailing context clause.
        return prefix, normalized[marker.start() :].strip()

    # A leading context clause ends once both mutually exclusive roles have
    # been introduced. The remaining text carries any explicit target role.
    context_end = distinct[-1].end()
    return normalized[context_end:].strip(), normalized[marker.start() : context_end].strip()


def _percent_request_requires_reference_base(normalized: str, operation: str | None) -> bool:
    padded = f" {normalized} "
    comparative = any(
        marker in padded
        for marker in (
            " cao hon ",
            " thap hon ",
            " lon hon ",
            " nho hon ",
            " it hon ",
            " nhieu hon ",
            " tang bao nhieu ",
            " giam bao nhieu ",
            " thay doi bao nhieu ",
            " vuot ",
            " con thieu ",
            " chua dat ",
        )
    )
    ratio_request = any(
        marker in padded
        for marker in (
            " chiem bao nhieu phan tram ",
            " dat bao nhieu phan tram ",
            " hoan thanh bao nhieu phan tram ",
        )
    )
    return comparative or ratio_request or operation in {"RATIO_PERCENT", "PERCENT_CHANGE"}


def _question_names_distinct_reference_periods(normalized: str) -> bool:
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", normalized))
    if len(years) >= 2:
        return True
    return (
        "so voi" in normalized
        and (
            "cuoi nam" in normalized and "dau nam" in normalized
            or "cuoi ky" in normalized and "dau ky" in normalized
            or "nam nay" in normalized and "nam truoc" in normalized
        )
    )


def _compile_answer_mode(normalized: str) -> tuple[str, int]:
    """Detect explicit multi-output requests without splitting arithmetic leaves.

    Conjunctions such as ``A và B chiếm bao nhiêu phần trăm C`` still request
    one scalar and must not be treated as multiple answers.  Only explicit
    output-distribution language activates the all-or-abstain contract.
    """

    padded = f" {normalized} "
    explicit_multi = any(
        marker in padded
        for marker in (
            " hai ket qua ",
            " hai gia tri ",
            " tung ket qua ",
            " tung gia tri ",
            " lan luot la bao nhieu ",
            " tra loi rieng ",
            " liet ke cac gia tri ",
        )
    )
    if not explicit_multi:
        return "SINGLE_SCALAR", 1
    return "MULTI_OUTPUT_ALL_OR_ABSTAIN", 2


def _compile_abstention_reason(
    normalized: str,
    *,
    asks_percent: bool,
    operation: str | None,
    direction_intent: DirectionIntent | None,
    reference_base: str | None,
    reference_base_named: bool,
    requested_column_roles: tuple[str, ...],
    contextual_column_roles: tuple[str, ...],
) -> str:
    """Compile a fail-closed, evidence-independent underspecification reason."""

    # Alternative periods name a choice, not an ordered pair of operands.
    # Do not silently select either year or relabel the question as absent data.
    alternatives = re.search(
        r"\b((?:19|20)\d{2})\s+(?:hoac|or)\s+(?:(?:nam|year)\s+)?((?:19|20)\d{2})\b", normalized)
    if alternatives and alternatives.group(1) != alternatives.group(2):
        return "ambiguous_requested_period"
    padded = f" {normalized} "
    names_ambiguous_columns = any(
        phrase in padded
        for phrase in (
            " bang co nhieu cot ",
            " bao cao co nhieu cot ",
            " bang gom nhieu cot ",
            " bao cao gom nhieu cot ",
        )
    )
    explicitly_omits_role = any(
        phrase in padded
        for phrase in (
            " khong xac dinh vai tro cot ",
            " khong neu vai tro cot ",
            " khong chi ro vai tro cot ",
            " chua xac dinh vai tro cot ",
        )
    )
    if (
        re.search(r"(?<!\bbang )\bso sanh\b", normalized)
        and "bao nhieu" not in normalized
        and not any(
            marker in normalized
            for marker in (
                "chenh lech",
                "thay doi",
                "tang",
                "giam",
                "cao hon",
                "thap hon",
                "lon hon",
                "nho hon",
                "ty le",
                "phan tram",
            )
        )
    ):
        return "ambiguous_comparison_measure"
    if names_ambiguous_columns and explicitly_omits_role:
        return "ambiguous_metric_or_column_role"
    if (
        contextual_column_roles
        and len(set(contextual_column_roles)) >= 2
        and not requested_column_roles
    ):
        return "missing_column_role"
    if (
        asks_percent
        and reference_base is None
        and not reference_base_named
        and not _question_names_distinct_reference_periods(normalized)
        and _percent_request_requires_reference_base(normalized, operation)
    ):
        return "missing_reference_base"
    if direction_intent is DirectionIntent.AMBIGUOUS:
        return "ambiguous_signed_direction"
    return ""


def resolve_public_question(question: str) -> str:
    """Apply the public ACTUAL default to intent only; never relabel evidence.

    The lexical compiler and training inputs retain their existing semantics.
    Calling this twice is harmless because the resolved role is explicit.
    """
    question = normalize_question_framing(question)
    plan = compile_question_plan(question)
    normalized = normalize_text(question)
    if (plan.column_roles or not plan.requested_years
            or not _has_financial_role_anchor(normalized)
            or plan.abstention_reason not in {"", "missing_column_role"}
            or re.search(r"\b(?:nếu|neu)\b", question, re.I)
            or re.search(r"\b(?:gia su|se ghi nhan|chua ghi nhan|khong ghi nhan|if|will|would)\b", normalized)
            or re.search(r"\b(?:khong|chua) (?:xac dinh|neu|chi ro) vai tro cot\b", normalized)):
        return question
    start = question.find(",") + 1 if normalized.startswith("theo ") and "," in question else 0
    focus = question[start:]
    match = re.search(r"\b(?:năm|nam|year)\s+(?:19|20)\d{2}\b", focus, re.IGNORECASE)
    if match is None:
        # A bare calendar year in a financial lookup has the same public
        # actual default as "năm Y". A date fragment is not rewritten.
        match = next((m for m in re.finditer(r"(?<![/\d])\b(?:19|20)\d{2}\b", focus)
                      if m[0] in plan.requested_years), None)
    if match is None:
        return question
    offset = start + match.start()
    # Insert ACTUAL before the complete temporal phrase, never between
    # "cuối" and "năm"; the endpoint parser needs that phrase intact.
    endpoint_prefix = re.search(r"\b(?:(?:số|so)\s+)?(?:cuối|cuoi|đầu|dau)\s+$", question[:offset], re.I)
    if endpoint_prefix and not re.search(r"\b(?:ban|bat|lan)$", normalize_text(question[:endpoint_prefix.start()])):
        offset = endpoint_prefix.start()
    direction_prefix = re.search(r"\b(?:từ|tu|giữa|giua|đến|den|sang)\s+$", question[:offset], re.I)
    if direction_prefix:
        offset = direction_prefix.start()
    calendar_prefix = re.search(r"\b(?:vào|vao|trong|tại|tai|ở|o|cho|của|cua|theo)\s+$", question[:offset], re.I)
    if calendar_prefix:
        offset = calendar_prefix.start()
    return question[:offset] + "thực hiện " + question[offset:]


def _compile_column_roles(normalized: str) -> tuple[str, ...]:
    """Return explicit column roles in operand order.

    Only lexical role evidence is compiled. A bare year or left/right position
    never creates a role. Revised-plan and forecast phrases take precedence
    over the generic plan token.
    """

    patterns = (
        (
            "revised_plan",
            r"\b(?:ke hoach dieu chinh|ke hoach sua doi|revised plan|"
            r"adjusted plan)\b",
        ),
        ("forecast", r"\b(?:du bao|du kien|uoc tinh|forecast|estimated?)\b"),
        (
            "actual",
            r"\b(?:thuc hien|thuc te|actual|realized|realised|"
            r"ket qua dat duoc)\b|\bth(?=\s+(?:nam\s+)?(?:19|20)\d{2}\b)",
        ),
        ("plan", r"\b(?:ke hoach|muc tieu|plan|target)\b|\bkh(?=\s+(?:nam\s+)?(?:19|20)\d{2}\b)"),
    )
    matches: list[tuple[int, int, str]] = []
    occupied: list[tuple[int, int]] = []
    for role, pattern in patterns:
        for match in re.finditer(pattern, normalized):
            span = match.span()
            if any(span[0] < end and span[1] > start for start, end in occupied):
                continue
            occupied.append(span)
            matches.append((span[0], span[1], role))
    ordered = [role for _, _, role in sorted(matches)]
    if not ordered:
        # Completed/recorded results name ACTUAL in ordinary questions too.
        # A bare year is still role-unspecified; forecasts/plans above win.
        if _has_financial_role_anchor(normalized) and re.search(
            r"\b(?:da ghi nhan|ghi nhan|da dat|recorded)\b", normalized
        ) and not re.search(r"\b(?:se|co the|chua|khong|neu|may|will|would|not)\b", normalized):
            return ("ACTUAL",)
        return ()

    # Repeated role mentions commonly include a second restatement of the
    # same operand. Preserve a two-role arithmetic pair, otherwise collapse
    # exact repetitions to a unary lookup contract.
    ordered_roles = [role.upper() for role in ordered]
    unique_ordered = list(dict.fromkeys(ordered_roles))
    if len(unique_ordered) == 1:
        return tuple(ordered_roles)
    if set(unique_ordered) == {"ACTUAL", "PLAN"}:
        if any(
            marker in normalized
            for marker in ("thap hon", "chua dat", "con thieu")
        ):
            # ``A thấp hơn B`` requests the positive shortfall B - A. Reverse
            # the lexical operands instead of hard-coding PLAN - ACTUAL so a
            # role-swapped counterfactual remains well typed.
            return tuple(reversed(unique_ordered[:2]))
        return tuple(unique_ordered[:2])
    return tuple(unique_ordered)


def _align_column_roles(
    roles: tuple[str, ...],
    *,
    operation: str | None,
    requested_years: tuple[str, ...],
    operand_count: int | None = None,
) -> tuple[str, ...]:
    if not roles or operation is None:
        return roles
    arity = (
        operand_count
        if operation == "SUM_N" and operand_count is not None
        else {
            "SELECT": 1,
            "ADD": 2,
            "SUBTRACT": 2,
            "MULTIPLY": 2,
            "DIVIDE": 2,
            "RATIO_PERCENT": 2,
            "PERCENT_CHANGE": 2,
            "AVERAGE": 2,
            "SUM3": 3,
            "SUM4": 4,
        }.get(operation)
    )
    if arity is None:
        return roles
    if arity == 1:
        return (roles[0],)
    if len(roles) >= arity:
        return tuple(roles[:arity])
    if len(roles) == 1 and operation == "SUM_N":
        # One explicit lane may scope the complete enumerated sum. This copies
        # question-side intent only; it never infers a role from column order.
        return roles * arity
    if (
        len(roles) == 1
        and arity == 2
        and operation in {"SUBTRACT", "PERCENT_CHANGE", "AVERAGE"}
        and len(requested_years) >= 2
    ):
        return roles * 2
    return roles


_FINANCIAL_ROLE_ANCHORS = (
    "doanh thu",
    "gia von",
    "loi nhuan",
    "chi phi",
    "tai san",
    "no ngan han",
    "no dai han",
    "no phai tra",
    "von chu so huu",
    "von dieu le",
    "thu nhap",
    "nguon von",
    "hang ton kho",
    "khoan phai thu",
    "khoan phai tra",
    "co phieu",
    "co phan",
)


def _strip_nonsemantic_metric_surface(text: str) -> str:
    """Remove proven framing from the metric surface, never typed intent.

    The full normalized question continues to drive operation, period, role,
    scope, unit, direction and reference-base compilation.  This helper only
    prevents source framing or an explicitly qualitative exclusion from being
    copied into the row metric stored in immutable operand slots.
    """

    cleaned = re.sub(r"\s+", " ", _strip_output_unit_directives(text)).strip()

    anchor_starts = [
        match.start()
        for anchor in _FINANCIAL_ROLE_ANCHORS
        for match in re.finditer(rf"\b{re.escape(anchor)}\b", cleaned)
    ]
    if anchor_starts:
        core_start = min(anchor_starts)
        role_before_metric = re.search(
            r"\b(?:ke\s+hoach\s+dieu\s+chinh|ke\s+hoach\s+sua\s+doi|"
            r"revised\s+plan|adjusted\s+plan|du\s+bao|uoc\s+tinh|forecast|"
            r"thuc\s+hien|thuc\s+te|actual|ke\s+hoach|plan)\s+$",
            cleaned[:core_start],
        )
        if role_before_metric is not None:
            core_start = role_before_metric.start()
        prefix = cleaned[:core_start].strip()
        prefix_tokens = set(prefix.split())
        discourse_tokens = {
            "theo",
            "dua",
            "tren",
            "can",
            "cu",
            "trong",
            "pham",
            "vi",
            "cung",
            "bao",
            "cao",
            "bang",
            "so",
            "lieu",
            "da",
            "duoc",
            "cong",
            "bo",
            "trinh",
            "bay",
            "tai",
            "chinh",
            "va",
            "chi",
            "dung",
            "mot",
            "nay",
        }
        starts_as_discourse = any(
            prefix.startswith(marker)
            for marker in ("theo ", "dua tren ", "can cu ", "trong pham vi ")
        )
        names_source_frame = "bao cao" in prefix or "so lieu" in prefix
        if (
            prefix
            and starts_as_discourse
            and names_source_frame
            and prefix_tokens.issubset(discourse_tokens)
        ):
            cleaned = cleaned[core_start:].strip()

    source_gloss_tokens = {
        "as",
        "reported",
        "in",
        "source",
        "unit",
        "units",
        "don",
        "vi",
        "theo",
        "nguon",
    }
    source_gloss = re.search(
        r"\b(?:as\s+reported|reported\s+in\s+source|"
        r"don\s+vi\s+theo\s+nguon|theo\s+nguon\s+source)\b",
        cleaned,
    )
    if source_gloss is not None:
        suffix_tokens = set(cleaned[source_gloss.start() :].split())
        if suffix_tokens and suffix_tokens.issubset(source_gloss_tokens):
            cleaned = cleaned[: source_gloss.start()].strip()

    exclusion = re.search(r"\bkhong\s+(?:yeu\s+cau|can)\b", cleaned)
    if exclusion is not None:
        qualitative_tokens = {
            "khong",
            "yeu",
            "cau",
            "can",
            "phan",
            "tich",
            "giai",
            "thich",
            "nguyen",
            "nhan",
            "bien",
            "dong",
            "binh",
            "luan",
            "dinh",
            "xu",
            "huong",
        }
        suffix_tokens = set(cleaned[exclusion.start() :].split())
        if suffix_tokens and suffix_tokens.issubset(qualitative_tokens):
            cleaned = cleaned[: exclusion.start()].strip()

    return re.sub(r"\s+", " ", cleaned).strip(" ?.,")


def _strip_question_suffix(text: str) -> str:
    text = re.split(
        r"\s+(?:(?:tai|vao|trong)\s+)?(?:thuc hien\s+)?(?:cuoi|dau)\s+(?:nam|ky)\b",
        text,
        maxsplit=1,
    )[0]
    text = re.split(
        r"\s+(?:(?:tai|vao|trong)\s+)?nam\s+"
        r"(?:nay|hien\s+tai|truoc)\b",
        text,
        maxsplit=1,
    )[0]
    text = re.split(
        r"\s+(?:tai|vao|trong)?\s*"
        r"(?:\d{1,2}\s+\d{1,2}\s+)?"
        r"(?:nam\s+)?(?:19|20)\d{2}\b",
        text,
        maxsplit=1,
    )[0]
    text = re.split(r"\s+la bao nhieu\b", text, maxsplit=1)[0]
    text = re.sub(r"\s+(?:thuc hien|thuc te|actual|ke hoach|plan)$", "", text)
    return text.strip(" ?.,")


def _has_financial_role_anchor(text: str) -> bool:
    return any(anchor in text for anchor in _FINANCIAL_ROLE_ANCHORS)


def _has_compound_financial_metric(metric_text: str) -> bool:
    """Return true only when both sides of a conjunction name metrics."""

    for conjunction in re.finditer(r"\s+va\s+", metric_text):
        left = metric_text[: conjunction.start()].strip(" ?.,")
        right = metric_text[conjunction.end() :].strip(" ?.,")
        if (
            left
            and right
            and _has_financial_role_anchor(left)
            and _has_financial_role_anchor(right)
        ):
            return True
    return False


def requests_financial_identity_computation(normalized: str) -> bool:
    """Keep explicit derivations open to the identity synthesizer.

    Merely naming a metric that also has a known accounting identity is still
    a scalar lookup.  Only question-side language that asks for construction
    or component checking suppresses the primary SELECT intent.
    """

    padded = f" {normalized} "
    explicit_formula = bool(
        re.search(
            r"\b(?:duoc\s+)?tinh\s+(?:tu|dua tren|bang)\b",
            normalized,
        )
        or re.search(
            r"^tinh\b.*\b(?:tu|dua tren)\b",
            normalized,
        )
        or re.match(r"^tinh\s+(?:he so|ty le|bien|menh gia)\b", normalized)
    )
    return (
        explicit_formula
        or " hay liet ke " in padded
        or " de kiem tra " in padded
        or " bao gom " in padded
        or " gom " in padded
        or " trong do " in padded
        or " chi tiet " in padded
        or " duoc cau thanh " in padded
    )


def _addition_role_hypotheses(
    normalized: str,
) -> tuple[tuple[str, str], ...]:
    """Return every structural two-role parse for an explicit ADD request.

    A conjunction may be part of a metric name, so question text alone cannot
    choose one split.  The typed synthesizer and verifier keep only parses for
    which two independent evidence cells prove both roles.
    """

    expression = re.sub(
        r"^(?:tai|vao|trong)\s+"
        r"(?:nam nay|nam truoc|cuoi nam|dau nam|cuoi ky|dau ky|"
        r"(?:nam\s+)?(?:19|20)\d{2})\s*,?\s*",
        "",
        normalized,
        count=1,
    )
    request_starts = [
        match.start()
        for match in re.finditer(r"\b(?:tinh\s+)?tong\s+", expression)
    ]
    for match in re.finditer(
        r"\b(?:tinh\s+)?cong\s+(.+?)\s+voi\s+", expression
    ):
        if _has_financial_role_anchor(match.group(1)):
            request_starts.append(match.start())
    if request_starts:
        request_start = min(request_starts)
        prefix = expression[:request_start].strip(" ,:;-")
        source_frame = any(
            prefix.startswith(marker)
            for marker in ("theo ", "dua tren ", "can cu ", "trong pham vi ")
        )
        names_source = "bao cao" in prefix or "so lieu" in prefix or "bang" in prefix
        if prefix and source_frame and names_source:
            expression = expression[request_start:]
    expression = _strip_question_suffix(expression)
    direct = re.match(r"^(?:tinh\s+)?cong\s+(.+?)\s+voi\s+(.+)$", expression)
    bodies: list[tuple[str, str]] = []
    if direct is not None:
        left, right = (part.strip(" ?.,") for part in direct.groups())
        if left and right:
            bodies.append((left, right))
    else:
        total = re.match(r"^(?:tinh\s+)?tong\s+(.+)$", expression)
        if total is not None:
            expression = total.group(1).strip()
        else:
            trailing = re.match(
                r"^(.+?)\s+co\s+tong\s+gia\s+tri\s+"
                r"(?:la\s+)?bao\s+nhieu(?:\s+.*)?$",
                normalized,
            )
            if trailing is None:
                return ()
            expression = trailing.group(1).strip()
        for match in re.finditer(r"\s+va\s+", expression):
            left = expression[: match.start()].strip(" ?.,")
            right = expression[match.end() :].strip(" ?.,")
            if left and right:
                bodies.append((left, right))
    return tuple(dict.fromkeys(bodies))


def _explicit_addition_roles(normalized: str) -> tuple[str, str] | None:
    hypotheses = _addition_role_hypotheses(normalized)
    if len(hypotheses) != 1:
        return None
    left, right = hypotheses[0]
    strong_frame = (
        normalized.startswith("cong ")
        or normalized.startswith("tinh cong ")
        or " co tong gia tri " in f" {normalized} "
    )
    if strong_frame or (
        _has_financial_role_anchor(left)
        and _has_financial_role_anchor(right)
    ):
        return left, right
    return None


def _explicit_sum_operand_roles(question: str) -> tuple[str, ...]:
    """Compile an explicitly punctuated, bounded sum list from question text.

    Punctuation is marked before normalization because normalization removes
    it. A bare conjunction is not enough: it may be part of one accounting
    metric. Quoted inventories are also explicit boundaries.
    """

    normalized = normalize_text(question)
    if not re.match(r"^(?:hay\s+)?(?:tinh\s+)?tong\b", normalized):
        return ()

    quoted_items = re.finditer(
        r'(?:"([^"]+)")|(?:\u201c([^\u201d]+)\u201d)',
        question,
    )
    quoted = tuple(
        normalized_item
        for match in quoted_items
        for item in (match.group(1) or match.group(2) or "",)
        for normalized_item in (normalize_text(item).strip(),)
        if normalized_item
    )
    if 2 <= len(quoted) <= 8 and len(set(quoted)) == len(quoted):
        return quoted

    marked = normalize_text(
        question.replace(",", " V14COMMA ").replace(";", " V14COMMA ")
    )
    body_match = re.match(
        r"^(?:hay\s+)?(?:tinh\s+)?tong"
        r"(?:\s+cua)?(?:\s+(?:hai|ba|bon|nam|sau|bay|tam|[2-8]))?\s+(.+)$",
        marked,
    )
    if body_match is None or " v14comma " not in f" {marked} ":
        return ()
    body = _strip_question_suffix(body_match.group(1))
    head, separator, tail = body.rpartition(" va ")
    if not separator:
        return ()
    raw_parts = (*head.split(" v14comma "), tail)
    roles: list[str] = []
    for position, part in enumerate(raw_parts):
        cleaned = _clean_comparison_operand(part)
        if position == 0:
            cleaned = re.sub(
                r"^(?:(?:so hoc|phep cong|gia tri) cua )?"
                r"(?:cac |hai |ba |bon |nam |sau |bay |tam )?"
                r"(?:khoan |chi tieu )?",
                "",
                cleaned,
            ).strip()
        if not cleaned:
            return ()
        roles.append(cleaned)
    if not 2 <= len(roles) <= 8 or len(set(roles)) != len(roles):
        return ()
    return tuple(roles)


def _clean_comparison_operand(value: str) -> str:
    cleaned = _strip_relative_period_surface(value)
    cleaned = re.sub(
        r"\s+(?:la\s+)?bao\s+nhieu(?:\s+.*)?$",
        " ",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(?:tai|vao|tinh\s+den)?\s*(?:ngay\s+)?"
        r"\d{1,2}\s+\d{1,2}\s+(?:19|20)\d{2}\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\b(?:19|20)\d{2}\b", " ", cleaned)
    cleaned = re.sub(
        r"\b(?:ke\s+hoach\s+dieu\s+chinh|ke\s+hoach\s+sua\s+doi|"
        r"revised\s+plan|adjusted\s+plan|thuc\s+hien|thuc\s+te|actual|"
        r"ke\s+hoach|plan|du\s+bao|uoc\s+tinh|forecast)\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(?:nam|nghin\s+dong|trieu\s+dong|ty\s+dong|vnd)\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(
        r"^(?:hay\s+cho\s+biet|hay\s+xac\s+dinh|xac\s+dinh|tinh)\s+",
        "",
        cleaned,
    )
    return re.sub(r"\s+", " ", cleaned).strip(" ?.,")


def _explicit_subtraction_roles(normalized: str) -> tuple[str, str] | None:
    """Extract ordered A-B roles only from explicit subtraction syntax."""

    patterns = (
        r"^(?:tinh\s+)?hieu(?:\s+so)?\s+cua\s+(.+?)\s+va\s+(.+)$",
        r"^(?:tinh\s+)?(?:lay\s+)?(.+?)\s+tru\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, normalized)
        if match is None:
            continue
        left, right = (
            _clean_comparison_operand(part) for part in match.groups()
        )
        if not left or not right:
            continue
        if _has_financial_role_anchor(left) and _has_financial_role_anchor(right):
            return left, right
    return None


def _relative_difference_roles(normalized: str) -> tuple[str, str] | None:
    """Compile comparison wording into the operand order it explicitly asks for."""

    patterns = (
        # A lower than B by how much means the positive decrease B - A.
        (r"^(.+?)\s+(?:thap hon|nho hon|it hon)\s+(.+)$", True),
        # A higher than B by how much means the signed difference A - B.
        (r"^(.+?)\s+(?:cao hon|lon hon|nhieu hon)\s+(.+)$", False),
    )
    for pattern, reverse in patterns:
        match = re.match(pattern, normalized)
        if match is None:
            continue
        left, right = (
            _clean_comparison_operand(part) for part in match.groups()
        )
        if not left or not right:
            continue
        if _has_financial_role_anchor(left) and _has_financial_role_anchor(right):
            return (right, left) if reverse else (left, right)
    return None


def _symmetric_difference_roles(normalized: str) -> tuple[str, str] | None:
    """Extract two metrics while leaving an unspecified sign fail-closed."""

    patterns = (
        r"^(?:tinh\s+)?(?:chenh\s+lech|do\s+lech)"
        r"(?:\s+giua)?\s+(.+?)\s+va\s+(.+)$",
        r"^(.+?)\s+va\s+(.+?)\s+"
        r"(?:chenh\s+lech|khac\s+nhau)(?:\s+.*)?$",
    )
    for pattern in patterns:
        match = re.match(pattern, normalized)
        if match is None:
            continue
        left, right = (
            _clean_comparison_operand(part) for part in match.groups()
        )
        if (
            left
            and right
            and _has_financial_role_anchor(left)
            and _has_financial_role_anchor(right)
        ):
            return left, right
    return None


def _compile_direction_intent(
    normalized: str,
    *,
    operation: str | None,
    temporal_change: bool,
    from_year: str | None,
    to_year: str | None,
    operand_roles: tuple[str, ...],
    column_roles: tuple[str, ...],
) -> DirectionIntent | None:
    padded = f" {normalized} "
    if (
        operation == "RATIO_PERCENT"
        and len(column_roles) == 2
        and set(column_roles) == {"ACTUAL", "PLAN"}
    ):
        if any(marker in normalized for marker in ("con thieu", "chua dat")):
            return (
                DirectionIntent.PLAN_MINUS_ACTUAL
                if column_roles == ("PLAN", "ACTUAL")
                else DirectionIntent.ACTUAL_MINUS_PLAN
            )
        if "vuot" in normalized:
            return (
                DirectionIntent.ACTUAL_MINUS_PLAN
                if column_roles == ("ACTUAL", "PLAN")
                else DirectionIntent.PLAN_MINUS_ACTUAL
            )
    directional_question = operation in {"SUBTRACT", "PERCENT_CHANGE"} or any(
        phrase in normalized
        for phrase in (
            "chenh lech",
            "do lech",
            "hieu so",
            " tru ",
            "cao hon",
            "thap hon",
            "lon hon",
            "nho hon",
            "it hon",
            "nhieu hon",
            "vuot",
        )
    )
    if not directional_question:
        return None
    if any(
        phrase in normalized
        for phrase in (
            "chenh lech tuyet doi",
            "do lech tuyet doi",
            "gia tri tuyet doi cua chenh lech",
            "khong xet dau",
            "khoang cach giua hai gia tri",
        )
    ):
        return DirectionIntent.ABSOLUTE_GAP
    if (
        operation == "SUBTRACT"
        and _symmetric_difference_roles(normalized) is not None
    ):
        return DirectionIntent.AMBIGUOUS
    if any(
        phrase in normalized
        for phrase in (
            "giam bao nhieu",
            "muc giam",
            "thap hon",
            "nho hon",
            "it hon",
        )
    ):
        return DirectionIntent.DECREASE_AMOUNT
    if (
        operation in {"SUBTRACT", "PERCENT_CHANGE"}
        and len(column_roles) == 2
        and len(set(column_roles)) == 2
    ):
        # A comparison between distinct semantic roles is ordered by the
        # language, even when the two roles happen to belong to different
        # years. Calendar recency must not override PLAN-vs-ACTUAL order.
        return DirectionIntent.LEFT_MINUS_RIGHT
    if from_year and to_year:
        return DirectionIntent.NEW_MINUS_OLD
    if temporal_change:
        return DirectionIntent.NEW_MINUS_OLD
    if len(operand_roles) == 2:
        return DirectionIntent.LEFT_MINUS_RIGHT
    if " tru " in padded or any(
        phrase in normalized
        for phrase in ("hieu cua", "hieu so cua", "cao hon", "lon hon", "nhieu hon")
    ):
        return DirectionIntent.LEFT_MINUS_RIGHT
    return DirectionIntent.AMBIGUOUS


def _comparison_difference_roles(normalized: str) -> tuple[str, str] | None:
    """Extract X - Y from a same-period balance/equality check."""

    if not any(marker in normalized for marker in ("chenh lech", "neu lech")):
        return None
    match = re.search(r"(.+?)\s+co\s+bang\s+(.+?)\s+khong\b", normalized)
    if match is None:
        return None

    def clean(role: str) -> str:
        role = re.sub(
            r"\b(?:tai|vao|tinh\s+den)?\s*(?:ngay\s+)?"
            r"\d{1,2}\s+\d{1,2}\s+(?:19|20)\d{2}\b",
            " ",
            role,
        )
        role = re.sub(r"\b(?:19|20)\d{2}\b", " ", role)
        role = re.sub(r"^(?:tinh|kiem tra|xac dinh)\s+", "", role)
        return re.sub(r"\s+", " ", role).strip(" ?.,")

    left, right = clean(match.group(1)), clean(match.group(2))
    if not left or not right:
        return None
    if not (_has_financial_role_anchor(left) and _has_financial_role_anchor(right)):
        return None
    return left, right


def _coefficient_roles(normalized: str) -> tuple[str, str] | None:
    if "he so" not in normalized or " tren " not in f" {normalized} ":
        return None
    expression = normalized.rsplit("he so", 1)[-1].strip()
    if " tren " not in f" {expression} ":
        return None
    left, right = expression.split(" tren ", 1)
    left = left.strip()
    right = _strip_question_suffix(right)
    if not left or not right:
        return None
    return left, right


def _par_value_roles(normalized: str) -> tuple[str, str] | None:
    marker = "gia tri theo menh gia cua"
    if marker not in normalized:
        return None
    metric = _strip_question_suffix(normalized.split(marker, 1)[-1])
    if not metric or not any(term in metric for term in ("co phieu", "co phan")):
        return None
    return metric, "menh gia"


def _ratio_roles(normalized: str) -> tuple[str, str] | None:
    padded = f" {normalized} "
    marker = next(
        (
            item
            for item in (
                " chiem bao nhieu phan tram ",
                " bang bao nhieu phan tram ",
                " bang bao nhieu lan ",
                " chia cho ",
                " tren ",
                " trong tong ",
            )
            if item in padded
        ),
        None,
    )
    if marker is None:
        return None
    left, right = padded.split(marker, 1)
    left = re.sub(
        r"^(?:tinh\s+)?(?:ty le\s+|ty trong\s+|lay\s+)?",
        "",
        left.strip(),
    )
    if " he so " in f" {left} ":
        left = left.rsplit("he so", 1)[-1].strip()
    right = _strip_question_suffix(right)
    return (left, right) if left and right else None


def _multiplication_roles(normalized: str) -> tuple[str, str] | None:
    padded = f" {normalized} "
    marker = next(
        (item for item in (" nhan voi ", " nhan cho ") if item in padded),
        None,
    )
    if marker is None:
        return None
    left, right = padded.split(marker, 1)
    left = left.strip()
    if " bang " in f" {left} ":
        left = left.rsplit(" bang ", 1)[-1].strip()
    left = re.sub(r"^(?:tinh\s+)?", "", left)
    right = _strip_question_suffix(right)
    return (left, right) if left and right else None
