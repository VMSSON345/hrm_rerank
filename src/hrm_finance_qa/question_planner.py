from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from .financial_identities import match_financial_identities
from .ontology import normalize_text


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
    column_roles: tuple[str, ...] = ()
    addition_role_hypotheses: tuple[tuple[str, str], ...] = ()
    operand_slots: tuple[OperandSlot, ...] = ()
    operand_schema_version: str = "question-operand-slot-v1"
    reference_base: str | None = None
    program_dag: tuple[ProgramIntentNode, ...] = ()
    program_schema_version: str = "question-program-intent-dag-v1"
    program_identity_ids: tuple[str, ...] = ()
    abstention_reason: str = ""
    answer_mode: str = "SINGLE_SCALAR"
    requested_output_count: int = 1

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

    if plan.column_roles or plan.operand_slots:
        required.add("column_role_match")
        applicable.add("column_role_match")

    # QP3-CORE applies the provenance contract only to questions for which the
    # compiler produced immutable typed slots.  This keeps non-core legacy
    # questions outside the declared support lane instead of pretending that
    # sparse metadata is complete.
    if plan.operand_slots:
        required.update({"logical_cell_unique", "source_provenance_complete"})
        applicable.update({"logical_cell_unique", "source_provenance_complete"})

    if plan.requested_years:
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


def _compile_requested_scale(normalized: str) -> str | None:
    """Compile an explicit monetary output scale from question text only."""

    padded = f" {normalized} "
    scales: set[str] = set()
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


def compile_question_plan(question: str) -> QuestionPlan:
    normalized = normalize_text(question)
    padded = f" {normalized} "
    requested_scale = _compile_requested_scale(normalized)
    years = tuple(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", question)))
    asks_percent = (
        "%" in question
        or "phan tram" in normalized
        or "ty le tang" in normalized
        or "ty le giam" in normalized
        or "toc do tang truong" in normalized
    )
    requested_unit_family = ""
    if asks_percent:
        requested_unit_family = "percent"
    elif "he so" in normalized:
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
            "so sanh",
            "cao hon",
            "thap hon",
            "lon hon",
            "nho hon",
            "it hon",
            "nhieu hon",
        )
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

    addition_role_hypotheses = _addition_role_hypotheses(normalized)
    addition_roles = _explicit_addition_roles(normalized)
    difference_roles = (
        _comparison_difference_roles(normalized)
        or _explicit_subtraction_roles(normalized)
        or _relative_difference_roles(normalized)
    )
    coefficient_roles = _coefficient_roles(normalized)
    par_value_roles = _par_value_roles(normalized)
    ratio_roles = _ratio_roles(normalized)
    multiplication_roles = _multiplication_roles(normalized)
    role_request_text, role_context_text = _partition_column_role_context(normalized)
    lexical_column_roles = _compile_column_roles(role_request_text)
    contextual_column_roles = _compile_column_roles(role_context_text)
    column_roles = lexical_column_roles
    direct_metric_lookup = coefficient_roles is not None
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
    if shortfall_percent or excess_percent:
        # These are multi-step ratios: (target-observed)/base or
        # (observed-target)/base. QP2B identifies the root operation and
        # direction; QP2C will materialize the SUBTRACT->RATIO_PERCENT DAG.
        operation = "RATIO_PERCENT"
    elif explicit_role_subtract or excess_amount:
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
    elif difference_roles is not None:
        operation = "SUBTRACT"
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
    )

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
        operand_roles = addition_roles or difference_roles or par_value_roles or ()
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
        r"(?=\s+(?:nam|tu nam|giua nam|tai nam|thay doi|tang|giam|"
        r"chenh lech|chiem|la bao nhieu)\b|[?.!,]|$)",
        normalized,
    )
    if entity_match:
        entity_text = entity_match.group(1).strip()
    entity_terms = tuple(
        token
        for token in re.findall(r"[a-z0-9]+", entity_text)
        if len(token) >= 3 and token not in _ENTITY_STOPWORDS
    )

    metric_text = normalized
    if role_context_text:
        metric_text = metric_text.replace(role_context_text, " ")
    if entity_match:
        metric_text = (
            metric_text[: entity_match.start()]
            + " "
            + metric_text[entity_match.end() :]
        )
    metric_text = _strip_nonsemantic_metric_surface(metric_text)
    metric_text = re.sub(r"\b(?:19|20)\d{2}\b", " ", metric_text)
    metric_text = re.sub(
        r"\b(?:nghin\s+dong|trieu\s+dong|ty\s+dong|vnd)\b|"
        r"(?<!hoat )(?<!co )\bdong\b",
        " ",
        metric_text,
    )
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
            r"\b(?:ke hoach dieu chinh|ke hoach sua doi|revised plan|"
            r"adjusted plan|thuc hien|thuc te|actual|ke hoach|plan|"
            r"du bao|uoc tinh|forecast)\b",
            " ",
            metric_text,
        )
        metric_text = re.sub(
            r"\b(?:hoan thanh|dat duoc|dat|ket qua)\b", " ", metric_text
        )
    metric_text = re.sub(r"\s+", " ", metric_text).strip(" ?.,")
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

    entity_mode = "named" if entity_text else "reporting"
    operand_slots = _compile_operand_slots(
        normalized,
        metric_text=metric_text,
        operation=operation,
        requested_years=years,
        # Preserve the lexical operand order for the new slot contract. The
        # legacy completion-ratio branch above intentionally canonicalizes
        # runtime column_roles to ACTUAL/PLAN; QP2A must not inherit that loss
        # on a counterfactual PLAN/ACTUAL question.
        column_roles=lexical_column_roles,
        requested_unit_family=requested_unit_family,
        entity_mode=entity_mode,
        entity_text=entity_text,
    )
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
        column_roles=column_roles,
        addition_role_hypotheses=addition_role_hypotheses,
        operand_slots=operand_slots,
        reference_base=reference_base,
        program_dag=program_dag,
        program_identity_ids=program_identity_ids,
        abstention_reason=abstention_reason,
        answer_mode=answer_mode,
        requested_output_count=requested_output_count,
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

    arity = _PROGRAM_INTENT_ARITY.get(operation)
    if arity is None or len(slot_ids) != arity:
        return None, ()
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
        if arity is None:
            errors.append(f"unsupported_operation:{node.operation}")
        elif len(node.inputs) != arity:
            errors.append(f"wrong_arity:{node.id}")
        if any(item not in available for item in node.inputs):
            errors.append(f"unresolved_or_non_topological_input:{node.id}")
        input_units = [units.get(item, "") for item in node.inputs]
        if node.operation in {"ADD", "SUBTRACT", "AVERAGE"}:
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


def _clean_operand_metric(metric_text: str) -> str:
    """Remove arithmetic wording while retaining the requested row metric."""

    cleaned = metric_text
    if " hay xac dinh " in f" {cleaned} ":
        cleaned = cleaned.rsplit("hay xac dinh", 1)[-1].strip()
    cleaned = re.sub(
        r"^(?:hay cho biet|theo bao cao|dua tren so lieu duoc cong bo|"
        r"doi voi cung chi tieu va pham vi bao cao)\s+",
        "",
        cleaned,
    )
    cleaned = re.split(
        r"\s+(?:tru|thap hon|cao hon|nho hon|it hon|nhieu hon|vuot|"
        r"con thieu|chua dat|thay doi|tang|giam|so voi)\b",
        cleaned,
        maxsplit=1,
    )[0]
    return re.sub(r"\s+", " ", cleaned).strip(" ?.,")


def _slot_unit_family(operation: str | None, requested_unit_family: str) -> str:
    # The requested answer may be a percent/ratio while its leaf operands are
    # monetary values. A direct SELECT can preserve an explicitly requested
    # unit or defer to the source cell.
    if operation == "SELECT":
        return requested_unit_family or "currency_or_source"
    return "currency"


def _slot_id(column_role: str, year: str, *, multi_year: bool) -> str:
    base = column_role.lower()
    return f"{base}_{year}" if multi_year else base


def _compile_interest_bearing_debt_slots(
    normalized: str,
    *,
    requested_years: tuple[str, ...],
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
    if not (asks_share and has_short and has_long and has_capital and requested_years):
        return ()
    year = requested_years[0]
    common_sources = (
        "year:question_explicit",
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
        ),
        OperandSlot(
            id="long_term_borrowings",
            metric="vay va no thue tai chinh dai han",
            year=year,
            column_role="ACTUAL",
            entity_scope=entity_scope,
            unit_family="currency",
            intent_sources=("metric:long_term_borrowing_lexical",) + common_sources,
        ),
        OperandSlot(
            id="total_capital",
            metric="tong nguon von",
            year=year,
            column_role="ACTUAL",
            entity_scope=entity_scope,
            unit_family="currency",
            intent_sources=("metric:total_capital_lexical",) + common_sources,
        ),
    )


def _compile_operand_slots(
    normalized: str,
    *,
    metric_text: str,
    operation: str | None,
    requested_years: tuple[str, ...],
    column_roles: tuple[str, ...],
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

    entity_scope = "reporting" if entity_mode == "reporting" else f"named:{entity_text}"
    identity_slots = _compile_interest_bearing_debt_slots(
        normalized,
        requested_years=requested_years,
        entity_scope=entity_scope,
    )
    if identity_slots:
        return identity_slots
    if not requested_years or not column_roles:
        return ()

    roles = column_roles
    years: tuple[str, ...]
    if len(requested_years) == 1:
        years = (requested_years[0],) * len(roles)
    elif len(roles) == len(requested_years):
        years = requested_years
    elif len(roles) == 1:
        # Do not duplicate a role unless the existing column-role compiler has
        # explicitly aligned it to every requested period.
        return ()
    else:
        return ()

    metric = _clean_operand_metric(metric_text)
    if not metric:
        return ()
    multi_year = len(set(years)) > 1
    unit_family = _slot_unit_family(operation, requested_unit_family)
    role_source = "column_role:question_explicit"
    year_source = "year:question_explicit"
    slots = tuple(
        OperandSlot(
            id=_slot_id(role, year, multi_year=multi_year),
            metric=metric,
            year=year,
            column_role=role,
            entity_scope=entity_scope,
            unit_family=unit_family,
            intent_sources=(
                "metric:question_lexical",
                year_source,
                role_source,
                "entity:question_scope",
                "unit:operation_contract",
            ),
        )
        for role, year in zip(roles, years)
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
        ("forecast", r"\b(?:du bao|uoc tinh|forecast|estimated?)\b"),
        (
            "actual",
            r"\b(?:thuc hien|thuc te|actual|realized|realised|"
            r"ket qua dat duoc)\b",
        ),
        ("plan", r"\b(?:ke hoach|muc tieu|plan|target)\b"),
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
) -> tuple[str, ...]:
    if not roles or operation is None:
        return roles
    arity = {
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
    if arity is None:
        return roles
    if arity == 1:
        return (roles[0],)
    if len(roles) >= arity:
        return tuple(roles[:arity])
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

    cleaned = re.sub(r"\s+", " ", text).strip()

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
        r"\s+(?:tai|vao|trong)\s+(?:cuoi|dau)\s+(?:nam|ky)\b",
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
    return text.strip(" ?.,")


def _has_financial_role_anchor(text: str) -> bool:
    return any(anchor in text for anchor in _FINANCIAL_ROLE_ANCHORS)


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


def _explicit_subtraction_roles(normalized: str) -> tuple[str, str] | None:
    """Extract ordered A-B roles only from explicit subtraction syntax."""

    expression = _strip_question_suffix(normalized)
    patterns = (
        r"^(?:tinh\s+)?hieu(?:\s+so)?\s+cua\s+(.+?)\s+va\s+(.+)$",
        r"^(?:tinh\s+)?(?:lay\s+)?(.+?)\s+tru\s+(.+)$",
    )
    for pattern in patterns:
        match = re.match(pattern, expression)
        if match is None:
            continue
        left, right = (part.strip(" ?.,") for part in match.groups())
        if not left or not right:
            continue
        if _has_financial_role_anchor(left) and _has_financial_role_anchor(right):
            return left, right
    return None


def _relative_difference_roles(normalized: str) -> tuple[str, str] | None:
    """Compile comparison wording into the operand order it explicitly asks for."""

    expression = _strip_question_suffix(normalized)
    patterns = (
        # A lower than B by how much means the positive decrease B - A.
        (r"^(.+?)\s+(?:thap hon|nho hon|it hon)\s+(.+)$", True),
        # A higher than B by how much means the signed difference A - B.
        (r"^(.+?)\s+(?:cao hon|lon hon|nhieu hon)\s+(.+)$", False),
    )
    for pattern, reverse in patterns:
        match = re.match(pattern, expression)
        if match is None:
            continue
        left, right = (part.strip(" ?.,") for part in match.groups())
        left = re.sub(r"\s+(?:la\s+)?bao nhieu$", "", left).strip()
        right = re.sub(r"\s+(?:la\s+)?bao nhieu$", "", right).strip()
        if not left or not right:
            continue
        if _has_financial_role_anchor(left) and _has_financial_role_anchor(right):
            return (right, left) if reverse else (left, right)
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
                " chia cho ",
                " tren ",
            )
            if item in padded
        ),
        None,
    )
    if marker is None:
        return None
    left, right = padded.split(marker, 1)
    left = re.sub(r"^(?:tinh\s+)?(?:ty le\s+|lay\s+)?", "", left.strip())
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
