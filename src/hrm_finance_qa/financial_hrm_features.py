"""Report-independent structural features for the V13 financial HRM path.

The feature extractors consume only question intent and source-derived
candidate metadata.  They never inspect expected answers, company-specific
identifiers, page constants, or numeric magnitudes to infer semantic identity.
"""

from __future__ import annotations

from .physical_coordinates import physical_coordinate_key, is_versioned_coordinate

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
import re
from typing import Sequence

from .ontology import normalize_text
from .program_data import NumericCandidate, currency_unit_scale
from .question_planner import OperandSlot, QuestionPlan


FINANCIAL_CANDIDATE_FEATURES = (
    "row_present",
    "column_present",
    "period_present",
    "table_present",
    "section_present",
    "entity_or_scope_present",
    "unit_present",
    "source_span_present",
    "row_id_present",
    "column_id_present",
    "source_bbox_present",
    "row_is_total",
    "row_is_subtotal",
    "role_is_actual",
    "role_is_plan",
    "scope_is_consolidated",
    "scope_is_parent_company",
    "currency_scale_explicit",
    "physical_identity_unique",
    "logical_identity_unique",
    "same_table_density",
    "same_row_density",
    "query_metric_row_overlap",
    "query_section_overlap",
)
FINANCIAL_CANDIDATE_FEATURE_SIZE = len(FINANCIAL_CANDIDATE_FEATURES)

FINANCIAL_SLOT_FEATURES = (
    "typed_slot_declared",
    "metric_overlap",
    "metric_exact",
    "year_match",
    "year_mismatch",
    "column_role_match",
    "column_role_mismatch",
    "entity_scope_match",
    "entity_scope_mismatch",
    "unit_family_match",
    "unit_family_mismatch",
    "unit_scale_match",
    "unit_scale_mismatch",
    "physical_provenance_complete",
    "physical_identity_unique",
    "logical_identity_unique",
    "period_explicit",
    "column_role_explicit",
    "entity_scope_explicit",
    "unit_explicit",
    "same_physical_duplicate",
    "same_logical_duplicate",
    "candidate_relevance",
    "slot_position",
    "relative_period_role_match",
    "relative_period_role_mismatch",
)
FINANCIAL_SLOT_FEATURE_SIZE = len(FINANCIAL_SLOT_FEATURES)
FINANCIAL_SLOT_MISMATCH_INDICES = (4, 6, 8, 10, 12, 25)


_TOKEN_STOPWORDS = frozenset(
    {
        "bao",
        "bao cao",
        "bang",
        "cua",
        "don vi",
        "gia tri",
        "la",
        "nam",
        "tai",
        "theo",
        "tinh",
        "va",
    }
)


@dataclass(frozen=True)
class FinancialSlotIntent:
    metric: str
    year: str
    period_role: str
    column_role: str
    entity_scope: str
    unit_family: str
    requested_scale: str | None
    declared: bool


def _tokens(value: str) -> set[str]:
    normalized = normalize_text(value)
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 1 and token not in _TOKEN_STOPWORDS and not token.isdigit()
    }


def _overlap(left: str, right: str) -> float:
    expected = _tokens(left)
    if not expected:
        return 0.0
    return len(expected & _tokens(right)) / len(expected)


def _normalized_exact(left: str, right: str) -> bool:
    a = " ".join(normalize_text(left).split())
    b = " ".join(normalize_text(right).split())
    return bool(a and a == b)


def _row_label(candidate: NumericCandidate) -> str:
    return candidate.row_label.strip()


def _column_label(candidate: NumericCandidate) -> str:
    return candidate.column_label.strip()


def _candidate_years(candidate: NumericCandidate) -> set[str]:
    return set(
        re.findall(
            r"\b(?:19|20)\d{2}\b",
            " ".join((candidate.period, candidate.column_label, candidate.context)),
        )
    )


def _candidate_role(candidate: NumericCandidate) -> str:
    explicit = normalize_text(candidate.column_role).replace(" ", "_")
    if explicit in {"actual", "plan"}:
        return explicit.upper()
    surface = normalize_text(
        " ".join((candidate.column_label, candidate.period, candidate.context))
    )
    actual = any(
        marker in f" {surface} "
        for marker in (" thuc hien ", " thuc te ", " actual ", " realized ")
    ) or bool(re.search(r"(?:^|\W)th(?:\W|$)", surface))
    plan = any(
        marker in f" {surface} "
        for marker in (" ke hoach ", " plan ", " target ", " muc tieu ")
    ) or bool(re.search(r"(?:^|\W)kh(?:\W|$)", surface))
    if actual == plan:
        return ""
    return "ACTUAL" if actual else "PLAN"


def _candidate_period_role(candidate: NumericCandidate) -> str:
    surface = normalize_text(
        " ".join((candidate.period, candidate.column_label))
    )
    descriptor = f" {surface} "
    if any(
        marker in descriptor
        for marker in (" nam truoc ", " dau nam ", " dau ky ", " 01 01 ")
    ):
        return "prior"
    if any(
        marker in descriptor
        for marker in (" nam nay ", " cuoi nam ", " cuoi ky ", " 31 12 ")
    ):
        return "current"
    return ""


def _candidate_scope(candidate: NumericCandidate) -> str:
    return normalize_text(
        " ".join((candidate.scope, candidate.entity_id, candidate.section))
    )


def _is_generic_source_entity(candidate: NumericCandidate) -> bool:
    entity = normalize_text(candidate.entity_id)
    return bool(entity) and entity.startswith("entity source ") and not any(
        marker in entity for marker in (" scope ", " parent ", " consolidated ")
    )


def _candidate_unit_family(candidate: NumericCandidate) -> str:
    unit = normalize_text(candidate.unit)
    if candidate.is_percent or "%" in candidate.unit or "phan tram" in unit:
        return "percent"
    if any(marker in unit for marker in ("share", "co phieu", "co phan")):
        return "share_count"
    if any(marker in unit for marker in ("ratio", "he so", "lan")):
        return "ratio"
    if currency_unit_scale(candidate.unit) is not None:
        return "currency"
    return ""


def _scale_value(requested_scale: str | None) -> Decimal | None:
    return {
        "vnd": Decimal(1),
        "thousand_vnd": Decimal(1000),
        "million_vnd": Decimal(1000000),
        "billion_vnd": Decimal(1000000000),
    }.get(str(requested_scale or ""))


def _physical_key(candidate: NumericCandidate) -> tuple[object, ...] | None:
    if is_versioned_coordinate(candidate):
        return physical_coordinate_key(candidate)
    if not candidate.source_name or candidate.page is None or candidate.source_span is None:
        return None
    source = normalize_text(candidate.source_name)
    span = tuple(candidate.source_span)
    row = candidate.row_id or normalize_text(candidate.row_label)
    column = candidate.column_id or normalize_text(candidate.column_label)
    if candidate.table_id and row and column:
        return (
            "table_cell",
            source,
            int(candidate.page),
            candidate.table_id,
            row,
            column,
            span,
        )
    if candidate.line_number >= 0:
        return (
            "text_span",
            source,
            int(candidate.page),
            int(candidate.line_number),
            span,
        )
    return None


def financial_physical_identity(
    candidate: NumericCandidate,
) -> tuple[object, ...] | None:
    """Return a proof-safe physical cell/span identity when provenance is complete."""

    return _physical_key(candidate)


def financial_same_physical_pairs(
    candidates: Sequence[NumericCandidate],
) -> list[list[bool]]:
    """Mark duplicate candidate records that resolve to the same physical evidence."""

    keys = [_physical_key(candidate) for candidate in candidates]
    return [
        [
            left != right
            and keys[left] is not None
            and keys[left] == keys[right]
            for right in range(len(candidates))
        ]
        for left in range(len(candidates))
    ]


def _logical_key(candidate: NumericCandidate) -> tuple[object, ...] | None:
    if not (candidate.row_id or candidate.row_label) or not (
        candidate.column_id or candidate.column_label or candidate.period
    ):
        return None
    return (
        normalize_text(candidate.source_name),
        candidate.table_id,
        candidate.row_id or normalize_text(candidate.row_label),
        candidate.column_id
        or normalize_text(candidate.column_label)
        or normalize_text(candidate.period),
        normalize_text(candidate.entity_id or candidate.scope),
    )


def _identity_count(
    candidate: NumericCandidate,
    candidates: Sequence[NumericCandidate],
    key_fn,
) -> int:
    key = key_fn(candidate)
    if key is None:
        return 0
    return sum(key_fn(item) == key for item in candidates)


def _pad(values: list[float], feature_size: int) -> list[float]:
    if feature_size < 1:
        raise ValueError("feature_size must be positive")
    if len(values) < feature_size:
        values.extend([0.0] * (feature_size - len(values)))
    return values[:feature_size]


def financial_candidate_features(
    plan: QuestionPlan,
    candidate: NumericCandidate,
    candidates: Sequence[NumericCandidate],
    *,
    feature_size: int = FINANCIAL_CANDIDATE_FEATURE_SIZE,
) -> list[float]:
    row = _row_label(candidate)
    column = _column_label(candidate)
    role = _candidate_role(candidate)
    scope = _candidate_scope(candidate)
    physical_count = _identity_count(candidate, candidates, _physical_key)
    logical_count = _identity_count(candidate, candidates, _logical_key)
    same_table = sum(
        bool(candidate.table_id and item.table_id == candidate.table_id)
        for item in candidates
    )
    same_row = sum(
        bool(row and normalize_text(item.row_label) == normalize_text(row))
        for item in candidates
    )
    denominator = max(1, len(candidates))
    row_type = normalize_text(candidate.row_type)
    values = [
        float(bool(row)),
        float(bool(column)),
        float(bool(candidate.period)),
        float(bool(candidate.table_id)),
        float(bool(candidate.section)),
        float(bool(candidate.entity_id or candidate.scope)),
        float(bool(candidate.unit)),
        float(candidate.source_span is not None),
        float(bool(candidate.row_id)),
        float(bool(candidate.column_id)),
        float(candidate.source_bbox is not None),
        float(row_type == "total"),
        float(row_type == "subtotal"),
        float(role == "ACTUAL"),
        float(role == "PLAN"),
        float("hop nhat" in scope or "consolidated" in scope),
        float("cong ty me" in scope or "parent company" in scope),
        float(currency_unit_scale(candidate.unit) is not None),
        float(physical_count == 1),
        float(logical_count == 1),
        same_table / denominator,
        same_row / denominator,
        _overlap(plan.metric_text, row),
        _overlap(plan.normalized, candidate.section),
    ]
    return _pad(values, feature_size)


def financial_slot_intent(plan: QuestionPlan, slot_index: int) -> FinancialSlotIntent:
    normalized_question = f" {plan.normalized} "
    explicit_entity_scope = ""
    if any(
        marker in normalized_question
        for marker in (" hop nhat ", " consolidated ")
    ):
        explicit_entity_scope = "named:consolidated"
    elif any(
        marker in normalized_question
        for marker in (" cong ty me ", " parent company ")
    ):
        explicit_entity_scope = "named:parent_company"

    if slot_index < len(plan.operand_slots):
        slot: OperandSlot = plan.operand_slots[slot_index]
        return FinancialSlotIntent(
            metric=slot.metric,
            year=slot.year,
            period_role=slot.period_role,
            column_role=slot.column_role,
            entity_scope=explicit_entity_scope or slot.entity_scope,
            unit_family=slot.unit_family,
            requested_scale=plan.requested_scale,
            declared=True,
        )

    years = plan.required_operand_years or plan.requested_years
    year = years[slot_index] if slot_index < len(years) else years[0] if len(years) == 1 else ""
    period_roles = plan.requested_period_roles
    period_role = (
        period_roles[slot_index]
        if slot_index < len(period_roles)
        else period_roles[0]
        if len(period_roles) == 1
        else ""
    )
    roles = plan.column_roles
    role = roles[slot_index] if slot_index < len(roles) else roles[0] if len(roles) == 1 else ""
    metrics = plan.operand_roles
    metric = metrics[slot_index] if slot_index < len(metrics) else plan.metric_text
    entity_scope = explicit_entity_scope or (
        "reporting"
        if plan.entity_mode == "reporting"
        else f"named:{plan.entity_text}"
    )
    unit_family = (
        plan.requested_unit_family or "currency_or_source"
        if plan.operation == "SELECT"
        else "currency"
    )
    return FinancialSlotIntent(
        metric=metric,
        year=year,
        period_role=period_role,
        column_role=role,
        entity_scope=entity_scope,
        unit_family=unit_family,
        requested_scale=plan.requested_scale,
        declared=False,
    )


def _entity_compatibility(
    intent: FinancialSlotIntent, candidate: NumericCandidate
) -> tuple[bool, bool, bool]:
    surface = _candidate_scope(candidate)
    explicit = bool(surface and not _is_generic_source_entity(candidate))
    requested = normalize_text(intent.entity_scope)
    if not requested or requested == "reporting":
        return False, False, explicit
    if "parent company" in requested or "parent_company" in intent.entity_scope:
        match = "cong ty me" in surface or "parent company" in surface
        mismatch = explicit and ("hop nhat" in surface or "consolidated" in surface)
        return match, mismatch, explicit
    if "consolidated" in requested or "hop nhat" in requested:
        match = "hop nhat" in surface or "consolidated" in surface
        mismatch = explicit and ("cong ty me" in surface or "parent company" in surface)
        return match, mismatch, explicit
    requested_terms = _tokens(requested.removeprefix("named "))
    observed_terms = _tokens(surface)
    match = bool(requested_terms and requested_terms <= observed_terms)
    mismatch = bool(explicit and requested_terms and observed_terms and not match)
    return match, mismatch, explicit


def financial_slot_features(
    plan: QuestionPlan,
    slot_index: int,
    candidate: NumericCandidate,
    candidates: Sequence[NumericCandidate],
    *,
    feature_size: int = FINANCIAL_SLOT_FEATURE_SIZE,
) -> tuple[list[float], bool]:
    intent = financial_slot_intent(plan, slot_index)
    row = _row_label(candidate)
    metric_overlap = _overlap(intent.metric, row)
    years = _candidate_years(candidate)
    year_match = bool(intent.year and intent.year in years)
    year_mismatch = bool(intent.year and years and intent.year not in years)
    role = _candidate_role(candidate)
    expected_role = normalize_text(intent.column_role).upper().replace(" ", "_")
    role_match = bool(expected_role and role == expected_role)
    role_mismatch = bool(expected_role and role and role != expected_role)
    entity_match, entity_mismatch, entity_explicit = _entity_compatibility(
        intent, candidate
    )
    family = _candidate_unit_family(candidate)
    expected_family = normalize_text(intent.unit_family).replace(" ", "_")
    if expected_family == "currency_or_source":
        unit_match = bool(candidate.unit)
        unit_mismatch = False
    else:
        unit_match = bool(expected_family and family == expected_family)
        unit_mismatch = bool(expected_family and family and family != expected_family)
    expected_scale = _scale_value(intent.requested_scale)
    observed_scale = currency_unit_scale(candidate.unit)
    scale_match = bool(expected_scale is not None and observed_scale == expected_scale)
    scale_mismatch = bool(
        expected_scale is not None
        and observed_scale is not None
        and observed_scale != expected_scale
    )
    physical_count = _identity_count(candidate, candidates, _physical_key)
    logical_count = _identity_count(candidate, candidates, _logical_key)
    physical_complete = _physical_key(candidate) is not None
    observed_period_role = _candidate_period_role(candidate)
    period_role_match = bool(
        intent.period_role and observed_period_role == intent.period_role
    )
    period_role_mismatch = bool(
        intent.period_role
        and observed_period_role
        and observed_period_role != intent.period_role
    )
    values = [
        float(intent.declared),
        metric_overlap,
        float(_normalized_exact(intent.metric, row)),
        float(year_match),
        float(year_mismatch),
        float(role_match),
        float(role_mismatch),
        float(entity_match),
        float(entity_mismatch),
        float(unit_match),
        float(unit_mismatch),
        float(scale_match),
        float(scale_mismatch),
        float(physical_complete),
        float(physical_count == 1),
        float(logical_count == 1),
        float(bool(years or candidate.period)),
        float(bool(role)),
        float(entity_explicit),
        float(bool(candidate.unit)),
        float(physical_count > 1),
        float(logical_count > 1),
        max(-1.0, min(1.0, float(candidate.relevance))),
        min(1.0, slot_index / 3.0),
        float(period_role_match),
        float(period_role_mismatch),
    ]
    mismatch = any(values[index] > 0.5 for index in FINANCIAL_SLOT_MISMATCH_INDICES)
    return _pad(values, feature_size), mismatch


@dataclass(frozen=True)
class _FinancialCandidateFacts:
    row: str
    column: str
    normalized_row: str
    row_tokens: frozenset[str]
    section_tokens: frozenset[str]
    years: frozenset[str]
    period_role: str
    role: str
    scope: str
    scope_tokens: frozenset[str]
    entity_explicit: bool
    unit_family: str
    observed_scale: Decimal | None
    physical_key: tuple[object, ...] | None
    logical_key: tuple[object, ...] | None
    row_type: str


@dataclass(frozen=True)
class _FinancialSlotFacts:
    intent: FinancialSlotIntent
    metric_tokens: frozenset[str]
    normalized_metric: str
    expected_role: str
    expected_family: str
    expected_scale: Decimal | None
    requested_entity: str
    requested_entity_tokens: frozenset[str]


@dataclass(frozen=True)
class FinancialFeatureContext:
    """Per-example immutable cache for report-independent financial features."""

    plan: QuestionPlan
    candidates: tuple[NumericCandidate, ...]
    candidate_facts: tuple[_FinancialCandidateFacts, ...]
    slot_facts: tuple[_FinancialSlotFacts, ...]
    physical_counts: dict[tuple[object, ...], int]
    logical_counts: dict[tuple[object, ...], int]
    table_counts: dict[str, int]
    row_counts: dict[str, int]
    plan_metric_tokens: frozenset[str]
    plan_question_tokens: frozenset[str]
    same_physical_mask: tuple[tuple[bool, ...], ...]


@dataclass(frozen=True)
class FinancialFeatureMatrices:
    """Dense Python matrices copied into one torch tensor per feature family."""

    candidate_features: tuple[tuple[float, ...], ...]
    slot_features: tuple[tuple[tuple[float, ...], ...], ...]
    slot_mismatch_mask: tuple[tuple[bool, ...], ...]
    same_physical_mask: tuple[tuple[bool, ...], ...]


def _normalized_text(value: str) -> str:
    return " ".join(normalize_text(value).split())


def _same_physical_mask_from_keys(
    keys: Sequence[tuple[object, ...] | None],
) -> tuple[tuple[bool, ...], ...]:
    size = len(keys)
    output = [[False] * size for _ in range(size)]
    groups: dict[tuple[object, ...], list[int]] = {}
    for index, key in enumerate(keys):
        if key is not None:
            groups.setdefault(key, []).append(index)
    for positions in groups.values():
        if len(positions) < 2:
            continue
        for left in positions:
            for right in positions:
                if left != right:
                    output[left][right] = True
    return tuple(tuple(row) for row in output)


def build_financial_feature_context(
    plan: QuestionPlan,
    candidates: Sequence[NumericCandidate],
    *,
    slot_count: int = 4,
) -> FinancialFeatureContext:
    """Precompute normalized candidate facts and identity counts once."""

    if slot_count < 0:
        raise ValueError("slot_count must be non-negative")
    candidate_tuple = tuple(candidates)
    candidate_facts: list[_FinancialCandidateFacts] = []
    for candidate in candidate_tuple:
        row = _row_label(candidate)
        scope = _candidate_scope(candidate)
        candidate_facts.append(
            _FinancialCandidateFacts(
                row=row,
                column=_column_label(candidate),
                normalized_row=_normalized_text(row),
                row_tokens=frozenset(_tokens(row)),
                section_tokens=frozenset(_tokens(candidate.section)),
                years=frozenset(_candidate_years(candidate)),
                period_role=_candidate_period_role(candidate),
                role=_candidate_role(candidate),
                scope=scope,
                scope_tokens=frozenset(_tokens(scope)),
                entity_explicit=bool(
                    scope and not _is_generic_source_entity(candidate)
                ),
                unit_family=_candidate_unit_family(candidate),
                observed_scale=currency_unit_scale(candidate.unit),
                physical_key=_physical_key(candidate),
                logical_key=_logical_key(candidate),
                row_type=normalize_text(candidate.row_type),
            )
        )

    slot_facts: list[_FinancialSlotFacts] = []
    for slot in range(slot_count):
        intent = financial_slot_intent(plan, slot)
        requested_entity = normalize_text(intent.entity_scope)
        slot_facts.append(
            _FinancialSlotFacts(
                intent=intent,
                metric_tokens=frozenset(_tokens(intent.metric)),
                normalized_metric=_normalized_text(intent.metric),
                expected_role=normalize_text(intent.column_role)
                .upper()
                .replace(" ", "_"),
                expected_family=normalize_text(intent.unit_family).replace(
                    " ", "_"
                ),
                expected_scale=_scale_value(intent.requested_scale),
                requested_entity=requested_entity,
                requested_entity_tokens=frozenset(
                    _tokens(requested_entity.removeprefix("named "))
                ),
            )
        )

    physical_keys = tuple(fact.physical_key for fact in candidate_facts)
    logical_keys = tuple(fact.logical_key for fact in candidate_facts)
    return FinancialFeatureContext(
        plan=plan,
        candidates=candidate_tuple,
        candidate_facts=tuple(candidate_facts),
        slot_facts=tuple(slot_facts),
        physical_counts=dict(
            Counter(key for key in physical_keys if key is not None)
        ),
        logical_counts=dict(
            Counter(key for key in logical_keys if key is not None)
        ),
        table_counts=dict(
            Counter(
                candidate.table_id
                for candidate in candidate_tuple
                if candidate.table_id
            )
        ),
        row_counts=dict(
            Counter(
                fact.normalized_row
                for fact in candidate_facts
                if fact.row
            )
        ),
        plan_metric_tokens=frozenset(_tokens(plan.metric_text)),
        plan_question_tokens=frozenset(_tokens(plan.normalized)),
        same_physical_mask=_same_physical_mask_from_keys(physical_keys),
    )


def _token_overlap(
    expected: frozenset[str], observed: frozenset[str]
) -> float:
    if not expected:
        return 0.0
    return len(expected & observed) / len(expected)


def _cached_entity_compatibility(
    slot: _FinancialSlotFacts,
    candidate: _FinancialCandidateFacts,
) -> tuple[bool, bool, bool]:
    requested = slot.requested_entity
    explicit = candidate.entity_explicit
    surface = candidate.scope
    if not requested or requested == "reporting":
        return False, False, explicit
    if (
        "parent company" in requested
        or "parent_company" in slot.intent.entity_scope
    ):
        match = "cong ty me" in surface or "parent company" in surface
        mismatch = explicit and (
            "hop nhat" in surface or "consolidated" in surface
        )
        return match, mismatch, explicit
    if (
        "consolidated" in requested
        or "hop nhat" in requested
    ):
        match = "hop nhat" in surface or "consolidated" in surface
        mismatch = explicit and (
            "cong ty me" in surface or "parent company" in surface
        )
        return match, mismatch, explicit
    requested_terms = slot.requested_entity_tokens
    observed_terms = candidate.scope_tokens
    match = bool(requested_terms and requested_terms <= observed_terms)
    mismatch = bool(
        explicit and requested_terms and observed_terms and not match
    )
    return match, mismatch, explicit


def _cached_candidate_features(
    context: FinancialFeatureContext,
    index: int,
    *,
    feature_size: int,
) -> list[float]:
    candidate = context.candidates[index]
    fact = context.candidate_facts[index]
    physical_count = (
        context.physical_counts.get(fact.physical_key, 0)
        if fact.physical_key is not None
        else 0
    )
    logical_count = (
        context.logical_counts.get(fact.logical_key, 0)
        if fact.logical_key is not None
        else 0
    )
    denominator = max(1, len(context.candidates))
    values = [
        float(bool(fact.row)),
        float(bool(fact.column)),
        float(bool(candidate.period)),
        float(bool(candidate.table_id)),
        float(bool(candidate.section)),
        float(bool(candidate.entity_id or candidate.scope)),
        float(bool(candidate.unit)),
        float(candidate.source_span is not None),
        float(bool(candidate.row_id)),
        float(bool(candidate.column_id)),
        float(candidate.source_bbox is not None),
        float(fact.row_type == "total"),
        float(fact.row_type == "subtotal"),
        float(fact.role == "ACTUAL"),
        float(fact.role == "PLAN"),
        float("hop nhat" in fact.scope or "consolidated" in fact.scope),
        float("cong ty me" in fact.scope or "parent company" in fact.scope),
        float(fact.observed_scale is not None),
        float(physical_count == 1),
        float(logical_count == 1),
        context.table_counts.get(candidate.table_id, 0) / denominator
        if candidate.table_id
        else 0.0,
        context.row_counts.get(fact.normalized_row, 0) / denominator
        if fact.row
        else 0.0,
        _token_overlap(context.plan_metric_tokens, fact.row_tokens),
        _token_overlap(context.plan_question_tokens, fact.section_tokens),
    ]
    return _pad(values, feature_size)


def _cached_slot_features(
    context: FinancialFeatureContext,
    slot_index: int,
    candidate_index: int,
    *,
    feature_size: int,
) -> tuple[list[float], bool]:
    slot = context.slot_facts[slot_index]
    candidate = context.candidates[candidate_index]
    fact = context.candidate_facts[candidate_index]
    intent = slot.intent
    metric_overlap = _token_overlap(slot.metric_tokens, fact.row_tokens)
    year_match = bool(intent.year and intent.year in fact.years)
    year_mismatch = bool(
        intent.year and fact.years and intent.year not in fact.years
    )
    role_match = bool(
        slot.expected_role and fact.role == slot.expected_role
    )
    role_mismatch = bool(
        slot.expected_role and fact.role and fact.role != slot.expected_role
    )
    entity_match, entity_mismatch, entity_explicit = (
        _cached_entity_compatibility(slot, fact)
    )
    if slot.expected_family == "currency_or_source":
        unit_match = bool(candidate.unit)
        unit_mismatch = False
    else:
        unit_match = bool(
            slot.expected_family
            and fact.unit_family == slot.expected_family
        )
        unit_mismatch = bool(
            slot.expected_family
            and fact.unit_family
            and fact.unit_family != slot.expected_family
        )
    scale_match = bool(
        slot.expected_scale is not None
        and fact.observed_scale == slot.expected_scale
    )
    scale_mismatch = bool(
        slot.expected_scale is not None
        and fact.observed_scale is not None
        and fact.observed_scale != slot.expected_scale
    )
    physical_count = (
        context.physical_counts.get(fact.physical_key, 0)
        if fact.physical_key is not None
        else 0
    )
    logical_count = (
        context.logical_counts.get(fact.logical_key, 0)
        if fact.logical_key is not None
        else 0
    )
    period_role_match = bool(
        intent.period_role and fact.period_role == intent.period_role
    )
    period_role_mismatch = bool(
        intent.period_role
        and fact.period_role
        and fact.period_role != intent.period_role
    )
    values = [
        float(intent.declared),
        metric_overlap,
        float(
            bool(slot.normalized_metric)
            and slot.normalized_metric == fact.normalized_row
        ),
        float(year_match),
        float(year_mismatch),
        float(role_match),
        float(role_mismatch),
        float(entity_match),
        float(entity_mismatch),
        float(unit_match),
        float(unit_mismatch),
        float(scale_match),
        float(scale_mismatch),
        float(fact.physical_key is not None),
        float(physical_count == 1),
        float(logical_count == 1),
        float(bool(fact.years or candidate.period)),
        float(bool(fact.role)),
        float(entity_explicit),
        float(bool(candidate.unit)),
        float(physical_count > 1),
        float(logical_count > 1),
        max(-1.0, min(1.0, float(candidate.relevance))),
        min(1.0, slot_index / 3.0),
        float(period_role_match),
        float(period_role_mismatch),
    ]
    mismatch = any(
        values[feature_index] > 0.5
        for feature_index in FINANCIAL_SLOT_MISMATCH_INDICES
    )
    return _pad(values, feature_size), mismatch


def build_financial_feature_matrices(
    plan: QuestionPlan,
    candidates: Sequence[NumericCandidate],
    *,
    slot_count: int = 4,
    candidate_feature_size: int = FINANCIAL_CANDIDATE_FEATURE_SIZE,
    slot_feature_size: int = FINANCIAL_SLOT_FEATURE_SIZE,
) -> FinancialFeatureMatrices:
    """Build all V13 features with one normalized/context scan per example."""

    context = build_financial_feature_context(
        plan, candidates, slot_count=slot_count
    )
    candidate_features = tuple(
        tuple(
            _cached_candidate_features(
                context, index, feature_size=candidate_feature_size
            )
        )
        for index in range(len(context.candidates))
    )
    slot_rows: list[tuple[tuple[float, ...], ...]] = []
    mismatch_rows: list[tuple[bool, ...]] = []
    for slot in range(slot_count):
        values: list[tuple[float, ...]] = []
        mismatches: list[bool] = []
        for candidate_index in range(len(context.candidates)):
            features, mismatch = _cached_slot_features(
                context,
                slot,
                candidate_index,
                feature_size=slot_feature_size,
            )
            values.append(tuple(features))
            mismatches.append(mismatch)
        slot_rows.append(tuple(values))
        mismatch_rows.append(tuple(mismatches))
    return FinancialFeatureMatrices(
        candidate_features=candidate_features,
        slot_features=tuple(slot_rows),
        slot_mismatch_mask=tuple(mismatch_rows),
        same_physical_mask=context.same_physical_mask,
    )
