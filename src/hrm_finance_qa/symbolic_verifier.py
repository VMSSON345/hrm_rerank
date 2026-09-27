from __future__ import annotations

from .physical_coordinates import LEGACY_SPAN, physical_coordinate_key, is_versioned_coordinate

from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
import re
from typing import Sequence

from .evidence_schema import EvidenceCell, EvidenceGraph, EvidenceRelation
from .direction_contract import assess_program_direction
from .financial_identities import (
    FinancialMetricRole,
    get_financial_identity,
    metric_alias_score,
    role_matches_metric,
    target_matches_metric,
)
from .ontology import normalize_text
from .relative_measurement import same_row_measurement_basis
from .source_assertions import source_assertion_matches_cell
from .report_context import report_field_matches
from .program_data import (
    _is_ratio_metric,
    COMMUTATIVE,
    ProgramDataError,
    ReasoningDAGProgram,
    ReasoningProgram,
    currency_unit_scale,
    is_sum_operation,
    operation_accepts_arity,
)
from .question_planner import (
    DirectionIntent,
    OperandSlot,
    QuestionPlan,
    QuestionProofContract,
    _has_financial_role_anchor,
    compile_question_proof_contract,
)


class VerificationState(str, Enum):
    VALID = "VALID"
    UNKNOWN = "UNKNOWN"
    INVALID = "INVALID"


@dataclass(frozen=True)
class ConstraintResult:
    name: str
    state: VerificationState


@dataclass(frozen=True)
class ColumnRoleProofAssessment:
    state: VerificationState
    action: str
    applicable: bool
    reason: str


@dataclass(frozen=True)
class PhysicalMapping:
    source_name: str
    page: int
    source_span: tuple[int, int]
    table_id: str
    row_id: str
    column_id: str
    canonical_identity: tuple[object, ...] = ()
    source_span_scope: str = LEGACY_SPAN
    source_line_sha256: str = ""
    line_number: int | None = None

    @property
    def identity(self) -> tuple[object, ...]:
        if self.canonical_identity:
            return self.canonical_identity
        return physical_coordinate_key(self) or ()


@dataclass(frozen=True)
class OperandMappingAssessment:
    slot_id: str
    selected_state: VerificationState
    valid_physical_mappings: tuple[PhysicalMapping, ...]
    unresolved_physical_mappings: tuple[PhysicalMapping, ...]
    selected_physical_is_valid: bool
    has_incomplete_candidate: bool = False
    conflict_reasons: tuple[str, ...] = ()
    equivalent_disclosures: bool = False
    equivalence_basis: str = ""
    equivalence_reasons: tuple[str, ...] = ()

    @property
    def semantic_mapping_count(self) -> int:
        return 1 if self.equivalent_disclosures else self.physical_mapping_count

    @property
    def physical_mapping_count(self) -> int:
        return len(
            {
                mapping.identity
                for mapping in (
                    *self.valid_physical_mappings,
                    *self.unresolved_physical_mappings,
                )
            }
        )


@dataclass(frozen=True)
class EvidenceResolutionAssessment:
    resolution: str
    operand_mappings: tuple[OperandMappingAssessment, ...]
    reason_codes: tuple[str, ...]
    selected_binding_state: VerificationState


@dataclass(frozen=True)
class ProofObligation:
    name: str
    applicability: str
    state: VerificationState | None


@dataclass(frozen=True)
class VerificationResult:
    state: VerificationState
    constraints: tuple[ConstraintResult, ...]
    proof_contract: QuestionProofContract | None = None
    obligation_states: tuple[ConstraintResult, ...] = ()
    mapping_assessment: EvidenceResolutionAssessment | None = None
    applicability_details: tuple[dict, ...] = ()

    @property
    def assessed_states(self) -> tuple[ConstraintResult, ...]:
        merged = {item.name: item for item in self.constraints}
        merged.update({item.name: item for item in self.obligation_states})
        return tuple(merged.values())

    @property
    def accepted(self) -> bool:
        """UNKNOWN evidence is eligible; only a contradiction is rejected."""

        return self.state is not VerificationState.INVALID

    @property
    def reasons(self) -> tuple[str, ...]:
        aliases = {
            "metric_match": "temporal_metric_mismatch",
            "period_coverage": "missing_requested_period",
            "reporting_entity": "reporting_entity_mismatch",
            "named_entity": "named_entity_mismatch",
            "denominator_safe": "zero_baseline_percent_change",
            "unit_algebra": "invalid_unit_algebra",
            "financial_identity": "financial_identity_mismatch",
            "column_role_match": "column_role_mismatch",
            "column_role_unique": "ambiguous_column_role_evidence",
            "logical_cell_unique": "duplicate_or_unproven_logical_cell",
            "source_provenance_complete": "incomplete_source_provenance",
            "answer_mode_complete": "incomplete_answer_mode",
            "reference_base_match": "reference_base_mismatch",
            "all_leaf_nodes_valid": "invalid_dag_leaf",
            "all_intermediate_nodes_valid": "invalid_dag_intermediate",
            "dag_execution_valid": "invalid_dag_execution",
        }
        reasons: list[str] = []
        for item in self.assessed_states:
            if item.state is not VerificationState.INVALID:
                continue
            reasons.append(aliases.get(item.name, item.name))
        return tuple(dict.fromkeys(reasons))

    @property
    def unknown_reasons(self) -> tuple[str, ...]:
        return tuple(
            item.name
            for item in self.assessed_states
            if item.state is VerificationState.UNKNOWN
        )

    @property
    def feature_vector(self) -> dict[str, float]:
        values = {
            VerificationState.INVALID: -1.0,
            VerificationState.UNKNOWN: 0.0,
            VerificationState.VALID: 1.0,
        }
        return {item.name: values[item.state] for item in self.assessed_states}

    @property
    def proof_score(self) -> float:
        if not self.assessed_states:
            return 0.0
        values = self.feature_vector.values()
        valid = sum(value > 0 for value in values)
        invalid = sum(value < 0 for value in values)
        return (valid - 2.0 * invalid) / len(self.assessed_states)

    @property
    def proof_obligations(self) -> tuple[ProofObligation, ...]:
        if self.proof_contract is None:
            return tuple(
                ProofObligation(item.name, "required", item.state)
                for item in self.constraints
            )
        states = {item.name: item.state for item in self.constraints}
        states.update(
            {item.name: item.state for item in self.obligation_states}
        )
        output: list[ProofObligation] = []
        for applicability, names in (
            ("required", self.proof_contract.required),
            ("optional", self.proof_contract.optional),
            ("not_applicable", self.proof_contract.not_applicable),
        ):
            output.extend(
                ProofObligation(
                    name,
                    applicability,
                    states.get(name, VerificationState.UNKNOWN)
                    if applicability != "not_applicable"
                    else None,
                )
                for name in names
            )
        return tuple(output)

    @property
    def proof_contract_state(self) -> VerificationState:
        required = [
            item.state
            for item in self.proof_obligations
            if item.applicability == "required"
        ]
        if not required:
            return self.state
        if any(state is VerificationState.INVALID for state in required):
            return VerificationState.INVALID
        if any(state is not VerificationState.VALID for state in required):
            return VerificationState.UNKNOWN
        return VerificationState.VALID

    @property
    def required_valid_fraction(self) -> float:
        required = [
            item for item in self.proof_obligations
            if item.applicability == "required"
        ]
        return (
            sum(item.state is VerificationState.VALID for item in required)
            / len(required)
            if required
            else 0.0
        )

    @property
    def required_unknown_fraction(self) -> float:
        required = [
            item for item in self.proof_obligations
            if item.applicability == "required"
        ]
        return (
            sum(item.state is VerificationState.UNKNOWN for item in required)
            / len(required)
            if required
            else 0.0
        )

    @property
    def optional_support_fraction(self) -> float:
        optional = [
            item for item in self.proof_obligations
            if item.applicability == "optional"
        ]
        return (
            sum(item.state is VerificationState.VALID for item in optional)
            / len(optional)
            if optional
            else 0.0
        )

    @property
    def rank_key(self) -> tuple[int, float, int, int]:
        valid = sum(
            item.state is VerificationState.VALID for item in self.assessed_states
        )
        unknown = sum(
            item.state is VerificationState.UNKNOWN for item in self.assessed_states
        )
        return (
            int(self.accepted),
            round(self.proof_score, 6),
            valid,
            -unknown,
        )

    @property
    def training_support_state(self) -> VerificationState:
        """Use only required obligations; runtime ``state`` stays unchanged."""

        return self.proof_contract_state


def _overall_state(
    constraints: Sequence[ConstraintResult],
) -> VerificationState:
    if any(item.state is VerificationState.INVALID for item in constraints):
        return VerificationState.INVALID
    if any(item.state is VerificationState.UNKNOWN for item in constraints):
        return VerificationState.UNKNOWN
    return VerificationState.VALID


_FIXED_SUM_ARITY = {"ADD": 2, "SUM3": 3, "SUM4": 4}


def _operation_matches_plan(
    expected: str,
    program: ReasoningProgram,
) -> bool:
    if expected == program.operation:
        return True
    if not is_sum_operation(expected) or not is_sum_operation(program.operation):
        return False
    if not operation_accepts_arity(program.operation, len(program.operands)):
        return False
    expected_arity = _FIXED_SUM_ARITY.get(expected)
    if expected_arity is not None:
        return len(program.operands) == expected_arity
    actual_arity = _FIXED_SUM_ARITY.get(program.operation)
    return actual_arity is None or len(program.operands) == actual_arity


def _sum_set_completeness_state(
    plan: QuestionPlan,
    program: ReasoningProgram,
    selected: Sequence[EvidenceCell],
) -> VerificationState:
    if not is_sum_operation(program.operation):
        return VerificationState.UNKNOWN
    if not operation_accepts_arity(program.operation, len(program.operands)):
        return VerificationState.INVALID
    intended_counts: set[int] = set()
    if plan.operation in _FIXED_SUM_ARITY:
        intended_counts.add(_FIXED_SUM_ARITY[plan.operation])
    if plan.operand_slots:
        intended_counts.add(len(plan.operand_slots))
    roles = plan.proof_operand_roles or plan.operand_roles
    if roles:
        intended_counts.add(len(roles))
    if len(intended_counts) > 1:
        return VerificationState.INVALID
    if not intended_counts:
        return VerificationState.UNKNOWN
    intended = next(iter(intended_counts))
    return (
        VerificationState.VALID
        if len(program.operands) == len(selected) == intended
        else VerificationState.INVALID
    )


def _assignment_state(
    states: Sequence[Sequence[VerificationState]],
) -> VerificationState:
    if not states or any(len(row) != len(states) for row in states):
        return VerificationState.INVALID
    valid = [
        [state is VerificationState.VALID for state in row]
        for row in states
    ]
    if _bounded_bipartite_assignments(valid, limit=1):
        return VerificationState.VALID
    non_invalid = [
        [state is not VerificationState.INVALID for state in row]
        for row in states
    ]
    return (
        VerificationState.UNKNOWN
        if _bounded_bipartite_assignments(non_invalid, limit=1)
        else VerificationState.INVALID
    )


def _bounded_bipartite_assignments(
    allowed: Sequence[Sequence[bool]],
    *,
    limit: int = 2,
) -> tuple[tuple[int, ...], ...]:
    """Return at most limit perfect assignments without factorial materialization."""

    size = len(allowed)
    if (
        size == 0
        or limit < 1
        or any(len(row) != size for row in allowed)
    ):
        return ()
    row_order = sorted(
        range(size),
        key=lambda row: sum(bool(value) for value in allowed[row]),
    )
    if any(not any(allowed[row]) for row in row_order):
        return ()
    assignment = [-1] * size
    output: list[tuple[int, ...]] = []

    def visit(depth: int, used_mask: int) -> None:
        if len(output) >= limit:
            return
        if depth == size:
            output.append(tuple(assignment))
            return
        row = row_order[depth]
        for column, permitted in enumerate(allowed[row]):
            if not permitted or used_mask & (1 << column):
                continue
            assignment[row] = column
            visit(depth + 1, used_mask | (1 << column))
            assignment[row] = -1
            if len(output) >= limit:
                return

    visit(0, 0)
    return tuple(output)


def _maximum_bottleneck_assignment(
    weights: Sequence[Sequence[float]],
) -> float:
    """Maximum possible minimum edge weight for a square assignment."""

    size = len(weights)
    if size == 0 or any(len(row) != size for row in weights):
        return 0.0
    states: dict[int, float] = {0: float("inf")}
    for row in range(size):
        next_states: dict[int, float] = {}
        for mask, score in states.items():
            for column in range(size):
                bit = 1 << column
                if mask & bit:
                    continue
                next_mask = mask | bit
                candidate = min(score, float(weights[row][column]))
                next_states[next_mask] = max(
                    next_states.get(next_mask, float("-inf")),
                    candidate,
                )
        states = next_states
    return states.get((1 << size) - 1, 0.0)


def _slot_period_state(
    plan: QuestionPlan,
    slot: OperandSlot,
    cell: EvidenceCell,
) -> VerificationState:
    """Match calendar years and relative statement endpoints independently."""

    requested_date = re.search(r"\b(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})\b", plan.question)
    if requested_date and source_assertion_matches_cell(cell):
        observed_date = re.search(r"\b(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})\b", cell.column)
        if not observed_date:
            return VerificationState.UNKNOWN
        if tuple(map(int, observed_date.groups())) != tuple(map(int, requested_date.groups())):
            return VerificationState.INVALID
    if plan.requested_value_column:
        # A named source column can identify a lookup without a calendar year.
        # If the question also names a period, the period checks still apply.
        if not cell.column:
            return VerificationState.UNKNOWN
        if normalize_text(cell.column) != plan.requested_value_column:
            return VerificationState.INVALID
        if not slot.year and not slot.period_role:
            return VerificationState.VALID
    if plan.requested_report_year:
        # An explicitly dated event is not an undated profile field. Keep it
        # for questions about that event, but not as an unresolved alternative
        # to the report-field lookup merely because its number is monetary.
        if (source_assertion_matches_cell(cell) and cell.period
                and not any(year != plan.requested_report_year for year in plan.requested_years)):
            return VerificationState.INVALID
        return (VerificationState.VALID if report_field_matches(cell) and cell.report_field["report_year"] == plan.requested_report_year
                else VerificationState.INVALID if report_field_matches(cell) else VerificationState.UNKNOWN)
    if not slot.year and not slot.period_role and report_field_matches(cell):
        # No temporal selector was requested for this proven undated field.
        # VALID means the binding meets the empty temporal requirement; it
        # does not supply a fact year or change period_coverage from N/A.
        return VerificationState.VALID
    # An undated report field has a report-year witness, not a fact period.
    # It cannot compete with a dated slot merely because its unit is usable.
    if (slot.year or slot.period_role) and report_field_matches(cell):
        return VerificationState.INVALID
    from .column_context import is_annual_operand_column
    if (slot.year or slot.period_role) and not is_annual_operand_column(" / ".join(cell.header_path) or cell.column, slot.metric):
        return VerificationState.INVALID
    states: list[ConstraintResult] = []
    if slot.year:
        years = cell.effective_years(plan)
        states.append(
            ConstraintResult(
                "slot_year",
                VerificationState.VALID
                if slot.year in years
                else VerificationState.INVALID
                if years
                else VerificationState.UNKNOWN,
            )
        )
    if slot.period_role:
        observed = cell.period_role
        states.append(
            ConstraintResult(
                "slot_period_role",
                VerificationState.VALID
                if observed == slot.period_role
                else VerificationState.INVALID
                if observed
                else VerificationState.UNKNOWN,
            )
        )
    if not states:
        return VerificationState.UNKNOWN
    return _overall_state(states)


def _operand_period_match_state(
    plan: QuestionPlan,
    selected: Sequence[EvidenceCell],
) -> VerificationState:
    if not selected or not (
        plan.requested_years or plan.requested_period_roles
    ):
        return VerificationState.UNKNOWN
    if plan.operand_slots and len(plan.operand_slots) == len(selected):
        matrix = tuple(
            tuple(
                _slot_period_state(plan, slot, cell)
                for slot in plan.operand_slots
            )
            for cell in selected
        )
        return _assignment_state(matrix)
    requested = set(plan.requested_years)
    requested_roles = set(plan.requested_period_roles)
    states = tuple(
        VerificationState.VALID
        if (
            requested
            and cell.effective_years(plan) & requested
            or requested_roles
            and cell.period_role in requested_roles
        )
        else VerificationState.INVALID
        if cell.effective_years(plan) or cell.period_role
        else VerificationState.UNKNOWN
        for cell in selected
    )
    return _overall_state(
        tuple(ConstraintResult("operand_period", state) for state in states)
    )


def _operand_entity_match_state(
    plan: QuestionPlan,
    selected: Sequence[EvidenceCell],
) -> VerificationState:
    if not selected:
        return VerificationState.UNKNOWN
    states: list[VerificationState] = []
    identities: set[str] = set()
    for cell in selected:
        identity = normalize_text(cell.entity_id or cell.entity)
        if identity:
            identities.add(identity)
        from .entity_ownership import query_owner_state
        owner_state = query_owner_state(plan, cell)
        if owner_state is not None:
            states.append(VerificationState(owner_state))
        elif plan.entity_mode == "reporting":
            states.append(
                VerificationState.INVALID
                if cell.embedded_entity
                else VerificationState.VALID
                if identity
                else VerificationState.UNKNOWN
            )
        else:
            observed = set(
                normalize_text(
                    f"{cell.entity_id} {cell.entity} {cell.section} "
                    f"{cell.source_name}"
                ).split()
            )
            states.append(
                VerificationState.VALID
                if plan.entity_terms and observed.intersection(plan.entity_terms)
                else VerificationState.INVALID
                if identity
                else VerificationState.UNKNOWN
            )
    if len(identities) > 1:
        return VerificationState.INVALID
    return _overall_state(
        tuple(ConstraintResult("operand_entity", state) for state in states)
    )


def _metric_obligation_state(
    expected: str, observed: str
) -> VerificationState:
    if not expected or not observed:
        return VerificationState.UNKNOWN
    from .program_data import metric_qualifiers_compatible
    from .metric_semantics import normalize_metric_text
    expected_normalized = normalize_metric_text(expected)
    observed_normalized = normalize_metric_text(observed)
    if (not metric_qualifiers_compatible(expected, observed)
            or not metric_qualifiers_compatible(expected_normalized, observed_normalized)):
        return VerificationState.INVALID
    from .metric_semantics import relation
    meaning = relation(expected, observed)
    if meaning == "SAME": return VerificationState.VALID
    if meaning == "RELATED": return VerificationState.UNKNOWN
    if meaning == "DIFFERENT": return VerificationState.INVALID
    score = metric_alias_score(observed_normalized, (expected_normalized,))
    if score >= 0.72:
        return VerificationState.VALID
    if score <= 0.10:
        return VerificationState.INVALID
    return VerificationState.UNKNOWN


def _cell_metric_obligation_state(expected: str, cell: EvidenceCell) -> VerificationState:
    # Referential company-field wording is equivalent only with its validated
    # field witness. Other owners, dates and scopes are checked independently.
    from .metric_semantics import identity, report_field_identity
    if report_field_matches(cell) and identity(expected) == report_field_identity(cell.metric_label):
        return VerificationState.VALID
    return _metric_obligation_state(expected, cell.metric_label)


def _query_metric_obligation_state(
    plan: QuestionPlan,
    graph: EvidenceGraph,
    selected: Sequence[EvidenceCell],
) -> VerificationState:
    """Verify a query metric from row text and provenanced header context."""

    if len(selected) != 1:
        return VerificationState.UNKNOWN

    def assess(cell: EvidenceCell) -> tuple[VerificationState, bool]:
        row_state = _cell_metric_obligation_state(plan.metric_text, cell)
        if row_state is not VerificationState.UNKNOWN or not cell.metric_label:
            return row_state, False
        if (
            not cell.header_path
            or len(cell.header_path) != len(cell.header_path_provenance)
            or not all(cell.header_path_provenance)
            or cell.column_role_conflict
        ):
            return VerificationState.UNKNOWN, False
        normalized_expected = normalize_text(plan.metric_text)
        normalized_row = normalize_text(cell.metric_label)
        row_tokens = set(normalized_row.split()) - {
            "1",
            "2",
            "3",
            "4",
            "cac",
            "chi",
            "cua",
            "khoan",
            "muc",
        }
        expected_tokens = set(normalized_expected.split())
        header_tokens = set(normalize_text(" ".join(cell.header_path)).split())
        relative_period_corroboration = bool(
            {"dau", "cuoi"} & expected_tokens & header_tokens
        )
        if (
            len(row_tokens) >= 3
            and row_tokens.issubset(expected_tokens)
            and relative_period_corroboration
            and _metric_obligation_state(cell.metric_label, plan.metric_text)
            is VerificationState.VALID
        ):
            return VerificationState.VALID, True
        composite = " ".join(
            part
            for part in (
                cell.section,
                *cell.header_path,
                cell.metric_label,
            )
            if part
        )
        return _metric_obligation_state(plan.metric_text, composite), True

    def explicit_components(cell: EvidenceCell) -> frozenset[str]:
        expected = f" {normalize_text(plan.metric_text)} "
        components: set[str] = set()
        for raw_component in (cell.section, *cell.header_path, cell.metric_label):
            component = normalize_text(raw_component)
            component = re.sub(r"\b(?:19|20)\d{2}\b", " ", component)
            component = re.sub(r"\s+", " ", component).strip()
            semantic_tokens = set(component.split()) - {
                "chi",
                "gia",
                "nam",
                "so",
                "tai",
                "tri",
            }
            if len(semantic_tokens) >= 2 and f" {component} " in expected:
                components.add(component)
        return frozenset(components)

    selected_state, used_composite = assess(selected[0])
    if selected_state is not VerificationState.VALID or not used_composite:
        return selected_state
    selected_components = explicit_components(selected[0])
    requested = set(plan.requested_years)
    requested_period_roles = set(plan.requested_period_roles)
    selected_key = _logical_cell_key(plan, selected[0])
    for cell in graph.cells:
        if cell.index == selected[0].index:
            continue
        state, _ = assess(cell)
        if state is not VerificationState.VALID:
            continue
        years = cell.effective_years(plan)
        if requested and years and requested.isdisjoint(years):
            continue
        if (
            requested_period_roles
            and cell.period_role
            and cell.period_role not in requested_period_roles
        ):
            continue
        if plan.column_roles and cell.column_role:
            if cell.column_role not in {role.lower() for role in plan.column_roles}:
                continue
        other_key = _logical_cell_key(plan, cell)
        if selected_key is not None and selected_key == other_key:
            continue
        competing_components = explicit_components(cell)
        if (
            selected_components
            and competing_components < selected_components
        ):
            continue
        return VerificationState.UNKNOWN
    return VerificationState.VALID


def _operand_role_state(
    plan: QuestionPlan,
    operation: str,
    selected: Sequence[EvidenceCell],
) -> VerificationState:
    labels = tuple(cell.metric_label for cell in selected)
    role_hypotheses = (
        plan.addition_role_hypotheses
        if operation == "ADD" and plan.addition_role_hypotheses
        else (plan.proof_operand_roles or plan.operand_roles,)
    )
    role_hypotheses = tuple(
        roles for roles in role_hypotheses if len(roles) == len(labels)
    )
    if not role_hypotheses or not all(labels):
        return VerificationState.UNKNOWN
    def role_score(role: str, label: str) -> float:
        forward = metric_alias_score(label, (role,))
        if operation != "ADD":
            return forward
        normalized_role = normalize_text(role)
        normalized_label = normalize_text(label)
        semantic_role_tokens = {
            token
            for token in normalized_role.split()
            if token not in {"cac", "cua", "khoan", "va"}
            and not token.isdigit()
        }
        if (
            " va " in f" {normalized_role} "
            and not semantic_role_tokens.issubset(set(normalized_label.split()))
        ):
            return 0.0
        # A single disclosed row must not prove a wider conjunction merely
        # because its compact text is a substring of the requested role.
        return min(forward, metric_alias_score(role, (label,)))

    best_assignment_score = 0.0
    has_assignment_scores = False
    for roles in role_hypotheses:
        has_assignment_scores = True
        if operation in COMMUTATIVE:
            weights = [
                [role_score(role, label) for label in labels]
                for role in roles
            ]
            score = _maximum_bottleneck_assignment(weights)
        else:
            score = min(
                (
                    role_score(role, label)
                    for role, label in zip(roles, labels)
                ),
                default=0.0,
            )
        best_assignment_score = max(best_assignment_score, score)
    if best_assignment_score >= 0.72:
        if operation == "ADD":
            requested_relative_roles = set(
                re.findall(
                    r"\b(?:cuoi|dau)\s+(?:nam|ky)\b|"
                    r"\bnam\s+(?:nay|hien tai|truoc)\b",
                    plan.normalized,
                )
            )
            if len(requested_relative_roles) == 1:
                requested = next(iter(requested_relative_roles))
                if requested.startswith("cuoi "):
                    requested_endpoint = "end"
                elif requested.startswith("dau "):
                    requested_endpoint = "start"
                elif requested in {"nam nay", "nam hien tai"}:
                    requested_endpoint = "current"
                else:
                    requested_endpoint = "prior"
                observed_endpoints: list[str] = []
                for cell in selected:
                    observed = normalize_text(cell.period)
                    if re.search(r"\bcuoi\s+(?:nam|ky)\b|\b31\s*/\s*12\b", observed):
                        observed_endpoints.append("end")
                    elif re.search(r"\bdau\s+(?:nam|ky)\b|\b0?1\s*/\s*0?1\b", observed):
                        observed_endpoints.append("start")
                    elif observed in {"nam nay", "nam hien tai"}:
                        observed_endpoints.append("current")
                    elif observed == "nam truoc":
                        observed_endpoints.append("prior")
                    else:
                        observed_endpoints.append("")
                if any(
                    endpoint and endpoint != requested_endpoint
                    for endpoint in observed_endpoints
                ):
                    return VerificationState.INVALID
                if not all(observed_endpoints):
                    return VerificationState.UNKNOWN
        return VerificationState.VALID
    if has_assignment_scores and best_assignment_score <= 0.10:
        return VerificationState.INVALID
    return VerificationState.UNKNOWN


def _same_year_column_ambiguity(
    plan: QuestionPlan,
    selected: Sequence[EvidenceCell],
    cells: Sequence[EvidenceCell],
) -> bool:
    """Detect an unasked-for endpoint choice within one requested year."""

    requested = set(plan.requested_years)
    if len(requested) != 1:
        return False
    normalized_question = plan.normalized
    if (
        re.search(r"\b(?:cuoi|dau)\s+(?:nam|ky)\b", normalized_question)
        or re.search(r"\b(?:0?1\s+0?1|31\s+12)\b", normalized_question)
    ):
        return False
    requested_year = next(iter(requested))
    for chosen in selected:
        chosen_metric = chosen.metric_key or normalize_text(chosen.metric_label)
        chosen_column = chosen.column_id or normalize_text(chosen.column)
        if not chosen_metric or not chosen.table_id or not chosen_column:
            continue
        for alternative in cells:
            if alternative.index == chosen.index or alternative.value == chosen.value:
                continue
            alternative_metric = (
                alternative.metric_key or normalize_text(alternative.metric_label)
            )
            alternative_column = (
                alternative.column_id or normalize_text(alternative.column)
            )
            if (
                alternative_metric == chosen_metric
                and alternative.source_name == chosen.source_name
                and alternative.page == chosen.page
                and alternative.table_id == chosen.table_id
                and normalize_text(alternative.entity_id or alternative.entity)
                == normalize_text(chosen.entity_id or chosen.entity)
                and alternative_column
                and alternative_column != chosen_column
                and requested_year in alternative.effective_years(plan)
                and requested_year in chosen.effective_years(plan)
            ):
                return True
    return False


def _column_role_obligation_state(
    plan: QuestionPlan,
    operation: str,
    selected: Sequence[EvidenceCell],
) -> VerificationState:
    required = tuple(role.lower() for role in plan.column_roles)
    if not required:
        return VerificationState.UNKNOWN
    if (
        len(required) == 1
        and len(selected) > 1
        and plan.operation == "SELECT"
        and operation != "SELECT"
        and plan.financial_identity_ids
    ):
        # A disclosed target lane governs every component when the target is
        # reconstructed from a catalog identity. It does not authorize an
        # ACTUAL/PLAN default: the single role remains question-explicit.
        required = required * len(selected)
    if len(required) != len(selected):
        return VerificationState.INVALID
    observed = tuple(cell.column_role for cell in selected)
    if any(cell.column_role_conflict for cell in selected):
        return VerificationState.INVALID
    if any(not role for role in observed):
        return VerificationState.UNKNOWN
    if (
        Counter(required) == Counter(observed)
        if operation in COMMUTATIVE
        else required == observed
    ):
        return VerificationState.VALID
    return VerificationState.INVALID


def _column_role_logical_key(
    plan: QuestionPlan,
    cell: EvidenceCell,
) -> tuple[object, ...] | None:
    years = tuple(sorted(cell.effective_years(plan)))
    period_identity: tuple[str, ...] = years or (
        (f"role:{cell.period_role}",) if cell.period_role else ()
    )
    metric = cell.metric_key or normalize_text(cell.metric_label)
    row = cell.row_id or metric
    if (
        not cell.source_name
        or not cell.table_id
        or not row
        or not metric
        or not period_identity
    ):
        return None
    return (
        cell.source_name,
        cell.table_id,
        row,
        metric,
        period_identity,
        normalize_text(cell.entity_id or cell.entity),
        cell.unit_key,
    )


def _column_role_physical_key(cell: EvidenceCell) -> tuple[object, ...]:
    if is_versioned_coordinate(cell):
        return physical_coordinate_key(cell) or ("unresolved", cell.index)
    if cell.source_span is None:
        return ("cell", cell.index)
    return (
        "span",
        cell.source_name,
        cell.page,
        cell.source_span,
        str(cell.value),
    )


def assess_column_role_proof(
    plan: QuestionPlan,
    evidence: EvidenceGraph | Sequence[EvidenceCell],
    *,
    selected_indices: Sequence[int] | None = None,
    operation: str | None = None,
) -> ColumnRoleProofAssessment:
    """Assess explicit role match or implicit role uniqueness from evidence.

    Candidate scores and order are deliberately absent. For a no-role query,
    distinct roles are counted only inside the same logical identity after
    physical-equivalent evidence is checked for extractor disagreement.
    """

    graph = (
        evidence
        if isinstance(evidence, EvidenceGraph)
        else EvidenceGraph(cells=tuple(evidence), edges=())
    )
    by_index = {cell.index: cell for cell in graph.cells}
    if selected_indices is None:
        selected = tuple(graph.cells)
    else:
        selected = tuple(
            by_index[index] for index in selected_indices if index in by_index
        )

    if (selected and not plan.column_roles and not plan.requested_value_column
            and all(report_field_matches(cell) or source_assertion_matches_cell(cell) for cell in selected)):
        return ColumnRoleProofAssessment(
            VerificationState.UNKNOWN, "DEFER_TO_OTHER_PROOFS", False,
            "source_bound_field_has_no_table_column",
        )

    if plan.column_roles:
        state = _column_role_obligation_state(
            plan,
            operation or plan.operation or "SELECT",
            selected,
        )
        action = (
            "DEFER_TO_OTHER_PROOFS"
            if state is VerificationState.VALID
            else "REJECT"
            if state is VerificationState.INVALID
            else "ABSTAIN"
        )
        return ColumnRoleProofAssessment(
            state=state,
            action=action,
            applicable=True,
            reason="explicit_query_column_role",
        )

    selected_keys = {
        key
        for cell in selected
        if (key := _column_role_logical_key(plan, cell)) is not None
    }
    relevant = tuple(
        cell
        for cell in graph.cells
        if _column_role_logical_key(plan, cell) in selected_keys
        and _query_section_state(plan, cell) is not VerificationState.INVALID
        and _query_accounting_scope_state(plan, cell) is not VerificationState.INVALID
    )
    if any(cell.column_role_conflict for cell in relevant):
        return ColumnRoleProofAssessment(
            VerificationState.INVALID,
            "ABSTAIN",
            True,
            "cross_extractor_role_conflict",
        )

    physical_roles: dict[tuple[object, ...], set[str]] = {}
    for cell in relevant:
        physical_roles.setdefault(_column_role_physical_key(cell), set()).add(
            cell.column_role
        )
    if any(len({role for role in roles if role}) > 1 for roles in physical_roles.values()):
        return ColumnRoleProofAssessment(
            VerificationState.INVALID,
            "ABSTAIN",
            True,
            "physical_role_conflict",
        )

    logical_roles: dict[tuple[object, ...], set[str]] = {}
    for cell in relevant:
        key = _column_role_logical_key(plan, cell)
        if key is None:
            continue
        logical_roles.setdefault(key, set()).add(cell.column_role)
    explicit_logical_roles = {
        key: {role for role in roles if role}
        for key, roles in logical_roles.items()
    }
    if any(len(roles) > 1 for roles in explicit_logical_roles.values()):
        return ColumnRoleProofAssessment(
            VerificationState.INVALID,
            "ABSTAIN",
            True,
            "multiple_column_roles_for_logical_identity",
        )

    relevant_table_keys = {
        (cell.source_name, cell.table_id)
        for cell in relevant
        if cell.source_name and cell.table_id
    }
    scoped_inventories = tuple(
        item
        for item in graph.table_role_inventories
        if (item.source_name, item.table_id) in relevant_table_keys
    )
    if scoped_inventories:
        inventory = (
            "EXPLICIT_COMPLETE"
            if all(
                item.state == "EXPLICIT_COMPLETE"
                for item in scoped_inventories
            )
            else "UNKNOWN"
        )
    elif graph.table_role_inventories:
        inventory = ""
    else:
        inventory = graph.column_role_inventory_state
    if inventory == "UNKNOWN":
        return ColumnRoleProofAssessment(
            VerificationState.UNKNOWN,
            "ABSTAIN",
            True,
            "column_role_inventory_unknown",
        )
    if inventory == "EXPLICIT_COMPLETE":
        state = (
            VerificationState.VALID
            if explicit_logical_roles
            and all(len(roles) == 1 for roles in explicit_logical_roles.values())
            else VerificationState.UNKNOWN
        )
        return ColumnRoleProofAssessment(
            state,
            "DEFER_TO_OTHER_PROOFS" if state is VerificationState.VALID else "ABSTAIN",
            True,
            "unique_complete_column_role_inventory" if logical_roles else "missing_logical_role_group",
        )
    return ColumnRoleProofAssessment(
        VerificationState.UNKNOWN,
        "DEFER_TO_OTHER_PROOFS",
        False,
        "column_role_inventory_not_declared",
    )


def _unit_compatibility_state(
    plan: QuestionPlan,
    operation: str,
    selected: Sequence[EvidenceCell],
) -> VerificationState:
    if len(selected) < 2:
        return VerificationState.UNKNOWN
    units = tuple(cell.unit_key for cell in selected)
    if not all(units):
        if any(is_versioned_coordinate(cell) for cell in selected):
            return (VerificationState.VALID
                    if (operation != "DIVIDE" or plan.requested_unit_family == "ratio")
                    and same_row_measurement_basis(operation, selected)
                    else VerificationState.UNKNOWN)
        # For a ratio, two physical cells in the same explicit table row share
        # the row's measurement basis even when OCR loses the printed unit
        # label.  This proves compatibility (the unit cancels), not the name or
        # scale of that missing unit.  Different rows/tables remain UNKNOWN.
        ratio_operation = operation in {"RATIO_PERCENT", "PERCENT_CHANGE"} or (
            operation == "DIVIDE" and plan.requested_unit_family == "ratio"
        )
        same_explicit_row = bool(
            ratio_operation
            and all(not unit for unit in units)
            and len(selected) == 2
            and selected[0].source_name
            and selected[0].source_name == selected[1].source_name
            and selected[0].table_id
            and selected[0].table_id == selected[1].table_id
            and selected[0].row_id
            and selected[0].row_id == selected[1].row_id
        )
        if same_explicit_row:
            return VerificationState.VALID
        return VerificationState.UNKNOWN
    if len(set(units)) == 1:
        if (
            units[0] in {"ratio", "percent"}
            and operation
            in {"ADD", "SUBTRACT", "AVERAGE", "SUM3", "SUM4", "SUM_N"}
            and plan.requested_unit_family not in {"ratio", "percent"}
        ):
            return VerificationState.INVALID
        return VerificationState.VALID
    if (
        operation == "DIVIDE"
        and units == ("currency", "share_count")
        and plan.requested_unit_family in {"", "currency"}
    ):
        return VerificationState.VALID
    if operation in {"RATIO_PERCENT", "PERCENT_CHANGE"} or (
        operation == "DIVIDE" and plan.requested_unit_family == "ratio"
    ):
        return VerificationState.INVALID
    if is_sum_operation(operation):
        return VerificationState.INVALID
    return VerificationState.UNKNOWN


def _unit_scale_match_state(
    plan: QuestionPlan,
    operation: str,
    selected: Sequence[EvidenceCell],
) -> VerificationState:
    requested = plan.requested_scale
    if requested is None:
        return VerificationState.UNKNOWN
    if requested == "conflict" or plan.requested_unit_family in {"percent", "ratio"}:
        return VerificationState.INVALID

    expected_unit = {
        "vnd": "VND",
        "thousand_vnd": "nghìn đồng",
        "million_vnd": "triệu đồng",
        "billion_vnd": "tỷ đồng",
    }.get(requested)
    if expected_unit is None or not selected:
        return VerificationState.UNKNOWN
    expected_scale = currency_unit_scale(expected_unit)

    observed_scales = []
    missing_scale = False
    for cell in selected:
        scale = currency_unit_scale(cell.unit)
        if scale is not None:
            observed_scales.append(scale)
            continue
        if (
            operation in {"MULTIPLY", "DIVIDE"}
            and cell.unit_key == "share_count"
        ):
            continue
        if cell.unit_key in {"", "currency", "currency_per_share"}:
            missing_scale = True
            continue
        return VerificationState.INVALID

    if missing_scale or not observed_scales:
        return VerificationState.UNKNOWN
    if operation in {
        "SELECT",
        "ADD",
        "SUBTRACT",
        "AVERAGE",
        "SUM3",
        "SUM4",
        "SUM_N",
    }:
        # Runtime replays every explicitly scaled monetary leaf in canonical
        # VND, then converts the result to the requested target scale.
        return VerificationState.VALID
    if len(set(observed_scales)) > 1:
        return VerificationState.INVALID
    # A missing display unit remains UNKNOWN: never infer scale from value
    # magnitude, company identity, or a neighboring unrelated table.
    if requested == "vnd":
        return VerificationState.VALID
    return (
        VerificationState.VALID
        if observed_scales[0] == expected_scale
        else VerificationState.UNKNOWN
    )


def _logical_cell_key(
    plan: QuestionPlan, cell: EvidenceCell
) -> tuple[object, ...] | None:
    if report_field_matches(cell):
        return ("report_field", physical_coordinate_key(cell), cell.report_field["report_year"], cell.entity_id)
    if source_assertion_matches_cell(cell):
        return ("source_assertion", physical_coordinate_key(cell), cell.period, cell.entity_id)
    years = tuple(sorted(cell.effective_years(plan)))
    period_identity: tuple[str, ...] = years or (
        (f"role:{cell.period_role}",) if cell.period_role else ()
    )
    if (not period_identity and plan.requested_value_column
            and normalize_text(cell.column) == plan.requested_value_column):
        # The explicitly requested source column identifies this cell without
        # manufacturing a calendar year. Its column/provenance proofs remain required.
        period_identity = ("named_column:" + normalize_text(cell.column),)
    entity = normalize_text(cell.entity_id or cell.entity)
    if not entity and not cell.embedded_entity:
        entity = "reporting"
    role = cell.column_role
    role_required = bool(plan.column_roles or any(slot.column_role for slot in plan.operand_slots))
    if (not role and not role_required and not cell.column_role_conflict
            and is_versioned_coordinate(cell) and physical_coordinate_key(cell) is not None):
        # Identity sentinel, not an inferred ACTUAL/PLAN semantic role.
        role = "__role_unspecified__"
    required = (
        cell.source_name,
        cell.page,
        cell.table_id,
        cell.row_id,
        cell.column_id,
        cell.metric_key,
        period_identity,
        role,
        entity,
    )
    if any(value in {None, "", ()} for value in required):
        return None
    # Unit is an observed property, not part of physical/logical cell
    # identity. Unit compatibility is proved independently. Requiring it here
    # would make an otherwise exact source/table/row/column identity UNKNOWN
    # merely because a Markdown page omitted its display unit.
    return required


def _physical_cell_key(cell: EvidenceCell) -> tuple[object, ...] | None:
    return physical_coordinate_key(cell)


def _structural_physical_key(
    cell: EvidenceCell,
) -> tuple[object, ...] | None:
    if is_versioned_coordinate(cell):
        return physical_coordinate_key(cell)
    if (
        not cell.source_name
        or cell.page is None
        or not cell.table_id
        or not cell.row_id
        or not cell.column_id
    ):
        return None
    return (
        "structural",
        cell.source_name,
        cell.page,
        cell.table_id,
        cell.subtable_id,
        cell.row_id,
        cell.column_id,
    )


def _same_physical_evidence(left: EvidenceCell, right: EvidenceCell) -> bool:
    left_span = _physical_cell_key(left)
    right_span = _physical_cell_key(right)
    if left_span is not None and left_span == right_span:
        return True
    if is_versioned_coordinate(left) or is_versioned_coordinate(right):
        return False
    left_structural = _structural_physical_key(left)
    right_structural = _structural_physical_key(right)
    return (
        left_structural is not None
        and left_structural == right_structural
    )


def _equivalent_disclosures(cells: Sequence[EvidenceCell]) -> bool:
    from .disclosure_equivalence import compare_disclosures
    return compare_disclosures(cells)["equivalent"]


def _logical_cell_unique_state(
    plan: QuestionPlan,
    graph: EvidenceGraph,
    selected: Sequence[EvidenceCell],
) -> VerificationState:
    keys = [_logical_cell_key(plan, cell) for cell in selected]
    if any(key is None for key in keys):
        return VerificationState.UNKNOWN
    complete_keys = [key for key in keys if key is not None]
    if len(complete_keys) != len(set(complete_keys)):
        return VerificationState.INVALID
    for key in complete_keys:
        renditions = [
            cell for cell in graph.cells if _logical_cell_key(plan, cell) == key
        ]
        physical = {_physical_cell_key(cell) for cell in renditions}
        if None in physical:
            return VerificationState.UNKNOWN
        if len(physical) > 1:
            return VerificationState.INVALID
        if len({cell.value for cell in renditions}) > 1:
            return VerificationState.INVALID
    return VerificationState.VALID


def _source_provenance_state(
    selected: Sequence[EvidenceCell],
    *, check_number_format: bool = True,
) -> VerificationState:
    if not selected:
        return VerificationState.UNKNOWN
    for cell in selected:
        from .column_context import column_kind, resolved_relative_year
        if ((cell.table_semantics or {}).get("statement_period")
                and column_kind(cell.column) == "relative_year" and not resolved_relative_year(cell)):
            return VerificationState.INVALID
        from .table_semantics import bound_column_context
        if ((cell.table_semantics or {}).get('period_origin')=='local_table_context'
                and bound_column_context(cell)[0]!=cell.period):
            return VerificationState.INVALID
        from .unit_defaults import valid_assumption
        if not valid_assumption(cell):
            return VerificationState.INVALID
        from .number_formats import resolution_state
        numeric_state = resolution_state(cell)
        from .entity_ownership import validate_owner
        ownership = validate_owner(cell)
        if ownership is not None and ownership != "VALID": return VerificationState(ownership)
        if check_number_format and numeric_state != "VALID": return VerificationState(numeric_state)
        if source_assertion_matches_cell(cell) or report_field_matches(cell):
            if cell.column_role_conflict:
                return VerificationState.INVALID
            continue
        if (
            not cell.source_name
            or cell.page is None
            or not cell.table_id
            or not cell.row_id
            or not cell.column_id
            or cell.source_span is None
        ):
            return VerificationState.UNKNOWN
        if cell.column_role_conflict:
            return VerificationState.INVALID
    return VerificationState.VALID


def _slot_binding_state(
    plan: QuestionPlan,
    cells: Sequence[EvidenceCell],
    slot_bindings: Sequence[tuple[str, int]],
    *,
    allow_missing_unit: bool = False,
) -> VerificationState:
    expected = {slot.id: slot for slot in plan.operand_slots}
    bindings = dict(slot_bindings)
    if set(bindings) != set(expected) or len(bindings) != len(slot_bindings):
        return VerificationState.INVALID
    states: list[VerificationState] = []
    for slot_id, slot in expected.items():
        index = bindings[slot_id]
        if not 0 <= index < len(cells):
            return VerificationState.INVALID
        states.append(
            _slot_candidate_state(
                plan,
                slot,
                cells[index],
                allow_missing_unit=allow_missing_unit,
            )
        )
    return _overall_state(
        tuple(ConstraintResult("slot", state) for state in states)
    )


def _column_scope_constraint_state(plan: QuestionPlan, cell) -> VerificationState | None:
    """Only reject an explicit contradictory column; never promote its proof.

    A table's shared HEADER can mention both scopes. Read only complete scope
    labels in the selected column path, not beneficiary names in row labels.
    """
    requested = set()
    if plan.entity_text == "parent_company":
        requested.add("parent")
    scope = plan.requested_accounting_scope
    if scope in {"separate", "consolidated"}:
        requested.add("parent" if scope == "separate" else "consolidated")
    if len(requested) != 1 or scope == "conflict":
        return None
    column = getattr(cell, "column", "") or getattr(cell, "column_label", "")
    labels = {re.sub(r"\b(?:nam|year|th|kh|(?:19|20)\d{2})\b", " ",
                     normalize_text(part)).strip() for part in column.split("/")}
    observed = {kind for label, kind in (("cong ty me", "parent"), ("parent company", "parent"),
                                        ("hop nhat", "consolidated"), ("consolidated", "consolidated"))
                if label in labels}
    if len(observed) > 1:
        return VerificationState.UNKNOWN
    return VerificationState.INVALID if observed and observed != requested else None


def _query_accounting_scope_state(plan: QuestionPlan, cell: EvidenceCell) -> VerificationState:
    column_state = _column_scope_constraint_state(plan, cell)
    if column_state is not None:
        return column_state
    requested = plan.requested_accounting_scope
    if not requested:
        return VerificationState.VALID
    semantic = cell.table_semantics or {}
    if source_assertion_matches_cell(cell):
        assertion_scope = cell.source_assertion["binding"].get("accounting_scope")
        if assertion_scope:
            return VerificationState.VALID if assertion_scope == requested else VerificationState.INVALID
    if requested == "conflict" or semantic.get("scope_conflict"):
        return VerificationState.INVALID
    scope = semantic.get("accounting_scope")
    if not scope or not semantic.get("scope_evidence"):
        return VerificationState.UNKNOWN
    return VerificationState.VALID if scope == requested else VerificationState.INVALID


def _query_source_frame_state(plan: QuestionPlan, cell: EvidenceCell) -> VerificationState:
    kind=plan.requested_source_kind
    if plan.requested_event:
        if not source_assertion_matches_cell(cell):
            return VerificationState.INVALID if report_field_matches(cell) or cell.table_id else VerificationState.UNKNOWN
        observed=cell.source_assertion["binding"].get("event_kind")
        if observed!=plan.requested_event:
            return VerificationState.INVALID if observed else VerificationState.UNKNOWN
    if not kind:
        return VerificationState.VALID
    field=report_field_matches(cell)
    narrative=source_assertion_matches_cell(cell)
    if kind=="table" and (field or narrative or not cell.table_id):
        return VerificationState.INVALID if field or narrative else VerificationState.UNKNOWN
    if kind=="profile" and not field:
        return VerificationState.INVALID if cell.table_id or narrative else VerificationState.UNKNOWN
    if kind=="narrative" and not narrative:
        return VerificationState.INVALID if cell.table_id or field else VerificationState.UNKNOWN
    if plan.requested_source_years or plan.requested_source_year_count:
        years=set(re.findall(r"\b(?:19|20)\d{2}\b", " ".join(cell.raw_header)))
        if not years:
            header = normalize_text(" ".join(cell.raw_header))
            if ("nam nay" in header and "nam truoc" in header
                    and (len(plan.requested_source_years) > 2 or plan.requested_source_year_count > 2)):
                return VerificationState.INVALID
            return VerificationState.UNKNOWN
        if plan.requested_source_years and years!=set(plan.requested_source_years):
            return VerificationState.INVALID
        if plan.requested_source_year_count and len(years)!=plan.requested_source_year_count:
            return VerificationState.INVALID
    heading=plan.requested_source_heading
    if heading:
        path=" / ".join(cell.source_section_path)
        if heading not in path:
            return VerificationState.UNKNOWN if not path else VerificationState.INVALID
    return VerificationState.VALID


def _query_section_state(plan: QuestionPlan, cell: EvidenceCell) -> VerificationState:
    if not plan.requested_section:
        return VerificationState.VALID
    if not cell.source_section_path:
        return VerificationState.UNKNOWN
    return (VerificationState.VALID if plan.requested_section in cell.source_section_path
            else VerificationState.INVALID)


def _slot_candidate_state(
    plan: QuestionPlan,
    slot: OperandSlot,
    cell: EvidenceCell,
    *,
    allow_missing_unit: bool = False,
) -> VerificationState:
    states = [_cell_metric_obligation_state(slot.metric, cell), _query_section_state(plan, cell), _query_accounting_scope_state(plan, cell), _query_source_frame_state(plan, cell)]
    if any((cell.table_semantics or {}).get(key) for key in ("entity_conflict", "scope_conflict")):
        states.append(VerificationState.INVALID)
    states.append(_slot_period_state(plan, slot, cell))
    if not slot.column_role:
        # Role-unspecified means every non-conflicting physical lane remains a
        # candidate. Mapping cardinality, rather than column order, decides
        # whether the evidence is unique or ambiguous.
        states.append(
            VerificationState.INVALID
            if cell.column_role_conflict
            else VerificationState.VALID
        )
    else:
        states.append(
            VerificationState.VALID
            if cell.column_role == slot.column_role.lower()
            else VerificationState.INVALID
            if cell.column_role or cell.column_role_conflict
            else VerificationState.UNKNOWN
        )
    from .entity_ownership import query_owner_state
    owner_state = query_owner_state(plan, cell)
    if owner_state is not None:
        states.append(VerificationState(owner_state))
    elif slot.entity_scope == "reporting":
        states.append(
            VerificationState.INVALID
            if cell.embedded_entity
            else VerificationState.VALID
        )
    else:
        observed_entity = normalize_text(
            f"{cell.entity_id} {cell.entity} {cell.section} {cell.source_name}"
        )
        expected_terms = set(normalize_text(slot.entity_scope).split()) - {
            "named",
            "company",
            "corporation",
            "corp",
            "group",
            "cong",
            "ty",
        }
        explicit_entity = normalize_text(cell.entity_id or cell.entity)
        states.append(
            VerificationState.VALID
            if expected_terms.intersection(observed_entity.split())
            else VerificationState.INVALID
            if explicit_entity
            else VerificationState.UNKNOWN
        )
    if slot.unit_family == "currency_or_source":
        states.append(
            VerificationState.VALID if cell.unit_key else VerificationState.UNKNOWN
        )
    else:
        states.append(
            VerificationState.VALID
            if cell.unit_key == slot.unit_family
            else VerificationState.VALID
            if allow_missing_unit and not cell.unit_key
            else VerificationState.INVALID
            if cell.unit_key
            else VerificationState.UNKNOWN
        )
    return _overall_state(
        tuple(ConstraintResult(f"slot:{slot.id}", state) for state in states)
    )


def _physical_mapping(
    cell: EvidenceCell,
    *,
    canonical_identity: tuple[object, ...] = (),
) -> PhysicalMapping | None:
    if (
        _source_provenance_state((cell,), check_number_format=False) is not VerificationState.VALID
        or cell.page is None
        or cell.source_span is None
        or physical_coordinate_key(cell) is None
    ):
        return None
    return PhysicalMapping(
        source_name=cell.source_name,
        page=cell.page,
        source_span=cell.source_span,
        source_span_scope=cell.source_span_scope,
        source_line_sha256=cell.source_line_sha256,
        line_number=cell.line_number,
        table_id=cell.table_id,
        row_id=cell.row_id,
        column_id=cell.column_id,
        canonical_identity=canonical_identity,
    )


def _program_slot_binding_indices(
    plan: QuestionPlan,
    program: ReasoningProgram,
    slot_bindings: Sequence[tuple[str, int]],
) -> dict[str, int]:
    slots = tuple(plan.operand_slots)
    if slot_bindings:
        bindings = dict(slot_bindings)
        return bindings if len(bindings) == len(slot_bindings) else {}
    if len(slots) != len(program.operands):
        return {}
    if len(slots) == 2 and plan.reference_base:
        reference_slots = [slot for slot in slots if slot.id == plan.reference_base]
        numerator_slots = [slot for slot in slots if slot.id != plan.reference_base]
        if len(reference_slots) != 1 or len(numerator_slots) != 1:
            return {}
        return {
            numerator_slots[0].id: program.operands[0],
            reference_slots[0].id: program.operands[1],
        }
    return {
        slot.id: index for slot, index in zip(slots, program.operands)
    }


def _role_agnostic_mapping_slots(
    plan: QuestionPlan,
    program: ReasoningProgram,
) -> tuple[OperandSlot, ...]:
    """Compile bounded physical-mapping intent without inventing a column role.

    QP2A intentionally leaves ``plan.operand_slots`` empty when a question does
    not name ACTUAL/PLAN.  Plain annual-report questions nevertheless provide
    an explicit metric and period.  Mapping diagnostics may use that immutable
    question intent, but must consider every non-conflicting source role so an
    ACTUAL/PLAN pair becomes AMBIGUOUS instead of silently defaulting to one.
    """

    if plan.operand_slots or plan.abstention_reason:
        return ()
    metric = plan.metric_text.strip()
    # Report references supply bounded lookup intent without requiring a
    # word from the vocabulary used to infer financial column roles. Each
    # candidate still needs a source-bound report-field witness and matching
    # report year in _slot_period_state; no fact date is manufactured.
    report_field_lookup = bool(plan.requested_report_year and program.operation == "SELECT")
    from .unit_defaults import monetary_label
    if not metric or not (_has_financial_role_anchor(metric) or report_field_lookup or monetary_label(metric)):
        return ()
    years = tuple(plan.required_operand_years) or ((plan.requested_report_year,) if plan.requested_report_year else ())
    if program.operation == "SELECT":
        if not years and plan.requested_value_column:
            years = ("",)  # No year inferred; identity is the requested column.
        if len(program.operands) != 1 or len(years) != 1:
            return ()
    elif program.operation in {
        "SUBTRACT",
        "PERCENT_CHANGE",
        "RATIO_PERCENT",
        "DIVIDE",
        "AVERAGE",
    }:
        if len(years) < 2 or len(program.operands) != len(years):
            return ()
    else:
        return ()
    if len(set(years)) != len(years):
        return ()

    entity_scope = (
        "reporting"
        if plan.entity_mode == "reporting"
        else f"named:{plan.entity_text}"
    )
    # An output percent is not a percent requirement for the input cells.
    # Growth can compare disclosed amounts or disclosed rates. The separate
    # unit-compatibility proof still enforces compatible operands.
    unit_family = (
        plan.requested_unit_family or "currency_or_source"
        if program.operation == "SELECT"
        else "currency_or_source" if program.operation == "PERCENT_CHANGE"
        else "currency"
    )
    return tuple(
        OperandSlot(
            id=f"mapping_period_{year}" if year else "mapping_column",
            metric=metric,
            year=year,
            # This sentinel is never compared with source evidence.  The
            # assessment helper below deliberately accepts every explicit or
            # absent role, while rejecting role-conflict cells.
            column_role="__role_agnostic__",
            entity_scope=entity_scope,
            unit_family=unit_family,
            intent_sources=(
                "metric:question_lexical",
                "year:question_explicit" if year else "column:question_explicit",
                "column_role:role_agnostic_mapping_only",
                "entity:question_scope",
                "unit:operation_contract",
            ),
        )
        for year in years
    )


def _role_agnostic_mapping_candidate_state(
    plan: QuestionPlan,
    slot: OperandSlot,
    cell: EvidenceCell,
    *,
    allow_missing_unit: bool,
) -> VerificationState:
    # Match the candidate's own role only to reuse the metric/year/entity/unit
    # checker.  This does not select or infer that role: every role-matching
    # physical cell is retained and therefore contributes to cardinality.
    candidate_role = (
        "__conflicting_source_role__"
        if cell.column_role_conflict
        else cell.column_role
    )
    return _slot_candidate_state(
        plan,
        replace(slot, column_role=candidate_role),
        cell,
        allow_missing_unit=allow_missing_unit,
    )


def _financial_identity_mapping_slots(
    plan: QuestionPlan,
    program: ReasoningProgram,
    graph: EvidenceGraph,
    *,
    financial_identity_catalog_enabled: bool,
) -> tuple[tuple[OperandSlot, ...], tuple[FinancialMetricRole, ...]]:
    """Bind one catalog formula to physical cells without inventing operands.

    Selected labels must identify exactly one ordered catalog-role tuple.
    Multiple formulas or role permutations therefore remain fail-closed.
    """

    target_slot: OperandSlot | None = None
    if plan.operand_slots:
        if not (
            plan.operation == "SELECT"
            and len(plan.operand_slots) == 1
            and len(program.operands) > 1
        ):
            return (), ()
        target_slot = plan.operand_slots[0]
    if (
        not financial_identity_catalog_enabled
        or plan.abstention_reason
        or plan.operation not in {None, "SELECT"}
        or not plan.financial_identity_ids
        or any(not 0 <= index < len(graph.cells) for index in program.operands)
    ):
        return (), ()
    years = tuple(
        dict.fromkeys(
            (target_slot.year,)
            if target_slot is not None
            else plan.required_operand_years
        )
    )
    if len(years) != 1:
        return (), ()
    selected_labels = tuple(
        graph.cells[index].metric_label for index in program.operands
    )
    if not all(selected_labels):
        return (), ()

    assignments: dict[
        tuple[tuple[str, tuple[str, ...], str], ...],
        tuple[FinancialMetricRole, ...],
    ] = {}
    for identity_id in plan.financial_identity_ids:
        identity = get_financial_identity(identity_id)
        if (
            not _operation_matches_plan(identity.operation, program)
            or len(identity.operands) != len(program.operands)
        ):
            continue
        if program.operation in COMMUTATIVE:
            allowed = [
                [
                    role_matches_metric(role, label)
                    for role in identity.operands
                ]
                for label in selected_labels
            ]
            role_orders = tuple(
                tuple(identity.operands[index] for index in assignment)
                for assignment in _bounded_bipartite_assignments(
                    allowed,
                    limit=2,
                )
            )
        else:
            role_orders = (identity.operands,)
        for role_order in role_orders:
            if not all(
                role_matches_metric(role, label)
                for role, label in zip(role_order, selected_labels)
            ):
                continue
            key = tuple(
                (role.key, role.aliases, role.unit_family)
                for role in role_order
            )
            assignments.setdefault(key, tuple(role_order))
    if len(assignments) != 1:
        return (), ()

    roles = next(iter(assignments.values()))
    year = years[0]
    entity_scope = (
        target_slot.entity_scope
        if target_slot is not None
        else "reporting"
        if plan.entity_mode == "reporting"
        else f"named:{plan.entity_text}"
    )
    column_role = (
        target_slot.column_role
        if target_slot is not None
        else "__role_agnostic__"
    )
    role_intent_source = (
        "column_role:question_explicit_identity_target"
        if target_slot is not None
        else "column_role:role_agnostic_mapping_only"
    )
    slots = tuple(
        OperandSlot(
            id=(
                f"identity_operand_{position + 1}_"
                + re.sub(r"[^a-z0-9]+", "_", role.key).strip("_")
            ),
            metric=role.preferred_text,
            year=year,
            column_role=column_role,
            entity_scope=entity_scope,
            unit_family=role.unit_family,
            intent_sources=(
                "metric:financial_identity_catalog",
                "year:question_explicit",
                role_intent_source,
                "entity:question_scope",
                "unit:financial_identity_operand",
            ),
        )
        for position, role in enumerate(roles)
    )
    return slots, roles


def assess_operand_mapping_resolution(
    plan: QuestionPlan,
    program: ReasoningProgram,
    graph: EvidenceGraph,
    *,
    slot_bindings: Sequence[tuple[str, int]] = (),
    financial_identity_catalog_enabled: bool = False,
) -> EvidenceResolutionAssessment | None:
    """Diagnose per-operand physical cardinality without changing proof state."""

    if plan.abstention_reason == "ambiguous_requested_period":
        return EvidenceResolutionAssessment(
            resolution="AMBIGUOUS", operand_mappings=(),
            reason_codes=("question_ambiguous_requested_period",),
            selected_binding_state=VerificationState.UNKNOWN,
        )
    typed_slots = tuple(plan.operand_slots)
    identity_slots, identity_roles = _financial_identity_mapping_slots(
        plan,
        program,
        graph,
        financial_identity_catalog_enabled=financial_identity_catalog_enabled,
    )
    slots = (
        identity_slots
        or typed_slots
        or _role_agnostic_mapping_slots(plan, program)
    )
    if not slots:
        return None
    using_typed_slots = bool(typed_slots and not identity_slots)
    mapping_plan = (
        plan if using_typed_slots else replace(plan, operand_slots=slots)
    )
    bindings = _program_slot_binding_indices(
        mapping_plan,
        program,
        slot_bindings if using_typed_slots else (),
    )
    allow_missing_unit = program.operation in {
        "DIVIDE",
        "RATIO_PERCENT",
        "PERCENT_CHANGE",
    }
    role_agnostic = not using_typed_slots
    identity_role_by_slot = {
        slot.id: role for slot, role in zip(identity_slots, identity_roles)
    }
    span_counts: dict[tuple[object, ...], int] = {}
    for cell in graph.cells:
        span_key = _physical_cell_key(cell)
        if span_key is not None:
            span_counts[span_key] = span_counts.get(span_key, 0) + 1

    def mapping_for(cell: EvidenceCell) -> PhysicalMapping | None:
        span_key = _physical_cell_key(cell)
        if span_key is None:
            return _physical_mapping(cell)
        structural_key = _structural_physical_key(cell)
        if is_versioned_coordinate(cell):
            return _physical_mapping(cell, canonical_identity=span_key)
        canonical = (
            ("physical", *span_key)
            if span_counts.get(span_key, 0) > 1
            else structural_key or ("physical", *span_key)
        )
        return _physical_mapping(cell, canonical_identity=canonical)

    def metric_state(slot: OperandSlot, cell: EvidenceCell) -> VerificationState:
        role = identity_role_by_slot.get(slot.id)
        if role is None:
            return _metric_obligation_state(slot.metric, cell.metric_label)
        if not cell.metric_label:
            return VerificationState.UNKNOWN
        return (
            VerificationState.VALID
            if role_matches_metric(role, cell.metric_label)
            else VerificationState.INVALID
        )

    def semantic_candidate_state(slot: OperandSlot, cell: EvidenceCell) -> VerificationState:
        chosen = bindings.get(slot.id)
        field_lookup = (
            program.operation == "SELECT" and not slot.year and not slot.period_role
            and not slot.column_role
            and chosen is not None and 0 <= chosen < len(graph.cells)
            and report_field_matches(graph.cells[chosen])
        )
        if field_lookup and not report_field_matches(cell):
            # A role-labelled table lane is not an undated company-profile
            # field, even when its year has not been extracted. Requiring a
            # period here made missing year metadata turn PLAN/ACTUAL lanes
            # into competing profile facts. Keep unknown untyped evidence and
            # all conflicting profile fields visible to the mapping checks.
            explicit_role_lane = (
                cell.column_role in {"actual", "plan", "revised_plan", "forecast"}
                and cell.column_role_confidence in {"EXPLICIT", "INHERITED"}
                and bool(cell.column_role_evidence)
                and not cell.column_role_conflict
            )
            if cell.period or cell.period_role or explicit_role_lane:
                return VerificationState.INVALID
        role = identity_role_by_slot.get(slot.id)
        if role is not None:
            structural_slot = replace(
                slot,
                metric=cell.metric_label or slot.metric,
            )
            structural_state = (
                _role_agnostic_mapping_candidate_state(
                    mapping_plan,
                    structural_slot,
                    cell,
                    allow_missing_unit=allow_missing_unit,
                )
                if slot.column_role == "__role_agnostic__"
                else _slot_candidate_state(
                    mapping_plan,
                    structural_slot,
                    cell,
                    allow_missing_unit=allow_missing_unit,
                )
            )
            return _overall_state(
                (
                    ConstraintResult(
                        f"identity_metric:{slot.id}",
                        metric_state(slot, cell),
                    ),
                    ConstraintResult(
                        f"identity_structure:{slot.id}",
                        structural_state,
                    ),
                )
            )
        if role_agnostic:
            return _role_agnostic_mapping_candidate_state(
                mapping_plan,
                slot,
                cell,
                allow_missing_unit=allow_missing_unit,
            )
        return _slot_candidate_state(
            mapping_plan,
            slot,
            cell,
            allow_missing_unit=allow_missing_unit,
        )

    def candidate_state(slot: OperandSlot, cell: EvidenceCell) -> VerificationState:
        from .number_formats import resolution_state
        semantic = semantic_candidate_state(slot, cell)
        if semantic is VerificationState.INVALID:
            return semantic
        numeric = VerificationState(resolution_state(cell))
        return _overall_state((ConstraintResult("mapping_semantics", semantic), ConstraintResult("number_format", numeric)))

    if set(bindings) == {slot.id for slot in slots}:
        selected_binding_state = _overall_state(
            tuple(
                ConstraintResult(
                    f"mapping:{slot.id}",
                    candidate_state(slot, graph.cells[bindings[slot.id]])
                    if 0 <= bindings[slot.id] < len(graph.cells)
                    else VerificationState.INVALID,
                )
                for slot in slots
            )
        )
    else:
        selected_binding_state = VerificationState.INVALID
    assessments: list[OperandMappingAssessment] = []
    duplicate_renditions_collapsed = False
    role_clarification_required = False
    scope_excluded = False
    for slot in slots:
        selected_index = bindings.get(slot.id)
        selected_state = (
            candidate_state(slot, graph.cells[selected_index])
            if selected_index is not None and 0 <= selected_index < len(graph.cells)
            else VerificationState.INVALID
        )
        valid: dict[tuple[object, ...], PhysicalMapping] = {}
        unresolved: dict[tuple[object, ...], PhysicalMapping] = {}
        relevant_by_physical: dict[
            tuple[object, ...], list[EvidenceCell]
        ] = {}
        incomplete_candidate = False
        valid_cells: list[EvidenceCell] = []
        proven_scope_keys={physical_coordinate_key(c) for c in graph.cells
                           if mapping_plan.requested_accounting_scope
                           and _query_accounting_scope_state(mapping_plan,c) is VerificationState.VALID
                           and candidate_state(slot,c) is VerificationState.VALID}
        for cell in graph.cells:
            if (proven_scope_keys and _query_accounting_scope_state(mapping_plan,cell) is VerificationState.UNKNOWN
                    and physical_coordinate_key(cell) not in proven_scope_keys):
                scope_excluded = True
                continue
            cell_metric_state = metric_state(slot, cell)
            period_state = _slot_period_state(mapping_plan, slot, cell)
            metric_period_relevant = (
                cell_metric_state is VerificationState.VALID
                and _query_section_state(mapping_plan, cell) is not VerificationState.INVALID
                and _query_accounting_scope_state(mapping_plan, cell) is not VerificationState.INVALID
                and period_state is not VerificationState.INVALID
            )
            state = candidate_state(slot, cell)
            mapping = mapping_for(cell)
            if metric_period_relevant and mapping is not None:
                relevant_by_physical.setdefault(mapping.identity, []).append(cell)
            elif metric_period_relevant and mapping is None and state is not VerificationState.INVALID:
                incomplete_candidate = True
            if state is VerificationState.VALID:
                if mapping is None:
                    incomplete_candidate = True
                else:
                    valid.setdefault(mapping.identity, mapping)
                    valid_cells.append(cell)
            elif (
                state is VerificationState.UNKNOWN
                and metric_period_relevant
                and mapping is not None
            ):
                unresolved.setdefault(mapping.identity, mapping)

        if not plan.column_roles and {"actual", "plan"}.issubset(
            {cell.column_role for cell in valid_cells}
        ):
            role_clarification_required = True
        conflict_reasons: list[str] = []
        for renditions in relevant_by_physical.values():
            group_conflict = False
            if len({cell.value for cell in renditions}) > 1:
                conflict_reasons.append("same_physical_mapping_value_conflict")
                group_conflict = True
            roles = {cell.column_role for cell in renditions if cell.column_role}
            if len(roles) > 1 or any(
                cell.column_role_conflict for cell in renditions
            ):
                conflict_reasons.append("same_physical_mapping_role_conflict")
                group_conflict = True
            entities = {
                normalize_text(cell.entity_id or cell.entity)
                for cell in renditions
                if cell.entity_id or cell.entity
            }
            if len(entities) > 1:
                conflict_reasons.append("same_physical_mapping_entity_conflict")
                group_conflict = True
            explicit_units = {
                (
                    cell.unit_key,
                    currency_unit_scale(cell.unit)
                    if currency_unit_scale(cell.unit) is not None
                    else normalize_text(cell.unit),
                )
                for cell in renditions
                if cell.unit
            }
            if len(explicit_units) > 1:
                conflict_reasons.append("same_physical_mapping_unit_conflict")
                group_conflict = True
            if len(renditions) > 1 and not group_conflict:
                duplicate_renditions_collapsed = True
        selected_mapping = (
            mapping_for(graph.cells[selected_index])
            if selected_index is not None and 0 <= selected_index < len(graph.cells)
            else None
        )
        from .disclosure_equivalence import compare_disclosures
        equivalence = compare_disclosures(valid_cells)
        assessments.append(
            OperandMappingAssessment(
                slot_id=slot.id,
                selected_state=selected_state,
                valid_physical_mappings=tuple(
                    valid[key] for key in sorted(valid, key=repr)
                ),
                unresolved_physical_mappings=tuple(
                    unresolved[key]
                    for key in sorted(unresolved, key=repr)
                    if key not in valid
                ),
                selected_physical_is_valid=(
                    selected_mapping is not None
                    and selected_mapping.identity in valid
                ),
                has_incomplete_candidate=incomplete_candidate,
                conflict_reasons=tuple(dict.fromkeys(conflict_reasons)),
                equivalent_disclosures=(
                    not conflict_reasons and not incomplete_candidate
                    and not (unresolved.keys() - valid.keys())
                    and equivalence["equivalent"]
                ),
                equivalence_basis=equivalence["basis"],
                equivalence_reasons=equivalence["reasons"],
            )
        )

    temporal_distinct_required = program.operation in {
        "SUBTRACT",
        "PERCENT_CHANGE",
    } or (
        program.operation in {"DIVIDE", "RATIO_PERCENT"}
        and len(set(plan.requested_years)) >= 2
    )
    selected_cells = tuple(
        graph.cells[bindings[slot.id]]
        for slot in slots
        if slot.id in bindings
        and 0 <= bindings[slot.id] < len(graph.cells)
    )
    structured_multi_operand = bool(
        len(selected_cells) >= 2
        and any(
            cell.source_name not in {"", "context", "synthetic"}
            for cell in selected_cells
        )
    )
    selected_operand_reuse = bool(
        (
            structured_multi_operand
            or temporal_distinct_required
            or bool(identity_slots)
        )
        and len(selected_cells) == len(slots)
        and len({cell.index for cell in selected_cells}) == len(selected_cells)
        and any(
            _same_physical_evidence(selected_cells[left], selected_cells[right])
            for left in range(len(selected_cells))
            for right in range(left + 1, len(selected_cells))
        )
    )
    if selected_operand_reuse:
        assessments = [
            replace(
                item,
                conflict_reasons=tuple(
                    dict.fromkeys(
                        (
                            *item.conflict_reasons,
                            "selected_operand_physical_reuse",
                        )
                    )
                ),
            )
            for item in assessments
        ]

    reasons: list[str] = ["unproven_requested_scope_not_selected"] if scope_excluded else []
    if any(item.conflict_reasons for item in assessments):
        resolution = "CONFLICT"
        reasons.append("same_physical_mapping_conflict")
    elif any(item.semantic_mapping_count > 1 for item in assessments):
        resolution = "AMBIGUOUS"
        reasons.append("multiple_physical_mappings")
        from .number_formats import resolution_state
        unknown_keys={m.identity for a in assessments for m in a.unresolved_physical_mappings}
        if any(physical_coordinate_key(c) in unknown_keys and resolution_state(c)!="VALID" for c in graph.cells):
            reasons.append("disclosure_number_format_unresolved")
        for item in assessments:
            if item.semantic_mapping_count > 1:
                reasons.extend(r for r in item.equivalence_reasons if r not in reasons)
    elif any(
        item.physical_mapping_count == 0 and not item.has_incomplete_candidate
        for item in assessments
    ):
        resolution = "MISSING"
        reasons.append("missing_operand_mapping")
    elif any(
        (
            not item.valid_physical_mappings
            and item.has_incomplete_candidate
        )
        or (
            not item.valid_physical_mappings
            and item.unresolved_physical_mappings
        )
        or item.selected_state is VerificationState.UNKNOWN
        for item in assessments
    ):
        resolution = "UNKNOWN"
        reasons.append("incomplete_operand_mapping")
    elif (
        selected_binding_state is VerificationState.INVALID
        or any(
            item.selected_state is VerificationState.INVALID
            or not item.selected_physical_is_valid
            for item in assessments
        )
    ):
        resolution = "CONFLICT"
        reasons.append("selected_mapping_conflict")
    elif selected_binding_state is VerificationState.UNKNOWN:
        resolution = "UNKNOWN"
        reasons.append("selected_mapping_unresolved")
    else:
        resolution = "UNIQUE"
        reasons.append("unique_physical_mapping_per_operand")
    if any(item.equivalent_disclosures for item in assessments):
        reasons.append("equivalent_disclosures_same_fact")
        if any(item.equivalent_disclosures and item.equivalence_basis in {"source_declared_nearest_rounding", "compatible_display_precision"} for item in assessments):
            reasons.append("disclosures_compatible_at_source_precision")
        if resolution == "UNIQUE":
            reasons = [r for r in reasons if r != "unique_physical_mapping_per_operand"]
            reasons.append("unique_semantic_fact_per_operand")
    from .number_formats import resolution_state
    if any(resolution_state(cell)!="VALID" for cell in selected_cells):
        reasons.append("source_number_format_unresolved")
    if role_clarification_required:
        reasons.append("question_column_role_unspecified")
    if selected_operand_reuse:
        reasons.append("selected_operand_physical_reuse")
    if duplicate_renditions_collapsed:
        reasons.append("duplicate_extractor_renditions_collapsed")
    if identity_slots:
        reasons.append("financial_identity_operand_mapping")
    return EvidenceResolutionAssessment(
        resolution=resolution,
        operand_mappings=tuple(assessments),
        reason_codes=tuple(reasons),
        selected_binding_state=selected_binding_state,
    )


def _unique_bound_slot_state(
    plan: QuestionPlan,
    operation: str,
    graph: EvidenceGraph,
    binding_indices: dict[str, int],
) -> VerificationState:
    """Require one physical source cell for every already-bound intent slot."""

    slots = tuple(plan.operand_slots)
    if not slots or set(binding_indices) != {slot.id for slot in slots}:
        return VerificationState.INVALID
    selected_bindings = tuple(
        (slot.id, binding_indices[slot.id]) for slot in slots
    )
    selected_state = _slot_binding_state(
        plan,
        graph.cells,
        selected_bindings,
        allow_missing_unit=operation
        in {"DIVIDE", "RATIO_PERCENT", "PERCENT_CHANGE"},
    )
    if selected_state is not VerificationState.VALID:
        return selected_state

    if len(slots)==2 and operation in {"DIVIDE", "RATIO_PERCENT", "PERCENT_CHANGE"}:
        # Use the same evidence identity/scope/equivalence contract as the
        # public mapping gate; a second, stricter physical-only pass drifted
        # from it and rejected otherwise proven denominators.
        program=ReasoningProgram(operation,tuple(binding_indices[slot.id] for slot in slots))
        mapping=assess_operand_mapping_resolution(plan,program,graph,slot_bindings=selected_bindings)
        if mapping is None:return VerificationState.UNKNOWN
        if mapping.selected_binding_state is VerificationState.INVALID:return VerificationState.INVALID
        return VerificationState.VALID if mapping.resolution=="UNIQUE" and mapping.selected_binding_state is VerificationState.VALID else VerificationState.UNKNOWN

    for slot in slots:
        valid_physical: set[tuple[object, ...]] = set()
        valid_cells: list[EvidenceCell] = []
        has_unknown = False
        for cell in graph.cells:
            trial = tuple(
                (
                    current.id,
                    cell.index
                    if current.id == slot.id
                    else binding_indices[current.id],
                )
                for current in slots
            )
            state = _slot_binding_state(
                plan,
                graph.cells,
                trial,
                allow_missing_unit=operation
                in {"DIVIDE", "RATIO_PERCENT", "PERCENT_CHANGE"},
            )
            if state is VerificationState.VALID:
                physical = _physical_cell_key(cell)
                if physical is None:
                    has_unknown = True
                else:
                    valid_physical.add(physical)
                    valid_cells.append(cell)
            elif (
                state is VerificationState.UNKNOWN
                and _source_provenance_state((cell,))
                is VerificationState.VALID
                and _metric_obligation_state(slot.metric, cell.metric_label)
                is VerificationState.VALID
                and _slot_period_state(plan, slot, cell)
                is not VerificationState.INVALID
            ):
                # A fully provenanced cell that matches the metric/period but
                # lacks one identity component (commonly role) is a plausible
                # alternate mapping. It cannot be silently ignored.
                has_unknown = True
        if has_unknown or (
            len(valid_physical) != 1 and not _equivalent_disclosures(valid_cells)
        ):
            return VerificationState.UNKNOWN
        selected_physical = _physical_cell_key(
            graph.cells[binding_indices[slot.id]]
        )
        if selected_physical not in valid_physical:
            return VerificationState.INVALID
    return VerificationState.VALID


def _selected_unique_slot_binding_state(
    plan: QuestionPlan,
    program: ReasoningProgram,
    graph: EvidenceGraph,
) -> VerificationState:
    """Prove flat selected operands against unique compiler slots.

    This is a verifier-only fallback for older checkpoints that do not emit
    slot bindings. It never searches for a better program: it validates the
    already selected operands and fails closed if any slot has zero or
    multiple physically distinct VALID matches in the full evidence graph.
    """

    slots = tuple(plan.operand_slots)
    if not slots or len(slots) != len(program.operands):
        return VerificationState.UNKNOWN
    if len(set(program.operands)) != len(program.operands):
        return VerificationState.INVALID
    if any(not 0 <= index < len(graph.cells) for index in program.operands):
        return VerificationState.INVALID

    binding_indices: dict[str, int] = {}
    if len(slots) == 2 and plan.reference_base:
        reference_slots = [slot for slot in slots if slot.id == plan.reference_base]
        numerator_slots = [slot for slot in slots if slot.id != plan.reference_base]
        if len(reference_slots) != 1 or len(numerator_slots) != 1:
            return VerificationState.UNKNOWN
        binding_indices[numerator_slots[0].id] = program.operands[0]
        binding_indices[reference_slots[0].id] = program.operands[1]
    else:
        binding_indices = {
            slot.id: index for slot, index in zip(slots, program.operands)
        }
    return _unique_bound_slot_state(
        plan,
        program.operation,
        graph,
        binding_indices,
    )


def _explicit_unique_slot_binding_state(
    plan: QuestionPlan,
    program: ReasoningProgram,
    graph: EvidenceGraph,
    slot_bindings: Sequence[tuple[str, int]],
) -> VerificationState:
    """Apply the same uniqueness proof to an authoritative typed DAG binding."""

    bindings = dict(slot_bindings)
    if len(bindings) != len(slot_bindings):
        return VerificationState.INVALID
    return _unique_bound_slot_state(plan, program.operation, graph, bindings)


def _dag_input_intent_reference(
    reference: object,
    *,
    kind: str,
    reverse_bindings: dict[int, str],
) -> str | None:
    if kind == "node":
        return str(reference)
    if kind == "candidate" and isinstance(reference, int):
        return reverse_bindings.get(reference)
    return None


def _intermediate_nodes_state(
    plan: QuestionPlan,
    dag: ReasoningDAGProgram | None,
    slot_bindings: Sequence[tuple[str, int]],
    *,
    candidate_count: int,
) -> VerificationState:
    if dag is None:
        return VerificationState.UNKNOWN
    try:
        dag.validate(candidate_count)
    except (ProgramDataError, ValueError):
        return VerificationState.INVALID
    expected = {node.id: node for node in plan.program_dag}
    observed = {node.node_id: node for node in dag.nodes}
    if set(expected) != set(observed) or dag.output_node != plan.program_dag[-1].id:
        return VerificationState.INVALID
    reverse_bindings = {index: slot_id for slot_id, index in slot_bindings}
    unknown = False
    for node_id, intent_node in expected.items():
        node = observed[node_id]
        if node.operation != intent_node.operation:
            return VerificationState.INVALID
        input_ids = tuple(
            _dag_input_intent_reference(
                item.reference,
                kind=item.kind,
                reverse_bindings=reverse_bindings,
            )
            for item in node.inputs
        )
        if None in input_ids or input_ids != intent_node.inputs:
            return VerificationState.INVALID
        if not node.result_unit_family:
            unknown = True
        elif node.result_unit_family != intent_node.output_unit_family:
            return VerificationState.INVALID
    return VerificationState.UNKNOWN if unknown else VerificationState.VALID


def _execute_dag_state(
    dag: ReasoningDAGProgram | None,
    cells: Sequence[EvidenceCell],
) -> VerificationState:
    if dag is None:
        return VerificationState.UNKNOWN
    try:
        dag.validate(len(cells))
        results: dict[str, object] = {}
        for node in dag.nodes:
            values = [
                cells[int(item.reference)].value
                if item.kind == "candidate"
                else results[str(item.reference)]
                for item in node.inputs
            ]
            operation = node.operation
            if operation == "SELECT":
                value = values[0]
            elif operation == "ADD":
                value = values[0] + values[1]
            elif operation == "SUBTRACT":
                value = values[0] - values[1]
            elif operation == "MULTIPLY":
                value = values[0] * values[1]
            elif operation in {"DIVIDE", "RATIO_PERCENT", "SHARE_OF_TOTAL"}:
                if values[1] == 0:
                    return VerificationState.INVALID
                value = values[0] / values[1]
                if operation in {"RATIO_PERCENT", "SHARE_OF_TOTAL"}:
                    value *= 100
            elif operation == "PERCENT_CHANGE":
                if values[1] == 0:
                    return VerificationState.INVALID
                value = (values[0] - values[1]) / values[1] * 100
            elif operation == "AVERAGE":
                value = (values[0] + values[1]) / 2
            elif operation in {"SUM3", "SUM4", "SUM_N"}:
                value = sum(values)
            else:
                return VerificationState.INVALID
            results[node.node_id] = value
        if dag.output_node not in results:
            return VerificationState.INVALID
    except (ArithmeticError, KeyError, ProgramDataError, TypeError, ValueError):
        return VerificationState.INVALID
    return VerificationState.VALID


def _reference_base_state(
    plan: QuestionPlan,
    dag: ReasoningDAGProgram | None,
    slot_bindings: Sequence[tuple[str, int]],
    program: ReasoningProgram,
    selected: Sequence[EvidenceCell],
    evidence_graph: EvidenceGraph,
    financial_identity_state: VerificationState,
) -> VerificationState:
    if plan.reference_base is None:
        # Outside the thesis-core typed-slot lane, an explicit ordered ratio
        # phrase (A over/of B) can still prove the denominator through the
        # established operand-role contract.  Missing roles remain UNKNOWN.
        role_state = _operand_role_state(plan, program.operation, selected)
        if role_state is not VerificationState.UNKNOWN:
            return role_state
        # Catalog identities validate an ordered operation/operand assignment,
        # so a VALID identity also proves its denominator role.
        if financial_identity_state is VerificationState.VALID:
            return VerificationState.VALID
        # For percent change, a supported NEW-vs-OLD direction proves that the
        # second operand is the requested reference period.
        if (
            program.operation == "PERCENT_CHANGE"
            and assess_program_direction(plan, program, evidence_graph).outcome
            == "direction_supported"
        ):
            return VerificationState.VALID
        return VerificationState.UNKNOWN
    bindings = dict(slot_bindings)
    if plan.reference_base not in bindings:
        return VerificationState.UNKNOWN
    if dag is None:
        # Flat binary programs still expose their denominator explicitly.
        # The bound-slot uniqueness/source checks below remain mandatory.
        if len(program.operands)!=2:return VerificationState.UNKNOWN
        return VerificationState.VALID if program.operands[1]==bindings[plan.reference_base] else VerificationState.INVALID
    if not dag.nodes:
        return VerificationState.UNKNOWN
    final = next(
        (node for node in dag.nodes if node.node_id == dag.output_node), None
    )
    if final is None or len(final.inputs) < 2:
        return VerificationState.INVALID
    denominator = final.inputs[1]
    if denominator.kind != "candidate":
        return VerificationState.INVALID
    return (
        VerificationState.VALID
        if denominator.reference == bindings[plan.reference_base]
        else VerificationState.INVALID
    )


def verify_program(
    plan: QuestionPlan,
    program: ReasoningProgram,
    evidence: EvidenceGraph | Sequence[EvidenceCell],
    *,
    financial_identity_catalog_enabled: bool = False,
    program_dag: ReasoningDAGProgram | None = None,
    slot_bindings: Sequence[tuple[str, int]] = (),
) -> VerificationResult:
    """Return a typed proof without treating missing OCR metadata as false.

    VALID means explicit evidence supports a constraint, INVALID means explicit
    evidence contradicts it, and UNKNOWN means the document parser did not
    recover enough metadata.  This separation is essential for applying the
    same verifier to unseen report layouts.
    """

    evidence_graph = (
        evidence
        if isinstance(evidence, EvidenceGraph)
        else EvidenceGraph(cells=tuple(evidence), edges=())
    )
    cells = evidence_graph.cells
    constraints: list[ConstraintResult] = []
    selected: list[EvidenceCell] = []
    identities = ()
    column_role_assessment: ColumnRoleProofAssessment | None = None

    def add(name: str, state: VerificationState) -> None:
        constraints.append(ConstraintResult(name, state))

    def finish() -> VerificationResult:
        identity_required = bool(
            identities and plan.operation in {None, "SELECT"}
        )
        contract = compile_question_proof_contract(
            plan,
            program.operation,
            financial_identity_required=identity_required,
        )
        from .entity_ownership import owner_of, query_owner_state
        if any(owner_of(cell) for cell in cells):
            contract = replace(contract, required=contract.required + ("query_entity_context",))
            owner_states = []
            for cell in selected:
                state = query_owner_state(plan, cell)
                if state is None:
                    state = ("VALID" if plan.entity_mode == "reporting" and not cell.embedded_entity else "UNKNOWN")
                owner_states.append(ConstraintResult("query_entity_context", VerificationState(state)))
            add("query_entity_context", _overall_state(tuple(owner_states)) if owner_states else VerificationState.UNKNOWN)
        if plan.requested_report_year:
            contract = replace(contract, required=contract.required + ("report_year_match",))
            add("report_year_match", VerificationState.VALID if selected and all(
                report_field_matches(cell) and cell.report_field["report_year"] == plan.requested_report_year for cell in selected
            ) else VerificationState.UNKNOWN)
        if any((cell.table_semantics or {}).get(key) for cell in selected for key in ("entity_conflict", "scope_conflict")):
            contract = replace(contract, required=contract.required + ("table_context_consistent",))
            add("table_context_consistent", VerificationState.INVALID)
        if plan.requested_accounting_scope or any(_column_scope_constraint_state(plan, cell) is not None for cell in selected):
            contract = replace(contract, required=contract.required + ("query_accounting_scope",))
            add("query_accounting_scope", _overall_state(tuple(
                ConstraintResult("query_accounting_scope", _query_accounting_scope_state(plan, cell)) for cell in selected
            )) if selected else VerificationState.UNKNOWN)
        if plan.requested_source_kind or plan.requested_event:
            contract = replace(contract, required=contract.required + ("query_source_context",))
            add("query_source_context", _overall_state(tuple(
                ConstraintResult("query_source_context", _query_source_frame_state(plan, cell)) for cell in selected
            )) if selected else VerificationState.UNKNOWN)
        if plan.requested_section:
            contract = replace(contract, required=contract.required + ("query_section",))
            add("query_section", _overall_state(tuple(
                ConstraintResult("query_section", _query_section_state(plan, cell)) for cell in selected
            )) if selected else VerificationState.UNKNOWN)
        if (
            column_role_assessment is not None
            and column_role_assessment.applicable
            and not plan.column_roles
            and "column_role_unique" not in contract.required
        ):
            contract = QuestionProofContract(
                required=contract.required + ("column_role_unique",),
                optional=tuple(
                    name
                    for name in contract.optional
                    if name != "column_role_unique"
                ),
                not_applicable=tuple(
                    name
                    for name in contract.not_applicable
                    if name != "column_role_unique"
                ),
            )
        legacy_unstructured_selection = bool(selected) and all(
            cell.source_name in {"", "context", "synthetic"}
            for cell in selected
        )
        if legacy_unstructured_selection and plan.operand_slots:
            # Historical context-only examples have stable candidate indices
            # but no document-level physical provenance. Preserve their
            # legacy proof lane without weakening real report evidence: any
            # source-grounded selection still requires these obligations.
            relaxed = {"logical_cell_unique", "source_provenance_complete"}
            contract = QuestionProofContract(
                required=tuple(
                    name for name in contract.required if name not in relaxed
                ),
                optional=tuple(
                    name for name in contract.optional if name not in relaxed
                ),
                not_applicable=tuple(
                    dict.fromkeys(
                        (*contract.not_applicable, *sorted(relaxed))
                    )
                ),
            )
        states = {item.name: item.state for item in constraints}
        structured_multi_operand = bool(
            len(selected) >= 2
            and any(
                cell.source_name not in {"", "context", "synthetic"}
                for cell in selected
            )
        )
        if len(set(program.operands)) != len(program.operands):
            pointer_state = VerificationState.INVALID
        elif structured_multi_operand and len(selected) >= 2:
            complete_physical_identity = all(
                _physical_cell_key(cell) is not None
                or _structural_physical_key(cell) is not None
                for cell in selected
            )
            physical_collision = any(
                _same_physical_evidence(selected[left], selected[right])
                for left in range(len(selected))
                for right in range(left + 1, len(selected))
            )
            pointer_state = (
                VerificationState.UNKNOWN
                if not complete_physical_identity
                else VerificationState.INVALID
                if physical_collision
                else VerificationState.VALID
            )
        else:
            pointer_state = VerificationState.VALID
        derived = [ConstraintResult("pointer_uniqueness", pointer_state)]
        if "set_completeness" in contract.required:
            derived.append(
                ConstraintResult(
                    "set_completeness",
                    _sum_set_completeness_state(plan, program, selected),
                )
            )
        if "operand_period_match" in contract.required:
            derived.append(
                ConstraintResult(
                    "operand_period_match",
                    _operand_period_match_state(plan, selected),
                )
            )
        if "operand_entity_match" in contract.required:
            derived.append(
                ConstraintResult(
                    "operand_entity_match",
                    _operand_entity_match_state(plan, selected),
                )
            )
        if (
            states.get("operation_match") is VerificationState.UNKNOWN
            and program.operation == "SELECT"
            and not plan.operation_candidates
            and plan.operation is None
        ):
            derived.append(
                ConstraintResult("operation_match", VerificationState.VALID)
            )
        if "query_metric" in contract.required or "query_metric" in contract.optional:
            derived.append(
                ConstraintResult(
                    "query_metric",
                    _query_metric_obligation_state(
                        plan, evidence_graph, selected
                    ),
                )
            )
        if "operand_roles" in contract.required or "operand_roles" in contract.optional:
            role_state = _operand_role_state(
                plan, program.operation, selected
            )
            unique_slot_state = (
                _explicit_unique_slot_binding_state(
                    plan,
                    program,
                    evidence_graph,
                    slot_bindings,
                )
                if slot_bindings
                else _selected_unique_slot_binding_state(
                    plan,
                    program,
                    evidence_graph,
                )
                if plan.operand_slots
                else VerificationState.UNKNOWN
            )
            if (
                role_state is VerificationState.UNKNOWN
                and plan.operand_slots
                and not slot_bindings
            ):
                role_state = _selected_unique_slot_binding_state(
                    plan, program, evidence_graph
                )
            if (
                role_state is VerificationState.UNKNOWN
                and plan.operand_slots
                and slot_bindings
            ):
                bindings = dict(slot_bindings)
                bound_roles = []
                for slot in plan.operand_slots:
                    index = bindings.get(slot.id)
                    bound_roles.append(
                        VerificationState.VALID
                        if index is not None
                        and 0 <= index < len(cells)
                        and (
                            not slot.column_role
                            and not cells[index].column_role_conflict
                            or cells[index].column_role
                            == slot.column_role.lower()
                        )
                        else VerificationState.INVALID
                        if index is not None
                        and 0 <= index < len(cells)
                        and (
                            cells[index].column_role
                            or cells[index].column_role_conflict
                        )
                        else VerificationState.UNKNOWN
                    )
                role_state = _overall_state(
                    tuple(
                        ConstraintResult("typed_operand_role", state)
                        for state in bound_roles
                    )
                )
            if (
                role_state is VerificationState.UNKNOWN
                and states.get("financial_identity")
                is VerificationState.VALID
            ):
                role_state = VerificationState.VALID
            if plan.operand_slots and any(
                cell.source_name not in {"", "context", "synthetic"}
                for cell in selected
            ):
                # Immutable compiler slots are the authoritative operand
                # identity contract.  The legacy free-text role parser is
                # only a fallback for plans without slots; its phrases may
                # include question framing such as "hãy cho biết" or
                # "mức ... cùng năm" even when role, metric and period are
                # all proved uniquely by the typed slots.
                role_state = unique_slot_state
            derived.append(
                ConstraintResult(
                    "operand_roles",
                    role_state,
                )
            )
        if "unit_compatibility" in contract.required:
            derived.append(
                ConstraintResult(
                    "unit_compatibility",
                    _unit_compatibility_state(
                        plan, program.operation, selected
                    ),
                )
            )
        if "unit_scale_match" in contract.required:
            derived.append(
                ConstraintResult(
                    "unit_scale_match",
                    _unit_scale_match_state(plan, program.operation, selected),
                )
            )
        if "add_grounding" in contract.required:
            role_state = next(
                (
                    item.state
                    for item in derived
                    if item.name == "operand_roles"
                ),
                VerificationState.UNKNOWN,
            )
            identity_state = states.get(
                "financial_identity", VerificationState.UNKNOWN
            )
            physical_keys = [_physical_cell_key(cell) for cell in selected]
            physical_collision = (
                len(physical_keys) >= 2
                and all(key is not None for key in physical_keys)
                and len(set(physical_keys)) != len(physical_keys)
            )
            ambiguous_year_column = _same_year_column_ambiguity(
                plan, selected, cells
            )
            explicit_subtotal_overlap = any(
                evidence_graph.has_relation(
                    selected[left].index,
                    selected[right].index,
                    EvidenceRelation.PART_OF_TOTAL,
                )
                or evidence_graph.has_relation(
                    selected[right].index,
                    selected[left].index,
                    EvidenceRelation.PART_OF_TOTAL,
                )
                for left in range(len(selected))
                for right in range(left + 1, len(selected))
            )
            if physical_collision:
                add_state = VerificationState.INVALID
            elif explicit_subtotal_overlap:
                # Absence of a PART_OF_TOTAL edge is not proof of independence.
                # Only extractor-proven overlap creates a hard contradiction.
                add_state = VerificationState.INVALID
            elif ambiguous_year_column:
                add_state = VerificationState.UNKNOWN
            elif VerificationState.VALID in {role_state, identity_state}:
                add_state = VerificationState.VALID
            elif (
                role_state is VerificationState.INVALID
                or identity_state is VerificationState.INVALID
            ):
                add_state = VerificationState.INVALID
            else:
                add_state = VerificationState.UNKNOWN
            derived.append(ConstraintResult("add_grounding", add_state))
        if "logical_cell_unique" in contract.required:
            derived.append(
                ConstraintResult(
                    "logical_cell_unique",
                    _logical_cell_unique_state(plan, evidence_graph, selected),
                )
            )
        if "source_provenance_complete" in contract.required:
            derived.append(
                ConstraintResult(
                    "source_provenance_complete",
                    _source_provenance_state(selected),
                )
            )
        if "answer_mode_complete" in contract.required:
            # The current executor returns one scalar. An explicitly multi-
            # output request is all-or-abstain until a complete output object
            # is implemented; returning a partial scalar is never valid.
            derived.append(
                ConstraintResult(
                    "answer_mode_complete",
                    VerificationState.UNKNOWN,
                )
            )
        if "reference_base_match" in contract.required:
            reference_state = _reference_base_state(
                plan,
                program_dag,
                slot_bindings,
                program,
                selected,
                evidence_graph,
                states.get("financial_identity", VerificationState.UNKNOWN),
            )
            if (
                reference_state is VerificationState.UNKNOWN
                and plan.reference_base
                and plan.operand_slots
                and not slot_bindings
            ):
                reference_state = _selected_unique_slot_binding_state(
                    plan, program, evidence_graph
                )
            elif plan.reference_base and plan.operand_slots:
                unique_reference_state = (
                    _explicit_unique_slot_binding_state(
                        plan,
                        program,
                        evidence_graph,
                        slot_bindings,
                    )
                    if slot_bindings
                    else _selected_unique_slot_binding_state(
                        plan,
                        program,
                        evidence_graph,
                    )
                )
                if unique_reference_state is not VerificationState.VALID:
                    reference_state = unique_reference_state
            derived.append(
                ConstraintResult(
                    "reference_base_match",
                    reference_state,
                )
            )
        if "all_leaf_nodes_valid" in contract.required:
            derived.append(
                ConstraintResult(
                    "all_leaf_nodes_valid",
                    _slot_binding_state(plan, cells, slot_bindings),
                )
            )
        if "all_intermediate_nodes_valid" in contract.required:
            derived.append(
                ConstraintResult(
                    "all_intermediate_nodes_valid",
                    _intermediate_nodes_state(
                        plan,
                        program_dag,
                        slot_bindings,
                        candidate_count=len(cells),
                    ),
                )
            )
        if "dag_execution_valid" in contract.required:
            derived.append(
                ConstraintResult(
                    "dag_execution_valid",
                    _execute_dag_state(program_dag, cells),
                )
            )
        result = VerificationResult(
            state=_overall_state(constraints),
            constraints=tuple(constraints),
            proof_contract=contract,
            obligation_states=tuple(derived),
            mapping_assessment=assess_operand_mapping_resolution(
                plan,
                program,
                evidence_graph,
                slot_bindings=slot_bindings,
                financial_identity_catalog_enabled=(
                    financial_identity_catalog_enabled
                ),
            ),
        )
        field = bool(selected) and all(report_field_matches(c) or source_assertion_matches_cell(c) for c in selected)
        kind = "DERIVED" if program.operation != "SELECT" else "FIELD_VALUE" if field else "TABLE_CELL" if selected and all(c.table_id for c in selected) else "UNKNOWN"
        witnesses = tuple({"source_name": c.source_name, "page": c.page,
                           "line_number": c.line_number, "source_span": c.source_span,
                           "source_line_sha256": c.source_line_sha256} for c in selected)
        details = []
        for obligation in result.proof_obligations:
            reason = "question_and_operation_contract"
            if field and obligation.name in {"column_role_unique", "column_role_match"} and obligation.applicability == "not_applicable":
                reason = "source_bound_field_has_no_table_column"
            elif field and obligation.name == "period_coverage" and obligation.applicability == "not_applicable":
                reason = "field_period_not_requested"
            elif obligation.applicability == "not_applicable":
                reason = "not_required_by_" + program.operation.lower() + "_intent"
            state = obligation.state.value if obligation.state is not None else None
            details.append({"name": obligation.name, "applicability": obligation.applicability,
                            "applicability_reason": reason, "state": state,
                            "witness_refs": witnesses, "evidence_kind": kind,
                            "reason_code": reason if state is None else obligation.name + ":" + state.lower()})
        return replace(result, applicability_details=tuple(details))

    try:
        program.validate(len(cells))
    except ValueError:
        add("execution_valid", VerificationState.INVALID)
        return finish()
    if program_dag is not None:
        dag_execution_state = _execute_dag_state(program_dag, cells)
        add("execution_valid", dag_execution_state)
        if dag_execution_state is VerificationState.INVALID:
            return finish()
    else:
        add("execution_valid", VerificationState.VALID)

    identities = (
        tuple(
            get_financial_identity(identity_id)
            for identity_id in plan.financial_identity_ids
        )
        if financial_identity_catalog_enabled
        else ()
    )
    identity_operations = (
        {identity.operation for identity in identities}
        if plan.operation in {None, "SELECT"}
        else set()
    )
    if plan.operation:
        allowed_operations = {plan.operation} | identity_operations
        add(
            "operation_match",
            VerificationState.VALID
            if any(
                _operation_matches_plan(expected, program)
                for expected in allowed_operations
            )
            else VerificationState.INVALID,
        )
    elif plan.operation_candidates:
        allowed_operations = set(plan.operation_candidates) | identity_operations
        add(
            "operation_match",
            VerificationState.VALID
            if any(
                _operation_matches_plan(expected, program)
                for expected in allowed_operations
            )
            else VerificationState.INVALID,
        )
    elif identities:
        add(
            "operation_match",
            VerificationState.VALID
            if program.operation == "SELECT"
            or any(
                _operation_matches_plan(expected, program)
                for expected in identity_operations
            )
            else VerificationState.INVALID,
        )
    else:
        add("operation_match", VerificationState.UNKNOWN)

    if (
        program.operation == "SELECT"
        and len(plan.operand_roles) > 1
        and not plan.direct_metric_lookup
    ):
        add("operand_arity_intent", VerificationState.INVALID)
    else:
        add("operand_arity_intent", VerificationState.VALID)

    selected_indices = (
        program_dag.leaf_operands if program_dag is not None else program.operands
    )
    selected = [cells[index] for index in selected_indices]
    column_role_assessment = assess_column_role_proof(
        plan,
        evidence_graph,
        selected_indices=selected_indices,
        operation=program.operation,
    )
    if plan.requested_value_column:
        column_states = [
            VerificationState.VALID if normalize_text(cell.column) == plan.requested_value_column
            else VerificationState.INVALID if cell.column else VerificationState.UNKNOWN
            for cell in selected
        ]
        add("column_role_match", _overall_state(tuple(
            ConstraintResult("requested_value_column", state) for state in column_states)))
    elif plan.column_roles:
        add(
            "column_role_match",
            column_role_assessment.state,
        )
    elif plan.operand_slots and slot_bindings:
        bindings = dict(slot_bindings)
        role_states: list[VerificationState] = []
        for slot in plan.operand_slots:
            index = bindings.get(slot.id)
            if index is None or not 0 <= index < len(cells):
                role_states.append(VerificationState.INVALID)
                continue
            cell = cells[index]
            role_states.append(
                VerificationState.VALID
                if (
                    not slot.column_role
                    and not cell.column_role_conflict
                    or cell.column_role == slot.column_role.lower()
                )
                else VerificationState.INVALID
                if cell.column_role or cell.column_role_conflict
                else VerificationState.UNKNOWN
            )
        add(
            "column_role_match",
            _overall_state(
                tuple(
                    ConstraintResult("slot_column_role", state)
                    for state in role_states
                )
            ),
        )
    elif column_role_assessment.applicable:
        add("column_role_unique", column_role_assessment.state)
    if identities and plan.operation in {None, "SELECT"}:
        if program.operation == "SELECT":
            label = selected[0].metric_label if selected else ""
            add(
                "financial_identity",
                VerificationState.VALID
                if label and any(
                    target_matches_metric(identity, label)
                    for identity in identities
                )
                else VerificationState.INVALID
                if label
                else VerificationState.UNKNOWN,
            )
        else:
            compatible_identities = tuple(
                identity
                for identity in identities
                if _operation_matches_plan(identity.operation, program)
                and len(identity.operands) == len(selected)
            )
            labels = tuple(cell.metric_label for cell in selected)
            matched = False
            for identity in compatible_identities:
                if program.operation in COMMUTATIVE:
                    allowed = [
                        [
                            role_matches_metric(role, label)
                            for label in labels
                        ]
                        for role in identity.operands
                    ]
                    identity_matches = bool(
                        _bounded_bipartite_assignments(
                            allowed,
                            limit=1,
                        )
                    )
                else:
                    identity_matches = all(
                        role_matches_metric(role, label)
                        for role, label in zip(identity.operands, labels)
                    )
                if identity_matches:
                    matched = True
                    break
            add(
                "financial_identity",
                VerificationState.VALID
                if matched
                else VerificationState.INVALID
                if compatible_identities and all(labels)
                else VerificationState.UNKNOWN,
            )
    if len(selected) >= 2:
        metric_keys = [cell.metric_key for cell in selected]
        from .metric_semantics import identity as metric_identity
        if all(cell.metric_label for cell in selected) and len({metric_identity(cell.metric_label) for cell in selected}) == 1:
            metric_keys = [metric_identity(cell.metric_label) for cell in selected]
        if all(metric_keys):
            add(
                "metric_match",
                VerificationState.VALID
                if len(set(metric_keys)) == 1
                else (
                    VerificationState.INVALID
                    if plan.temporal_change
                    else VerificationState.UNKNOWN
                ),
            )
        else:
            add("metric_match", VerificationState.UNKNOWN)

        entities = [
            normalize_text(cell.entity_id or cell.entity)
            for cell in selected
        ]
        known_entities = {entity for entity in entities if entity}
        if len(known_entities) > 1:
            add("entity_match", VerificationState.INVALID)
        elif all(entities):
            add("entity_match", VerificationState.VALID)
        else:
            add("entity_match", VerificationState.UNKNOWN)

        unit_keys = [cell.unit_key for cell in selected]
        known_units = {unit for unit in unit_keys if unit}
        comparable = program.operation in {
            "ADD",
            "SUBTRACT",
            "RATIO_PERCENT",
            "PERCENT_CHANGE",
            "AVERAGE",
            "SUM3",
            "SUM4",
            "SUM_N",
        }
        unrequested_ratio_addition = (
            comparable
            and known_units in ({"ratio"}, {"percent"})
            and plan.requested_unit_family not in {"ratio", "percent"}
        )
        if comparable and (len(known_units) > 1 or unrequested_ratio_addition):
            add("unit_match", VerificationState.INVALID)
        elif all(unit_keys):
            add("unit_match", VerificationState.VALID)
        else:
            add("unit_match", VerificationState.UNKNOWN)

        if program.operation == "MULTIPLY" and len(unit_keys) >= 2:
            if unit_keys[0] == unit_keys[1] == "share_count":
                add("unit_algebra", VerificationState.INVALID)
            elif set(unit_keys) == {"share_count", "currency_per_share"}:
                add("unit_algebra", VerificationState.VALID)
            else:
                add("unit_algebra", VerificationState.UNKNOWN)

        tables = [cell.table_id for cell in selected]
        if all(tables):
            add(
                "same_table",
                VerificationState.VALID
                if len(set(tables)) == 1
                else VerificationState.UNKNOWN,
            )
        else:
            add("same_table", VerificationState.UNKNOWN)

        rows = [cell.row_id for cell in selected]
        if all(rows):
            add(
                "same_row",
                VerificationState.VALID
                if len(set(rows)) == 1
                # The same logical metric may be repeated in separate tables
                # or across a page break. A different physical row is weak
                # evidence, never a contradiction by itself.
                else VerificationState.UNKNOWN,
            )
        else:
            add("same_row", VerificationState.UNKNOWN)

    requested = set(plan.requested_years)
    requested_period_roles = set(plan.requested_period_roles)
    selected_years = [cell.effective_years(plan) for cell in selected]
    covered = set().union(*selected_years) if selected_years else set()
    selected_period_roles = {
        cell.period_role for cell in selected if cell.period_role
    }
    if requested or requested_period_roles:
        if (
            program.operation == "ADD"
            and requested
            and _same_year_column_ambiguity(plan, selected, cells)
        ):
            add("period_coverage", VerificationState.UNKNOWN)
        elif (
            (not requested or requested.issubset(covered))
            and (
                not requested_period_roles
                or requested_period_roles.issubset(selected_period_roles)
            )
        ):
            add("period_coverage", VerificationState.VALID)
        elif (
            (not requested or all(selected_years))
            and (
                not requested_period_roles
                or len(selected_period_roles) == len(requested_period_roles)
            )
        ):
            add("period_coverage", VerificationState.INVALID)
        else:
            add("period_coverage", VerificationState.UNKNOWN)
    else:
        add("period_coverage", VerificationState.UNKNOWN)

    direction_outcome = "not_applicable"
    if program.operation in {"SUBTRACT", "PERCENT_CHANGE"}:
        direction = assess_program_direction(plan, program, evidence_graph)
        direction_outcome = direction.outcome
        if direction.outcome == "direction_supported":
            add("temporal_direction", VerificationState.VALID)
        elif direction.outcome in {
            "direction_reversed",
            "ambiguous_wording",
            "unsupported_absolute_gap",
        }:
            add("temporal_direction", VerificationState.INVALID)
        else:
            add("temporal_direction", VerificationState.UNKNOWN)
    else:
        add("temporal_direction", VerificationState.UNKNOWN)

    if program.operation in {"DIVIDE", "RATIO_PERCENT", "PERCENT_CHANGE"}:
        denominator_cell = None
        if plan.reference_base is not None and slot_bindings:
            denominator_index = dict(slot_bindings).get(plan.reference_base)
            if denominator_index is not None and 0 <= denominator_index < len(cells):
                denominator_cell = cells[denominator_index]
        if denominator_cell is None and len(selected) >= 2:
            denominator_cell = selected[1]
        add(
            "denominator_safe",
            VerificationState.INVALID
            if denominator_cell is not None and denominator_cell.value == 0
            else VerificationState.VALID
            if denominator_cell is not None
            else VerificationState.UNKNOWN,
        )
    else:
        add("denominator_safe", VerificationState.VALID)

    if plan.entity_mode == "reporting" and any(
        cell.embedded_entity for cell in selected
    ):
        selected_metrics = {cell.metric_key for cell in selected if cell.metric_key}
        reporting = [
            cell
            for cell in cells
            if not cell.embedded_entity
            and (not selected_metrics or cell.metric_key in selected_metrics)
        ]
        reporting_years = (
            set().union(*(cell.effective_years(plan) for cell in reporting))
            if reporting
            else set()
        )
        add(
            "reporting_entity",
            VerificationState.INVALID
            if reporting and (not requested or requested.issubset(reporting_years))
            else VerificationState.UNKNOWN,
        )
    elif plan.entity_mode == "reporting":
        add("reporting_entity", VerificationState.VALID)

    if plan.entity_mode == "named" and plan.entity_terms:
        matching_embedded = {
            cell.index
            for cell in cells
            if (
                cell.embedded_entity
                or (
                    plan.entity_text == "parent_company"
                    and normalize_text(cell.entity) == "parent company"
                )
            )
            and set(
                normalize_text(
                    f"{cell.entity_id} {cell.entity} {cell.section} "
                    f"{cell.source_name}"
                ).split()
            ).intersection(plan.entity_terms)
        }
        from .entity_ownership import query_owner_state
        matching_embedded.update(cell.index for cell in cells if query_owner_state(plan, cell) == "VALID")
        if matching_embedded:
            add(
                "named_entity",
                VerificationState.VALID
                if all(cell.index in matching_embedded for cell in selected)
                else VerificationState.INVALID,
            )
        else:
            add("named_entity", VerificationState.UNKNOWN)

    # The graph contributes explicit relational proof features without
    # becoming a new source of hard failures when extraction metadata is absent.
    if isinstance(evidence, EvidenceGraph) and len(selected) == 2:
        left, right = selected
        for relation, name in (
            (EvidenceRelation.SAME_METRIC, "graph_same_metric"),
            (EvidenceRelation.SAME_ENTITY, "graph_same_entity"),
            (EvidenceRelation.SAME_TABLE, "graph_same_table"),
            (EvidenceRelation.SAME_ROW, "graph_same_row"),
            (EvidenceRelation.SAME_UNIT, "graph_same_unit"),
        ):
            if (
                relation is EvidenceRelation.SAME_UNIT
                and program.operation == "MULTIPLY"
            ):
                add(name, VerificationState.UNKNOWN)
                continue
            add(
                name,
                VerificationState.VALID
                if evidence.has_relation(left.index, right.index, relation)
                else VerificationState.UNKNOWN,
            )
        if plan.temporal_change:
            expected_relation = (
                EvidenceRelation.NEXT_PERIOD
                if plan.direction_intent is DirectionIntent.DECREASE_AMOUNT
                else EvidenceRelation.PREVIOUS_PERIOD
            )
            opposite_relation = (
                EvidenceRelation.PREVIOUS_PERIOD
                if expected_relation is EvidenceRelation.NEXT_PERIOD
                else EvidenceRelation.NEXT_PERIOD
            )
            if (
                direction_outcome == "direction_supported"
                and evidence.has_relation(left.index, right.index, expected_relation)
            ):
                add("graph_temporal_order", VerificationState.VALID)
            elif (
                direction_outcome == "direction_reversed"
                and evidence.has_relation(left.index, right.index, opposite_relation)
            ):
                add("graph_temporal_order", VerificationState.INVALID)
            else:
                add("graph_temporal_order", VerificationState.UNKNOWN)

    return finish()
