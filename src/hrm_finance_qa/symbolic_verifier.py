from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import permutations
import re
from typing import Sequence

from .evidence_schema import EvidenceCell, EvidenceGraph, EvidenceRelation
from .direction_contract import assess_program_direction
from .financial_identities import (
    get_financial_identity,
    metric_alias_score,
    role_matches_metric,
    target_matches_metric,
)
from .ontology import normalize_text
from .program_data import (
    COMMUTATIVE,
    ProgramDataError,
    ReasoningDAGProgram,
    ReasoningProgram,
    currency_unit_scale,
)
from .question_planner import (
    DirectionIntent,
    QuestionPlan,
    QuestionProofContract,
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


def _metric_obligation_state(
    expected: str, observed: str
) -> VerificationState:
    if not expected or not observed:
        return VerificationState.UNKNOWN
    score = metric_alias_score(observed, (expected,))
    if score >= 0.72:
        return VerificationState.VALID
    if score <= 0.10:
        return VerificationState.INVALID
    return VerificationState.UNKNOWN


def _query_metric_obligation_state(
    plan: QuestionPlan,
    graph: EvidenceGraph,
    selected: Sequence[EvidenceCell],
) -> VerificationState:
    """Verify a query metric from row text and provenanced header context."""

    if len(selected) != 1:
        return VerificationState.UNKNOWN

    def assess(cell: EvidenceCell) -> tuple[VerificationState, bool]:
        row_state = _metric_obligation_state(plan.metric_text, cell.metric_label)
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
    assignments = (labels,)
    if operation in COMMUTATIVE:
        assignments = tuple(permutations(labels))

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

    assignment_scores = tuple(
        tuple(
            role_score(role, label)
            for role, label in zip(roles, assignment)
        )
        for roles in role_hypotheses
        for assignment in assignments
    )
    if any(all(score >= 0.72 for score in scores) for scores in assignment_scores):
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
    if assignment_scores and max(min(scores) for scores in assignment_scores) <= 0.10:
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
    if len(required) != len(selected):
        return VerificationState.INVALID
    observed = tuple(cell.column_role for cell in selected)
    if any(cell.column_role_conflict for cell in selected):
        return VerificationState.INVALID
    if any(not role for role in observed):
        return VerificationState.UNKNOWN
    assignments = (observed,)
    if operation in COMMUTATIVE:
        assignments = tuple(permutations(observed))
    if required in assignments:
        return VerificationState.VALID
    return VerificationState.INVALID


def _column_role_logical_key(
    plan: QuestionPlan,
    cell: EvidenceCell,
) -> tuple[object, ...] | None:
    years = tuple(sorted(cell.effective_years(plan)))
    metric = cell.metric_key or normalize_text(cell.metric_label)
    row = cell.row_id or metric
    if not cell.source_name or not cell.table_id or not row or not metric or not years:
        return None
    return (
        cell.source_name,
        cell.table_id,
        row,
        metric,
        years,
        normalize_text(cell.entity_id or cell.entity),
        cell.unit_key,
    )


def _column_role_physical_key(cell: EvidenceCell) -> tuple[object, ...]:
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
    units = tuple(cell.unit_key for cell in selected[:2])
    if not all(units):
        return VerificationState.UNKNOWN
    if len(set(units)) == 1:
        if (
            units[0] in {"ratio", "percent"}
            and operation in {"ADD", "SUBTRACT", "AVERAGE", "SUM3", "SUM4"}
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
        if operation == "MULTIPLY" and cell.unit_key == "share_count":
            continue
        if cell.unit_key in {"", "currency", "currency_per_share"}:
            missing_scale = True
            continue
        return VerificationState.INVALID

    if len(set(observed_scales)) > 1:
        return VerificationState.INVALID
    if missing_scale or not observed_scales:
        return VerificationState.UNKNOWN
    return (
        VerificationState.VALID
        if observed_scales[0] == expected_scale
        else VerificationState.UNKNOWN
    )


def _logical_cell_key(
    plan: QuestionPlan, cell: EvidenceCell
) -> tuple[object, ...] | None:
    years = tuple(sorted(cell.effective_years(plan)))
    entity = normalize_text(cell.entity_id or cell.entity)
    if not entity and not cell.embedded_entity:
        entity = "reporting"
    required = (
        cell.source_name,
        cell.page,
        cell.table_id,
        cell.row_id,
        cell.column_id,
        cell.metric_key,
        years,
        cell.column_role,
        entity,
        cell.unit_key,
    )
    if any(value in {None, "", ()} for value in required):
        return None
    return required


def _physical_cell_key(cell: EvidenceCell) -> tuple[object, ...] | None:
    if not cell.source_name or cell.page is None or cell.source_span is None:
        return None
    return cell.source_name, cell.page, cell.source_span


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
) -> VerificationState:
    if not selected:
        return VerificationState.UNKNOWN
    for cell in selected:
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
        cell = cells[index]
        states.append(_metric_obligation_state(slot.metric, cell.metric_label))
        years = cell.effective_years(plan)
        states.append(
            VerificationState.VALID
            if slot.year in years
            else VerificationState.INVALID
            if years
            else VerificationState.UNKNOWN
        )
        states.append(
            VerificationState.VALID
            if cell.column_role == slot.column_role.lower()
            else VerificationState.INVALID
            if cell.column_role or cell.column_role_conflict
            else VerificationState.UNKNOWN
        )
        if slot.entity_scope == "reporting":
            states.append(
                VerificationState.INVALID
                if cell.embedded_entity
                else VerificationState.VALID
            )
        else:
            observed_entity = normalize_text(
                f"{cell.entity_id} {cell.entity} {cell.section} {cell.source_name}"
            )
            expected_terms = set(normalize_text(slot.entity_scope).split())
            states.append(
                VerificationState.VALID
                if expected_terms.intersection(observed_entity.split())
                else VerificationState.UNKNOWN
            )
        if slot.unit_family == "currency_or_source":
            states.append(
                VerificationState.VALID
                if cell.unit_key
                else VerificationState.UNKNOWN
            )
        else:
            states.append(
                VerificationState.VALID
                if cell.unit_key == slot.unit_family
                else VerificationState.INVALID
                if cell.unit_key
                else VerificationState.UNKNOWN
            )
    return _overall_state(
        tuple(ConstraintResult("slot", state) for state in states)
    )


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
            elif operation in {"SUM3", "SUM4"}:
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
    if plan.reference_base not in bindings or dag is None or not dag.nodes:
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
        identity_required = bool(identities and plan.operation is None)
        contract = compile_question_proof_contract(
            plan,
            program.operation,
            financial_identity_required=identity_required,
        )
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
        states = {item.name: item.state for item in constraints}
        pointer_state = (
            VerificationState.VALID
            if len(set(program.operands)) == len(program.operands)
            else VerificationState.INVALID
        )
        derived = [ConstraintResult("pointer_uniqueness", pointer_state)]
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
                        and cells[index].column_role == slot.column_role.lower()
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
            if physical_collision:
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
            derived.append(
                ConstraintResult(
                    "reference_base_match",
                    _reference_base_state(
                        plan,
                        program_dag,
                        slot_bindings,
                        program,
                        selected,
                        evidence_graph,
                        states.get("financial_identity", VerificationState.UNKNOWN),
                    ),
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
        return VerificationResult(
            state=_overall_state(constraints),
            constraints=tuple(constraints),
            proof_contract=contract,
            obligation_states=tuple(derived),
        )

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
    identity_operations = {
        identity.operation for identity in identities
    } if plan.operation is None else set()
    if plan.operation:
        add(
            "operation_match",
            VerificationState.VALID
            if program.operation == plan.operation
            else VerificationState.INVALID,
        )
    elif plan.operation_candidates:
        allowed_operations = set(plan.operation_candidates) | identity_operations
        add(
            "operation_match",
            VerificationState.VALID
            if program.operation in allowed_operations
            else VerificationState.INVALID,
        )
    elif identities:
        add(
            "operation_match",
            VerificationState.VALID
            if program.operation == "SELECT"
            or program.operation in identity_operations
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
    if plan.column_roles:
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
                if cell.column_role == slot.column_role.lower()
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
    if identities and plan.operation is None:
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
                if identity.operation == program.operation
                and len(identity.operands) == len(selected)
            )
            labels = tuple(cell.metric_label for cell in selected)
            matched = False
            for identity in compatible_identities:
                assignments = (labels,)
                if program.operation in COMMUTATIVE:
                    assignments = tuple(permutations(labels))
                if any(
                    all(
                        role_matches_metric(role, label)
                        for role, label in zip(identity.operands, assignment)
                    )
                    for assignment in assignments
                ):
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
    selected_years = [cell.effective_years(plan) for cell in selected]
    covered = set().union(*selected_years) if selected_years else set()
    if requested:
        if (
            program.operation == "ADD"
            and _same_year_column_ambiguity(plan, selected, cells)
        ):
            add("period_coverage", VerificationState.UNKNOWN)
        elif requested.issubset(covered):
            add("period_coverage", VerificationState.VALID)
        elif all(selected_years):
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
            if cell.embedded_entity
            and set(
                normalize_text(
                    f"{cell.entity_id} {cell.entity} {cell.section} "
                    f"{cell.source_name}"
                ).split()
            ).intersection(plan.entity_terms)
        }
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
