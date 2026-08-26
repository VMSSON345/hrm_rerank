from __future__ import annotations

from dataclasses import dataclass

from .evidence_schema import EvidenceCell, EvidenceGraph
from .financial_identities import metric_alias_score
from .program_data import ReasoningProgram
from .question_planner import DirectionIntent, QuestionPlan


@dataclass(frozen=True)
class DirectionAssessment:
    """Auditable result for the operand order required by a question."""

    outcome: str
    expected: tuple[str, str] = ()
    observed: tuple[str, str] = ()


def assess_program_direction(
    plan: QuestionPlan,
    program: ReasoningProgram,
    graph: EvidenceGraph,
) -> DirectionAssessment:
    """Assess direction from language and evidence metadata, never magnitude.

    This function is diagnostic in S1. It does not mutate, execute or accept a
    program. S2 can reuse the same result when all pipeline branches are wired
    to the shared direction contract.
    """

    intent = plan.direction_intent
    if program.operation not in {"SUBTRACT", "PERCENT_CHANGE"}:
        return DirectionAssessment("not_applicable")
    if intent is None:
        return DirectionAssessment("direction_unknown")
    if intent is DirectionIntent.AMBIGUOUS:
        return DirectionAssessment("ambiguous_wording")
    if intent is DirectionIntent.ABSOLUTE_GAP:
        return DirectionAssessment("unsupported_absolute_gap")
    if len(program.operands) != 2:
        return DirectionAssessment("direction_unknown")

    cells_by_index = {cell.index: cell for cell in graph.cells}
    try:
        left, right = (cells_by_index[index] for index in program.operands)
    except KeyError:
        return DirectionAssessment("direction_unknown")

    if intent is DirectionIntent.LEFT_MINUS_RIGHT:
        return _assess_named_roles(plan, left, right)
    if intent is DirectionIntent.NEW_MINUS_OLD:
        return _assess_temporal(plan, left, right, new_first=True)
    if intent is DirectionIntent.DECREASE_AMOUNT:
        if len(plan.operand_roles) == 2 or (
            len(plan.operand_slots) == 2
            and len({slot.column_role for slot in plan.operand_slots}) == 2
        ):
            return _assess_named_roles(plan, left, right)
        return _assess_temporal(plan, left, right, new_first=False)
    return DirectionAssessment("direction_unknown")


def orient_program_direction(
    plan: QuestionPlan,
    program: ReasoningProgram,
    graph: EvidenceGraph,
) -> ReasoningProgram:
    """Orient a selected pair only when the shared contract proves reversal."""

    if program.operation not in {"SUBTRACT", "PERCENT_CHANGE"}:
        return program
    assessment = assess_program_direction(plan, program, graph)
    if assessment.outcome != "direction_reversed":
        return program
    return ReasoningProgram(program.operation, tuple(reversed(program.operands)))


def reversed_pair_is_proven_negative(
    plan: QuestionPlan,
    program: ReasoningProgram,
    graph: EvidenceGraph,
) -> bool:
    """Return true only when the question proves both gold and reversed order."""

    if (
        program.operation not in {"SUBTRACT", "PERCENT_CHANGE"}
        or len(program.operands) != 2
    ):
        return False
    direct = assess_program_direction(plan, program, graph)
    if direct.outcome != "direction_supported":
        return False
    reversed_program = ReasoningProgram(
        program.operation, tuple(reversed(program.operands))
    )
    reverse = assess_program_direction(plan, reversed_program, graph)
    return reverse.outcome == "direction_reversed"


def _assess_named_roles(
    plan: QuestionPlan,
    left: EvidenceCell,
    right: EvidenceCell,
) -> DirectionAssessment:
    if len(plan.operand_roles) == 2:
        expected_left, expected_right = plan.operand_roles
        observed = (left.metric_label, right.metric_label)
        direct = (
            _role_matches(expected_left, left.metric_label)
            and _role_matches(expected_right, right.metric_label)
        )
        reverse = (
            _role_matches(expected_left, right.metric_label)
            and _role_matches(expected_right, left.metric_label)
        )
    elif len(plan.operand_slots) == 2 and all(
        slot.column_role for slot in plan.operand_slots
    ):
        # Typed report columns (ACTUAL/PLAN/FORECAST/...) carry the same
        # ordering evidence as named metric roles.  Use the immutable compiler
        # slots so direction proof does not depend on values or candidate rank.
        expected_left, expected_right = (
            slot.column_role.lower() for slot in plan.operand_slots
        )
        observed = (left.column_role, right.column_role)
        direct = observed == (expected_left, expected_right)
        reverse = observed == (expected_right, expected_left)
    else:
        return DirectionAssessment("direction_unknown")
    if direct and not reverse:
        outcome = "direction_supported"
    elif reverse and not direct:
        outcome = "direction_reversed"
    elif not left.metric_label or not right.metric_label:
        outcome = "wrong_or_missing_metric"
    else:
        outcome = "wrong_metric"
    return DirectionAssessment(
        outcome,
        expected=(expected_left, expected_right),
        observed=observed,
    )


def _role_matches(role: str, metric_label: str) -> bool:
    return metric_alias_score(metric_label, (role,)) >= 0.72


def _assess_temporal(
    plan: QuestionPlan,
    left: EvidenceCell,
    right: EvidenceCell,
    *,
    new_first: bool,
) -> DirectionAssessment:
    period_pair = _expected_period_pair(plan, new_first=new_first)
    if period_pair is not None:
        first_period, second_period = period_pair
        direct = (
            first_period in left.effective_years(plan)
            and second_period in right.effective_years(plan)
        )
        reverse = (
            first_period in right.effective_years(plan)
            and second_period in left.effective_years(plan)
        )
        observed = (
            ",".join(sorted(left.effective_years(plan))),
            ",".join(sorted(right.effective_years(plan))),
        )
        if direct and not reverse:
            outcome = "direction_supported"
        elif reverse and not direct:
            outcome = "direction_reversed"
        elif not observed[0] or not observed[1]:
            outcome = "missing_period"
        else:
            outcome = "direction_unknown"
        return DirectionAssessment(outcome, period_pair, observed)

    expected_roles = ("current", "prior") if new_first else ("prior", "current")
    observed_roles = (left.period_role, right.period_role)
    if observed_roles == expected_roles:
        outcome = "direction_supported"
    elif observed_roles == tuple(reversed(expected_roles)):
        outcome = "direction_reversed"
    elif not all(observed_roles):
        outcome = "missing_period"
    else:
        outcome = "direction_unknown"
    return DirectionAssessment(outcome, expected_roles, observed_roles)


def _expected_period_pair(
    plan: QuestionPlan,
    *,
    new_first: bool,
) -> tuple[str, str] | None:
    if plan.from_year and plan.to_year:
        old, new = plan.from_year, plan.to_year
    elif len(plan.requested_years) >= 2:
        ordered = sorted(set(plan.requested_years), key=int)
        old, new = ordered[0], ordered[-1]
    else:
        return None
    return (new, old) if new_first else (old, new)
