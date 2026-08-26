from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import product
import math
import re
from typing import Sequence

from .evidence_schema import EvidenceCell, EvidenceGraph, EvidenceRelation
from .ontology import normalize_text
from .program_data import (
    COMMUTATIVE,
    OPERATION_ARITY,
    ProgramDAGInput,
    ProgramDAGNode,
    ReasoningDAGProgram,
    ReasoningProgram,
)
from .question_planner import DirectionIntent, QuestionPlan


_SLOT_STOPWORDS = {
    "bao",
    "cao",
    "cua",
    "den",
    "duoc",
    "gia",
    "giua",
    "la",
    "muc",
    "nam",
    "nhieu",
    "phan",
    "tai",
    "thay doi",
    "theo",
    "tinh",
    "tong",
    "tram",
    "tri",
    "tu",
    "va",
}


@dataclass(frozen=True)
class TypedProgramProposal:
    program: ReasoningProgram
    type_score: float
    proofs: tuple[str, ...]
    program_dag: ReasoningDAGProgram | None = None
    slot_bindings: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class TypedSynthesisResult:
    operation: str
    template: str
    proposals: tuple[TypedProgramProposal, ...]
    slot_candidate_counts: tuple[int, ...]
    status: str

    @property
    def programs(self) -> tuple[ReasoningProgram, ...]:
        return tuple(item.program for item in self.proposals)


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_text(text))
        if len(token) > 1
        and not token.isdigit()
        and token not in _SLOT_STOPWORDS
    }


def _metric_score(
    expected: str,
    cell: EvidenceCell,
    *,
    observed_text: str = "",
) -> float:
    expected_terms = _terms(expected)
    observed_terms = _terms(observed_text or cell.metric_label)
    if not expected_terms or not observed_terms:
        return 0.0
    overlap = 0.0
    for term in expected_terms:
        overlap += max(
            (
                1.0
                if term == candidate
                else 0.90
                if len(term) >= 3 and term in candidate
                else 0.82
                if len(candidate) >= 4 and candidate in term
                else 0.0
                for candidate in observed_terms
            ),
            default=0.0,
        )
    expected_compact = "".join(sorted(expected_terms))
    observed_compact = "".join(sorted(observed_terms))
    similarity = SequenceMatcher(
        None, expected_compact, observed_compact
    ).ratio()
    density = overlap / math.sqrt(max(1, len(observed_terms)))
    return density + 2.0 * similarity


def _metric_context_score(expected: str, cell: EvidenceCell) -> float:
    """Score row semantics plus source-provenanced structural context."""

    score = _metric_score(expected, cell)
    if (
        not cell.metric_label
        or not cell.header_path
        or len(cell.header_path) != len(cell.header_path_provenance)
        or not all(cell.header_path_provenance)
        or cell.column_role_conflict
    ):
        return score
    composite = " ".join(
        part
        for part in (
            cell.section,
            *cell.header_path,
            cell.metric_label,
        )
        if part
    )
    return max(score, _metric_score(expected, cell, observed_text=composite))


def _entity_text(cell: EvidenceCell) -> str:
    return normalize_text(
        f"{cell.entity_id} {cell.entity} {cell.section} {cell.source_name}"
    )


def _entity_key(cell: EvidenceCell) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_text(cell.entity_id or cell.entity))


def _named_entity_match(plan: QuestionPlan, cell: EvidenceCell) -> bool:
    if not plan.entity_terms:
        return False
    observed = set(_entity_text(cell).split())
    return bool(observed.intersection(plan.entity_terms))


def _year_state(cell: EvidenceCell, year: str, plan: QuestionPlan) -> int:
    if not year:
        return 0
    years = cell.effective_years(plan)
    if year in years:
        return 1
    return -1 if years else 0


def _role_state(cell: EvidenceCell, role: str) -> int:
    if not role:
        return 0
    if cell.period_role == role:
        return 1
    return -1 if cell.period_role else 0


def _column_role_state(cell: EvidenceCell, role: str) -> int:
    if not role:
        return 0
    role = role.lower()
    if cell.column_role_conflict:
        return -1
    if cell.column_role == role:
        return 1
    return -1 if cell.column_role else 0


def _slot_candidates(
    graph: EvidenceGraph,
    plan: QuestionPlan,
    *,
    metric_text: str,
    year: str = "",
    period_role: str = "",
    column_role: str = "",
    required_unit_family: str = "",
    limit: int = 20,
) -> list[tuple[EvidenceCell, float, tuple[str, ...]]]:
    rows: list[tuple[EvidenceCell, float, tuple[str, ...]]] = []
    for cell in graph.cells:
        if cell.is_year_like:
            continue
        year_state = _year_state(cell, year, plan)
        role_state = _role_state(cell, period_role)
        column_role_state = _column_role_state(cell, column_role)
        if year_state < 0 or role_state < 0 or column_role_state < 0:
            continue
        # Explicit plan/actual lanes are strict: absence of recovered role is
        # not a candidate assignment. The verifier separately records UNKNOWN
        # for neural baselines that still point at such a cell.
        if column_role and column_role_state == 0:
            continue
        if (
            required_unit_family
            and cell.unit_key
            and cell.unit_key != required_unit_family
        ):
            continue
        metric = _metric_context_score(metric_text, cell)
        score = (
            metric
            + 2.5 * max(0, year_state)
            + 1.5 * max(0, role_state)
            + 2.5 * max(0, column_role_state)
        )
        score += 0.15 * max(0.0, cell.relevance)
        proofs: list[str] = []
        if metric >= 0.5:
            proofs.append("metric_slot_match")
        if year_state > 0:
            proofs.append("period_slot_match")
        elif role_state > 0:
            proofs.append("period_role_match")
        if column_role_state > 0:
            proofs.append("column_role_match")
        if plan.entity_mode == "named" and _named_entity_match(plan, cell):
            score += 2.0
            proofs.append("named_entity_match")
        elif plan.entity_mode == "reporting" and not cell.embedded_entity:
            score += 0.75
            proofs.append("reporting_entity_match")
        if cell.unit_key:
            score += 0.10
        if required_unit_family and cell.unit_key == required_unit_family:
            score += 0.50
            proofs.append("unit_slot_match")
        rows.append((cell, score, tuple(proofs)))

    if not rows:
        return []
    best_metric = max(_metric_context_score(metric_text, row[0]) for row in rows)
    if best_metric < 0.5:
        return []
    rows = [
        row
        for row in rows
        if _metric_context_score(metric_text, row[0]) >= best_metric - 0.75
    ]
    if year:
        requested_period_rows = [
            row for row in rows if _year_state(row[0], year, plan) > 0
        ]
        if requested_period_rows:
            # Apply period dominance only after metric-slot narrowing.  This
            # keeps an unrelated requested-year cell from suppressing sparse
            # OCR evidence for the actual operand role.
            rows = requested_period_rows
    if plan.entity_mode == "named":
        named = [row for row in rows if _named_entity_match(plan, row[0])]
        if named:
            rows = named
    elif plan.entity_mode == "reporting":
        reporting = [row for row in rows if not row[0].embedded_entity]
        if reporting:
            rows = reporting
    return sorted(
        rows,
        key=lambda row: (row[1], row[0].relevance, -row[0].index),
        reverse=True,
    )[:limit]


def _pair_compatible(
    graph: EvidenceGraph,
    left: EvidenceCell,
    right: EvidenceCell,
    *,
    operation: str,
    require_same_metric: bool,
) -> tuple[bool, float, tuple[str, ...]]:
    if left.index == right.index:
        return False, 0.0, ()
    if require_same_metric and left.metric_key and right.metric_key:
        if left.metric_key != right.metric_key:
            return False, 0.0, ()
    left_entity = _entity_key(left)
    right_entity = _entity_key(right)
    if left_entity and right_entity and left_entity != right_entity:
        return False, 0.0, ()
    if (
        operation == "MULTIPLY"
        and left.unit_key == right.unit_key == "share_count"
    ):
        return False, 0.0, ()
    comparable = operation in {
        "ADD",
        "SUBTRACT",
        "RATIO_PERCENT",
        "PERCENT_CHANGE",
        "AVERAGE",
        "SUM3",
        "SUM4",
    }
    if comparable and left.unit_key and right.unit_key:
        if left.unit_key != right.unit_key:
            return False, 0.0, ()

    score = 0.0
    proofs: list[str] = []
    relation_features = [
        (EvidenceRelation.SAME_METRIC, "same_metric", 2.0),
        (EvidenceRelation.SAME_ENTITY, "same_entity", 1.0),
        (EvidenceRelation.SAME_TABLE, "same_table", 0.75),
        (EvidenceRelation.SAME_ROW, "same_row", 1.5),
    ]
    # Equal units are useful for additive/ratio reasoning, but are not positive
    # evidence for multiplication: currency * currency has a different
    # dimension. Keep it UNKNOWN until operand-role unit algebra is available.
    if operation != "MULTIPLY":
        relation_features.append(
            (EvidenceRelation.SAME_UNIT, "same_unit", 0.5)
        )
    for relation, name, weight in relation_features:
        if graph.has_relation(left.index, right.index, relation):
            score += weight
            proofs.append(name)
    return True, score, tuple(proofs)


def _temporal_slots(plan: QuestionPlan) -> tuple[tuple[str, str], tuple[str, str]]:
    years = plan.required_operand_years
    if len(years) >= 2:
        return (years[0], ""), (years[1], "")
    if plan.direction_intent is DirectionIntent.DECREASE_AMOUNT:
        return ("", "prior"), ("", "current")
    return ("", "current"), ("", "prior")


def _temporal_proposals(
    graph: EvidenceGraph,
    plan: QuestionPlan,
    operation: str,
    focus_text: str,
    *,
    max_programs: int,
) -> TypedSynthesisResult:
    left_period, right_period = _temporal_slots(plan)
    required_unit = (
        "currency" if plan.requested_unit_family == "currency" else ""
    )
    left = _slot_candidates(
        graph,
        plan,
        metric_text=focus_text,
        year=left_period[0],
        period_role=left_period[1],
        required_unit_family=required_unit,
    )
    right = _slot_candidates(
        graph,
        plan,
        metric_text=focus_text,
        year=right_period[0],
        period_role=right_period[1],
        required_unit_family=required_unit,
    )
    proposals: dict[ReasoningProgram, TypedProgramProposal] = {}
    for left_row, right_row in product(left, right):
        left_cell, left_score, left_proofs = left_row
        right_cell, right_score, right_proofs = right_row
        compatible, relation_score, relation_proofs = _pair_compatible(
            graph,
            left_cell,
            right_cell,
            operation=operation,
            require_same_metric=True,
        )
        if not compatible:
            continue
        operands = (left_cell.index, right_cell.index)
        if operation in COMMUTATIVE:
            operands = tuple(sorted(operands))
        program = ReasoningProgram(operation, tuple(operands))
        proofs = tuple(
            dict.fromkeys((*left_proofs, *right_proofs, *relation_proofs))
        )
        score = left_score + right_score + relation_score
        current = proposals.get(program)
        if current is None or score > current.type_score:
            proposals[program] = TypedProgramProposal(program, score, proofs)
    ranked = tuple(
        sorted(
            proposals.values(),
            key=lambda item: (item.type_score, tuple(-i for i in item.program.operands)),
            reverse=True,
        )[:max_programs]
    )
    return TypedSynthesisResult(
        operation=operation,
        template="temporal_same_metric_pair",
        proposals=ranked,
        slot_candidate_counts=(len(left), len(right)),
        status="ok" if ranked else "no_typed_assignment",
    )


def _role_proposals(
    graph: EvidenceGraph,
    plan: QuestionPlan,
    operation: str,
    role_texts: Sequence[str],
    *,
    required_unit_families: Sequence[str] | None = None,
    max_programs: int,
) -> TypedSynthesisResult:
    arity = OPERATION_ARITY[operation]
    if len(role_texts) != arity:
        return TypedSynthesisResult(
            operation, "explicit_operand_roles", (), (), "missing_operand_roles"
        )
    if (
        required_unit_families is None
        and operation == "MULTIPLY"
        and arity == 2
        and "menh gia" in normalize_text(role_texts[1])
    ):
        required_unit_families = ("share_count", "currency_per_share")
    requested_year = plan.requested_years[-1] if plan.requested_years else ""
    if required_unit_families is not None and len(required_unit_families) != arity:
        return TypedSynthesisResult(
            operation, "explicit_operand_roles", (), (), "invalid_unit_roles"
        )
    default_required_unit = (
        "currency" if plan.requested_unit_family == "currency" else ""
    )
    slots = []
    for index, text in enumerate(role_texts):
        rows = _slot_candidates(
            graph,
            plan,
            metric_text=text,
            year=requested_year,
            required_unit_family=(
                required_unit_families[index]
                if required_unit_families is not None
                else default_required_unit
            ),
            limit=12 if arity <= 2 else 6,
        )
        normalized_role = normalize_text(text)
        if " va " in f" {normalized_role} ":
            role_terms = set(normalized_role.split()) - {
                "cac",
                "cua",
                "khoan",
                "va",
            }
            rows = [
                row
                for row in rows
                if not (
                    (label := normalize_text(row[0].metric_label))
                    and not role_terms.issubset(set(label.split()))
                )
            ]
        slots.append(rows)
    if any(not slot for slot in slots):
        return TypedSynthesisResult(
            operation,
            "explicit_operand_roles",
            (),
            tuple(len(slot) for slot in slots),
            "empty_typed_slot",
        )

    beams: list[tuple[tuple[EvidenceCell, ...], float, tuple[str, ...]]] = [
        ((), 0.0, ())
    ]
    for slot in slots:
        expanded: list[tuple[tuple[EvidenceCell, ...], float, tuple[str, ...]]] = []
        for cells, score, proofs in beams:
            for cell, cell_score, cell_proofs in slot:
                if any(item.index == cell.index for item in cells):
                    continue
                if cells:
                    compatible, relation_score, relation_proofs = _pair_compatible(
                        graph,
                        cells[0],
                        cell,
                        operation=operation,
                        require_same_metric=False,
                    )
                    if not compatible:
                        continue
                else:
                    relation_score, relation_proofs = 0.0, ()
                expanded.append(
                    (
                        (*cells, cell),
                        score + cell_score + relation_score,
                        tuple(
                            dict.fromkeys(
                                (*proofs, *cell_proofs, *relation_proofs)
                            )
                        ),
                    )
                )
        beams = sorted(expanded, key=lambda item: item[1], reverse=True)[:max_programs]

    proposals: dict[ReasoningProgram, TypedProgramProposal] = {}
    for cells, score, proofs in beams:
        operands = tuple(cell.index for cell in cells)
        if operation in COMMUTATIVE:
            operands = tuple(sorted(operands))
        program = ReasoningProgram(operation, operands)
        current = proposals.get(program)
        if current is None or score > current.type_score:
            proposals[program] = TypedProgramProposal(program, score, proofs)
    ranked = tuple(
        sorted(
            proposals.values(), key=lambda item: item.type_score, reverse=True
        )[:max_programs]
    )
    return TypedSynthesisResult(
        operation=operation,
        template="explicit_operand_roles",
        proposals=ranked,
        slot_candidate_counts=tuple(len(slot) for slot in slots),
        status="ok" if ranked else "no_typed_assignment",
    )


def _column_role_proposals(
    graph: EvidenceGraph,
    plan: QuestionPlan,
    operation: str,
    focus_text: str,
    *,
    max_programs: int,
) -> TypedSynthesisResult:
    arity = OPERATION_ARITY[operation]
    if len(plan.column_roles) != arity:
        return TypedSynthesisResult(
            operation,
            "explicit_column_roles",
            (),
            (),
            "missing_column_roles",
        )
    requested_year = plan.requested_years[-1] if plan.requested_years else ""
    required_unit = (
        "currency" if plan.requested_unit_family == "currency" else ""
    )
    slots = [
        _slot_candidates(
            graph,
            plan,
            metric_text=focus_text,
            year=requested_year,
            column_role=role,
            required_unit_family=required_unit,
            limit=16,
        )
        for role in plan.column_roles
    ]
    if any(not slot for slot in slots):
        return TypedSynthesisResult(
            operation,
            "explicit_column_roles",
            (),
            tuple(len(slot) for slot in slots),
            "empty_column_role_slot",
        )

    proposals: dict[ReasoningProgram, TypedProgramProposal] = {}
    for assignment in product(*slots):
        cells = tuple(item[0] for item in assignment)
        if len({cell.index for cell in cells}) != len(cells):
            continue
        relation_score = 0.0
        relation_proofs: tuple[str, ...] = ()
        if len(cells) >= 2:
            compatible, relation_score, relation_proofs = _pair_compatible(
                graph,
                cells[0],
                cells[1],
                operation=operation,
                require_same_metric=True,
            )
            if not compatible:
                continue
        operands = tuple(cell.index for cell in cells)
        if operation in COMMUTATIVE:
            operands = tuple(sorted(operands))
        program = ReasoningProgram(operation, operands)
        score = sum(item[1] for item in assignment) + relation_score
        proofs = tuple(
            dict.fromkeys(
                (
                    *(proof for item in assignment for proof in item[2]),
                    *relation_proofs,
                )
            )
        )
        current = proposals.get(program)
        if current is None or score > current.type_score:
            proposals[program] = TypedProgramProposal(program, score, proofs)
    ranked = tuple(
        sorted(
            proposals.values(),
            key=lambda item: (
                item.type_score,
                tuple(-index for index in item.program.operands),
            ),
            reverse=True,
        )[:max_programs]
    )
    return TypedSynthesisResult(
        operation,
        "explicit_column_roles",
        ranked,
        tuple(len(slot) for slot in slots),
        "ok" if ranked else "no_typed_assignment",
    )


def _question_anchored_add_proposals(
    graph: EvidenceGraph,
    plan: QuestionPlan,
    focus_text: str,
    *,
    max_programs: int,
) -> TypedSynthesisResult:
    """Recover ADD pairs when ``va`` also occurs inside metric labels.

    Splitting a Vietnamese question on the final conjunction is ambiguous for
    rows such as ``Tien gui va vay ...``.  Full row labels explicitly present
    in the question are stronger report-independent anchors.  This path only
    proposes distinct metrics from the same table with compatible units; it
    broadens recall but does not bypass verifier/runtime gates.
    """

    normalized_question = f" {normalize_text(focus_text)} "
    asks_current = " cuoi nam " in normalized_question
    asks_prior = " dau nam " in normalized_question
    anchored: list[EvidenceCell] = []
    for cell in graph.cells:
        label = normalize_text(cell.metric_label).strip()
        if len(_terms(label)) < 2 or f" {label} " not in normalized_question:
            continue
        if asks_current and not asks_prior and cell.period_role == "prior":
            continue
        if asks_prior and not asks_current and cell.period_role == "current":
            continue
        anchored.append(cell)

    if plan.entity_mode == "named":
        named = [cell for cell in anchored if _named_entity_match(plan, cell)]
        if named:
            anchored = named
    elif plan.entity_mode == "reporting":
        reporting = [cell for cell in anchored if not cell.embedded_entity]
        if reporting:
            anchored = reporting

    proposals: dict[ReasoningProgram, TypedProgramProposal] = {}
    for left_index, left in enumerate(anchored):
        for right in anchored[left_index + 1 :]:
            if left.metric_key and left.metric_key == right.metric_key:
                continue
            if not graph.has_relation(
                left.index, right.index, EvidenceRelation.SAME_TABLE
            ):
                continue
            compatible, relation_score, relation_proofs = _pair_compatible(
                graph,
                left,
                right,
                operation="ADD",
                require_same_metric=False,
            )
            if not compatible:
                continue
            program = ReasoningProgram("ADD", tuple(sorted((left.index, right.index))))
            proofs = tuple(
                dict.fromkeys(
                    (
                        "question_metric_anchor",
                        *relation_proofs,
                    )
                )
            )
            score = 4.0 + relation_score + left.relevance + right.relevance
            proposals[program] = TypedProgramProposal(program, score, proofs)
    ranked = tuple(
        sorted(
            proposals.values(),
            key=lambda item: (item.type_score, tuple(-i for i in item.program.operands)),
            reverse=True,
        )[:max_programs]
    )
    return TypedSynthesisResult(
        operation="ADD",
        template="question_metric_anchors",
        proposals=ranked,
        slot_candidate_counts=(len(anchored), len(anchored)),
        status="ok" if ranked else "no_question_metric_anchors",
    )


def _merge_synthesis_results(
    primary: TypedSynthesisResult,
    additive: TypedSynthesisResult,
    *,
    max_programs: int,
) -> TypedSynthesisResult:
    proposals: dict[ReasoningProgram, TypedProgramProposal] = {
        item.program: item for item in primary.proposals
    }
    for item in additive.proposals:
        current = proposals.get(item.program)
        if current is None or item.type_score > current.type_score:
            proposals[item.program] = item
    ranked = tuple(
        sorted(
            proposals.values(),
            key=lambda item: (item.type_score, tuple(-i for i in item.program.operands)),
            reverse=True,
        )[:max_programs]
    )
    return TypedSynthesisResult(
        operation=primary.operation,
        template=f"{primary.template}+{additive.template}",
        proposals=ranked,
        slot_candidate_counts=primary.slot_candidate_counts,
        status="ok" if ranked else primary.status,
    )


def _slot_required_unit(unit_family: str) -> str:
    if unit_family == "currency":
        return "currency"
    if unit_family in {"percent", "ratio", "share_count", "currency_per_share"}:
        return unit_family
    return ""


def _materialize_typed_intent_dag(
    plan: QuestionPlan,
    bindings: dict[str, int],
    *,
    candidate_count: int,
) -> ReasoningDAGProgram:
    nodes: list[ProgramDAGNode] = []
    produced: set[str] = set()
    for intent_node in plan.program_dag:
        inputs: list[ProgramDAGInput] = []
        for reference in intent_node.inputs:
            if reference in bindings:
                inputs.append(ProgramDAGInput.candidate(bindings[reference]))
            elif reference in produced:
                inputs.append(ProgramDAGInput.node(reference))
            else:
                raise ValueError(f"unresolved typed DAG input: {reference}")
        nodes.append(
            ProgramDAGNode(
                node_id=intent_node.id,
                operation=intent_node.operation,
                inputs=tuple(inputs),
                result_unit_family=intent_node.output_unit_family,
            )
        )
        produced.add(intent_node.id)
    dag = ReasoningDAGProgram(tuple(nodes), plan.program_dag[-1].id)
    dag.validate(candidate_count)
    return dag


def _carrier_program(
    plan: QuestionPlan,
    operation: str,
    bindings: dict[str, int],
) -> ReasoningProgram:
    """Build a flat compatibility carrier while the attached DAG is authoritative."""

    arity = OPERATION_ARITY[operation]
    ordered = [bindings[slot.id] for slot in plan.operand_slots]
    unique = list(dict.fromkeys(ordered))
    if arity == 1:
        operands = tuple(unique[:1])
    elif arity == 2 and plan.reference_base in bindings:
        denominator = bindings[plan.reference_base]
        numerator = next((item for item in unique if item != denominator), None)
        if numerator is None:
            raise ValueError("typed DAG carrier has no distinct numerator")
        operands = (numerator, denominator)
    else:
        operands = tuple(unique[:arity])
    if len(operands) != arity:
        raise ValueError("typed DAG carrier arity is incomplete")
    if operation in COMMUTATIVE:
        operands = tuple(sorted(operands))
    return ReasoningProgram(operation, operands)


def _typed_operand_slot_dag_proposals(
    graph: EvidenceGraph,
    plan: QuestionPlan,
    operation: str,
    *,
    max_programs: int,
) -> TypedSynthesisResult:
    """Bind immutable compiler slots and lower the compiler DAG before ranking."""

    if not plan.operand_slots or not plan.program_dag:
        return TypedSynthesisResult(
            operation, "typed_operand_slot_dag", (), (), "missing_compiler_dag"
        )
    if plan.program_dag[-1].operation != operation:
        return TypedSynthesisResult(
            operation, "typed_operand_slot_dag", (), (), "root_operation_mismatch"
        )
    slots = [
        _slot_candidates(
            graph,
            plan,
            metric_text=slot.metric,
            year=slot.year,
            column_role=slot.column_role,
            required_unit_family=_slot_required_unit(slot.unit_family),
            limit=12 if len(plan.operand_slots) <= 2 else 6,
        )
        for slot in plan.operand_slots
    ]
    if any(not rows for rows in slots):
        return TypedSynthesisResult(
            operation,
            "typed_operand_slot_dag",
            (),
            tuple(len(rows) for rows in slots),
            "empty_typed_operand_slot",
        )

    proposals: dict[ReasoningProgram, TypedProgramProposal] = {}
    for assignment in product(*slots):
        cells = tuple(item[0] for item in assignment)
        if len({cell.index for cell in cells}) != len(cells):
            continue
        score = sum(item[1] for item in assignment)
        proofs: list[str] = [proof for item in assignment for proof in item[2]]
        compatible = True
        for left_index, left in enumerate(cells):
            for right in cells[left_index + 1 :]:
                pair_ok, pair_score, pair_proofs = _pair_compatible(
                    graph,
                    left,
                    right,
                    operation=operation,
                    require_same_metric=False,
                )
                if not pair_ok:
                    compatible = False
                    break
                score += pair_score
                proofs.extend(pair_proofs)
            if not compatible:
                break
        if not compatible:
            continue
        bindings = {
            slot.id: cell.index for slot, cell in zip(plan.operand_slots, cells)
        }
        try:
            dag = _materialize_typed_intent_dag(
                plan, bindings, candidate_count=len(graph.cells)
            )
            program = _carrier_program(plan, operation, bindings)
            program.validate(len(graph.cells))
        except ValueError:
            continue
        proposal = TypedProgramProposal(
            program=program,
            type_score=score,
            proofs=tuple(dict.fromkeys((*proofs, "typed_operand_slot_dag"))),
            program_dag=dag,
            slot_bindings=tuple(
                (slot.id, bindings[slot.id]) for slot in plan.operand_slots
            ),
        )
        current = proposals.get(program)
        if current is None or proposal.type_score > current.type_score:
            proposals[program] = proposal
    ranked = tuple(
        sorted(
            proposals.values(),
            key=lambda item: (
                item.type_score,
                tuple(-index for index in item.program.operands),
            ),
            reverse=True,
        )[:max_programs]
    )
    return TypedSynthesisResult(
        operation=operation,
        template="typed_operand_slot_dag",
        proposals=ranked,
        slot_candidate_counts=tuple(len(rows) for rows in slots),
        status="ok" if ranked else "no_typed_dag_assignment",
    )


def synthesize_typed_programs(
    plan: QuestionPlan,
    operation: str,
    graph: EvidenceGraph,
    *,
    focus_text: str,
    operand_role_texts: Sequence[str] | None = None,
    operand_unit_families: Sequence[str] | None = None,
    max_programs: int = 128,
) -> TypedSynthesisResult:
    """Compile a typed query into executable program candidates.

    The synthesizer is report-independent: it uses semantic metric text and
    explicit period/entity/unit/row relations, never issuer names, page numbers
    or fixed years. Missing metadata remains an unknown soft signal. Explicit
    contradictions are excluded before neural scoring.
    """

    if operation not in OPERATION_ARITY:
        return TypedSynthesisResult(
            operation, "unsupported", (), (), "unsupported_operation"
        )
    arity = OPERATION_ARITY[operation]
    if not graph.cells:
        return TypedSynthesisResult(operation, "empty", (), (), "empty_graph")
    if operation in {"SUBTRACT", "PERCENT_CHANGE"} and plan.direction_intent in {
        DirectionIntent.AMBIGUOUS,
        DirectionIntent.ABSOLUTE_GAP,
    }:
        return TypedSynthesisResult(
            operation,
            "direction_contract",
            (),
            tuple(0 for _ in range(arity)),
            "direction_not_executable",
        )
    if plan.operand_slots and plan.program_dag:
        return _typed_operand_slot_dag_proposals(
            graph,
            plan,
            operation,
            max_programs=max_programs,
        )
    if operation == "SELECT":
        requested_year = plan.requested_years[-1] if plan.requested_years else ""
        requested_column_role = (
            plan.column_roles[0] if len(plan.column_roles) == 1 else ""
        )
        slot = _slot_candidates(
            graph,
            plan,
            metric_text=focus_text,
            year=requested_year,
            column_role=requested_column_role,
            required_unit_family=plan.requested_unit_family,
            limit=max_programs,
        )
        proposals = tuple(
            TypedProgramProposal(
                ReasoningProgram("SELECT", (cell.index,)), score, proofs
            )
            for cell, score, proofs in slot
        )
        return TypedSynthesisResult(
            operation,
            "typed_lookup",
            proposals,
            (len(slot),),
            "ok" if proposals else "empty_typed_slot",
        )
    if plan.column_roles:
        return _column_role_proposals(
            graph,
            plan,
            operation,
            focus_text,
            max_programs=max_programs,
        )
    temporal = operation in {"SUBTRACT", "PERCENT_CHANGE", "AVERAGE"} and (
        plan.temporal_change
        or len(plan.requested_years) >= 2
        or (
            operand_role_texts is None
            and bool(
                {cell.period_role for cell in graph.cells}
                >= {"current", "prior"}
            )
        )
    )
    if temporal:
        return _temporal_proposals(
            graph,
            plan,
            operation,
            focus_text,
            max_programs=max_programs,
        )
    if operation == "ADD" and plan.addition_role_hypotheses:
        results = [
            _role_proposals(
                graph,
                plan,
                operation,
                hypothesis,
                required_unit_families=operand_unit_families,
                max_programs=max_programs,
            )
            for hypothesis in plan.addition_role_hypotheses
        ]
        merged = results[0]
        for result in results[1:]:
            merged = _merge_synthesis_results(
                merged, result, max_programs=max_programs
            )
        return merged
    if operand_role_texts is not None:
        role_result = _role_proposals(
            graph,
            plan,
            operation,
            operand_role_texts,
            required_unit_families=operand_unit_families,
            max_programs=max_programs,
        )
        if operation == "ADD":
            return _merge_synthesis_results(
                role_result,
                _question_anchored_add_proposals(
                    graph, plan, focus_text, max_programs=max_programs
                ),
                max_programs=max_programs,
            )
        return role_result
    if operation == "ADD":
        return _question_anchored_add_proposals(
            graph, plan, focus_text, max_programs=max_programs
        )
    return TypedSynthesisResult(
        operation,
        "unresolved",
        (),
        tuple(0 for _ in range(arity)),
        "missing_operand_roles",
    )
