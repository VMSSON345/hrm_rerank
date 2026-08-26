from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
import re
from typing import Sequence

from .evidence_schema import EvidenceCell, EvidenceGraph, build_evidence_graph
from .direction_contract import (
    assess_program_direction,
    reversed_pair_is_proven_negative,
)
from .ontology import normalize_text
from .program_data import (
    COMMUTATIVE,
    ProgramExample,
    ReasoningProgram,
    canonical_program_equal,
)
from .question_planner import QuestionPlan, compile_question_plan
from .symbolic_verifier import VerificationState, verify_program


RANK_POSITIVE_CLASSES = frozenset({"canonical_exact", "grounded_equivalent"})
PHASE_C_SCHEMA = "v12b2-canonical-multi-positive-v3"

_NON_VALUE_ROW_TYPES = frozenset(
    {"header", "section_header", "column_header", "metadata", "narrative"}
)
_NON_VALUE_METRIC_LABELS = frozenset(
    {"chi tieu", "ma so", "thuyet minh", "don vi tinh", "noi dung"}
)


def parse_program_key(value: str) -> ReasoningProgram:
    operation, separator, raw_operands = value.partition(":")
    if not separator or not operation:
        raise ValueError(f"Invalid program key: {value}")
    operands = tuple(
        int(item) for item in raw_operands.split(",") if item != ""
    )
    return ReasoningProgram(operation, operands)


def canonical_program_match(
    predicted: ReasoningProgram, gold: ReasoningProgram
) -> bool:
    return canonical_program_equal(predicted, gold)


def _cell_signature(cell: EvidenceCell, plan: QuestionPlan) -> tuple[object, ...]:
    entity = re.sub(
        r"[^a-z0-9]+",
        "",
        normalize_text(cell.entity_id or cell.entity),
    )
    return (
        cell.metric_key,
        tuple(sorted(cell.effective_years(plan))),
        cell.period_role,
        entity,
        cell.unit_key,
        str(cell.value),
    )


def semantic_program_signature(
    program: ReasoningProgram,
    cells: Sequence[EvidenceCell],
    plan: QuestionPlan,
) -> tuple[object, ...]:
    signatures = tuple(_cell_signature(cells[index], plan) for index in program.operands)
    if program.operation in COMMUTATIVE:
        signatures = tuple(sorted(signatures))
    return program.operation, signatures


def semantic_signature_sha256(signature: tuple[object, ...]) -> str:
    encoded = json.dumps(signature, ensure_ascii=False, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def supervision_integrity_issues(
    example: ProgramExample,
    *,
    graph: EvidenceGraph | None = None,
    plan: QuestionPlan | None = None,
) -> tuple[str, ...]:
    """Return hard contradictions that make a gold program unsafe to train.

    This gate is intentionally report-independent.  It validates the gold
    program against the same typed evidence and question contract used at
    runtime, and applies a stricter role/unit obligation to par-value
    multiplication where UNKNOWN provenance would otherwise teach a wrong
    table-neighbour pointer.
    """

    graph = graph or build_evidence_graph(example.candidates)
    plan = plan or compile_question_plan(example.question)
    issues: list[str] = []

    direction = assess_program_direction(plan, example.program, graph)
    direction_issue = {
        "direction_reversed": "gold_direction_reversed",
        "ambiguous_wording": "gold_direction_ambiguous",
        "unsupported_absolute_gap": "gold_direction_unsupported_absolute_gap",
        "wrong_metric": "gold_direction_wrong_metric",
        "wrong_or_missing_metric": "gold_endpoint_missing_metric",
    }.get(direction.outcome)
    if direction_issue:
        issues.append(direction_issue)

    selected = tuple(graph.cells[index] for index in example.program.operands)
    for cell in selected:
        row_type = normalize_text(cell.row_type)
        metric = normalize_text(cell.metric_label).strip()
        if row_type in _NON_VALUE_ROW_TYPES:
            issues.append(f"gold_endpoint_non_value_row:{row_type}")
        if cell.is_year_like:
            issues.append("gold_endpoint_year_like")
        if metric in _NON_VALUE_METRIC_LABELS:
            issues.append("gold_endpoint_header_label")
        if not metric and (cell.column or cell.period or cell.table_id):
            issues.append("gold_endpoint_missing_metric")

    verification = verify_program(plan, example.program, graph)
    if verification.state is VerificationState.INVALID:
        issues.extend(
            f"gold_verifier:{reason}"
            for reason in verification.reasons
            # Planner operation granularity (notably ADD versus SUM3/SUM4) is
            # audited separately. It is not evidence that the gold pointers
            # themselves are unsafe for a fixed-operation reranker.
            if reason != "operation_match"
            and not (reason == "temporal_direction" and direction_issue)
        )

    par_value_multiply = (
        example.program.operation == "MULTIPLY"
        and len(plan.operand_roles) == 2
        and any("menh gia" in normalize_text(role) for role in plan.operand_roles)
    )
    if par_value_multiply:
        selected_units = {
            graph.cells[index].unit_key for index in example.program.operands
        }
        if selected_units != {"share_count", "currency_per_share"}:
            issues.append("multiply_par_value_unit_algebra_unproven")

    return tuple(dict.fromkeys(issues))


def positive_class_for_program(
    program: ReasoningProgram,
    example: ProgramExample,
    *,
    execution_correct: bool,
    training_state: str,
    graph: EvidenceGraph | None = None,
    plan: QuestionPlan | None = None,
) -> str:
    if canonical_program_match(program, example.program):
        return "canonical_exact"
    graph = graph or build_evidence_graph(example.candidates)
    plan = plan or compile_question_plan(example.question)
    if (
        execution_correct
        and training_state == "VALID"
        and semantic_program_signature(program, graph.cells, plan)
        == semantic_program_signature(example.program, graph.cells, plan)
    ):
        return "grounded_equivalent"
    if execution_correct:
        return "execution_only"
    return "negative"


def _operand_cells(
    program: ReasoningProgram, cells: Sequence[EvidenceCell]
) -> tuple[EvidenceCell, ...]:
    return tuple(cells[index] for index in program.operands)


def _hard_negative_types(
    candidate: dict[str, object],
    program: ReasoningProgram,
    example: ProgramExample,
    *,
    high_neural_rank: bool,
    graph: EvidenceGraph | None = None,
    plan: QuestionPlan | None = None,
) -> tuple[str, ...]:
    graph = graph or build_evidence_graph(example.candidates)
    plan = plan or compile_question_plan(example.question)
    gold = example.program
    types: list[str] = []
    if bool(candidate.get("execution_correct")):
        types.append("execution_only_same_result")
    if len(set(program.operands)) != len(program.operands):
        types.append("duplicate_pointer")
    if program.operation != gold.operation:
        types.append("wrong_operation")
    selected = _operand_cells(program, graph.cells)
    gold_cells = _operand_cells(gold, graph.cells)
    if program.operation == gold.operation and len(selected) == len(gold_cells):
        selected_signatures = tuple(_cell_signature(cell, plan) for cell in selected)
        gold_signatures = tuple(_cell_signature(cell, plan) for cell in gold_cells)
        exact_reversal = (
            program.operation not in COMMUTATIVE
            and len(selected_signatures) == 2
            and selected_signatures == tuple(reversed(gold_signatures))
        )
        if exact_reversal:
            if program.operation in {"SUBTRACT", "PERCENT_CHANGE"}:
                if reversed_pair_is_proven_negative(plan, gold, graph):
                    types.append("reversed_direction_or_roles")
                else:
                    return ()
            else:
                types.append("reversed_direction_or_roles")

        selected_periods = {
            (signature[1], signature[2]) for signature in selected_signatures
        }
        gold_periods = {(signature[1], signature[2]) for signature in gold_signatures}
        if selected_periods != gold_periods:
            types.append("wrong_period")

        selected_metrics = {signature[0] for signature in selected_signatures}
        gold_metrics = {signature[0] for signature in gold_signatures}
        if selected_metrics != gold_metrics:
            selected_tables = {cell.table_id for cell in selected if cell.table_id}
            gold_tables = {cell.table_id for cell in gold_cells if cell.table_id}
            types.append(
                "wrong_metric_same_table"
                if selected_tables & gold_tables
                else "wrong_metric"
            )

        selected_entities = {signature[3] for signature in selected_signatures}
        gold_entities = {signature[3] for signature in gold_signatures}
        if selected_entities != gold_entities:
            types.append("wrong_entity_or_scope")

        selected_units = {signature[4] for signature in selected_signatures}
        gold_units = {signature[4] for signature in gold_signatures}
        if selected_units != gold_units:
            types.append("wrong_unit_or_constant")

    if high_neural_rank and (
        str(candidate.get("proof_contract_state", "")) == "UNKNOWN"
        or float(candidate.get("required_unknown_fraction", 0.0)) > 0.0
    ):
        types.append("high_neural_incomplete_proof")
    return tuple(dict.fromkeys(types or ["other_incorrect"]))


def build_phase_c_record(
    record: dict[str, object], example: ProgramExample
) -> tuple[dict[str, object], dict[str, object]]:
    raw_candidates = record.get("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("Phase-C record candidates must be a list")
    graph = build_evidence_graph(example.candidates)
    plan = compile_question_plan(example.question)
    integrity_issues = supervision_integrity_issues(
        example, graph=graph, plan=plan
    )
    candidate_count = len(raw_candidates)
    high_neural_limit = max(1, (candidate_count + 3) // 4)
    output_candidates: list[dict[str, object]] = []
    positive_program_keys: set[str] = set()
    positive_classes: Counter[str] = Counter()
    hard_negative_types: Counter[str] = Counter()
    reversed_pair_negatives_withheld = 0
    label_contradictions = 0
    execution_only_promoted = 0

    for position, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            continue
        candidate = dict(raw_candidate)
        members = candidate.get("semantic_equivalent_members")
        if not isinstance(members, list) or not members:
            members = [
                {
                    "program_key": candidate.get("program_key", ""),
                    "positive_class": candidate.get("positive_class", "negative"),
                    "exact_gold": candidate.get("exact_gold", False),
                    "execution_correct": candidate.get("execution_correct", False),
                    "training_state": candidate.get("training_state", "UNKNOWN"),
                }
            ]
        validated_members: list[dict[str, object]] = []
        member_positive = False
        for raw_member in members:
            if not isinstance(raw_member, dict):
                continue
            program_key = str(raw_member.get("program_key", ""))
            program = parse_program_key(program_key)
            computed = positive_class_for_program(
                program,
                example,
                execution_correct=bool(raw_member.get("execution_correct")),
                training_state=str(raw_member.get("training_state", "UNKNOWN")),
                graph=graph,
                plan=plan,
            )
            source_class = str(raw_member.get("positive_class", "negative"))
            label_contradictions += int(computed != source_class)
            is_positive = (
                computed in RANK_POSITIVE_CLASSES and not integrity_issues
            )
            execution_only_promoted += int(
                computed == "execution_only" and is_positive
            )
            member_positive = member_positive or is_positive
            if is_positive:
                positive_program_keys.add(program_key)
                positive_classes[computed] += 1
            validated_members.append(
                {
                    **raw_member,
                    "computed_positive_class": computed,
                    "rank_positive": is_positive,
                }
            )

        representative = parse_program_key(str(candidate.get("program_key", "")))
        signature = semantic_program_signature(representative, graph.cells, plan)
        compact_candidate: dict[str, object] = {
            "program_key": str(candidate.get("program_key", "")),
            "semantic_signature_sha256": semantic_signature_sha256(signature),
            "semantic_equivalent_members": validated_members,
            "rank_positive": member_positive,
            "source_positive_class": str(
                candidate.get("positive_class", "negative")
            ),
            "proof_contract_state": str(
                candidate.get("proof_contract_state", "")
            ),
            "supervision_only": bool(candidate.get("supervision_only", False)),
        }
        if integrity_issues:
            compact_candidate["hard_negative_types"] = []
            compact_candidate["supervision_excluded"] = True
        elif member_positive:
            compact_candidate["hard_negative_types"] = []
        else:
            types = _hard_negative_types(
                candidate,
                representative,
                example,
                high_neural_rank=position < high_neural_limit,
                graph=graph,
                plan=plan,
            )
            compact_candidate["hard_negative_types"] = list(types)
            representative_cells = _operand_cells(representative, graph.cells)
            gold_cells = _operand_cells(example.program, graph.cells)
            exact_unordered_reversal = (
                representative.operation == example.program.operation
                and representative.operation not in COMMUTATIVE
                and len(representative_cells) == len(gold_cells) == 2
                and tuple(
                    _cell_signature(cell, plan) for cell in representative_cells
                )
                == tuple(
                    reversed(
                        tuple(_cell_signature(cell, plan) for cell in gold_cells)
                    )
                )
            )
            if exact_unordered_reversal and not types:
                compact_candidate["hard_negative_withheld"] = (
                    "unproven_direction"
                )
                reversed_pair_negatives_withheld += 1
            hard_negative_types.update(types)
        output_candidates.append(compact_candidate)

    output = {
        "phase_c_schema": PHASE_C_SCHEMA,
        "example_id": str(record.get("example_id", "")),
        "operation": str(record.get("operation", "")),
        "positive_program_keys": sorted(positive_program_keys),
        "has_rank_positive": bool(positive_program_keys),
        "has_hard_negative": bool(hard_negative_types),
        "supervision_integrity_issues": list(integrity_issues),
        "candidates": output_candidates,
    }
    audit = {
        "records": 1,
        "candidates": len(output_candidates),
        "positive_programs": len(positive_program_keys),
        "positive_classes": dict(positive_classes),
        "hard_negative_types": dict(hard_negative_types),
        "label_contradictions": label_contradictions,
        "execution_only_promoted_to_positive": execution_only_promoted,
        "reversed_pair_negatives_withheld": reversed_pair_negatives_withheld,
        "supervision_integrity_eligible": not integrity_issues,
        "supervision_integrity_issues": list(integrity_issues),
        "eligible_positive_and_hard_negative": bool(positive_program_keys)
        and bool(hard_negative_types)
        and not integrity_issues,
    }
    return output, audit
