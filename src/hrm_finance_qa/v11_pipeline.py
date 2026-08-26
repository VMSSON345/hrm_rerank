from __future__ import annotations

from collections import Counter
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import random
import re
from typing import Callable, Sequence

from .evidence_schema import EvidenceGraph, build_evidence_graph
from .ontology import normalize_text
from .question_planner import QuestionPlan, compile_question_plan
from .document_hrm import (
    DocumentHRMBundle,
    ProgramCandidateTrace,
    _example_group_key,
    _execution_is_correct,
)
from .program_data import (
    COMMUTATIVE,
    OPERATIONS,
    ProgramExample,
    ReasoningProgram,
    canonical_program_equal,
)
from .program_reranker import (
    FEATURE_INDEX,
    RERANKER_TRAIN_OPERATIONS,
    RerankerCandidateRecord,
    RerankerExampleRecord,
    features_with_training_state,
)


class V11PipelineError(RuntimeError):
    pass


def _v11_group_key(example: ProgramExample) -> str:
    key = _example_group_key(example)
    if key.startswith(("company:", "ticker:")):
        return key
    sources = sorted(
        {
            candidate.source_name.strip().lower()
            for candidate in example.candidates
            if candidate.source_name.strip()
            and candidate.source_name.strip().lower() != "context"
        }
    )
    if sources:
        return "source:" + "|".join(sources)
    return key


def checkpoint_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _program_key(trace: ProgramCandidateTrace) -> str:
    operands = ",".join(str(index) for index in trace.program.operands)
    return f"{trace.program.operation}:{operands}"


def _canonical_program_match(
    predicted: ReasoningProgram, gold: ReasoningProgram
) -> bool:
    return canonical_program_equal(predicted, gold)


def _cell_signature(
    graph: EvidenceGraph,
    plan: QuestionPlan,
    index: int,
) -> tuple[object, ...]:
    cell = graph.cells[index]
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


def _grounded_program_equivalent(
    predicted: ReasoningProgram,
    gold: ReasoningProgram,
    graph: EvidenceGraph,
    plan: QuestionPlan,
) -> bool:
    if predicted.operation != gold.operation:
        return False
    predicted_signatures = tuple(
        _cell_signature(graph, plan, index) for index in predicted.operands
    )
    gold_signatures = tuple(
        _cell_signature(graph, plan, index) for index in gold.operands
    )
    if predicted.operation in COMMUTATIVE:
        return sorted(predicted_signatures) == sorted(gold_signatures)
    return predicted_signatures == gold_signatures


def _candidate_positive_class(
    trace: ProgramCandidateTrace,
    example: ProgramExample,
    graph: EvidenceGraph,
    plan: QuestionPlan,
    *,
    execution_correct: bool,
) -> str:
    if _canonical_program_match(trace.program, example.program):
        return "canonical_exact"
    if (
        execution_correct
        and _trace_training_state(trace) == "VALID"
        and _grounded_program_equivalent(
            trace.program, example.program, graph, plan
        )
    ):
        return "grounded_equivalent"
    if execution_correct:
        return "execution_only"
    return "negative"


def _diverse_top_k_traces(
    trace: Sequence[ProgramCandidateTrace], *, top_k: int
) -> list[ProgramCandidateTrace]:
    normal = [
        item
        for item in sorted(trace, key=lambda row: row.neural_rank)
        if (
            _trace_training_state(item) != "INVALID"
            and item.reranker_features
            and item.program.operation in RERANKER_TRAIN_OPERATIONS
        )
    ]
    probes = [
        item
        for item in sorted(trace, key=lambda row: row.neural_rank)
        if (
            _trace_training_state(item) == "INVALID"
            and item.reranker_features
            and "directional_reversal_probe" in item.generation_sources
        )
    ]
    if len(normal) <= top_k:
        output = list(normal)
    else:
        output = []
    legacy = next((item for item in normal if item.legacy_winner), None)
    buckets: dict[tuple[str, str, str, int], list[ProgramCandidateTrace]] = {}
    for item in normal:
        if item is legacy:
            continue
        first = item.program.operands[0] if item.program.operands else -1
        buckets.setdefault(
            (
                item.program.operation,
                _trace_training_state(item),
                _trace_generation_family(item),
                first,
            ),
            [],
        ).append(item)
    if not output:
        output = [legacy] if legacy is not None else []
        keys = sorted(
            buckets,
            key=lambda key: buckets[key][0].neural_rank,
        )
        while len(output) < top_k:
            progressed = False
            for key in keys:
                if not buckets[key]:
                    continue
                output.append(buckets[key].pop(0))
                progressed = True
                if len(output) >= top_k:
                    break
            if not progressed:
                break
    # A reversal probe is useful only beside the supported direct program.
    # Preserve that direct candidate and the legacy winner if Top-K is full.
    for probe in probes:
        direct = ReasoningProgram(
            probe.program.operation, tuple(reversed(probe.program.operands))
        )
        if not any(item.program == direct for item in output):
            continue
        if len(output) >= top_k:
            removable = [
                item
                for item in output
                if not item.legacy_winner
                and item.program != direct
                and "directional_reversal_probe" not in item.generation_sources
            ]
            if not removable:
                continue
            output.remove(max(removable, key=lambda item: item.neural_rank))
        output.append(probe)
    output.sort(key=lambda item: item.neural_rank)
    return output


def _trace_training_state(trace: ProgramCandidateTrace) -> str:
    state = trace.training_state or trace.verification_state
    if state not in {"VALID", "UNKNOWN", "INVALID"}:
        return "INVALID"
    if not trace.training_state and not trace.eligible:
        return "INVALID"
    return state


def _cached_candidate_training_state(
    candidate: RerankerCandidateRecord,
) -> str:
    """Upgrade V12-B cache rows to the evidence-supported tri-state policy."""

    if candidate.training_state == "INVALID":
        return "INVALID"
    if candidate.rejection_reasons:
        return "UNKNOWN"
    features = candidate.features
    if (
        candidate.training_state == "VALID"
        and features[FEATURE_INDEX["verifier_state_valid"]] > 0.5
    ):
        return "VALID"
    if (
        features[FEATURE_INDEX["execution_valid"]] < 0.5
        or features[FEATURE_INDEX["operation_match"]] < 0.5
    ):
        return "UNKNOWN"
    strong = (
        "period_coverage",
        "temporal_direction",
    )
    if any(features[FEATURE_INDEX[name]] > 0.5 for name in strong):
        return "VALID"
    relational = (
        "metric_match",
        "entity_match",
        "unit_match",
        "same_table",
        "same_row",
        "graph_same_metric",
        "graph_same_entity",
        "graph_same_table",
        "graph_same_row",
        "graph_same_unit",
        "graph_temporal_order",
    )
    supported = sum(
        features[FEATURE_INDEX[name]] > 0.5 for name in relational
    )
    return "VALID" if supported >= 2 else "UNKNOWN"


def upgrade_reranker_records_to_supported_tristate(
    records: Sequence[RerankerExampleRecord],
) -> tuple[list[RerankerExampleRecord], dict[str, int]]:
    """Migrate cached V12-B evidence to V12-B.1 without rerunning frozen V8."""

    output: list[RerankerExampleRecord] = []
    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    changed = 0
    for record in records:
        candidates: list[RerankerCandidateRecord] = []
        for candidate in record.candidates:
            before[candidate.training_state] += 1
            state = _cached_candidate_training_state(candidate)
            after[state] += 1
            changed += int(
                state != candidate.training_state
                or candidate.features[FEATURE_INDEX["verifier_state_valid"]]
                != float(state == "VALID")
                or candidate.features[FEATURE_INDEX["verifier_state_unknown"]]
                != float(state == "UNKNOWN")
            )
            candidates.append(
                replace(
                    candidate,
                    features=features_with_training_state(
                        candidate.features, state
                    ),
                    training_state=state,
                )
            )
        output.append(replace(record, candidates=tuple(candidates)))
    audit = {
        "records": len(records),
        "candidates": sum(len(item.candidates) for item in records),
        "changed_candidates": changed,
        **{f"before_{key.lower()}": value for key, value in before.items()},
        **{f"after_{key.lower()}": value for key, value in after.items()},
    }
    return output, audit


def _trace_generation_family(trace: ProgramCandidateTrace) -> str:
    sources = set(trace.generation_sources)
    if "v12_3_reference_winner" in sources:
        return "legacy_reference"
    if any(source.startswith("financial_identity:") for source in sources):
        return "financial_identity"
    if "typed_slot_synthesis" in sources:
        return "typed"
    if "directional_reversal_probe" in sources:
        return "directional_reversal"
    if "neural_operation_conditioned_beam" in sources:
        return "operation_conditioned"
    if "neural_baseline" in sources:
        return "neural_baseline"
    return "other"


def build_reranker_records(
    bundle: DocumentHRMBundle,
    examples: Sequence[ProgramExample],
    *,
    top_k: int = 30,
    batch_size: int = 16,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[RerankerExampleRecord], dict[str, object]]:
    if bundle.program_reranker is not None:
        raise V11PipelineError("Candidate cache must be generated by frozen HRM only")
    if top_k < 2:
        raise V11PipelineError("top_k must be at least 2")
    records: list[RerankerExampleRecord] = []
    generated = candidate_sets = gold_in_top_k = execution_in_top_k = no_trace = 0
    rerankable_gold_in_top_k = rerankable_execution_in_top_k = 0
    positive_classes: Counter[str] = Counter()
    synthesis_statuses: Counter[str] = Counter()
    failure_modes: Counter[str] = Counter()
    failure_examples: dict[str, list[dict[str, object]]] = {}
    typed_candidates = typed_gold_candidates = 0
    training_trace_sets = 0
    pre_prune_candidates = post_prune_candidates = 0
    pre_prune_gold = post_prune_gold = pruning_gold_loss = 0
    training_state_counts: Counter[str] = Counter()
    retained_state_counts: Counter[str] = Counter()
    proof_contract_state_counts: Counter[str] = Counter()
    proof_obligation_counts: Counter[str] = Counter()
    gold_false_rejection_constraints: Counter[str] = Counter()
    proof_contradiction_examples: dict[str, list[dict[str, object]]] = {}
    hard_invalid_dropped = soft_unknown_retained = 0
    required_valid_total = required_unknown_total = optional_support_total = 0.0
    gold_verifier_generated = gold_verifier_retained = 0
    per_operation: dict[str, Counter[str]] = {
        operation: Counter() for operation in OPERATIONS
    }

    def record_failure(
        mode: str,
        example: ProgramExample,
        trace: ProgramCandidateTrace | None = None,
    ) -> None:
        rows = failure_examples.setdefault(mode, [])
        if len(rows) >= 40:
            return
        payload: dict[str, object] = {
            "example_id": example.example_id,
            "operation": example.program.operation,
            "question": example.question,
            "gold_program": (
                f"{example.program.operation}:"
                + ",".join(str(index) for index in example.program.operands)
            ),
        }
        if trace is not None:
            payload.update(
                {
                    "candidate_program": _program_key(trace),
                    "rejection_reasons": trace.rejection_reasons,
                    "unknown_reasons": trace.unknown_reasons,
                    "generation_sources": trace.generation_sources,
                    "required_valid_fraction": trace.required_valid_fraction,
                    "required_unknown_fraction": trace.required_unknown_fraction,
                    "optional_support_fraction": trace.optional_support_fraction,
                    "proof_obligations": trace.proof_obligations,
                }
            )
        rows.append(payload)
    for start in range(0, len(examples), batch_size):
        current = list(examples[start : start + batch_size])
        predictions = bundle.predict_many(current)
        for example, prediction in zip(current, predictions):
            stats = per_operation[example.program.operation]
            stats["examples"] += 1
            gold_operands_available = all(
                0 <= index < len(example.candidates)
                for index in example.program.operands
            )
            for status in getattr(prediction, "typed_synthesis_status", ()):
                synthesis_statuses[status] += 1
            if example.program.operation not in RERANKER_TRAIN_OPERATIONS:
                stats["excluded_from_reranker"] += 1
                continue
            if not prediction.program_search_trace:
                no_trace += 1
                stats["no_trace"] += 1
                mode = (
                    "no_trace_gold_operands_available"
                    if gold_operands_available
                    else "no_trace_gold_operands_missing"
                )
                failure_modes[mode] += 1
                stats[mode] += 1
                record_failure(mode, example)
                if getattr(
                    prediction, "typed_program_synthesis_enabled", False
                ):
                    stats["typed_synthesis_no_trace"] += 1
                continue
            canonical_gold_trace = next(
                (
                    item
                    for item in prediction.program_search_trace
                    if _canonical_program_match(
                        item.program, example.program
                    )
                ),
                None,
            )
            if canonical_gold_trace is None:
                mode = (
                    "gold_program_not_generated_with_operands_available"
                    if gold_operands_available
                    else "gold_program_not_generated_operands_missing"
                )
                failure_modes[mode] += 1
                stats[mode] += 1
                record_failure(mode, example)
            else:
                gold_verifier_generated += 1
                stats["gold_verifier_generated"] += 1
                gold_contract_state = (
                    canonical_gold_trace.proof_contract_state
                    or canonical_gold_trace.training_state
                    or canonical_gold_trace.verification_state
                )
                gold_verifier_retained += int(
                    gold_contract_state != "INVALID"
                )
                stats["gold_verifier_retained"] += int(
                    gold_contract_state != "INVALID"
                )
                if gold_contract_state == "INVALID":
                    for obligation in canonical_gold_trace.proof_obligations:
                        if not obligation.startswith("required:") or not obligation.endswith(":INVALID"):
                            continue
                        constraint = obligation.split(":", 2)[1]
                        gold_false_rejection_constraints[constraint] += 1
                        stats[f"gold_false_rejection__{constraint}"] += 1
            if (
                canonical_gold_trace is not None
                and not canonical_gold_trace.eligible
            ):
                failure_modes["gold_program_generated_but_ineligible"] += 1
                stats["gold_program_generated_but_ineligible"] += 1
                record_failure(
                    "gold_program_generated_but_ineligible",
                    example,
                    canonical_gold_trace,
                )
            trainable_trace = [
                item
                for item in prediction.program_search_trace
                if item.reranker_features
                and item.program.operation in RERANKER_TRAIN_OPERATIONS
                and _trace_training_state(item) != "INVALID"
            ]
            training_trace_sets += 1
            for item in prediction.program_search_trace:
                state = _trace_training_state(item)
                training_state_counts[state] += 1
                if state == "INVALID" and (
                    "directional_reversal_probe" not in item.generation_sources
                ):
                    hard_invalid_dropped += 1
                contract_state = (
                    item.proof_contract_state
                    or item.training_state
                    or item.verification_state
                )
                proof_contract_state_counts[contract_state] += 1
                if contract_state != "INVALID":
                    continue
                for obligation in item.proof_obligations:
                    if not obligation.startswith("required:") or not obligation.endswith(":INVALID"):
                        continue
                    constraint = obligation.split(":", 2)[1]
                    rows = proof_contradiction_examples.setdefault(constraint, [])
                    if len(rows) < 40:
                        rows.append(
                            {
                                "example_id": example.example_id,
                                "operation": item.program.operation,
                                "question": example.question,
                                "candidate_program": _program_key(item),
                                "canonical_gold": _canonical_program_match(
                                    item.program, example.program
                                ),
                                "execution_correct": _execution_is_correct(
                                    item.program, example
                                ),
                                "proof_obligations": item.proof_obligations,
                            }
                        )
            pre_prune_candidates += len(trainable_trace)
            pre_gold_found = any(
                _canonical_program_match(item.program, example.program)
                for item in trainable_trace
            )
            pre_prune_gold += int(pre_gold_found)
            selected = _diverse_top_k_traces(
                prediction.program_search_trace, top_k=top_k
            )
            post_prune_candidates += len(selected)
            for item in selected:
                state = _trace_training_state(item)
                retained_state_counts[state] += 1
                soft_unknown_retained += int(state == "UNKNOWN")
                required_valid_total += item.required_valid_fraction
                required_unknown_total += item.required_unknown_fraction
                optional_support_total += item.optional_support_fraction
                proof_obligation_counts.update(item.proof_obligations)
            post_gold_found = any(
                _canonical_program_match(item.program, example.program)
                for item in selected
            )
            post_prune_gold += int(post_gold_found)
            if pre_gold_found and not post_gold_found:
                pruning_gold_loss += 1
                failure_modes["gold_program_lost_by_pruning"] += 1
                stats["gold_program_lost_by_pruning"] += 1
                record_failure("gold_program_lost_by_pruning", example)
            if not selected:
                no_trace += 1
                stats["no_eligible_trace"] += 1
                failure_modes["trace_without_eligible_program"] += 1
                record_failure("trace_without_eligible_program", example)
                continue
            graph = build_evidence_graph(example.candidates)
            plan = compile_question_plan(example.question)
            candidate_rows: list[RerankerCandidateRecord] = []
            for item in selected:
                execution_correct = _execution_is_correct(
                    item.program, example
                )
                training_state = _trace_training_state(item)
                positive_class = _candidate_positive_class(
                    item,
                    example,
                    graph,
                    plan,
                    execution_correct=execution_correct,
                )
                candidate_rows.append(
                    RerankerCandidateRecord(
                        features=features_with_training_state(
                            item.reranker_features, training_state
                        ),
                        program_key=_program_key(item),
                        exact_gold=_canonical_program_match(
                            item.program, example.program
                        ),
                        execution_correct=execution_correct,
                        positive_class=positive_class,
                        generation_sources=item.generation_sources,
                        runtime_eligible=item.eligible,
                        training_state=training_state,
                        proof_contract_state=item.proof_contract_state,
                        rejection_reasons=item.rejection_reasons,
                        unknown_reasons=item.unknown_reasons,
                        required_valid_fraction=item.required_valid_fraction,
                        required_unknown_fraction=item.required_unknown_fraction,
                        optional_support_fraction=item.optional_support_fraction,
                        proof_obligations=item.proof_obligations,
                        supervision_only=(
                            training_state == "INVALID"
                            and "directional_reversal_probe"
                            in item.generation_sources
                        ),
                    )
                )
                positive_classes[positive_class] += 1
                stats[f"positive_class__{positive_class}"] += 1
                if "typed_slot_synthesis" in item.generation_sources:
                    typed_candidates += 1
                    stats["typed_candidates"] += 1
                    if positive_class == "canonical_exact":
                        typed_gold_candidates += 1
                        stats["typed_gold_candidates"] += 1
            candidates = tuple(candidate_rows)
            candidate_sets += 1
            exact_found = any(item.exact_gold for item in candidates)
            canonical_found = any(
                item.positive_class == "canonical_exact"
                for item in candidates
            )
            execution_found = any(item.execution_correct for item in candidates)
            gold_in_top_k += int(exact_found)
            execution_in_top_k += int(execution_found)
            stats["candidate_sets"] += 1
            stats["gold_in_top_k"] += int(exact_found)
            stats["execution_in_top_k"] += int(execution_found)
            stats["candidates"] += len(candidates)
            if len(selected) < 2:
                stats["legacy_fallback_single_candidate"] += 1
                if canonical_found:
                    failure_modes["single_candidate_correct"] += 1
                    stats["single_candidate_correct"] += 1
                elif (
                    canonical_gold_trace is not None
                    and not canonical_gold_trace.eligible
                ):
                    failure_modes["single_candidate_wrong_gold_ineligible"] += 1
                    stats["single_candidate_wrong_gold_ineligible"] += 1
                    record_failure(
                        "single_candidate_wrong_gold_ineligible",
                        example,
                        canonical_gold_trace,
                    )
                else:
                    failure_modes["single_candidate_wrong_gold_not_generated"] += 1
                    stats["single_candidate_wrong_gold_not_generated"] += 1
                    record_failure(
                        "single_candidate_wrong_gold_not_generated",
                        example,
                    )
                continue
            legacy_winner = next(
                (item for item in selected if item.legacy_winner), None
            )
            if canonical_found and (
                legacy_winner is None
                or not _canonical_program_match(
                    legacy_winner.program, example.program
                )
            ):
                failure_modes["ranking_failure_with_gold_eligible"] += 1
                stats["ranking_failure_with_gold_eligible"] += 1
                record_failure(
                    "ranking_failure_with_gold_eligible",
                    example,
                    canonical_gold_trace,
                )
            record = RerankerExampleRecord(
                example_id=example.example_id,
                operation=example.program.operation,
                candidates=candidates,
            )
            records.append(record)
            generated += 1
            rerankable_gold_in_top_k += int(exact_found)
            rerankable_execution_in_top_k += int(execution_found)
            stats["generated"] += 1
        if progress_callback is not None:
            progress_callback(min(start + len(current), len(examples)), len(examples))
    audit = {
        "examples": len(examples),
        "records": generated,
        "candidate_sets": candidate_sets,
        "top_k": top_k,
        "no_trace": no_trace,
        "gold_program_recall_at_k": (
            gold_in_top_k / candidate_sets if candidate_sets else 0.0
        ),
        "execution_recall_at_k": (
            execution_in_top_k / candidate_sets if candidate_sets else 0.0
        ),
        "rerankable_gold_program_recall_at_k": (
            rerankable_gold_in_top_k / generated if generated else 0.0
        ),
        "rerankable_execution_recall_at_k": (
            rerankable_execution_in_top_k / generated if generated else 0.0
        ),
        "invalid_candidates_retained": 0,
        "proven_supervision_invalid_candidates_retained": sum(
            candidate.supervision_only
            for record in records
            for candidate in record.candidates
        ),
        "unknown_as_negative": False,
        "training_state_policy": "applicability-aware-required-only-v3",
        "training_state_counts": dict(training_state_counts),
        "retained_training_state_counts": dict(retained_state_counts),
        "proof_contract_state_counts": dict(proof_contract_state_counts),
        "hard_invalid_candidates_dropped": hard_invalid_dropped,
        "soft_unknown_candidates_retained": soft_unknown_retained,
        "mean_required_valid_fraction": (
            required_valid_total / post_prune_candidates
            if post_prune_candidates
            else 0.0
        ),
        "mean_required_unknown_fraction": (
            required_unknown_total / post_prune_candidates
            if post_prune_candidates
            else 0.0
        ),
        "mean_optional_support_fraction": (
            optional_support_total / post_prune_candidates
            if post_prune_candidates
            else 0.0
        ),
        "proof_obligation_counts": dict(proof_obligation_counts),
        "gold_verifier_generated": gold_verifier_generated,
        "gold_verifier_retained": gold_verifier_retained,
        "gold_verifier_retention": (
            gold_verifier_retained / gold_verifier_generated
            if gold_verifier_generated
            else 0.0
        ),
        "gold_false_rejection_constraints": dict(
            gold_false_rejection_constraints
        ),
        "proof_contradiction_examples": proof_contradiction_examples,
        "pre_prune_candidates": pre_prune_candidates,
        "post_prune_candidates": post_prune_candidates,
        "training_trace_sets": training_trace_sets,
        "mean_pre_prune_candidates": (
            pre_prune_candidates / training_trace_sets
            if training_trace_sets
            else 0.0
        ),
        "mean_post_prune_candidates": (
            post_prune_candidates / training_trace_sets
            if training_trace_sets
            else 0.0
        ),
        "pre_prune_gold_program_recall": (
            pre_prune_gold / training_trace_sets if training_trace_sets else 0.0
        ),
        "post_prune_gold_program_recall": (
            post_prune_gold / training_trace_sets if training_trace_sets else 0.0
        ),
        "pruning_gold_loss_count": pruning_gold_loss,
        "positive_classes": dict(positive_classes),
        "typed_candidates": typed_candidates,
        "typed_gold_candidates": typed_gold_candidates,
        "typed_synthesis_statuses": dict(synthesis_statuses),
        "candidate_failure_modes": dict(failure_modes),
        "failure_examples": failure_examples,
        "per_operation": {
            operation: dict(stats)
            for operation, stats in per_operation.items()
        },
    }
    return records, audit


def split_reranker_development(
    examples: Sequence[ProgramExample],
    validation_examples: Sequence[ProgramExample],
    *,
    calibration_fraction: float = 0.20,
    seed: int = 17,
) -> tuple[list[ProgramExample], list[ProgramExample], dict[str, object]]:
    if not 0.05 <= calibration_fraction <= 0.40:
        raise V11PipelineError("calibration_fraction must be within [0.05, 0.40]")
    validation_ids = {item.example_id for item in validation_examples}
    missing = validation_ids - {item.example_id for item in examples}
    if missing:
        raise V11PipelineError(
            f"Validation IDs missing from development dataset: {len(missing)}"
        )
    validation_groups = {
        _v11_group_key(item) for item in validation_examples
    }
    # Exclude the whole company/report group, not merely the exact validation
    # IDs. This prevents questions from the same source report leaking into the
    # sidecar train or calibration portion.
    development = [
        item
        for item in examples
        if _v11_group_key(item) not in validation_groups
    ]
    validation_group_examples_excluded = len(examples) - len(development)
    groups: dict[str, list[ProgramExample]] = {}
    for example in development:
        groups.setdefault(_v11_group_key(example), []).append(example)
    if len(groups) < 2:
        raise V11PipelineError("Need at least two report/company groups")
    target_total = max(1, round(len(development) * calibration_fraction))
    totals = Counter(item.program.operation for item in development)
    target_ops = {
        operation: totals[operation] * calibration_fraction
        for operation in OPERATIONS
    }
    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda key: len(groups[key]), reverse=True)
    selected: set[str] = set()
    selected_count = 0
    selected_ops: Counter[str] = Counter()

    def cost(count: int, operations: Counter[str]) -> float:
        size_cost = abs(count - target_total) / max(1, target_total)
        operation_cost = sum(
            abs(operations[operation] - target_ops[operation])
            / max(1.0, target_ops[operation])
            for operation in OPERATIONS
            if totals[operation]
        )
        return size_cost + 0.30 * operation_cost

    for key in keys:
        group = groups[key]
        proposed_ops = selected_ops + Counter(
            item.program.operation for item in group
        )
        proposed_count = selected_count + len(group)
        if (
            selected_count < target_total * 0.70
            or cost(proposed_count, proposed_ops) < cost(selected_count, selected_ops)
        ):
            selected.add(key)
            selected_count = proposed_count
            selected_ops = proposed_ops
    if not selected or len(selected) == len(groups):
        selected = {min(keys, key=lambda key: abs(len(groups[key]) - target_total))}
    calibration = [
        item for key in selected for item in groups[key]
    ]
    train = [
        item for key, group in groups.items() if key not in selected for item in group
    ]
    train_groups = {_v11_group_key(item) for item in train}
    calibration_groups = {_v11_group_key(item) for item in calibration}
    if train_groups & calibration_groups:
        raise V11PipelineError("Report/company leakage in calibration split")
    train_counts = Counter(item.program.operation for item in train)
    calibration_counts = Counter(item.program.operation for item in calibration)
    required_operations = {
        operation
        for operation in RERANKER_TRAIN_OPERATIONS
        if totals[operation]
    }
    missing_train = required_operations - set(train_counts)
    missing_calibration = required_operations - set(calibration_counts)
    if missing_train or missing_calibration:
        raise V11PipelineError(
            "Company/report split lacks operation coverage: "
            f"train={sorted(missing_train)}, "
            f"calibration={sorted(missing_calibration)}"
        )
    rng.shuffle(train)
    rng.shuffle(calibration)
    audit = {
        "input_examples": len(examples),
        "validation_ids_excluded": len(validation_ids),
        "validation_groups_excluded": len(validation_groups),
        "validation_group_examples_excluded": (
            validation_group_examples_excluded
        ),
        "development_examples": len(development),
        "reranker_train_examples": len(train),
        "calibration_examples": len(calibration),
        "reranker_train_groups": len(train_groups),
        "calibration_groups": len(calibration_groups),
        "group_overlap": 0,
        "train_operation_counts": dict(train_counts),
        "calibration_operation_counts": dict(calibration_counts),
    }
    return train, calibration, audit


def write_reranker_records(
    path: str | Path, records: Sequence[RerankerExampleRecord]
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "example_id": record.example_id,
                        "operation": record.operation,
                        "candidates": [
                            {
                                "features": candidate.features,
                                "program_key": candidate.program_key,
                                "exact_gold": candidate.exact_gold,
                                "execution_correct": candidate.execution_correct,
                                "positive_class": candidate.positive_class,
                                "generation_sources": candidate.generation_sources,
                                "runtime_eligible": candidate.runtime_eligible,
                                "training_state": candidate.training_state,
                                "proof_contract_state": (
                                    candidate.proof_contract_state
                                ),
                                "rejection_reasons": candidate.rejection_reasons,
                                "unknown_reasons": candidate.unknown_reasons,
                                "required_valid_fraction": (
                                    candidate.required_valid_fraction
                                ),
                                "required_unknown_fraction": (
                                    candidate.required_unknown_fraction
                                ),
                                "optional_support_fraction": (
                                    candidate.optional_support_fraction
                                ),
                                "proof_obligations": candidate.proof_obligations,
                                "supervision_only": candidate.supervision_only,
                            }
                            for candidate in record.candidates
                        ],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return destination


def read_reranker_records(path: str | Path) -> list[RerankerExampleRecord]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    records: list[RerankerExampleRecord] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                candidates = tuple(
                    RerankerCandidateRecord(
                        features=tuple(float(value) for value in item["features"]),
                        program_key=str(item["program_key"]),
                        exact_gold=bool(item["exact_gold"]),
                        execution_correct=bool(item["execution_correct"]),
                        positive_class=str(item.get("positive_class", "negative")),
                        generation_sources=tuple(item.get("generation_sources", ())),
                        runtime_eligible=bool(item.get("runtime_eligible", True)),
                        training_state=str(item.get("training_state", "VALID")),
                        proof_contract_state=str(
                            item.get("proof_contract_state", "")
                        ),
                        rejection_reasons=tuple(item.get("rejection_reasons", ())),
                        unknown_reasons=tuple(item.get("unknown_reasons", ())),
                        required_valid_fraction=float(
                            item.get("required_valid_fraction", 0.0)
                        ),
                        required_unknown_fraction=float(
                            item.get("required_unknown_fraction", 0.0)
                        ),
                        optional_support_fraction=float(
                            item.get("optional_support_fraction", 0.0)
                        ),
                        proof_obligations=tuple(
                            item.get("proof_obligations", ())
                        ),
                        supervision_only=bool(
                            item.get("supervision_only", False)
                        ),
                    )
                    for item in payload["candidates"]
                )
                records.append(
                    RerankerExampleRecord(
                        example_id=str(payload["example_id"]),
                        operation=str(payload["operation"]),
                        candidates=candidates,
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise V11PipelineError(
                    f"Invalid reranker cache row {source}:{line_number}"
                ) from exc
    return records
