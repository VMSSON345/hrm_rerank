from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import time
from typing import Callable, Sequence

from .document_hrm import (
    DocumentHRMBundle,
    DocumentHRMError,
    ProgramPrediction,
    _apply_symbolic_operation_constraint,
    _abstention_threshold_for_operation,
    _abstention_threshold_for_prediction,
    _candidate_period_years,
    _grounding_gate_reasons,
    _is_groundable_document_candidate,
    _metric_compatible_indices,
    _prediction_calibration_accepts,
    _prediction_required_proof_rejection_reasons,
    _proof_conditioned_acceptance_eligible,
    _role_compatible_pair_indices,
    _temporal_role_compatible_indices,
)
from .document_page_index import (
    DocumentPageIndexError,
    PageIndexStats,
    build_page_indexed_chunks,
)
from .document_sources import (
    DocumentSourceError,
    EvidenceChunk,
    OCRConfig,
    build_chunks,
    discover_documents,
    expand_neighbor_chunks,
    rank_chunks,
)
from .evidence_schema import build_evidence_graph
from .program_data import (
    SUM_N_OPERATION,
    NumericCandidate,
    ReasoningDAGProgram,
    ReasoningProgram,
    currency_unit_scale,
    execute_program,
    execute_program_dag,
    is_sum_operation,
    materialize_program_dag,
    merge_candidates,
)
from .ontology import normalize_text
from .financial_identities import metric_alias_score
from .question_planner import _has_financial_role_anchor, compile_question_plan
from .question_request_planner import compile_question_request
from .request_answer import VerifiedRequestAnswer, build_verified_request_answer
from .source_action_policy import (classify_source_action, AnswerDecision,
    CALIBRATED_ACTION_POLICY, EVIDENCE_ACTION_POLICY, validate_publication_policy)
from .document_hrm import _prediction_action_decision, _uses_logical_sum_evaluation
from .symbolic_verifier import (
    OperandMappingAssessment,
    VerificationState,
    verify_program,
)


def _flat_verified_program_acceptance_eligible(
    plan: object,
    prediction: object,
    verification: object,
    *,
    missing_periods: Sequence[str] = (),
    derived_formula_evidence: bool = False,
    grounding_rejections: Sequence[str] = (),
) -> bool:
    """Allow a flat computation only when the full required proof is complete."""

    program = getattr(prediction, "program", None)
    if (
        program is None
        or program.operation == "ADD"
        or missing_periods
        or derived_formula_evidence
        or grounding_rejections
    ):
        return False
    if program.operation != "SELECT" and len(program.operands) < 2:
        return False
    mapping = getattr(verification, "mapping_assessment", None)
    operand_mappings = tuple(getattr(mapping, "operand_mappings", ()))
    if (
        getattr(mapping, "resolution", "UNKNOWN") != "UNIQUE"
        or not operand_mappings
        or len(operand_mappings) != len(program.operands)
        or any(
            not getattr(item, "selected_physical_is_valid", False)
            or bool(getattr(item, "conflict_reasons", ()))
            for item in operand_mappings
        )
    ):
        return False
    required = tuple(
        item.state
        for item in getattr(verification, "proof_obligations", ())
        if item.applicability == "required"
    )
    return bool(required) and all(
        state is VerificationState.VALID for state in required
    )


def _classify_evidence_resolution(
    *,
    missing_periods: Sequence[str],
    required_states: Sequence[str],
    mapping_assessment: object | None,
) -> tuple[str, tuple[str, ...]]:
    if missing_periods:
        return "MISSING", ("missing_requested_period",)
    mapping_resolution = str(
        getattr(mapping_assessment, "resolution", "UNKNOWN")
    )
    mapping_reasons = tuple(
        str(item)
        for item in getattr(mapping_assessment, "reason_codes", ())
    )
    if mapping_resolution == "CONFLICT":
        return "CONFLICT", mapping_reasons
    if VerificationState.INVALID.value in required_states:
        return "CONFLICT", (
            "required_proof_invalid",
            *(reason for reason in mapping_reasons if reason == "question_column_role_unspecified"),
        )
    if mapping_resolution == "AMBIGUOUS":
        return "AMBIGUOUS", mapping_reasons
    if mapping_resolution == "MISSING":
        return "MISSING", mapping_reasons
    if (
        mapping_resolution == "UNIQUE"
        and required_states
        and all(state == VerificationState.VALID.value for state in required_states)
    ):
        return "UNIQUE", mapping_reasons
    return (
        "UNKNOWN",
        mapping_reasons or ("mapping_assessment_unavailable",),
    )


def _classify_source_action(
    *,
    missing_periods: Sequence[str],
    verification_state: str,
    required_states: Sequence[str],
    mapping_assessment: object | None,
    abstained: bool,
) -> str:
    """Compatibility adapter for the proof-only public action policy."""

    mapping_resolution = str(
        getattr(mapping_assessment, "resolution", "UNKNOWN")
    )
    return classify_source_action(
        missing_periods=missing_periods,
        verification_state=verification_state,
        required_states=required_states,
        mapping_resolution=mapping_resolution,
        abstained=abstained,
    )


@dataclass(frozen=True)
class SelectedEvidence:
    operand_number: int
    candidate: NumericCandidate

    def render(self) -> str:
        page = f"page {self.candidate.page}" if self.candidate.page else f"line {self.candidate.line_number}"
        context = " | ".join(
            line.strip() for line in self.candidate.context.splitlines() if line.strip()
        )
        if len(context) > 320:
            context = context[:317].rstrip() + "..."
        return (
            f"operand[{self.operand_number}]={self.candidate.value} "
            f"({self.candidate.unit or 'unitless'}) | {self.candidate.source_name}, {page}\n"
            f"  {context}"
        )


@dataclass(frozen=True)
class DocumentReasoningAnswer:
    question: str
    value: Decimal | None
    unit: str
    prediction: ProgramPrediction
    evidence: tuple[SelectedEvidence, ...]
    retrieved_chunks: tuple[EvidenceChunk, ...]
    document_count: int
    candidate_count: int
    abstained: bool
    confidence_threshold: float
    warnings: tuple[str, ...]
    program_dag: ReasoningDAGProgram | None = None
    page_index_stats: PageIndexStats | None = None
    stage_timings: tuple[tuple[str, float], ...] = ()
    action_decision: AnswerDecision | None = None
    source_action: str = "ABSTAIN"
    verification_state: str = "UNKNOWN"
    proof_obligations: tuple[tuple[str, str, str | None], ...] = ()
    verification_reasons: tuple[str, ...] = ()
    acceptance_basis: tuple[str, ...] = ()
    missing_periods: tuple[str, ...] = ()
    evidence_resolution: str = "UNKNOWN"
    evidence_resolution_reasons: tuple[str, ...] = ()
    operand_mapping_assessments: tuple[OperandMappingAssessment, ...] = ()
    applicability_details: tuple[dict, ...] = ()
    lookup_recovery: dict | None = None

    @property
    def program(self) -> ReasoningProgram:
        return self.prediction.program

    def verified_envelope(self) -> object:
        from .answer_presentation import build_verified_answer_envelope

        return build_verified_answer_envelope(self)

    def render_verified(self) -> str:
        from .answer_presentation import render_verified_answer

        return render_verified_answer(self.verified_envelope())

    def render(self) -> str:
        candidate_value = _format_value(self.value)
        suffix = f" {self.unit}" if self.unit else ""
        if self.abstained and self.value is not None:
            answer = (
                "ABSTAIN (unverified candidate: "
                f"{candidate_value}{suffix})"
            )
        else:
            answer = (
                _format_value(None if self.abstained else self.value) + suffix
            )
        lines = [
            f"Answer: {answer}",
            (
                "Status: ABSTAIN - manual evidence review required"
                if self.abstained
                else "Status: ACCEPTED"
            ),
            f"Program: {self.program.operation}{self.program.operands}",
            (
                "HRM confidence: "
                f"overall={self.prediction.confidence:.3f}, "
                f"operation={self.prediction.operation_confidence:.3f}, "
                f"segments={self.prediction.segments}"
            ),
            (
                "ACT: "
                f"halt_q={self.prediction.halt_q:.3f}, "
                f"continue_q={self.prediction.continue_q:.3f}"
            ),
            (
                "Neuro-symbolic search: "
                f"programs={getattr(self.prediction, 'program_candidates_considered', 1)}, "
                f"rejected={getattr(self.prediction, 'symbolic_program_rejections', 0)}, "
                f"proof={getattr(self.prediction, 'symbolic_proof_score', 0.0):.3f}, "
                f"unknown={getattr(self.prediction, 'symbolic_unknown_constraints', 0)}"
            ),
            (
                "Source path: MinerU OCR -> persisted Markdown -> retrieval -> "
                "numeric candidates -> H/L latent recurrence -> "
                "verified executable program"
            ),
            f"Documents read: {self.document_count}; numeric candidates considered: {self.candidate_count}",
            "Evidence:",
        ]
        if self.program_dag is not None:
            lines.insert(3, f"Program DAG: {self.program_dag.render()}")
        if self.page_index_stats is not None:
            lines.insert(7, "Page index: " + self.page_index_stats.render())
        if self.stage_timings:
            lines.insert(
                8 if self.page_index_stats is not None else 7,
                "Stage timing: "
                + ", ".join(
                    f"{stage}={seconds:.3f}s"
                    for stage, seconds in self.stage_timings
                ),
            )
        if self.prediction.grounding_constraints_applied:
            lines.insert(
                4,
                "Grounding constraints: "
                + ", ".join(self.prediction.grounding_constraints_applied)
                + " | neural operation confidence="
                + f"{self.prediction.neural_operation_confidence:.3f}",
            )
        lines.extend(f"- {item.render()}" for item in self.evidence)
        if self.retrieved_chunks:
            lines.append("Retrieved source chunks:")
            for chunk in self.retrieved_chunks[:5]:
                lines.append(
                    f"- score={chunk.score:.3f} | {chunk.citation()} | {chunk.chunk_id}"
                )
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        if self.abstained and self.value is not None:
            lines.append(
                "Candidate under review (not an accepted answer): "
                f"{candidate_value}{suffix}"
            )
        return "\n".join(lines)


from .metric_semantics import isolated_metric_labels


class DocumentHRMReasoningEngine:
    """End-to-end numerical QA over prepared MinerU Markdown evidence."""

    def __init__(self, bundle: DocumentHRMBundle, *,
                 publication_policy: str = CALIBRATED_ACTION_POLICY) -> None:
        self.publication_policy = validate_publication_policy(publication_policy)
        self.bundle = bundle

    def answer_request(
        self,
        question: str,
        document_path: str | Path,
        *,
        top_k_chunks: int = 8,
        max_candidates: int | None = None,
        chunk_lines_count: int = 28,
        overlap_lines: int = 6,
        neighbor_chunks: int = 1,
        ocr: bool = True,
        force_ocr: bool = False,
        mineru_output_dir: str | Path | None = None,
        mineru_page_offset: int = 0,
        mineru_cache_dir: str | Path = "cache/mineru",
        mineru_command: str | Path = ".venv-mineru/bin/mineru",
        mineru_api_url: str | None = None,
        mineru_api_timeout_seconds: float = 7200.0,
        mineru_backend: str = "hybrid-engine",
        mineru_effort: str = "medium",
        mineru_auto_generate: bool = True,
        use_page_index: bool = True,
        page_index_path: str | Path = "cache/document_page_index.sqlite3",
        rebuild_page_index: bool = False,
        page_top_k: int = 6,
        page_neighbor_pages: int = 1,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> VerifiedRequestAnswer:
        """Answer each explicitly separable output with an independent proof."""

        request_plan = compile_question_request(question)
        if request_plan.fail_closed:
            return build_verified_request_answer(request_plan)
        scalar_answers: list[DocumentReasoningAnswer] = []
        for output in request_plan.outputs:
            from .comparison_overview import STRATEGY
            if (request_plan.split_strategy == STRATEGY
                    and len(scalar_answers) == 2
                    and self.publication_policy == EVIDENCE_ACTION_POLICY):
                scalar_answers.append(self._comparison_difference(output, scalar_answers))
                continue
            child_progress = None
            if progress_callback is not None:
                def child_progress(
                    stage: str,
                    elapsed: float,
                    *,
                    output_id: str = output.output_id,
                ) -> None:
                    progress_callback(f"{output_id}:{stage}", elapsed)
            scalar_answers.append(
                self.answer(
                    output.question,
                    document_path,
                    top_k_chunks=top_k_chunks,
                    max_candidates=max_candidates,
                    chunk_lines_count=chunk_lines_count,
                    overlap_lines=overlap_lines,
                    neighbor_chunks=neighbor_chunks,
                    ocr=ocr,
                    force_ocr=force_ocr,
                    mineru_output_dir=mineru_output_dir,
                    mineru_page_offset=mineru_page_offset,
                    mineru_cache_dir=mineru_cache_dir,
                    mineru_command=mineru_command,
                    mineru_api_url=mineru_api_url,
                    mineru_api_timeout_seconds=mineru_api_timeout_seconds,
                    mineru_backend=mineru_backend,
                    mineru_effort=mineru_effort,
                    mineru_auto_generate=mineru_auto_generate,
                    use_page_index=use_page_index,
                    page_index_path=page_index_path,
                    rebuild_page_index=rebuild_page_index,
                    page_top_k=page_top_k,
                    page_neighbor_pages=page_neighbor_pages,
                    progress_callback=child_progress,
                )
            )
        return build_verified_request_answer(request_plan, scalar_answers)

    @isolated_metric_labels
    def answer(
        self,
        question: str,
        document_path: str | Path,
        *,
        top_k_chunks: int = 8,
        max_candidates: int | None = None,
        chunk_lines_count: int = 28,
        overlap_lines: int = 6,
        neighbor_chunks: int = 1,
        ocr: bool = True,
        force_ocr: bool = False,
        mineru_output_dir: str | Path | None = None,
        mineru_page_offset: int = 0,
        mineru_cache_dir: str | Path = "cache/mineru",
        mineru_command: str | Path = ".venv-mineru/bin/mineru",
        mineru_api_url: str | None = None,
        mineru_api_timeout_seconds: float = 7200.0,
        mineru_backend: str = "hybrid-engine",
        mineru_effort: str = "medium",
        mineru_auto_generate: bool = True,
        use_page_index: bool = True,
        page_index_path: str | Path = "cache/document_page_index.sqlite3",
        rebuild_page_index: bool = False,
        page_top_k: int = 6,
        page_neighbor_pages: int = 1,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> DocumentReasoningAnswer:
        answer_started = time.perf_counter()
        stage_timings: dict[str, float] = {}

        def record_stage(stage: str, started_at: float) -> None:
            stage_timings[stage] = stage_timings.get(stage, 0.0) + (
                time.perf_counter() - started_at
            )
            if progress_callback is not None:
                try:
                    progress_callback(stage, time.perf_counter() - answer_started)
                except Exception:
                    # Observability must never change the reasoning outcome.
                    pass

        if not question.strip():
            raise DocumentHRMError("Question cannot be empty")
        from .question_planner import resolve_public_question, public_question_assumptions
        resolved_question = resolve_public_question(question)
        default_actual_applied = "column_role:default_actual" in public_question_assumptions(question)
        question = resolved_question
        stage_started = time.perf_counter()
        documents = discover_documents(document_path)
        record_stage("document_discovery", stage_started)
        prepared_sources = all(
            item.suffix.lower() in {".md", ".txt"} for item in documents
        )
        use_page_index = use_page_index and not prepared_sources
        ocr_config = OCRConfig(
            enabled=ocr or force_ocr,
            force=force_ocr,
            mineru_output_dir=(
                Path(mineru_output_dir) if mineru_output_dir is not None else None
            ),
            mineru_page_offset=mineru_page_offset,
            mineru_cache_dir=Path(mineru_cache_dir),
            mineru_command=Path(mineru_command),
            mineru_api_url=mineru_api_url,
            mineru_api_timeout_seconds=mineru_api_timeout_seconds,
            mineru_backend=mineru_backend,
            mineru_effort=mineru_effort,
            mineru_auto_generate=mineru_auto_generate,
        )
        limit = max_candidates or self.bundle.model.config.max_candidates
        attempt_configs = (ocr_config,)
        page_index_stats: PageIndexStats | None = None
        retrieval_warnings: list[str] = []
        ranked: list[EvidenceChunk] = []
        candidates: list[NumericCandidate] = []
        prediction: ProgramPrediction
        program_dag: ReasoningDAGProgram | None = None
        value: Decimal | None
        missing_periods: tuple[str, ...] = ()
        lookup_recovery = None

        for attempt_index, attempt_config in enumerate(attempt_configs):
            attempt_name = (
                "markdown"
                if prepared_sources
                else "mineru"
            )
            try:
                stage_started = time.perf_counter()
                chunks, attempt_stats, attempt_warnings = self._retrieve_chunks(
                    documents,
                    question,
                    ocr_config=attempt_config,
                    use_page_index=use_page_index,
                    page_index_path=page_index_path,
                    rebuild_page_index=rebuild_page_index,
                    page_top_k=page_top_k,
                    page_neighbor_pages=page_neighbor_pages,
                    chunk_lines_count=chunk_lines_count,
                    overlap_lines=overlap_lines,
                    prefer_mineru_for_selected_pages=(attempt_index > 0),
                )
                record_stage(f"{attempt_name}_retrieval", stage_started)
            except DocumentSourceError as exc:
                record_stage(f"{attempt_name}_retrieval", stage_started)
                raise

            stage_started = time.perf_counter()
            retrieval_question = question
            retrieval_plan = compile_question_plan(question)
            from .unit_defaults import monetary_label
            if (retrieval_plan.operation == "SELECT" and retrieval_plan.entity_mode == "reporting"
                    and not retrieval_plan.requested_years and not retrieval_plan.requested_period_roles
                    and not retrieval_plan.column_roles and not retrieval_plan.requested_source_kind
                    and not retrieval_plan.requested_value_column and monetary_label(retrieval_plan.metric_text)):
                retrieval_question += " thông tin chung thông tin khái quát"
            ranked = rank_chunks(retrieval_question, chunks, top_k=top_k_chunks)
            ranked = expand_neighbor_chunks(
                ranked,
                chunks,
                neighbor_window=neighbor_chunks,
                max_chunks=top_k_chunks + max(0, neighbor_chunks * 4),
            )
            record_stage(f"{attempt_name}_chunk_ranking", stage_started)
            if not ranked:
                raise DocumentSourceError("No source chunk matched the question")

            stage_started = time.perf_counter()
            # Page retrieval is the coarse relevance boundary. Once a page has
            # been selected, inspect all of its chunks for numeric candidates;
            # the smaller ranked set remains the concise display/evidence view.
            # Legacy retrieval has no page-level boundary, so it keeps using
            # the ranked chunks to avoid scanning an entire document.
            candidate_chunks = chunks if use_page_index else ranked
            candidates = merge_candidates(
                question,
                candidate_chunks,
                max_candidates=limit,
                legacy_mode=(
                    self.bundle.model.config.hierarchical_input_mode == "legacy"
                ),
                financial_identity_catalog_enabled=(
                    self.bundle.metadata.get(
                        "financial_identity_catalog_enabled", False
                    )
                    is True
                ),
            )
            candidates = _augment_financial_identity_candidates(
                question,
                candidates,
                max_candidates=limit,
            )
            record_stage(f"{attempt_name}_candidate_extraction", stage_started)
            if not candidates:
                raise DocumentHRMError(
                    "No numerical values were extracted from the matched source chunks"
                )

            from .metric_semantics import set_source_metric_labels
            set_source_metric_labels(c.row_label for c in candidates)
            missing_periods = _missing_requested_periods(question, candidates)

            stage_started = time.perf_counter()
            prediction = self.bundle.predict(question, candidates)
            record_stage(f"{attempt_name}_prediction", stage_started)
            if self.publication_policy == EVIDENCE_ACTION_POLICY:
                recovery_started = time.perf_counter()
                candidates, prediction, lookup_recovery = self._recover_lookup_binding(
                    question, candidates, prediction, chunks, limit=limit,
                )
                if lookup_recovery is not None:
                    record_stage("lookup_verification_recovery", recovery_started)
                    missing_periods = _missing_requested_periods(question, candidates)
            program_dag = getattr(prediction, "program_dag", None)
            if program_dag is None:
                try:
                    program_dag = materialize_program_dag(
                        prediction.program, candidates
                    )
                except (ValueError, IndexError):
                    program_dag = None
            preexecution_rejections = _grounding_gate_reasons(
                question,
                prediction.program,
                candidates,
                financial_identity_catalog_enabled=(
                    "financial_identity_catalog"
                    in prediction.grounding_constraints_applied
                ),
                program_dag=program_dag,
            )
            preexecution_rejections = tuple(
                dict.fromkeys(
                    (
                        *preexecution_rejections,
                        *_prediction_required_proof_rejection_reasons(
                            question, prediction, candidates
                        ),
                    )
                )
            )
            # Never calculate a provisional answer when the requested period
            # is absent or any required grounding/proof gate has failed. The
            # selected candidates remain available for audit, but no arithmetic
            # result is produced for an unverified mapping.
            if missing_periods or preexecution_rejections:
                value = None
            else:
                try:
                    stage_started = time.perf_counter()
                    value = (
                        execute_program_dag(program_dag, candidates)
                        if program_dag is not None
                        else execute_program(prediction.program, candidates)
                    )
                    record_stage(f"{attempt_name}_execution", stage_started)
                except (ZeroDivisionError, InvalidOperation) as exc:
                    record_stage(f"{attempt_name}_execution", stage_started)
                    raise DocumentHRMError(
                        f"Predicted program is not executable: {exc}"
                    ) from exc

            page_index_stats = attempt_stats
            retrieval_warnings = list(attempt_warnings)
            break
        else:  # pragma: no cover - every terminal branch returns or raises
            raise DocumentSourceError("Document reasoning produced no usable evidence")

        evidence_indices = (
            program_dag.leaf_operands
            if program_dag is not None
            else prediction.program.operands
        )
        evidence = tuple(
            SelectedEvidence(position + 1, candidates[index])
            for position, index in enumerate(evidence_indices)
        )
        value, normalized_currency_unit = _normalize_mixed_currency_result(
            question,
            prediction.program,
            program_dag,
            candidates,
            evidence_indices,
            value,
        )
        unit = normalized_currency_unit or _infer_answer_unit(
            prediction.program, evidence
        )
        threshold = _abstention_threshold_for_prediction(
            self.bundle,
            prediction,
            self._confidence_threshold(prediction.program.operation),
        )
        derived_formula_evidence = _uses_derived_table_formula(evidence)
        grounding_rejections = _grounding_gate_reasons(
            question,
            prediction.program,
            candidates,
            financial_identity_catalog_enabled=(
                "financial_identity_catalog"
                in getattr(
                    prediction,
                    "grounding_constraints_applied",
                    (),
                )
            ),
            program_dag=program_dag,
        )
        grounding_rejections = tuple(
            dict.fromkeys(
                (
                    *grounding_rejections,
                    *_prediction_required_proof_rejection_reasons(
                        question, prediction, candidates
                    ),
                )
            )
        )
        proof_conditioned_acceptance = (
            _proof_conditioned_acceptance_eligible(prediction)
        )
        v14_policy_enabled = self.publication_policy == EVIDENCE_ACTION_POLICY or _uses_logical_sum_evaluation(self.bundle) or isinstance(
            self.bundle.metadata.get("abstention_policies_by_operation"),
            dict,
        )
        calibration_accepted = (
            _prediction_calibration_accepts(self.bundle, prediction)
            if v14_policy_enabled
            else (
                prediction.confidence >= threshold
                or proof_conditioned_acceptance
            )
        )
        abstained = (
            (not calibration_accepted and self.publication_policy != EVIDENCE_ACTION_POLICY)
            or bool(missing_periods)
            or derived_formula_evidence
            or bool(grounding_rejections)
        )
        warnings: list[str] = list(retrieval_warnings)
        if default_actual_applied:
            warnings.append("public_default_actual")
        if self._uses_calibration_fallback():
            warnings.append(
                "The checkpoint has no valid calibrated threshold; runtime is using "
                "the experimental fallback threshold 0.75."
            )
        if self.publication_policy == EVIDENCE_ACTION_POLICY:
            warnings.append("Experimental evidence publication policy; calibration remains diagnostic.")
        elif v14_policy_enabled and not calibration_accepted:
            warnings.append(
                "ABSTAIN: the V14 operation calibration policy is disabled, "
                "uncalibrated, or its real confidence threshold was not met."
            )
        elif prediction.confidence < threshold and not proof_conditioned_acceptance:
            warnings.append(
                "ABSTAIN: program confidence is below the validation-calibrated threshold "
                f"({prediction.confidence:.3f} < {threshold:.3f})."
            )
        elif prediction.confidence < threshold and not abstained:
            warnings.append(
                "ACCEPTED: an executable multi-step typed program satisfied every "
                "required proof obligation; the calibrated neural-confidence "
                "threshold remains unchanged."
            )
        if missing_periods:
            warnings.append(
                "ABSTAIN: retrieved evidence does not cover requested period(s): "
                + ", ".join(missing_periods)
                + "."
            )
        if derived_formula_evidence:
            warnings.append(
                "ABSTAIN: at least one operand was reconstructed from an explicit "
                "same-table formula; verify the source cells manually."
            )
        if grounding_rejections:
            if "zero_baseline_percent_change" in grounding_rejections:
                warnings.append(
                    "ABSTAIN: percent change is undefined because the baseline "
                    "period value is zero."
                )
            warnings.append(
                "ABSTAIN: grounding gate rejected the selected evidence: "
                + ", ".join(grounding_rejections)
                + "."
            )
        if prediction.operation_confidence < 0.35:
            warnings.append("The H-module was uncertain about the arithmetic operation.")
        if self.bundle.metadata.get("abstention_target_met") is False:
            warnings.append(
                "The checkpoint did not reach its requested calibration precision; "
                "treat ACCEPTED as experimental rather than production-safe."
            )
        operand_units = {
            item.candidate.unit for item in evidence if item.candidate.unit
        }
        convertible_currency_scales = bool(
            operand_units
            and all(
                currency_unit_scale(unit) is not None
                for unit in operand_units
            )
        )
        if (
            (
                is_sum_operation(prediction.program.operation)
                or prediction.program.operation in {"SUBTRACT", "AVERAGE"}
            )
            and len(operand_units) > 1
            and not convertible_currency_scales
        ):
            warnings.append("Selected operands have incompatible units.")
        if len({item.candidate.source_name for item in evidence}) > 1:
            warnings.append("Operands came from more than one source document.")
        evidence_methods = {
            item.candidate.extraction_method.lower() for item in evidence
        }
        if any("mineru" in method for method in evidence_methods):
            warnings.append(
                "MinerU OCR-derived evidence was used. Verify selected digits "
                "against the source scan for high-stakes use."
            )
        elif any("ocr" in method for method in evidence_methods):
            warnings.append(
                "OCR-derived evidence was used. Verify every selected digit against the scan."
            )
        verification_plan = compile_question_plan(question)
        verification = verify_program(
            verification_plan,
            prediction.program,
            build_evidence_graph(candidates),
            financial_identity_catalog_enabled=(
                "financial_identity_catalog"
                in getattr(prediction, "grounding_constraints_applied", ())
            ),
            program_dag=program_dag,
            slot_bindings=getattr(prediction, "slot_bindings", ()),
        )
        proof_obligations = tuple(
            (
                obligation.name,
                obligation.applicability,
                obligation.state.value if obligation.state is not None else None,
            )
            for obligation in verification.proof_obligations
        )
        required_states = tuple(
            state
            for _, applicability, state in proof_obligations
            if applicability == "required"
        )
        mapping_assessment = verification.mapping_assessment
        flat_proof_acceptance = _flat_verified_program_acceptance_eligible(
            verification_plan,
            prediction,
            verification,
            missing_periods=missing_periods,
            derived_formula_evidence=derived_formula_evidence,
            grounding_rejections=grounding_rejections,
        )
        if abstained and flat_proof_acceptance and not v14_policy_enabled:
            abstained = False
            warnings = [
                warning
                for warning in warnings
                if not warning.startswith(
                    "ABSTAIN: program confidence is below"
                )
            ]
            warnings.append(
                "ACCEPTED: the selected flat computation uniquely satisfied "
                "every required metric/role/period/provenance proof; the "
                "calibrated neural-confidence threshold remains unchanged."
            )
        # Absence of a requested period remains an evidence-coverage gap, not
        # contradictory evidence.  Otherwise an answer is public only after
        # the selected operands have one unique physical mapping each.
        source_action = _classify_source_action(
            missing_periods=missing_periods,
            verification_state=verification.state.value,
            required_states=required_states,
            mapping_assessment=mapping_assessment,
            abstained=abstained,
        )
        action_decision = None
        if v14_policy_enabled:
            action_decision = _prediction_action_decision(
                self.bundle, question, prediction, candidates, publication_policy=self.publication_policy)
            if not action_decision.policy_accepted or derived_formula_evidence:
                source_action = "REJECT" if source_action == "REJECT" else "ABSTAIN"
                abstained = True
                value = None
        if source_action != "ANSWER":
            abstained = True
        evidence_resolution, evidence_resolution_reasons = (
            _classify_evidence_resolution(
                missing_periods=missing_periods,
                required_states=required_states,
                mapping_assessment=mapping_assessment,
            )
        )
        if "question_column_role_unspecified" in evidence_resolution_reasons:
            warnings.append(
                "Cần làm rõ: báo cáo có cả số kế hoạch và thực hiện cho chỉ tiêu/kỳ này. "
                "Bạn muốn số thực hiện hay số kế hoạch?"
            )
        acceptance_basis = (
            "runtime_action_preserved",
            "full_candidate_evidence_graph_verification",
            *(
                ("flat_unique_slot_proof",)
                if flat_proof_acceptance
                else ()
            ),
            (
                "required_proof_valid"
                if source_action == "ANSWER"
                else "fail_closed"
            ),
        )
        if action_decision is not None:
            acceptance_basis = action_decision.acceptance_basis or ("policy_not_accepted",)
        stage_timings["total"] = time.perf_counter() - answer_started
        if progress_callback is not None:
            try:
                progress_callback("complete", stage_timings["total"])
            except Exception:
                pass
        return DocumentReasoningAnswer(
            question=question,
            value=value,
            unit=unit,
            prediction=prediction,
            evidence=evidence,
            retrieved_chunks=tuple(ranked),
            document_count=len(documents),
            candidate_count=len(candidates),
            abstained=abstained,
            confidence_threshold=threshold,
            warnings=tuple(warnings),
            program_dag=program_dag,
            page_index_stats=page_index_stats,
            stage_timings=tuple(stage_timings.items()),
            action_decision=action_decision,
            source_action=source_action,
            # Public answer state follows the required proof contract. Optional
            # layout diagnostics may remain UNKNOWN without contradicting an
            # ANSWER whose every required obligation is VALID.
            verification_state=(
                VerificationState.UNKNOWN.value
                if missing_periods
                else verification.proof_contract_state.value
            ),
            proof_obligations=proof_obligations,
            applicability_details=verification.applicability_details,
            lookup_recovery=lookup_recovery,
            verification_reasons=tuple(
                dict.fromkeys((*verification.reasons, *verification.unknown_reasons))
            ),
            acceptance_basis=acceptance_basis,
            missing_periods=tuple(missing_periods),
            evidence_resolution=evidence_resolution,
            evidence_resolution_reasons=tuple(evidence_resolution_reasons),
            operand_mapping_assessments=(
                mapping_assessment.operand_mappings
                if mapping_assessment is not None
                else ()
            ),
        )

    def _recover_relative_binding(self, question, candidates, prediction, plan):
        """Rebind a failed relative operation within its existing evidence pool.

        Three full-proof checks at most; no extra model call, values, retrieval
        filtering or relaxed verification. Conflicting source facts stay present.
        """
        from itertools import product
        from .symbolic_verifier import _slot_candidate_state, _source_provenance_state
        from .physical_coordinates import physical_coordinate_key
        graph=build_evidence_graph(candidates)
        trace={"budget":{"binding_proposals":3,"context_recoveries":0,"additional_model_calls":0},
               "attempts":[],"recovered":False,"context_recoveries":0,"additional_model_calls":0}
        def check(proposal,stage):
            result=verify_program(plan,proposal.program,graph,
                program_dag=proposal.program_dag,slot_bindings=proposal.slot_bindings)
            mapping=result.mapping_assessment
            reasons=tuple(dict.fromkeys((*_grounding_gate_reasons(question,proposal.program,candidates,program_dag=proposal.program_dag),
                *_prediction_required_proof_rejection_reasons(question,proposal,candidates))))
            valid=result.proof_contract_state is VerificationState.VALID and mapping is not None and mapping.resolution=="UNIQUE" and not reasons
            trace['attempts'].append({'stage':stage,'operands':list(proposal.program.operands),'eligible':valid,
                'required_state':result.proof_contract_state.value,'resolution':mapping.resolution if mapping else 'UNKNOWN',
                'reasons':list(dict.fromkeys((*reasons,*(mapping.reason_codes if mapping else ())))),'conflict_pool_size':len(candidates)})
            return valid
        initial_valid=check(prediction,'initial')
        from .disclosure_equivalence import _quantum
        from .program_data import currency_unit_scale
        def precision(c):
            scale=currency_unit_scale(c.unit)
            return (_quantum(graph.cells[c.index],scale) if scale is not None else None) or Decimal('Infinity')
        pools=[]
        for slot in plan.operand_slots:
            eligible=[c for c in candidates if _slot_candidate_state(plan,slot,graph.cells[c.index]) is VerificationState.VALID
                      and _source_provenance_state((graph.cells[c.index],)) is VerificationState.VALID]
            eligible.sort(key=lambda c:(precision(c),-c.relevance,repr(physical_coordinate_key(c)),c.index))
            seen=set();unique=[]
            for c in eligible:
                key=physical_coordinate_key(c)
                if key not in seen:seen.add(key);unique.append(c)
            pools.append(unique)
        if initial_valid and all(not pool or precision(candidates[index])<=precision(pool[0]) for index,pool in zip(prediction.program.operands,pools)):
            return candidates,prediction,trace
        # When the full proof established equivalent disclosures, prefer the
        # finest source precision; never substitute numbers into a source cell.
        pairs=sorted(((a,b) for a,b in product(*pools) if physical_coordinate_key(a)!=physical_coordinate_key(b)),
                     key=lambda p:(precision(p[0]),precision(p[1]),-(p[0].relevance+p[1].relevance),p[0].index,p[1].index))
        diagnostic=None
        for pair in pairs:
            operands=tuple(c.index for c in pair)
            if operands==prediction.program.operands:continue
            program=ReasoningProgram(plan.operation,operands)
            proposal=replace(prediction,program=program,program_dag=materialize_program_dag(program,candidates),
                slot_bindings=tuple((slot.id,c.index) for slot,c in zip(plan.operand_slots,pair)),
                confidence=0.0,pointer_confidences=(0.0,0.0),
                grounding_constraints_applied=tuple(dict.fromkeys((*prediction.grounding_constraints_applied,'bounded_relative_binding_recovery'))))
            if check(proposal,'existing_pool'):
                trace['recovered']=True
                return candidates,proposal,trace
            diagnostic=diagnostic or proposal
            if len(trace['attempts'])>=3:break
        # Explain a proven conflicting fact instead of a discarded wrong-role
        # model choice. The ordinary proof gate still withholds the value.
        return candidates,prediction if initial_valid else diagnostic or prediction,trace

    def _recover_lookup_binding(self, question, candidates, prediction, chunks, *, limit):
        """At most three deep binding checks; the HRM beam stays unchanged.

        Reuse the original inference. One local context reconstruction is
        allowed, and all semantically relevant retrieved evidence is retained
        for conflict checks. No reference values or conversation cache enter
        the ranking or verifier.
        """
        from .physical_coordinates import physical_coordinate_key
        from .report_context import validated_report_field
        from .symbolic_verifier import _slot_candidate_state, _source_provenance_state, _role_agnostic_mapping_slots
        from .unit_defaults import monetary_label
        plan = compile_question_plan(question)
        if plan.operation in {"PERCENT_CHANGE", "RATIO_PERCENT", "DIVIDE"} and len(plan.operand_slots)==2:
            return self._recover_relative_binding(question, candidates, prediction, plan)
        slots = plan.operand_slots or _role_agnostic_mapping_slots(plan, ReasoningProgram("SELECT", (0,)))
        if (plan.operation != "SELECT" or len(slots) != 1
                or not monetary_label(slots[0].metric)):
            return candidates, prediction, None
        started = time.perf_counter()
        slot = replace(slots[0], column_role="") if slots[0].column_role == "__role_agnostic__" else slots[0]
        trace = {"budget": {"binding_proposals": 3, "context_recoveries": 1,
                            "additional_model_calls": 1},
                 "attempts": [], "context_recoveries": 0, "additional_model_calls": 0,
                 "recovered": False, "kept_pool": [], "discarded_pool": []}
        original_candidates, original_prediction = candidates, prediction
        seen = set()
        ranks = {row.program.operands[0]: row.neural_rank
                 for row in prediction.program_search_trace
                 if row.program.operation == "SELECT" and len(row.program.operands) == 1}
        def key(c):
            return (physical_coordinate_key(c), c.row_label, c.column_label,
                    c.period, c.unit, c.entity_id)
        original_rank = {key(c): ranks.get(c.index, 100000 + c.index) for c in candidates}
        def sort_key(c, graph):
            state = _slot_candidate_state(plan, slot, graph.cells[c.index])
            complete = _source_provenance_state((graph.cells[c.index],))
            field = bool(validated_report_field(c)) and not (slot.year or slot.period_role or slot.column_role)
            return ({VerificationState.VALID: 0, VerificationState.UNKNOWN: 1, VerificationState.INVALID: 2}[state],
                    0 if field else 1, 0 if complete is VerificationState.VALID else 1,
                    original_rank.get(key(c), 200000), repr(physical_coordinate_key(c)))
        def signature(c):
            return {"source_name": c.source_name, "page": c.page,
                    "line_number": c.line_number, "source_span": c.source_span,
                    "source_line_sha256": c.source_line_sha256,
                    "row_label": c.row_label, "column_label": c.column_label}
        def check(pool, proposal, stage):
            if len(trace["attempts"]) >= 3:
                return False, proposal
            if len(proposal.program.operands) != 1:
                trace["attempts"].append({"ordinal":len(trace["attempts"])+1,
                    "stage":stage,"binding":{"operands":[signature(pool[i]) for i in proposal.program.operands if 0 <= i < len(pool)]},
                    "eligible":False,"required_state":"INVALID","resolution":"UNKNOWN",
                    "reasons":["lookup_requires_one_operand"],"applicability":[],"conflict_pool_size":len(pool)})
                return False, proposal
            c = pool[proposal.program.operands[0]]
            if key(c) in seen:
                return False, proposal
            seen.add(key(c))
            graph = build_evidence_graph(pool)
            verification = verify_program(plan, proposal.program, graph,
                program_dag=proposal.program_dag, slot_bindings=proposal.slot_bindings)
            mapping = verification.mapping_assessment
            reasons = tuple(dict.fromkeys((*_grounding_gate_reasons(question, proposal.program, pool,
                program_dag=proposal.program_dag),
                *_prediction_required_proof_rejection_reasons(question, proposal, pool))))
            eligible = (verification.proof_contract_state is VerificationState.VALID
                        and mapping is not None and mapping.resolution == "UNIQUE" and not reasons)
            trace["attempts"].append({"ordinal": len(trace["attempts"]) + 1,
                "stage": stage, "binding": signature(c), "eligible": eligible,
                "required_state": verification.proof_contract_state.value,
                "resolution": mapping.resolution if mapping else "UNKNOWN",
                "reasons": list(dict.fromkeys((*reasons, *(mapping.reason_codes if mapping else ())))),
                "applicability": verification.applicability_details,
                "conflict_pool_size": len(pool)})
            if len(trace["attempts"])>1:
                previous={x["name"]:x.get("state") for x in trace["attempts"][0]["applicability"]}
                trace["attempts"][-1]["repaired_obligations"]=[
                    {"name":x["name"],"before":previous.get(x["name"]),"after":x.get("state"),"witness_refs":x.get("witness_refs",[])}
                    for x in verification.applicability_details
                    if x.get("state")=="VALID" and previous.get(x["name"])!="VALID"]
            if eligible:
                proposal = replace(proposal, symbolic_proof_score=verification.proof_score,
                    symbolic_unknown_constraints=sum(o.state is VerificationState.UNKNOWN
                        for o in verification.proof_obligations if o.applicability == "required"))
            return eligible, proposal
        def proposal_for(c, pool):
            program = ReasoningProgram("SELECT", (c.index,))
            return replace(original_prediction, program=program,
                program_dag=materialize_program_dag(program, pool),
                slot_bindings=((slot.id, c.index),) if plan.operand_slots else (),
                grounding_constraints_applied=tuple(dict.fromkeys((
                    *original_prediction.grounding_constraints_applied, "bounded_lookup_recovery_v2"))))
        def finish(pool, proposal, recovered):
            trace["recovered"] = recovered
            trace["binding_proposals"] = len(trace["attempts"])
            trace["seconds"] = time.perf_counter() - started
            return pool, proposal, trace
        accepted, selected = check(candidates, prediction, "initial")
        if accepted:
            return finish(candidates, prediction, False)
        graph = build_evidence_graph(candidates)
        alternatives = [c for c in sorted(candidates, key=lambda c: sort_key(c, graph))
                        if key(c) not in seen
                        and _slot_candidate_state(plan, slot, graph.cells[c.index]) is not VerificationState.INVALID]
        if alternatives:
            candidate = alternatives[0]
            accepted, selected = check(candidates, proposal_for(candidate, candidates), "existing_pool")
            if accepted:
                return finish(candidates, selected, True)
        # Recover only around a binding already examined; this never scans a
        # different file or jumps to a page selected using a reference answer.
        anchor = alternatives[0] if alternatives else candidates[prediction.program.operands[0] if prediction.program.operands else 0]
        same_source = sorted((c for c in chunks if c.source_name == anchor.source_name),
                             key=lambda c: (c.line_start, c.line_end, c.chunk_id))
        containing = [i for i,c in enumerate(same_source)
                      if c.line_start <= anchor.line_number <= c.line_end]
        if containing and len(trace["attempts"]) < 3:
            center = min(containing, key=lambda i: (same_source[i].line_end-same_source[i].line_start,
                                                   same_source[i].chunk_id))
            region = same_source[max(0, center-1):center+2]
            trace["context_recoveries"] = 1
            trace["recovery_chunks"] = [{"source_name": c.source_name, "chunk_id": c.chunk_id,
                                          "line_start": c.line_start, "line_end": c.line_end} for c in region]
            fresh = merge_candidates(question, region, max_candidates=limit,
                legacy_mode=self.bundle.model.config.hierarchical_input_mode == "legacy",
                financial_identity_catalog_enabled=self.bundle.metadata.get("financial_identity_catalog_enabled", False) is True)
            merged = {key(c): c for c in candidates}
            for c in fresh:
                previous = merged.get(key(c))
                if previous is not None and previous.value != c.value:
                    from .number_formats import resolution_state
                    if resolution_state(previous) == resolution_state(c) == "VALID":
                        trace["recovery_blocked"] = "conflicting_reconstructed_values"
                        return finish(original_candidates, original_prediction, False)
                merged[key(c)] = c
            union = [replace(c, index=i) for i,c in enumerate(merged.values())]
            union_graph = build_evidence_graph(union)
            ordered = sorted(union, key=lambda c: sort_key(c, union_graph))
            kept, discarded = ordered[:limit], ordered[limit:]
            trace["kept_pool"] = [signature(c) for c in kept]
            trace["discarded_pool"] = [signature(c) for c in discarded]
            # Refuse rather than lose an unresolved competing binding at the
            # input cap. Every relevant retrieved record must remain visible.
            if any(_slot_candidate_state(plan, slot, union_graph.cells[c.index]) is not VerificationState.INVALID for c in discarded):
                trace["recovery_blocked"] = "relevant_evidence_would_exceed_input_limit"
                return finish(original_candidates, original_prediction, False)
            candidates = [replace(c, index=i) for i,c in enumerate(kept)]
            graph = build_evidence_graph(candidates)
            alternatives = [c for c in sorted(candidates, key=lambda c: sort_key(c, graph))
                            if key(c) not in seen and _slot_candidate_state(plan, slot, graph.cells[c.index]) is not VerificationState.INVALID]
        if alternatives and len(trace["attempts"]) < 3:
            candidate = next((c for c in alternatives if key(c) not in seen), None)
            if candidate is not None:
                accepted, selected = check(candidates, proposal_for(candidate, candidates), "bounded_context" if trace["context_recoveries"] else "existing_pool")
                if accepted:
                    return finish(candidates, selected, True)
        return finish(original_candidates, original_prediction, False)

    def _retrieve_chunks(
        self,
        documents: Sequence[Path],
        question: str,
        *,
        ocr_config: OCRConfig,
        use_page_index: bool,
        page_index_path: str | Path,
        rebuild_page_index: bool,
        page_top_k: int,
        page_neighbor_pages: int,
        chunk_lines_count: int,
        overlap_lines: int,
        prefer_mineru_for_selected_pages: bool = False,
    ) -> tuple[list[EvidenceChunk], PageIndexStats | None, list[str]]:
        warnings: list[str] = []
        retrieval_question = _expanded_retrieval_question(question)
        if use_page_index:
            try:
                indexed = build_page_indexed_chunks(
                    documents,
                    retrieval_question,
                    index_path=page_index_path,
                    ocr_config=ocr_config,
                    top_k_pages=page_top_k,
                    neighbor_pages=page_neighbor_pages,
                    chunk_lines_count=chunk_lines_count,
                    overlap_lines=overlap_lines,
                    rebuild=rebuild_page_index,
                    prefer_mineru_for_selected_pages=(
                        prefer_mineru_for_selected_pages
                    ),
                )
                warnings.extend(indexed.warnings)
                return list(indexed.chunks), indexed.stats, warnings
            except DocumentPageIndexError as exc:
                warnings.append(
                    f"Page-level index unavailable; used legacy full-document path: {exc}"
                )
        chunks = build_chunks(
            documents,
            chunk_lines_count=chunk_lines_count,
            overlap_lines=overlap_lines,
            ocr_config=ocr_config,
        )
        return chunks, None, warnings

    def answer_context(
        self,
        question: str,
        context: str,
        *,
        source_name: str = "context.txt",
    ) -> DocumentReasoningAnswer:
        from .document_sources import EvidenceChunk

        chunk = EvidenceChunk(
            chunk_id="context:1",
            source_path=source_name,
            source_name=source_name,
            page_start=None,
            page_end=None,
            line_start=1,
            line_end=max(1, len(context.splitlines())),
            text=context,
            score=1.0,
        )
        candidates = merge_candidates(
            question,
            [chunk],
            max_candidates=self.bundle.model.config.max_candidates,
            legacy_mode=self.bundle.model.config.hierarchical_input_mode == "legacy",
            financial_identity_catalog_enabled=(
                self.bundle.metadata.get(
                    "financial_identity_catalog_enabled", False
                )
                is True
            ),
        )
        candidates = _augment_financial_identity_candidates(
            question,
            candidates,
            max_candidates=self.bundle.model.config.max_candidates,
        )
        if not candidates:
            raise DocumentHRMError("No numerical values in context")
        prediction = self.bundle.predict(question, candidates)
        return self._answer_candidates(question, candidates, prediction, (chunk,))

    def _comparison_difference(self, output, answers):
        """Reuse two certified SELECT inputs; never ask HRM to select them again."""
        from .comparison_overview import verified_comparison_inputs
        from .source_action_policy import decide_answer

        candidates, blocker = verified_comparison_inputs(answers)
        # The validated overview compiler puts the base first and the target
        # second (earlier/later or plan/actual). It always requests target-base.
        prediction = ProgramPrediction(
            program=ReasoningProgram("SUBTRACT", (1, 0)),
            operation_confidence=0.0, pointer_confidences=(0.0, 0.0),
            confidence=0.0, segments=0, halt_q=0.0, continue_q=0.0,
            operation_probabilities=(), neural_operation_confidence=0.0,
            neural_pointer_confidences=(), confidence_schema="verified-comparison-inputs-v1",
            grounding_constraints_applied=("verified_comparison_inputs",),
            slot_bindings=tuple((s.id, i) for s, i in zip(output.plan.operand_slots, (1, 0))),
        )
        started = time.perf_counter()
        result = self._answer_candidates(output.question, candidates, prediction,
                                         tuple(chunk for a in answers for chunk in a.retrieved_chunks))
        if blocker:
            # A fresh proof over two cells cannot undo a failed full-pool proof
            # of either parent, or authorize a comparison of different scopes.
            blocked_action = "REJECT" if result.source_action == "REJECT" else "ABSTAIN"
            result = replace(result, value=None, abstained=True, source_action=blocked_action,
                evidence_resolution="CONFLICT" if blocked_action == "REJECT" else
                    "MISSING" if any(a.evidence_resolution == "MISSING" for a in answers) else "UNKNOWN",
                evidence_resolution_reasons=(blocker,), acceptance_basis=(),
                action_decision=decide_answer(blocked_action, None, 0.0,
                    policy_id=self.publication_policy, reason_codes=(blocker,)))
        return replace(result, stage_timings=(("verified_input_difference", time.perf_counter()-started),))

    def _answer_candidates(self, question, candidates, prediction, retrieved_chunks):
        """Shared verification/execution path for context and certified inputs."""
        program_dag = getattr(prediction, "program_dag", None)
        if program_dag is None:
            try:
                program_dag = materialize_program_dag(
                    prediction.program, candidates
                )
            except (ValueError, IndexError):
                program_dag = None
        grounding_rejections = _grounding_gate_reasons(
            question,
            prediction.program,
            candidates,
            financial_identity_catalog_enabled=(
                "financial_identity_catalog"
                in prediction.grounding_constraints_applied
            ),
            program_dag=program_dag,
        )
        grounding_rejections = tuple(
            dict.fromkeys(
                (
                    *grounding_rejections,
                    *_prediction_required_proof_rejection_reasons(
                        question, prediction, candidates
                    ),
                )
            )
        )
        value = (
            None
            if grounding_rejections
            else (
                execute_program_dag(program_dag, candidates)
                if program_dag is not None
                else execute_program(prediction.program, candidates)
            )
        )
        evidence_indices = (
            program_dag.leaf_operands
            if program_dag is not None
            else prediction.program.operands
        )
        evidence = tuple(
            SelectedEvidence(position + 1, candidates[index])
            for position, index in enumerate(evidence_indices)
        )
        value, normalized_currency_unit = _normalize_mixed_currency_result(
            question,
            prediction.program,
            program_dag,
            candidates,
            evidence_indices,
            value,
        )
        threshold = _abstention_threshold_for_prediction(
            self.bundle,
            prediction,
            self._confidence_threshold(prediction.program.operation),
        )
        derived_formula_evidence = _uses_derived_table_formula(evidence)
        proof_conditioned_acceptance = (
            _proof_conditioned_acceptance_eligible(prediction)
        )
        abstained = (
            (
                prediction.confidence < threshold
                and not proof_conditioned_acceptance
            )
            or derived_formula_evidence
            or bool(grounding_rejections)
        )
        warning_items: list[str] = []
        if self._uses_calibration_fallback():
            warning_items.append(
                "The checkpoint has no valid calibrated threshold; runtime is using "
                "the experimental fallback threshold 0.75."
            )
        if self.publication_policy == EVIDENCE_ACTION_POLICY:
            warning_items.append("Experimental evidence publication policy; calibration remains diagnostic.")
        elif prediction.confidence < threshold and not proof_conditioned_acceptance:
            warning_items.append(
                "ABSTAIN: program confidence is below the validation-calibrated threshold "
                f"({prediction.confidence:.3f} < {threshold:.3f})."
            )
        elif prediction.confidence < threshold and not abstained:
            warning_items.append(
                "ACCEPTED: an executable multi-step typed program satisfied every "
                "required proof obligation; the calibrated neural-confidence "
                "threshold remains unchanged."
            )
        if derived_formula_evidence:
            warning_items.append(
                "ABSTAIN: at least one operand was reconstructed from an explicit "
                "same-table formula; verify the source cells manually."
            )
        if grounding_rejections:
            warning_items.append(
                "ABSTAIN: grounding gate rejected the selected evidence: "
                + ", ".join(grounding_rejections)
                + "."
            )
        action_fields = {}
        if self.publication_policy == EVIDENCE_ACTION_POLICY or _uses_logical_sum_evaluation(self.bundle) or isinstance(
                self.bundle.metadata.get("abstention_policies_by_operation"), dict):
            verification = verify_program(
                compile_question_plan(question), prediction.program,
                build_evidence_graph(candidates),
                financial_identity_catalog_enabled=(
                    "financial_identity_catalog" in prediction.grounding_constraints_applied),
                program_dag=prediction.program_dag, slot_bindings=prediction.slot_bindings)
            decision = _prediction_action_decision(self.bundle, question, prediction, candidates, publication_policy=self.publication_policy)
            missing = _missing_requested_periods(question, candidates)
            abstained = not decision.policy_accepted or derived_formula_evidence or bool(grounding_rejections)
            if abstained:
                value = None
            mapping = verification.mapping_assessment
            action_fields = dict(
                action_decision=decision,
                source_action="ANSWER" if not abstained else (
                    "REJECT" if decision.final_action == "REJECT" else "ABSTAIN"),
                verification_state=VerificationState.UNKNOWN.value if missing else verification.proof_contract_state.value,
                proof_obligations=tuple((p.name,p.applicability,p.state.value if p.state else None)
                                        for p in verification.proof_obligations),
                verification_reasons=verification.reasons,
                missing_periods=missing,
                evidence_resolution=mapping.resolution if mapping else "UNKNOWN",
                evidence_resolution_reasons=mapping.reason_codes if mapping else ("mapping_assessment_unavailable",),
                acceptance_basis=decision.acceptance_basis,
            )
            warning_items.extend(decision.reason_codes)
        warnings = tuple(warning_items)
        return DocumentReasoningAnswer(
            **action_fields,
            question=question,
            value=value,
            unit=normalized_currency_unit
            or _infer_answer_unit(prediction.program, evidence),
            prediction=prediction,
            evidence=evidence,
            retrieved_chunks=tuple(retrieved_chunks),
            document_count=1,
            candidate_count=len(candidates),
            abstained=abstained,
            confidence_threshold=threshold,
            warnings=warnings,
            program_dag=program_dag,
        )

    def _uses_calibration_fallback(self) -> bool:
        if self.publication_policy == EVIDENCE_ACTION_POLICY:
            return False
        if isinstance(
            self.bundle.metadata.get("abstention_policies_by_operation"),
            dict,
        ):
            return False
        raw = self.bundle.metadata.get("abstention_threshold", 0.75)
        try:
            threshold = float(raw)
        except (TypeError, ValueError):
            return True
        return not 0.0 <= threshold <= 1.0

    def _confidence_threshold(self, operation: str | None = None) -> float:
        raw = self.bundle.metadata.get("abstention_threshold", 0.75)
        try:
            threshold = float(raw)
        except (TypeError, ValueError):
            return 0.75
        # Legacy checkpoints used 1.000001 to mean "no operating point",
        # which made every answer abstain forever. Use a visible experimental
        # fallback while retaining the warning that calibration failed.
        if not 0.0 <= threshold <= 1.0:
            return 0.75
        if operation is None:
            return threshold
        return _abstention_threshold_for_operation(
            self.bundle, operation, threshold
        )


def _metric_coverage_indices(
    question: str,
    candidates: Sequence[NumericCandidate],
) -> set[int]:
    pool = set(range(len(candidates)))
    compatible = _metric_compatible_indices(question, candidates, pool)
    if compatible:
        return compatible
    # Synthetic/plaintext fixtures may carry a structured ROW without an
    # extraction label. Reuse the same metric matcher for coverage diagnosis;
    # this does not admit or select a runtime operand.
    labeled = tuple(
        candidate
        if candidate.extraction_method
        else replace(candidate, extraction_method="native")
        for candidate in candidates
    )
    return _metric_compatible_indices(question, labeled, pool)


def _missing_requested_periods(
    question: str, candidates: Sequence[NumericCandidate]
) -> tuple[str, ...]:
    from .report_context import validated_report_field
    report_year = compile_question_plan(question).requested_report_year
    if report_year:
        compatible = _metric_coverage_indices(question, candidates)
        return () if any((field := validated_report_field(candidates[i])) and field["report_year"] == report_year for i in compatible) else (report_year,)
    requested = set(re.findall(r"\b(?:19|20)\d{2}\b", question))
    if not requested:
        return ()
    period_candidates: Sequence[NumericCandidate] = candidates
    plan = compile_question_plan(question)
    planned_operation = plan.operation or _apply_symbolic_operation_constraint(
        question, "SELECT"
    )
    if planned_operation == "SELECT":
        # A phone number, filing date or other metadata can mention the
        # requested year without proving that the requested financial metric
        # is disclosed for that year. Direct-fact coverage is therefore
        # measured only over candidates compatible with the named metric.
        compatible = _metric_coverage_indices(question, candidates)
        if not compatible:
            return tuple(sorted(requested))
        period_candidates = tuple(candidates[index] for index in compatible)
    covered: set[str] = set()
    for candidate in period_candidates:
        covered.update(_candidate_period_years(candidate, question))
    return tuple(sorted(requested - covered))


def _uses_derived_table_formula(evidence: Sequence[SelectedEvidence]) -> bool:
    return any(
        "VALUE_ORIGIN: DERIVED_FROM_TABLE_FORMULA" in item.candidate.context
        for item in evidence
    )


def _expanded_retrieval_question(question: str) -> str:
    plan = compile_question_plan(question)
    if plan.interest_bearing_debt_share:
        return (
            question
            + " vay và nợ thuê tài chính ngắn hạn"
            + " vay và nợ thuê tài chính dài hạn"
            + " tổng nguồn vốn nợ vay chịu lãi"
        )
    normalized = normalize_text(question)
    if (
        "gia tri theo menh gia" in normalized
        and any(term in normalized for term in ("co phieu", "co phan"))
    ):
        return (
            question
            + " vốn điều lệ số lượng cổ phần đã phát hành "
            + "số lượng cổ phần đang lưu hành mệnh giá"
        )
    return question


def _candidate_metric_label(candidate: NumericCandidate) -> str:
    if candidate.row_label:
        return candidate.row_label
    match = re.search(
        r"(?im)(?:^|\n)ROW:\s*(.*?)(?=\n[A-Z_]+:|$)",
        candidate.context,
    )
    return match.group(1).strip() if match else ""


_DEBT_SHARE_ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "short_term_borrowings": (
        "vay và nợ thuê tài chính ngắn hạn",
        "nợ vay ngắn hạn",
        "vay ngắn hạn",
    ),
    "long_term_borrowings": (
        "vay và nợ thuê tài chính dài hạn",
        "nợ vay dài hạn",
        "vay dài hạn",
    ),
    "total_capital": (
        "tổng nguồn vốn",
        "tổng cộng nguồn vốn",
    ),
}


def _debt_share_role_anchor(label: str, role: str) -> bool:
    """Require the defining accounting concept before accepting fuzzy aliases.

    Endpoint words such as ``ngắn hạn`` and ``dài hạn`` are not sufficient to
    turn a broad liability row into an interest-bearing borrowing. This check
    intentionally uses only semantic families; it has no issuer/year/layout
    dependency and remains tolerant of the common ``nợ vay`` word order.
    """

    normalized = normalize_text(label)
    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    if role in {"short_term_borrowings", "long_term_borrowings"}:
        if "vay" not in tokens:
            return False
        # Receivable/lending rows contain ``cho vay`` but are assets, not the
        # reporting entity's interest-bearing borrowings.
        if re.search(r"\b(?:phai thu|cho)\s+(?:ve\s+)?vay\b", normalized):
            return False
        endpoint = "ngan" if role == "short_term_borrowings" else "dai"
        opposite = "dai" if endpoint == "ngan" else "ngan"
        return endpoint in tokens and opposite not in tokens
    if role == "total_capital":
        return (
            "tong" in tokens
            and "nguon" in tokens
            and "von" in tokens
            and "tai" not in tokens
            and "san" not in tokens
        )
    return False


def _debt_share_role(candidate: NumericCandidate) -> str | None:
    label = _candidate_metric_label(candidate)
    if not label or candidate.is_percent or candidate.is_year_like:
        return None
    scored = [
        (metric_alias_score(label, aliases), role)
        for role, aliases in _DEBT_SHARE_ROLE_ALIASES.items()
        if _debt_share_role_anchor(label, role)
    ]
    score, role = max(scored, default=(0.0, ""))
    return role if score >= 0.82 else None


def _derived_lineage_closure(
    candidates: Sequence[NumericCandidate], protected: set[int]
) -> set[int]:
    by_index = {candidate.index: candidate for candidate in candidates}
    pending = list(protected)
    while pending:
        index = pending.pop()
        candidate = by_index.get(index)
        if candidate is None:
            continue
        for operand in candidate.derived_operands:
            if operand not in protected:
                protected.add(operand)
                pending.append(operand)
    return protected


def _augment_interest_bearing_debt_share(
    question: str,
    candidates: Sequence[NumericCandidate],
    *,
    max_candidates: int,
) -> tuple[NumericCandidate, ...]:
    plan = compile_question_plan(question)
    if not plan.interest_bearing_debt_share or len(plan.requested_years) != 1:
        return tuple(candidates)
    if any(
        candidate.derivation_id == "interest_bearing_debt"
        for candidate in candidates
    ):
        return tuple(candidates)

    requested_year = plan.requested_years[0]
    grouped: dict[
        tuple[str, str, str, str, Decimal],
        dict[str, list[NumericCandidate]],
    ] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        role = _debt_share_role(candidate)
        scale = currency_unit_scale(candidate.unit)
        explicit_years = _candidate_period_years(candidate, question)
        if role is None or scale is None or requested_year not in explicit_years:
            continue
        location = (
            candidate.table_id
            or candidate.subtable_id
            or f"page:{candidate.page or 0}"
        )
        key = (
            candidate.source_name,
            candidate.entity_id,
            location,
            candidate.scope,
            scale,
        )
        grouped[key][role].append(candidate)

    derivations: list[
        tuple[NumericCandidate, NumericCandidate, NumericCandidate]
    ] = []
    required_roles = tuple(_DEBT_SHARE_ROLE_ALIASES)
    for role_map in grouped.values():
        if all(len(role_map.get(role, ())) == 1 for role in required_roles):
            derivations.append(
                tuple(role_map[role][0] for role in required_roles)  # type: ignore[arg-type]
            )
    if len(derivations) != 1:
        return tuple(candidates)

    short_term, long_term, total_capital = derivations[0]
    borrowings = short_term.decimal + long_term.decimal
    value_text = (
        str(int(borrowings))
        if borrowings == borrowings.to_integral_value()
        else format(borrowings, "f").rstrip("0").rstrip(".")
    )

    kept = list(candidates)
    if len(kept) >= max_candidates:
        protected = _derived_lineage_closure(
            kept,
            {
                short_term.index,
                long_term.index,
                total_capital.index,
                *(
                    candidate.index
                    for candidate in kept
                    if candidate.derived_operation
                ),
            },
        )
        removable = [
            candidate for candidate in kept if candidate.index not in protected
        ]
        if not removable:
            return tuple(candidates)
        kept.remove(
            min(removable, key=lambda candidate: (candidate.relevance, -candidate.index))
        )

    old_to_new = {
        candidate.index: index for index, candidate in enumerate(kept)
    }
    if any(
        source.index not in old_to_new
        for source in (short_term, long_term, total_capital)
    ):
        return tuple(candidates)
    reindexed = [
        replace(
            candidate,
            index=index,
            derived_operands=tuple(
                old_to_new[operand] for operand in candidate.derived_operands
            ),
        )
        for index, candidate in enumerate(kept)
    ]
    short_index = old_to_new[short_term.index]
    long_index = old_to_new[long_term.index]
    context = "\n".join(
        (
            "ROW: Tổng nợ vay tài chính chịu lãi gồm vay ngắn hạn và dài hạn",
            "ROW_TYPE: derived_identity",
            f"COLUMN: {total_capital.column_label or requested_year}",
            f"PERIOD: {requested_year}",
            f"TABLE: {total_capital.table_id}",
            f"SCOPE: {total_capital.scope}",
            "VALUE_ORIGIN: DERIVED_FROM_EXPLICIT_FINANCIAL_IDENTITY",
            "DAG_NODE: interest_bearing_debt=ADD(short_term_borrowings,long_term_borrowings)",
            f"SOURCE_SHORT_TERM: candidate[{short_index}]={short_term.value}",
            f"SOURCE_LONG_TERM: candidate[{long_index}]={long_term.value}",
            f"CELL: <VALUE> {value_text} </VALUE> {total_capital.unit}",
        )
    )
    derived = NumericCandidate(
        index=len(reindexed),
        value=value_text,
        raw=value_text,
        context=context,
        line_number=min(short_term.line_number, long_term.line_number),
        char_start=0,
        char_end=len(value_text),
        unit=total_capital.unit,
        is_percent=False,
        is_year_like=False,
        relevance=max(short_term.relevance, long_term.relevance) + 0.25,
        source_name=total_capital.source_name,
        page=total_capital.page,
        extraction_method="symbolic-derived",
        row_label=(
            "Tổng nợ vay tài chính chịu lãi gồm vay ngắn hạn và dài hạn"
        ),
        column_label=total_capital.column_label or requested_year,
        period=requested_year,
        row_type="derived_identity",
        table_id=total_capital.table_id,
        section=total_capital.section,
        scope=total_capital.scope,
        entity_id=total_capital.entity_id,
        subtable_id=total_capital.subtable_id,
        derived_operation="ADD",
        derived_operands=(short_index, long_index),
        derivation_id="interest_bearing_debt",
    )
    return tuple((*reindexed, derived))


def _augment_financial_identity_candidates(
    question: str,
    candidates: Sequence[NumericCandidate],
    *,
    max_candidates: int,
) -> tuple[NumericCandidate, ...]:
    """Materialize a provenance-carrying par-value intermediate when safe.

    The frozen V8 DSL remains unchanged. A deterministic first step derives
    par value from two disclosed cells in the same source; the existing typed
    MULTIPLY then consumes outstanding shares and this intermediate cell.
    """

    debt_augmented = _augment_interest_bearing_debt_share(
        question, candidates, max_candidates=max_candidates
    )
    if debt_augmented != tuple(candidates):
        return debt_augmented

    normalized_question = normalize_text(question)
    if not (
        "gia tri theo menh gia" in normalized_question
        and any(
            term in normalized_question for term in ("co phieu", "co phan")
        )
    ):
        return tuple(candidates)

    rows = {
        candidate.index: normalize_text(_candidate_metric_label(candidate))
        for candidate in candidates
    }
    if any("menh gia" in row for row in rows.values()):
        return tuple(candidates)
    capital = [
        candidate
        for candidate in candidates
        if "von dieu le" in rows.get(candidate.index, "")
        and candidate.decimal > 0
    ]
    issued = [
        candidate
        for candidate in candidates
        if any(
            phrase in rows.get(candidate.index, "")
            for phrase in (
                "so luong co phieu da phat hanh",
                "so luong co phan da phat hanh",
                "co phieu da phat hanh",
                "co phan da phat hanh",
            )
        )
        and candidate.decimal > 0
    ]
    outstanding = [
        candidate
        for candidate in candidates
        if any(
            phrase in rows.get(candidate.index, "")
            for phrase in (
                "so luong co phieu dang luu hanh",
                "so luong co phan dang luu hanh",
                "co phieu dang luu hanh",
                "co phan dang luu hanh",
            )
        )
        and candidate.decimal > 0
    ]
    if not capital or not issued or not outstanding:
        return tuple(candidates)

    derivations: list[
        tuple[float, NumericCandidate, NumericCandidate, Decimal]
    ] = []
    for capital_cell in capital:
        for issued_cell in issued:
            if capital_cell.source_name != issued_cell.source_name:
                continue
            quotient = capital_cell.decimal / issued_cell.decimal
            if quotient <= 0 or quotient != quotient.to_integral_value():
                continue
            derivations.append(
                (
                    capital_cell.relevance + issued_cell.relevance,
                    capital_cell,
                    issued_cell,
                    quotient,
                )
            )
    if not derivations:
        return tuple(candidates)
    _, capital_cell, issued_cell, par_value = max(
        derivations, key=lambda item: item[0]
    )
    same_source_outstanding = [
        cell
        for cell in outstanding
        if cell.source_name == capital_cell.source_name
    ]
    if not same_source_outstanding:
        return tuple(candidates)

    value_text = (
        str(int(par_value))
        if par_value == par_value.to_integral_value()
        else format(par_value, "f").rstrip("0").rstrip(".")
    )
    context = "\n".join(
        (
            "ROW: Mệnh giá cổ phần",
            "ROW_TYPE: derived_identity",
            "VALUE_ORIGIN: DERIVED_FROM_EXPLICIT_FINANCIAL_IDENTITY",
            "FORMULA: Vốn điều lệ / Số lượng cổ phần đã phát hành",
            f"SOURCE_LEFT: {capital_cell.value} VND page={capital_cell.page}",
            f"SOURCE_RIGHT: {issued_cell.value} shares page={issued_cell.page}",
            f"CELL: <VALUE> {value_text} </VALUE> VND/share",
        )
    )
    derived = NumericCandidate(
        index=len(candidates),
        value=value_text,
        raw=value_text,
        context=context,
        line_number=capital_cell.line_number,
        char_start=0,
        char_end=len(value_text),
        unit="VND/share",
        is_percent=False,
        is_year_like=False,
        relevance=max(capital_cell.relevance, issued_cell.relevance) + 0.25,
        source_name=capital_cell.source_name,
        page=capital_cell.page,
        extraction_method="symbolic-derived",
        row_label="Mệnh giá cổ phần",
        row_type="derived_identity",
        section="Derived financial identity",
        entity_id=capital_cell.entity_id or issued_cell.entity_id,
    )

    kept = list(candidates)
    if len(kept) >= max_candidates:
        protected = {id(capital_cell), id(issued_cell)} | {
            id(cell) for cell in same_source_outstanding
        }
        removable = [cell for cell in kept if id(cell) not in protected]
        if not removable:
            return tuple(candidates)
        drop = min(removable, key=lambda cell: (cell.relevance, -cell.index))
        kept.remove(drop)
    kept.append(derived)
    return tuple(
        replace(candidate, index=index)
        for index, candidate in enumerate(kept[:max_candidates])
    )


def _missing_requested_period_qualifiers(
    question: str, candidates: Sequence[NumericCandidate]
) -> tuple[str, ...]:
    normalized_question = normalize_text(question)
    requested = {
        item
        for item in ("cuoi nam", "cuoi ky", "dau nam", "dau ky")
        if item in normalized_question
    }
    if not requested:
        return ()
    observed: set[str] = set()
    for candidate in candidates:
        period_text = " ".join(
            line.removeprefix("PERIOD:").strip()
            for line in candidate.context.splitlines()
            if line.startswith("PERIOD:")
        )
        normalized_period = normalize_text(period_text)
        observed.update(item for item in requested if item in normalized_period)
    return tuple(sorted(requested - observed))


def _native_structured_evidence_gap(
    question: str, candidates: Sequence[NumericCandidate]
) -> str | None:
    """Explain why native evidence should be retried with MinerU evidence.

    This gate checks semantic coverage only while a fallback is available. It
    does not use model confidence, preserving the lazy-OCR contract for fully
    native documents.
    """

    pool = set(range(len(candidates)))
    operation = _apply_symbolic_operation_constraint(question, "SELECT")
    if operation == "SELECT":
        if not _metric_compatible_indices(question, candidates, pool):
            return "native evidence did not cover the requested metric"
    elif operation == "ADD":
        # OCR can recover a scanned table when native extraction cannot even
        # materialize the two structured rows required by binary addition.
        # Once two or more rows exist, role ambiguity belongs to binding/proof
        # and must remain UNKNOWN/ABSTAIN rather than launching OCR as a repair.
        if _predicted_add_requires_ocr_fallback(operation, candidates):
            return "native evidence did not cover both requested operand roles"
    elif operation == "RATIO_PERCENT":
        if _role_compatible_pair_indices(
            question, operation, candidates, pool
        ) is None:
            return "native evidence did not cover both requested operand roles"
    elif operation in {"SUBTRACT", "PERCENT_CHANGE"}:
        if _temporal_role_compatible_indices(
            question, operation, candidates, pool
        ) is None:
            return "native evidence did not cover the requested metric pair"
    missing_qualifiers = _missing_requested_period_qualifiers(question, candidates)
    if missing_qualifiers:
        return (
            "native evidence did not cover requested period qualifier(s): "
            + ", ".join(missing_qualifiers)
        )
    return None


def _predicted_add_requires_ocr_fallback(
    operation: str,
    candidates: Sequence[NumericCandidate],
) -> bool:
    """Retry OCR only when an ADD proposal cannot meet its structural arity."""

    return operation == "ADD" and sum(
        _is_groundable_document_candidate(candidate)
        and _has_financial_role_anchor(
            normalize_text(_candidate_metric_label(candidate))
        )
        for candidate in candidates
    ) < 2


def _infer_answer_unit(
    program: ReasoningProgram,
    evidence: Sequence[SelectedEvidence],
) -> str:
    if program.operation in {"RATIO_PERCENT", "PERCENT_CHANGE"}:
        return "%"
    if program.operation in {"DIVIDE"}:
        return "x"
    units = [item.candidate.unit for item in evidence if item.candidate.unit]
    normalized_units = {normalize_text(unit) for unit in units}
    if program.operation == "MULTIPLY" and normalized_units == {
        "shares",
        "vnd share",
    }:
        return "VND"
    if units and all(unit == units[0] for unit in units):
        return units[0]
    return ""


def _normalize_mixed_currency_result(
    question: str,
    program: ReasoningProgram,
    program_dag: ReasoningDAGProgram | None,
    candidates: Sequence[NumericCandidate],
    evidence_indices: Sequence[int],
    value: Decimal | None,
) -> tuple[Decimal | None, str]:
    """Replay explicit mixed-scale currency leaves on one canonical basis."""

    output_operation = program.operation
    if program_dag is not None:
        output_node = next(
            (
                node
                for node in program_dag.nodes
                if node.node_id == program_dag.output_node
            ),
            None,
        )
        if output_node is not None:
            output_operation = output_node.operation
    supported = {
        "SELECT",
        "ADD",
        "SUBTRACT",
        "AVERAGE",
        "SUM3",
        "SUM4",
        SUM_N_OPERATION,
        "DIVIDE",
        "RATIO_PERCENT",
        "PERCENT_CHANGE",
        "SHARE_OF_TOTAL",
    }
    if value is None or output_operation not in supported:
        return value, ""
    scales = tuple(
        currency_unit_scale(candidates[index].unit) for index in evidence_indices
    )
    if not scales or any(scale is None for scale in scales):
        return value, ""
    plan = compile_question_plan(question)
    targets = {
        "vnd": (Decimal(1), "VND"),
        "thousand_vnd": (Decimal(1000), "nghìn đồng"),
        "million_vnd": (Decimal(1000000), "triệu đồng"),
        "billion_vnd": (Decimal(1000000000), "tỷ đồng"),
    }
    target = targets.get(plan.requested_scale)
    currency_output = (
        output_operation in {"SELECT", "SUBTRACT", "AVERAGE"}
        or is_sum_operation(output_operation)
    )
    mixed_source_scales = len(set(scales)) > 1
    heterogeneous_source_units = len(
        {
            normalize_text(candidates[index].unit)
            for index in evidence_indices
            if candidates[index].unit
        }
    ) > 1
    target_differs = bool(
        currency_output
        and target is not None
        and any(scale != target[0] for scale in scales)
    )
    if (
        not mixed_source_scales
        and not heterogeneous_source_units
        and not target_differs
    ):
        return value, ""
    converted = tuple(
        replace(candidate, value=format(candidate.decimal * scale, "f"))
        if (scale := currency_unit_scale(candidate.unit)) is not None
        else candidate
        for candidate in candidates
    )
    normalized = (
        execute_program_dag(program_dag, converted)
        if program_dag is not None
        else execute_program(program, converted)
    )
    if output_operation in {
        "RATIO_PERCENT",
        "PERCENT_CHANGE",
        "SHARE_OF_TOTAL",
    }:
        return normalized, "%"
    if output_operation == "DIVIDE":
        return normalized, "x"

    target_scale, target_unit = targets.get(
        plan.requested_scale, (Decimal(1), "VND")
    )
    return normalized / target_scale, target_unit


def _format_value(value: Decimal | None) -> str:
    if value is None:
        return "not disclosed"
    if value == value.to_integral_value():
        return f"{int(value):,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")
