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
    NumericCandidate,
    ReasoningDAGProgram,
    ReasoningProgram,
    currency_unit_scale,
    execute_program,
    execute_program_dag,
    materialize_program_dag,
    merge_candidates,
)
from .ontology import normalize_text
from .financial_identities import metric_alias_score
from .question_planner import _has_financial_role_anchor, compile_question_plan
from .symbolic_verifier import VerificationState, verify_program


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
    source_action: str = "ABSTAIN"
    verification_state: str = "UNKNOWN"
    proof_obligations: tuple[tuple[str, str, str | None], ...] = ()
    verification_reasons: tuple[str, ...] = ()
    acceptance_basis: tuple[str, ...] = ()

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
        value = _format_value(None if self.abstained else self.value)
        candidate_value = _format_value(self.value)
        suffix = f" {self.unit}" if self.unit else ""
        lines = [
            f"Answer: {value}{suffix}",
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
                "Source path: real PDF/TXT -> native-first retrieval -> optional "
                "MinerU fallback -> numeric candidates -> H/L latent recurrence -> "
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
                "Rejected candidate answer (technical audit only): "
                f"{candidate_value}{suffix}"
            )
        return "\n".join(lines)


class DocumentHRMReasoningEngine:
    """End-to-end numerical QA over actual PDF/TXT source text."""

    def __init__(self, bundle: DocumentHRMBundle) -> None:
        self.bundle = bundle

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
        stage_started = time.perf_counter()
        documents = discover_documents(document_path)
        record_stage("document_discovery", stage_started)
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
        native_first = ocr_config.enabled and not ocr_config.force
        attempt_configs = (
            (replace(ocr_config, enabled=False, force=False), ocr_config)
            if native_first
            else (ocr_config,)
        )
        fallback_reason: str | None = None
        page_index_stats: PageIndexStats | None = None
        retrieval_warnings: list[str] = []
        ranked: list[EvidenceChunk] = []
        candidates: list[NumericCandidate] = []
        prediction: ProgramPrediction
        program_dag: ReasoningDAGProgram | None = None
        value: Decimal
        missing_periods: tuple[str, ...] = ()

        for attempt_index, attempt_config in enumerate(attempt_configs):
            has_fallback = attempt_index + 1 < len(attempt_configs)
            attempt_name = "native" if not attempt_config.enabled else "mineru"
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
                if has_fallback:
                    fallback_reason = f"native document extraction failed: {exc}"
                    continue
                raise

            stage_started = time.perf_counter()
            ranked = rank_chunks(question, chunks, top_k=top_k_chunks)
            ranked = expand_neighbor_chunks(
                ranked,
                chunks,
                neighbor_window=neighbor_chunks,
                max_chunks=top_k_chunks + max(0, neighbor_chunks * 4),
            )
            record_stage(f"{attempt_name}_chunk_ranking", stage_started)
            if not ranked:
                if has_fallback:
                    fallback_reason = "no native source chunk matched the question"
                    continue
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
                if has_fallback:
                    fallback_reason = (
                        "native retrieval produced no numerical candidates"
                    )
                    continue
                raise DocumentHRMError(
                    "No numerical values were extracted from the matched source chunks"
                )

            missing_periods = _missing_requested_periods(question, candidates)
            if missing_periods and has_fallback:
                fallback_reason = (
                    "native evidence did not cover requested period(s): "
                    + ", ".join(missing_periods)
                )
                continue
            native_gap = (
                _native_structured_evidence_gap(question, candidates)
                if has_fallback
                else None
            )
            if native_gap:
                fallback_reason = native_gap
                continue

            stage_started = time.perf_counter()
            prediction = self.bundle.predict(question, candidates)
            record_stage(f"{attempt_name}_prediction", stage_started)
            if has_fallback and _predicted_add_requires_ocr_fallback(
                prediction.program.operation,
                candidates,
            ):
                fallback_reason = (
                    "native ADD proposal had fewer than two structured "
                    "operand rows"
                )
                continue
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
            if "zero_baseline_percent_change" in preexecution_rejections:
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
                    if has_fallback:
                        fallback_reason = f"native program was not executable: {exc}"
                        continue
                    raise DocumentHRMError(
                        f"Predicted program is not executable: {exc}"
                    ) from exc

            page_index_stats = attempt_stats
            retrieval_warnings = list(attempt_warnings)
            if attempt_index > 0 and fallback_reason:
                retrieval_warnings.insert(
                    0,
                    "Native evidence was insufficient; retried with MinerU-enabled "
                    f"index ({fallback_reason}).",
                )
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
        unit = _infer_answer_unit(prediction.program, evidence)
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
        abstained = (
            (
                prediction.confidence < threshold
                and not proof_conditioned_acceptance
            )
            or bool(missing_periods)
            or derived_formula_evidence
            or bool(grounding_rejections)
        )
        warnings: list[str] = list(retrieval_warnings)
        if self._uses_calibration_fallback():
            warnings.append(
                "The checkpoint has no valid calibrated threshold; runtime is using "
                "the experimental fallback threshold 0.75."
            )
        if prediction.confidence < threshold and not proof_conditioned_acceptance:
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
        if prediction.program.operation in {"ADD", "SUBTRACT", "AVERAGE", "SUM3", "SUM4"} and len(operand_units) > 1:
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
        verification = verify_program(
            compile_question_plan(question),
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
        if (
            verification.state is VerificationState.INVALID
            or VerificationState.INVALID.value in required_states
        ):
            source_action = "REJECT"
        elif (
            abstained
            or not required_states
            or any(state != VerificationState.VALID.value for state in required_states)
        ):
            source_action = "ABSTAIN"
        else:
            source_action = "ANSWER"
        acceptance_basis = (
            "runtime_action_preserved",
            "full_candidate_evidence_graph_verification",
            (
                "required_proof_valid"
                if source_action == "ANSWER"
                else "fail_closed"
            ),
        )
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
            source_action=source_action,
            verification_state=verification.state.value,
            proof_obligations=proof_obligations,
            verification_reasons=tuple(
                dict.fromkeys((*verification.reasons, *verification.unknown_reasons))
            ),
            acceptance_basis=acceptance_basis,
        )

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
            if "zero_baseline_percent_change" in grounding_rejections
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
        if prediction.confidence < threshold and not proof_conditioned_acceptance:
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
        warnings = tuple(warning_items)
        return DocumentReasoningAnswer(
            question=question,
            value=value,
            unit=_infer_answer_unit(prediction.program, evidence),
            prediction=prediction,
            evidence=evidence,
            retrieved_chunks=(chunk,),
            document_count=1,
            candidate_count=len(candidates),
            abstained=abstained,
            confidence_threshold=threshold,
            warnings=warnings,
            program_dag=program_dag,
        )

    def _uses_calibration_fallback(self) -> bool:
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


def _missing_requested_periods(
    question: str, candidates: Sequence[NumericCandidate]
) -> tuple[str, ...]:
    requested = set(re.findall(r"\b(?:19|20)\d{2}\b", question))
    if not requested:
        return ()
    covered: set[str] = set()
    for candidate in candidates:
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


def _format_value(value: Decimal | None) -> str:
    if value is None:
        return "not disclosed"
    if value == value.to_integral_value():
        return f"{int(value):,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")
