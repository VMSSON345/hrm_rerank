from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from decimal import Decimal
from pathlib import Path

from .dense_retrieval import (
    DEFAULT_DENSE_MODEL,
    DenseEmbeddingIndex,
    DenseRetrievalError,
    HybridFinancialDocumentIndex,
)
from .document_agent import DocumentEvidenceResult, DocumentEvidenceRetriever
from .document_search import (
    DocumentSearchError,
    SearchHit,
    VietnameseFinancialDocumentIndex,
)
from .document_hrm import (
    analyze_document_hrm_errors,
    DocumentHRMConfig,
    DocumentHRMBundle,
    DocumentHRMError,
    checkpoint_validation_examples,
    evaluate_document_hrm,
    model_summary as document_hrm_summary,
    train_document_hrm,
)
from .document_reasoning_engine import DocumentHRMReasoningEngine
from .document_sources import (
    DocumentSourceError,
    OCRConfig,
    materialize_mineru_markdown,
)
from .external_evaluation import audit_external_test_set
from .program_data import (
    OPERATIONS,
    SUM_N_OPERATION,
    build_program_dataset,
    generate_synthetic_program_examples,
    load_program_examples,
    write_program_examples,
)
from .program_reranker import ProgramRerankerBundle
from .engine import FinancialQAEngine, QAError
from .hrm import NeuroSymbolicHRMEngine
from .reasoning import (
    HRMConfig,
    HierarchicalReasoningBundle,
    ReasoningModelError,
    evaluate_reasoning_bundle,
    model_summary_json,
    train_reasoning_bundle,
)
from .reasoning_engine import HierarchicalReasoningQAEngine
from .neural import (
    NeuralModelBundle,
    NeuralModelError,
    evaluate_model_bundle,
    train_model_bundle,
)
from .ontology import FORMULAS, normalize_text
from .repository import FinancialRepository


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "vnstock_output_500_core"
DEFAULT_MODEL_PATH = ROOT / "models" / "hrm_reasoner.pt"
DEFAULT_DOCUMENT_HRM_PATH = (
    ROOT / "models" / "document_hrm_V8_candidate.pt"
)
DEFAULT_HRM_INSTRUCTION_JSONL = ROOT / "data" / "hrm_instruction_1k.jsonl"
DEFAULT_HRM_PROGRAM_DATASET = ROOT / "data" / "hrm_programs_clean_v5.jsonl"
DEFAULT_LEGACY_MODEL_PATH = ROOT / "models" / "hrm_neural.json"
DEFAULT_DOCUMENT_TEXT_DIR = (
    ROOT
    / "Instruction_Data_of_Vietnamese_Financial_report-main"
    / "folder_2.6_txt"
)
DEFAULT_DOCUMENT_INDEX_PATH = ROOT / "models" / "instruction_document_index.json"
DEFAULT_DENSE_INDEX_PATH = ROOT / "models" / "instruction_dense_index.npz"
DEFAULT_EMBEDDING_CACHE_DIR = ROOT / "models" / "embedding_cache"


def _validate_document_hrm_training_output(
    output_path: str | Path,
    *,
    initialize_from: str | Path | None,
    overwrite: bool,
) -> Path:
    destination = Path(output_path).expanduser().resolve()
    frozen_v8 = DEFAULT_DOCUMENT_HRM_PATH.resolve()
    if destination == frozen_v8:
        raise DocumentHRMError(
            "Refusing to use the frozen V8 checkpoint as a training output; "
            "choose a new --document-model-path"
        )
    if (
        initialize_from is not None
        and destination == Path(initialize_from).expanduser().resolve()
    ):
        raise DocumentHRMError(
            "Training output cannot overwrite initialize_from"
        )
    if destination.exists() and not overwrite:
        raise DocumentHRMError(
            "Training output already exists; choose a new path or pass "
            "--overwrite-output explicitly"
        )
    return destination


def _load_document_hrm_runtime(args: argparse.Namespace) -> DocumentHRMBundle:
    bundle = DocumentHRMBundle.load(
        args.document_model_path, device=args.device
    )
    typed_synthesis = getattr(args, "typed_program_synthesis", None)
    if typed_synthesis is not None:
        bundle.metadata = dict(bundle.metadata)
        bundle.metadata["typed_program_synthesis_enabled"] = bool(
            typed_synthesis
        )
    identity_catalog = getattr(args, "financial_identity_catalog", None)
    if identity_catalog is not None:
        bundle.metadata = dict(bundle.metadata)
        bundle.metadata["financial_identity_catalog_enabled"] = bool(
            identity_catalog
        )
    candidate_recall = getattr(args, "candidate_recall_v12_4", None)
    if candidate_recall is not None:
        bundle.metadata = dict(bundle.metadata)
        bundle.metadata["candidate_recall_v12_4_enabled"] = bool(
            candidate_recall
        )
    reranker_path = getattr(args, "program_reranker_path", None)
    if reranker_path is not None:
        reranker = ProgramRerankerBundle.load(reranker_path, device=args.device)
        reranker.validate_base_model(args.document_model_path)
        bundle.attach_program_reranker(reranker)
    return bundle


def _wilson_interval(successes: int, total: int) -> dict[str, object]:
    if total <= 0:
        return {
            "successes": successes,
            "total": total,
            "lower": 0.0,
            "upper": 0.0,
            "defined": False,
            "value": None,
        }
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * (
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        ** 0.5
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "lower": max(0.0, center - margin),
        "defined": True,
        "value": proportion,
        "upper": min(1.0, center + margin),
    }


def _evaluation_report_operations(
    bundle: DocumentHRMBundle,
    metrics: dict[str, float],
) -> tuple[str, ...]:
    """Select the checkpoint's logical metric schema without changing legacy."""

    raw = bundle.metadata.get("logical_program_operations")
    if isinstance(raw, list):
        configured = tuple(
            dict.fromkeys(item for item in raw if isinstance(item, str) and item)
        )
        if configured and all(
            f"program_operation_{operation.lower()}_support" in metrics
            for operation in configured
        ):
            return configured
    return tuple(
        operation
        for operation in (*OPERATIONS, SUM_N_OPERATION)
        if f"program_operation_{operation.lower()}_support" in metrics
    ) or OPERATIONS


def _document_hrm_evaluation_report(
    bundle: DocumentHRMBundle,
    metrics: dict[str, float],
    *,
    checkpoint: Path,
    dataset: Path,
    mode: str,
    example_count: int,
    adaptive_inference: bool = False,
) -> dict[str, object]:
    per_operation = {}
    report_operations = _evaluation_report_operations(bundle, metrics)
    for operation in report_operations:
        classification_prefix = f"operation_class_{operation.lower()}"
        quality_prefix = f"program_operation_{operation.lower()}"
        per_operation[operation] = {
            "support": int(metrics[f"{quality_prefix}_support"]),
            "classification": {
                "precision": metrics[f"{classification_prefix}_precision"],
                "recall": metrics[f"{classification_prefix}_recall"],
                "f1": metrics[f"{classification_prefix}_f1"],
            },
            "operation_accuracy": metrics[f"{quality_prefix}_operation_accuracy"],
            "pointer_pair_accuracy": metrics[
                f"{quality_prefix}_pointer_pair_accuracy"
            ],
            "raw_pointer_pair_accuracy": metrics[
                f"{quality_prefix}_raw_pointer_pair_accuracy"
            ],
            "program_exact_accuracy": metrics[
                f"{quality_prefix}_program_exact_accuracy"
            ],
            "raw_program_exact_accuracy": metrics[
                f"{quality_prefix}_raw_program_exact_accuracy"
            ],
            "grounded_semantic_program_accuracy": metrics[
                f"{quality_prefix}_grounded_semantic_program_accuracy"
            ],
            "execution_accuracy": metrics[
                f"{quality_prefix}_execution_accuracy"
            ],
            "accepted_coverage": metrics[f"{quality_prefix}_accepted_coverage"],
            "accepted_execution_precision": metrics[
                f"{quality_prefix}_accepted_execution_precision"
            ],
            "accepted_program_precision": metrics[
                f"{quality_prefix}_accepted_program_precision"
            ],
            "accepted_precision_defined": bool(
                metrics[f"{quality_prefix}_accepted_precision_defined"]
            ),
            "accepted_execution_precision_value": (
                metrics[f"{quality_prefix}_accepted_execution_precision"]
                if bool(
                    metrics[f"{quality_prefix}_accepted_execution_precision_defined"]
                )
                else None
            ),
            "proof_eligible_coverage": metrics[
                f"{quality_prefix}_proof_eligible_coverage"
            ],
            "proof_eligible_execution_precision": metrics[
                f"{quality_prefix}_proof_eligible_execution_precision"
            ],
            "proof_eligible_program_precision": metrics[
                f"{quality_prefix}_proof_eligible_program_precision"
            ],
            "proof_eligible_precision_defined": bool(
                metrics[f"{quality_prefix}_proof_eligible_precision_defined"]
            ),
            "proof_eligible_count": int(
                metrics[f"{quality_prefix}_proof_eligible_count"]
            ),
            "confidence_intervals_95": {
                "execution_accuracy": _wilson_interval(
                    round(
                        metrics[f"{quality_prefix}_execution_accuracy"]
                        * metrics[f"{quality_prefix}_support"]
                    ),
                    int(metrics[f"{quality_prefix}_support"]),
                ),
                "accepted_execution_precision": _wilson_interval(
                    int(
                        metrics[
                            f"{quality_prefix}_accepted_execution_correct_count"
                        ]
                    ),
                    int(metrics[f"{quality_prefix}_accepted_count"]),
                ),
            },
        }
    return {
        "model": {
            "checkpoint": str(checkpoint.resolve()),
            "architecture": bundle.metadata.get("architecture"),
            "parameters": bundle.parameter_count,
            "input_encoder": bundle.model.config.input_encoder_mode,
            "semantic_model": bundle.model.config.semantic_model_name,
            "program_reranker": bundle.metadata.get(
                "program_reranker_architecture"
            ),
            "program_reranker_parameters": (
                bundle.program_reranker.parameter_count
                if bundle.program_reranker is not None
                else 0
            ),
        },
        "evaluation": {
            "mode": mode,
            "dataset": str(dataset.resolve()),
            "examples": example_count,
            "validation_split": bundle.metadata.get("validation_split"),
            "program_decoder": (
                str(
                    bundle.metadata.get(
                        "program_reranker_architecture",
                        "safe-residual-frozen-hrm-reranker",
                    )
                )
                if bundle.program_reranker is not None
                else "training-free-proof-guided-search-v2"
            ),
            "typed_evidence_graph": bundle.metadata.get(
                "typed_evidence_graph_enabled", True
            ),
            "proof_guided_reranking": bundle.metadata.get(
                "proof_guided_reranking_enabled", True
            ),
            "reasoning_policy": (
                "adaptive_serving"
                if adaptive_inference
                else "fixed_max_segments_batch"
            ),
        },
        "metrics": {
            "operation_accuracy": metrics["operation_accuracy"],
            "operation_macro": {
                "precision": metrics["operation_macro_precision"],
                "recall": metrics["operation_macro_recall"],
                "f1": metrics["operation_macro_f1"],
            },
            "operation_weighted": {
                "precision": metrics["operation_weighted_precision"],
                "recall": metrics["operation_weighted_recall"],
                "f1": metrics["operation_weighted_f1"],
            },
            "program_exact_accuracy": metrics["program_exact_accuracy"],
            "raw_program_exact_accuracy": metrics[
                "raw_program_exact_accuracy"
            ],
            "grounded_semantic_program_accuracy": metrics[
                "grounded_semantic_program_accuracy"
            ],
            "execution_accuracy": metrics["execution_accuracy"],
            "pointer_pair_accuracy": metrics["pointer_pair_accuracy"],
            "raw_pointer_pair_accuracy": metrics[
                "raw_pointer_pair_accuracy"
            ],
            "pointer_slot_accuracy": [
                metrics[f"pointer_slot_{slot + 1}_accuracy"]
                for slot in range(bundle.model.config.max_program_operands)
            ],
            "pointer_slot_support": [
                int(metrics[f"pointer_slot_{slot + 1}_support"])
                for slot in range(bundle.model.config.max_program_operands)
            ],
            "pointer_slot_accuracy_defined": [
                bool(metrics[f"pointer_slot_{slot + 1}_accuracy_defined"])
                for slot in range(bundle.model.config.max_program_operands)
            ],
            "pointer_slot_accuracy_or_null": [
                metrics[f"pointer_slot_{slot + 1}_accuracy"]
                if bool(metrics[f"pointer_slot_{slot + 1}_accuracy_defined"])
                else None
                for slot in range(bundle.model.config.max_program_operands)
            ],
            "mean_reasoning_segments": metrics["mean_segments"],
            "program_search_change_rate": metrics[
                "program_search_change_rate"
            ],
            "mean_program_candidates_considered": metrics[
                "mean_program_candidates_considered"
            ],
            "mean_symbolic_program_rejections": metrics[
                "mean_symbolic_program_rejections"
            ],
            "mean_symbolic_unknown_constraints": metrics[
                "mean_symbolic_unknown_constraints"
            ],
            "gold_program_recall_at_5": metrics[
                "gold_program_recall_at_5"
            ],
            "gold_program_recall_at_10": metrics[
                "gold_program_recall_at_10"
            ],
            "gold_physical_program_recall_at_5": metrics.get(
                "gold_physical_program_recall_at_5", 0.0
            ),
            "gold_physical_program_recall_at_10": metrics.get(
                "gold_physical_program_recall_at_10", 0.0
            ),
            "gold_operand_recall_at_5": metrics[
                "gold_operand_recall_at_5"
            ],
            "gold_operand_recall_at_10": metrics[
                "gold_operand_recall_at_10"
            ],
            "operand_candidate_recall_at_10": metrics.get(
                "operand_candidate_recall_at_10", 0.0
            ),
            "operand_candidate_recall_at_20": metrics.get(
                "operand_candidate_recall_at_20", 0.0
            ),
            "operand_candidate_full_coverage_at_10": metrics.get(
                "operand_candidate_full_coverage_at_10", 0.0
            ),
            "operand_candidate_full_coverage_at_20": metrics.get(
                "operand_candidate_full_coverage_at_20", 0.0
            ),
            "operand_candidate_recall_at_10_given_gold_operation": metrics.get(
                "operand_candidate_recall_at_10_given_gold_operation", 0.0
            ),
            "operand_candidate_recall_at_20_given_gold_operation": metrics.get(
                "operand_candidate_recall_at_20_given_gold_operation", 0.0
            ),
            "operand_candidate_gold_operand_count": metrics.get(
                "operand_candidate_gold_operand_count", 0.0
            ),
            "operand_candidate_gold_operation_operand_count": metrics.get(
                "operand_candidate_gold_operation_operand_count", 0.0
            ),
            "diverse_beam_diagnostic_examples": metrics.get(
                "diverse_beam_diagnostic_examples", 0.0
            ),
            "program_search_failure_buckets": {
                "denominator": metrics.get(
                    "diverse_beam_diagnostic_examples", 0.0
                ),
                "bucket_total": metrics.get(
                    "program_search_failure_bucket_total", 0.0
                ),
                "operation_not_considered": metrics.get(
                    "gold_operation_not_considered_count", 0.0
                ),
                "operand_outside_slot_pool_at_20": metrics.get(
                    "gold_operand_outside_slot_pool_at_20_count", 0.0
                ),
                "program_not_generated": metrics.get(
                    "gold_program_not_generated_count", 0.0
                ),
                "program_below_preverifier_top10": metrics.get(
                    "gold_program_below_preverifier_top10_count", 0.0
                ),
                "top10_verifier_invalid": metrics.get(
                    "gold_top10_verifier_invalid_count", 0.0
                ),
                "top10_runtime_gate_rejected": metrics.get(
                    "gold_top10_runtime_gate_rejected_count", 0.0
                ),
                "top10_eligible": metrics.get(
                    "gold_top10_eligible_count", 0.0
                ),
                "diagnostic_unavailable": metrics.get(
                    "gold_pipeline_diagnostic_unavailable_count", 0.0
                ),
            },
            "exact_physical_set_by_arity": {
                str(arity): {
                    "support": metrics.get(
                        f"exact_physical_set_arity_{arity}_support", 0.0
                    ),
                    "accuracy": metrics.get(
                        f"exact_physical_set_arity_{arity}_accuracy", 0.0
                    ),
                }
                for arity in range(2, 9)
            },
            "exact_physical_program_by_arity": {
                str(arity): {
                    "support": metrics.get(
                        f"exact_physical_program_arity_{arity}_support", 0.0
                    ),
                    "accuracy": metrics.get(
                        f"exact_physical_program_arity_{arity}_accuracy", 0.0
                    ),
                }
                for arity in range(2, 9)
            },
            "verifier_gold_generated_count": metrics[
                "verifier_gold_generated_count"
            ],
            "verifier_gold_false_rejection_count": metrics[
                "verifier_gold_false_rejection_count"
            ],
            "verifier_gold_false_rejection_rate": metrics[
                "verifier_gold_false_rejection_rate"
            ],
            "search_pipeline_gold_rejection_rate": metrics[
                "search_pipeline_gold_rejection_rate"
            ],
            "accepted_coverage": metrics["accepted_coverage"],
            "accepted_precision": metrics["accepted_precision"],
            "accepted_execution_precision": metrics[
                "accepted_execution_precision"
            ],
            "accepted_program_precision": metrics["accepted_program_precision"],
            "accepted_grounded_semantic_program_precision": metrics[
                "accepted_grounded_semantic_program_precision"
            ],
            "accepted_precision_defined": bool(metrics["accepted_precision_defined"]),
            "accepted_execution_precision_defined": bool(
                metrics["accepted_execution_precision_defined"]
            ),
            "accepted_precision_value": (
                metrics["accepted_precision"]
                if bool(metrics["accepted_precision_defined"])
                else None
            ),
            "proof_eligible_coverage": metrics["proof_eligible_coverage"],
            "proof_eligible_precision": metrics["proof_eligible_precision"],
            "proof_eligible_execution_precision": metrics[
                "proof_eligible_execution_precision"
            ],
            "proof_eligible_program_precision": metrics[
                "proof_eligible_program_precision"
            ],
            "proof_eligible_precision_defined": bool(
                metrics["proof_eligible_precision_defined"]
            ),
            "proof_eligible_count": int(metrics["proof_eligible_count"]),
        },
        "metric_definitions": {
            "per_operation_grouping": (
                "Rows are grouped by the gold operation; support is gold support."
            ),
            "classification": (
                "Precision/recall/F1 measure only the predicted operation label."
            ),
            "program_exact_accuracy": (
                "The operation and operand pointers must match gold after "
                "canonicalizing operand order for commutative operations."
            ),
            "raw_program_exact_accuracy": (
                "The operation and serialized operand pointer order must match gold."
            ),
            "grounded_semantic_program_accuracy": (
                "A canonical exact match, or an execution-equivalent program "
                "whose operation and typed evidence signatures match gold."
            ),
            "pointer_pair_accuracy": (
                "Operand pointers must match gold after canonicalizing operand "
                "order for commutative operations."
            ),
            "raw_pointer_pair_accuracy": (
                "Serialized operand pointer order must match gold."
            ),
            "execution_accuracy": (
                "The predicted program must execute to the gold numeric answer."
            ),
            "accepted_precision": (
                "Backward-compatible alias of accepted_execution_precision."
            ),
            "pointer_slot_support": (
                "Support is the number of gold examples declaring that operand slot; "
                "accuracy is N/A when support is zero."
            ),
            "accepted_precision_defined": (
                "False when no policy-calibrated answer was accepted; precision is N/A."
            ),
            "proof_eligible_coverage": (
                "Coverage passing the public proof/source-action gate before "
                "calibration policy acceptance."
            ),
            "gold_program_recall_at_k": (
                "Legacy pointer-exact gold program recall in the ranked "
                "search trace; retained for historical comparison."
            ),
            "gold_physical_program_recall_at_k": (
                "A conflict-aware physical-cell-equivalent gold program "
                "appears in the first K diverse programs before verification."
            ),
            "gold_operand_recall_at_k": (
                "Every gold operand appears in the union of operands from the "
                "K highest neural-scored programs."
            ),
            "operand_candidate_recall_at_k": (
                "Micro recall: matched conflict-aware gold physical operands "
                "divided by all gold operands. Pools are conditioned on the "
                "logical gold operation and matched injectively to slots."
            ),
            "operand_candidate_full_coverage_at_k": (
                "Fraction of all evaluated examples whose complete gold "
                "operand set can be injectively assigned to the operation-"
                "conditioned per-slot pools."
            ),
            "program_search_failure_buckets": (
                "Exhaustive exclusive buckets over every diverse-beam example: "
                "operation miss, pool miss, composition miss, pre-verifier "
                "rank miss, verifier INVALID, other runtime-gate rejection, "
                "eligible success, or unavailable telemetry."
            ),
            "exact_physical_set_by_arity": (
                "For gold SUM_N examples only, unordered conflict-aware "
                "physical operand-set accuracy, independent of operation label."
            ),
            "exact_physical_program_by_arity": (
                "Conflict-aware physical program accuracy for every operation; "
                "includes operation and directed operand order where applicable."
            ),
            "verifier_gold_false_rejection_rate": (
                "Among generated gold programs, fraction marked INVALID by "
                "the three-state verifier; UNKNOWN is not a rejection."
            ),
        },
        "confidence_intervals_95": {
            "program_exact_accuracy": _wilson_interval(
                int(metrics["program_exact_correct_count"]),
                int(metrics["example_count"]),
            ),
            "raw_program_exact_accuracy": _wilson_interval(
                int(metrics["raw_program_exact_correct_count"]),
                int(metrics["example_count"]),
            ),
            "grounded_semantic_program_accuracy": _wilson_interval(
                int(metrics["grounded_semantic_program_correct_count"]),
                int(metrics["example_count"]),
            ),
            "execution_accuracy": _wilson_interval(
                int(metrics["execution_correct_count"]),
                int(metrics["example_count"]),
            ),
            "accepted_execution_precision": _wilson_interval(
                int(metrics["accepted_execution_correct_count"]),
                int(metrics["accepted_count"]),
            ),
            "accepted_program_precision": _wilson_interval(
                int(metrics["accepted_program_correct_count"]),
                int(metrics["accepted_count"]),
            ),
        },
        "per_operation": per_operation,
        "calibration": {
            "confidence_schema": bundle.metadata.get(
                "confidence_schema", "raw-neural-v1"
            ),
            "mode": bundle.metadata.get("abstention_calibration_mode"),
            "threshold": bundle.metadata.get("abstention_threshold"),
            "thresholds_by_operation": bundle.metadata.get(
                "abstention_thresholds_by_operation", {}
            ),
            "policies_by_operation": bundle.metadata.get(
                "abstention_policies_by_operation", {}
            ),
            "target_precision": bundle.metadata.get("abstention_target_precision"),
            "target_met": bundle.metadata.get("abstention_target_met"),
            "stability_errors": bundle.metadata.get(
                "abstention_stability_errors", 0
            ),
            "program_reranker": (
                {
                    "confidence_schema": bundle.metadata.get(
                        "program_reranker_confidence_schema"
                    ),
                    "mode": bundle.metadata.get(
                        "program_reranker_calibration_mode"
                    ),
                    "thresholds_by_operation": bundle.metadata.get(
                        "program_reranker_thresholds_by_operation", {}
                    ),
                }
                if bundle.program_reranker is not None
                else None
            ),
        },
    }


def _format_document_hrm_evaluation_summary(
    report: dict[str, object],
) -> str:
    evaluation = report["evaluation"]
    metrics = report["metrics"]
    intervals = report["confidence_intervals_95"]
    per_operation = report["per_operation"]

    execution_interval = intervals["execution_accuracy"]
    accepted_interval = intervals["accepted_execution_precision"]
    accepted_execution_text = (
        f"{metrics['accepted_execution_precision']:.2%}"
        if metrics.get("accepted_precision_defined", True)
        else "N/A"
    )
    accepted_program_text = (
        f"{metrics['accepted_program_precision']:.2%}"
        if metrics.get("accepted_precision_defined", True)
        else "N/A"
    )
    proof_precision_text = (
        f"{metrics['proof_eligible_execution_precision']:.2%}"
        if metrics.get("proof_eligible_precision_defined", False)
        else "N/A"
    )
    accepted_interval_text = (
        (
            f"{accepted_interval['lower']:.2%}-"
            f"{accepted_interval['upper']:.2%}"
        )
        if accepted_interval.get("defined", False)
        else "N/A"
    )
    lines = [
        (
            f"Document HRM evaluation | n={evaluation['examples']} | "
            f"{evaluation['mode']}"
        ),
        (
            "Overall  "
            f"operation={metrics['operation_accuracy']:.2%}  "
            f"program={metrics['program_exact_accuracy']:.2%}  "
            f"raw_program={metrics['raw_program_exact_accuracy']:.2%}  "
            f"semantic={metrics['grounded_semantic_program_accuracy']:.2%}  "
            f"execution={metrics['execution_accuracy']:.2%}  "
            f"pointer={metrics['pointer_pair_accuracy']:.2%}  "
            f"raw_pointer={metrics['raw_pointer_pair_accuracy']:.2%}"
        ),
        (
            "Accepted "
            f"coverage={metrics['accepted_coverage']:.2%}  "
            f"execution_precision="
            f"{accepted_execution_text}  "
            f"program_precision={accepted_program_text}"
        ),
        (
            "Proof    "
            f"coverage={metrics['proof_eligible_coverage']:.2%}  "
            f"execution_precision={proof_precision_text}"
        ),
        (
            "Search   "
            f"changed={metrics['program_search_change_rate']:.2%}  "
            f"candidates={metrics['mean_program_candidates_considered']:.1f}  "
            f"rejected={metrics['mean_symbolic_program_rejections']:.1f}"
        ),
        (
            "Recall   "
            f"physical_program@10="
            f"{metrics.get('gold_physical_program_recall_at_10', 0.0):.2%}  "
            f"operands@10={metrics['gold_operand_recall_at_10']:.2%}  "
            f"candidate@10="
            f"{metrics.get('operand_candidate_recall_at_10', 0.0):.2%}  "
            f"candidate@20="
            f"{metrics.get('operand_candidate_recall_at_20', 0.0):.2%}  "
            f"full@20="
            f"{metrics.get('operand_candidate_full_coverage_at_20', 0.0):.2%}"
        ),
        (
            "Verifier "
            f"gold_false_reject="
            f"{metrics['verifier_gold_false_rejection_rate']:.2%}  "
            f"pipeline_gold_reject="
            f"{metrics['search_pipeline_gold_rejection_rate']:.2%}"
        ),
        (
            "95% CI   "
            f"execution={execution_interval['lower']:.2%}-"
            f"{execution_interval['upper']:.2%}  "
            f"accepted_execution_precision={accepted_interval_text}"
        ),
        "",
        "Operation          n       op  program execution",
    ]
    for operation, scores in per_operation.items():
        if scores["support"] <= 0:
            continue
        lines.append(
            f"{operation:<15} {scores['support']:>4}  "
            f"{scores['operation_accuracy']:>7.2%}  "
            f"{scores['program_exact_accuracy']:>7.2%}  "
            f"{scores['execution_accuracy']:>8.2%}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hierarchical retrieval + neuro-symbolic QA tai chinh Viet Nam"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--document-model-path", type=Path, default=DEFAULT_DOCUMENT_HRM_PATH
    )
    parser.add_argument(
        "--program-reranker-path",
        type=Path,
        help=(
            "Optional program-reranker sidecar. The Document HRM checkpoint is "
            "loaded read-only and its weights are not modified."
        ),
    )
    parser.add_argument(
        "--typed-program-synthesis",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable the experimental V12 typed-slot program synthesizer. "
            "Disabled by default until its validation gate passes."
        ),
    )
    parser.add_argument(
        "--financial-identity-catalog",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable the experimental YAML financial-identity catalog. "
            "Disabled by default; use together with typed synthesis."
        ),
    )
    parser.add_argument(
        "--candidate-recall-v12-4",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable experimental V12.4 operation-conditioned candidate union "
            "and trace-complete fallback."
        ),
    )
    parser.add_argument(
        "--legacy-model-path", type=Path, default=DEFAULT_LEGACY_MODEL_PATH
    )
    parser.add_argument(
        "--engine",
        choices=("auto", "hrm", "legacy", "symbolic"),
        default="auto",
        help="Reasoning backend. auto prefers recurrent HRM, then legacy classifier.",
    )
    parser.add_argument(
        "--symbolic-only",
        action="store_true",
        help="Disable neural ranking and use deterministic symbolic retrieval",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)


    prepare_document_hrm = subparsers.add_parser(
        "prepare-document-hrm-data",
        help="Induce executable programs from the 1k financial QA corpus",
    )
    prepare_document_hrm.add_argument(
        "--source", type=Path, default=DEFAULT_HRM_INSTRUCTION_JSONL
    )
    prepare_document_hrm.add_argument(
        "--output", type=Path, default=DEFAULT_HRM_PROGRAM_DATASET
    )
    prepare_document_hrm.add_argument(
        "--synthetic-per-operation",
        type=int,
        default=25,
        help="Balanced synthetic examples per operation; defaults to 25 to avoid swamping real QA",
    )
    prepare_document_hrm.add_argument("--max-candidates", type=int, default=64)
    prepare_document_hrm.add_argument(
        "--audit-report",
        type=Path,
        help="Write source filtering and rejection counts as JSON",
    )
    prepare_document_hrm.add_argument(
        "--allowed-task",
        action="append",
        dest="allowed_tasks",
        help="Source task allowed for program induction; defaults to QA only",
    )

    train_document_hrm = subparsers.add_parser(
        "train-document-hrm",
        help="Train source-grounded H/L HRM to emit arithmetic programs",
    )
    train_document_hrm.add_argument(
        "--dataset", type=Path, default=DEFAULT_HRM_PROGRAM_DATASET
    )
    train_document_hrm.add_argument("--epochs", type=int, default=6)
    train_document_hrm.add_argument("--batch-size", type=int, default=32)
    train_document_hrm.add_argument("--learning-rate", type=float, default=0.0015)
    train_document_hrm.add_argument(
        "--seed",
        type=int,
        default=31,
        help="Seed for model initialization, data order and augmentation",
    )
    train_document_hrm.add_argument(
        "--backbone-learning-rate",
        type=float,
        help=(
            "Optional lower learning rate for the shared backbone during "
            "staged pair or unary-pointer fine-tuning"
        ),
    )
    train_document_hrm.add_argument("--vocab-size", type=int, default=16384)
    train_document_hrm.add_argument("--embedding-dim", type=int, default=64)
    train_document_hrm.add_argument("--hidden-size", type=int, default=96)
    train_document_hrm.add_argument("--high-cycles", type=int, default=1)
    train_document_hrm.add_argument("--low-steps", type=int, default=2)
    train_document_hrm.add_argument("--max-segments", type=int, default=3)
    train_document_hrm.add_argument("--max-candidates", type=int, default=64)
    train_document_hrm.add_argument("--attention-heads", type=int, default=4)
    train_document_hrm.add_argument("--ff-multiplier", type=int, default=3)
    train_document_hrm.add_argument("--validation-fraction", type=float, default=0.15)
    train_document_hrm.add_argument(
        "--synthetic-fraction",
        type=float,
        default=1.0,
        help="Fraction of available synthetic examples sampled per epoch",
    )
    train_document_hrm.add_argument("--relevance-prior-max", type=float, default=2.0)
    train_document_hrm.add_argument(
        "--act-mode", choices=("q_learning", "supervised"), default="q_learning"
    )
    train_document_hrm.add_argument("--minimum-halt-segments", type=int, default=2)
    train_document_hrm.add_argument(
        "--input-encoder", choices=("semantic", "hash"), default="semantic"
    )
    train_document_hrm.add_argument(
        "--semantic-model", default=DEFAULT_DENSE_MODEL
    )
    train_document_hrm.add_argument("--semantic-dim", type=int, default=384)
    train_document_hrm.add_argument(
        "--semantic-encoder-backend",
        choices=("fastembed", "sentence-transformers"),
        default="fastembed",
    )
    train_document_hrm.add_argument(
        "--semantic-model-revision",
        default="",
        help="Exact Hugging Face commit for non-legacy semantic encoders",
    )
    train_document_hrm.add_argument(
        "--semantic-local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the pinned encoder only from the local snapshot cache",
    )
    train_document_hrm.add_argument(
        "--semantic-encoder-dim",
        type=int,
        default=0,
        help="Raw frozen encoder width before the trainable HRM adapter",
    )
    train_document_hrm.add_argument(
        "--semantic-word-segmentation",
        choices=("none", "pyvi"),
        default="none",
    )
    train_document_hrm.add_argument(
        "--semantic-segmenter-version",
        default="",
    )
    train_document_hrm.add_argument(
        "--semantic-prefix-mode",
        choices=("none", "e5-v1"),
        default="none",
    )
    train_document_hrm.add_argument(
        "--semantic-query-encoding-mode",
        choices=("document", "query"),
        default="document",
    )
    train_document_hrm.add_argument(
        "--semantic-text-template",
        choices=("legacy-raw-v1", "finance-vi-v1"),
        default="legacy-raw-v1",
    )
    train_document_hrm.add_argument(
        "--semantic-adapter-mode",
        choices=("identity", "layernorm-linear-l2-v1"),
        default="identity",
    )
    train_document_hrm.add_argument(
        "--candidate-semantic-mode",
        choices=("context", "structured"),
        default="context",
        help="Encode full candidate context or only stable row/column/period fields",
    )
    train_document_hrm.add_argument(
        "--financial-structure-mode",
        choices=("legacy", "typed-hierarchy-v1"),
        default="legacy",
        help=(
            "Use the legacy flat candidate representation or the V13 typed "
            "financial hierarchy"
        ),
    )
    train_document_hrm.add_argument(
        "--financial-hierarchy-layers",
        type=int,
        default=1,
        help="Number of typed financial hierarchy refinement layers",
    )
    train_document_hrm.add_argument(
        "--variable-sum-mode",
        choices=("disabled", "sum-n-v1"),
        default="disabled",
        help=(
            "Keep legacy ADD/SUM3/SUM4 decoding or enable the V14 logical "
            "SUM_N count and unordered-membership heads"
        ),
    )
    train_document_hrm.add_argument(
        "--max-program-operands",
        type=int,
        default=4,
        help="Maximum pointer width; V14 SUM_N supports values from 4 through 8",
    )
    train_document_hrm.add_argument(
        "--program-search-strategy",
        choices=("legacy", "diverse-set-beam-v1"),
        default="legacy",
    )
    train_document_hrm.add_argument(
        "--pointer-conditioning",
        choices=(
            "independent",
            "operation-sequential",
            "later-sequential",
            "joint-pair",
        ),
        default="joint-pair",
        help="Choose independent, sequential, or relational joint-pair pointer scoring",
    )
    train_document_hrm.add_argument(
        "--pair-backbone-mode",
        choices=(
            "detached",
            "end-to-end",
            "normalized-detached",
            "normalized-end-to-end",
            "warmup-end-to-end",
        ),
        default="detached",
        help="Keep pair reranking isolated or train candidate/pointer representations end-to-end",
    )
    train_document_hrm.add_argument(
        "--pair-backbone-warmup-epochs",
        type=int,
        default=0,
        help="Keep pair gradients detached from the backbone for this many epochs",
    )
    train_document_hrm.add_argument(
        "--pair-loss-weight",
        type=float,
        default=1.0,
        help="Final joint-pair loss weight",
    )
    train_document_hrm.add_argument(
        "--pair-loss-warmup-epochs",
        type=int,
        default=0,
        help="Linearly ramp pair loss from zero over this many epochs",
    )
    train_document_hrm.add_argument(
        "--pair-targeted-negative-weight",
        type=float,
        default=0.0,
        help="Margin-loss weight for same-row/wrong-period and same-period/wrong-row pairs",
    )
    train_document_hrm.add_argument(
        "--pointer-teacher-forcing",
        type=float,
        default=0.50,
        help="Gold/predicted previous-pointer context blend used during training",
    )
    train_document_hrm.add_argument(
        "--pointer-hard-negative-weight",
        type=float,
        default=0.0,
        help="Margin-loss weight against the strongest wrong candidate per pointer slot",
    )
    train_document_hrm.add_argument(
        "--pointer-hard-negative-margin",
        type=float,
        default=0.50,
    )
    train_document_hrm.add_argument(
        "--pointer-targeted-negative-weight",
        type=float,
        default=0.0,
        help=(
            "Margin-loss weight for unary same-row/wrong-period and "
            "same-period/wrong-row candidates"
        ),
    )
    train_document_hrm.add_argument(
        "--verifier-guided-pair-weight",
        type=float,
        default=0.0,
        help=(
            "Margin-loss weight against only three-state verifier INVALID "
            "binary programs; UNKNOWN and SUM3/SUM4 are excluded"
        ),
    )
    train_document_hrm.add_argument(
        "--verifier-guided-unary-weight",
        type=float,
        default=0.0,
        help=(
            "Margin-loss weight against only verifier INVALID SELECT pointers"
        ),
    )
    train_document_hrm.add_argument(
        "--verifier-guided-margin",
        type=float,
        default=0.75,
    )
    train_document_hrm.add_argument(
        "--add-oversample-factor",
        type=float,
        default=1.0,
        help="Oversample ADD after the train/validation split; validation is unchanged",
    )
    train_document_hrm.add_argument(
        "--add-select-contrastive-weight",
        type=float,
        default=0.0,
        help="Margin-loss weight separating ADD from SELECT operation logits",
    )
    train_document_hrm.add_argument(
        "--add-select-contrastive-margin", type=float, default=0.75
    )
    train_document_hrm.add_argument(
        "--abstention-target-precision",
        type=float,
        default=0.80,
        help="Minimum validation precision required for abstention calibration",
    )
    train_document_hrm.add_argument(
        "--abstention-stability-errors",
        type=int,
        default=1,
        help="Extra validation errors tolerated by abstention calibration",
    )
    validation_source = train_document_hrm.add_mutually_exclusive_group()
    validation_source.add_argument(
        "--validation-ids-from",
        type=Path,
        help="Reuse validation_example_ids from an existing checkpoint",
    )
    validation_source.add_argument(
        "--validation-dataset",
        type=Path,
        help=(
            "Use a physically separate validation JSONL. Example IDs and "
            "company/report groups must be disjoint from --dataset"
        ),
    )
    train_document_hrm.add_argument(
        "--initialize-from",
        type=Path,
        help="Initialize from an architecture-compatible checkpoint",
    )
    train_document_hrm.add_argument(
        "--inherit-initial-config",
        action="store_true",
        help=(
            "Reuse the exact architecture/config stored in --initialize-from; "
            "avoids restating every checkpoint-shape option for fine-tuning"
        ),
    )
    train_document_hrm.add_argument(
        "--initialization-mode",
        choices=("exact", "compatible-backbone"),
        default="exact",
        help=(
            "Require an exact config match, or copy only state tensors with "
            "matching names and shapes into the requested architecture"
        ),
    )
    train_document_hrm.add_argument(
        "--overwrite-output",
        action="store_true",
        help=(
            "Allow replacing an existing non-frozen output checkpoint. The "
            "frozen V8 checkpoint and initialize_from are always protected"
        ),
    )
    train_document_hrm.add_argument(
        "--pair-head-only-epochs",
        type=int,
        default=0,
        help="Freeze operation/pointer/backbone and train only pair_* parameters",
    )
    train_document_hrm.add_argument(
        "--pointer-head-only-epochs",
        type=int,
        default=0,
        help="Train only unary pointer parameters for the first N epochs",
    )
    train_document_hrm.add_argument(
        "--pointer-heads-only",
        action="store_true",
        help=(
            "Freeze the shared backbone, operation and Q heads; train only "
            "unary pointer and joint-pair heads"
        ),
    )
    train_document_hrm.add_argument(
        "--retrieval-heads-only",
        action="store_true",
        help=(
            "Freeze operation/shared HRM and train the semantic adapter, "
            "typed financial, pointer, pair, SUM count and membership heads"
        ),
    )
    train_document_hrm.add_argument(
        "--freeze-operation-head",
        action="store_true",
        help="Keep the initialized operation classifier fixed during all stages",
    )
    train_document_hrm.add_argument(
        "--semantic-contrastive-weight",
        type=float,
        default=0.0,
    )
    train_document_hrm.add_argument(
        "--semantic-contrastive-temperature",
        type=float,
        default=0.07,
    )
    train_document_hrm.add_argument(
        "--semantic-contrastive-margin",
        type=float,
        default=0.20,
    )
    train_document_hrm.add_argument(
        "--freeze-pair-head",
        action="store_true",
        help="Keep the initialized joint-pair reranker fixed during all stages",
    )
    train_document_hrm.add_argument(
        "--preserve-initial-threshold",
        action="store_true",
        help="Do not calibrate below initialize_from's abstention threshold",
    )
    train_document_hrm.add_argument(
        "--select-pointer-only",
        action="store_true",
        help=(
            "Train only the operation-specific SELECT residual in the existing "
            "joint-pair bilinear tensor"
        ),
    )
    train_document_hrm.add_argument(
        "--select-best-epoch",
        action="store_true",
        help="Restore the epoch with the best development-validation program score",
    )
    train_document_hrm.add_argument(
        "--best-training-state",
        type=Path,
        help="Atomically update this resumable state on strict best-epoch improvement",
    )
    train_document_hrm.add_argument(
        "--last-training-state",
        type=Path,
        help="Atomically update this resumable state after every completed epoch",
    )
    train_document_hrm.add_argument(
        "--resume-training-state",
        type=Path,
        help=(
            "Resume model, optimizer, best selection and RNG from a locked "
            "last/best training-state artifact"
        ),
    )


    train_document_hrm.add_argument(
        "--legacy-hl-inputs",
        action="store_true",
        help="Use the old mode where H directly sees question and candidates",
    )
    train_document_hrm.add_argument("--device", default="cpu")

    evaluate_document_hrm_parser = subparsers.add_parser(
        "evaluate-document-hrm", help="Evaluate operation/program/answer accuracy"
    )
    evaluate_document_hrm_parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_HRM_PROGRAM_DATASET
    )
    evaluate_document_hrm_parser.add_argument(
        "--all-examples",
        action="store_true",
        help="Evaluate the entire dataset, including training examples",
    )
    evaluate_document_hrm_parser.add_argument("--device", default="cpu")
    evaluate_document_hrm_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete verbose report as JSON",
    )
    evaluate_document_hrm_parser.add_argument(
        "--output-report",
        type=Path,
        help="Write the complete JSON report to this path",
    )
    evaluate_document_hrm_parser.add_argument(
        "--adaptive-inference",
        action="store_true",
        help="Use the serving-time ACT halt policy instead of fixed max segments",
    )

    evaluate_external_parser = subparsers.add_parser(
        "evaluate-document-hrm-external",
        help="Audit leakage, then evaluate a company/report-disjoint test set",
    )
    evaluate_external_parser.add_argument(
        "--dataset", type=Path, required=True,
        help="Real-only external ProgramExample JSONL",
    )
    evaluate_external_parser.add_argument(
        "--reference-dataset",
        type=Path,
        action="append",
        required=True,
        help=(
            "Train/dev/test JSONL used before freezing this external set; "
            "repeat the option to audit against every prior dataset"
        ),
    )
    evaluate_external_parser.add_argument("--device", default="cpu")
    evaluate_external_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete verbose report as JSON",
    )
    evaluate_external_parser.add_argument(
        "--output-report", type=Path,
        help="Write metrics and the leakage audit as JSON",
    )
    evaluate_external_parser.add_argument(
        "--audit-report", type=Path,
        help="Write the standalone integrity/leakage audit as JSON",
    )
    evaluate_external_parser.add_argument(
        "--adaptive-inference",
        action="store_true",
        help="Use the serving-time ACT halt policy instead of fixed max segments",
    )

    diagnose_document_hrm_parser = subparsers.add_parser(
        "diagnose-document-hrm",
        help="Show pointer, ACT, coverage and execution bottlenecks",
    )
    diagnose_document_hrm_parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_HRM_PROGRAM_DATASET
    )
    diagnose_document_hrm_parser.add_argument(
        "--all-examples", action="store_true"
    )
    diagnose_document_hrm_parser.add_argument(
        "--limit", type=int, default=24,
        help="Number of examples for a quick diagnosis; use 0 for the full set",
    )
    diagnose_document_hrm_parser.add_argument("--device", default="cpu")

    analyze_document_hrm_parser = subparsers.add_parser(
        "analyze-document-hrm-errors",
        help="Export per-example Document HRM errors and pointer diagnostics",
    )
    analyze_document_hrm_parser.add_argument(
        "--dataset", type=Path, default=DEFAULT_HRM_PROGRAM_DATASET
    )
    analyze_document_hrm_parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "document_hrm_errors.jsonl",
    )
    analyze_document_hrm_parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "reports" / "document_hrm_error_summary.json",
    )
    analyze_document_hrm_parser.add_argument(
        "--all-examples",
        action="store_true",
        help="Analyze the entire dataset instead of checkpoint holdout IDs",
    )
    analyze_document_hrm_parser.add_argument(
        "--include-correct",
        action="store_true",
        help="Include exact-correct examples in the JSONL output",
    )
    analyze_document_hrm_parser.add_argument(
        "--limit", type=int, default=0, help="Analyze only the first N examples"
    )
    analyze_document_hrm_parser.add_argument(
        "--operation",
        action="append",
        choices=OPERATIONS,
        help="Analyze only the selected gold operation; repeat for multiple operations",
    )
    analyze_document_hrm_parser.add_argument("--batch-size", type=int, default=32)
    analyze_document_hrm_parser.add_argument("--device", default="cpu")

    document_hrm_info = subparsers.add_parser(
        "document-hrm-info", help="Show source-grounded Adaptive-HRM architecture"
    )
    document_hrm_info.add_argument("--device", default="cpu")

    prepare_document = subparsers.add_parser(
        "prepare-document",
        help="Run MinerU OCR once and save a page-marked Markdown artifact",
    )
    prepare_document.add_argument("--document", type=Path, required=True)
    prepare_document.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "mineru_markdown",
        help="Output .md file or directory",
    )
    prepare_document.add_argument("--mineru-output-dir", type=Path)
    prepare_document.add_argument(
        "--mineru-cache-dir",
        type=Path,
        default=ROOT / "cache" / "mineru",
    )
    prepare_document.add_argument(
        "--mineru-command",
        type=Path,
        default=ROOT / ".venv-mineru" / "bin" / "mineru",
    )
    prepare_document.add_argument("--mineru-api-url")
    prepare_document.add_argument(
        "--mineru-api-timeout",
        type=float,
        default=7200.0,
    )
    prepare_document.add_argument(
        "--mineru-backend",
        choices=("hybrid-engine", "pipeline"),
        default="hybrid-engine",
    )
    prepare_document.add_argument(
        "--mineru-effort",
        choices=("low", "medium", "high"),
        default="medium",
    )
    prepare_document.add_argument(
        "--no-mineru-auto",
        action="store_false",
        dest="mineru_auto_generate",
        default=True,
    )

    answer_document = subparsers.add_parser(
        "answer-document",
        help="Answer from a prepared MinerU Markdown artifact",
    )
    answer_document.add_argument("question")
    answer_document.add_argument(
        "--document", type=Path, required=True,
        help="A MinerU .md file or directory containing prepared Markdown",
    )
    answer_document.add_argument("--top-k-chunks", type=int, default=8)
    answer_document.add_argument("--max-candidates", type=int, default=64)
    answer_document.add_argument(
        "--neighbor-chunks", type=int, default=1,
        help="Include adjacent chunks around top retrieval hits",
    )
    answer_document.add_argument(
        "--ocr",
        action="store_true",
        default=True,
        help="MinerU configuration; prepare PDF to Markdown before answering",
    )
    answer_document.add_argument(
        "--no-ocr",
        action="store_false",
        dest="ocr",
        help="Use already prepared Markdown/TXT without OCR generation",
    )
    answer_document.add_argument(
        "--force-ocr",
        action="store_true",
        help="Preparation compatibility flag; PDF content always comes from MinerU",
    )
    answer_document.add_argument(
        "--mineru-output-dir",
        type=Path,
        help="Existing MinerU output root or exact *_content_list.json artifact",
    )
    answer_document.add_argument(
        "--mineru-page-offset",
        type=int,
        default=0,
        help="Add this offset to MinerU's zero-based page_idx for partial PDF runs",
    )
    answer_document.add_argument(
        "--mineru-cache-dir",
        type=Path,
        default=ROOT / "cache" / "mineru",
        help="Cache auto-generated fingerprinted MinerU artifacts",
    )
    answer_document.add_argument(
        "--mineru-command",
        type=Path,
        default=ROOT / ".venv-mineru" / "bin" / "mineru",
        help="MinerU executable from the isolated environment",
    )
    answer_document.add_argument(
        "--mineru-api-url",
        help="Use a long-lived MinerU API instead of launching a CLI subprocess",
    )
    answer_document.add_argument(
        "--mineru-api-timeout",
        type=float,
        default=7200.0,
        help="MinerU API request timeout in seconds",
    )
    answer_document.add_argument(
        "--mineru-backend",
        choices=("hybrid-engine", "pipeline"),
        default="hybrid-engine",
    )
    answer_document.add_argument(
        "--mineru-effort",
        choices=("low", "medium", "high"),
        default="medium",
    )
    answer_document.add_argument(
        "--no-mineru-auto",
        action="store_false",
        dest="mineru_auto_generate",
        default=True,
        help="Require --mineru-output-dir instead of launching MinerU automatically",
    )
    answer_document.add_argument(
        "--page-index",
        action="store_true",
        default=True,
        help="Compatibility option for a prepared MinerU page index",
    )
    answer_document.add_argument(
        "--no-page-index",
        action="store_false",
        dest="page_index",
        help="Read the prepared Markdown/TXT document directly",
    )
    answer_document.add_argument(
        "--page-index-path",
        type=Path,
        default=ROOT / "cache" / "document_page_index.sqlite3",
        help="Persistent SQLite page index",
    )
    answer_document.add_argument(
        "--rebuild-page-index",
        action="store_true",
        help="Rebuild indexed pages even when the source fingerprint is unchanged",
    )
    answer_document.add_argument(
        "--page-top-k",
        type=int,
        default=6,
        help="Number of pages ranked before page-neighbor expansion",
    )
    answer_document.add_argument(
        "--page-neighbor-pages",
        type=int,
        default=1,
        help="Include this many adjacent pages around each page hit",
    )
    answer_document.add_argument("--device", default="cpu")

    subparsers.add_parser("analyze-data", help="Thong ke cau truc bo CSV")
    subparsers.add_parser("metrics", help="Liet ke chi so duoc ho tro")

    train_hrm = subparsers.add_parser(
        "train-hrm", help="Train the recurrent H/L Hierarchical Reasoning Model"
    )
    train_hrm.add_argument("--epochs", type=int, default=20)
    train_hrm.add_argument("--learning-rate", type=float, default=0.002)
    train_hrm.add_argument("--batch-size", type=int, default=64)
    train_hrm.add_argument("--vocab-size", type=int, default=8192)
    train_hrm.add_argument("--embedding-dim", type=int, default=96)
    train_hrm.add_argument("--hidden-size", type=int, default=128)
    train_hrm.add_argument("--high-cycles", type=int, default=2)
    train_hrm.add_argument("--low-steps", type=int, default=3)
    train_hrm.add_argument("--max-segments", type=int, default=4)
    train_hrm.add_argument("--max-tickers", type=int, default=160)
    train_hrm.add_argument("--device", default="cpu")

    evaluate_hrm = subparsers.add_parser(
        "evaluate-hrm", help="Evaluate recurrent HRM heads"
    )
    evaluate_hrm.add_argument("--max-tickers", type=int, default=100)
    evaluate_hrm.add_argument("--device", default="cpu")
    hrm_info = subparsers.add_parser("hrm-info", help="Show recurrent HRM architecture")
    hrm_info.add_argument("--device", default="cpu")

    train = subparsers.add_parser("train-neural", help="Train legacy hashed-softmax bundle")
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--feature-dim", type=int, default=4096)
    train.add_argument("--learning-rate", type=float, default=0.08)
    train.add_argument("--max-tickers", type=int, default=160)

    evaluate = subparsers.add_parser("evaluate-neural", help="Evaluate legacy neural heads")
    evaluate.add_argument("--max-tickers", type=int, default=100)

    subparsers.add_parser("model-info", help="Show legacy classifier information")

    answer = subparsers.add_parser("answer", help="Tra loi cau hoi tai chinh")
    answer.add_argument("question", nargs="?", default="")
    answer.add_argument("--ticker")
    answer.add_argument("--year", type=int)
    answer.add_argument("--metric")

    validate = subparsers.add_parser("validate", help="Kiem tra rang buoc ke toan")
    validate.add_argument("--ticker", required=True)
    validate.add_argument("--year", required=True, type=int)
    validate.add_argument("--tolerance", type=Decimal, default=Decimal("0.01"))

    what_if = subparsers.add_parser("what-if", help="Tinh kich ban thay doi bien dau vao")
    what_if.add_argument("question", nargs="?", default="")
    what_if.add_argument("--ticker")
    what_if.add_argument("--year", type=int)
    what_if.add_argument("--metric")
    what_if.add_argument(
        "--set",
        dest="adjustments",
        action="append",
        required=True,
        help="Vi du: current_assets=+10%% (co the dung nhieu --set)",
    )

    build_index = subparsers.add_parser(
        "build-document-index",
        help="Lap chi muc truc tiep tu cac file text da trich xuat",
    )
    build_index.add_argument("--text-dir", type=Path, default=DEFAULT_DOCUMENT_TEXT_DIR)
    build_index.add_argument(
        "--output", type=Path, default=DEFAULT_DOCUMENT_INDEX_PATH
    )
    build_index.add_argument("--chunk-lines", type=int, default=80)
    build_index.add_argument("--overlap-lines", type=int, default=15)

    build_dense = subparsers.add_parser(
        "build-dense-index",
        help="Tao pretrained embedding index cho table/chunk retrieval",
    )
    build_dense.add_argument(
        "--index", type=Path, default=DEFAULT_DOCUMENT_INDEX_PATH
    )
    build_dense.add_argument(
        "--output", type=Path, default=DEFAULT_DENSE_INDEX_PATH
    )
    build_dense.add_argument("--model", default=DEFAULT_DENSE_MODEL)
    build_dense.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_EMBEDDING_CACHE_DIR
    )
    build_dense.add_argument("--batch-size", type=int, default=32)

    search_documents = subparsers.add_parser(
        "search-documents",
        help="Truy hoi bang chung nhieu vong trong bao cao tai chinh",
    )
    search_documents.add_argument("question")
    search_documents.add_argument(
        "--index", type=Path, default=DEFAULT_DOCUMENT_INDEX_PATH
    )
    search_documents.add_argument("--document-id")
    search_documents.add_argument("--top-k", type=int, default=15)
    search_documents.add_argument("--max-rounds", type=int, default=5)
    search_documents.add_argument("--show-hits", type=int, default=10)
    search_documents.add_argument(
        "--dense-index", type=Path, default=DEFAULT_DENSE_INDEX_PATH
    )
    search_documents.add_argument(
        "--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE_DIR
    )
    search_documents.add_argument(
        "--sparse-only",
        action="store_true",
        help="Chi dung BM25/phrase, bo qua pretrained embedding",
    )
    search_documents.add_argument(
        "--exact",
        action="store_true",
        help="Chi tim cum tu chinh xac, khong mo rong cong thuc",
    )

    list_documents = subparsers.add_parser(
        "document-list", help="Liet ke tai lieu trong document index"
    )
    list_documents.add_argument(
        "--index", type=Path, default=DEFAULT_DOCUMENT_INDEX_PATH
    )
    list_documents.add_argument("--filter", default="")
    list_documents.add_argument("--limit", type=int, default=100)

    outline = subparsers.add_parser(
        "document-outline", help="Xem outline section cua mot tai lieu"
    )
    outline.add_argument("--index", type=Path, default=DEFAULT_DOCUMENT_INDEX_PATH)
    outline.add_argument("--document-id", required=True)
    outline.add_argument("--limit", type=int, default=80)

    section = subparsers.add_parser(
        "document-section", help="Doc toan bo mot section kem so dong"
    )
    section.add_argument("--index", type=Path, default=DEFAULT_DOCUMENT_INDEX_PATH)
    section.add_argument("--document-id", required=True)
    section.add_argument("--section-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-document-hrm-data":
            real_output = args.output.with_name(args.output.stem + "_real.jsonl")
            allowed_tasks = tuple(args.allowed_tasks or ("QA",))
            report = build_program_dataset(
                args.source,
                real_output,
                max_candidates=args.max_candidates,
                allowed_tasks=allowed_tasks,
            )
            real_examples = load_program_examples(real_output)
            synthetic = generate_synthetic_program_examples(
                count_per_operation=args.synthetic_per_operation,
                max_candidates=args.max_candidates,
            )
            destination = write_program_examples(
                [*real_examples, *synthetic], args.output
            )
            audit_output = args.audit_report or args.output.with_name(
                args.output.stem + "_audit.json"
            )
            audit_output = audit_output.expanduser().resolve()
            audit_output.parent.mkdir(parents=True, exist_ok=True)
            audit_payload = {
                **asdict(report),
                "allowed_tasks": list(allowed_tasks),
                "max_candidates": args.max_candidates,
                "synthetic_per_operation": args.synthetic_per_operation,
                "synthetic_question_schema": "v2-financial-paraphrases",
                "real_output": str(real_output.resolve()),
                "combined_output": str(destination),
                "synthetic_examples": len(synthetic),
                "total_examples": len(real_examples) + len(synthetic),
            }
            audit_output.write_text(
                json.dumps(audit_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved program dataset: {destination}")
            print(f"Saved real-only dataset: {real_output.resolve()}")
            print(f"Saved data audit: {audit_output}")
            print(
                f"Source rows={report.source_rows}, eligible rows={report.eligible_rows}, "
                f"real programs={len(real_examples)}, "
                f"synthetic programs={len(synthetic)}, total={len(real_examples) + len(synthetic)}"
            )
            print("Source task counts:")
            for task, count in report.source_task_counts.items():
                print(f"- {task}: {count}")
            print("Rejection counts:")
            for reason, count in report.rejection_counts.items():
                print(f"- {reason}: {count}")
            print("Real operation counts:")
            for operation, count in sorted(report.operation_counts.items()):
                print(f"- {operation}: {count}")
            return 0

        if args.command == "train-document-hrm":
            output_path = _validate_document_hrm_training_output(
                args.document_model_path,
                initialize_from=args.initialize_from,
                overwrite=args.overwrite_output,
            )
            state_outputs = tuple(
                Path(path).expanduser().resolve()
                for path in (
                    args.best_training_state,
                    args.last_training_state,
                )
                if path is not None
            )
            if output_path in state_outputs:
                raise DocumentHRMError(
                    "Final model output and training-state output must differ"
                )
            if len(state_outputs) != len(set(state_outputs)):
                raise DocumentHRMError(
                    "Best and last training-state outputs must differ"
                )

            if args.inherit_initial_config and args.initialize_from is None:
                raise DocumentHRMError(
                    "--inherit-initial-config requires --initialize-from"
                )
            if (
                args.inherit_initial_config
                and args.initialization_mode != "exact"
            ):
                raise DocumentHRMError(
                    "--inherit-initial-config requires --initialization-mode exact"
                )
            validation_ids: set[str] | None = None
            allow_synthetic_validation = False
            if args.validation_ids_from:
                reference = DocumentHRMBundle.load(args.validation_ids_from)
                raw_ids = reference.metadata.get("validation_example_ids", [])
                if not isinstance(raw_ids, list) or not raw_ids:
                    raise DocumentHRMError(
                        "Reference checkpoint has no validation_example_ids"
                    )
                validation_ids = {str(item) for item in raw_ids}
            if args.inherit_initial_config:
                initial = DocumentHRMBundle.load(
                    args.initialize_from, device=args.device
                )
                config = replace(initial.model.config, seed=args.seed)
            else:
                config = DocumentHRMConfig(
                    vocab_size=args.vocab_size,
                    embedding_dim=args.embedding_dim,
                    hidden_size=args.hidden_size,
                    high_cycles=args.high_cycles,
                    low_steps=args.low_steps,
                    max_segments=args.max_segments,
                    max_candidates=args.max_candidates,
                    attention_heads=args.attention_heads,
                    feed_forward_multiplier=args.ff_multiplier,
                    hierarchical_input_mode=(
                        "legacy" if args.legacy_hl_inputs else "strict"
                    ),
                    relevance_prior_max=args.relevance_prior_max,
                    act_mode=args.act_mode,
                    minimum_halt_segments=args.minimum_halt_segments,
                    input_encoder_mode=args.input_encoder,
                    semantic_model_name=args.semantic_model,
                    semantic_embedding_size=args.semantic_dim,
                    semantic_encoder_backend=(
                        args.semantic_encoder_backend.replace("-", "_")
                    ),
                    semantic_model_revision=args.semantic_model_revision,
                    semantic_local_files_only=(
                        args.semantic_local_files_only
                    ),
                    semantic_encoder_output_size=(
                        args.semantic_encoder_dim
                    ),
                    semantic_word_segmentation=(
                        args.semantic_word_segmentation
                    ),
                    semantic_segmenter_version=(
                        args.semantic_segmenter_version
                    ),
                    semantic_prefix_mode=(
                        args.semantic_prefix_mode.replace("-", "_")
                    ),
                    semantic_query_encoding_mode=(
                        args.semantic_query_encoding_mode
                    ),
                    semantic_text_template_version=(
                        args.semantic_text_template.replace("-", "_")
                    ),
                    semantic_adapter_mode=(
                        args.semantic_adapter_mode.replace("-", "_")
                    ),
                    candidate_semantic_mode=args.candidate_semantic_mode,
                    financial_structure_mode=(
                        args.financial_structure_mode.replace("-", "_")
                    ),
                    financial_hierarchy_layers=args.financial_hierarchy_layers,
                    variable_sum_mode=args.variable_sum_mode.replace("-", "_"),
                    max_program_operands=args.max_program_operands,
                    program_search_strategy=(
                        args.program_search_strategy.replace("-", "_")
                    ),
                    pointer_conditioning_mode=(
                        args.pointer_conditioning.replace("-", "_")
                    ),
                    pair_backbone_mode=args.pair_backbone_mode.replace("-", "_"),
                    pair_backbone_warmup_epochs=args.pair_backbone_warmup_epochs,
                    pair_loss_weight=args.pair_loss_weight,
                    pair_loss_warmup_epochs=args.pair_loss_warmup_epochs,
                    pair_targeted_negative_weight=(
                        args.pair_targeted_negative_weight
                    ),
                    pointer_teacher_forcing=args.pointer_teacher_forcing,
                    pointer_hard_negative_weight=args.pointer_hard_negative_weight,
                    pointer_hard_negative_margin=args.pointer_hard_negative_margin,
                    seed=args.seed,
                )
            bundle, report = train_document_hrm(
                args.dataset,
                config=config,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                validation_fraction=args.validation_fraction,
                validation_example_ids=validation_ids,
                validation_dataset_path=args.validation_dataset,
                allow_synthetic_validation=allow_synthetic_validation,
                synthetic_fraction=args.synthetic_fraction,
                select_best_epoch=args.select_best_epoch,
                best_training_state_path=args.best_training_state,
                last_training_state_path=args.last_training_state,
                resume_training_state_path=args.resume_training_state,
                initialize_from=args.initialize_from,
                initialization_mode=(
                    args.initialization_mode.replace("-", "_")
                ),
                pointer_heads_only=args.pointer_heads_only,
                retrieval_heads_only=args.retrieval_heads_only,
                pair_head_only_epochs=args.pair_head_only_epochs,
                pointer_head_only_epochs=args.pointer_head_only_epochs,
                pointer_targeted_negative_weight=(
                    args.pointer_targeted_negative_weight
                ),
                verifier_guided_pair_weight=(
                    args.verifier_guided_pair_weight
                ),
                verifier_guided_unary_weight=(
                    args.verifier_guided_unary_weight
                ),
                verifier_guided_margin=args.verifier_guided_margin,
                add_oversample_factor=args.add_oversample_factor,
                add_select_contrastive_weight=args.add_select_contrastive_weight,
                add_select_contrastive_margin=args.add_select_contrastive_margin,
                semantic_contrastive_weight=(
                    args.semantic_contrastive_weight
                ),
                semantic_contrastive_temperature=(
                    args.semantic_contrastive_temperature
                ),
                semantic_contrastive_margin=(
                    args.semantic_contrastive_margin
                ),
                backbone_learning_rate=args.backbone_learning_rate,
                freeze_operation_head=args.freeze_operation_head,
                freeze_pair_head=args.freeze_pair_head,
                preserve_initial_threshold=args.preserve_initial_threshold,
                select_pointer_only=args.select_pointer_only,
                abstention_target_precision=args.abstention_target_precision,
                abstention_stability_errors=args.abstention_stability_errors,
                device=args.device,
            )
            destination = bundle.save(
                output_path, overwrite=args.overwrite_output
            )
            print(f"Saved Document HRM: {destination}")
            print(f"Trainable parameters: {bundle.parameter_count:,}")
            print(
                f"Train={report.train_examples}, validation={report.validation_examples}, "
                f"epochs={report.epochs}, loss={report.final_loss:.5f}"
            )
            print(f"Operation accuracy: {report.operation_accuracy:.2%}")
            print(
                "Operation macro P/R/F1: "
                f"{report.operation_macro_precision:.2%} / "
                f"{report.operation_macro_recall:.2%} / "
                f"{report.operation_macro_f1:.2%}"
            )
            print(
                "Operation weighted P/R/F1: "
                f"{report.operation_weighted_precision:.2%} / "
                f"{report.operation_weighted_recall:.2%} / "
                f"{report.operation_weighted_f1:.2%}"
            )
            print(f"Program exact accuracy: {report.program_exact_accuracy:.2%}")
            print(f"Execution accuracy: {report.execution_accuracy:.2%}")
            print(f"Pointer pair accuracy: {report.pointer_pair_accuracy:.2%}")
            slot_support = (
                getattr(report, "pointer_slot_support", ())
                or tuple(1 for _ in report.pointer_slot_accuracy)
            )
            slot_summary = ", ".join(
                (
                    f"p{index + 1}=N/A (n=0)"
                    if support <= 0
                    else f"p{index + 1}={value:.2%} (n={support})"
                )
                for index, (value, support) in enumerate(
                    zip(report.pointer_slot_accuracy, slot_support)
                )
            )
            print("Pointer slot accuracy: " + slot_summary)
            accepted_precision_text = (
                f"{report.accepted_precision:.2%}"
                if getattr(report, "accepted_precision_defined", True)
                else "N/A (accepted=0)"
            )
            print(
                f"Accepted coverage/precision: {report.accepted_coverage:.2%} / "
                f"{accepted_precision_text}"
            )
            return 0

        if args.command in {
            "evaluate-document-hrm",
            "evaluate-document-hrm-external",
            "diagnose-document-hrm",
            "document-hrm-info",
        }:
            bundle = _load_document_hrm_runtime(args)
            if args.command == "document-hrm-info":
                print(f"Model: {args.document_model_path.resolve()}")
                print(document_hrm_summary(bundle))
            else:
                external_audit = None
                audit_output = None
                if args.command == "evaluate-document-hrm-external":
                    examples = load_program_examples(args.dataset)
                    reference_examples = [
                        example
                        for reference_path in args.reference_dataset
                        for example in load_program_examples(reference_path)
                    ]
                    external_audit = audit_external_test_set(
                        examples, reference_examples
                    )
                    external_audit["reference_datasets"] = [
                        str(path.expanduser().resolve())
                        for path in args.reference_dataset
                    ]
                    audit_output = args.audit_report or args.dataset.with_name(
                        args.dataset.stem + "_audit.json"
                    )
                    audit_output = audit_output.expanduser().resolve()
                    audit_output.parent.mkdir(parents=True, exist_ok=True)
                    audit_output.write_text(
                        json.dumps(external_audit, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    if not external_audit["passed"]:
                        if args.json:
                            print(json.dumps(external_audit, ensure_ascii=False, indent=2))
                        raise DocumentHRMError(
                            "External test audit failed; metrics were not computed. "
                            f"See {audit_output}"
                        )
                    mode = "external company/report-disjoint test"
                else:
                    examples = (
                        load_program_examples(args.dataset)
                        if args.all_examples
                        else checkpoint_validation_examples(bundle, args.dataset)
                    )
                    mode = "all examples" if args.all_examples else "checkpoint real holdout"
                if args.command == "diagnose-document-hrm" and args.limit > 0:
                    examples = examples[: args.limit]
                    mode += " diagnostic sample"
                adaptive_inference = bool(
                    getattr(args, "adaptive_inference", False)
                )
                metrics = evaluate_document_hrm(
                    bundle,
                    examples,
                    adaptive_inference=adaptive_inference,
                )
                evaluation_report = _document_hrm_evaluation_report(
                    bundle,
                    metrics,
                    checkpoint=args.document_model_path,
                    dataset=args.dataset,
                    mode=mode,
                    example_count=len(examples),
                    adaptive_inference=adaptive_inference,
                )
                if external_audit is not None:
                    evaluation_report["external_test_audit"] = external_audit
                output_report = getattr(args, "output_report", None)
                if output_report is not None:
                    output_report = output_report.expanduser().resolve()
                    output_report.parent.mkdir(parents=True, exist_ok=True)
                    output_report.write_text(
                        json.dumps(evaluation_report, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                if getattr(args, "json", False):
                    print(json.dumps(evaluation_report, ensure_ascii=False, indent=2))
                    return 0
                print(_format_document_hrm_evaluation_summary(evaluation_report))
                if output_report is not None:
                    print(f"Saved JSON report: {output_report}")
                if audit_output is not None:
                    print(f"Saved external-test audit: {audit_output}")
                if args.command == "diagnose-document-hrm":
                    print("Diagnosis:")
                    if metrics["operation_accuracy"] > metrics["pointer_pair_accuracy"] + 0.15:
                        print("- Main bottleneck: operand pointer selection, not operation classification.")
                    if metrics["accepted_coverage"] == 0:
                        print("- Abstention calibration accepts no examples; retrain/recalibrate.")
                    if metrics["execution_accuracy"] < 0.40:
                        print("- Do not increase parameters yet; improve real data and candidate retrieval first.")
            return 0

        if args.command == "prepare-document":
            destination = materialize_mineru_markdown(
                args.document,
                args.output,
                ocr_config=OCRConfig(
                    enabled=True,
                    force=True,
                    mineru_output_dir=args.mineru_output_dir,
                    mineru_cache_dir=args.mineru_cache_dir,
                    mineru_command=args.mineru_command,
                    mineru_api_url=args.mineru_api_url,
                    mineru_api_timeout_seconds=args.mineru_api_timeout,
                    mineru_backend=args.mineru_backend,
                    mineru_effort=args.mineru_effort,
                    mineru_auto_generate=args.mineru_auto_generate,
                ),
            )
            print(f"Saved MinerU Markdown: {destination}")
            return 0

        if args.command == "answer-document":
            bundle = _load_document_hrm_runtime(args)
            answer = DocumentHRMReasoningEngine(bundle).answer_request(
                args.question,
                args.document,
                top_k_chunks=args.top_k_chunks,
                max_candidates=args.max_candidates,
                neighbor_chunks=args.neighbor_chunks,
                ocr=args.ocr,
                force_ocr=args.force_ocr,
                mineru_output_dir=args.mineru_output_dir,
                mineru_page_offset=args.mineru_page_offset,
                mineru_cache_dir=args.mineru_cache_dir,
                mineru_command=args.mineru_command,
                mineru_api_url=args.mineru_api_url,
                mineru_api_timeout_seconds=args.mineru_api_timeout,
                mineru_backend=args.mineru_backend,
                mineru_effort=args.mineru_effort,
                mineru_auto_generate=args.mineru_auto_generate,
                use_page_index=args.page_index,
                page_index_path=args.page_index_path,
                rebuild_page_index=args.rebuild_page_index,
                page_top_k=args.page_top_k,
                page_neighbor_pages=args.page_neighbor_pages,
            )
            print(answer.render_verified())
            return 0

        if args.command == "analyze-document-hrm-errors":
            bundle = _load_document_hrm_runtime(args)
            examples = (
                load_program_examples(args.dataset)
                if args.all_examples
                else checkpoint_validation_examples(bundle, args.dataset)
            )
            if args.operation:
                requested_operations = set(args.operation)
                examples = [
                    example
                    for example in examples
                    if example.program.operation in requested_operations
                ]
            if args.limit > 0:
                examples = examples[: args.limit]
            summary, records = analyze_document_hrm_errors(
                bundle,
                examples,
                batch_size=args.batch_size,
                include_correct=args.include_correct,
            )
            output_path = args.output.expanduser().resolve()
            summary_path = args.summary.expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8", newline="\n") as handle:
                for record in records:
                    handle.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            metrics = summary["metrics"]
            categories = summary["error_categories"]
            candidate_diagnostics = summary["candidate_diagnostics"]
            print(f"Analyzed examples: {summary['examples']}")
            print(f"Error records written: {summary['records_written']}")
            print(
                "Operation / pointer pair / execution: "
                f"{metrics['operation_accuracy']:.2%} / "
                f"{metrics['pointer_pair_accuracy']:.2%} / "
                f"{metrics['execution_accuracy']:.2%}"
            )
            print(
                "Error categories: "
                + ", ".join(
                    f"{name}={count}" for name, count in categories.items()
                )
            )
            print(
                "Stored gold operand coverage: "
                f"{candidate_diagnostics['stored_gold_operand_coverage']:.2%} "
                "(dataset integrity only; not end-to-end retrieval recall)"
            )
            print(f"Saved JSONL: {output_path}")
            print(f"Saved summary: {summary_path}")
            return 0

        if args.command == "build-document-index":
            index = VietnameseFinancialDocumentIndex.build_from_text_dir(
                args.text_dir,
                chunk_lines=args.chunk_lines,
                overlap_lines=args.overlap_lines,
            )
            destination = index.save(args.output)
            _print_document_index_stats(index, destination)
            return 0

        if args.command == "build-dense-index":
            sparse_index = VietnameseFinancialDocumentIndex.load(args.index)
            dense_index = DenseEmbeddingIndex.build(
                sparse_index,
                model_name=args.model,
                cache_dir=args.cache_dir,
                batch_size=args.batch_size,
            )
            destination = dense_index.save(args.output)
            print(f"Saved dense index: {destination}")
            print(f"Embedding model: {dense_index.model_name}")
            print(f"Chunks: {len(dense_index.chunk_ids):,}")
            print(f"Dimensions: {dense_index.embeddings.shape[1]:,}")
            return 0

        if args.command in {
            "search-documents",
            "document-list",
            "document-outline",
            "document-section",
        }:
            index = VietnameseFinancialDocumentIndex.load(args.index)
            if args.command == "search-documents":
                if args.exact:
                    hits = index.exact_search(
                        args.question,
                        document_id=args.document_id,
                        top_k=args.top_k,
                    )
                    print(f"Exact matches: {len(hits)}")
                    _print_document_hits(hits, args.show_hits)
                else:
                    search_index = index
                    hybrid_index = None
                    if not args.sparse_only and args.dense_index.is_file():
                        dense_index = DenseEmbeddingIndex.load(
                            args.dense_index,
                            cache_dir=args.embedding_cache,
                        )
                        hybrid_index = HybridFinancialDocumentIndex(
                            index, dense_index
                        )
                        search_index = hybrid_index
                    elif not args.sparse_only:
                        print(
                            f"Dense index chua co: {args.dense_index}; "
                            "fallback=BM25"
                        )
                    result = DocumentEvidenceRetriever(search_index).retrieve(
                        args.question,
                        document_id=args.document_id,
                        top_k=args.top_k,
                        max_rounds=args.max_rounds,
                    )
                    _print_document_evidence(result, args.show_hits)
                    if hybrid_index and hybrid_index.last_warning:
                        print(
                            f"Dense retrieval warning: {hybrid_index.last_warning}; "
                            "fallback=BM25"
                        )
            elif args.command == "document-list":
                _print_document_list(index, args.filter, args.limit)
            elif args.command == "document-outline":
                _print_document_outline(index, args.document_id, args.limit)
            else:
                print(index.get_section_content(args.document_id, args.section_id))
            return 0

        model_only_commands = {
            "train-hrm",
            "evaluate-hrm",
            "hrm-info",
            "train-neural",
            "evaluate-neural",
            "model-info",
            "prepare-document-hrm-data",
            "train-document-hrm",
            "evaluate-document-hrm",
            "evaluate-document-hrm-external",
            "diagnose-document-hrm",
            "analyze-document-hrm-errors",
            "document-hrm-info",
            "answer-document",
        }
        if args.data_dir.is_dir():
            repository = FinancialRepository(args.data_dir)
        elif args.command in model_only_commands:
            repository = None
        else:
            repository = FinancialRepository(args.data_dir)
        if args.command == "train-hrm":
            config = HRMConfig(
                vocab_size=args.vocab_size,
                embedding_dim=args.embedding_dim,
                hidden_size=args.hidden_size,
                high_cycles=args.high_cycles,
                low_steps=args.low_steps,
                max_segments=args.max_segments,
            )
            bundle, report = train_reasoning_bundle(
                repository,
                config=config,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                max_tickers=args.max_tickers,
                device=args.device,
            )
            destination = bundle.save(args.model_path)
            print(f"Saved recurrent HRM: {destination}")
            print(f"Trainable parameters: {bundle.parameter_count:,}")
            print(
                f"Samples={report.samples}, epochs={report.epochs}, "
                f"loss={report.final_loss:.5f}"
            )
            for task, accuracy in report.accuracy.items():
                print(f"- {task}: ontology accuracy={accuracy:.2%}")
            return 0

        if args.command in {"evaluate-hrm", "hrm-info"}:
            bundle = HierarchicalReasoningBundle.load(
                args.model_path, device=args.device
            )
            if args.command == "hrm-info":
                print(f"Model: {args.model_path.resolve()}")
                print(model_summary_json(bundle))
            else:
                _print_hrm_evaluation(
                    evaluate_reasoning_bundle(
                        bundle, repository, max_tickers=args.max_tickers
                    )
                )
            return 0

        if args.command == "train-neural":
            bundle, reports = train_model_bundle(
                repository,
                feature_dim=args.feature_dim,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                max_tickers=args.max_tickers,
            )
            destination = bundle.save(args.legacy_model_path)
            print(f"Saved legacy neural model: {destination}")
            print(f"Trainable parameters: {bundle.parameter_count:,}")
            for name, report in reports.items():
                print(
                    f"- {name}: samples={report.samples}, epochs={report.epochs}, "
                    f"loss={report.loss:.5f}, train_accuracy={report.accuracy:.2%}"
                )
            return 0

        if args.command in {"evaluate-neural", "model-info"}:
            bundle = NeuralModelBundle.load(args.legacy_model_path)
            if args.command == "model-info":
                print(f"Model: {args.legacy_model_path.resolve()}")
                print(f"Architecture: {bundle.metadata.get('architecture', 'unknown')}")
                print(f"Trainable parameters: {bundle.parameter_count:,}")
                print(f"Non-zero parameters: {bundle.nonzero_parameter_count:,}")
                print("Metadata:")
                print(json.dumps(bundle.metadata, ensure_ascii=False, indent=2))
            else:
                _print_neural_evaluation(
                    evaluate_model_bundle(
                        bundle, repository, max_tickers=args.max_tickers
                    )
                )
            return 0

        engine_mode = "symbolic" if args.symbolic_only else args.engine
        if engine_mode == "hrm" or (
            engine_mode == "auto" and args.model_path.is_file()
        ):
            engine = HierarchicalReasoningQAEngine(
                repository, HierarchicalReasoningBundle.load(args.model_path)
            )
        elif engine_mode == "legacy" or (
            engine_mode == "auto" and args.legacy_model_path.is_file()
        ):
            engine = NeuroSymbolicHRMEngine(
                repository, NeuralModelBundle.load(args.legacy_model_path)
            )
        else:
            engine = FinancialQAEngine(repository)
        if args.command == "analyze-data":
            _print_profile(repository.profile(), repository.data_dir)
        elif args.command == "metrics":
            for formula in FORMULAS.values():
                print(f"{formula.key:16} | {formula.label:32} | {formula.expression}")
        elif args.command == "answer":
            answer = engine.answer(
                args.question, ticker=args.ticker, year=args.year, metric=args.metric
            )
            print(answer.render())
        elif args.command == "what-if":
            answer = engine.answer(
                args.question,
                ticker=args.ticker,
                year=args.year,
                metric=args.metric,
                adjustments=tuple(args.adjustments),
            )
            print(answer.render())
        elif args.command == "validate":
            _print_checks(engine.validate(args.ticker.upper(), args.year, args.tolerance))
        return 0
    except (
        QAError,
        NeuralModelError,
        ReasoningModelError,
        DocumentHRMError,
        DocumentSourceError,
        DocumentSearchError,
        DenseRetrievalError,
        FileNotFoundError,
    ) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2


def _print_profile(profile: dict[str, object], data_dir: Path) -> None:
    years = profile["years"]
    year_text = "khong co" if not years else f"{years[0]}-{years[-1]}"
    print(f"Thu muc: {data_dir}")
    print(f"Ma co phieu co BCTC: {profile['tickers']}")
    print(f"File BCTC: {profile['financial_files']}")
    print(f"File thi truong: {profile['market_files']}")
    print(f"Tong dong BCTC: {profile['financial_rows']:,}")
    print(f"Khoang nam: {year_text}")
    print("Phan bo statement:")
    for statement, count in profile["statements"].most_common():
        print(f"- {statement}: {count:,}")
    print("15 item_id pho bien:")
    for item_id, count in profile["top_item_ids"]:
        print(f"- {item_id}: {count:,}")


def _print_checks(checks: tuple[object, ...]) -> None:
    for check in checks:
        if check.status == "skipped":
            print(f"SKIP | {check.name} | thieu du lieu")
            continue
        error = check.relative_error * 100
        print(
            f"{check.status.upper():4} | {check.name} | "
            f"actual={check.actual:,.4f} | expected={check.expected:,.4f} | error={error:.3f}%"
        )


def _print_hrm_evaluation(result: dict[str, object]) -> None:
    print(f"Trainable parameters: {int(result['parameter_count']):,}")
    for task, accuracy in result["accuracy"].items():
        print(f"{task}: ontology/paraphrase accuracy={float(accuracy):.2%}")
    held_out = result.get("held_out", {})
    variable_total = int(held_out.get("variable_total", 0))
    statement_total = int(held_out.get("statement_total", 0))
    if variable_total:
        correct = int(held_out["variable_correct"])
        print(f"Held-out variable cells: {correct}/{variable_total} ({correct / variable_total:.2%})")
    if statement_total:
        correct = int(held_out["statement_correct"])
        print(f"Held-out statements: {correct}/{statement_total} ({correct / statement_total:.2%})")


def _print_neural_evaluation(result: dict[str, object]) -> None:
    metric_correct = int(result["metric_correct"])
    metric_total = int(result["metric_total"])
    print(
        f"Metric holdout: {metric_correct}/{metric_total} "
        f"({metric_correct / metric_total:.2%})"
    )
    for item in result["metric_details"]:
        marker = "PASS" if item["correct"] else "FAIL"
        print(
            f"- {marker} expected={item['expected']} predicted={item['predicted']} "
            f"confidence={item['confidence']:.3f}"
        )
    variable_total = int(result.get("variable_total", 0))
    statement_total = int(result.get("statement_total", 0))
    if variable_total:
        variable_correct = int(result["variable_correct"])
        print(
            f"Variable/cell classification: {variable_correct}/{variable_total} "
            f"({variable_correct / variable_total:.2%})"
        )
    if statement_total:
        statement_correct = int(result["statement_correct"])
        print(
            f"Statement classification: {statement_correct}/{statement_total} "
            f"({statement_correct / statement_total:.2%})"
        )


def _print_document_index_stats(
    index: VietnameseFinancialDocumentIndex, destination: Path
) -> None:
    stats = index.stats()
    print(f"Saved document index: {destination}")
    print(f"Documents: {stats['documents']}")
    print(f"Sections: {stats['sections']:,}")
    print(f"Chunks: {stats['chunks']:,}")
    print(f"Source lines: {stats['lines']:,}")
    print(f"Search terms: {stats['terms']:,}")


def _print_document_evidence(
    result: DocumentEvidenceResult, show_hits: int
) -> None:
    if result.metric_key:
        print(f"Metric: {result.metric_key}")
        print(f"Formula: {result.formula_expression}")
        coverage = f"{len(result.covered_variables)}/{len(result.required_variables)}"
        print(f"Account-value evidence coverage: {coverage}")
        if result.missing_variables:
            print(f"Missing evidence: {', '.join(result.missing_variables)}")
    if result.selected_document_id:
        print(f"Selected document: {result.selected_document_id}")
    if result.reporting_year:
        print(f"Reporting year: {result.reporting_year}")
    if result.year_mismatch:
        print(
            f"YEAR MISMATCH: question={result.requested_year}, "
            f"report={result.reporting_year}; evidence is not valid for the requested year"
        )
    print(f"Retrieval rounds: {len(result.rounds)}")
    for item in result.rounds:
        scope = item.document_id or "all documents"
        covered = ", ".join(item.newly_covered) or "none"
        missing = ", ".join(item.missing_after) or "none"
        print(
            f"- Round {item.number}: scope={scope}, hits={len(item.hit_ids)}, "
            f"new={covered}, missing={missing}"
        )
    _print_document_hits(result.hits, show_hits)


def _print_document_hits(hits: tuple[SearchHit, ...], show_hits: int) -> None:
    if not hits:
        print("Khong tim thay bang chung phu hop")
        return
    limit = max(0, show_hits)
    print(f"Evidence hits shown: {min(len(hits), limit)}/{len(hits)}")
    for position, hit in enumerate(hits[:limit], start=1):
        chunk = hit.chunk
        exact = " exact" if hit.exact_match else ""
        components = ""
        if hit.retrieval_mode == "hybrid":
            sparse = "-" if hit.sparse_score is None else f"{hit.sparse_score:.3f}"
            dense = "-" if hit.dense_score is None else f"{hit.dense_score:.3f}"
            components = f" sparse={sparse} dense={dense}"
        print(
            f"[{position}] mode={hit.retrieval_mode} score={hit.score:.3f}{exact}"
            f"{components} | {chunk.source_name} | "
            f"{chunk.section_id} | lines {chunk.start_line}-{chunk.end_line}"
        )
        print(f"    Heading: {chunk.heading}")
        print(f"    Match: {', '.join(hit.matched_terms[:12])}")
        print(f"    {_document_excerpt(hit)}")


def _document_excerpt(hit: SearchHit, max_chars: int = 420) -> str:
    lines = [line.strip() for line in hit.chunk.text.splitlines() if line.strip()]
    selected: list[str] = []
    terms = set(hit.matched_terms)
    best_position = -1
    best_score = 0
    for position, line in enumerate(lines):
        normalized_line = normalize_text(line)
        phrase_score = sum(
            len(phrase.split()) + 3
            for phrase in hit.matched_phrases
            if phrase in normalized_line
        )
        term_score = len(terms.intersection(normalized_line.split()))
        score = phrase_score + term_score
        if score > best_score:
            best_score = score
            best_position = position
    if best_position >= 0:
        start = max(0, best_position - 1)
        selected = lines[start : best_position + 2]
    if not selected:
        selected = lines[:3]
    text = " | ".join(selected)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _print_document_list(
    index: VietnameseFinancialDocumentIndex, filter_text: str, limit: int
) -> None:
    normalized_filter = normalize_text(filter_text)
    limit = max(0, limit)
    if limit == 0:
        print("Documents shown: 0")
        return
    shown = 0
    for document_id in index.document_ids():
        outline = index.outline(document_id)
        haystack = normalize_text(f"{document_id} {outline.source_name}")
        if normalized_filter and normalized_filter not in haystack:
            continue
        year = outline.reporting_year if outline.reporting_year else "unknown"
        print(
            f"{document_id} | {outline.source_name} | year={year} | "
            f"sections={len(outline.sections)}"
        )
        shown += 1
        if shown >= limit:
            break
    print(f"Documents shown: {shown}")


def _print_document_outline(
    index: VietnameseFinancialDocumentIndex, document_id: str, limit: int
) -> None:
    outline = index.outline(document_id)
    print(f"Document: {outline.document_id}")
    print(f"Source: {outline.source_name}")
    print(f"Reporting year: {outline.reporting_year or 'unknown'}")
    for section in outline.sections[: max(0, limit)]:
        print(
            f"{section.section_id} | lines {section.start_line}-{section.end_line} | "
            f"{section.heading}"
        )
    print(
        f"Sections shown: {min(len(outline.sections), max(0, limit))}/"
        f"{len(outline.sections)}"
    )
