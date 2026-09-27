from __future__ import annotations

from .source_binding import program_binding_issues

import json
import math
import random
import re
import sqlite3
import threading
import unicodedata
from copy import deepcopy
from collections import Counter
from dataclasses import asdict, dataclass, replace
from decimal import InvalidOperation
from difflib import SequenceMatcher
from hashlib import blake2b, sha256
from itertools import combinations, permutations
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for Adaptive-HRM. Install requirements.txt first."
    ) from exc

from .ontology import normalize_text
from .dense_retrieval import DEFAULT_DENSE_MODEL
from .semantic_encoder import (
    FINANCIAL_VI_SEMANTIC_TEMPLATE,
    LEGACY_SEMANTIC_TEMPLATE,
    SemanticEncoderError,
    SemanticEncoderSpec,
    build_role_aware_encoder,
)
from .program_data import (
    _is_ratio_metric,
    COMMUTATIVE,
    MAX_OPERANDS,
    OPERATION_ARITY,
    OPERATION_TO_ID,
    OPERATIONS,
    SUM_N_MAX_OPERANDS,
    SUM_N_OPERATION,
    NumericCandidate,
    ProgramDataError,
    ProgramExample,
    ReasoningDAGProgram,
    ReasoningProgram,
    canonical_pointer_equal,
    canonical_program_equal,
    canonical_sum_family_equal,
    currency_unit_scale,
    execute_program,
    execute_program_dag,
    is_sum_operation,
    operation_accepts_arity,
    load_program_examples,
    materialize_program_dag,
)
from .calibration_policy import (
    CalibrationPolicy,
    CalibrationPolicyError,
    CalibrationStatus,
    wilson_lower_bound,
)
from .source_action_policy import classify_source_action
from .evidence_schema import EvidenceGraph, build_evidence_graph, evidence_cells, _unit_key
from .direction_contract import (
    orient_program_direction,
    reversed_pair_is_proven_negative,
)
from .financial_identities import (
    FinancialIdentity,
    get_financial_identity,
    target_matches_metric,
)
from .financial_hrm_features import (
    FINANCIAL_CANDIDATE_FEATURE_SIZE,
    FINANCIAL_SLOT_FEATURES,
    FINANCIAL_SLOT_FEATURE_SIZE,
    build_financial_feature_matrices,
    financial_physical_identity,
    financial_same_physical_pairs,
)
from .question_planner import (
    QuestionPlan,
    compile_question_plan,
    requests_financial_identity_computation,
)
from .training_state import (
    TRAINING_STATE_SCHEMA,
    TrainingStateError,
    atomic_save_training_state,
    canonical_payload_sha256,
    decode_python_random_state,
    encode_python_random_state,
    load_training_state,
    restore_optimizer_state_device,
)
from .symbolic_verifier import (
    _metric_obligation_state,
    VerificationResult,
    VerificationState,
    verify_program,
)
from .program_reranker import (
    PROGRAM_RERANKER_FEATURE_SCHEMA,
    PROGRAM_RERANKER_FEATURES,
    RERANKER_TRAIN_OPERATIONS,
    ProgramRerankerBundle,
)
from .program_synthesis import (
    TypedProgramProposal,
    TypedSynthesisResult,
    synthesize_typed_programs,
)
from .v12b2_supervision import semantic_program_signature


DOCUMENT_HRM_SCHEMA_VERSION = 13
LEGACY_DOCUMENT_HRM_SCHEMA_VERSIONS = {4, 5, 6, 7, 8, 9, 10, 11, 12}
DEFAULT_SEMANTIC_CACHE_DIR = (
    Path(__file__).resolve().parents[2] / "models" / "embedding_cache"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROTECTED_TRAINING_INPUTS = frozenset(
    {
        (_PROJECT_ROOT / "data" / "test.jsonl").resolve(),
        (_PROJECT_ROOT / "data" / "canonical_v13_v1" / "test.jsonl").resolve(),
    }
)
PAIR_FEATURE_SIZE = 20
DEFAULT_ABSTENTION_STABILITY_ERRORS = 1
MULTI_OPERAND_OPERATIONS = frozenset({"SUM3", "SUM4", SUM_N_OPERATION})
SUM_FAMILY_OPERATION_IDS = tuple(
    OPERATION_TO_ID[name] for name in ("ADD", "SUM3", "SUM4")
)
VERIFIER_GUIDED_OPERATIONS = frozenset(
    {
        "SELECT",
        "ADD",
        "SUBTRACT",
        "MULTIPLY",
        "DIVIDE",
        "RATIO_PERCENT",
        "PERCENT_CHANGE",
        "AVERAGE",
    }
)
MAX_VERIFIER_GUIDED_PAIR_CANDIDATES = 24


class DocumentHRMError(ValueError):
    pass


def _reject_protected_training_input(path: str | Path, *, role: str) -> Path:
    """Resolve a training input and reject the frozen canonical test split."""

    resolved = Path(path).expanduser().resolve()
    if resolved in _PROTECTED_TRAINING_INPUTS:
        raise DocumentHRMError(
            f"Protected test split cannot be used as {role}: {resolved}"
        )
    return resolved


@dataclass(frozen=True)
class DocumentHRMConfig:
    vocab_size: int = 16384
    embedding_dim: int = 96
    hidden_size: int = 160
    numeric_feature_size: int = 18
    question_feature_size: int = 12
    max_question_tokens: int = 96
    max_candidate_tokens: int = 72
    max_candidates: int = 64
    attention_heads: int = 4
    feed_forward_multiplier: int = 3
    high_cycles: int = 2
    low_steps: int = 3
    max_segments: int = 4
    dropout: float = 0.10
    hierarchical_input_mode: str = "strict"
    relevance_prior_max: float = 2.0
    act_mode: str = "q_learning"
    minimum_halt_segments: int = 2
    input_encoder_mode: str = "semantic"
    semantic_model_name: str = DEFAULT_DENSE_MODEL
    semantic_embedding_size: int = 384
    semantic_encoder_backend: str = "fastembed"
    semantic_model_revision: str = ""
    semantic_local_files_only: bool = False
    semantic_encoder_output_size: int = 0
    semantic_word_segmentation: str = "none"
    semantic_segmenter_version: str = ""
    semantic_prefix_mode: str = "none"
    semantic_query_encoding_mode: str = "document"
    semantic_text_template_version: str = LEGACY_SEMANTIC_TEMPLATE
    semantic_adapter_mode: str = "identity"
    candidate_semantic_mode: str = "context"
    pointer_conditioning_mode: str = "independent"
    pair_backbone_mode: str = "detached"
    pair_backbone_warmup_epochs: int = 0
    pair_loss_weight: float = 1.0
    pair_loss_warmup_epochs: int = 0
    pair_targeted_negative_weight: float = 0.0
    pointer_teacher_forcing: float = 0.50
    pointer_hard_negative_weight: float = 0.0
    pointer_hard_negative_margin: float = 0.50
    financial_structure_mode: str = "legacy"
    financial_hierarchy_layers: int = 1
    variable_sum_mode: str = "disabled"
    max_program_operands: int = MAX_OPERANDS
    program_search_strategy: str = "legacy"
    seed: int = 31

    @property
    def semantic_encoder_size(self) -> int:
        return (
            self.semantic_encoder_output_size
            if self.semantic_encoder_output_size > 0
            else self.semantic_embedding_size
        )

    def validate(self) -> None:
        if self.vocab_size < 1024:
            raise DocumentHRMError("vocab_size must be at least 1024")
        if self.embedding_dim < 32 or self.hidden_size < 64:
            raise DocumentHRMError("embedding_dim/hidden_size are too small")
        if self.hidden_size % 2:
            raise DocumentHRMError("hidden_size must be even for the bidirectional encoder")
        if self.max_candidates < 8 or self.max_candidates > 256:
            raise DocumentHRMError("max_candidates must be in [8, 256]")
        if self.attention_heads < 1 or self.hidden_size % self.attention_heads:
            raise DocumentHRMError("hidden_size must be divisible by attention_heads")
        if self.feed_forward_multiplier < 2:
            raise DocumentHRMError("feed_forward_multiplier must be at least 2")
        if self.high_cycles < 1 or self.low_steps < 1 or self.max_segments < 1:
            raise DocumentHRMError("recurrence counts must be positive")
        if self.hierarchical_input_mode not in {"strict", "legacy"}:
            raise DocumentHRMError("hierarchical_input_mode must be strict or legacy")
        if self.relevance_prior_max < 0:
            raise DocumentHRMError("relevance_prior_max must be non-negative")
        if self.act_mode not in {"q_learning", "supervised"}:
            raise DocumentHRMError("act_mode must be q_learning or supervised")
        if not 1 <= self.minimum_halt_segments <= self.max_segments:
            raise DocumentHRMError(
                "minimum_halt_segments must be within [1, max_segments]"
            )
        if self.input_encoder_mode not in {"semantic", "hash"}:
            raise DocumentHRMError("input_encoder_mode must be semantic or hash")
        if self.semantic_embedding_size < 128:
            raise DocumentHRMError("semantic_embedding_size must be at least 128")
        if self.semantic_encoder_size < 128:
            raise DocumentHRMError(
                "semantic_encoder_output_size must be at least 128"
            )
        try:
            SemanticEncoderSpec(
                backend=self.semantic_encoder_backend,
                model_name=self.semantic_model_name,
                model_revision=self.semantic_model_revision,
                output_size=self.semantic_encoder_size,
                local_files_only=self.semantic_local_files_only,
                word_segmentation=self.semantic_word_segmentation,
                segmenter_version=self.semantic_segmenter_version,
                prefix_mode=self.semantic_prefix_mode,
                query_encoding_mode=self.semantic_query_encoding_mode,
                text_template_version=self.semantic_text_template_version,
            ).validate()
        except SemanticEncoderError as exc:
            raise DocumentHRMError(str(exc)) from exc
        if self.semantic_adapter_mode not in {
            "identity",
            "layernorm_linear_l2_v1",
        }:
            raise DocumentHRMError(
                "semantic_adapter_mode must be identity or "
                "layernorm_linear_l2_v1"
            )
        if (
            self.semantic_adapter_mode == "identity"
            and self.semantic_encoder_size
            != self.semantic_embedding_size
        ):
            raise DocumentHRMError(
                "identity semantic adapter requires equal raw and HRM dimensions"
            )
        if self.candidate_semantic_mode not in {"context", "structured"}:
            raise DocumentHRMError(
                "candidate_semantic_mode must be context or structured"
            )
        if self.pointer_conditioning_mode not in {
            "independent",
            "operation_sequential",
            "later_sequential",
            "joint_pair",
        }:
            raise DocumentHRMError(
                "pointer_conditioning_mode must be independent, "
                "operation_sequential, later_sequential or joint_pair"
            )
        if self.pair_backbone_mode not in {
            "detached",
            "end_to_end",
            "normalized_detached",
            "normalized_end_to_end",
            "warmup_end_to_end",
        }:
            raise DocumentHRMError(
                "pair_backbone_mode must be detached, end_to_end, "
                "normalized_detached, normalized_end_to_end or "
                "warmup_end_to_end"
            )
        if (
            self.pair_backbone_mode
            in {
                "end_to_end",
                "normalized_detached",
                "normalized_end_to_end",
                "warmup_end_to_end",
            }
            and self.pointer_conditioning_mode != "joint_pair"
        ):
            raise DocumentHRMError(
                "relational pair backbone requires joint_pair conditioning"
            )
        if self.pair_loss_weight < 0:
            raise DocumentHRMError("pair_loss_weight must be non-negative")
        if self.pair_backbone_warmup_epochs < 0:
            raise DocumentHRMError(
                "pair_backbone_warmup_epochs must be non-negative"
            )
        if self.pair_loss_warmup_epochs < 0:
            raise DocumentHRMError(
                "pair_loss_warmup_epochs must be non-negative"
            )
        if self.pair_targeted_negative_weight < 0:
            raise DocumentHRMError(
                "pair_targeted_negative_weight must be non-negative"
            )
        if not 0.0 <= self.pointer_teacher_forcing <= 1.0:
            raise DocumentHRMError("pointer_teacher_forcing must be within [0, 1]")
        if self.pointer_hard_negative_weight < 0:
            raise DocumentHRMError(
                "pointer_hard_negative_weight must be non-negative"
            )
        if self.pointer_hard_negative_margin < 0:
            raise DocumentHRMError(
                "pointer_hard_negative_margin must be non-negative"
            )
        if self.financial_structure_mode not in {
            "legacy",
            "typed_hierarchy_v1",
        }:
            raise DocumentHRMError(
                "financial_structure_mode must be legacy or typed_hierarchy_v1"
            )
        if not 1 <= self.financial_hierarchy_layers <= 3:
            raise DocumentHRMError(
                "financial_hierarchy_layers must be within [1, 3]"
            )
        if self.variable_sum_mode not in {"disabled", "sum_n_v1"}:
            raise DocumentHRMError(
                "variable_sum_mode must be disabled or sum_n_v1"
            )
        if not MAX_OPERANDS <= self.max_program_operands <= SUM_N_MAX_OPERANDS:
            raise DocumentHRMError(
                f"max_program_operands must be within "
                f"[{MAX_OPERANDS}, {SUM_N_MAX_OPERANDS}]"
            )
        if (
            self.variable_sum_mode == "disabled"
            and self.max_program_operands != MAX_OPERANDS
        ):
            raise DocumentHRMError(
                "disabled variable_sum_mode requires the legacy operand width"
            )
        if self.program_search_strategy not in {
            "legacy",
            "diverse_set_beam_v1",
        }:
            raise DocumentHRMError(
                "program_search_strategy must be legacy or diverse_set_beam_v1"
            )
        if (
            self.financial_structure_mode == "typed_hierarchy_v1"
            and self.pointer_conditioning_mode != "joint_pair"
        ):
            raise DocumentHRMError(
                "typed_hierarchy_v1 requires joint_pair pointer conditioning"
            )


@dataclass
class DocumentHRMState:
    high: Tensor
    low: Tensor

    def detach(self) -> "DocumentHRMState":
        return DocumentHRMState(self.high.detach(), self.low.detach())


@dataclass(frozen=True)
class DocumentHRMForward:
    state: DocumentHRMState
    operation_logits: Tensor
    pointer_logits: Tensor
    q_values: Tensor
    pair_logits: Tensor | None = None
    select_pointer_logits: Tensor | None = None
    operand_count_logits: Tensor | None = None
    sum_membership_logits: Tensor | None = None


@dataclass(frozen=True)
class ProgramPrediction:
    program: ReasoningProgram
    operation_confidence: float
    pointer_confidences: tuple[float, ...]
    confidence: float
    segments: int
    halt_q: float
    continue_q: float
    operation_probabilities: tuple[float, ...]
    neural_operation_confidence: float
    neural_pointer_confidences: tuple[float, ...]
    confidence_schema: str
    grounding_constraints_applied: tuple[str, ...]
    operand_count_confidence: float | None = None
    program_candidates_considered: int = 1
    symbolic_program_rejections: int = 0
    symbolic_proof_score: float = 0.0
    symbolic_unknown_constraints: int = 0
    program_search_trace: tuple["ProgramCandidateTrace", ...] = ()
    reranker_score: float | None = None
    reranker_confidence: float | None = None
    typed_program_synthesis_enabled: bool = False
    synthesized_program_candidates: int = 0
    typed_synthesis_status: tuple[str, ...] = ()
    program_dag: ReasoningDAGProgram | None = None
    slot_bindings: tuple[tuple[str, int], ...] = ()
    slot_candidate_pools_at_10: tuple[tuple[int, ...], ...] = ()
    slot_candidate_pools_at_20: tuple[tuple[int, ...], ...] = ()
    operation_slot_candidate_pools_at_10: tuple[
        tuple[str, tuple[tuple[int, ...], ...]], ...
    ] = ()
    operation_slot_candidate_pools_at_20: tuple[
        tuple[str, tuple[tuple[int, ...], ...]], ...
    ] = ()
    operation_beam: tuple[str, ...] = ()
    preverification_programs: tuple[ReasoningProgram, ...] = ()
    selected_for_verification: tuple[ReasoningProgram, ...] = ()
    generated_programs_before_physical_dedup: int = 0
    generated_programs_after_physical_dedup: int = 0


def _safe_materialize_program_dag(
    program: ReasoningProgram,
    candidates: Sequence[NumericCandidate],
) -> ReasoningDAGProgram | None:
    try:
        return materialize_program_dag(program, candidates)
    except (ProgramDataError, IndexError):
        return None


@dataclass(frozen=True)
class SymbolicQuerySpec:
    """Typed, report-independent constraints compiled from a finance question."""

    focus_text: str
    requested_years: tuple[str, ...]
    required_operation: str | None
    has_structured_evidence: bool
    allowed_operations: tuple[str, ...] = ()


@dataclass(frozen=True)
class SymbolicProgramSearchResult:
    program: ReasoningProgram
    pointer_confidences: tuple[float, ...]
    neural_pointer_confidences: tuple[float, ...]
    constraints_applied: tuple[str, ...]
    candidates_considered: int
    candidates_rejected: int
    proof_score: float
    unknown_constraints: int
    trace: tuple["ProgramCandidateTrace", ...]
    reranker_score: float | None = None
    reranker_confidence: float | None = None
    synthesized_program_candidates: int = 0
    typed_synthesis_status: tuple[str, ...] = ()
    grounded_confidence_floor: float | None = None
    program_dag: ReasoningDAGProgram | None = None
    slot_bindings: tuple[tuple[str, int], ...] = ()
    slot_candidate_pools_at_10: tuple[tuple[int, ...], ...] = ()
    slot_candidate_pools_at_20: tuple[tuple[int, ...], ...] = ()
    operation_slot_candidate_pools_at_10: tuple[
        tuple[str, tuple[tuple[int, ...], ...]], ...
    ] = ()
    operation_slot_candidate_pools_at_20: tuple[
        tuple[str, tuple[tuple[int, ...], ...]], ...
    ] = ()
    operation_beam: tuple[str, ...] = ()
    preverification_programs: tuple[ReasoningProgram, ...] = ()
    selected_for_verification: tuple[ReasoningProgram, ...] = ()
    generated_programs_before_physical_dedup: int = 0
    generated_programs_after_physical_dedup: int = 0
    # Internal memo shared by the additive V12.4 search. It avoids verifying
    # every V12.3 program twice while preserving the exact legacy winner.
    verification_cache: tuple[
        tuple[
            ReasoningProgram,
            bool,
            VerificationResult,
            tuple[str, ...],
        ],
        ...,
    ] = ()


@dataclass(frozen=True)
class ProgramCandidateTrace:
    program: ReasoningProgram
    neural_score: float
    neural_rank: int
    verification_state: str
    proof_score: float
    eligible: bool
    rejection_reasons: tuple[str, ...]
    unknown_reasons: tuple[str, ...]
    selected_for_reranking: bool = False
    reranker_features: tuple[float, ...] = ()
    reranker_score: float | None = None
    reranker_confidence: float | None = None
    legacy_rank: int = 0
    legacy_rank_score: float = 0.0
    legacy_winner: bool = False
    generation_sources: tuple[str, ...] = ()
    # Runtime eligibility remains deliberately strict.  Training state is a
    # separate tri-state label so incomplete metadata is not silently turned
    # into a hard negative while constructing reranker caches.
    training_state: str = ""
    proof_contract_state: str = ""
    required_valid_fraction: float = 0.0
    required_unknown_fraction: float = 0.0
    optional_support_fraction: float = 0.0
    proof_obligations: tuple[str, ...] = ()


@dataclass
class _SearchProgramCandidate:
    program: ReasoningProgram
    neural_score: float
    operation_probability: float
    pointer_confidences: tuple[float, ...]
    verification: VerificationResult
    rejection_reasons: tuple[str, ...]
    neural_rank: int = 0
    reranker_features: tuple[float, ...] = ()
    selected_for_reranking: bool = False
    reranker_score: float | None = None
    reranker_confidence: float | None = None
    legacy_rank: int = 0
    legacy_rank_score: float = 0.0
    legacy_winner: bool = False
    generation_sources: tuple[str, ...] = ()
    identity_constraint_required: bool = False
    program_dag: ReasoningDAGProgram | None = None
    slot_bindings: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class CandidateGenerationResult:
    programs: tuple[ReasoningProgram, ...]
    program_scores: tuple[tuple[ReasoningProgram, float], ...]
    slot_pools_at_10: tuple[tuple[int, ...], ...]
    slot_pools_at_20: tuple[tuple[int, ...], ...]
    generated_before_physical_dedup: int
    generated_after_physical_dedup: int


_TRAINING_UNKNOWN_REJECTION_REASONS = frozenset(
    {
        "unproven_physical_mapping",
        "source_cell_binding_ambiguous",
        "missing_structured_row",
        "missing_requested_operand_roles",
        "operand_role_mismatch",
        "missing_direct_metric_evidence",
        "unary_grounding_mismatch",
    }
)


def _serialized_proof_obligations(
    verification: VerificationResult,
) -> tuple[str, ...]:
    return tuple(
        f"{item.applicability}:{item.name}:"
        f"{item.state.value if item.state is not None else 'NOT_APPLICABLE'}"
        for item in verification.proof_obligations
    )


def _program_candidate_training_state(
    verification: VerificationResult,
    rejection_reasons: Sequence[str],
) -> str:
    """Map strict runtime gates to a cache-only tri-state decision.

    Only required proof obligations determine the verifier tri-state. Runtime
    rejection stays separate and unchanged.
    """

    if verification.proof_contract_state is VerificationState.INVALID:
        return VerificationState.INVALID.value
    reasons = set(rejection_reasons) - set(verification.reasons)
    hard_reasons = reasons - _TRAINING_UNKNOWN_REJECTION_REASONS
    if hard_reasons:
        return VerificationState.INVALID.value
    if reasons:
        return VerificationState.UNKNOWN.value
    return verification.training_support_state.value


@dataclass(frozen=True)
class DocumentHRMTrainingReport:
    train_examples: int
    validation_examples: int
    epochs: int
    final_loss: float
    operation_accuracy: float
    operation_macro_precision: float
    operation_macro_recall: float
    operation_macro_f1: float
    operation_weighted_precision: float
    operation_weighted_recall: float
    operation_weighted_f1: float
    program_exact_accuracy: float
    execution_accuracy: float
    pointer_pair_accuracy: float
    pointer_slot_accuracy: tuple[float, ...]
    accepted_coverage: float
    accepted_precision: float
    operation_counts: dict[str, int]
    pointer_slot_support: tuple[int, ...] = ()
    accepted_precision_defined: bool = True


@dataclass(frozen=True)
class EncodedBatch:
    question_ids: Tensor
    question_mask: Tensor
    question_features: Tensor
    candidate_ids: Tensor
    candidate_token_mask: Tensor
    numeric_features: Tensor
    candidate_mask: Tensor
    question_semantic: Tensor | None = None
    candidate_semantic: Tensor | None = None
    slot_semantic: Tensor | None = None
    operation_targets: Tensor | None = None
    pointer_targets: Tensor | None = None
    semantic_pointer_targets: Tensor | None = None
    semantic_slot_declared: Tensor | None = None
    operand_count_targets: Tensor | None = None
    sum_family_targets: Tensor | None = None
    pair_features: Tensor | None = None
    unary_structural_negatives: Tensor | None = None
    pair_structural_negatives: Tensor | None = None
    verifier_unary_invalid_negatives: Tensor | None = None
    verifier_pair_invalid_negatives: Tensor | None = None
    financial_candidate_features: Tensor | None = None
    financial_slot_features: Tensor | None = None
    financial_slot_mismatch_mask: Tensor | None = None
    financial_same_physical_mask: Tensor | None = None


def _semantic_encoder_spec_from_config(
    config: DocumentHRMConfig,
) -> SemanticEncoderSpec:
    return SemanticEncoderSpec(
        backend=config.semantic_encoder_backend,
        model_name=config.semantic_model_name,
        model_revision=config.semantic_model_revision,
        output_size=config.semantic_encoder_size,
        local_files_only=config.semantic_local_files_only,
        word_segmentation=config.semantic_word_segmentation,
        segmenter_version=config.semantic_segmenter_version,
        prefix_mode=config.semantic_prefix_mode,
        query_encoding_mode=config.semantic_query_encoding_mode,
        text_template_version=config.semantic_text_template_version,
    )


class SemanticEmbeddingCache:
    """Frozen multilingual sentence encoder with an in-memory text cache."""

    def __init__(
        self,
        model_name: str,
        embedding_size: int,
        cache_dir: str | Path | None = None,
        *,
        spec: SemanticEncoderSpec | None = None,
        encoder: object | None = None,
    ) -> None:
        self.spec = spec or SemanticEncoderSpec(
            backend="fastembed",
            model_name=model_name,
            model_revision="",
            output_size=embedding_size,
            text_template_version=LEGACY_SEMANTIC_TEMPLATE,
        )
        self.spec.validate()
        if self.spec.model_name != model_name:
            raise DocumentHRMError(
                "Semantic cache model name does not match its encoder spec"
            )
        if self.spec.output_size != embedding_size:
            raise DocumentHRMError(
                "Semantic cache dimension does not match its encoder spec"
            )
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.encoder = encoder or build_role_aware_encoder(
            self.spec, cache_dir
        )
        self._vectors: dict[str, np.ndarray] = {}
        self._lock = threading.RLock()
        cache_root = Path(cache_dir or DEFAULT_SEMANTIC_CACHE_DIR)
        cache_root.mkdir(parents=True, exist_ok=True)
        self._database = sqlite3.connect(
            cache_root / "hrm_semantic_vectors.sqlite3",
            check_same_thread=False,
        )
        self._database.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "model TEXT NOT NULL, cache_key TEXT NOT NULL, vector BLOB NOT NULL, "
            "PRIMARY KEY (model, cache_key))"
        )

    def _database_model_key(self, role: str) -> str:
        if (
            self.spec.backend == "fastembed"
            and not self.spec.model_revision
            and self.spec.word_segmentation == "none"
            and self.spec.prefix_mode == "none"
            and self.spec.query_encoding_mode == "document"
            and self.spec.text_template_version
            == LEGACY_SEMANTIC_TEMPLATE
        ):
            return self.model_name
        return self.spec.cache_namespace(role)

    def _load_cached_vectors(
        self, keys: Sequence[str], *, role: str
    ) -> None:
        """Load only requested vectors instead of scanning the whole SQLite cache."""
        with self._lock:
            pending = list(
                dict.fromkeys(key for key in keys if key not in self._vectors)
            )
            # Stay below SQLite's variable limit and keep lookups friendly to caches
            # stored on /mnt/<drive>, where a full-table scan is particularly slow.
            for start in range(0, len(pending), 400):
                current = pending[start : start + 400]
                if not current:
                    continue
                placeholders = ",".join("?" for _ in current)
                query = (
                    "SELECT cache_key, vector FROM embeddings "
                    f"WHERE model = ? AND cache_key IN ({placeholders})"
                )
                for cache_key, blob in self._database.execute(
                    query, (self._database_model_key(role), *current)
                ):
                    vector = np.frombuffer(blob, dtype=np.float32).copy()
                    norm = float(np.linalg.norm(vector))
                    legacy_namespace = (
                        self._database_model_key(role) == self.model_name
                    )
                    if (
                        vector.shape == (self.embedding_size,)
                        and np.isfinite(vector).all()
                        and norm > 0
                        and (
                            legacy_namespace
                            or math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-4)
                        )
                    ):
                        self._vectors[cache_key] = vector

    def _cache_key(self, text: str, *, role: str = "evidence") -> str:
        if role not in {"query", "evidence"}:
            raise DocumentHRMError("Semantic cache role is invalid")
        digest = blake2b(digest_size=20)
        digest.update(self._database_model_key(role).encode("utf-8"))
        digest.update(b"\0")
        if self._database_model_key(role) != self.model_name:
            digest.update(role.encode("ascii"))
            digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 64,
        role: str = "evidence",
    ) -> np.ndarray:
        with self._lock:
            normalized = [text.strip() or "[empty]" for text in texts]
            keys = [
                self._cache_key(text, role=role) for text in normalized
            ]
            self._load_cached_vectors(keys, role=role)
            missing_map = {
                key: text
                for key, text in zip(keys, normalized)
                if key not in self._vectors
            }
            missing = list(missing_map)
            if missing:
                raw_texts = [missing_map[key] for key in missing]
                if role == "query" and hasattr(
                    self.encoder, "encode_queries"
                ):
                    vectors = self.encoder.encode_queries(
                        raw_texts, batch_size=batch_size
                    )
                else:
                    vectors = self.encoder.encode_documents(
                        raw_texts, batch_size=batch_size
                    )
                pending: list[tuple[str, str, bytes]] = []
                for key, raw in zip(missing, vectors):
                    vector = np.asarray(raw, dtype=np.float32)
                    if vector.shape != (self.embedding_size,):
                        raise DocumentHRMError(
                            f"Semantic encoder returned {vector.shape}; "
                            f"expected ({self.embedding_size},)"
                        )
                    if not np.isfinite(vector).all():
                        raise DocumentHRMError(
                            "Semantic encoder returned a non-finite vector"
                        )
                    norm = float(np.linalg.norm(vector))
                    if norm <= 0:
                        raise DocumentHRMError(
                            "Semantic encoder returned a zero vector"
                        )
                    self._vectors[key] = vector / norm
                    pending.append(
                        (
                            self._database_model_key(role),
                            key,
                            self._vectors[key].tobytes(),
                        )
                    )
                self._database.executemany(
                    "INSERT OR REPLACE INTO embeddings(model, cache_key, vector) "
                    "VALUES (?, ?, ?)",
                    pending,
                )
                self._database.commit()
            return np.stack([self._vectors[key] for key in keys], axis=0)

    def warm_examples(
        self,
        examples: Sequence[ProgramExample],
        *,
        batch_size: int = 64,
        candidate_semantic_mode: str = "context",
    ) -> None:
        texts: list[str] = []
        query_texts: list[str] = []
        slot_query_texts: list[str] = []
        for example in examples:
            plan = compile_question_plan(example.question)
            query_texts.append(
                _question_semantic_text(
                    example.question,
                    plan,
                    template_version=self.spec.text_template_version,
                )
            )
            slot_query_texts.extend(
                _slot_semantic_text(
                    example.question,
                    plan,
                    slot,
                    template_version=self.spec.text_template_version,
                )
                for slot in range(len(example.program.operands))
            )
            texts.extend(
                _candidate_semantic_text(
                    candidate,
                    mode=candidate_semantic_mode,
                    template_version=self.spec.text_template_version,
                )
                for candidate in example.candidates
            )
        for start in range(0, len(query_texts), batch_size * 8):
            self.encode(
                query_texts[start : start + batch_size * 8],
                batch_size=batch_size,
                role="query",
            )
        for start in range(0, len(slot_query_texts), batch_size * 8):
            self.encode(
                slot_query_texts[start : start + batch_size * 8],
                batch_size=batch_size,
                role="query",
            )
        for start in range(0, len(texts), batch_size * 8):
            self.encode(
                texts[start : start + batch_size * 8],
                batch_size=batch_size,
                role="evidence",
            )


class HashSequenceTokenizer:
    """Order-preserving hashed tokenizer with no external vocabulary file."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str, max_tokens: int) -> list[int]:
        normalized = normalize_text(text)
        words = normalized.split()
        if not words:
            return [0]
        features: list[str] = []
        previous = ""
        for word in words:
            features.append(f"w:{word}")
            if previous:
                features.append(f"b:{previous}_{word}")
            if len(word) >= 5:
                features.append(f"c:{word[:3]}")
                features.append(f"c:{word[-3:]}")
            previous = word
            if len(features) >= max_tokens:
                break
        return [self._hash(item) for item in features[:max_tokens]] or [0]

    def _hash(self, feature: str) -> int:
        digest = blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return 1 + int.from_bytes(digest, "little") % (self.vocab_size - 1)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.epsilon = epsilon

    def forward(self, inputs: Tensor) -> Tensor:
        scale = inputs.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(scale + self.epsilon).to(inputs.dtype)
        return normalized * self.weight


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dropout: float,
        use_rotary_position: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_size = hidden_size // heads
        if self.head_size % 2:
            raise DocumentHRMError("attention head size must be even for RoPE")
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.use_rotary_position = use_rotary_position
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.head_size, 2, dtype=torch.float32) / self.head_size)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope(self, values: Tensor) -> Tensor:
        positions = torch.arange(values.shape[-2], device=values.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq.to(values.device))
        cosine = torch.repeat_interleave(angles.cos(), 2, dim=-1)[None, None]
        sine = torch.repeat_interleave(angles.sin(), 2, dim=-1)[None, None]
        even, odd = values[..., 0::2], values[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
        return values * cosine.to(values.dtype) + rotated * sine.to(values.dtype)

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        batch, length, _ = inputs.shape
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)
        reshape = lambda item: item.view(batch, length, self.heads, self.head_size).transpose(1, 2)
        query = reshape(query)
        key = reshape(key)
        if self.use_rotary_position:
            query = self._rope(query)
            key = self._rope(key)
        value = reshape(value)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(mask[:, None, None, :].eq(0), -1e9)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value).transpose(1, 2).contiguous()
        attended = attended.view(batch, length, self.hidden_size)
        return self.output(attended) * mask.unsqueeze(-1)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, multiplier: int) -> None:
        super().__init__()
        expanded = hidden_size * multiplier
        self.gate = nn.Linear(hidden_size, expanded, bias=False)
        self.up = nn.Linear(hidden_size, expanded, bias=False)
        self.down = nn.Linear(expanded, hidden_size, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(inputs)) * self.up(inputs))


class HRMBlock(nn.Module):
    """Transformer-style recurrent block used independently by H and L."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        feed_forward_multiplier: int,
        dropout: float,
        use_rotary_position: bool = True,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size)
        self.attention = RotarySelfAttention(
            hidden_size,
            heads,
            dropout,
            use_rotary_position=use_rotary_position,
        )
        self.feed_forward_norm = RMSNorm(hidden_size)
        self.feed_forward = SwiGLU(hidden_size, feed_forward_multiplier)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor, mask: Tensor) -> Tensor:
        hidden = inputs + self.dropout(self.attention(self.attention_norm(inputs), mask))
        hidden = hidden + self.dropout(self.feed_forward(self.feed_forward_norm(hidden)))
        return hidden * mask.unsqueeze(-1)


class DocumentHierarchicalReasoningModel(nn.Module):
    """Source-grounded H/L recurrent program inducer.

    H carries the slow operation/strategy state. L repeatedly attends to numeric
    evidence and refines operand pointers. Heads emit a complete executable
    program, not a metric class that is later expanded by a hand-written
    financial ontology.
    """

    def __init__(self, config: DocumentHRMConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        torch.manual_seed(config.seed)
        h = config.hidden_size
        e = config.embedding_dim
        if config.input_encoder_mode == "hash":
            self.token_embedding = nn.Embedding(config.vocab_size, e, padding_idx=0)
            self.question_encoder = nn.GRU(
                e,
                h // 2,
                batch_first=True,
                bidirectional=True,
            )
            self.candidate_text = nn.Sequential(
                nn.Linear(e * 2, h),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.LayerNorm(h),
            )
            self.semantic_question = None
            self.semantic_adapter_norm = None
            self.semantic_adapter_projection = None
        else:
            self.token_embedding = None
            self.question_encoder = None
            if config.semantic_adapter_mode == "layernorm_linear_l2_v1":
                self.semantic_adapter_norm = nn.LayerNorm(
                    config.semantic_encoder_size
                )
                self.semantic_adapter_projection = nn.Linear(
                    config.semantic_encoder_size,
                    config.semantic_embedding_size,
                )
            else:
                self.semantic_adapter_norm = None
                self.semantic_adapter_projection = None
            self.semantic_question = nn.Sequential(
                nn.Linear(config.semantic_embedding_size, h * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(h * 2, h),
            )
            self.candidate_text = nn.Sequential(
                nn.Linear(config.semantic_embedding_size, h * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(h * 2, h),
                nn.LayerNorm(h),
            )
        self.question_feature_encoder = nn.Sequential(
            nn.Linear(config.question_feature_size, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.question_norm = nn.LayerNorm(h)
        self.numeric_encoder = nn.Sequential(
            nn.Linear(config.numeric_feature_size, h),
            nn.GELU(),
            nn.Linear(h, h),
        )
        self.candidate_fusion = nn.Sequential(
            nn.Linear(h * 5, h * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(h * 2, h),
        )
        self.candidate_norm = nn.LayerNorm(h)

        block_args = (
            h,
            config.attention_heads,
            config.feed_forward_multiplier,
            config.dropout,
            config.financial_structure_mode == "legacy",
        )
        self.comparative_block = HRMBlock(*block_args)
        self.low_module = HRMBlock(*block_args)
        self.high_module = HRMBlock(*block_args)
        self.initial_high = nn.Parameter(torch.zeros(h))
        self.initial_low = nn.Parameter(torch.zeros(h))

        self.output_fusion = nn.Sequential(
            nn.Linear(h * 3, h * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(h * 2),
        )
        self.operation_head = nn.Linear(h * 2, len(OPERATIONS))
        self.pointer_queries = nn.ModuleList(
            nn.Linear(h * 2, h) for _ in range(config.max_program_operands)
        )
        if config.pointer_conditioning_mode in {
            "operation_sequential",
            "later_sequential",
        }:
            # Operand selection is relational: the required row/period for p2
            # depends on both the operation and the evidence selected for p1.
            # Keep these modules behind a config flag so schema-v4/v5/v6
            # checkpoints retain their exact parameter layout.
            self.pointer_operation_embedding = nn.Embedding(len(OPERATIONS), h)
            self.pointer_conditioner = nn.Sequential(
                nn.Linear(h * 2, h * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(h * 2, h * 2),
            )
            self.pointer_condition_norm = nn.LayerNorm(h * 2)
            self.pointer_slot_embedding = nn.Embedding(
                config.max_program_operands, h * 2
            )
        else:
            self.pointer_operation_embedding = None
            self.pointer_conditioner = None
            self.pointer_condition_norm = None
            self.pointer_slot_embedding = None
        self.pointer_teacher_forcing_weight = config.pointer_teacher_forcing
        self.pair_backbone_detached = config.pair_backbone_mode in {
            "detached",
            "normalized_detached",
            "warmup_end_to_end",
        }
        # This parameter is transformed through tanh and capped at
        # config.relevance_prior_max. It can no longer grow unbounded and replace
        # learned pointer reasoning with a hand-written relevance score.
        self.pointer_relevance_scale = nn.Parameter(
            torch.full((config.max_program_operands,), 0.25)
        )
        self.q_head = nn.Linear(h * 2, 2)
        if config.act_mode == "q_learning":
            # HRM initializes ACT values near zero so bootstrap targets do not
            # dominate the program loss at the beginning of training.
            nn.init.zeros_(self.q_head.weight)
            nn.init.constant_(self.q_head.bias, -5.0)
        # Initialize the detached pair reranker after every baseline module so
        # joint_pair preserves the exact seeded initialization of independent
        # HRM parameters (including q_head).
        if config.pointer_conditioning_mode == "joint_pair":
            rng_state = torch.random.get_rng_state()
            pair_size = max(32, h // 2)
            self.pair_left = nn.Linear(h, pair_size)
            self.pair_right = nn.Linear(h, pair_size)
            self.pair_operation_bilinear = nn.Parameter(
                torch.zeros(len(OPERATIONS), pair_size, pair_size)
            )
            self.pair_feature_weights = nn.Parameter(
                torch.zeros(len(OPERATIONS), PAIR_FEATURE_SIZE)
            )
            self.pair_relation_scale = nn.Parameter(torch.tensor(0.25))
            torch.random.set_rng_state(rng_state)
        else:
            self.pair_left = None
            self.pair_right = None
            self.pair_operation_bilinear = None
            self.pair_feature_weights = None
            self.pair_relation_scale = None

        if config.financial_structure_mode == "typed_hierarchy_v1":
            # These modules are appended after every legacy parameter so a
            # compatible warm start can copy the old HRM exactly.  Zero-valued
            # residual gates make the initial typed path behavior-preserving;
            # training then learns how strongly each structural signal matters.
            hierarchy_hidden = max(32, h // 2)
            self.financial_candidate_encoder = nn.Sequential(
                nn.Linear(FINANCIAL_CANDIDATE_FEATURE_SIZE, h),
                nn.GELU(),
                nn.Linear(h, h),
                nn.LayerNorm(h),
            )
            self.financial_candidate_scale = nn.Parameter(torch.zeros(()))
            self.financial_hierarchy_scorers = nn.ModuleList(
                nn.Linear(PAIR_FEATURE_SIZE, 1, bias=False)
                for _ in range(config.financial_hierarchy_layers)
            )
            self.financial_hierarchy_projections = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(h * 2, h),
                    nn.GELU(),
                    nn.Linear(h, h),
                )
                for _ in range(config.financial_hierarchy_layers)
            )
            self.financial_hierarchy_norms = nn.ModuleList(
                nn.LayerNorm(h)
                for _ in range(config.financial_hierarchy_layers)
            )
            self.financial_hierarchy_scales = nn.Parameter(
                torch.zeros(config.financial_hierarchy_layers)
            )
            self.financial_operation_embedding = nn.Embedding(len(OPERATIONS), h)
            self.financial_slot_embedding = nn.Embedding(
                config.max_program_operands, h
            )
            self.financial_pointer_conditioner = nn.Sequential(
                nn.Linear(h * 4, h * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(h * 2, h * 2),
            )
            self.financial_pointer_norm = nn.LayerNorm(h * 2)
            self.financial_pointer_scale = nn.Parameter(torch.zeros(()))
            self.financial_slot_scorer = nn.Sequential(
                nn.Linear(FINANCIAL_SLOT_FEATURE_SIZE, hierarchy_hidden),
                nn.GELU(),
                nn.Linear(hierarchy_hidden, 1),
            )
            self.financial_slot_scales = nn.Parameter(
                torch.zeros(config.max_program_operands)
            )
        else:
            self.financial_candidate_encoder = None
            self.financial_candidate_scale = None
            self.financial_hierarchy_scorers = None
            self.financial_hierarchy_projections = None
            self.financial_hierarchy_norms = None
            self.financial_hierarchy_scales = None
            self.financial_operation_embedding = None
            self.financial_slot_embedding = None
            self.financial_pointer_conditioner = None
            self.financial_pointer_norm = None
            self.financial_pointer_scale = None
            self.financial_slot_scorer = None
            self.financial_slot_scales = None
        if config.variable_sum_mode == "sum_n_v1":
            # The legacy operation classifier remains untouched. These V14-only
            # heads predict bounded set cardinality and candidate membership.
            self.sum_operand_count_head = nn.Linear(
                h * 2, config.max_program_operands - 1
            )
            self.sum_membership_query = nn.Linear(h * 2, h)
        else:
            self.sum_operand_count_head = None
            self.sum_membership_query = None

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def effective_relevance_scales(self) -> Tensor:
        return self.config.relevance_prior_max * torch.tanh(
            self.pointer_relevance_scale
        )

    def adapt_semantic_embeddings(self, values: Tensor) -> Tensor:
        """Project frozen raw encoder vectors into the stable HRM semantic ABI."""

        if self.semantic_adapter_projection is None:
            if values.shape[-1] != self.config.semantic_embedding_size:
                raise DocumentHRMError(
                    "Identity semantic adapter received the wrong dimension"
                )
            return values
        assert self.semantic_adapter_norm is not None
        if values.shape[-1] != self.config.semantic_encoder_size:
            raise DocumentHRMError(
                "Semantic adapter received the wrong raw encoder dimension"
            )
        projected = self.semantic_adapter_projection(
            self.semantic_adapter_norm(values)
        )
        return F.normalize(projected, p=2, dim=-1, eps=1e-12)

    def _encode_financial_hierarchy(
        self,
        candidates: Tensor,
        pair_features: Tensor,
        candidate_mask: Tensor,
    ) -> Tensor:
        if self.financial_hierarchy_scorers is None:
            return candidates
        assert self.financial_hierarchy_projections is not None
        assert self.financial_hierarchy_norms is not None
        assert self.financial_hierarchy_scales is not None
        # Equality relations are content-independent edges.  Missing metadata
        # produces zero relation bits and is therefore never pooled together.
        relation_indices = (0, 10, 11, 12, 13, 16, 17)
        relation_mask = pair_features[..., relation_indices].sum(dim=-1).gt(0)
        valid = (
            candidate_mask.unsqueeze(2).bool()
            & candidate_mask.unsqueeze(1).bool()
        )
        diagonal = torch.eye(
            valid.shape[-1], dtype=torch.bool, device=valid.device
        ).unsqueeze(0)
        relation_mask = relation_mask & valid & ~diagonal
        hidden = candidates
        for layer, scorer in enumerate(self.financial_hierarchy_scorers):
            logits = scorer(pair_features).squeeze(-1)
            logits = logits.masked_fill(~relation_mask, -1e9)
            weights = torch.softmax(logits, dim=-1) * relation_mask.to(logits.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            message = torch.einsum("bnm,bmh->bnh", weights, hidden)
            update = self.financial_hierarchy_norms[layer](
                self.financial_hierarchy_projections[layer](
                    torch.cat((hidden, message), dim=-1)
                )
            )
            hidden = hidden + torch.tanh(
                self.financial_hierarchy_scales[layer]
            ) * update
            hidden = hidden * candidate_mask.unsqueeze(-1)
        return hidden

    def encode_inputs(self, batch: EncodedBatch) -> tuple[Tensor, Tensor]:
        if self.config.input_encoder_mode == "hash":
            assert self.token_embedding is not None and self.question_encoder is not None
            question_embedded = self.token_embedding(batch.question_ids)
            question_lengths = batch.question_mask.sum(dim=1).long().clamp_min(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                question_embedded,
                question_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.question_encoder(packed)
            question = torch.cat((hidden[-2], hidden[-1]), dim=-1)
            b, n, t = batch.candidate_ids.shape
            candidate_embedded = self.token_embedding(
                batch.candidate_ids.reshape(b * n, t)
            )
            token_mask = batch.candidate_token_mask.reshape(b * n, t).unsqueeze(-1)
            mean = (candidate_embedded * token_mask).sum(dim=1) / token_mask.sum(
                dim=1
            ).clamp_min(1.0)
            masked = candidate_embedded.masked_fill(token_mask.eq(0), -1e4)
            maximum = masked.max(dim=1).values
            maximum = torch.where(
                torch.isfinite(maximum), maximum, torch.zeros_like(maximum)
            )
            text = self.candidate_text(torch.cat((mean, maximum), dim=-1)).reshape(
                b, n, -1
            )
        else:
            if batch.question_semantic is None or batch.candidate_semantic is None:
                raise DocumentHRMError("Semantic embeddings are missing from the batch")
            assert self.semantic_question is not None
            question_semantic = self.adapt_semantic_embeddings(
                batch.question_semantic
            )
            candidate_semantic = self.adapt_semantic_embeddings(
                batch.candidate_semantic
            )
            question = self.semantic_question(question_semantic)
            text = self.candidate_text(candidate_semantic)
        question = self.question_norm(
            question + self.question_feature_encoder(batch.question_features)
        )
        b, n = batch.candidate_mask.shape
        numeric = self.numeric_encoder(batch.numeric_features)
        if self.config.financial_structure_mode == "typed_hierarchy_v1":
            if batch.financial_candidate_features is None:
                raise DocumentHRMError(
                    "Financial candidate features are missing from the batch"
                )
            assert self.financial_candidate_encoder is not None
            assert self.financial_candidate_scale is not None
            structure = self.financial_candidate_encoder(
                batch.financial_candidate_features
            )
            numeric = numeric + torch.tanh(
                self.financial_candidate_scale
            ) * structure
        q_bias = question.unsqueeze(1).expand(-1, n, -1)
        candidates = self.candidate_norm(
            self.candidate_fusion(
                torch.cat(
                    (text, numeric, q_bias, text * q_bias, torch.abs(text - q_bias)),
                    dim=-1,
                )
            )
        )
        candidates = candidates * batch.candidate_mask.unsqueeze(-1)
        candidates = self.comparative_block(candidates, batch.candidate_mask)
        if self.config.financial_structure_mode == "typed_hierarchy_v1":
            if batch.pair_features is None:
                raise DocumentHRMError(
                    "Pair features are missing from the typed hierarchy batch"
                )
            candidates = self._encode_financial_hierarchy(
                candidates,
                batch.pair_features,
                batch.candidate_mask,
            )
        return question, candidates

    def initial_state(self, candidate_mask: Tensor) -> DocumentHRMState:
        batch_size, candidate_count = candidate_mask.shape
        shape = (batch_size, candidate_count, -1)
        high = self.initial_high.view(1, 1, -1).expand(*shape)
        low = self.initial_low.view(1, 1, -1).expand(*shape)
        high = high * candidate_mask.unsqueeze(-1)
        low = low * candidate_mask.unsqueeze(-1)
        return DocumentHRMState(high, low)

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _low_update(
        self, low: Tensor, high: Tensor, question: Tensor, candidates: Tensor, mask: Tensor
    ) -> Tensor:
        inputs = low + high + candidates + question.unsqueeze(1)
        return self.low_module(inputs, mask)

    def _high_update(
        self, high: Tensor, low: Tensor, question: Tensor, candidates: Tensor, mask: Tensor
    ) -> Tensor:
        if self.config.hierarchical_input_mode == "legacy":
            inputs = high + low + candidates + question.unsqueeze(1)
        else:
            # Strict HRM coupling: H receives the refined L state, not a second
            # direct copy of the source evidence/question. L remains the fast
            # evidence processor and H becomes the slower strategy state.
            inputs = high + low
        return self.high_module(inputs, mask)

    def forward_segment(
        self,
        batch: EncodedBatch,
        state: DocumentHRMState | None = None,
        encoded: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[DocumentHRMForward, tuple[Tensor, Tensor]]:
        question, candidates = encoded or self.encode_inputs(batch)
        if state is None:
            state = self.initial_state(batch.candidate_mask)
        high, low = state.high, state.low
        total_steps = self.config.high_cycles * self.config.low_steps

        with torch.no_grad():
            for step in range(max(0, total_steps - 1)):
                low = self._low_update(
                    low, high, question, candidates, batch.candidate_mask
                )
                if (step + 1) % self.config.low_steps == 0:
                    high = self._high_update(
                        high, low, question, candidates, batch.candidate_mask
                    )

        low = self._low_update(low, high, question, candidates, batch.candidate_mask)
        high = self._high_update(high, low, question, candidates, batch.candidate_mask)

        high_pool = self._masked_mean(high, batch.candidate_mask)
        low_pool = self._masked_mean(low, batch.candidate_mask)
        fused = self.output_fusion(torch.cat((high_pool, low_pool, question), dim=-1))
        operation_logits = self.operation_head(fused)
        reasoned_candidates = self.candidate_norm(candidates + high + low)
        pointer_logits = []
        relevance = batch.numeric_features[:, :, 4]
        operation_context: Tensor | None = None
        previous_pointer_context: Tensor | None = None
        financial_operation_context: Tensor | None = None
        if self.config.pointer_conditioning_mode in {
            "operation_sequential",
            "later_sequential",
        }:
            assert self.pointer_operation_embedding is not None
            operation_probabilities = torch.softmax(operation_logits, dim=-1)
            operation_context = operation_probabilities @ (
                self.pointer_operation_embedding.weight
            )
            previous_pointer_context = torch.zeros_like(operation_context)
        if self.config.financial_structure_mode == "typed_hierarchy_v1":
            if batch.financial_slot_features is None:
                raise DocumentHRMError(
                    "Financial slot features are missing from the batch"
                )
            assert self.financial_operation_embedding is not None
            financial_operation_context = torch.softmax(
                operation_logits, dim=-1
            ) @ self.financial_operation_embedding.weight
        for slot, pointer_query in enumerate(self.pointer_queries):
            pointer_input = fused
            condition_this_slot = (
                operation_context is not None
                and previous_pointer_context is not None
                and not (
                    self.config.pointer_conditioning_mode == "later_sequential"
                    and slot == 0
                )
            )
            if condition_this_slot:
                assert self.pointer_conditioner is not None
                assert self.pointer_condition_norm is not None
                assert self.pointer_slot_embedding is not None
                slot_ids = torch.full(
                    (fused.shape[0],), slot, dtype=torch.long, device=fused.device
                )
                condition = self.pointer_conditioner(
                    torch.cat((operation_context, previous_pointer_context), dim=-1)
                )
                pointer_input = self.pointer_condition_norm(
                    fused + condition + self.pointer_slot_embedding(slot_ids)
                )
            if financial_operation_context is not None:
                assert self.financial_slot_embedding is not None
                assert self.financial_pointer_conditioner is not None
                assert self.financial_pointer_norm is not None
                assert self.financial_pointer_scale is not None
                slot_ids = torch.full(
                    (fused.shape[0],), slot, dtype=torch.long, device=fused.device
                )
                slot_context = self.financial_slot_embedding(slot_ids)
                condition = self.financial_pointer_conditioner(
                    torch.cat(
                        (fused, financial_operation_context, slot_context),
                        dim=-1,
                    )
                )
                pointer_input = self.financial_pointer_norm(
                    fused
                    + torch.tanh(self.financial_pointer_scale) * condition
                )
            query = pointer_query(pointer_input)
            scores = torch.einsum("bd,bnd->bn", query, reasoned_candidates) / math.sqrt(
                reasoned_candidates.shape[-1]
            )
            scale = self.effective_relevance_scales()[slot]
            scores = scores + scale * relevance
            if financial_operation_context is not None:
                assert self.financial_slot_scorer is not None
                assert self.financial_slot_scales is not None
                slot_bias = self.financial_slot_scorer(
                    batch.financial_slot_features[:, slot]
                ).squeeze(-1)
                scores = scores + torch.tanh(
                    self.financial_slot_scales[slot]
                ) * slot_bias
            masked_scores = scores.masked_fill(batch.candidate_mask.eq(0), -1e9)
            pointer_logits.append(masked_scores)
            if operation_context is not None:
                pointer_probabilities = torch.softmax(masked_scores, dim=-1)
                predicted_context = torch.einsum(
                    "bn,bnd->bd", pointer_probabilities, reasoned_candidates
                )
                previous_pointer_context = predicted_context
                if self.training and batch.pointer_targets is not None:
                    targets = batch.pointer_targets[:, slot]
                    valid = targets.ge(0)
                    if valid.any() and self.pointer_teacher_forcing_weight > 0:
                        safe_targets = targets.clamp_min(0)
                        rows = torch.arange(
                            reasoned_candidates.shape[0], device=fused.device
                        )
                        gold_context = reasoned_candidates[rows, safe_targets]
                        weight = self.pointer_teacher_forcing_weight
                        blended = (
                            weight * gold_context + (1.0 - weight) * predicted_context
                        )
                        previous_pointer_context = torch.where(
                            valid.unsqueeze(-1), blended, predicted_context
                        )
        operand_count_logits: Tensor | None = None
        sum_membership_logits: Tensor | None = None
        if self.config.variable_sum_mode == "sum_n_v1":
            assert self.sum_operand_count_head is not None
            assert self.sum_membership_query is not None
            operand_count_logits = self.sum_operand_count_head(fused)
            sum_query = self.sum_membership_query(fused)
            sum_membership_logits = torch.einsum(
                "bd,bnd->bn", sum_query, reasoned_candidates
            ) / math.sqrt(reasoned_candidates.shape[-1])
            sum_membership_logits = sum_membership_logits.masked_fill(
                batch.candidate_mask.eq(0), -1e9
            )
        pair_logits: Tensor | None = None
        select_pointer_logits: Tensor | None = None
        if self.config.pointer_conditioning_mode == "joint_pair":
            if batch.pair_features is None:
                raise DocumentHRMError("Pair features are missing from the batch")
            assert self.pair_left is not None and self.pair_right is not None
            assert self.pair_operation_bilinear is not None
            assert self.pair_feature_weights is not None
            assert self.pair_relation_scale is not None
            pair_candidates = (
                reasoned_candidates.detach()
                if self.pair_backbone_detached
                else reasoned_candidates
            )
            left = self.pair_left(pair_candidates)
            right = self.pair_right(pair_candidates)
            bilinear = self.pair_operation_bilinear
            if self.config.pair_backbone_mode in {
                "normalized_detached",
                "normalized_end_to_end",
            }:
                left = F.normalize(left, dim=-1)
                right = F.normalize(right, dim=-1)
                bilinear_norm = torch.linalg.vector_norm(
                    bilinear, dim=(1, 2), keepdim=True
                ).clamp_min(1.0)
                bilinear = bilinear / bilinear_norm
            relation = torch.einsum(
                "bnr,ors,bms->bonm",
                left,
                bilinear,
                right,
            )
            if self.config.pair_backbone_mode not in {
                "normalized_detached",
                "normalized_end_to_end",
            }:
                relation = relation / math.sqrt(left.shape[-1])
            structured = torch.einsum(
                "bnmk,ok->bonm",
                batch.pair_features,
                self.pair_feature_weights,
            )
            unary_left = pointer_logits[0]
            unary_right = pointer_logits[1]
            if self.pair_backbone_detached:
                unary_left = unary_left.detach()
                unary_right = unary_right.detach()
            unary_pair = unary_left.unsqueeze(2) + unary_right.unsqueeze(1)
            pair_logits = (
                unary_pair.unsqueeze(1)
                + torch.tanh(self.pair_relation_scale) * relation
                + structured
            )
            valid_pair = (
                batch.candidate_mask.unsqueeze(2)
                * batch.candidate_mask.unsqueeze(1)
            ).bool()
            diagonal = torch.eye(
                valid_pair.shape[-1], dtype=torch.bool, device=fused.device
            ).unsqueeze(0)
            valid_pair = valid_pair & ~diagonal
            if (
                self.config.financial_structure_mode == "typed_hierarchy_v1"
                and batch.financial_same_physical_mask is not None
            ):
                valid_pair = valid_pair & ~batch.financial_same_physical_mask.bool()
            pair_logits = pair_logits.masked_fill(
                ~valid_pair.unsqueeze(1), -1e9
            )
            select_id = OPERATION_TO_ID["SELECT"]
            select_residual = relation[:, select_id].diagonal(dim1=-2, dim2=-1)
            select_pointer_logits = (
                pointer_logits[0]
                + torch.tanh(self.pair_relation_scale) * select_residual
            ).masked_fill(batch.candidate_mask.eq(0), -1e9)
        q_input = (
            fused
            if self.config.hierarchical_input_mode == "legacy"
            else torch.cat((high_pool, high_pool), dim=-1)
        )
        return (
            DocumentHRMForward(
                state=DocumentHRMState(high, low),
                operation_logits=operation_logits,
                pointer_logits=torch.stack(pointer_logits, dim=1),
                q_values=self.q_head(q_input),
                pair_logits=pair_logits,
                select_pointer_logits=select_pointer_logits,
                operand_count_logits=operand_count_logits,
                sum_membership_logits=sum_membership_logits,
            ),
            (question, candidates),
        )


def _semantic_adapter_state_sha256(
    model: DocumentHierarchicalReasoningModel,
) -> str:
    digest = sha256()
    found = False
    for name, tensor in sorted(model.state_dict().items()):
        if not name.startswith("semantic_adapter_"):
            continue
        found = True
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest() if found else "identity"


SEMANTIC_CHECKPOINT_INTEGRITY_SCHEMA = "semantic-checkpoint-integrity-v2"
SEMANTIC_ADAPTER_IMPLEMENTATION_VERSION = "document-hrm-adapter-v1"


def _effective_semantic_fingerprint(
    config: DocumentHRMConfig,
    *,
    frontend_fingerprint: str,
    adapter_state_sha256: str,
) -> str:
    payload = {
        "schema": SEMANTIC_CHECKPOINT_INTEGRITY_SCHEMA,
        "frontend_fingerprint": frontend_fingerprint,
        "adapter_state_sha256": adapter_state_sha256,
        "adapter_mode": config.semantic_adapter_mode,
        "adapter_implementation": SEMANTIC_ADAPTER_IMPLEMENTATION_VERSION,
        "encoder_output_size": config.semantic_encoder_size,
        "hrm_semantic_size": config.semantic_embedding_size,
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


class DocumentHRMBundle:
    def __init__(
        self,
        model: DocumentHierarchicalReasoningModel,
        tokenizer: HashSequenceTokenizer | None = None,
        metadata: dict[str, object] | None = None,
        semantic_encoder: SemanticEmbeddingCache | None = None,
        pointer_bundle: "DocumentHRMBundle | None" = None,
        pointer_secondary_bundle: "DocumentHRMBundle | None" = None,
        program_reranker: ProgramRerankerBundle | None = None,
    ) -> None:
        self.model = model
        self.metadata = metadata or {}
        self.pointer_bundle = pointer_bundle
        self.pointer_secondary_bundle = pointer_secondary_bundle
        self.program_reranker = program_reranker
        if model.config.input_encoder_mode == "hash":
            self.tokenizer = tokenizer or HashSequenceTokenizer(model.config.vocab_size)
            self.semantic_encoder = None
        else:
            self.tokenizer = None
            self.semantic_encoder = semantic_encoder or SemanticEmbeddingCache(
                model.config.semantic_model_name,
                model.config.semantic_encoder_size,
                DEFAULT_SEMANTIC_CACHE_DIR,
                spec=_semantic_encoder_spec_from_config(model.config),
            )

    def attach_program_reranker(
        self, reranker: ProgramRerankerBundle
    ) -> "DocumentHRMBundle":
        """Attach a reranker sidecar without changing or re-saving HRM weights."""

        required_flags = (
            "typed_program_synthesis_enabled",
            "financial_identity_catalog_enabled",
            "candidate_recall_v12_4_enabled",
        )
        missing_flags = [
            name
            for name in required_flags
            if reranker.metadata.get(name) is True
            and self.metadata.get(name) is not True
        ]
        if missing_flags:
            raise DocumentHRMError(
                "Program reranker runtime mismatch; enable required flags: "
                + ", ".join(missing_flags)
            )
        self.program_reranker = reranker
        self.metadata = dict(self.metadata)
        self.metadata.update(
            {
                "program_reranker_enabled": True,
                "program_reranker_architecture": reranker.metadata.get(
                    "architecture", "v11-safe-residual-frozen-hrm-reranker"
                ),
                "program_reranker_feature_schema": reranker.metadata.get(
                    "feature_schema"
                ),
                "program_reranker_confidence_schema": (
                    "v11-selected-program-confidence-head"
                ),
                "program_reranker_calibration_mode": (
                    "v11-company-disjoint-safe-residual"
                ),
                "program_reranker_thresholds_by_operation": reranker.metadata.get(
                    "confidence_thresholds_by_operation", {}
                ),
                "abstention_target_precision": reranker.metadata.get(
                    "target_precision", 0.90
                ),
                "abstention_target_met": reranker.metadata.get(
                    "calibration_target_met", False
                ),
            }
        )
        return self

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def parameter_count(self) -> int:
        count = self.model.parameter_count()
        if self.pointer_bundle is not None:
            count += self.pointer_bundle.model.parameter_count()
        if self.pointer_secondary_bundle is not None:
            count += self.pointer_secondary_bundle.model.parameter_count()
        return count

    def predict(
        self,
        question: str,
        candidates: Sequence[NumericCandidate],
        *,
        max_segments: int | None = None,
        min_segments: int = 1,
    ) -> ProgramPrediction:
        if not candidates:
            raise DocumentHRMError("No numeric evidence candidates were extracted")
        max_segments = max_segments or self.model.config.max_segments
        configured_minimum = self.model.config.minimum_halt_segments
        min_segments = max(min_segments, configured_minimum)
        min_segments = max(1, min(min_segments, max_segments))
        final, used_segments = self._predict_single_output(
            question, candidates, max_segments, min_segments
        )
        pointer_output = None
        pointer_secondary_output = None
        if self.pointer_bundle is not None:
            pointer_max = self.pointer_bundle.model.config.max_segments
            pointer_min = self.pointer_bundle.model.config.minimum_halt_segments
            pointer_output, _ = self.pointer_bundle._predict_single_output(
                question, candidates, pointer_max, pointer_min
            )
        if self.pointer_secondary_bundle is not None:
            pointer_max = self.pointer_secondary_bundle.model.config.max_segments
            pointer_min = self.pointer_secondary_bundle.model.config.minimum_halt_segments
            pointer_secondary_output, _ = (
                self.pointer_secondary_bundle._predict_single_output(
                    question, candidates, pointer_max, pointer_min
                )
            )
        return self._decode_prediction(
            final,
            0,
            used_segments,
            pointer_output=pointer_output,
            pointer_secondary_output=pointer_secondary_output,
            question=question,
            candidates=candidates,
        )

    def _predict_single_output(
        self,
        question: str,
        candidates: Sequence[NumericCandidate],
        max_segments: int,
        min_segments: int,
    ) -> tuple[DocumentHRMForward, int]:
        batch = encode_inference_batch(
            question,
            candidates,
            self.tokenizer,
            self.semantic_encoder,
            self.model.config,
            self.device,
        )
        self.model.eval()
        state: DocumentHRMState | None = None
        encoded: tuple[Tensor, Tensor] | None = None
        final: DocumentHRMForward | None = None
        used_segments = max_segments
        with torch.no_grad():
            for segment in range(1, max_segments + 1):
                final, encoded = self.model.forward_segment(batch, state, encoded)
                state = final.state.detach()
                if segment >= min_segments and final.q_values[0, 0] >= final.q_values[0, 1]:
                    used_segments = segment
                    break
        assert final is not None
        return final, used_segments

    def predict_many(
        self,
        examples: Sequence[ProgramExample],
        *,
        max_segments: int | None = None,
    ) -> list[ProgramPrediction]:
        """Batch evaluation at the final reasoning segment.

        The official HRM evaluation path runs the maximum number of ACT steps.
        Doing the same here also makes holdout evaluation substantially faster
        than invoking single-example adaptive inference repeatedly.
        """
        if not examples:
            return []
        segments = max_segments or self.model.config.max_segments
        final = self._predict_many_output(examples, segments)
        pointer_output = None
        pointer_secondary_output = None
        if self.pointer_bundle is not None:
            pointer_segments = self.pointer_bundle.model.config.max_segments
            pointer_output = self.pointer_bundle._predict_many_output(
                examples, pointer_segments
            )
        if self.pointer_secondary_bundle is not None:
            pointer_segments = self.pointer_secondary_bundle.model.config.max_segments
            pointer_secondary_output = (
                self.pointer_secondary_bundle._predict_many_output(
                    examples, pointer_segments
                )
            )
        return [
            self._decode_prediction(
                final,
                row,
                segments,
                pointer_output=pointer_output,
                pointer_secondary_output=pointer_secondary_output,
                question=example.question,
                candidates=example.candidates,
            )
            for row, example in enumerate(examples)
        ]

    def _predict_many_output(
        self,
        examples: Sequence[ProgramExample],
        segments: int,
    ) -> DocumentHRMForward:
        batch = encode_examples(
            examples,
            self.tokenizer,
            self.semantic_encoder,
            self.model.config,
            self.device,
        )
        self.model.eval()
        state: DocumentHRMState | None = None
        encoded: tuple[Tensor, Tensor] | None = None
        final: DocumentHRMForward | None = None
        with torch.no_grad():
            for _segment in range(segments):
                final, encoded = self.model.forward_segment(batch, state, encoded)
                state = final.state.detach()
        assert final is not None
        return final

    def _decode_prediction(
        self,
        output: DocumentHRMForward,
        row: int,
        segments: int,
        *,
        pointer_output: DocumentHRMForward | None = None,
        pointer_secondary_output: DocumentHRMForward | None = None,
        question: str = "",
        candidates: Sequence[NumericCandidate] = (),
    ) -> ProgramPrediction:
        confidence_schema = str(
            self.metadata.get("confidence_schema", "raw-neural-v1")
        )
        use_grounded_confidence = confidence_schema == "grounded-constraints-v2"
        operation_probs = torch.softmax(output.operation_logits[row], dim=-1)
        neural_operation_id = int(operation_probs.argmax().item())
        variable_sum_enabled = (
            self.model.config.variable_sum_mode == "sum_n_v1"
        )
        sum_family_probability = float(
            operation_probs[list(SUM_FAMILY_OPERATION_IDS)].sum().item()
        )
        best_sum_operation_id = max(
            SUM_FAMILY_OPERATION_IDS,
            key=lambda index: float(operation_probs[index].item()),
        )
        non_sum_operation_ids = tuple(
            index
            for index in range(len(OPERATIONS))
            if index not in SUM_FAMILY_OPERATION_IDS
        )
        best_non_sum_operation_id = max(
            non_sum_operation_ids,
            key=lambda index: float(operation_probs[index].item()),
        )
        if (
            variable_sum_enabled
            and sum_family_probability
            >= float(operation_probs[best_non_sum_operation_id].item())
        ):
            operation = SUM_N_OPERATION
            operation_id = best_sum_operation_id
        else:
            operation_id = neural_operation_id
            operation = OPERATIONS[operation_id]
        grounding_constraints: list[str] = []
        operation_was_constrained = False
        if question:
            constrained_operation = _apply_symbolic_operation_constraint(
                question, operation
            )
            if variable_sum_enabled and is_sum_operation(constrained_operation):
                constrained_operation = SUM_N_OPERATION
            if constrained_operation != operation:
                grounding_constraints.append("operation_wording")
                operation_was_constrained = True
            operation = constrained_operation
            operation_id = (
                best_sum_operation_id
                if operation == SUM_N_OPERATION
                else OPERATION_TO_ID[operation]
            )
        operand_count_confidence: float | None = None
        if operation == SUM_N_OPERATION:
            if (
                output.operand_count_logits is None
                or output.sum_membership_logits is None
            ):
                raise DocumentHRMError(
                    "SUM_N decoding requires count and membership heads"
                )
            count_probabilities = torch.softmax(
                output.operand_count_logits[row], dim=-1
            )
            explicit_count = _explicit_sum_operand_count(question)
            if explicit_count is not None:
                if explicit_count > self.model.config.max_program_operands:
                    raise DocumentHRMError(
                        f"Explicit SUM_N arity {explicit_count} exceeds configured "
                        f"maximum {self.model.config.max_program_operands}"
                    )
                arity = explicit_count
                operand_count_confidence = 1.0
                grounding_constraints.append("explicit_sum_operand_count")
            else:
                count_index = int(count_probabilities.argmax().item())
                arity = count_index + 2
                operand_count_confidence = float(
                    count_probabilities[count_index].item()
                )
                grounding_constraints.append("predicted_sum_operand_count")
        else:
            arity = OPERATION_ARITY[operation]
        if candidates and len(candidates) < arity:
            raise DocumentHRMError(
                f"{operation} requires {arity} distinct evidence values, "
                f"but only {len(candidates)} candidate(s) were available"
            )
        operands: list[int] = []
        pointer_confidences: list[float] = []
        neural_pointer_confidences: list[float] = []
        first_unfilled_slot = 0
        same_physical_pairs: list[list[bool]] | None = None
        if operation == SUM_N_OPERATION and candidates:
            same_physical_pairs = financial_same_physical_pairs(candidates)
            representatives: list[int] = []
            for index in range(len(candidates)):
                if any(
                    same_physical_pairs[index][existing]
                    for existing in representatives
                ):
                    continue
                representatives.append(index)
            if len(representatives) < arity:
                raise DocumentHRMError(
                    f"SUM_N requires {arity} physically unique evidence values, "
                    f"but only {len(representatives)} were available"
                )
        multi_operand_allowed: set[int] | None = None
        if operation in MULTI_OPERAND_OPERATIONS:
            compatible = _period_compatible_indices(
                question, operation, candidates, arity
            )
            if len(compatible) >= arity:
                scoped = _prefer_reporting_entity_indices(
                    question, candidates, compatible
                )
                multi_operand_allowed = (
                    scoped if len(scoped) >= arity else compatible
                )
                grounding_constraints.append("multi_operand_period_scope")
        if (
            output.pair_logits is not None
            and arity >= 2
            and operation not in MULTI_OPERAND_OPERATIONS
        ):
            pair_scores = output.pair_logits[row, operation_id].clone()
            # The model masks self-pairs with a large negative finite value.
            # Subsequent symbolic masks can make every pair equally negative,
            # allowing argmax to select a diagonal cell again.  A diagonal
            # represents reusing one evidence candidate as both operands and
            # is structurally invalid, so keep it at true negative infinity
            # throughout decoding.
            pair_diagonal = torch.eye(
                pair_scores.shape[-1],
                dtype=torch.bool,
                device=pair_scores.device,
            )
            pair_forbidden = pair_diagonal
            if (
                self.model.config.financial_structure_mode
                == "typed_hierarchy_v1"
            ):
                observed_duplicate_physical = torch.tensor(
                    financial_same_physical_pairs(candidates),
                    dtype=torch.bool,
                    device=pair_scores.device,
                )
                duplicate_physical = torch.zeros_like(pair_forbidden)
                observed_count = min(
                    len(candidates), duplicate_physical.shape[-1]
                )
                duplicate_physical[
                    :observed_count, :observed_count
                ] = observed_duplicate_physical[
                    :observed_count, :observed_count
                ]
                pair_forbidden = pair_forbidden | duplicate_physical
            pair_scores = pair_scores.masked_fill(pair_forbidden, -torch.inf)
            unconstrained_pair_scores = pair_scores.clone()
            raw_pair_probabilities = torch.softmax(pair_scores.flatten(), dim=-1)
            has_semantic_pair_constraint = False
            compatible = _period_compatible_indices(
                question, operation, candidates, arity
            )
            if compatible:
                grounding_constraints.append("period")
                allowed = torch.zeros(
                    pair_scores.shape[-1], dtype=torch.bool, device=pair_scores.device
                )
                allowed[list(compatible)] = True
                pair_scores = pair_scores.masked_fill(
                    ~(allowed[:, None] & allowed[None, :]), -1e9
                )
                if operation in {"SUBTRACT", "PERCENT_CHANGE"}:
                    requested_years = _requested_years(question)
                    if len(requested_years) >= 2:
                        candidate_years = [
                            _candidate_period_years(candidate, question)
                            for candidate in candidates
                        ]
                        for first in compatible:
                            for second in compatible:
                                if candidate_years[first] & candidate_years[second]:
                                    pair_scores[first, second] = -1e9
            role_compatible = _role_compatible_pair_indices(
                question,
                operation,
                candidates,
                set(range(len(candidates))),
            )
            if role_compatible is not None:
                left_allowed, right_allowed = role_compatible
                if compatible:
                    period_left = left_allowed & compatible
                    period_right = right_allowed & compatible
                    if period_left and period_right:
                        left_allowed, right_allowed = period_left, period_right
                    else:
                        # Explicit semantic anchors are stronger than sparse or
                        # unrelated period metadata. Applying both disjoint
                        # masks would otherwise make every pair impossible.
                        pair_scores = unconstrained_pair_scores.clone()
                        if "period" in grounding_constraints:
                            grounding_constraints.remove("period")
                scoped_left = _prefer_reporting_entity_indices(
                    question, candidates, left_allowed
                )
                scoped_right = _prefer_reporting_entity_indices(
                    question, candidates, right_allowed
                )
                if scoped_left != left_allowed or scoped_right != right_allowed:
                    if "scope" not in grounding_constraints:
                        grounding_constraints.append("scope")
                    left_allowed, right_allowed = scoped_left, scoped_right
                grounding_constraints.append("operand_role")
                has_semantic_pair_constraint = True
                role_mask = torch.zeros_like(pair_scores, dtype=torch.bool)
                role_mask[
                    torch.tensor(sorted(left_allowed), device=pair_scores.device)[:, None],
                    torch.tensor(sorted(right_allowed), device=pair_scores.device)[None, :],
                ] = True
                pair_scores = pair_scores.masked_fill(~role_mask, -1e9)
            temporal_allowed = _temporal_role_compatible_indices(
                question,
                operation,
                candidates,
                set(range(len(candidates))),
            )
            if temporal_allowed is not None:
                if compatible:
                    period_temporal = temporal_allowed & compatible
                    if len(period_temporal) >= 2:
                        temporal_allowed = period_temporal
                    else:
                        pair_scores = unconstrained_pair_scores.clone()
                        if "period" in grounding_constraints:
                            grounding_constraints.remove("period")
                scoped_temporal = _prefer_reporting_entity_indices(
                    question, candidates, temporal_allowed
                )
                if scoped_temporal != temporal_allowed:
                    if "scope" not in grounding_constraints:
                        grounding_constraints.append("scope")
                    temporal_allowed = scoped_temporal
                grounding_constraints.append("metric")
                has_semantic_pair_constraint = True
                temporal_mask = torch.zeros_like(pair_scores, dtype=torch.bool)
                allowed_tensor = torch.tensor(
                    sorted(temporal_allowed), device=pair_scores.device
                )
                temporal_mask[allowed_tensor[:, None], allowed_tensor[None, :]] = True
                pair_scores = pair_scores.masked_fill(~temporal_mask, -1e9)
            column_allowed = _column_scope_compatible_indices(
                question, candidates, set(range(len(candidates)))
            )
            if len(column_allowed) >= arity and len(column_allowed) < len(candidates):
                scope_mask = torch.zeros(pair_scores.shape[-1], dtype=torch.bool, device=pair_scores.device)
                scope_mask[list(column_allowed)] = True
                pair_scores = pair_scores.masked_fill(~(scope_mask[:, None] & scope_mask[None, :]), -1e9)
                grounding_constraints.append("column_scope")
                has_semantic_pair_constraint = True
            pair_scores = pair_scores.masked_fill(pair_forbidden, -torch.inf)
            pair_probabilities = torch.softmax(pair_scores.flatten(), dim=-1)
            pair_index = int(pair_probabilities.argmax().item())
            pair_width = pair_scores.shape[-1]
            operands.extend((pair_index // pair_width, pair_index % pair_width))
            # Retain the original score for diagnostics/backward compatibility.
            # New checkpoints calibrate the masked, grounded score on the fixed
            # real holdout instead of reusing a threshold learned for raw logits.
            joint_confidence = float(raw_pair_probabilities[pair_index].item())
            neural_slot_confidence = math.sqrt(max(joint_confidence, 1e-8))
            effective_joint_confidence = float(pair_probabilities[pair_index].item())
            effective_slot_confidence = math.sqrt(
                max(effective_joint_confidence, 1e-8)
            )
            selected_slot_confidence = (
                effective_slot_confidence
                if use_grounded_confidence and has_semantic_pair_constraint
                else neural_slot_confidence
            )
            pointer_confidences.extend(
                (selected_slot_confidence, selected_slot_confidence)
            )
            neural_pointer_confidences.extend(
                (neural_slot_confidence, neural_slot_confidence)
            )
            first_unfilled_slot = 2
        for slot in range(first_unfilled_slot, arity):
            use_membership = operation in COMMUTATIVE and arity > 2
            use_sum_membership = (
                operation == SUM_N_OPERATION
                and output.sum_membership_logits is not None
            )
            logits = (
                output.sum_membership_logits[row]
                if use_sum_membership
                else
                _commutative_pointer_membership_logits(
                    output.pointer_logits[row], arity
                )
                if use_membership
                else _operation_pointer_logits(
                    output, row, slot, operation
                )
            ).clone()
            if multi_operand_allowed is not None:
                allowed_mask = torch.zeros_like(logits, dtype=torch.bool)
                allowed_mask[list(multi_operand_allowed)] = True
                logits = logits.masked_fill(~allowed_mask, -1e9)
            if pointer_output is not None:
                primary_weight = _metadata_float(
                    self.metadata, "pointer_primary_weight", 0.60
                )
                auxiliary_weight = _metadata_float(
                    self.metadata, "pointer_auxiliary_weight", 0.40
                )
                secondary_weight = (
                    _metadata_float(
                        self.metadata, "pointer_secondary_weight", 0.0
                    )
                    if pointer_secondary_output is not None
                    else 0.0
                )
                relevance_weight = _metadata_float(
                    self.metadata, "pointer_relevance_weight", 1.0
                )
                weight_total = max(
                    1e-8, primary_weight + auxiliary_weight + secondary_weight
                )
                logits = (
                    primary_weight * torch.log_softmax(logits, dim=-1)
                    + auxiliary_weight
                    * torch.log_softmax(
                        pointer_output.sum_membership_logits[row]
                        if (
                            use_sum_membership
                            and pointer_output.sum_membership_logits is not None
                        )
                        else
                        _commutative_pointer_membership_logits(
                            pointer_output.pointer_logits[row], arity
                        )
                        if use_membership
                        else _operation_pointer_logits(
                            pointer_output, row, slot, operation
                        ),
                        dim=-1,
                    )
                ) / weight_total
                if pointer_secondary_output is not None and secondary_weight:
                    logits = logits + (
                        secondary_weight
                        / weight_total
                        * torch.log_softmax(
                            pointer_secondary_output.sum_membership_logits[row]
                            if (
                                use_sum_membership
                                and pointer_secondary_output.sum_membership_logits
                                is not None
                            )
                            else
                            _commutative_pointer_membership_logits(
                                pointer_secondary_output.pointer_logits[row],
                                arity,
                            )
                            if use_membership
                            else _operation_pointer_logits(
                                pointer_secondary_output,
                                row,
                                slot,
                                operation,
                            ),
                            dim=-1,
                        )
                    )
                if candidates and relevance_weight:
                    relevance = torch.zeros_like(logits)
                    relevance[: len(candidates)] = torch.tensor(
                        [candidate.relevance for candidate in candidates],
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                    logits = logits + relevance_weight * relevance
            neural_logits = logits.clone()
            if arity == 1:
                metric_allowed = _metric_compatible_indices(
                    question, candidates, set(range(len(candidates)))
                )
                period_allowed = _period_compatible_indices(
                    question, operation, candidates, arity
                )
                (
                    unary_allowed,
                    metric_constraint_applied,
                    period_constraint_applied,
                ) = _combine_unary_grounding_constraints(
                    metric_allowed, period_allowed
                )
                narrowed_unary = _narrow_unary_period_by_row_order(
                    question, candidates, unary_allowed
                )
                if narrowed_unary != unary_allowed:
                    unary_allowed = narrowed_unary
                    period_constraint_applied = True
                unary_allowed = _prefer_requested_context_indices(question, candidates, unary_allowed)
                scoped_unary = _prefer_reporting_entity_indices(
                    question, candidates, unary_allowed
                )
                if scoped_unary != unary_allowed:
                    if "scope" not in grounding_constraints:
                        grounding_constraints.append("scope")
                    unary_allowed = scoped_unary
                if unary_allowed:
                    if metric_constraint_applied:
                        grounding_constraints.append("metric")
                    if period_constraint_applied:
                        grounding_constraints.append("period")
                    metric_mask = torch.zeros_like(logits, dtype=torch.bool)
                    metric_mask[list(unary_allowed)] = True
                    logits = logits.masked_fill(~metric_mask, -1e9)
            if candidates and len(candidates) < logits.shape[0]:
                logits[len(candidates) :] = -torch.inf
            forbidden_operands = set(operands)
            if same_physical_pairs is not None:
                forbidden_operands.update(
                    candidate_index
                    for selected in operands
                    for candidate_index in range(len(candidates))
                    if same_physical_pairs[selected][candidate_index]
                )
            if forbidden_operands:
                # True negative infinity prevents an already-used operand from
                # winning a tie when every remaining logit was hard-masked.
                logits[
                    torch.tensor(
                        sorted(forbidden_operands), device=logits.device
                    )
                ] = -torch.inf
            if not torch.isfinite(logits[: len(candidates)]).any():
                raise DocumentHRMError(
                    "SUM_N could not fill every slot with physically unique evidence"
                    if operation == SUM_N_OPERATION
                    else "No valid evidence candidate remained during decoding"
                )
            probabilities = torch.softmax(logits, dim=-1)
            pointer = int(probabilities.argmax().item())
            operands.append(pointer)
            pointer_confidence = float(probabilities[pointer].item())
            neural_probabilities = torch.softmax(neural_logits, dim=-1)
            neural_pointer_confidence = float(
                neural_probabilities[pointer].item()
            )
            if (
                use_grounded_confidence
                and arity == 1
                and not metric_allowed
            ):
                pointer_confidence = neural_pointer_confidence
            pointer_confidences.append(pointer_confidence)
            neural_pointer_confidences.append(neural_pointer_confidence)
        grounded_multi = _explicit_multi_operand_indices(
            question, operation, candidates
        )
        if grounded_multi is not None:
            operands = list(grounded_multi)
            grounding_constraints.append("multi_operand_metric")
            grounded_confidences = []
            for slot, pointer in enumerate(operands):
                slot_probabilities = torch.softmax(
                    _operation_pointer_logits(
                        output, row, slot, operation
                    ),
                    dim=-1,
                )
                grounded_confidences.append(
                    float(slot_probabilities[pointer].item())
                )
            pointer_confidences = grounded_confidences
            neural_pointer_confidences = list(grounded_confidences)
        if operation in COMMUTATIVE:
            operands.sort()
        elif question and candidates:
            operands = list(
                _align_directed_operands(
                    question, operation, operands, candidates
                )
            )
        program = ReasoningProgram(operation, tuple(operands))
        program_candidates_considered = 1
        symbolic_program_rejections = 0
        symbolic_proof_score = 0.0
        symbolic_unknown_constraints = 0
        program_search_trace: tuple[ProgramCandidateTrace, ...] = ()
        reranker_score: float | None = None
        reranker_confidence: float | None = None
        grounded_confidence_floor: float | None = None
        # Typed slot synthesis is a deterministic proof/reranking layer, not a
        # learned checkpoint capability. Older checkpoints persisted ``False``
        # before the fail-closed unique-slot verifier existed; runtime now
        # enables the layer and still requires every final proof obligation.
        typed_synthesis_enabled = True
        financial_identity_catalog_enabled = (
            self.metadata.get("financial_identity_catalog_enabled", False) is True
        )
        candidate_recall_v12_4_enabled = (
            self.metadata.get("candidate_recall_v12_4_enabled", False) is True
        )
        diverse_set_beam_enabled = (
            self.model.config.program_search_strategy
            == "diverse_set_beam_v1"
        )
        if diverse_set_beam_enabled:
            candidate_recall_v12_4_enabled = True
        synthesized_program_candidates = 0
        typed_synthesis_status: tuple[str, ...] = ()
        selected_program_dag: ReasoningDAGProgram | None = None
        selected_slot_bindings: tuple[tuple[str, int], ...] = ()
        slot_candidate_pools_at_10: tuple[tuple[int, ...], ...] = ()
        slot_candidate_pools_at_20: tuple[tuple[int, ...], ...] = ()
        operation_slot_candidate_pools_at_10: tuple[
            tuple[str, tuple[tuple[int, ...], ...]], ...
        ] = ()
        operation_slot_candidate_pools_at_20: tuple[
            tuple[str, tuple[tuple[int, ...], ...]], ...
        ] = ()
        operation_beam: tuple[str, ...] = ()
        preverification_programs: tuple[ReasoningProgram, ...] = ()
        selected_for_verification: tuple[ReasoningProgram, ...] = ()
        generated_before_physical_dedup = 0
        generated_after_physical_dedup = 0
        if (
            question
            and candidates
            and self.metadata.get("program_search_enabled", True) is not False
        ):
            search_result = _search_symbolic_programs(
                output,
                row,
                question,
                candidates,
                program,
                proof_guided=(
                    self.metadata.get("proof_guided_reranking_enabled", True)
                    is not False
                ),
                graph_enabled=(
                    self.metadata.get("typed_evidence_graph_enabled", True)
                    is not False
                ),
                typed_synthesis_enabled=typed_synthesis_enabled,
                financial_identity_catalog_enabled=(
                    financial_identity_catalog_enabled
                ),
                candidate_recall_v12_4_enabled=(
                    candidate_recall_v12_4_enabled
                ),
                diverse_set_beam_enabled=diverse_set_beam_enabled,
                reranker=self.program_reranker,
            )
            if search_result is not None:
                program_candidates_considered = (
                    search_result.candidates_considered
                )
                symbolic_program_rejections = (
                    search_result.candidates_rejected
                )
                symbolic_proof_score = search_result.proof_score
                symbolic_unknown_constraints = (
                    search_result.unknown_constraints
                )
                program_search_trace = search_result.trace
                reranker_score = search_result.reranker_score
                reranker_confidence = search_result.reranker_confidence
                grounded_confidence_floor = (
                    search_result.grounded_confidence_floor
                )
                synthesized_program_candidates = (
                    search_result.synthesized_program_candidates
                )
                typed_synthesis_status = search_result.typed_synthesis_status
                selected_program_dag = search_result.program_dag
                selected_slot_bindings = search_result.slot_bindings
                slot_candidate_pools_at_10 = (
                    search_result.slot_candidate_pools_at_10
                )
                slot_candidate_pools_at_20 = (
                    search_result.slot_candidate_pools_at_20
                )
                operation_slot_candidate_pools_at_10 = (
                    search_result.operation_slot_candidate_pools_at_10
                )
                operation_slot_candidate_pools_at_20 = (
                    search_result.operation_slot_candidate_pools_at_20
                )
                operation_beam = search_result.operation_beam
                preverification_programs = (
                    search_result.preverification_programs
                )
                selected_for_verification = (
                    search_result.selected_for_verification
                )
                generated_before_physical_dedup = (
                    search_result.generated_programs_before_physical_dedup
                )
                generated_after_physical_dedup = (
                    search_result.generated_programs_after_physical_dedup
                )
                if search_result.program != program:
                    program = search_result.program
                    operation = program.operation
                    operation_id = OPERATION_TO_ID[operation]
                    operation_was_constrained = (
                        operation_was_constrained
                        or operation_id != neural_operation_id
                    )
                    pointer_confidences = list(
                        search_result.pointer_confidences
                    )
                    neural_pointer_confidences = list(
                        search_result.neural_pointer_confidences
                    )
                    grounding_constraints.extend(
                        search_result.constraints_applied
                    )
        neural_operation_confidence = (
            sum_family_probability
            if operation == SUM_N_OPERATION
            else float(operation_probs[operation_id].item())
        )
        operation_confidence = (
            1.0
            if use_grounded_confidence and operation_was_constrained
            else neural_operation_confidence
        )
        confidence_parts = [operation_confidence, *pointer_confidences]
        if operand_count_confidence is not None:
            confidence_parts.append(operand_count_confidence)
        confidence = math.exp(
            sum(math.log(max(value, 1e-8)) for value in confidence_parts)
            / len(confidence_parts)
        )
        if reranker_confidence is not None:
            confidence = reranker_confidence
            confidence_schema = "v11-selected-program-confidence-head"
        elif grounded_confidence_floor is not None:
            confidence = max(confidence, grounded_confidence_floor)
            confidence_schema = "grounded-symbolic-fusion-v12.1"
        return ProgramPrediction(
            program=program,
            operation_confidence=operation_confidence,
            pointer_confidences=tuple(pointer_confidences),
            confidence=confidence,
            segments=segments,
            halt_q=float(output.q_values[row, 0].item()),
            continue_q=float(output.q_values[row, 1].item()),
            operation_probabilities=tuple(
                float(item) for item in operation_probs.tolist()
            ),
            neural_operation_confidence=neural_operation_confidence,
            neural_pointer_confidences=tuple(neural_pointer_confidences),
            confidence_schema=confidence_schema,
            grounding_constraints_applied=tuple(
                dict.fromkeys(grounding_constraints)
            ),
            operand_count_confidence=operand_count_confidence,
            program_candidates_considered=program_candidates_considered,
            symbolic_program_rejections=symbolic_program_rejections,
            symbolic_proof_score=symbolic_proof_score,
            symbolic_unknown_constraints=symbolic_unknown_constraints,
            program_search_trace=program_search_trace,
            reranker_score=reranker_score,
            reranker_confidence=reranker_confidence,
            typed_program_synthesis_enabled=typed_synthesis_enabled,
            synthesized_program_candidates=synthesized_program_candidates,
            typed_synthesis_status=typed_synthesis_status,
            program_dag=(
                selected_program_dag
                or (
                    _safe_materialize_program_dag(program, candidates)
                    if candidates
                    else None
                )
            ),
            slot_bindings=selected_slot_bindings,
            slot_candidate_pools_at_10=slot_candidate_pools_at_10,
            slot_candidate_pools_at_20=slot_candidate_pools_at_20,
            operation_slot_candidate_pools_at_10=(
                operation_slot_candidate_pools_at_10
            ),
            operation_slot_candidate_pools_at_20=(
                operation_slot_candidate_pools_at_20
            ),
            operation_beam=operation_beam,
            preverification_programs=preverification_programs,
            selected_for_verification=selected_for_verification,
            generated_programs_before_physical_dedup=(
                generated_before_physical_dedup
            ),
            generated_programs_after_physical_dedup=(
                generated_after_physical_dedup
            ),
        )

    def save(self, path: str | Path, *, overwrite: bool = True) -> Path:
        destination = Path(path).expanduser().resolve()
        if destination.exists() and not overwrite:
            raise DocumentHRMError(
                f"Refusing to overwrite existing Document HRM checkpoint: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        adapter_hash = _semantic_adapter_state_sha256(self.model)
        semantic_frontend_fingerprint = (
            _semantic_encoder_spec_from_config(
                self.model.config
            ).fingerprint()
            if self.model.config.input_encoder_mode == "semantic"
            else "hash-input"
        )
        effective_fingerprint = _effective_semantic_fingerprint(
            self.model.config,
            frontend_fingerprint=semantic_frontend_fingerprint,
            adapter_state_sha256=adapter_hash,
        )
        metadata = dict(self.metadata)
        metadata.update(
            {
                "semantic_frontend_fingerprint": (
                    semantic_frontend_fingerprint
                ),
                "semantic_adapter_state_sha256": adapter_hash,
                "semantic_checkpoint_integrity_schema": (
                    SEMANTIC_CHECKPOINT_INTEGRITY_SCHEMA
                ),
                "semantic_adapter_implementation": (
                    SEMANTIC_ADAPTER_IMPLEMENTATION_VERSION
                ),
                "effective_semantic_fingerprint": effective_fingerprint,
                "financial_slot_feature_names": list(
                    FINANCIAL_SLOT_FEATURES
                ),
            }
        )
        self.metadata = metadata
        torch.save(
            {
                "schema_version": DOCUMENT_HRM_SCHEMA_VERSION,
                "config": asdict(self.model.config),
                "metadata": metadata,
                "state_dict": self.model.state_dict(),
            },
            destination,
        )
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "DocumentHRMBundle":
        source = Path(path).expanduser().resolve()
        if str(device).startswith("cpu") and torch.get_num_threads() > 4:
            torch.set_num_threads(4)
        if not source.is_file():
            raise FileNotFoundError(f"Document HRM checkpoint not found: {source}")
        try:
            payload = torch.load(source, map_location=device, weights_only=True)
        except TypeError:  # pragma: no cover
            payload = torch.load(source, map_location=device)
        schema_version = payload.get("schema_version")
        if schema_version not in {
            DOCUMENT_HRM_SCHEMA_VERSION,
            *LEGACY_DOCUMENT_HRM_SCHEMA_VERSIONS,
        }:
            raise DocumentHRMError("Unsupported Document HRM checkpoint schema")
        raw_config = dict(payload["config"])
        if schema_version in LEGACY_DOCUMENT_HRM_SCHEMA_VERSIONS:
            raw_config.setdefault("input_encoder_mode", "hash")
            raw_config.setdefault("semantic_model_name", DEFAULT_DENSE_MODEL)
            raw_config.setdefault("semantic_embedding_size", 384)
        if schema_version <= 12:
            raw_config.setdefault("semantic_encoder_backend", "fastembed")
            raw_config.setdefault("semantic_model_revision", "")
            raw_config.setdefault(
                "semantic_encoder_output_size",
                raw_config.get("semantic_embedding_size", 384),
            )
            raw_config.setdefault("semantic_word_segmentation", "none")
            raw_config.setdefault("semantic_segmenter_version", "")
            raw_config.setdefault("semantic_prefix_mode", "none")
            raw_config.setdefault(
                "semantic_query_encoding_mode", "document"
            )
            raw_config.setdefault(
                "semantic_text_template_version",
                LEGACY_SEMANTIC_TEMPLATE,
            )
            raw_config.setdefault("semantic_adapter_mode", "identity")
            raw_config.setdefault("program_search_strategy", "legacy")
        if schema_version <= 6:
            raw_config.setdefault("pointer_conditioning_mode", "independent")
            raw_config.setdefault("pointer_teacher_forcing", 0.50)
        if schema_version <= 7:
            raw_config.setdefault("pair_backbone_mode", "detached")
        if schema_version <= 8:
            raw_config.setdefault("pair_loss_weight", 1.0)
            raw_config.setdefault("pair_loss_warmup_epochs", 0)
        if schema_version <= 9:
            raw_config.setdefault("pair_backbone_warmup_epochs", 0)
            raw_config.setdefault("pair_targeted_negative_weight", 0.0)
        if schema_version == 4:
            # Preserve exact v3 behavior for packaged checkpoints while making
            # newly trained models use strict H/L coupling and bounded
            # relevance priors.
            raw_config.setdefault("hierarchical_input_mode", "legacy")
            raw_config.setdefault("relevance_prior_max", 20.0)
            raw_config.setdefault("act_mode", "supervised")
            raw_config.setdefault("minimum_halt_segments", 1)
        if schema_version <= 11:
            raw_config.setdefault("variable_sum_mode", "disabled")
            raw_config.setdefault("max_program_operands", MAX_OPERANDS)
        config = DocumentHRMConfig(**raw_config)
        model = DocumentHierarchicalReasoningModel(config)
        state_dict = dict(payload["state_dict"])
        pair_weights = state_dict.get("pair_feature_weights")
        if (
            isinstance(pair_weights, Tensor)
            and model.pair_feature_weights is not None
            and pair_weights.shape[1] < model.pair_feature_weights.shape[1]
        ):
            expanded = torch.zeros_like(model.pair_feature_weights)
            expanded[:, : pair_weights.shape[1]] = pair_weights
            state_dict["pair_feature_weights"] = expanded
        slot_weight_name = "financial_slot_scorer.0.weight"
        slot_weight = state_dict.get(slot_weight_name)
        target_slot_weight = model.state_dict().get(slot_weight_name)
        if (
            isinstance(slot_weight, Tensor)
            and isinstance(target_slot_weight, Tensor)
            and slot_weight.ndim == 2
            and target_slot_weight.ndim == 2
            and slot_weight.shape[0] == target_slot_weight.shape[0]
            and slot_weight.shape[1] < target_slot_weight.shape[1]
        ):
            expanded_slot_weight = torch.zeros_like(target_slot_weight)
            expanded_slot_weight[:, : slot_weight.shape[1]] = slot_weight
            state_dict[slot_weight_name] = expanded_slot_weight
        model.load_state_dict(state_dict)
        model.to(device)
        metadata = dict(payload.get("metadata", {}))
        metadata.setdefault("checkpoint_schema_version", schema_version)
        metadata.setdefault("runtime_schema_version", DOCUMENT_HRM_SCHEMA_VERSION)
        metadata.setdefault(
            "financial_slot_feature_names",
            list(FINANCIAL_SLOT_FEATURES),
        )
        expected_adapter_hash = metadata.get(
            "semantic_adapter_state_sha256"
        )
        actual_adapter_hash = _semantic_adapter_state_sha256(model)
        strict_semantic_integrity = (
            schema_version == DOCUMENT_HRM_SCHEMA_VERSION
            and config.input_encoder_mode == "semantic"
        )
        required_integrity_fields = {
            "semantic_frontend_fingerprint",
            "semantic_adapter_state_sha256",
            "semantic_checkpoint_integrity_schema",
            "semantic_adapter_implementation",
            "effective_semantic_fingerprint",
        }
        missing_integrity_fields = sorted(
            required_integrity_fields - set(metadata)
        )
        if strict_semantic_integrity and missing_integrity_fields:
            raise DocumentHRMError(
                "Schema-13 semantic checkpoint is missing integrity fields: "
                + ", ".join(missing_integrity_fields)
            )
        if expected_adapter_hash is not None and (
            str(expected_adapter_hash) != actual_adapter_hash
        ):
            raise DocumentHRMError(
                "Semantic adapter fingerprint does not match checkpoint weights"
            )
        if strict_semantic_integrity:
            if metadata["semantic_checkpoint_integrity_schema"] != (
                SEMANTIC_CHECKPOINT_INTEGRITY_SCHEMA
            ):
                raise DocumentHRMError(
                    "Unsupported semantic checkpoint integrity schema"
                )
            if metadata["semantic_adapter_implementation"] != (
                SEMANTIC_ADAPTER_IMPLEMENTATION_VERSION
            ):
                raise DocumentHRMError(
                    "Semantic adapter implementation fingerprint mismatch"
                )
            actual_frontend_fingerprint = (
                _semantic_encoder_spec_from_config(config).fingerprint()
            )
            if str(metadata["semantic_frontend_fingerprint"]) != (
                actual_frontend_fingerprint
            ):
                raise DocumentHRMError(
                    "Semantic frontend fingerprint does not match config"
                )
            actual_effective_fingerprint = _effective_semantic_fingerprint(
                config,
                frontend_fingerprint=actual_frontend_fingerprint,
                adapter_state_sha256=actual_adapter_hash,
            )
            if str(metadata["effective_semantic_fingerprint"]) != (
                actual_effective_fingerprint
            ):
                raise DocumentHRMError(
                    "Effective semantic fingerprint does not match checkpoint"
                )
        bundle = cls(model, metadata=metadata)
        pointer_checkpoint = metadata.get("pointer_checkpoint")
        if pointer_checkpoint:
            pointer_path = Path(str(pointer_checkpoint)).expanduser()
            if not pointer_path.is_absolute():
                pointer_path = source.parent / pointer_path
            if pointer_path.resolve() == source:
                raise DocumentHRMError("pointer_checkpoint cannot reference itself")
            bundle.pointer_bundle = cls.load(pointer_path, device=device)
            if (
                bundle.semantic_encoder is not None
                and bundle.pointer_bundle.semantic_encoder is not None
                and _semantic_encoder_spec_from_config(bundle.model.config)
                == _semantic_encoder_spec_from_config(
                    bundle.pointer_bundle.model.config
                )
            ):
                bundle.pointer_bundle.semantic_encoder = bundle.semantic_encoder
        pointer_secondary_checkpoint = metadata.get("pointer_secondary_checkpoint")
        if pointer_secondary_checkpoint:
            pointer_path = Path(str(pointer_secondary_checkpoint)).expanduser()
            if not pointer_path.is_absolute():
                pointer_path = source.parent / pointer_path
            if pointer_path.resolve() == source:
                raise DocumentHRMError(
                    "pointer_secondary_checkpoint cannot reference itself"
                )
            bundle.pointer_secondary_bundle = cls.load(pointer_path, device=device)
            if (
                bundle.semantic_encoder is not None
                and bundle.pointer_secondary_bundle.semantic_encoder is not None
                and _semantic_encoder_spec_from_config(bundle.model.config)
                == _semantic_encoder_spec_from_config(
                    bundle.pointer_secondary_bundle.model.config
                )
            ):
                bundle.pointer_secondary_bundle.semantic_encoder = bundle.semantic_encoder
        return bundle


def _verifier_guided_invalid_masks(
    example: ProgramExample,
    *,
    operand_width: int = MAX_OPERANDS,
) -> tuple[list[list[bool]], list[list[bool]], bool]:
    """Mine only verifier-certain negatives for common one/two-step programs.

    UNKNOWN candidates are deliberately excluded. If the verifier contradicts
    the stored gold program, the whole example is skipped for this auxiliary
    loss so parser noise can never turn gold supervision into a negative.
    """

    count = len(example.candidates)
    unary = [[False] * count for _ in range(operand_width)]
    pairs = [[False] * count for _ in range(count)]
    operation = example.program.operation
    if operation not in VERIFIER_GUIDED_OPERATIONS:
        return unary, pairs, False
    arity = OPERATION_ARITY[operation]
    if arity > 2:
        return unary, pairs, False

    plan = compile_question_plan(example.question)
    graph = build_evidence_graph(example.candidates)
    gold_verification = verify_program(plan, example.program, graph)
    if gold_verification.state is VerificationState.INVALID:
        return unary, pairs, False

    if arity == 1:
        gold_pointer = example.program.operands[0]
        for index in range(count):
            if index == gold_pointer:
                continue
            result = verify_program(
                plan,
                ReasoningProgram(operation, (index,)),
                graph,
            )
            unary[0][index] = result.state is VerificationState.INVALID
        return unary, pairs, True

    positive_pairs = {tuple(example.program.operands)}
    if operation in COMMUTATIVE:
        positive_pairs.add(tuple(reversed(example.program.operands)))
    reversed_gold_pair = (
        tuple(reversed(example.program.operands))
        if arity == 2 and operation not in COMMUTATIVE
        else ()
    )
    reversed_gold_is_proven = (
        reversed_pair_is_proven_negative(plan, example.program, graph)
        if operation in {"SUBTRACT", "PERCENT_CHANGE"}
        else True
    )
    pair_pool = set(example.program.operands)
    pair_pool.update(
        sorted(
            range(count),
            key=lambda index: (
                example.candidates[index].relevance,
                -index,
            ),
            reverse=True,
        )[:MAX_VERIFIER_GUIDED_PAIR_CANDIDATES]
    )
    for left in pair_pool:
        for right in pair_pool:
            if left == right or (left, right) in positive_pairs:
                continue
            if (
                (left, right) == reversed_gold_pair
                and not reversed_gold_is_proven
            ):
                continue
            result = verify_program(
                plan,
                ReasoningProgram(operation, (left, right)),
                graph,
            )
            pairs[left][right] = result.state is VerificationState.INVALID
    return unary, pairs, True


def encode_examples(
    examples: Sequence[ProgramExample],
    tokenizer: HashSequenceTokenizer | None,
    semantic_encoder: SemanticEmbeddingCache | None,
    config: DocumentHRMConfig,
    device: torch.device,
    *,
    include_verifier_negatives: bool = False,
) -> EncodedBatch:
    if not examples:
        raise DocumentHRMError("Cannot encode an empty batch")
    questions = [example.question for example in examples]
    needs_question_plan = (
        config.financial_structure_mode == "typed_hierarchy_v1"
        or config.semantic_text_template_version
        == FINANCIAL_VI_SEMANTIC_TEMPLATE
    )
    plans: list[QuestionPlan | None] = [
        compile_question_plan(question)
        if needs_question_plan
        else None
        for question in questions
    ]
    question_semantic: Tensor | None = None
    candidate_semantic: Tensor | None = None
    slot_semantic: Tensor | None = None
    if config.input_encoder_mode == "hash":
        if tokenizer is None:
            raise DocumentHRMError("Hash tokenizer is missing")
        question_ids, question_mask = _encode_text_matrix(
            questions,
            tokenizer,
            config.max_question_tokens,
            device,
        )
    else:
        if semantic_encoder is None:
            raise DocumentHRMError("Semantic encoder is missing")
        question_ids = torch.zeros((len(examples), 1), dtype=torch.long, device=device)
        question_mask = torch.ones((len(examples), 1), dtype=torch.float32, device=device)
        query_texts = [
            _question_semantic_text(
                question,
                plan,
                template_version=config.semantic_text_template_version,
            )
            for question, plan in zip(questions, plans)
        ]
        question_semantic = torch.from_numpy(
            semantic_encoder.encode(query_texts, role="query")
        ).to(device)
    question_features = torch.tensor(
        [_question_features(question) for question in questions],
        dtype=torch.float32,
        device=device,
    )
    b = len(examples)
    n = config.max_candidates
    operand_width = config.max_program_operands
    t = config.max_candidate_tokens if config.input_encoder_mode == "hash" else 1
    candidate_ids = torch.zeros((b, n, t), dtype=torch.long, device=device)
    candidate_token_mask = torch.zeros((b, n, t), dtype=torch.float32, device=device)
    if config.input_encoder_mode == "semantic":
        candidate_semantic = torch.zeros(
            (b, n, config.semantic_encoder_size),
            dtype=torch.float32,
            device=device,
        )
        slot_semantic = torch.zeros(
            (b, operand_width, config.semantic_encoder_size),
            dtype=torch.float32,
            device=device,
        )
    numeric_features = torch.zeros(
        (b, n, config.numeric_feature_size), dtype=torch.float32, device=device
    )
    candidate_mask = torch.zeros((b, n), dtype=torch.float32, device=device)
    operation_targets = torch.empty((b,), dtype=torch.long, device=device)
    pointer_targets = torch.full(
        (b, operand_width), -100, dtype=torch.long, device=device
    )
    semantic_pointer_targets = torch.full(
        (b, operand_width), -100, dtype=torch.long, device=device
    )
    semantic_slot_declared = torch.zeros(
        (b, operand_width), dtype=torch.bool, device=device
    )
    operand_count_targets = torch.full(
        (b,), -100, dtype=torch.long, device=device
    )
    sum_family_targets = torch.zeros((b,), dtype=torch.bool, device=device)
    unary_structural_negatives = torch.zeros(
        (b, operand_width, n), dtype=torch.bool, device=device
    )
    pair_structural_negatives = torch.zeros(
        (b, n, n), dtype=torch.bool, device=device
    )
    verifier_unary_invalid_negatives = torch.zeros(
        (b, operand_width, n), dtype=torch.bool, device=device
    )
    verifier_pair_invalid_negatives = torch.zeros(
        (b, n, n), dtype=torch.bool, device=device
    )
    pair_features = (
        torch.zeros(
            (b, n, n, PAIR_FEATURE_SIZE), dtype=torch.float32, device=device
        )
        if config.pointer_conditioning_mode == "joint_pair"
        else None
    )
    financial_candidate_feature_tensor = (
        torch.zeros(
            (b, n, FINANCIAL_CANDIDATE_FEATURE_SIZE),
            dtype=torch.float32,
            device=device,
        )
        if config.financial_structure_mode == "typed_hierarchy_v1"
        else None
    )
    financial_slot_feature_tensor = (
        torch.zeros(
            (b, operand_width, n, FINANCIAL_SLOT_FEATURE_SIZE),
            dtype=torch.float32,
            device=device,
        )
        if config.financial_structure_mode == "typed_hierarchy_v1"
        else None
    )
    financial_slot_mismatch_mask = (
        torch.zeros(
            (b, operand_width, n), dtype=torch.bool, device=device
        )
        if config.financial_structure_mode == "typed_hierarchy_v1"
        else None
    )
    financial_same_physical_mask = (
        torch.zeros((b, n, n), dtype=torch.bool, device=device)
        if config.financial_structure_mode == "typed_hierarchy_v1"
        else None
    )
    semantic_positions: list[tuple[int, int]] = []
    semantic_texts: list[str] = []
    slot_semantic_positions: list[tuple[int, int]] = []
    slot_semantic_texts: list[str] = []
    for row, example in enumerate(examples):
        if len(example.candidates) > n:
            raise DocumentHRMError(
                f"Example {example.example_id} has {len(example.candidates)} candidates; max={n}"
            )
        example.program.validate(len(example.candidates))
        if len(example.program.operands) > operand_width:
            raise DocumentHRMError(
                f"Example {example.example_id} has "
                f"{len(example.program.operands)} operands; max={operand_width}"
            )
        sum_family = (
            config.variable_sum_mode == "sum_n_v1"
            and is_sum_operation(example.program.operation)
        )
        if (
            example.program.operation == SUM_N_OPERATION
            and config.variable_sum_mode != "sum_n_v1"
        ):
            raise DocumentHRMError(
                "SUM_N examples require variable_sum_mode=sum_n_v1"
            )
        operation_targets[row] = OPERATION_TO_ID[
            "ADD" if sum_family else example.program.operation
        ]
        if sum_family:
            sum_family_targets[row] = True
            operand_count_targets[row] = len(example.program.operands) - 2
        plan = plans[row]
        semantic_gold_operands = _supervision_gold_operand_indices(example)
        for slot, pointer in enumerate(semantic_gold_operands):
            semantic_pointer_targets[row, slot] = pointer
            semantic_slot_declared[row, slot] = bool(
                plan is not None and slot < len(plan.operand_slots)
            )
        if config.input_encoder_mode == "semantic":
            for slot in range(len(example.program.operands)):
                slot_semantic_positions.append((row, slot))
                slot_semantic_texts.append(
                    _slot_semantic_text(
                        example.question,
                        plan,
                        slot,
                        template_version=(
                            config.semantic_text_template_version
                        ),
                    )
                )
        if (
            plan is not None
            and config.financial_structure_mode == "typed_hierarchy_v1"
        ):
            financial_matrices = build_financial_feature_matrices(
                plan,
                example.candidates,
                slot_count=operand_width,
            )
            same_physical = torch.tensor(
                financial_matrices.same_physical_mask,
                dtype=torch.bool,
                device=device,
            )
            assert financial_same_physical_mask is not None
            assert financial_candidate_feature_tensor is not None
            assert financial_slot_feature_tensor is not None
            assert financial_slot_mismatch_mask is not None
            candidate_count = len(example.candidates)
            if candidate_count:
                financial_candidate_feature_tensor[
                    row, :candidate_count
                ] = torch.tensor(
                    financial_matrices.candidate_features,
                    dtype=torch.float32,
                    device=device,
                )
                financial_slot_feature_tensor[
                    row, :, :candidate_count
                ] = torch.tensor(
                    financial_matrices.slot_features,
                    dtype=torch.float32,
                    device=device,
                )
                financial_slot_mismatch_mask[
                    row, :, :candidate_count
                ] = torch.tensor(
                    financial_matrices.slot_mismatch_mask,
                    dtype=torch.bool,
                    device=device,
                )
            financial_same_physical_mask[
                row, :candidate_count, :candidate_count
            ] = same_physical
            pair_structural_negatives[
                row, :candidate_count, :candidate_count
            ] |= same_physical
            for left_slot, right_slot in combinations(
                range(len(example.program.operands)), 2
            ):
                left = example.program.operands[left_slot]
                right = example.program.operands[right_slot]
                if bool(same_physical[left, right].item()):
                    raise DocumentHRMError(
                        f"Example {example.example_id} reuses one physical "
                        "evidence cell for multiple operands"
                    )
        if include_verifier_negatives:
            verifier_unary, verifier_pairs, _ = (
                _verifier_guided_invalid_masks(
                    example, operand_width=operand_width
                )
            )
            verifier_unary_invalid_negatives[
                row, :, : len(example.candidates)
            ] = torch.tensor(
                verifier_unary,
                dtype=torch.bool,
                device=device,
            )
            verifier_pair_invalid_negatives[
                row, : len(example.candidates), : len(example.candidates)
            ] = torch.tensor(
                verifier_pairs,
                dtype=torch.bool,
                device=device,
            )
        for slot, pointer in enumerate(example.program.operands):
            pointer_targets[row, slot] = pointer
            unary_structural_negatives[row, slot, : len(example.candidates)] = (
                torch.tensor(
                    _unary_structural_negative_mask(example.candidates, pointer),
                    dtype=torch.bool,
                    device=device,
                )
            )
        explicit_unary, explicit_pairs = _structured_negative_indices(
            example, operand_width=operand_width
        )
        for slot, candidate_indices in enumerate(explicit_unary):
            if candidate_indices:
                unary_structural_negatives[row, slot, list(candidate_indices)] = True
        for left_index, right_index in explicit_pairs:
            pair_structural_negatives[row, left_index, right_index] = True
        for column, candidate in enumerate(example.candidates):
            if config.input_encoder_mode == "hash":
                assert tokenizer is not None
                ids = tokenizer.encode(
                    _candidate_semantic_text(
                        candidate,
                        mode=config.candidate_semantic_mode,
                        template_version=(
                            config.semantic_text_template_version
                        ),
                    ),
                    t,
                )
                candidate_ids[row, column, : len(ids)] = torch.tensor(ids, device=device)
                candidate_token_mask[row, column, : len(ids)] = 1.0
            else:
                semantic_positions.append((row, column))
                semantic_texts.append(
                    _candidate_semantic_text(
                        candidate,
                        mode=config.candidate_semantic_mode,
                        template_version=(
                            config.semantic_text_template_version
                        ),
                    )
                )
            numeric_values = _numeric_features(
                candidate,
                column,
                n,
                question=example.question,
                candidates=example.candidates,
                feature_size=config.numeric_feature_size,
            )
            if (
                plan is not None
                and config.financial_structure_mode
                == "typed_hierarchy_v1"
                and len(numeric_values) > 5
            ):
                # Candidate order is a retrieval implementation detail, not a
                # financial relation.  V13 uses provenance edges instead.
                numeric_values[5] = 0.0
            numeric_features[row, column] = torch.tensor(
                numeric_values, device=device
            )
            if (
                plan is not None
                and config.financial_structure_mode
                == "typed_hierarchy_v1"
            ):
                assert financial_slot_mismatch_mask is not None
                for slot in range(operand_width):
                    if (
                        bool(
                            financial_slot_mismatch_mask[
                                row, slot, column
                            ].item()
                        )
                        and slot < len(example.program.operands)
                        and column != example.program.operands[slot]
                    ):
                        unary_structural_negatives[row, slot, column] = True
            candidate_mask[row, column] = 1.0
        if pair_features is not None:
            for left_index, left_candidate in enumerate(example.candidates):
                for right_index, right_candidate in enumerate(example.candidates):
                    pair_values = _candidate_pair_features(
                        left_candidate,
                        right_candidate,
                        left_index,
                        right_index,
                    )
                    if (
                        plan is not None
                        and config.financial_structure_mode
                        == "typed_hierarchy_v1"
                    ):
                        pair_values[9] = 0.0
                    pair_features[row, left_index, right_index] = torch.tensor(
                        pair_values, device=device
                    )
    if semantic_texts:
        assert semantic_encoder is not None and candidate_semantic is not None
        vectors = torch.from_numpy(
            semantic_encoder.encode(semantic_texts, role="evidence")
        ).to(device)
        for index, (row, column) in enumerate(semantic_positions):
            candidate_semantic[row, column] = vectors[index]
            if config.numeric_feature_size > 17 and question_semantic is not None:
                numeric_features[row, column, 17] = torch.dot(
                    question_semantic[row], vectors[index]
                )
    if slot_semantic_texts:
        assert semantic_encoder is not None and slot_semantic is not None
        slot_vectors = torch.from_numpy(
            semantic_encoder.encode(
                slot_semantic_texts,
                role="query",
            )
        ).to(device)
        for index, (row, slot) in enumerate(slot_semantic_positions):
            slot_semantic[row, slot] = slot_vectors[index]
    return EncodedBatch(
        question_ids=question_ids,
        question_mask=question_mask,
        question_features=question_features,
        candidate_ids=candidate_ids,
        candidate_token_mask=candidate_token_mask,
        numeric_features=numeric_features,
        candidate_mask=candidate_mask,
        question_semantic=question_semantic,
        candidate_semantic=candidate_semantic,
        slot_semantic=slot_semantic,
        operation_targets=operation_targets,
        pointer_targets=pointer_targets,
        semantic_pointer_targets=semantic_pointer_targets,
        semantic_slot_declared=semantic_slot_declared,
        operand_count_targets=operand_count_targets,
        sum_family_targets=sum_family_targets,
        pair_features=pair_features,
        unary_structural_negatives=unary_structural_negatives,
        pair_structural_negatives=pair_structural_negatives,
        verifier_unary_invalid_negatives=(
            verifier_unary_invalid_negatives
        ),
        verifier_pair_invalid_negatives=(
            verifier_pair_invalid_negatives
        ),
        financial_candidate_features=financial_candidate_feature_tensor,
        financial_slot_features=financial_slot_feature_tensor,
        financial_slot_mismatch_mask=financial_slot_mismatch_mask,
        financial_same_physical_mask=financial_same_physical_mask,
    )


def encode_inference_batch(
    question: str,
    candidates: Sequence[NumericCandidate],
    tokenizer: HashSequenceTokenizer | None,
    semantic_encoder: SemanticEmbeddingCache | None,
    config: DocumentHRMConfig,
    device: torch.device,
) -> EncodedBatch:
    if len(candidates) > config.max_candidates:
        candidates = candidates[: config.max_candidates]
    dummy = ProgramExample(
        example_id="inference",
        question=question,
        context="",
        answer_text="",
        candidates=tuple(candidates),
        program=ReasoningProgram("SELECT", (0,)),
        answer_value="0",
    )
    encoded = encode_examples(
        [dummy], tokenizer, semantic_encoder, config, device
    )
    return EncodedBatch(
        question_ids=encoded.question_ids,
        question_mask=encoded.question_mask,
        question_features=encoded.question_features,
        candidate_ids=encoded.candidate_ids,
        candidate_token_mask=encoded.candidate_token_mask,
        numeric_features=encoded.numeric_features,
        candidate_mask=encoded.candidate_mask,
        question_semantic=encoded.question_semantic,
        candidate_semantic=encoded.candidate_semantic,
        slot_semantic=encoded.slot_semantic,
        pair_features=encoded.pair_features,
        financial_candidate_features=encoded.financial_candidate_features,
        financial_slot_features=encoded.financial_slot_features,
        financial_slot_mismatch_mask=encoded.financial_slot_mismatch_mask,
        financial_same_physical_mask=encoded.financial_same_physical_mask,
    )


def _load_compatible_training_bundle(
    checkpoint: str | Path,
    config: DocumentHRMConfig,
    device: str | torch.device,
    *,
    initialization_mode: str = "exact",
) -> DocumentHRMBundle:
    if initialization_mode not in {"exact", "compatible_backbone"}:
        raise DocumentHRMError(
            "initialization_mode must be exact or compatible_backbone"
        )
    source_path = Path(checkpoint).expanduser().resolve()
    source_sha256 = _sha256_file(source_path)

    # Loading a checkpoint constructs its source architecture before restoring
    # weights. Preserve RNG state so compatible initialization gives every new
    # target-only parameter the same initialization as a scratch model with the
    # same seed.
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    source = DocumentHRMBundle.load(source_path, device=device)
    if initialization_mode == "compatible_backbone":
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)

    if source.pointer_bundle is not None or source.pointer_secondary_bundle is not None:
        raise DocumentHRMError(
            "initialize_from must reference a single compatible HRM checkpoint"
        )
    source_config = asdict(source.model.config)
    target_config = asdict(config)
    if initialization_mode == "exact":
        changed = [
            name
            for name, value in target_config.items()
            if name != "seed" and source_config.get(name) != value
        ]
        if changed:
            raise DocumentHRMError(
                "initialize_from architecture/config mismatch: "
                + ", ".join(changed[:8])
            )
        # Seed controls the new training run, not checkpoint tensor shapes.
        source.model.config = config
        source.metadata = dict(source.metadata)
        source.metadata["_training_initialization_manifest"] = {
            "mode": "exact",
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": source_sha256,
            "copied_state_keys": len(source.model.state_dict()),
            "target_state_keys": len(source.model.state_dict()),
            "missing_target_keys": [],
            "shape_mismatch_keys": [],
            "unused_source_keys": [],
        }
        return source

    target_model = DocumentHierarchicalReasoningModel(config).to(device)
    source_state = source.model.state_dict()
    target_state = target_model.state_dict()
    copied_keys: list[str] = []
    prefix_expanded_keys: list[str] = []
    missing_keys: list[str] = []
    shape_mismatches: list[dict[str, object]] = []
    copied_numel = 0
    for name, target_tensor in target_state.items():
        source_tensor = source_state.get(name)
        if source_tensor is None:
            missing_keys.append(name)
            continue
        if source_tensor.shape != target_tensor.shape:
            prefix_safe = name in {
                "pointer_relevance_scale",
                "pointer_slot_embedding.weight",
                "financial_slot_embedding.weight",
                "financial_slot_scales",
            }
            prefix_compatible = (
                prefix_safe
                and source_tensor.ndim == target_tensor.ndim
                and source_tensor.shape[0] < target_tensor.shape[0]
                and source_tensor.shape[1:] == target_tensor.shape[1:]
            )
            if prefix_compatible:
                expanded = target_tensor.detach().clone()
                expanded[: source_tensor.shape[0]] = source_tensor.detach()
                target_state[name] = expanded
                copied_keys.append(name)
                prefix_expanded_keys.append(name)
                copied_numel += source_tensor.numel()
                continue
            shape_mismatches.append(
                {
                    "name": name,
                    "source_shape": list(source_tensor.shape),
                    "target_shape": list(target_tensor.shape),
                }
            )
            continue
        target_state[name] = source_tensor.detach().clone()
        copied_keys.append(name)
        copied_numel += source_tensor.numel()
    target_model.load_state_dict(target_state, strict=True)

    unused_source_keys = sorted(set(source_state) - set(target_state))
    target_numel = sum(tensor.numel() for tensor in target_state.values())
    manifest: dict[str, object] = {
        "mode": "compatible_backbone",
        "copy_rule": "state_dict_name_and_shape_match",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": source_sha256,
        "source_schema_version": source.metadata.get("checkpoint_schema_version"),
        "copied_state_keys": len(copied_keys),
        "target_state_keys": len(target_state),
        "copied_numel": copied_numel,
        "target_numel": target_numel,
        "copied_fraction": copied_numel / max(1, target_numel),
        "copied_keys": sorted(copied_keys),
        "prefix_expanded_keys": sorted(prefix_expanded_keys),
        "missing_target_keys": sorted(missing_keys),
        "shape_mismatch_keys": shape_mismatches,
        "unused_source_keys": unused_source_keys,
    }
    same_semantic_encoder = (
        config.input_encoder_mode == "semantic"
        and source.model.config.input_encoder_mode == "semantic"
        and _semantic_encoder_spec_from_config(config)
        == _semantic_encoder_spec_from_config(source.model.config)
    )
    same_hash_tokenizer = (
        config.input_encoder_mode == "hash"
        and source.model.config.input_encoder_mode == "hash"
        and config.vocab_size == source.model.config.vocab_size
    )
    metadata = dict(source.metadata)
    metadata["_training_initialization_manifest"] = manifest
    return DocumentHRMBundle(
        target_model,
        tokenizer=source.tokenizer if same_hash_tokenizer else None,
        metadata=metadata,
        semantic_encoder=(
            source.semantic_encoder if same_semantic_encoder else None
        ),
    )


def _configure_pair_training_stage(
    model: DocumentHierarchicalReasoningModel,
    *,
    pair_head_only: bool,
    freeze_operation_head: bool = False,
) -> None:
    pair_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("pair_")
    ]
    if pair_head_only and not pair_parameters:
        raise DocumentHRMError(
            "pair-head-only training requires joint-pair parameters"
        )
    for name, parameter in model.named_parameters():
        trainable = not pair_head_only or name.startswith("pair_")
        if freeze_operation_head and name.startswith("operation_head."):
            trainable = False
        parameter.requires_grad_(trainable)


def _is_unary_pointer_parameter(name: str) -> bool:
    return name == "pointer_relevance_scale" or name.startswith("pointer_")


def _is_pointer_head_parameter(name: str) -> bool:
    return _is_unary_pointer_parameter(name) or name.startswith("pair_")


def _configure_pointer_heads_only_training(
    model: DocumentHierarchicalReasoningModel,
) -> None:
    unary_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if _is_unary_pointer_parameter(name)
    ]
    pair_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("pair_")
    ]
    if not unary_parameters or not pair_parameters:
        raise DocumentHRMError(
            "pointer-heads-only training requires unary and joint-pair parameters"
        )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_pointer_head_parameter(name))


def _is_retrieval_training_parameter(name: str) -> bool:
    return name.startswith(
        (
            "semantic_adapter_",
            "pointer_",
            "pair_",
            "sum_",
            "financial_",
        )
    )


def _configure_retrieval_heads_only_training(
    model: DocumentHierarchicalReasoningModel,
) -> None:
    selected = [
        name
        for name, _parameter in model.named_parameters()
        if _is_retrieval_training_parameter(name)
    ]
    required_prefixes = (
        "semantic_adapter_",
        "pointer_",
        "sum_operand_count_head.",
        "sum_membership_query.",
        "financial_slot_",
    )
    missing = [
        prefix
        for prefix in required_prefixes
        if not any(name.startswith(prefix) for name in selected)
    ]
    if missing:
        raise DocumentHRMError(
            "retrieval-heads-only architecture is incomplete: "
            + ", ".join(missing)
        )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(_is_retrieval_training_parameter(name))


def _build_retrieval_heads_only_optimizer(
    model: DocumentHierarchicalReasoningModel,
    *,
    learning_rate: float,
) -> torch.optim.Optimizer:
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        raise DocumentHRMError(
            "retrieval-heads-only training has no trainable parameters"
        )
    return torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=2e-4
    )


def _configure_unary_pointer_training_stage(
    model: DocumentHierarchicalReasoningModel,
    *,
    pointer_head_only: bool,
    freeze_operation_head: bool = False,
    freeze_pair_head: bool = False,
) -> None:
    pointer_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if _is_unary_pointer_parameter(name)
    ]
    if pointer_head_only and not pointer_parameters:
        raise DocumentHRMError(
            "pointer-head-only training requires unary pointer parameters"
        )
    for name, parameter in model.named_parameters():
        trainable = not pointer_head_only or _is_unary_pointer_parameter(name)
        if freeze_operation_head and name.startswith("operation_head."):
            trainable = False
        if freeze_pair_head and name.startswith("pair_"):
            trainable = False
        parameter.requires_grad_(trainable)


def _build_staged_pair_optimizer(
    model: DocumentHierarchicalReasoningModel,
    *,
    learning_rate: float,
    backbone_learning_rate: float | None,
) -> torch.optim.Optimizer:
    if backbone_learning_rate is None:
        return torch.optim.AdamW(
            model.parameters(), learning_rate, weight_decay=2e-4
        )
    pair_parameters: list[Tensor] = []
    backbone_parameters: list[Tensor] = []
    for name, parameter in model.named_parameters():
        target = pair_parameters if name.startswith("pair_") else backbone_parameters
        target.append(parameter)
    if not pair_parameters:
        raise DocumentHRMError(
            "a separate backbone learning rate requires joint-pair parameters"
        )
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_learning_rate},
            {"params": pair_parameters, "lr": learning_rate},
        ],
        weight_decay=2e-4,
    )


def _build_staged_pointer_optimizer(
    model: DocumentHierarchicalReasoningModel,
    *,
    learning_rate: float,
    backbone_learning_rate: float | None,
) -> torch.optim.Optimizer:
    if backbone_learning_rate is None:
        return torch.optim.AdamW(
            model.parameters(), learning_rate, weight_decay=2e-4
        )
    pointer_parameters: list[Tensor] = []
    backbone_parameters: list[Tensor] = []
    for name, parameter in model.named_parameters():
        target = (
            pointer_parameters
            if _is_unary_pointer_parameter(name)
            else backbone_parameters
        )
        target.append(parameter)
    if not pointer_parameters:
        raise DocumentHRMError(
            "a separate pointer learning rate requires unary pointer parameters"
        )
    return torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": backbone_learning_rate},
            {"params": pointer_parameters, "lr": learning_rate},
        ],
        weight_decay=2e-4,
    )


def _build_pointer_heads_only_optimizer(
    model: DocumentHierarchicalReasoningModel,
    *,
    learning_rate: float,
) -> torch.optim.Optimizer:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise DocumentHRMError(
            "pointer-heads-only optimizer has no trainable parameters"
        )
    return torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=0.0,
    )


def _training_segment_loss(
    *,
    operation_loss: Tensor,
    pointer_loss: Tensor,
    pair_loss: Tensor,
    q_loss: Tensor,
    pair_loss_scale: float,
    pointer_heads_only: bool,
) -> Tensor:
    if pointer_heads_only:
        return 2.00 * pointer_loss + pair_loss_scale * pair_loss
    return (
        operation_loss
        + 2.00 * pointer_loss
        + pair_loss_scale * pair_loss
        + 0.20 * q_loss
    )


def _combine_segment_losses(segment_losses: Sequence[Tensor]) -> Tensor:
    """Favor later HRM refinements while retaining early deep supervision."""

    if not segment_losses:
        raise DocumentHRMError("segment losses cannot be empty")
    stacked = torch.stack(tuple(segment_losses))
    weights = torch.arange(
        1,
        len(segment_losses) + 1,
        dtype=stacked.dtype,
        device=stacked.device,
    )
    return (stacked * weights).sum() / weights.sum()


def _configure_select_pointer_training(
    model: DocumentHierarchicalReasoningModel,
) -> None:
    if model.pair_operation_bilinear is None:
        raise DocumentHRMError(
            "SELECT residual training requires joint-pair parameters"
        )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name == "pair_operation_bilinear")


def _build_select_pointer_optimizer(
    model: DocumentHierarchicalReasoningModel,
    *,
    learning_rate: float,
) -> torch.optim.Optimizer:
    if model.pair_operation_bilinear is None:
        raise DocumentHRMError(
            "SELECT residual training requires joint-pair parameters"
        )
    return torch.optim.AdamW(
        [model.pair_operation_bilinear],
        lr=learning_rate,
        weight_decay=0.0,
    )


def _common_operation_selection_score(
    metrics: dict[str, float],
) -> tuple[float, float, float, float]:
    """Weight model selection by common operations, excluding SUM3/SUM4."""

    def weighted(metric: str) -> float:
        numerator = 0.0
        denominator = 0.0
        for operation in sorted(VERIFIER_GUIDED_OPERATIONS):
            prefix = f"program_operation_{operation.lower()}"
            support = metrics[f"{prefix}_support"]
            numerator += metrics[f"{prefix}_{metric}"] * support
            denominator += support
        return numerator / denominator if denominator else 0.0

    return (
        weighted("execution_accuracy"),
        weighted("pointer_pair_accuracy"),
        weighted("program_exact_accuracy"),
        metrics["operation_macro_f1"],
    )


def train_document_hrm(
    dataset_path: str | Path,
    *,
    config: DocumentHRMConfig | None = None,
    epochs: int = 12,
    batch_size: int = 32,
    learning_rate: float = 1.5e-3,
    validation_fraction: float = 0.15,
    validation_example_ids: set[str] | None = None,
    validation_dataset_path: str | Path | None = None,
    allow_synthetic_validation: bool = False,
    synthetic_fraction: float = 1.0,
    select_best_epoch: bool = False,
    initialize_from: str | Path | None = None,
    initialization_mode: str = "exact",
    pointer_heads_only: bool = False,
    retrieval_heads_only: bool = False,
    pair_head_only_epochs: int = 0,
    pointer_head_only_epochs: int = 0,
    pointer_targeted_negative_weight: float = 0.0,
    verifier_guided_pair_weight: float = 0.0,
    verifier_guided_unary_weight: float = 0.0,
    verifier_guided_margin: float = 0.75,
    add_oversample_factor: float = 1.0,
    add_select_contrastive_weight: float = 0.0,
    add_select_contrastive_margin: float = 0.75,
    semantic_contrastive_weight: float = 0.0,
    semantic_contrastive_temperature: float = 0.07,
    semantic_contrastive_margin: float = 0.20,
    backbone_learning_rate: float | None = None,
    freeze_operation_head: bool = False,
    freeze_pair_head: bool = False,
    preserve_initial_threshold: bool = False,
    select_pointer_only: bool = False,
    abstention_target_precision: float = 0.80,
    abstention_stability_errors: int = DEFAULT_ABSTENTION_STABILITY_ERRORS,
    best_training_state_path: str | Path | None = None,
    last_training_state_path: str | Path | None = None,
    resume_training_state_path: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> tuple[DocumentHRMBundle, DocumentHRMTrainingReport]:
    config = config or DocumentHRMConfig()
    config.validate()
    if initialization_mode not in {"exact", "compatible_backbone"}:
        raise DocumentHRMError(
            "initialization_mode must be exact or compatible_backbone"
        )
    if initialization_mode != "exact" and initialize_from is None:
        raise DocumentHRMError(
            "compatible_backbone initialization requires initialize_from"
        )
    if resume_training_state_path is not None and initialize_from is not None:
        raise DocumentHRMError(
            "resume_training_state_path and initialize_from are mutually exclusive"
        )
    if best_training_state_path is not None and not select_best_epoch:
        raise DocumentHRMError(
            "best_training_state_path requires select_best_epoch"
        )

    if validation_dataset_path is not None and validation_example_ids is not None:
        raise DocumentHRMError(
            "validation_dataset_path and validation_example_ids are mutually exclusive"
        )
    if epochs < 1 or batch_size < 1:
        raise DocumentHRMError("epochs and batch_size must be positive")
    if not 0.0 <= synthetic_fraction <= 1.0:
        raise DocumentHRMError("synthetic_fraction must be within [0, 1]")
    if not 0.0 < abstention_target_precision <= 1.0:
        raise DocumentHRMError("abstention_target_precision must be within (0, 1]")
    if abstention_stability_errors < 0:
        raise DocumentHRMError("abstention_stability_errors must be non-negative")
    if pair_head_only_epochs < 0 or pair_head_only_epochs >= epochs:
        raise DocumentHRMError(
            "pair_head_only_epochs must be within [0, epochs)"
        )
    if pointer_head_only_epochs < 0 or pointer_head_only_epochs >= epochs:
        raise DocumentHRMError(
            "pointer_head_only_epochs must be within [0, epochs)"
        )
    if pair_head_only_epochs and pointer_head_only_epochs:
        raise DocumentHRMError(
            "pair-head-only and pointer-head-only schedules are mutually exclusive"
        )
    if pointer_heads_only and (
        pair_head_only_epochs
        or pointer_head_only_epochs
        or select_pointer_only
        or freeze_operation_head
        or freeze_pair_head
    ):
        raise DocumentHRMError(
            "pointer-heads-only training cannot use another freeze/head-only mode"
        )
    if retrieval_heads_only and (
        pointer_heads_only
        or pair_head_only_epochs
        or pointer_head_only_epochs
        or select_pointer_only
        or freeze_pair_head
    ):
        raise DocumentHRMError(
            "retrieval-heads-only training cannot use another head-only mode"
        )
    if retrieval_heads_only and initialize_from is None:
        raise DocumentHRMError(
            "retrieval-heads-only training requires initialize_from"
        )
    if (
        retrieval_heads_only
        and config.semantic_adapter_mode != "layernorm_linear_l2_v1"
    ):
        raise DocumentHRMError(
            "retrieval-heads-only training requires the semantic adapter"
        )
    if retrieval_heads_only and (
        config.variable_sum_mode != "sum_n_v1"
        or config.financial_structure_mode != "typed_hierarchy_v1"
    ):
        raise DocumentHRMError(
            "retrieval-heads-only training requires SUM_N and typed hierarchy"
        )
    if pointer_heads_only and initialize_from is None:
        raise DocumentHRMError(
            "pointer-heads-only training requires initialize_from"
        )
    if pointer_heads_only and config.pointer_conditioning_mode != "joint_pair":
        raise DocumentHRMError(
            "pointer-heads-only training requires joint_pair conditioning"
        )
    if pointer_heads_only and config.pair_loss_weight <= 0:
        raise DocumentHRMError(
            "pointer-heads-only training requires a positive pair loss weight"
        )
    if pointer_heads_only and backbone_learning_rate is not None:
        raise DocumentHRMError(
            "pointer-heads-only training does not update the backbone"
        )
    if pointer_heads_only and not preserve_initial_threshold:
        raise DocumentHRMError(
            "pointer-heads-only training requires preserve_initial_threshold"
        )
    if select_pointer_only and (pair_head_only_epochs or pointer_head_only_epochs):
        raise DocumentHRMError(
            "SELECT residual training cannot use another head-only schedule"
        )
    if select_pointer_only and config.pointer_conditioning_mode != "joint_pair":
        raise DocumentHRMError(
            "SELECT residual training requires joint_pair conditioning"
        )
    if select_pointer_only and initialize_from is None:
        raise DocumentHRMError(
            "SELECT residual training requires initialize_from"
        )
    if select_pointer_only and backbone_learning_rate is not None:
        raise DocumentHRMError(
            "SELECT residual training does not update the backbone"
        )
    if pair_head_only_epochs and config.pointer_conditioning_mode != "joint_pair":
        raise DocumentHRMError(
            "pair-head-only training requires joint_pair conditioning"
        )
    if pair_head_only_epochs and config.pair_loss_weight <= 0:
        raise DocumentHRMError(
            "pair-head-only training requires a positive pair loss weight"
        )
    if backbone_learning_rate is not None and backbone_learning_rate <= 0:
        raise DocumentHRMError("backbone_learning_rate must be positive")
    if pointer_targeted_negative_weight < 0:
        raise DocumentHRMError(
            "pointer_targeted_negative_weight must be non-negative"
        )
    if verifier_guided_pair_weight < 0 or verifier_guided_unary_weight < 0:
        raise DocumentHRMError(
            "verifier-guided loss weights must be non-negative"
        )
    if verifier_guided_margin < 0:
        raise DocumentHRMError(
            "verifier-guided margin must be non-negative"
        )
    if (
        verifier_guided_pair_weight > 0
        and config.pointer_conditioning_mode != "joint_pair"
    ):
        raise DocumentHRMError(
            "verifier-guided pair loss requires joint_pair conditioning"
        )
    if add_oversample_factor < 1.0:
        raise DocumentHRMError("add_oversample_factor must be at least 1")
    if add_select_contrastive_weight < 0:
        raise DocumentHRMError(
            "add_select_contrastive_weight must be non-negative"
        )
    if add_select_contrastive_margin < 0:
        raise DocumentHRMError(
            "add_select_contrastive_margin must be non-negative"
        )
    if semantic_contrastive_weight < 0:
        raise DocumentHRMError(
            "semantic_contrastive_weight must be non-negative"
        )
    if semantic_contrastive_temperature <= 0:
        raise DocumentHRMError(
            "semantic_contrastive_temperature must be positive"
        )
    if semantic_contrastive_margin < 0:
        raise DocumentHRMError(
            "semantic_contrastive_margin must be non-negative"
        )
    if (
        semantic_contrastive_weight > 0
        and config.semantic_adapter_mode != "layernorm_linear_l2_v1"
    ):
        raise DocumentHRMError(
            "semantic contrastive loss requires a trainable semantic adapter"
        )
    if preserve_initial_threshold and initialize_from is None:
        raise DocumentHRMError(
            "preserve_initial_threshold requires initialize_from"
        )
    if preserve_initial_threshold and initialization_mode != "exact":
        raise DocumentHRMError(
            "preserve_initial_threshold requires exact initialization"
        )
    training_source_path = _reject_protected_training_input(
        dataset_path, role="training data"
    )
    validation_source_path = (
        _reject_protected_training_input(
            validation_dataset_path, role="validation data"
        )
        if validation_dataset_path is not None
        else None
    )
    best_training_state_destination = (
        Path(best_training_state_path).expanduser().resolve()
        if best_training_state_path is not None
        else None
    )
    last_training_state_destination = (
        Path(last_training_state_path).expanduser().resolve()
        if last_training_state_path is not None
        else None
    )
    resume_training_state_source = (
        Path(resume_training_state_path).expanduser().resolve()
        if resume_training_state_path is not None
        else None
    )
    if (
        best_training_state_destination is not None
        and best_training_state_destination == last_training_state_destination
    ):
        raise DocumentHRMError(
            "best and last training-state paths must be different"
        )
    protected_state_targets = {
        training_source_path,
        *(
            (validation_source_path,)
            if validation_source_path is not None
            else ()
        ),
        *(
            (Path(initialize_from).expanduser().resolve(),)
            if initialize_from is not None
            else ()
        ),
    }
    for destination in (
        best_training_state_destination,
        last_training_state_destination,
    ):
        if destination is not None and destination in protected_state_targets:
            raise DocumentHRMError(
                "Training-state output overlaps an input artifact"
            )
        if (
            destination is not None
            and destination.exists()
            and resume_training_state_source is None
        ):
            raise DocumentHRMError(
                f"Training-state output already exists: {destination}"
            )



    examples = load_program_examples(training_source_path)
    if len(examples) < 50:
        raise DocumentHRMError("At least 50 induced program examples are required")
    if str(device).startswith("cpu") and torch.get_num_threads() > 4:
        torch.set_num_threads(4)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if validation_source_path is not None:
        train_examples = list(examples)
        validation_examples = load_program_examples(validation_source_path)
        split_rng = random.Random(config.seed)
        split_rng.shuffle(train_examples)
        split_rng.shuffle(validation_examples)
        validation_split_name = "explicit-company-disjoint-dataset"
    elif validation_example_ids:
        train_examples, validation_examples = _split_from_validation_ids(
            examples,
            validation_example_ids,
            config.seed,
            allow_synthetic_validation=allow_synthetic_validation,
        )
        validation_split_name = (
            "fixed-company-disjoint-explicit"
            if allow_synthetic_validation
            else "fixed-checkpoint-real-holdout"
        )
    else:
        train_examples, validation_examples = _grouped_real_split(
            examples, validation_fraction, config.seed
        )
        validation_split_name = "company-grouped-balanced-real-only"
    _validate_training_split_boundaries(train_examples, validation_examples)
    real_train = [example for example in train_examples if not _is_synthetic(example)]
    synthetic_train = [example for example in train_examples if _is_synthetic(example)]
    synthetic_per_epoch = round(len(synthetic_train) * synthetic_fraction)
    initial_bundle = (
        _load_compatible_training_bundle(
            initialize_from,
            config,
            device,
            initialization_mode=initialization_mode,
        )
        if initialize_from is not None
        else None
    )
    tokenizer = (
        initial_bundle.tokenizer
        if initial_bundle is not None
        else HashSequenceTokenizer(config.vocab_size)
        if config.input_encoder_mode == "hash"
        else None
    )
    semantic_encoder = (
        initial_bundle.semantic_encoder
        if initial_bundle is not None
        else SemanticEmbeddingCache(
            config.semantic_model_name,
            config.semantic_encoder_size,
            DEFAULT_SEMANTIC_CACHE_DIR,
            spec=_semantic_encoder_spec_from_config(config),
        )
        if config.input_encoder_mode == "semantic"
        else None
    )
    if semantic_encoder is not None:
        semantic_encoder.warm_examples(
            [*train_examples, *validation_examples],
            batch_size=batch_size,
            candidate_semantic_mode=config.candidate_semantic_mode,
        )
    model = (
        initial_bundle.model
        if initial_bundle is not None
        else DocumentHierarchicalReasoningModel(config).to(device)
    )
    counts = Counter(
        (
            "ADD"
            if config.variable_sum_mode == "sum_n_v1"
            and is_sum_operation(example.program.operation)
            else example.program.operation
        )
        for example in [
            *synthetic_train[:synthetic_per_epoch],
            *real_train,
            *real_train,
            *real_train,
        ]
    )
    operation_weights = torch.tensor(
        [1.0 / math.sqrt(max(1, counts[name])) for name in OPERATIONS],
        dtype=torch.float32,
        device=device,
    )
    operation_weights = operation_weights / operation_weights.mean()
    unary_pointer_schedule = bool(pointer_head_only_epochs or freeze_pair_head)
    if retrieval_heads_only:
        _configure_retrieval_heads_only_training(model)
        optimizer = _build_retrieval_heads_only_optimizer(
            model,
            learning_rate=learning_rate,
        )
    elif pointer_heads_only:
        _configure_pointer_heads_only_training(model)
        optimizer = _build_pointer_heads_only_optimizer(
            model,
            learning_rate=learning_rate,
        )
    elif select_pointer_only:
        optimizer = _build_select_pointer_optimizer(
            model,
            learning_rate=learning_rate,
        )
    elif unary_pointer_schedule:
        optimizer = _build_staged_pointer_optimizer(
            model,
            learning_rate=learning_rate,
            backbone_learning_rate=backbone_learning_rate,
        )
    else:
        optimizer = _build_staged_pair_optimizer(
            model,
            learning_rate=learning_rate,
            backbone_learning_rate=backbone_learning_rate,
        )
    rng = random.Random(config.seed)
    final_loss = 0.0
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_loss = float("inf")
    best_score = (-1.0, -1.0, -1.0, -1.0)
    verifier_guided_training = (
        verifier_guided_pair_weight > 0
        or verifier_guided_unary_weight > 0
    )

    def selection_score(metrics: dict[str, float]) -> tuple[float, float, float, float]:
        if retrieval_heads_only:
            return (
                metrics["gold_operation_conditioned_candidate_recall_at_20"],
                metrics["gold_operation_conditioned_candidate_recall_at_10"],
                metrics["gold_operation_conditioned_full_coverage_at_20"],
                metrics["gold_operation_conditioned_full_coverage_at_10"],
            )
        if verifier_guided_training:
            return _common_operation_selection_score(metrics)
        return (
            metrics["execution_accuracy"],
            metrics["pointer_pair_accuracy"],
            metrics["program_exact_accuracy"],
            metrics["operation_macro_f1"],
        )
    training_contract = {
        "schema": "document-hrm-training-contract-v1",
        "runtime": {
            "document_hrm_sha256": _sha256_file(Path(__file__).resolve()),
            "training_state_sha256": _sha256_file(
                Path(__file__).resolve().with_name("training_state.py")
            ),
        },
        "config": asdict(config),
        "data": {
            "training_path": str(training_source_path),
            "training_sha256": _sha256_file(training_source_path),
            "validation_path": (
                str(validation_source_path)
                if validation_source_path is not None
                else None
            ),
            "validation_sha256": (
                _sha256_file(validation_source_path)
                if validation_source_path is not None
                else None
            ),
            "validation_example_ids": (
                sorted(validation_example_ids)
                if validation_example_ids is not None
                else None
            ),
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "validation_fraction": validation_fraction,
            "allow_synthetic_validation": allow_synthetic_validation,
            "synthetic_fraction": synthetic_fraction,
            "select_best_epoch": select_best_epoch,
            "initialization_mode": (
                initialization_mode if initialize_from is not None else "random"
            ),
            "initialize_from_sha256": (
                _sha256_file(Path(initialize_from).expanduser().resolve())
                if initialize_from is not None
                else None
            ),
            "pointer_heads_only": pointer_heads_only,
            "retrieval_heads_only": retrieval_heads_only,
            "pair_head_only_epochs": pair_head_only_epochs,
            "pointer_head_only_epochs": pointer_head_only_epochs,
            "pointer_targeted_negative_weight": pointer_targeted_negative_weight,
            "verifier_guided_pair_weight": verifier_guided_pair_weight,
            "verifier_guided_unary_weight": verifier_guided_unary_weight,
            "verifier_guided_margin": verifier_guided_margin,
            "add_oversample_factor": add_oversample_factor,
            "add_select_contrastive_weight": add_select_contrastive_weight,
            "add_select_contrastive_margin": add_select_contrastive_margin,
            "semantic_contrastive_weight": semantic_contrastive_weight,
            "semantic_contrastive_temperature": semantic_contrastive_temperature,
            "semantic_contrastive_margin": semantic_contrastive_margin,
            "backbone_learning_rate": backbone_learning_rate,
            "freeze_operation_head": freeze_operation_head,
            "freeze_pair_head": freeze_pair_head,
            "preserve_initial_threshold": preserve_initial_threshold,
            "select_pointer_only": select_pointer_only,
            "abstention_target_precision": abstention_target_precision,
            "abstention_stability_errors": abstention_stability_errors,
            "selection_metric_schema": (
                "v14-logical-sum-family-selection-v1"
                if config.variable_sum_mode == "sum_n_v1"
                else "legacy-operation-label-selection-v1"
            ),
            "device_type": torch.device(device).type,
            "optimizer_type": type(optimizer).__qualname__,
            "optimizer_group_parameter_counts": [
                len(group["params"]) for group in optimizer.param_groups
            ],
        },
    }
    training_contract_sha256 = canonical_payload_sha256(training_contract)
    start_epoch = 0
    resumed_training_state_sha256: str | None = None
    if resume_training_state_source is not None:
        resumed_training_state_sha256 = _sha256_file(
            resume_training_state_source
        )
        try:
            resume_payload = load_training_state(resume_training_state_source)
        except TrainingStateError as exc:
            raise DocumentHRMError(str(exc)) from exc
        if (
            resume_payload["training_contract_sha256"]
            != training_contract_sha256
        ):
            raise DocumentHRMError(
                "Resume training contract differs from the current run"
            )
        start_epoch = int(resume_payload["completed_epoch"])
        if start_epoch > epochs:
            raise DocumentHRMError(
                "Resume state completed epoch exceeds requested epochs"
            )
        try:
            model.load_state_dict(resume_payload["model_state_dict"])
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise DocumentHRMError(
                "Resume model or optimizer state is incompatible"
            ) from exc
        restore_optimizer_state_device(optimizer, device)
        raw_best_state = resume_payload.get("best_state_dict")
        best_state = (
            {
                str(name): tensor.detach().cpu().clone()
                for name, tensor in raw_best_state.items()
            }
            if isinstance(raw_best_state, dict)
            else None
        )
        raw_best_score = resume_payload.get("best_score")
        if not isinstance(raw_best_score, list) or len(raw_best_score) != 4:
            raise DocumentHRMError("Resume state has an invalid best score")
        best_score = tuple(float(value) for value in raw_best_score)
        best_epoch = int(resume_payload["best_epoch"])
        best_loss = float(resume_payload["best_loss"])
        final_loss = float(resume_payload["final_loss"])
        if select_best_epoch and best_state is None:
            raise DocumentHRMError("Resume state is missing the best model")
        random.setstate(
            decode_python_random_state(resume_payload["python_random_state"])
        )
        rng.setstate(
            decode_python_random_state(resume_payload["training_rng_state"])
        )
        torch.random.set_rng_state(resume_payload["torch_cpu_rng_state"])
        raw_cuda_rng = resume_payload.get("torch_cuda_rng_states")
        if torch.device(device).type == "cuda":
            if not isinstance(raw_cuda_rng, list):
                raise DocumentHRMError(
                    "Resume state is missing CUDA RNG state"
                )
            torch.cuda.set_rng_state_all(raw_cuda_rng)
        print(
            f"Resumed training after epoch {start_epoch}/{epochs}: "
            f"{resume_training_state_source}",
            flush=True,
        )






    if (
        select_best_epoch
        and initial_bundle is not None
        and resume_training_state_source is None
    ):
        initial_metrics = (
            evaluate_document_hrm_candidate_retrieval(
                initial_bundle,
                validation_examples,
                batch_size=batch_size,
            )
            if retrieval_heads_only
            else evaluate_document_hrm(
                initial_bundle,
                validation_examples,
                batch_size=batch_size,
            )
        )
        best_score = selection_score(initial_metrics)
        best_loss = 0.0
        best_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        if retrieval_heads_only:
            print(
                "Initialized epoch 0: "
                f"gold_op_candidate@20={best_score[0]:.2%}, "
                f"gold_op_candidate@10={best_score[1]:.2%}, "
                f"full@20={best_score[2]:.2%}, "
                f"full@10={best_score[3]:.2%}"
            )
        else:
            print(
                "Initialized epoch 0: "
                f"execution={best_score[0]:.2%}, pair={best_score[1]:.2%}, "
                f"program={best_score[2]:.2%}, macro_f1={best_score[3]:.2%}"
            )
    for _epoch in range(start_epoch, epochs):
        print(
            f"Epoch {_epoch + 1}/{epochs}: training started",
            flush=True,
        )

        pair_head_only = _epoch < pair_head_only_epochs
        pointer_head_only = _epoch < pointer_head_only_epochs
        if retrieval_heads_only:
            _configure_retrieval_heads_only_training(model)
        elif pointer_heads_only:
            _configure_pointer_heads_only_training(model)
        elif select_pointer_only:
            _configure_select_pointer_training(model)
        elif unary_pointer_schedule:
            _configure_unary_pointer_training_stage(
                model,
                pointer_head_only=pointer_head_only,
                freeze_operation_head=freeze_operation_head,
                freeze_pair_head=freeze_pair_head,
            )
        else:
            _configure_pair_training_stage(
                model,
                pair_head_only=pair_head_only,
                freeze_operation_head=freeze_operation_head,
            )
        if retrieval_heads_only:
            model.pair_backbone_detached = True
        elif pointer_heads_only:
            model.pair_backbone_detached = True
        elif pointer_head_only:
            model.pair_backbone_detached = True
        elif pair_head_only:
            model.pair_backbone_detached = True
        elif config.pair_backbone_mode == "warmup_end_to_end":
            model.pair_backbone_detached = (
                _epoch < config.pair_backbone_warmup_epochs
            )
        if retrieval_heads_only:
            pair_loss_scale = config.pair_loss_weight
        elif pointer_heads_only:
            pair_loss_scale = config.pair_loss_weight
        elif select_pointer_only or unary_pointer_schedule:
            pair_loss_scale = 0.0
        elif pair_head_only:
            pair_loss_scale = config.pair_loss_weight
        else:
            pair_loss_scale = _pair_loss_scale(config, _epoch)
        if config.pointer_conditioning_mode in {
            "operation_sequential",
            "later_sequential",
        }:
            # Start with partial gold context to bootstrap p2+, then remove it
            # by the final epoch so training converges under inference-like
            # predicted-pointer context.
            remaining = 1.0 - (_epoch / max(1, epochs - 1))
            model.pointer_teacher_forcing_weight = (
                config.pointer_teacher_forcing * remaining
            )
        epoch_synthetic = (
            rng.sample(synthetic_train, synthetic_per_epoch)
            if synthetic_per_epoch < len(synthetic_train)
            else list(synthetic_train)
        )
        epoch_examples = [
            *epoch_synthetic,
            *real_train,
            *(_permute_candidates(example, rng) for example in real_train),
            *(_permute_candidates(example, rng) for example in real_train),
        ]
        add_examples = [
            example
            for example in epoch_examples
            if example.program.operation == "ADD"
            and _question_features(example.question)[2] > 0.5
        ]
        select_count = sum(
            example.program.operation == "SELECT" for example in epoch_examples
        )
        desired_add_count = min(
            select_count,
            round(len(add_examples) * add_oversample_factor),
        )
        for index in range(max(0, desired_add_count - len(add_examples))):
            epoch_examples.append(
                _permute_candidates(add_examples[index % len(add_examples)], rng)
            )
        if pair_head_only:
            epoch_examples = [
                example
                for example in epoch_examples
                if len(example.program.operands) >= 2
            ]
            if not epoch_examples:
                raise DocumentHRMError(
                    "pair-head-only stage has no multi-operand training examples"
                )
        if select_pointer_only:
            epoch_examples = [
                example
                for example in epoch_examples
                if example.program.operation == "SELECT"
            ]
            if not epoch_examples:
                raise DocumentHRMError(
                    "SELECT residual training has no SELECT examples"
                )
        rng.shuffle(epoch_examples)
        model.train()
        running = 0.0
        updates = 0
        epoch_batches = max(1, math.ceil(len(epoch_examples) / batch_size))
        progress_every = max(1, math.ceil(epoch_batches / 20))


        for start in range(0, len(epoch_examples), batch_size):
            current = epoch_examples[start : start + batch_size]
            batch = encode_examples(
                current,
                tokenizer,
                semantic_encoder,
                config,
                torch.device(device),
                include_verifier_negatives=(
                    verifier_guided_pair_weight > 0
                    or verifier_guided_unary_weight > 0
                ),
            )
            semantic_loss = (
                _semantic_adapter_contrastive_loss(
                    model,
                    batch,
                    temperature=semantic_contrastive_temperature,
                    hard_negative_margin=semantic_contrastive_margin,
                )
                if semantic_contrastive_weight > 0
                else torch.zeros((), device=device)
            )
            state: DocumentHRMState | None = None
            encoded: tuple[Tensor, Tensor] | None = None
            outputs: list[DocumentHRMForward] = []
            for _segment in range(config.max_segments):
                output, encoded = model.forward_segment(batch, state, encoded)
                outputs.append(output)
                state = output.state.detach()

            segment_losses: list[Tensor] = []
            for segment, output in enumerate(outputs):
                assert batch.sum_family_targets is not None
                assert batch.operand_count_targets is not None
                operation_loss = _operation_training_loss(
                    output.operation_logits,
                    batch.operation_targets,
                    batch.sum_family_targets,
                    weight=operation_weights,
                )
                if output.operand_count_logits is not None:
                    count_valid = batch.operand_count_targets.ge(0)
                    if count_valid.any():
                        operation_loss = operation_loss + F.cross_entropy(
                            output.operand_count_logits[count_valid],
                            batch.operand_count_targets[count_valid],
                        )
                if add_select_contrastive_weight > 0:
                    operation_loss = operation_loss + (
                        add_select_contrastive_weight
                        * _add_select_contrastive_loss(
                            output.operation_logits,
                            batch.operation_targets,
                            margin=add_select_contrastive_margin,
                        )
                    )
                pointer_losses: list[Tensor] = []
                commutative_rows = torch.tensor(
                    [
                        OPERATIONS[int(target)] in COMMUTATIVE
                        for target in batch.operation_targets.tolist()
                    ],
                    dtype=torch.bool,
                    device=device,
                )
                for slot in range(config.max_program_operands):
                    valid = (
                        batch.pointer_targets[:, slot].ge(0)
                        & ~commutative_rows
                    )
                    if valid.any():
                        slot_logits = _training_pointer_slot_logits(
                            output, batch.operation_targets, slot
                        )
                        slot_loss = _hard_negative_cross_entropy(
                            slot_logits[valid],
                            batch.pointer_targets[valid, slot],
                            margin=config.pointer_hard_negative_margin,
                            weight=config.pointer_hard_negative_weight,
                        )
                        if pointer_targeted_negative_weight > 0:
                            assert batch.unary_structural_negatives is not None
                            slot_loss = slot_loss + (
                                pointer_targeted_negative_weight
                                * _targeted_unary_margin_loss(
                                    slot_logits[valid],
                                    batch.pointer_targets[valid, slot],
                                    batch.unary_structural_negatives[valid, slot],
                                    margin=config.pointer_hard_negative_margin,
                                )
                            )
                        if verifier_guided_unary_weight > 0:
                            assert (
                                batch.verifier_unary_invalid_negatives
                                is not None
                            )
                            slot_loss = slot_loss + (
                                verifier_guided_unary_weight
                                * _targeted_unary_margin_loss(
                                    slot_logits[valid],
                                    batch.pointer_targets[valid, slot],
                                    batch.verifier_unary_invalid_negatives[
                                        valid, slot
                                    ],
                                    margin=verifier_guided_margin,
                                )
                            )
                        pointer_losses.append(slot_loss)
                if commutative_rows.any():
                    pointer_losses.append(
                        _commutative_pointer_set_loss(
                            output.pointer_logits,
                            batch.operation_targets,
                            batch.pointer_targets,
                            batch.candidate_mask,
                            operand_count_targets=batch.operand_count_targets,
                            sum_family_targets=batch.sum_family_targets,
                            sum_membership_logits=output.sum_membership_logits,
                        )
                    )
                pointer_loss = torch.stack(pointer_losses).mean()
                pair_loss = torch.zeros((), device=device)
                if output.pair_logits is not None:
                    pair_valid = (
                        batch.pointer_targets[:, 1].ge(0)
                        & _binary_operation_rows(batch.operation_targets)
                        & ~batch.sum_family_targets
                    )
                    if pair_valid.any():
                        rows = torch.arange(
                            batch.operation_targets.shape[0], device=device
                        )
                        operation_pairs = output.pair_logits[
                            rows, batch.operation_targets
                        ]
                        flat_pairs = operation_pairs[pair_valid].flatten(1)
                        pair_width = operation_pairs.shape[-1]
                        pair_targets = (
                            batch.pointer_targets[pair_valid, 0] * pair_width
                            + batch.pointer_targets[pair_valid, 1]
                        )
                        pair_loss = _pair_program_loss(
                            flat_pairs,
                            batch.pointer_targets[pair_valid],
                            batch.operation_targets[pair_valid],
                            margin=0.50,
                            hard_negative_weight=0.25,
                        )
                        if config.pair_targeted_negative_weight > 0:
                            assert batch.pair_features is not None
                            directed = torch.tensor(
                                [
                                    OPERATIONS[int(target)]
                                    not in COMMUTATIVE
                                    for target in batch.operation_targets[
                                        pair_valid
                                    ].tolist()
                                ],
                                dtype=torch.bool,
                                device=device,
                            )
                            if directed.any():
                                pair_loss = pair_loss + (
                                    config.pair_targeted_negative_weight
                                    * _targeted_pair_margin_loss(
                                        flat_pairs[directed],
                                        pair_targets[directed],
                                        batch.pair_features[pair_valid][directed],
                                        batch.pair_structural_negatives[pair_valid][directed]
                                        if batch.pair_structural_negatives is not None
                                        else None,
                                    )
                                )
                        if verifier_guided_pair_weight > 0:
                            assert (
                                batch.verifier_pair_invalid_negatives
                                is not None
                            )
                            pair_loss = pair_loss + (
                                verifier_guided_pair_weight
                                * _verifier_guided_pair_margin_loss(
                                    flat_pairs,
                                    batch.pointer_targets[pair_valid],
                                    batch.operation_targets[pair_valid],
                                    batch.verifier_pair_invalid_negatives[
                                        pair_valid
                                    ],
                                    margin=verifier_guided_margin,
                                )
                            )

                predicted_operation = _operation_training_predictions(
                    output.operation_logits,
                    aggregate_sum_family=(
                        config.variable_sum_mode == "sum_n_v1"
                    ),
                )
                predicted_pointers = _training_pointer_predictions(
                    output,
                    batch.operation_targets,
                    batch.pointer_targets,
                    use_pair=config.pair_backbone_mode
                    in {"end_to_end", "normalized_end_to_end"},
                    operand_count_targets=batch.operand_count_targets,
                    sum_family_targets=batch.sum_family_targets,
                )
                operation_correct = predicted_operation.eq(batch.operation_targets)
                pointer_correct = torch.ones_like(operation_correct)
                for slot in range(config.max_program_operands):
                    valid = batch.pointer_targets[:, slot].ge(0)
                    pointer_correct = pointer_correct & (
                        ~valid
                        | predicted_pointers[:, slot].eq(batch.pointer_targets[:, slot])
                    )
                count_correct = torch.ones_like(operation_correct)
                if output.operand_count_logits is not None:
                    count_valid = batch.operand_count_targets.ge(0)
                    predicted_count = output.operand_count_logits.argmax(dim=-1)
                    count_correct = (
                        ~count_valid
                        | predicted_count.eq(batch.operand_count_targets)
                    )
                exact_correct = (
                    operation_correct & pointer_correct & count_correct
                ).float().detach()

                if config.act_mode == "q_learning":
                    # q_values are [halt, continue]. Q(halt) learns the immediate
                    # exact-program reward. Q(continue) bootstraps from the next
                    # reasoning segment, following the ACT objective in HRM.
                    halt_loss = F.binary_cross_entropy_with_logits(
                        output.q_values[:, 0], exact_correct
                    )
                    if segment + 1 < len(outputs):
                        with torch.no_grad():
                            next_values = torch.sigmoid(outputs[segment + 1].q_values)
                            continue_target = next_values.max(dim=-1).values
                    else:
                        continue_target = torch.zeros_like(exact_correct)
                    continue_loss = F.binary_cross_entropy_with_logits(
                        output.q_values[:, 1], continue_target
                    )
                    q_loss = 0.5 * (halt_loss + continue_loss)
                else:
                    halt_target = torch.where(
                        exact_correct.bool(),
                        torch.zeros_like(batch.operation_targets),
                        torch.ones_like(batch.operation_targets),
                    )
                    if segment == config.max_segments - 1:
                        halt_target = torch.zeros_like(halt_target)
                    q_loss = F.cross_entropy(output.q_values, halt_target)

                segment_loss = _training_segment_loss(
                        operation_loss=operation_loss,
                        pointer_loss=pointer_loss,
                        pair_loss=pair_loss,
                        q_loss=q_loss,
                        pair_loss_scale=pair_loss_scale,
                        pointer_heads_only=pointer_heads_only,
                    )
                if semantic_contrastive_weight > 0:
                    segment_loss = segment_loss + (
                        semantic_contrastive_weight * semantic_loss
                    )
                segment_losses.append(
                    segment_loss
                )
            loss = _combine_segment_losses(segment_losses)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if select_pointer_only:
                assert model.pair_operation_bilinear is not None
                gradient = model.pair_operation_bilinear.grad
                if gradient is not None:
                    select_gradient = gradient[OPERATION_TO_ID["SELECT"]].clone()
                    gradient.zero_()
                    gradient[OPERATION_TO_ID["SELECT"]] = select_gradient
            if (
                config.pointer_conditioning_mode == "joint_pair"
                and config.pair_backbone_mode != "normalized_end_to_end"
            ):
                pair_parameters: list[Tensor] = []
                base_parameters: list[Tensor] = []
                for name, parameter in model.named_parameters():
                    target = (
                        pair_parameters
                        if name.startswith("pair_")
                        else base_parameters
                    )
                    target.append(parameter)
                # Keep pair gradients from changing the clipping scale applied
                # to the otherwise identical independent-HRM update.
                nn.utils.clip_grad_norm_(base_parameters, 1.0)
                nn.utils.clip_grad_norm_(pair_parameters, 1.0)
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item())
            updates += 1
            if (
                updates == 1
                or updates % progress_every == 0
                or updates == epoch_batches
            ):
                print(
                    f"Epoch {_epoch + 1}/{epochs}: batch {updates}/{epoch_batches}, "
                    f"average_loss={running / updates:.5f}",
                    flush=True,
                )
        final_loss = running / max(1, updates)
        improved = False
        if select_best_epoch:
            epoch_bundle = DocumentHRMBundle(
                model,
                tokenizer,
                metadata={
                    "confidence_schema": "grounded-constraints-v2",
                    "evaluation_operation_schema": (
                        "logical-sum-family-v1"
                        if config.variable_sum_mode == "sum_n_v1"
                        else "legacy-operation-label-v1"
                    ),
                },
                semantic_encoder=semantic_encoder,
            )
            epoch_metrics = (
                evaluate_document_hrm_candidate_retrieval(
                    epoch_bundle,
                    validation_examples,
                    batch_size=batch_size,
                )
                if retrieval_heads_only
                else evaluate_document_hrm(
                    epoch_bundle,
                    validation_examples,
                    batch_size=batch_size,
                )
            )
            epoch_score = selection_score(epoch_metrics)
            if retrieval_heads_only:
                print(
                    f"Epoch {_epoch + 1}/{epochs}: loss={final_loss:.5f}, "
                    f"gold_op_candidate@20={epoch_score[0]:.2%}, "
                    f"gold_op_candidate@10={epoch_score[1]:.2%}, "
                    f"full@20={epoch_score[2]:.2%}, "
                    f"full@10={epoch_score[3]:.2%}",
                    flush=True,
                )
            else:
                print(
                    f"Epoch {_epoch + 1}/{epochs}: loss={final_loss:.5f}, "
                    f"execution={epoch_score[0]:.2%}, pair={epoch_score[1]:.2%}, "
                    f"program={epoch_score[2]:.2%}, macro_f1={epoch_score[3]:.2%}",
                    flush=True,
                )
            if epoch_score > best_score:
                improved = True
                best_score = epoch_score
                best_epoch = _epoch + 1
                best_loss = final_loss
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
        if (
            best_training_state_destination is not None
            or last_training_state_destination is not None
        ):
            current_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
            training_state_payload = {
                "schema": TRAINING_STATE_SCHEMA,
                "training_contract": training_contract,
                "training_contract_sha256": training_contract_sha256,
                "completed_epoch": _epoch + 1,
                "model_state_dict": current_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_state_dict": best_state,
                "best_score": list(best_score),
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "final_loss": final_loss,
                "python_random_state": encode_python_random_state(
                    random.getstate()
                ),
                "training_rng_state": encode_python_random_state(
                    rng.getstate()
                ),
                "torch_cpu_rng_state": torch.random.get_rng_state().clone(),
                "torch_cuda_rng_states": (
                    [state.cpu() for state in torch.cuda.get_rng_state_all()]
                    if torch.cuda.is_available()
                    else None
                ),
            }
            if improved and best_training_state_destination is not None:
                atomic_save_training_state(
                    training_state_payload,
                    best_training_state_destination,
                )
                print(
                    f"Saved best training state: {best_training_state_destination}",
                    flush=True,
                )
            if last_training_state_destination is not None:
                atomic_save_training_state(
                    training_state_payload,
                    last_training_state_destination,
                )
                print(
                    f"Saved last training state: {last_training_state_destination}",
                    flush=True,
                )



    if select_best_epoch:
        if best_state is None:
            raise DocumentHRMError(
                "Training completed without a selectable checkpoint"
            )
        model.load_state_dict(best_state)
        final_loss = best_loss
        if retrieval_heads_only:
            print(
                f"Selected best epoch: {best_epoch}/{epochs} "
                f"(gold_op_candidate@20={best_score[0]:.2%}, "
                f"gold_op_candidate@10={best_score[1]:.2%}, "
                f"full@20={best_score[2]:.2%}, "
                f"full@10={best_score[3]:.2%})",
                flush=True,
            )
        else:
            print(
                f"Selected best epoch: {best_epoch}/{epochs} "
                f"(execution={best_score[0]:.2%}, pair={best_score[1]:.2%}, "
                f"program={best_score[2]:.2%}, macro_f1={best_score[3]:.2%})",
                flush=True,
            )

    initialization_manifest = (
        initial_bundle.metadata.get("_training_initialization_manifest")
        if initial_bundle is not None
        else None
    )
    initialized_from_sha256: str | None = None
    if initialize_from is not None:
        initialized_from_sha256 = _sha256_file(
            Path(initialize_from).expanduser().resolve()
        )
        expected_source_sha256 = (
            initialization_manifest.get("source_checkpoint_sha256")
            if isinstance(initialization_manifest, dict)
            else None
        )
        if (
            isinstance(expected_source_sha256, str)
            and initialized_from_sha256 != expected_source_sha256
        ):
            raise DocumentHRMError(
                "initialize_from checkpoint changed during training"
            )

    bundle = DocumentHRMBundle(
        model,
        tokenizer,
        metadata={
            "architecture": (
                "source-grounded-financial-HRM-v14-variable-sum-typed-hierarchy-v1"
                if (
                    config.financial_structure_mode == "typed_hierarchy_v1"
                    and config.variable_sum_mode == "sum_n_v1"
                )
                else "source-grounded-financial-HRM-v13-typed-hierarchy-v1"
                if config.financial_structure_mode == "typed_hierarchy_v1"
                else "source-grounded-semantic-HRM-v10-verifier-guided-joint-pair"
                if verifier_guided_training
                else "source-grounded-semantic-HRM-v9-structured-"
                f"{config.pointer_conditioning_mode.replace('_', '-')}"
                if config.candidate_semantic_mode == "structured"
                else "source-grounded-semantic-HRM-v10-end-to-end-joint-pair"
                if (
                    config.pointer_conditioning_mode == "joint_pair"
                    and config.pair_backbone_mode == "end_to_end"
                )
                else "source-grounded-semantic-HRM-v10-normalized-end-to-end-joint-pair"
                if (
                    config.pointer_conditioning_mode == "joint_pair"
                    and config.pair_backbone_mode == "normalized_end_to_end"
                )
                else "source-grounded-semantic-HRM-v10-normalized-detached-joint-pair"
                if (
                    config.pointer_conditioning_mode == "joint_pair"
                    and config.pair_backbone_mode == "normalized_detached"
                )
                else "source-grounded-semantic-HRM-v10-warmup-end-to-end-joint-pair"
                if (
                    config.pointer_conditioning_mode == "joint_pair"
                    and config.pair_backbone_mode == "warmup_end_to_end"
                )
                else {
                    "independent": "source-grounded-semantic-HRM-v8-independent-pointer",
                    "operation_sequential": "source-grounded-semantic-HRM-v8-sequential-pointer",
                    "later_sequential": "source-grounded-semantic-HRM-v8-later-sequential-pointer",
                    "joint_pair": "source-grounded-semantic-HRM-v8-joint-pair",
                }[config.pointer_conditioning_mode]
            ),
            "confidence_schema": "grounded-constraints-v2",
            "grounding_gate_schema": "structured-evidence-v1",
            "training_dataset": training_source_path.name,
            "training_dataset_path": str(training_source_path),
            "training_dataset_sha256": _sha256_file(training_source_path),
            "training_examples": len(train_examples),
            "validation_examples": len(validation_examples),
            "validation_dataset_path": (
                str(validation_source_path)
                if validation_source_path is not None
                else None
            ),
            "validation_dataset_sha256": (
                _sha256_file(validation_source_path)
                if validation_source_path is not None
                else None
            ),
            "training_real_examples": len(real_train),
            "training_synthetic_examples": len(synthetic_train),
            "training_synthetic_examples_per_epoch": synthetic_per_epoch,
            "training_synthetic_fraction": synthetic_fraction,
            "select_best_epoch": select_best_epoch,
            "selected_best_epoch": best_epoch if select_best_epoch else epochs,
            "validation_real_examples": len(validation_examples),
            "validation_non_synthetic_examples": sum(
                not _is_synthetic(example) for example in validation_examples
            ),
            "validation_example_ids": [
                example.example_id for example in validation_examples
            ],
            "validation_split": validation_split_name,
            "training_seed": config.seed,
            "training_state_schema": TRAINING_STATE_SCHEMA,
            "best_training_state_path": (
                str(best_training_state_destination)
                if best_training_state_destination is not None
                else None
            ),
            "last_training_state_path": (
                str(last_training_state_destination)
                if last_training_state_destination is not None
                else None
            ),
            "resumed_training_state_path": (
                str(resume_training_state_source)
                if resume_training_state_source is not None
                else None
            ),
            "resumed_training_state_sha256": resumed_training_state_sha256,
            "resumed_after_epoch": start_epoch,
            "training_contract_sha256": training_contract_sha256,
            "training_epoch_checkpoint_policy": (
                "atomic_best_on_strict_improvement_and_last_every_epoch"
                if last_training_state_destination is not None
                else "disabled"
            ),
            "operations": list(OPERATIONS),
            "logical_program_operations": (
                list(_logical_evaluation_operations())
                if config.variable_sum_mode == "sum_n_v1"
                else list(OPERATIONS)
            ),
            # Metric semantics are part of the model/runtime contract. They
            # must not be inferred from calibration metadata, which is only
            # attached after epoch selection has already run.
            "evaluation_operation_schema": (
                "logical-sum-family-v1"
                if config.variable_sum_mode == "sum_n_v1"
                else "legacy-operation-label-v1"
            ),
            "variable_sum_mode": config.variable_sum_mode,
            "max_program_operands": config.max_program_operands,
            "sum_count_head_enabled": config.variable_sum_mode == "sum_n_v1",
            "sum_shared_membership_head_enabled": (
                config.variable_sum_mode == "sum_n_v1"
            ),
            "uses_financial_formula_ontology": False,
            "uses_item_id_shortcut": False,
            "reasoning_blocks": (
                "H/L recurrent permutation-equivariant self-attention + "
                "typed relation message passing + RMSNorm + SwiGLU"
                if config.financial_structure_mode == "typed_hierarchy_v1"
                else "H/L recurrent RoPE self-attention + RMSNorm + SwiGLU"
            ),
            "one_step_gradient_approximation": True,
            "deep_supervision_segments": config.max_segments,
            "training_objective_schema": (
                "v14-variable-sum-count-set-verifier-guided-v1"
                if (
                    config.variable_sum_mode == "sum_n_v1"
                    and verifier_guided_training
                )
                else "v14-variable-sum-count-set-v1"
                if config.variable_sum_mode == "sum_n_v1"
                else "verifier-guided-common-operation-pair-ranking-v1"
                if verifier_guided_pair_weight > 0
                or verifier_guided_unary_weight > 0
                else "late-weighted-deep-supervision-multi-operand-set-v1"
            ),
            "hierarchical_input_mode": config.hierarchical_input_mode,
            "bounded_relevance_prior": config.relevance_prior_max,
            "adaptive_halting": config.act_mode,
            "input_encoder_mode": config.input_encoder_mode,
            "semantic_model_name": config.semantic_model_name,
            "semantic_embedding_size": config.semantic_embedding_size,
            "semantic_encoder_backend": config.semantic_encoder_backend,
            "semantic_model_revision": config.semantic_model_revision,
            "semantic_local_files_only": config.semantic_local_files_only,
            "semantic_encoder_output_size": (
                config.semantic_encoder_size
            ),
            "semantic_word_segmentation": (
                config.semantic_word_segmentation
            ),
            "semantic_segmenter_version": (
                config.semantic_segmenter_version
            ),
            "semantic_prefix_mode": config.semantic_prefix_mode,
            "semantic_query_encoding_mode": (
                config.semantic_query_encoding_mode
            ),
            "semantic_text_template_version": (
                config.semantic_text_template_version
            ),
            "semantic_adapter_mode": config.semantic_adapter_mode,
            "semantic_frontend_fingerprint": (
                _semantic_encoder_spec_from_config(config).fingerprint()
            ),
            "semantic_encoder_frozen": True,
            "semantic_adapter_trainable": (
                config.semantic_adapter_mode
                == "layernorm_linear_l2_v1"
            ),
            "candidate_semantic_mode": config.candidate_semantic_mode,
            "financial_structure_mode": config.financial_structure_mode,
            "financial_hierarchy_layers": config.financial_hierarchy_layers,
            "financial_candidate_feature_size": FINANCIAL_CANDIDATE_FEATURE_SIZE,
            "financial_slot_feature_size": FINANCIAL_SLOT_FEATURE_SIZE,
            "financial_slot_feature_names": list(
                FINANCIAL_SLOT_FEATURES
            ),
            "program_search_strategy": config.program_search_strategy,
            "financial_same_physical_pair_mask": (
                config.financial_structure_mode == "typed_hierarchy_v1"
            ),
            "candidate_position_shortcut_enabled": (
                config.financial_structure_mode == "legacy"
            ),
            "pointer_conditioning_mode": config.pointer_conditioning_mode,
            "pair_backbone_mode": config.pair_backbone_mode,
            "pair_backbone_warmup_epochs": config.pair_backbone_warmup_epochs,
            "pair_loss_weight": config.pair_loss_weight,
            "pair_loss_warmup_epochs": config.pair_loss_warmup_epochs,
            "pair_targeted_negative_weight": config.pair_targeted_negative_weight,
            "initialized_from_checkpoint": (
                str(Path(initialize_from).expanduser().resolve())
                if initialize_from is not None
                else None
            ),
            "initialized_from_checkpoint_sha256": (
                initialized_from_sha256
            ),
            "initialization_mode": (
                initialization_mode
                if initialize_from is not None
                else "random"
            ),
            "initialization_manifest": initialization_manifest,
            "pair_head_only_epochs": pair_head_only_epochs,
            "pointer_head_only_epochs": pointer_head_only_epochs,
            "pointer_heads_only": pointer_heads_only,
            "retrieval_heads_only": retrieval_heads_only,
            "training_operation_loss_weight": (
                0.0
                if pointer_heads_only or retrieval_heads_only
                else 1.0
            ),
            "training_q_loss_weight": (
                0.0
                if pointer_heads_only or retrieval_heads_only
                else 0.20
            ),
            "training_pointer_loss_weight": 2.0,
            "training_pair_loss_weight": config.pair_loss_weight,
            "training_trainable_parameters": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "head_learning_rate": learning_rate,
            "pair_head_learning_rate": learning_rate,
            "backbone_learning_rate": (
                0.0
                if pointer_heads_only or retrieval_heads_only
                else backbone_learning_rate
                if backbone_learning_rate is not None
                else learning_rate
            ),
            "operation_head_frozen_during_training": (
                freeze_operation_head
                or pointer_heads_only
                or retrieval_heads_only
            ),
            "pair_head_frozen_during_training": freeze_pair_head,
            "select_pointer_residual_only": select_pointer_only,
            "gradient_clip_mode": (
                "global"
                if config.pair_backbone_mode == "normalized_end_to_end"
                else "separate_pair_and_base"
                if config.pointer_conditioning_mode == "joint_pair"
                else "global"
            ),
            "pointer_teacher_forcing": config.pointer_teacher_forcing,
            "pointer_hard_negative_weight": config.pointer_hard_negative_weight,
            "pointer_hard_negative_margin": config.pointer_hard_negative_margin,
            "pointer_targeted_negative_weight": pointer_targeted_negative_weight,
            "verifier_guided_pair_weight": verifier_guided_pair_weight,
            "verifier_guided_unary_weight": verifier_guided_unary_weight,
            "verifier_guided_margin": verifier_guided_margin,
            "verifier_guided_operations": sorted(
                VERIFIER_GUIDED_OPERATIONS
            ),
            "verifier_guided_unknown_as_negative": False,
            "verifier_guided_multi_operand_optimization": False,
            "verifier_guided_pair_candidate_cap": (
                MAX_VERIFIER_GUIDED_PAIR_CANDIDATES
            ),
            "model_selection_operations": (
                sorted(VERIFIER_GUIDED_OPERATIONS)
                if verifier_guided_training
                else list(OPERATIONS)
            ),
            "model_selection_objective": (
                "gold_operation_conditioned_candidate_recall_20_10_then_full_20_10"
                if retrieval_heads_only
                else "common_operation_execution_pointer_program_macro_f1"
                if verifier_guided_training
                else "execution_pointer_program_macro_f1"
            ),
            "add_oversample_factor": add_oversample_factor,
            "add_select_contrastive_weight": add_select_contrastive_weight,
            "add_select_contrastive_margin": add_select_contrastive_margin,
            "semantic_contrastive_weight": semantic_contrastive_weight,
            "semantic_contrastive_temperature": (
                semantic_contrastive_temperature
            ),
            "semantic_contrastive_margin": semantic_contrastive_margin,
            "semantic_contrastive_gold_policy": (
                "typed-slot-or-multi-positive-set-v1"
            ),
            "pointer_teacher_forcing_schedule": (
                "linear_to_zero"
                if config.pointer_conditioning_mode
                in {"operation_sequential", "later_sequential"}
                else "disabled"
            ),
            "joint_pair_relation_features": (
                PAIR_FEATURE_SIZE
                if config.pointer_conditioning_mode == "joint_pair"
                else 0
            ),
            "joint_pair_backbone_detached": (
                config.pointer_conditioning_mode == "joint_pair"
                and (
                    pointer_heads_only
                    or retrieval_heads_only
                    or config.pair_backbone_mode
                    in {"detached", "normalized_detached"}
                    or (
                        config.pair_backbone_mode == "warmup_end_to_end"
                        and config.pair_backbone_warmup_epochs >= epochs
                    )
                )
            ),
        },
        semantic_encoder=semantic_encoder,
    )
    v14_policy_mode = config.variable_sum_mode == "sum_n_v1"
    minimum_threshold: float | None = None
    if preserve_initial_threshold and not v14_policy_mode:
        assert initial_bundle is not None
        raw_threshold = initial_bundle.metadata.get("abstention_threshold")
        if not isinstance(raw_threshold, (int, float)):
            raise DocumentHRMError(
                "initialize_from has no numeric abstention_threshold"
            )
        minimum_threshold = float(raw_threshold)
    if v14_policy_mode:
        if pointer_heads_only:
            assert initial_bundle is not None
            raw_policies = initial_bundle.metadata.get(
                "abstention_policies_by_operation"
            )
            if not isinstance(raw_policies, dict):
                raise DocumentHRMError(
                    "V14 pointer-head-only training requires initial "
                    "abstention_policies_by_operation"
                )
            try:
                policies_by_operation = {
                    operation: CalibrationPolicy.from_dict(
                        raw_policies[operation]
                    )
                    for operation in _logical_evaluation_operations()
                }
            except (KeyError, CalibrationPolicyError, TypeError) as exc:
                raise DocumentHRMError(
                    "Initial V14 calibration policies are incomplete or invalid"
                ) from exc
        else:
            policies_by_operation = calibrate_operation_abstention_policies(
                bundle,
                validation_examples,
                target_precision=abstention_target_precision,
                batch_size=batch_size,
            )
        bundle.metadata.update(
            {
                "abstention_policies_by_operation": {
                    operation: policy.to_dict()
                    for operation, policy in policies_by_operation.items()
                },
                "abstention_calibration_mode": "v14-wilson-public-gate-v1",
            }
        )
        threshold = 0.0
        thresholds_by_operation: dict[str, float] = {}
    elif pointer_heads_only:
        assert minimum_threshold is not None
        threshold = minimum_threshold
        raw_thresholds = (
            initial_bundle.metadata.get(
                "abstention_thresholds_by_operation"
            )
            if initial_bundle is not None
            else None
        )
        thresholds_by_operation = (
            {
                str(operation): float(value)
                for operation, value in raw_thresholds.items()
                if operation in OPERATIONS
                and isinstance(value, (int, float))
                and 0.0 <= float(value) <= 1.0
            }
            if isinstance(raw_thresholds, dict)
            else {}
        )
    else:
        (
            thresholds_by_operation,
            accepted,
            precision,
            target_met,
        ) = calibrate_operation_abstention_thresholds(
            bundle,
            validation_examples,
            target_precision=abstention_target_precision,
            stability_errors=abstention_stability_errors,
            minimum_threshold=minimum_threshold,
            batch_size=batch_size,
        )
        threshold = max(
            thresholds_by_operation.values(),
            default=minimum_threshold if minimum_threshold is not None else 1.0,
        )
    if not v14_policy_mode:
        bundle.metadata.update(
            {
                "abstention_threshold": float(threshold),
                "abstention_thresholds_by_operation": thresholds_by_operation,
                "abstention_calibration_mode": "validation-operation-dp-v1",
            }
        )
    metrics = evaluate_document_hrm(
        bundle, validation_examples, batch_size=batch_size
    )
    accepted = round(metrics["accepted_count"])
    accepted_correct = round(metrics["accepted_execution_correct_count"])
    precision = metrics["accepted_execution_precision"]
    if v14_policy_mode:
        enabled_policies = tuple(
            policy
            for policy in policies_by_operation.values()
            if policy.status is CalibrationStatus.CALIBRATED
        )
        aggregate_policy_accepted = sum(
            policy.accepted for policy in enabled_policies
        )
        aggregate_policy_correct = sum(
            policy.correct for policy in enabled_policies
        )
        aggregate_lower_bound = wilson_lower_bound(
            aggregate_policy_correct,
            aggregate_policy_accepted,
        )
        target_met = bool(enabled_policies) and all(
            policy.lower_bound >= abstention_target_precision
            for policy in enabled_policies
        ) and aggregate_lower_bound >= abstention_target_precision
    else:
        target_met = accepted >= 5 and _precision_survives_errors(
            accepted_correct,
            accepted,
            target_precision=abstention_target_precision,
            stability_errors=abstention_stability_errors,
        )
    calibration_metadata: dict[str, object] = {
            "validation_operation_accuracy": round(metrics["operation_accuracy"], 6),
            "validation_program_exact_accuracy": round(
                metrics["program_exact_accuracy"], 6
            ),
            "validation_execution_accuracy": round(metrics["execution_accuracy"], 6),
            "validation_pointer_pair_accuracy": round(metrics["pointer_pair_accuracy"], 6),
            "validation_pointer_slot_accuracy": [
                round(metrics[f"pointer_slot_{slot + 1}_accuracy"], 6)
                for slot in range(config.max_program_operands)
            ],
            "validation_pointer_slot_support": [
                int(metrics[f"pointer_slot_{slot + 1}_support"])
                for slot in range(config.max_program_operands)
            ],
            "validation_pointer_slot_accuracy_defined": [
                bool(metrics[f"pointer_slot_{slot + 1}_accuracy_defined"])
                for slot in range(config.max_program_operands)
            ],
            "validation_proof_eligible_coverage": round(
                metrics["proof_eligible_coverage"], 6
            ),
            "validation_proof_eligible_precision_defined": bool(
                metrics["proof_eligible_precision_defined"]
            ),
            "abstention_validation_precision_defined": bool(
                metrics["accepted_execution_precision_defined"]
            ),
            "abstention_target_precision": abstention_target_precision,
            "abstention_validation_accepted": accepted,
            "abstention_validation_precision": round(precision, 6),
            "abstention_target_met": target_met,
            "abstention_stability_errors": abstention_stability_errors,
            "initial_threshold_preserved": preserve_initial_threshold,
            "initial_threshold_frozen": pointer_heads_only,
            "minimum_abstention_threshold": minimum_threshold,
    }
    if v14_policy_mode:
        compatible_thresholds = {
            operation: policy.threshold
            for operation, policy in policies_by_operation.items()
            if policy.status is CalibrationStatus.CALIBRATED
            and policy.threshold is not None
        }
        calibration_metadata.update(
            {
                "abstention_policies_by_operation": {
                    operation: policy.to_dict()
                    for operation, policy in policies_by_operation.items()
                },
                "abstention_calibration_mode": "v14-wilson-public-gate-v1",
                "abstention_policy_statuses_by_operation": {
                    operation: policy.status.value
                    for operation, policy in policies_by_operation.items()
                },
                "abstention_policy_support_by_operation": {
                    operation: policy.support
                    for operation, policy in policies_by_operation.items()
                },
                "abstention_policy_enabled_count": len(enabled_policies),
                "abstention_policy_aggregate_accepted": (
                    aggregate_policy_accepted
                ),
                "abstention_policy_aggregate_correct": (
                    aggregate_policy_correct
                ),
                "abstention_policy_aggregate_lower_bound": (
                    aggregate_lower_bound
                ),
                # Compatibility-only projection. V14 runtime/evaluation reads
                # the typed policies above and never these numeric thresholds.
                "abstention_thresholds_by_operation": compatible_thresholds,
            }
        )
        if compatible_thresholds:
            calibration_metadata["abstention_threshold"] = max(
                compatible_thresholds.values()
            )
    else:
        # This threshold is one of the observed confidence values. Rounding it
        # can move the boundary above that sample after save + reload.
        calibration_metadata.update(
            {
                "abstention_threshold": float(threshold),
                "abstention_thresholds_by_operation": thresholds_by_operation,
                "abstention_calibration_mode": "validation-operation-dp-v1",
            }
        )
    bundle.metadata.update(calibration_metadata)
    return bundle, DocumentHRMTrainingReport(
        train_examples=len(train_examples),
        validation_examples=len(validation_examples),
        epochs=epochs,
        final_loss=final_loss,
        operation_accuracy=metrics["operation_accuracy"],
        operation_macro_precision=metrics["operation_macro_precision"],
        operation_macro_recall=metrics["operation_macro_recall"],
        operation_macro_f1=metrics["operation_macro_f1"],
        operation_weighted_precision=metrics["operation_weighted_precision"],
        operation_weighted_recall=metrics["operation_weighted_recall"],
        operation_weighted_f1=metrics["operation_weighted_f1"],
        program_exact_accuracy=metrics["program_exact_accuracy"],
        execution_accuracy=metrics["execution_accuracy"],
        pointer_pair_accuracy=metrics["pointer_pair_accuracy"],
        pointer_slot_accuracy=tuple(
            metrics[f"pointer_slot_{slot + 1}_accuracy"]
            for slot in range(config.max_program_operands)
        ),
        accepted_coverage=metrics["accepted_coverage"],
        accepted_precision=metrics["accepted_precision"],
        operation_counts=dict(counts),
        pointer_slot_support=tuple(
            int(metrics[f"pointer_slot_{slot + 1}_support"])
            for slot in range(config.max_program_operands)
        ),
        accepted_precision_defined=bool(
            metrics["accepted_precision_defined"]
        ),
    )


def _bundle_operand_width(bundle: DocumentHRMBundle) -> int:
    """Read dynamic V14 width while tolerating legacy/mock bundles."""

    raw = getattr(
        getattr(bundle.model, "config", None),
        "max_program_operands",
        MAX_OPERANDS,
    )
    if type(raw) is not int or not MAX_OPERANDS <= raw <= SUM_N_MAX_OPERANDS:
        return MAX_OPERANDS
    return raw


def _physical_program_matches(
    candidate_program: ReasoningProgram,
    gold_program: ReasoningProgram,
    candidates: Sequence[NumericCandidate],
) -> bool:
    candidate_operation = _logical_operation_name(
        candidate_program.operation
    )
    gold_operation = _logical_operation_name(gold_program.operation)
    if (
        candidate_operation != gold_operation
        or len(candidate_program.operands) != len(gold_program.operands)
    ):
        return False
    if any(
        index < 0 or index >= len(candidates)
        for index in (
            *candidate_program.operands,
            *gold_program.operands,
        )
    ):
        return False
    conflicts = _conflicting_physical_indices(candidates)

    def evaluation_key(index: int) -> tuple[object, ...]:
        return (
            ("conflict_pointer", index)
            if index in conflicts
            else _beam_physical_key(index, candidates)
        )

    candidate_keys = [evaluation_key(index) for index in candidate_program.operands]
    gold_keys = [evaluation_key(index) for index in gold_program.operands]
    if (
        candidate_program.operation in COMMUTATIVE
        or candidate_program.operation == SUM_N_OPERATION
        or gold_program.operation in COMMUTATIVE
        or gold_program.operation == SUM_N_OPERATION
    ):
        return sorted(candidate_keys, key=repr) == sorted(
            gold_keys, key=repr
        )
    return candidate_keys == gold_keys


def _physical_operand_set_matches(
    candidate_program: ReasoningProgram,
    gold_program: ReasoningProgram,
    candidates: Sequence[NumericCandidate],
) -> bool:
    if len(candidate_program.operands) != len(gold_program.operands):
        return False
    if any(
        index < 0 or index >= len(candidates)
        for index in (*candidate_program.operands, *gold_program.operands)
    ):
        return False
    conflicts = _conflicting_physical_indices(candidates)

    def evaluation_key(index: int) -> tuple[object, ...]:
        return (
            ("conflict_pointer", index)
            if index in conflicts
            else _beam_physical_key(index, candidates)
        )

    return sorted(
        (evaluation_key(index) for index in candidate_program.operands),
        key=repr,
    ) == sorted(
        (evaluation_key(index) for index in gold_program.operands),
        key=repr,
    )


def _gold_operand_slot_pool_match_count(
    example: ProgramExample,
    pools: Sequence[Sequence[int]],
) -> int:
    gold = _supervision_gold_operand_indices(example)
    if not gold or not pools:
        return 0
    if any(index < 0 or index >= len(example.candidates) for index in gold):
        return 0
    conflicts = _conflicting_physical_indices(example.candidates)

    def evaluation_key(index: int) -> tuple[object, ...]:
        return (
            ("conflict_pointer", index)
            if index in conflicts
            else _beam_physical_key(index, example.candidates)
        )

    gold_keys = [evaluation_key(index) for index in gold]
    if len(set(gold_keys)) != len(gold_keys):
        return 0
    commutative = (
        example.program.operation in COMMUTATIVE
        or example.program.operation == SUM_N_OPERATION
    )
    pool_keys = [
        {
            evaluation_key(index)
            for index in pool
            if 0 <= index < len(example.candidates)
        }
        for pool in pools[: len(gold)]
    ]
    pool_keys.extend(set() for _ in range(len(gold) - len(pool_keys)))
    if not commutative:
        return sum(
            gold_key in pool_keys[slot]
            for slot, gold_key in enumerate(gold_keys)
        )

    adjacency = [
        [slot for slot, keys in enumerate(pool_keys) if gold_key in keys]
        for gold_key in gold_keys
    ]
    slot_to_gold: dict[int, int] = {}

    def augment(gold_slot: int, seen_slots: set[int]) -> bool:
        for pool_slot in adjacency[gold_slot]:
            if pool_slot in seen_slots:
                continue
            seen_slots.add(pool_slot)
            assigned_gold = slot_to_gold.get(pool_slot)
            if assigned_gold is None or augment(assigned_gold, seen_slots):
                slot_to_gold[pool_slot] = gold_slot
                return True
        return False

    return sum(
        augment(gold_slot, set()) for gold_slot in range(len(gold_keys))
    )


def _gold_operands_in_slot_pools(
    example: ProgramExample,
    pools: Sequence[Sequence[int]],
) -> bool:
    gold = _supervision_gold_operand_indices(example)
    return bool(gold) and _gold_operand_slot_pool_match_count(
        example, pools
    ) == len(gold)


def evaluate_document_hrm_candidate_retrieval(
    bundle: DocumentHRMBundle,
    examples_or_path: Sequence[ProgramExample] | str | Path,
    *,
    batch_size: int = 32,
) -> dict[str, float]:
    """Evaluate operation-conditioned slot pools without composing programs.

    This development-only metric is used to select retrieval-head epochs. It
    never changes inference and full symbolic evaluation still runs afterward.
    """

    examples = (
        load_program_examples(examples_or_path)
        if isinstance(examples_or_path, (str, Path))
        else list(examples_or_path)
    )
    matched = {10: 0, 20: 0}
    full = {10: 0, 20: 0}
    gold_operand_total = 0
    bundle.model.eval()
    for start in range(0, len(examples), batch_size):
        current = examples[start : start + batch_size]
        output = bundle._predict_many_output(
            current, bundle.model.config.max_segments
        )
        for row, example in enumerate(current):
            gold = _supervision_gold_operand_indices(example)
            arity = len(gold)
            gold_operand_total += arity
            operation = _logical_operation_name(example.program.operation)
            generated = _rank_slot_candidate_pools(
                output,
                row,
                operation,
                example.question,
                example.candidates,
                arity=arity,
            )
            for cutoff, pools in (
                (10, generated[0]),
                (20, generated[1]),
            ):
                count = _gold_operand_slot_pool_match_count(example, pools)
                matched[cutoff] += count
                full[cutoff] += int(arity > 0 and count == arity)
    example_total = len(examples)
    return {
        "gold_operation_conditioned_candidate_recall_at_10": (
            matched[10] / gold_operand_total if gold_operand_total else 0.0
        ),
        "gold_operation_conditioned_candidate_recall_at_20": (
            matched[20] / gold_operand_total if gold_operand_total else 0.0
        ),
        "gold_operation_conditioned_full_coverage_at_10": (
            full[10] / example_total if example_total else 0.0
        ),
        "gold_operation_conditioned_full_coverage_at_20": (
            full[20] / example_total if example_total else 0.0
        ),
        "gold_operation_conditioned_operand_count": float(gold_operand_total),
        "example_count": float(example_total),
    }


def evaluate_document_hrm(
    bundle: DocumentHRMBundle,
    examples_or_path: Sequence[ProgramExample] | str | Path,
    *,
    batch_size: int = 32,
    adaptive_inference: bool = False,
) -> dict[str, float]:
    examples = (
        load_program_examples(examples_or_path)
        if isinstance(examples_or_path, (str, Path))
        else list(examples_or_path)
    )
    logical_sum_evaluation = _uses_logical_sum_evaluation(bundle)
    # V14 acceptance stays fail-closed before typed policies are calibrated;
    # metric labels are nevertheless stable on every in-memory epoch bundle.
    typed_policy_evaluation = logical_sum_evaluation or isinstance(
        bundle.metadata.get("abstention_policies_by_operation"), dict
    )
    evaluation_operations = (
        _logical_evaluation_operations()
        if logical_sum_evaluation
        else OPERATIONS
    )
    operand_width = _bundle_operand_width(bundle)
    empty = {
        "operation_accuracy": 0.0,
        "operation_macro_precision": 0.0,
        "operation_macro_recall": 0.0,
        "operation_macro_f1": 0.0,
        "operation_weighted_precision": 0.0,
        "operation_weighted_recall": 0.0,
        "operation_weighted_f1": 0.0,
        "program_exact_accuracy": 0.0,
        "raw_program_exact_accuracy": 0.0,
        "grounded_semantic_program_accuracy": 0.0,
        "execution_accuracy": 0.0,
        "pointer_pair_accuracy": 0.0,
        "raw_pointer_pair_accuracy": 0.0,
        "accepted_coverage": 0.0,
        "accepted_precision": 0.0,
        "accepted_execution_precision": 0.0,
        "accepted_program_precision": 0.0,
        "accepted_grounded_semantic_program_precision": 0.0,
        "accepted_precision_defined": False,
        "accepted_execution_precision_defined": False,
        "accepted_program_precision_defined": False,
        "accepted_grounded_semantic_program_precision_defined": False,
        "proof_eligible_coverage": 0.0,
        "proof_eligible_precision": 0.0,
        "proof_eligible_execution_precision": 0.0,
        "proof_eligible_program_precision": 0.0,
        "proof_eligible_grounded_semantic_program_precision": 0.0,
        "proof_eligible_precision_defined": False,
        "proof_eligible_execution_precision_defined": False,
        "proof_eligible_program_precision_defined": False,
        "proof_eligible_grounded_semantic_program_precision_defined": False,
        "mean_segments": 0.0,
        "program_search_change_rate": 0.0,
        "mean_program_candidates_considered": 0.0,
        "mean_symbolic_program_rejections": 0.0,
        "mean_symbolic_unknown_constraints": 0.0,
        "gold_program_recall_at_5": 0.0,
        "gold_program_recall_at_10": 0.0,
        "gold_physical_program_recall_at_5": 0.0,
        "gold_physical_program_recall_at_10": 0.0,
        "gold_operand_recall_at_5": 0.0,
        "gold_operand_recall_at_10": 0.0,
        "operand_candidate_recall_at_10": 0.0,
        "operand_candidate_recall_at_20": 0.0,
        "operand_candidate_full_coverage_at_10": 0.0,
        "operand_candidate_full_coverage_at_20": 0.0,
        "operand_candidate_recall_at_10_given_gold_operation": 0.0,
        "operand_candidate_recall_at_20_given_gold_operation": 0.0,
        "operand_candidate_gold_operand_count": 0.0,
        "operand_candidate_gold_operation_operand_count": 0.0,
        "diverse_beam_diagnostic_examples": 0.0,
        "gold_operation_not_considered_count": 0.0,
        "gold_operand_outside_slot_pool_at_20_count": 0.0,
        "gold_program_not_generated_count": 0.0,
        "gold_program_below_preverifier_top10_count": 0.0,
        "gold_top10_verifier_rejected_count": 0.0,
        "gold_top10_verifier_invalid_count": 0.0,
        "gold_top10_runtime_gate_rejected_count": 0.0,
        "gold_top10_eligible_count": 0.0,
        "gold_pipeline_diagnostic_unavailable_count": 0.0,
        "program_search_failure_bucket_total": 0.0,
        "gold_operation_not_considered_rate": 0.0,
        "gold_operand_outside_slot_pool_at_20_rate": 0.0,
        "gold_program_not_generated_rate": 0.0,
        "gold_program_below_preverifier_top10_rate": 0.0,
        "gold_top10_verifier_rejected_rate": 0.0,
        "gold_top10_runtime_gate_rejected_rate": 0.0,
        "gold_top10_eligible_rate": 0.0,
        "gold_pipeline_diagnostic_unavailable_rate": 0.0,
        "mean_generated_programs_before_physical_dedup": 0.0,
        "mean_generated_programs_after_physical_dedup": 0.0,
        "gold_symbolic_program_recall_at_5": 0.0,
        "gold_symbolic_program_recall_at_10": 0.0,
        "gold_program_generated_recall": 0.0,
        "gold_eligible_program_recall": 0.0,
        "typed_synthesis_gold_generation_rate": 0.0,
        "typed_synthesis_example_rate": 0.0,
        "mean_synthesized_program_candidates": 0.0,
        "program_search_trace_coverage": 0.0,
        "single_eligible_program_rate": 0.0,
        "verifier_gold_generated_count": 0.0,
        "verifier_gold_false_rejection_count": 0.0,
        "verifier_gold_false_rejection_rate": 0.0,
        "search_pipeline_gold_rejection_rate": 0.0,
        "example_count": 0.0,
        "program_exact_correct_count": 0.0,
        "raw_program_exact_correct_count": 0.0,
        "grounded_semantic_program_correct_count": 0.0,
        "execution_correct_count": 0.0,
        "accepted_count": 0.0,
        "accepted_execution_correct_count": 0.0,
        "accepted_program_correct_count": 0.0,
        "accepted_grounded_semantic_program_correct_count": 0.0,
        "proof_eligible_count": 0.0,
        "proof_eligible_execution_correct_count": 0.0,
        "proof_eligible_program_correct_count": 0.0,
        "proof_eligible_grounded_semantic_program_correct_count": 0.0,
    }
    empty.update(
        {
            f"pointer_slot_{slot + 1}_accuracy": 0.0
            for slot in range(operand_width)
        }
    )
    for slot in range(operand_width):
        empty[f"pointer_slot_{slot + 1}_support"] = 0.0
        empty[f"pointer_slot_{slot + 1}_accuracy_defined"] = False

    for arity in range(2, SUM_N_MAX_OPERANDS + 1):
        empty[f"exact_physical_set_arity_{arity}_accuracy"] = 0.0
        empty[f"exact_physical_set_arity_{arity}_support"] = 0.0
        empty[f"exact_physical_program_arity_{arity}_accuracy"] = 0.0
        empty[f"exact_physical_program_arity_{arity}_support"] = 0.0
    for operation in evaluation_operations:
        prefix = f"operation_class_{operation.lower()}"
        empty.update(
            {
                f"{prefix}_precision": 0.0,
                f"{prefix}_recall": 0.0,
                f"{prefix}_f1": 0.0,
                f"{prefix}_support": 0.0,
            }
        )
        quality_prefix = f"program_operation_{operation.lower()}"
        empty.update(
            {
                f"{quality_prefix}_support": 0.0,
                f"{quality_prefix}_operation_accuracy": 0.0,
                f"{quality_prefix}_pointer_pair_accuracy": 0.0,
                f"{quality_prefix}_raw_pointer_pair_accuracy": 0.0,
                f"{quality_prefix}_program_exact_accuracy": 0.0,
                f"{quality_prefix}_raw_program_exact_accuracy": 0.0,
                f"{quality_prefix}_grounded_semantic_program_accuracy": 0.0,
                f"{quality_prefix}_execution_accuracy": 0.0,
                f"{quality_prefix}_accepted_coverage": 0.0,
                f"{quality_prefix}_accepted_execution_precision": 0.0,
                f"{quality_prefix}_accepted_program_precision": 0.0,
                f"{quality_prefix}_accepted_grounded_semantic_program_precision": 0.0,
                f"{quality_prefix}_accepted_count": 0.0,
                f"{quality_prefix}_accepted_execution_correct_count": 0.0,
                f"{quality_prefix}_accepted_program_correct_count": 0.0,
                f"{quality_prefix}_accepted_precision_defined": False,
                f"{quality_prefix}_accepted_execution_precision_defined": False,
                f"{quality_prefix}_accepted_program_precision_defined": False,
                f"{quality_prefix}_proof_eligible_coverage": 0.0,
                f"{quality_prefix}_proof_eligible_execution_precision": 0.0,
                f"{quality_prefix}_proof_eligible_program_precision": 0.0,
                f"{quality_prefix}_proof_eligible_precision_defined": False,
                f"{quality_prefix}_proof_eligible_count": 0.0,
                f"{quality_prefix}_proof_eligible_execution_correct_count": 0.0,
                f"{quality_prefix}_proof_eligible_program_correct_count": 0.0,
            }
        )
    if not examples:
        return empty

    bundle.model.eval()
    operation_correct = program_correct = execution_correct = total = 0
    raw_program_correct = grounded_semantic_program_correct = 0
    pointer_pair_correct = raw_pointer_pair_correct = 0
    pointer_correct = [0] * operand_width
    pointer_total = [0] * operand_width
    accepted = accepted_execution_correct = accepted_program_correct = segment_total = 0
    accepted_grounded_semantic_program_correct = 0
    proof_eligible = 0
    proof_eligible_execution_correct = 0
    proof_eligible_program_correct = 0
    proof_eligible_grounded_semantic_program_correct = 0
    program_search_changes = 0
    program_candidates_total = 0
    symbolic_rejections_total = 0
    symbolic_unknown_total = 0
    program_recall = {5: 0, 10: 0}
    physical_program_recall = {5: 0, 10: 0}
    operand_recall = {5: 0, 10: 0}
    symbolic_program_recall = {5: 0, 10: 0}
    operand_candidate_matches = {10: 0, 20: 0}
    operand_candidate_full_coverage = {10: 0, 20: 0}
    operand_candidate_matches_given_gold_operation = {10: 0, 20: 0}
    operand_candidate_gold_operands = 0
    operand_candidate_gold_operation_operands = 0
    diverse_beam_diagnostic_examples = 0
    operation_misses = pool_misses = generation_misses = ranking_misses = 0
    verifier_invalid = runtime_gate_rejections = eligible_successes = 0
    diagnostic_unavailable = 0
    generated_before_total = generated_after_total = 0
    physical_program_by_arity: dict[int, Counter[str]] = {
        arity: Counter() for arity in range(2, SUM_N_MAX_OPERANDS + 1)
    }
    sum_physical_set_by_arity: dict[int, Counter[str]] = {
        arity: Counter() for arity in range(2, SUM_N_MAX_OPERANDS + 1)
    }
    verifier_gold_generated = 0
    eligible_gold_generated = 0
    typed_gold_generated = 0
    typed_synthesis_examples = 0
    synthesized_candidates_total = 0
    search_trace_examples = 0
    single_eligible_programs = 0
    verifier_gold_false_rejections = 0
    pipeline_gold_rejections = 0
    pipeline_gold_rejection_reasons: Counter[str] = Counter()
    operation_quality: dict[str, Counter[str]] = {
        operation: Counter() for operation in evaluation_operations
    }
    expected_operations: list[str] = []
    predicted_operations: list[str] = []
    default_threshold = _metadata_float(
        bundle.metadata, "abstention_threshold", 0.75
    )
    diverse_beam_metrics_enabled = (
        getattr(bundle.model.config, "program_search_strategy", "legacy")
        == "diverse_set_beam_v1"
    )

    for start in range(0, len(examples), batch_size):
        current = examples[start : start + batch_size]
        predictions = (
            [
                bundle.predict(example.question, example.candidates)
                for example in current
            ]
            if adaptive_inference
            else bundle.predict_many(current)
        )
        for example, prediction in zip(current, predictions):
            grounding_constraints = getattr(
                prediction, "grounding_constraints_applied", ()
            )
            trace = getattr(prediction, "program_search_trace", ())
            synthesized_program_candidates = int(
                getattr(prediction, "synthesized_program_candidates", 0)
            )
            legacy_pools_at_10 = tuple(
                getattr(prediction, "slot_candidate_pools_at_10", ())
            )
            legacy_pools_at_20 = tuple(
                getattr(prediction, "slot_candidate_pools_at_20", ())
            )
            operation_pools_at_10 = {
                _logical_operation_name(operation): tuple(pools)
                for operation, pools in getattr(
                    prediction,
                    "operation_slot_candidate_pools_at_10",
                    (),
                )
            }
            operation_pools_at_20 = {
                _logical_operation_name(operation): tuple(pools)
                for operation, pools in getattr(
                    prediction,
                    "operation_slot_candidate_pools_at_20",
                    (),
                )
            }
            operation_beam = tuple(
                _logical_operation_name(operation)
                for operation in getattr(prediction, "operation_beam", ())
            )
            preverification = tuple(
                getattr(prediction, "preverification_programs", ())
            )
            verified_beam = tuple(
                getattr(prediction, "selected_for_verification", ())
            )
            if diverse_beam_metrics_enabled:
                diverse_beam_diagnostic_examples += 1
                gold_operation = _logical_operation_name(
                    example.program.operation
                )
                telemetry_available = bool(
                    operation_beam
                    or operation_pools_at_10
                    or operation_pools_at_20
                    or preverification
                )
                gold_operation_considered = gold_operation in operation_beam
                if operation_pools_at_10 or operation_pools_at_20:
                    pools_at_10 = operation_pools_at_10.get(
                        gold_operation, ()
                    )
                    pools_at_20 = operation_pools_at_20.get(
                        gold_operation, ()
                    )
                else:
                    # Compatibility for predictions created before the
                    # operation-conditioned telemetry fields existed.
                    pools_at_10 = legacy_pools_at_10
                    pools_at_20 = legacy_pools_at_20
                    gold_operation_considered = telemetry_available
                gold_operand_count = len(
                    _supervision_gold_operand_indices(example)
                )
                operand_candidate_gold_operands += gold_operand_count
                match_counts = {
                    10: _gold_operand_slot_pool_match_count(
                        example, pools_at_10
                    ),
                    20: _gold_operand_slot_pool_match_count(
                        example, pools_at_20
                    ),
                }
                for cutoff in (10, 20):
                    operand_candidate_matches[cutoff] += match_counts[cutoff]
                    operand_candidate_full_coverage[cutoff] += int(
                        gold_operand_count > 0
                        and match_counts[cutoff] == gold_operand_count
                    )
                if gold_operation_considered:
                    operand_candidate_gold_operation_operands += (
                        gold_operand_count
                    )
                    for cutoff in (10, 20):
                        operand_candidate_matches_given_gold_operation[
                            cutoff
                        ] += match_counts[cutoff]
                generated_before_total += int(
                    getattr(
                        prediction,
                        "generated_programs_before_physical_dedup",
                        0,
                    )
                )
                generated_after_total += int(
                    getattr(
                        prediction,
                        "generated_programs_after_physical_dedup",
                        0,
                    )
                )
                gold_preverification_ranks = [
                    rank
                    for rank, program in enumerate(
                        preverification, start=1
                    )
                    if _physical_program_matches(
                        program,
                        example.program,
                        example.candidates,
                    )
                ]
                gold_selected = any(
                    _physical_program_matches(
                        program, example.program, example.candidates
                    )
                    for program in verified_beam
                )
                gold_selected_trace = [
                    item
                    for item in trace
                    if _physical_program_matches(
                        item.program,
                        example.program,
                        example.candidates,
                    )
                ]
                if not telemetry_available:
                    diagnostic_unavailable += 1
                elif not gold_operation_considered:
                    operation_misses += 1
                elif match_counts[20] != gold_operand_count:
                    pool_misses += 1
                elif not gold_preverification_ranks:
                    generation_misses += 1
                elif not gold_selected:
                    ranking_misses += 1
                elif not gold_selected_trace:
                    diagnostic_unavailable += 1
                elif any(item.eligible for item in gold_selected_trace):
                    eligible_successes += 1
                elif any(
                    item.verification_state == "INVALID"
                    for item in gold_selected_trace
                ):
                    verifier_invalid += 1
                else:
                    runtime_gate_rejections += 1
            total += 1
            expected_operations.append(
                _logical_operation_name(example.program.operation)
                if logical_sum_evaluation
                else example.program.operation
            )
            predicted_operations.append(
                _logical_operation_name(prediction.program.operation)
                if logical_sum_evaluation
                else prediction.program.operation
            )
            segment_total += int(getattr(prediction, "segments", 0))
            program_candidates_total += int(
                getattr(prediction, "program_candidates_considered", 0)
            )
            symbolic_rejections_total += int(
                getattr(prediction, "symbolic_program_rejections", 0)
            )
            symbolic_unknown_total += int(
                getattr(prediction, "symbolic_unknown_constraints", 0)
            )
            program_search_changes += int(
                "symbolic_program_search" in grounding_constraints
            )
            operation_matches = (
                _arity_normalized_operation_equal(
                    prediction.program, example.program
                )
                if logical_sum_evaluation
                else prediction.program.operation == example.program.operation
            )
            raw_program_matches = prediction.program == example.program
            program_matches = (
                canonical_sum_family_equal(
                    prediction.program, example.program
                )
                if logical_sum_evaluation
                else canonical_program_equal(
                    prediction.program, example.program
                )
            )
            raw_pointer_pair_matches = tuple(
                prediction.program.operands
            ) == tuple(example.program.operands)
            pointer_pair_matches = canonical_pointer_equal(
                example.program.operation,
                prediction.program.operands,
                example.program.operands,
            )
            execution_matches = _execution_is_correct(prediction.program, example)
            arity = len(example.program.operands)
            if arity in physical_program_by_arity:
                physical_program_by_arity[arity]["support"] += 1
                physical_program_by_arity[arity]["correct"] += int(
                    _physical_program_matches(
                        prediction.program,
                        example.program,
                        example.candidates,
                    )
                )
                if _logical_operation_name(
                    example.program.operation
                ) == SUM_N_OPERATION:
                    sum_physical_set_by_arity[arity]["support"] += 1
                    sum_physical_set_by_arity[arity]["correct"] += int(
                        _physical_operand_set_matches(
                            prediction.program,
                            example.program,
                            example.candidates,
                        )
                    )
            grounded_semantic_program_matches = program_matches
            if (
                not grounded_semantic_program_matches
                and execution_matches
                and operation_matches
            ):
                graph = build_evidence_graph(example.candidates)
                plan = compile_question_plan(example.question)
                grounded_semantic_program_matches = (
                    semantic_program_signature(
                        prediction.program, graph.cells, plan
                    )
                    == semantic_program_signature(
                        example.program, graph.cells, plan
                    )
                )
            synthesized_candidates_total += synthesized_program_candidates
            typed_synthesis_examples += int(
                synthesized_program_candidates > 0
            )
            search_trace_examples += int(bool(trace))
            eligible_trace = [item for item in trace if item.eligible]
            single_eligible_programs += int(len(eligible_trace) == 1)
            if trace:
                ranked_programs = [
                    item.program
                    for item in sorted(trace, key=lambda item: item.neural_rank)
                ]
                symbolic_ranked_programs = [
                    item.program
                    for item in sorted(
                        eligible_trace,
                        key=lambda item: item.legacy_rank,
                    )
                ]
            else:
                ranked_programs = [prediction.program]
                symbolic_ranked_programs = [prediction.program]
            physical_ranked_programs = (
                list(preverification) if preverification else ranked_programs
            )
            for cutoff in (5, 10):
                top_programs = ranked_programs[:cutoff]
                program_recall[cutoff] += int(
                    any(
                        (
                            canonical_sum_family_equal(
                                candidate, example.program
                            )
                            if logical_sum_evaluation
                            else canonical_program_equal(
                                candidate, example.program
                            )
                        )
                        for candidate in top_programs
                    )
                )
                physical_program_recall[cutoff] += int(
                    any(
                        _physical_program_matches(
                            candidate,
                            example.program,
                            example.candidates,
                        )
                        for candidate in physical_ranked_programs[:cutoff]
                    )
                )
                covered_operands = {
                    operand
                    for candidate_program in top_programs
                    for operand in candidate_program.operands
                }
                operand_recall[cutoff] += int(
                    set(example.program.operands).issubset(covered_operands)
                )
                symbolic_program_recall[cutoff] += int(
                    any(
                        (
                            canonical_sum_family_equal(
                                candidate, example.program
                            )
                            if logical_sum_evaluation
                            else canonical_program_equal(
                                candidate, example.program
                            )
                        )
                        for candidate in symbolic_ranked_programs[:cutoff]
                    )
                )
            gold_trace = next(
                (
                    item
                    for item in trace
                    if (
                        canonical_sum_family_equal(
                            item.program, example.program
                        )
                        if logical_sum_evaluation
                        else canonical_program_equal(
                            item.program, example.program
                        )
                    )
                ),
                None,
            )
            if gold_trace is not None:
                verifier_gold_generated += 1
                eligible_gold_generated += int(gold_trace.eligible)
                typed_gold_generated += int(
                    "typed_slot_synthesis" in gold_trace.generation_sources
                )
                verifier_gold_false_rejections += int(
                    gold_trace.verification_state == "INVALID"
                )
                pipeline_gold_rejections += int(not gold_trace.eligible)
                if not gold_trace.eligible:
                    pipeline_gold_rejection_reasons.update(
                        gold_trace.rejection_reasons
                    )
            quality = operation_quality[
                _logical_operation_name(example.program.operation)
                if logical_sum_evaluation
                else example.program.operation
            ]
            quality["support"] += 1
            quality["operation_correct"] += int(operation_matches)
            if operation_matches:
                operation_correct += 1
            if program_matches:
                program_correct += 1
                quality["program_correct"] += 1
            if raw_program_matches:
                raw_program_correct += 1
                quality["raw_program_correct"] += 1
            if grounded_semantic_program_matches:
                grounded_semantic_program_correct += 1
                quality["grounded_semantic_program_correct"] += 1
            if execution_matches:
                execution_correct += 1
                quality["execution_correct"] += 1
            if pointer_pair_matches:
                pointer_pair_correct += 1
                quality["pointer_pair_correct"] += 1
            if raw_pointer_pair_matches:
                raw_pointer_pair_correct += 1
                quality["raw_pointer_pair_correct"] += 1
            for slot, target in enumerate(example.program.operands):
                pointer_total[slot] += 1
                predicted = (
                    prediction.program.operands[slot]
                    if slot < len(prediction.program.operands)
                    else None
                )
                if predicted == target:
                    pointer_correct[slot] += 1
            grounding_rejections = _grounding_gate_reasons(
                example.question,
                prediction.program,
                example.candidates,
                financial_identity_catalog_enabled=(
                    "financial_identity_catalog" in grounding_constraints
                ),
            )
            threshold = _abstention_threshold_for_prediction(
                bundle, prediction, default_threshold
            )
            proof_eligible_by_runtime = (
                _prediction_public_source_action(
                    example.question,
                    prediction,
                    example.candidates,
                )
                == "ANSWER"
            )
            if proof_eligible_by_runtime:
                proof_eligible += 1
                proof_eligible_execution_correct += int(execution_matches)
                proof_eligible_program_correct += int(program_matches)
                proof_eligible_grounded_semantic_program_correct += int(
                    grounded_semantic_program_matches
                )
                quality["proof_eligible"] += 1
                quality["proof_eligible_execution_correct"] += int(execution_matches)
                quality["proof_eligible_program_correct"] += int(program_matches)
                quality["proof_eligible_grounded_semantic_program_correct"] += int(
                    grounded_semantic_program_matches
                )

            accepted_by_policy = (
                _prediction_action_decision(
                    bundle, example.question, prediction, example.candidates
                ).policy_accepted
                if typed_policy_evaluation
                else prediction.confidence >= threshold
                and not grounding_rejections
            )
            if accepted_by_policy:
                accepted += 1
                accepted_execution_correct += int(execution_matches)
                accepted_program_correct += int(program_matches)
                accepted_grounded_semantic_program_correct += int(
                    grounded_semantic_program_matches
                )
                quality["accepted"] += 1
                quality["accepted_execution_correct"] += int(execution_matches)
                quality["accepted_program_correct"] += int(program_matches)
                quality["accepted_grounded_semantic_program_correct"] += int(
                    grounded_semantic_program_matches
                )

    failure_bucket_total = (
        operation_misses
        + pool_misses
        + generation_misses
        + ranking_misses
        + verifier_invalid
        + runtime_gate_rejections
        + eligible_successes
        + diagnostic_unavailable
    )
    if diverse_beam_metrics_enabled and (
        diverse_beam_diagnostic_examples != total
        or failure_bucket_total != total
    ):
        raise DocumentHRMError(
            "V14.1 search diagnostics are not exhaustive: "
            f"examples={total}, diagnostic_examples="
            f"{diverse_beam_diagnostic_examples}, buckets="
            f"{failure_bucket_total}"
        )

    metrics = {
        "operation_accuracy": operation_correct / total,
        "program_exact_accuracy": program_correct / total,
        "raw_program_exact_accuracy": raw_program_correct / total,
        "grounded_semantic_program_accuracy": (
            grounded_semantic_program_correct / total
        ),
        "execution_accuracy": execution_correct / total,
        "pointer_pair_accuracy": pointer_pair_correct / total,
        "raw_pointer_pair_accuracy": raw_pointer_pair_correct / total,
        "accepted_coverage": accepted / total,
        # Backward-compatible alias: historically accepted_precision meant
        # execution-equivalent correctness, not exact program correctness.
        "accepted_precision": (
            accepted_execution_correct / accepted if accepted else 0.0
        ),
        "accepted_execution_precision": (
            accepted_execution_correct / accepted if accepted else 0.0
        ),
        "accepted_program_precision": (
            accepted_program_correct / accepted if accepted else 0.0
        ),
        "accepted_grounded_semantic_program_precision": (
            accepted_grounded_semantic_program_correct / accepted
            if accepted
            else 0.0
        ),
        "accepted_precision_defined": bool(accepted),
        "accepted_execution_precision_defined": bool(accepted),
        "accepted_program_precision_defined": bool(accepted),
        "accepted_grounded_semantic_program_precision_defined": bool(accepted),
        "proof_eligible_coverage": proof_eligible / total,
        # Backward-compatible execution-equivalence alias for proof eligibility.
        "proof_eligible_precision": (
            proof_eligible_execution_correct / proof_eligible
            if proof_eligible
            else 0.0
        ),
        "proof_eligible_execution_precision": (
            proof_eligible_execution_correct / proof_eligible
            if proof_eligible
            else 0.0
        ),
        "proof_eligible_program_precision": (
            proof_eligible_program_correct / proof_eligible
            if proof_eligible
            else 0.0
        ),
        "proof_eligible_grounded_semantic_program_precision": (
            proof_eligible_grounded_semantic_program_correct / proof_eligible
            if proof_eligible
            else 0.0
        ),
        "proof_eligible_precision_defined": bool(proof_eligible),
        "proof_eligible_execution_precision_defined": bool(proof_eligible),
        "proof_eligible_program_precision_defined": bool(proof_eligible),
        "proof_eligible_grounded_semantic_program_precision_defined": bool(proof_eligible),
        "mean_segments": segment_total / total,
        "program_search_change_rate": program_search_changes / total,
        "mean_program_candidates_considered": (
            program_candidates_total / total
        ),
        "mean_symbolic_program_rejections": (
            symbolic_rejections_total / total
        ),
        "mean_symbolic_unknown_constraints": symbolic_unknown_total / total,
        "gold_program_recall_at_5": program_recall[5] / total,
        "gold_program_recall_at_10": program_recall[10] / total,
        "gold_physical_program_recall_at_5": (
            physical_program_recall[5] / total
        ),
        "gold_physical_program_recall_at_10": (
            physical_program_recall[10] / total
        ),
        "gold_operand_recall_at_5": operand_recall[5] / total,
        "gold_operand_recall_at_10": operand_recall[10] / total,
        "operand_candidate_recall_at_10": (
            operand_candidate_matches[10] / operand_candidate_gold_operands
            if operand_candidate_gold_operands
            else 0.0
        ),
        "operand_candidate_recall_at_20": (
            operand_candidate_matches[20] / operand_candidate_gold_operands
            if operand_candidate_gold_operands
            else 0.0
        ),
        "operand_candidate_full_coverage_at_10": (
            operand_candidate_full_coverage[10]
            / diverse_beam_diagnostic_examples
            if diverse_beam_diagnostic_examples
            else 0.0
        ),
        "operand_candidate_full_coverage_at_20": (
            operand_candidate_full_coverage[20]
            / diverse_beam_diagnostic_examples
            if diverse_beam_diagnostic_examples
            else 0.0
        ),
        "operand_candidate_recall_at_10_given_gold_operation": (
            operand_candidate_matches_given_gold_operation[10]
            / operand_candidate_gold_operation_operands
            if operand_candidate_gold_operation_operands
            else 0.0
        ),
        "operand_candidate_recall_at_20_given_gold_operation": (
            operand_candidate_matches_given_gold_operation[20]
            / operand_candidate_gold_operation_operands
            if operand_candidate_gold_operation_operands
            else 0.0
        ),
        "operand_candidate_gold_operand_count": float(
            operand_candidate_gold_operands
        ),
        "operand_candidate_gold_operation_operand_count": float(
            operand_candidate_gold_operation_operands
        ),
        "diverse_beam_diagnostic_examples": float(
            diverse_beam_diagnostic_examples
        ),
        "gold_operation_not_considered_count": float(operation_misses),
        "gold_operand_outside_slot_pool_at_20_count": float(
            pool_misses
        ),
        "gold_program_not_generated_count": float(generation_misses),
        "gold_program_below_preverifier_top10_count": float(
            ranking_misses
        ),
        "gold_top10_verifier_rejected_count": float(verifier_invalid),
        "gold_top10_verifier_invalid_count": float(verifier_invalid),
        "gold_top10_runtime_gate_rejected_count": float(
            runtime_gate_rejections
        ),
        "gold_top10_eligible_count": float(eligible_successes),
        "gold_pipeline_diagnostic_unavailable_count": float(
            diagnostic_unavailable
        ),
        "program_search_failure_bucket_total": float(failure_bucket_total),
        "gold_operation_not_considered_rate": (
            operation_misses / diverse_beam_diagnostic_examples
            if diverse_beam_diagnostic_examples
            else 0.0
        ),
        "gold_operand_outside_slot_pool_at_20_rate": (
            pool_misses / diverse_beam_diagnostic_examples
            if diverse_beam_diagnostic_examples
            else 0.0
        ),
        "gold_program_not_generated_rate": (
            generation_misses / diverse_beam_diagnostic_examples
            if diverse_beam_diagnostic_examples
            else 0.0
        ),
        "gold_program_below_preverifier_top10_rate": (
            ranking_misses / diverse_beam_diagnostic_examples
            if diverse_beam_diagnostic_examples
            else 0.0
        ),
        "gold_top10_verifier_rejected_rate": (
            verifier_invalid / diverse_beam_diagnostic_examples
            if diverse_beam_diagnostic_examples
            else 0.0
        ),
        "gold_top10_runtime_gate_rejected_rate": (
            runtime_gate_rejections / diverse_beam_diagnostic_examples
            if diverse_beam_diagnostic_examples
            else 0.0
        ),
        "gold_top10_eligible_rate": (
            eligible_successes / diverse_beam_diagnostic_examples
            if diverse_beam_diagnostic_examples
            else 0.0
        ),
        "gold_pipeline_diagnostic_unavailable_rate": (
            diagnostic_unavailable / diverse_beam_diagnostic_examples
            if diverse_beam_diagnostic_examples
            else 0.0
        ),
        "mean_generated_programs_before_physical_dedup": (
            generated_before_total / diverse_beam_diagnostic_examples
            if diverse_beam_diagnostic_examples
            else 0.0
        ),
        "mean_generated_programs_after_physical_dedup": (
            generated_after_total / diverse_beam_diagnostic_examples
            if diverse_beam_diagnostic_examples
            else 0.0
        ),
        "gold_symbolic_program_recall_at_5": (
            symbolic_program_recall[5] / total
        ),
        "gold_symbolic_program_recall_at_10": (
            symbolic_program_recall[10] / total
        ),
        "gold_program_generated_recall": verifier_gold_generated / total,
        "gold_eligible_program_recall": eligible_gold_generated / total,
        "typed_synthesis_gold_generation_rate": typed_gold_generated / total,
        "typed_synthesis_example_rate": typed_synthesis_examples / total,
        "mean_synthesized_program_candidates": (
            synthesized_candidates_total / total
        ),
        "program_search_trace_coverage": search_trace_examples / total,
        "single_eligible_program_rate": single_eligible_programs / total,
        "verifier_gold_generated_count": float(verifier_gold_generated),
        "verifier_gold_false_rejection_count": float(
            verifier_gold_false_rejections
        ),
        "verifier_gold_false_rejection_rate": (
            verifier_gold_false_rejections / verifier_gold_generated
            if verifier_gold_generated
            else 0.0
        ),
        "search_pipeline_gold_rejection_rate": (
            pipeline_gold_rejections / verifier_gold_generated
            if verifier_gold_generated
            else 0.0
        ),
        "example_count": float(total),
        "program_exact_correct_count": float(program_correct),
        "raw_program_exact_correct_count": float(raw_program_correct),
        "grounded_semantic_program_correct_count": float(
            grounded_semantic_program_correct
        ),
        "execution_correct_count": float(execution_correct),
        "accepted_count": float(accepted),
        "accepted_execution_correct_count": float(accepted_execution_correct),
        "accepted_program_correct_count": float(accepted_program_correct),
        "accepted_grounded_semantic_program_correct_count": float(
            accepted_grounded_semantic_program_correct
        ),
        "proof_eligible_count": float(proof_eligible),
        "proof_eligible_execution_correct_count": float(proof_eligible_execution_correct),
        "proof_eligible_program_correct_count": float(proof_eligible_program_correct),
        "proof_eligible_grounded_semantic_program_correct_count": float(
            proof_eligible_grounded_semantic_program_correct
        ),
    }
    metrics.update(
        _operation_classification_metrics(
            expected_operations,
            predicted_operations,
            operations=evaluation_operations,
        )
    )
    metrics.update(
        {
            f"gold_pipeline_rejection_reason__{reason}": float(count)
            for reason, count in sorted(
                pipeline_gold_rejection_reasons.items()
            )
        }
    )
    metrics.update(
        {
            f"pointer_slot_{slot + 1}_accuracy": (
                pointer_correct[slot] / pointer_total[slot]
                if pointer_total[slot]
                else 0.0
            )
            for slot in range(operand_width)
        }
    )
    for slot in range(operand_width):
        metrics[f"pointer_slot_{slot + 1}_support"] = float(pointer_total[slot])
        metrics[f"pointer_slot_{slot + 1}_accuracy_defined"] = bool(pointer_total[slot])

    for arity, counts_by_arity in sum_physical_set_by_arity.items():
        support = int(counts_by_arity["support"])
        metrics[f"exact_physical_set_arity_{arity}_support"] = float(
            support
        )
        metrics[f"exact_physical_set_arity_{arity}_accuracy"] = (
            counts_by_arity["correct"] / support if support else 0.0
        )
    for arity, counts_by_arity in physical_program_by_arity.items():
        support = int(counts_by_arity["support"])
        metrics[f"exact_physical_program_arity_{arity}_support"] = float(
            support
        )
        metrics[f"exact_physical_program_arity_{arity}_accuracy"] = (
            counts_by_arity["correct"] / support if support else 0.0
        )
    for operation, quality in operation_quality.items():
        support = quality["support"]
        accepted_for_operation = quality["accepted"]
        prefix = f"program_operation_{operation.lower()}"
        proof_for_operation = quality["proof_eligible"]
        metrics.update(
            {
                f"{prefix}_support": float(support),
                f"{prefix}_operation_accuracy": (
                    quality["operation_correct"] / support if support else 0.0
                ),
                f"{prefix}_pointer_pair_accuracy": (
                    quality["pointer_pair_correct"] / support if support else 0.0
                ),
                f"{prefix}_raw_pointer_pair_accuracy": (
                    quality["raw_pointer_pair_correct"] / support
                    if support
                    else 0.0
                ),
                f"{prefix}_program_exact_accuracy": (
                    quality["program_correct"] / support if support else 0.0
                ),
                f"{prefix}_raw_program_exact_accuracy": (
                    quality["raw_program_correct"] / support if support else 0.0
                ),
                f"{prefix}_grounded_semantic_program_accuracy": (
                    quality["grounded_semantic_program_correct"] / support
                    if support
                    else 0.0
                ),
                f"{prefix}_execution_accuracy": (
                    quality["execution_correct"] / support if support else 0.0
                ),
                f"{prefix}_accepted_coverage": (
                    accepted_for_operation / support if support else 0.0
                ),
                f"{prefix}_accepted_execution_precision": (
                    quality["accepted_execution_correct"] / accepted_for_operation
                    if accepted_for_operation
                    else 0.0
                ),
                f"{prefix}_accepted_program_precision": (
                    quality["accepted_program_correct"] / accepted_for_operation
                    if accepted_for_operation
                    else 0.0
                ),
                f"{prefix}_accepted_grounded_semantic_program_precision": (
                    quality["accepted_grounded_semantic_program_correct"]
                    / accepted_for_operation
                    if accepted_for_operation
                    else 0.0
                ),
                f"{prefix}_accepted_count": float(accepted_for_operation),
                f"{prefix}_accepted_execution_correct_count": float(
                    quality["accepted_execution_correct"]
                ),
                f"{prefix}_accepted_program_correct_count": float(
                    quality["accepted_program_correct"]
                ),
                f"{prefix}_accepted_precision_defined": bool(accepted_for_operation),
                f"{prefix}_accepted_execution_precision_defined": bool(accepted_for_operation),
                f"{prefix}_accepted_program_precision_defined": bool(accepted_for_operation),
                f"{prefix}_proof_eligible_coverage": (
                    proof_for_operation / support if support else 0.0
                ),
                f"{prefix}_proof_eligible_execution_precision": (
                    quality["proof_eligible_execution_correct"] / proof_for_operation
                    if proof_for_operation
                    else 0.0
                ),
                f"{prefix}_proof_eligible_program_precision": (
                    quality["proof_eligible_program_correct"] / proof_for_operation
                    if proof_for_operation
                    else 0.0
                ),
                f"{prefix}_proof_eligible_precision_defined": bool(proof_for_operation),
                f"{prefix}_proof_eligible_execution_precision_defined": bool(proof_for_operation),
                f"{prefix}_proof_eligible_program_precision_defined": bool(proof_for_operation),
                f"{prefix}_proof_eligible_count": float(proof_for_operation),
                f"{prefix}_proof_eligible_execution_correct_count": float(
                    quality["proof_eligible_execution_correct"]
                ),
                f"{prefix}_proof_eligible_program_correct_count": float(
                    quality["proof_eligible_program_correct"]
                ),
            }
        )
    return metrics


def analyze_document_hrm_errors(
    bundle: DocumentHRMBundle,
    examples_or_path: Sequence[ProgramExample] | str | Path,
    *,
    batch_size: int = 32,
    include_correct: bool = False,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Return aggregate diagnostics and JSON-serializable example records.

    Program datasets already contain gold pointers into their stored candidate
    lists, so candidate coverage here is a label-integrity check rather than an
    end-to-end retrieval recall measurement. The detailed records are intended
    to reveal whether failures come from operation classification, pointer
    selection, candidate ranking, or an execution-equivalent alternative.
    """
    examples = (
        load_program_examples(examples_or_path)
        if isinstance(examples_or_path, (str, Path))
        else list(examples_or_path)
    )
    if batch_size < 1:
        raise DocumentHRMError("batch_size must be positive")
    operand_width = _bundle_operand_width(bundle)

    category_counts: Counter[str] = Counter()
    per_operation: dict[str, Counter[str]] = {
        operation: Counter() for operation in OPERATIONS
    }
    pointer_correct = [0] * operand_width
    pointer_total = [0] * operand_width
    confidence_bins: dict[str, Counter[str]] = {
        "[0.0,0.2)": Counter(),
        "[0.2,0.4)": Counter(),
        "[0.4,0.6)": Counter(),
        "[0.6,0.8)": Counter(),
        "[0.8,1.0]": Counter(),
    }
    candidate_counts: list[int] = []
    gold_ranks: list[int] = []
    stored_gold_operands = 0
    covered_gold_operands = 0
    total = operation_correct = pointer_pair_correct = 0
    program_exact = execution_correct = 0
    records: list[dict[str, object]] = []

    def confidence_bin(value: float) -> str:
        if value < 0.2:
            return "[0.0,0.2)"
        if value < 0.4:
            return "[0.2,0.4)"
        if value < 0.6:
            return "[0.4,0.6)"
        if value < 0.8:
            return "[0.6,0.8)"
        return "[0.8,1.0]"

    def safe_execute(
        program: ReasoningProgram, example: ProgramExample
    ) -> str | None:
        try:
            return str(execute_program(program, example.candidates))
        except Exception:
            return None

    for start in range(0, len(examples), batch_size):
        current = examples[start : start + batch_size]
        predictions = bundle.predict_many(current)
        for example, prediction in zip(current, predictions):
            total += 1
            candidate_counts.append(len(example.candidates))
            expected = example.program
            predicted = prediction.program
            operation_matches = predicted.operation == expected.operation
            pointer_matches = tuple(predicted.operands) == tuple(expected.operands)
            exact_matches = predicted == expected
            execution_matches = _execution_is_correct(predicted, example)
            operation_correct += int(operation_matches)
            pointer_pair_correct += int(pointer_matches)
            program_exact += int(exact_matches)
            execution_correct += int(execution_matches)

            if exact_matches:
                category = "exact"
            elif execution_matches:
                category = "execution_equivalent"
            elif not operation_matches and not pointer_matches:
                category = "operation_and_pointer"
            elif not operation_matches:
                category = "operation_only"
            else:
                category = "pointer_only"
            category_counts[category] += 1

            operation_stats = per_operation[expected.operation]
            operation_stats["support"] += 1
            operation_stats["operation_correct"] += int(operation_matches)
            operation_stats["pointer_pair_correct"] += int(pointer_matches)
            operation_stats["program_exact"] += int(exact_matches)
            operation_stats["execution_correct"] += int(execution_matches)

            bin_stats = confidence_bins[confidence_bin(prediction.confidence)]
            bin_stats["total"] += 1
            bin_stats["execution_correct"] += int(execution_matches)

            relevance_order = sorted(
                range(len(example.candidates)),
                key=lambda index: (
                    example.candidates[index].relevance,
                    not example.candidates[index].is_year_like,
                    -example.candidates[index].line_number,
                ),
                reverse=True,
            )
            relevance_rank = {
                candidate_index: rank
                for rank, candidate_index in enumerate(relevance_order, start=1)
            }
            example_gold_ranks: list[int | None] = []
            for slot, pointer in enumerate(expected.operands):
                stored_gold_operands += 1
                covered = 0 <= pointer < len(example.candidates)
                covered_gold_operands += int(covered)
                rank = relevance_rank.get(pointer)
                example_gold_ranks.append(rank)
                if rank is not None:
                    gold_ranks.append(rank)
                pointer_total[slot] += 1
                predicted_pointer = (
                    predicted.operands[slot]
                    if slot < len(predicted.operands)
                    else None
                )
                pointer_correct[slot] += int(predicted_pointer == pointer)

            if exact_matches and not include_correct:
                continue

            gold_slots: dict[int, list[int]] = {}
            predicted_slots: dict[int, list[int]] = {}
            for slot, pointer in enumerate(expected.operands, start=1):
                gold_slots.setdefault(pointer, []).append(slot)
            for slot, pointer in enumerate(predicted.operands, start=1):
                predicted_slots.setdefault(pointer, []).append(slot)
            candidates = []
            for candidate in example.candidates:
                candidates.append(
                    {
                        "index": candidate.index,
                        "value": candidate.value,
                        "raw": candidate.raw,
                        "context": candidate.context,
                        "line_number": candidate.line_number,
                        "unit": candidate.unit,
                        "is_percent": candidate.is_percent,
                        "is_year_like": candidate.is_year_like,
                        "relevance": candidate.relevance,
                        "relevance_rank": relevance_rank.get(candidate.index),
                        "source_name": candidate.source_name,
                        "page": candidate.page,
                        "row_label": candidate.row_label,
                        "column_label": candidate.column_label,
                        "period": candidate.period,
                        "table_id": candidate.table_id,
                        "section": candidate.section,
                        "scope": candidate.scope,
                        "gold_slots": gold_slots.get(candidate.index, []),
                        "predicted_slots": predicted_slots.get(candidate.index, []),
                    }
                )
            records.append(
                {
                    "example_id": example.example_id,
                    "task": example.task,
                    "difficulty": example.difficulty,
                    "question": example.question,
                    "context": example.context,
                    "answer_text": example.answer_text,
                    "answer_value": example.answer_value,
                    "candidate_count": len(example.candidates),
                    "error_category": category,
                    "checks": {
                        "operation_correct": operation_matches,
                        "pointer_pair_correct": pointer_matches,
                        "program_exact": exact_matches,
                        "execution_correct": execution_matches,
                        "stored_gold_operands_present": all(
                            rank is not None for rank in example_gold_ranks
                        ),
                    },
                    "expected": {
                        "operation": expected.operation,
                        "operands": list(expected.operands),
                        "value": safe_execute(expected, example),
                        "gold_operand_relevance_ranks": example_gold_ranks,
                    },
                    "predicted": {
                        "operation": predicted.operation,
                        "operands": list(predicted.operands),
                        "value": safe_execute(predicted, example),
                        "confidence": prediction.confidence,
                        "operation_confidence": prediction.operation_confidence,
                        "pointer_confidences": list(
                            prediction.pointer_confidences
                        ),
                        "neural_operation_confidence": (
                            prediction.neural_operation_confidence
                        ),
                        "neural_pointer_confidences": list(
                            prediction.neural_pointer_confidences
                        ),
                        "confidence_schema": prediction.confidence_schema,
                        "grounding_constraints_applied": list(
                            prediction.grounding_constraints_applied
                        ),
                        "program_candidates_considered": (
                            prediction.program_candidates_considered
                        ),
                        "symbolic_program_rejections": (
                            prediction.symbolic_program_rejections
                        ),
                        "reranker_score": prediction.reranker_score,
                        "reranker_confidence": (
                            prediction.reranker_confidence
                        ),
                        "segments": prediction.segments,
                        "halt_q": prediction.halt_q,
                        "continue_q": prediction.continue_q,
                        "operation_probabilities": {
                            operation: probability
                            for operation, probability in zip(
                                OPERATIONS, prediction.operation_probabilities
                            )
                        },
                    },
                    "candidates": candidates,
                }
            )

    def rate(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    operation_summary: dict[str, object] = {}
    for operation, stats in per_operation.items():
        support = stats["support"]
        operation_summary[operation] = {
            "support": support,
            "operation_accuracy": rate(stats["operation_correct"], support),
            "pointer_pair_accuracy": rate(
                stats["pointer_pair_correct"], support
            ),
            "program_exact_accuracy": rate(stats["program_exact"], support),
            "execution_accuracy": rate(stats["execution_correct"], support),
        }

    confidence_summary = {
        name: {
            "total": stats["total"],
            "execution_correct": stats["execution_correct"],
            "execution_accuracy": rate(
                stats["execution_correct"], stats["total"]
            ),
        }
        for name, stats in confidence_bins.items()
    }
    rank_summary = {
        "mean": sum(gold_ranks) / len(gold_ranks) if gold_ranks else 0.0,
        "max": max(gold_ranks) if gold_ranks else 0,
        "top_1_rate": rate(sum(rank <= 1 for rank in gold_ranks), len(gold_ranks)),
        "top_5_rate": rate(sum(rank <= 5 for rank in gold_ranks), len(gold_ranks)),
        "top_10_rate": rate(
            sum(rank <= 10 for rank in gold_ranks), len(gold_ranks)
        ),
    }
    summary: dict[str, object] = {
        "examples": total,
        "records_written": len(records),
        "include_correct": include_correct,
        "metrics": {
            "operation_accuracy": rate(operation_correct, total),
            "pointer_pair_accuracy": rate(pointer_pair_correct, total),
            "program_exact_accuracy": rate(program_exact, total),
            "execution_accuracy": rate(execution_correct, total),
            "pointer_slot_accuracy": [
                rate(pointer_correct[slot], pointer_total[slot])
                for slot in range(operand_width)
            ],
            "pointer_slot_support": pointer_total,
        },
        "error_categories": dict(category_counts),
        "per_operation": operation_summary,
        "candidate_diagnostics": {
            "candidate_count_mean": (
                sum(candidate_counts) / len(candidate_counts)
                if candidate_counts
                else 0.0
            ),
            "candidate_count_min": min(candidate_counts) if candidate_counts else 0,
            "candidate_count_max": max(candidate_counts) if candidate_counts else 0,
            "stored_gold_operand_coverage": rate(
                covered_gold_operands, stored_gold_operands
            ),
            "stored_gold_operand_count": stored_gold_operands,
            "gold_operand_relevance_rank": rank_summary,
            "coverage_note": (
                "Gold pointers are part of each stored ProgramExample, so this "
                "coverage is a dataset integrity check, not PDF/TXT retrieval recall."
            ),
        },
        "confidence_bins": confidence_summary,
    }
    return summary, records


def _operation_classification_metrics(
    expected: Sequence[str],
    predicted: Sequence[str],
    *,
    operations: Sequence[str] = OPERATIONS,
) -> dict[str, float]:
    """Compute sklearn-compatible macro and support-weighted operation scores."""
    rows: list[tuple[float, float, float, int]] = []
    result: dict[str, float] = {}
    for operation in operations:
        prefix = f"operation_class_{operation.lower()}"
        result.update(
            {
                f"{prefix}_precision": 0.0,
                f"{prefix}_recall": 0.0,
                f"{prefix}_f1": 0.0,
                f"{prefix}_support": 0.0,
            }
        )
    for operation in operations:
        true_positive = sum(
            truth == operation and guess == operation
            for truth, guess in zip(expected, predicted)
        )
        false_positive = sum(
            truth != operation and guess == operation
            for truth, guess in zip(expected, predicted)
        )
        false_negative = sum(
            truth == operation and guess != operation
            for truth, guess in zip(expected, predicted)
        )
        support = true_positive + false_negative
        if support == 0 and false_positive == 0:
            continue
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, support)
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        prefix = f"operation_class_{operation.lower()}"
        result.update(
            {
                f"{prefix}_precision": precision,
                f"{prefix}_recall": recall,
                f"{prefix}_f1": f1,
                f"{prefix}_support": float(support),
            }
        )
        rows.append((precision, recall, f1, support))
    total_support = sum(row[3] for row in rows)
    if not rows or not total_support:
        result.update({
            "operation_macro_precision": 0.0,
            "operation_macro_recall": 0.0,
            "operation_macro_f1": 0.0,
            "operation_weighted_precision": 0.0,
            "operation_weighted_recall": 0.0,
            "operation_weighted_f1": 0.0,
        })
        return result
    result.update({
        "operation_macro_precision": sum(row[0] for row in rows) / len(rows),
        "operation_macro_recall": sum(row[1] for row in rows) / len(rows),
        "operation_macro_f1": sum(row[2] for row in rows) / len(rows),
        "operation_weighted_precision": sum(row[0] * row[3] for row in rows)
        / total_support,
        "operation_weighted_recall": sum(row[1] * row[3] for row in rows)
        / total_support,
        "operation_weighted_f1": sum(row[2] * row[3] for row in rows)
        / total_support,
    })
    return result


_OPERAND_ALIGNMENT_STOPWORDS = {
    "bao",
    "bang",
    "cho",
    "context",
    "cua",
    "cuoi",
    "dau",
    "duoc",
    "den",
    "giam",
    "gia",
    "la",
    "line",
    "nam",
    "nhieu",
    "phan",
    "sang",
    "so",
    "tang",
    "thay",
    "tinh",
    "tri",
    "tu",
    "ty",
    "value",
    "voi",
}


def _apply_symbolic_operation_constraint(
    question: str, predicted_operation: str
) -> str:
    """Apply only explicit arithmetic wording; ambiguous questions stay neural."""
    text = normalize_text(question)
    explicit_sum = _explicit_sum_operation(question, text)
    if explicit_sum is not None:
        return explicit_sum
    plan = compile_question_plan(question)
    planned_operation = plan.operation
    if planned_operation is not None:
        return planned_operation
    if plan.operation_candidates:
        if predicted_operation in plan.operation_candidates:
            return predicted_operation
        return plan.operation_candidates[0]
    padded = f" {text} "
    percent_wording = "phan tram" in text or "%" in question
    if percent_wording and any(
        phrase in text
        for phrase in (
            "tang bao nhieu",
            "giam bao nhieu",
            "thay doi bao nhieu",
            "muc thay doi",
        )
    ):
        return "PERCENT_CHANGE"
    if "chiem bao nhieu phan tram" in text:
        return "RATIO_PERCENT"
    if any(
        phrase in text
        for phrase in (
            "ty le tang truong",
            "toc do tang truong",
            "phan tram tang",
            "phan tram giam",
        )
    ):
        return "PERCENT_CHANGE"
    period_comparison = (
        len(_requested_years(question)) >= 2
        or ("cuoi ky" in text and "dau ky" in text)
        or ("cuoi nam" in text and "dau nam" in text)
        or ("nam nay" in text and "nam truoc" in text)
    )
    if "so sanh" in text and period_comparison and not percent_wording:
        return "SUBTRACT"
    if any(
        phrase in text
        for phrase in (
            "giam bao nhieu",
            "tang bao nhieu",
            "muc thay doi la bao nhieu",
            "chenh lech",
        )
    ):
        return "SUBTRACT"
    if "ty le" in text and " tren " in padded:
        return "RATIO_PERCENT"
    if "ty trong" in text and " trong tong " in padded:
        return "RATIO_PERCENT"
    if " chia cho " in padded:
        return "RATIO_PERCENT" if percent_wording else "DIVIDE"
    if "he so" in text and (" tren " in padded or "chia" in text):
        return "DIVIDE"
    if (
        any(term in text for term in ("binh quan", "trung binh"))
        and (
            " giua " in padded
            or (
                any(term in text for term in ("dau nam", "dau ky"))
                and any(term in text for term in ("cuoi nam", "cuoi ky"))
            )
        )
    ):
        return "AVERAGE"
    if (
        any(
            phrase in padded
            for phrase in (
                " nhan voi ",
                " nhan cho ",
                " tich cua ",
                " tich giua ",
                " tinh tich ",
            )
        )
        or (
            "gia tri theo menh gia" in text
            and any(term in text for term in ("co phieu", "co phan"))
        )
    ):
        return "MULTIPLY"
    if _question_requests_addition(text):
        explicit_sum = _explicit_sum_operation(question, text)
        if explicit_sum is not None:
            return explicit_sum
        if predicted_operation in {"SUM3", "SUM4"}:
            return predicted_operation
        return "ADD"
    if _question_requests_single_value_lookup(question):
        return "SELECT"
    return predicted_operation


def _plan_financial_identities(plan: QuestionPlan) -> tuple[FinancialIdentity, ...]:
    # A direct identity target is a primary SELECT, but reports that omit the
    # disclosed total may still prove it from catalogued components.  Other
    # explicit operations must not activate unrelated identity alternatives.
    if plan.operation not in {None, "SELECT"}:
        return ()
    return tuple(
        get_financial_identity(identity_id)
        for identity_id in plan.financial_identity_ids
    )


def _has_direct_financial_identity_evidence(
    plan: QuestionPlan,
    graph: EvidenceGraph,
    identities: Sequence[FinancialIdentity],
) -> bool:
    """Prefer a disclosed target cell over reconstructing the same metric."""

    if _question_requires_financial_identity_computation(plan):
        return False

    requested_years = set(plan.requested_years)
    for cell in graph.cells:
        if not any(
            target_matches_metric(identity, cell.metric_label)
            for identity in identities
        ):
            continue
        effective_years = cell.effective_years(plan)
        if requested_years and effective_years and not (
            requested_years & effective_years
        ):
            continue
        if plan.entity_mode == "reporting" and cell.embedded_entity:
            continue
        if plan.entity_mode == "named" and plan.entity_terms:
            entity_tokens = set(
                normalize_text(
                    f"{cell.entity_id} {cell.entity} {cell.section} "
                    f"{cell.source_name}"
                ).split()
            )
            if not entity_tokens.intersection(plan.entity_terms):
                continue
        if (
            plan.requested_unit_family
            and cell.unit_key
            and cell.unit_key != plan.requested_unit_family
        ):
            continue
        return True
    return False


def _question_requires_financial_identity_computation(
    plan: QuestionPlan,
) -> bool:
    """Distinguish a requested derivation from a plain disclosed-value lookup."""

    return requests_financial_identity_computation(plan.normalized)


def _explicit_sum_operation(
    question: str, normalized_question: str
) -> str | None:
    """Return SUM3/SUM4 only when the question exposes a clear list arity."""
    if re.search(r"\btong(?: cua)? (?:bon|4)\b", normalized_question):
        return "SUM4"
    if re.search(r"\btong(?: cua)? (?:ba|3)\b", normalized_question):
        return "SUM3"

    # Commas and ``va`` also occur inside metric labels and between requested
    # periods (for example, an AVERAGE question). Treat a punctuation list as
    # a sum only when the question itself explicitly starts with a total
    # request. This prevents row text from silently changing the operation.
    if not re.match(r"^(?:tinh\s+)?tong\b", normalized_question):
        return None

    quoted_items = re.findall(r'["“][^"”]+["”]', question)
    if len(quoted_items) == 4:
        return "SUM4"
    if len(quoted_items) == 3:
        return "SUM3"

    # Preserve commas while normalizing Vietnamese so a list such as
    # ``A, B và C`` can be distinguished from a binary metric whose label
    # itself contains a comma after the conjunction.
    marked = normalize_text(question.replace(",", " codexcomma "))
    final_and = marked.rfind(" va ")
    if final_and >= 0:
        separators = marked[:final_and].count(" codexcomma ")
        if separators >= 2:
            return "SUM4"
        if separators == 1:
            return "SUM3"
    return None


def _explicit_sum_operand_count(question: str) -> int | None:
    """Return a source-worded SUM cardinality without guessing missing items."""

    if not question:
        return None
    normalized = normalize_text(question)
    count_words = {
        "hai": 2,
        "ba": 3,
        "bon": 4,
        "nam": 5,
        "sau": 6,
        "bay": 7,
        "tam": 8,
    }
    count_match = re.search(
        r"\btong(?: cua)?\s+(hai|ba|bon|nam|sau|bay|tam|[2-8])\b",
        normalized,
    )
    if count_match:
        token = count_match.group(1)
        return int(token) if token.isdigit() else count_words[token]

    legacy_operation = _explicit_sum_operation(question, normalized)
    if legacy_operation is not None:
        return OPERATION_ARITY[legacy_operation]
    if not re.match(r"^(?:tinh\s+)?tong\b", normalized):
        return None
    quoted_count = len(
        re.findall(r'(?:"[^"]+"|\u201c[^\u201d]+\u201d)', question)
    )
    if 2 <= quoted_count <= SUM_N_MAX_OPERANDS:
        return quoted_count
    for arity in range(2, SUM_N_MAX_OPERANDS + 1):
        if _explicit_multi_operand_parts(question, arity) is not None:
            return arity
    return 2 if _question_requests_addition(normalized) else None


def _align_directed_operands(
    question: str,
    operation: str,
    operands: Sequence[int],
    candidates: Sequence[NumericCandidate],
) -> tuple[int, ...]:
    """Orient an HRM-selected pair using the shared question contract."""
    if len(operands) != 2:
        return tuple(operands)
    if operation in {"SUBTRACT", "PERCENT_CHANGE"}:
        plan = compile_question_plan(question)
        program = orient_program_direction(
            plan,
            ReasoningProgram(operation, tuple(operands)),
            EvidenceGraph(cells=evidence_cells(candidates), edges=()),
        )
        return program.operands
    if operation not in {"RATIO_PERCENT", "DIVIDE"}:
        return tuple(operands)
    normalized = f" {normalize_text(question)} "
    marker = next(
        (
            item
            for item in (
                " chiem bao nhieu phan tram ",
                " chia cho ",
                " tren ",
            )
            if item in normalized
        ),
        None,
    )
    if marker is None:
        return tuple(operands)
    left, right = normalized.split(marker, 1)

    left_terms = _alignment_terms(left)
    right_terms = _alignment_terms(right)
    if not left_terms or not right_terms:
        return tuple(operands)
    first, second = operands
    if first >= len(candidates) or second >= len(candidates):
        return tuple(operands)
    first_terms = _alignment_terms(_candidate_row_label(candidates[first]))
    second_terms = _alignment_terms(_candidate_row_label(candidates[second]))
    current_score = _fuzzy_term_score(left_terms, first_terms) + _fuzzy_term_score(
        right_terms, second_terms
    )
    reversed_score = _fuzzy_term_score(left_terms, second_terms) + _fuzzy_term_score(
        right_terms, first_terms
    )
    if reversed_score > current_score:
        return second, first
    return first, second


def _requested_years(question: str) -> tuple[str, ...]:
    return compile_question_plan(question).requested_years


def _candidate_period_descriptor(candidate: NumericCandidate) -> str:
    period = _context_field(candidate.context, "PERIOD") or candidate.period
    column = _context_field(candidate.context, "COLUMN") or candidate.column_label
    from .column_context import normalize_period_text
    return normalize_period_text(f"{period} {column}")


def _candidate_period_role(candidate: NumericCandidate) -> str:
    descriptor = f" {_candidate_period_descriptor(candidate)} "
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


def _candidate_exact_date_years(candidate: NumericCandidate) -> set[str]:
    """Return calendar years explicitly encoded by a full candidate date.

    ``01/01/YYYY`` is an opening endpoint, but it is still inside calendar
    year ``YYYY``.  Period-role reasoning may distinguish it from 31 December;
    it must not decrement the already-explicit year a second time.
    """

    descriptor = _candidate_period_descriptor(candidate)
    return set(
        re.findall(
            r"\b(?:0?[1-9]|[12]\d|3[01])\s+"
            r"(?:0?[1-9]|1[0-2])\s+((?:19|20)\d{2})\b",
            descriptor,
        )
    )


def _candidate_period_years(
    candidate: NumericCandidate, question: str = ""
) -> set[str]:
    descriptor = _candidate_period_descriptor(candidate)
    explicit = set(re.findall(r"\b(?:19|20)\d{2}\b", descriptor))
    exact_date_years = _candidate_exact_date_years(candidate)
    if exact_date_years:
        return exact_date_years
    requested = sorted(_requested_years(question), key=int)
    role = _candidate_period_role(candidate)
    if role == "current":
        if explicit:
            return explicit
        if requested:
            return {requested[-1]}
    if role == "prior":
        if "nam truoc" in descriptor and requested:
            return {requested[-2] if len(requested) >= 2 else str(int(requested[-1]) - 1)}
        base = max((int(year) for year in explicit), default=None)
        if base is not None:
            return {str(base - 1)}
        if requested:
            return {str(int(requested[-1]) - 1)}
    return explicit


def _period_compatible_indices(
    question: str,
    operation: str,
    candidates: Sequence[NumericCandidate],
    arity: int,
) -> set[int]:
    if operation not in {
        "SELECT",
        "ADD",
        "SUBTRACT",
        "DIVIDE",
        "RATIO_PERCENT",
        "PERCENT_CHANGE",
        "AVERAGE",
        "SUM3",
        "SUM4",
        SUM_N_OPERATION,
    }:
        return set()
    requested = set(_requested_years(question))
    normalized = f" {normalize_text(question)} "
    relative_pair = any(
        marker in normalized
        for marker in (" cuoi nam ", " cuoi ky ", " so cuoi nam ")
    ) and any(
        marker in normalized
        for marker in (" dau nam ", " dau ky ", " so dau nam ")
    )
    if relative_pair and operation in {"SUBTRACT", "PERCENT_CHANGE"}:
        compatible = {
            index
            for index, candidate in enumerate(candidates)
            if _candidate_period_role(candidate) in {"current", "prior"}
        }
        if len(compatible) >= arity:
            return compatible
    if not requested:
        return set()
    candidate_years = [
        _candidate_period_years(candidate, question) for candidate in candidates
    ]
    compatible = {
        index for index, years in enumerate(candidate_years) if years & requested
    }
    if len(compatible) < arity:
        return set()
    if len(requested) >= 2 and operation in {"SUBTRACT", "PERCENT_CHANGE"}:
        covered = set().union(*(candidate_years[index] for index in compatible))
        if not requested.issubset(covered):
            return set()
    return compatible


def _fuzzy_term_score(expected: set[str], observed: set[str]) -> float:
    score = 0.0
    for term in expected:
        best = max(
            (
                1.0
                if term == candidate
                else 0.92
                if len(term) >= 3 and term in candidate
                else 0.85
                if len(candidate) >= 4 and candidate in term
                else SequenceMatcher(None, term, candidate).ratio()
                for candidate in observed
            ),
            default=0.0,
        )
        if best >= 0.72:
            score += best
    return score


def _alignment_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_text(text))
        if len(token) > 1
        and token not in _OPERAND_ALIGNMENT_STOPWORDS
        and not token.isdigit()
    }


def _candidate_row_label(candidate: NumericCandidate) -> str:
    return _structured_candidate_row_label(candidate) or candidate.context


def _structured_candidate_row_label(candidate: NumericCandidate) -> str:
    return _context_field(candidate.context, "ROW") or candidate.row_label


def _is_groundable_document_candidate(candidate: NumericCandidate) -> bool:
    """Return true for structured evidence produced by document ingestion.

    Empty extraction methods belong to training/synthetic contexts.  Keeping
    those on the learned path preserves checkpoint validation behavior while
    native PDF rows receive the same safeguards as OCR rows at runtime.
    """

    return bool(_structured_candidate_row_label(candidate)) and (
        candidate.extraction_method.lower()
        in {
            "native",
            "text",
            "ocr",
            "mineru",
            "mineru_markdown",
            "symbolic-derived",
        }
    )


def _operand_role_terms(
    question: str, operation: str
) -> tuple[set[str], set[str]] | None:
    parts = _operand_role_parts(question, operation)
    if parts is None:
        return None
    left, right = parts
    left_terms = _alignment_terms(left)
    right_terms = _alignment_terms(right)
    if not left_terms or not right_terms:
        return None
    return left_terms, right_terms


def _operand_role_parts(
    question: str, operation: str
) -> tuple[str, str] | None:
    plan = compile_question_plan(question)
    if (
        len(plan.operand_roles) == 2
        and operation in plan.operation_candidates
        and operation != "SELECT"
    ):
        return plan.operand_roles[0], plan.operand_roles[1]
    if operation == "ADD":
        focus = _question_focus_text(question)
        quoted = [
            left or right
            for left, right in re.findall(
                r"'([^']+)'|\"([^\"]+)\"",
                focus,
            )
        ]
        if len(quoted) == 2:
            return quoted[0].strip(), quoted[1].strip()

        normalized = normalize_text(focus)
        if not _question_requests_addition(normalized):
            return None
        expression = re.split(
            r"\s+(?:tai|vao)\s+(?=(?:nam\s+)?(?:cuoi|dau|ngay|"
            r"(?:19|20)\d{2}))|\s+(?:trong nam|nam)\s+(?=(?:19|20)\d{2})",
            normalized,
            maxsplit=1,
        )[0]
        expression = re.sub(
            r"^(?:tinh\s+)?(?:tong|cong)\s+",
            "",
            expression,
            count=1,
        ).strip()
        for marker in (" va ", " voi "):
            if marker not in expression:
                continue
            left, right = expression.rsplit(marker, 1)
            if left.strip() and right.strip():
                return left.strip(), right.strip()
        return None
    if operation == "MULTIPLY":
        normalized = f" {normalize_text(_question_focus_text(question))} "
        par_value_marker = " gia tri theo menh gia cua "
        if par_value_marker in normalized:
            _, metric = normalized.split(par_value_marker, 1)
            metric = re.split(
                r"\s+(?:tai|nam|vao)\s+", metric, maxsplit=1
            )[0]
            if metric.strip():
                return metric.strip(), "menh gia"
        return None
    if operation not in {"RATIO_PERCENT", "DIVIDE"}:
        return None
    normalized = f" {normalize_text(_question_focus_text(question))} "
    marker = next(
        (
            item
            for item in (
                " chiem bao nhieu phan tram ",
                " chia cho ",
                " tren ",
            )
            if item in normalized
        ),
        None,
    )
    if marker is None:
        return None
    left, right = normalized.split(marker, 1)
    if " ty le " in left:
        left = left.rsplit(" ty le ", 1)[-1]
    return left.strip(), right.strip()


def _question_focus_text(question: str) -> str:
    """Remove a leading statement qualifier such as ``Theo báo cáo...,``."""

    if "," in question and normalize_text(question).startswith("theo "):
        return question.split(",", 1)[-1].strip()
    return question


def _metric_compatible_indices(
    question: str,
    candidates: Sequence[NumericCandidate],
    pool: set[int],
) -> set[int]:
    """Constrain document SELECT pointers to the explicitly named metric."""

    plan = compile_question_plan(question)
    # Exclude known dimensional contradictions before lexical scores can
    # promote an unrelated quantity. Unknown units remain for proof to resolve.
    nonmonetary = {"m2", "m3", "ha", "kg", "tan", "nguoi", "share_count", "percent", "ratio"}
    structured_pool = {
        index
        for index in pool
        if _is_groundable_document_candidate(candidates[index])
        and not (
            plan.requested_unit_family == "currency"
            and _unit_key(candidates[index].unit, is_percent=candidates[index].is_percent)
            in nonmonetary
        )
        and not _metric_modifier_conflicts(
            question, _structured_candidate_row_label(candidates[index])
        )
        and _metric_anchor_compatible(
            # An explicit report-field owner is a separate query dimension;
            # it need not be repeated inside the source metric label.
            plan.metric_text if (plan.requested_accounting_scope or (plan.requested_report_year and plan.entity_mode == "named"))
            else _question_focus_text(question),
            _structured_candidate_row_label(candidates[index]),
        )
    }
    if not structured_pool:
        return set()
    # A row already satisfying the verifier's semantic metric obligation
    # must not lose to a long paragraph matching issuer/currency surface words.
    # Keep every valid row: selection cannot establish authority by score/value.
    proven_metric = {
        index for index in structured_pool
        if _metric_obligation_state(
            plan.metric_text, _structured_candidate_row_label(candidates[index])
        ) is VerificationState.VALID
    }
    if proven_metric:
        return proven_metric
    target_terms = _alignment_terms(_question_focus_text(question))
    if not target_terms:
        return set()
    scores = {
        index: _fuzzy_term_score(
            target_terms,
            _alignment_terms(_structured_candidate_row_label(candidates[index])),
        )
        for index in structured_pool
    }
    best = max(scores.values(), default=0.0)
    if best < 1.5:
        return set()
    return {index for index, score in scores.items() if score >= best - 0.30}


def _explicit_multi_operand_parts(
    question: str, arity: int
) -> tuple[str, ...] | None:
    """Extract an explicit three/four-metric list without guessing its labels."""
    quoted = [
        left or right
        for left, right in re.findall(
            r'"([^"]+)"|“([^”]+)”',
            question,
        )
    ]
    if len(quoted) == arity:
        return tuple(quoted)

    focus = _question_focus_text(question)
    listed = re.split(
        r"\s+(?:tại|trong năm|năm)\s+",
        focus,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    listed = re.sub(
        r"^\s*(?:tính\s+)?tổng"
        r"(?:\s+(?:ba|bốn|3|4))?"
        r"(?:\s+số\s+học\s+của)?\s+",
        "",
        listed,
        count=1,
        flags=re.IGNORECASE,
    )
    head, separator, tail = listed.rpartition(" và ")
    if not separator:
        head, separator, tail = listed.rpartition(" VÀ ")
    if not separator:
        return None
    parts = [*head.split(","), tail]
    cleaned = tuple(part.strip(' "\t') for part in parts if part.strip(' "\t'))
    return cleaned if len(cleaned) == arity else None


def _explicit_multi_operand_indices(
    question: str,
    operation: str,
    candidates: Sequence[NumericCandidate],
    *,
    arity: int | None = None,
) -> tuple[int, ...] | None:
    """Ground an explicit additive list when every metric has one clear row."""

    if operation == SUM_N_OPERATION:
        arity = arity or _explicit_sum_operand_count(question)
        if arity is None or not 2 <= arity <= SUM_N_MAX_OPERANDS:
            return None
    elif operation in {"SUM3", "SUM4"}:
        arity = OPERATION_ARITY[operation]
    else:
        return None
    assert arity is not None
    parts = _explicit_multi_operand_parts(question, arity)
    if parts is None:
        return None

    requested_years = set(_requested_years(question))
    pool = set(range(len(candidates)))
    selected: list[int] = []
    for part in parts:
        compact_part = re.sub(r"[^a-z0-9]+", "", normalize_text(part))
        exact = {
            index
            for index in pool
            if compact_part
            and re.sub(
                r"[^a-z0-9]+",
                "",
                normalize_text(_structured_candidate_row_label(candidates[index])),
            )
            == compact_part
        }
        allowed = exact or _metric_compatible_indices(part, candidates, pool)
        if requested_years and allowed:
            period_allowed = {
                index
                for index in allowed
                if _candidate_period_years(candidates[index], question)
                & requested_years
            }
            if period_allowed:
                allowed = period_allowed
        allowed = _narrow_unary_period_by_row_order(
            question, candidates, allowed
        )
        allowed = _prefer_reporting_entity_indices(
            question, candidates, allowed
        )
        allowed -= set(selected)
        if len(allowed) != 1:
            return None
        selected.append(next(iter(allowed)))
        pool -= set(selected)
    return tuple(sorted(selected))


_ENTITY_MATCH_STOPWORDS = {
    "bao",
    "cao",
    "co",
    "cong",
    "dich",
    "doan",
    "don",
    "dong",
    "nam",
    "phan",
    "san",
    "tai",
    "tap",
    "thuong",
    "tnhh",
    "tong",
    "ty",
    "vi",
    "vu",
    "xuat",
}


def _column_scope_compatible_indices(question, candidates, allowed):
    from .symbolic_verifier import _column_scope_constraint_state
    plan = compile_question_plan(question)
    compatible = {i for i in allowed
                  if _column_scope_constraint_state(plan, candidates[i]) is not VerificationState.INVALID}
    # With no compatible evidence, preserve a diagnostic prediction. The same
    # contradiction is required and rejected by the verifier, not published.
    return compatible or allowed


def _prefer_reporting_entity_indices(
    question: str,
    candidates: Sequence[NumericCandidate],
    allowed: set[int],
) -> set[int]:
    """Prefer the report owner unless the question names an embedded entity."""
    allowed = _column_scope_compatible_indices(question, candidates, allowed)
    embedded = {
        index
        for index in allowed
        if str(candidates[index].scope).strip().lower().startswith("entity:")
    }
    reporting = allowed - embedded
    if not embedded or not reporting:
        return allowed

    question_tokens = set(normalize_text(question).split())
    for index in embedded:
        candidate = candidates[index]
        identity_tokens = set(
            normalize_text(f"{candidate.scope} {candidate.section}").split()
        )
        meaningful = {
            token
            for token in identity_tokens
            if len(token) >= 3 and token not in _ENTITY_MATCH_STOPWORDS
        }
        if question_tokens & meaningful:
            return allowed
    return reporting


def _metric_modifier_conflicts(question: str, row_label: str) -> bool:
    """Reject opposite financial qualifiers before fuzzy row scoring."""

    if _is_ratio_metric(row_label) and not _is_ratio_metric(question):
        return True
    from .metric_semantics import normalize_metric_text
    expected = normalize_metric_text(_question_focus_text(question))
    observed = normalize_metric_text(row_label)
    if (
        "sau thue" in expected
        and "truoc thue" in observed
        or "truoc thue" in expected
        and "sau thue" in observed
    ):
        return True
    # Prevent a weak shared token such as ``thu`` from making revenue look like
    # profit. A shortened row like ``công ty mẹ`` remains eligible because it
    # does not assert a conflicting family.
    families = (
        "loi nhuan",
        "doanh thu",
        "tai san",
        "no phai tra",
        "chi phi",
        "tien",
    )
    expected_families = {item for item in families if item in expected}
    observed_families = {item for item in families if item in observed}
    return bool(
        expected_families
        and observed_families
        and expected_families.isdisjoint(observed_families)
    )


def _metric_anchor_compatible(expected_text: str, row_label: str) -> bool:
    """Require explicit financial anchors before applying a hard mask.

    Relative fuzzy scores are useful for OCR damage, but they are unsafe by
    themselves: one weak shared word can still be the best match in a retrieved
    candidate pool.  A hard mask also renormalizes probabilities and can turn
    that weak match into an apparently high-confidence answer.  These compact
    anchors therefore act as an absolute admission check.  They deliberately
    cover broad accounting concepts rather than issuer-specific wording.
    """

    from .metric_semantics import normalize_metric_text
    expected = normalize_metric_text(expected_text)
    observed = normalize_metric_text(row_label)
    expected_compact = re.sub(r"[^a-z0-9]+", "", expected)
    observed_compact = re.sub(r"[^a-z0-9]+", "", observed)

    # A question asking for a disclosed coefficient must not be grounded to
    # one of its component rows merely because that row shares words such as
    # ``no`` or ``tai san``. DIVIDE remains a separate allowed hypothesis.
    if "he so" in expected and "he so" not in observed:
        return False
    for debt_phrase in ("no ngan han", "no dai han"):
        if debt_phrase in expected and debt_phrase not in observed:
            return False
    if expected.strip() == "no" and not re.search(r"\bno\b", observed):
        return False

    # A parent-company qualifier is more specific than a surrounding profit
    # family. Native statement rows are often shortened to just ``Công ty mẹ``.
    if "congtyme" in expected_compact:
        return "congtyme" in observed_compact

    if "nophaitra" in expected_compact:
        debt_aliases = ("nophaitra", "nphaitra", "nphitra")
        if not any(alias in observed_compact for alias in debt_aliases):
            return False

    family_anchors = (
        ("loi nhuan", ("loinhuan",)),
        ("doanh thu", ("doanhthu",)),
        ("tai san", ("taisan", "taisn")),
        ("chi phi", ("chiphi",)),
        ("tien", ("tien",)),
    )
    for phrase, aliases in family_anchors:
        if phrase not in expected:
            continue
        if not any(alias in observed_compact for alias in aliases):
            return False
        # Distinguish a requested statement total from arbitrary asset rows.
        if phrase == "tai san" and "tong" in expected and "tong" not in observed:
            return False

    return True


def _prefer_requested_context_indices(
    question: str, candidates: Sequence[NumericCandidate], allowed: set[int],
) -> set[int]:
    """Prefer source-supported intent before neural ranking, without proof promotion."""
    allowed = _column_scope_compatible_indices(question, candidates, allowed)
    plan = compile_question_plan(question)
    if plan.requested_value_column:
        matching = {i for i in allowed if normalize_text(candidates[i].column_label) == plan.requested_value_column}
        if matching:
            allowed = matching
    roles = {role.lower() for role in plan.column_roles}
    if len(roles) > 1:
        roles = set()
    cells = evidence_cells(candidates)
    compatible = {
        index for index in allowed
        if not (roles and (cells[index].column_role_conflict
                           or cells[index].column_role and cells[index].column_role not in roles))
        and not (plan.requested_section and cells[index].source_section_path
                 and plan.requested_section not in cells[index].source_section_path)
    }
    from .entity_ownership import query_owner_state
    entity_compatible = {i for i in compatible if query_owner_state(plan, cells[i]) not in {"INVALID", "UNKNOWN"}}
    if entity_compatible:
        compatible = entity_compatible
    if plan.entity_mode == "named":
        proven = {i for i in compatible if query_owner_state(plan, cells[i]) == "VALID"}
        if proven:
            compatible = proven
    from .symbolic_verifier import _query_accounting_scope_state
    if plan.requested_accounting_scope:
        proven={i for i in compatible if _query_accounting_scope_state(plan,cells[i]) is VerificationState.VALID}
        if proven:return proven
    return compatible or allowed


def _combine_unary_grounding_constraints(
    metric_allowed: set[int], period_allowed: set[int]
) -> tuple[set[int], bool, bool]:
    """Combine SELECT masks without dropping a strong OCR metric on conflict.

    Relative headers such as ``Số cuối năm`` may not resolve to an absolute
    requested year when other retrieved pages do contain explicit years.  In
    that case the two masks are both non-empty but disjoint.  Falling back to
    unconstrained neural selection is less safe than retaining the explicit
    metric named by the question.
    """

    if metric_allowed and period_allowed:
        intersection = metric_allowed & period_allowed
        if intersection:
            return intersection, True, True
        return set(metric_allowed), True, False
    if metric_allowed:
        return set(metric_allowed), True, False
    if period_allowed:
        return set(period_allowed), False, True
    return set(), False, False


def _narrow_unary_period_by_row_order(
    question: str,
    candidates: Sequence[NumericCandidate],
    allowed: set[int],
) -> set[int]:
    """Resolve a two-value native row when its column header was separated.

    The fallback is intentionally narrow. It only compares values on the exact
    same source row and requires explicit current/prior wording. Absolute years
    must come from candidate period metadata; filenames are never evidence.
    """

    if len(allowed) < 2:
        return set(allowed)
    normalized = f" {normalize_text(question)} "
    desired_role = ""
    if any(marker in normalized for marker in (" cuoi nam ", " nam nay ", " 31 12 ")):
        desired_role = "current"
    elif any(marker in normalized for marker in (" dau nam ", " nam truoc ", " 01 01 ")):
        desired_role = "prior"
    if not desired_role:
        return set(allowed)

    groups: dict[tuple[str, int | None, int, str], list[int]] = {}
    for index in allowed:
        candidate = candidates[index]
        key = (
            candidate.source_name,
            candidate.page,
            candidate.line_number,
            normalize_text(_structured_candidate_row_label(candidate)),
        )
        groups.setdefault(key, []).append(index)

    narrowed: set[int] = set()
    changed = False
    for indices in groups.values():
        role_matches = [
            index
            for index in indices
            if _candidate_period_role(candidates[index]) == desired_role
        ]
        if len(role_matches) == 1:
            narrowed.add(role_matches[0])
            changed = True
            continue
        positions = {candidates[index].char_start for index in indices}
        if len(indices) < 2 or len(positions) < 2:
            narrowed.update(indices)
            continue
        chooser = min if desired_role == "current" else max
        narrowed.add(chooser(indices, key=lambda index: candidates[index].char_start))
        changed = True
    return narrowed if changed else set(allowed)


def _prefer_requested_period_indices(
    question: str,
    candidates: Sequence[NumericCandidate],
    allowed: set[int],
) -> set[int]:
    """Prefer explicit requested-period evidence without guessing missing metadata.

    A table often repeats the same row label in absolute-value columns and in
    a trailing change/percentage column.  Missing period metadata remains
    admissible when it is the only evidence available, but it must not outrank
    a role-equivalent cell whose period explicitly matches the question.
    """

    requested = set(_requested_years(question))
    if not requested or not allowed:
        return set(allowed)
    matching = {
        index
        for index in allowed
        if _candidate_period_years(candidates[index], question) & requested
    }
    return matching or set(allowed)


def _role_compatible_pair_indices(
    question: str,
    operation: str,
    candidates: Sequence[NumericCandidate],
    pool: set[int],
) -> tuple[set[int], set[int]] | None:
    if not pool:
        return None
    # Apply fuzzy role masking only to rows emitted by real document ingestion.
    # Training/synthetic contexts keep the learned pair scorer unchanged.
    structured_pool = {
        index
        for index in pool
        if _is_groundable_document_candidate(candidates[index])
    }
    if len(structured_pool) < 2:
        return None
    plan = compile_question_plan(question)
    role_parts = _operand_role_parts(question, operation)
    if operation == "ADD" and plan.addition_role_hypotheses:
        if len(plan.addition_role_hypotheses) == 1:
            role_parts = plan.addition_role_hypotheses[0]
        else:
            row_identities = {
                _role_identity(_structured_candidate_row_label(candidates[index]))
                for index in structured_pool
            }
            viable = [
                hypothesis
                for hypothesis in plan.addition_role_hypotheses
                if _role_identity(hypothesis[0]) in row_identities
                and _role_identity(hypothesis[1]) in row_identities
            ]
            role_parts = viable[0] if len(viable) == 1 else None
    if role_parts is None:
        return None
    roles = tuple(_alignment_terms(part) for part in role_parts)
    if not all(roles):
        return None
    left_terms, right_terms = roles
    left_text, right_text = role_parts
    left_identity = _role_identity(left_text)
    right_identity = _role_identity(right_text)
    row_terms = {
        index: _alignment_terms(_candidate_row_label(candidates[index]))
        for index in structured_pool
    }
    exact_left = {
        index
        for index in structured_pool
        if left_identity
        and _role_identity(_structured_candidate_row_label(candidates[index]))
        == left_identity
    }
    exact_right = {
        index
        for index in structured_pool
        if right_identity
        and _role_identity(_structured_candidate_row_label(candidates[index]))
        == right_identity
    }
    left_scores = {
        index: _role_match_score(
            left_text, left_terms, _candidate_row_label(candidates[index]), terms
        )
        for index, terms in row_terms.items()
        if _metric_anchor_compatible(
            left_text, _structured_candidate_row_label(candidates[index])
        )
    }
    right_scores = {
        index: _role_match_score(
            right_text, right_terms, _candidate_row_label(candidates[index]), terms
        )
        for index, terms in row_terms.items()
        if _metric_anchor_compatible(
            right_text, _structured_candidate_row_label(candidates[index])
        )
    }
    best_left = max(left_scores.values(), default=0.0)
    best_right = max(right_scores.values(), default=0.0)
    if best_left < 1.2 or best_right < 1.2:
        return None
    left_allowed = exact_left or {
        index for index, score in left_scores.items() if score >= best_left - 0.30
    }
    right_allowed = exact_right or {
        index for index, score in right_scores.items() if score >= best_right - 0.30
    }
    left_allowed = _prefer_requested_period_indices(
        question, candidates, left_allowed
    )
    right_allowed = _prefer_requested_period_indices(
        question, candidates, right_allowed
    )
    left_allowed = _narrow_unary_period_by_row_order(
        question, candidates, left_allowed
    )
    right_allowed = _narrow_unary_period_by_row_order(
        question, candidates, right_allowed
    )
    if not left_allowed or not right_allowed:
        return None
    return left_allowed, right_allowed


def _role_identity(text: str) -> str:
    normalized = normalize_text(text)
    normalized = re.sub(
        r"^(?:[ivxlcdm]+|\d+)[.)\s-]+", "", normalized
    )
    return re.sub(r"[^a-z0-9]+", "", normalized)


def _role_match_score(
    expected_text: str,
    expected_terms: set[str],
    row_label: str,
    row_terms: set[str],
) -> float:
    term_score = _fuzzy_term_score(expected_terms, row_terms)
    density_score = term_score / math.sqrt(max(1, len(row_terms)))
    expected_compact = "".join(
        token
        for token in re.findall(r"[a-z0-9]+", normalize_text(expected_text))
        if token not in _OPERAND_ALIGNMENT_STOPWORDS and not token.isdigit()
    )
    observed_compact = "".join(
        token
        for token in re.findall(r"[a-z0-9]+", normalize_text(row_label))
        if token not in _OPERAND_ALIGNMENT_STOPWORDS and not token.isdigit()
    )
    compact_similarity = (
        SequenceMatcher(None, expected_compact, observed_compact).ratio()
        if expected_compact and observed_compact
        else 0.0
    )
    return density_score + 2.0 * compact_similarity


def _temporal_role_compatible_indices(
    question: str,
    operation: str,
    candidates: Sequence[NumericCandidate],
    pool: set[int],
) -> set[int] | None:
    """Ground temporal arithmetic to the metric named by the question.

    This is deliberately limited to structured document-ingestion candidates.
    The learned pair scorer remains unchanged for train/dev contexts, while
    native and scanned reports avoid comparing an unrelated row merely because
    its two values have plausible periods.
    """

    if operation not in {"SUBTRACT", "PERCENT_CHANGE"} or not pool:
        return None
    metric_focus = (
        compile_question_plan(question).metric_text
        or _question_focus_text(question)
    )
    structured_pool = {
        index
        for index in pool
        if _is_groundable_document_candidate(candidates[index])
        and _metric_anchor_compatible(
            metric_focus,
            _structured_candidate_row_label(candidates[index]),
        )
    }
    if len(structured_pool) < 2:
        return None
    target_terms = _alignment_terms(metric_focus)
    if not target_terms:
        return None
    scores = {
        index: _role_match_score(
            metric_focus,
            target_terms,
            _candidate_row_label(candidates[index]),
            _alignment_terms(_candidate_row_label(candidates[index])),
        )
        for index in structured_pool
    }
    best = max(scores.values(), default=0.0)
    if best < 1.5:
        return None
    allowed = {index for index, score in scores.items() if score >= best - 0.30}
    return allowed if len(allowed) >= 2 else None


def _grounding_gate_reasons(
    question: str,
    program: ReasoningProgram,
    candidates: Sequence[NumericCandidate],
    verification: VerificationResult | None = None,
    *,
    financial_identity_catalog_enabled: bool = False,
    program_dag: ReasoningDAGProgram | None = None,
) -> tuple[str, ...]:
    """Return structural reasons that make an otherwise confident program unsafe."""

    selected = [
        candidates[index]
        for index in program.operands
        if 0 <= index < len(candidates)
    ]
    if len(selected) != len(program.operands):
        return ("invalid_operand_pointer",)

    reasons: list[str] = list(program_binding_issues(program, candidates, program_dag))
    structured = any(
        candidate.row_label
        or candidate.column_label
        or candidate.period
        or candidate.row_type
        for candidate in selected
    )
    plan = compile_question_plan(question)
    if plan.abstention_reason:
        reasons.append(f"question_abstention:{plan.abstention_reason}")
    requested_years = set(_requested_years(question))
    selected_derived = any(
        candidate.derived_operation.strip() for candidate in selected
    )
    if (
        plan.interest_bearing_debt_share
        and not selected_derived
        and program_dag is None
    ):
        reasons.append("missing_required_multi_step_derivation")
    if selected_derived or program_dag is not None:
        try:
            dag = program_dag or materialize_program_dag(program, candidates)
            if dag is None:
                reasons.append("invalid_derived_program")
            else:
                dag_value = execute_program_dag(dag, candidates)
                flat_value = (
                    execute_program(program, candidates)
                    if program_dag is None
                    else dag_value
                )
                if program_dag is None and flat_value != dag_value:
                    reasons.append("derived_value_mismatch")
                if plan.interest_bearing_debt_share:
                    leaves = [candidates[index] for index in dag.leaf_operands]
                    if (
                        len(leaves) != 3
                        or dag.nodes[-1].operation != "SHARE_OF_TOTAL"
                    ):
                        reasons.append("invalid_share_of_total_dag")
                    if requested_years and any(
                        not (
                            _candidate_period_years(candidate, question)
                            & requested_years
                        )
                        for candidate in leaves
                    ):
                        reasons.append("multi_step_period_unproven")
                    sources = {
                        candidate.source_name
                        for candidate in leaves
                        if candidate.source_name
                    }
                    if len(sources) != 1:
                        reasons.append("multi_step_source_mismatch")
                    leaf_units = {
                        currency_unit_scale(candidate.unit)
                        for candidate in leaves
                        if candidate.unit
                    }
                    if None in leaf_units or len(leaf_units) != 1:
                        reasons.append("multi_step_unit_scale_mismatch")
        except (
            ProgramDataError,
            IndexError,
            InvalidOperation,
            ZeroDivisionError,
        ):
            reasons.append("invalid_derived_program")
    if requested_years and structured:
        selected_years = [
            _candidate_period_years(candidate, question) for candidate in selected
        ]
        covered_years = set().union(*selected_years)
        # Sparse OCR metadata is absence of evidence, not evidence of a wrong
        # period. Reject only when every selected operand carries a period and
        # those explicit periods contradict the request.
        if all(selected_years) and not requested_years.issubset(covered_years):
            reasons.append("missing_requested_period")
        if (
            len(program.operands) >= 2
            and len(requested_years) == 1
            and all(selected_years)
            and any(not (years & requested_years) for years in selected_years)
        ):
            reasons.append("operand_outside_requested_period")
        if any(
            (candidate.column_label or candidate.period)
            and not candidate.row_label
            for candidate in selected
        ):
            reasons.append("missing_structured_row")

    if plan.column_roles:
        selected_cells = evidence_cells(selected)
        if any(cell.column_role_conflict for cell in selected_cells):
            reasons.append("column_role_conflict")
        elif any(not cell.column_role for cell in selected_cells):
            reasons.append("missing_requested_column_role")
        else:
            observed_roles = tuple(cell.column_role for cell in selected_cells)
            required_roles = tuple(role.lower() for role in plan.column_roles)
            direct_match = observed_roles == required_roles
            swapped_match = (
                program.operation in COMMUTATIVE
                and tuple(reversed(observed_roles)) == required_roles
            )
            if not direct_match and not swapped_match:
                reasons.append("column_role_mismatch")

    metadata_labels = {"ma so", "maso", "thuyet minh", "thuyetminh", "stt"}
    for candidate in selected:
        row = normalize_text(candidate.row_label)
        column = normalize_text(candidate.column_label)
        row_type = normalize_text(candidate.row_type)
        if (
            row_type == "metadata"
            or column in metadata_labels
            or any(column.startswith(label + " ") for label in metadata_labels)
            or bool(re.fullmatch(r"\d+(?:[.,]\d+)?", row))
        ):
            reasons.append("metadata_operand")
            break

    comparable_operations = {
        "ADD",
        "SUBTRACT",
        "RATIO_PERCENT",
        "PERCENT_CHANGE",
        "AVERAGE",
        "SUM3",
        "SUM4",
    }
    units = {candidate.unit for candidate in selected if candidate.unit}
    selected_unit_keys = {
        cell.unit_key for cell in evidence_cells(selected) if cell.unit_key
    }
    explicitly_convertible_scales = bool(
        len(selected_unit_keys) == 1
        and selected_unit_keys
        <= {"currency", "currency_per_share"}
        and units
        and all(currency_unit_scale(unit) is not None for unit in units)
    )
    if (
        program.operation in comparable_operations
        and len(units) > 1
        and not explicitly_convertible_scales
    ):
        reasons.append("incompatible_operand_units")
    identity_operations = (
        {
            identity.operation for identity in _plan_financial_identities(plan)
        }
        if financial_identity_catalog_enabled
        else set()
    )
    if (
        program.operation == "ADD"
        and not _question_requests_addition(normalize_text(question))
        and program.operation not in identity_operations
        and any(
            _is_groundable_document_candidate(candidate)
            for candidate in selected
        )
    ):
        reasons.append("missing_explicit_addition_intent")

    pool = set(range(len(candidates)))
    if len(program.operands) >= 2:
        role_compatible = _role_compatible_pair_indices(
            question, program.operation, candidates, pool
        )
        if (
            role_compatible is None
            and (
                len(plan.operand_roles) == 2
                or program.operation == "ADD"
                and bool(plan.addition_role_hypotheses)
            )
            and any(
                _is_groundable_document_candidate(candidate)
                for candidate in candidates
            )
        ):
            # Never let a neural fallback or confidence fusion turn an
            # arbitrary pair into an answer when the question explicitly
            # names two financial roles. Missing evidence must remain ABSTAIN.
            reasons.append("missing_requested_operand_roles")
        if role_compatible is not None:
            left_allowed, right_allowed = role_compatible
            direct_match = (
                program.operands[0] in left_allowed
                and program.operands[1] in right_allowed
            )
            swapped_match = (
                program.operation in COMMUTATIVE
                and program.operands[1] in left_allowed
                and program.operands[0] in right_allowed
            )
            if not direct_match and not swapped_match:
                reasons.append("operand_role_mismatch")
            scoped_left = _prefer_reporting_entity_indices(
                question, candidates, left_allowed
            )
            scoped_right = _prefer_reporting_entity_indices(
                question, candidates, right_allowed
            )
            if scoped_left != left_allowed or scoped_right != right_allowed:
                scoped_direct = (
                    program.operands[0] in scoped_left
                    and program.operands[1] in scoped_right
                )
                scoped_swapped = (
                    program.operation in COMMUTATIVE
                    and program.operands[1] in scoped_left
                    and program.operands[0] in scoped_right
                )
                if not scoped_direct and not scoped_swapped:
                    reasons.append("reporting_entity_mismatch")

        # A same-period comparison can explicitly name two different balance
        # sheet roles (for example, total assets versus total sources).  In
        # that case the role pair is the grounding contract; applying the
        # temporal same-metric mask would incorrectly force two periods of
        # only the first role.
        explicit_same_period_roles = (
            (
                len(plan.operand_roles) == 2
                or program.operation == "ADD"
                and bool(plan.addition_role_hypotheses)
            )
            and len(requested_years) <= 1
        )
        temporal_allowed = (
            None
            if explicit_same_period_roles
            else _temporal_role_compatible_indices(
                question, program.operation, candidates, pool
            )
        )
        if temporal_allowed is not None:
            if any(
                index not in temporal_allowed for index in program.operands[:2]
            ):
                reasons.append("temporal_metric_mismatch")
            scoped_temporal = _prefer_reporting_entity_indices(
                question, candidates, temporal_allowed
            )
            if scoped_temporal != temporal_allowed and any(
                index not in scoped_temporal for index in program.operands[:2]
            ):
                reasons.append("reporting_entity_mismatch")

        # A temporal change without two explicitly named metric roles must use
        # two periods of the same accounting row.  This catches unsafe pairs
        # such as ``Nợ dài hạn`` with ``Tài sản dở dang dài hạn`` even when
        # sparse native-PDF metadata prevents the temporal candidate mask from
        # finding two eligible values.
        if (
            program.operation in {"SUBTRACT", "PERCENT_CHANGE"}
            and len(requested_years) >= 2
            and role_compatible is None
        ):
            from .metric_semantics import identity as metric_identity
            identities = {
                metric_identity(_structured_candidate_row_label(candidates[index]))
                for index in program.operands[:2]
                if _candidate_metric_identity(candidates[index])
            }
            if len(identities) > 1:
                reasons.append("temporal_metric_mismatch")

    if verification is None:
        verification = verify_program(
            plan,
            program,
            build_evidence_graph(candidates),
            financial_identity_catalog_enabled=(
                financial_identity_catalog_enabled
            ),
        )
    reasons.extend(verification.reasons)
    if any(
        obligation.name == "column_role_unique"
        and obligation.applicability == "required"
        and obligation.state is VerificationState.UNKNOWN
        for obligation in verification.proof_obligations
    ):
        reasons.append("missing_column_role_inventory")

    return tuple(dict.fromkeys(reasons))


def _compile_symbolic_query_spec(
    question: str,
    candidates: Sequence[NumericCandidate],
) -> SymbolicQuerySpec:
    """Compile report-independent constraints from a finance question."""

    plan = compile_question_plan(question)
    normalized_question = f" {normalize_text(question)} "
    anchored_labels = {
        normalize_text(_structured_candidate_row_label(candidate)).strip()
        for candidate in candidates
        if _is_groundable_document_candidate(candidate)
        and len(
            _alignment_terms(_structured_candidate_row_label(candidate))
        )
        >= 2
        and (
            f" {normalize_text(_structured_candidate_row_label(candidate)).strip()} "
            in normalized_question
        )
    }
    maximal_anchors = {
        label
        for label in anchored_labels
        if not any(
            label != other and f" {label} " in f" {other} "
            for other in anchored_labels
        )
    }
    anchored_binary_add = (
        _question_requests_addition(normalize_text(question))
        and len(maximal_anchors) == 2
    )
    constrained = {
        _apply_symbolic_operation_constraint(question, operation)
        for operation in OPERATIONS
    }
    required_operation = (
        "ADD"
        if anchored_binary_add
        else next(iter(constrained))
        if len(constrained) == 1
        else None
    )
    if required_operation is None:
        normalized = normalize_text(question)
        explicit_sum = _explicit_sum_operation(question, normalized)
        if explicit_sum is not None:
            required_operation = explicit_sum
        elif (
            _question_requests_addition(normalized)
            and _operand_role_parts(question, "ADD") is not None
        ):
            required_operation = "ADD"
    # A pair of exact document-row anchors is stronger than punctuation in a
    # row label. Keep the legacy binary alias here; V14 search normalizes it
    # to SUM_N only after the model mode is known.
    allowed_operations = (
        ("ADD",) if anchored_binary_add else plan.operation_candidates
    )
    if not allowed_operations and (plan.operation or required_operation):
        allowed_operations = (plan.operation or required_operation,)
    return SymbolicQuerySpec(
        focus_text=plan.metric_text or _question_focus_text(question),
        requested_years=tuple(sorted(plan.requested_years, key=int)),
        required_operation=(
            allowed_operations[0] if len(allowed_operations) == 1 else None
        ),
        has_structured_evidence=any(
            candidate.row_label
            or candidate.column_label
            or candidate.period
            or candidate.row_type
            for candidate in candidates
        ),
        allowed_operations=tuple(allowed_operations),
    )


def _candidate_metric_identity(candidate: NumericCandidate) -> str:
    return re.sub(
        r"^(?:\d+[.)]?\s*)+",
        "",
        normalize_text(_structured_candidate_row_label(candidate)),
    )


def _program_search_rejection_reasons(
    spec: SymbolicQuerySpec,
    question: str,
    program: ReasoningProgram,
    candidates: Sequence[NumericCandidate],
    verification: VerificationResult | None = None,
    *,
    financial_identity_catalog_enabled: bool = False,
    program_dag: ReasoningDAGProgram | None = None,
) -> tuple[str, ...]:
    reasons: list[str] = list(program_binding_issues(program, candidates, program_dag))
    plan = compile_question_plan(question)
    identity_operations = (
        {
            identity.operation for identity in _plan_financial_identities(plan)
        }
        if financial_identity_catalog_enabled
        else set()
    )
    identity_operation = program.operation in identity_operations
    try:
        program.validate(len(candidates))
    except (ValueError, IndexError):
        return ("invalid_program",)

    if (
        financial_identity_catalog_enabled
        and identity_operations
        and _question_requires_financial_identity_computation(plan)
        and not identity_operation
    ):
        reasons.append("financial_identity_computation_required")

    if (
        spec.allowed_operations
        and program.operation not in spec.allowed_operations
        and not identity_operation
    ):
        reasons.append("operation_mismatch")
    elif (
        spec.required_operation is not None
        and program.operation != spec.required_operation
        and not identity_operation
    ):
        reasons.append("operation_mismatch")
    wording_operation = _apply_symbolic_operation_constraint(
        question, program.operation
    )
    wording_matches = wording_operation == program.operation or (
        program.operation == SUM_N_OPERATION
        and is_sum_operation(wording_operation)
    )
    if not wording_matches and not identity_operation:
        reasons.append("operation_wording")
    reasons.extend(
        _grounding_gate_reasons(
            question,
            program,
            candidates,
            verification,
            financial_identity_catalog_enabled=(
                financial_identity_catalog_enabled
            ),
            program_dag=program_dag,
        )
    )

    pool = set(range(len(candidates)))
    if len(program.operands) == 1:
        metric_allowed = _metric_compatible_indices(question, candidates, pool)
        period_allowed = _period_compatible_indices(
            question, program.operation, candidates, 1
        )
        unary_allowed, _, _ = _combine_unary_grounding_constraints(
            metric_allowed, period_allowed
        )
        unary_allowed = _narrow_unary_period_by_row_order(
            question, candidates, unary_allowed
        )
        unary_allowed = _prefer_requested_context_indices(question, candidates, unary_allowed)
        unary_allowed = _prefer_reporting_entity_indices(
            question, candidates, unary_allowed
        )
        if (
            plan.direct_metric_lookup
            and program.operation == "SELECT"
            and not metric_allowed
        ):
            reasons.append("missing_direct_metric_evidence")
        if unary_allowed and program.operands[0] not in unary_allowed:
            reasons.append("unary_grounding_mismatch")

    explicit_multi = _explicit_multi_operand_indices(
        question,
        program.operation,
        candidates,
        arity=len(program.operands),
    )
    if (
        explicit_multi is not None
        and tuple(sorted(program.operands)) != tuple(sorted(explicit_multi))
    ):
        reasons.append("explicit_operand_mismatch")

    if program.operation == "AVERAGE" and len(program.operands) == 2:
        normalized = f" {normalize_text(question)} "
        temporal_average = (
            len(spec.requested_years) >= 2
            or (
                any(
                    marker in normalized
                    for marker in (" cuoi nam ", " cuoi ky ")
                )
                and any(
                    marker in normalized
                    for marker in (" dau nam ", " dau ky ")
                )
            )
        )
        if temporal_average:
            row_identities = {
                _candidate_metric_identity(candidates[index])
                for index in program.operands
            }
            if len(row_identities) != 1:
                reasons.append("average_metric_mismatch")

    if verification is not None:
        for obligation in verification.proof_obligations:
            if obligation.applicability != "required":
                continue
            if obligation.state is VerificationState.UNKNOWN:
                reasons.append(f"required_proof_unknown:{obligation.name}")
            elif obligation.state is VerificationState.INVALID:
                reasons.append(f"required_proof_invalid:{obligation.name}")

    from .physical_coordinates import is_versioned_coordinate
    if any(is_versioned_coordinate(candidate) for candidate in candidates):
        mapping = getattr(verification, "mapping_assessment", None)
        resolution = getattr(mapping, "resolution", "UNKNOWN")
        if resolution != "UNIQUE":
            reasons.append("physical_mapping_conflict" if resolution == "CONFLICT"
                           else "unproven_physical_mapping")
    return tuple(dict.fromkeys(reasons))


def _program_metric_alignment_score(
    spec: SymbolicQuerySpec,
    question: str,
    program: ReasoningProgram,
    candidates: Sequence[NumericCandidate],
) -> float:
    selected = [candidates[index] for index in program.operands]
    labels = [_structured_candidate_row_label(candidate) for candidate in selected]
    role_parts = _operand_role_parts(question, program.operation)

    def match(expected: str, observed: str) -> float:
        return _role_match_score(
            expected,
            _alignment_terms(expected),
            observed,
            _alignment_terms(observed),
        )

    if role_parts is not None and len(selected) == 2:
        direct = (
            match(role_parts[0], labels[0])
            + match(role_parts[1], labels[1])
        ) / 2.0
        if program.operation in COMMUTATIVE:
            swapped = (
                match(role_parts[0], labels[1])
                + match(role_parts[1], labels[0])
            ) / 2.0
            return max(direct, swapped)
        return direct

    multi_parts = _explicit_multi_operand_parts(question, len(selected))
    if multi_parts is not None and len(selected) > 2:
        size = len(selected)
        weights = [
            [match(expected, observed) for observed in labels]
            for expected in multi_parts
        ]
        # Exact maximum-weight assignment in O(k^2 * 2^k), bounded by k<=8.
        # This replaces the previous k! permutation loop for SUM_N.
        states: dict[int, float] = {0: 0.0}
        for expected_index in range(size):
            next_states: dict[int, float] = {}
            for mask, score in states.items():
                for observed_index in range(size):
                    bit = 1 << observed_index
                    if mask & bit:
                        continue
                    next_mask = mask | bit
                    candidate_score = (
                        score + weights[expected_index][observed_index]
                    )
                    next_states[next_mask] = max(
                        next_states.get(next_mask, -math.inf),
                        candidate_score,
                    )
            states = next_states
        return states.get((1 << size) - 1, 0.0) / size

    return (
        sum(match(spec.focus_text, label) for label in labels) / len(labels)
        if labels
        else 0.0
    )


def _program_period_score(
    spec: SymbolicQuerySpec,
    question: str,
    program: ReasoningProgram,
    candidates: Sequence[NumericCandidate],
) -> float:
    requested = set(spec.requested_years)
    if not requested:
        normalized = f" {normalize_text(question)} "
        relative = {
            _candidate_period_role(candidates[index])
            for index in program.operands
            if _candidate_period_role(candidates[index])
        }
        expects_relative_pair = (
            any(marker in normalized for marker in (" cuoi nam ", " cuoi ky "))
            and any(marker in normalized for marker in (" dau nam ", " dau ky "))
        )
        return (
            1.0
            if expects_relative_pair and relative == {"current", "prior"}
            else 0.0
        )

    selected_years = [
        _candidate_period_years(candidates[index], question)
        for index in program.operands
    ]
    covered = set().union(*selected_years) if selected_years else set()
    coverage = len(covered & requested) / len(requested)
    operand_match = (
        sum(bool(years & requested) for years in selected_years)
        / len(selected_years)
        if selected_years
        else 0.0
    )
    return coverage + operand_match


def _program_structural_score(
    program: ReasoningProgram,
    candidates: Sequence[NumericCandidate],
) -> int:
    selected = [candidates[index] for index in program.operands]
    if len(selected) < 2:
        return 0
    score = 0
    structural_fields = ["source_name", "entity_id", "table_id", "section"]
    if program.operation != "MULTIPLY":
        structural_fields.insert(0, "unit")
    for field in structural_fields:
        values = [
            str(getattr(candidate, field) or "").strip()
            for candidate in selected
        ]
        if all(values) and len(set(values)) == 1:
            score += 1
    if program.operation in {"SUBTRACT", "PERCENT_CHANGE", "AVERAGE"}:
        rows = {
            normalize_text(_structured_candidate_row_label(candidate))
            for candidate in selected
        }
        if len(rows) == 1:
            score += 2
    return score


def _program_neural_evidence(
    output: DocumentHRMForward,
    row: int,
    program: ReasoningProgram,
) -> tuple[float, float, tuple[float, ...]]:
    operation_probabilities = torch.softmax(
        output.operation_logits[row], dim=-1
    )
    arity = len(program.operands)
    if program.operation == SUM_N_OPERATION:
        operation_id = OPERATION_TO_ID["ADD"]
        operation_probability = float(
            operation_probabilities[list(SUM_FAMILY_OPERATION_IDS)].sum().item()
        )
        if output.operand_count_logits is None:
            count_probability = 1e-12
        else:
            count_probabilities = torch.softmax(
                output.operand_count_logits[row], dim=-1
            )
            count_index = arity - 2
            count_probability = (
                float(count_probabilities[count_index].item())
                if 0 <= count_index < count_probabilities.shape[0]
                else 1e-12
            )
    else:
        operation_id = OPERATION_TO_ID[program.operation]
        operation_probability = float(
            operation_probabilities[operation_id].item()
        )
        count_probability = 1.0

    if (
        output.pair_logits is not None
        and arity == 2
        and program.operation != SUM_N_OPERATION
    ):
        pair_probabilities = torch.softmax(
            output.pair_logits[row, operation_id].flatten(), dim=-1
        )
        width = output.pair_logits.shape[-1]
        left, right = program.operands
        probability = float(
            pair_probabilities[left * width + right].item()
        )
        if program.operation in COMMUTATIVE:
            probability += float(
                pair_probabilities[right * width + left].item()
            )
        probability = min(1.0, probability)
        slot_probability = math.sqrt(max(probability, 1e-12))
        score = math.log(max(operation_probability, 1e-12)) + math.log(
            max(probability, 1e-12)
        )
        return operation_probability, score, (
            slot_probability,
            slot_probability,
        )

    pointer_probabilities: list[float] = []
    if program.operation == SUM_N_OPERATION:
        logits = (
            output.sum_membership_logits[row]
            if output.sum_membership_logits is not None
            else _commutative_pointer_membership_logits(
                output.pointer_logits[row], arity
            )
        )
        probabilities = torch.softmax(logits, dim=-1)
        pointer_probabilities = [
            float(probabilities[index].item()) for index in program.operands
        ]
    elif program.operation in COMMUTATIVE and arity > 2:
        logits = _commutative_pointer_membership_logits(
            output.pointer_logits[row], arity
        )
        probabilities = torch.softmax(logits, dim=-1)
        pointer_probabilities = [
            float(probabilities[index].item()) for index in program.operands
        ]
    else:
        for slot, index in enumerate(program.operands):
            probabilities = torch.softmax(
                _operation_pointer_logits(
                    output, row, slot, program.operation
                ),
                dim=-1,
            )
            pointer_probabilities.append(float(probabilities[index].item()))
    score = (
        math.log(max(operation_probability, 1e-12))
        + math.log(max(count_probability, 1e-12))
        + sum(
        math.log(max(probability, 1e-12))
        for probability in pointer_probabilities
        )
    )
    return operation_probability, score, tuple(pointer_probabilities)


def _symbolic_program_search_key(
    spec: SymbolicQuerySpec,
    question: str,
    program: ReasoningProgram,
    candidates: Sequence[NumericCandidate],
    neural_score: float,
    verification: VerificationResult | None = None,
    *,
    proof_guided: bool = True,
) -> tuple[float | int, ...]:
    # SUM_N is proof-ranked only after the verifier has established its
    # set-completeness/physical-uniqueness contract.
    use_verified_proof = (
        proof_guided
        and verification is not None
        and (
            len(program.operands) <= 2
            or (
                program.operation == SUM_N_OPERATION
                and verification.proof_contract_state
                is VerificationState.VALID
            )
        )
    )
    proof_key: tuple[float | int, ...] = (
        verification.rank_key if use_verified_proof else ()
    )
    return (*proof_key,
        round(
            _program_period_score(spec, question, program, candidates),
            3,
        ),
        _program_structural_score(program, candidates),
        round(
            _program_metric_alignment_score(
                spec, question, program, candidates
            ),
            3,
        ),
        neural_score,
    )


def _top_pair_programs(
    output: DocumentHRMForward,
    row: int,
    operation: str,
    candidates: Sequence[NumericCandidate],
    question: str,
    *,
    slot_pointer_union_enabled: bool = False,
) -> set[ReasoningProgram]:
    operation_id = OPERATION_TO_ID[operation]
    count = len(candidates)
    programs: set[ReasoningProgram] = set()
    if output.pair_logits is not None:
        pair_scores = output.pair_logits[row, operation_id]
        width = pair_scores.shape[-1]
        scores = pair_scores.flatten()
        take = min(64, int(scores.numel()))
        for flat_index in torch.topk(scores, k=take).indices.tolist():
            left, right = divmod(int(flat_index), width)
            if left >= count or right >= count or left == right:
                continue
            operands = (left, right)
            if operation in COMMUTATIVE:
                operands = tuple(sorted(operands))
            else:
                operands = _align_directed_operands(
                    question, operation, operands, candidates
                )
            programs.add(ReasoningProgram(operation, tuple(operands)))

    if output.pair_logits is None or slot_pointer_union_enabled:
        # The joint pair head and the two slot heads make different errors.
        # V12.4 takes their union; older runtimes retain the historical branch.
        take = min(10 if slot_pointer_union_enabled else 12, count)
        left_indices = torch.topk(
            _operation_pointer_logits(output, row, 0, operation),
            k=take,
        ).indices.tolist()
        right_indices = torch.topk(
            _operation_pointer_logits(output, row, 1, operation),
            k=take,
        ).indices.tolist()
        for left in left_indices:
            for right in right_indices:
                if left >= count or right >= count or left == right:
                    continue
                operands = (left, right)
                if operation in COMMUTATIVE:
                    operands = tuple(sorted(operands))
                else:
                    operands = _align_directed_operands(
                        question, operation, operands, candidates
                    )
                programs.add(ReasoningProgram(operation, tuple(operands)))

    pool = set(range(count))
    structural_pairs: set[tuple[int, int]] = set()
    role_compatible = _role_compatible_pair_indices(
        question, operation, candidates, pool
    )
    if role_compatible is not None:
        left_allowed, right_allowed = role_compatible
        structural_pairs.update(
            (left, right)
            for left in left_allowed
            for right in right_allowed
            if left != right
        )
    temporal_allowed = _temporal_role_compatible_indices(
        question, operation, candidates, pool
    )
    if temporal_allowed is not None:
        structural_pairs.update(permutations(temporal_allowed, 2))
    period_allowed = _period_compatible_indices(
        question, operation, candidates, 2
    )
    if period_allowed and len(period_allowed) <= 16:
        structural_pairs.update(permutations(period_allowed, 2))

    for left, right in structural_pairs:
        operands = (left, right)
        if operation in COMMUTATIVE:
            operands = tuple(sorted(operands))
        else:
            operands = _align_directed_operands(
                question, operation, operands, candidates
            )
        programs.add(ReasoningProgram(operation, tuple(operands)))
    return programs


def _operation_beam_for_search(
    output: DocumentHRMForward,
    row: int,
    baseline_operation: str,
    *,
    limit: int = 3,
) -> tuple[str, ...]:
    """Return a bounded neural operation beam without inventing new DSL ops."""

    probabilities = torch.softmax(output.operation_logits[row], dim=-1)
    sum_family_probability = float(
        probabilities[list(SUM_FAMILY_OPERATION_IDS)].sum().item()
    )
    take = min(max(1, limit), len(RERANKER_TRAIN_OPERATIONS))
    ranked_ids = torch.topk(probabilities, k=min(take + 2, len(OPERATIONS))).indices
    top_probability = float(probabilities[ranked_ids[0]].item())
    operations = [baseline_operation]
    if (
        output.operand_count_logits is not None
        and output.sum_membership_logits is not None
        and sum_family_probability >= max(0.02, top_probability * 0.08)
        and SUM_N_OPERATION not in operations
    ):
        operations.append(SUM_N_OPERATION)
    for operation_id in ranked_ids.tolist():
        operation = OPERATIONS[int(operation_id)]
        if (
            output.operand_count_logits is not None
            and is_sum_operation(operation)
        ):
            operation = SUM_N_OPERATION
            probability = sum_family_probability
        else:
            probability = float(probabilities[operation_id].item())
        if (
            operation not in RERANKER_TRAIN_OPERATIONS
            and operation != SUM_N_OPERATION
        ):
            continue
        if probability < max(0.02, top_probability * 0.08):
            continue
        if operation not in operations:
            operations.append(operation)
        if len(operations) >= take:
            break
    return tuple(operations)


def _sum_arity_for_search(
    output: DocumentHRMForward,
    row: int,
    question: str,
    *,
    fallback: int | None = None,
) -> int | None:
    explicit = _explicit_sum_operand_count(question)
    if explicit is not None:
        return explicit
    if output.operand_count_logits is not None:
        return int(output.operand_count_logits[row].argmax().item()) + 2
    return fallback


def _physically_unique_operand_set(
    operands: Sequence[int],
    candidates: Sequence[NumericCandidate],
) -> bool:
    duplicate_pairs = financial_same_physical_pairs(candidates)
    return not any(
        duplicate_pairs[left][right]
        for position, left in enumerate(operands)
        for right in operands[position + 1 :]
    )


def _candidate_programs_for_search(
    output: DocumentHRMForward,
    row: int,
    operation: str,
    question: str,
    candidates: Sequence[NumericCandidate],
    *,
    arity: int | None = None,
    candidate_recall_v12_4_enabled: bool = False,
) -> set[ReasoningProgram]:
    if operation == SUM_N_OPERATION:
        arity = arity or _sum_arity_for_search(
            output, row, question
        )
        if arity is None or not 2 <= arity <= SUM_N_MAX_OPERANDS:
            return set()
    else:
        arity = OPERATION_ARITY[operation]
    if arity == 1:
        return {
            ReasoningProgram(operation, (index,))
            for index in range(len(candidates))
        }
    if arity == 2 and operation != SUM_N_OPERATION:
        return _top_pair_programs(
            output,
            row,
            operation,
            candidates,
            question,
            slot_pointer_union_enabled=candidate_recall_v12_4_enabled,
        )

    explicit = _explicit_multi_operand_indices(
        question, operation, candidates, arity=arity
    )
    if explicit is not None:
        if (
            operation == SUM_N_OPERATION
            and not _physically_unique_operand_set(explicit, candidates)
        ):
            return set()
        return {
            ReasoningProgram(operation, tuple(sorted(explicit)))
        }

    programs: set[ReasoningProgram] = set()
    logits = (
        output.sum_membership_logits[row]
        if (
            operation == SUM_N_OPERATION
            and output.sum_membership_logits is not None
        )
        else _commutative_pointer_membership_logits(
            output.pointer_logits[row], arity
        )
    )
    pool_limit = min(12, max(arity + 2, 8))
    take = min(pool_limit, len(candidates))
    pool = set(torch.topk(logits, k=take).indices.tolist())
    period_allowed = _period_compatible_indices(
        question, operation, candidates, arity
    )
    if period_allowed and len(period_allowed) <= pool_limit + 2:
        pool.update(period_allowed)
    if len(pool) > pool_limit:
        pool = set(
            sorted(
                pool,
                key=lambda index: (
                    float(logits[index].item()),
                    candidates[index].relevance,
                ),
                reverse=True,
            )[:pool_limit]
        )
    ranked_combinations = sorted(
        (
            (
                sum(float(logits[index].item()) for index in indices),
                tuple(indices),
            )
            for indices in combinations(sorted(pool), arity)
            if (
                operation != SUM_N_OPERATION
                or _physically_unique_operand_set(indices, candidates)
            )
        ),
        key=lambda item: (item[0], tuple(-index for index in item[1])),
        reverse=True,
    )[:256]
    programs.update(
        ReasoningProgram(operation, indices)
        for _score, indices in ranked_combinations
    )
    return programs


def _beam_physical_key(
    index: int,
    candidates: Sequence[NumericCandidate],
) -> tuple[object, ...]:
    identity = financial_physical_identity(candidates[index])
    return identity if identity is not None else ("pointer", index)


def _conflicting_physical_indices(
    candidates: Sequence[NumericCandidate],
) -> set[int]:
    groups: dict[tuple[object, ...], list[int]] = {}
    for index, candidate in enumerate(candidates):
        identity = financial_physical_identity(candidate)
        if identity is not None:
            groups.setdefault(identity, []).append(index)
    conflicts: set[int] = set()
    for indices in groups.values():
        claims = {
            (
                candidates[index].value,
                normalize_text(candidates[index].unit),
                normalize_text(candidates[index].period),
                normalize_text(candidates[index].column_role),
            )
            for index in indices
        }
        if len(claims) > 1:
            conflicts.update(indices)
    return conflicts


def _program_physical_signature(
    program: ReasoningProgram,
    candidates: Sequence[NumericCandidate],
    *,
    conflict_indices: set[int] | None = None,
) -> tuple[object, ...]:
    conflicts = conflict_indices or set()
    keys = [
        (
            "conflict_pointer",
            index,
        )
        if index in conflicts
        else _beam_physical_key(index, candidates)
        for index in program.operands
    ]
    if program.operation in COMMUTATIVE or program.operation == SUM_N_OPERATION:
        keys = sorted(keys, key=repr)
    return (program.operation, len(program.operands), *keys)


def _slot_structural_beam_score(features: Sequence[float]) -> float:
    positive_indices = (1, 2, 3, 5, 7, 9, 11, 13, 14, 15, 24)
    mismatch_indices = (4, 6, 8, 10, 12, 25)
    positive = sum(
        float(features[index])
        for index in positive_indices
        if index < len(features)
    )
    mismatch = sum(
        float(features[index])
        for index in mismatch_indices
        if index < len(features)
    )
    relevance = float(features[22]) if len(features) > 22 else 0.0
    return positive + 0.25 * relevance - 3.0 * mismatch


def _round_robin_candidate_orders(
    orders: Sequence[Sequence[int]],
    *,
    limit: int,
) -> tuple[int, ...]:
    selected: list[int] = []
    seen: set[int] = set()
    positions = [0 for _ in orders]
    while len(selected) < limit:
        progressed = False
        for source, order in enumerate(orders):
            while positions[source] < len(order):
                index = int(order[positions[source]])
                positions[source] += 1
                if index in seen:
                    continue
                seen.add(index)
                selected.append(index)
                progressed = True
                break
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return tuple(selected)


def _rank_slot_candidate_pools(
    output: DocumentHRMForward,
    row: int,
    operation: str,
    question: str,
    candidates: Sequence[NumericCandidate],
    *,
    arity: int,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[dict[int, float], ...],
]:
    if not candidates:
        empty = tuple(() for _ in range(arity))
        return empty, empty, tuple({} for _ in range(arity))
    plan = compile_question_plan(question)
    matrices = build_financial_feature_matrices(
        plan,
        candidates,
        slot_count=arity,
    )
    conflict_indices = _conflicting_physical_indices(candidates)
    context_allowed=_prefer_requested_context_indices(question,candidates,set(range(len(candidates))))
    shared_membership = (
        output.sum_membership_logits[row]
        if (
            operation == SUM_N_OPERATION
            and output.sum_membership_logits is not None
        )
        else None
    )
    pools_10: list[tuple[int, ...]] = []
    pools_20: list[tuple[int, ...]] = []
    score_maps: list[dict[int, float]] = []
    count = len(candidates)
    for slot in range(arity):
        pointer_logits = _operation_pointer_logits(
            output,
            row,
            min(slot, output.pointer_logits.shape[1] - 1),
            "ADD" if operation == SUM_N_OPERATION else operation,
        )[:count]
        neural = pointer_logits.clone()
        if shared_membership is not None:
            neural = neural + shared_membership[:count]
        neural_order = torch.argsort(neural, descending=True).tolist()
        neural_rank = {
            int(index): 1.0 - rank / max(1, count - 1)
            for rank, index in enumerate(neural_order)
        }
        scores = {
            index: (
                neural_rank[index]
                + _slot_structural_beam_score(
                    matrices.slot_features[slot][index]
                )
            )
            for index in range(count)
            if index not in conflict_indices and index in context_allowed
        }
        combined_representatives: dict[tuple[object, ...], int] = {}
        for index in sorted(
            scores,
            key=lambda item: (scores[item], -item),
            reverse=True,
        ):
            key = _beam_physical_key(index, candidates)
            combined_representatives.setdefault(key, index)
        combined_order = tuple(combined_representatives.values())
        neural_physical_order: list[int] = []
        seen_neural_physical: set[tuple[object, ...]] = set()
        for index in neural_order:
            if index in conflict_indices or index not in context_allowed:
                continue
            key = _beam_physical_key(index, candidates)
            if key in seen_neural_physical:
                continue
            seen_neural_physical.add(key)
            neural_physical_order.append(combined_representatives[key])
        ranked = _round_robin_candidate_orders(
            (tuple(neural_physical_order), combined_order),
            limit=20,
        )
        pools_10.append(ranked[:10])
        pools_20.append(ranked)
        score_maps.append(scores)
    return tuple(pools_10), tuple(pools_20), tuple(score_maps)


def _incremental_set_relation_score(
    selected: Sequence[int],
    candidate_index: int,
    candidates: Sequence[NumericCandidate],
) -> float:
    candidate = candidates[candidate_index]
    score = 0.0
    for prior_index in selected:
        prior = candidates[prior_index]
        if candidate.table_id and candidate.table_id == prior.table_id:
            score += 0.20
        if candidate.section and candidate.section == prior.section:
            score += 0.10
        if candidate.unit and candidate.unit == prior.unit:
            score += 0.10
        explicit_parent_component = (
            prior_index in candidate.derived_operands
            or candidate_index in prior.derived_operands
        )
        if explicit_parent_component:
            score -= 2.0
    return score


def _candidate_programs_for_diverse_search(
    output: DocumentHRMForward,
    row: int,
    operation: str,
    question: str,
    candidates: Sequence[NumericCandidate],
    *,
    arity: int,
) -> CandidateGenerationResult:
    pools_10, pools_20, slot_scores = _rank_slot_candidate_pools(
        output,
        row,
        operation,
        question,
        candidates,
        arity=arity,
    )
    explicit = _explicit_multi_operand_indices(
        question,
        operation,
        candidates,
        arity=arity,
    )
    if explicit is not None:
        program = ReasoningProgram(
            operation,
            tuple(sorted(explicit))
            if operation in COMMUTATIVE or operation == SUM_N_OPERATION
            else tuple(explicit),
        )
        valid = _physically_unique_operand_set(
            program.operands, candidates
        )
        programs = (program,) if valid else ()
        return CandidateGenerationResult(
            programs=programs,
            program_scores=((program, 0.0),) if valid else (),
            slot_pools_at_10=pools_10,
            slot_pools_at_20=pools_20,
            generated_before_physical_dedup=len(programs),
            generated_after_physical_dedup=len(programs),
        )

    generation_pools = [
        pool[:8] if pool else tuple(range(min(8, len(candidates))))
        for pool in pools_20
    ]
    partials: list[tuple[tuple[int, ...], float]] = [((), 0.0)]
    conflict_indices = _conflicting_physical_indices(candidates)
    for slot in range(arity):
        expanded: dict[tuple[object, ...], tuple[tuple[int, ...], float]] = {}
        for selected, score in partials:
            selected_physical = {
                _beam_physical_key(index, candidates) for index in selected
            }
            for index in generation_pools[slot]:
                if index in selected or index in conflict_indices:
                    continue
                physical = _beam_physical_key(index, candidates)
                if physical in selected_physical:
                    continue
                next_selected = (*selected, index)
                next_score = (
                    score
                    + slot_scores[slot].get(index, -10.0)
                    + _incremental_set_relation_score(
                        selected, index, candidates
                    )
                )
                if operation in COMMUTATIVE or operation == SUM_N_OPERATION:
                    key = tuple(
                        sorted(
                            (
                                _beam_physical_key(item, candidates)
                                for item in next_selected
                            ),
                            key=repr,
                        )
                    )
                else:
                    key = tuple(
                        _beam_physical_key(item, candidates)
                        for item in next_selected
                    )
                current = expanded.get(key)
                if current is None or next_score > current[1]:
                    expanded[key] = (next_selected, next_score)
        partials = sorted(
            expanded.values(),
            key=lambda item: (item[1], tuple(-x for x in item[0])),
            reverse=True,
        )[:64]
        if not partials:
            break

    completed: list[tuple[ReasoningProgram, float]] = []
    for operands, score in partials[:256]:
        if len(operands) != arity:
            continue
        if operation in COMMUTATIVE or operation == SUM_N_OPERATION:
            operands = tuple(sorted(operands))
        completed.append((ReasoningProgram(operation, operands), score))
    before = len(completed)
    deduplicated: dict[tuple[object, ...], tuple[ReasoningProgram, float]] = {}
    for program, score in completed:
        signature = _program_physical_signature(
            program,
            candidates,
            conflict_indices=conflict_indices,
        )
        current = deduplicated.get(signature)
        if current is None or score > current[1]:
            deduplicated[signature] = (program, score)
    ranked = sorted(
        deduplicated.values(),
        key=lambda item: (item[1], tuple(-x for x in item[0].operands)),
        reverse=True,
    )[:256]
    return CandidateGenerationResult(
        programs=tuple(program for program, _score in ranked),
        program_scores=tuple(ranked),
        slot_pools_at_10=pools_10,
        slot_pools_at_20=pools_20,
        generated_before_physical_dedup=before,
        generated_after_physical_dedup=len(ranked),
    )


def _diverse_preverification_order(
    programs: Sequence[ReasoningProgram],
    program_sources: dict[ReasoningProgram, set[str]],
    program_scores: dict[ReasoningProgram, float],
    candidates: Sequence[NumericCandidate],
    baseline: ReasoningProgram,
) -> tuple[ReasoningProgram, ...]:
    conflict_indices = _conflicting_physical_indices(candidates)
    representatives: dict[tuple[object, ...], ReasoningProgram] = {}
    for program in programs:
        signature = _program_physical_signature(
            program,
            candidates,
            conflict_indices=conflict_indices,
        )
        current = representatives.get(signature)
        if current is None or (
            "typed_slot_synthesis" in program_sources.get(program, set()),
            program_scores.get(program, -math.inf),
            tuple(-index for index in program.operands),
        ) > (
            "typed_slot_synthesis" in program_sources.get(current, set()),
            program_scores.get(current, -math.inf),
            tuple(-index for index in current.operands),
        ):
            representatives[signature] = program
    deduplicated = list(dict.fromkeys(representatives.values()))
    buckets: dict[tuple[object, ...], list[ReasoningProgram]] = {}
    for program in deduplicated:
        selected = [candidates[index] for index in program.operands]
        tables = tuple(
            sorted(
                {
                    (candidate.table_id, candidate.section)
                    for candidate in selected
                }
            )
        )
        roles = tuple(
            sorted(
                {
                    normalize_text(candidate.column_role)
                    for candidate in selected
                    if candidate.column_role
                }
            )
        )
        sources = program_sources.get(program, set())
        primary_source = (
            "typed"
            if "typed_slot_synthesis" in sources
            else "neural"
            if any(source.startswith("neural") for source in sources)
            else "identity"
            if any(source.startswith("financial_identity:") for source in sources)
            else "baseline"
        )
        key = (
            program.operation,
            len(program.operands),
            tables,
            roles,
            primary_source,
        )
        buckets.setdefault(key, []).append(program)
    for bucket in buckets.values():
        bucket.sort(
            key=lambda program: (
                program_scores.get(program, -math.inf),
                tuple(-index for index in program.operands),
            ),
            reverse=True,
        )
    ordered: list[ReasoningProgram] = []
    reserved = [baseline]
    best_typed = max(
        (
            program
            for program in deduplicated
            if "typed_slot_synthesis" in program_sources.get(program, set())
        ),
        key=lambda program: program_scores.get(program, -math.inf),
        default=None,
    )
    if best_typed is not None:
        reserved.append(best_typed)
    for program in reserved:
        if program in deduplicated and program not in ordered:
            ordered.append(program)
            for bucket in buckets.values():
                if program in bucket:
                    bucket.remove(program)
    ordered_keys = sorted(
        buckets,
        key=lambda key: (
            program_scores.get(buckets[key][0], -math.inf)
            if buckets[key]
            else -math.inf
        ),
        reverse=True,
    )
    while True:
        progressed = False
        for key in ordered_keys:
            if not buckets[key]:
                continue
            ordered.append(buckets[key].pop(0))
            progressed = True
        if not progressed:
            break
    return tuple(ordered)


def _has_symbolic_program_anchors(
    spec: SymbolicQuerySpec,
    question: str,
    operation: str,
    candidates: Sequence[NumericCandidate],
    *,
    arity: int | None = None,
) -> bool:
    if operation == SUM_N_OPERATION:
        arity = arity or _explicit_sum_operand_count(question)
        if arity is None:
            return False
    else:
        arity = OPERATION_ARITY[operation]
    pool = set(range(len(candidates)))
    if arity == 1:
        return bool(
            _metric_compatible_indices(question, candidates, pool)
        )
    if arity == 2:
        if (
            _role_compatible_pair_indices(
                question, operation, candidates, pool
            )
            is not None
        ):
            return True
        if (
            _temporal_role_compatible_indices(
                question, operation, candidates, pool
            )
            is not None
        ):
            return True
        if operation == "AVERAGE":
            normalized = f" {normalize_text(question)} "
            has_period_pair = (
                len(spec.requested_years) >= 2
                or (
                    any(
                        marker in normalized
                        for marker in (" cuoi nam ", " cuoi ky ")
                    )
                    and any(
                        marker in normalized
                        for marker in (" dau nam ", " dau ky ")
                    )
                )
            )
            if not has_period_pair:
                return False
            if len(
                _metric_compatible_indices(question, candidates, pool)
            ) >= 2:
                return True
            focus_terms = _alignment_terms(spec.focus_text)
            row_groups: dict[str, list[NumericCandidate]] = {}
            for candidate in candidates:
                identity = _candidate_metric_identity(candidate)
                if identity:
                    row_groups.setdefault(identity, []).append(candidate)
            return any(
                {
                    _candidate_period_role(candidate)
                    for candidate in group
                }
                >= {"current", "prior"}
                and _fuzzy_term_score(
                    focus_terms,
                    _alignment_terms(identity),
                )
                >= 1.5
                for identity, group in row_groups.items()
            )
        return False
    return _explicit_multi_operand_parts(question, arity) is not None


def _program_reranker_feature_vector(
    spec: SymbolicQuerySpec,
    question: str,
    program: ReasoningProgram,
    candidates: Sequence[NumericCandidate],
    verification: VerificationResult,
    *,
    neural_score: float,
    neural_rank: int,
    total_programs: int,
    operation_probability: float,
    pointer_confidences: Sequence[float],
    legacy_rank: int,
    legacy_rank_score: float,
    legacy_winner: bool,
    feature_schema: str = PROGRAM_RERANKER_FEATURE_SCHEMA,
) -> tuple[float, ...]:
    """Build the versioned V11 feature vector from explicit evidence only."""

    constraint_values = verification.feature_vector
    constraints = verification.constraints
    count = max(1, len(constraints))
    valid_count = sum(
        item.state is VerificationState.VALID for item in constraints
    )
    unknown_count = sum(
        item.state is VerificationState.UNKNOWN for item in constraints
    )
    invalid_count = sum(
        item.state is VerificationState.INVALID for item in constraints
    )
    selected = [candidates[index] for index in program.operands]
    pointer_values = [max(float(value), 1e-12) for value in pointer_confidences]
    pointer_geometric_mean = (
        math.exp(sum(math.log(value) for value in pointer_values) / len(pointer_values))
        if pointer_values
        else 0.0
    )
    try:
        execute_program(program, candidates)
        execution_valid = 1.0
    except (ArithmeticError, ValueError):
        execution_valid = 0.0
    scopes = [normalize_text(item.scope) for item in selected]
    known_scopes = {scope for scope in scopes if scope}
    same_scope = (
        1.0
        if scopes and all(scopes) and len(known_scopes) == 1
        else -1.0
        if len(known_scopes) > 1
        else 0.0
    )
    relevances = [float(item.relevance) for item in selected]

    def constraint(name: str) -> float:
        return float(constraint_values.get(name, 0.0))

    support_state = (
        verification.training_support_state
        if feature_schema == PROGRAM_RERANKER_FEATURE_SCHEMA
        else verification.state
    )
    values = (
        float(neural_score),
        1.0 / max(1, neural_rank),
        1.0 - (max(1, neural_rank) - 1) / max(1, total_programs),
        float(legacy_rank_score),
        float(legacy_winner),
        1.0 / max(1, legacy_rank),
        float(operation_probability),
        pointer_geometric_mean,
        min(pointer_values) if pointer_values else 0.0,
        float(verification.proof_score),
        valid_count / count,
        unknown_count / count,
        invalid_count / count,
        float(support_state is VerificationState.VALID),
        float(support_state is VerificationState.UNKNOWN),
        constraint("operation_match"),
        constraint("metric_match"),
        constraint("period_coverage"),
        constraint("temporal_direction"),
        constraint("entity_match"),
        constraint("unit_match"),
        constraint("same_table"),
        constraint("same_row"),
        constraint("denominator_safe"),
        execution_valid,
        constraint("graph_same_metric"),
        constraint("graph_same_entity"),
        constraint("graph_same_table"),
        constraint("graph_same_row"),
        constraint("graph_same_unit"),
        constraint("graph_temporal_order"),
        float(
            _program_metric_alignment_score(
                spec, question, program, candidates
            )
        ),
        float(_program_period_score(spec, question, program, candidates)),
        float(_program_structural_score(program, candidates)),
        sum(relevances) / len(relevances) if relevances else 0.0,
        min(relevances) if relevances else 0.0,
        same_scope,
        (
            len(program.operands) / SUM_N_MAX_OPERANDS
            if program.operation == SUM_N_OPERATION
            else OPERATION_ARITY[program.operation] / MAX_OPERANDS
        ),
        unknown_count / count,
    )
    if len(values) != len(PROGRAM_RERANKER_FEATURES):
        raise DocumentHRMError("V11 reranker feature schema is inconsistent")
    return tuple(float(value) for value in values)


def _select_diverse_reranker_candidates(
    rows: Sequence[_SearchProgramCandidate],
    *,
    limit: int = 30,
) -> list[_SearchProgramCandidate]:
    """Round-robin neural candidates across operation/state/first-pointer beams."""

    eligible = [item for item in rows if not item.rejection_reasons]
    if len(eligible) <= limit:
        return eligible
    legacy = next((item for item in eligible if item.legacy_winner), None)
    buckets: dict[
        tuple[str, str, int], list[_SearchProgramCandidate]
    ] = {}
    for item in eligible:
        first_pointer = item.program.operands[0] if item.program.operands else -1
        key = (
            item.program.operation,
            item.verification.training_support_state.value,
            first_pointer,
        )
        buckets.setdefault(key, []).append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item.neural_score, reverse=True)
    selected: list[_SearchProgramCandidate] = []
    ordered_keys = sorted(
        buckets,
        key=lambda key: buckets[key][0].neural_score,
        reverse=True,
    )
    while len(selected) < limit:
        progressed = False
        for key in ordered_keys:
            bucket = buckets[key]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    if legacy is not None and legacy not in selected:
        selected[-1] = legacy
    selected.sort(key=lambda item: item.neural_rank)
    return selected


def _preserve_safe_residual_reference(
    legacy_reference_result: SymbolicProgramSearchResult | None,
    eligible_rows: Sequence[_SearchProgramCandidate],
    chosen: _SearchProgramCandidate,
    *,
    reranker_overridden: bool,
) -> tuple[_SearchProgramCandidate, bool]:
    """Keep the frozen reference unless the sidecar made a real override."""

    if legacy_reference_result is None or reranker_overridden:
        return chosen, False
    reference_row = next(
        (
            item
            for item in eligible_rows
            if item.program == legacy_reference_result.program
        ),
        None,
    )
    if reference_row is None:
        return chosen, False
    if (
        reference_row.program.operation == SUM_N_OPERATION
        and reference_row.verification.proof_contract_state
        is not VerificationState.VALID
    ):
        # SUM_N has no frozen legacy winner. Preserve it only when the current
        # verifier proves the complete set contract, never merely by recursion.
        return chosen, False
    return reference_row, chosen.program != reference_row.program


def _verified_confidence_floor(
    plan: QuestionPlan,
    program: ReasoningProgram,
    verification: VerificationResult,
    generation_sources: Sequence[str],
) -> float | None:
    """Fuse explicit symbolic proof into confidence without issuer rules.

    The frozen V8 thresholds were calibrated on neural confidence. A typed
    program with all intent-critical constraints explicitly VALID should not be
    rejected solely because its neural pointer rank was low. Optional layout
    metadata may remain UNKNOWN; any INVALID constraint still disables fusion.
    """

    if verification.proof_contract_state is not VerificationState.VALID:
        return None
    if "typed_slot_synthesis" not in generation_sources:
        return None
    if program.operation == "SELECT" and not plan.direct_metric_lookup:
        return None

    states = {item.name: item.state for item in verification.constraints}
    required = {
        "execution_valid",
        "operation_match",
        "operand_arity_intent",
        "denominator_safe",
    }
    if plan.requested_years:
        required.add("period_coverage")
    if plan.temporal_change:
        required.update(("metric_match", "temporal_direction"))
    elif program.operation in {"AVERAGE"} and len(plan.requested_years) >= 2:
        required.add("metric_match")
    if program.operation == "MULTIPLY" and plan.operand_roles:
        required.add("unit_algebra")
    if (
        any(
            source.startswith("financial_identity:")
            for source in generation_sources
        )
        and plan.operation is None
        and program.operation != "SELECT"
    ):
        required.add("financial_identity")
    if any(states.get(name) is not VerificationState.VALID for name in required):
        return None

    valid = sum(
        item.state is VerificationState.VALID
        for item in verification.constraints
    )
    completeness = valid / max(1, len(verification.constraints))
    return min(0.95, 0.76 + 0.20 * completeness)


def _search_symbolic_programs(
    output: DocumentHRMForward,
    row: int,
    question: str,
    candidates: Sequence[NumericCandidate],
    baseline: ReasoningProgram,
    *,
    proof_guided: bool = True,
    graph_enabled: bool = True,
    typed_synthesis_enabled: bool = True,
    financial_identity_catalog_enabled: bool = False,
    candidate_recall_v12_4_enabled: bool = False,
    diverse_set_beam_enabled: bool = False,
    reranker: ProgramRerankerBundle | None = None,
) -> SymbolicProgramSearchResult | None:
    """Select the best verified program instead of accepting only neural top-1."""

    spec = _compile_symbolic_query_spec(question, candidates)
    if not spec.has_structured_evidence:
        return None
    sum_n_search = baseline.operation == SUM_N_OPERATION
    if sum_n_search:
        logical_allowed = tuple(
            dict.fromkeys(
                SUM_N_OPERATION if is_sum_operation(operation) else operation
                for operation in spec.allowed_operations
            )
        )
        logical_required = (
            SUM_N_OPERATION
            if (
                spec.required_operation is not None
                and is_sum_operation(spec.required_operation)
            )
            else spec.required_operation
        )
        spec = replace(
            spec,
            required_operation=logical_required,
            allowed_operations=logical_allowed,
        )
    legacy_reference_result: SymbolicProgramSearchResult | None = None
    legacy_verification_cache: dict[
        tuple[ReasoningProgram, bool],
        tuple[VerificationResult, tuple[str, ...]],
    ] = {}
    if candidate_recall_v12_4_enabled:
        # Candidate recall is an additive research layer. Preserve the exact
        # V12.3 winner whenever it already has an eligible proof; V12.4-only
        # programs are collected for audit/reranker training and may serve as
        # a runtime fallback only when V12.3 has no selectable result.
        legacy_reference_result = _search_symbolic_programs(
            output,
            row,
            question,
            candidates,
            baseline,
            proof_guided=proof_guided,
            graph_enabled=graph_enabled,
            typed_synthesis_enabled=typed_synthesis_enabled,
            financial_identity_catalog_enabled=(
                financial_identity_catalog_enabled
            ),
            candidate_recall_v12_4_enabled=False,
            diverse_set_beam_enabled=False,
            reranker=None,
        )
        if legacy_reference_result is not None:
            legacy_verification_cache = {
                (program, identity_required): (verification, reasons)
                for (
                    program,
                    identity_required,
                    verification,
                    reasons,
                ) in legacy_reference_result.verification_cache
            }
    graph = build_evidence_graph(candidates)
    plan = compile_question_plan(question)
    financial_identities = (
        _plan_financial_identities(plan)
        if financial_identity_catalog_enabled
        else ()
    )
    has_direct_identity_evidence = _has_direct_financial_identity_evidence(
        plan, graph, financial_identities
    )

    if spec.allowed_operations:
        operations = spec.allowed_operations
    elif spec.required_operation is not None:
        operations = (spec.required_operation,)
    else:
        # Operation top-1 is already strong, but a bounded Top-K recovers the
        # small set of questions where intent is implicit and the correct
        # operation is neural rank 2/3. An operation can only materialize a
        # free beam below when it also has a structural anchor.
        operations = (
            _operation_beam_for_search(output, row, baseline.operation)
            if candidate_recall_v12_4_enabled
            else (baseline.operation,)
        )
    if (
        typed_synthesis_enabled
        and financial_identities
        and not has_direct_identity_evidence
    ):
        operations = tuple(
            dict.fromkeys(
                (
                    *operations,
                    *(identity.operation for identity in financial_identities),
                )
            )
        )
    if has_direct_identity_evidence:
        # A disclosed target adds a safe SELECT hypothesis. It must never
        # delete the frozen neural baseline or another V12.1 hypothesis: an
        # alias match is supporting evidence, not proof that every alternative
        # operation is wrong.
        operations = tuple(dict.fromkeys((*operations, "SELECT")))
    if sum_n_search:
        # An identity catalog may serialize a two/three/four-term sum with a
        # legacy alias. Keep every additive hypothesis on the logical SUM_N
        # proof contract so an ADD proposal cannot bypass set completeness or
        # per-operand proof obligations.
        operations = tuple(
            dict.fromkeys(
                SUM_N_OPERATION if is_sum_operation(operation) else operation
                for operation in operations
            )
        )
    if baseline.operation not in operations:
        operations = (*operations, baseline.operation)
    operations = tuple(dict.fromkeys(operations))
    operation_arities: dict[str, int] = {}
    for operation in operations:
        if operation == SUM_N_OPERATION:
            arity = _sum_arity_for_search(
                output,
                row,
                question,
                fallback=len(baseline.operands),
            )
            if arity is None or not 2 <= arity <= SUM_N_MAX_OPERANDS:
                continue
            operation_arities[operation] = arity
        else:
            operation_arities[operation] = OPERATION_ARITY[operation]
    operations = tuple(
        operation for operation in operations if operation in operation_arities
    )
    operation_has_anchors = {
        operation: _has_symbolic_program_anchors(
            spec,
            question,
            operation,
            candidates,
            arity=operation_arities[operation],
        )
        for operation in operations
    }
    legacy_has_anchors = any(operation_has_anchors.values())
    explicit_operations = set(spec.allowed_operations)
    if spec.required_operation is not None:
        explicit_operations.add(spec.required_operation)
    identity_operations = {
        (
            SUM_N_OPERATION
            if sum_n_search and is_sum_operation(identity.operation)
            else identity.operation
        )
        for identity in financial_identities
    }
    synthesis_results: list[TypedSynthesisResult] = []
    if typed_synthesis_enabled:
        for operation in operations:
            if candidate_recall_v12_4_enabled and not (
                operation == baseline.operation
                or operation in explicit_operations
                or operation_has_anchors[operation]
                or operation in identity_operations
            ):
                continue
            arity = operation_arities[operation]
            role_texts: Sequence[str] | None = None
            if operation == SUM_N_OPERATION:
                role_texts = _explicit_multi_operand_parts(question, arity)
            elif arity == 2:
                role_texts = _operand_role_parts(question, operation)
            elif arity > 2:
                role_texts = _explicit_multi_operand_parts(question, arity)
            synthesis_results.append(
                synthesize_typed_programs(
                    plan,
                    operation,
                    graph,
                    focus_text=spec.focus_text,
                    operand_role_texts=role_texts,
                )
            )
        if not has_direct_identity_evidence:
            for identity in financial_identities:
                identity_operation = (
                    SUM_N_OPERATION
                    if sum_n_search and is_sum_operation(identity.operation)
                    else identity.operation
                )
                result = synthesize_typed_programs(
                    plan,
                    identity_operation,
                    graph,
                    focus_text=spec.focus_text,
                    operand_role_texts=identity.operand_texts,
                    operand_unit_families=tuple(
                        role.unit_family for role in identity.operands
                    ),
                )
                synthesis_results.append(
                    replace(
                        result,
                        template=f"financial_identity:{identity.identity_id}",
                    )
                )
    if (
        not candidate_recall_v12_4_enabled
        and not legacy_has_anchors
        and not any(result.proposals for result in synthesis_results)
    ):
        return None
    program_sources: dict[ReasoningProgram, set[str]] = {
        baseline: {"neural_baseline"}
    }
    generation_scores: dict[ReasoningProgram, float] = {}
    slot_pool_10: dict[int, list[int]] = {}
    slot_pool_20: dict[int, list[int]] = {}
    operation_slot_pool_10: dict[str, tuple[tuple[int, ...], ...]] = {}
    operation_slot_pool_20: dict[str, tuple[tuple[int, ...], ...]] = {}
    generated_before_physical_dedup = 0
    generated_after_physical_dedup = 0
    # Typed synthesis may recover a question for which the legacy decoder had
    # no reliable structural anchor. In that case it must not accidentally
    # unlock the entire free neural beam: only the neural baseline and typed
    # slot assignments are admissible. Otherwise an unrelated high-proof pair
    # can win merely because typed synthesis activated search at all.
    for operation in operations:
        if diverse_set_beam_enabled:
            # V14.1 always retrieves a bounded pool for the neural baseline or
            # an explicitly requested operation. Unrelated operation-beam
            # hypotheses still require a structural/identity anchor, and every
            # composed program remains subject to the unchanged verifier.
            use_operation_beam = (
                operation == baseline.operation
                or operation in explicit_operations
                or operation_has_anchors[operation]
                or operation in identity_operations
            )
        else:
            use_operation_beam = (
                operation_has_anchors[operation]
                if candidate_recall_v12_4_enabled
                else legacy_has_anchors
            )
        if use_operation_beam:
            if diverse_set_beam_enabled:
                generated = _candidate_programs_for_diverse_search(
                    output,
                    row,
                    operation,
                    question,
                    candidates,
                    arity=operation_arities[operation],
                )
                operation_programs = generated.programs
                generation_scores.update(dict(generated.program_scores))
                generated_before_physical_dedup += (
                    generated.generated_before_physical_dedup
                )
                generated_after_physical_dedup += (
                    generated.generated_after_physical_dedup
                )
                operation_slot_pool_10[operation] = (
                    generated.slot_pools_at_10
                )
                operation_slot_pool_20[operation] = (
                    generated.slot_pools_at_20
                )
                for slot, pool in enumerate(
                    generated.slot_pools_at_10
                ):
                    values = slot_pool_10.setdefault(slot, [])
                    values.extend(
                        index for index in pool if index not in values
                    )
                for slot, pool in enumerate(
                    generated.slot_pools_at_20
                ):
                    values = slot_pool_20.setdefault(slot, [])
                    values.extend(
                        index for index in pool if index not in values
                    )
            else:
                operation_programs = _candidate_programs_for_search(
                    output,
                    row,
                    operation,
                    question,
                    candidates,
                    arity=operation_arities[operation],
                    candidate_recall_v12_4_enabled=(
                        candidate_recall_v12_4_enabled
                    ),
                )
            for program_candidate in operation_programs:
                program_sources.setdefault(program_candidate, set()).add(
                    "diverse_set_beam_v1"
                    if diverse_set_beam_enabled
                    else "neural_operation_conditioned_beam"
                    if candidate_recall_v12_4_enabled
                    else "neural_structural_beam"
                )
    synthesized_programs: set[ReasoningProgram] = set()
    typed_proposals: dict[ReasoningProgram, TypedProgramProposal] = {}
    for result in synthesis_results:
        for proposal in result.proposals:
            proposal_program = (
                ReasoningProgram(
                    SUM_N_OPERATION,
                    tuple(sorted(proposal.program.operands)),
                )
                if (
                    result.operation == SUM_N_OPERATION
                    and proposal.program.operation != SUM_N_OPERATION
                )
                else proposal.program
            )
            synthesized_programs.add(proposal_program)
            current = typed_proposals.get(proposal_program)
            if current is None or proposal.type_score > current.type_score:
                typed_proposals[proposal_program] = replace(
                    proposal, program=proposal_program
                )
            program_sources.setdefault(proposal_program, set()).add(
                "typed_slot_synthesis"
            )
            if result.template.startswith("financial_identity:"):
                program_sources[proposal_program].add(result.template)
    if legacy_reference_result is not None:
        program_sources.setdefault(
            legacy_reference_result.program, set()
        ).add("v12_3_reference_winner")
    if financial_identities:
        # A unary answer to an identity question is safe only when the selected
        # row is the disclosed target itself. Tag those direct disclosures so
        # they receive the same auditable constraint and final grounding gate
        # as a catalog-derived program. Other neural operations remain legacy
        # hypotheses and are not invalidated merely by catalog activation.
        for program, sources in program_sources.items():
            if program.operation != "SELECT" or not program.operands:
                continue
            cell = graph.cells[program.operands[0]]
            if any(
                target_matches_metric(identity, cell.metric_label)
                for identity in financial_identities
            ):
                sources.add("financial_identity:direct")
    # Add the opposite operand order only when the shared language/evidence
    # contract proves that the current directional program is supported and
    # its reversal is contradicted.  The probe remains runtime-ineligible;
    # Phase-C may retain it as a typed hard negative with real model/verifier
    # features.  Ambiguous or unsupported questions never create this row.
    for program, sources in list(program_sources.items()):
        if not reversed_pair_is_proven_negative(plan, program, graph):
            continue
        reversed_program = ReasoningProgram(
            program.operation, tuple(reversed(program.operands))
        )
        program_sources.setdefault(reversed_program, set()).add(
            "directional_reversal_probe"
        )
    programs = set(program_sources)
    preverification_programs: tuple[ReasoningProgram, ...] = ()
    selected_for_verification: tuple[ReasoningProgram, ...] = ()
    if diverse_set_beam_enabled:
        program_count_before_dedup = len(programs)
        for program in programs:
            _operation_probability, neural_score, _pointers = (
                _program_neural_evidence(output, row, program)
            )
            score = neural_score + 0.05 * generation_scores.get(
                program, 0.0
            )
            proposal = typed_proposals.get(program)
            if proposal is not None:
                score += 0.10 * proposal.type_score
            generation_scores[program] = score
        preverification_programs = _diverse_preverification_order(
            tuple(programs),
            program_sources,
            generation_scores,
            candidates,
            baseline,
        )
        conflict_indices = _conflicting_physical_indices(candidates)
        signature_to_representative = {
            _program_physical_signature(
                program,
                candidates,
                conflict_indices=conflict_indices,
            ): program
            for program in preverification_programs
        }
        merged_sources: dict[ReasoningProgram, set[str]] = {
            program: set() for program in preverification_programs
        }
        for program, sources in program_sources.items():
            signature = _program_physical_signature(
                program,
                candidates,
                conflict_indices=conflict_indices,
            )
            representative = signature_to_representative.get(signature)
            if representative is None:
                continue
            merged_sources[representative].update(sources)
            proposal = typed_proposals.get(program)
            current = typed_proposals.get(representative)
            if proposal is not None and (
                current is None or proposal.type_score > current.type_score
            ):
                typed_proposals[representative] = replace(
                    proposal,
                    program=representative,
                )
        program_sources = merged_sources
        selected_for_verification = preverification_programs[:10]
        programs = set(selected_for_verification)
        generated_before_physical_dedup = max(
            generated_before_physical_dedup,
            program_count_before_dedup,
        )
        generated_after_physical_dedup = len(
            preverification_programs
        )

    verification_evidence: EvidenceGraph | Sequence = (
        graph if graph_enabled else graph.cells
    )
    rejected = 0
    search_rows: list[_SearchProgramCandidate] = []
    ordered_programs = (
        list(selected_for_verification)
        if diverse_set_beam_enabled
        else [
            baseline,
            *sorted(
                programs - {baseline},
                key=lambda program: (
                    OPERATION_TO_ID[
                        "ADD"
                        if program.operation == SUM_N_OPERATION
                        else program.operation
                    ],
                    program.operands,
                ),
            ),
        ]
    )
    for program in ordered_programs:
        sources = tuple(sorted(program_sources.get(program, ())))
        typed_proposal = typed_proposals.get(program)
        proposal_dag = (
            typed_proposal.program_dag if typed_proposal is not None else None
        )
        proposal_bindings = (
            typed_proposal.slot_bindings if typed_proposal is not None else ()
        )
        identity_catalog_candidate = any(
            source.startswith("financial_identity:")
            for source in sources
        )
        identity_constraint_required = (
            identity_catalog_candidate
            or (
                bool(financial_identities)
                and program.operation == "SELECT"
            )
        )
        cached = (
            legacy_verification_cache.get(
                (program, identity_constraint_required)
            )
            if proposal_dag is None
            else None
        )
        if cached is not None:
            verification, reasons = cached
        else:
            verification = verify_program(
                plan,
                program,
                verification_evidence,
                financial_identity_catalog_enabled=(
                    financial_identity_catalog_enabled
                    and identity_constraint_required
                ),
                program_dag=proposal_dag,
                slot_bindings=proposal_bindings,
            )
            reasons = _program_search_rejection_reasons(
                spec,
                question,
                program,
                candidates,
                verification,
                financial_identity_catalog_enabled=(
                    financial_identity_catalog_enabled
                    and identity_constraint_required
                ),
                program_dag=proposal_dag,
            )
        operation_probability, neural_score, pointer_confidences = (
            _program_neural_evidence(output, row, program)
        )
        search_rows.append(
            _SearchProgramCandidate(
                program=program,
                neural_score=neural_score,
                operation_probability=operation_probability,
                pointer_confidences=pointer_confidences,
                verification=verification,
                rejection_reasons=reasons,
                generation_sources=sources,
                identity_constraint_required=identity_constraint_required,
                program_dag=proposal_dag,
                slot_bindings=proposal_bindings,
            )
        )
        if reasons:
            rejected += 1

    if diverse_set_beam_enabled:
        preverification_rank = {
            program: rank
            for rank, program in enumerate(
                preverification_programs, start=1
            )
        }
        ranked_rows = sorted(
            search_rows,
            key=lambda item: preverification_rank.get(
                item.program, len(preverification_rank) + 1
            ),
        )
        for item in ranked_rows:
            item.neural_rank = preverification_rank.get(
                item.program, len(preverification_rank) + 1
            )
    else:
        ranked_rows = sorted(
            search_rows,
            key=lambda item: (
                item.neural_score,
                -OPERATION_TO_ID[
                    "ADD"
                    if item.program.operation == SUM_N_OPERATION
                    else item.program.operation
                ],
                tuple(-index for index in item.program.operands),
            ),
            reverse=True,
        )
        for rank, item in enumerate(ranked_rows, start=1):
            item.neural_rank = rank

    pool_width = (
        max((*slot_pool_10.keys(), *slot_pool_20.keys()), default=-1) + 1
    )
    slot_candidate_pools_at_10 = tuple(
        tuple(slot_pool_10.get(slot, ())[:10])
        for slot in range(pool_width)
    )
    slot_candidate_pools_at_20 = tuple(
        tuple(slot_pool_20.get(slot, ())[:20])
        for slot in range(pool_width)
    )
    operation_slot_candidate_pools_at_10 = tuple(
        (operation, operation_slot_pool_10[operation])
        for operation in operations
        if operation in operation_slot_pool_10
    )
    operation_slot_candidate_pools_at_20 = tuple(
        (operation, operation_slot_pool_20[operation])
        for operation in operations
        if operation in operation_slot_pool_20
    )
    eligible_rows = [item for item in ranked_rows if not item.rejection_reasons]
    if not eligible_rows:
        if not candidate_recall_v12_4_enabled:
            return None
        fallback = next(
            (
                item
                for item in ranked_rows
                if item.program == baseline
            ),
            ranked_rows[0],
        )
        trace = tuple(
            ProgramCandidateTrace(
                program=item.program,
                neural_score=item.neural_score,
                neural_rank=item.neural_rank,
                verification_state=item.verification.state.value,
                proof_score=item.verification.proof_score,
                eligible=False,
                rejection_reasons=item.rejection_reasons,
                unknown_reasons=item.verification.unknown_reasons,
                generation_sources=item.generation_sources,
                training_state=_program_candidate_training_state(
                    item.verification, item.rejection_reasons
                ),
                proof_contract_state=(
                    item.verification.proof_contract_state.value
                ),
                required_valid_fraction=(
                    item.verification.required_valid_fraction
                ),
                required_unknown_fraction=(
                    item.verification.required_unknown_fraction
                ),
                optional_support_fraction=(
                    item.verification.optional_support_fraction
                ),
                proof_obligations=_serialized_proof_obligations(
                    item.verification
                ),
            )
            for item in ranked_rows
        )
        return SymbolicProgramSearchResult(
            program=fallback.program,
            pointer_confidences=fallback.pointer_confidences,
            neural_pointer_confidences=fallback.pointer_confidences,
            constraints_applied=("trace_complete_fallback",),
            candidates_considered=(
                len(preverification_programs)
                if diverse_set_beam_enabled
                else len(programs)
            ),
            candidates_rejected=rejected,
            proof_score=fallback.verification.proof_score,
            unknown_constraints=len(fallback.verification.unknown_reasons),
            trace=trace,
            synthesized_program_candidates=len(synthesized_programs),
            typed_synthesis_status=tuple(
                f"{result.operation}:{result.template}:{result.status}:"
                f"{len(result.proposals)}"
                for result in synthesis_results
            ),
            verification_cache=tuple(
                (
                    item.program,
                    item.identity_constraint_required,
                    item.verification,
                    item.rejection_reasons,
                )
                for item in search_rows
            ),
            program_dag=fallback.program_dag,
            slot_bindings=fallback.slot_bindings,
            slot_candidate_pools_at_10=slot_candidate_pools_at_10,
            slot_candidate_pools_at_20=slot_candidate_pools_at_20,
            operation_slot_candidate_pools_at_10=(
                operation_slot_candidate_pools_at_10
            ),
            operation_slot_candidate_pools_at_20=(
                operation_slot_candidate_pools_at_20
            ),
            operation_beam=operations,
            preverification_programs=preverification_programs,
            selected_for_verification=selected_for_verification,
            generated_programs_before_physical_dedup=(
                generated_before_physical_dedup
            ),
            generated_programs_after_physical_dedup=(
                generated_after_physical_dedup
            ),
        )
    legacy_rows = sorted(
        eligible_rows,
        key=lambda item: _symbolic_program_search_key(
            spec,
            question,
            item.program,
            candidates,
            item.neural_score,
            item.verification,
            proof_guided=proof_guided,
        ),
        reverse=True,
    )
    legacy_count = len(legacy_rows)
    for legacy_rank, item in enumerate(legacy_rows, start=1):
        item.legacy_rank = legacy_rank
        item.legacy_winner = legacy_rank == 1
        item.legacy_rank_score = (
            1.0
            if legacy_count == 1
            else 1.0 - (legacy_rank - 1) / (legacy_count - 1)
        )
    for item in ranked_rows:
        if not item.legacy_rank:
            item.legacy_rank = legacy_count + 1
            item.legacy_rank_score = 0.0
        item.reranker_features = _program_reranker_feature_vector(
            spec,
            question,
            item.program,
            candidates,
            item.verification,
            neural_score=item.neural_score,
            neural_rank=item.neural_rank,
            total_programs=len(ranked_rows),
            operation_probability=item.operation_probability,
            pointer_confidences=item.pointer_confidences,
            legacy_rank=item.legacy_rank,
            legacy_rank_score=item.legacy_rank_score,
            legacy_winner=item.legacy_winner,
            feature_schema=(
                str(
                    reranker.metadata.get(
                        "feature_schema", PROGRAM_RERANKER_FEATURE_SCHEMA
                    )
                )
                if reranker is not None
                else PROGRAM_RERANKER_FEATURE_SCHEMA
            ),
        )

    legacy_chosen = legacy_rows[0]
    chosen: _SearchProgramCandidate = legacy_chosen
    reranker_supported_rows = [
        item
        for item in ranked_rows
        if (
            item.program.operation in RERANKER_TRAIN_OPERATIONS
            and not item.rejection_reasons
        )
    ]
    reranker_active = (
        reranker is not None
        and legacy_chosen.program.operation in RERANKER_TRAIN_OPERATIONS
        and len(reranker_supported_rows) >= 2
    )
    reranker_overridden = False
    if reranker_active:
        raw_top_k = reranker.metadata.get("top_k", 30)
        reranker_top_k = (
            int(raw_top_k) if isinstance(raw_top_k, (int, float)) else 30
        )
        reranker_top_k = max(2, min(64, reranker_top_k))
        reranker_rows = _select_diverse_reranker_candidates(
            reranker_supported_rows, limit=reranker_top_k
        )
        (
            winner,
            combined_scores,
            reranker_confidences,
            reranker_overridden,
            _,
        ) = reranker.select(
            [item.reranker_features for item in reranker_rows],
            operation=legacy_chosen.program.operation,
        )
        for item, score, confidence in zip(
            reranker_rows, combined_scores, reranker_confidences
        ):
            item.selected_for_reranking = True
            item.reranker_score = score
            item.reranker_confidence = confidence
        chosen = reranker_rows[winner]

    chosen, safe_residual_preserved = _preserve_safe_residual_reference(
        legacy_reference_result,
        eligible_rows,
        chosen,
        reranker_overridden=reranker_overridden,
    )

    # Only within a source-proven equivalent SELECT group, prefer the fullest
    # reported amount when no coarser display unit was requested. This is a
    # source precision preference, not a new score threshold or a numeric oracle.
    if chosen.program.operation == "SELECT" and plan.requested_scale in {None, "vnd"}:
        from .disclosure_equivalence import _quantum
        def equivalent_precision(item):
            mapping = item.verification.mapping_assessment
            if item.program.operation != "SELECT" or mapping is None or len(mapping.operand_mappings) != 1:
                return None
            operand = mapping.operand_mappings[0]
            if not operand.equivalent_disclosures:
                return None
            cell = graph.cells[item.program.operands[0]]
            scale = currency_unit_scale(cell.unit)
            quantum = _quantum(cell, scale) if scale is not None else None
            if quantum is None:
                return None
            group = tuple(sorted(repr(m.identity) for m in operand.valid_physical_mappings))
            return group, quantum
        current_precision = equivalent_precision(chosen)
        if current_precision is not None:
            fuller = [(precision[1], item.neural_rank, item) for item in eligible_rows
                      if (precision := equivalent_precision(item)) is not None
                      and precision[0] == current_precision[0] and precision[1] < current_precision[1]]
            if fuller:
                chosen = min(fuller, key=lambda item: (item[0], item[1], item[2].program.operands))[2]
                chosen.generation_sources = tuple(dict.fromkeys((*chosen.generation_sources, "equivalent_disclosure_source_precision")))

    trace = tuple(
        ProgramCandidateTrace(
            program=item.program,
            neural_score=item.neural_score,
            neural_rank=item.neural_rank,
            verification_state=item.verification.state.value,
            proof_score=item.verification.proof_score,
            eligible=not item.rejection_reasons,
            rejection_reasons=item.rejection_reasons,
            unknown_reasons=item.verification.unknown_reasons,
            selected_for_reranking=item.selected_for_reranking,
            reranker_features=item.reranker_features,
            reranker_score=item.reranker_score,
            reranker_confidence=item.reranker_confidence,
            legacy_rank=item.legacy_rank,
            legacy_rank_score=item.legacy_rank_score,
            legacy_winner=item.legacy_winner,
            generation_sources=item.generation_sources,
            training_state=_program_candidate_training_state(
                item.verification, item.rejection_reasons
            ),
            proof_contract_state=(
                item.verification.proof_contract_state.value
            ),
            required_valid_fraction=(
                item.verification.required_valid_fraction
            ),
            required_unknown_fraction=(
                item.verification.required_unknown_fraction
            ),
            optional_support_fraction=(
                item.verification.optional_support_fraction
            ),
            proof_obligations=_serialized_proof_obligations(
                item.verification
            ),
        )
        for item in ranked_rows
    )
    applied_constraints: list[str] = []
    if chosen.program != baseline:
        applied_constraints.append("symbolic_program_search")
    if "typed_slot_synthesis" in chosen.generation_sources:
        applied_constraints.append("typed_program_synthesis")
    if any(
        source.startswith("financial_identity:")
        for source in chosen.generation_sources
    ):
        applied_constraints.append("financial_identity_catalog")
    if reranker_overridden:
        applied_constraints.append("program_reranker")
    if safe_residual_preserved:
        applied_constraints.append("v12_4_safe_residual_preserved")
    if diverse_set_beam_enabled:
        applied_constraints.append("diverse_set_beam_v1")
    return SymbolicProgramSearchResult(
        program=chosen.program,
        pointer_confidences=chosen.pointer_confidences,
        neural_pointer_confidences=chosen.pointer_confidences,
        constraints_applied=tuple(applied_constraints),
        candidates_considered=(
            len(preverification_programs)
            if diverse_set_beam_enabled
            else len(programs)
        ),
        candidates_rejected=rejected,
        proof_score=chosen.verification.proof_score,
        unknown_constraints=len(chosen.verification.unknown_reasons),
        trace=trace,
        reranker_score=(
            chosen.reranker_score if reranker_overridden else None
        ),
        reranker_confidence=(
            chosen.reranker_confidence if reranker_overridden else None
        ),
        synthesized_program_candidates=len(synthesized_programs),
        typed_synthesis_status=tuple(
            f"{result.operation}:{result.template}:{result.status}:"
            f"{len(result.proposals)}"
            for result in synthesis_results
        ),
        grounded_confidence_floor=_verified_confidence_floor(
            plan,
            chosen.program,
            chosen.verification,
            chosen.generation_sources,
        ),
        verification_cache=tuple(
            (
                item.program,
                item.identity_constraint_required,
                item.verification,
                item.rejection_reasons,
            )
            for item in search_rows
        ),
        program_dag=chosen.program_dag,
        slot_bindings=chosen.slot_bindings,
        slot_candidate_pools_at_10=slot_candidate_pools_at_10,
        slot_candidate_pools_at_20=slot_candidate_pools_at_20,
        operation_slot_candidate_pools_at_10=(
            operation_slot_candidate_pools_at_10
        ),
        operation_slot_candidate_pools_at_20=(
            operation_slot_candidate_pools_at_20
        ),
        operation_beam=operations,
        preverification_programs=preverification_programs,
        selected_for_verification=selected_for_verification,
        generated_programs_before_physical_dedup=(
            generated_before_physical_dedup
        ),
        generated_programs_after_physical_dedup=(
            generated_after_physical_dedup
        ),
    )


def _metadata_float(metadata: dict[str, object], key: str, default: float) -> float:
    try:
        value = float(metadata.get(key, default))
    except (TypeError, ValueError):
        return default
    if key == "abstention_threshold" and not 0.0 <= value <= 1.0:
        return default
    if key in {
        "pointer_primary_weight",
        "pointer_auxiliary_weight",
        "pointer_secondary_weight",
    } and not 0.0 <= value <= 1.0:
        return default
    if key == "pointer_relevance_weight" and not 0.0 <= value <= 10.0:
        return default
    return value


def _logical_operation_name(operation: str) -> str:
    return SUM_N_OPERATION if is_sum_operation(operation) else operation


def _logical_evaluation_operations() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            _logical_operation_name(operation)
            for operation in (*OPERATIONS, SUM_N_OPERATION)
        )
    )

def _uses_logical_sum_evaluation(bundle: DocumentHRMBundle) -> bool:
    """Select metric labels from architecture, never calibration timing.

    ``variable_sum_mode`` is authoritative when present. Stable schema
    metadata supports architecture-light consumers, while the policy probe is
    retained only for older mock/checkpoint bundles that predate both fields.
    """

    config = getattr(getattr(bundle, "model", None), "config", None)
    variable_sum_mode = getattr(config, "variable_sum_mode", None)
    if variable_sum_mode == "sum_n_v1":
        return True
    if variable_sum_mode == "disabled":
        return False

    metadata = getattr(bundle, "metadata", {})
    if not isinstance(metadata, dict):
        return False
    schema = metadata.get("evaluation_operation_schema")
    if schema == "logical-sum-family-v1":
        return True
    if schema == "legacy-operation-label-v1":
        return False
    logical_operations = metadata.get("logical_program_operations")
    if isinstance(logical_operations, (list, tuple)):
        return SUM_N_OPERATION in logical_operations

    # Compatibility only: V14 checkpoints/mocks created before the stable
    # architecture/schema fields used the existence of typed policies as the
    # sole signal. New bundles never reach this fallback.
    return isinstance(
        metadata.get("abstention_policies_by_operation"), dict
    )



def _arity_normalized_operation_equal(
    predicted: ReasoningProgram,
    expected: ReasoningProgram,
) -> bool:
    if is_sum_operation(predicted.operation) and is_sum_operation(expected.operation):
        return (
            len(predicted.operands) == len(expected.operands)
            and operation_accepts_arity(
                predicted.operation, len(predicted.operands)
            )
            and operation_accepts_arity(
                expected.operation, len(expected.operands)
            )
        )
    return predicted.operation == expected.operation


def _prediction_action_decision(bundle, question, prediction, candidates, *, publication_policy=None):
    from .source_action_policy import decide_answer, CALIBRATED_ACTION_POLICY
    raw = bundle.metadata.get("abstention_policies_by_operation", {})
    payload = raw.get(_logical_operation_name(prediction.program.operation)) if isinstance(raw, dict) else None
    reasons = (
        *_prediction_required_proof_rejection_reasons(question, prediction, candidates),
        *(f"missing_requested_period:{p}" for p in _evaluation_missing_requested_periods(question, candidates)),
    )
    return decide_answer(_prediction_public_source_action(question, prediction, candidates),
                         payload, prediction.confidence, reason_codes=reasons,
                         policy_id=CALIBRATED_ACTION_POLICY if publication_policy is None else publication_policy)


def _prediction_calibration_accepts(
    bundle: DocumentHRMBundle,
    prediction: ProgramPrediction,
) -> bool:
    """Apply V14 policy metadata without falling back to numeric sentinels."""

    raw = bundle.metadata.get("abstention_policies_by_operation")
    if not isinstance(raw, dict):
        return False
    payload = raw.get(_logical_operation_name(prediction.program.operation))
    if not isinstance(payload, dict):
        return False
    try:
        policy = CalibrationPolicy.from_dict(payload)
    except CalibrationPolicyError:
        return False
    return policy.accepts(prediction.confidence)


def _evaluation_missing_requested_periods(
    question: str,
    candidates: Sequence[NumericCandidate],
) -> tuple[str, ...]:
    """Mirror public period coverage without importing the reasoning engine."""

    requested = set(_requested_years(question))
    if not requested:
        return ()
    period_candidates: Sequence[NumericCandidate] = candidates
    plan = compile_question_plan(question)
    if plan.operation == "SELECT":
        pool = set(range(len(candidates)))
        compatible = _metric_compatible_indices(question, candidates, pool)
        if not compatible:
            labeled = tuple(
                candidate
                if candidate.extraction_method
                else replace(candidate, extraction_method="native")
                for candidate in candidates
            )
            compatible = _metric_compatible_indices(question, labeled, pool)
        if not compatible:
            return tuple(sorted(requested))
        period_candidates = tuple(candidates[index] for index in compatible)
    covered: set[str] = set()
    for candidate in period_candidates:
        covered.update(_candidate_period_years(candidate, question))
    return tuple(sorted(requested - covered))


def _prediction_public_source_action(
    question: str,
    prediction: ProgramPrediction,
    candidates: Sequence[NumericCandidate],
) -> str:
    """Replay the proof/mapping portion of the public gate for eval/calibration."""

    plan = compile_question_plan(question)
    graph = build_evidence_graph(candidates)
    program_dag = getattr(prediction, "program_dag", None)
    identity_enabled = (
        "financial_identity_catalog"
        in getattr(prediction, "grounding_constraints_applied", ())
    )
    verification = verify_program(
        plan,
        prediction.program,
        graph,
        financial_identity_catalog_enabled=identity_enabled,
        program_dag=program_dag,
        slot_bindings=getattr(prediction, "slot_bindings", ()),
    )
    grounding_rejections = _grounding_gate_reasons(
        question,
        prediction.program,
        candidates,
        verification,
        financial_identity_catalog_enabled=identity_enabled,
        program_dag=program_dag,
    )
    required_states = tuple(
        obligation.state.value
        if obligation.state is not None
        else VerificationState.UNKNOWN.value
        for obligation in verification.proof_obligations
        if obligation.applicability == "required"
    )
    mapping_resolution = str(
        getattr(verification.mapping_assessment, "resolution", "UNKNOWN")
    )
    return classify_source_action(
        missing_periods=_evaluation_missing_requested_periods(
            question, candidates
        ),
        verification_state=verification.state.value,
        required_states=required_states,
        mapping_resolution=mapping_resolution,
        abstained=bool(grounding_rejections),
    )


def checkpoint_validation_examples(
    bundle: DocumentHRMBundle,
    examples_or_path: Sequence[ProgramExample] | str | Path,
) -> list[ProgramExample]:
    examples = (
        load_program_examples(examples_or_path)
        if isinstance(examples_or_path, (str, Path))
        else list(examples_or_path)
    )
    raw_ids = bundle.metadata.get("validation_example_ids", [])
    if not isinstance(raw_ids, list) or not raw_ids:
        raise DocumentHRMError(
            "Checkpoint does not contain validation IDs; use --all-examples explicitly"
        )
    validation_ids = {str(item) for item in raw_ids}
    selected = [item for item in examples if item.example_id in validation_ids]
    if len(selected) != len(validation_ids):
        raise DocumentHRMError(
            "Evaluation dataset does not contain every checkpoint validation example"
        )
    return selected


def _abstention_threshold_for_operation(
    bundle: DocumentHRMBundle,
    operation: str,
    default: float | None = None,
) -> float:
    fallback = (
        _metadata_float(bundle.metadata, "abstention_threshold", 0.75)
        if default is None
        else default
    )
    raw = bundle.metadata.get("abstention_thresholds_by_operation")
    if not isinstance(raw, dict):
        return fallback
    value = raw.get(operation)
    if not isinstance(value, (int, float)):
        return fallback
    threshold = float(value)
    return threshold if 0.0 <= threshold <= 1.0 else fallback


def _abstention_threshold_for_prediction(
    bundle: DocumentHRMBundle,
    prediction: ProgramPrediction,
    default: float | None = None,
) -> float:
    """Use sidecar calibration only when the sidecar supplied confidence."""

    if (
        getattr(prediction, "reranker_confidence", None) is not None
        and getattr(bundle, "program_reranker", None) is not None
    ):
        return bundle.program_reranker.threshold_for_operation(
            prediction.program.operation
        )
    return _abstention_threshold_for_operation(
        bundle,
        prediction.program.operation,
        default,
    )


def _proof_conditioned_acceptance_eligible(
    prediction: ProgramPrediction,
) -> bool:
    """Return whether a multi-step winner has a complete required proof.

    This is deliberately narrower than treating every verifier ``VALID`` as
    accepted.  The selected program must be a typed-synthesis winner backed by
    an executable derived DAG, with no search rejection and every *required*
    proof obligation explicitly VALID.  Optional layout constraints may stay
    UNKNOWN; direct SELECT/ADD winners and flat neural programs remain governed
    by their calibrated confidence threshold.
    """

    dag = getattr(prediction, "program_dag", None)
    if dag is None or len(getattr(dag, "nodes", ())) < 2:
        return False
    if prediction.program.operation in {"SELECT", "ADD"}:
        return False
    selected = next(
        (
            item
            for item in getattr(prediction, "program_search_trace", ())
            if item.program == prediction.program
        ),
        None,
    )
    if selected is None:
        return False
    if "typed_slot_synthesis" not in selected.generation_sources:
        return False
    if not selected.eligible or selected.rejection_reasons:
        return False
    if selected.proof_contract_state != VerificationState.VALID.value:
        return False
    if selected.required_valid_fraction != 1.0:
        return False
    if selected.required_unknown_fraction != 0.0:
        return False
    required = tuple(
        item
        for item in selected.proof_obligations
        if item.startswith("required:")
    )
    return bool(required) and all(item.endswith(":VALID") for item in required)


def _prediction_required_proof_rejection_reasons(
    question: str,
    prediction: ProgramPrediction,
    candidates: Sequence[NumericCandidate],
) -> tuple[str, ...]:
    """Recheck the selected program so neural confidence cannot bypass QP3 proof."""

    plan = compile_question_plan(question)
    if plan.abstention_reason:
        return (f"question_abstention:{plan.abstention_reason}",)
    verification = verify_program(
        plan,
        prediction.program,
        build_evidence_graph(candidates),
        financial_identity_catalog_enabled=(
            "financial_identity_catalog"
            in getattr(prediction, "grounding_constraints_applied", ())
        ),
        program_dag=getattr(prediction, "program_dag", None),
        slot_bindings=getattr(prediction, "slot_bindings", ()),
    )
    reasons: list[str] = list(program_binding_issues(prediction.program, candidates, getattr(prediction, "program_dag", None)))
    for obligation in verification.proof_obligations:
        if obligation.applicability != "required":
            continue
        if obligation.state is VerificationState.UNKNOWN:
            reasons.append(f"required_proof_unknown:{obligation.name}")
        elif obligation.state is VerificationState.INVALID:
            reasons.append(f"required_proof_invalid:{obligation.name}")
    return tuple(dict.fromkeys(reasons))


def calibrate_operation_abstention_policies(
    bundle: DocumentHRMBundle,
    examples: Sequence[ProgramExample],
    *,
    target_precision: float = 0.90,
    minimum_accepted: int = 5,
    batch_size: int = 32,
) -> dict[str, CalibrationPolicy]:
    """Calibrate explicit V14 policies against the public structural gate."""

    if not 0.0 < target_precision <= 1.0:
        raise DocumentHRMError("target_precision must be within (0, 1]")
    if minimum_accepted < 1:
        raise DocumentHRMError("minimum_accepted must be positive")
    if batch_size < 1:
        raise DocumentHRMError("batch_size must be positive")

    records: dict[str, list[tuple[float, bool, bool]]] = {
        operation: [] for operation in _logical_evaluation_operations()
    }
    for start in range(0, len(examples), batch_size):
        current = list(examples[start : start + batch_size])
        predictions = bundle.predict_many(current)
        for example, prediction in zip(current, predictions):
            operation = _logical_operation_name(prediction.program.operation)
            records.setdefault(operation, []).append(
                (
                    prediction.confidence,
                    _execution_is_correct(prediction.program, example),
                    _prediction_public_source_action(
                        example.question,
                        prediction,
                        example.candidates,
                    )
                    == "ANSWER",
                )
            )

    policies: dict[str, CalibrationPolicy] = {}
    for operation in _logical_evaluation_operations():
        rows = records.get(operation, [])
        support = len(rows)
        if not support:
            policies[operation] = CalibrationPolicy(
                status=CalibrationStatus.UNCALIBRATED,
                threshold=None,
                support=0,
                accepted=0,
                correct=0,
                precision=0.0,
                lower_bound=0.0,
                target=target_precision,
            )
            continue

        choices: list[tuple[int, int, float, float]] = []
        thresholds = sorted(
            {
                confidence
                for confidence, _, structurally_eligible in rows
                if structurally_eligible
            }
        )
        for threshold in thresholds:
            accepted_rows = tuple(
                correct
                for confidence, correct, structurally_eligible in rows
                if structurally_eligible and confidence >= threshold
            )
            accepted = len(accepted_rows)
            if accepted < minimum_accepted:
                continue
            correct = sum(accepted_rows)
            lower_bound = wilson_lower_bound(correct, accepted)
            if lower_bound >= target_precision:
                choices.append((accepted, correct, threshold, lower_bound))
        if not choices:
            policies[operation] = CalibrationPolicy(
                status=CalibrationStatus.DISABLED,
                threshold=None,
                support=support,
                accepted=0,
                correct=0,
                precision=0.0,
                lower_bound=0.0,
                target=target_precision,
            )
            continue
        accepted, correct, threshold, lower_bound = max(
            choices,
            key=lambda item: (item[0], item[1], -item[2]),
        )
        policies[operation] = CalibrationPolicy(
            status=CalibrationStatus.CALIBRATED,
            threshold=threshold,
            support=support,
            accepted=accepted,
            correct=correct,
            precision=correct / accepted,
            lower_bound=lower_bound,
            target=target_precision,
        )
    return policies


def calibrate_abstention_threshold(
    bundle: DocumentHRMBundle,
    examples: Sequence[ProgramExample],
    *,
    target_precision: float = 0.80,
    minimum_accepted: int = 5,
    stability_errors: int = DEFAULT_ABSTENTION_STABILITY_ERRORS,
    minimum_threshold: float | None = None,
    batch_size: int = 32,
) -> tuple[float, int, float, bool]:
    if stability_errors < 0:
        raise DocumentHRMError("stability_errors must be non-negative")
    if minimum_threshold is not None and not 0.0 <= minimum_threshold <= 1.0:
        raise DocumentHRMError("minimum_threshold must be within [0, 1]")
    records: list[tuple[float, bool, bool]] = []
    for start in range(0, len(examples), batch_size):
        current = list(examples[start : start + batch_size])
        predictions = bundle.predict_many(current)
        records.extend(
            (
                prediction.confidence,
                _execution_is_correct(prediction.program, example),
                not _grounding_gate_reasons(
                    example.question,
                    prediction.program,
                    example.candidates,
                    financial_identity_catalog_enabled=(
                        "financial_identity_catalog"
                        in getattr(
                            prediction,
                            "grounding_constraints_applied",
                            (),
                        )
                    ),
                ),
            )
            for example, prediction in zip(current, predictions)
        )
    if not records:
        return 1.0, 0, 0.0, False

    threshold_values = {
        confidence
        for confidence, _, eligible in records
        if eligible
        and (minimum_threshold is None or confidence >= minimum_threshold)
    }
    if minimum_threshold is not None:
        threshold_values.add(minimum_threshold)
    candidates: list[tuple[float, int, int, float]] = []
    for threshold in sorted(threshold_values):
        accepted = [
            correct
            for confidence, correct, eligible in records
            if eligible and confidence >= threshold
        ]
        if len(accepted) < minimum_accepted:
            continue
        correct = sum(accepted)
        precision = correct / len(accepted)
        candidates.append((threshold, len(accepted), correct, precision))

    meeting_target = [
        item
        for item in candidates
        if _precision_survives_errors(
            item[2],
            item[1],
            target_precision=target_precision,
            stability_errors=stability_errors,
        )
    ]
    if meeting_target:
        # Lowest threshold gives the greatest coverage while preserving precision.
        threshold, accepted, _, precision = min(
            meeting_target, key=lambda item: item[0]
        )
        return threshold, accepted, precision, True

    if candidates:
        # A weak checkpoint should not create an impossible >1 threshold that
        # silently abstains forever. Use the most precise available operating
        # point and record that the requested target was not reached.
        threshold, accepted, _, precision = max(
            candidates, key=lambda item: (item[3], item[1], item[0])
        )
        return threshold, accepted, precision, False

    eligible_records = [
        (confidence, correct)
        for confidence, correct, eligible in records
        if eligible
        and (minimum_threshold is None or confidence >= minimum_threshold)
    ]
    if not eligible_records:
        threshold = minimum_threshold if minimum_threshold is not None else 1.0
        return threshold, 0, 0.0, False
    best_confidence = max(confidence for confidence, _ in eligible_records)
    accepted = [
        correct
        for confidence, correct in eligible_records
        if confidence >= best_confidence
    ]
    precision = sum(accepted) / len(accepted) if accepted else 0.0
    return best_confidence, len(accepted), precision, False


def calibrate_operation_abstention_thresholds(
    bundle: DocumentHRMBundle,
    examples: Sequence[ProgramExample],
    *,
    target_precision: float = 0.80,
    minimum_accepted: int = 5,
    stability_errors: int = DEFAULT_ABSTENTION_STABILITY_ERRORS,
    minimum_threshold: float | None = None,
    batch_size: int = 32,
) -> tuple[dict[str, float], int, float, bool]:
    """Maximize validation coverage with one threshold per predicted operation."""

    if stability_errors < 0:
        raise DocumentHRMError("stability_errors must be non-negative")
    if minimum_threshold is not None and not 0.0 <= minimum_threshold <= 1.0:
        raise DocumentHRMError("minimum_threshold must be within [0, 1]")
    records: dict[str, list[tuple[float, bool, bool]]] = {
        operation: [] for operation in OPERATIONS
    }
    for start in range(0, len(examples), batch_size):
        current = list(examples[start : start + batch_size])
        predictions = bundle.predict_many(current)
        for example, prediction in zip(current, predictions):
            records[prediction.program.operation].append(
                (
                    prediction.confidence,
                    _execution_is_correct(prediction.program, example),
                    not _grounding_gate_reasons(
                        example.question,
                        prediction.program,
                        example.candidates,
                        financial_identity_catalog_enabled=(
                            "financial_identity_catalog"
                            in getattr(
                                prediction,
                                "grounding_constraints_applied",
                                (),
                            )
                        ),
                    ),
                )
            )
    if not examples:
        return {operation: 1.0 for operation in OPERATIONS}, 0, 0.0, False

    operation_options: dict[
        str, list[tuple[int, int, float]]
    ] = {}
    for operation in OPERATIONS:
        current = records[operation]
        threshold_values = {
            confidence
            for confidence, _, eligible in current
            if eligible
            and (
                minimum_threshold is None
                or confidence >= minimum_threshold
            )
        }
        threshold_values.add(
            minimum_threshold if minimum_threshold is not None else 1.0
        )
        by_outcome: dict[tuple[int, int], float] = {}
        for threshold in threshold_values:
            accepted = [
                correct
                for confidence, correct, eligible in current
                if eligible and confidence >= threshold
            ]
            outcome = (len(accepted), sum(accepted))
            # The higher boundary is safer for validation-tied outcomes.
            by_outcome[outcome] = max(
                threshold, by_outcome.get(outcome, 0.0)
            )
        operation_options[operation] = [
            (accepted, correct, threshold)
            for (accepted, correct), threshold in by_outcome.items()
        ]

    states: dict[tuple[int, int], dict[str, float]] = {(0, 0): {}}
    for operation in OPERATIONS:
        updated: dict[tuple[int, int], dict[str, float]] = {}
        for (accepted_so_far, correct_so_far), thresholds in states.items():
            for accepted, correct, threshold in operation_options[operation]:
                key = (
                    accepted_so_far + accepted,
                    correct_so_far + correct,
                )
                candidate = {**thresholds, operation: float(threshold)}
                existing = updated.get(key)
                if existing is None or sum(candidate.values()) > sum(
                    existing.values()
                ):
                    updated[key] = candidate
        states = updated

    meeting_target = [
        (accepted, correct, thresholds)
        for (accepted, correct), thresholds in states.items()
        if accepted >= minimum_accepted
        and _precision_survives_errors(
            correct,
            accepted,
            target_precision=target_precision,
            stability_errors=stability_errors,
        )
    ]
    if meeting_target:
        accepted, correct, thresholds = max(
            meeting_target,
            key=lambda item: (item[0], item[1]),
        )
        return thresholds, accepted, correct / accepted, True

    nonempty = [
        (accepted, correct, thresholds)
        for (accepted, correct), thresholds in states.items()
        if accepted > 0
    ]
    if not nonempty:
        return {operation: 1.0 for operation in OPERATIONS}, 0, 0.0, False
    accepted, correct, thresholds = max(
        nonempty,
        key=lambda item: (
            item[1] / item[0],
            item[0],
            item[1],
        ),
    )
    return thresholds, accepted, correct / accepted, False


def _precision_survives_errors(
    correct: int,
    accepted: int,
    *,
    target_precision: float,
    stability_errors: int,
) -> bool:
    if accepted <= 0:
        return False
    conservative_correct = max(0, correct - stability_errors)
    return conservative_correct / accepted >= target_precision


def _execution_is_correct(program: ReasoningProgram, example: ProgramExample) -> bool:
    try:
        predicted_value = execute_program(program, example.candidates)
        expected_value = execute_program(example.program, example.candidates)
        difference = abs(predicted_value - expected_value)
        tolerance = max(0.035, float(abs(expected_value)) * 1e-9)
        return float(difference) <= tolerance
    except Exception:
        return False


def model_summary(bundle: DocumentHRMBundle) -> str:
    return json.dumps(
        {
            "schema_version": DOCUMENT_HRM_SCHEMA_VERSION,
            "architecture": bundle.metadata.get("architecture"),
            "parameters": bundle.parameter_count,
            "config": asdict(bundle.model.config),
            "metadata": bundle.metadata,
        },
        ensure_ascii=False,
        indent=2,
    )


def _encode_text_matrix(
    texts: Sequence[str],
    tokenizer: HashSequenceTokenizer,
    max_tokens: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    encoded = [tokenizer.encode(text, max_tokens) for text in texts]
    width = max(len(item) for item in encoded)
    ids = torch.zeros((len(texts), width), dtype=torch.long, device=device)
    mask = torch.zeros((len(texts), width), dtype=torch.float32, device=device)
    for row, values in enumerate(encoded):
        ids[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
        mask[row, : len(values)] = 1.0
    return ids, mask


def _question_features(question: str) -> list[float]:
    text = normalize_text(question)
    years = len(set(re.findall(r"\b20\d{2}\b", text)))
    percent_change = any(
        term in text
        for term in (
            "tang truong",
            "tang giam",
            "bien dong",
            "thay doi",
            "muc thay doi",
            "tu dau nam den cuoi nam",
            "tang bao nhieu phan tram",
            "giam bao nhieu phan tram",
        )
    )
    ratio = any(
        term in text
        for term in (
            "ty le",
            "ty suat",
            "ty trong",
            "tren tong",
            "chiem bao nhieu phan tram",
        )
    ) or ("phan tram" in text and not percent_change)
    addition = _question_requests_addition(text)
    subtraction = any(
        term in text for term in ("chenh lech", "giam bao nhieu", "tang bao nhieu")
    )
    direct_lookup = any(
        term in text for term in ("la bao nhieu", "so du", "gia tri")
    ) and not any((ratio, percent_change, addition, subtraction))
    return [
        float(ratio),
        float(percent_change),
        float(addition),
        float(subtraction),
        float(any(term in text for term in ("binh quan", "trung binh"))),
        float(any(term in text for term in ("nhan", "tich", "so luong ban", "gia ban"))),
        float(any(term in text for term in ("he so", "bao nhieu lan", "chia"))),
        float(direct_lookup),
        float("%" in question or "phan tram" in text),
        float(any(term in text for term in ("so sanh", "nam truoc", "dau nam", "cuoi nam"))),
        min(1.0, years / 2.0),
        min(1.0, len(text.split()) / 40.0),
    ]


def _question_requests_addition(normalized_question: str) -> bool:
    if compile_question_plan(normalized_question).operation == "ADD":
        return True
    text = f" {normalized_question} "
    return bool(
        (
            normalized_question.startswith("tinh tong ")
            and not normalized_question.startswith("tinh tong gia tri ")
        )
        or " cong lai " in text
        or " co tong gia tri " in text
        or (
            normalized_question.startswith("cong ")
            and " voi " in text
            and not normalized_question.startswith("cong ty ")
        )
        or (
            normalized_question.startswith("tong ")
            and " va " in text
            and " tai " in text
            and " la bao nhieu" in text
        )
    )


def _question_requests_single_value_lookup(question: str) -> bool:
    """Recognize a single financial metric requested for one absolute year.

    A bare lookup such as ``Doanh thu thuần năm 2023`` contains no arithmetic
    verb, so leaving its operation entirely neural can turn a one-cell request
    into an apparently confident ADD. Keep this override deliberately narrow:
    exactly one explicit year and exactly one broad accounting metric family.
    Explicit arithmetic wording is handled before this helper is called.
    """

    if len(_requested_years(question)) != 1:
        return False
    text = normalize_text(_question_focus_text(question))
    if _question_requests_addition(text):
        return False
    metric_families = (
        "doanh thu",
        "loi nhuan",
        "tai san",
        "no phai tra",
        "von chu so huu",
        "chi phi",
        "gia von",
        "hang ton kho",
        "khoan phai thu",
        "khoan phai tra",
        "dong tien",
        "tien va cac khoan tuong duong tien",
        "ebitda",
    )
    matched = {family for family in metric_families if family in text}
    # Nested labels such as "tiền và các khoản tương đương tiền" also contain
    # the broad word "tiền"; the family list intentionally includes only the
    # full cash label so one disclosed row still counts as a single metric.
    if "tien va cac khoan tuong duong tien" in matched:
        matched.discard("dong tien")
    return len(matched) == 1


def _candidate_pair_features(
    left: NumericCandidate,
    right: NumericCandidate,
    left_index: int,
    right_index: int,
) -> list[float]:
    left_row = normalize_text(_context_field(left.context, "ROW") or left.row_label)
    right_row = normalize_text(_context_field(right.context, "ROW") or right.row_label)
    left_column = normalize_text(
        _context_field(left.context, "COLUMN") or left.column_label
    )
    right_column = normalize_text(
        _context_field(right.context, "COLUMN") or right.column_label
    )
    left_period = normalize_text(_context_field(left.context, "PERIOD") or left.period)
    right_period = normalize_text(
        _context_field(right.context, "PERIOD") or right.period
    )
    left_type = normalize_text(_context_field(left.context, "ROW_TYPE") or left.row_type)
    right_type = normalize_text(
        _context_field(right.context, "ROW_TYPE") or right.row_type
    )
    left_table = normalize_text(_context_field(left.context, "TABLE") or left.table_id)
    right_table = normalize_text(
        _context_field(right.context, "TABLE") or right.table_id
    )
    left_section = normalize_text(
        _context_field(left.context, "SECTION") or left.section
    )
    right_section = normalize_text(
        _context_field(right.context, "SECTION") or right.section
    )
    left_scope = normalize_text(_context_field(left.context, "SCOPE") or left.scope)
    right_scope = normalize_text(
        _context_field(right.context, "SCOPE") or right.scope
    )
    left_entity = normalize_text(
        _context_field(left.context, "ENTITY") or left.entity_id
    )
    right_entity = normalize_text(
        _context_field(right.context, "ENTITY") or right.entity_id
    )
    left_subtable = normalize_text(
        _context_field(left.context, "SUBTABLE") or left.subtable_id
    )
    right_subtable = normalize_text(
        _context_field(right.context, "SUBTABLE") or right.subtable_id
    )
    return [
        float(bool(left_row and left_row == right_row)),
        float(bool(left_column and left_column == right_column)),
        float(bool(left_period and left_period == right_period)),
        float(left.line_number == right.line_number),
        float(left.unit == right.unit),
        float(left_type == "total"),
        float(right_type == "total"),
        float(left_type == "subtotal"),
        float(right_type == "subtotal"),
        float(abs(left_index - right_index) == 1),
        float(left.page is not None and left.page == right.page),
        float(bool(left_table and left_table == right_table)),
        float(bool(left_section and left_section == right_section)),
        float(bool(left_scope and left_scope == right_scope)),
        float(bool(left_table and right_table and left_table != right_table)),
        float(bool(left_scope and right_scope and left_scope != right_scope)),
        float(bool(left_entity and left_entity == right_entity)),
        float(bool(left_subtable and left_subtable == right_subtable)),
        float(bool(left_entity and right_entity and left_entity != right_entity)),
        float(bool(left_subtable and right_subtable and left_subtable != right_subtable)),
    ]


def _add_select_contrastive_loss(
    operation_logits: Tensor,
    operation_targets: Tensor,
    *,
    margin: float = 0.75,
) -> Tensor:
    """Separate ADD from SELECT without changing labels or validation data."""

    add_id = OPERATION_TO_ID["ADD"]
    select_id = OPERATION_TO_ID["SELECT"]
    relevant = operation_targets.eq(add_id) | operation_targets.eq(select_id)
    if not relevant.any():
        return torch.zeros((), device=operation_logits.device)
    logits = operation_logits[relevant]
    targets = operation_targets[relevant]
    gold = torch.where(
        targets.eq(add_id), logits[:, add_id], logits[:, select_id]
    )
    contrast = torch.where(
        targets.eq(add_id), logits[:, select_id], logits[:, add_id]
    )
    return F.relu(margin - gold + contrast).mean()


def _unary_structural_negative_mask(
    candidates: Sequence[NumericCandidate], gold_pointer: int
) -> list[bool]:
    """Identify general row/period near misses without guessing missing metadata."""

    if gold_pointer < 0 or gold_pointer >= len(candidates):
        raise DocumentHRMError("gold pointer is outside the candidate list")
    gold = candidates[gold_pointer]
    gold_row = normalize_text(_structured_candidate_row_label(gold))
    gold_period = _candidate_period_descriptor(gold)
    targeted: list[bool] = []
    for index, candidate in enumerate(candidates):
        row = normalize_text(_structured_candidate_row_label(candidate))
        period = _candidate_period_descriptor(candidate)
        complete_relation = bool(gold_row and row and gold_period and period)
        same_row_wrong_period = (
            complete_relation and gold_row == row and gold_period != period
        )
        same_period_wrong_row = (
            complete_relation and gold_period == period and gold_row != row
        )
        targeted.append(
            index != gold_pointer
            and (same_row_wrong_period or same_period_wrong_row)
        )
    return targeted


def _numeric_features(
    candidate: NumericCandidate,
    position: int,
    max_candidates: int,
    *,
    question: str = "",
    candidates: Sequence[NumericCandidate] = (),
    feature_size: int = 8,
) -> list[float]:
    value = float(candidate.decimal)
    magnitude = math.log1p(abs(value)) / 32.0
    raw_digits = len([char for char in candidate.raw if char.isdigit()]) / 18.0
    unit_vnd = 1.0 if candidate.unit == "VND" else 0.0
    features = [
        magnitude,
        1.0 if value < 0 else 0.0,
        1.0 if candidate.is_percent else 0.0,
        1.0 if candidate.is_year_like else 0.0,
        max(-1.0, min(1.0, candidate.relevance)),
        position / max(1, max_candidates - 1),
        raw_digits,
        unit_vnd,
    ]
    if feature_size <= len(features):
        return features[:feature_size]

    question_terms = _grounding_terms(question)
    context_terms = _grounding_terms(candidate.context)
    row = _context_field(candidate.context, "ROW")
    column = _context_field(candidate.context, "COLUMN")
    value_line = _context_field(candidate.context, "VALUE_LINE")
    row_terms = _grounding_terms(row)
    column_terms = _grounding_terms(column)
    value_line_terms = _grounding_terms(value_line)
    question_years = set(re.findall(r"\b20\d{2}\b", question))
    column_years = set(re.findall(r"\b20\d{2}\b", column))
    normalized_question = normalize_text(question)
    normalized_column = normalize_text(column)
    normalized_row = normalize_text(row)
    row_exact_match = bool(
        normalized_row
        and f" {normalized_row} " in f" {normalized_question} "
    )
    relevance_rank = 0.0
    if candidates:
        relevance_rank = sum(
            other.relevance <= candidate.relevance for other in candidates
        ) / len(candidates)
    extras = [
        len(question_terms & context_terms) / max(1, len(question_terms)),
        len(question_terms & row_terms) / max(1, len(question_terms)),
        len(question_terms & column_terms) / max(1, len(question_terms)),
        len(question_terms & value_line_terms) / max(1, len(question_terms)),
        float(bool(question_years and question_years & column_years)),
        float(
            any(term in normalized_question for term in ("cuoi nam", "31/12", "31.12"))
            and any(term in normalized_column for term in ("cuoi nam", "31/12", "31.12"))
        ),
        float(
            any(term in normalized_question for term in ("dau nam", "01/01", "01.01"))
            and any(term in normalized_column for term in ("dau nam", "01/01", "01.01"))
        ),
        float(row_exact_match),
        relevance_rank,
        0.0,  # Frozen semantic cosine is filled after batch encoding.
    ]
    combined = features + extras
    if len(combined) < feature_size:
        combined.extend([0.0] * (feature_size - len(combined)))
    return combined[:feature_size]


_GROUNDING_STOPWORDS = {
    "bao",
    "bao cao",
    "bang",
    "cho",
    "chi",
    "cua",
    "duoc",
    "gia tri",
    "la",
    "nam",
    "nhieu",
    "phan",
    "so",
    "tai",
    "theo",
    "tinh",
    "tren",
    "va",
    "vao",
}


def _grounding_terms(text: str) -> set[str]:
    normalized = normalize_text(text)
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalized)
        if len(token) > 1
        and token not in _GROUNDING_STOPWORDS
        and not token.isdigit()
    }


def _context_field(context: str, name: str) -> str:
    match = re.search(
        rf"(?im)(?:^|\n){re.escape(name)}:\s*(.*?)(?=\n[A-Z_]+:|$)",
        context,
    )
    return match.group(1).strip() if match else ""


def _semantic_scope_label(plan: QuestionPlan | None) -> str:
    if plan is None:
        return ""
    normalized = f" {plan.normalized} "
    if " hop nhat " in normalized or " consolidated " in normalized:
        return "hợp nhất"
    if " cong ty me " in normalized or " parent company " in normalized:
        return "công ty mẹ"
    return "đơn vị báo cáo" if plan.entity_mode == "reporting" else ""


def _semantic_question_surface(
    question: str, plan: QuestionPlan | None
) -> str:
    surface = unicodedata.normalize("NFC", question.strip())
    if plan is None or not plan.entity_text:
        return surface
    # The issuer identifier is document routing metadata, not financial
    # semantics. Remove the planner-resolved named entity from the encoder
    # surface while retaining generic consolidated/parent-company scope.
    return re.sub(
        re.escape(plan.entity_text),
        " ",
        surface,
        flags=re.IGNORECASE,
    ).strip()


def _question_semantic_text(
    question: str,
    plan: QuestionPlan | None,
    *,
    template_version: str,
) -> str:
    if template_version == LEGACY_SEMANTIC_TEMPLATE:
        return question
    if template_version != FINANCIAL_VI_SEMANTIC_TEMPLATE:
        raise DocumentHRMError("Unsupported semantic question template")
    active_plan = plan or compile_question_plan(question)
    periods = (
        *active_plan.requested_years,
        *active_plan.requested_period_roles,
    )
    fields = (
        ("Câu hỏi tài chính", _semantic_question_surface(question, active_plan)),
        ("Chỉ tiêu", active_plan.metric_text),
        ("Kỳ", ", ".join(periods)),
        ("Vai trò cột", ", ".join(active_plan.column_roles)),
        ("Phạm vi", _semantic_scope_label(active_plan)),
        ("Đơn vị", active_plan.requested_unit_family),
        ("Phép toán", active_plan.operation or "chưa xác định"),
    )
    return unicodedata.normalize(
        "NFC",
        "\n".join(f"{label}: {value}" for label, value in fields if value),
    )


def _slot_semantic_text(
    question: str,
    plan: QuestionPlan | None,
    slot_index: int,
    *,
    template_version: str,
) -> str:
    if template_version == LEGACY_SEMANTIC_TEMPLATE:
        return question
    active_plan = plan or compile_question_plan(question)
    slot = (
        active_plan.operand_slots[slot_index]
        if slot_index < len(active_plan.operand_slots)
        else None
    )
    metric = (
        slot.metric
        if slot is not None
        else active_plan.operand_roles[slot_index]
        if slot_index < len(active_plan.operand_roles)
        else active_plan.metric_text
    )
    year = slot.year if slot is not None else ""
    period_role = slot.period_role if slot is not None else ""
    column_role = slot.column_role if slot is not None else ""
    unit_family = (
        slot.unit_family
        if slot is not None
        else active_plan.requested_unit_family
    )
    fields = (
        ("Câu hỏi tài chính", _semantic_question_surface(question, active_plan)),
        ("Chỉ tiêu", metric),
        ("Kỳ", ", ".join(value for value in (year, period_role) if value)),
        ("Vai trò cột", column_role),
        ("Phạm vi", _semantic_scope_label(active_plan)),
        ("Đơn vị", unit_family),
        ("Phép toán", active_plan.operation or "chưa xác định"),
        ("Operand", str(slot_index + 1)),
    )
    return unicodedata.normalize(
        "NFC",
        "\n".join(f"{label}: {value}" for label, value in fields if value),
    )


def _candidate_semantic_text(
    candidate: NumericCandidate,
    *,
    mode: str,
    template_version: str = LEGACY_SEMANTIC_TEMPLATE,
) -> str:
    if template_version == FINANCIAL_VI_SEMANTIC_TEMPLATE:
        scope_surface = normalize_text(
            " ".join((candidate.scope, candidate.section))
        )
        if "hop nhat" in scope_surface or "consolidated" in scope_surface:
            scope = "hợp nhất"
        elif "cong ty me" in scope_surface or "parent company" in scope_surface:
            scope = "công ty mẹ"
        else:
            scope = ""
        fields = (
            ("Dòng", candidate.row_label),
            ("Loại dòng", candidate.row_type),
            ("Cột", candidate.column_label),
            ("Kỳ", candidate.period),
            ("Vai trò cột", candidate.column_role),
            ("Đơn vị", candidate.unit),
            ("Phần báo cáo", candidate.section),
            ("Phạm vi", scope),
        )
        structured = "\n".join(
            f"{label}: {value}" for label, value in fields if value
        )
        prose = candidate.context
        prose = re.sub(
            r"(?im)^(?:PAGE|TABLE|ENTITY|SOURCE|SUBTABLE|VALUE|RAW|"
            r"SOURCE_SPAN|SOURCE_BBOX):.*$",
            " ",
            prose,
        )
        for sensitive in (
            candidate.raw,
            candidate.value,
            candidate.source_name,
            candidate.entity_id,
        ):
            if sensitive:
                prose = prose.replace(str(sensitive), " ")
        prose = " ".join(prose.split())
        return unicodedata.normalize(
            "NFC",
            structured or prose or "Bằng chứng văn bản",
        )
    if template_version != LEGACY_SEMANTIC_TEMPLATE:
        raise DocumentHRMError("Unsupported semantic evidence template")
    provenance = "\n".join(
        f"{label}: {value}"
        for label, value in (
            ("PAGE", candidate.page),
            ("TABLE", candidate.table_id),
            ("SECTION", candidate.section),
            ("SCOPE", candidate.scope),
            ("ENTITY", candidate.entity_id),
            ("SUBTABLE", candidate.subtable_id),
        )
        if value not in (None, "")
    )
    if mode == "context":
        return "\n".join(part for part in (provenance, candidate.context) if part)
    context_fields = {
        name: _context_field(candidate.context, name)
        for name in (
            "ROW",
            "ROW_TYPE",
            "COLUMN",
            "PERIOD",
            "TABLE",
            "SECTION",
            "SCOPE",
            "ENTITY",
            "SUBTABLE",
        )
    }
    if not any(context_fields.values()) and not provenance:
        return candidate.context
    fields = (
        ("Chỉ tiêu", context_fields["ROW"] or candidate.row_label),
        ("Loại dòng", context_fields["ROW_TYPE"] or candidate.row_type),
        ("Cột", context_fields["COLUMN"] or candidate.column_label),
        ("Kỳ", context_fields["PERIOD"] or candidate.period),
        ("PAGE", candidate.page),
        ("TABLE", context_fields["TABLE"] or candidate.table_id),
        ("SECTION", context_fields["SECTION"] or candidate.section),
        ("SCOPE", context_fields["SCOPE"] or candidate.scope),
        ("ENTITY", context_fields["ENTITY"] or candidate.entity_id),
        ("SUBTABLE", context_fields["SUBTABLE"] or candidate.subtable_id),
    )
    structured = "\n".join(
        f"{label}: {value}" for label, value in fields if value
    )
    # Prose candidates do not have stable table fields; retaining their full
    # context avoids discarding evidence outside row/cell layouts.
    return structured or candidate.context


def _pair_loss_scale(config: DocumentHRMConfig, epoch: int) -> float:
    if config.pair_loss_warmup_epochs <= 0:
        return config.pair_loss_weight
    progress = min(1.0, max(0.0, epoch / config.pair_loss_warmup_epochs))
    return config.pair_loss_weight * progress


def _targeted_pair_margin_loss(
    scores: Tensor,
    targets: Tensor,
    pair_features: Tensor,
    explicit_negatives: Tensor | None = None,
    *,
    margin: float = 0.75,
) -> Tensor:
    same_row = pair_features[..., 0].gt(0.5)
    same_period = pair_features[..., 2].gt(0.5)
    targeted = (same_row & ~same_period) | (same_period & ~same_row)
    if explicit_negatives is not None:
        targeted = targeted | explicit_negatives.bool()
    targeted = targeted.flatten(1)
    targeted.scatter_(1, targets.unsqueeze(1), False)
    has_targeted = targeted.any(dim=1)
    if not has_targeted.any():
        return torch.zeros((), device=scores.device)
    gold_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
    targeted_scores = scores.masked_fill(~targeted, -1e9)
    hardest_targeted = targeted_scores[has_targeted].max(dim=1).values
    return F.relu(
        margin - gold_scores[has_targeted] + hardest_targeted
    ).mean()


def _verifier_guided_pair_margin_loss(
    scores: Tensor,
    pointer_targets: Tensor,
    operation_targets: Tensor,
    invalid_negatives: Tensor,
    *,
    margin: float = 0.75,
) -> Tensor:
    """Rank gold pairs above only verifier-certain INVALID programs."""

    losses: list[Tensor] = []
    pair_width = int(math.isqrt(scores.shape[1]))
    if pair_width * pair_width != scores.shape[1]:
        raise DocumentHRMError("pair score width is not square")
    invalid_flat = invalid_negatives.bool().flatten(1)
    for row, operation_id in enumerate(operation_targets.tolist()):
        operation = OPERATIONS[operation_id]
        if operation not in VERIFIER_GUIDED_OPERATIONS:
            continue
        if OPERATION_ARITY[operation] != 2:
            continue
        operands = pointer_targets[row, :2]
        if operands.lt(0).any():
            continue
        left, right = (int(value) for value in operands.tolist())
        positives = {left * pair_width + right}
        if operation in COMMUTATIVE:
            positives.add(right * pair_width + left)
        positive_indices = torch.tensor(
            sorted(positives),
            dtype=torch.long,
            device=scores.device,
        )
        negative_mask = invalid_flat[row].clone()
        negative_mask[positive_indices] = False
        if not negative_mask.any():
            continue
        positive_score = torch.logsumexp(
            scores[row, positive_indices], dim=0
        ) - math.log(len(positives))
        hardest_invalid = scores[row].masked_fill(
            ~negative_mask, -1e9
        ).max()
        losses.append(F.relu(margin - positive_score + hardest_invalid))
    if not losses:
        return torch.zeros((), device=scores.device)
    return torch.stack(losses).mean()


def _targeted_unary_margin_loss(
    scores: Tensor,
    targets: Tensor,
    targeted_negatives: Tensor,
    *,
    margin: float = 0.75,
) -> Tensor:
    targeted = targeted_negatives.bool().clone()
    targeted.scatter_(1, targets.unsqueeze(1), False)
    has_targeted = targeted.any(dim=1)
    if not has_targeted.any():
        return torch.zeros((), device=scores.device)
    gold_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
    targeted_scores = scores.masked_fill(~targeted, -1e9)
    hardest_targeted = targeted_scores[has_targeted].max(dim=1).values
    return F.relu(
        margin - gold_scores[has_targeted] + hardest_targeted
    ).mean()


def _hard_negative_cross_entropy(
    scores: Tensor,
    targets: Tensor,
    *,
    margin: float,
    weight: float,
) -> Tensor:
    cross_entropy = F.cross_entropy(scores, targets)
    if weight <= 0 or scores.shape[1] < 2:
        return cross_entropy
    gold_scores = scores.gather(1, targets.unsqueeze(1)).squeeze(1)
    negatives = scores.clone()
    negatives.scatter_(1, targets.unsqueeze(1), -1e9)
    hardest_negative = negatives.max(dim=1).values
    hard_margin = F.relu(margin - gold_scores + hardest_negative).mean()
    return cross_entropy + weight * hard_margin


def _operation_training_loss(
    operation_logits: Tensor,
    operation_targets: Tensor,
    sum_family_targets: Tensor,
    *,
    weight: Tensor | None = None,
) -> Tensor:
    """Use one additive family probability without changing legacy class IDs."""

    if not sum_family_targets.any():
        return F.cross_entropy(
            operation_logits, operation_targets, weight=weight
        )
    log_probabilities = torch.log_softmax(operation_logits, dim=-1)
    row_losses = -log_probabilities.gather(
        1, operation_targets.unsqueeze(1)
    ).squeeze(1)
    sum_family_log_probability = torch.logsumexp(
        log_probabilities[:, list(SUM_FAMILY_OPERATION_IDS)], dim=-1
    )
    row_losses = torch.where(
        sum_family_targets,
        -sum_family_log_probability,
        row_losses,
    )
    if weight is None:
        return row_losses.mean()
    row_weights = weight[operation_targets]
    return (row_losses * row_weights).sum() / row_weights.sum().clamp_min(1e-8)


def _operation_training_predictions(
    operation_logits: Tensor,
    *,
    aggregate_sum_family: bool,
) -> Tensor:
    if not aggregate_sum_family:
        return operation_logits.argmax(dim=-1)
    family_scores = torch.logsumexp(
        operation_logits[:, list(SUM_FAMILY_OPERATION_IDS)], dim=-1
    )
    scores = operation_logits.clone()
    scores[:, OPERATION_TO_ID["ADD"]] = family_scores
    scores[:, OPERATION_TO_ID["SUM3"]] = -torch.inf
    scores[:, OPERATION_TO_ID["SUM4"]] = -torch.inf
    return scores.argmax(dim=-1)


def _commutative_pointer_membership_logits(
    pointer_logits: Tensor,
    arity: int,
) -> Tensor:
    """Combine arbitrary slot scores into one candidate-membership score."""

    if arity < 2 or arity > pointer_logits.shape[-2]:
        raise DocumentHRMError("invalid commutative pointer arity")
    return torch.logsumexp(pointer_logits[..., :arity, :], dim=-2) - math.log(
        arity
    )


def _semantic_adapter_contrastive_loss(
    model: DocumentHierarchicalReasoningModel,
    batch: EncodedBatch,
    *,
    temperature: float = 0.07,
    hard_negative_margin: float = 0.20,
) -> Tensor:
    """Contrast typed query slots against gold evidence without false negatives."""

    if temperature <= 0 or hard_negative_margin < 0:
        raise DocumentHRMError(
            "Semantic contrastive temperature/margin are invalid"
        )
    if (
        batch.slot_semantic is None
        or batch.candidate_semantic is None
        or batch.semantic_pointer_targets is None
        or batch.semantic_slot_declared is None
    ):
        return torch.zeros((), device=batch.candidate_mask.device)
    slot_vectors = model.adapt_semantic_embeddings(batch.slot_semantic)
    candidate_vectors = model.adapt_semantic_embeddings(
        batch.candidate_semantic
    )
    losses: list[Tensor] = []
    for row in range(batch.candidate_mask.shape[0]):
        valid_candidates = batch.candidate_mask[row].bool()
        semantic_targets = batch.semantic_pointer_targets[row]
        target_slots = semantic_targets.ge(0)
        if not target_slots.any():
            continue
        gold_set = semantic_targets[target_slots].unique()
        sum_family = bool(
            batch.sum_family_targets is not None
            and batch.sum_family_targets[row].item()
        )
        for slot in target_slots.nonzero(as_tuple=False).flatten().tolist():
            scores = torch.einsum(
                "h,nh->n",
                slot_vectors[row, slot],
                candidate_vectors[row],
            ) / temperature
            scores = scores.masked_fill(~valid_candidates, -torch.inf)
            target = int(semantic_targets[slot].item())
            typed_slot = bool(
                batch.semantic_slot_declared[row, slot].item()
            )
            if sum_family and not typed_slot:
                positive_score = torch.logsumexp(
                    scores[gold_set], dim=0
                )
                loss = -(
                    positive_score - torch.logsumexp(scores, dim=0)
                )
            else:
                positive_score = scores[target]
                loss = F.cross_entropy(
                    scores.unsqueeze(0),
                    torch.tensor(
                        [target],
                        dtype=torch.long,
                        device=scores.device,
                    ),
                )
            negative_mask = valid_candidates.clone()
            negative_mask[gold_set] = False
            if batch.unary_structural_negatives is not None:
                negative_mask &= batch.unary_structural_negatives[
                    row, slot
                ].bool()
            if negative_mask.any():
                hardest_negative = scores.masked_fill(
                    ~negative_mask, -torch.inf
                ).max()
                loss = loss + F.softplus(
                    hard_negative_margin
                    - positive_score
                    + hardest_negative
                )
            losses.append(loss)
    if not losses:
        return torch.zeros((), device=batch.candidate_mask.device)
    return torch.stack(losses).mean()


def _multi_operand_set_ranking_loss(
    logits: Tensor,
    gold: Tensor,
    candidate_mask: Tensor,
    *,
    margin: float = 0.50,
) -> Tensor:
    """Rank every gold set member above the hardest valid non-member."""

    valid = candidate_mask.bool()
    negatives = valid.clone()
    negatives[gold] = False
    if not negatives.any():
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    hardest_negative = logits.masked_fill(~negatives, -1e9).max()
    return F.softplus(margin - logits[gold] + hardest_negative).mean()


def _commutative_pointer_set_loss(
    pointer_logits: Tensor,
    operation_targets: Tensor,
    pointer_targets: Tensor,
    candidate_mask: Tensor,
    *,
    operand_count_targets: Tensor | None = None,
    sum_family_targets: Tensor | None = None,
    sum_membership_logits: Tensor | None = None,
) -> Tensor:
    """Teach every gold member without assigning an arbitrary slot order."""

    losses: list[Tensor] = []
    for row, operation_id in enumerate(operation_targets.tolist()):
        operation = OPERATIONS[operation_id]
        if operation not in COMMUTATIVE:
            continue
        sum_family_row = bool(
            sum_family_targets is not None
            and sum_family_targets[row].item()
        )
        if sum_family_row:
            if operand_count_targets is None:
                continue
            count_target = int(operand_count_targets[row].item())
            if count_target < 0:
                continue
            arity = count_target + 2
        else:
            arity = OPERATION_ARITY[operation]
        gold = pointer_targets[row, :arity]
        if gold.lt(0).any():
            continue
        if sum_family_row and sum_membership_logits is not None:
            logits = sum_membership_logits[row]
        else:
            logits = _commutative_pointer_membership_logits(
                pointer_logits[row], arity
            )
        logits = logits.masked_fill(candidate_mask[row].eq(0), -1e9)
        if sum_family_row or operation in MULTI_OPERAND_OPERATIONS:
            losses.append(
                _multi_operand_set_ranking_loss(
                    logits,
                    gold,
                    candidate_mask[row],
                )
            )
            continue
        log_probabilities = torch.log_softmax(logits, dim=-1)
        losses.append(-log_probabilities[gold].mean())
    if not losses:
        return torch.zeros((), device=pointer_logits.device)
    return torch.stack(losses).mean()


def _pair_program_loss(
    scores: Tensor,
    pointer_targets: Tensor,
    operation_targets: Tensor,
    *,
    margin: float = 0.50,
    hard_negative_weight: float = 0.25,
) -> Tensor:
    """Ordered CE for directed programs; set-positive loss for commutative ones."""

    losses: list[Tensor] = []
    pair_width = int(math.isqrt(scores.shape[1]))
    if pair_width * pair_width != scores.shape[1]:
        raise DocumentHRMError("pair score width is not square")
    for row, operation_id in enumerate(operation_targets.tolist()):
        operation = OPERATIONS[operation_id]
        arity = OPERATION_ARITY[operation]
        operands = pointer_targets[row, :arity]
        if operands.lt(0).any():
            continue
        if operation in COMMUTATIVE:
            positives = torch.tensor(
                [
                    int(left) * pair_width + int(right)
                    for left in operands.tolist()
                    for right in operands.tolist()
                    if left != right
                ],
                dtype=torch.long,
                device=scores.device,
            )
        else:
            positives = (
                operands[0] * pair_width + operands[1]
            ).view(1)
        row_scores = scores[row]
        log_probabilities = torch.log_softmax(row_scores, dim=-1)
        positive_log_probability = torch.logsumexp(
            log_probabilities[positives], dim=0
        )
        loss = -positive_log_probability
        if hard_negative_weight > 0 and positives.numel() < row_scores.numel():
            positive_score = torch.logsumexp(
                row_scores[positives], dim=0
            ) - math.log(positives.numel())
            negative_mask = torch.ones_like(row_scores, dtype=torch.bool)
            negative_mask[positives] = False
            hardest_negative = row_scores.masked_fill(
                ~negative_mask, -1e9
            ).max()
            loss = loss + hard_negative_weight * F.relu(
                margin - positive_score + hardest_negative
            )
        losses.append(loss)
    if not losses:
        return torch.zeros((), device=scores.device)
    return torch.stack(losses).mean()


def _operation_pointer_logits(
    output: DocumentHRMForward,
    row: int,
    slot: int,
    operation: str,
) -> Tensor:
    if (
        operation == "SELECT"
        and slot == 0
        and output.select_pointer_logits is not None
    ):
        return output.select_pointer_logits[row]
    return output.pointer_logits[row, slot]


def _training_pointer_slot_logits(
    output: DocumentHRMForward,
    operation_targets: Tensor,
    slot: int,
) -> Tensor:
    logits = output.pointer_logits[:, slot]
    if slot != 0 or output.select_pointer_logits is None:
        return logits
    select_rows = operation_targets.eq(OPERATION_TO_ID["SELECT"]).unsqueeze(1)
    return torch.where(select_rows, output.select_pointer_logits, logits)


def _binary_operation_rows(operation_targets: Tensor) -> Tensor:
    return torch.tensor(
        [
            OPERATION_ARITY[OPERATIONS[int(operation_id)]] == 2
            for operation_id in operation_targets.tolist()
        ],
        dtype=torch.bool,
        device=operation_targets.device,
    )


def _training_pointer_predictions(
    output: DocumentHRMForward,
    operation_targets: Tensor,
    pointer_targets: Tensor,
    *,
    use_pair: bool = True,
    operand_count_targets: Tensor | None = None,
    sum_family_targets: Tensor | None = None,
) -> Tensor:
    predicted = output.pointer_logits.argmax(dim=-1)
    if output.select_pointer_logits is not None:
        select_rows = operation_targets.eq(OPERATION_TO_ID["SELECT"])
        predicted[select_rows, 0] = output.select_pointer_logits[
            select_rows
        ].argmax(dim=-1)
    predicted = predicted.clone()
    if output.pair_logits is not None and use_pair:
        pair_rows = (
            pointer_targets[:, 1].ge(0)
            & _binary_operation_rows(operation_targets)
        )
        if sum_family_targets is not None:
            pair_rows = pair_rows & ~sum_family_targets
        if pair_rows.any():
            rows = torch.arange(
                operation_targets.shape[0], device=operation_targets.device
            )
            pair_scores = output.pair_logits[rows, operation_targets].flatten(1)
            pair_width = output.pair_logits.shape[-1]
            pair_indices = pair_scores.argmax(dim=-1)
            predicted[pair_rows, 0] = pair_indices[pair_rows] // pair_width
            predicted[pair_rows, 1] = pair_indices[pair_rows] % pair_width
    for row, operation_id in enumerate(operation_targets.tolist()):
        operation = OPERATIONS[operation_id]
        if operation not in COMMUTATIVE:
            continue
        sum_family_row = bool(
            sum_family_targets is not None
            and sum_family_targets[row].item()
        )
        if sum_family_row:
            if operand_count_targets is None:
                continue
            count_target = int(operand_count_targets[row].item())
            if count_target < 0:
                continue
            arity = count_target + 2
        else:
            arity = OPERATION_ARITY[operation]
        membership = (
            output.sum_membership_logits[row]
            if sum_family_row and output.sum_membership_logits is not None
            else _commutative_pointer_membership_logits(
                output.pointer_logits[row], arity
            )
        )
        selected = (
            {
                int(predicted[row, slot])
                for slot in range(min(2, arity))
            }
            if (
                output.pair_logits is not None
                and use_pair
                and arity == 2
                and not sum_family_row
            )
            else set()
        )
        for slot in range(len(selected), arity):
            scores = membership.clone()
            if selected:
                scores[
                    torch.tensor(sorted(selected), device=scores.device)
                ] = -1e9
            pointer = int(scores.argmax().item())
            predicted[row, slot] = pointer
            selected.add(pointer)
        predicted[row, :arity] = predicted[row, :arity].sort().values
    return predicted


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_synthetic(example: ProgramExample) -> bool:
    return example.example_id.startswith("synthetic-")


def _permute_candidates(
    example: ProgramExample, rng: random.Random
) -> ProgramExample:
    order = list(range(len(example.candidates)))
    rng.shuffle(order)
    old_to_new = {old: new for new, old in enumerate(order)}
    candidates = tuple(
        replace(example.candidates[old], index=new)
        for new, old in enumerate(order)
    )
    operands = tuple(old_to_new[index] for index in example.program.operands)
    if example.program.operation in COMMUTATIVE:
        operands = tuple(sorted(operands))
    return replace(
        example,
        candidates=candidates,
        program=ReasoningProgram(example.program.operation, operands),
        supervision=_remap_supervision_candidate_indices(
            example.supervision, old_to_new
        ),
    )


def _structured_negative_indices(
    example: ProgramExample,
    *,
    operand_width: int = MAX_OPERANDS,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, int], ...]]:
    """Resolve audited hard negatives into unary slots and ordered pair targets."""

    unary: list[set[int]] = [set() for _ in range(operand_width)]
    pairs: set[tuple[int, int]] = set()
    supervision = example.supervision
    if not isinstance(supervision, dict):
        return tuple(tuple() for _ in range(operand_width)), tuple()
    hard_negatives = supervision.get("hard_negatives")
    if not isinstance(hard_negatives, list):
        return tuple(tuple() for _ in range(operand_width)), tuple()

    candidate_count = len(example.candidates)
    gold_operands = _supervision_gold_operand_indices(example)
    gold_pair = tuple(gold_operands[:2]) if len(gold_operands) >= 2 else None
    for raw_negative in hard_negatives:
        if not isinstance(raw_negative, dict):
            continue
        raw_indices = raw_negative.get("candidate_indices")
        if not isinstance(raw_indices, list):
            continue
        indices = tuple(
            index
            for index in raw_indices
            if isinstance(index, int) and 0 <= index < candidate_count
        )
        raw_slot = raw_negative.get("operand_slot")
        slot = raw_slot if isinstance(raw_slot, int) else None
        if slot is not None and 0 <= slot < operand_width:
            gold = gold_operands[slot] if slot < len(gold_operands) else None
            unary[slot].update(index for index in indices if index != gold)
            if gold_pair is not None and slot < 2:
                for index in indices:
                    pair = list(gold_pair)
                    pair[slot] = index
                    pairs.add((pair[0], pair[1]))
        elif len(indices) == 2:
            pairs.add((indices[0], indices[1]))

    if gold_pair is not None:
        pairs.discard((gold_pair[0], gold_pair[1]))
    return (
        tuple(tuple(sorted(indices)) for indices in unary),
        tuple(sorted(pairs)),
    )


def _supervision_gold_operand_indices(
    example: ProgramExample,
) -> tuple[int, ...]:
    """Return semantic slot order without changing commutative program order."""

    supervision = example.supervision
    if not isinstance(supervision, dict):
        return example.program.operands
    raw_gold = supervision.get("gold_operands")
    if not isinstance(raw_gold, list):
        return example.program.operands
    ordered: list[int] = []
    for item in raw_gold:
        if not isinstance(item, dict):
            return example.program.operands
        index = item.get("candidate_index")
        if not isinstance(index, int) or not 0 <= index < len(example.candidates):
            return example.program.operands
        ordered.append(index)
    if (
        len(ordered) != len(example.program.operands)
        or set(ordered) != set(example.program.operands)
    ):
        return example.program.operands
    return tuple(ordered)


def _remap_supervision_candidate_indices(
    supervision: dict[str, object] | None,
    old_to_new: dict[int, int],
) -> dict[str, object] | None:
    if supervision is None:
        return None
    remapped = deepcopy(supervision)
    gold_operands = remapped.get("gold_operands")
    if isinstance(gold_operands, list):
        for operand in gold_operands:
            if not isinstance(operand, dict):
                continue
            index = operand.get("candidate_index")
            if isinstance(index, int) and index in old_to_new:
                operand["candidate_index"] = old_to_new[index]
    hard_negatives = remapped.get("hard_negatives")
    if isinstance(hard_negatives, list):
        for negative in hard_negatives:
            if not isinstance(negative, dict):
                continue
            indices = negative.get("candidate_indices")
            if isinstance(indices, list):
                negative["candidate_indices"] = [
                    old_to_new[index]
                    for index in indices
                    if isinstance(index, int) and index in old_to_new
                ]
    return remapped


def _example_group_key(example: ProgramExample) -> str:
    # Prefer the company identity. Some reports explicitly contain
    # "Mã cổ phiếu: Không có"; treating "không" as a ticker previously merged
    # hundreds of unrelated companies into one validation group.
    company_match = re.search(
        r"(?im)^\s*c[oô]ng ty\s*:\s*(.+?)\s*$",
        example.context,
    )
    if company_match:
        company = normalize_text(company_match.group(1))
        company = re.sub(r"\s+", " ", company).strip(" -:|")
        if company:
            return f"company:{company}"

    normalized = normalize_text(example.context)
    ticker = re.search(r"\bma co phieu\s*:?\s*([a-z0-9]{2,10})\b", normalized)
    invalid_tickers = {"khong", "none", "null", "na", "n/a", "chua"}
    if ticker and ticker.group(1) not in invalid_tickers:
        return f"ticker:{ticker.group(1)}"

    # Stable report-level fallback instead of example_id, so questions from the
    # same unidentified report still stay on one side of the split.
    metadata_lines = "\n".join(example.context.splitlines()[:8])
    digest = blake2b(normalize_text(metadata_lines).encode("utf-8"), digest_size=8).hexdigest()
    return f"report:{digest}"


def _validate_training_split_boundaries(
    train_examples: Sequence[ProgramExample],
    validation_examples: Sequence[ProgramExample],
) -> None:
    if not train_examples:
        raise DocumentHRMError("Training split contains no examples")
    if not validation_examples:
        raise DocumentHRMError("Validation split contains no examples")

    def ids_and_duplicates(
        examples: Sequence[ProgramExample],
    ) -> tuple[set[str], list[str]]:
        counts = Counter(example.example_id for example in examples)
        return set(counts), sorted(
            example_id for example_id, count in counts.items() if count > 1
        )

    train_ids, train_duplicates = ids_and_duplicates(train_examples)
    validation_ids, validation_duplicates = ids_and_duplicates(validation_examples)
    if train_duplicates:
        raise DocumentHRMError(
            "Training split contains duplicate example IDs: "
            + ", ".join(train_duplicates[:5])
        )
    if validation_duplicates:
        raise DocumentHRMError(
            "Validation split contains duplicate example IDs: "
            + ", ".join(validation_duplicates[:5])
        )
    overlapping_ids = sorted(train_ids & validation_ids)
    if overlapping_ids:
        raise DocumentHRMError(
            "Training and validation example IDs overlap: "
            + ", ".join(overlapping_ids[:5])
        )

    train_groups = {_example_group_key(example) for example in train_examples}
    validation_groups = {
        _example_group_key(example) for example in validation_examples
    }
    overlapping_groups = sorted(train_groups & validation_groups)
    if overlapping_groups:
        raise DocumentHRMError(
            "Training and validation company/report groups overlap: "
            + ", ".join(overlapping_groups[:5])
        )


def _split_from_validation_ids(
    examples: Sequence[ProgramExample],
    validation_ids: set[str],
    seed: int,
    *,
    allow_synthetic_validation: bool = False,
) -> tuple[list[ProgramExample], list[ProgramExample]]:
    validation = [
        example
        for example in examples
        if (
            (allow_synthetic_validation or not _is_synthetic(example))
            and example.example_id in validation_ids
        )
    ]
    found = {example.example_id for example in validation}
    missing = validation_ids - found
    if missing:
        raise DocumentHRMError(
            f"Fixed validation IDs are missing from the dataset: {len(missing)}"
        )
    train = [
        example
        for example in examples
        if (
            (
                not allow_synthetic_validation
                and _is_synthetic(example)
            )
            or example.example_id not in validation_ids
        )
    ]
    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def _grouped_real_split(
    examples: Sequence[ProgramExample], fraction: float, seed: int
) -> tuple[list[ProgramExample], list[ProgramExample]]:
    real_examples = [example for example in examples if not _is_synthetic(example)]
    synthetic_examples = [example for example in examples if _is_synthetic(example)]
    if len(real_examples) < 10:
        return _stratified_split(examples, fraction, seed)

    groups: dict[str, list[ProgramExample]] = {}
    for example in real_examples:
        groups.setdefault(_example_group_key(example), []).append(example)

    target_total = max(1, int(round(len(real_examples) * fraction)))
    total_by_operation = Counter(example.program.operation for example in real_examples)
    target_by_operation = {
        operation: total_by_operation[operation] * fraction for operation in OPERATIONS
    }

    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)
    # Large groups are placed first; the cost function keeps operation coverage
    # close to the real-data distribution without leaking a company across sets.
    keys.sort(key=lambda key: len(groups[key]), reverse=True)
    selected: set[str] = set()
    selected_total = 0
    selected_by_operation: Counter[str] = Counter()

    def cost(total: int, counts: Counter[str]) -> float:
        size_error = abs(total - target_total) / max(1, target_total)
        operation_error = sum(
            abs(counts[operation] - target_by_operation[operation])
            / max(1.0, target_by_operation[operation])
            for operation in OPERATIONS
            if total_by_operation[operation]
        )
        return size_error + 0.35 * operation_error

    for key in keys:
        group = groups[key]
        candidate_total = selected_total + len(group)
        candidate_counts = selected_by_operation + Counter(
            item.program.operation for item in group
        )
        must_fill = selected_total < target_total * 0.75
        if must_fill or cost(candidate_total, candidate_counts) < cost(
            selected_total, selected_by_operation
        ):
            selected.add(key)
            selected_total = candidate_total
            selected_by_operation = candidate_counts

    if not selected:
        selected.add(keys[0])
    if len(selected) == len(keys) and len(keys) > 1:
        selected.remove(keys[-1])

    validation = [
        example
        for key in keys
        if key in selected
        for example in groups[key]
    ]
    train = [
        *synthetic_examples,
        *(
            example
            for key in keys
            if key not in selected
            for example in groups[key]
        ),
    ]
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation

def _stratified_split(
    examples: Sequence[ProgramExample], fraction: float, seed: int
) -> tuple[list[ProgramExample], list[ProgramExample]]:
    rng = random.Random(seed)
    grouped: dict[str, list[ProgramExample]] = {name: [] for name in OPERATIONS}
    for example in examples:
        grouped[example.program.operation].append(example)
    train: list[ProgramExample] = []
    validation: list[ProgramExample] = []
    for group in grouped.values():
        rng.shuffle(group)
        count = max(1, int(round(len(group) * fraction))) if len(group) > 4 else 1
        validation.extend(group[:count])
        train.extend(group[count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation
