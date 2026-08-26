from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import random
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


PROGRAM_RERANKER_SCHEMA_VERSION = 3
LEGACY_PROGRAM_RERANKER_FEATURE_SCHEMA = "typed-evidence-safe-residual-v2"
PROGRAM_RERANKER_FEATURE_SCHEMA = "typed-evidence-safe-residual-v3"
SUPPORTED_PROGRAM_RERANKER_FEATURE_SCHEMAS = frozenset(
    {
        LEGACY_PROGRAM_RERANKER_FEATURE_SCHEMA,
        PROGRAM_RERANKER_FEATURE_SCHEMA,
    }
)
DISABLE_OVERRIDE_MARGIN = 1_000_000_000.0

PROGRAM_RERANKER_FEATURES = (
    "neural_log_score",
    "reciprocal_neural_rank",
    "relative_neural_rank",
    "legacy_rank_score",
    "legacy_winner",
    "legacy_reciprocal_rank",
    "operation_probability",
    "pointer_geometric_mean",
    "pointer_minimum",
    "verifier_proof_score",
    "verifier_valid_fraction",
    "verifier_unknown_fraction",
    "verifier_invalid_fraction",
    "verifier_state_valid",
    "verifier_state_unknown",
    "operation_match",
    "metric_match",
    "period_coverage",
    "temporal_direction",
    "entity_match",
    "unit_match",
    "same_table",
    "same_row",
    "denominator_safe",
    "execution_valid",
    "graph_same_metric",
    "graph_same_entity",
    "graph_same_table",
    "graph_same_row",
    "graph_same_unit",
    "graph_temporal_order",
    "metric_alignment_score",
    "period_score",
    "structural_score",
    "candidate_relevance_mean",
    "candidate_relevance_min",
    "same_scope",
    "normalized_arity",
    "unknown_constraint_fraction",
)

FEATURE_INDEX = {
    name: index for index, name in enumerate(PROGRAM_RERANKER_FEATURES)
}


def features_with_training_state(
    features: Sequence[float], state: str
) -> tuple[float, ...]:
    """Return a feature row whose verifier-state flags match cache tri-state."""

    if len(features) != len(PROGRAM_RERANKER_FEATURES):
        raise ProgramRerankerError("Cannot rewrite an incompatible feature row")
    if state not in {"VALID", "UNKNOWN", "INVALID"}:
        raise ProgramRerankerError(f"Unsupported training state: {state}")
    values = [float(value) for value in features]
    values[FEATURE_INDEX["verifier_state_valid"]] = float(state == "VALID")
    values[FEATURE_INDEX["verifier_state_unknown"]] = float(state == "UNKNOWN")
    return tuple(values)
RERANKER_TRAIN_OPERATIONS = frozenset(
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


class ProgramRerankerError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProgramRerankerConfig:
    feature_dim: int = len(PROGRAM_RERANKER_FEATURES)
    hidden_size: int = 48
    dropout: float = 0.10

    def validate(self) -> None:
        if self.feature_dim != len(PROGRAM_RERANKER_FEATURES):
            raise ProgramRerankerError(
                "Reranker feature dimension does not match the runtime schema"
            )
        if self.hidden_size < 4:
            raise ProgramRerankerError("hidden_size must be at least 4")
        if not 0.0 <= self.dropout < 1.0:
            raise ProgramRerankerError("dropout must be within [0, 1)")


class ProgramRerankerModel(nn.Module):
    """Predict a bounded correction; legacy symbolic rank stays external."""

    def __init__(self, config: ProgramRerankerConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.rank_backbone = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
        )
        self.confidence_backbone = nn.Sequential(
            nn.Linear(config.feature_dim, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
        )
        self.rank_head = nn.Linear(config.hidden_size, 1)
        self.confidence_head = nn.Linear(config.hidden_size, 1)
        # Epoch 0 is exactly the legacy symbolic policy. This makes a real
        # no-regression checkpoint available to model selection.
        nn.init.zeros_(self.rank_head.weight)
        nn.init.zeros_(self.rank_head.bias)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        rank_hidden = self.rank_backbone(features)
        confidence_hidden = self.confidence_backbone(features)
        correction = torch.tanh(self.rank_head(rank_hidden)).squeeze(-1)
        confidence_logits = self.confidence_head(confidence_hidden).squeeze(-1)
        return correction, confidence_logits


@dataclass(frozen=True)
class RerankerCandidateRecord:
    features: tuple[float, ...]
    program_key: str
    exact_gold: bool
    execution_correct: bool
    positive_class: str = "negative"
    generation_sources: tuple[str, ...] = ()
    runtime_eligible: bool = True
    training_state: str = "VALID"
    proof_contract_state: str = ""
    rejection_reasons: tuple[str, ...] = ()
    unknown_reasons: tuple[str, ...] = ()
    required_valid_fraction: float = 0.0
    required_unknown_fraction: float = 0.0
    optional_support_fraction: float = 0.0
    proof_obligations: tuple[str, ...] = ()
    supervision_only: bool = False

    def __post_init__(self) -> None:
        if len(self.features) != len(PROGRAM_RERANKER_FEATURES):
            raise ProgramRerankerError(
                f"Expected {len(PROGRAM_RERANKER_FEATURES)} features, "
                f"received {len(self.features)}"
            )
        if self.positive_class == "negative" and self.exact_gold:
            object.__setattr__(self, "positive_class", "canonical_exact")
        elif self.positive_class == "negative" and self.execution_correct:
            object.__setattr__(self, "positive_class", "execution_only")
        if self.positive_class not in {
            "canonical_exact",
            "grounded_equivalent",
            "execution_only",
            "negative",
        }:
            raise ProgramRerankerError(
                f"Unsupported candidate positive class: {self.positive_class}"
            )
        if self.training_state not in {"VALID", "UNKNOWN", "INVALID"}:
            raise ProgramRerankerError(
                f"Unsupported candidate training state: {self.training_state}"
            )
        if self.proof_contract_state and self.proof_contract_state not in {
            "VALID",
            "UNKNOWN",
            "INVALID",
        }:
            raise ProgramRerankerError(
                "Unsupported candidate proof-contract state: "
                f"{self.proof_contract_state}"
            )
        if self.training_state == "INVALID" and not (
            self.supervision_only
            and not self.runtime_eligible
            and self.positive_class in {"negative", "execution_only"}
        ):
            raise ProgramRerankerError(
                "INVALID candidates require a proven supervision-only negative"
            )

    @property
    def rank_positive(self) -> bool:
        return self.positive_class in {
            "canonical_exact",
            "grounded_equivalent",
        }


@dataclass(frozen=True)
class RerankerExampleRecord:
    example_id: str
    operation: str
    candidates: tuple[RerankerCandidateRecord, ...]

    @property
    def has_exact_gold(self) -> bool:
        return any(item.exact_gold for item in self.candidates)

    @property
    def has_rank_positive(self) -> bool:
        return any(item.rank_positive for item in self.candidates)

    @property
    def has_canonical_positive(self) -> bool:
        return any(
            item.positive_class == "canonical_exact"
            for item in self.candidates
        )


@dataclass(frozen=True)
class ProgramRerankerTrainingReport:
    train_examples: int
    train_rank_examples: int
    calibration_examples: int
    calibration_rank_examples: int
    epochs: int
    selected_epoch: int
    final_loss: float
    calibration_legacy_program_exact: float
    calibration_legacy_execution: float
    calibration_legacy_coverage: float
    calibration_legacy_accepted_precision: float
    calibration_legacy_target_coverage: float
    calibration_legacy_target_precision: float
    calibration_program_exact: float
    calibration_execution: float
    calibration_overrides: int
    override_margin: float
    override_margins_by_operation: dict[str, float]
    calibration_coverage: float
    calibration_accepted_precision: float
    confidence_epochs: int
    ranking_regression_guard_passed: bool
    confidence_target_met: bool
    confidence_coverage_guard_passed: bool
    regression_guard_passed: bool


class ProgramRerankerBundle:
    def __init__(
        self,
        model: ProgramRerankerModel,
        *,
        feature_mean: Sequence[float] | Tensor | None = None,
        feature_scale: Sequence[float] | Tensor | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.model = model
        width = model.config.feature_dim
        self.feature_mean = torch.as_tensor(
            feature_mean if feature_mean is not None else [0.0] * width,
            dtype=torch.float32,
        )
        self.feature_scale = torch.as_tensor(
            feature_scale if feature_scale is not None else [1.0] * width,
            dtype=torch.float32,
        ).clamp_min(1e-6)
        if self.feature_mean.shape != (width,) or self.feature_scale.shape != (width,):
            raise ProgramRerankerError("Invalid reranker normalization vectors")
        self.metadata = metadata or {}
        schema = self.metadata.get(
            "feature_schema", PROGRAM_RERANKER_FEATURE_SCHEMA
        )
        if schema not in SUPPORTED_PROGRAM_RERANKER_FEATURE_SCHEMAS:
            raise ProgramRerankerError(
                f"Unsupported reranker feature schema: {schema!r}"
            )
        self.metadata["feature_schema"] = schema

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @property
    def residual_scale(self) -> float:
        raw = self.metadata.get("residual_scale", 0.25)
        return max(0.0, float(raw)) if isinstance(raw, (int, float)) else 0.25

    @property
    def override_margin(self) -> float:
        raw = self.metadata.get("override_margin", DISABLE_OVERRIDE_MARGIN)
        return (
            float(raw)
            if isinstance(raw, (int, float))
            else DISABLE_OVERRIDE_MARGIN
        )

    def override_margin_for_operation(self, operation: str | None) -> float:
        raw = self.metadata.get("override_margins_by_operation", {})
        if operation is not None and isinstance(raw, dict):
            value = raw.get(operation)
            if isinstance(value, (int, float)):
                return float(value)
        return self.override_margin

    def score_many(
        self, feature_rows: Sequence[Sequence[float]]
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return learned corrections and confidence probabilities."""

        if not feature_rows:
            return (), ()
        tensor = torch.tensor(feature_rows, dtype=torch.float32, device=self.device)
        mean = self.feature_mean.to(self.device)
        scale = self.feature_scale.to(self.device)
        self.model.eval()
        with torch.no_grad():
            corrections, confidence_logits = self.model((tensor - mean) / scale)
        return (
            tuple(float(value) for value in corrections.cpu().tolist()),
            tuple(
                float(value)
                for value in torch.sigmoid(confidence_logits).cpu().tolist()
            ),
        )

    def select(
        self,
        feature_rows: Sequence[Sequence[float]],
        *,
        operation: str | None = None,
    ) -> tuple[int, tuple[float, ...], tuple[float, ...], bool, float]:
        """Apply legacy + correction, guarded by the calibrated override margin."""

        if not feature_rows:
            raise ProgramRerankerError("Cannot select from an empty program list")
        corrections, confidences = self.score_many(feature_rows)
        legacy_scores = [
            float(row[FEATURE_INDEX["legacy_rank_score"]])
            for row in feature_rows
        ]
        legacy_candidates = [
            index
            for index, row in enumerate(feature_rows)
            if float(row[FEATURE_INDEX["legacy_winner"]]) >= 0.5
        ]
        if len(legacy_candidates) != 1:
            raise ProgramRerankerError(
                "Every reranker group must contain exactly one legacy winner"
            )
        legacy_index = legacy_candidates[0]
        combined = tuple(
            legacy + self.residual_scale * correction
            for legacy, correction in zip(legacy_scores, corrections)
        )
        alternative = max(range(len(combined)), key=lambda index: combined[index])
        improvement = combined[alternative] - combined[legacy_index]
        overridden = (
            alternative != legacy_index
            and improvement > 0.0
            and improvement >= self.override_margin_for_operation(operation)
        )
        winner = alternative if overridden else legacy_index
        return winner, combined, confidences, overridden, float(improvement)

    def threshold_for_operation(self, operation: str) -> float:
        raw = self.metadata.get("confidence_thresholds_by_operation", {})
        if isinstance(raw, dict):
            value = raw.get(operation)
            if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
                return float(value)
        value = self.metadata.get("confidence_threshold", 0.5)
        return float(value) if isinstance(value, (int, float)) else 0.5

    def validate_base_model(self, path: str | Path) -> None:
        expected = self.metadata.get("base_model_sha256")
        if not isinstance(expected, str) or not expected:
            return
        digest = sha256()
        with Path(path).expanduser().resolve().open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ProgramRerankerError(
                "V11 sidecar was trained for a different frozen HRM checkpoint"
            )

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        feature_schema = str(
            self.metadata.get(
                "feature_schema", PROGRAM_RERANKER_FEATURE_SCHEMA
            )
        )
        schema_version = (
            PROGRAM_RERANKER_SCHEMA_VERSION
            if feature_schema == PROGRAM_RERANKER_FEATURE_SCHEMA
            else 2
        )
        torch.save(
            {
                "schema_version": schema_version,
                "feature_schema": feature_schema,
                "feature_names": PROGRAM_RERANKER_FEATURES,
                "config": asdict(self.model.config),
                "state_dict": self.model.state_dict(),
                "feature_mean": self.feature_mean,
                "feature_scale": self.feature_scale,
                "metadata": self.metadata,
            },
            destination,
        )
        return destination

    @classmethod
    def load(
        cls, path: str | Path, *, device: str | torch.device = "cpu"
    ) -> "ProgramRerankerBundle":
        source = Path(path).expanduser().resolve()
        payload = torch.load(source, map_location=device, weights_only=False)
        schema_version = payload.get("schema_version")
        feature_schema = payload.get("feature_schema")
        if schema_version not in {2, PROGRAM_RERANKER_SCHEMA_VERSION}:
            raise ProgramRerankerError(
                "Unsupported program reranker checkpoint; regenerate the sidecar"
            )
        if feature_schema not in SUPPORTED_PROGRAM_RERANKER_FEATURE_SCHEMAS:
            raise ProgramRerankerError("Reranker feature schema is incompatible")
        if (
            schema_version == 2
            and feature_schema != LEGACY_PROGRAM_RERANKER_FEATURE_SCHEMA
        ) or (
            schema_version == PROGRAM_RERANKER_SCHEMA_VERSION
            and feature_schema != PROGRAM_RERANKER_FEATURE_SCHEMA
        ):
            raise ProgramRerankerError(
                "Reranker checkpoint schema and feature semantics disagree"
            )
        if tuple(payload.get("feature_names", ())) != PROGRAM_RERANKER_FEATURES:
            raise ProgramRerankerError("Reranker feature order is incompatible")
        config = ProgramRerankerConfig(**payload["config"])
        model = ProgramRerankerModel(config).to(device)
        model.load_state_dict(payload["state_dict"])
        return cls(
            model,
            feature_mean=payload["feature_mean"],
            feature_scale=payload["feature_scale"],
            metadata={
                **dict(payload.get("metadata", {})),
                "feature_schema": feature_schema,
            },
        )


def _normalization(
    records: Sequence[RerankerExampleRecord],
) -> tuple[Tensor, Tensor]:
    rows = [candidate.features for item in records for candidate in item.candidates]
    if not rows:
        raise ProgramRerankerError("No candidate features were provided")
    tensor = torch.tensor(rows, dtype=torch.float32)
    return tensor.mean(dim=0), tensor.std(dim=0, unbiased=False).clamp_min(1e-4)


def _balanced_epoch_examples(
    records: Sequence[RerankerExampleRecord], rng: random.Random
) -> list[RerankerExampleRecord]:
    groups: dict[str, list[RerankerExampleRecord]] = {}
    for record in records:
        if record.operation in RERANKER_TRAIN_OPERATIONS:
            groups.setdefault(record.operation, []).append(record)
    if not groups:
        return []
    target = max(len(group) for group in groups.values())
    output: list[RerankerExampleRecord] = []
    for operation in sorted(groups):
        group = groups[operation]
        output.extend(group)
        output.extend(rng.choice(group) for _ in range(target - len(group)))
    rng.shuffle(output)
    return output


def _legacy_index(record: RerankerExampleRecord) -> int:
    winners = [
        index
        for index, candidate in enumerate(record.candidates)
        if candidate.features[FEATURE_INDEX["legacy_winner"]] >= 0.5
    ]
    if len(winners) != 1:
        raise ProgramRerankerError(
            f"{record.example_id}: expected exactly one legacy winner"
        )
    return winners[0]


def _selection_metrics(
    bundle: ProgramRerankerBundle,
    records: Sequence[RerankerExampleRecord],
    *,
    use_thresholds: bool,
    training_operations_only: bool = False,
) -> tuple[float, float, float, float, int]:
    exact = execution = accepted = accepted_correct = overrides = 0
    considered = 0
    for record in records:
        if (
            training_operations_only
            and record.operation not in RERANKER_TRAIN_OPERATIONS
        ):
            continue
        if not record.candidates:
            continue
        winner, _, confidences, overridden, _ = bundle.select(
            [candidate.features for candidate in record.candidates],
            operation=record.operation,
        )
        candidate = record.candidates[winner]
        considered += 1
        exact += int(candidate.exact_gold)
        execution += int(candidate.execution_correct)
        overrides += int(overridden)
        is_accepted = (
            candidate.runtime_eligible
            and (
                not use_thresholds
                or confidences[winner]
                >= bundle.threshold_for_operation(record.operation)
            )
        )
        if is_accepted:
            accepted += 1
            accepted_correct += int(candidate.execution_correct)
    if not considered:
        return 0.0, 0.0, 0.0, 0.0, 0
    return (
        exact / considered,
        execution / considered,
        accepted / considered,
        accepted_correct / accepted if accepted else 0.0,
        overrides,
    )


def _legacy_metrics(
    records: Sequence[RerankerExampleRecord],
) -> tuple[float, float]:
    selected = [
        record.candidates[_legacy_index(record)]
        for record in records
        if record.operation in RERANKER_TRAIN_OPERATIONS and record.candidates
    ]
    if not selected:
        return 0.0, 0.0
    return (
        sum(item.exact_gold for item in selected) / len(selected),
        sum(item.execution_correct for item in selected) / len(selected),
    )


def _legacy_runtime_metrics(
    records: Sequence[RerankerExampleRecord],
) -> tuple[float, float]:
    selected = [
        record.candidates[_legacy_index(record)]
        for record in records
        if record.operation in RERANKER_TRAIN_OPERATIONS and record.candidates
    ]
    if not selected:
        return 0.0, 0.0
    accepted = [item for item in selected if item.runtime_eligible]
    coverage = len(accepted) / len(selected)
    precision = (
        sum(item.execution_correct for item in accepted) / len(accepted)
        if accepted
        else 0.0
    )
    return coverage, precision


def calibrate_override_margin(
    bundle: ProgramRerankerBundle,
    records: Sequence[RerankerExampleRecord],
) -> tuple[float, float, float, int]:
    """Choose a margin that cannot regress exact/execution on calibration."""

    legacy_exact, legacy_execution = _legacy_metrics(records)
    legacy_coverage, legacy_precision = _legacy_runtime_metrics(records)
    raw_margins: list[float] = []
    original_margin = bundle.metadata.get(
        "override_margin", DISABLE_OVERRIDE_MARGIN
    )
    bundle.metadata["override_margin"] = 0.0
    for record in records:
        if record.operation not in RERANKER_TRAIN_OPERATIONS or not record.candidates:
            continue
        _, _, _, overridden, margin = bundle.select(
            [candidate.features for candidate in record.candidates],
            operation=record.operation,
        )
        if overridden and margin > 0.0:
            raw_margins.append(margin)

    best = (
        legacy_execution,
        legacy_exact,
        legacy_coverage,
        legacy_precision,
        0,
        DISABLE_OVERRIDE_MARGIN,
    )
    for margin in [DISABLE_OVERRIDE_MARGIN, *sorted(set(raw_margins))]:
        bundle.metadata["override_margin"] = margin
        exact, execution, coverage, precision, overrides = _selection_metrics(
            bundle,
            records,
            use_thresholds=False,
            training_operations_only=True,
        )
        if (
            execution < legacy_execution
            or exact < legacy_exact
            or coverage < legacy_coverage
            or precision < legacy_precision
        ):
            continue
        candidate = (
            execution,
            exact,
            coverage,
            precision,
            -overrides,
            margin,
        )
        if candidate[:5] > best[:5]:
            best = candidate
    bundle.metadata["override_margin"] = original_margin
    return float(best[5]), float(best[1]), float(best[0]), int(-best[4])


def calibrate_reranker_thresholds(
    bundle: ProgramRerankerBundle,
    records: Sequence[RerankerExampleRecord],
    *,
    target_precision: float = 0.90,
    minimum_accepted: int = 3,
    stability_errors: int = 1,
    force_legacy: bool = False,
) -> dict[str, float]:
    if not 0.0 < target_precision <= 1.0:
        raise ProgramRerankerError("target_precision must be within (0, 1]")
    if stability_errors < 0:
        raise ProgramRerankerError("stability_errors must be non-negative")
    outcomes: dict[str, list[tuple[float, bool]]] = {}
    for record in records:
        if record.operation not in RERANKER_TRAIN_OPERATIONS or not record.candidates:
            continue
        feature_rows = [candidate.features for candidate in record.candidates]
        if force_legacy:
            winner = _legacy_index(record)
            _, confidences = bundle.score_many(feature_rows)
        else:
            winner, _, confidences, _, _ = bundle.select(
                feature_rows,
                operation=record.operation,
            )
        if not record.candidates[winner].runtime_eligible:
            continue
        outcomes.setdefault(record.operation, []).append(
            (confidences[winner], record.candidates[winner].execution_correct)
        )
    thresholds: dict[str, float] = {}
    for operation, rows in outcomes.items():
        choices: list[tuple[int, float, float]] = []
        for threshold in sorted({confidence for confidence, _ in rows}):
            selected = [
                correct for confidence, correct in rows if confidence >= threshold
            ]
            if len(selected) < minimum_accepted:
                continue
            precision = sum(selected) / len(selected)
            conservative = max(0, sum(selected) - stability_errors) / len(selected)
            if conservative >= target_precision:
                choices.append((len(selected), threshold, precision))
        if choices:
            _, threshold, _ = max(
                choices, key=lambda item: (item[0], item[2], item[1])
            )
        else:
            ranked = sorted(rows, reverse=True)
            best = max(
                (
                    (sum(correct for _, correct in ranked[:index]) / index, index)
                    for index in range(1, len(ranked) + 1)
                ),
                key=lambda item: (item[0], item[1]),
            )
            threshold = ranked[best[1] - 1][0]
        thresholds[operation] = float(threshold)
    return thresholds


def _reranker_thresholds_meet_target(
    bundle: ProgramRerankerBundle,
    records: Sequence[RerankerExampleRecord],
    *,
    target_precision: float,
    stability_errors: int = 1,
    thresholds: dict[str, float] | None = None,
    force_legacy: bool = False,
) -> bool:
    outcomes: dict[str, list[bool]] = {}
    for record in records:
        if record.operation not in RERANKER_TRAIN_OPERATIONS or not record.candidates:
            continue
        feature_rows = [candidate.features for candidate in record.candidates]
        if force_legacy:
            winner = _legacy_index(record)
            _, confidences = bundle.score_many(feature_rows)
        else:
            winner, _, confidences, _, _ = bundle.select(
                feature_rows,
                operation=record.operation,
            )
        threshold = (
            float(thresholds.get(record.operation, 1.0))
            if thresholds is not None
            else bundle.threshold_for_operation(record.operation)
        )
        if (
            record.candidates[winner].runtime_eligible
            and confidences[winner] >= threshold
        ):
            outcomes.setdefault(record.operation, []).append(
                record.candidates[winner].execution_correct
            )
    accepted = [correct for rows in outcomes.values() for correct in rows]
    if not accepted:
        return False
    return (
        max(0, sum(accepted) - stability_errors) / len(accepted)
        >= target_precision
    )


def _operation_threshold_target_status(
    bundle: ProgramRerankerBundle,
    records: Sequence[RerankerExampleRecord],
    *,
    target_precision: float,
    thresholds: dict[str, float] | None = None,
    force_legacy: bool = False,
    stability_errors: int = 1,
) -> dict[str, bool]:
    outcomes: dict[str, list[bool]] = {}
    for record in records:
        if (
            record.operation not in RERANKER_TRAIN_OPERATIONS
            or not record.candidates
        ):
            continue
        feature_rows = [candidate.features for candidate in record.candidates]
        if force_legacy:
            winner = _legacy_index(record)
            _, confidences = bundle.score_many(feature_rows)
        else:
            winner, _, confidences, _, _ = bundle.select(
                feature_rows, operation=record.operation
            )
        threshold = (
            float(thresholds.get(record.operation, 1.0))
            if thresholds is not None
            else bundle.threshold_for_operation(record.operation)
        )
        candidate = record.candidates[winner]
        if candidate.runtime_eligible and confidences[winner] >= threshold:
            outcomes.setdefault(record.operation, []).append(
                candidate.execution_correct
            )
    return {
        operation: (
            bool(rows)
            and max(0, sum(rows) - stability_errors) / len(rows)
            >= target_precision
        )
        for operation, rows in outcomes.items()
    }


def _confidence_metrics(
    bundle: ProgramRerankerBundle,
    records: Sequence[RerankerExampleRecord],
    thresholds: dict[str, float],
    *,
    force_legacy: bool,
) -> tuple[float, float]:
    considered = accepted = correct = 0
    for record in records:
        if (
            record.operation not in RERANKER_TRAIN_OPERATIONS
            or not record.candidates
        ):
            continue
        considered += 1
        feature_rows = [candidate.features for candidate in record.candidates]
        if force_legacy:
            winner = _legacy_index(record)
            _, confidences = bundle.score_many(feature_rows)
        else:
            winner, _, confidences, _, _ = bundle.select(
                feature_rows, operation=record.operation
            )
        candidate = record.candidates[winner]
        threshold = float(thresholds.get(record.operation, 1.0))
        if candidate.runtime_eligible and confidences[winner] >= threshold:
            accepted += 1
            correct += int(candidate.execution_correct)
    return (
        accepted / considered if considered else 0.0,
        correct / accepted if accepted else 0.0,
    )


def _train_confidence_head(
    bundle: ProgramRerankerBundle,
    records: Sequence[RerankerExampleRecord],
    *,
    epochs: int,
    learning_rate: float,
) -> None:
    if epochs <= 0:
        return
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    for parameter in bundle.model.confidence_backbone.parameters():
        parameter.requires_grad_(True)
    for parameter in bundle.model.confidence_head.parameters():
        parameter.requires_grad_(True)
    confidence_parameters = [
        *bundle.model.confidence_backbone.parameters(),
        *bundle.model.confidence_head.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        confidence_parameters,
        lr=learning_rate,
        weight_decay=1e-4,
    )
    mean = bundle.feature_mean.to(bundle.device)
    scale = bundle.feature_scale.to(bundle.device)
    for _ in range(epochs):
        bundle.model.train()
        for record in records:
            if record.operation not in RERANKER_TRAIN_OPERATIONS or not record.candidates:
                continue
            winner, _, _, _, _ = bundle.select(
                [candidate.features for candidate in record.candidates],
                operation=record.operation,
            )
            candidate = record.candidates[winner]
            bundle.model.train()
            features = torch.tensor(
                [candidate.features], dtype=torch.float32, device=bundle.device
            )
            _, confidence_logits = bundle.model((features - mean) / scale)
            target = torch.tensor(
                [float(candidate.execution_correct)],
                dtype=torch.float32,
                device=bundle.device,
            )
            loss = F.binary_cross_entropy_with_logits(confidence_logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    bundle.model.eval()


def train_program_reranker(
    train_records: Sequence[RerankerExampleRecord],
    calibration_records: Sequence[RerankerExampleRecord],
    *,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    hidden_size: int = 48,
    dropout: float = 0.10,
    listwise_weight: float = 1.0,
    pairwise_weight: float = 0.35,
    pairwise_margin: float = 0.5,
    residual_scale: float = 0.25,
    confidence_epochs: int = 5,
    confidence_learning_rate: float = 1e-3,
    target_precision: float = 0.90,
    seed: int = 17,
    device: str | torch.device = "cpu",
    metadata: dict[str, object] | None = None,
) -> tuple[ProgramRerankerBundle, ProgramRerankerTrainingReport]:
    if epochs < 1:
        raise ProgramRerankerError("epochs must be positive")
    if residual_scale <= 0:
        raise ProgramRerankerError("residual_scale must be positive")
    if not train_records or not calibration_records:
        raise ProgramRerankerError("Train and calibration records are required")
    train_rank_examples = sum(
        record.has_rank_positive for record in train_records
    )
    calibration_rank_examples = sum(
        record.has_rank_positive for record in calibration_records
    )
    if not train_rank_examples:
        raise ProgramRerankerError(
            "No training candidate group contains a canonical or grounded positive"
        )
    random.seed(seed)
    torch.manual_seed(seed)
    mean, scale = _normalization(train_records)
    config = ProgramRerankerConfig(hidden_size=hidden_size, dropout=dropout)
    model = ProgramRerankerModel(config).to(device)
    rank_parameters = [
        *model.rank_backbone.parameters(),
        *model.rank_head.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        rank_parameters, lr=learning_rate, weight_decay=1e-4
    )
    rng = random.Random(seed)
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    legacy_exact, legacy_execution = _legacy_metrics(calibration_records)
    legacy_coverage, legacy_precision = _legacy_runtime_metrics(
        calibration_records
    )
    best_state = initial_state
    best_score = (legacy_execution, legacy_exact, 0)
    best_epoch = 0
    best_margin = DISABLE_OVERRIDE_MARGIN
    best_margins_by_operation = {
        operation: DISABLE_OVERRIDE_MARGIN
        for operation in RERANKER_TRAIN_OPERATIONS
    }
    best_overrides = 0
    final_loss = 0.0
    mean_device = mean.to(device)
    scale_device = scale.to(device)

    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for record in _balanced_epoch_examples(train_records, rng):
            if not record.candidates or not record.has_rank_positive:
                continue
            features = torch.tensor(
                [candidate.features for candidate in record.candidates],
                dtype=torch.float32,
                device=device,
            )
            corrections, _ = model((features - mean_device) / scale_device)
            legacy_scores = features[:, FEATURE_INDEX["legacy_rank_score"]]
            combined = legacy_scores + residual_scale * corrections
            use_canonical = record.has_canonical_positive
            positive_mask = torch.tensor(
                [
                    candidate.positive_class == "canonical_exact"
                    if use_canonical
                    else candidate.positive_class == "grounded_equivalent"
                    for candidate in record.candidates
                ],
                dtype=torch.bool,
                device=device,
            )
            gold_indices = torch.nonzero(
                positive_mask, as_tuple=False
            ).flatten()
            gold_index = gold_indices[combined[gold_indices].argmax()]
            loss = listwise_weight * F.cross_entropy(
                combined.unsqueeze(0), gold_index.unsqueeze(0)
            )
            wrong_mask = ~positive_mask
            if bool(wrong_mask.any()):
                hardest_wrong = combined[wrong_mask].max()
                loss = loss + pairwise_weight * F.relu(
                    pairwise_margin - combined[gold_index] + hardest_wrong
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rank_parameters, 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        final_loss = sum(losses) / len(losses) if losses else 0.0
        current = ProgramRerankerBundle(
            model,
            feature_mean=mean,
            feature_scale=scale,
            metadata={
                "feature_schema": PROGRAM_RERANKER_FEATURE_SCHEMA,
                "residual_scale": residual_scale,
            },
        )
        operation_margins: dict[str, float] = {}
        for operation in sorted(RERANKER_TRAIN_OPERATIONS):
            operation_records = [
                record
                for record in calibration_records
                if record.operation == operation
            ]
            if not operation_records:
                continue
            margin, _, _, _ = calibrate_override_margin(
                current, operation_records
            )
            operation_margins[operation] = margin
        current.metadata["override_margins_by_operation"] = operation_margins
        exact, execution, _, _, overrides = _selection_metrics(
            current,
            calibration_records,
            use_thresholds=False,
            training_operations_only=True,
        )
        score = (execution, exact, -overrides)
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_margins_by_operation = operation_margins
            best_overrides = overrides
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    output_metadata = {
        "architecture": "v11-safe-residual-frozen-hrm-reranker",
        "feature_schema": PROGRAM_RERANKER_FEATURE_SCHEMA,
        "train_operations": sorted(RERANKER_TRAIN_OPERATIONS),
        "sum3_sum4_optimized": False,
        "unknown_as_automatic_negative": False,
        "selected_epoch": best_epoch,
        "target_precision": target_precision,
        "residual_scale": residual_scale,
        "override_margin": best_margin,
        "override_margins_by_operation": best_margins_by_operation,
        "legacy_fallback_enabled": True,
        **(metadata or {}),
    }
    bundle = ProgramRerankerBundle(
        model,
        feature_mean=mean,
        feature_scale=scale,
        metadata=output_metadata,
    )
    _train_confidence_head(
        bundle,
        train_records,
        epochs=confidence_epochs,
        learning_rate=confidence_learning_rate,
    )
    thresholds = calibrate_reranker_thresholds(
        bundle,
        calibration_records,
        target_precision=target_precision,
    )
    bundle.metadata["confidence_thresholds_by_operation"] = thresholds
    legacy_target_thresholds = calibrate_reranker_thresholds(
        bundle,
        calibration_records,
        target_precision=target_precision,
        force_legacy=True,
    )
    exact, execution, coverage, precision, overrides = _selection_metrics(
        bundle,
        calibration_records,
        use_thresholds=True,
        training_operations_only=True,
    )
    ranking_regression_guard_passed = (
        exact >= legacy_exact
        and execution >= legacy_execution
    )
    confidence_target_met = _reranker_thresholds_meet_target(
        bundle,
        calibration_records,
        target_precision=target_precision,
    )
    legacy_target_met = _reranker_thresholds_meet_target(
        bundle,
        calibration_records,
        target_precision=target_precision,
        thresholds=legacy_target_thresholds,
        force_legacy=True,
    )
    legacy_target_coverage, legacy_target_precision = _confidence_metrics(
        bundle,
        calibration_records,
        legacy_target_thresholds,
        force_legacy=True,
    )
    confidence_coverage_guard_passed = (
        confidence_target_met
        and (
            not legacy_target_met
            or coverage >= legacy_target_coverage
        )
    )
    regression_guard_passed = (
        ranking_regression_guard_passed
        and confidence_coverage_guard_passed
    )
    bundle.metadata.update(
        {
            "calibration_legacy_program_exact": legacy_exact,
            "calibration_legacy_execution": legacy_execution,
            "calibration_legacy_coverage": legacy_coverage,
            "calibration_legacy_accepted_precision": legacy_precision,
            "calibration_legacy_target_thresholds_by_operation": (
                legacy_target_thresholds
            ),
            "calibration_legacy_target_met": legacy_target_met,
            "calibration_legacy_target_coverage": legacy_target_coverage,
            "calibration_legacy_target_precision": legacy_target_precision,
            "calibration_program_exact": exact,
            "calibration_execution": execution,
            "calibration_overrides": overrides,
            "calibration_coverage": coverage,
            "calibration_accepted_precision": precision,
            "calibration_target_met": confidence_target_met,
            "calibration_target_met_by_operation": (
                _operation_threshold_target_status(
                    bundle,
                    calibration_records,
                    target_precision=target_precision,
                )
            ),
            "ranking_regression_guard_passed": (
                ranking_regression_guard_passed
            ),
            "confidence_coverage_guard_passed": (
                confidence_coverage_guard_passed
            ),
            "calibration_regression_guard": (
                ranking_regression_guard_passed
            ),
            "regression_guard_passed": regression_guard_passed,
        }
    )
    return bundle, ProgramRerankerTrainingReport(
        train_examples=len(train_records),
        train_rank_examples=train_rank_examples,
        calibration_examples=len(calibration_records),
        calibration_rank_examples=calibration_rank_examples,
        epochs=epochs,
        selected_epoch=best_epoch,
        final_loss=final_loss,
        calibration_legacy_program_exact=legacy_exact,
        calibration_legacy_execution=legacy_execution,
        calibration_legacy_coverage=legacy_coverage,
        calibration_legacy_accepted_precision=legacy_precision,
        calibration_legacy_target_coverage=legacy_target_coverage,
        calibration_legacy_target_precision=legacy_target_precision,
        calibration_program_exact=exact,
        calibration_execution=execution,
        calibration_overrides=best_overrides,
        override_margin=best_margin,
        override_margins_by_operation=best_margins_by_operation,
        calibration_coverage=coverage,
        calibration_accepted_precision=precision,
        confidence_epochs=confidence_epochs,
        ranking_regression_guard_passed=ranking_regression_guard_passed,
        confidence_target_met=confidence_target_met,
        confidence_coverage_guard_passed=(
            confidence_coverage_guard_passed
        ),
        regression_guard_passed=regression_guard_passed,
    )
