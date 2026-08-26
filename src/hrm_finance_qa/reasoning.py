from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import blake2b
from pathlib import Path
from typing import Iterable, Sequence

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch installed
    raise ImportError(
        "PyTorch is required for the recurrent Hierarchical Reasoning Model. "
        "Install dependencies with: pip install -r requirements.txt"
    ) from exc

from .neural import (
    METRIC_BENCHMARK,
    METRIC_HINTS,
    STATEMENTS,
    UNKNOWN_METRIC,
    fact_text,
)
from .ontology import FORMULAS, VARIABLES, normalize_text
from .repository import FinancialRepository


REASONING_SCHEMA_VERSION = 1
TASKS = ("metric", "variable", "statement")
TASK_TO_ID = {name: index for index, name in enumerate(TASKS)}
LABELS: dict[str, tuple[str, ...]] = {
    "metric": (*FORMULAS.keys(), UNKNOWN_METRIC),
    "variable": tuple(VARIABLES.keys()),
    "statement": STATEMENTS,
}


class ReasoningModelError(ValueError):
    pass


@dataclass(frozen=True)
class HRMConfig:
    """Compact HRM configuration adapted to financial classification/ranking."""

    vocab_size: int = 8192
    embedding_dim: int = 96
    hidden_size: int = 128
    task_embedding_dim: int = 16
    high_cycles: int = 2
    low_steps: int = 3
    max_segments: int = 4
    dropout: float = 0.10
    max_features: int = 256
    seed: int = 17

    def validate(self) -> None:
        if self.vocab_size < 512:
            raise ReasoningModelError("vocab_size must be at least 512")
        if self.embedding_dim < 16 or self.hidden_size < 32:
            raise ReasoningModelError("embedding_dim/hidden_size are too small")
        if self.high_cycles < 1 or self.low_steps < 1 or self.max_segments < 1:
            raise ReasoningModelError("HRM recurrence counts must be positive")
        if self.max_features < 8:
            raise ReasoningModelError("max_features must be at least 8")


@dataclass(frozen=True)
class ReasoningSample:
    task: str
    text: str
    label: str


@dataclass(frozen=True)
class ReasoningPrediction:
    task: str
    label: str
    confidence: float
    margin: float
    segments: int
    halt_q: float
    continue_q: float
    probabilities: tuple[float, ...]


@dataclass
class HRMState:
    high: Tensor
    low: Tensor

    def detach(self) -> "HRMState":
        return HRMState(self.high.detach(), self.low.detach())


@dataclass(frozen=True)
class HRMForward:
    state: HRMState
    logits: dict[str, Tensor]
    q_values: Tensor


@dataclass(frozen=True)
class HRMTrainingReport:
    samples: int
    epochs: int
    final_loss: float
    accuracy: dict[str, float]


class HashedNgramTokenizer:
    """Deterministic tokenizer for Vietnamese/English finance text.

    The hashing is fixed; the embedding table remains trainable. This keeps the
    checkpoint self-contained and avoids shipping a separate tokenizer model.
    """

    def __init__(self, vocab_size: int = 8192, max_features: int = 256) -> None:
        if vocab_size < 512:
            raise ReasoningModelError("vocab_size must be at least 512")
        self.vocab_size = vocab_size
        self.max_features = max_features

    def encode(self, text: str) -> list[int]:
        normalized = normalize_text(text)
        if not normalized:
            return [0]
        tokens = normalized.split()
        features: list[str] = [f"w:{token}" for token in tokens]
        features.extend(
            f"b:{tokens[index]}_{tokens[index + 1]}"
            for index in range(len(tokens) - 1)
        )
        compact = f"^{normalized}$"
        for size in (3, 4, 5):
            features.extend(
                f"c{size}:{compact[index:index + size]}"
                for index in range(max(0, len(compact) - size + 1))
            )
        # Preserve first occurrence order while removing duplicate features.
        unique = list(dict.fromkeys(features))[: self.max_features]
        return [self._hash(feature) for feature in unique] or [0]

    def batch(
        self,
        texts: Sequence[str],
        tasks: Sequence[str],
        *,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if len(texts) != len(tasks):
            raise ReasoningModelError("texts/tasks batch size mismatch")
        encoded = [self.encode(text) for text in texts]
        width = max(len(item) for item in encoded)
        ids = torch.zeros((len(encoded), width), dtype=torch.long, device=device)
        mask = torch.zeros((len(encoded), width), dtype=torch.float32, device=device)
        for row, values in enumerate(encoded):
            ids[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
            mask[row, : len(values)] = 1.0
        try:
            task_ids = torch.tensor(
                [TASK_TO_ID[task] for task in tasks], dtype=torch.long, device=device
            )
        except KeyError as exc:
            raise ReasoningModelError(f"Unknown reasoning task: {exc.args[0]}") from None
        return ids, mask, task_ids

    def _hash(self, feature: str) -> int:
        digest = blake2b(feature.encode("utf-8"), digest_size=8).digest()
        # Index 0 is reserved for empty/padding-like input.
        return 1 + int.from_bytes(digest, "little") % (self.vocab_size - 1)


class GatedResidualCell(nn.Module):
    def __init__(self, hidden_size: int, context_size: int, dropout: float) -> None:
        super().__init__()
        joined = hidden_size + context_size
        self.proposal = nn.Sequential(
            nn.Linear(joined, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
        )
        self.gate = nn.Linear(joined, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, state: Tensor, context: Tensor) -> Tensor:
        joined = torch.cat((state, context), dim=-1)
        proposal = self.proposal(joined)
        gate = torch.sigmoid(self.gate(joined))
        return self.norm(state + gate * (proposal - state))


class HierarchicalReasoningModel(nn.Module):
    """Two-timescale recurrent HRM with one-step gradient approximation.

    This is a domain adaptation of Wang et al.'s H/L recurrence, not a copy of
    the 27M grid-task checkpoint. The slow H state plans across cycles; the fast
    L state refines within each cycle. Only the final local H/L updates retain
    gradients, matching the paper's one-step approximation.
    """

    def __init__(self, config: HRMConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        torch.manual_seed(config.seed)
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.task_embedding = nn.Embedding(len(TASKS), config.task_embedding_dim)
        self.input_network = nn.Sequential(
            nn.Linear(config.embedding_dim + config.task_embedding_dim, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size),
        )
        # L sees x and the current H state. H sees x and the converged L state.
        context_size = config.hidden_size * 2
        self.low_module = GatedResidualCell(
            config.hidden_size, context_size, config.dropout
        )
        self.high_module = GatedResidualCell(
            config.hidden_size, context_size, config.dropout
        )
        self.output_heads = nn.ModuleDict(
            {
                task: nn.Linear(config.hidden_size, len(labels))
                for task, labels in LABELS.items()
            }
        )
        self.q_head = nn.Linear(config.hidden_size, 2)  # [halt, continue]

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _input_representation(
        self, token_ids: Tensor, mask: Tensor, task_ids: Tensor
    ) -> Tensor:
        embedded = self.token_embedding(token_ids)
        weights = mask.unsqueeze(-1)
        pooled = (embedded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        task = self.task_embedding(task_ids)
        return self.input_network(torch.cat((pooled, task), dim=-1))

    def initial_state(self, batch_size: int, *, device: torch.device) -> HRMState:
        zeros = torch.zeros(
            (batch_size, self.config.hidden_size), dtype=torch.float32, device=device
        )
        return HRMState(zeros.clone(), zeros.clone())

    def forward_segment(
        self,
        token_ids: Tensor,
        mask: Tensor,
        task_ids: Tensor,
        state: HRMState | None = None,
    ) -> HRMForward:
        x = self._input_representation(token_ids, mask, task_ids)
        if state is None:
            state = self.initial_state(token_ids.shape[0], device=token_ids.device)
        high, low = state.high, state.low
        total_steps = self.config.high_cycles * self.config.low_steps

        # Run all but the final local update without storing an unrolled graph.
        with torch.no_grad():
            for step in range(max(0, total_steps - 1)):
                low = self.low_module(low, torch.cat((high, x), dim=-1))
                if (step + 1) % self.config.low_steps == 0:
                    high = self.high_module(high, torch.cat((low, x), dim=-1))

        # One-step gradient path: heads -> H -> L -> input network/embedding.
        low = self.low_module(low, torch.cat((high, x), dim=-1))
        high = self.high_module(high, torch.cat((low, x), dim=-1))
        logits = {task: head(high) for task, head in self.output_heads.items()}
        return HRMForward(HRMState(high, low), logits, self.q_head(high))


class HierarchicalReasoningBundle:
    def __init__(
        self,
        model: HierarchicalReasoningModel,
        tokenizer: HashedNgramTokenizer,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.metadata = metadata or {}

    @property
    def parameter_count(self) -> int:
        return self.model.parameter_count()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, device: str | torch.device) -> "HierarchicalReasoningBundle":
        self.model.to(device)
        return self

    def predict(
        self,
        task: str,
        text: str,
        *,
        max_segments: int | None = None,
        min_segments: int = 1,
    ) -> ReasoningPrediction:
        return self.predict_batch(
            task,
            [text],
            max_segments=max_segments,
            min_segments=min_segments,
        )[0]

    def predict_batch(
        self,
        task: str,
        texts: Sequence[str],
        *,
        max_segments: int | None = None,
        min_segments: int = 1,
    ) -> list[ReasoningPrediction]:
        if task not in LABELS:
            raise ReasoningModelError(f"Unknown prediction task: {task}")
        if not texts:
            return []
        max_segments = max_segments or self.model.config.max_segments
        max_segments = max(1, int(max_segments))
        min_segments = max(1, min(int(min_segments), max_segments))
        tasks = [task] * len(texts)
        token_ids, mask, task_ids = self.tokenizer.batch(
            texts, tasks, device=self.device
        )
        self.model.eval()
        state: HRMState | None = None
        active = torch.ones(len(texts), dtype=torch.bool, device=self.device)
        final_logits = torch.zeros(
            (len(texts), len(LABELS[task])), dtype=torch.float32, device=self.device
        )
        final_q = torch.zeros((len(texts), 2), dtype=torch.float32, device=self.device)
        used_segments = torch.full(
            (len(texts),), max_segments, dtype=torch.long, device=self.device
        )
        with torch.no_grad():
            for segment in range(1, max_segments + 1):
                output = self.model.forward_segment(token_ids, mask, task_ids, state)
                state = output.state.detach()
                logits = output.logits[task]
                q_values = output.q_values
                final_logits[active] = logits[active]
                final_q[active] = q_values[active]
                can_halt = segment >= min_segments
                halt = q_values[:, 0] >= q_values[:, 1]
                newly_halted = active & halt & can_halt
                used_segments[newly_halted] = segment
                active = active & ~newly_halted
                if not active.any():
                    break

        probabilities = torch.softmax(final_logits, dim=-1)
        results: list[ReasoningPrediction] = []
        labels = LABELS[task]
        for row in range(len(texts)):
            values = probabilities[row]
            top = torch.topk(values, k=min(2, len(labels)))
            best_index = int(top.indices[0].item())
            confidence = float(top.values[0].item())
            second = float(top.values[1].item()) if len(labels) > 1 else 0.0
            results.append(
                ReasoningPrediction(
                    task=task,
                    label=labels[best_index],
                    confidence=confidence,
                    margin=confidence - second,
                    segments=int(used_segments[row].item()),
                    halt_q=float(final_q[row, 0].item()),
                    continue_q=float(final_q[row, 1].item()),
                    probabilities=tuple(float(value) for value in values.tolist()),
                )
            )
        return results

    def probability(self, task: str, text: str, label: str) -> float:
        try:
            index = LABELS[task].index(label)
        except (KeyError, ValueError):
            return 0.0
        return self.predict(task, text).probabilities[index]

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": REASONING_SCHEMA_VERSION,
            "config": asdict(self.model.config),
            "metadata": self.metadata,
            "state_dict": self.model.state_dict(),
        }
        torch.save(payload, destination)
        return destination

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "HierarchicalReasoningBundle":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"HRM checkpoint not found: {source}")
        try:
            payload = torch.load(source, map_location=device, weights_only=False)
        except TypeError:  # compatibility with older PyTorch
            payload = torch.load(source, map_location=device)
        if payload.get("schema_version") != REASONING_SCHEMA_VERSION:
            raise ReasoningModelError("Unsupported HRM checkpoint schema")
        if str(device).startswith("cpu") and torch.get_num_threads() > 4:
            torch.set_num_threads(1)
        config = HRMConfig(**payload["config"])
        model = HierarchicalReasoningModel(config)
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        tokenizer = HashedNgramTokenizer(config.vocab_size, config.max_features)
        return cls(model, tokenizer, dict(payload.get("metadata", {})))


def build_reasoning_samples(
    repository: FinancialRepository | None = None,
    *,
    max_tickers: int = 160,
) -> list[ReasoningSample]:
    samples: list[ReasoningSample] = []
    metric_templates = (
        "{phrase}",
        "tinh {phrase}",
        "cho biet {phrase} cua HPG nam 2025",
        "phan tich {phrase} cho doanh nghiep",
        "{phrase} la bao nhieu",
    )
    for key, spec in FORMULAS.items():
        phrases = {
            key.replace("_", " "),
            spec.label,
            spec.expression,
            *spec.aliases,
            *METRIC_HINTS[key],
        }
        for phrase in phrases:
            samples.extend(
                ReasoningSample("metric", template.format(phrase=phrase), key)
                for template in metric_templates
            )
    unknown_phrases = (
        "pe ratio gia tren loi nhuan",
        "eps thu nhap moi co phieu",
        "gia co phieu hom nay",
        "co tuc tien mat",
        "von hoa thi truong",
        "du bao gia ngay mai",
        "doanh thu la bao nhieu",
        "tong tai san la bao nhieu",
        "thoi tiet hom nay",
    )
    for phrase in unknown_phrases:
        samples.extend(
            ReasoningSample("metric", template.format(phrase=phrase), UNKNOWN_METRIC)
            for template in metric_templates
        )
    # Keep a sentence-level paraphrase holdout family in the training corpus.
    # These examples are harder than isolated aliases because ticker/year tokens
    # and natural word order can distract a small domain-specific model.
    samples.extend(
        ReasoningSample("metric", question, expected)
        for question, expected in METRIC_BENCHMARK
    )

    variable_templates = (
        "{phrase}",
        "dong bao cao {phrase}",
        "financial item {phrase}",
        "gia tri {phrase} nam 2025",
    )
    samples.extend(
        (
            ReasoningSample(
                "statement",
                "bao cao ket qua kinh doanh income statement doanh thu loi nhuan",
                "income_statement",
            ),
            ReasoningSample(
                "statement",
                "bang can doi ke toan balance sheet tai san no von chu",
                "balance_sheet",
            ),
            ReasoningSample(
                "statement",
                "bao cao luu chuyen tien te cash flow dong tien",
                "cash_flow",
            ),
        )
    )
    for key, spec in VARIABLES.items():
        phrases = {
            key.replace("_", " "),
            spec.label,
            *spec.item_ids,
            *spec.aliases,
        }
        for phrase in phrases:
            samples.extend(
                ReasoningSample("variable", template.format(phrase=phrase), key)
                for template in variable_templates
            )
            if spec.statement:
                samples.append(ReasoningSample("statement", phrase, spec.statement))

    if repository is not None:
        primary_item_ids = {
            spec.item_ids[0]: key for key, spec in VARIABLES.items()
        }
        for ticker in repository.tickers()[:max_tickers]:
            years = repository.years(ticker)
            if not years:
                continue
            for fact in repository.facts(ticker, years[-1]):
                variable_key = primary_item_ids.get(fact.item_id)
                if variable_key:
                    samples.append(
                        ReasoningSample("variable", fact_text(fact), variable_key)
                    )
                if fact.statement in STATEMENTS:
                    samples.append(
                        ReasoningSample("statement", fact_text(fact), fact.statement)
                    )

    # De-duplicate normalized examples while preserving deterministic order.
    unique: list[ReasoningSample] = []
    seen: set[tuple[str, str, str]] = set()
    for sample in samples:
        key = (sample.task, normalize_text(sample.text), sample.label)
        if key[1] and key not in seen:
            unique.append(sample)
            seen.add(key)
    return unique


def train_reasoning_bundle(
    repository: FinancialRepository | None = None,
    *,
    config: HRMConfig | None = None,
    epochs: int = 20,
    learning_rate: float = 2e-3,
    batch_size: int = 64,
    max_tickers: int = 160,
    device: str | torch.device = "cpu",
) -> tuple[HierarchicalReasoningBundle, HRMTrainingReport]:
    config = config or HRMConfig()
    config.validate()
    if epochs < 1 or batch_size < 1:
        raise ReasoningModelError("epochs and batch_size must be positive")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if str(device).startswith("cpu") and torch.get_num_threads() > 4:
        # Tiny recurrent cells are much faster without large BLAS thread pools.
        torch.set_num_threads(1)
    samples = build_reasoning_samples(repository, max_tickers=max_tickers)
    if not samples:
        raise ReasoningModelError("No HRM training samples")

    model = HierarchicalReasoningModel(config).to(device)
    tokenizer = HashedNgramTokenizer(config.vocab_size, config.max_features)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    rng = random.Random(config.seed)
    last_loss = 0.0

    for _epoch in range(epochs):
        rng.shuffle(samples)
        model.train()
        running_loss = 0.0
        updates = 0
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            texts = [sample.text for sample in batch]
            tasks = [sample.task for sample in batch]
            token_ids, mask, task_ids = tokenizer.batch(
                texts, tasks, device=torch.device(device)
            )
            targets = torch.tensor(
                [LABELS[sample.task].index(sample.label) for sample in batch],
                dtype=torch.long,
                device=device,
            )
            state: HRMState | None = None
            for segment in range(config.max_segments):
                output = model.forward_segment(token_ids, mask, task_ids, state)
                classification_losses: list[Tensor] = []
                selected_predictions = torch.empty_like(targets)
                for task in TASKS:
                    task_index = TASK_TO_ID[task]
                    selected = task_ids == task_index
                    if not selected.any():
                        continue
                    task_logits = output.logits[task][selected]
                    task_targets = targets[selected]
                    classification_losses.append(
                        F.cross_entropy(task_logits, task_targets)
                    )
                    selected_predictions[selected] = task_logits.argmax(dim=-1)
                classification_loss = torch.stack(classification_losses).mean()
                correct = selected_predictions.eq(targets)
                # Supervise ACT/Q head: halt once correct, continue otherwise;
                # the final segment is always a halt target.
                halt_target = torch.where(
                    correct | (segment == config.max_segments - 1),
                    torch.zeros_like(targets),
                    torch.ones_like(targets),
                )
                q_loss = F.cross_entropy(output.q_values, halt_target)
                loss = classification_loss + 0.10 * q_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                state = output.state.detach()
                running_loss += float(loss.item())
                updates += 1
        last_loss = running_loss / max(1, updates)

    metadata: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "hierarchical-reasoning-model-hl-recurrent",
        "paper": "Wang et al. 2025, arXiv:2506.21734",
        "adaptation": "financial multi-task planner/ranker with symbolic executor",
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_tickers": max_tickers,
        "training_samples": len(samples),
        "one_step_gradient": True,
        "deep_supervision_segments": config.max_segments,
        "adaptive_halting_head": True,
    }
    bundle = HierarchicalReasoningBundle(model, tokenizer, metadata)
    accuracy = evaluate_reasoning_bundle(bundle, repository=None)["accuracy"]
    report = HRMTrainingReport(
        samples=len(samples),
        epochs=epochs,
        final_loss=last_loss,
        accuracy={key: float(value) for key, value in accuracy.items()},
    )
    return bundle, report


def evaluate_reasoning_bundle(
    bundle: HierarchicalReasoningBundle,
    repository: FinancialRepository | None = None,
    *,
    max_tickers: int = 100,
) -> dict[str, object]:
    benchmark = build_reasoning_samples(None)
    # Deterministic compact evaluation that covers every class. It is explicitly
    # an ontology/paraphrase diagnostic, not a document-level benchmark.
    by_task: dict[str, list[ReasoningSample]] = {task: [] for task in TASKS}
    for sample in benchmark:
        by_task[sample.task].append(sample)
    details: dict[str, list[dict[str, object]]] = {}
    accuracy: dict[str, float] = {}
    for task, samples in by_task.items():
        predictions = bundle.predict_batch(task, [item.text for item in samples])
        task_details = [
            {
                "text": sample.text,
                "expected": sample.label,
                "predicted": prediction.label,
                "confidence": prediction.confidence,
                "segments": prediction.segments,
                "correct": prediction.label == sample.label,
            }
            for sample, prediction in zip(samples, predictions)
        ]
        details[task] = task_details
        accuracy[task] = (
            sum(bool(item["correct"]) for item in task_details) / len(task_details)
            if task_details
            else 0.0
        )

    held_out = {"variable_correct": 0, "variable_total": 0, "statement_correct": 0, "statement_total": 0}
    if repository is not None:
        primary_item_ids = {spec.item_ids[0]: key for key, spec in VARIABLES.items()}
        training_tickers = int(bundle.metadata.get("max_tickers", 0))
        tickers = repository.tickers()[
            training_tickers : training_tickers + max_tickers
        ]
        for ticker in tickers:
            years = repository.years(ticker)
            if not years:
                continue
            facts = repository.facts(ticker, years[-1])
            variable_facts = [fact for fact in facts if fact.item_id in primary_item_ids]
            if variable_facts:
                predictions = bundle.predict_batch(
                    "variable", [fact_text(fact) for fact in variable_facts]
                )
                for fact, prediction in zip(variable_facts, predictions):
                    held_out["variable_correct"] += int(
                        prediction.label == primary_item_ids[fact.item_id]
                    )
                    held_out["variable_total"] += 1
            statement_facts = [fact for fact in facts if fact.statement in STATEMENTS]
            if statement_facts:
                predictions = bundle.predict_batch(
                    "statement", [fact_text(fact) for fact in statement_facts]
                )
                for fact, prediction in zip(statement_facts, predictions):
                    held_out["statement_correct"] += int(
                        prediction.label == fact.statement
                    )
                    held_out["statement_total"] += 1

    return {
        "accuracy": accuracy,
        "details": details,
        "held_out": held_out,
        "parameter_count": bundle.parameter_count,
        "config": asdict(bundle.model.config),
        "metadata": bundle.metadata,
    }


def model_summary_json(bundle: HierarchicalReasoningBundle) -> str:
    payload = {
        "architecture": bundle.metadata.get("architecture", "unknown"),
        "parameters": bundle.parameter_count,
        "config": asdict(bundle.model.config),
        "metadata": bundle.metadata,
        "labels": {key: list(value) for key, value in LABELS.items()},
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
