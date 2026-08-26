from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import blake2b
from pathlib import Path
from typing import Iterable

from .ontology import FORMULAS, VARIABLES, normalize_text
from .repository import Fact, FinancialRepository


MODEL_SCHEMA_VERSION = 1
STATEMENTS = ("income_statement", "balance_sheet", "cash_flow")
UNKNOWN_METRIC = "__unknown__"
METRIC_LABELS = (*FORMULAS.keys(), UNKNOWN_METRIC)


class NeuralModelError(ValueError):
    pass


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    margin: float


@dataclass(frozen=True)
class TrainingReport:
    samples: int
    epochs: int
    loss: float
    accuracy: float


class HashedSoftmaxClassifier:
    """A small trainable text classifier with hashed sparse n-gram inputs."""

    def __init__(
        self,
        labels: Iterable[str],
        feature_dim: int = 4096,
        seed: int = 17,
    ) -> None:
        self.labels = tuple(labels)
        if len(self.labels) < 2:
            raise NeuralModelError("Classifier needs at least two labels")
        if feature_dim < 128:
            raise NeuralModelError("feature_dim must be at least 128")
        self.feature_dim = feature_dim
        self.seed = seed
        rng = random.Random(seed)
        self.weights = [
            [rng.uniform(-0.002, 0.002) for _ in range(feature_dim)]
            for _ in self.labels
        ]
        self.bias = [0.0 for _ in self.labels]
        self._label_index = {label: index for index, label in enumerate(self.labels)}

    @property
    def parameter_count(self) -> int:
        return len(self.labels) * (self.feature_dim + 1)

    @property
    def nonzero_parameter_count(self) -> int:
        weights = sum(
            1 for row in self.weights for value in row if abs(value) > 1e-12
        )
        return weights + sum(1 for value in self.bias if abs(value) > 1e-12)

    def fit(
        self,
        samples: Iterable[tuple[str, str]],
        *,
        epochs: int = 60,
        learning_rate: float = 0.08,
        l2: float = 1e-5,
    ) -> TrainingReport:
        prepared = self._prepare_balanced(samples)
        if not prepared:
            raise NeuralModelError("No valid training samples")
        encoded = [
            (self._features(text), self._label_index[label])
            for text, label in prepared
        ]
        rng = random.Random(self.seed)
        last_loss = 0.0
        for epoch in range(epochs):
            rng.shuffle(encoded)
            rate = learning_rate / (1.0 + 0.035 * epoch)
            total_loss = 0.0
            for features, target in encoded:
                probabilities = self._probabilities_from_features(features)
                total_loss -= math.log(max(probabilities[target], 1e-12))
                for class_index, probability in enumerate(probabilities):
                    error = probability - (1.0 if class_index == target else 0.0)
                    self.bias[class_index] -= rate * error
                    row = self.weights[class_index]
                    for feature_index, value in features.items():
                        gradient = error * value + l2 * row[feature_index]
                        row[feature_index] -= rate * gradient
            last_loss = total_loss / len(encoded)
        correct = sum(
            self._predict_from_features(features).label == self.labels[target]
            for features, target in encoded
        )
        self._probabilities_for_text.cache_clear()
        return TrainingReport(
            samples=len(encoded),
            epochs=epochs,
            loss=last_loss,
            accuracy=correct / len(encoded),
        )

    def predict(self, text: str) -> Prediction:
        probabilities = self._probabilities_for_text(text)
        return self._prediction_from_probabilities(probabilities)

    def probability(self, text: str, label: str) -> float:
        if label not in self._label_index:
            return 0.0
        probabilities = self._probabilities_for_text(text)
        return probabilities[self._label_index[label]]

    def to_dict(self) -> dict[str, object]:
        return {
            "labels": self.labels,
            "feature_dim": self.feature_dim,
            "seed": self.seed,
            "weights": [
                [round(value, 9) for value in row]
                for row in self.weights
            ],
            "bias": [round(value, 9) for value in self.bias],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> HashedSoftmaxClassifier:
        model = cls(
            labels=payload["labels"],  # type: ignore[arg-type]
            feature_dim=int(payload["feature_dim"]),
            seed=int(payload.get("seed", 17)),
        )
        weights = payload.get("weights")
        bias = payload.get("bias")
        if not isinstance(weights, list) or not isinstance(bias, list):
            raise NeuralModelError("Invalid classifier weights")
        if len(weights) != len(model.labels) or len(bias) != len(model.labels):
            raise NeuralModelError("Classifier label/weight shape mismatch")
        parsed_weights = [[float(value) for value in row] for row in weights]
        if any(len(row) != model.feature_dim for row in parsed_weights):
            raise NeuralModelError("Classifier feature dimension mismatch")
        model.weights = parsed_weights
        model.bias = [float(value) for value in bias]
        return model

    def _prepare_balanced(
        self, samples: Iterable[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()
        for text, label in samples:
            normalized = normalize_text(text)
            pair = (normalized, label)
            if normalized and label in self._label_index and pair not in seen:
                grouped[label].append(normalized)
                seen.add(pair)
        missing = [label for label in self.labels if not grouped[label]]
        if missing:
            raise NeuralModelError("Missing samples for labels: " + ", ".join(missing))
        target_size = max(len(values) for values in grouped.values())
        balanced: list[tuple[str, str]] = []
        for label in self.labels:
            values = grouped[label]
            for index in range(target_size):
                balanced.append((values[index % len(values)], label))
        return balanced

    @lru_cache(maxsize=8192)
    def _features(self, text: str) -> dict[int, float]:
        normalized = normalize_text(text)
        if not normalized:
            return {}
        tokens = normalized.split()
        feature_names: list[str] = [f"w:{token}" for token in tokens]
        feature_names.extend(
            f"b:{tokens[index]}_{tokens[index + 1]}"
            for index in range(len(tokens) - 1)
        )
        compact = f"^{normalized}$"
        for size in (3, 4, 5):
            feature_names.extend(
                f"c{size}:{compact[index:index + size]}"
                for index in range(max(0, len(compact) - size + 1))
            )
        counts: Counter[int] = Counter(self._hash(name) for name in feature_names)
        norm = math.sqrt(sum(count * count for count in counts.values())) or 1.0
        return {index: count / norm for index, count in counts.items()}

    def _hash(self, feature: str) -> int:
        digest = blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.feature_dim

    def _scores(self, features: dict[int, float]) -> list[float]:
        return [
            self.bias[class_index]
            + sum(row[index] * value for index, value in features.items())
            for class_index, row in enumerate(self.weights)
        ]

    def _probabilities_from_features(
        self, features: dict[int, float]
    ) -> list[float]:
        scores = self._scores(features)
        maximum = max(scores)
        exponentials = [math.exp(score - maximum) for score in scores]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    def _predict_from_features(self, features: dict[int, float]) -> Prediction:
        probabilities = self._probabilities_from_features(features)
        return self._prediction_from_probabilities(probabilities)

    @lru_cache(maxsize=8192)
    def _probabilities_for_text(self, text: str) -> tuple[float, ...]:
        return tuple(self._probabilities_from_features(self._features(text)))

    def _prediction_from_probabilities(
        self, probabilities: list[float] | tuple[float, ...]
    ) -> Prediction:
        order = sorted(range(len(probabilities)), key=probabilities.__getitem__, reverse=True)
        best = order[0]
        second = order[1]
        return Prediction(
            label=self.labels[best],
            confidence=probabilities[best],
            margin=probabilities[best] - probabilities[second],
        )


@dataclass
class NeuralModelBundle:
    metric_model: HashedSoftmaxClassifier
    variable_model: HashedSoftmaxClassifier
    statement_model: HashedSoftmaxClassifier
    metadata: dict[str, object]

    @property
    def parameter_count(self) -> int:
        return (
            self.metric_model.parameter_count
            + self.variable_model.parameter_count
            + self.statement_model.parameter_count
        )

    @property
    def nonzero_parameter_count(self) -> int:
        return (
            self.metric_model.nonzero_parameter_count
            + self.variable_model.nonzero_parameter_count
            + self.statement_model.nonzero_parameter_count
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "metadata": self.metadata,
            "metric_model": self.metric_model.to_dict(),
            "variable_model": self.variable_model.to_dict(),
            "statement_model": self.statement_model.to_dict(),
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> NeuralModelBundle:
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Neural model not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != MODEL_SCHEMA_VERSION:
            raise NeuralModelError("Unsupported neural model schema")
        return cls(
            metric_model=HashedSoftmaxClassifier.from_dict(payload["metric_model"]),
            variable_model=HashedSoftmaxClassifier.from_dict(payload["variable_model"]),
            statement_model=HashedSoftmaxClassifier.from_dict(payload["statement_model"]),
            metadata=dict(payload.get("metadata", {})),
        )


METRIC_HINTS: dict[str, tuple[str, ...]] = {
    "gross_margin": (
        "lai gop tren doanh thu",
        "phan tram lai gop moi dong doanh thu",
        "hieu qua lai gop",
    ),
    "net_margin": (
        "loi nhuan sau thue tren doanh thu",
        "moi dong doanh thu giu lai bao nhieu lai rong",
        "ty le lai rong",
    ),
    "roa": (
        "loi nhuan so voi tong tai san",
        "hieu qua su dung tai san tao loi nhuan",
        "suc sinh loi cua tai san",
    ),
    "roe": (
        "loi nhuan so voi von co dong",
        "hieu qua von chu tao loi nhuan",
        "suc sinh loi cua von chu",
    ),
    "current_ratio": (
        "tai san ngan han bao phu no ngan han",
        "kha nang tra no ngan han",
        "thanh khoan ngan han bao nhieu lan",
        "tai san luu dong so voi cac khoan no den han",
        "nguon luc ngan han bao phu nghia vu ngan han",
        "kha nang thanh toan cac khoan no sap den han",
    ),
    "debt_to_equity": (
        "no phai tra so voi von chu",
        "no gap may lan von co dong",
        "don bay no tren von chu",
    ),
    "debt_ratio": (
        "no phai tra chiem bao nhieu tong tai san",
        "tai san duoc tai tro boi no",
        "ty trong no trong tai san",
    ),
    "ebitda": (
        "loi nhuan hoat dong cong khau hao",
        "loi nhuan truoc lai vay thue va khau hao",
        "dong tien ebitda uoc tinh",
    ),
    "dscr": (
        "dong tien bao phu nghia vu no",
        "kha nang tra lai va no goc",
        "nguon tien hoat dong so voi dich vu no",
    ),
}


METRIC_BENCHMARK: tuple[tuple[str, str], ...] = (
    ("HPG tao duoc bao nhieu lai gop tren mot dong ban hang nam 2025", "gross_margin"),
    ("FPT giu lai bao nhieu lai sau thue tu mot dong ban hang nam 2025", "net_margin"),
    ("tai san cua HPG tao ra loi nhuan hieu qua den dau nam 2025", "roa"),
    ("von cua co dong HPG sinh ra loi nhuan the nao nam 2025", "roe"),
    ("tai san luu dong co du bao phu cac khoan no sap den han cua HPG 2025", "current_ratio"),
    ("tong no cua HPG gap may lan phan von cua co dong nam 2025", "debt_to_equity"),
    ("bao nhieu phan tai san HPG nam 2025 duoc hinh thanh tu no", "debt_ratio"),
    ("uoc tinh loi nhuan hoat dong truoc khau hao cua HPG nam 2025", "ebitda"),
    ("nguon tien HPG co bao phu du lai vay va no goc nam 2025 khong", "dscr"),
)


def fact_text(fact: Fact) -> str:
    return " ".join(
        part
        for part in (fact.item_id, fact.item, fact.item_en, fact.statement)
        if part
    )


def train_model_bundle(
    repository: FinancialRepository | None = None,
    *,
    feature_dim: int = 4096,
    epochs: int = 60,
    learning_rate: float = 0.08,
    max_tickers: int = 160,
    seed: int = 17,
) -> tuple[NeuralModelBundle, dict[str, TrainingReport]]:
    metric_samples = _metric_samples()
    variable_samples, statement_samples = _variable_and_statement_samples(
        repository, max_tickers=max_tickers
    )
    metric_model = HashedSoftmaxClassifier(METRIC_LABELS, feature_dim, seed)
    variable_model = HashedSoftmaxClassifier(VARIABLES, feature_dim, seed + 1)
    statement_model = HashedSoftmaxClassifier(STATEMENTS, feature_dim, seed + 2)
    reports = {
        "metric": metric_model.fit(
            metric_samples, epochs=epochs, learning_rate=learning_rate
        ),
        "variable": variable_model.fit(
            variable_samples, epochs=epochs, learning_rate=learning_rate
        ),
        "statement": statement_model.fit(
            statement_samples, epochs=epochs, learning_rate=learning_rate
        ),
    }
    metadata: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": "hashed-ngram-softmax-3head",
        "feature_dim": feature_dim,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "max_tickers": max_tickers,
        "seed": seed,
        "training_samples": {
            name: report.samples for name, report in reports.items()
        },
        "training_accuracy": {
            name: report.accuracy for name, report in reports.items()
        },
    }
    return NeuralModelBundle(
        metric_model=metric_model,
        variable_model=variable_model,
        statement_model=statement_model,
        metadata=metadata,
    ), reports


def evaluate_model_bundle(
    bundle: NeuralModelBundle,
    repository: FinancialRepository | None = None,
    *,
    max_tickers: int = 100,
) -> dict[str, object]:
    metric_details = []
    for question, expected in METRIC_BENCHMARK:
        prediction = bundle.metric_model.predict(question)
        metric_details.append(
            {
                "question": question,
                "expected": expected,
                "predicted": prediction.label,
                "confidence": prediction.confidence,
                "correct": prediction.label == expected,
            }
        )
    result: dict[str, object] = {
        "metric_correct": sum(item["correct"] for item in metric_details),
        "metric_total": len(metric_details),
        "metric_details": metric_details,
    }
    if repository is None:
        return result

    primary_item_ids = {
        spec.item_ids[0]: key for key, spec in VARIABLES.items()
    }
    variable_correct = 0
    variable_total = 0
    statement_correct = 0
    statement_total = 0
    training_tickers = int(bundle.metadata.get("max_tickers", 0))
    held_out_tickers = repository.tickers()[
        training_tickers:training_tickers + max_tickers
    ]
    for ticker in held_out_tickers:
        years = repository.years(ticker)
        if not years:
            continue
        for fact in repository.facts(ticker, years[-1]):
            expected_variable = primary_item_ids.get(fact.item_id)
            if expected_variable:
                prediction = bundle.variable_model.predict(fact_text(fact))
                variable_correct += prediction.label == expected_variable
                variable_total += 1
            if fact.statement in STATEMENTS:
                prediction = bundle.statement_model.predict(fact_text(fact))
                statement_correct += prediction.label == fact.statement
                statement_total += 1
    result.update(
        {
            "variable_correct": variable_correct,
            "variable_total": variable_total,
            "statement_correct": statement_correct,
            "statement_total": statement_total,
        }
    )
    return result


def _metric_samples() -> list[tuple[str, str]]:
    templates = (
        "{phrase}",
        "tinh {phrase}",
        "cho biet {phrase} cua HPG nam 2025",
        "phan tich {phrase} cho doanh nghiep",
        "{phrase} la bao nhieu",
    )
    samples: list[tuple[str, str]] = []
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
                (template.format(phrase=phrase), key) for template in templates
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
            (template.format(phrase=phrase), UNKNOWN_METRIC)
            for template in templates
        )
    return samples


def _variable_and_statement_samples(
    repository: FinancialRepository | None,
    *,
    max_tickers: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    templates = (
        "{phrase}",
        "dong bao cao {phrase}",
        "financial item {phrase}",
        "gia tri {phrase} nam 2025",
    )
    variable_samples: list[tuple[str, str]] = []
    statement_samples: list[tuple[str, str]] = [
        ("bao cao ket qua kinh doanh income statement doanh thu loi nhuan", "income_statement"),
        ("bang can doi ke toan balance sheet tai san no von chu", "balance_sheet"),
        ("bao cao luu chuyen tien te cash flow dong tien", "cash_flow"),
    ]
    for key, spec in VARIABLES.items():
        phrases = {
            key.replace("_", " "),
            spec.label,
            spec.item_ids[0],
            *spec.aliases,
        }
        for phrase in phrases:
            variable_samples.extend(
                (template.format(phrase=phrase), key) for template in templates
            )
            if spec.statement:
                statement_samples.append((phrase, spec.statement))

    if repository is None:
        return variable_samples, statement_samples

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
                variable_samples.append((fact_text(fact), variable_key))
            if fact.statement in STATEMENTS:
                statement_samples.append((fact_text(fact), fact.statement))
    return variable_samples, statement_samples
