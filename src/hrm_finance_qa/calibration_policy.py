from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Mapping


class CalibrationPolicyError(ValueError):
    """Raised when a serialized or in-memory calibration policy is invalid."""


class CalibrationStatus(str, Enum):
    CALIBRATED = "CALIBRATED"
    DISABLED = "DISABLED"
    UNCALIBRATED = "UNCALIBRATED"


_POLICY_FIELDS = frozenset(
    {
        "status",
        "threshold",
        "support",
        "accepted",
        "correct",
        "precision",
        "lower_bound",
        "target",
    }
)


def _bounded_probability(name: str, value: object, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationPolicyError(f"{name} must be a finite number")
    result = float(value)
    lower_valid = result > 0.0 if positive else result >= 0.0
    if not math.isfinite(result) or not lower_valid or result > 1.0:
        interval = "(0, 1]" if positive else "[0, 1]"
        raise CalibrationPolicyError(f"{name} must be within {interval}")
    return result


def _nonnegative_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CalibrationPolicyError(f"{name} must be a non-negative integer")
    return value


def wilson_lower_bound(correct: int, total: int, *, z: float = 1.959963984540054) -> float:
    """Return the two-sided 95% Wilson lower bound by default.

    The bound is deterministic and dependency-free, which keeps checkpoint
    calibration reproducible across the training and public-runtime paths.
    """

    successes = _nonnegative_count("correct", correct)
    trials = _nonnegative_count("total", total)
    if successes > trials:
        raise CalibrationPolicyError("correct cannot exceed total")
    if isinstance(z, bool) or not isinstance(z, (int, float)):
        raise CalibrationPolicyError("z must be a positive finite number")
    z_value = float(z)
    if not math.isfinite(z_value) or z_value <= 0.0:
        raise CalibrationPolicyError("z must be a positive finite number")
    if trials == 0:
        return 0.0
    proportion = successes / trials
    z_squared = z_value * z_value
    denominator = 1.0 + z_squared / trials
    center = proportion + z_squared / (2.0 * trials)
    radius = z_value * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + z_squared / (4.0 * trials * trials)
    )
    return max(0.0, (center - radius) / denominator)


@dataclass(frozen=True)
class CalibrationPolicy:
    """One explicit, auditable confidence operating point.

    ``threshold`` is deliberately absent for disabled and uncalibrated
    policies.  This prevents a numeric sentinel such as ``1.0`` from silently
    changing an unavailable operating point into a runtime acceptance rule.
    """

    status: CalibrationStatus
    threshold: float | None
    support: int
    accepted: int
    correct: int
    precision: float
    lower_bound: float
    target: float

    def __post_init__(self) -> None:
        try:
            status = CalibrationStatus(self.status)
        except (TypeError, ValueError) as exc:
            raise CalibrationPolicyError(f"unknown calibration status: {self.status!r}") from exc
        object.__setattr__(self, "status", status)

        support = _nonnegative_count("support", self.support)
        accepted = _nonnegative_count("accepted", self.accepted)
        correct = _nonnegative_count("correct", self.correct)
        if accepted > support:
            raise CalibrationPolicyError("accepted cannot exceed support")
        if correct > accepted:
            raise CalibrationPolicyError("correct cannot exceed accepted")

        precision = _bounded_probability("precision", self.precision)
        lower_bound = _bounded_probability("lower_bound", self.lower_bound)
        target = _bounded_probability("target", self.target, positive=True)
        expected_precision = correct / accepted if accepted else 0.0
        if not math.isclose(
            precision,
            expected_precision,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise CalibrationPolicyError(
                "precision must equal correct / accepted (or 0 when accepted is 0)"
            )
        if lower_bound > precision + 1e-12:
            raise CalibrationPolicyError("lower_bound cannot exceed precision")

        if status is CalibrationStatus.CALIBRATED:
            if self.threshold is None:
                raise CalibrationPolicyError(
                    "a CALIBRATED policy requires a numeric threshold"
                )
            threshold = _bounded_probability("threshold", self.threshold)
            if accepted == 0:
                raise CalibrationPolicyError(
                    "a CALIBRATED policy requires at least one accepted example"
                )
            if lower_bound + 1e-12 < target:
                raise CalibrationPolicyError(
                    "a CALIBRATED policy must meet its target lower bound"
                )
            object.__setattr__(self, "threshold", threshold)
        elif self.threshold is not None:
            raise CalibrationPolicyError(
                f"a {status.value} policy must use a null threshold"
            )

        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "lower_bound", lower_bound)
        object.__setattr__(self, "target", target)

    @property
    def is_calibrated(self) -> bool:
        return self.status is CalibrationStatus.CALIBRATED

    def accepts(self, confidence: float) -> bool:
        """Return whether ``confidence`` reaches an enabled operating point."""

        if not self.is_calibrated:
            return False
        value = _bounded_probability("confidence", confidence)
        assert self.threshold is not None
        return value >= self.threshold

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "threshold": self.threshold,
            "support": self.support,
            "accepted": self.accepted,
            "correct": self.correct,
            "precision": self.precision,
            "lower_bound": self.lower_bound,
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "CalibrationPolicy":
        if not isinstance(payload, Mapping):
            raise CalibrationPolicyError("calibration policy must be a JSON object")
        fields = set(payload)
        missing = _POLICY_FIELDS - fields
        extra = fields - _POLICY_FIELDS
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append("missing=" + ",".join(sorted(missing)))
            if extra:
                details.append("extra=" + ",".join(sorted(extra)))
            raise CalibrationPolicyError(
                "invalid calibration policy fields: " + "; ".join(details)
            )
        return cls(
            status=payload["status"],  # type: ignore[arg-type]
            threshold=payload["threshold"],  # type: ignore[arg-type]
            support=payload["support"],  # type: ignore[arg-type]
            accepted=payload["accepted"],  # type: ignore[arg-type]
            correct=payload["correct"],  # type: ignore[arg-type]
            precision=payload["precision"],  # type: ignore[arg-type]
            lower_bound=payload["lower_bound"],  # type: ignore[arg-type]
            target=payload["target"],  # type: ignore[arg-type]
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str) -> "CalibrationPolicy":
        try:
            decoded = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise CalibrationPolicyError("invalid calibration policy JSON") from exc
        if not isinstance(decoded, dict):
            raise CalibrationPolicyError("calibration policy JSON must be an object")
        return cls.from_dict(decoded)


__all__ = [
    "CalibrationPolicy",
    "CalibrationPolicyError",
    "CalibrationStatus",
    "wilson_lower_bound",
]
