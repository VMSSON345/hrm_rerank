"""Exact, unit-aware execution and independently checkable proof certificates.

The neural model is responsible for proposing an operation and evidence
pointers.  This module deliberately contains no model logic: it executes an
already selected program with rational arithmetic, validates explicit units,
and records enough information to replay the calculation.  Decimal values are
used only at the display/API boundary; binary floating point is never accepted
as a financial operand.

Lean 4 support is optional.  Exporting a theorem never requires Lean to be
installed, and attempting verification reports ``UNAVAILABLE`` instead of
changing the runtime answer path.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unicodedata
from typing import Mapping, Sequence


CERTIFICATE_SCHEMA_VERSION = "finhrm-exact-proof-v1"
DEFAULT_DECIMAL_PRECISION = 34


class ExactExecutionError(ValueError):
    """Base class for fail-closed exact-execution failures."""


class UnsupportedOperationError(ExactExecutionError):
    pass


class OperandArityError(ExactExecutionError):
    pass


class InvalidNumericValueError(ExactExecutionError):
    pass


class MissingUnitError(ExactExecutionError):
    pass


class UnitMismatchError(ExactExecutionError):
    pass


class MissingEvidenceError(ExactExecutionError):
    pass


class ExactZeroDivisionError(ExactExecutionError, ZeroDivisionError):
    pass


class InvalidProofCertificateError(ExactExecutionError):
    pass


@dataclass(frozen=True)
class UnitSpec:
    """An explicit unit and its exact conversion to canonical dimensions."""

    key: str
    symbol: str
    dimensions: tuple[tuple[str, int], ...]
    scale_to_canonical: Fraction
    disclosed: str

    @property
    def canonical_symbol(self) -> str:
        return _canonical_unit_symbol(self.dimensions)

    @property
    def is_dimensionless(self) -> bool:
        return not self.dimensions


@dataclass(frozen=True)
class ExactOperand:
    """One verified program input.

    ``value`` must be an exact decimal/rational representation.  ``float`` is
    rejected even when it happens to display as a short decimal.
    """

    value: Decimal | Fraction | int | str
    unit: str | UnitSpec | None
    evidence_ref: Mapping[str, object] | str | None
    operand_id: str = ""


@dataclass(frozen=True)
class ProofCertificate:
    schema_version: str
    operation: str
    expression: str
    normalized_inputs: tuple[dict[str, object], ...]
    result: dict[str, object]
    evidence_refs: tuple[dict[str, object], ...]
    decimal_precision: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "expression": self.expression,
            "normalized_inputs": [dict(item) for item in self.normalized_inputs],
            "result": dict(self.result),
            "evidence_refs": [dict(item) for item in self.evidence_refs],
            "decimal_precision": self.decimal_precision,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProofCertificate":
        try:
            normalized_inputs = tuple(
                dict(item) for item in payload["normalized_inputs"]  # type: ignore[arg-type]
            )
            evidence_refs = tuple(
                dict(item) for item in payload["evidence_refs"]  # type: ignore[arg-type]
            )
            result = dict(payload["result"])  # type: ignore[arg-type]
            precision = int(payload["decimal_precision"])
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidProofCertificateError(
                "Malformed exact proof certificate"
            ) from exc
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            operation=str(payload.get("operation", "")),
            expression=str(payload.get("expression", "")),
            normalized_inputs=normalized_inputs,
            result=result,
            evidence_refs=evidence_refs,
            decimal_precision=precision,
        )


@dataclass(frozen=True)
class ExactExecutionResult:
    """Exact canonical result plus a Decimal-compatible display value."""

    operation: str
    exact_value: Fraction
    canonical_value: Fraction
    unit: UnitSpec
    decimal_value: Decimal
    certificate: ProofCertificate

    @property
    def decimal_text(self) -> str:
        return _decimal_text(self.decimal_value)


@dataclass(frozen=True)
class LeanVerificationResult:
    status: str
    verified: bool
    backend: str | None
    source: str
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None


_ARITY: dict[str, tuple[int, int | None]] = {
    "SELECT": (1, 1),
    "ADD": (2, 2),
    "SUM3": (3, 3),
    "SUM4": (4, 4),
    "SUM_N": (2, 8),
    "SUBTRACT": (2, 2),
    "MULTIPLY": (2, 2),
    "DIVIDE": (2, 2),
    "RATIO": (2, 2),
    "RATIO_PERCENT": (2, 2),
    "SHARE_OF_TOTAL": (2, 2),
    "PERCENT_CHANGE": (2, 2),
    "AVERAGE": (2, 2),
}
_SUM_OPERATIONS = frozenset({"ADD", "SUM3", "SUM4", "SUM_N"})
_PERCENT_OPERATIONS = frozenset(
    {"RATIO_PERCENT", "SHARE_OF_TOTAL", "PERCENT_CHANGE"}
)


def _normalize_unit_text(raw: str) -> str:
    decomposed = unicodedata.normalize("NFD", raw.casefold().replace("đ", "d"))
    text = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    text = text.replace("₫", " dong ").replace("%", " percent ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _dimensions(**values: int) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((key, exponent) for key, exponent in values.items() if exponent))


def _scaled_prefix(normalized: str) -> tuple[Fraction, str]:
    tokens = normalized.split()
    scale = Fraction(1)
    removable: set[str] = {"don", "vi", "tinh", "dvt", "unit"}
    if any(token in {"billion", "billionvnd"} for token in tokens) or (
        "ty" in tokens and "le" not in tokens
    ):
        scale = Fraction(1_000_000_000)
        removable.update({"billion", "billionvnd", "ty"})
    elif any(token in {"million", "millionvnd", "trieu"} for token in tokens):
        scale = Fraction(1_000_000)
        removable.update({"million", "millionvnd", "trieu"})
    elif any(
        token in {"thousand", "thousandvnd", "nghin", "ngan", "000"}
        for token in tokens
    ):
        scale = Fraction(1_000)
        removable.update({"thousand", "thousandvnd", "nghin", "ngan", "000"})
    base = " ".join(token for token in tokens if token not in removable)
    return scale, base


def parse_unit(unit: str | UnitSpec | None) -> UnitSpec:
    """Parse an explicit financial unit without guessing from numeric size."""

    if isinstance(unit, UnitSpec):
        return unit
    disclosed = "" if unit is None else str(unit).strip()
    if not disclosed:
        raise MissingUnitError("An explicit operand unit is required")
    normalized = _normalize_unit_text(disclosed)
    if not normalized:
        raise MissingUnitError("An explicit operand unit is required")

    unit_tokens = set(normalized.split())
    if "percent" in unit_tokens or "percentage" in unit_tokens or normalized in {
        "percent",
        "percentage",
        "phan tram",
        "pct",
    }:
        return UnitSpec("percent", "%", (), Fraction(1, 100), disclosed)
    if normalized in {
        "ratio",
        "scalar",
        "dimensionless",
        "lan",
        "x",
        "ty le",
    }:
        return UnitSpec("ratio", "ratio", (), Fraction(1), disclosed)

    scale, base = _scaled_prefix(normalized)
    currency = any(
        marker in {"vnd", "dong", "d", "currency"}
        for marker in base.split()
    )
    shares = (
        any(marker in base.split() for marker in {"share", "shares", "stock", "cp"})
        or "co phieu" in base
        or "co phan" in base
        or "share count" in base
    )
    per_share = shares and (
        "/" in disclosed
        or " per " in f" {disclosed.casefold()} "
        or "tren" in base.split()
        or "moi" in base.split()
        or "currency per share" in base
    )

    if currency and per_share:
        return UnitSpec(
            "currency_per_share",
            "VND/share" if scale == 1 else disclosed,
            _dimensions(currency=1, share=-1),
            scale,
            disclosed,
        )
    if currency:
        key = {
            Fraction(1): "vnd",
            Fraction(1_000): "thousand_vnd",
            Fraction(1_000_000): "million_vnd",
            Fraction(1_000_000_000): "billion_vnd",
        }.get(scale, "currency")
        return UnitSpec(key, disclosed, _dimensions(currency=1), scale, disclosed)
    if shares:
        return UnitSpec("share_count", disclosed, _dimensions(share=1), scale, disclosed)

    generic = base or normalized
    return UnitSpec(
        f"unit:{generic.replace(' ', '_')}",
        disclosed,
        _dimensions(**{f"unit:{generic}": 1}),
        scale,
        disclosed,
    )


def _canonical_unit_symbol(dimensions: tuple[tuple[str, int], ...]) -> str:
    if not dimensions:
        return "ratio"
    if dimensions == (("currency", 1),):
        return "VND"
    if dimensions == (("share", 1),):
        return "share"
    if dimensions == (("currency", 1), ("share", -1)):
        return "VND/share"
    numerator: list[str] = []
    denominator: list[str] = []
    for name, exponent in dimensions:
        clean = name.removeprefix("unit:")
        destination = numerator if exponent > 0 else denominator
        power = abs(exponent)
        destination.append(clean if power == 1 else f"{clean}^{power}")
    top = "*".join(numerator) or "1"
    return top if not denominator else f"{top}/{'*'.join(denominator)}"


def _canonical_unit(dimensions: tuple[tuple[str, int], ...]) -> UnitSpec:
    symbol = _canonical_unit_symbol(dimensions)
    key = "ratio" if not dimensions else f"canonical:{symbol}"
    return UnitSpec(key, symbol, dimensions, Fraction(1), symbol)


def _combine_dimensions(
    left: tuple[tuple[str, int], ...],
    right: tuple[tuple[str, int], ...],
    *,
    subtract: bool = False,
) -> tuple[tuple[str, int], ...]:
    result = dict(left)
    direction = -1 if subtract else 1
    for name, exponent in right:
        result[name] = result.get(name, 0) + direction * exponent
    return tuple(sorted((name, exponent) for name, exponent in result.items() if exponent))


def _fraction(value: Decimal | Fraction | int | str) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise InvalidNumericValueError(
            "Binary float operands are forbidden; use Decimal, Fraction, int, or str"
        )
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    try:
        decimal = value if isinstance(value, Decimal) else Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise InvalidNumericValueError(f"Invalid exact numeric value: {value!r}") from exc
    if not decimal.is_finite():
        raise InvalidNumericValueError("NaN and infinite financial operands are forbidden")
    return Fraction(decimal)


def _fraction_decimal(value: Fraction, precision: int) -> Decimal:
    if precision < 1:
        raise ExactExecutionError("decimal_precision must be positive")
    with localcontext() as context:
        context.prec = precision
        return Decimal(value.numerator) / Decimal(value.denominator)


def _decimal_text(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _rational_payload(value: Fraction, precision: int) -> dict[str, str]:
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        "decimal": _decimal_text(_fraction_decimal(value, precision)),
    }


def _safe_evidence_ref(reference: Mapping[str, object] | str | None) -> dict[str, object]:
    if reference is None:
        return {}
    if isinstance(reference, str):
        return {"id": reference} if reference.strip() else {}
    output: dict[str, object] = {}
    for key in sorted(reference):
        value = reference[key]
        if value is None or value == "" or value == () or value == []:
            continue
        if isinstance(value, (str, int, bool)):
            output[str(key)] = value
        elif isinstance(value, (Decimal, Fraction, float)):
            output[str(key)] = str(value)
        elif isinstance(value, (tuple, list)):
            output[str(key)] = [str(item) if isinstance(item, float) else item for item in value]
        else:
            output[str(key)] = str(value)
    return output


def evidence_reference_from_candidate(candidate: object) -> dict[str, object]:
    """Build a physical evidence reference without depending on model classes."""

    from .physical_coordinates import is_versioned_coordinate, physical_coordinate_key
    if is_versioned_coordinate(candidate) and physical_coordinate_key(candidate) is None:
        raise MissingEvidenceError("Unresolved physical coordinate provenance")
    fields = (
        "source_name",
        "page",
        "table_id",
        "subtable_id",
        "row_id",
        "column_id",
        "row_label",
        "column_label",
        "period",
        "source_span",
        "source_bbox",
        "extraction_method",
    )
    payload = {
        "candidate_index": getattr(candidate, "index", None),
        **({"source_span_scope": getattr(candidate, "source_span_scope"),
            "line_number": getattr(candidate, "line_number", None),
            "source_line_sha256": getattr(candidate, "source_line_sha256", "")}
           if getattr(candidate, "source_span_scope", "legacy_page_v1") != "legacy_page_v1" else {}),
        **{field: getattr(candidate, field, None) for field in fields},
    }
    return _safe_evidence_ref(payload)


def operand_from_candidate(candidate: object, *, operand_id: str = "") -> ExactOperand:
    return ExactOperand(
        value=getattr(candidate, "value"),
        unit=getattr(candidate, "unit", None),
        evidence_ref=evidence_reference_from_candidate(candidate),
        operand_id=operand_id,
    )


def _validate_arity(operation: str, count: int) -> None:
    bounds = _ARITY.get(operation)
    if bounds is None:
        raise UnsupportedOperationError(f"Unsupported exact operation: {operation}")
    minimum, maximum = bounds
    if count < minimum or (maximum is not None and count > maximum):
        expected = str(minimum) if maximum == minimum else (
            f"at least {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        )
        raise OperandArityError(f"{operation} needs {expected} operands, got {count}")


def _require_same_dimensions(operation: str, units: Sequence[UnitSpec]) -> None:
    if any(item.dimensions != units[0].dimensions for item in units[1:]):
        disclosed = ", ".join(item.disclosed for item in units)
        raise UnitMismatchError(
            f"{operation} requires compatible operand units, got: {disclosed}"
        )


def _expression(operation: str, count: int) -> str:
    names = [f"x{index + 1}" for index in range(count)]
    if operation == "SELECT":
        return names[0]
    if operation in _SUM_OPERATIONS:
        return "(" + " + ".join(names) + ")"
    if operation == "SUBTRACT":
        return f"({names[0]} - {names[1]})"
    if operation == "MULTIPLY":
        return f"({names[0]} * {names[1]})"
    if operation in {"DIVIDE", "RATIO", "RATIO_PERCENT", "SHARE_OF_TOTAL"}:
        return f"({names[0]} / {names[1]})"
    if operation == "PERCENT_CHANGE":
        return f"(({names[0]} - {names[1]}) / {names[1]})"
    if operation == "AVERAGE":
        return f"(({' + '.join(names)}) / {count})"
    raise UnsupportedOperationError(f"Unsupported exact operation: {operation}")


def _compute_canonical(operation: str, values: Sequence[Fraction]) -> Fraction:
    if operation == "SELECT":
        return values[0]
    if operation in _SUM_OPERATIONS:
        return sum(values, Fraction(0))
    if operation == "SUBTRACT":
        return values[0] - values[1]
    if operation == "MULTIPLY":
        return values[0] * values[1]
    if operation in {"DIVIDE", "RATIO", "RATIO_PERCENT", "SHARE_OF_TOTAL"}:
        if values[1] == 0:
            raise ExactZeroDivisionError(f"{operation} denominator is zero")
        return values[0] / values[1]
    if operation == "PERCENT_CHANGE":
        if values[1] == 0:
            raise ExactZeroDivisionError("PERCENT_CHANGE baseline is zero")
        return (values[0] - values[1]) / values[1]
    if operation == "AVERAGE":
        return sum(values, Fraction(0)) / len(values)
    raise UnsupportedOperationError(f"Unsupported exact operation: {operation}")


def _result_dimensions(operation: str, units: Sequence[UnitSpec]) -> tuple[tuple[str, int], ...]:
    if operation in _SUM_OPERATIONS or operation in {"SELECT", "SUBTRACT", "AVERAGE"}:
        return units[0].dimensions
    if operation == "MULTIPLY":
        return _combine_dimensions(units[0].dimensions, units[1].dimensions)
    if operation in {"DIVIDE", "RATIO"}:
        return _combine_dimensions(units[0].dimensions, units[1].dimensions, subtract=True)
    if operation in _PERCENT_OPERATIONS:
        return ()
    raise UnsupportedOperationError(f"Unsupported exact operation: {operation}")


def _default_output_unit(
    operation: str,
    units: Sequence[UnitSpec],
    dimensions: tuple[tuple[str, int], ...],
) -> UnitSpec:
    if operation in _PERCENT_OPERATIONS:
        return parse_unit("%")
    if operation == "SELECT":
        return units[0]
    if operation in _SUM_OPERATIONS or operation in {"SUBTRACT", "AVERAGE"}:
        first = units[0]
        if all(
            item.dimensions == first.dimensions
            and item.scale_to_canonical == first.scale_to_canonical
            for item in units[1:]
        ):
            return first
        return _canonical_unit(dimensions)
    if operation == "MULTIPLY":
        if units[0].is_dimensionless and not units[1].is_dimensionless:
            return units[1]
        if units[1].is_dimensionless and not units[0].is_dimensionless:
            return units[0]
    if operation in {"DIVIDE", "RATIO"} and units[1].is_dimensionless:
        return units[0]
    return _canonical_unit(dimensions)


def execute_financial_program(
    operation: str,
    operands: Sequence[ExactOperand],
    *,
    output_unit: str | UnitSpec | None = None,
    require_units: bool = True,
    require_evidence: bool = True,
    decimal_precision: int = DEFAULT_DECIMAL_PRECISION,
) -> ExactExecutionResult:
    """Execute verified finance operands and emit a replayable certificate."""

    normalized_operation = operation.strip().upper()
    _validate_arity(normalized_operation, len(operands))
    units: list[UnitSpec] = []
    raw_values: list[Fraction] = []
    canonical_values: list[Fraction] = []
    evidence_refs: list[dict[str, object]] = []
    normalized_inputs: list[dict[str, object]] = []

    for index, operand in enumerate(operands):
        raw_value = _fraction(operand.value)
        if operand.unit is None or (isinstance(operand.unit, str) and not operand.unit.strip()):
            if require_units:
                raise MissingUnitError(f"Operand {index + 1} has no explicit unit")
            unit = parse_unit("ratio")
        else:
            unit = parse_unit(operand.unit)
        evidence_ref = _safe_evidence_ref(operand.evidence_ref)
        if require_evidence and not evidence_ref:
            raise MissingEvidenceError(f"Operand {index + 1} has no evidence reference")
        canonical_value = raw_value * unit.scale_to_canonical
        operand_id = operand.operand_id.strip() or f"operand_{index + 1}"
        raw_values.append(raw_value)
        units.append(unit)
        canonical_values.append(canonical_value)
        evidence_refs.append(evidence_ref)
        normalized_inputs.append(
            {
                "operand_id": operand_id,
                "raw_value": _rational_payload(raw_value, decimal_precision),
                "raw_unit": unit.disclosed,
                "unit_key": unit.key,
                "scale_to_canonical": _rational_payload(
                    unit.scale_to_canonical, decimal_precision
                ),
                "canonical_value": _rational_payload(
                    canonical_value, decimal_precision
                ),
                "canonical_unit": unit.canonical_symbol,
                "evidence_ref": evidence_ref,
            }
        )

    if normalized_operation in _SUM_OPERATIONS or normalized_operation in {
        "SUBTRACT",
        "AVERAGE",
        "PERCENT_CHANGE",
        "RATIO",
        "RATIO_PERCENT",
        "SHARE_OF_TOTAL",
    }:
        _require_same_dimensions(normalized_operation, units)

    canonical_result = _compute_canonical(normalized_operation, canonical_values)
    dimensions = _result_dimensions(normalized_operation, units)
    selected_output_unit = (
        parse_unit(output_unit)
        if output_unit is not None
        else _default_output_unit(normalized_operation, units, dimensions)
    )
    if selected_output_unit.dimensions != dimensions:
        raise UnitMismatchError(
            f"Requested output unit {selected_output_unit.disclosed!r} is not "
            f"compatible with {_canonical_unit_symbol(dimensions)!r}"
        )
    display_result = canonical_result / selected_output_unit.scale_to_canonical
    decimal_result = _fraction_decimal(display_result, decimal_precision)
    result_payload: dict[str, object] = {
        "value": _rational_payload(display_result, decimal_precision),
        "unit": selected_output_unit.symbol,
        "unit_key": selected_output_unit.key,
        "scale_to_canonical": _rational_payload(
            selected_output_unit.scale_to_canonical, decimal_precision
        ),
        "canonical_value": _rational_payload(canonical_result, decimal_precision),
        "canonical_unit": _canonical_unit_symbol(dimensions),
        "dimensions": [
            {"name": name, "exponent": exponent}
            for name, exponent in dimensions
        ],
    }
    certificate = ProofCertificate(
        schema_version=CERTIFICATE_SCHEMA_VERSION,
        operation=normalized_operation,
        expression=_expression(normalized_operation, len(operands)),
        normalized_inputs=tuple(normalized_inputs),
        result=result_payload,
        evidence_refs=tuple(evidence_refs),
        decimal_precision=decimal_precision,
    )
    return ExactExecutionResult(
        operation=normalized_operation,
        exact_value=display_result,
        canonical_value=canonical_result,
        unit=selected_output_unit,
        decimal_value=decimal_result,
        certificate=certificate,
    )


def execute_reasoning_program(
    program: object,
    candidates: Sequence[object],
    *,
    output_unit: str | UnitSpec | None = None,
    require_units: bool = True,
    require_evidence: bool = True,
    decimal_precision: int = DEFAULT_DECIMAL_PRECISION,
) -> ExactExecutionResult:
    """Adapter from the existing pointer program/candidate representation."""

    operation = str(getattr(program, "operation"))
    pointers = tuple(int(item) for item in getattr(program, "operands"))
    if any(pointer < 0 or pointer >= len(candidates) for pointer in pointers):
        raise OperandArityError("Operand pointer is outside the candidate list")
    if len(set(pointers)) != len(pointers):
        raise ExactExecutionError("A program cannot reuse the same candidate pointer")
    operands = tuple(
        operand_from_candidate(candidates[pointer], operand_id=f"p{slot + 1}")
        for slot, pointer in enumerate(pointers)
    )
    return execute_financial_program(
        operation,
        operands,
        output_unit=output_unit,
        require_units=require_units,
        require_evidence=require_evidence,
        decimal_precision=decimal_precision,
    )


def execute_decimal_operation(
    operation: str,
    values: Sequence[Decimal | Fraction | int | str],
    *,
    decimal_precision: int = DEFAULT_DECIMAL_PRECISION,
) -> Decimal:
    """Backward-compatible exact arithmetic for existing runtime consumers.

    This deliberately bypasses unit/evidence checks.  Call
    :func:`execute_financial_program` after evidence verification when a proof
    certificate is required.
    """

    operands = tuple(
        ExactOperand(value=value, unit="ratio", evidence_ref=f"runtime:{index}")
        for index, value in enumerate(values)
    )
    return execute_financial_program(
        operation,
        operands,
        require_units=False,
        require_evidence=False,
        decimal_precision=decimal_precision,
    ).decimal_value


def _certificate_operands(certificate: ProofCertificate) -> tuple[ExactOperand, ...]:
    operands: list[ExactOperand] = []
    for item in certificate.normalized_inputs:
        try:
            raw_record = item["raw_value"]
            assert isinstance(raw_record, Mapping)
            numerator = int(str(raw_record["numerator"]))
            denominator = int(str(raw_record["denominator"]))
            raw_value = Fraction(numerator, denominator)
            raw_unit = str(item["raw_unit"])
            evidence_ref = item["evidence_ref"]
            assert isinstance(evidence_ref, Mapping)
        except (KeyError, TypeError, ValueError, AssertionError, ZeroDivisionError) as exc:
            raise InvalidProofCertificateError(
                "Malformed normalized certificate input"
            ) from exc
        operands.append(
            ExactOperand(
                value=raw_value,
                unit=raw_unit,
                evidence_ref=dict(evidence_ref),
                operand_id=str(item.get("operand_id", "")),
            )
        )
    return tuple(operands)


def verify_proof_certificate(
    certificate: ProofCertificate | Mapping[str, object],
) -> bool:
    """Replay and compare every deterministic certificate field."""

    try:
        parsed = (
            certificate
            if isinstance(certificate, ProofCertificate)
            else ProofCertificate.from_dict(certificate)
        )
        if parsed.schema_version != CERTIFICATE_SCHEMA_VERSION:
            return False
        result_dimensions = tuple(
            (str(item["name"]), int(item["exponent"]))
            for item in parsed.result["dimensions"]
        )
        output_scale = _fraction_from_payload(parsed.result["scale_to_canonical"])
        result_unit = UnitSpec(
            key=str(parsed.result["unit_key"]), symbol=str(parsed.result["unit"]),
            dimensions=result_dimensions, scale_to_canonical=output_scale,
            disclosed=str(parsed.result["unit"]),
        )
        rebuilt = execute_financial_program(
            parsed.operation,
            _certificate_operands(parsed),
            output_unit=result_unit,
            require_units=True,
            require_evidence=True,
            decimal_precision=parsed.decimal_precision,
        ).certificate
        return rebuilt.to_dict() == parsed.to_dict()
    except (ExactExecutionError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def _lean_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return f"({value.numerator} : ℚ)"
    return f"(({value.numerator} : ℚ) / {value.denominator})"


def _fraction_from_payload(payload: object) -> Fraction:
    if not isinstance(payload, Mapping):
        raise InvalidProofCertificateError("Expected a rational payload")
    try:
        return Fraction(int(str(payload["numerator"])), int(str(payload["denominator"])))
    except (KeyError, ValueError, ZeroDivisionError) as exc:
        raise InvalidProofCertificateError("Invalid rational payload") from exc


def export_lean4(
    certificate: ProofCertificate | Mapping[str, object],
    *,
    theorem_name: str = "finhrm_exact_result",
) -> str:
    """Export an exact arithmetic theorem; does not invoke Lean."""

    parsed = (
        certificate
        if isinstance(certificate, ProofCertificate)
        else ProofCertificate.from_dict(certificate)
    )
    if not verify_proof_certificate(parsed):
        raise InvalidProofCertificateError("Refusing to export an invalid certificate")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", theorem_name):
        raise InvalidProofCertificateError("Unsafe Lean theorem name")
    values = [
        _fraction_from_payload(item["canonical_value"])
        for item in parsed.normalized_inputs
    ]
    names = [f"({_lean_fraction(value)})" for value in values]
    operation = parsed.operation
    if operation == "SELECT":
        expression = names[0]
    elif operation in _SUM_OPERATIONS:
        expression = "(" + " + ".join(names) + ")"
    elif operation == "SUBTRACT":
        expression = f"({names[0]} - {names[1]})"
    elif operation == "MULTIPLY":
        expression = f"({names[0]} * {names[1]})"
    elif operation in {"DIVIDE", "RATIO", "RATIO_PERCENT", "SHARE_OF_TOTAL"}:
        expression = f"({names[0]} / {names[1]})"
    elif operation == "PERCENT_CHANGE":
        expression = f"(({names[0]} - {names[1]}) / {names[1]})"
    elif operation == "AVERAGE":
        expression = f"(({' + '.join(names)}) / {len(names)})"
    else:
        raise UnsupportedOperationError(f"Unsupported Lean operation: {operation}")
    result = _fraction_from_payload(parsed.result["canonical_value"])
    certificate_json = json.dumps(
        parsed.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    return (
        "import Mathlib\n\n"
        f"-- FinHRM certificate: {certificate_json}\n"
        f"theorem {theorem_name} : {expression} = {_lean_fraction(result)} := by\n"
        "  norm_num\n"
    )


class Lean4Verifier:
    """Optional Lean verifier isolated from the mandatory runtime path."""

    def __init__(self, command: Sequence[str] | None = None) -> None:
        self.command = tuple(command) if command is not None else None

    def _resolved_command(self) -> tuple[str, ...] | None:
        if self.command:
            return self.command if shutil.which(self.command[0]) else None
        if shutil.which("lake"):
            return ("lake", "env", "lean")
        if shutil.which("lean"):
            return ("lean",)
        return None

    def verify(
        self,
        certificate: ProofCertificate | Mapping[str, object],
        *,
        theorem_name: str = "finhrm_exact_result",
        cwd: str | Path | None = None,
        timeout_seconds: int = 30,
    ) -> LeanVerificationResult:
        source = export_lean4(certificate, theorem_name=theorem_name)
        command = self._resolved_command()
        if command is None:
            return LeanVerificationResult(
                status="UNAVAILABLE",
                verified=False,
                backend=None,
                source=source,
                stderr="Lean 4 executable not found; exact runtime result remains valid",
            )
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".lean", encoding="utf-8", delete=False
            ) as handle:
                handle.write(source)
                temporary_path = Path(handle.name)
            completed = subprocess.run(
                [*command, str(temporary_path)],
                cwd=None if cwd is None else str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            return LeanVerificationResult(
                status="VERIFIED" if completed.returncode == 0 else "FAILED",
                verified=completed.returncode == 0,
                backend=" ".join(command),
                source=source,
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return LeanVerificationResult(
                status="FAILED",
                verified=False,
                backend=" ".join(command),
                source=source,
                stderr=str(exc),
            )
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

