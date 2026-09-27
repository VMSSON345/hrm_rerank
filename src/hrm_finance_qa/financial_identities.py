from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import yaml

from .ontology import normalize_text


FINANCIAL_IDENTITY_CATALOG_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "financial_identities.yaml"
)
FINANCIAL_IDENTITY_SCHEMA_VERSION = 1
_OPERATION_ARITIES = {
    "SELECT": 1,
    "ADD": 2,
    "SUBTRACT": 2,
    "MULTIPLY": 2,
    "DIVIDE": 2,
    "RATIO_PERCENT": 2,
    "PERCENT_CHANGE": 2,
    "AVERAGE": 2,
    "SUM3": 3,
    "SUM4": 4,
}
_ALLOWED_STATUSES = {"active", "conditional"}


class FinancialIdentityCatalogError(ValueError):
    pass


@dataclass(frozen=True)
class FinancialMetricRole:
    key: str
    aliases: tuple[str, ...]
    unit_family: str

    @property
    def preferred_text(self) -> str:
        return self.aliases[0]


@dataclass(frozen=True)
class FinancialIdentity:
    """A report-independent accounting identity used as a program schema."""

    identity_id: str
    target_key: str
    target_aliases: tuple[str, ...]
    operation: str
    operands: tuple[FinancialMetricRole, ...]
    result_unit_family: str
    status: str = "active"
    prefer_direct_value: bool = True
    description: str = ""

    @property
    def operand_texts(self) -> tuple[str, ...]:
        return tuple(role.preferred_text for role in self.operands)


def _required_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FinancialIdentityCatalogError(f"{field} must be a mapping")
    return value


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinancialIdentityCatalogError(f"{field} must be non-empty text")
    return value.strip()


def _required_texts(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FinancialIdentityCatalogError(f"{field} must be a text list")
    output = tuple(_required_text(item, field) for item in value)
    if not output:
        raise FinancialIdentityCatalogError(f"{field} cannot be empty")
    return output


def _parse_identity(raw: object, index: int) -> FinancialIdentity:
    row = _required_mapping(raw, f"identities[{index}]")
    identity_id = _required_text(row.get("id"), f"identities[{index}].id")
    target = _required_mapping(
        row.get("target"), f"identities[{index}].target"
    )
    operation = _required_text(
        row.get("operation"), f"{identity_id}.operation"
    ).upper()
    if operation not in _OPERATION_ARITIES:
        raise FinancialIdentityCatalogError(
            f"{identity_id}.operation is unsupported: {operation}"
        )
    raw_operands = row.get("operands")
    if not isinstance(raw_operands, Sequence) or isinstance(
        raw_operands, (str, bytes)
    ):
        raise FinancialIdentityCatalogError(
            f"{identity_id}.operands must be a list"
        )
    operands: list[FinancialMetricRole] = []
    for operand_index, raw_operand in enumerate(raw_operands):
        operand = _required_mapping(
            raw_operand, f"{identity_id}.operands[{operand_index}]"
        )
        operands.append(
            FinancialMetricRole(
                key=_required_text(
                    operand.get("key"),
                    f"{identity_id}.operands[{operand_index}].key",
                ),
                aliases=_required_texts(
                    operand.get("aliases"),
                    f"{identity_id}.operands[{operand_index}].aliases",
                ),
                unit_family=_required_text(
                    operand.get("unit_family"),
                    f"{identity_id}.operands[{operand_index}].unit_family",
                ),
            )
        )
    if len(operands) != _OPERATION_ARITIES[operation]:
        raise FinancialIdentityCatalogError(
            f"{identity_id} requires {_OPERATION_ARITIES[operation]} operands "
            f"for {operation}, got {len(operands)}"
        )
    status = str(row.get("status", "active")).strip().lower()
    if status not in _ALLOWED_STATUSES:
        raise FinancialIdentityCatalogError(
            f"{identity_id}.status must be active or conditional"
        )
    prefer_direct = row.get("prefer_direct_value", True)
    if not isinstance(prefer_direct, bool):
        raise FinancialIdentityCatalogError(
            f"{identity_id}.prefer_direct_value must be boolean"
        )
    return FinancialIdentity(
        identity_id=identity_id,
        target_key=_required_text(target.get("key"), f"{identity_id}.target.key"),
        target_aliases=_required_texts(
            target.get("aliases"), f"{identity_id}.target.aliases"
        ),
        operation=operation,
        operands=tuple(operands),
        result_unit_family=_required_text(
            row.get("result_unit_family"), f"{identity_id}.result_unit_family"
        ),
        status=status,
        prefer_direct_value=prefer_direct,
        description=str(row.get("description", "")).strip(),
    )


def load_financial_identity_catalog(
    path: str | Path = FINANCIAL_IDENTITY_CATALOG_PATH,
) -> tuple[FinancialIdentity, ...]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Financial identity catalog not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    root = _required_mapping(payload, "catalog")
    if root.get("schema_version") != FINANCIAL_IDENTITY_SCHEMA_VERSION:
        raise FinancialIdentityCatalogError(
            "Unsupported financial identity catalog schema"
        )
    raw_identities = root.get("identities")
    if not isinstance(raw_identities, Sequence) or isinstance(
        raw_identities, (str, bytes)
    ):
        raise FinancialIdentityCatalogError("identities must be a list")
    identities = tuple(
        _parse_identity(raw, index)
        for index, raw in enumerate(raw_identities)
    )
    ids = [identity.identity_id for identity in identities]
    if len(ids) != len(set(ids)):
        raise FinancialIdentityCatalogError(
            "Financial identity IDs must be unique"
        )
    return identities


FINANCIAL_IDENTITIES = load_financial_identity_catalog()
_BY_ID = {identity.identity_id: identity for identity in FINANCIAL_IDENTITIES}


_MATCH_STOPWORDS = {
    "bao",
    "cao",
    "cho",
    "cua",
    "la",
    "nam",
    "nhieu",
    "theo",
    "tinh",
    "tri",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_text(text))
        if len(token) > 1 and not token.isdigit() and token not in _MATCH_STOPWORDS
    }


def _normalize_metric_ocr_text(text: str) -> str:
    """Repair only explicit, generic OCR fractures of financial vocabulary."""

    normalized = normalize_text(text).strip()
    normalized = re.sub(
        r"\b(?:lai|loi nhuan)\s+tr\s+c\s+thu\b",
        "loi nhuan truoc thue",
        normalized,
    )
    normalized = re.sub(
        r"\b(?:lai|loi nhuan)\s+sau\s+thu\b",
        "loi nhuan sau thue",
        normalized,
    )
    return normalized


def _exclusive_metric_modifiers(text: str) -> dict[str, str]:
    """Return source-explicit metric qualifiers that cannot be synonyms."""

    padded = f" {_normalize_metric_ocr_text(text)} "
    modifiers: dict[str, str] = {}
    if " truoc thue " in padded:
        modifiers["tax_basis"] = "before_tax"
    elif " sau thue " in padded:
        modifiers["tax_basis"] = "after_tax"
    if " ngan han " in padded:
        modifiers["maturity"] = "short_term"
    elif " dai han " in padded:
        modifiers["maturity"] = "long_term"
    return modifiers


def metric_alias_score(text: str, aliases: Iterable[str]) -> float:
    """Score a metric against auditable aliases without issuer-specific data."""

    observed = _normalize_metric_ocr_text(text)
    observed_tokens = _tokens(observed)
    observed_compact = re.sub(r"[^a-z0-9]+", "", observed)
    observed_modifiers = _exclusive_metric_modifiers(observed)
    best = 0.0
    for alias in aliases:
        expected = _normalize_metric_ocr_text(alias)
        expected_tokens = _tokens(expected)
        expected_compact = re.sub(r"[^a-z0-9]+", "", expected)
        expected_modifiers = _exclusive_metric_modifiers(expected)
        if any(
            name in observed_modifiers
            and name in expected_modifiers
            and observed_modifiers[name] != expected_modifiers[name]
            for name in observed_modifiers.keys() | expected_modifiers.keys()
        ):
            continue
        if not expected_tokens or not observed_tokens:
            continue
        if observed_compact == expected_compact:
            best = max(best, 1.0)
            continue
        common = expected_tokens & observed_tokens
        coverage = len(common) / len(expected_tokens)
        precision = len(common) / len(observed_tokens)
        compact = (
            expected_compact in observed_compact
            or observed_compact in expected_compact
        )
        score = 0.65 * coverage + 0.35 * precision
        if compact:
            score = max(
                score,
                0.90
                * min(
                    1.0,
                    len(expected_compact) / max(1, len(observed_compact)),
                ),
            )
        best = max(best, score)
    return best


_TARGET_CONTEXT_TOKENS = {
    "bang",
    "cuoi",
    "dau",
    "gia",
    "gia tri",
    "he",
    "ky",
    "muc",
    "so",
    "tai",
    "thoi diem",
    "ty",
    "le",
}


def target_activation_score(text: str, aliases: Iterable[str]) -> float:
    """Match a requested target without promoting a narrower sub-metric.

    Formula activation is intentionally stricter than operand retrieval. For
    example, ``no phai tra dai han`` must not activate the broader
    ``total_liabilities`` identity merely because all words in ``no phai tra``
    occur in the question. Benign query wrappers are tolerated, but additional
    financial qualifiers are not.
    """

    observed = normalize_text(text).strip()
    observed_tokens = _tokens(observed)
    observed_compact = re.sub(r"[^a-z0-9]+", "", observed)
    best = 0.0
    for alias in aliases:
        expected = normalize_text(alias).strip()
        expected_tokens = _tokens(expected)
        expected_compact = re.sub(r"[^a-z0-9]+", "", expected)
        if not expected_tokens or not observed_tokens:
            continue
        if observed_compact == expected_compact:
            best = 1.0
            continue
        if not expected_tokens.issubset(observed_tokens):
            continue
        extra = observed_tokens - expected_tokens
        if extra - _TARGET_CONTEXT_TOKENS:
            continue
        best = max(best, metric_alias_score(observed, (expected,)))
    return best


def get_financial_identity(identity_id: str) -> FinancialIdentity:
    return _BY_ID[identity_id]


def match_financial_identities(
    metric_text: str,
    *,
    include_conditional: bool = False,
) -> tuple[FinancialIdentity, ...]:
    """Return only the most specific target identity family for a question."""

    scored = [
        (target_activation_score(metric_text, identity.target_aliases), identity)
        for identity in FINANCIAL_IDENTITIES
        if include_conditional or identity.status == "active"
    ]
    best = max((score for score, _ in scored), default=0.0)
    if best < 0.88:
        return ()
    threshold = max(0.88, best - 0.03)
    target_keys = {
        identity.target_key
        for score, identity in scored
        if score >= threshold
    }
    return tuple(
        identity
        for score, identity in scored
        if identity.target_key in target_keys and score >= threshold
    )


def role_matches_metric(role: FinancialMetricRole, metric_label: str) -> bool:
    return metric_alias_score(metric_label, role.aliases) >= 0.72


def target_matches_metric(identity: FinancialIdentity, metric_label: str) -> bool:
    return metric_alias_score(metric_label, identity.target_aliases) >= 0.82
