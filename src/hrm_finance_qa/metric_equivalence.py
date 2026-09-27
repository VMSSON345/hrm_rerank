"""Reviewed metric-key equivalences for diagnostics, never value inference.

Only complete identities are registered. Qualified, parent-company, current,
and total measures must not become equivalent through string similarity.
"""
from __future__ import annotations

# The source runtime intentionally uses a literal key for this exact label,
# while reviewed gold uses the financial-ontology key. This pair is the
# audited mismatch; keep both source labels and citations unchanged.
_APPROVED = {
    "literal:tai san ngan han": "current_assets",
    "current_assets": "current_assets",
}


def diagnostic_metric_key(metric: str) -> str:
    """Canonicalize only an approved complete metric identity."""
    return _APPROVED.get(metric, metric)


def diagnostic_fact_signature(signature: tuple) -> tuple:
    """Preserve physical token, time, role, scope, and unit-state constraints."""
    if not signature or signature[0] == "PDF_VISUAL":
        return signature
    return (*signature[:3], diagnostic_metric_key(signature[3]),
            *signature[4:])
