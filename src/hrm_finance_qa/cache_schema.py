"""Lightweight cache identity constants shared by builders and audits."""

from __future__ import annotations

from .evidence_schema import (
    EVIDENCE_SCHEMA_FINGERPRINT,
    EVIDENCE_SCHEMA_VERSION,
)


CANDIDATE_CACHE_SCHEMA = "v12b2-applicability-proof-candidate-cache-v5"


def candidate_cache_schema_identity() -> dict[str, str]:
    return {
        "cache_schema": CANDIDATE_CACHE_SCHEMA,
        "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
        "evidence_schema_fingerprint": EVIDENCE_SCHEMA_FINGERPRINT,
    }
