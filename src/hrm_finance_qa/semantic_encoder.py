"""Versioned, role-aware semantic encoders used by Document HRM.

This module is intentionally separate from dense_retrieval. The frozen
dense-index v1 format stores only a model name, while HRM V14.1 needs an exact
frontend identity covering revision, Vietnamese segmentation and query versus
evidence preprocessing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import re
from pathlib import Path
from typing import Any, Iterable, Protocol

from .dense_retrieval import FastEmbedEncoder


LEGACY_SEMANTIC_TEMPLATE = "legacy_raw_v1"
FINANCIAL_VI_SEMANTIC_TEMPLATE = "finance_vi_v1"
SEMANTIC_FRONTEND_IMPLEMENTATION_VERSION = "semantic-frontend-v3"
BKAI_VIETNAMESE_BI_ENCODER = (
    "bkai-foundation-models/vietnamese-bi-encoder"
)
BKAI_VIETNAMESE_BI_ENCODER_REVISION = (
    "84f9d9ada0d1a3c37557398b9ae9fcedcdf40be0"
)


class SemanticEncoderError(RuntimeError):
    pass


@dataclass(frozen=True)
class SemanticEncoderSpec:
    backend: str
    model_name: str
    model_revision: str
    output_size: int
    local_files_only: bool = False
    word_segmentation: str = "none"
    segmenter_version: str = ""
    prefix_mode: str = "none"
    query_encoding_mode: str = "document"
    pooling: str = "mean"
    normalization: str = "l2"
    text_template_version: str = LEGACY_SEMANTIC_TEMPLATE

    def validate(self) -> None:
        if self.backend not in {"fastembed", "sentence_transformers"}:
            raise SemanticEncoderError(
                "semantic backend must be fastembed or sentence_transformers"
            )
        if not self.model_name:
            raise SemanticEncoderError("semantic model name must be non-empty")
        if self.output_size < 1:
            raise SemanticEncoderError("semantic output size must be positive")
        if self.word_segmentation not in {"none", "pyvi"}:
            raise SemanticEncoderError(
                "semantic word segmentation must be none or pyvi"
            )
        if self.prefix_mode not in {"none", "e5_v1"}:
            raise SemanticEncoderError(
                "semantic prefix mode must be none or e5_v1"
            )
        if self.query_encoding_mode not in {"document", "query"}:
            raise SemanticEncoderError(
                "query encoding mode must be document or query"
            )
        if self.pooling != "mean" or self.normalization != "l2":
            raise SemanticEncoderError(
                "V14.1 supports mean pooling and L2 normalization only"
            )
        if self.backend == "sentence_transformers" and not re.fullmatch(
            r"[0-9a-fA-F]{40}", self.model_revision
        ):
            raise SemanticEncoderError(
                "sentence_transformers backend requires an exact 40-hex "
                "commit revision"
            )
        if self.backend == "fastembed" and self.model_revision:
            raise SemanticEncoderError(
                "FastEmbed legacy frontend cannot claim an unenforced revision"
            )
        if self.backend == "fastembed" and self.word_segmentation != "none":
            raise SemanticEncoderError(
                "FastEmbed frontend cannot provide versioned word segmentation"
            )
        if self.word_segmentation == "pyvi" and not self.segmenter_version:
            raise SemanticEncoderError(
                "PyVi preprocessing requires a pinned segmenter version"
            )
        if self.text_template_version not in {
            LEGACY_SEMANTIC_TEMPLATE,
            FINANCIAL_VI_SEMANTIC_TEMPLATE,
        }:
            raise SemanticEncoderError(
                "unsupported semantic text template version"
            )

    def canonical_payload(self) -> dict[str, object]:
        self.validate()
        return {
            "frontend_implementation_version": (
                SEMANTIC_FRONTEND_IMPLEMENTATION_VERSION
            ),
            **asdict(self),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.canonical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def cache_namespace(self, role: str) -> str:
        if role not in {"query", "evidence"}:
            raise SemanticEncoderError("semantic cache role is invalid")
        return f"semantic-raw-v3:{self.fingerprint()}:{role}"


class RoleAwareEmbeddingEncoder(Protocol):
    spec: SemanticEncoderSpec

    def encode_documents(
        self, documents: Iterable[str], *, batch_size: int
    ) -> Any: ...

    def encode_queries(
        self, queries: Iterable[str], *, batch_size: int
    ) -> Any: ...


def _prefix(text: str, *, role: str, mode: str) -> str:
    cleaned = text.replace("|", " ")
    if mode != "e5_v1":
        return cleaned
    return ("query: " if role == "query" else "passage: ") + cleaned


class FastEmbedRoleAwareEncoder:
    """Compatibility wrapper preserving the historical document embedding."""

    def __init__(
        self,
        spec: SemanticEncoderSpec,
        cache_dir: str | Path | None,
    ) -> None:
        spec.validate()
        self.spec = spec
        self._encoder = FastEmbedEncoder(spec.model_name, cache_dir)

    def encode_documents(
        self, documents: Iterable[str], *, batch_size: int
    ) -> Any:
        return self._encoder.encode_documents(
            (
                _prefix(text, role="evidence", mode=self.spec.prefix_mode)
                for text in documents
            ),
            batch_size=batch_size,
        )

    def encode_queries(
        self, queries: Iterable[str], *, batch_size: int
    ) -> Any:
        prepared = [
            _prefix(text, role="query", mode=self.spec.prefix_mode)
            for text in queries
        ]
        if self.spec.query_encoding_mode == "document":
            return self._encoder.encode_documents(
                prepared, batch_size=batch_size
            )
        return [self._encoder.encode_query(text) for text in prepared]


class SentenceTransformerRoleAwareEncoder:
    """Pinned Sentence-Transformers frontend with explicit local preprocessing."""

    def __init__(
        self,
        spec: SemanticEncoderSpec,
        cache_dir: str | Path | None,
    ) -> None:
        spec.validate()
        self.spec = spec
        self.cache_dir = (
            str(Path(cache_dir).expanduser().resolve()) if cache_dir else None
        )
        self._model: Any = None
        self._segmenter: Any = None

    def _prepare(self, text: str, *, role: str) -> str:
        value = text.replace("|", " ")
        if self.spec.word_segmentation == "pyvi":
            if self._segmenter is None:
                try:
                    from pyvi import ViTokenizer
                except ImportError as exc:
                    raise SemanticEncoderError(
                        "Missing pyvi for Vietnamese word segmentation"
                    ) from exc
                try:
                    installed_version = version("pyvi")
                except PackageNotFoundError as exc:
                    raise SemanticEncoderError(
                        "Cannot resolve the installed PyVi version"
                    ) from exc
                if installed_version != self.spec.segmenter_version:
                    raise SemanticEncoderError(
                        "PyVi version does not match the pinned encoder spec"
                    )
                self._segmenter = ViTokenizer
            value = str(self._segmenter.tokenize(value))
        return _prefix(value, role=role, mode=self.spec.prefix_mode)

    def _encode(
        self,
        texts: Iterable[str],
        *,
        role: str,
        batch_size: int,
    ) -> Any:
        prepared = [self._prepare(text, role=role) for text in texts]
        if not prepared:
            return []
        try:
            return self._load_model().encode(
                prepared,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise SemanticEncoderError(
                f"Cannot encode {role} with pinned SentenceTransformer: {exc}"
            ) from exc

    def encode_documents(
        self, documents: Iterable[str], *, batch_size: int
    ) -> Any:
        return self._encode(
            documents, role="evidence", batch_size=batch_size
        )

    def encode_queries(
        self, queries: Iterable[str], *, batch_size: int
    ) -> Any:
        return self._encode(queries, role="query", batch_size=batch_size)

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise SemanticEncoderError(
                    "Missing sentence-transformers; install requirements.txt"
                ) from exc
            try:
                self._model = SentenceTransformer(
                    self.spec.model_name,
                    cache_folder=self.cache_dir,
                    revision=self.spec.model_revision,
                    local_files_only=self.spec.local_files_only,
                    trust_remote_code=False,
                    model_kwargs={"use_safetensors": True},
                )
            except Exception as exc:
                raise SemanticEncoderError(
                    "Cannot load the pinned SentenceTransformer snapshot: "
                    f"{exc}"
                ) from exc
        return self._model


def build_role_aware_encoder(
    spec: SemanticEncoderSpec,
    cache_dir: str | Path | None = None,
) -> RoleAwareEmbeddingEncoder:
    spec.validate()
    if spec.backend == "fastembed":
        return FastEmbedRoleAwareEncoder(spec, cache_dir)
    return SentenceTransformerRoleAwareEncoder(spec, cache_dir)
