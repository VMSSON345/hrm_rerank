from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterable, Protocol

from .document_search import (
    DocumentSearchError,
    SearchHit,
    VietnameseFinancialDocumentIndex,
)


DENSE_INDEX_SCHEMA_VERSION = 1
DEFAULT_DENSE_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class DenseRetrievalError(RuntimeError):
    pass


def _find_cached_fastembed_snapshot(
    cache_dir: str | Path | None, model_name: str
) -> Path | None:
    """Return a complete local HF snapshot without invoking cache validation.

    FastEmbed metadata created on Windows stores relative paths with backslashes.
    Reading the same cache from WSL can therefore trigger a full metadata check
    and a misleading size-mismatch warning on every process start. A complete
    snapshot is already immutable and can safely be passed as
    ``specific_model_path``.
    """
    if cache_dir is None:
        return None
    root = Path(cache_dir).expanduser().resolve()
    if not root.is_dir():
        return None
    model_slug = model_name.rsplit("/", 1)[-1].lower()
    repositories = sorted(
        path
        for path in root.glob("models--*")
        if path.is_dir() and model_slug in path.name.lower()
    )
    for repository in repositories:
        candidates: list[Path] = []
        main_ref = repository / "refs" / "main"
        if main_ref.is_file():
            revision = main_ref.read_text(encoding="utf-8").strip()
            if revision:
                candidates.append(repository / "snapshots" / revision)
        snapshots = repository / "snapshots"
        if snapshots.is_dir():
            candidates.extend(
                path for path in snapshots.iterdir() if path.is_dir()
            )
        for snapshot in dict.fromkeys(candidates):
            required = (
                snapshot / "config.json",
                snapshot / "tokenizer.json",
                snapshot / "tokenizer_config.json",
            )
            if all(path.is_file() for path in required) and any(
                snapshot.rglob("*.onnx")
            ):
                return snapshot
    return None


class EmbeddingEncoder(Protocol):
    model_name: str

    def encode_documents(
        self, documents: Iterable[str], *, batch_size: int
    ) -> Any: ...

    def encode_query(self, query: str) -> Any: ...


class FastEmbedEncoder:
    def __init__(self, model_name: str, cache_dir: str | Path | None = None) -> None:
        self.model_name = model_name
        self.cache_dir = str(Path(cache_dir).resolve()) if cache_dir else None
        self._model: Any = None

    def encode_documents(
        self, documents: Iterable[str], *, batch_size: int
    ) -> Any:
        try:
            return list(self._load_model().embed(documents, batch_size=batch_size))
        except Exception as exc:
            raise DenseRetrievalError(f"Khong encode duoc document: {exc}") from exc

    def encode_query(self, query: str) -> Any:
        try:
            return next(iter(self._load_model().query_embed(query.replace("|", " "))))
        except Exception as exc:
            raise DenseRetrievalError(f"Khong encode duoc query: {exc}") from exc

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise DenseRetrievalError(
                    "Thieu fastembed; chay pip install -r requirements.txt"
                ) from exc
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*now uses mean pooling instead of CLS embedding.*",
                )
                local_snapshot = _find_cached_fastembed_snapshot(
                    self.cache_dir, self.model_name
                )
                options: dict[str, Any] = {}
                if local_snapshot is not None:
                    options.update(
                        local_files_only=True,
                        specific_model_path=str(local_snapshot),
                    )
                self._model = TextEmbedding(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir,
                    **options,
                )
        return self._model


class DenseEmbeddingIndex:
    def __init__(
        self,
        *,
        model_name: str,
        source_fingerprint: str,
        chunk_ids: tuple[str, ...],
        embeddings: Any,
        encoder: EmbeddingEncoder | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        numpy = _numpy()
        vectors = numpy.asarray(embeddings, dtype=numpy.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunk_ids):
            raise DenseRetrievalError(
                "Dense index khong khop so chunk va embedding"
            )
        norms = numpy.linalg.norm(vectors, axis=1, keepdims=True)
        if numpy.any(norms == 0):
            raise DenseRetrievalError("Dense index co vector zero")
        self.model_name = model_name
        self.source_fingerprint = source_fingerprint
        self.chunk_ids = chunk_ids
        self.embeddings = vectors / norms
        self.cache_dir = cache_dir
        self._encoder = encoder

    @classmethod
    def build(
        cls,
        sparse_index: VietnameseFinancialDocumentIndex,
        *,
        model_name: str = DEFAULT_DENSE_MODEL,
        cache_dir: str | Path | None = None,
        batch_size: int = 32,
        encoder: EmbeddingEncoder | None = None,
    ) -> "DenseEmbeddingIndex":
        if batch_size < 1:
            raise DenseRetrievalError("batch_size phai lon hon hoac bang 1")
        active_encoder = encoder or FastEmbedEncoder(model_name, cache_dir)
        chunks = sparse_index.chunks()
        documents = (
            f"{chunk.heading}\n{chunk.text}" for chunk in chunks
        )
        embeddings = active_encoder.encode_documents(
            documents, batch_size=batch_size
        )
        return cls(
            model_name=active_encoder.model_name,
            source_fingerprint=sparse_index.fingerprint(),
            chunk_ids=tuple(chunk.chunk_id for chunk in chunks),
            embeddings=embeddings,
            encoder=active_encoder,
            cache_dir=cache_dir,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        cache_dir: str | Path | None = None,
        encoder: EmbeddingEncoder | None = None,
    ) -> "DenseEmbeddingIndex":
        numpy = _numpy()
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Khong tim thay dense index: {source}")
        with numpy.load(source, allow_pickle=False) as payload:
            schema_version = int(payload["schema_version"][0])
            if schema_version != DENSE_INDEX_SCHEMA_VERSION:
                raise DenseRetrievalError(
                    f"Phien ban dense index khong duoc ho tro: {schema_version}"
                )
            model_name = str(payload["model_name"][0])
            source_fingerprint = str(payload["source_fingerprint"][0])
            chunk_ids = tuple(str(item) for item in payload["chunk_ids"])
            embeddings = payload["embeddings"].copy()
        return cls(
            model_name=model_name,
            source_fingerprint=source_fingerprint,
            chunk_ids=chunk_ids,
            embeddings=embeddings,
            encoder=encoder,
            cache_dir=cache_dir,
        )

    def save(self, path: str | Path) -> Path:
        numpy = _numpy()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            numpy.savez_compressed(
                handle,
                schema_version=numpy.asarray(
                    [DENSE_INDEX_SCHEMA_VERSION], dtype=numpy.int32
                ),
                model_name=numpy.asarray([self.model_name]),
                source_fingerprint=numpy.asarray([self.source_fingerprint]),
                chunk_ids=numpy.asarray(self.chunk_ids),
                embeddings=self.embeddings,
            )
        return destination.resolve()

    def validate_against(
        self, sparse_index: VietnameseFinancialDocumentIndex
    ) -> None:
        if self.source_fingerprint != sparse_index.fingerprint():
            raise DenseRetrievalError(
                "Dense index khong khop document index; hay build lai dense index"
            )
        available_chunk_ids = {chunk.chunk_id for chunk in sparse_index.chunks()}
        missing = [
            chunk_id
            for chunk_id in self.chunk_ids
            if chunk_id not in available_chunk_ids
        ]
        if missing:
            raise DenseRetrievalError(
                f"Dense index tham chieu chunk khong ton tai: {missing[0]}"
            )

    def search(
        self,
        query: str,
        sparse_index: VietnameseFinancialDocumentIndex,
        *,
        document_id: str | None = None,
        top_k: int = 15,
    ) -> tuple[SearchHit, ...]:
        if top_k < 1:
            raise DocumentSearchError("top_k phai lon hon hoac bang 1")
        self.validate_against(sparse_index)
        resolved_document = (
            sparse_index.resolve_document_id(document_id) if document_id else None
        )
        encoder = self._encoder or FastEmbedEncoder(
            self.model_name, self.cache_dir
        )
        self._encoder = encoder
        numpy = _numpy()
        query_vector = numpy.asarray(
            encoder.encode_query(query), dtype=numpy.float32
        )
        query_norm = float(numpy.linalg.norm(query_vector))
        if query_norm == 0:
            raise DenseRetrievalError("Query embedding la vector zero")
        scores = self.embeddings @ (query_vector / query_norm)
        candidates: list[SearchHit] = []
        for position, chunk_id in enumerate(self.chunk_ids):
            chunk = sparse_index.get_chunk(chunk_id)
            if resolved_document and chunk.document_id != resolved_document:
                continue
            dense_score = float(scores[position])
            candidates.append(
                SearchHit(
                    chunk=chunk,
                    score=dense_score,
                    matched_terms=(),
                    exact_match=False,
                    retrieval_mode="dense",
                    dense_score=dense_score,
                )
            )
        candidates.sort(key=lambda hit: hit.score, reverse=True)
        return tuple(candidates[:top_k])


class HybridFinancialDocumentIndex:
    def __init__(
        self,
        sparse_index: VietnameseFinancialDocumentIndex,
        dense_index: DenseEmbeddingIndex,
        *,
        sparse_weight: float = 0.30,
        dense_weight: float = 0.70,
        rrf_k: int = 60,
    ) -> None:
        if sparse_weight <= 0 or dense_weight <= 0:
            raise DenseRetrievalError("Hybrid weights phai lon hon 0")
        dense_index.validate_against(sparse_index)
        self.sparse_index = sparse_index
        self.dense_index = dense_index
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.last_warning: str | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sparse_index, name)

    def search(
        self,
        query: str,
        *,
        document_id: str | None = None,
        top_k: int = 15,
    ) -> tuple[SearchHit, ...]:
        candidate_k = max(top_k, top_k * 3)
        sparse_hits = self.sparse_index.search(
            query, document_id=document_id, top_k=candidate_k
        )
        try:
            dense_hits = self.dense_index.search(
                query,
                self.sparse_index,
                document_id=document_id,
                top_k=candidate_k,
            )
            self.last_warning = None
        except DenseRetrievalError as exc:
            self.last_warning = str(exc)
            return sparse_hits[:top_k]

        sparse_by_id = {hit.chunk.chunk_id: hit for hit in sparse_hits}
        dense_by_id = {hit.chunk.chunk_id: hit for hit in dense_hits}
        fused_scores: dict[str, float] = {}
        for rank, hit in enumerate(sparse_hits, start=1):
            fused_scores[hit.chunk.chunk_id] = (
                fused_scores.get(hit.chunk.chunk_id, 0.0)
                + self.sparse_weight / (self.rrf_k + rank)
            )
        for rank, hit in enumerate(dense_hits, start=1):
            fused_scores[hit.chunk.chunk_id] = (
                fused_scores.get(hit.chunk.chunk_id, 0.0)
                + self.dense_weight / (self.rrf_k + rank)
            )

        fused: list[SearchHit] = []
        for chunk_id, score in fused_scores.items():
            sparse_hit = sparse_by_id.get(chunk_id)
            dense_hit = dense_by_id.get(chunk_id)
            base = sparse_hit or dense_hit
            assert base is not None
            fused.append(
                SearchHit(
                    chunk=base.chunk,
                    score=score,
                    matched_terms=sparse_hit.matched_terms if sparse_hit else (),
                    exact_match=sparse_hit.exact_match if sparse_hit else False,
                    matched_phrases=(
                        sparse_hit.matched_phrases if sparse_hit else ()
                    ),
                    retrieval_mode="hybrid",
                    sparse_score=sparse_hit.score if sparse_hit else None,
                    dense_score=dense_hit.score if dense_hit else None,
                )
            )
        fused.sort(
            key=lambda hit: (hit.score, hit.exact_match), reverse=True
        )
        return tuple(fused[:top_k])


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:
        raise DenseRetrievalError(
            "Thieu numpy/fastembed; chay pip install -r requirements.txt"
        ) from exc
    return numpy
