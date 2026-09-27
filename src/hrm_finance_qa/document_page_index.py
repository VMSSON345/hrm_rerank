from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Sequence

from .document_sources import (
    DocumentSourceError,
    EvidenceChunk,
    OCRConfig,
    SourceLine,
    _clean_line,
    _load_or_generate_mineru_pages,
    chunk_lines,
    _pdf_page_count,
    rank_chunks,
)


PAGE_INDEX_SCHEMA_VERSION = 2
PAGE_INDEX_CONTENT_VERSION = 9


class DocumentPageIndexError(DocumentSourceError):
    pass


@dataclass(frozen=True)
class PageRecord:
    """Index schema compatibility: native fields are empty in all new records."""

    document_fingerprint: str
    source_path: str
    source_name: str
    source_size: int
    source_mtime_ns: int
    page_number: int
    native_usable: bool
    native_score: float
    native_reason: str
    native_text: str
    extraction_method: str
    searchable_text: str
    ocr_state: str


@dataclass(frozen=True)
class PageIndexStats:
    total_pages: int = 0
    native_pages: int = 0
    mineru_pages: int = 0
    selected_mineru_pages: int = 0
    cache_hits: int = 0
    pages_ranked: int = 0
    pages_selected: int = 0
    documents_indexed: int = 0
    documents_rebuilt: int = 0
    index_seconds: float = 0.0
    selected_mineru_seconds: float = 0.0
    total_seconds: float = 0.0

    def render(self) -> str:
        return (
            f"total_pages={self.total_pages}, native_pages={self.native_pages}, "
            f"mineru_pages={self.mineru_pages}, "
            f"selected_mineru_pages={self.selected_mineru_pages}, "
            f"cache_hits={self.cache_hits}, pages_ranked={self.pages_ranked}, "
            f"pages_used_by_hrm={self.pages_selected}, "
            f"documents_indexed={self.documents_indexed}, "
            f"documents_rebuilt={self.documents_rebuilt}, "
            f"index_seconds={self.index_seconds:.3f}, "
            f"selected_mineru_seconds={self.selected_mineru_seconds:.3f}, "
            f"total_seconds={self.total_seconds:.3f}"
        )


@dataclass(frozen=True)
class PageIndexedChunks:
    chunks: tuple[EvidenceChunk, ...]
    stats: PageIndexStats
    warnings: tuple[str, ...] = ()
    ranked_pages: tuple[int, ...] = ()


class DocumentPageIndex:
    """Persistent MinerU-only index keyed by PDF fingerprint."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    @staticmethod
    def fingerprint(path: str | Path) -> str:
        source = Path(path).expanduser().resolve()
        stat = source.stat()
        identity = "|".join(
            (
                str(PAGE_INDEX_CONTENT_VERSION),
                str(source),
                str(stat.st_size),
                str(stat.st_mtime_ns),
            )
        )
        return blake2b(identity.encode("utf-8"), digest_size=16).hexdigest()

    def load_or_build(
        self,
        path: str | Path,
        *,
        ocr_config: OCRConfig,
        rebuild: bool = False,
    ) -> tuple[list[PageRecord], PageIndexStats]:
        source = Path(path).expanduser().resolve()
        if source.suffix.lower() != ".pdf":
            raise DocumentPageIndexError(
                f"Page-level indexing currently supports PDF only: {source.name}"
            )
        ocr_config.validate()
        if not ocr_config.enabled:
            raise DocumentPageIndexError("PDF indexing requires MinerU; prepare Markdown first. Native PDF reading is disabled.")
        fingerprint = self.fingerprint(source)
        started_at = time.perf_counter()
        try:
            with closing(self._connect()) as connection, connection:
                if not rebuild:
                    cached = self._load_records(connection, fingerprint)
                    complete = cached and self._is_complete(
                        connection, fingerprint, len(cached)
                    )
                    missing_mineru = ocr_config.enabled and any(
                        not item.native_usable and item.ocr_state == "unavailable"
                        for item in cached
                    )
                    if complete and not missing_mineru and all(item.extraction_method == "mineru" for item in cached):
                        return cached, PageIndexStats(
                            total_pages=len(cached),
                            native_pages=sum(item.native_usable for item in cached),
                            cache_hits=len(cached),
                            pages_ranked=len(cached),
                            documents_indexed=1,
                            index_seconds=time.perf_counter() - started_at,
                        )
                records, mineru_count = self._build_records(
                    source,
                    fingerprint,
                    ocr_config=ocr_config,
                )
                self._replace_document(connection, source, fingerprint, records)
                return records, PageIndexStats(
                    total_pages=len(records),
                    native_pages=sum(item.native_usable for item in records),
                    mineru_pages=mineru_count,
                    pages_ranked=len(records),
                    documents_indexed=1,
                    documents_rebuilt=1,
                    index_seconds=time.perf_counter() - started_at,
                )
        except (OSError, sqlite3.Error) as exc:
            raise DocumentPageIndexError(
                f"Could not update page index {self.path}: {exc}"
            ) from exc

    def update_records(self, records: Sequence[PageRecord]) -> None:
        if not records:
            return
        try:
            with closing(self._connect()) as connection, connection:
                connection.executemany(
                    """
                    UPDATE pages
                    SET extraction_method = ?, searchable_text = ?, ocr_state = ?,
                        updated_at = ?
                    WHERE document_fingerprint = ? AND page_number = ?
                    """,
                    [
                        (
                            item.extraction_method,
                            item.searchable_text,
                            item.ocr_state,
                            time.time(),
                            item.document_fingerprint,
                            item.page_number,
                        )
                        for item in records
                    ],
                )
        except (OSError, sqlite3.Error) as exc:
            raise DocumentPageIndexError(
                f"Could not persist selected MinerU pages in {self.path}: {exc}"
            ) from exc

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version == 1:
            connection.executescript(
                "DROP TABLE IF EXISTS pages; DROP TABLE IF EXISTS documents;"
            )
            version = 0
        if version not in {0, PAGE_INDEX_SCHEMA_VERSION}:
            connection.close()
            raise DocumentPageIndexError(
                f"Unsupported page-index schema {version}; rebuild {self.path}"
            )
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_fingerprint TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_size INTEGER NOT NULL,
                source_mtime_ns INTEGER NOT NULL,
                page_count INTEGER NOT NULL,
                indexed_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS documents_source_path
                ON documents(source_path);
            CREATE TABLE IF NOT EXISTS pages (
                document_fingerprint TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                native_usable INTEGER NOT NULL,
                native_score REAL NOT NULL,
                native_reason TEXT NOT NULL,
                native_text TEXT NOT NULL,
                extraction_method TEXT NOT NULL,
                searchable_text TEXT NOT NULL,
                ocr_state TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY(document_fingerprint, page_number),
                FOREIGN KEY(document_fingerprint)
                    REFERENCES documents(document_fingerprint) ON DELETE CASCADE
            );
            """
        )
        connection.execute(f"PRAGMA user_version = {PAGE_INDEX_SCHEMA_VERSION}")
        return connection

    @staticmethod
    def _is_complete(
        connection: sqlite3.Connection, fingerprint: str, record_count: int
    ) -> bool:
        row = connection.execute(
            "SELECT page_count FROM documents WHERE document_fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        return row is not None and int(row["page_count"]) == record_count

    @staticmethod
    def _load_records(
        connection: sqlite3.Connection, fingerprint: str
    ) -> list[PageRecord]:
        rows = connection.execute(
            """
            SELECT p.*, d.source_path, d.source_name, d.source_size,
                   d.source_mtime_ns
            FROM pages p
            JOIN documents d USING(document_fingerprint)
            WHERE p.document_fingerprint = ?
            ORDER BY p.page_number
            """,
            (fingerprint,),
        ).fetchall()
        return [
            PageRecord(
                document_fingerprint=str(row["document_fingerprint"]),
                source_path=str(row["source_path"]),
                source_name=str(row["source_name"]),
                source_size=int(row["source_size"]),
                source_mtime_ns=int(row["source_mtime_ns"]),
                page_number=int(row["page_number"]),
                native_usable=bool(row["native_usable"]),
                native_score=float(row["native_score"]),
                native_reason=str(row["native_reason"]),
                native_text=str(row["native_text"]),
                extraction_method=str(row["extraction_method"]),
                searchable_text=str(row["searchable_text"]),
                ocr_state=str(row["ocr_state"]),
            )
            for row in rows
        ]

    def _build_records(
        self, path: Path, fingerprint: str, *, ocr_config: OCRConfig,
    ) -> tuple[list[PageRecord], int]:
        page_count = _pdf_page_count(path)
        pages = _load_or_generate_mineru_pages(path, [True] * page_count, ocr_config)
        if not any(pages.values()):
            raise DocumentPageIndexError(f"MinerU found no usable content in {path.name}")
        stat = path.stat()
        records = [
            PageRecord(
                document_fingerprint=fingerprint, source_path=str(path),
                source_name=path.name, source_size=stat.st_size,
                source_mtime_ns=stat.st_mtime_ns, page_number=number,
                native_usable=False, native_score=0.0, native_reason="disabled",
                native_text="", extraction_method="mineru",
                searchable_text="\n".join(pages.get(number, [])),
                ocr_state="high_quality",
            )
            for number in range(1, page_count + 1)
        ]
        return records, page_count

    @staticmethod
    def _replace_document(
        connection: sqlite3.Connection,
        source: Path,
        fingerprint: str,
        records: Sequence[PageRecord],
    ) -> None:
        stat = source.stat()
        connection.execute(
            "DELETE FROM documents WHERE source_path = ?",
            (str(source),),
        )
        connection.execute(
            """
            INSERT INTO documents(
                document_fingerprint, source_path, source_name, source_size,
                source_mtime_ns, page_count, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fingerprint,
                str(source),
                source.name,
                stat.st_size,
                stat.st_mtime_ns,
                len(records),
                time.time(),
            ),
        )
        connection.executemany(
            """
            INSERT INTO pages(
                document_fingerprint, page_number, native_usable, native_score,
                native_reason, native_text, extraction_method, searchable_text,
                ocr_state, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item.document_fingerprint,
                    item.page_number,
                    int(item.native_usable),
                    item.native_score,
                    item.native_reason,
                    item.native_text,
                    item.extraction_method,
                    item.searchable_text,
                    item.ocr_state,
                    time.time(),
                )
                for item in records
            ],
        )


def build_page_indexed_chunks(
    documents: Sequence[str | Path],
    question: str,
    *,
    index_path: str | Path,
    ocr_config: OCRConfig,
    top_k_pages: int = 6,
    neighbor_pages: int = 1,
    chunk_lines_count: int = 28,
    overlap_lines: int = 6,
    rebuild: bool = False,
    prefer_mineru_for_selected_pages: bool = False,
) -> PageIndexedChunks:
    started_at = time.perf_counter()
    if top_k_pages <= 0:
        raise DocumentPageIndexError("top_k_pages must be positive")
    if neighbor_pages < 0:
        raise DocumentPageIndexError("neighbor_pages cannot be negative")
    sources = [Path(item).expanduser().resolve() for item in documents]
    if any(source.suffix.lower() != ".pdf" for source in sources):
        raise DocumentPageIndexError(
            "Page-level index requires PDF; read prepared Markdown/TXT with build_chunks"
        )

    index = DocumentPageIndex(index_path)
    records: list[PageRecord] = []
    stats_items: list[PageIndexStats] = []
    for source in sources:
        index_ocr_config = ocr_config
        document_records, document_stats = index.load_or_build(
            source,
            ocr_config=index_ocr_config,
            rebuild=rebuild,
        )
        records.extend(document_records)
        stats_items.append(document_stats)

    ranked_records = (
        list(records)
        if ocr_config.force
        else _rank_page_records(
            question,
            records,
            top_k=top_k_pages,
            ocr_enabled=ocr_config.enabled,
        )
    )
    warnings: list[str] = []
    if not ranked_records and records:
        ranked_records = records[:top_k_pages]
        warnings.append(
            "Page retrieval found no lexical match; using the first indexed pages."
        )
    selected = (
        list(records)
        if ocr_config.force
        else _expand_page_neighbors(ranked_records, records, neighbor_pages)
    )
    mineru_started_at = time.perf_counter()
    (
        materialized,
        selected_mineru_pages,
        selected_cache_hits,
        materialize_warnings,
    ) = _materialize_selected_pages(
        selected,
        all_records=records,
        index=index,
        ocr_config=ocr_config,
        prefer_mineru_for_selected_pages=prefer_mineru_for_selected_pages,
    )
    warnings.extend(materialize_warnings)
    lines = _records_to_lines(materialized, ocr_enabled=ocr_config.enabled)
    chunks = chunk_lines(
        lines,
        chunk_lines_count=chunk_lines_count,
        overlap_lines=overlap_lines,
    )
    stats = PageIndexStats(
        total_pages=sum(item.total_pages for item in stats_items),
        native_pages=sum(item.native_pages for item in stats_items),
        mineru_pages=sum(item.mineru_pages for item in stats_items),
        selected_mineru_pages=selected_mineru_pages,
        cache_hits=sum(item.cache_hits for item in stats_items) + selected_cache_hits,
        pages_ranked=len(records),
        pages_selected=len({(item.source_path, item.page) for item in lines}),
        documents_indexed=len(stats_items),
        documents_rebuilt=sum(item.documents_rebuilt for item in stats_items),
        index_seconds=sum(item.index_seconds for item in stats_items),
        selected_mineru_seconds=time.perf_counter() - mineru_started_at,
        total_seconds=time.perf_counter() - started_at,
    )
    return PageIndexedChunks(
        tuple(chunks),
        stats,
        tuple(warnings),
        tuple(item.page_number for item in ranked_records),
    )


def _rank_page_records(
    question: str,
    records: Sequence[PageRecord],
    *,
    top_k: int,
    ocr_enabled: bool,
) -> list[PageRecord]:
    by_id: dict[str, PageRecord] = {}
    chunks: list[EvidenceChunk] = []
    for item in records:
        chunk_id = f"page-index:{item.document_fingerprint}:{item.page_number}"
        by_id[chunk_id] = item
        chunks.append(
            EvidenceChunk(
                chunk_id=chunk_id,
                source_path=item.source_path,
                source_name=item.source_name,
                page_start=item.page_number,
                page_end=item.page_number,
                line_start=item.page_number * 100_000,
                line_end=item.page_number * 100_000,
                text=item.searchable_text,
                extraction_methods=(item.extraction_method,),
            )
        )
    return [by_id[chunk.chunk_id] for chunk in rank_chunks(question, chunks, top_k=top_k)]


def _expand_page_neighbors(
    ranked: Sequence[PageRecord],
    all_records: Sequence[PageRecord],
    neighbor_pages: int,
) -> list[PageRecord]:
    lookup = {
        (item.document_fingerprint, item.page_number): item for item in all_records
    }
    selected: dict[tuple[str, int], PageRecord] = {}
    for hit in ranked:
        for offset in range(-neighbor_pages, neighbor_pages + 1):
            key = (hit.document_fingerprint, hit.page_number + offset)
            if key in lookup:
                selected.setdefault(key, lookup[key])
    return sorted(
        selected.values(), key=lambda item: (item.source_path, item.page_number)
    )


def _materialize_selected_pages(
    selected: Sequence[PageRecord], *, all_records: Sequence[PageRecord],
    index: DocumentPageIndex, ocr_config: OCRConfig,
    prefer_mineru_for_selected_pages: bool = False,
) -> tuple[list[PageRecord], int, int, list[str]]:
    # Every page is prepared with MinerU before retrieval; there is no native override.
    if not ocr_config.enabled or any(item.extraction_method != "mineru" for item in selected):
        raise DocumentPageIndexError("Only prepared MinerU pages may supply PDF evidence")
    return list(selected), 0, len(selected), []


def _has_compatible_high_quality_ocr(
    item: PageRecord, config: OCRConfig
) -> bool:
    del config
    return item.ocr_state == "high_quality" and item.extraction_method == "mineru"


def _records_to_lines(
    records: Sequence[PageRecord], *, ocr_enabled: bool,
) -> list[SourceLine]:
    if not ocr_enabled or any(item.extraction_method != "mineru" for item in records):
        raise DocumentPageIndexError("Only prepared MinerU pages may supply PDF evidence")
    output: list[SourceLine] = []
    for item in records:
        for local_line, raw in enumerate(item.searchable_text.splitlines(), start=1):
            if cleaned := _clean_line(raw):
                output.append(SourceLine(
                    source_path=item.source_path, source_name=item.source_name,
                    page=item.page_number, line_number=item.page_number * 100_000 + local_line,
                    text=cleaned, extraction_method="mineru",
                ))
    return output
