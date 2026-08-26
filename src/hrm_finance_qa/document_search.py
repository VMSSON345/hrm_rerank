from __future__ import annotations

import json
import hashlib
import math
import re
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .ontology import normalize_text


INDEX_SCHEMA_VERSION = 1

_HEADING_MARKERS = (
    "muc luc",
    "table of contents",
    "bao cao tai chinh",
    "financial statements",
    "bao cao cua hoi dong quan tri",
    "statement of the board",
    "bao cao kiem toan",
    "independent auditor",
    "bang can doi ke toan",
    "balance sheet",
    "statement of financial position",
    "bao cao ket qua hoat dong kinh doanh",
    "income statement",
    "statement of income",
    "bao cao luu chuyen tien te",
    "cash flow statement",
    "thuyet minh bao cao tai chinh",
    "notes to the financial statements",
    "co so lap bao cao",
    "basis of preparation",
    "chinh sach ke toan",
    "accounting policies",
)

_STRICT_HEADING_MARKERS = (
    "bao cao tinh hinh tai chinh",
    "bao cao tinh hinh tai chinh hop nhat",
    "bao cao tinh hinh tai chinh rieng",
    "bao cao luu chuyen tien te rieng",
    "bao cao tinh hinh bien dong von chu so huu",
)

_ALL_HEADING_MARKERS = _HEADING_MARKERS + _STRICT_HEADING_MARKERS

_HEADING_NOISE = (
    "digitally signed",
    "foxit reader",
    "location",
    "reason i am",
    "telephone",
    "website",
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "cac",
    "cho",
    "co",
    "cua",
    "da",
    "de",
    "duoc",
    "he",
    "la",
    "mot",
    "nam",
    "nay",
    "nhung",
    "so",
    "theo",
    "tinh",
    "trong",
    "tu",
    "va",
    "ve",
}


class DocumentSearchError(ValueError):
    pass


@dataclass(frozen=True)
class DocumentSection:
    section_id: str
    heading: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class DocumentOutline:
    document_id: str
    source_name: str
    reporting_year: int | None
    sections: tuple[DocumentSection, ...]


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    source_name: str
    section_id: str
    heading: str
    start_line: int
    end_line: int
    text: str


@dataclass(frozen=True)
class SearchHit:
    chunk: DocumentChunk
    score: float
    matched_terms: tuple[str, ...]
    exact_match: bool
    matched_phrases: tuple[str, ...] = ()
    retrieval_mode: str = "sparse"
    sparse_score: float | None = None
    dense_score: float | None = None


@dataclass(frozen=True)
class _DocumentRecord:
    document_id: str
    source_name: str
    reporting_year: int | None
    lines: tuple[str, ...]
    sections: tuple[DocumentSection, ...]


class VietnameseFinancialDocumentIndex:
    """Searchable structure over already-extracted financial report text."""

    def __init__(
        self,
        documents: tuple[_DocumentRecord, ...],
        *,
        chunk_lines: int = 80,
        overlap_lines: int = 15,
        source_dir: str | None = None,
    ) -> None:
        if not documents:
            raise DocumentSearchError("Khong co tai lieu de lap chi muc")
        if chunk_lines < 10:
            raise DocumentSearchError("chunk_lines phai lon hon hoac bang 10")
        if overlap_lines < 0 or overlap_lines >= chunk_lines:
            raise DocumentSearchError(
                "overlap_lines phai nam trong khoang 0 den chunk_lines - 1"
            )

        self.chunk_lines = chunk_lines
        self.overlap_lines = overlap_lines
        self.source_dir = source_dir
        self._documents = {item.document_id: item for item in documents}
        if len(self._documents) != len(documents):
            raise DocumentSearchError("document_id bi trung trong chi muc")

        self._chunks = self._build_chunks(documents)
        self._chunk_lookup = {item.chunk_id: item for item in self._chunks}
        self._fingerprint: str | None = None
        self._build_bm25_statistics()

    @classmethod
    def build_from_text_dir(
        cls,
        text_dir: str | Path,
        *,
        chunk_lines: int = 80,
        overlap_lines: int = 15,
    ) -> "VietnameseFinancialDocumentIndex":
        source = Path(text_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"Khong tim thay thu muc text: {source}")

        paths = sorted(source.rglob("*.txt"), key=lambda item: item.name.lower())
        if not paths:
            raise DocumentSearchError(f"Khong co file .txt trong {source}")

        records: list[_DocumentRecord] = []
        used_ids: set[str] = set()
        for path in paths:
            document_id = path.stem
            if document_id in used_ids:
                relative = path.relative_to(source).as_posix()
                document_id = normalize_text(relative).replace(" ", "_")
            used_ids.add(document_id)
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            lines = tuple(text.splitlines())
            if not lines:
                continue
            records.append(
                _DocumentRecord(
                    document_id=document_id,
                    source_name=path.name,
                    reporting_year=_infer_reporting_year(path.name, lines),
                    lines=lines,
                    sections=_detect_sections(lines),
                )
            )

        return cls(
            tuple(records),
            chunk_lines=chunk_lines,
            overlap_lines=overlap_lines,
            source_dir=str(source.resolve()),
        )

    @classmethod
    def load(cls, path: str | Path) -> "VietnameseFinancialDocumentIndex":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Khong tim thay document index: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema_version") != INDEX_SCHEMA_VERSION:
            raise DocumentSearchError(
                f"Phien ban document index khong duoc ho tro: "
                f"{payload.get('schema_version')}"
            )

        records: list[_DocumentRecord] = []
        for raw_document in payload.get("documents", ()):
            lines = tuple(raw_document["lines"])
            sections = tuple(
                DocumentSection(
                    section_id=item["section_id"],
                    heading=item["heading"],
                    start_line=int(item["start_line"]),
                    end_line=int(item["end_line"]),
                )
                for item in raw_document["sections"]
            )
            records.append(
                _DocumentRecord(
                    document_id=raw_document["document_id"],
                    source_name=raw_document["source_name"],
                    reporting_year=raw_document.get("reporting_year")
                    or _infer_reporting_year(raw_document["source_name"], lines),
                    lines=lines,
                    sections=sections,
                )
            )
        return cls(
            tuple(records),
            chunk_lines=int(payload["chunk_lines"]),
            overlap_lines=int(payload["overlap_lines"]),
            source_dir=payload.get("source_dir"),
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "source_dir": self.source_dir,
            "chunk_lines": self.chunk_lines,
            "overlap_lines": self.overlap_lines,
            "documents": [
                {
                    "document_id": document.document_id,
                    "source_name": document.source_name,
                    "reporting_year": document.reporting_year,
                    "lines": document.lines,
                    "sections": [
                        {
                            "section_id": section.section_id,
                            "heading": section.heading,
                            "start_line": section.start_line,
                            "end_line": section.end_line,
                        }
                        for section in document.sections
                    ],
                }
                for document in self._documents.values()
            ],
        }
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        return destination.resolve()

    def stats(self) -> dict[str, int]:
        return {
            "documents": len(self._documents),
            "sections": sum(len(item.sections) for item in self._documents.values()),
            "chunks": len(self._chunks),
            "lines": sum(len(item.lines) for item in self._documents.values()),
            "terms": len(self._document_frequency),
        }

    def document_ids(self) -> tuple[str, ...]:
        return tuple(self._documents)

    def chunks(self) -> tuple[DocumentChunk, ...]:
        return self._chunks

    def get_chunk(self, chunk_id: str) -> DocumentChunk:
        try:
            return self._chunk_lookup[chunk_id]
        except KeyError:
            raise DocumentSearchError(f"Khong tim thay chunk_id: {chunk_id}") from None

    def fingerprint(self) -> str:
        if self._fingerprint is not None:
            return self._fingerprint
        digest = hashlib.sha256()
        for chunk in self._chunks:
            digest.update(chunk.chunk_id.encode("utf-8"))
            digest.update(b"\0")
            digest.update(chunk.text.encode("utf-8"))
            digest.update(b"\0")
        self._fingerprint = digest.hexdigest()
        return self._fingerprint

    def resolve_document_id(self, value: str) -> str:
        if value in self._documents:
            return value
        normalized = normalize_text(value)
        matches = [
            document_id
            for document_id, document in self._documents.items()
            if normalized
            and (
                normalized == normalize_text(document_id)
                or normalized in normalize_text(document_id)
                or normalized in normalize_text(document.source_name)
            )
        ]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise DocumentSearchError(f"Khong tim thay document_id: {value}")
        examples = ", ".join(matches[:5])
        raise DocumentSearchError(
            f"document_id '{value}' khong duy nhat; vi du: {examples}"
        )

    def outline(self, document_id: str) -> DocumentOutline:
        resolved = self.resolve_document_id(document_id)
        document = self._documents[resolved]
        return DocumentOutline(
            document_id=resolved,
            source_name=document.source_name,
            reporting_year=document.reporting_year,
            sections=document.sections,
        )

    def get_section_content(self, document_id: str, section_id: str) -> str:
        resolved = self.resolve_document_id(document_id)
        document = self._documents[resolved]
        section = next(
            (item for item in document.sections if item.section_id == section_id), None
        )
        if section is None:
            raise DocumentSearchError(
                f"Khong tim thay section {section_id} trong {resolved}"
            )
        return self.get_line_range(
            resolved, section.start_line, section.end_line, numbered=True
        )

    def get_line_range(
        self,
        document_id: str,
        start_line: int,
        end_line: int,
        *,
        numbered: bool = True,
    ) -> str:
        resolved = self.resolve_document_id(document_id)
        document = self._documents[resolved]
        if start_line < 1 or end_line < start_line or end_line > len(document.lines):
            raise DocumentSearchError(
                f"Khoang dong khong hop le: {start_line}-{end_line}; "
                f"tai lieu co {len(document.lines)} dong"
            )
        selected = document.lines[start_line - 1 : end_line]
        if not numbered:
            return "\n".join(selected)
        return "\n".join(
            f"{line_number}: {line}"
            for line_number, line in enumerate(selected, start=start_line)
        )

    def search(
        self,
        query: str,
        *,
        document_id: str | None = None,
        top_k: int = 15,
    ) -> tuple[SearchHit, ...]:
        if top_k < 1:
            raise DocumentSearchError("top_k phai lon hon hoac bang 1")
        query_tokens = _tokenize(query)
        if not query_tokens:
            raise DocumentSearchError("Truy van khong co tu khoa hop le")

        resolved_document = (
            self.resolve_document_id(document_id) if document_id else None
        )
        query_counts = Counter(query_tokens)
        query_phrases = tuple(
            dict.fromkeys(
                phrase
                for phrase in (normalize_text(part) for part in query.split("|"))
                if len(phrase) >= 4
            )
        )
        total_chunks = len(self._chunks)
        scored: list[SearchHit] = []

        for position, chunk in enumerate(self._chunks):
            if resolved_document and chunk.document_id != resolved_document:
                continue
            term_counts = self._term_counts[position]
            chunk_length = self._chunk_lengths[position]
            score = 0.0
            matched: list[str] = []
            for term, query_frequency in query_counts.items():
                frequency = term_counts.get(term, 0)
                if not frequency:
                    continue
                matched.append(term)
                document_frequency = self._document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0
                    + (total_chunks - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                normalization = frequency + 1.5 * (
                    1.0
                    - 0.75
                    + 0.75 * chunk_length / self._average_chunk_length
                )
                score += (
                    inverse_document_frequency
                    * (frequency * 2.5 / normalization)
                    * min(query_frequency, 2)
                )

            if not matched:
                continue
            normalized_chunk = self._normalized_chunks[position]
            matched_phrases = tuple(
                phrase for phrase in query_phrases if phrase in normalized_chunk
            )
            exact_match = bool(matched_phrases)
            if exact_match:
                score += sum(
                    3.0 + min(6, len(phrase.split())) * 0.5
                    for phrase in matched_phrases
                )
            normalized_heading = normalize_text(chunk.heading)
            heading_matches = sum(
                1 for term in set(query_tokens) if term in normalized_heading.split()
            )
            score += heading_matches * 0.75
            scored.append(
                SearchHit(
                    chunk=chunk,
                    score=score,
                    matched_terms=tuple(dict.fromkeys(matched)),
                    exact_match=exact_match,
                    matched_phrases=matched_phrases,
                )
            )

        scored.sort(
            key=lambda item: (
                item.score,
                item.exact_match,
                -item.chunk.start_line,
            ),
            reverse=True,
        )
        return tuple(scored[:top_k])

    def exact_search(
        self,
        keyword: str,
        *,
        document_id: str | None = None,
        top_k: int = 50,
    ) -> tuple[SearchHit, ...]:
        if top_k < 1:
            raise DocumentSearchError("top_k phai lon hon hoac bang 1")
        normalized_keyword = normalize_text(keyword)
        if not normalized_keyword:
            raise DocumentSearchError("Tu khoa chinh xac khong hop le")
        resolved_document = (
            self.resolve_document_id(document_id) if document_id else None
        )
        results: list[SearchHit] = []
        keyword_terms = tuple(dict.fromkeys(_tokenize(keyword)))
        for position, chunk in enumerate(self._chunks):
            if resolved_document and chunk.document_id != resolved_document:
                continue
            if normalized_keyword not in self._normalized_chunks[position]:
                continue
            results.append(
                SearchHit(
                    chunk=chunk,
                    score=1.0,
                    matched_terms=keyword_terms,
                    exact_match=True,
                    matched_phrases=(normalized_keyword,),
                )
            )
            if len(results) >= top_k:
                break
        return tuple(results)

    def _build_chunks(
        self, documents: tuple[_DocumentRecord, ...]
    ) -> tuple[DocumentChunk, ...]:
        chunks: list[DocumentChunk] = []
        step = self.chunk_lines - self.overlap_lines
        for document in documents:
            section_starts = [item.start_line for item in document.sections]
            chunk_number = 1
            for start_line in range(1, len(document.lines) + 1, step):
                end_line = min(start_line + self.chunk_lines - 1, len(document.lines))
                section_position = max(0, bisect_right(section_starts, start_line) - 1)
                section = document.sections[section_position]
                text = "\n".join(document.lines[start_line - 1 : end_line]).strip()
                if not text:
                    continue
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document.document_id}::C{chunk_number:04d}",
                        document_id=document.document_id,
                        source_name=document.source_name,
                        section_id=section.section_id,
                        heading=section.heading,
                        start_line=start_line,
                        end_line=end_line,
                        text=text,
                    )
                )
                chunk_number += 1
        if not chunks:
            raise DocumentSearchError("Khong tao duoc chunk tu tai lieu")
        return tuple(chunks)

    def _build_bm25_statistics(self) -> None:
        term_counts: list[Counter[str]] = []
        document_frequency: Counter[str] = Counter()
        chunk_lengths: list[int] = []
        normalized_chunks: list[str] = []
        for chunk in self._chunks:
            normalized_chunk = normalize_text(chunk.text)
            normalized_chunks.append(normalized_chunk)
            tokens = _tokenize_normalized(normalized_chunk)
            counts = Counter(tokens)
            counts.update(_tokenize(chunk.heading))
            term_counts.append(counts)
            document_frequency.update(counts.keys())
            chunk_lengths.append(sum(counts.values()))

        self._term_counts = tuple(term_counts)
        self._document_frequency = document_frequency
        self._chunk_lengths = tuple(chunk_lengths)
        self._normalized_chunks = tuple(normalized_chunks)
        self._average_chunk_length = max(
            1.0, sum(chunk_lengths) / len(chunk_lengths)
        )


def _detect_sections(lines: tuple[str, ...]) -> tuple[DocumentSection, ...]:
    headings: list[tuple[int, str]] = []
    for line_number, raw_line in enumerate(lines, start=1):
        heading = _clean_heading(raw_line)
        if _is_heading(heading):
            headings.append((line_number, heading))

    if not headings or headings[0][0] > 1:
        headings.insert(0, (1, "Document start"))

    sections: list[DocumentSection] = []
    for position, (start_line, heading) in enumerate(headings):
        end_line = (
            headings[position + 1][0] - 1
            if position + 1 < len(headings)
            else len(lines)
        )
        sections.append(
            DocumentSection(
                section_id=f"S{position + 1:04d}",
                heading=heading,
                start_line=start_line,
                end_line=max(start_line, end_line),
            )
        )
    return tuple(sections)


def _infer_reporting_year(source_name: str, lines: tuple[str, ...]) -> int | None:
    name_without_id = re.sub(r"^\d+_", "", source_name)
    name_years = [
        int(value)
        for value in re.findall(r"20\d{2}", name_without_id)
        if 2000 <= int(value) <= 2035
    ]
    if name_years:
        return max(name_years)

    candidates: Counter[int] = Counter()
    report_markers = (
        "bao cao tai chinh",
        "financial report",
        "financial statements",
        "year ended",
        "nam tai chinh ket thuc",
    )
    for line in lines[:300]:
        normalized = normalize_text(line)
        if not any(marker in normalized for marker in report_markers):
            continue
        for raw_year in re.findall(r"20\d{2}", normalized):
            year = int(raw_year)
            if 2000 <= year <= 2035:
                candidates[year] += 1
    return candidates.most_common(1)[0][0] if candidates else None


def _clean_heading(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^\d{1,3}(?=[^\W\d_])", "", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip(" -_|^")


def _is_heading(value: str) -> bool:
    if len(value) < 4 or len(value) > 180:
        return False
    normalized = normalize_text(value)
    if not normalized or any(noise in normalized for noise in _HEADING_NOISE):
        return False
    if "@" in value or "http" in normalized or value.count(":") > 1:
        return False

    letters = [character for character in value if character.isalpha()]
    if len(letters) < 4:
        return False
    words = normalized.split()
    if len(words) < 2 or len(words) > 18:
        return False
    if _looks_like_toc_entry(value, normalized):
        return False
    uppercase_ratio = sum(character.isupper() for character in letters) / len(letters)
    marker_match = any(marker in normalized for marker in _HEADING_MARKERS)
    if _matches_known_heading(normalized):
        return True
    if marker_match and (uppercase_ratio >= 0.45 or value.istitle()):
        return True
    if re.match(r"^\d{1,2}[.)]\s+", value) and uppercase_ratio >= 0.55:
        return True
    if value.endswith((".", ";", ",")):
        return False
    return uppercase_ratio >= 0.82


def _matches_known_heading(normalized: str) -> bool:
    return normalized in _ALL_HEADING_MARKERS


def _looks_like_toc_entry(value: str, normalized: str) -> bool:
    if re.search(r"(?:\.{2,}|…+|(?:[-_]\s*){3,})\s*\d{1,4}\s*$", value):
        return True
    words = normalized.split()
    if not words or not words[-1].isdigit() or len(words[-1]) > 3:
        return False
    heading_without_page = " ".join(words[:-1])
    return any(marker in heading_without_page for marker in _ALL_HEADING_MARKERS)


def _tokenize(value: str) -> tuple[str, ...]:
    return _tokenize_normalized(normalize_text(value))


def _tokenize_normalized(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in value.split()
        if token not in _STOPWORDS and (len(token) > 1 or token.isdigit())
    )
