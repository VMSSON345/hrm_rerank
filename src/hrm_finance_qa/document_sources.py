from __future__ import annotations

import json
import math
import re
import subprocess
import urllib.error
import urllib.request
import uuid
from collections import Counter
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from hashlib import blake2b
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Sequence

from .ontology import decode_pdf_unicode_glyph_names, normalize_text
from .pdf_metadata import pdf_page_count


SUPPORTED_DOCUMENT_SUFFIXES = {".md", ".txt"}
MINERU_CACHE_VERSION = 1


class DocumentSourceError(ValueError):
    pass


@dataclass(frozen=True)
class OCRConfig:
    enabled: bool = False
    force: bool = False
    mineru_output_dir: Path | None = None
    mineru_page_offset: int = 0
    mineru_cache_dir: Path = Path("cache/mineru")
    mineru_command: Path = Path(".venv-mineru/bin/mineru")
    mineru_api_url: str | None = None
    mineru_api_timeout_seconds: float = 7200.0
    mineru_api_page_batch_size: int = 4
    mineru_backend: str = "hybrid-engine"
    mineru_effort: str = "medium"
    mineru_auto_generate: bool = True

    def validate(self) -> None:
        if self.mineru_page_offset < 0:
            raise DocumentSourceError("MinerU page offset cannot be negative")
        if self.mineru_backend not in {"hybrid-engine", "pipeline"}:
            raise DocumentSourceError(
                "MinerU backend must be 'hybrid-engine' or 'pipeline'"
            )
        if self.mineru_effort not in {"low", "medium", "high"}:
            raise DocumentSourceError("MinerU effort must be low, medium, or high")
        if self.mineru_api_timeout_seconds <= 0:
            raise DocumentSourceError("MinerU API timeout must be positive")
        if self.mineru_api_page_batch_size <= 0:
            raise DocumentSourceError("MinerU API page batch size must be positive")
        if self.mineru_api_url is not None and not self.mineru_api_url.startswith(
            ("http://", "https://")
        ):
            raise DocumentSourceError("MinerU API URL must use http:// or https://")
        if self.mineru_api_url is not None and self.mineru_effort == "low":
            raise DocumentSourceError(
                "Persistent MinerU API supports medium or high effort"
            )




@dataclass(frozen=True)
class StructuredSourceCell:
    page: int
    bbox: tuple[float, float, float, float]
    source_span: tuple[int, int]
    table_id: str
    row_id: str
    column_id: str
    row_label: str
    column_label: str
    column_role: str
    period: str
    raw_value: str
    value: str
    unit: str
    extraction_method: str = "structured_table"


@dataclass(frozen=True)
class SourceLine:
    source_path: str
    source_name: str
    page: int | None
    line_number: int
    text: str
    extraction_method: str = "native"
    structured_cells: tuple[StructuredSourceCell, ...] = ()


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    source_path: str
    source_name: str
    page_start: int | None
    page_end: int | None
    line_start: int
    line_end: int
    text: str
    score: float = 0.0
    extraction_methods: tuple[str, ...] = ()
    structured_cells: tuple[StructuredSourceCell, ...] = ()

    source_annotations: tuple[dict, ...] = ()

    def citation(self) -> str:
        if "mineru" in self.extraction_methods:
            method = " [MinerU OCR]"
        elif "ocr" in self.extraction_methods:
            method = " [OCR]"
        else:
            method = ""
        if self.page_start is not None:
            if self.page_end == self.page_start:
                return f"{self.source_name}, page {self.page_start}{method}"
            return f"{self.source_name}, pages {self.page_start}-{self.page_end}{method}"
        return f"{self.source_name}, lines {self.line_start}-{self.line_end}{method}"


def discover_documents(path: str | Path) -> list[Path]:
    root = Path(path).expanduser().resolve()
    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_DOCUMENT_SUFFIXES:
            if root.suffix.lower() == ".pdf":
                raise DocumentSourceError(
                    "Direct PDF question answering is disabled. Run "
                    "prepare-document to create a MinerU Markdown artifact first."
                )
            raise DocumentSourceError(f"Unsupported document type: {root.suffix}")
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Document path not found: {root}")
    files = [
        item
        for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_DOCUMENT_SUFFIXES
    ]
    if not files:
        raise DocumentSourceError(f"No Markdown/TXT documents found under: {root}")
    return sorted(files)


def read_document(
    path: str | Path,
    *,
    ocr_config: OCRConfig | None = None,
) -> list[SourceLine]:
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix in {".md", ".txt"}:
        return _read_text(source)
    if suffix == ".pdf":
        raise DocumentSourceError(
            "Direct PDF question answering is disabled. Run prepare-document "
            "to create a MinerU Markdown artifact first."
        )
    raise DocumentSourceError(f"Unsupported document type: {suffix}")












def _pdf_page_count(path: Path) -> int:
    try:
        return pdf_page_count(path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise DocumentSourceError(str(exc)) from exc


def materialize_mineru_markdown(
    path: str | Path,
    output: str | Path,
    *,
    ocr_config: OCRConfig | None = None,
) -> Path:
    """Run MinerU for the complete PDF and persist deterministic Markdown."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PDF not found: {source}")
    if source.suffix.lower() != ".pdf":
        raise DocumentSourceError(
            f"MinerU preparation requires a PDF input: {source.name}"
        )
    config = replace(ocr_config or OCRConfig(), enabled=True, force=True)
    config.validate()
    page_count = _pdf_page_count(source)
    pages = _load_or_generate_mineru_pages(
        source,
        [True] * page_count,
        config,
    )
    if not any(lines for lines in pages.values()):
        raise DocumentSourceError(
            f"MinerU found no usable text in {source.name}. Check its artifact."
        )

    requested = Path(output).expanduser().resolve()
    destination = (
        requested
        if requested.suffix.lower() == ".md"
        else requested / f"{source.stem}.md"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = [
        f"# {source.stem}",
        "",
        f"<!-- source: {source.name}; extractor: MinerU OCR -->",
    ]
    for page_number in range(1, page_count + 1):
        rendered.extend(("", f"## Page {page_number}", ""))
        rendered.extend(pages.get(page_number, ()))
    destination.write_text(
        "\n".join(rendered).rstrip() + "\n",
        encoding="utf-8",
    )
    return destination


def _load_or_generate_mineru_pages(
    path: Path,
    selected_pages: Sequence[bool],
    config: OCRConfig,
    *,
    _allow_api_split: bool = True,
) -> dict[int, list[str]]:
    """Load a supplied MinerU artifact or create a fingerprinted cached job.

    A fully scanned PDF has no searchable representation before OCR. In that
    case MinerU must process the selected page range once; the SQLite page index
    then makes subsequent questions artifact-only and source-scan-free.
    """
    selected_indexes = [
        index for index, selected in enumerate(selected_pages) if selected
    ]
    if not selected_indexes:
        return {}
    if config.mineru_output_dir is not None:
        return _load_mineru_pages(path, config)

    start_page = min(selected_indexes)
    end_page = max(selected_indexes)
    stat = path.stat()
    identity = "|".join(
        (
            str(MINERU_CACHE_VERSION),
            str(path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(start_page),
            str(end_page),
            config.mineru_backend,
            config.mineru_effort,
            (
                f"api:{config.mineru_api_url.rstrip('/')}"
                if config.mineru_api_url
                else f"cli:{Path(config.mineru_command).expanduser()}"
            ),
        )
    )
    digest = blake2b(identity.encode("utf-8"), digest_size=12).hexdigest()
    job_root = (
        Path(config.mineru_cache_dir).expanduser().resolve()
        / f"{path.stem}-{digest}"
    )
    generated_config = replace(
        config,
        mineru_output_dir=job_root,
        mineru_page_offset=start_page,
    )
    try:
        return _load_mineru_pages(path, generated_config)
    except DocumentSourceError:
        pass
    contiguous_runs: list[list[int]] = []
    for page_index in selected_indexes:
        if (
            not contiguous_runs
            or page_index != contiguous_runs[-1][-1] + 1
        ):
            contiguous_runs.append([])
        contiguous_runs[-1].append(page_index)
    if len(contiguous_runs) > 1:
        combined: dict[int, list[str]] = {}
        for run in contiguous_runs:
            run_selected = [
                page_index in set(run)
                for page_index in range(len(selected_pages))
            ]
            combined.update(
                _load_or_generate_mineru_pages(
                    path,
                    run_selected,
                    config,
                    _allow_api_split=_allow_api_split,
                )
            )
        return combined
    if not config.mineru_auto_generate:
        raise DocumentSourceError(
            "MinerU OCR is required but no compatible cached artifact exists "
            "and auto-generation is disabled"
        )

    if config.mineru_api_url is not None:
        page_count = end_page - start_page + 1
        if _allow_api_split and page_count > config.mineru_api_page_batch_size:
            combined: dict[int, list[str]] = {}
            for batch_start in range(
                start_page, end_page + 1, config.mineru_api_page_batch_size
            ):
                batch_end = min(
                    end_page,
                    batch_start + config.mineru_api_page_batch_size - 1,
                )
                batch_selected = [
                    selected
                    and batch_start <= page_index <= batch_end
                    for page_index, selected in enumerate(selected_pages)
                ]
                if not any(batch_selected):
                    continue
                combined.update(
                    _load_or_generate_mineru_pages(
                        path,
                        batch_selected,
                        config,
                        _allow_api_split=False,
                    )
                )
            return combined
        _generate_mineru_artifact_via_api(
            path,
            job_root=job_root,
            start_page=start_page,
            end_page=end_page,
            config=config,
        )
        return _load_mineru_pages(path, generated_config)

    command = Path(config.mineru_command).expanduser()
    if not command.is_absolute():
        command = (Path.cwd() / command).resolve()
    if not command.is_file():
        raise DocumentSourceError(
            f"MinerU executable not found: {command}. Run scripts/setup_mineru_venv.sh"
        )
    job_root.mkdir(parents=True, exist_ok=True)
    try:
        process = subprocess.run(
            [
                str(command),
                "-p",
                str(path),
                "-o",
                str(job_root),
                "-b",
                config.mineru_backend,
                "--effort",
                config.mineru_effort,
                "-m",
                "ocr",
                "-s",
                str(start_page),
                "-e",
                str(end_page),
                "-t",
                "true",
                "-f",
                "false",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise DocumentSourceError(
            f"Could not launch MinerU executable {command}: {exc}"
        ) from exc
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "unknown MinerU error").strip()
        raise DocumentSourceError(
            f"MinerU failed for {path.name}, pages {start_page + 1}-{end_page + 1}: "
            f"{detail[-1600:]}"
        )
    return _load_mineru_pages(path, generated_config)


def _generate_mineru_artifact_via_api(
    path: Path,
    *,
    job_root: Path,
    start_page: int,
    end_page: int,
    config: OCRConfig,
) -> None:
    """Request a long-lived MinerU API and persist its content-list response."""
    boundary = f"----hrm-mineru-{uuid.uuid4().hex}"
    body = bytearray()

    def add_field(name: str, value: str) -> None:
        body.extend(f"--{boundary}\r\n".encode("ascii"))
        body.extend(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(
                "ascii"
            )
        )
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    for name, value in (
        ("backend", config.mineru_backend),
        ("effort", config.mineru_effort),
        ("parse_method", "ocr"),
        ("formula_enable", "false"),
        ("table_enable", "true"),
        ("image_analysis", "false"),
        ("return_md", "false"),
        ("return_content_list", "true"),
        ("return_images", "false"),
        ("response_format_zip", "false"),
        ("start_page_id", str(start_page)),
        ("end_page_id", str(end_page)),
    ):
        add_field(name, value)
    safe_filename = path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
    body.extend(f"--{boundary}\r\n".encode("ascii"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="files"; filename="{safe_filename}"\r\n'
            "Content-Type: application/pdf\r\n\r\n"
        ).encode("utf-8")
    )
    try:
        body.extend(path.read_bytes())
    except OSError as exc:
        raise DocumentSourceError(f"Could not read PDF for MinerU API: {exc}") from exc
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("ascii"))

    endpoint = config.mineru_api_url.rstrip("/") + "/file_parse"
    request = urllib.request.Request(
        endpoint,
        data=bytes(body),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=config.mineru_api_timeout_seconds
        ) as response:
            raw_response = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise DocumentSourceError(
            f"MinerU API returned HTTP {exc.code}: {detail[-1600:]}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DocumentSourceError(
            f"MinerU API request failed at {endpoint}: {exc}"
        ) from exc
    try:
        payload = json.loads(raw_response)
        results = payload["results"]
        if not isinstance(results, dict):
            raise TypeError("results is not an object")
        document_result = results.get(path.stem)
        if document_result is None and len(results) == 1:
            document_result = next(iter(results.values()))
        content_list = document_result["content_list"]
        parsed_content = (
            json.loads(content_list) if isinstance(content_list, str) else content_list
        )
        if not isinstance(parsed_content, list):
            raise TypeError("content_list is not a JSON array")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise DocumentSourceError(
            "MinerU API response did not contain a valid content_list"
        ) from exc

    artifact_dir = job_root / path.stem / "hybrid_ocr"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / f"{path.stem}_content_list.json"
    artifact_path.write_text(
        json.dumps(parsed_content, ensure_ascii=False), encoding="utf-8"
    )


@dataclass(frozen=True)
class _MinerUTableCell:
    text: str
    rowspan: int = 1
    colspan: int = 1


class _MinerUTableParser(HTMLParser):
    """Small dependency-free reader for MinerU's table_body HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_MinerUTableCell]] = []
        self._row: list[_MinerUTableCell] | None = None
        self._cell: list[str] | None = None
        self._cell_rowspan = 1
        self._cell_colspan = 1

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
            attributes = {name.lower(): value for name, value in attrs}
            self._cell_rowspan = _mineru_span(attributes.get("rowspan"))
            self._cell_colspan = _mineru_span(attributes.get("colspan"))
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None:
            assert self._row is not None
            self._row.append(
                _MinerUTableCell(
                    _clean_line("".join(self._cell)),
                    rowspan=self._cell_rowspan,
                    colspan=self._cell_colspan,
                )
            )
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if any(cell.text for cell in self._row):
                self.rows.append(self._row)
            self._row = None


def _mineru_span(raw: str | None) -> int:
    try:
        value = int(raw or "1")
    except ValueError:
        return 1
    return max(1, min(100, value))


def _expand_mineru_table_rows(
    source_rows: Sequence[Sequence[_MinerUTableCell]],
) -> list[list[str]]:
    """Expand HTML row/column spans without inventing numeric cell values."""

    active: dict[int, tuple[str, int]] = {}
    output: list[list[str]] = []
    for source_row in source_rows:
        row: list[str] = []
        column = 0

        def consume_active() -> bool:
            nonlocal column
            carried = active.get(column)
            if carried is None:
                return False
            text, remaining = carried
            row.append(text)
            if remaining <= 1:
                del active[column]
            else:
                active[column] = (text, remaining - 1)
            column += 1
            return True

        for cell in source_row:
            while consume_active():
                pass
            for offset in range(cell.colspan):
                while consume_active():
                    pass
                # A colspan occupies real grid columns, but repeating a number
                # would create artificial duplicate operands. The first column
                # retains the value and the remaining columns are placeholders.
                text = cell.text if offset == 0 else ""
                row.append(text)
                if cell.rowspan > 1:
                    active[column] = (text, cell.rowspan - 1)
                column += 1

        while active:
            if consume_active():
                continue
            next_column = min(active)
            if column >= next_column:
                break
            row.extend("" for _ in range(next_column - column))
            column = next_column
        if any(row):
            output.append(row)
    return output


def _find_mineru_content_list(path: Path, output_dir: Path) -> Path:
    root = output_dir.expanduser().resolve()
    if root.is_file():
        if root.name.endswith("_content_list.json"):
            return root
        raise DocumentSourceError(
            f"MinerU artifact must be a *_content_list.json file: {root}"
        )
    if not root.is_dir():
        raise DocumentSourceError(f"MinerU output directory not found: {root}")
    expected_name = f"{path.stem}_content_list.json"
    matches = sorted(root.rglob(expected_name))
    if not matches:
        raise DocumentSourceError(
            f"No {expected_name} found under MinerU output directory: {root}"
        )
    if len(matches) > 1:
        exact = [item for item in matches if item.parent.parent.name == path.stem]
        if len(exact) == 1:
            return exact[0]
        raise DocumentSourceError(
            f"Multiple MinerU content lists match {path.name}; pass the exact JSON file"
        )
    return matches[0]


def _mineru_table_to_lines(table_body: str) -> list[str]:
    parser = _MinerUTableParser()
    try:
        parser.feed(table_body)
        parser.close()
    except Exception as exc:
        raise DocumentSourceError(f"Could not parse MinerU table HTML: {exc}") from exc
    rows = [row for row in _expand_mineru_table_rows(parser.rows) if row]
    if not rows:
        return []

    header_count = 0
    for row in rows[:3]:
        if _mineru_row_has_financial_value(row):
            break
        header_count += 1
    header_count = max(1, header_count)
    rows = _repair_mineru_dropped_label_header(rows, header_count)
    width = max(len(row) for row in rows)
    headers = [_align_mineru_header_row(row, width) for row in rows[:header_count]]
    period_labels = _mineru_period_labels(headers, width)
    data_rows = [
        _align_mineru_data_row(row, width, period_labels)
        for row in rows[header_count:]
    ]
    data_rows = _derive_mineru_formula_cells(data_rows, period_labels)
    metadata_labels = [""] * width
    for header in headers:
        offset = max(0, width - len(header))
        for index, cell in enumerate(header):
            normalized = normalize_text(cell)
            if normalized in {"ma so", "thuyet minh", "stt"}:
                metadata_labels[offset + index] = _clean_line(cell)
    separator = ["---"] * width

    def render(row: Sequence[str]) -> str:
        return " | ".join(_clean_line(cell) for cell in row)

    def enrich(row: Sequence[str]) -> list[str]:
        cells = list(row)
        offset = max(0, width - len(cells))
        for index, cell in enumerate(cells):
            column_index = offset + index
            if (
                column_index < len(period_labels)
                and period_labels[column_index]
                and _mineru_row_has_financial_value((cell,))
            ):
                cells[index] = f"{period_labels[column_index]}: {cell}"
            elif (
                cell
                and column_index < len(metadata_labels)
                and metadata_labels[column_index]
            ):
                cells[index] = f"{metadata_labels[column_index]}: {cell}"
        return cells

    if not data_rows:
        return [render(row) for row in rows]
    output: list[str] = []
    # Repeat headers so every regular 28-line evidence chunk retains period
    # metadata even for long statement tables.
    for start in range(0, len(data_rows), 18):
        output.extend(render(row) for row in headers)
        output.append(render(separator))
        output.extend(render(enrich(row)) for row in data_rows[start : start + 18])
    return output


def _align_mineru_header_row(row: Sequence[str], width: int) -> list[str]:
    cells = list(row[:width])
    # MinerU commonly emits a second header row containing only the two period
    # labels. Those labels describe the rightmost value columns.
    return [""] * max(0, width - len(cells)) + cells


def _repair_mineru_dropped_label_header(
    rows: Sequence[Sequence[str]], header_count: int
) -> list[list[str]]:
    """Restore a missing description column proven by mixed data-row shapes.

    Some table extractors emit ``code | note | current | prior`` as the header
    even though data rows are logically ``code | description | note | current |
    prior``. A complete value row then places two financial values in the last
    two cells, while a row with an explicit note code places ``note | current``
    there and loses the trailing prior-period cell. Requiring both shapes keeps
    ordinary four-column tables unchanged.
    """

    output = [list(row) for row in rows]
    if not output or header_count <= 0 or header_count >= len(output):
        return output
    width = max(len(row) for row in output)
    headers = [
        _align_mineru_header_row(row, width)
        for row in output[:header_count]
    ]
    period_labels = _mineru_period_labels(headers, width)
    period_columns = [index for index, label in enumerate(period_labels) if label]
    if len(period_columns) != 2 or period_columns != [width - 2, width - 1]:
        return output
    first_period = period_columns[0]
    metadata_headers = [
        normalize_text(cell) for row in headers for cell in row[:first_period]
    ]
    if not any(label in {"ma so", "maso", "stt"} for label in metadata_headers):
        return output
    note_column = next(
        (
            index
            for index in range(first_period - 1, -1, -1)
            if any(
                normalize_text(row[index])
                in {"thuyet minh", "thuyetminh", "ghi chu"}
                for row in headers
                if index < len(row)
            )
        ),
        None,
    )
    if note_column is None:
        return output

    data_rows = [row for row in output[header_count:] if len(row) == width]
    complete_rows = [
        row
        for row in data_rows
        if all(
            _mineru_row_has_financial_value((row[index],))
            for index in period_columns
        )
    ]
    shifted_note_rows = [
        row
        for row in data_rows
        if (
            note_column < len(row)
            and re.search(r"[^\W\d_]", row[note_column], re.UNICODE)
            and row[first_period].strip()
            and not _mineru_row_has_financial_value((row[first_period],))
            and _mineru_row_has_financial_value((row[first_period + 1],))
        )
    ]
    if not complete_rows or not shifted_note_rows:
        return output

    repaired: list[list[str]] = []
    for row_index, row in enumerate(output):
        cells = list(row)
        if row_index < header_count:
            cells.insert(note_column, "")
        elif len(cells) == width and all(
            _mineru_row_has_financial_value((cells[index],))
            for index in period_columns
        ):
            cells.insert(first_period, "")
        elif len(cells) == width:
            cells.append("")
        repaired.append(cells)
    return repaired


def _align_mineru_data_row(
    row: Sequence[str], width: int, period_labels: Sequence[str]
) -> list[str]:
    cells = list(row[:width])
    if len(cells) >= width:
        return cells
    financial = [
        index
        for index, cell in enumerate(cells)
        if _mineru_row_has_financial_value((cell,))
    ]
    period_columns = [index for index, label in enumerate(period_labels) if label]
    if financial and period_columns:
        first_value = min(financial)
        first_period = min(period_columns)
        if first_value >= first_period:
            # A truncated row such as code | label | note | current-value is
            # missing a trailing prior-period cell, not a leading metadata cell.
            return cells + [""] * (width - len(cells))
        if len(financial) <= len(period_columns):
            # When MinerU drops empty metadata cells, restore them immediately
            # before the ordered period values instead of shifting the periods.
            prefix = cells[:first_value]
            suffix = cells[first_value:]
            aligned = prefix + [""] * max(0, first_period - len(prefix)) + suffix
            return (aligned + [""] * width)[:width]
    if period_columns and re.search(
        r"(?<!\d)\d{1,3}\s*=\s*\d{1,3}\s*[+-]\s*\d{1,3}(?!\d)",
        _mineru_formula_text(cells),
    ):
        # A formula row may have lost every value cell. Its code/label metadata
        # remains left-anchored and the missing periods belong on the right.
        return cells + [""] * (width - len(cells))
    # Text-only continuation rows retain the historical right alignment.
    return [""] * (width - len(cells)) + cells


def _mineru_row_code(row: Sequence[str], first_period: int) -> str:
    for cell in row[:first_period]:
        normalized = normalize_text(cell).strip()
        if re.fullmatch(r"\d{1,3}", normalized):
            return str(int(normalized))
    return ""


def _mineru_formula_reference(
    row: Sequence[str], first_period: int
) -> tuple[str, str, str, str] | None:
    text = _mineru_formula_text(row[:first_period])
    match = re.search(
        r"(?<!\d)(\d{1,3})\s*=\s*(\d{1,3})\s*([+-])\s*(\d{1,3})(?!\d)",
        text,
    )
    if match is None:
        return None
    target, left, operator, right = match.groups()
    return str(int(target)), str(int(left)), operator, str(int(right))


def _mineru_formula_text(cells: Sequence[str]) -> str:
    """Keep arithmetic operators while normalizing common OCR glyph variants."""

    return (
        " ".join(cells)
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("＋", "+")
        .replace("＝", "=")
    )


def _mineru_cell_decimal(cell: str) -> Decimal | None:
    if "DERIVED_FROM_TABLE_FORMULA" in cell:
        # Do not chain inferred cells into additional inferred cells.
        return None
    tokens = re.findall(r"(?<!\d)\(?-?\d[\d.,\s]*\d\)?(?!\d)", cell)
    values: list[Decimal] = []
    for token in tokens:
        digits = re.sub(r"\D", "", token)
        if len(digits) < 4 or re.fullmatch(r"(?:19|20)\d{2}", digits):
            continue
        compact = re.sub(r"\s+", "", token.strip())
        negative = compact.startswith("-") or (
            compact.startswith("(") and compact.endswith(")")
        )
        compact = compact.strip("()+-")
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", compact):
            compact = compact.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?", compact):
            compact = compact.replace(",", "")
        elif "," in compact and "." not in compact:
            compact = compact.replace(",", ".")
        try:
            value = Decimal(compact)
        except InvalidOperation:
            continue
        values.append(-value if negative else value)
    return values[0] if len(values) == 1 else None


def _format_mineru_decimal(value: Decimal) -> str:
    sign = "-" if value < 0 else ""
    absolute = abs(value)
    if absolute == absolute.to_integral_value():
        return sign + f"{int(absolute):,}".replace(",", ".")
    rendered = format(absolute, "f").rstrip("0").rstrip(".")
    integer, _, fraction = rendered.partition(".")
    grouped = f"{int(integer):,}".replace(",", ".")
    return sign + grouped + ("," + fraction if fraction else "")


def _derive_mineru_formula_cells(
    rows: Sequence[Sequence[str]], period_labels: Sequence[str]
) -> list[list[str]]:
    """Fill only missing cells backed by an explicit same-table row formula.

    The marker is carried into candidate evidence. Runtime treats selected
    derived operands as review-required, so this recovery cannot silently turn
    an uncertain OCR reconstruction into an accepted answer.
    """

    period_columns = [index for index, label in enumerate(period_labels) if label]
    if not rows or not period_columns:
        return [list(row) for row in rows]
    first_period = min(period_columns)
    by_code: dict[str, Sequence[str]] = {}
    for row in rows:
        if code := _mineru_row_code(row, first_period):
            by_code.setdefault(code, row)

    output = [list(row) for row in rows]
    for row in output:
        formula = _mineru_formula_reference(row, first_period)
        if formula is None:
            continue
        target, left_code, operator, right_code = formula
        if _mineru_row_code(row, first_period) != target:
            continue
        left_row = by_code.get(left_code)
        right_row = by_code.get(right_code)
        if left_row is None or right_row is None:
            continue
        for column in period_columns:
            if _mineru_cell_decimal(row[column]) is not None:
                continue
            left = _mineru_cell_decimal(left_row[column])
            right = _mineru_cell_decimal(right_row[column])
            if left is None or right is None:
                continue
            value = left + right if operator == "+" else left - right
            row[column] = (
                f"{_format_mineru_decimal(value)} [DERIVED_FROM_TABLE_FORMULA]"
            )
    return output


def _mineru_period_labels(
    headers: Sequence[Sequence[str]], width: int
) -> list[str]:
    labels = [""] * width
    for row in headers:
        offset = max(0, width - len(row))
        for index, cell in enumerate(row):
            normalized = normalize_text(cell)
            if re.search(r"\b(?:19|20)\d{2}\b", normalized) or any(
                marker in normalized
                for marker in (
                    "nam nay",
                    "nam truoc",
                    "cuoi nam",
                    "cuoi ky",
                    "dau nam",
                    "dau ky",
                    "so cuoi nam",
                    "so dau nam",
                )
            ):
                labels[offset + index] = _clean_line(cell)
    return labels


def _mineru_row_has_financial_value(row: Sequence[str]) -> bool:
    for cell in row:
        for token in re.findall(r"(?<!\d)\(?-?\d[\d.,\s]*\d\)?(?!\d)", cell):
            digits = re.sub(r"\D", "", token)
            if len(digits) >= 4 and not re.fullmatch(r"(?:19|20)\d{2}", digits):
                return True
    return False


def _mineru_table_bbox(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = item.get("bbox")
    if not isinstance(raw, list) or len(raw) != 4:
        return None
    try:
        left, top, right, bottom = (float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _mineru_table_continuations(
    payload: Sequence[Any], item: dict[str, Any]
) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Find empty table placeholders created by a MinerU multi-page merge.

    Hybrid MinerU can attach the complete HTML body to the first physical page
    and emit bbox-only table records on following pages.  Without recognizing
    those placeholders every row is cited as the first page of the table.
    """

    raw_page = item.get("page_idx")
    anchor = _mineru_table_bbox(item)
    body = item.get("table_body")
    if (
        not isinstance(raw_page, int)
        or raw_page < 0
        or anchor is None
        or not isinstance(body, str)
        or not body.strip()
        or anchor[3] < 800
    ):
        return []

    by_page: dict[int, list[dict[str, Any]]] = {}
    for candidate in payload:
        if not isinstance(candidate, dict) or candidate.get("type") != "table":
            continue
        page = candidate.get("page_idx")
        if isinstance(page, int):
            by_page.setdefault(page, []).append(candidate)

    output: list[tuple[int, tuple[float, float, float, float]]] = []
    previous = anchor
    page = raw_page + 1
    while True:
        match: tuple[float, float, float, float] | None = None
        for candidate in by_page.get(page, ()):
            candidate_body = candidate.get("table_body")
            bbox = _mineru_table_bbox(candidate)
            if (
                (isinstance(candidate_body, str) and candidate_body.strip())
                or bbox is None
                or bbox[1] > 200
            ):
                continue
            overlap = max(
                0.0, min(previous[2], bbox[2]) - max(previous[0], bbox[0])
            )
            narrower = min(previous[2] - previous[0], bbox[2] - bbox[0])
            if narrower > 0 and overlap / narrower >= 0.75:
                match = bbox
                break
        if match is None:
            break
        output.append((page, match))
        previous = match
        page += 1
    return output


def _partition_mineru_table_lines(
    lines: Sequence[str],
    anchor_page: int,
    anchor_bbox: tuple[float, float, float, float],
    continuations: Sequence[tuple[int, tuple[float, float, float, float]]],
) -> dict[int, list[str]]:
    pages = [(anchor_page, anchor_bbox), *continuations]
    output: dict[int, list[str]] = {}
    start = 0
    for position, (page, bbox) in enumerate(pages):
        # MinerU bboxes use an approximately 0-1000 page coordinate space.
        # Allocate rows sequentially at a conservative 20 units per rendered
        # line. Proportional normalization across the whole span shifts middle
        # rows one page too far when the final continuation is only partially
        # occupied.
        capacity = max(1, round((bbox[3] - bbox[1]) / 20.0))
        end = (
            len(lines)
            if position + 1 == len(pages)
            else start + capacity
        )
        end = max(start, min(len(lines), end))
        output[page] = list(lines[start:end])
        start = end
    return output



def _mineru_unit_header_preludes(payload: list[dict]) -> tuple[dict[int, list[str]], set[int]]:
    """Restore a monetary unit header physically above its table on the same page.

    MinerU sometimes emits headers after body blocks in content_list order.
    Only an explicit currency declaration with a verified bounding box above
    a nearby table is moved; ordinary running headers keep their original order.
    """
    table_tops: dict[int, list[float]] = {}
    for item in payload:
        if not isinstance(item, dict) or item.get("type") != "table":
            continue
        page = item.get("page_idx")
        bbox = _mineru_table_bbox(item)
        if isinstance(page, int) and bbox is not None:
            table_tops.setdefault(page, []).append(float(bbox[1]))
    found: dict[int, list[tuple[float, int, str]]] = {}
    moved: set[int] = set()
    unit_pattern = re.compile(
        r"^(?:dvt|don vi(?: tinh)?|units?) "
        r"(?:(?:nghin|trieu|ty|ti|thousand|million|billion) )?"
        r"(?:dong|vnd|usd|eur|jpy|gbp|cny)$"
    )
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or item.get("type") != "header":
            continue
        page, bbox, raw = item.get("page_idx"), item.get("bbox"), item.get("text")
        if not isinstance(page, int) or not isinstance(raw, str):
            continue
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        if not all(isinstance(value, (int, float)) for value in bbox):
            continue
        cleaned = _clean_line(raw)
        if not unit_pattern.fullmatch(normalize_text(cleaned)):
            continue
        if not any(0 <= top - float(bbox[3]) <= 200 for top in table_tops.get(page, ())):
            continue
        found.setdefault(page, []).append((float(bbox[1]), index, cleaned))
        moved.add(index)
    return {
        page: [text for _, _, text in sorted(items)]
        for page, items in found.items()
    }, moved

def _load_mineru_pages(path: Path, config: OCRConfig) -> dict[int, list[str]]:
    assert config.mineru_output_dir is not None
    content_path = _find_mineru_content_list(path, config.mineru_output_dir)
    try:
        payload = json.loads(content_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DocumentSourceError(
            f"Could not read MinerU content list {content_path}: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise DocumentSourceError(
            f"MinerU content list must contain a JSON array: {content_path}"
        )

    unit_preludes, moved_unit_headers = _mineru_unit_header_preludes(payload)
    pages: dict[int, list[str]] = {
        page + 1 + config.mineru_page_offset: list(lines)
        for page, lines in unit_preludes.items()
    }
    for index, item in enumerate(payload):
        if index in moved_unit_headers or not isinstance(item, dict):
            continue
        raw_page = item.get("page_idx")
        if not isinstance(raw_page, int) or raw_page < 0:
            continue
        page_number = raw_page + 1 + config.mineru_page_offset
        lines = pages.setdefault(page_number, [])
        item_type = str(item.get("type", ""))
        if item_type == "table":
            raw_captions = item.get("table_caption", ())
            captions = (
                (raw_captions,)
                if isinstance(raw_captions, str)
                else raw_captions if isinstance(raw_captions, list) else ()
            )
            for caption in captions:
                cleaned = _clean_line(str(caption))
                if cleaned:
                    lines.append(cleaned)
            footnote_page = page_number
            body = item.get("table_body")
            if isinstance(body, str) and body.strip():
                table_lines = _mineru_table_to_lines(body)
                bbox = _mineru_table_bbox(item)
                continuations = _mineru_table_continuations(payload, item)
                if bbox is not None and continuations:
                    partitioned = _partition_mineru_table_lines(
                        table_lines, raw_page, bbox, continuations
                    )
                    # ``pages`` is one-based and may include a source-page offset.
                    footnote_page = max(partitioned) + 1 + config.mineru_page_offset
                    lines.extend(partitioned.pop(raw_page, ()))
                    for continuation_page, continuation_lines in partitioned.items():
                        pages.setdefault(
                            continuation_page + 1 + config.mineru_page_offset, []
                        ).extend(continuation_lines)
                else:
                    lines.extend(table_lines)
            notes = item.get("table_footnote", ())
            notes = (notes,) if isinstance(notes, str) else notes if isinstance(notes, list) else ()
            for note in notes:
                if isinstance(note, str) and (cleaned := _clean_line(note)):
                    # Blockquote retains table-footnote ownership for downstream binding.
                    pages.setdefault(footnote_page, []).append("> " + cleaned)
            continue
        text = item.get("text")
        if isinstance(text, str):
            lines.extend(
                cleaned
                for raw in text.splitlines()
                if (cleaned := _clean_line(raw))
            )
    return pages


def _repair_wrapped_numeric_lines(lines: list[str]) -> list[str]:
    """Reattach OCR digits wrapped below an incomplete grouped-number cell.

    Financial tables sometimes render the last digit on the next visual line.
    The repair is intentionally narrow: the previous row must be table-like,
    exactly one cell must end in an incomplete number with at least two group
    separators, and the continuation line may contain only one or two digits.
    """

    output: list[str] = []
    incomplete = re.compile(r"^([(+\-]?)(?:\d{1,3}[.,]){2,}\d{1,2}(\)?)$")
    for line in lines:
        continuation = line.strip()
        if output and re.fullmatch(r"\d{1,2}", continuation):
            cells = [cell.strip() for cell in output[-1].split("|")]
            if len(cells) >= 3:
                matches = [index for index, cell in enumerate(cells) if incomplete.fullmatch(cell)]
                if len(matches) == 1:
                    cell_index = matches[0]
                    closing = ")" if cells[cell_index].endswith(")") else ""
                    base = cells[cell_index][:-1] if closing else cells[cell_index]
                    cells[cell_index] = base + continuation + closing
                    output[-1] = " | ".join(cells)
                    continue
        output.append(line)
    return output


def _read_text(path: Path) -> list[SourceLine]:
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="utf-8", errors="replace")
    output: list[SourceLine] = []
    page: int | None = None
    # This historical tag describes the Markdown input format, not verified
    # extractor lineage. Independent evaluations verify MinerU sidecars before
    # loading the checkpoint (see mineru_provenance_gate.py).
    method = "mineru_markdown" if path.suffix.lower() == ".md" else "text"
    page_pattern = re.compile(
        r"^\s*(?:#{1,6}\s*)?(?:[-=]{2,}\s*)?(?:page|trang)\s+(\d+)",
        re.I,
    )
    for line_number, raw in enumerate(content.splitlines(), start=1):
        match = page_pattern.match(raw)
        if match:
            page = int(match.group(1))
        cleaned = _clean_line(raw)
        if not cleaned:
            continue
        output.append(
            SourceLine(
                source_path=str(path),
                source_name=path.name,
                page=page,
                line_number=line_number,
                text=cleaned,
                extraction_method=method,
            )
        )
    return output


def _clean_line(text: str) -> str:
    text = decode_pdf_unicode_glyph_names(text)
    text = text.replace("\x00", " ").replace("\u00a0", " ")
    return re.sub(r"[ \t]+", " ", text).strip()


def _clean_layout_line(text: str) -> str:
    """Clean native layout text without destroying horizontal provenance."""

    text = decode_pdf_unicode_glyph_names(text)
    text = text.replace("\x00", " ").replace("\u00a0", " ").expandtabs(4).rstrip()
    return text if text.strip() else ""


def chunk_lines(
    lines: Sequence[SourceLine],
    *,
    chunk_lines_count: int = 28,
    overlap_lines: int = 6,
) -> list[EvidenceChunk]:
    if chunk_lines_count < 4:
        raise DocumentSourceError("chunk_lines_count must be at least 4")
    if overlap_lines < 0 or overlap_lines >= chunk_lines_count:
        raise DocumentSourceError("overlap_lines must be between 0 and chunk_lines_count-1")
    if not lines:
        return []
    from .source_bindings import source_annotations
    annotations = source_annotations(lines)
    annotation_by_identity = {id(line): annotations[i] for i, line in enumerate(lines)}
    chunks: list[EvidenceChunk] = []
    step = chunk_lines_count - overlap_lines
    groups: list[list[SourceLine]] = []
    for line in lines:
        key = (line.source_path, line.page)
        if not groups or (groups[-1][0].source_path, groups[-1][0].page) != key:
            groups.append([])
        groups[-1].append(line)
    for group in groups:
        for start in range(0, len(group), step):
            selected = group[start : start + chunk_lines_count]
            if not selected:
                continue
            first, last = selected[0], selected[-1]
            # SourceLine numbers refer to original source lines. Retain gaps so
            # extraction offsets and overlapping chunks name the same physical row.
            source_text = {item.line_number: item.text for item in selected}
            text = "\n".join(source_text.get(number, "") for number in range(
                first.line_number, last.line_number + 1
            ))
            chunk_id = (
                f"{Path(first.source_name).stem}:"
                f"{first.line_number}-{last.line_number}"
            )
            chunks.append(
                EvidenceChunk(
                    chunk_id=chunk_id,
                    source_path=first.source_path,
                    source_name=first.source_name,
                    page_start=first.page,
                    page_end=last.page,
                    line_start=first.line_number,
                    line_end=last.line_number,
                    text=text,
                    source_annotations=tuple(annotation_by_identity[id(item)] for item in selected),
                    extraction_methods=tuple(
                        dict.fromkeys(item.extraction_method for item in selected)
                    ),
                    structured_cells=tuple(
                        {
                            (
                                cell.page,
                                cell.table_id,
                                cell.row_id,
                                cell.column_id,
                            ): cell
                            for item in selected
                            for cell in item.structured_cells
                        }.values()
                    ),
                )
            )
            if start + chunk_lines_count >= len(group):
                break
    return chunks


def build_chunks(
    documents: Iterable[str | Path],
    *,
    chunk_lines_count: int = 28,
    overlap_lines: int = 6,
    ocr_config: OCRConfig | None = None,
) -> list[EvidenceChunk]:
    output: list[EvidenceChunk] = []
    for path in documents:
        output.extend(
            chunk_lines(
                read_document(path, ocr_config=ocr_config),
                chunk_lines_count=chunk_lines_count,
                overlap_lines=overlap_lines,
            )
        )
    return output


def rank_chunks(
    question: str,
    chunks: Sequence[EvidenceChunk],
    *,
    top_k: int = 8,
) -> list[EvidenceChunk]:
    """Small dependency-free BM25 ranker over real PDF/TXT chunks."""
    if not chunks or top_k <= 0:
        return []
    query = normalize_text(question)
    query_terms = [term for term in query.split() if len(term) > 1]
    if not query_terms:
        return list(chunks[:top_k])
    documents = [normalize_text(chunk.text).split() for chunk in chunks]
    doc_freq: Counter[str] = Counter()
    for terms in documents:
        doc_freq.update(set(terms))
    avg_len = sum(len(terms) for terms in documents) / max(1, len(documents))
    n_docs = len(documents)
    query_counts = Counter(query_terms)
    phrase_candidates = _query_phrases(query_terms)
    scored: list[EvidenceChunk] = []
    for chunk, terms in zip(chunks, documents):
        term_counts = Counter(terms)
        length = max(1, len(terms))
        score = 0.0
        for term, qtf in query_counts.items():
            tf = term_counts.get(term, 0)
            if not tf:
                continue
            idf = math.log(1.0 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denominator = tf + 1.2 * (1.0 - 0.75 + 0.75 * length / max(1.0, avg_len))
            score += idf * (tf * 2.2 / denominator) * (1.0 + 0.1 * min(qtf, 3))
        normalized_text = " ".join(terms)
        for phrase in phrase_candidates:
            if phrase in normalized_text:
                score += 1.5 + 0.35 * len(phrase.split())
        # Numbers and ticker/year tokens are unusually useful in financial reports.
        for token in query_terms:
            if (token.isdigit() or (token.isupper() and 2 <= len(token) <= 5)) and token in normalized_text:
                score += 1.0
        if score > 0:
            scored.append(EvidenceChunk(**{**chunk.__dict__, "score": score}))
    # An explicit source section is an intent constraint, not another BM25
    # keyword. Preserve matching chunks before top-k drops them. The verifier
    # still checks the exact cell; a chunk can straddle a section boundary.
    from .source_bindings import requested_section
    section = requested_section(question)
    if section and scored:
        priority = max(item.score for item in scored) + 1.0
        scored = [replace(item, score=item.score + priority)
                  if any(section in annotation.get("source_section_path", ())
                         for annotation in item.source_annotations)
                  else item for item in scored]
    scored.sort(key=lambda item: (item.score, -item.line_start), reverse=True)
    return scored[:top_k]



def expand_neighbor_chunks(
    ranked: Sequence[EvidenceChunk],
    all_chunks: Sequence[EvidenceChunk],
    *,
    neighbor_window: int = 1,
    max_chunks: int | None = None,
) -> list[EvidenceChunk]:
    """Add adjacent chunks around the strongest retrieval hits.

    Financial table headers and operands often sit just outside a fixed chunk.
    Neighbor expansion improves evidence completeness without changing the
    sparse ranker or introducing an ontology shortcut.
    """
    if neighbor_window <= 0 or not ranked:
        return list(ranked[:max_chunks] if max_chunks else ranked)
    by_source: dict[str, list[EvidenceChunk]] = {}
    for chunk in all_chunks:
        by_source.setdefault(chunk.source_path, []).append(chunk)
    for chunks in by_source.values():
        chunks.sort(key=lambda item: item.line_start)

    output: dict[str, EvidenceChunk] = {chunk.chunk_id: chunk for chunk in ranked}
    # Every ranked hit may be the visible half of a split financial table.
    # Apply the same neighbor policy to all hits, then enforce ``max_chunks``
    # after score decay instead of privileging an arbitrary first four.
    for hit in ranked:
        source_chunks = by_source.get(hit.source_path, [])
        positions = [
            index for index, chunk in enumerate(source_chunks)
            if chunk.chunk_id == hit.chunk_id
        ]
        if not positions:
            continue
        position = positions[0]
        for distance in range(1, neighbor_window + 1):
            for neighbor_position in (position - distance, position + distance):
                if not 0 <= neighbor_position < len(source_chunks):
                    continue
                neighbor = source_chunks[neighbor_position]
                if neighbor.chunk_id in output:
                    continue
                output[neighbor.chunk_id] = EvidenceChunk(
                    **{
                        **neighbor.__dict__,
                        "score": max(0.0, hit.score * (0.65 ** distance)),
                    }
                )

    expanded = list(output.values())
    expanded.sort(key=lambda item: (item.score, -item.line_start), reverse=True)
    return expanded[:max_chunks] if max_chunks else expanded

def _query_phrases(tokens: Sequence[str]) -> tuple[str, ...]:
    phrases: list[str] = []
    for size in (4, 3, 2):
        for index in range(len(tokens) - size + 1):
            phrase_tokens = tokens[index : index + size]
            if any(len(token) > 2 for token in phrase_tokens):
                phrases.append(" ".join(phrase_tokens))
    return tuple(dict.fromkeys(phrases))

