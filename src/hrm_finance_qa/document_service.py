from __future__ import annotations

import hashlib
import inspect
import json
import mimetypes
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

from .document_hrm import DocumentHRMBundle, DocumentHRMError
from .document_page_index import DocumentPageIndex, PageIndexStats
from .document_reasoning_engine import DocumentHRMReasoningEngine, DocumentReasoningAnswer
from .document_sources import DocumentSourceError, OCRConfig, discover_documents


_DOCUMENT_HRM_SOURCE_PATH = Path(
    inspect.getsourcefile(DocumentHRMBundle) or ""
).resolve()
_DOCUMENT_HRM_SOURCE_SHA256 = (
    hashlib.sha256(_DOCUMENT_HRM_SOURCE_PATH.read_bytes()).hexdigest()
    if _DOCUMENT_HRM_SOURCE_PATH.is_file()
    else "unavailable"
)
_RUNTIME_SOURCE_NAMES = (
    "document_hrm.py",
    "document_page_index.py",
    "document_reasoning_engine.py",
    "document_service.py",
    "document_sources.py",
    "ontology.py",
    "program_data.py",
)
_RUNTIME_SOURCE_PATHS = tuple(
    _DOCUMENT_HRM_SOURCE_PATH.parent / name for name in _RUNTIME_SOURCE_NAMES
)


def _source_set_sha256(paths: Sequence[Path]) -> str:
    """Hash the complete runtime source set, including stable file names."""

    if not paths or any(not path.is_file() for path in paths):
        return "unavailable"
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


_RUNTIME_SOURCE_SHA256 = _source_set_sha256(_RUNTIME_SOURCE_PATHS)


class DocumentServiceError(RuntimeError):
    pass


class DocumentNotReadyError(DocumentServiceError):
    def __init__(self, message: str, *, job: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.job = job


@dataclass
class IngestionJob:
    job_id: str
    document: str
    rebuild: bool
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    error: str | None = None
    documents_total: int = 0
    documents_completed: int = 0
    index_results: list[dict[str, Any]] = field(default_factory=list)

    def render(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "document": self.document,
            "rebuild": self.rebuild,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "documents_total": self.documents_total,
            "documents_completed": self.documents_completed,
            "index_results": list(self.index_results),
        }


class DocumentIngestionManager:
    """Single-worker ingestion queue that keeps OCR outside answer requests."""

    def __init__(
        self,
        *,
        index_path: str | Path,
        ocr_config: OCRConfig,
        allowed_roots: Sequence[str | Path],
        start_worker: bool = True,
    ) -> None:
        self.index_path = Path(index_path).expanduser().resolve()
        self.ocr_config = ocr_config
        self.ocr_config.validate()
        self.allowed_roots = tuple(
            Path(item).expanduser().resolve() for item in allowed_roots
        )
        if not self.allowed_roots:
            raise DocumentServiceError("At least one document root is required")
        self._jobs: dict[str, IngestionJob] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._worker: threading.Thread | None = None
        if start_worker:
            self.start()

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="document-ingestion-worker",
            daemon=True,
        )
        self._worker.start()

    def shutdown(self, *, timeout: float = 5.0) -> None:
        if self._worker is None:
            return
        self._queue.put(None)
        self._worker.join(timeout=timeout)

    def resolve_document(self, document: str | Path) -> Path:
        candidate = Path(document).expanduser()
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        resolved = candidate.resolve()
        if not any(
            resolved == root or root in resolved.parents for root in self.allowed_roots
        ):
            roots = ", ".join(str(item) for item in self.allowed_roots)
            raise DocumentServiceError(
                f"Document is outside configured roots ({roots}): {resolved}"
            )
        return resolved

    def submit(
        self, document: str | Path, *, rebuild: bool = False
    ) -> tuple[dict[str, Any], bool]:
        resolved = self.resolve_document(document)
        with self._lock:
            for job in self._jobs.values():
                if (
                    job.document == str(resolved)
                    and job.rebuild == rebuild
                    and job.status in {"queued", "processing"}
                ):
                    return job.render(), False
            job = IngestionJob(
                job_id=uuid.uuid4().hex,
                document=str(resolved),
                rebuild=rebuild,
            )
            self._jobs[job.job_id] = job
            self._queue.put(job.job_id)
            return job.render(), True

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.render() if job is not None else None

    def active_for(self, document: str | Path) -> dict[str, Any] | None:
        resolved = self.resolve_document(document)
        with self._lock:
            active = [
                job
                for job in self._jobs.values()
                if (
                    job.document == str(resolved)
                    or Path(job.document) in resolved.parents
                )
                and job.status in {"queued", "processing"}
            ]
            if not active:
                return None
            active.sort(key=lambda item: item.created_at, reverse=True)
            return active[0].render()

    def summary(self) -> dict[str, int]:
        counts = {"queued": 0, "processing": 0, "ready": 0, "failed": 0}
        with self._lock:
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
        return counts

    def mineru_health(self) -> dict[str, Any]:
        if not self.ocr_config.mineru_api_url:
            return {"status": "disabled", "mode": "cli"}
        endpoint = self.ocr_config.mineru_api_url.rstrip("/") + "/health"
        try:
            with urllib.request.urlopen(endpoint, timeout=2.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return {"status": "healthy", "endpoint": endpoint, "details": payload}
        except Exception as exc:
            return {"status": "unavailable", "endpoint": endpoint, "error": str(exc)}

    def _worker_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                self._run_job(job_id)
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = "processing"
            job.started_at = time.time()
            job.error = None
        try:
            documents = discover_documents(job.document)
            pdf_documents = [item for item in documents if item.suffix.lower() == ".pdf"]
            with self._lock:
                job.documents_total = len(pdf_documents)
            index = DocumentPageIndex(self.index_path)
            for document in pdf_documents:
                _records, stats = index.load_or_build(
                    document,
                    ocr_config=self.ocr_config,
                    rebuild=job.rebuild,
                )
                with self._lock:
                    job.documents_completed += 1
                    job.index_results.append(
                        {
                            "document": str(document),
                            "stats": asdict(stats),
                        }
                    )
            with self._lock:
                job.status = "ready"
                job.completed_at = time.time()
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.error = str(exc)
                job.completed_at = time.time()


class PersistentDocumentHRMService:
    def __init__(
        self,
        bundle: DocumentHRMBundle,
        ingestion: DocumentIngestionManager,
        *,
        model_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.engine = DocumentHRMReasoningEngine(bundle)
        self.ingestion = ingestion
        self._inference_lock = threading.Lock()
        self._runtime_started_at = time.time()
        self._device = device
        self._model_path = (
            Path(model_path).expanduser().resolve() if model_path is not None else None
        )
        bundle_metadata = getattr(bundle, "metadata", {}) or {}
        self._model_metadata = {
            key: bundle_metadata.get(key)
            for key in (
                "architecture",
                "checkpoint_schema_version",
                "runtime_schema_version",
                "confidence_schema",
                "program_reranker_enabled",
                "program_reranker_architecture",
                "program_reranker_feature_schema",
                "program_reranker_confidence_schema",
                "program_reranker_calibration_mode",
                "typed_program_synthesis_enabled",
            )
            if bundle_metadata.get(key) is not None
        }

    def health(self) -> dict[str, Any]:
        mineru = self.ingestion.mineru_health()
        model: dict[str, Any] = {
            "loaded": True,
            "device": self._device,
            **self._model_metadata,
        }
        if self._model_path is not None:
            model.update(
                {
                    "checkpoint": self._model_path.name,
                    "path": str(self._model_path),
                    "size_bytes": (
                        self._model_path.stat().st_size
                        if self._model_path.is_file()
                        else None
                    ),
                }
            )
        return {
            "status": (
                "healthy" if mineru.get("status") == "healthy" else "degraded"
            ),
            "model_loaded": True,
            "model": model,
            "runtime_started_at": self._runtime_started_at,
            "document_hrm_source": str(_DOCUMENT_HRM_SOURCE_PATH),
            "document_hrm_source_sha256": _DOCUMENT_HRM_SOURCE_SHA256,
            "runtime_source_sha256": _RUNTIME_SOURCE_SHA256,
            "runtime_source_files": [str(path) for path in _RUNTIME_SOURCE_PATHS],
            "inference_busy": self._inference_lock.locked(),
            "ingestion_jobs": self.ingestion.summary(),
            "mineru": mineru,
        }

    def answer(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", "")).strip()
        if not question:
            raise DocumentServiceError("question is required")
        raw_document = str(payload.get("document", "")).strip()
        if not raw_document:
            raise DocumentServiceError("document is required")
        document = self.ingestion.resolve_document(raw_document)
        active = self.ingestion.active_for(document)
        if active is not None:
            raise DocumentNotReadyError(
                "Document ingestion is still in progress", job=active
            )
        started_at = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]

        def report_progress(stage: str, elapsed: float) -> None:
            print(
                f"document-service inference={request_id} stage={stage} "
                f"elapsed={elapsed:.3f}s",
                flush=True,
            )

        try:
            with self._inference_lock:
                answer = self.engine.answer(
                    question,
                    document,
                    top_k_chunks=int(payload.get("top_k_chunks", 8)),
                    neighbor_chunks=int(payload.get("neighbor_chunks", 1)),
                    ocr=True,
                    force_ocr=False,
                    mineru_cache_dir=self.ingestion.ocr_config.mineru_cache_dir,
                    mineru_command=self.ingestion.ocr_config.mineru_command,
                    mineru_api_url=self.ingestion.ocr_config.mineru_api_url,
                    mineru_api_timeout_seconds=(
                        self.ingestion.ocr_config.mineru_api_timeout_seconds
                    ),
                    mineru_backend=self.ingestion.ocr_config.mineru_backend,
                    mineru_effort=self.ingestion.ocr_config.mineru_effort,
                    mineru_auto_generate=False,
                    use_page_index=True,
                    page_index_path=self.ingestion.index_path,
                    rebuild_page_index=False,
                    page_top_k=int(payload.get("page_top_k", 6)),
                    page_neighbor_pages=int(payload.get("page_neighbor_pages", 1)),
                    progress_callback=report_progress,
                )
        except (DocumentSourceError, DocumentHRMError) as exc:
            raise DocumentNotReadyError(
                "No ready native/indexed evidence. Submit /v1/ingest for scanned PDFs. "
                f"Details: {exc}"
            ) from exc
        return serialize_answer(answer, time.perf_counter() - started_at)


def serialize_answer(
    answer: DocumentReasoningAnswer, service_answer_seconds: float
) -> dict[str, Any]:
    candidate_answer = None if answer.value is None else str(answer.value)
    return {
        "answer": None if answer.abstained else candidate_answer,
        "candidate_answer": candidate_answer if answer.abstained else None,
        "unit": answer.unit,
        "status": "ABSTAIN" if answer.abstained else "ACCEPTED",
        "program": {
            "operation": answer.program.operation,
            "operands": list(answer.program.operands),
        },
        "confidence": answer.prediction.confidence,
        "program_candidates_considered": (
            answer.prediction.program_candidates_considered
        ),
        "symbolic_program_rejections": (
            answer.prediction.symbolic_program_rejections
        ),
        "symbolic_proof_score": answer.prediction.symbolic_proof_score,
        "symbolic_unknown_constraints": (
            answer.prediction.symbolic_unknown_constraints
        ),
        "service_answer_seconds": service_answer_seconds,
        "document_count": answer.document_count,
        "candidate_count": answer.candidate_count,
        "stage_timings": dict(answer.stage_timings),
        "page_index": (
            asdict(answer.page_index_stats)
            if isinstance(answer.page_index_stats, PageIndexStats)
            else None
        ),
        "evidence": [
            {
                "operand_number": item.operand_number,
                "value": item.candidate.value,
                "unit": item.candidate.unit,
                "source_name": item.candidate.source_name,
                "page": item.candidate.page,
                "context": item.candidate.context,
                "extraction_method": item.candidate.extraction_method,
            }
            for item in answer.evidence
        ],
        "warnings": list(answer.warnings),
        "rendered": answer.render(),
    }


class _ServiceHTTPServer(ThreadingHTTPServer):
    """Keep diagnostics responsive while serialized model inference runs."""

    daemon_threads = True


def make_handler(
    service: PersistentDocumentHRMService,
    *,
    web_root: str | Path | None = None,
    upload_root: str | Path | None = None,
    max_upload_bytes: int = 512 * 1024 * 1024,
) -> type[BaseHTTPRequestHandler]:
    resolved_web_root = (
        Path(web_root).expanduser().resolve() if web_root is not None else None
    )
    resolved_upload_root = (
        Path(upload_root).expanduser().resolve() if upload_root is not None else None
    )
    static_routes = {
        "/": "index.html",
        "/index.html": "index.html",
        "/styles.css": "styles.css",
        "/app.js": "app.js",
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "DocumentHRM/2"

        def do_GET(self) -> None:  # noqa: N802
            route = urllib.parse.urlsplit(self.path).path
            if route == "/health":
                self._send_json(HTTPStatus.OK, service.health())
                return
            if route.startswith("/v1/jobs/"):
                job_id = route.removeprefix("/v1/jobs/")
                job = service.ingestion.get(job_id)
                if job is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "job not found"})
                else:
                    self._send_json(HTTPStatus.OK, job)
                return
            if resolved_web_root is not None and route in static_routes:
                self._send_static(resolved_web_root / static_routes[route])
                return
            if route == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self._send_security_headers()
                self.end_headers()
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "route not found"})

        def do_POST(self) -> None:  # noqa: N802
            try:
                parsed = urllib.parse.urlsplit(self.path)
                route = parsed.path
                if route == "/v1/upload":
                    self._handle_upload(parsed.query)
                    return
                payload = self._read_json()
                if route == "/v1/ingest":
                    document = str(payload.get("document", ""))
                    if not document:
                        raise DocumentServiceError("document is required")
                    job, created = service.ingestion.submit(
                        document, rebuild=bool(payload.get("rebuild", False))
                    )
                    self._send_json(
                        HTTPStatus.ACCEPTED if created else HTTPStatus.OK,
                        {**job, "deduplicated": not created},
                    )
                    return
                if route == "/v1/answer":
                    self._send_json(HTTPStatus.OK, service.answer(payload))
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "route not found"})
            except DocumentNotReadyError as exc:
                body: dict[str, Any] = {"error": str(exc), "status": "not_ready"}
                if exc.job is not None:
                    body["job"] = exc.job
                self._send_json(HTTPStatus.CONFLICT, body)
            except (DocumentServiceError, ValueError, TypeError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive server boundary
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"internal service error: {exc}"},
                )

        def _read_json(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise DocumentServiceError("invalid Content-Length") from exc
            if length <= 0 or length > 1024 * 1024:
                raise DocumentServiceError("JSON body must be between 1 byte and 1 MiB")
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise DocumentServiceError("request body must be valid UTF-8 JSON") from exc
            if not isinstance(payload, dict):
                raise DocumentServiceError("JSON body must be an object")
            return payload

        def _handle_upload(self, query: str) -> None:
            if resolved_upload_root is None:
                raise DocumentServiceError("document upload is disabled")
            parameters = urllib.parse.parse_qs(query)
            raw_name = str(parameters.get("filename", [""])[0]).strip()
            filename = Path(raw_name).name
            if not filename:
                raise DocumentServiceError("filename is required")
            filename = re.sub(r"[^\w .()\-]", "_", filename, flags=re.UNICODE)
            suffix = Path(filename).suffix.lower()
            if suffix not in {".pdf", ".txt"}:
                raise DocumentServiceError("only PDF and TXT documents are supported")
            raw_length = self.headers.get("Content-Length", "0")
            try:
                length = int(raw_length)
            except ValueError as exc:
                raise DocumentServiceError("invalid Content-Length") from exc
            if length <= 0:
                raise DocumentServiceError("uploaded document is empty")
            if length > max_upload_bytes:
                limit_mib = max_upload_bytes // (1024 * 1024)
                raise DocumentServiceError(
                    f"uploaded document exceeds the {limit_mib} MiB limit"
                )

            resolved_upload_root.mkdir(parents=True, exist_ok=True)
            destination = resolved_upload_root / f"{uuid.uuid4().hex[:10]}-{filename}"
            destination = service.ingestion.resolve_document(destination)
            temporary = resolved_upload_root / f".{uuid.uuid4().hex}.part"
            remaining = length
            try:
                with temporary.open("wb") as stream:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise DocumentServiceError(
                                "upload ended before Content-Length bytes arrived"
                            )
                        stream.write(chunk)
                        remaining -= len(chunk)
                os.replace(temporary, destination)
            finally:
                if temporary.exists():
                    temporary.unlink()
            self._send_json(
                HTTPStatus.CREATED,
                {
                    "document": str(destination),
                    "filename": filename,
                    "size_bytes": length,
                },
            )

        def _send_static(self, path: Path) -> None:
            if not path.is_file():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "asset not found"})
                return
            body = path.read_bytes()
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            # Demo assets change frequently during local iteration. Avoid serving
            # a stale app.js/styles.css after the service is restarted.
            self.send_header("Cache-Control", "no-store")
            self._send_security_headers()
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self._send_security_headers()
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # The inference may outlive a client's timeout. Keep the server and
                # warmed caches healthy even when that client is no longer listening.
                return

        def _send_security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; "
                "style-src 'self'; script-src 'self'; connect-src 'self'; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            )

        def log_message(self, format: str, *args: object) -> None:
            message = format % args
            print(f"document-service {self.address_string()} {message}")

    return Handler


def serve(
    service: PersistentDocumentHRMService,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    web_root: str | Path | None = None,
    upload_root: str | Path | None = None,
    max_upload_bytes: int = 512 * 1024 * 1024,
) -> None:
    server = _ServiceHTTPServer(
        (host, port),
        make_handler(
            service,
            web_root=web_root,
            upload_root=upload_root,
            max_upload_bytes=max_upload_bytes,
        ),
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        service.ingestion.shutdown()
