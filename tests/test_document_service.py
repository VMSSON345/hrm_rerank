from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from hrm_finance_qa.document_page_index import PageIndexStats
from hrm_finance_qa.document_service import (
    DocumentIngestionManager,
    DocumentNotReadyError,
    DocumentServiceError,
    PersistentDocumentHRMService,
    _ServiceHTTPServer,
    make_handler,
)
from hrm_finance_qa.document_sources import OCRConfig


class DocumentIngestionManagerTests(unittest.TestCase):
    def test_http_diagnostics_remain_responsive_during_inference(self) -> None:
        self.assertTrue(issubclass(_ServiceHTTPServer, ThreadingHTTPServer))
        self.assertTrue(_ServiceHTTPServer.daemon_threads)

    def _manager(self, root: Path) -> DocumentIngestionManager:
        return DocumentIngestionManager(
            index_path=root / "page-index.sqlite3",
            ocr_config=OCRConfig(
                enabled=True,
                mineru_cache_dir=root / "mineru",
                mineru_api_url="http://127.0.0.1:8000",
            ),
            allowed_roots=[root / "documents"],
            start_worker=False,
        )

    def test_restricts_documents_to_configured_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "documents").mkdir()
            manager = self._manager(root)

            allowed = manager.resolve_document(root / "documents" / "report.pdf")
            self.assertEqual(allowed, (root / "documents" / "report.pdf").resolve())
            with self.assertRaises(DocumentServiceError):
                manager.resolve_document(root / "outside.pdf")

    def test_deduplicates_active_ingestion_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            documents.mkdir()
            source = documents / "report.pdf"
            source.write_bytes(b"pdf")
            manager = self._manager(root)

            first, first_created = manager.submit(source)
            second, second_created = manager.submit(source)

            self.assertTrue(first_created)
            self.assertFalse(second_created)
            self.assertEqual(first["job_id"], second["job_id"])

    def test_worker_persists_ready_index_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            documents.mkdir()
            source = documents / "report.pdf"
            source.write_bytes(b"pdf")
            manager = self._manager(root)
            job, _created = manager.submit(source)
            with patch(
                "hrm_finance_qa.document_service.discover_documents",
                return_value=[source],
            ), patch(
                "hrm_finance_qa.document_service.DocumentPageIndex.load_or_build",
                return_value=([], PageIndexStats(total_pages=1, native_pages=1)),
            ) as load_or_build:
                manager._run_job(job["job_id"])

            completed = manager.get(job["job_id"])
            assert completed is not None
            self.assertEqual(completed["status"], "ready")
            self.assertEqual(completed["documents_completed"], 1)
            self.assertTrue(load_or_build.call_args.kwargs["ocr_config"].enabled)

    def test_answer_returns_not_ready_while_ingestion_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            documents.mkdir()
            source = documents / "report.pdf"
            source.write_bytes(b"pdf")
            manager = self._manager(root)
            job, _created = manager.submit(source)
            service = PersistentDocumentHRMService(SimpleNamespace(), manager)

            with self.assertRaises(DocumentNotReadyError) as raised:
                service.answer(
                    {"document": str(source), "question": "Doanh thu là bao nhiêu?"}
                )

            self.assertEqual(raised.exception.job["job_id"], job["job_id"])

    def test_health_identifies_loaded_runtime_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = self._manager(Path(directory))
            service = PersistentDocumentHRMService(SimpleNamespace(), manager)

            with patch.object(
                manager, "mineru_health", return_value={"status": "unavailable"}
            ):
                health = service.health()

            self.assertEqual(len(health["document_hrm_source_sha256"]), 64)
            self.assertEqual(len(health["runtime_source_sha256"]), 64)
            self.assertGreaterEqual(len(health["runtime_source_files"]), 5)
            self.assertTrue(
                any(
                    path.endswith("document_page_index.py")
                    for path in health["runtime_source_files"]
                )
            )
            self.assertTrue(health["document_hrm_source"].endswith("document_hrm.py"))
            self.assertGreater(health["runtime_started_at"], 0)

    def test_health_exposes_packaged_checkpoint_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = self._manager(root)
            checkpoint = root / "document_hrm_V8_candidate.pt"
            checkpoint.write_bytes(b"checkpoint")
            bundle = SimpleNamespace(
                metadata={
                    "architecture": "source-grounded-semantic-HRM-v10",
                    "checkpoint_schema_version": 10,
                }
            )
            service = PersistentDocumentHRMService(
                bundle,
                manager,
                model_path=checkpoint,
                device="cpu",
            )

            with patch.object(
                manager, "mineru_health", return_value={"status": "unavailable"}
            ):
                health = service.health()

            self.assertEqual(
                health["model"]["checkpoint"], "document_hrm_V8_candidate.pt"
            )
            self.assertEqual(health["model"]["device"], "cpu")
            self.assertEqual(health["model"]["size_bytes"], len(b"checkpoint"))
            self.assertEqual(
                health["model"]["architecture"],
                "source-grounded-semantic-HRM-v10",
            )

    def test_demo_handler_serves_ui_and_accepts_bounded_pdf_upload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            uploads = documents / "uploads"
            web_root = root / "web"
            documents.mkdir()
            web_root.mkdir()
            (web_root / "index.html").write_text(
                "<!doctype html><title>FinHRM</title>",
                encoding="utf-8",
            )
            manager = self._manager(root)
            service = PersistentDocumentHRMService(
                SimpleNamespace(metadata={}),
                manager,
            )
            server = _ServiceHTTPServer(
                ("127.0.0.1", 0),
                make_handler(
                    service,
                    web_root=web_root,
                    upload_root=uploads,
                    max_upload_bytes=1024,
                ),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=3
            )
            try:
                connection.request("GET", "/")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn(b"FinHRM", response.read())
                self.assertEqual(response.getheader("X-Frame-Options"), "DENY")
                self.assertEqual(response.getheader("Cache-Control"), "no-store")

                payload = b"%PDF-1.7 demo"
                connection.request(
                    "POST",
                    "/v1/upload?filename=bao-cao.pdf",
                    body=payload,
                    headers={"Content-Type": "application/pdf"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 201)
                upload = json.loads(response.read().decode("utf-8"))
                uploaded_path = Path(upload["document"])
                self.assertEqual(uploaded_path.read_bytes(), payload)
                self.assertEqual(uploaded_path.parent, uploads.resolve())
                self.assertEqual(upload["filename"], "bao-cao.pdf")
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
