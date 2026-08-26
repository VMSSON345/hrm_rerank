from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.document_hrm import DocumentHRMBundle
from hrm_finance_qa.program_reranker import ProgramRerankerBundle
from hrm_finance_qa.document_service import (
    DocumentIngestionManager,
    PersistentDocumentHRMService,
    serve,
)
from hrm_finance_qa.document_sources import OCRConfig


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a persistent HRM API with background MinerU ingestion"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "models" / "document_hrm_V8_candidate.pt",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--program-reranker-path",
        type=Path,
        help="Optional program-reranker sidecar; V8 remains frozen and read-only",
    )
    parser.add_argument(
        "--typed-program-synthesis",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable experimental V12 typed-slot program synthesis. "
            "It remains disabled unless explicitly requested."
        ),
    )
    parser.add_argument(
        "--financial-identity-catalog",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the experimental YAML financial-identity catalog on top "
            "of typed synthesis. It remains disabled unless explicitly requested."
        ),
    )
    parser.add_argument(
        "--candidate-recall-v12-4",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable experimental V12.4 operation-conditioned candidate union "
            "and trace-complete fallback. Disabled by default."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--document-root",
        action="append",
        type=Path,
        dest="document_roots",
        help="Allowed document root; repeat for multiple roots",
    )
    parser.add_argument(
        "--page-index-path",
        type=Path,
        default=ROOT / "cache" / "document_page_index.sqlite3",
    )
    parser.add_argument(
        "--mineru-api-url",
        default="http://127.0.0.1:8000",
        help="Long-lived mineru-api base URL",
    )
    parser.add_argument(
        "--mineru-api-timeout",
        type=float,
        default=7200.0,
    )
    parser.add_argument(
        "--mineru-cache-dir",
        type=Path,
        default=ROOT / "cache" / "mineru",
    )
    parser.add_argument(
        "--mineru-backend",
        choices=("hybrid-engine", "pipeline"),
        default="hybrid-engine",
    )
    parser.add_argument(
        "--mineru-effort",
        choices=("medium", "high"),
        default="medium",
    )
    parser.add_argument(
        "--web-root",
        type=Path,
        default=ROOT / "web_demo",
        help="Static local demo UI directory",
    )
    parser.add_argument(
        "--upload-root",
        type=Path,
        default=ROOT / "tmp" / "demo_uploads",
        help="Private local directory for documents uploaded from the demo UI",
    )
    parser.add_argument(
        "--max-upload-mib",
        type=int,
        default=512,
        help="Maximum PDF/TXT upload size in MiB",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the local demo UI after the checkpoint is ready",
    )
    args = parser.parse_args()

    args.upload_root.mkdir(parents=True, exist_ok=True)
    configured_roots = args.document_roots or [ROOT / "pdf", ROOT / "examples"]
    roots = list(
        dict.fromkeys(
            Path(item).expanduser().resolve()
            for item in [*configured_roots, args.upload_root]
        )
    )
    load_started = time.perf_counter()
    bundle = DocumentHRMBundle.load(args.model_path, device=args.device)
    bundle.metadata = dict(bundle.metadata)
    bundle.metadata["typed_program_synthesis_enabled"] = bool(
        args.typed_program_synthesis
    )
    bundle.metadata["financial_identity_catalog_enabled"] = bool(
        args.financial_identity_catalog
    )
    bundle.metadata["candidate_recall_v12_4_enabled"] = bool(
        args.candidate_recall_v12_4
    )
    if args.program_reranker_path is not None:
        reranker = ProgramRerankerBundle.load(
            args.program_reranker_path, device=args.device
        )
        reranker.validate_base_model(args.model_path)
        bundle.attach_program_reranker(reranker)
    load_seconds = time.perf_counter() - load_started
    ingestion = DocumentIngestionManager(
        index_path=args.page_index_path,
        ocr_config=OCRConfig(
            enabled=True,
            mineru_cache_dir=args.mineru_cache_dir,
            mineru_api_url=args.mineru_api_url,
            mineru_api_timeout_seconds=args.mineru_api_timeout,
            mineru_backend=args.mineru_backend,
            mineru_effort=args.mineru_effort,
        ),
        allowed_roots=roots,
    )
    service = PersistentDocumentHRMService(
        bundle,
        ingestion,
        model_path=args.model_path,
        device=args.device,
    )
    print(
        f"HRM checkpoint loaded once in {load_seconds:.3f}s: {args.model_path}",
        flush=True,
    )
    if args.program_reranker_path is not None:
        print(
            f"V11 sidecar loaded: {args.program_reranker_path}",
            flush=True,
        )
    print(
        f"Document roots: {', '.join(str(Path(item).resolve()) for item in roots)}",
        flush=True,
    )
    local_url = f"http://{args.host}:{args.port}"
    print(f"Document HRM API: {local_url}", flush=True)
    if args.web_root.is_dir():
        print(f"Local demo UI: {local_url}", flush=True)
    print(f"MinerU API: {args.mineru_api_url}", flush=True)
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            "WARNING: no authentication is implemented; bind to loopback only.",
            flush=True,
        )
    if args.open_browser and args.web_root.is_dir():
        threading.Timer(0.8, webbrowser.open, args=(local_url,)).start()
    try:
        serve(
            service,
            host=args.host,
            port=args.port,
            web_root=args.web_root,
            upload_root=args.upload_root,
            max_upload_bytes=args.max_upload_mib * 1024 * 1024,
        )
    except KeyboardInterrupt:
        print("Stopping document HRM service", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
