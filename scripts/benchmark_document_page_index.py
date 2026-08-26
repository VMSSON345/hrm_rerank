from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.document_page_index import build_page_indexed_chunks
from hrm_finance_qa.document_sources import OCRConfig


def _run_once(
    document: Path,
    question: str,
    *,
    index_path: Path,
    mineru_cache_dir: Path,
    mineru_command: Path,
    mineru_api_url: str | None,
    mineru_api_timeout_seconds: float,
    mineru_backend: str,
    mineru_effort: str,
    top_k_pages: int,
    neighbor_pages: int,
    rebuild: bool,
) -> dict[str, object]:
    tracemalloc.reset_peak()
    started_at = time.perf_counter()
    result = build_page_indexed_chunks(
        [document],
        question,
        index_path=index_path,
        ocr_config=OCRConfig(
            enabled=True,
            mineru_cache_dir=mineru_cache_dir,
            mineru_command=mineru_command,
            mineru_api_url=mineru_api_url,
            mineru_api_timeout_seconds=mineru_api_timeout_seconds,
            mineru_backend=mineru_backend,
            mineru_effort=mineru_effort,
        ),
        top_k_pages=top_k_pages,
        neighbor_pages=neighbor_pages,
        rebuild=rebuild,
    )
    elapsed = time.perf_counter() - started_at
    _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    return {
        "wall_seconds": elapsed,
        "python_peak_memory_mb": peak_bytes / (1024 * 1024),
        "chunk_count": len(result.chunks),
        "stats": asdict(result.stats),
        "warnings": list(result.warnings),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure isolated cold/warm page-index retrieval on one PDF"
    )
    parser.add_argument("document", type=Path)
    parser.add_argument("question")
    parser.add_argument(
        "--mineru-command",
        type=Path,
        default=ROOT / ".venv-mineru" / "bin" / "mineru",
    )
    parser.add_argument(
        "--mineru-api-url",
        help=(
            "Use an already-running MinerU API, for example "
            "http://127.0.0.1:8000; avoids model startup in the benchmark"
        ),
    )
    parser.add_argument("--mineru-api-timeout", type=float, default=7200.0)
    parser.add_argument(
        "--mineru-backend",
        choices=("hybrid-engine", "pipeline"),
        default="hybrid-engine",
    )
    parser.add_argument(
        "--mineru-effort",
        choices=("low", "medium", "high"),
        default="medium",
    )
    parser.add_argument("--top-k-pages", type=int, default=6)
    parser.add_argument("--neighbor-pages", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    document = args.document.expanduser().resolve()
    if not document.is_file() or document.suffix.lower() != ".pdf":
        parser.error(f"PDF not found: {document}")

    with tempfile.TemporaryDirectory(prefix="page-index-benchmark-") as directory:
        temporary = Path(directory)
        index_path = temporary / "document_page_index.sqlite3"
        mineru_cache_dir = temporary / "mineru"
        tracemalloc.start()
        try:
            cold = _run_once(
                document,
                args.question,
                index_path=index_path,
                mineru_cache_dir=mineru_cache_dir,
                mineru_command=args.mineru_command,
                mineru_api_url=args.mineru_api_url,
                mineru_api_timeout_seconds=args.mineru_api_timeout,
                mineru_backend=args.mineru_backend,
                mineru_effort=args.mineru_effort,
                top_k_pages=args.top_k_pages,
                neighbor_pages=args.neighbor_pages,
                rebuild=True,
            )
            warm = _run_once(
                document,
                args.question,
                index_path=index_path,
                mineru_cache_dir=mineru_cache_dir,
                mineru_command=args.mineru_command,
                mineru_api_url=args.mineru_api_url,
                mineru_api_timeout_seconds=args.mineru_api_timeout,
                mineru_backend=args.mineru_backend,
                mineru_effort=args.mineru_effort,
                top_k_pages=args.top_k_pages,
                neighbor_pages=args.neighbor_pages,
                rebuild=False,
            )
        finally:
            tracemalloc.stop()

    report = {
        "document": str(document),
        "question": args.question,
        "mineru_mode": "persistent_api" if args.mineru_api_url else "subprocess",
        "measurement_note": (
            "python_peak_memory_mb is parent-process tracemalloc memory; memory of "
            "the external MinerU API/subprocess is not included"
        ),
        "cold": cold,
        "warm": warm,
        "warm_speedup": (
            cold["wall_seconds"] / warm["wall_seconds"]
            if warm["wall_seconds"]
            else None
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
