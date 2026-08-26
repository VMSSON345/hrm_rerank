#!/usr/bin/env python3
"""Inspect cached page retrieval and numeric-candidate truncation without HRM."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hrm_finance_qa.document_page_index import build_page_indexed_chunks
from hrm_finance_qa.document_sources import (
    OCRConfig,
    discover_documents,
    expand_neighbor_chunks,
    rank_chunks,
)
from hrm_finance_qa.program_data import (
    _ocr_row_metric_score,
    merge_candidates,
)


def row_label(context: str) -> str:
    return next(
        (
            line.removeprefix("ROW:").strip()
            for line in context.splitlines()
            if line.startswith("ROW:")
        ),
        "",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("document", type=Path)
    parser.add_argument("question")
    parser.add_argument("--value", action="append", default=[])
    parser.add_argument("--page-index-path", type=Path, default=ROOT / "cache" / "document_page_index.sqlite3")
    parser.add_argument("--mineru-cache-dir", type=Path, default=ROOT / "cache" / "mineru")
    parser.add_argument("--top-k-chunks", type=int, default=12)
    parser.add_argument("--neighbor-chunks", type=int, default=2)
    parser.add_argument("--page-top-k", type=int, default=6)
    parser.add_argument("--page-neighbor-pages", type=int, default=1)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--legacy-mode", action="store_true")
    args = parser.parse_args()

    documents = discover_documents(args.document)
    indexed = build_page_indexed_chunks(
        documents,
        args.question,
        index_path=args.page_index_path,
        ocr_config=OCRConfig(
            enabled=True,
            mineru_cache_dir=args.mineru_cache_dir,
            mineru_api_url="http://127.0.0.1:8000",
        ),
        top_k_pages=args.page_top_k,
        neighbor_pages=args.page_neighbor_pages,
    )
    ranked = rank_chunks(args.question, indexed.chunks, top_k=args.top_k_chunks)
    ranked = expand_neighbor_chunks(
        ranked,
        indexed.chunks,
        neighbor_window=args.neighbor_chunks,
        max_chunks=args.top_k_chunks + max(0, args.neighbor_chunks * 4),
    )
    candidates = merge_candidates(
        args.question,
        indexed.chunks,
        max_candidates=args.max_candidates,
        legacy_mode=args.legacy_mode,
    )
    scores = [_ocr_row_metric_score(args.question, item) for item in candidates]
    best = max(scores, default=0.0)
    requested_values = set(args.value)
    selected = [
        (candidate, score)
        for candidate, score in zip(candidates, scores)
        if score >= best - 0.30 or candidate.value in requested_values
    ]
    payload = {
        "page_index": asdict(indexed.stats),
        "chunks": [
            {
                "chunk_id": item.chunk_id,
                "page": item.page_start,
                "score": item.score,
            }
            for item in ranked
        ],
        "candidate_extraction_pool": "all_chunks_from_selected_pages",
        "candidate_chunk_count": len(indexed.chunks),
        "candidate_count": len(candidates),
        "legacy_mode": args.legacy_mode,
        "metric_best": best,
        "requested_values_present": {
            value: any(item.value == value for item in candidates)
            for value in requested_values
        },
        "metric_candidates": [
            {
                "index": item.index,
                "value": item.value,
                "page": item.page,
                "row": row_label(item.context),
                "period": next(
                    (
                        line.removeprefix("PERIOD:").strip()
                        for line in item.context.splitlines()
                        if line.startswith("PERIOD:")
                    ),
                    "",
                ),
                "relevance": item.relevance,
                "metric_score": score,
                "extraction_method": item.extraction_method,
            }
            for item, score in selected
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
