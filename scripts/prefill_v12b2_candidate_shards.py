#!/usr/bin/env python3
"""Safely prefill disjoint V12-B.2 candidate-cache shards."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from hrm_finance_qa.program_data import load_program_examples  # noqa: E402
from hrm_finance_qa.v11_pipeline import (  # noqa: E402
    build_reranker_records,
    read_reranker_records,
    split_reranker_development,
    write_reranker_records,
)
from train_document_hrm_v12b_reranker import (  # noqa: E402
    CACHE_SCHEMA,
    _digest,
    _expected_manifest,
    _reject_protected,
    _runtime_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--label", choices=("train", "calibration"), required=True)
    parser.add_argument("--first-shard", type=int, required=True)
    parser.add_argument("--last-shard", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    _reject_protected(args.train)
    if args.first_shard < 0 or args.last_shard < args.first_shard:
        raise ValueError("Invalid shard range")

    development = load_program_examples(args.train)
    train_examples, calibration_examples, split_audit = split_reranker_development(
        development,
        [],
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    calibration_examples.sort(key=lambda example: example.example_id)
    examples = train_examples if args.label == "train" else calibration_examples
    expected_manifest = _expected_manifest(
        base_model=args.base_model,
        train_path=args.train,
        validation_path=None,
        top_k=args.top_k,
        shard_size=args.shard_size,
        split_audit=split_audit,
    )
    cache_identity = sha256(
        json.dumps(
            expected_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    bundle = _runtime_bundle(args.base_model, device=args.device)
    shard_root = args.cache_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    total_shards = (len(examples) + args.shard_size - 1) // args.shard_size
    if args.last_shard >= total_shards:
        raise ValueError(
            f"Shard range ends at {args.last_shard}, but {args.label} has "
            f"only {total_shards} shards"
        )

    for shard_index in range(args.first_shard, args.last_shard + 1):
        start = shard_index * args.shard_size
        current = examples[start : start + args.shard_size]
        end = start + len(current)
        stem = f"{args.label}_{start:05d}_{end:05d}"
        records_path = shard_root / f"{stem}.jsonl"
        audit_path = shard_root / f"{stem}.json"
        fingerprint = sha256(
            "\n".join(item.example_id for item in current).encode("utf-8")
        ).hexdigest()
        if records_path.is_file() and audit_path.is_file():
            manifest = json.loads(audit_path.read_text(encoding="utf-8"))
            if not (
                manifest.get("cache_identity") == cache_identity
                and manifest.get("example_fingerprint") == fingerprint
                and manifest.get("records_sha256") == _digest(records_path)
            ):
                raise RuntimeError(f"Existing shard is inconsistent: {stem}")
            read_reranker_records(records_path)
            print(f"[prefill {args.label}] {shard_index + 1}/{total_shards} reused")
            continue
        if records_path.exists() or audit_path.exists():
            raise RuntimeError(f"Partial shard exists; refusing concurrent write: {stem}")
        records, audit = build_reranker_records(
            bundle,
            current,
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
        write_reranker_records(records_path, records)
        payload = {
            "cache_schema": CACHE_SCHEMA,
            "cache_identity": cache_identity,
            "label": args.label,
            "start": start,
            "end": end,
            "example_fingerprint": fingerprint,
            "records": len(records),
            "records_sha256": _digest(records_path),
            "audit": audit,
        }
        audit_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[prefill {args.label}] {shard_index + 1}/{total_shards} saved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
