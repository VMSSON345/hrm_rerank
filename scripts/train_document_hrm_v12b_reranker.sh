#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TOP_K="${TOP_K:-64}"
EPOCHS="${EPOCHS:-24}"
SHARD_SIZE="${SHARD_SIZE:-64}"

PYTHONPATH=src "$PYTHON_BIN" scripts/train_document_hrm_v12b_reranker.py \
  --base-model models/document_hrm_V8_candidate.pt \
  --train data/train.jsonl \
  --validation data/validation.jsonl \
  --cache-dir cache/v12b2_phase_a \
  --sidecar models/document_hrm_V12B2_phase_a_ablation.pt \
  --report reports/document_hrm_V12B2_phase_a_training.json \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --top-k "$TOP_K" \
  --shard-size "$SHARD_SIZE" \
  --epochs "$EPOCHS" \
  "$@"
