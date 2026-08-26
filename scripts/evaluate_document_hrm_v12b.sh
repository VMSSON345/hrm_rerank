#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SIDECAR="${SIDECAR:-models/document_hrm_V12B1_program_reranker.pt}"

if [[ ! -f "$SIDECAR" ]]; then
  echo "Khong tim thay V12-B sidecar: $SIDECAR" >&2
  exit 1
fi

PYTHONPATH=src "$PYTHON_BIN" scripts/evaluate_document_hrm_v12_synthesis.py \
  --base-model models/document_hrm_V8_candidate.pt \
  --validation data/validation.jsonl \
  --financial-identity-catalog \
  --candidate-recall-v12-4 \
  --program-reranker-path "$SIDECAR" \
  --same-runtime-reference \
  --report reports/document_hrm_V12B1_validation.json \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  "$@"
