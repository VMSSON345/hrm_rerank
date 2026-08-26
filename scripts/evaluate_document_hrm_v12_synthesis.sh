#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
DEVICE="${DEVICE:-cpu}"
BASE_MODEL="${BASE_MODEL:-models/document_hrm_V8_candidate.pt}"
VALIDATION_DATA="${VALIDATION_DATA:-data/validation.jsonl}"
REPORT_PATH="${REPORT_PATH:-reports/document_hrm_V12_typed_synthesis_validation.json}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TOP_K="${TOP_K:-30}"

if [[ "${RUN_BLIND:-0}" == "1" ]]; then
  echo "Refusing to use the consumed blind test for V12 development." >&2
  exit 2
fi

echo "[1/2] Running typed synthesis and symbolic regression tests"
PYTHONPATH=src "$PYTHON_BIN" -m unittest \
  tests.test_program_synthesis \
  tests.test_question_planner \
  tests.test_program_reranker \
  tests.test_document_hrm.SymbolicGroundingConstraintTests

echo "[2/2] Running validation-only V8 vs V12 synthesis ablation"
PYTHONPATH=src "$PYTHON_BIN" scripts/evaluate_document_hrm_v12_synthesis.py \
  --base-model "$BASE_MODEL" \
  --validation "$VALIDATION_DATA" \
  --report "$REPORT_PATH" \
  --batch-size "$BATCH_SIZE" \
  --top-k "$TOP_K" \
  --device "$DEVICE"

echo "Saved report: $REPORT_PATH"
echo "Frozen base: $BASE_MODEL"
echo "Blind test was not touched."
