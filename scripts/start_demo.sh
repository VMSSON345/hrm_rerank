#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DEMO_VENV="${DEMO_VENV:-$PROJECT_ROOT/.venv-demo}"
DEVICE="${DEVICE:-cpu}"
PORT="${PORT:-8765}"
MINERU_API_URL="${MINERU_API_URL:-http://127.0.0.1:8000}"
CHECKPOINT="${MODEL_PATH:-$PROJECT_ROOT/models/document_hrm_V8_candidate.pt}"
PROGRAM_RERANKER_PATH="${PROGRAM_RERANKER_PATH:-}"
TYPED_PROGRAM_SYNTHESIS="${TYPED_PROGRAM_SYNTHESIS:-0}"
FINANCIAL_IDENTITY_CATALOG="${FINANCIAL_IDENTITY_CATALOG:-0}"
CANDIDATE_RECALL_V12_4="${CANDIDATE_RECALL_V12_4:-0}"

if [[ ! -x "$DEMO_VENV/bin/python" ]]; then
  echo "Chưa có môi trường demo. Chạy: bash scripts/setup_demo.sh" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Không tìm thấy checkpoint V8: $CHECKPOINT" >&2
  exit 1
fi
if [[ -n "$PROGRAM_RERANKER_PATH" && ! -f "$PROGRAM_RERANKER_PATH" ]]; then
  echo "Khong tim thay program-reranker sidecar: $PROGRAM_RERANKER_PATH" >&2
  exit 1
fi

echo "Khởi động FinHRM local tại http://127.0.0.1:$PORT"
echo "Nhấn Ctrl+C để dừng."
ARGS=(
  "$PROJECT_ROOT/scripts/serve_document_hrm.py"
  --model-path "$CHECKPOINT"
  --device "$DEVICE"
  --host 127.0.0.1
  --port "$PORT"
  --mineru-api-url "$MINERU_API_URL"
)
if [[ -n "$PROGRAM_RERANKER_PATH" ]]; then
  ARGS+=(--program-reranker-path "$PROGRAM_RERANKER_PATH")
fi
if [[ "$TYPED_PROGRAM_SYNTHESIS" == "1" ]]; then
  ARGS+=(--typed-program-synthesis)
fi
if [[ "$FINANCIAL_IDENTITY_CATALOG" == "1" ]]; then
  ARGS+=(--financial-identity-catalog)
fi
if [[ "$CANDIDATE_RECALL_V12_4" == "1" ]]; then
  ARGS+=(--candidate-recall-v12-4)
fi
exec "$DEMO_VENV/bin/python" "${ARGS[@]}"
