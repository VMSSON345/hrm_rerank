#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MINERU_API_HOST="${MINERU_API_HOST:-127.0.0.1}"
MINERU_API_PORT="${MINERU_API_PORT:-8000}"
MINERU_VLM_MAX_CONCURRENCY="${MINERU_VLM_MAX_CONCURRENCY:-1}"
export MINERU_API_OUTPUT_ROOT="${MINERU_API_OUTPUT_ROOT:-$PROJECT_ROOT/cache/mineru-api}"
export MINERU_API_DISABLE_ACCESS_LOG="${MINERU_API_DISABLE_ACCESS_LOG:-1}"
export MINERU_PROCESSING_WINDOW_SIZE="${MINERU_PROCESSING_WINDOW_SIZE:-4}"
export MINERU_API_MAX_CONCURRENT_REQUESTS="${MINERU_API_MAX_CONCURRENT_REQUESTS:-1}"

if [[ ! -x "$PROJECT_ROOT/.venv-mineru/bin/mineru-api" ]]; then
  echo "MinerU API executable not found. Run: bash scripts/setup_mineru_venv.sh" >&2
  exit 1
fi

echo "Starting persistent MinerU API at http://$MINERU_API_HOST:$MINERU_API_PORT"
echo "MinerU API output: $MINERU_API_OUTPUT_ROOT"
echo "MinerU processing window: $MINERU_PROCESSING_WINDOW_SIZE page(s)"
echo "MinerU max concurrent requests: $MINERU_API_MAX_CONCURRENT_REQUESTS"
echo "MinerU VLM max concurrency: $MINERU_VLM_MAX_CONCURRENCY"
exec "$PROJECT_ROOT/.venv-mineru/bin/mineru-api" \
  --host "$MINERU_API_HOST" \
  --port "$MINERU_API_PORT" \
  --max-concurrency "$MINERU_VLM_MAX_CONCURRENCY"
