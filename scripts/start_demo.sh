#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${DEMO_PYTHON:-$PROJECT_ROOT/.venv-demo/bin/python}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:$PROJECT_ROOT/venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m finhrm_web.launcher "$@"
