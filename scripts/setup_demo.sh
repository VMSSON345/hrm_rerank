#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DEMO_VENV="${DEMO_VENV:-$PROJECT_ROOT/.venv-demo}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"

"$PYTHON_BOOTSTRAP" -m venv "$DEMO_VENV"
"$DEMO_VENV/bin/python" -m pip install --upgrade pip
"$DEMO_VENV/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt"
if [[ "${SKIP_WARMUP:-0}" != "1" ]]; then
  "$DEMO_VENV/bin/python" "$PROJECT_ROOT/scripts/warmup_document_hrm_demo.py"
fi

echo "Thiết lập hoàn tất."
echo "Chạy demo bằng: bash scripts/start_demo.sh"
