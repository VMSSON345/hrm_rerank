#!/usr/bin/env bash
set -euo pipefail

# Keep MinerU and its large ML dependency tree out of the HRM project's venv.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MINERU_VENV="${MINERU_VENV:-$PROJECT_ROOT/.venv-mineru}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"

if [[ ! -f "$MINERU_VENV/bin/activate" ]]; then
  if [[ -d "$MINERU_VENV" ]]; then
    echo "Incomplete MinerU environment detected; rebuilding: $MINERU_VENV"
    "$PYTHON_BOOTSTRAP" -m venv --clear "$MINERU_VENV"
  else
    echo "Creating isolated MinerU environment: $MINERU_VENV"
    "$PYTHON_BOOTSTRAP" -m venv "$MINERU_VENV"
  fi
fi

if [[ ! -f "$MINERU_VENV/bin/activate" ]]; then
  echo "Failed to create $MINERU_VENV/bin/activate." >&2
  echo "On Ubuntu/Debian, install the matching venv package, for example:" >&2
  echo "  sudo apt install python3-venv" >&2
  exit 1
fi

if [[ ! -x "$MINERU_VENV/bin/python" ]]; then
  echo "Invalid Python executable: $MINERU_VENV/bin/python" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$MINERU_VENV/bin/activate"
python -m pip install --upgrade pip uv
uv pip install --upgrade "mineru[all]"
mineru --version

echo "MinerU environment is ready: $MINERU_VENV"
echo "Activate with: source $MINERU_VENV/bin/activate"
echo "Start API: bash scripts/run_mineru_api.sh"
echo "Run persistent API: bash scripts/run_mineru_api.sh"
