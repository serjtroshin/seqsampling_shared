#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$ROOT/evaluation/mt"
UV_CACHE_DIR_DEFAULT="/tmp/${USER:-user}/uv-cache"

# Some systems mount $HOME cache locations as read-only. Keep uv cache writable.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$UV_CACHE_DIR_DEFAULT}"
mkdir -p "$UV_CACHE_DIR"
echo "[install] uv cache: $UV_CACHE_DIR"

echo "[install] core env: $ROOT/.venv"
(
  cd "$ROOT"
  uv venv
  uv sync --extra vllm --extra openai --extra analysis --extra dev
)

# echo "[install] eval env: $EVAL_DIR/.venv"
# (
#   cd "$EVAL_DIR"
#   uv venv .venv
#   uv sync
# )

echo "[ok] environments are ready."
echo "[hint] activate core env: source $ROOT/.venv/bin/activate"
echo "[hint] activate eval env: source $EVAL_DIR/.venv/bin/activate"
