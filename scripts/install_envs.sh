#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$ROOT/evaluation/mt"

echo "[install] core env: $ROOT/.venv"
(
  cd "$ROOT"
  uv venv
  uv sync --extra vllm --extra openai --extra analysis --extra dev
)

echo "[install] eval env: $EVAL_DIR/.venv"
(
  cd "$EVAL_DIR"
  uv venv .venv
  uv sync
)

echo "[ok] environments are ready."
echo "[hint] activate core env: source $ROOT/.venv/bin/activate"
echo "[hint] activate eval env: source $EVAL_DIR/.venv/bin/activate"
