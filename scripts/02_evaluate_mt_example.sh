#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_DIR="$ROOT/evaluation/mt"

SAMPLES_FILE="${1:-$ROOT/outputs/mt/en-de/multi_turn_please_translate_again_demo/samples_history1.jsonl}"

if [[ ! -d "$EVAL_DIR/.venv" ]]; then
  (
    cd "$EVAL_DIR"
    uv venv .venv
  )
fi

source "$EVAL_DIR/.venv/bin/activate"
(
  cd "$EVAL_DIR"
  uv pip install -e .
  mt-eval \
    --file "$SAMPLES_FILE" \
    --metric comet_kiwi_qe \
    --batch-size 8
)
