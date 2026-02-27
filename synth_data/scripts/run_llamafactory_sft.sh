#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-synth_data/configs/exp_base.yaml}"
TRAIN_JSONL="${2:-synth_data/outputs/processed/opus_train_pool.jsonl}"
BASE_MODEL="${3:-meta-llama/Llama-3.1-8B-Instruct}"
OUT_DIR="${4:-synth_data/outputs/models/sft_runs/iter_00}"
MODE="${5:-dry-run}" # dry-run | run

EXTRA_ARGS=()
if [[ "${MODE}" != "run" ]]; then
  EXTRA_ARGS+=(--dry-run)
fi

PYTHONPATH="${PYTHONPATH:-}:src" .venv/bin/python -m active_mt_distill.sft.llamafactory_runner \
  --config "${CONFIG_PATH}" \
  --train-jsonl "${TRAIN_JSONL}" \
  --base-model "${BASE_MODEL}" \
  --out-dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"
