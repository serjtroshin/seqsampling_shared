#!/usr/bin/env bash
set -euo pipefail

TRAIN_JSONL="${1:-synth_data/outputs/processed/opus_train_pool.jsonl}"
OUTPUT_DIR="${2:-synth_data/outputs/iterations/iter_00/lf_dataset}"
DATASET_NAME="${3:-active_mt_train}"

PYTHONPATH="${PYTHONPATH:-}:src" .venv/bin/python -m active_mt_distill.sft.dataset_export \
  --train-jsonl "${TRAIN_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset-name "${DATASET_NAME}"
