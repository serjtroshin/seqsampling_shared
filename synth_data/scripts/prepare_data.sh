#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-synth_data/configs/exp_base.yaml}"
shift || true

PYTHONPATH="${PYTHONPATH:-}:src" .venv/bin/python -m active_mt_distill.datasets.prepare \
  --config "${CONFIG_PATH}" \
  "$@"

