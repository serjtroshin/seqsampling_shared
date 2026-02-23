#!/usr/bin/env bash
set -euo pipefail

MODEL_CONFIG="${1:?Usage: $0 <model_config_path> [experiment_dir]}"
EXPERIMENT_DIR="${2:-}"

CMD=(.venv/bin/python experiments/mt/sweep_runner.py
  --config experiments/mt/sweep_configs/mt_multi_turn.yaml
  --submit
  --override "model_config=${MODEL_CONFIG}"
)
if [[ -n "${EXPERIMENT_DIR}" ]]; then
  CMD+=(--experiment-dir "${EXPERIMENT_DIR}")
fi
exec "${CMD[@]}"
