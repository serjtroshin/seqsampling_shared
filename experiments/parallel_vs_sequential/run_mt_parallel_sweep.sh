#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[error] Missing .venv/bin/python. Run ./scripts/install_envs.sh first."
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_config_path>"
  echo "Example: $0 configs/model_configs/gemma-3-4b-it.yaml"
  exit 2
fi

MODEL_CONFIG="$1"
if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "[error] Model config not found: ${MODEL_CONFIG}"
  exit 1
fi

TS="$(date +%Y%m%d-%H%M%S)"
MODEL_SLUG="$(basename "${MODEL_CONFIG}")"
MODEL_SLUG="${MODEL_SLUG%.yaml}"
WORK_DIR="outputs/mt/sweeps/mt_parallel/${MODEL_SLUG}_submit_${TS}"

source .venv/bin/activate

python experiments/mt/sweep_runner.py \
  --config experiments/mt/sweep_configs/mt_parallel.yaml \
  --work-dir "${WORK_DIR}" \
  --submit \
  --override "model_config=${MODEL_CONFIG}"

echo "Submitted mt_parallel sweep runner job."
echo "Work directory: ${WORK_DIR}"
echo "Model config: ${MODEL_CONFIG}"
echo "Runs are written under: outputs/mt/pipeline_runs/<lang>/mt_parallel/<model>"
