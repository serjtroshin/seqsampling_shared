#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODEL_CONFIG="configs/model_configs/qwen3-32b-instruct.yaml"
EXPERIMENT_DIR="clear_run_27feb"
EXPERIMENT_DIR_ARGS=()
if [[ -n "${EXPERIMENT_DIR}" ]]; then
  EXPERIMENT_DIR_ARGS=(--experiment-dir "${EXPERIMENT_DIR}")
fi

# mt_parallel sweep (normal dataset)
# .venv/bin/python experiments/mt/sweep_runner.py \
#   --config experiments/mt/sweep_configs/mt_parallel.yaml \
#   --submit \
#   --override "model_config=${MODEL_CONFIG}" \
#   "${EXPERIMENT_DIR_ARGS[@]}"

# mt_parallel sweep (doc dataset)
# .venv/bin/python experiments/mt/sweep_runner.py \
#   --config experiments/mt/sweep_configs/mt_parallel_doc.yaml \
#   --submit \
#   --override "model_config=${MODEL_CONFIG}" \
#   "${EXPERIMENT_DIR_ARGS[@]}"

# mt_multi_turn sweep (normal dataset)
# .venv/bin/python experiments/mt/sweep_runner.py \
#   --config experiments/mt/sweep_configs/mt_multi_turn.yaml \
#   --submit \
#   --override "model_config=${MODEL_CONFIG}" \
#   "${EXPERIMENT_DIR_ARGS[@]}"

# mt_multi_turn sweep (doc dataset)
# .venv/bin/python experiments/mt/sweep_runner.py \
#   --config experiments/mt/sweep_configs/mt_multi_turn_plus_doc.yaml \
#   --submit \
#   --override "model_config=${MODEL_CONFIG}" \
#   "${EXPERIMENT_DIR_ARGS[@]}"

# greedy (normal dataset)
.venv/bin/python experiments/mt/sweep_runner.py \
  --config experiments/mt/sweep_configs/mt_greedy.yaml \
  --submit \
  --override "model_config=${MODEL_CONFIG}" \
  "${EXPERIMENT_DIR_ARGS[@]}"

# greedy (doc dataset)
.venv/bin/python experiments/mt/sweep_runner.py \
  --config experiments/mt/sweep_configs/mt_greedy_doc.yaml \
  --submit \
  --override "model_config=${MODEL_CONFIG}" \
  "${EXPERIMENT_DIR_ARGS[@]}"


# tail -f \
#   outputs/mt/sweeps/mt_parallel/mt-parallel_20260221-174806/sweep.log \
#   outputs/mt/sweeps/mt_parallel_doc/mt-parallel-doc_20260223-000000/sweep.log \
#   outputs/mt/sweeps/mt_multi_turn/mt-multi-turn_20260221-174806/sweep.log \
#   outputs/mt/sweeps/mt_multi_turn_plus_doc/mt-multi-turn-plus-doc_20260221-174807/sweep.log
