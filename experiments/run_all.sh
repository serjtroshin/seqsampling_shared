#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODEL_CONFIG="configs/model_configs/qwen3-32b-instruct.yaml"
EXPERIMENT_DIR="clear_run_22feb"

# mt_parallel sweep (normal dataset)
bash experiments/parallel_vs_sequential/run_mt_parallel_sweep.sh "${MODEL_CONFIG}" "${EXPERIMENT_DIR}"

# mt_multi_turn sweep (normal dataset)
bash experiments/parallel_vs_sequential/run_mt_multi_turn_sweep.sh "${MODEL_CONFIG}" "${EXPERIMENT_DIR}"

# mt_multi_turn sweep (doc dataset)
bash experiments/parallel_vs_sequential/run_mt_multi_turn_plus_doc_sweep.sh "${MODEL_CONFIG}" "${EXPERIMENT_DIR}"


# tail -f \
#   outputs/mt/sweeps/mt_parallel/mt-parallel_20260221-174806/sweep.log \
#   outputs/mt/sweeps/mt_multi_turn/mt-multi-turn_20260221-174806/sweep.log \
#   outputs/mt/sweeps/mt_multi_turn_plus_doc/mt-multi-turn-plus-doc_20260221-174807/sweep.log
