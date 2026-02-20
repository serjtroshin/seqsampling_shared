#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_TARGET="outputs/mt/experiments_v6/ru/google-gemma-3-4b-it/20260210-000518_wmt-v6-model-priority_s07w000_0029_par100_seq5_h5/parallel"
TARGET_INPUT="${1:-$DEFAULT_TARGET}"
CONFIG_FILE="${2:-${SCRIPT_DIR}/metrics.example.json}"
VENV_PY="${VENV_PY:-${SCRIPT_DIR}/.venv/bin/python}"
VENV_UV="${SCRIPT_DIR}/.venv/bin/uv"

if [[ "${TARGET_INPUT}" = /* ]]; then
  TARGET_DIR="${TARGET_INPUT}"
else
  TARGET_DIR="${REPO_ROOT}/${TARGET_INPUT}"
fi

SAMPLES_FILE="${TARGET_DIR}/samples.jsonl"
OUTPUT_DIR="${TARGET_DIR}/evaluation"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "Missing python interpreter: ${VENV_PY}"
  echo "Create the eval env first:"
  echo "  cd evaluation/mt && uv venv .venv && source .venv/bin/activate && uv pip install -e ."
  exit 1
fi

if [[ ! -f "${SAMPLES_FILE}" ]]; then
  echo "Missing samples file: ${SAMPLES_FILE}"
  exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing config file: ${CONFIG_FILE}"
  exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${VENV_PY}" -m mt_evaluation.cli \
  --file "${SAMPLES_FILE}" \
  --config "${CONFIG_FILE}" \
  --output-dir "${OUTPUT_DIR}"

echo "Saved evaluation reports to: ${OUTPUT_DIR}"
