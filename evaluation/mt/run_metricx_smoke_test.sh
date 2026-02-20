#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

EVAL_PY="${PROJECT_ROOT}/evaluation/mt/metricx/.venv/bin/python"
SAMPLES_FILE="${1:-${PROJECT_ROOT}/evaluation/mt/test_data/metricx_smoke_samples.jsonl}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/evaluation/mt/test_outputs/metricx_smoke}"
CONFIG_FILE="${3:-${PROJECT_ROOT}/evaluation/mt/metricx_smoke_config.json}"

if [[ ! -x "${EVAL_PY}" ]]; then
  echo "Missing evaluator python: ${EVAL_PY}" >&2
  exit 1
fi
if [[ ! -f "${SAMPLES_FILE}" ]]; then
  echo "Missing samples file: ${SAMPLES_FILE}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Missing config file: ${CONFIG_FILE}" >&2
  exit 1
fi
if [[ ! -x "${PROJECT_ROOT}/evaluation/mt/metricx/.venv/bin/python" ]]; then
  echo "Missing MetricX env python: ${PROJECT_ROOT}/evaluation/mt/metricx/.venv/bin/python" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${PROJECT_ROOT}/evaluation/mt/src${PYTHONPATH:+:${PYTHONPATH}}"

echo "[metricx-smoke] running mt-eval..."
"${EVAL_PY}" -m mt_evaluation.cli \
  --file "${SAMPLES_FILE}" \
  --config "${CONFIG_FILE}" \
  --output-dir "${OUTPUT_DIR}"

REPORT_FILE="${OUTPUT_DIR}/metricx24_ref_smoke.json"
if [[ ! -f "${REPORT_FILE}" ]]; then
  echo "[metricx-smoke] missing expected report: ${REPORT_FILE}" >&2
  exit 1
fi

"${EVAL_PY}" - "${REPORT_FILE}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text(encoding="utf-8"))
scores = report.get("scores")
if not isinstance(scores, list) or not scores:
    raise SystemExit(f"invalid or empty scores in {path}")
for idx, row in enumerate(scores):
    if not isinstance(row, dict) or "score" not in row:
        raise SystemExit(f"missing score field at row {idx} in {path}")
    float(row["score"])
print(f"[metricx-smoke] ok: {path} with {len(scores)} scores")
PY

echo "[metricx-smoke] success"
