#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source .venv/bin/activate

SCENARIO="${1:-configs/mt/mt_multi_turn_please_translate_again.yaml}"
PROMPTS_JSONL="${ROOT}/data/mt_demo_en_de.jsonl"

if [[ ! -f "$SCENARIO" ]]; then
  echo "[error] Scenario file not found: $SCENARIO"
  exit 1
fi

# Small demo overrides for a quick run.
set +e
parseq-core \
  --scenario "$SCENARIO" \
  prompts_path="$PROMPTS_JSONL" \
  prompt_key=source \
  extra_data_fields=[source,target] \
  limit_prompts=2 \
  num_generations=2 \
  max_tokens=128 \
  output_dir=../../outputs/mt/en-de/multi_turn_please_translate_again_demo
status=$?
set -e

if [[ $status -ne 0 ]]; then
  echo "[error] Generation failed."
  echo "[hint] If this scenario uses vLLM, make sure the server is running:"
  echo "       parseq-core-vllm-server --scenario \"$SCENARIO\""
  exit $status
fi
