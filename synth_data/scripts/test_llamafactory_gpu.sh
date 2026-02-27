#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

IMAGE_DEFAULT="/fnwi_fs/ivi/irlab/projects/ltl-mt/sampling_for_mt_data/models/llamafactory_latest.sif"
CONFIG_DEFAULT="synth_data/configs/exp_base.yaml"
TRAIN_JSONL_DEFAULT="synth_data/data/processed/opus_train_pool.jsonl"
OUT_DIR_DEFAULT="synth_data/data/iterations/llamafactory_gpu_test"
BASE_MODEL_DEFAULT="meta-llama/Llama-3.1-8B-Instruct"
TINY_ROWS_DEFAULT=64

IMAGE="${IMAGE_DEFAULT}"
CONFIG_PATH="${CONFIG_DEFAULT}"
TRAIN_JSONL="${TRAIN_JSONL_DEFAULT}"
OUT_DIR="${OUT_DIR_DEFAULT}"
BASE_MODEL="${BASE_MODEL_DEFAULT}"
TINY_ROWS="${TINY_ROWS_DEFAULT}"
RUN_TRAIN=0
SKIP_GPU_CHECK=0

show_help() {
  cat <<'EOF'
Run LlamaFactory GPU smoke tests with Apptainer.

Default behavior:
1) checks image + CLI
2) checks CUDA visibility inside container
3) runs the SFT wrapper in dry-run mode on a tiny dataset

Optional:
- run a tiny real training job with --run-train

Options:
  --image <path>        Apptainer image path
  --config <path>       Config path (default: synth_data/configs/exp_base.yaml)
  --train-jsonl <path>  Source training jsonl
  --out-dir <path>      Output directory for test artifacts
  --base-model <id>     Base model for the SFT wrapper
  --tiny-rows <int>     Rows to sample for tiny smoke dataset
  --run-train           Run a real tiny training job (not just dry-run)
  --skip-gpu-check      Skip CUDA check (useful on CPU-only machines)
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="${2:?Missing value for --image}"
      shift 2
      ;;
    --config)
      CONFIG_PATH="${2:?Missing value for --config}"
      shift 2
      ;;
    --train-jsonl)
      TRAIN_JSONL="${2:?Missing value for --train-jsonl}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:?Missing value for --out-dir}"
      shift 2
      ;;
    --base-model)
      BASE_MODEL="${2:?Missing value for --base-model}"
      shift 2
      ;;
    --tiny-rows)
      TINY_ROWS="${2:?Missing value for --tiny-rows}"
      shift 2
      ;;
    --run-train)
      RUN_TRAIN=1
      shift
      ;;
    --skip-gpu-check)
      SKIP_GPU_CHECK=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "[error] Unknown arg: $1" >&2
      show_help
      exit 2
      ;;
  esac
done

if ! command -v apptainer >/dev/null 2>&1; then
  echo "[error] apptainer is not available in PATH" >&2
  exit 1
fi
if [[ ! -f "${IMAGE}" ]]; then
  echo "[error] Image not found: ${IMAGE}" >&2
  exit 1
fi
if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "[error] Training jsonl not found: ${TRAIN_JSONL}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[error] Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

echo "[info] image=${IMAGE}"
echo "[info] config=${CONFIG_PATH}"
echo "[info] train_jsonl=${TRAIN_JSONL}"
echo "[info] out_dir=${OUT_DIR}"
echo "[info] base_model=${BASE_MODEL}"

echo "[step] LlamaFactory CLI version in container"
apptainer exec "${IMAGE}" llamafactory-cli version

if [[ "${SKIP_GPU_CHECK}" -ne 1 ]]; then
  echo "[step] CUDA visibility check in container (--nv)"
  apptainer exec --nv "${IMAGE}" python - <<'PY'
import sys
import torch
ok = torch.cuda.is_available() and torch.cuda.device_count() > 0
print("cuda_available=", torch.cuda.is_available())
print("cuda_device_count=", torch.cuda.device_count())
if not ok:
    raise SystemExit("No CUDA devices visible inside container.")
PY
else
  echo "[step] Skipping CUDA visibility check (--skip-gpu-check set)"
fi

mkdir -p "${OUT_DIR}"
TINY_JSONL="${OUT_DIR}/tiny_train.jsonl"
head -n "${TINY_ROWS}" "${TRAIN_JSONL}" > "${TINY_JSONL}"
echo "[step] Tiny dataset prepared at ${TINY_JSONL} (${TINY_ROWS} rows)"

echo "[step] Wrapper dry-run"
bash synth_data/scripts/run_llamafactory_sft.sh \
  "${CONFIG_PATH}" \
  "${TINY_JSONL}" \
  "${BASE_MODEL}" \
  "${OUT_DIR}" \
  dry-run

if [[ "${RUN_TRAIN}" -eq 1 ]]; then
  echo "[step] Running tiny real training smoke test"
  bash synth_data/scripts/run_llamafactory_sft.sh \
    "${CONFIG_PATH}" \
    "${TINY_JSONL}" \
    "${BASE_MODEL}" \
    "${OUT_DIR}" \
    run
  echo "[ok] Training smoke test finished."
else
  echo "[ok] Dry-run smoke test finished. Use --run-train to execute real training."
fi

