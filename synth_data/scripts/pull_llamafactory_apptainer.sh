#!/usr/bin/env bash
set -euo pipefail

# Pull LlamaFactory Docker image into an Apptainer SIF file.
#
# Default output:
#   /fnwi_fs/ivi/irlab/projects/ltl-mt/sampling_for_mt_data/models/llamafactory_latest.sif
#
# Usage examples:
#   bash synth_data/scripts/pull_llamafactory_apptainer.sh
#   bash synth_data/scripts/pull_llamafactory_apptainer.sh --tag v0.9.3
#   bash synth_data/scripts/pull_llamafactory_apptainer.sh --output /path/to/llamafactory.sif --force

BASE_DATA_FOLDER_DEFAULT="/fnwi_fs/ivi/irlab/projects/ltl-mt/sampling_for_mt_data"
MODEL_STORE_DEFAULT="${BASE_DATA_FOLDER_DEFAULT}/models"
CACHE_DIR_DEFAULT="${BASE_DATA_FOLDER_DEFAULT}/.cache/apptainer"
TAG_DEFAULT="latest"

show_help() {
  cat <<'EOF'
Pull LlamaFactory Docker image into an Apptainer SIF.

Options:
  --tag <tag>         Docker tag (default: latest)
  --output <path>     Output .sif path
  --cache-dir <path>  Apptainer cache dir
  --force             Overwrite existing SIF
  -h, --help          Show this help message
EOF
}

TAG="${TAG_DEFAULT}"
OUTPUT_PATH=""
CACHE_DIR="${CACHE_DIR_DEFAULT}"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TAG="${2:?Missing value for --tag}"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="${2:?Missing value for --output}"
      shift 2
      ;;
    --cache-dir)
      CACHE_DIR="${2:?Missing value for --cache-dir}"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "[error] Unknown argument: $1" >&2
      show_help
      exit 2
      ;;
  esac
done

if ! command -v apptainer >/dev/null 2>&1; then
  echo "[error] apptainer not found in PATH." >&2
  exit 1
fi

if [[ -z "${OUTPUT_PATH}" ]]; then
  OUTPUT_PATH="${MODEL_STORE_DEFAULT}/llamafactory_${TAG}.sif"
fi

IMAGE_URI="docker://hiyouga/llamafactory:${TAG}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"
mkdir -p "${CACHE_DIR}"
export APPTAINER_CACHEDIR="${CACHE_DIR}"

echo "[info] image=${IMAGE_URI}"
echo "[info] output=${OUTPUT_PATH}"
echo "[info] cache=${APPTAINER_CACHEDIR}"

if [[ -f "${OUTPUT_PATH}" && "${FORCE}" -ne 1 ]]; then
  echo "[ok] SIF already exists (use --force to overwrite): ${OUTPUT_PATH}"
  exit 0
fi

PULL_ARGS=()
if [[ "${FORCE}" -eq 1 ]]; then
  PULL_ARGS+=(--force)
fi

apptainer pull "${PULL_ARGS[@]}" "${OUTPUT_PATH}" "${IMAGE_URI}"
echo "[ok] Pulled LlamaFactory image to ${OUTPUT_PATH}"

