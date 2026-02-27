#!/usr/bin/env bash
set -euo pipefail

# Probe whether nvcc is available on specific cluster nodes.
# Usage:
#   bash experiments/check_nvcc_nodes.sh
#   bash experiments/check_nvcc_nodes.sh ilps-cn116 ilps-cn117
#
# Optional env vars:
#   PARTITION=gpu
#   TIME_LIMIT=00:03:00
#   IMMEDIATE_SECS=30
#   GRES=gpu:nvidia_rtx_a6000:1

PARTITION="${PARTITION:-gpu}"
TIME_LIMIT="${TIME_LIMIT:-00:03:00}"
IMMEDIATE_SECS="${IMMEDIATE_SECS:-30}"
GRES="${GRES:-}"

if [[ "$#" -gt 0 ]]; then
  NODES=("$@")
else
  NODES=(ilps-cn116 ilps-cn117 ilps-cn118 ilps-cn119 ilps-cn120)
fi

echo "partition=${PARTITION} time_limit=${TIME_LIMIT} immediate_secs=${IMMEDIATE_SECS}"
if [[ -n "${GRES}" ]]; then
  echo "gres=${GRES}"
fi
echo

for node in "${NODES[@]}"; do
  srun_cmd=(
    srun
    --partition "${PARTITION}"
    --nodes 1
    --ntasks 1
    --cpus-per-task 1
    --time "${TIME_LIMIT}"
    --immediate="${IMMEDIATE_SECS}"
    --nodelist "${node}"
  )
  if [[ -n "${GRES}" ]]; then
    srun_cmd+=(--gres "${GRES}")
  fi

  probe_cmd=$'set -euo pipefail\nhost="$(hostname -s)"\npath_nvcc="$(command -v nvcc || true)"\ncandidates=(\n  "${path_nvcc}"\n  "/usr/local/cuda/bin/nvcc"\n  "/usr/bin/nvcc"\n  "/opt/cuda/bin/nvcc"\n)\nresolved=""\nfor cand in "${candidates[@]}"; do\n  if [[ -n "${cand}" && -x "${cand}" ]]; then\n    resolved="${cand}"\n    break\n  fi\ndone\nif [[ -n "${resolved}" ]]; then\n  version="$("${resolved}" --version 2>/dev/null | tail -n 1 || true)"\n  in_path=false\n  if [[ -n "${path_nvcc}" ]]; then in_path=true; fi\n  echo "status=ok host=${host} nvcc=${resolved} in_path=${in_path} version=${version}"\nelse\n  details=()\n  for cand in "/usr/local/cuda/bin/nvcc" "/usr/bin/nvcc" "/opt/cuda/bin/nvcc"; do\n    if [[ -e "${cand}" ]]; then\n      perms="$(ls -l "${cand}" 2>/dev/null || true)"\n      details+=("${cand}=>${perms}")\n    fi\n  done\n  echo "status=missing host=${host} nvcc_in_path=${path_nvcc} detail=${details[*]:-none}"\nfi'

  out=""
  if out="$("${srun_cmd[@]}" bash -lc "${probe_cmd}" 2>&1)"; then
    status_line="$(printf '%s\n' "${out}" | grep -E '^status=' | tail -n 1 || true)"
    if [[ -n "${status_line}" ]]; then
      printf '%s %s\n' "node=${node}" "${status_line}"
    else
      printf '%s status=unknown detail=%q\n' "node=${node}" "${out}"
    fi
  else
    err_line="$(printf '%s\n' "${out}" | tail -n 1)"
    printf '%s status=unreachable detail=%q\n' "node=${node}" "${err_line}"
  fi
done
