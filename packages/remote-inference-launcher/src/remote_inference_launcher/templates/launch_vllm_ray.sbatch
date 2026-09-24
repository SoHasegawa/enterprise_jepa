#!/usr/bin/env bash
@sbatch_directives

set -euo pipefail
OUT_DIR=@out_dir
MODEL_ID=@model
SERVED_MODEL_NAME=@served_model_name
REQUESTED_HEAD_NODE_VLLM_API_PORT=@head_node_port
REMOTE_INFERENCE_STATE_PATH=@state_path
RAY_PORT=@ray_port
REMOTE_VLLM_NUM_GPUS=@num_gpus
REMOTE_VLLM_PYTHON_BIN=@python_bin
REMOTE_VLLM_TARGET_DEVICE=@target_device
RUNTIME_TMP_ROOT=@runtime_tmp_root
REMOTE_HF_HOME=@hf_home
EXTRA_ARGS=@extra_args
VLLM_OPTIONS=@vllm_options

expand_home_path() {
  if [ "$1" = "~" ]; then
    printf "%s\n" "${HOME}"
  elif [ "${1#\~/}" != "$1" ]; then
    printf "%s/%s\n" "${HOME}" "${1#\~/}"
  else
    printf "%s\n" "$1"
  fi
}

prepend_python_package_library_paths() {
  local python_bin="$1"
  local venv_path
  local site_packages_dir
  local library_dir
  venv_path="$(cd "$(dirname "${python_bin}")/.." 2>/dev/null && pwd -P)" || return 0
  for site_packages_dir in "${venv_path}"/lib/python*/site-packages; do
    [ -d "${site_packages_dir}" ] || continue
    for library_dir in "${site_packages_dir}"/nvidia/*/lib "${site_packages_dir}"/torch/lib; do
      if [ -d "${library_dir}" ]; then
        export LD_LIBRARY_PATH="${library_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
      fi
    done
  done
}

select_free_tcp_port() {
  "${REMOTE_VLLM_PYTHON_BIN}" - <<'PY'
import socket
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("0.0.0.0", 0))
    print(sock.getsockname()[1])
PY
}

write_state_file() {
  "${REMOTE_VLLM_PYTHON_BIN}" - "$REMOTE_INFERENCE_STATE_PATH" "$HEAD_NODE_VLLM_API_PORT" <<'PY'
import json
import socket
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "remote_port": int(sys.argv[2]),
    "hostname": socket.gethostname(),
}
path.parent.mkdir(parents=True, exist_ok=True)
tmp_path = path.with_suffix(path.suffix + ".tmp")
tmp_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
tmp_path.replace(path)
PY
}

OUT_DIR="$(expand_home_path "${OUT_DIR}")"
LOG_PATH="${OUT_DIR}/vllm.log"
REMOTE_VLLM_PYTHON_BIN="$(expand_home_path "${REMOTE_VLLM_PYTHON_BIN}")"
REMOTE_INFERENCE_STATE_PATH="$(expand_home_path "${REMOTE_INFERENCE_STATE_PATH}")"
RUNTIME_TMP_ROOT="$(expand_home_path "${RUNTIME_TMP_ROOT}")"
REMOTE_HF_HOME="$(expand_home_path "${REMOTE_HF_HOME}")"
prepend_python_package_library_paths "${REMOTE_VLLM_PYTHON_BIN}"

if [ -z "${RUNTIME_TMP_ROOT}" ]; then
  RUNTIME_TMP_ROOT="/tmp/remote-inference-launcher-vllm-${USER:-unknown}"
fi
TMP_ROOT="${RUNTIME_TMP_ROOT%/}/${SLURM_JOB_ID:-$$}"
mkdir -p "${OUT_DIR}" "${TMP_ROOT}/ipc"
mkdir -p "${TMP_ROOT}/torchinductor" "${TMP_ROOT}/triton"
RAY_PIDS=()
cleanup_runtime_tmp() {
  set +e
  if [ "${#RAY_PIDS[@@]}" -gt 0 ]; then
    for pid in "${RAY_PIDS[@@]}"; do
      kill "${pid}" 2>/dev/null || true
    done
  fi
  rm -rf "${TMP_ROOT}"
}
trap cleanup_runtime_tmp EXIT
export TMPDIR="${TMP_ROOT}/ipc"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export TORCHINDUCTOR_CACHE_DIR="${TMP_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${TMP_ROOT}/triton"
if [ -n "${REMOTE_HF_HOME}" ]; then
  export HF_HOME="${REMOTE_HF_HOME}"
  export HF_HUB_CACHE="${HF_HOME}/hub"
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
  mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"
fi
if [ -n "${REMOTE_VLLM_TARGET_DEVICE}" ]; then
  export VLLM_TARGET_DEVICE="${REMOTE_VLLM_TARGET_DEVICE}"
fi
if [ "${REMOTE_VLLM_TARGET_DEVICE}" = "rocm" ]; then
  export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
  export ROCM_HOME="${ROCM_HOME:-${ROCM_PATH}}"
fi
@setup_cmd_block
if [ "${REMOTE_VLLM_TARGET_DEVICE}" = "rocm" ]; then
  REAL_REMOTE_VLLM_PYTHON_BIN="${REMOTE_VLLM_PYTHON_BIN}"
  ROCM_PYTHON_WRAPPER="${OUT_DIR}/python-rocm-visible-devices"
  {
    printf '%s\n' '#!/usr/bin/env bash'
    printf '%s\n' 'if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then'
    printf '%s\n' '  export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$ROCR_VISIBLE_DEVICES}"'
    printf '%s\n' '  unset ROCR_VISIBLE_DEVICES'
    printf '%s\n' 'fi'
    printf 'exec %q "$@@"\n' "${REAL_REMOTE_VLLM_PYTHON_BIN}"
  } > "${ROCM_PYTHON_WRAPPER}"
  chmod +x "${ROCM_PYTHON_WRAPPER}"
  REMOTE_VLLM_PYTHON_BIN="${ROCM_PYTHON_WRAPPER}"
fi
resolve_node_ip() {
  local node="$1"
  local ip
  ip="$(getent hosts "${node}" | awk '{print $1; exit}')"
  if [ -z "${ip}" ]; then
    echo "Could not resolve Slurm node to an IP address: ${node}" >&2
    exit 1
  fi
  printf "%s\n" "${ip}"
}

mapfile -t RAY_NODES < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
if [ "${#RAY_NODES[@@]}" -lt 2 ]; then
  echo "Expected at least two Slurm nodes for distributed vLLM; got ${#RAY_NODES[@@]}" >&2
  exit 1
fi
HEAD_NODE="${RAY_NODES[0]}"
HEAD_NODE_IP="$(resolve_node_ip "${HEAD_NODE}")"
RAY_ADDRESS="${HEAD_NODE_IP}:${RAY_PORT}"

for node in "${RAY_NODES[@@]}"; do
  srun --export=ALL,TMPDIR=/tmp,TMP=/tmp,TEMP=/tmp --nodes=1 --ntasks=1 -w "${node}" \
    mkdir -p "${TMP_ROOT}/ipc" "${TMP_ROOT}/torchinductor" "${TMP_ROOT}/triton"
done

: > "${LOG_PATH}"
echo "Starting Ray head on ${HEAD_NODE} (${HEAD_NODE_IP}):${RAY_PORT}" | tee -a "${LOG_PATH}"
srun --nodes=1 --ntasks=1 -w "${HEAD_NODE}" \
  "${REMOTE_VLLM_PYTHON_BIN}" -m ray.scripts.scripts start \
    --head \
    --node-ip-address="${HEAD_NODE_IP}" \
    --port="${RAY_PORT}" \
    --num-gpus="${REMOTE_VLLM_NUM_GPUS}" \
    --block \
  >> "${OUT_DIR}/ray-head.log" 2>&1 &
RAY_PIDS+=("$!")
sleep 10

for node in "${RAY_NODES[@@]:1}"; do
  node_ip="$(resolve_node_ip "${node}")"
  echo "Starting Ray worker on ${node} (${node_ip})" | tee -a "${LOG_PATH}"
  srun --nodes=1 --ntasks=1 -w "${node}" \
    "${REMOTE_VLLM_PYTHON_BIN}" -m ray.scripts.scripts start \
      --address="${RAY_ADDRESS}" \
      --node-ip-address="${node_ip}" \
      --num-gpus="${REMOTE_VLLM_NUM_GPUS}" \
      --block \
    >> "${OUT_DIR}/ray-worker-${node}.log" 2>&1 &
  RAY_PIDS+=("$!")
done

echo "Waiting for Ray cluster at ${RAY_ADDRESS}" | tee -a "${LOG_PATH}"
for _ in $(seq 1 60); do
  if "${REMOTE_VLLM_PYTHON_BIN}" -m ray.scripts.scripts status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
    break
  fi
  sleep 5
done
"${REMOTE_VLLM_PYTHON_BIN}" -m ray.scripts.scripts status --address="${RAY_ADDRESS}" >> "${LOG_PATH}" 2>&1

if [ -n "${REQUESTED_HEAD_NODE_VLLM_API_PORT}" ]; then
  HEAD_NODE_VLLM_API_PORT="${REQUESTED_HEAD_NODE_VLLM_API_PORT}"
else
  HEAD_NODE_VLLM_API_PORT="$(select_free_tcp_port)"
fi
write_state_file
echo "Running distributed remote vLLM on ${HEAD_NODE}:${HEAD_NODE_VLLM_API_PORT}"
echo "MODEL_ID = ${MODEL_ID}"
"${REMOTE_VLLM_PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_ID}" \
  --trust-remote-code \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --host 0.0.0.0 \
  --port "${HEAD_NODE_VLLM_API_PORT}" \
@api_key_line  "${VLLM_OPTIONS[@@]}" \
  "${EXTRA_ARGS[@@]}" \
  >> "${LOG_PATH}" 2>&1
