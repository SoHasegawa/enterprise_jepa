#!/usr/bin/env bash
@sbatch_directives

set -euo pipefail
OUT_DIR=@out_dir
MODEL_ID=@model
SERVED_MODEL_NAME=@served_model_name
REQUESTED_REMOTE_VLLM_API_PORT=@remote_port
REMOTE_INFERENCE_STATE_PATH=@state_path
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
  "${REMOTE_VLLM_PYTHON_BIN}" - "$REMOTE_INFERENCE_STATE_PATH" "$REMOTE_VLLM_API_PORT" <<'PY'
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
cleanup_runtime_tmp() {
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
  if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$ROCR_VISIBLE_DEVICES}"
    unset ROCR_VISIBLE_DEVICES
  fi
fi
@setup_cmd_block
: > "${LOG_PATH}"
if [ -n "${REQUESTED_REMOTE_VLLM_API_PORT}" ]; then
  REMOTE_VLLM_API_PORT="${REQUESTED_REMOTE_VLLM_API_PORT}"
else
  REMOTE_VLLM_API_PORT="$(select_free_tcp_port)"
fi
write_state_file
echo "Running remote vLLM on $(hostname):${REMOTE_VLLM_API_PORT}"
echo "MODEL_ID = ${MODEL_ID}"
"${REMOTE_VLLM_PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL_ID}" \
  --trust-remote-code \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --host 0.0.0.0 \
  --port "${REMOTE_VLLM_API_PORT}" \
@api_key_line  "${VLLM_OPTIONS[@@]}" \
  "${EXTRA_ARGS[@@]}" \
  >> "${LOG_PATH}" 2>&1
