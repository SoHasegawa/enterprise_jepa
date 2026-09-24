#!/usr/bin/env bash

# Shared helpers for Slurm jobs that serve vLLM inside the GPU allocation.
# The caller must define WORK_ROOT and the CLUSTER_VLLM_* / CLUSTER_LLM_* env vars.

VLLM_PID="${VLLM_PID:-}"

visible_gpu_count() {
  local devices="${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}"
  if [[ -n "${devices}" ]]; then
    local IFS=',' count=0 device
    for device in ${devices}; do
      device="${device// /}"
      [[ -n "${device}" ]] && count=$((count + 1))
    done
    echo "${count}"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l
    return 0
  fi
  echo 1
}

resolve_vllm_bin() {
  if [[ -n "${CLUSTER_VLLM_BIN:-}" ]]; then
    [[ -x "${CLUSTER_VLLM_BIN}" ]] || {
      echo "CLUSTER_VLLM_BIN is not executable: ${CLUSTER_VLLM_BIN}" >&2
      return 1
    }
    echo "${CLUSTER_VLLM_BIN}"
    return 0
  fi

  local venv candidate
  for venv in "${CLUSTER_VLLM_VENV:-}" "${VLLM_VENV:-}"; do
    [[ -n "${venv}" ]] || continue
    if [[ -x "${venv}/bin/vllm" ]]; then
      echo "${venv}/bin/vllm"
      return 0
    fi
  done

  if [[ -n "${CLUSTER_VLLM_VENV_CANDIDATES:-}" ]]; then
    local IFS=':'
    for candidate in ${CLUSTER_VLLM_VENV_CANDIDATES}; do
      [[ -n "${candidate}" ]] || continue
      if [[ -x "${candidate}/bin/vllm" ]]; then
        echo "${candidate}/bin/vllm"
        return 0
      fi
    done
  fi

  command -v vllm
}

stop_local_vllm() {
  [[ "${CLUSTER_STOP_VLLM_ON_EXIT:-1}" == "1" ]] || return 0
  [[ -n "${VLLM_PID}" ]] || return 0
  if kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[vllm] stopping pid=${VLLM_PID}"
    kill -TERM "${VLLM_PID}" 2>/dev/null || true
  fi
}

wait_for_local_vllm() {
  local deadline="$((SECONDS + CLUSTER_VLLM_READY_TIMEOUT))"
  local models
  while (( SECONDS < deadline )); do
    models="$(curl -s -m 5 "${CLUSTER_LLM_API_BASE%/}/models" 2>/dev/null || true)"
    if grep -q "\"id\":\"${CLUSTER_LLM_MODEL_NAME}\"" <<<"${models}"; then
      echo "[vllm] ready model=${CLUSTER_LLM_MODEL_NAME} endpoint=${CLUSTER_LLM_API_BASE}"
      return 0
    fi
    if [[ -n "${VLLM_PID}" ]] && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "[vllm] process exited before readiness; last log lines:" >&2
      tail -40 "${WORK_ROOT}/vllm/vllm.log" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "[vllm] timed out waiting for ${CLUSTER_LLM_API_BASE}/models; last log lines:" >&2
  tail -40 "${WORK_ROOT}/vllm/vllm.log" >&2 || true
  return 1
}

check_vllm_chat_completion() {
  python3 - "${CLUSTER_LLM_API_BASE%/}" "${CLUSTER_LLM_MODEL_NAME}" <<'PY_VLLM_CHAT'
import json
import sys
import urllib.error
import urllib.request

base_url = sys.argv[1].rstrip("/")
model_name = sys.argv[2]
payload = {
    "model": model_name,
    "messages": [{"role": "user", "content": "Reply with OK."}],
    "max_tokens": 4,
    "temperature": 0,
}
request = urllib.request.Request(
    f"{base_url}/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read(4096).decode("utf-8", errors="replace")
        if response.status != 200:
            print(f"chat preflight returned HTTP {response.status}: {body}", file=sys.stderr)
            sys.exit(1)
        data = json.loads(body)
        if not data.get("choices"):
            print(f"chat preflight returned no choices: {body}", file=sys.stderr)
            sys.exit(1)
except urllib.error.HTTPError as exc:
    body = exc.read(4096).decode("utf-8", errors="replace")
    print(f"chat preflight HTTP error {exc.code}: {body}", file=sys.stderr)
    sys.exit(1)
except Exception as exc:
    print(f"chat preflight connection error: {exc}", file=sys.stderr)
    sys.exit(1)
PY_VLLM_CHAT
}

wait_for_vllm_chat_completion() {
  local deadline="$((SECONDS + CLUSTER_VLLM_CHAT_READY_TIMEOUT))"
  local last_error="${WORK_ROOT}/vllm-chat-preflight.err"
  : >"${last_error}"
  while (( SECONDS < deadline )); do
    if check_vllm_chat_completion 2>"${last_error}"; then
      echo "[vllm] chat completions ready endpoint=${CLUSTER_LLM_API_BASE}"
      return 0
    fi
    if [[ -n "${VLLM_PID}" ]] && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "[vllm] process exited before chat preflight; last log lines:" >&2
      tail -40 "${WORK_ROOT}/vllm/vllm.log" >&2 || true
      return 1
    fi
    sleep 5
  done
  echo "[vllm] timed out waiting for ${CLUSTER_LLM_API_BASE}/chat/completions; last preflight error:" >&2
  cat "${last_error}" >&2 || true
  if [[ -n "${VLLM_PID}" ]]; then
    echo "[vllm] last local vLLM log lines:" >&2
    tail -40 "${WORK_ROOT}/vllm/vllm.log" >&2 || true
  fi
  return 1
}

start_local_vllm() {
  local vllm_bin tp_size vllm_dir
  vllm_bin="$(resolve_vllm_bin)" || {
    echo "vllm command not found; set CLUSTER_VLLM_BIN or CLUSTER_VLLM_VENV" >&2
    return 1
  }
  export PATH="$(dirname "${vllm_bin}"):${PATH}"
  tp_size="${CLUSTER_VLLM_TENSOR_PARALLEL_SIZE:-$(visible_gpu_count)}"
  vllm_dir="${WORK_ROOT}/vllm"
  mkdir -p "${vllm_dir}"
  : >"${vllm_dir}/vllm.log"
  echo "[vllm] starting ${CLUSTER_VLLM_MODEL_ID} as ${CLUSTER_LLM_MODEL_NAME}"
  echo "[vllm] bin=${vllm_bin}"
  echo "[vllm] port=${CLUSTER_VLLM_PORT} tensor_parallel_size=${tp_size} log=${vllm_dir}/vllm.log"
  "${vllm_bin}" serve "${CLUSTER_VLLM_MODEL_ID}" \
    --host 0.0.0.0 \
    --port "${CLUSTER_VLLM_PORT}" \
    --tensor-parallel-size "${tp_size}" \
    --served-model-name "${CLUSTER_LLM_MODEL_NAME}" \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser "${CLUSTER_VLLM_TOOL_CALL_PARSER}" \
    --language-model-only \
    --gdn-prefill-backend triton \
    --max-model-len "${CLUSTER_VLLM_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${CLUSTER_VLLM_GPU_MEMORY_UTILIZATION}" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    >>"${vllm_dir}/vllm.log" 2>&1 &
  VLLM_PID="$!"
  echo "${VLLM_PID}" >"${vllm_dir}/vllm.pid"
  wait_for_local_vllm
}

clear_stale_mcp_listeners() {
  echo "[slurm] clearing stale MCP listeners on expected ports"
  for port in 8001 8002 8003 8004 8006 8008 8009; do
    if command -v fuser >/dev/null 2>&1; then
      fuser -k -TERM "${port}/tcp" >/dev/null 2>&1 || true
    fi
  done
  sleep 2
  for port in 8001 8002 8003 8004 8006 8008 8009; do
    if command -v fuser >/dev/null 2>&1; then
      fuser -k -KILL "${port}/tcp" >/dev/null 2>&1 || true
    fi
  done
  echo "[slurm] stale MCP listener cleanup complete"
}
