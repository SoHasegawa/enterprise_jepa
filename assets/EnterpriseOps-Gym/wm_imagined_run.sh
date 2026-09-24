#!/usr/bin/env bash
# EnterpriseOps-Gym + generative Enterprise World Model (EWM), imagined strategy.
#
# One-command runner for WM_STRATEGY=imagined. It serves only the **EWM** vLLM
# (the one thing run.sh doesn't know about), exports the imagined-mode env, then
# **delegates everything else to run.sh** — which serves the AGENT vLLM via its
# tested per-model serve script (correct tool-call/reasoning parsers, thinking
# preset), starts the MCP tool containers, runs `ejepa bench run`, and tears down.
#
#   [A] EWM vLLM : Qwen/Qwen3.6-27B + LoRA (gymops_world_model) on :9000.
#   [B] run.sh   : serves the AGENT (Nemotron Super NVFP4, thinking) on :8020 via
#                  its MODEL preset, starts MCP gyms, runs ejepa (executor=mcp_react),
#                  inheriting the WM_STRATEGY=imagined env we export.
#   [C] stop the EWM vLLM (run.sh stops its own agent vLLM + MCP).
#
# Agent serving is reused from run.sh (MODEL preset + serve scripts), so the
# Nemotron tool-call/reasoning-parser/TP flags come from the tested baseline,
# not hand-maintained here.
#
# Usage:
#   EWM_LORA=/abs/path/to/sessions/gymops_binary_new/checkpoint-6900 \
#   AGENT_GPUS=0,1,2,3 EWM_GPUS=4 \
#     bash assets/EnterpriseOps-Gym/wm_imagined_run.sh
#
#   # reuse an already-running EWM endpoint (skip [A]):
#   START_EWM_VLLM=0 WM_VLLM_BASE_URL=http://127.0.0.1:9000/v1 \
#     bash assets/EnterpriseOps-Gym/wm_imagined_run.sh
#
# Imagined-strategy knobs (see wm_react.py / wm_ewm.py): WM_STATE, ACTION_OPTIMIZER,
# K_CONTROLLER, WM_IMAGINED_MAX_STEPS/TOP_K/CANDIDATE_ACTIONS, WM_STATE_HISTORY_SIZE.
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH_DIR="$ROOT/assets/EnterpriseOps-Gym"
cd "$ROOT"

export BENCHMARK_HOME="${BENCHMARK_HOME:-$ROOT/.cache/benchmark home}"
export BENCHMARK_A2A_CLIENT_TIMEOUT="${BENCHMARK_A2A_CLIENT_TIMEOUT:-3600}"

# ---- run.sh delegation (agent + MCP + ejepa) --------------------------------
RUN_MODE="${RUN_MODE:-eval-opsgym-80}"           # → target=opsgym_80_test
AGENT_PRESET="${AGENT_PRESET:-super-think}"      # Nemotron Super NVFP4, thinking ON (run.sh preset)
AGENT_GPUS="${AGENT_GPUS:-0,1,2,3}"              # CUDA_VISIBLE_DEVICES for run.sh's agent vLLM
MAX_ITERATIONS="${MAX_ITERATIONS:-50}"
MAX_PARALLEL="${MAX_PARALLEL:-10}"

# ---- imagined strategy knobs ----------------------------------------------
WM_STATE="${WM_STATE:-binary_error}"
ACTION_OPTIMIZER="${ACTION_OPTIMIZER:-}"         # ""/none → plain greedy imagine (matches evaluation.py binary2: num_rollouts=1+llm_judge→single temp-0 rollout); topk_search → beam (overshoots binary2)
K_CONTROLLER="${K_CONTROLLER:-}"                 # ""/none | react_wm_decide_k | react_wm_rl_k
WM_IMAGINED_MAX_STEPS="${WM_IMAGINED_MAX_STEPS:-6}"   # binary2 used --imagined-trajectory-max-steps 6
WM_IMAGINED_CANDIDATE_ACTIONS="${WM_IMAGINED_CANDIDATE_ACTIONS:-3}"
WM_IMAGINED_TOP_K="${WM_IMAGINED_TOP_K:-3}"
WM_IMAGINED_TEMPERATURE="${WM_IMAGINED_TEMPERATURE:-0.7}"
WM_STATE_HISTORY_SIZE="${WM_STATE_HISTORY_SIZE:-3}"
WM_EWM_MAX_NEW_TOKENS="${WM_EWM_MAX_NEW_TOKENS:-4096}"

# ---- EWM vLLM (Qwen3.6-27B base + gymops_world_model LoRA) ----------------
START_EWM_VLLM="${START_EWM_VLLM:-1}"
# Resolve the vllm binary: explicit EWM_VLLM_BIN > $VLLM_VENV/bin/vllm > PATH.
# (VLLM_VENV is also exported below so run.sh's agent serve finds vllm too.)
VLLM_VENV="${VLLM_VENV:-}"
if [ -n "${EWM_VLLM_BIN:-}" ]; then
  :
elif [ -n "$VLLM_VENV" ] && [ -x "$VLLM_VENV/bin/vllm" ]; then
  EWM_VLLM_BIN="$VLLM_VENV/bin/vllm"
elif command -v vllm >/dev/null 2>&1; then
  EWM_VLLM_BIN="$(command -v vllm)"
else
  EWM_VLLM_BIN="vllm"
fi
# Put the vLLM venv's bin/ on PATH so the `ninja` console script (shipped by the
# `ninja` pip pkg) is found — Qwen3.6 GDN linear-attention JIT-compiles a kernel
# via FlashInfer's subprocess('ninja'); without it on PATH, the engine crashes
# (EngineCore FileNotFoundError: 'ninja' → Connection refused/error). This PATH
# is inherited by run.sh's agent serve too.
_vllm_bin_dir=""
if [ -n "$VLLM_VENV" ] && [ -d "$VLLM_VENV/bin" ]; then
  _vllm_bin_dir="$VLLM_VENV/bin"
elif [ -x "$EWM_VLLM_BIN" ]; then
  _vllm_bin_dir="$(cd "$(dirname "$EWM_VLLM_BIN")" && pwd)"
fi
if [ -n "$_vllm_bin_dir" ]; then export PATH="$_vllm_bin_dir:$PATH"; fi
command -v ninja >/dev/null 2>&1 || \
  echo "[wm_imagined_run] WARN 'ninja' not on PATH — FlashInfer JIT (Qwen3.6 GDN) will crash the vLLM engine. Install it: ${_vllm_bin_dir%/bin}/bin/pip install ninja, or apt install ninja-build." >&2
EWM_BASE_MODEL="${EWM_BASE_MODEL:-Qwen/Qwen3.6-27B}"
WM_EWM_MODEL="${WM_EWM_MODEL:-gymops_world_model}"          # LoRA module name == client model id
EWM_LORA="${EWM_LORA:-sessions/gymops_binary_new/checkpoint-6900}"
EWM_HOST="${EWM_HOST:-127.0.0.1}"
WM_VLLM_SERVER_PORT="${WM_VLLM_SERVER_PORT:-9000}"
EWM_GPUS="${EWM_GPUS:-4}"                          # disjoint from AGENT_GPUS
EWM_TP="${EWM_TP:-1}"
EWM_MAX_MODEL_LEN="${EWM_MAX_MODEL_LEN:-32768}"
EWM_GPU_UTIL="${EWM_GPU_UTIL:-0.92}"
EWM_MAX_NUM_SEQS="${EWM_MAX_NUM_SEQS:-256}"
EWM_MAX_NUM_BATCHED="${EWM_MAX_NUM_BATCHED:-8192}"
EWM_VLLM_EXTRA_ARGS="${EWM_VLLM_EXTRA_ARGS:-}"

WM_VLLM_BASE_URL="${WM_VLLM_BASE_URL:-http://${EWM_HOST}:${WM_VLLM_SERVER_PORT}/v1}"

READY_TIMEOUT="${EWM_READY_TIMEOUT:-1800}"
LOG_DIR="${WM_LOG_DIR:-$BENCH_DIR/.wm_imagined_logs}"
mkdir -p "$LOG_DIR"
EWM_LOG="$LOG_DIR/ewm_vllm_$(date -u +%Y%m%dT%H%M%SZ).log"

log() { echo "[wm_imagined_run] $*"; }

echo "=== wm_imagined_run ==="
echo "agent      : run.sh preset MODEL=$AGENT_PRESET (gpus=$AGENT_GPUS) mode=$RUN_MODE"
echo "ewm        : $EWM_BASE_MODEL + LoRA '$WM_EWM_MODEL' ($EWM_LORA) @ $WM_VLLM_BASE_URL (gpus=$EWM_GPUS tp=$EWM_TP start=$START_EWM_VLLM)"
echo "imagined   : wm_state=$WM_STATE optimizer=${ACTION_OPTIMIZER:-none} k_controller=${K_CONTROLLER:-static} max_steps=$WM_IMAGINED_MAX_STEPS"
echo "ejepa        : executor=mcp_react max_iterations=$MAX_ITERATIONS max_parallel=$MAX_PARALLEL"

EWM_PID=""
stop_ewm() {
  if [ -n "$EWM_PID" ]; then
    log "stopping EWM vLLM (pid=$EWM_PID)"
    kill "$EWM_PID" 2>/dev/null || true
    wait "$EWM_PID" 2>/dev/null || true
    EWM_PID=""
  fi
}
trap stop_ewm EXIT INT TERM

wait_for_vllm() {  # base_url served_name pid log
  local base="${1%/}" want="$2" pid="$3" logf="$4" models
  for ((i = 0; i < READY_TIMEOUT / 5; i++)); do
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "EWM vLLM exited during startup:" >&2; tail -40 "$logf" >&2; return 1
    fi
    models="$(curl -s -m 5 "$base/models" 2>/dev/null || true)"
    if echo "$models" | grep -q "\"$want\""; then return 0; fi
    sleep 5
  done
  echo "EWM vLLM not ready in ${READY_TIMEOUT}s:" >&2; tail -40 "$logf" >&2; return 1
}

# ---- [A] start EWM vLLM ---------------------------------------------------
if [ "$START_EWM_VLLM" = 1 ]; then
  if ! command -v "$EWM_VLLM_BIN" >/dev/null 2>&1 && [ ! -x "$EWM_VLLM_BIN" ]; then
    echo "vllm binary not found ('$EWM_VLLM_BIN'). Set VLLM_VENV=/path/to/venv (uses" >&2
    echo "  \$VLLM_VENV/bin/vllm) or EWM_VLLM_BIN=/abs/path/to/vllm, or activate the venv." >&2
    exit 1
  fi
  [ -e "$EWM_LORA" ] || log "WARN EWM_LORA path not found from $(pwd): $EWM_LORA (set an absolute path)"
  log "[A] serving EWM ($EWM_VLLM_BIN) → $EWM_LOG"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$EWM_GPUS" nohup "$EWM_VLLM_BIN" serve "$EWM_BASE_MODEL" \
    --host 0.0.0.0 --port "$WM_VLLM_SERVER_PORT" --trust-remote-code \
    --enable-lora --lora-modules "${WM_EWM_MODEL}=${EWM_LORA}" \
    --tensor-parallel-size "$EWM_TP" \
    --max-model-len "$EWM_MAX_MODEL_LEN" \
    --reasoning-parser qwen3 --language-model-only \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --kv-cache-dtype fp8 --gpu-memory-utilization "$EWM_GPU_UTIL" \
    --enable-prefix-caching --enable-chunked-prefill \
    --max-num-seqs "$EWM_MAX_NUM_SEQS" --max-num-batched-tokens "$EWM_MAX_NUM_BATCHED" \
    $EWM_VLLM_EXTRA_ARGS >"$EWM_LOG" 2>&1 &
  EWM_PID=$!
  if wait_for_vllm "$WM_VLLM_BASE_URL" "$WM_EWM_MODEL" "$EWM_PID" "$EWM_LOG"; then
    log "EWM ready: $WM_EWM_MODEL @ $WM_VLLM_BASE_URL"
  else
    echo "EWM vLLM failed to become ready — aborting (would otherwise run with no world model)." >&2
    echo "  See $EWM_LOG (common cause: missing 'ninja' for FlashInfer JIT)." >&2
    exit 1
  fi
else
  curl -s -m 5 "${WM_VLLM_BASE_URL%/}/models" 2>/dev/null | grep -q "\"$WM_EWM_MODEL\"" \
    && log "reusing EWM endpoint: $WM_EWM_MODEL @ $WM_VLLM_BASE_URL" \
    || log "WARN EWM model '$WM_EWM_MODEL' not visible at ${WM_VLLM_BASE_URL%/}/models"
fi

# ---- imagined env (inherited by run.sh → ejepa → purple agent) --------------
# run.sh's configure_llm sets the AGENT (policy) LLM and does NOT touch these,
# so the orchestrator + WM endpoint pass straight through to the executor.
export ENTERPRISEOPS_ORCHESTRATOR=wm_react
export WM_STRATEGY=imagined
export WM_STATE ACTION_OPTIMIZER K_CONTROLLER
export WM_IMAGINED_MAX_STEPS WM_IMAGINED_CANDIDATE_ACTIONS WM_IMAGINED_TOP_K
export WM_IMAGINED_TEMPERATURE WM_STATE_HISTORY_SIZE WM_EWM_MAX_NEW_TOKENS
export WM_EWM_MODEL WM_VLLM_BASE_URL WM_VLLM_SERVER_PORT
export WM_VLLM_API_KEY="${WM_VLLM_API_KEY:-not-needed}"

# ---- [B] delegate agent serving + MCP + ejepa to run.sh ---------------------
log "[B] run.sh $RUN_MODE (MODEL=$AGENT_PRESET EXECUTOR=mcp_react) — serves agent + MCP + ejepa"
# VLLM_VENV (if set) is exported so run.sh's agent serve scripts inherit it.
[ -n "$VLLM_VENV" ] && export VLLM_VENV
MODEL="$AGENT_PRESET" \
EXECUTOR=mcp_react \
MAX_ITERATIONS="$MAX_ITERATIONS" \
MAX_PARALLEL="$MAX_PARALLEL" \
CUDA_VISIBLE_DEVICES="$AGENT_GPUS" \
BENCHMARK_A2A_CLIENT_TIMEOUT="$BENCHMARK_A2A_CLIENT_TIMEOUT" \
  bash "$BENCH_DIR/run.sh" "$RUN_MODE"

# ---- [C] stop EWM (run.sh stops its own agent vLLM + MCP) ------------------
stop_ewm
log "done. EWM log: $EWM_LOG"
