#!/usr/bin/env bash

#kill "$(cat build/vllm-qwen3.6-27b/vllm.pid)" 2>/dev/null || true
START_VLLM=0
STOP_VLLM_ON_EXIT=1

MAX_PARALLEL=20
CAPTURE_TRAJECTORY=true

ENTERPRISEOPS_LLM_TEMPERATURE=0.7
ENTERPRISEOPS_LLM_TOP_P=0.95 

HF_MODE=oracle # plus_15_tools

# MAX_TASKS_PER_DOMAIN=10  bash assets/EnterpriseOps-Gym/run.sh eval
VLLM_GPU_MEMORY_UTILIZATION=1 
CUDA_VISIBLE_DEVICES=1,2,3,4
VLLM_MODEL=qwen3.6-27b
DOMAINS='["calendar","csm","drive","email","hr","itsm","teams", "hybrid"]'
MAX_TASKS_PER_DOMAIN=null

# Run EnterpriseOps-Gym end-to-end with local vLLM and print evaluation results.
#
# Usage:
#   bash assets/EnterpriseOps-Gym/run.sh              # smoke: one bundled calendar task
#   bash assets/EnterpriseOps-Gym/run.sh eval         # HF corpus (see env vars below)
#   bash assets/EnterpriseOps-Gym/run.sh eval-opsgym-80  # fixed 80-task opsgym held-out split
#   bash assets/EnterpriseOps-Gym/run.sh smoke        # explicit smoke
#
# Local vLLM (default) uses per-model serve scripts under $EWM_VLLM_SCRIPTS (set it to
# your own launcher directory), or set START_VLLM=0 and point the run at a served endpoint.
#
#   VLLM_MODEL=nemotron-3-nano-30b-a3b bash assets/EnterpriseOps-Gym/run.sh smoke
#   VLLM_MODEL=qwen3.6-27b bash assets/EnterpriseOps-Gym/run.sh smoke
#
# VLLM_MODEL choices (must support tool calling for mcp_react):
#   nemotron-3-nano-30b-a3b   (default; nemo3_serve.sh)
#   nemotron-3-super-120b-a12b
#   nemotron-3-super-120b-a12b-nvfp4
#   gemma4-26b-a4b-it
#   command-a-plus
#   qwen3.6-27b
#
# Common vLLM env overrides:
#   VLLM_VENV=/path/to/vllm/venv
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#   VLLM_TENSOR_PARALLEL_SIZE=2   # auto-set from CUDA_VISIBLE_DEVICES if unset
#   LOCAL_VLLM_BASE=http://127.0.0.1:8020/v1
#   START_VLLM=0                # vLLM already running; only health-check
#   STOP_VLLM_ON_EXIT=1         # tear down :8020 when the script exits
#
# Eval-mode knobs (optional):
#   DOMAIN=calendar              # or DOMAINS='["calendar","teams"]'
#   HF_MODE=oracle               # oracle | plus_5_tools | plus_10_tools | plus_15_tools
#   MAX_TASKS_PER_DOMAIN=5
#   MAX_ITERATIONS=50            # eval-opsgym-80 default; upstream leaderboard uses 50
#   TASK_ID=task_...             # restrict to one HF task
#   MAX_PARALLEL=1
#   CAPTURE_TRAJECTORY=true
#
# eval-opsgym-80 runs target=opsgym_80_test (80 IDs in green/tasks/task_ids.toml).
# Task list matches ewm-enterprisearena trajectories/enterpriseops_gym_80_test_task_split.json.
#   bash assets/EnterpriseOps-Gym/run.sh eval-opsgym-80
#   OPSGYM_HF_MODE=plus_15_tools bash assets/EnterpriseOps-Gym/run.sh eval-opsgym-80
#   TASK_ID=task_20251201_113908_295_e81f6083_e770442c bash assets/EnterpriseOps-Gym/run.sh eval-opsgym-80
#
# Alternate LLM backends (set USE_LOCAL_VLLM=0):
#   export ENTERPRISEOPS_LLM_PROVIDER=openai
#   export ENTERPRISEOPS_LLM_MODEL=gpt-4.1-mini
#   export ENTERPRISEOPS_LLM_API_KEY="sk-..."
#
# Remote vLLM via ejepa inference config:
#   USE_LOCAL_VLLM=0 INFERENCE_CONFIG=inference-slurm-login.yaml bash assets/EnterpriseOps-Gym/run.sh smoke
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BENCH_DIR="$ROOT/assets/EnterpriseOps-Gym"
# Directory holding this site's per-model vLLM serve scripts (serve-<model>.sh). Only
# needed when this script serves the agent itself (START_VLLM=1); point it at your own
# launcher directory, or serve the model separately and pass its endpoint instead.
EWM_VLLM_SCRIPTS="${EWM_VLLM_SCRIPTS:-$ROOT/vllm-serve-scripts}"
cd "$ROOT"

MODE="${1:-smoke}"
if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  sed -n '2,55p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi
if [[ "$MODE" != "smoke" && "$MODE" != "eval" && "$MODE" != "eval-opsgym-80" ]]; then
  echo "Unknown mode: $MODE (expected smoke, eval, or eval-opsgym-80)" >&2
  exit 1
fi

# --- paths & defaults -------------------------------------------------------
unset HF_ENDPOINT 2>/dev/null || true

export ENTERPRISEOPS_GYM_REPO_PATH="${ENTERPRISEOPS_GYM_REPO_PATH:-$ROOT/.cache/EnterpriseOps-Gym}"
export BENCHMARK_HOME="${BENCHMARK_HOME:-$ROOT/.cache/benchmark home}"
export BENCHMARK_A2A_CLIENT_TIMEOUT="${BENCHMARK_A2A_CLIENT_TIMEOUT:-1800}"

EJEPA="${EJEPA:-$ROOT/.venv/bin/ejepa}"
EXECUTOR="${EXECUTOR:-mcp_react}"
SKIP_SETUP="${SKIP_SETUP:-0}"
START_MCP="${START_MCP:-1}"
START_VLLM="${START_VLLM:-1}"
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"
USE_LOCAL_VLLM="${USE_LOCAL_VLLM:-1}"
READY_TIMEOUT="${READY_TIMEOUT:-600}"
SHOW_LOGS="${SHOW_LOGS:-1}"

VLLM_MODEL="${VLLM_MODEL:-nemotron-3-nano-30b-a3b}"
LOCAL_VLLM_BASE="${LOCAL_VLLM_BASE:-http://127.0.0.1:8020/v1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

HF_MODE="${HF_MODE:-oracle}"
DOMAIN="${DOMAIN:-calendar}"
MAX_TASKS_PER_DOMAIN="${MAX_TASKS_PER_DOMAIN:-5}"
MAX_ITERATIONS="${MAX_ITERATIONS:-50}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
CAPTURE_TRAJECTORY="${CAPTURE_TRAJECTORY:-false}"
TASK_ID="${TASK_ID:-}"

LOG_DIR="${LOG_DIR:-$BENCHMARK_HOME/logs}"
mkdir -p "$LOG_DIR" "$BENCHMARK_HOME"
LOG_FILE="${LOG_FILE:-$LOG_DIR/enterpriseops_gym_${MODE}_$(date -u +%Y%m%dT%H%M%SZ).log}"

CURRENT_VLLM_MODEL=""

# Domain -> host port (for health checks / container startup)
declare -A DOMAIN_PORT=(
  [calendar]=8003
  [teams]=8002
  [csm]=8001
  [email]=8004
  [itsm]=8006
  [hr]=8008
  [drive]=8009
)
declare -A DOMAIN_IMAGE=(
  [calendar]=shivakrishnareddyma225/enterpriseops-gym-mcp-calendar:latest
  [teams]=shivakrishnareddyma225/enterpriseops-gym-mcp-teams:latest
  [csm]=shivakrishnareddyma225/enterpriseops-gym-mcp-csm:latest
  [email]=shivakrishnareddyma225/enterpriseops-gym-mcp-email:latest
  [itsm]=shivakrishnareddyma225/enterpriseops-gym-mcp-itsm:latest
  [hr]=shivakrishnareddyma225/enterpriseops-gym-mcp-hr:latest
  [drive]=shivakrishnareddyma225/enterpriseops-gym-mcp-drive:latest
)
declare -A DOMAIN_CONTAINER=(
  [calendar]=gym-calendar
  [teams]=gym-teams
  [csm]=gym-csm
  [email]=gym-email
  [itsm]=gym-itsm
  [hr]=gym-hr
  [drive]=gym-drive
)
declare -A DOMAIN_PUBLISH=(
  [calendar]=8003:8003
  [teams]=8002:8005
  [csm]=8001:8005
  [email]=8004:8005
  [itsm]=8006:8005
  [hr]=8008:8005
  [drive]=8009:8005
)
# HF split "hybrid" has no dedicated MCP container; tasks use multiple domain gyms.
MCP_DOMAINS=(calendar csm drive email hr itsm teams)

log() { printf '%s\n' "$*" | tee -a "$LOG_FILE"; }

parse_domains_list() {
  local -a domains=()
  if [[ "$MODE" == "smoke" ]]; then
    domains=(calendar)
  elif [[ "$MODE" == "eval-opsgym-80" ]]; then
    domains=(calendar csm drive email hr hybrid itsm teams)
  elif [[ -n "${DOMAINS:-}" ]]; then
    local raw="$DOMAINS"
    raw="${raw//[\[\]\"\' ]/}"
    IFS=',' read -r -a domains <<< "$raw"
  else
    domains=("$DOMAIN")
  fi
  printf '%s\n' "${domains[@]}"
}

resolve_mcp_domains() {
  # Map requested HF splits to MCP containers. "hybrid" requires every domain gym.
  local -a requested=("$@")
  local -a mcp=()
  local seen="" d
  for d in "${requested[@]}"; do
    if [[ "$d" == "hybrid" ]]; then
      local h
      for h in "${MCP_DOMAINS[@]}"; do
        if [[ ",$seen," != *",$h,"* ]]; then
          mcp+=("$h")
          seen="${seen},$h"
        fi
      done
    elif [[ -n "${DOMAIN_PORT[$d]:-}" ]]; then
      if [[ ",$seen," != *",$d,"* ]]; then
        mcp+=("$d")
        seen="${seen},$d"
      fi
    else
      echo "Unknown domain: $d (supported: calendar csm drive email hr hybrid itsm teams)" >&2
      return 1
    fi
  done
  printf '%s\n' "${mcp[@]}"
}

visible_gpu_count() {
  local devices="${CUDA_VISIBLE_DEVICES:-}"
  if [[ -z "$devices" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l
      return 0
    fi
    echo 1
    return 0
  fi
  local IFS=',' count=0 part
  for part in $devices; do
    part="${part// /}"
    [[ -n "$part" ]] && count=$((count + 1))
  done
  echo "$count"
}

ensure_vllm_tensor_parallel_size() {
  if [[ -n "${VLLM_TENSOR_PARALLEL_SIZE:-}" ]]; then
    return 0
  fi
  export VLLM_TENSOR_PARALLEL_SIZE="$(visible_gpu_count)"
  log "VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE} (from visible GPUs; override if needed)"
}

vllm_log_has_fatal_error() {
  local log_path="$1"
  [[ -f "$log_path" ]] || return 1
  grep -Eq 'Traceback|ValidationError|CUDA out of memory|No available memory|Address already in use|World size \([0-9]+\) is larger than the number of available GPUs' \
    "$log_path"
}

vllm_script_for_model() {
  case "$1" in
    qwen3.6-27b|local-qwen3.6-27b|local)
      echo "$EWM_VLLM_SCRIPTS/serve-qwen3.6-27b.sh"
      ;;
    gemma4-26b-a4b-it|local-gemma4-26b-a4b-it)
      echo "$EWM_VLLM_SCRIPTS/serve-gemma4-26b-a4b-it.sh"
      ;;
    command-a-plus|local-command-a-plus)
      echo "$EWM_VLLM_SCRIPTS/serve-command-a-plus-w4a4.sh"
      ;;
    nemotron-3-nano-30b-a3b|local-nemotron-3-nano-30b-a3b|nemo3)
      echo "$EWM_VLLM_SCRIPTS/nemo3_serve.sh"
      ;;
    nemotron-3-super-120b-a12b|local-nemotron-3-super-120b-a12b)
      echo "$EWM_VLLM_SCRIPTS/serve-nemotron-3-super-120b-a12b-bf16.sh"
      ;;
    nemotron-3-super-120b-a12b-nvfp4|local-nemotron-3-super-120b-a12b-nvfp4|nemo3-super-nvfp4)
      echo "$EWM_VLLM_SCRIPTS/serve-nemotron-3-super-120b-a12b-nvfp4.sh"
      ;;
    *)
      return 1
      ;;
  esac
}

vllm_runtime_dir_for_model() {
  case "$1" in
    qwen3.6-27b|local-qwen3.6-27b|local) echo "$ROOT/build/vllm-qwen3.6-27b" ;;
    gemma4-26b-a4b-it|local-gemma4-26b-a4b-it) echo "$ROOT/build/vllm-gemma4" ;;
    command-a-plus|local-command-a-plus) echo "$ROOT/build/vllm-command-a-plus" ;;
    nemotron-3-nano-30b-a3b|local-nemotron-3-nano-30b-a3b|nemo3)
      echo "$ROOT/build/vllm-nemotron3-nano"
      ;;
    nemotron-3-super-120b-a12b|local-nemotron-3-super-120b-a12b)
      echo "$ROOT/build/vllm-nemotron3-super"
      ;;
    nemotron-3-super-120b-a12b-nvfp4|local-nemotron-3-super-120b-a12b-nvfp4|nemo3-super-nvfp4)
      echo "$ROOT/build/vllm-nemotron3-super-nvfp4"
      ;;
    *)
      return 1
      ;;
  esac
}

vllm_served_model_id() {
  case "$1" in
    qwen3.6-27b|local-qwen3.6-27b|local) echo "Qwen3.6-27B" ;;
    gemma4-26b-a4b-it|local-gemma4-26b-a4b-it) echo "google/gemma-4-26B-A4B-it" ;;
    command-a-plus|local-command-a-plus) echo "CohereLabs/command-a-plus-05-2026-w4a4" ;;
    nemotron-3-nano-30b-a3b|local-nemotron-3-nano-30b-a3b|nemo3)
      echo "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
      ;;
    nemotron-3-super-120b-a12b|local-nemotron-3-super-120b-a12b)
      echo "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
      ;;
    nemotron-3-super-120b-a12b-nvfp4|local-nemotron-3-super-120b-a12b-nvfp4|nemo3-super-nvfp4)
      echo "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
      ;;
    *)
      return 1
      ;;
  esac
}

stop_local_vllm() {
  pkill -TERM -f "vllm serve.*--port 8020" 2>/dev/null || true
  pkill -TERM -f "vllm serve.*--port=8020" 2>/dev/null || true
  local i
  for i in $(seq 1 30); do
    curl -s -m 2 "${LOCAL_VLLM_BASE%/}/models" >/dev/null 2>&1 || {
      CURRENT_VLLM_MODEL=""
      return 0
    }
    sleep 2
  done
  pkill -KILL -f "vllm serve.*8020" 2>/dev/null || true
  sleep 3
  CURRENT_VLLM_MODEL=""
}

wait_local_vllm() {
  local want="$1"
  local log_path="${2:-}"
  local pid="${3:-}"
  local i models
  for i in $(seq 1 240); do
    models="$(curl -s -m 5 "${LOCAL_VLLM_BASE%/}/models" 2>/dev/null || true)"
    if echo "$models" | grep -q "\"id\":\"${want}\""; then
      log "vLLM ready: ${want} @ ${LOCAL_VLLM_BASE} (${i}x5s)"
      return 0
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "vLLM process exited before ${want} became ready (pid=$pid)" >&2
      if [[ -n "$log_path" && -f "$log_path" ]]; then
        echo "Last log lines from $log_path:" >&2
        tail -20 "$log_path" >&2
      fi
      return 1
    fi
    if [[ -n "$log_path" ]] && vllm_log_has_fatal_error "$log_path"; then
      echo "vLLM failed during startup; see $log_path" >&2
      tail -20 "$log_path" >&2
      return 1
    fi
    sleep 5
  done
  echo "vLLM did not expose ${want} at ${LOCAL_VLLM_BASE%/}/models within timeout" >&2
  if [[ -n "$log_path" && -f "$log_path" ]]; then
    tail -20 "$log_path" >&2
  fi
  return 1
}

ensure_local_vllm() {
  local model="$1" script runtime want
  want="$(vllm_served_model_id "$model")" || {
    echo "Unknown VLLM_MODEL: $model" >&2
    echo "Supported: nemotron-3-nano-30b-a3b gemma4-26b-a4b-it command-a-plus qwen3.6-27b nemotron-3-super-120b-a12b nemotron-3-super-120b-a12b-nvfp4" >&2
    return 1
  }
  if [[ "$CURRENT_VLLM_MODEL" == "$model" ]]; then
    wait_local_vllm "$want" "$(vllm_runtime_dir_for_model "$model")/vllm.log"
    return 0
  fi
  script="$(vllm_script_for_model "$model")" || {
    echo "No serve script for VLLM_MODEL=$model" >&2
    return 1
  }
  runtime="$(vllm_runtime_dir_for_model "$model")" || {
    echo "No runtime dir for VLLM_MODEL=$model" >&2
    return 1
  }
  if [[ ! -f "$script" ]]; then
    echo "Serve script missing: $script" >&2
    return 1
  fi
  ensure_vllm_tensor_parallel_size
  log "Starting local vLLM on :8020 -> $model ($want)"
  log "  script: $script"
  log "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}  VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE}"
  stop_local_vllm
  mkdir -p "$runtime"
  : >"$runtime/vllm.log"
  nohup bash "$script" >/dev/null 2>&1 &
  local vllm_pid=$!
  echo "$vllm_pid" >"$runtime/vllm.pid"
  wait_local_vllm "$want" "$runtime/vllm.log" "$vllm_pid"
  CURRENT_VLLM_MODEL="$model"
}

configure_vllm_llm_env() {
  local model="$1" served_id
  served_id="$(vllm_served_model_id "$model")"

  unset ENTERPRISEOPS_LLM_API_ENDPOINT ENTERPRISEOPS_LLM_API_KEY
  unset ENTERPRISEOPS_LLM_API_VERSION ENTERPRISEOPS_LLM_CONFIG_FILE 2>/dev/null || true
  unset AZURE_OPENAI_API_KEY AZURE_OPENAI_ENDPOINT 2>/dev/null || true

  export ENTERPRISEOPS_LLM_PROVIDER=vllm
  export ENTERPRISEOPS_LLM_MODEL="$served_id"
  export ENTERPRISEOPS_LLM_TEMPERATURE="${ENTERPRISEOPS_LLM_TEMPERATURE:-0.0}"
  export ENTERPRISEOPS_LLM_MAX_TOKENS="${ENTERPRISEOPS_LLM_MAX_TOKENS:-8192}"
  export OPENAI_BASE_URL="$LOCAL_VLLM_BASE"
  export OPENAI_MODEL_NAME="$served_id"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

  case "$model" in
    nemotron-3-nano-30b-a3b|local-nemotron-3-nano-30b-a3b|nemo3|nemotron-3-super-120b-a12b|local-nemotron-3-super-120b-a12b|nemotron-3-super-120b-a12b-nvfp4|local-nemotron-3-super-120b-a12b-nvfp4|nemo3-super-nvfp4)
      export ENTERPRISEOPS_LLM_EXTRA_BODY='{"chat_template_kwargs":{"enable_thinking":false}}'
      ;;
    *)
      unset ENTERPRISEOPS_LLM_EXTRA_BODY 2>/dev/null || true
      ;;
  esac
}

configure_llm() {
  if [[ "$USE_LOCAL_VLLM" == "1" && -z "${INFERENCE_CONFIG:-}" ]]; then
    local served_id
    served_id="$(vllm_served_model_id "$VLLM_MODEL")" || exit 1
    if [[ "$START_VLLM" == "1" ]]; then
      ensure_local_vllm "$VLLM_MODEL"
    else
      log "START_VLLM=0 — checking existing vLLM at $LOCAL_VLLM_BASE ..."
      wait_local_vllm "$served_id"
    fi
    configure_vllm_llm_env "$VLLM_MODEL"
    return 0
  fi

  if [[ -n "${INFERENCE_CONFIG:-}" ]]; then
    export ENTERPRISEOPS_LLM_PROVIDER="${ENTERPRISEOPS_LLM_PROVIDER:-vllm}"
    export ENTERPRISEOPS_LLM_TEMPERATURE="${ENTERPRISEOPS_LLM_TEMPERATURE:-0.0}"
    export ENTERPRISEOPS_LLM_MAX_TOKENS="${ENTERPRISEOPS_LLM_MAX_TOKENS:-8192}"
    unset ENTERPRISEOPS_LLM_API_ENDPOINT ENTERPRISEOPS_LLM_MODEL ENTERPRISEOPS_LLM_API_KEY
    unset ENTERPRISEOPS_LLM_API_VERSION ENTERPRISEOPS_LLM_CONFIG_FILE 2>/dev/null || true
    return 0
  fi

  if [[ -n "${ENTERPRISEOPS_LLM_CONFIG_FILE:-}" && -f "${ENTERPRISEOPS_LLM_CONFIG_FILE}" ]]; then
    return 0
  fi

  if [[ -n "${OPENAI_BASE_URL:-}" ]]; then
    export ENTERPRISEOPS_LLM_PROVIDER="${ENTERPRISEOPS_LLM_PROVIDER:-vllm}"
    export ENTERPRISEOPS_LLM_MODEL="${ENTERPRISEOPS_LLM_MODEL:-${OPENAI_MODEL_NAME:-}}"
    export ENTERPRISEOPS_LLM_TEMPERATURE="${ENTERPRISEOPS_LLM_TEMPERATURE:-0.0}"
    export ENTERPRISEOPS_LLM_MAX_TOKENS="${ENTERPRISEOPS_LLM_MAX_TOKENS:-8192}"
    return 0
  fi

  if [[ -n "${ENTERPRISEOPS_LLM_API_KEY:-}" || -n "${OPENAI_API_KEY:-}" || -n "${ANTHROPIC_API_KEY:-}" ]]; then
    export ENTERPRISEOPS_LLM_PROVIDER="${ENTERPRISEOPS_LLM_PROVIDER:-openai}"
    export ENTERPRISEOPS_LLM_TEMPERATURE="${ENTERPRISEOPS_LLM_TEMPERATURE:-0.0}"
    export ENTERPRISEOPS_LLM_MAX_TOKENS="${ENTERPRISEOPS_LLM_MAX_TOKENS:-4096}"
    if [[ -z "${ENTERPRISEOPS_LLM_API_KEY:-}" ]]; then
      if [[ "${ENTERPRISEOPS_LLM_PROVIDER}" == "anthropic" && -n "${ANTHROPIC_API_KEY:-}" ]]; then
        export ENTERPRISEOPS_LLM_API_KEY="$ANTHROPIC_API_KEY"
      elif [[ -n "${OPENAI_API_KEY:-}" ]]; then
        export ENTERPRISEOPS_LLM_API_KEY="$OPENAI_API_KEY"
      fi
    fi
    return 0
  fi

  cat >&2 <<EOF
Missing LLM configuration. Defaults to local vLLM via:
  $EWM_VLLM_SCRIPTS

Set VLLM_MODEL (default: $VLLM_MODEL) and ensure VLLM_VENV points at a vLLM install, or use:
  USE_LOCAL_VLLM=0 INFERENCE_CONFIG=/path/to/inference.yaml
  ENTERPRISEOPS_LLM_API_KEY + ENTERPRISEOPS_LLM_PROVIDER + ENTERPRISEOPS_LLM_MODEL
EOF
  exit 1
}

maybe_stop_vllm_on_exit() {
  if [[ "$STOP_VLLM_ON_EXIT" == "1" && "$USE_LOCAL_VLLM" == "1" && -z "${INFERENCE_CONFIG:-}" ]]; then
    log "Stopping local vLLM on :8020 ..."
    stop_local_vllm
  fi
}

ensure_upstream_repo() {
  if [[ -d "$ENTERPRISEOPS_GYM_REPO_PATH/.git" ]]; then
    log "Upstream repo: $ENTERPRISEOPS_GYM_REPO_PATH (existing clone)"
  elif [[ -f "$ENTERPRISEOPS_GYM_REPO_PATH/benchmark/executor.py" ]]; then
    # A prepared/vendored gym (e.g. bundled in an EWM checkout) is not a git
    # clone but is already usable — don't clone over it.
    log "Upstream repo: $ENTERPRISEOPS_GYM_REPO_PATH (prepared; not a git clone)"
  else
    log "Cloning upstream EnterpriseOps-Gym into $ENTERPRISEOPS_GYM_REPO_PATH ..."
    git clone --depth 1 https://github.com/ServiceNow/EnterpriseOps-Gym.git "$ENTERPRISEOPS_GYM_REPO_PATH"
  fi
  local db_root="$ENTERPRISEOPS_GYM_REPO_PATH/Domain Wise DBs and Task-DB Mappings"
  if [[ ! -d "$db_root" ]]; then
    if [[ -f "$ENTERPRISEOPS_GYM_REPO_PATH/gym_dbs.zip" ]]; then
      log "Unzipping seed databases ..."
      (cd "$ENTERPRISEOPS_GYM_REPO_PATH" && unzip -q gym_dbs.zip)
    else
      log "Downloading gym_dbs.zip ..."
      curl -fsSL -o "$ENTERPRISEOPS_GYM_REPO_PATH/gym_dbs.zip" \
        "https://github.com/ServiceNow/EnterpriseOps-Gym/raw/main/gym_dbs.zip"
      (cd "$ENTERPRISEOPS_GYM_REPO_PATH" && unzip -q gym_dbs.zip)
    fi
  fi
  if [[ ! -d "$db_root" ]]; then
    echo "Seed databases missing under $db_root" >&2
    exit 1
  fi
}

ensure_venvs() {
  if [[ "$SKIP_SETUP" == "1" ]]; then
    return 0
  fi
  log "Syncing benchmark environments ..."
  if [[ ! -x "$EJEPA" ]]; then
    "$ROOT/scripts/install.sh" enterpriseops-gym
  fi
  uv sync --project "$BENCH_DIR/purple" --extra mcp_react --extra openai
  if [[ "$MODE" == "eval" || "$MODE" == "eval-opsgym-80" ]]; then
    uv sync --project "$BENCH_DIR/green" --extra hf
  fi
}

check_port() {
  local port="$1"
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:${port}/" || true)"
  [[ -n "$code" && "$code" != "000" ]]
}

start_mcp_container() {
  local domain="$1"
  local name="${DOMAIN_CONTAINER[$domain]}"
  local image="${DOMAIN_IMAGE[$domain]}"
  local publish="${DOMAIN_PUBLISH[$domain]}"
  local port="${DOMAIN_PORT[$domain]}"

  if check_port "$port"; then
    log "  $domain MCP already reachable on port $port"
    return 0
  fi

  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker not found; start $domain MCP on port $port manually." >&2
    return 1
  fi

  "$ROOT/scripts/check-docker-access.sh" >/dev/null

  log "  Starting $domain MCP ($name) on port $port ..."
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d --name "$name" -p "$publish" "$image" >/dev/null

  local i
  for i in $(seq 1 30); do
    if check_port "$port"; then
      log "  $domain MCP ready (port $port)"
      return 0
    fi
    sleep 1
  done
  echo "Timed out waiting for $domain MCP on port $port" >&2
  return 1
}

ensure_mcp_servers() {
  local -a requested=()
  local -a mcp_domains=()
  mapfile -t requested < <(parse_domains_list)
  mapfile -t mcp_domains < <(resolve_mcp_domains "${requested[@]}") || exit 1

  if [[ "$START_MCP" != "1" ]]; then
    log "START_MCP=0 — checking MCP ports only ..."
    local missing=0 d p
    for d in "${mcp_domains[@]}"; do
      p="${DOMAIN_PORT[$d]}"
      if check_port "$p"; then
        log "  port $p ($d): OK"
      else
        log "  port $p ($d): FAIL"
        missing=1
      fi
    done
    if [[ "$missing" -ne 0 ]]; then
      echo "Start MCP containers first. See assets/EnterpriseOps-Gym/README.md section 3." >&2
      exit 1
    fi
    return 0
  fi

  if [[ " ${requested[*]} " == *" hybrid "* ]]; then
    log "Ensuring MCP servers for domains: ${requested[*]} (hybrid → all domain gyms)"
  else
    log "Ensuring MCP servers for domains: ${requested[*]}"
  fi
  local d
  for d in "${mcp_domains[@]}"; do
    start_mcp_container "$d"
  done
}

print_latest_results() {
  local experiments_root="$BENCHMARK_HOME/experiments"
  if [[ ! -d "$experiments_root" ]]; then
    log "No experiments directory at $experiments_root"
    return 0
  fi
  python3 - "$experiments_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifests = sorted(root.glob("**/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
if not manifests:
    print("No manifest.json found under experiments/")
    raise SystemExit(0)

manifest_path = manifests[0]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
detail_path = Path(manifest.get("detail_file_path") or manifest_path.parent / "detail.json")
detail = json.loads(detail_path.read_text(encoding="utf-8")) if detail_path.is_file() else {}

print("")
print("=== Evaluation results ===")
print(f"Result dir   : {manifest.get('result_dir', manifest_path.parent)}")
print(f"Status       : {manifest.get('status')}")
print(f"Score        : {manifest.get('score_summary')}")

eval_result = manifest.get("eval_result") or {}
if eval_result:
    print(f"Tasks        : {eval_result.get('total_tasks')} total, score_rate={eval_result.get('score_rate')}")
    if "avg_verifier_pass_rate" in eval_result:
        print(f"Avg verifier : {eval_result.get('avg_verifier_pass_rate')}")

task_results = detail.get("task_results") or eval_result.get("task_results") or []
if task_results:
    print("")
    print("Per-task:")
    for row in task_results:
        tid = row.get("task_id", "?")
        score = row.get("score", row.get("score_value", "?"))
        vpr = row.get("verifier_pass_rate")
        suffix = f", verifier_pass_rate={vpr}" if vpr is not None else ""
        err = row.get("error")
        if err:
            suffix += f", error={err!r}"
        print(f"  - {tid}: score={score}{suffix}")

print(f"Manifest     : {manifest_path}")
print(f"Detail       : {detail_path}")
PY
}

# --- main -------------------------------------------------------------------
log "=== EnterpriseOps-Gym run (mode=$MODE) ==="
log "Log file: $LOG_FILE"
log "Started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

trap maybe_stop_vllm_on_exit EXIT INT TERM

configure_llm
ensure_upstream_repo
ensure_venvs
ensure_mcp_servers

if [[ ! -x "$EJEPA" ]]; then
  echo "ejepa CLI not found at $EJEPA. Run: scripts/install.sh enterpriseops-gym" >&2
  exit 1
fi

EJEPA_ARGS=(
  bench run EnterpriseOps-Gym
  --executor "$EXECUTOR"
  --ready-timeout "$READY_TIMEOUT"
)

if [[ -n "${INFERENCE_CONFIG:-}" ]]; then
  EJEPA_ARGS+=(--inference-config "$INFERENCE_CONFIG")
fi
if [[ "$SHOW_LOGS" == "1" ]]; then
  EJEPA_ARGS+=(--show-logs)
fi

if [[ "$MODE" == "smoke" ]]; then
  EJEPA_ARGS+=(--config target=sample)
  if [[ -n "$TASK_ID" ]]; then
    EJEPA_ARGS+=(--task-id "$TASK_ID")
  else
    EJEPA_ARGS+=(--task-id enterpriseops_sample_calendar_001)
  fi
elif [[ "$MODE" == "eval-opsgym-80" ]]; then
  EJEPA_ARGS+=(--config target=opsgym_80_test)
  EJEPA_ARGS+=(--config "mode=${OPSGYM_HF_MODE:-oracle}")
  EJEPA_ARGS+=(--config orchestrator=react)
  EJEPA_ARGS+=(--config "max_iterations=$MAX_ITERATIONS")
  if [[ -n "$TASK_ID" ]]; then
    EJEPA_ARGS+=(--task-id "$TASK_ID")
  fi
  EJEPA_ARGS+=(--config "max_parallel=$MAX_PARALLEL")
  EJEPA_ARGS+=(--config "capture_trajectory=$CAPTURE_TRAJECTORY")
else
  EJEPA_ARGS+=(--config target=hf_dataset)
  EJEPA_ARGS+=(--config "mode=$HF_MODE")
  if [[ -n "${DOMAINS:-}" ]]; then
    if [[ "$DOMAINS" == \[* ]]; then
      EJEPA_ARGS+=(--config "domains=$DOMAINS")
    else
      EJEPA_ARGS+=(--config "domains=[\"$DOMAINS\"]")
    fi
  else
    EJEPA_ARGS+=(--config "domain=$DOMAIN")
  fi
  if [[ -n "$MAX_TASKS_PER_DOMAIN" && "$MAX_TASKS_PER_DOMAIN" != "null" ]]; then
    EJEPA_ARGS+=(--config "max_tasks_per_domain=$MAX_TASKS_PER_DOMAIN")
  fi
  if [[ -n "$TASK_ID" ]]; then
    EJEPA_ARGS+=(--task-id "$TASK_ID")
  fi
  EJEPA_ARGS+=(--config "max_parallel=$MAX_PARALLEL")
  EJEPA_ARGS+=(--config "capture_trajectory=$CAPTURE_TRAJECTORY")
fi

log "LLM: ${ENTERPRISEOPS_LLM_PROVIDER:-?} model=${ENTERPRISEOPS_LLM_MODEL:-?} base=${OPENAI_BASE_URL:-${ENTERPRISEOPS_LLM_API_ENDPOINT:-hosted}}"
log "Command: $EJEPA ${EJEPA_ARGS[*]}"
exec > >(tee -a "$LOG_FILE") 2>&1
"$EJEPA" "${EJEPA_ARGS[@]}"
print_latest_results
