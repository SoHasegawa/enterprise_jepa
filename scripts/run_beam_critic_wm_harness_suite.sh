#!/usr/bin/env bash
# Run the beam_critic WM harness once per benchmark with ONE shared configuration
# (the EnterpriseOps-Gym "jepa-separate" settings: critic thresholds 0.65/0.65,
# max-quiet-steps 8, 8 samples x horizon 4, SSoT diversity, advisory mode).
#
# Benchmarks are run SEQUENTIALLY: every run loads its own JEPA world model and
# drives the same vLLM policy endpoint, so overlapping them distorts latency.
#
# Usage:
#   scripts/run_beam_critic_wm_harness_suite.sh                 # run all five
#   DRY_RUN=1 scripts/run_beam_critic_wm_harness_suite.sh       # print/validate only
#   BENCHMARKS="WorkBench AutomationBench" scripts/run_beam_critic_wm_harness_suite.sh
#   WAIT_FOR_PID=2232308 scripts/run_beam_critic_wm_harness_suite.sh   # start after a running job exits
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --- policy ("agent") LLM served by vLLM -------------------------------------
# Same instance the current EnterpriseOps-Gym run uses (Qwen3.6-27B @ :9011).
AGENT_BASE_URL="${AGENT_BASE_URL:-http://127.0.0.1:9011/v1}"
AGENT_MODEL="${AGENT_MODEL:-wm_agent1}"
AGENT_API_KEY="${AGENT_API_KEY:-EMPTY}"

# --- world model -------------------------------------------------------------
JEPA_CHECKPOINT="${JEPA_CHECKPOINT:-checkpoints/jepa}"

# Imagined-rollout mode for the beam planner; matches the EnterpriseOps-Gym run.
ROLLOUT_MODE="${ROLLOUT_MODE:-open_loop}"

# The JEPA generator has no device knob -- it takes the first VISIBLE CUDA device
# (src/ejepa_wm/backends/_ewm_jepa.py:1456), so pinning is done with
# CUDA_VISIBLE_DEVICES: the executor process then sees GPU $JEPA_GPU as cuda:0.
# The policy LLM is a remote vLLM over HTTP and is unaffected.
JEPA_GPU="${JEPA_GPU:-7}"
export CUDA_VISIBLE_DEVICES="$JEPA_GPU"

# --- run control -------------------------------------------------------------
BENCHMARKS="${BENCHMARKS:-crmarenapro WorkBench AutomationBench Terminal-Bench-2.0}"
LABEL_SUFFIX="${LABEL_SUFFIX:-beam-critic-separate}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python3}"
EJEPA="${EJEPA:-$ROOT/.venv/bin/ejepa}"
LOG_DIR="${LOG_DIR:-$ROOT/results/logs/beam_critic_suite}"
mkdir -p "$LOG_DIR"

# --- per-benchmark agent credentials ----------------------------------------
# Each purple executor reads its OWN env var names; these are the ones the
# mcp_react executors actually resolve (first match wins in each chain).
export OPENAI_API_KEY="${OPENAI_API_KEY:-$AGENT_API_KEY}"

# crmarenapro -> baseline_crm_agent.Agent (LLM_PROVIDER must be openai_compatible,
# otherwise it defaults to Anthropic). assets/crmarenapro/purple-executors/baseline_crm_agent/agent.py:326,374
export LLM_PROVIDER=openai_compatible
export LLM_BASE_URL="$AGENT_BASE_URL"
export LLM_MODEL="$AGENT_MODEL"
export LLM_API_KEY="$AGENT_API_KEY"

# WorkBench -> WORKBENCH_VLLM_* registers a vLLM route; `--config model_name=`
# must equal WORKBENCH_VLLM_MODEL_NAME or WORKBENCH_VLLM_MODEL.
# assets/WorkBench/purple-executors/mcp_react/executor.py:72-75,190-200
export WORKBENCH_VLLM_BASE_URL="$AGENT_BASE_URL"
export WORKBENCH_VLLM_MODEL="$AGENT_MODEL"
export WORKBENCH_VLLM_MODEL_NAME="$AGENT_MODEL"
export WORKBENCH_VLLM_API_KEY="$AGENT_API_KEY"

# Terminal-Bench-2.0 -> TERMINAL_BENCH_LLM_* (shared shell-agent resolver)
# assets/Terminal-Bench-2.0/purple-executors/llm_shell/executor.py:181-258
export TERMINAL_BENCH_LLM_BASE_URL="$AGENT_BASE_URL"
export TERMINAL_BENCH_LLM_MODEL="$AGENT_MODEL"
export TERMINAL_BENCH_LLM_API_KEY="$AGENT_API_KEY"

# EnterpriseOps-Gym, for parity if it is added to BENCHMARKS.
export ENTERPRISEOPS_LLM_PROVIDER=vllm
export ENTERPRISEOPS_LLM_API_ENDPOINT="$AGENT_BASE_URL"
export ENTERPRISEOPS_LLM_MODEL="$AGENT_MODEL"
export ENTERPRISEOPS_LLM_API_KEY="${ENTERPRISEOPS_LLM_API_KEY:-$AGENT_API_KEY}"

# An Azure/OpenAI endpoint left in the environment wins over LLM_BASE_URL in the
# crmarenapro / Terminal-Bench resolvers, so clear those routes.
unset AZURE_OPENAI_API_KEY AZURE_OPENAI_ENDPOINT AZURE_OPENAI_API_VERSION \
      AZURE_OPENAI_DEPLOYMENT_NAME OPENAI_BASE_URL OPENAI_API_BASE_URL \
      OPENAI_MODEL_NAME 2>/dev/null || true

# --- harness configuration (identical for every benchmark) -------------------
HARNESS_ARGS=(
  --harnesses beam_critic
  --itp-max-k 4
  --beam-samples 8
  --beam-horizon 4
  --beam-execute-steps 4
  --beam-score-margin 0.10
  --beam-temperature 0.7
  --critic-failure-prob 0.65
  --critic-stall-prob 0.65
  --critic-max-quiet-steps 8
  --ssot-diversity
  --no-beam-hard-override
)
[[ "$STOP_ON_FAILURE" == "1" ]] && HARNESS_ARGS+=(--stop-on-failure)
[[ "$DRY_RUN" == "1" ]] && HARNESS_ARGS+=(--dry-run)
[[ -n "${RESULT_ROOT:-}" ]] && HARNESS_ARGS+=(--result-root "$RESULT_ROOT")

WM_ARGS=(
  --wm-ewm-jepa-checkpoint "$JEPA_CHECKPOINT"
  --wm-jepa-observation-backend canonical_event
  --wm-itp-fixed-k 4
  --wm-beam-plan-terminal-advice
  --wm-beam-plan-terminal-advice-threshold 0.75
  --wm-beam-mpc-execute-steps 4
  # open_loop: ONE candidate-generation request per re-plan (all samples from a
  # shared prompt, scored in one batched WM pass) instead of closed_loop's one
  # request per horizon depth. Also the mode in which --wm-beam-plan-ssot-diversity
  # actually applies (ewm_imagined.py:2115,2127).
  --wm-imagined-rollout-mode "$ROLLOUT_MODE"
)

# --- per-benchmark target / extra config -------------------------------------
# Targets and extra --config keys mirror each benchmark's last working run.
benchmark_target() {
  case "$1" in
    crmarenapro)        echo "world_model_test_longest_100" ;;
    WorkBench)          echo "longest_100_balanced" ;;
    Terminal-Bench-2.0) echo "all" ;;
    AutomationBench)    echo "all_domains" ;;
    EnterpriseOps-Gym)  echo "opsgym_80_test" ;;
    *) return 1 ;;
  esac
}

benchmark_label() {
  # "{benchmark name}-beam-critic-separate", e.g. WorkBench-beam-critic-separate.
  # run_wm_harnesses.safe_label keeps [A-Za-z0-9_.-] as-is.
  echo "$1-${LABEL_SUFFIX}"
}

# WorkBench green REQUIRES config.model_name and it must match the registered
# vLLM alias; Terminal-Bench carries it for provenance.
benchmark_extra_config() {
  case "$1" in
    WorkBench|Terminal-Bench-2.0) echo "--config model_name=$AGENT_MODEL" ;;
    *) echo "" ;;
  esac
}

# Per-benchmark env, applied to that run only (not exported globally).
# crmarenapro: MAX_TURNS is its ReAct budget -- BaselineAgent/WmReactAgent read
# os.getenv("MAX_TURNS", "8") (assets/crmarenapro/purple-executors/baseline_crm_agent/agent.py:406);
# there is no --config for it. No other benchmark reads MAX_TURNS.
CRMARENAPRO_MAX_TURNS="${CRMARENAPRO_MAX_TURNS:-20}"
benchmark_env() {
  case "$1" in
    crmarenapro) echo "MAX_TURNS=$CRMARENAPRO_MAX_TURNS" ;;
    *) echo "" ;;
  esac
}

# --- preflight ---------------------------------------------------------------
echo "=== beam_critic WM harness suite (sequential, one benchmark at a time) ==="
echo "agent LLM   : $AGENT_MODEL @ $AGENT_BASE_URL"
echo "JEPA ckpt   : $JEPA_CHECKPOINT"
echo "JEPA GPU    : $JEPA_GPU (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "rollout mode: $ROLLOUT_MODE"
echo "label       : {benchmark}-$LABEL_SUFFIX"
echo "benchmarks  : $BENCHMARKS"
echo "logs        : $LOG_DIR"
echo "dry run     : $DRY_RUN"

for binary in "$PYTHON" "$EJEPA"; do
  [[ -x "$binary" ]] || { echo "FATAL: not executable: $binary" >&2; exit 1; }
done
[[ -d "$JEPA_CHECKPOINT" ]] || { echo "FATAL: JEPA checkpoint dir not found: $JEPA_CHECKPOINT" >&2; exit 1; }

if command -v nvidia-smi >/dev/null; then
  gpu_info="$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i "$JEPA_GPU" 2>&1)" \
    || { echo "FATAL: GPU $JEPA_GPU not available: $gpu_info" >&2; exit 1; }
  echo "GPU $JEPA_GPU     : $gpu_info"
else
  echo "WARNING: nvidia-smi not found; cannot verify GPU $JEPA_GPU" >&2
fi

if ! curl -s --max-time 10 "$AGENT_BASE_URL/models" | grep -q "\"$AGENT_MODEL\""; then
  echo "FATAL: $AGENT_BASE_URL does not serve model '$AGENT_MODEL'" >&2
  echo "       served models: $(curl -s --max-time 10 "$AGENT_BASE_URL/models" || echo '<no response>')" >&2
  exit 1
fi
echo "endpoint    : OK (serves $AGENT_MODEL)"

for benchmark in $BENCHMARKS; do
  benchmark_target "$benchmark" >/dev/null || { echo "FATAL: unknown benchmark '$benchmark'" >&2; exit 1; }
done

if [[ -n "$WAIT_FOR_PID" ]]; then
  # Space-separated list: e.g. the in-flight harness wrappers for EnterpriseOps-Gym
  # Waiting avoids two runs sharing one vLLM endpoint / benchmark.
  echo "Waiting for PID(s) to exit before starting: $WAIT_FOR_PID"
  for pid in $WAIT_FOR_PID; do
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    echo "  PID $pid finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  done
  echo "All awaited PIDs finished; starting suite."
fi

# --- run ---------------------------------------------------------------------
declare -a SUMMARY=()
suite_status=0

for benchmark in $BENCHMARKS; do
  target="$(benchmark_target "$benchmark")"
  label="$(benchmark_label "$benchmark")"
  # shellcheck disable=SC2206
  extra_config=( $(benchmark_extra_config "$benchmark") )
  # shellcheck disable=SC2206
  extra_env=( $(benchmark_env "$benchmark") )
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log_file="$LOG_DIR/${label}-${stamp}.log"

  command=(
    "$PYTHON" scripts/run_wm_harnesses.py
    --label "$label"
    "${HARNESS_ARGS[@]}"
    --
    "$EJEPA" bench run "$benchmark"
    --executor mcp_react
    --config "target=$target"
    "${WM_ARGS[@]}"
    --config task_limit=0
    --config capture_trajectory=true
  )
  [[ ${#extra_config[@]} -gt 0 ]] && command+=("${extra_config[@]}")
  [[ ${#extra_env[@]} -gt 0 ]] && command=(env "${extra_env[@]}" "${command[@]}")

  echo
  echo "--- [$benchmark] target=$target label=$label"
  [[ ${#extra_env[@]} -gt 0 ]] && echo "    env: ${extra_env[*]}"
  echo "    log: $log_file"
  printf '    cmd:'; printf ' %q' "${command[@]}"; echo

  started=$SECONDS
  set +e
  "${command[@]}" >"$log_file" 2>&1
  status=$?
  set -e
  elapsed=$((SECONDS - started))

  if [[ $status -eq 0 ]]; then
    echo "    OK in ${elapsed}s"
  else
    echo "    FAILED (exit $status) after ${elapsed}s -- see $log_file"
    suite_status=1
    [[ "$STOP_ON_FAILURE" == "1" ]] && { SUMMARY+=("$benchmark exit=$status ${elapsed}s $log_file"); break; }
  fi
  SUMMARY+=("$benchmark exit=$status ${elapsed}s $log_file")
done

echo
echo "=== suite summary ==="
for row in "${SUMMARY[@]}"; do echo "  $row"; done
echo
echo "Per-run harness summaries: results/wm_harness_summaries/ejepa-wm-harnesses-*-${LABEL_SUFFIX}-*.json"
echo "Compare with: $PYTHON scripts/summarize_wm_harness_summaries.py --show-skipped"
exit $suite_status
