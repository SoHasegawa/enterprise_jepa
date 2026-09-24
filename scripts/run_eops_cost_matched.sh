#!/usr/bin/env bash
# Cost-matched world-model comparison on EnterpriseOps-Gym, run under the SAME
# conditions as the beam-search ablation (agent Qwen3.6-27B @ :18043, max_parallel=3,
# opsgym_80_test) so its JEPA centre run (s=8 h=3 open) is directly reusable.
#
# Arms produced here:
#   1. baseline            no world model
#   2. llmwm-s8-h3         LLM-WM (vLLM-served, :9015) at JEPA's configuration
#   3. llmwm-matched       LLM-WM at the configuration whose per-replan wall-clock
#                          equals JEPA's (set MATCHED_S / MATCHED_H from the
#                          measurement sweep; defaults below are placeholders)
#   4. (optional) jepa-s8-h3 repeat, REPEAT_JEPA=1
#
# Usage:
#   MATCHED_S=3 MATCHED_H=1 WAIT_FOR_PID=<ablation pid> nohup scripts/run_eops_cost_matched.sh \
#       > results/logs/beam_ablation/cost_matched.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

MATCHED_S="${MATCHED_S:-3}"
MATCHED_H="${MATCHED_H:-1}"
EXECUTE_STEPS="${EXECUTE_STEPS:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
TARGET="${TARGET:-opsgym_80_test}"
REPEAT_JEPA="${REPEAT_JEPA:-0}"
LABEL_PREFIX="${LABEL_PREFIX:-eops-costmatch}"

AGENT_MODEL="${AGENT_MODEL:-Qwen3.6-27B}"
AGENT_BASE_URL="${AGENT_BASE_URL:-http://127.0.0.1:18043/v1}"
AGENT_API_KEY="${AGENT_API_KEY:-EMPTY}"
WM_BASE_URL="${WM_BASE_URL:-http://127.0.0.1:9015/v1}"
WM_MODEL="${WM_MODEL:-world_model}"
JEPA_CHECKPOINT="${JEPA_CHECKPOINT:-checkpoints/jepa}"
JEPA_GPU="${JEPA_GPU:-7}"

export ENTERPRISEOPS_GYM_REPO_PATH="${ENTERPRISEOPS_GYM_REPO_PATH:-$PWD/upstreams/EnterpriseOps-Gym}"
export ENTERPRISEOPS_LLM_PROVIDER=vllm
export ENTERPRISEOPS_LLM_API_ENDPOINT="$AGENT_BASE_URL"
export ENTERPRISEOPS_LLM_MODEL="$AGENT_MODEL"
export ENTERPRISEOPS_LLM_API_KEY="$AGENT_API_KEY"
export WM_VLLM_BASE_URL="$WM_BASE_URL"
export WM_VLLM_API_KEY="$AGENT_API_KEY"
export CUDA_VISIBLE_DEVICES="$JEPA_GPU"

LOG_DIR="results/logs/beam_ablation"; SUMMARY_DIR="results/wm_harness_summaries"
mkdir -p "$LOG_DIR" "$SUMMARY_DIR/failed"

if [[ -n "${WAIT_FOR_PID:-}" ]]; then
  echo "[queue] waiting for PID $WAIT_FOR_PID..."
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do sleep 60; done
  echo "[queue] PID $WAIT_FOR_PID gone at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

# --- preflight ---------------------------------------------------------------
ok=1
[[ -d "$ENTERPRISEOPS_GYM_REPO_PATH/benchmark" ]] || { echo "PREFLIGHT: EnterpriseOps repo missing"; ok=0; }
curl -s -m 15 "$AGENT_BASE_URL/models" | grep -q "\"$AGENT_MODEL\"" || { echo "PREFLIGHT: agent $AGENT_MODEL not served at $AGENT_BASE_URL"; ok=0; }
curl -s -m 15 "$WM_BASE_URL/models" | grep -q "\"$WM_MODEL\"" || { echo "PREFLIGHT: world model $WM_MODEL not served at $WM_BASE_URL"; ok=0; }
[[ "$ok" == 1 ]] || { echo "PREFLIGHT FAILED"; exit 2; }
echo "preflight OK. matched LLM-WM config: s=$MATCHED_S h=$MATCHED_H (execute=$EXECUTE_STEPS)"

BASE=(ejepa bench run EnterpriseOps-Gym --executor mcp_react --config target="$TARGET"
      --config task_limit=0 --config capture_trajectory=true --config max_parallel="$MAX_PARALLEL"
      --config model_name="$AGENT_MODEL")

quarantine_if_empty() {  # a zero-score run finishing in < 120 s = every task failed at startup
  local label="$1" f
  f=$(compgen -G "$SUMMARY_DIR/ejepa-wm-harnesses-${label}-*.json" | tail -1) || return 0
  [[ -n "$f" ]] || return 0
  python3 - "$f" <<'PY' || { mv "$f" "$SUMMARY_DIR/failed/"; echo "  quarantined $f"; return 1; }
import json,sys
for r in json.load(open(sys.argv[1])).get("runs") or []:
    s=r.get("result_summary") or {}
    per_task=s.get("per_task") or []
    # all tasks failing in the executor (CUDA OOM, dead endpoint) is infrastructure,
    # not a result, and can outlast any duration threshold.
    broken=[t for t in per_task if "executor error" in str(t.get("reason") or "") or t.get("error")]
    if per_task and len(broken) >= 0.9*len(per_task): sys.exit(1)
    if (s.get("score_rate") in (0,0.0,None)) and float(r.get("wrapper_elapsed_seconds") or 0) < 120: sys.exit(1)
PY
}

run_wm() {  # label  extra-ejepa-args...
  local label="$1"; shift
  echo "=== $(date -u +%H:%M:%SZ) START $label"
  python scripts/run_wm_harnesses.py --label "$label" --harnesses beam_interval \
    --beam-score-margin 0.10 --beam-temperature 0.7 --beam-refinement-rounds 1 --beam-refinement-top-k 4 \
    --ssot-diversity --no-beam-hard-override "$@" > "$LOG_DIR/$label.log" 2>&1 \
    && quarantine_if_empty "$label" && echo "=== $(date -u +%H:%M:%SZ) OK $label" || echo "=== $(date -u +%H:%M:%SZ) FAIL $label (see $LOG_DIR/$label.log)"
}

# 1. baseline (no world model)
label="$LABEL_PREFIX-baseline"
echo "=== $(date -u +%H:%M:%SZ) START $label"
( unset WM_STRATEGY; python scripts/run_bench_repeated.py --runs 1 \
    --result-root "${BENCHMARK_HOME:-$BENCHMARK_HOME}/experiments" \
    --label "$label" -- "${BASE[@]}" ) > "$LOG_DIR/$label.log" 2>&1 \
  && echo "=== $(date -u +%H:%M:%SZ) OK $label" || echo "=== $(date -u +%H:%M:%SZ) FAIL $label"

# 2. LLM-WM at JEPA's configuration
run_wm "$LABEL_PREFIX-llmwm-s8-h3" --beam-samples 8 --beam-horizon 3 --beam-execute-steps "$EXECUTE_STEPS" -- \
  "${BASE[@]}" --wm-llm-ewm-mode llm_canonical_trained --wm-ewm-model "$WM_MODEL" \
  --wm-imagined-rollout-mode open_loop --wm-beam-mpc-execute-steps "$EXECUTE_STEPS" \
  --wm-beam-plan-terminal-advice --wm-beam-plan-terminal-advice-threshold 0.75

# 3. LLM-WM at JEPA's wall-clock budget
run_wm "$LABEL_PREFIX-llmwm-matched-s${MATCHED_S}-h${MATCHED_H}" --beam-samples "$MATCHED_S" --beam-horizon "$MATCHED_H" \
  --beam-execute-steps "$(( MATCHED_H < EXECUTE_STEPS ? MATCHED_H : EXECUTE_STEPS ))" -- \
  "${BASE[@]}" --wm-llm-ewm-mode llm_canonical_trained --wm-ewm-model "$WM_MODEL" \
  --wm-imagined-rollout-mode open_loop --wm-beam-mpc-execute-steps "$(( MATCHED_H < EXECUTE_STEPS ? MATCHED_H : EXECUTE_STEPS ))" \
  --wm-beam-plan-terminal-advice --wm-beam-plan-terminal-advice-threshold 0.75

# 4. optional JEPA repeat at the centre config
if [[ "$REPEAT_JEPA" == 1 ]]; then
  run_wm "$LABEL_PREFIX-jepa-s8-h3" --beam-samples 8 --beam-horizon 3 --beam-execute-steps "$EXECUTE_STEPS" -- \
    "${BASE[@]}" --wm-ewm-jepa-checkpoint "$JEPA_CHECKPOINT" --wm-jepa-observation-backend canonical_event \
    --wm-imagined-rollout-mode open_loop --wm-beam-mpc-execute-steps "$EXECUTE_STEPS" \
    --wm-beam-plan-terminal-advice --wm-beam-plan-terminal-advice-threshold 0.75
fi
echo "=== done ==="
