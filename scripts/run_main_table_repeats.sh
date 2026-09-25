#!/usr/bin/env bash
# Re-run the main table's world-model cells under one fixed configuration.
#
# Decision 2026-09-18: apart from EnterpriseOps-Gym, the existing world-model runs mix
# parallelism (WorkBench spans mp=1/3/10/20) and beam settings (h4/e4 vs h3/e2), so they
# are not comparable within a column. Every cell here is re-run with the protocol in
# results/analysis/main_table_protocol.md; baselines are NOT re-run (they carry no
# world-model configuration and are reused), and each cell gets REPEATS new runs.
#
# One benchmark per invocation so that several can run concurrently on separate agent
# endpoints and GPUs:
#
#   BENCH=WorkBench      AGENT_URL=http://127.0.0.1:9010/v1  AGENT_NAME=wm_agent    JEPA_GPU=3 \
#     scripts/run_main_table_repeats.sh
#   BENCH=AutomationBench AGENT_URL=http://127.0.0.1:18045/v1 AGENT_NAME=Qwen3.6-27B JEPA_GPU=6 \
#     scripts/run_main_table_repeats.sh
#
# Command shape follows the project's reference invocation: one run_wm_harnesses
# call per (world model, repeat) covering all three harnesses, with the critic settings
# --critic-failure-prob 0.30 --critic-stall-prob 0.40 --critic-max-quiet-steps 4.
#
# Labels: tab-<slug>-<jepa|llmwm>-r<N> (one summary holds all three harnesses).
# Resumable: a label whose summary already exists is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."

BENCH="${BENCH:?set BENCH to WorkBench|AutomationBench|crmarenapro|Terminal-Bench-2.0}"
AGENT_URL="${AGENT_URL:?set AGENT_URL}"
AGENT_NAME="${AGENT_NAME:?set AGENT_NAME}"
JEPA_GPU="${JEPA_GPU:-3}"
REPEATS="${REPEATS:-2}"
# First repeat index to run. Lets a caller drive one repeat at a time (REP_START=2
# REPEATS=2) so several repeats of the same cell can run concurrently on different
# agent endpoints instead of being serialised inside one invocation.
REP_START="${REP_START:-1}"
# Appended to the run label. Lets one harness be banked per invocation
# (HARNESSES=beam_interval LABEL_SUFFIX=-beam), so an endpoint dying during the second
# harness does not quarantine the summary of the first, which is how a completed
# AutomationBench beam run was lost.
LABEL_SUFFIX="${LABEL_SUFFIX:-}"
WORLD_MODELS="${WORLD_MODELS:-jepa llmwm}"
HARNESSES="${HARNESSES:-beam_interval revision itp_i}"   # one invocation runs all three
WM_URL="${WM_URL:-http://127.0.0.1:9015/v1}"
WM_NAME="${WM_NAME:-world_model}"
JEPA_CKPT="${JEPA_CKPT:-checkpoints/jepa}"

LOG_DIR=results/logs/beam_ablation
SUMMARY_DIR=results/wm_harness_summaries
mkdir -p "$LOG_DIR" "$SUMMARY_DIR/failed"

export CUDA_VISIBLE_DEVICES="$JEPA_GPU"
export WM_SHARE_MODEL_WEIGHTS=1
export WM_VLLM_BASE_URL="$WM_URL"
export WM_VLLM_API_KEY=EMPTY
unset WM_JEPA_PREDICTION_CONTROL WM_JEPA_COMPILE

case "$BENCH" in
  EnterpriseOps-Gym)
    SLUG=eops; MP=5
    # EnterpriseOps-Gym is verified insensitive to parallelism up to mp=5 (operations
    # paired test: 31/100 at mp=3 vs 32/100 at mp=5), so the fastest safe setting is used.
    export ENTERPRISEOPS_GYM_REPO_PATH="${ENTERPRISEOPS_GYM_REPO_PATH:-$PWD/upstreams/EnterpriseOps-Gym}"
    export ENTERPRISEOPS_LLM_PROVIDER=vllm ENTERPRISEOPS_LLM_API_ENDPOINT="$AGENT_URL"
    export ENTERPRISEOPS_LLM_MODEL="$AGENT_NAME" ENTERPRISEOPS_LLM_API_KEY=EMPTY
    BENCH_CFG=(--config target=opsgym_80_test --config model_name="$AGENT_NAME")
    ;;
  WorkBench)
    SLUG=wb; MP=1
    export WORKBENCH_REPO_PATH="${WORKBENCH_REPO_PATH:-$PWD/upstreams/WorkBench}"
    export WORKBENCH_VLLM_BASE_URL="$AGENT_URL" WORKBENCH_VLLM_MODEL="$AGENT_NAME" WORKBENCH_VLLM_API_KEY=EMPTY
    BENCH_CFG=(--config target=all --config max_turns=50 --config timeout=3600 --config model_name="$AGENT_NAME")
    ;;
  AutomationBench)
    SLUG=ab; MP=5
    export AUTOMATIONBENCH_REPO_PATH="${AUTOMATIONBENCH_REPO_PATH:-$PWD/upstreams/AutomationBench}"
    export AUTOMATIONBENCH_LLM_MODEL="$AGENT_NAME" AUTOMATIONBENCH_LLM_API_ENDPOINT="$AGENT_URL"
    export AUTOMATIONBENCH_LLM_API_KEY=EMPTY AUTOMATIONBENCH_LLM_TEMPERATURE=0
    export OPENAI_BASE_URL="$AGENT_URL" OPENAI_API_KEY=EMPTY
    BENCH_CFG=(--config target=all_domains --config toolset=limited_zapier --config max_turns=50)
    ;;
  crmarenapro)
    SLUG=crm; MP=3
    # No MAX_TURNS here: nothing in the crmarenapro asset reads it (it was copied from an
    # older launcher). The turn budget is green/agent.py's `max_turns = min(config.max_steps, 10)`,
    # i.e. hard-capped at 10, and max_steps itself is hardcoded to 10 unless leaderboard_mode=true.
    # `--config max_turns=50` below is likewise inert; it is kept only to match the reference command.
    export LLM_PROVIDER=openai_compatible LLM_MODEL="$AGENT_NAME" LLM_BASE_URL="$AGENT_URL" LLM_API_KEY=EMPTY
    export OPENAI_API_KEY=EMPTY
    # Matches the reference command. NOTE: crmarenapro only honours `timeout`/`max_steps`
    # when leaderboard_mode=true (assets/crmarenapro/green/agent.py), so with this setting
    # the effective per-task timeout is the hardcoded 300 s, not 3600 s. Kept as given so
    # every CRM cell shares one setting; set CRM_LEADERBOARD=1 to switch the column.
    BENCH_CFG=(--config target=world_model_test --config max_turns=50 --config timeout=3600 --config model_name="$AGENT_NAME")
    if [ "${CRM_LEADERBOARD:-0}" = 1 ]; then
      BENCH_CFG+=(--config leaderboard_mode=true --config max_steps=20)
    fi
    ;;
  Terminal-Bench-2.0)
    SLUG=tb; MP=1
    # Terminal-Bench uses the generic OpenAI-compatible route (LLM_BASE_URL/LLM_MODEL/
    # LLM_API_KEY per its README), not a TERMINAL_BENCH_LLM_API_ENDPOINT variable.
    # Set both families: the README says mcp_react reuses llm_shell's generic LLM_*
    # credentials, while the other executors read TERMINAL_BENCH_LLM_*. The first
    # attempt set only LLM_* and every task died with "Connection refused".
    export LLM_BASE_URL="$AGENT_URL" LLM_MODEL="$AGENT_NAME" LLM_API_KEY=EMPTY
    export TERMINAL_BENCH_LLM_BASE_URL="$AGENT_URL" TERMINAL_BENCH_LLM_MODEL="$AGENT_NAME"
    export TERMINAL_BENCH_LLM_API_KEY=EMPTY
    export OPENAI_BASE_URL="$AGENT_URL" OPENAI_API_KEY=EMPTY
    # The task repo is not at the default path on this machine; point at the
    # checkout explicitly or task_loader raises FileNotFoundError.
    export TERMINAL_BENCH_TASK_REPO="${TERMINAL_BENCH_TASK_REPO:-$PWD/upstreams/terminal-bench-2}"
    export TERMINAL_BENCH_WORKSPACE="${TERMINAL_BENCH_WORKSPACE:-/tmp/tb-workspace}"
    BENCH_CFG=(--config target=all --config timeout=3600 --config model_name="$AGENT_NAME")
    ;;
  *) echo "unknown BENCH=$BENCH"; exit 2 ;;
esac

curl -s -m 15 "$AGENT_URL/models" | grep -q "\"$AGENT_NAME\"" || { echo "PREFLIGHT: agent $AGENT_NAME not served at $AGENT_URL"; exit 2; }
[ -d "$JEPA_CKPT" ] || { echo "PREFLIGHT: JEPA checkpoint missing at $JEPA_CKPT"; exit 2; }
if [[ " $WORLD_MODELS " == *" llmwm "* || " $WORLD_MODELS " == *" agentworld "* ]]; then
  curl -s -m 15 "$WM_URL/models" | grep -q "\"$WM_NAME\"" || { echo "PREFLIGHT: world model $WM_NAME not served at $WM_URL"; exit 2; }
fi
echo "=== main-table repeats: $BENCH (mp=$MP), agent $AGENT_NAME @ $AGENT_URL, JEPA on GPU $JEPA_GPU, x$REPEATS"

run_cell() {  # world_model repeat   (runs every harness in one invocation)
  local wm="$1" rep="$2"
  local label="tab-${SLUG}-${wm}-r${rep}${LABEL_SUFFIX}"
  if compgen -G "$SUMMARY_DIR/ejepa-wm-harnesses-${label}-*.json" >/dev/null; then
    echo "SKIP  $label (summary exists)"; return 0
  fi
  # an endpoint can die mid-sweep; check before paying for a multi-hour run
  if ! curl -s -m 15 "$AGENT_URL/models" | grep -q "\"$AGENT_NAME\""; then
    echo "=== $(date -u +%H:%M:%SZ) ABORT $label: agent gone"; return 1
  fi
  local wm_flags=()
  case "$wm" in
    jepa)
      wm_flags=(--wm-ewm-jepa-checkpoint "$JEPA_CKPT" --wm-jepa-observation-backend canonical_event) ;;
    agentworld)
      # Qwen-AgentWorld judges raw tool output rather than predicting a canonical state.
      wm_flags=(--wm-llm-ewm-mode llm_tool_output_judge --wm-ewm-model "$WM_NAME") ;;
    *)
      wm_flags=(--wm-llm-ewm-mode llm_canonical_trained --wm-ewm-model "$WM_NAME") ;;
  esac
  echo "=== $(date -u +%H:%M:%SZ) START $label"
  # shellcheck disable=SC2086 -- $HARNESSES is an intentional word list
  python scripts/run_wm_harnesses.py --label "$label" --harnesses $HARNESSES \
    --itp-max-k 4 --beam-samples 8 --beam-horizon 3 --beam-execute-steps 2 \
    --beam-score-margin 0.10 --beam-temperature 0.7 \
    --critic-failure-prob 0.30 --critic-stall-prob 0.40 --critic-max-quiet-steps 4 \
    --ssot-diversity --no-beam-hard-override -- \
    ejepa bench run "$BENCH" --executor mcp_react \
      "${BENCH_CFG[@]}" --config task_limit=0 --config capture_trajectory=true \
      --config max_parallel="$MP" \
      "${wm_flags[@]}" --wm-itp-fixed-k 4 \
      --wm-imagined-rollout-mode open_loop --wm-beam-mpc-execute-steps 2 \
      --wm-beam-plan-terminal-advice --wm-beam-plan-terminal-advice-threshold 0.75 \
    > "$LOG_DIR/$label.log" 2>&1
  echo "=== $(date -u +%H:%M:%SZ) DONE  $label rc=$?"
  local f
  f=$(compgen -G "$SUMMARY_DIR/ejepa-wm-harnesses-${label}-*.json" | tail -1) || return 0
  [ -n "${f:-}" ] || return 0
  python3 - "$f" <<'PYEOF' || { mv "$f" "$SUMMARY_DIR/failed/"; echo "=== quarantined $label (all tasks failed)"; }
import json,sys
for r in json.load(open(sys.argv[1])).get("runs") or []:
    pt=(r.get("result_summary") or {}).get("per_task") or []
    bad=[t for t in pt if "executor error" in str(t.get("reason") or "") or t.get("error")]
    if pt and len(bad) >= 0.5*len(pt):
        print("   ", len(bad), "of", len(pt), "tasks failed:", str(bad[0].get("reason"))[:140]); sys.exit(1)
PYEOF
}

# JEPA before the LLM world model, and repeat 1 of both before repeat 2, so an
# interruption leaves a complete n=1 sweep rather than a half-finished column.
for rep in $(seq "$REP_START" "$REPEATS"); do
  for wm in $WORLD_MODELS; do
    run_cell "$wm" "$rep"
  done
done
echo "=== $(date -u +%H:%M:%SZ) main-table repeats done for $BENCH"
