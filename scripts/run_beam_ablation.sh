#!/usr/bin/env bash
# Beam-search planning ablation: candidates x horizon x rollout mode, on
# EnterpriseOps-Gym and AutomationBench, with the JEPA world model.
#
# One factor at a time around the paper's centre point, six configs per benchmark
# (paper Table 10):
#
#     centre            s=8  h=3  open_loop
#     candidates        s=4 | s=16        (h=3, open_loop)
#     horizon           h=2 | h=4         (s=8, open_loop)
#     rollout mode      closed_loop       (s=8, h=3)
#
# Execute-steps is held fixed at 2 so that "horizon" varies lookahead depth alone.
# GRID=1 runs the full 3 x 3 x 2 grid instead. REPEATS / EOPS_REPEATS / AB_REPEATS set
# the repeat count (the paper uses 3); AB_TARGETS="sales operations support" runs
# several AutomationBench domains.
#
# Labels encode the config so scripts/summarize_beam_ablation.py can recover it:
#     beam-ablation-<bench>[-<ab domain>]-s<S>-h<H>-e<E>-<open|closed>-r<REP>
#
# Usage:
#   DRY_RUN=1 scripts/run_beam_ablation.sh          # print the plan only
#   nohup scripts/run_beam_ablation.sh > results/logs/beam_ablation/run.out 2>&1 &
#   WAIT_FOR_PID=<pid> scripts/run_beam_ablation.sh # queue behind a running job
#
# Resumable: a config whose summary JSON already exists is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."

# --- design -------------------------------------------------------------------
CENTRE_S="${CENTRE_S:-8}"
CENTRE_H="${CENTRE_H:-3}"
CENTRE_LOOP="${CENTRE_LOOP:-open_loop}"
EXECUTE_STEPS="${EXECUTE_STEPS:-2}"
SAMPLE_LEVELS="${SAMPLE_LEVELS:-4 8 16}"
HORIZON_LEVELS="${HORIZON_LEVELS:-2 3 4}"
LOOP_LEVELS="${LOOP_LEVELS:-open_loop closed_loop}"
GRID="${GRID:-0}"
REPEATS="${REPEATS:-1}"                    # default repeats per benchmark
EOPS_REPEATS="${EOPS_REPEATS:-$REPEATS}"   # per-benchmark override
AB_REPEATS="${AB_REPEATS:-$REPEATS}"
BENCHMARKS="${BENCHMARKS:-EnterpriseOps-Gym AutomationBench}"

# --- benchmark settings -------------------------------------------------------
EOPS_TARGET="${EOPS_TARGET:-opsgym_80_test}"
# AutomationBench is run one domain (100 tasks, ~1h) at a time; each domain is a
# separate run whose label carries the domain (`beam-ablation-ab-<domain>-...`)
# and the summarizer pools them. `operations` was the original single domain
# (labels without a domain segment are read as operations); marketing/finance
# are excluded from the paper's AB numbers and `hr` is unmeasurable, leaving
# sales, operations and support as the measurable set.
AB_TARGETS="${AB_TARGETS:-${AB_TARGET:-operations}}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"

# --- agent LLM (vLLM) ---------------------------------------------------------
AGENT_MODEL="${AGENT_MODEL:-wm_agent}"
AGENT_BASE_URL="${AGENT_BASE_URL:-http://127.0.0.1:9010/v1}"
AGENT_API_KEY="${AGENT_API_KEY:-EMPTY}"

# --- world model --------------------------------------------------------------
JEPA_CHECKPOINT="${JEPA_CHECKPOINT:-checkpoints/jepa}"
JEPA_GPU="${JEPA_GPU:-7}"
export CUDA_VISIBLE_DEVICES="$JEPA_GPU"

# --- fixed harness settings (unchanged from the paper's beam_interval runs) ---
SCORE_MARGIN="${SCORE_MARGIN:-0.10}"
TEMPERATURE="${TEMPERATURE:-0.7}"
REFINEMENT_ROUNDS="${REFINEMENT_ROUNDS:-1}"
REFINEMENT_TOP_K="${REFINEMENT_TOP_K:-4}"

LOG_DIR="results/logs/beam_ablation"
SUMMARY_DIR="results/wm_harness_summaries"
mkdir -p "$LOG_DIR"

# --- environment for each benchmark's purple executor -------------------------
# Upstream checkouts. Without ENTERPRISEOPS_GYM_REPO_PATH every task fails in
# ~0.1 s with "repository not found" while `ejepa` still exits 0 and the harness
# records status=completed, score=0.0 -- see the preflight and post-run guards.
export ENTERPRISEOPS_GYM_REPO_PATH="${ENTERPRISEOPS_GYM_REPO_PATH:-$PWD/upstreams/EnterpriseOps-Gym}"
export ENTERPRISEOPS_LLM_PROVIDER=vllm
export ENTERPRISEOPS_LLM_API_ENDPOINT="$AGENT_BASE_URL"
export ENTERPRISEOPS_LLM_MODEL="$AGENT_MODEL"
export ENTERPRISEOPS_LLM_API_KEY="$AGENT_API_KEY"

export AUTOMATIONBENCH_REPO_PATH="${AUTOMATIONBENCH_REPO_PATH:-$PWD/upstreams/AutomationBench}"
export AUTOMATIONBENCH_LLM_MODEL="$AGENT_MODEL"
export AUTOMATIONBENCH_LLM_API_ENDPOINT="$AGENT_BASE_URL"
export AUTOMATIONBENCH_LLM_API_KEY="$AGENT_API_KEY"
export AUTOMATIONBENCH_LLM_TEMPERATURE=0
# Route upstream's chatgpt_* / salesforce-AI tools to the same vLLM so they do
# not 401 against api.openai.com.
export OPENAI_BASE_URL="$AGENT_BASE_URL"
export OPENAI_API_KEY="$AGENT_API_KEY"

# --- optional queueing --------------------------------------------------------
if [[ -n "${WAIT_FOR_PID:-}" ]]; then
  echo "[queue] waiting for PID $WAIT_FOR_PID to exit..."
  while kill -0 "$WAIT_FOR_PID" 2>/dev/null; do sleep 60; done
  echo "[queue] PID $WAIT_FOR_PID gone at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi

# --- config enumeration -------------------------------------------------------
configs=()   # "S H LOOP"
add_config() {
  local c="$1 $2 $3"
  for existing in "${configs[@]:-}"; do [[ "$existing" == "$c" ]] && return; done
  configs+=("$c")
}
if [[ "$GRID" == "1" ]]; then
  for s in $SAMPLE_LEVELS; do for h in $HORIZON_LEVELS; do for l in $LOOP_LEVELS; do
    add_config "$s" "$h" "$l"
  done; done; done
else
  add_config "$CENTRE_S" "$CENTRE_H" "$CENTRE_LOOP"
  for s in $SAMPLE_LEVELS;  do add_config "$s" "$CENTRE_H" "$CENTRE_LOOP"; done
  for h in $HORIZON_LEVELS; do add_config "$CENTRE_S" "$h" "$CENTRE_LOOP"; done
  for l in $LOOP_LEVELS;    do add_config "$CENTRE_S" "$CENTRE_H" "$l"; done
fi

bench_slug() { case "$1" in EnterpriseOps-Gym) echo eops ;; AutomationBench) echo ab ;; *) echo "$1" | tr 'A-Z' 'a-z' ;; esac; }
loop_slug()  { case "$1" in open_loop) echo open ;; closed_loop) echo closed ;; *) echo "$1" ;; esac; }

base_command() {
  local bench="$1" loop="$2" execute="$3" target="$4"
  local common=(
    ejepa bench run "$bench" --executor mcp_react
    --wm-ewm-jepa-checkpoint "$JEPA_CHECKPOINT"
    --wm-jepa-observation-backend canonical_event
    --wm-beam-plan-terminal-advice --wm-beam-plan-terminal-advice-threshold 0.75
    --wm-imagined-rollout-mode "$loop"
    --wm-beam-mpc-execute-steps "$execute"
    --config task_limit=0 --config capture_trajectory=true
    --config max_parallel="$MAX_PARALLEL"
  )
  case "$bench" in
    EnterpriseOps-Gym)
      printf '%q ' "${common[@]}" --config target="$target" --config model_name="$AGENT_MODEL" ;;
    AutomationBench)
      printf '%q ' "${common[@]}" --config target="$target" --config toolset=limited_zapier --config max_turns=50 ;;
  esac
}

# --- plan --------------------------------------------------------------------
# Each benchmark contributes configs x repeats x targets runs.
repeats_for() { case "$1" in EnterpriseOps-Gym) echo "$EOPS_REPEATS" ;; AutomationBench) echo "$AB_REPEATS" ;; *) echo "$REPEATS" ;; esac; }
targets_for() { case "$1" in EnterpriseOps-Gym) echo "$EOPS_TARGET" ;; AutomationBench) echo "$AB_TARGETS" ;; *) echo default ;; esac; }
total=0
for bench in $BENCHMARKS; do total=$(( total + ${#configs[@]} * $(repeats_for "$bench") * $(wc -w <<<"$(targets_for "$bench")") )); done
echo "=== beam-search planning ablation ($( [[ "$GRID" == 1 ]] && echo full grid || echo OFAT ), ${#configs[@]} configs; repeats EOPS=$EOPS_REPEATS AB=$AB_REPEATS; $total runs) ==="
echo "agent      : $AGENT_MODEL @ $AGENT_BASE_URL"
echo "world model: JEPA $JEPA_CHECKPOINT on GPU $JEPA_GPU"
echo "fixed      : execute_steps=$EXECUTE_STEPS margin=$SCORE_MARGIN temp=$TEMPERATURE max_parallel=$MAX_PARALLEL"
echo "targets    : EnterpriseOps-Gym=$EOPS_TARGET  AutomationBench=$AB_TARGETS"
for c in "${configs[@]}"; do echo "  config: samples=${c%% *}  horizon=$(cut -d' ' -f2 <<<"$c")  $(cut -d' ' -f3 <<<"$c")"; done
echo

# --- preflight: fail fast instead of producing 12 zero-score "completed" runs --
preflight_ok=1
for bench in $BENCHMARKS; do
  case "$bench" in
    EnterpriseOps-Gym) [[ -d "$ENTERPRISEOPS_GYM_REPO_PATH/benchmark" ]] || { echo "PREFLIGHT: EnterpriseOps-Gym repo not found at $ENTERPRISEOPS_GYM_REPO_PATH"; preflight_ok=0; } ;;
    AutomationBench)   [[ -d "$AUTOMATIONBENCH_REPO_PATH/automationbench" ]] || { echo "PREFLIGHT: AutomationBench repo not found at $AUTOMATIONBENCH_REPO_PATH"; preflight_ok=0; } ;;
  esac
done
[[ -d "$JEPA_CHECKPOINT" ]] || { echo "PREFLIGHT: JEPA checkpoint not found at $JEPA_CHECKPOINT"; preflight_ok=0; }
if ! curl -s -m 15 "$AGENT_BASE_URL/models" | grep -q "\"$AGENT_MODEL\""; then
  echo "PREFLIGHT: agent endpoint $AGENT_BASE_URL does not list model '$AGENT_MODEL'"; preflight_ok=0
fi
if [[ "$preflight_ok" != 1 ]]; then
  echo "PREFLIGHT FAILED -- nothing launched."; [[ "${DRY_RUN:-0}" == "1" ]] || exit 2
fi
echo "preflight  : OK (repos, checkpoint, agent endpoint)"
echo

# Quarantine a summary whose run finished implausibly fast with a zero score:
# that is the signature of every task failing at startup, not of a real result.
# Left in place it would also make the resume logic skip the config as "done".
MIN_RUN_SECONDS="${MIN_RUN_SECONDS:-120}"
check_summary() {
  local label="$1" f
  f=$(compgen -G "$SUMMARY_DIR/ejepa-wm-harnesses-${label}-*.json" | tail -1) || return 0
  [[ -n "$f" ]] || return 0
  python3 - "$f" "$MIN_RUN_SECONDS" <<'PY' || return 1
import json, sys
f, min_s = sys.argv[1], float(sys.argv[2])
for r in json.load(open(f)).get("runs") or []:
    s = r.get("result_summary") or {}
    score = s.get("score_rate")
    elapsed = float(r.get("wrapper_elapsed_seconds") or 0)
    per_task = s.get("per_task") or []
    # every task erroring out (CUDA OOM, missing repo, dead endpoint) is an
    # infrastructure failure, not a result -- and it can take minutes, so a
    # duration threshold alone is not enough.
    broken = [t for t in per_task if "executor error" in str(t.get("reason") or "") or t.get("error")]
    if per_task and len(broken) >= 0.9 * len(per_task):
        print(f"  suspicious: {len(broken)}/{len(per_task)} tasks failed in the executor -> quarantined")
        print(f"  first reason: {str(broken[0].get('reason') or broken[0].get('error'))[:160]}")
        sys.exit(1)
    if (score in (0, 0.0, None)) and elapsed < min_s:
        print(f"  suspicious: score={score} elapsed={elapsed:.0f}s (< {min_s:.0f}s) -> quarantined")
        sys.exit(1)
PY
}

# --- run ----------------------------------------------------------------------
mkdir -p "$SUMMARY_DIR/failed"
run_index=0
for bench in $BENCHMARKS; do
  for target in $(targets_for "$bench"); do
  for rep in $(seq 1 "$(repeats_for "$bench")"); do
    for c in "${configs[@]}"; do
      read -r s h loop <<<"$c"
      run_index=$((run_index + 1))
      slug="$(bench_slug "$bench")"
      # AutomationBench labels carry the domain; EnterpriseOps-Gym has one target.
      [[ "$bench" == AutomationBench ]] && slug="$slug-$target"
      label="beam-ablation-${slug}-s${s}-h${h}-e${EXECUTE_STEPS}-$(loop_slug "$loop")-r${rep}"
      # legacy label (before the domain segment) for the original operations runs
      legacy="beam-ablation-$(bench_slug "$bench")-s${s}-h${h}-e${EXECUTE_STEPS}-$(loop_slug "$loop")-r${rep}"
      if compgen -G "$SUMMARY_DIR/ejepa-wm-harnesses-${label}-*.json" >/dev/null || \
         { [[ "$bench" == AutomationBench && "$target" == operations ]] && compgen -G "$SUMMARY_DIR/ejepa-wm-harnesses-${legacy}-*.json" >/dev/null; }; then
        echo "[$run_index/$total] SKIP $label (summary exists)"; continue
      fi
      # SSoT diversity only applies to open-loop candidate generation.
      diversity=(); [[ "$loop" == "open_loop" ]] && diversity=(--ssot-diversity)
      cmd="python scripts/run_wm_harnesses.py --label $label --harnesses beam_interval \
        --beam-samples $s --beam-horizon $h --beam-execute-steps $EXECUTE_STEPS \
        --beam-score-margin $SCORE_MARGIN --beam-temperature $TEMPERATURE \
        --beam-refinement-rounds $REFINEMENT_ROUNDS --beam-refinement-top-k $REFINEMENT_TOP_K \
        ${diversity[*]} --no-beam-hard-override -- $(base_command "$bench" "$loop" "$EXECUTE_STEPS" "$target")"
      echo "[$run_index/$total] $(date -u +%H:%M:%SZ) START $label"
      if [[ "${DRY_RUN:-0}" == "1" ]]; then echo "  $cmd"; continue; fi
      log="$LOG_DIR/${label}.log"
      if bash -c "$cmd" > "$log" 2>&1 && check_summary "$label"; then
        echo "[$run_index/$total] $(date -u +%H:%M:%SZ) OK    $label"
      else
        echo "[$run_index/$total] $(date -u +%H:%M:%SZ) FAIL  $label (see $log)"
        # keep the evidence, but out of the summarizer's and the resume check's way
        for f in "$SUMMARY_DIR"/ejepa-wm-harnesses-"${label}"-*.json; do
          [[ -f "$f" ]] && mv "$f" "$SUMMARY_DIR/failed/"
        done
        grep -m1 -oE '"reason": "[^"]{0,160}' "$log" | sed 's/^/  first reason: /'
      fi
    done
  done
  done
done
echo "=== done. Summarise with: python scripts/summarize_beam_ablation.py ==="
