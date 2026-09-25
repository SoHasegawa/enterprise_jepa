#!/usr/bin/env bash
# "Is the planner doing the work?" controls, all benchmarks.
#
# The identical beam_interval planner (s=8, h=3, execute 2, open loop -- the paper's
# configuration) with the JEPA world model's per-step predictions replaced by
#   no_state  no predicted state reaches the planner at all (paper Table 4,
#             "No world model feedback")
#   shuffled  the model's own rows permuted across (plan, step) within each call
#             (paper Table 4, "Shuffled predictions")
#   uniform   1/K over every field's classes                    (not reported)
#   prior     training-set class priors                          (not reported)
#             (results/analysis/canonical_event_class_priors.json)
#
# Paper Table 4 is the EnterpriseOps-Gym column of the first two modes:
#   MODES="no_state shuffled" BENCHES=EnterpriseOps-Gym scripts/run_prediction_controls.sh
# (WM_JEPA_PREDICTION_CONTROL in src/ejepa_wm/backends/_ewm_jepa.py). Each benchmark
# uses the agent/target/settings of the real-prediction run it is compared with:
#   EnterpriseOps-Gym  opsgym_80_test, Qwen3.6-27B @18043, mp=3   (x EOPS_REPEATS)
#   AutomationBench    sales/operations/support, Qwen3.6-27B @18043, mp=3 (per domain)
#   WorkBench          all (690), Qwen3.6-27B @18043, mp=3, max_turns=50
#   crmarenapro        world_model_test (428), wm_agent @9010, mp=5, leaderboard_mode
# Labels: control-<bench>[-<domain>]-<mode>-s8-h3-r<N>. Resumable (existing summary = skip).
# Order: AutomationBench, WorkBench, EnterpriseOps-Gym repeats, crmarenapro.
#
#   DRY_RUN=1 scripts/run_prediction_controls.sh
#   nohup scripts/run_prediction_controls.sh > results/logs/beam_ablation/controls.out 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
MODES="${MODES:-no_state shuffled}"
BENCHES="${BENCHES:-EnterpriseOps-Gym}"
EOPS_REPEATS="${EOPS_REPEATS:-3}"
AB_TARGETS="${AB_TARGETS:-sales operations support}"
S="${S:-8}"; H="${H:-3}"; E="${E:-2}"
# Parallelism per benchmark. Controls are success-only, so these may exceed the
# mp of the real-prediction runs when the agent server's time budget is short.
EOPS_MP="${EOPS_MP:-3}"; AB_MP="${AB_MP:-3}"; WB_MP="${WB_MP:-3}"; CRM_MP="${CRM_MP:-5}"
QWEN_MODEL="${QWEN_MODEL:-Qwen3.6-27B}"; QWEN_URL="${QWEN_URL:-http://127.0.0.1:18043/v1}"
CRM_MODEL="${CRM_MODEL:-wm_agent}"; CRM_URL="${CRM_URL:-http://127.0.0.1:9010/v1}"
JEPA_CHECKPOINT="${JEPA_CHECKPOINT:-checkpoints/jepa}"
export WM_JEPA_PRIOR_PATH="${WM_JEPA_PRIOR_PATH:-$PWD/results/analysis/canonical_event_class_priors.json}"
export CUDA_VISIBLE_DEVICES="${JEPA_GPU:-7}" WM_SHARE_MODEL_WEIGHTS=1
# EnterpriseOps-Gym
export ENTERPRISEOPS_GYM_REPO_PATH="${ENTERPRISEOPS_GYM_REPO_PATH:-$PWD/upstreams/EnterpriseOps-Gym}"
export ENTERPRISEOPS_LLM_PROVIDER=vllm ENTERPRISEOPS_LLM_API_ENDPOINT="$QWEN_URL" ENTERPRISEOPS_LLM_MODEL="$QWEN_MODEL" ENTERPRISEOPS_LLM_API_KEY=EMPTY
# AutomationBench
export AUTOMATIONBENCH_REPO_PATH="${AUTOMATIONBENCH_REPO_PATH:-$PWD/upstreams/AutomationBench}"
export AUTOMATIONBENCH_LLM_MODEL="$QWEN_MODEL" AUTOMATIONBENCH_LLM_API_ENDPOINT="$QWEN_URL" AUTOMATIONBENCH_LLM_API_KEY=EMPTY AUTOMATIONBENCH_LLM_TEMPERATURE=0
export OPENAI_BASE_URL="$QWEN_URL" OPENAI_API_KEY=EMPTY
# WorkBench
export WORKBENCH_VLLM_BASE_URL="$QWEN_URL" WORKBENCH_VLLM_MODEL="$QWEN_MODEL" WORKBENCH_VLLM_API_KEY=EMPTY
# crmarenapro
export LLM_PROVIDER=openai_compatible LLM_MODEL="$CRM_MODEL" LLM_BASE_URL="$CRM_URL" LLM_API_KEY=EMPTY MAX_TURNS=20

LOG_DIR=results/logs/beam_ablation; SUMMARY_DIR=results/wm_harness_summaries; mkdir -p "$LOG_DIR"
case " $MODES " in
  *" prior "*) [[ -f "$WM_JEPA_PRIOR_PATH" ]] ||
    { echo "priors file missing: $WM_JEPA_PRIOR_PATH"; exit 2; } ;;
esac
need=()
[[ " $BENCHES " == *"crmarenapro"* ]] && need+=("$CRM_URL|$CRM_MODEL")
[[ " $BENCHES " == *"EnterpriseOps-Gym"* || " $BENCHES " == *"AutomationBench"* || " $BENCHES " == *"WorkBench"* ]] && need+=("$QWEN_URL|$QWEN_MODEL")
for u in "${need[@]}"; do
  curl -s -m 15 "${u%%|*}/models" | grep -q "\"${u##*|}\"" || { echo "PREFLIGHT: ${u##*|} not served at ${u%%|*}"; [[ "${DRY_RUN:-0}" == 1 ]] || exit 2; }
done

base_command() {  # bench target -> ejepa bench run ... (printf %q-quoted)
  local bench="$1" target="$2"
  local common=(ejepa bench run "$bench" --executor mcp_react
    --wm-ewm-jepa-checkpoint "$JEPA_CHECKPOINT" --wm-jepa-observation-backend canonical_event
    --wm-beam-plan-terminal-advice --wm-beam-plan-terminal-advice-threshold 0.75
    --wm-imagined-rollout-mode open_loop --wm-beam-mpc-execute-steps "$E"
    --config task_limit=0 --config capture_trajectory=true)
  case "$bench" in
    EnterpriseOps-Gym) printf '%q ' "${common[@]}" --config target="$target" --config max_parallel="$EOPS_MP" --config model_name="$QWEN_MODEL" ;;
    AutomationBench)   printf '%q ' "${common[@]}" --config target="$target" --config toolset=limited_zapier --config max_turns=50 --config max_parallel="$AB_MP" ;;
    WorkBench)         printf '%q ' "${common[@]}" --config target="$target" --config model_name="$QWEN_MODEL" --config max_turns=50 --config max_parallel="$WB_MP" --config timeout=3600 ;;
    crmarenapro)       printf '%q ' "${common[@]}" --config target="$target" --config max_parallel="$CRM_MP" --config leaderboard_mode=true --config timeout=3600 --config max_steps=20 ;;
  esac
}
slug() { case "$1" in EnterpriseOps-Gym) echo eops ;; AutomationBench) echo ab ;; WorkBench) echo wb ;; crmarenapro) echo crm ;; esac; }

run_one() {  # label bench target mode
  local label="$1" bench="$2" target="$3" mode="$4"
  if compgen -G "$SUMMARY_DIR/ejepa-wm-harnesses-${label}-*.json" >/dev/null; then echo "SKIP  $label (summary exists)"; return; fi
  echo "=== $(date -u +%H:%M:%SZ) START $label"
  [[ "${DRY_RUN:-0}" == 1 ]] && return
  # Per-arm preflight: a served endpoint can disappear mid-sweep (both Qwen tunnels
  # died at 12:36Z on 2026-09-18), after which every remaining arm "completes" in 60 s
  # with all tasks erroring, rc=0, and a summary that the resume logic then skips.
  local url="$QWEN_URL" model="$QWEN_MODEL"
  if [ "$bench" = crmarenapro ]; then url="$CRM_URL"; model="$CRM_MODEL"; fi
  if ! curl -s -m 15 "$url/models" | grep -q "\"$model\""; then
    echo "=== $(date -u +%H:%M:%SZ) ABORT $label: $model not served at $url"
    return 1
  fi
  WM_JEPA_PREDICTION_CONTROL="$mode" bash -c "python scripts/run_wm_harnesses.py --label $label --harnesses beam_interval \
    --beam-samples $S --beam-horizon $H --beam-execute-steps $E --beam-score-margin 0.10 --beam-temperature 0.7 \
    --beam-refinement-rounds 1 --beam-refinement-top-k 4 --ssot-diversity --no-beam-hard-override -- $(base_command "$bench" "$target")" \
    > "$LOG_DIR/$label.log" 2>&1
  echo "=== $(date -u +%H:%M:%SZ) DONE  $label rc=$?"
  # Quarantine a run whose tasks all failed in the executor, so it is redone rather
  # than silently treated as a result (and skipped on resume).
  local f
  f=$(compgen -G "$SUMMARY_DIR/ejepa-wm-harnesses-${label}-*.json" | tail -1) || return 0
  [ -n "${f:-}" ] || return 0
  python3 - "$f" <<'PYEOF' || { mkdir -p "$SUMMARY_DIR/failed"; mv "$f" "$SUMMARY_DIR/failed/"; echo "=== quarantined $label (all tasks failed)"; }
import json,sys
for r in json.load(open(sys.argv[1])).get("runs") or []:
    pt=(r.get("result_summary") or {}).get("per_task") or []
    broken=[t for t in pt if "executor error" in str(t.get("reason") or "") or t.get("error")]
    if pt and len(broken) >= 0.5*len(pt):
        print("   ", len(broken), "of", len(pt), "tasks failed:", str(broken[0].get("reason"))[:120]); sys.exit(1)
PYEOF
}

# Order (user request 2026-09-18): AutomationBench -> WorkBench first; EnterpriseOps-Gym
# repeats next; crmarenapro (428 tasks, slowest) last.
has() { [[ " $BENCHES " == *" $1 "* ]]; }
has AutomationBench && for d in $AB_TARGETS; do for mode in $MODES; do run_one "control-ab-${d}-${mode}-s${S}-h${H}-r1" AutomationBench "$d" "$mode"; done; done
has WorkBench && for mode in $MODES; do run_one "control-wb-${mode}-s${S}-h${H}-r1" WorkBench all "$mode"; done
has EnterpriseOps-Gym && for rep in $(seq 1 "$EOPS_REPEATS"); do for mode in $MODES; do run_one "control-eops-${mode}-s${S}-h${H}-r${rep}" EnterpriseOps-Gym opsgym_80_test "$mode"; done; done
has crmarenapro && for mode in $MODES; do run_one "control-crm-${mode}-s${S}-h${H}-r1" crmarenapro world_model_test "$mode"; done
echo "=== all controls done $(date -u +%H:%M:%SZ) ==="
