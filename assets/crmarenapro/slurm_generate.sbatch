#!/usr/bin/env bash
# Slurm array job: sharded multi-temperature trajectory generation for crmarenapro.
#
# One array task = one (temp, shard) cell of a single split, on its own GPU node
# with a local vLLM (default Qwen3.6-27B). Inside the cell, REPS repetitions run
# sequentially, each a full `ejepa bench run crmarenapro` (capture_trajectory=true)
# over the shard's task window [offset, offset+shard_size) of the split
# (crmarena_train / crmarena_valid — test-excluded, see green/crm/splits.py).
#
# Sharding keeps each cell small enough to finish inside a 6h wall
# (batch-8gpu-short), and every (shard, rep) is independently resumable.
#
# Array index mapping (single split per submission):
#   TEMP  = TEMPS[ ARRAY_ID / NUM_SHARDS ]
#   SHARD = ARRAY_ID % NUM_SHARDS      ; offset = SHARD * SHARD_SIZE
#
# Resumable: a (shard, rep) whose manifest.json has status=completed is skipped.
#
# Required env (set by submit_generate.sh):
#   CRMGEN_SPLIT       single split name, e.g. "train" or "valid"
#   CRMGEN_TEMPS       e.g. "0.2 0.4 0.6 0.8 1.0 1.2"
#   CRMGEN_NUM_SHARDS  number of shards for this split
#   CRMGEN_SHARD_SIZE  tasks per shard
#   CRMGEN_REPS        repetitions per cell (default 5)
# Optional:
#   CRMGEN_LEADERBOARD    1 -> leaderboard_mode (drift/rot off, MAX_TURNS=20)
#   CLUSTER_*            vLLM overrides as in serve-vllm-and-run-benchmark
#   CRMGEN_OUT_ROOT       results root (default <repo>/.cache/crmgen)
#   CRMGEN_EXECUTOR       purple executor (default baseline_crm_agent)

#SBATCH --job-name=crmgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=64G
#SBATCH --time=23:59:00
#SBATCH --output=slurm-%x-%A_%a.out
#SBATCH --export=NONE

set -euo pipefail

if ! command -v module >/dev/null 2>&1 && [[ -r /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  for m in ${CRMGEN_MODULES:-rocm/7.2.0}; do
    module load "$m" 2>/dev/null || echo "[warn] module load $m failed" >&2
  done
fi

REPO_ROOT="${CRMGEN_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_ROOT"

ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"

SPLIT="${CRMGEN_SPLIT:?CRMGEN_SPLIT required}"
read -r -a TEMPS <<<"${CRMGEN_TEMPS:?CRMGEN_TEMPS required}"
REPS="${CRMGEN_REPS:-5}"
NUM_SHARDS="${CRMGEN_NUM_SHARDS:?CRMGEN_NUM_SHARDS required}"
SHARD_SIZE="${CRMGEN_SHARD_SIZE:?CRMGEN_SHARD_SIZE required}"
cell_count=$(( ${#TEMPS[@]} * NUM_SHARDS ))
if (( ARRAY_ID >= cell_count )); then
  echo "array id ${ARRAY_ID} out of range (cells=${cell_count})" >&2
  exit 2
fi
TEMP="${TEMPS[$(( ARRAY_ID / NUM_SHARDS ))]}"
SHARD=$(( ARRAY_ID % NUM_SHARDS ))
OFFSET=$(( SHARD * SHARD_SIZE ))
SHARD_TAG="$(printf 'shard%02d' "$SHARD")"
TARGET="crmarena_${SPLIT}"
EXECUTOR="${CRMGEN_EXECUTOR:-baseline_crm_agent}"
OUT_ROOT="${CRMGEN_OUT_ROOT:-${REPO_ROOT}/.cache/crmgen}"

WORK_ROOT="${REPO_ROOT}/.cache/crmgen-slurm-${JOB_ID}_${ARRAY_ID}"
mkdir -p "$WORK_ROOT"
export WORK_ROOT

MAS_BIN="${CRMGEN_MAS_BIN:-${REPO_ROOT}/.venv/bin/ejepa}"
[[ -x "$MAS_BIN" ]] || { echo "ejepa not found: $MAS_BIN" >&2; exit 1; }

# --- node-local vLLM (same defaults as slurm/serve-vllm-and-run-benchmark) ---
export CLUSTER_VLLM_PORT="${CLUSTER_VLLM_PORT:-8000}"
export CLUSTER_VLLM_MODEL_ID="${CLUSTER_VLLM_MODEL_ID:-Qwen/Qwen3.6-27B}"
export CLUSTER_LLM_MODEL_NAME="${CLUSTER_LLM_MODEL_NAME:-Qwen3.6-27B}"
export CLUSTER_LLM_API_BASE="${CLUSTER_LLM_API_BASE:-http://127.0.0.1:${CLUSTER_VLLM_PORT}/v1}"
export CLUSTER_VLLM_READY_TIMEOUT="${CLUSTER_VLLM_READY_TIMEOUT:-2700}"
export CLUSTER_VLLM_CHAT_READY_TIMEOUT="${CLUSTER_VLLM_CHAT_READY_TIMEOUT:-300}"
export CLUSTER_VLLM_GPU_MEMORY_UTILIZATION="${CLUSTER_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export CLUSTER_VLLM_MAX_MODEL_LEN="${CLUSTER_VLLM_MAX_MODEL_LEN:-131072}"
export CLUSTER_VLLM_TOOL_CALL_PARSER="${CLUSTER_VLLM_TOOL_CALL_PARSER:-qwen3_xml}"
export CLUSTER_STOP_VLLM_ON_EXIT="${CLUSTER_STOP_VLLM_ON_EXIT:-1}"
if [[ -z "${CLUSTER_VLLM_BIN:-}" && -x "${HOME}/repo/servers/.venv/bin/vllm" ]]; then
  export CLUSTER_VLLM_BIN="${HOME}/repo/servers/.venv/bin/vllm"
fi

VLLM_PID=""
source "${REPO_ROOT}/assets/crmarenapro/slurm_vllm_lib.sh"
trap stop_local_vllm EXIT
start_local_vllm
wait_for_vllm_chat_completion

# --- executor-facing LLM env -------------------------------------------------
export LLM_PROVIDER=openai
export LLM_BASE_URL="${CLUSTER_LLM_API_BASE}"
export LLM_MODEL="${CLUSTER_LLM_MODEL_NAME}"
export OPENAI_BASE_URL="${CLUSTER_LLM_API_BASE}"
export OPENAI_MODEL_NAME="${CLUSTER_LLM_MODEL_NAME}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export TEMPERATURE="${TEMP}"
unset ANTHROPIC_API_KEY AZURE_OPENAI_API_KEY AZURE_OPENAI_ENDPOINT 2>/dev/null || true

# Shard window into the split. task_limit MUST be explicit (benchmark.toml
# defaults task_limit=1, which would otherwise truncate to a single task).
config_args=(--config "target=${TARGET}" --config capture_trajectory=true
             --config "task_offset=${OFFSET}" --config "task_limit=${SHARD_SIZE}")

# Leaderboard mode: disable adversarial drift/rot and raise the interaction
# budget (green max_steps + executor MAX_TURNS). Temperature still varies per
# cell (${TEMP}) — leaderboard_mode is orthogonal to the temperature sweep.
LB_MODE="off"
case "${CRMGEN_LEADERBOARD:-}" in
  1|true|yes|on|y|Y)
    config_args+=(--config leaderboard_mode=true)
    export MAX_TURNS="${MAX_TURNS:-20}"
    LB_MODE="on (MAX_TURNS=${MAX_TURNS})"
    ;;
esac

echo "=== crmgen cell: split=${SPLIT} temp=${TEMP} ${SHARD_TAG} offset=${OFFSET} size=${SHARD_SIZE} reps=${REPS} leaderboard=${LB_MODE} node=$(hostname) job=${JOB_ID}_${ARRAY_ID} ==="

for REP in $(seq 1 "$REPS"); do
  rep_dir="${OUT_ROOT}/gen_${SPLIT}/run_t${TEMP}/${SHARD_TAG}/rep${REP}"
  # Resume guard: skip (shard, rep) that already completed.
  if compgen -G "${rep_dir}/bm-*/manifest.json" >/dev/null && \
     grep -l '"status": *"completed"' "${rep_dir}"/bm-*/manifest.json >/dev/null 2>&1; then
    echo "[skip] ${SPLIT} t=${TEMP} ${SHARD_TAG} rep${REP}: completed manifest exists"
    continue
  fi
  mkdir -p "$rep_dir"
  echo "[run] ${SPLIT} t=${TEMP} ${SHARD_TAG} rep${REP} -> ${rep_dir}"
  "$MAS_BIN" --result-root "$rep_dir" bench run crmarenapro \
    --launcher local --ready-timeout 600 \
    --executor "$EXECUTOR" \
    "${config_args[@]}"
done

echo "=== crmgen cell done: split=${SPLIT} temp=${TEMP} ==="
