#!/usr/bin/env bash
# Submit the crmarenapro sharded multi-temperature trajectory-generation array.
#
# ONE split per invocation. The split is sharded into fixed-size task windows so
# each array cell (one temp x one shard, REPS runs looped in-cell) finishes
# inside a 6h wall on batch-8gpu-short. Every (shard, rep) is resumable, so
# re-running the same command continues where a killed/failed cell left off.
#
# Run ON the cluster login node, from the repo checkout to use:
#
#   assets/crmarenapro/submit_generate.sh \
#     --split valid --temps "0.2 0.4 0.6 0.8 1.0 1.2" \
#     --reps 5 --shard-size 60 --leaderboard \
#     --array-throttle 8 --partition batch-8gpu-short --slurm-time 05:59:00
#
#   assets/crmarenapro/submit_generate.sh \
#     --split train --temps "0.2 0.4 0.6 0.8 1.0 1.2" \
#     --reps 5 --shard-size 60 --leaderboard \
#     --array-throttle 8 --partition batch-8gpu-short --slurm-time 05:59:00
#
# Cells = #temps x ceil(split_size / shard_size). Size a shard so
# shard_size * reps * per_task_seconds + vLLM_boot < wall. On batch-8gpu-short
# (6h): clean mode ~47 s/task -> shard_size 60, reps 5 => ~3.9h. Resubmit the
# same command any number of times to fill remaining (shard, rep) cells.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SPLIT="valid"
TEMPS="0.2 0.4 0.6 0.8 1.0 1.2"
REPS=5
SHARD_SIZE=50
ARRAY_THROTTLE=8
PARTITION="batch-8gpu-short"
SLURM_TIME="05:59:00"
EXECUTOR="baseline_crm_agent"
OUT_ROOT=""
LEADERBOARD=0
DRY_RUN=0

while (( "$#" )); do
  case "$1" in
    --split)          SPLIT="$2"; shift 2 ;;
    --splits)         SPLIT="$2"; shift 2 ;;   # alias; single split only
    --temps)          TEMPS="$2"; shift 2 ;;
    --reps)           REPS="$2"; shift 2 ;;
    --shard-size)     SHARD_SIZE="$2"; shift 2 ;;
    --array-throttle) ARRAY_THROTTLE="$2"; shift 2 ;;
    --partition)      PARTITION="$2"; shift 2 ;;
    --slurm-time)     SLURM_TIME="$2"; shift 2 ;;
    --executor)       EXECUTOR="$2"; shift 2 ;;
    --out-root)       OUT_ROOT="$2"; shift 2 ;;
    --leaderboard)    LEADERBOARD=1; shift ;;   # clean mode: drift/rot off, MAX_TURNS=20
    --max-parallel)   echo "[note] --max-parallel ignored: crmarenapro evaluates sequentially" >&2; shift 2 ;;
    --dry-run)        DRY_RUN=1; shift ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
done

read -r -a temp_arr <<<"$TEMPS"
[[ "$SPLIT" == "train" || "$SPLIT" == "valid" ]] || { echo "invalid --split: $SPLIT (train|valid)" >&2; exit 2; }
[[ "$SHARD_SIZE" -gt 0 ]] || { echo "--shard-size must be > 0" >&2; exit 2; }

# Exact split size from the same code the green agent uses (stdlib only).
SPLIT_SIZE="$(python3 - "$REPO_ROOT" "$SPLIT" <<'PY'
import sys, json, importlib.util, pathlib
repo = pathlib.Path(sys.argv[1]); split = sys.argv[2]
green = repo / "assets" / "crmarenapro" / "green"
spec = importlib.util.spec_from_file_location("splits", green / "crm" / "splits.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
corpus = json.load(open(green / "data" / "crmarena_b2b_tasks.json"))
ids = [t["idx"] for t in corpus]
print(len(m.resolve_split_task_ids(f"crmarena_{split}", ids)))
PY
)"
NUM_SHARDS=$(( (SPLIT_SIZE + SHARD_SIZE - 1) / SHARD_SIZE ))
array_max=$(( ${#temp_arr[@]} * NUM_SHARDS - 1 ))

mkdir -p "${REPO_ROOT}/.cache/slurm-logs"

cmd=(sbatch
  --job-name="crmgen-${SPLIT}"
  --partition="$PARTITION"
  --time="$SLURM_TIME"
  --array="0-${array_max}%${ARRAY_THROTTLE}"
  --chdir="$REPO_ROOT"
  --output="${REPO_ROOT}/.cache/slurm-logs/%x-%A_%a.out"
  --export="NONE,HOME=${HOME},USER=${USER},CRMGEN_SPLIT=${SPLIT},CRMGEN_TEMPS=${TEMPS},CRMGEN_REPS=${REPS},CRMGEN_NUM_SHARDS=${NUM_SHARDS},CRMGEN_SHARD_SIZE=${SHARD_SIZE},CRMGEN_REPO_ROOT=${REPO_ROOT},CRMGEN_EXECUTOR=${EXECUTOR}${OUT_ROOT:+,CRMGEN_OUT_ROOT=${OUT_ROOT}}$([[ "$LEADERBOARD" == 1 ]] && echo ',CRMGEN_LEADERBOARD=1')"
  "${SCRIPT_DIR}/slurm_generate.sbatch")

echo "split=${SPLIT} size=${SPLIT_SIZE} shard_size=${SHARD_SIZE} -> ${NUM_SHARDS} shards x ${#temp_arr[@]} temps = $((array_max + 1)) cells (throttle ${ARRAY_THROTTLE}), reps/cell=${REPS}, leaderboard=${LEADERBOARD}"
echo "${cmd[*]}"
if (( DRY_RUN )); then
  echo "[dry-run] not submitting"
  exit 0
fi
"${cmd[@]}"
