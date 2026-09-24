#!/usr/bin/env bash
# Reproducible launcher: crmarenapro baseline (baseline_crm_agent) against a
# node-local Qwen3.6-27B vLLM server, N independent runs, one Slurm node each.
#
# Usage (from any checkout of this repo, on a machine with ssh access to the
# cluster login node):
#
#   scripts/run_crmarenapro_qwen36_baseline_slurm.sh                # 5 runs
#   NUM_RUNS=3 TASK_LIMIT=50 scripts/run_crmarenapro_qwen36_baseline_slurm.sh
#
# Each run is a self-contained sbatch job (slurm/serve-vllm-and-run-benchmark.sbatch.sh)
# that boots vLLM on its own 8-GPU node, waits for readiness, and runs
#   ejepa bench run crmarenapro --executor baseline_crm_agent
# Jobs are fully detached: closing this shell / ssh session does not kill them.
#
# Tunables (env):
#   CLUSTER_HOST   ssh alias of the cluster login node   (default gpu-cluster)
#   PARTITION      Slurm partition                       (default batch-8gpu-short)
#   NUM_RUNS       number of baseline runs               (default 5)
#   TASK_LIMIT     tasks per run (seeded sample, seed=42 (default 100)
#                  fixed in the green agent, so all runs share the task set)
#   TEMPERATURE    executor sampling temperature         (default 0.1)
#   BRANCH         git branch to run                     (default crmarenapro-qwen36-baseline)
#   REMOTE_BASE    benchmarks checkout on the cluster    (default ~/repo/benchmarks)
#   REMOTE_WORKTREE dedicated worktree for these runs    (default ~/repo/benchmarks-crmarenapro)
#   SLURM_TIME     per-job time limit                    (default 05:59:00, partition cap 6h)
#   SKIP_SETUP=1   skip remote fetch/worktree/install (resubmit only)

set -euo pipefail

CLUSTER_HOST="${CLUSTER_HOST:-gpu-cluster}"
PARTITION="${PARTITION:-batch-8gpu-short}"
NUM_RUNS="${NUM_RUNS:-5}"
TASK_LIMIT="${TASK_LIMIT:-100}"
TEMPERATURE="${TEMPERATURE:-0.1}"
BRANCH="${BRANCH:-crmarenapro-qwen36-baseline}"
REMOTE_BASE="${REMOTE_BASE:-\$HOME/repo/benchmarks}"
REMOTE_WORKTREE="${REMOTE_WORKTREE:-\$HOME/repo/benchmarks-crmarenapro}"
SLURM_TIME="${SLURM_TIME:-05:59:00}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

echo "=== crmarenapro Qwen3.6-27B baseline: ${NUM_RUNS} runs on ${CLUSTER_HOST}/${PARTITION} ==="

if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  echo "--- [1/2] preparing remote worktree (${BRANCH}) + environments ---"
  # shellcheck disable=SC2087
  ssh "$CLUSTER_HOST" bash -s <<EOF
set -euo pipefail
BASE="${REMOTE_BASE}"
WT="${REMOTE_WORKTREE}"
git -C "\$BASE" fetch origin "${BRANCH}"
if [[ ! -d "\$WT" ]]; then
  git -C "\$BASE" worktree add "\$WT" -B "${BRANCH}" "origin/${BRANCH}"
else
  git -C "\$WT" fetch origin "${BRANCH}"
  git -C "\$WT" checkout "${BRANCH}"
  git -C "\$WT" reset --hard "origin/${BRANCH}"
fi
cd "\$WT"
# Pre-seed venvs with uv: system python3.12's ensurepip is broken on the
# gpu-cluster login node, so let uv create the venvs (with pip) and have
# install.sh only bootstrap + sync into them.
UV="\${UV:-\$HOME/.local/bin/uv}"
for d in . assets/crmarenapro/green assets/crmarenapro/purple; do
  [[ -x "\$d/.venv/bin/python" ]] || "\$UV" venv --python 3.12 --seed "\$d/.venv"
done
# root venv (ejepa CLI) + crmarenapro green/purple venvs + bundled DBs
bash scripts/install.sh crmarenapro
test -x .venv/bin/ejepa
test -x assets/crmarenapro/green/.venv/bin/python
test -x assets/crmarenapro/purple/.venv/bin/python
mkdir -p .cache/slurm-logs
echo "remote setup OK: \$WT @ \$(git -C "\$WT" rev-parse --short HEAD)"
EOF
fi

echo "--- [2/2] submitting ${NUM_RUNS} sbatch jobs (one node each, detached) ---"
for run in $(seq 1 "$NUM_RUNS"); do
  tag="qwen36_baseline_${RUN_STAMP}_r${run}"
  # shellcheck disable=SC2087
  ssh "$CLUSTER_HOST" bash -s <<EOF
set -euo pipefail
WT="${REMOTE_WORKTREE}"
cd "\$WT"
sbatch \
  --job-name="crmpro-q36-r${run}" \
  --partition="${PARTITION}" \
  --time="${SLURM_TIME}" \
  --chdir="\$WT" \
  --output="\$WT/.cache/slurm-logs/%x-%j.out" \
  --export=NONE,HOME="\$HOME",USER="\$USER",\
BENCHMARK_SLURM_BENCHMARK_NAME=crmarenapro,\
BENCHMARK_SLURM_EXECUTOR_NAME=baseline_crm_agent,\
BENCHMARK_SLURM_CONFIG="task_limit=${TASK_LIMIT}",\
BENCHMARK_SLURM_RUN_TAG="${tag}",\
BENCHMARK_SLURM_TEMPERATURE="${TEMPERATURE}",\
BENCHMARK_SLURM_REPO_ROOT="\$WT" \
  slurm/serve-vllm-and-run-benchmark.sbatch.sh
EOF
done

cat <<EOF

Submitted. Monitor with:
  ssh ${CLUSTER_HOST} squeue -u \\\$USER -n $(for r in $(seq 1 "$NUM_RUNS"); do printf "crmpro-q36-r%s," "$r"; done | sed 's/,$//')
Logs:    ${REMOTE_WORKTREE}/.cache/slurm-logs/
Results: ${REMOTE_WORKTREE}/.cache/bench-results/crmarenapro/qwen36_baseline_${RUN_STAMP}_r{1..${NUM_RUNS}}/
EOF
