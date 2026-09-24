#!/usr/bin/env bash
# Smoke-run Terminal-Bench-2.0 sample task (fix-git) with Azure GPT-5.5.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TASK_ID="${TASK_ID:-fix-git}"
EXECUTOR="${EXECUTOR:-llm_shell}"
EJEPA="${EJEPA:-$ROOT/ejepa}"

# Local machine: avoid ejepa default $BENCHMARK_HOME (cluster-only).
export BENCHMARK_HOME="$ROOT/.cache/benchmark home"
export TERMINAL_BENCH_WORKSPACE="${TERMINAL_BENCH_WORKSPACE:-$ROOT/.cache/terminal-bench-workspace}"
mkdir -p "$BENCHMARK_HOME" "$TERMINAL_BENCH_WORKSPACE"
export BENCHMARK_A2A_CLIENT_TIMEOUT="${BENCHMARK_A2A_CLIENT_TIMEOUT:-1800}"

# Azure GPT-5.5 (unset inference / OpenAI-compatible overrides).
unset OPENAI_BASE_URL OPENAI_API_BASE_URL INFERENCE_DEFAULT_BASE_URL INFERENCE_DEFAULT_MODEL 2>/dev/null || true

export AZURE_OPENAI_DEPLOYMENT_NAME="${AZURE_OPENAI_DEPLOYMENT_NAME:-gpt-5.5}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-10-21}"
export AZURE_OPENAI_REASONING_EFFORT="${AZURE_OPENAI_REASONING_EFFORT:-low}"
export TERMINAL_BENCH_LLM_MAX_TOKENS="${TERMINAL_BENCH_LLM_MAX_TOKENS:-16384}"

if [[ ! -d "$ROOT/assets/Terminal-Bench-2.0/tasks/terminal-bench-2/$TASK_ID" ]]; then
  echo "Task repo missing. Clone with:" >&2
  echo "  git clone --depth 1 https://github.com/laude-institute/terminal-bench-2.git \\" >&2
  echo "    assets/Terminal-Bench-2.0/tasks/terminal-bench-2" >&2
  exit 1
fi

"$ROOT/scripts/check-docker-access.sh"

exec "$EJEPA" bench run Terminal-Bench-2.0 \
  --executor "$EXECUTOR" \
  --task-id "$TASK_ID" \
  --config capture_trajectories=true \
  --ready-timeout 600 \
  --show-logs
