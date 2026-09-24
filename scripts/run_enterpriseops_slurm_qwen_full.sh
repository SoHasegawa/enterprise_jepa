#!/usr/bin/env bash
# Run the full EnterpriseOps-Gym Hugging Face corpus with Slurm remote Qwen3.5 inference.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Broken mirrors (e.g. hf-mirror) can make `datasets.load_dataset` fail; use default Hub.
unset HF_ENDPOINT 2>/dev/null || true

export ENTERPRISEOPS_GYM_REPO_PATH="${ENTERPRISEOPS_GYM_REPO_PATH:-$ROOT/.cache/EnterpriseOps-Gym}"
export BENCHMARK_HOME="${BENCHMARK_HOME:-$ROOT/.cache/benchmark home}"

# Remote vLLM via ejepa --inference-config (do not point at Azure or other endpoints).
unset ENTERPRISEOPS_LLM_API_ENDPOINT ENTERPRISEOPS_LLM_MODEL ENTERPRISEOPS_LLM_API_KEY
unset ENTERPRISEOPS_LLM_API_VERSION ENTERPRISEOPS_LLM_CONFIG_FILE
unset AZURE_OPENAI_API_KEY AZURE_OPENAI_ENDPOINT 2>/dev/null || true

export ENTERPRISEOPS_LLM_PROVIDER=vllm
export ENTERPRISEOPS_LLM_TEMPERATURE=0.0
export ENTERPRISEOPS_LLM_MAX_TOKENS=8192
export BENCHMARK_A2A_CLIENT_TIMEOUT="${BENCHMARK_A2A_CLIENT_TIMEOUT:-1800}"

EJEPA="${EJEPA:-$ROOT/.venv/bin/ejepa}"
INFERENCE_CONFIG="${INFERENCE_CONFIG:-$ROOT/inference-slurm-login.yaml}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
LOG_DIR="${LOG_DIR:-$ROOT/.cache/benchmark home/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/enterpriseops_slurm-login_qwen_full_$(date -u +%Y%m%dT%H%M%SZ).log}"

DOMAINS_JSON='["calendar","csm","drive","email","hr","hybrid","itsm","teams"]'
MCP_PORTS=(8001 8002 8003 8004 8006 8008 8009)

echo "=== EnterpriseOps-Gym full run (Slurm Qwen3.5) ==="
echo "Log file: $LOG_FILE"
echo "Checking MCP host ports..."
missing=0
for port in "${MCP_PORTS[@]}"; do
  code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:${port}/" || true)"
  if [[ -z "$code" || "$code" == "000" ]]; then
    echo "  port $port: FAIL (HTTP $code)"
    missing=1
  else
    echo "  port $port: OK (HTTP $code)"
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "Start all domain MCP containers before running. See assets/EnterpriseOps-Gym/README.md section 3." >&2
  exit 1
fi

exec > >(tee -a "$LOG_FILE") 2>&1
echo "Started at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

"$EJEPA" bench run EnterpriseOps-Gym \
  --executor mcp_react \
  --inference-config "$INFERENCE_CONFIG" \
  --config target=hf_dataset \
  --config mode=oracle \
  --config "domains=$DOMAINS_JSON" \
  --config capture_trajectory=true \
  --config max_parallel="$MAX_PARALLEL" \
  --ready-timeout 600 \
  --show-logs
