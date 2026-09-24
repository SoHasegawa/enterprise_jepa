#!/usr/bin/env bash
set -e

# mcp_react/ (holds executor.py + the orchestrator modules)
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# The gym repo (benchmark/ + the seed-database SQL the verifiers read) is the
# only external repo dependency.
: "${ENTERPRISEOPS_GYM_REPO_PATH:?set ENTERPRISEOPS_GYM_REPO_PATH to the EnterpriseOps-Gym repo}"

# Agent LLM (same vars as the A2A executor), e.g.:
#   export ENTERPRISEOPS_LLM_PROVIDER=openai ENTERPRISEOPS_LLM_MODEL=gpt-4o-mini ENTERPRISEOPS_LLM_API_KEY=sk-...
#   export ENTERPRISEOPS_LLM_PROVIDER=vllm   ENTERPRISEOPS_LLM_MODEL=agent ENTERPRISEOPS_LLM_API_ENDPOINT=http://127.0.0.1:9001/v1
#
# Orchestrator (default react). For the World Model:
#   export ENTERPRISEOPS_ORCHESTRATOR=wm_react
#   # best_of_n / prompt_injection (classifier scorer):
#   export WM_STRATEGY=best_of_n WM_SCORER_URL=http://127.0.0.1:8030
#   # imagined (generative EWM over vLLM):
#   export WM_STRATEGY=imagined WM_STATE=binary_error ACTION_OPTIMIZER=topk_search \
#          WM_EWM_MODEL=gymops_world_model WM_VLLM_BASE_URL=http://127.0.0.1:9000/v1

MCP_HOST="${MCP_HOST:-0.0.0.0}" \
MCP_PORT="${MCP_PORT:-12072}" \
python server/mcp_react/mcp_server.py
