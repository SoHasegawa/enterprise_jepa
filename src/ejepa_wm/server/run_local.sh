#!/usr/bin/env bash
set -e

# Repo `src/` (so `import ejepa_wm` resolves) — server lives at src/ejepa_wm/server/.
SRC_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="$SRC_DIR:$PYTHONPATH"

# EWM model backend (pick one):
#   vLLM (proxy to an OpenAI-compatible world-model server):
#     export EWM_WORLD_MODEL_METHOD=vllm/gymops_world_model
#     export EWM_VLLM_SERVER_PORT=9000          # or WM_VLLM_BASE_URL=http://host:9000/v1
#   transformers (load a local HuggingFace checkpoint in-process):
#     export EWM_WORLD_MODEL_PATH=/path/to/world_model_checkpoint
#     # (requires torch + transformers installed; uncomment them in requirements.txt)
#
# Prediction mode (what the WM predicts):
#   export WM_STATE=binary_error      # | binary_error_stage | tool_output

MCP_HOST="${MCP_HOST:-0.0.0.0}" \
MCP_PORT="${MCP_PORT:-12072}" \
python -m ejepa_wm.server.ewm_predict
