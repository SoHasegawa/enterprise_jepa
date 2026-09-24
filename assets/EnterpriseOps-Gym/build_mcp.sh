#!/usr/bin/env bash
#
# build_mcp.sh — bring up the two servers the EWM-over-MCP path needs:
#
#   1. the EWM world model on vLLM  (Qwen3.6-27B-FP8 + the gymops LoRA, served at :9000), and
#   2. the EWM predict MCP server   (FastMCP, streamable-HTTP /mcp at :12072), which the agent
#      reaches for predict_state / generate / info and which proxies the vLLM model.
#
# Lives in assets/EnterpriseOps-Gym/ but is meant to be run from the benchmarks repo root, e.g.
#
#     ./assets/EnterpriseOps-Gym/build_mcp.sh
#
# Both servers start in the background; logs + pids land in $RUNTIME_DIR. Re-running is safe —
# a server already listening on its port is left alone. Then run ./assets/EnterpriseOps-Gym/run_mcp.sh
#
set -euo pipefail

# Repo root = two levels up from this script (assets/EnterpriseOps-Gym/<script>), independent of CWD.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SRC_DIR="$REPO_ROOT/src"

# ---- config (override via env) ---------------------------------------------------------
VLLM_PORT="${VLLM_PORT:-9000}"
MCP_HOST="${MCP_HOST:-127.0.0.1}"
MCP_PORT="${MCP_PORT:-12072}"
EWM_VLLM_MODEL="${EWM_VLLM_MODEL:-Qwen/Qwen3.6-27B-FP8}"
# The EWM LoRA adapter ships in-repo (via Git LFS) so this works from a fresh checkout — run
# `git lfs pull` first. Override EWM_LORA to point elsewhere if you keep the adapter outside.
EWM_LORA="${EWM_LORA:-gymops_world_model=$REPO_ROOT/assets/EnterpriseOps-Gym/models/gymops_world_model}"
# vLLM resolves a *relative* LoRA path from this dir; the default EWM_LORA is absolute so it
# resolves regardless of VLLM_CWD.
VLLM_CWD="${VLLM_CWD:-$PWD}"
# Python that can import fastmcp + uvicorn (the EWM engine itself is stdlib-only):
MCP_PYTHON="${MCP_PYTHON:-python}"
WM_STATE="${WM_STATE:-binary_error}"
RUNTIME_DIR="${RUNTIME_DIR:-/tmp/ewm_mcp_runtime}"
mkdir -p "$RUNTIME_DIR"

_listening() { ss -ltn 2>/dev/null | grep -q ":$1 " ; }

# ---- 1. EWM world-model vLLM server ----------------------------------------------------
# NOTE: the served model id is gymops_world_model (the redundant duplicate
# `--served-model-name world_model` from the original command is dropped — argparse keeps
# only the last value anyway, and the EWM stack expects gymops_world_model).
if _listening "$VLLM_PORT"; then
  echo "[vllm]  already listening on :$VLLM_PORT — skipping launch"
else
  lora_path="${EWM_LORA#*=}"
  resolved="$lora_path"; [ -e "$resolved" ] || resolved="$VLLM_CWD/$lora_path"
  if [ -n "$lora_path" ] && [ ! -e "$resolved" ]; then
    echo "[vllm]  WARN: LoRA checkpoint '$lora_path' not found under VLLM_CWD=$VLLM_CWD"
    echo "        set VLLM_CWD to the dir containing it, or pass an absolute EWM_LORA path."
  elif [ -f "$resolved/adapter_model.safetensors" ] \
       && head -c 64 "$resolved/adapter_model.safetensors" 2>/dev/null | grep -q "git-lfs"; then
    echo "[vllm]  WARN: LoRA weights at '$resolved' are an unpulled Git LFS pointer."
    echo "        run: git lfs pull   (then re-run this script)"
  fi
  echo "[vllm]  starting $EWM_VLLM_MODEL (LoRA: $EWM_LORA) on :$VLLM_PORT (cwd: $VLLM_CWD)"
  ( cd "$VLLM_CWD" && nohup vllm serve "$EWM_VLLM_MODEL" \
        --enable-lora \
        --lora-modules "$EWM_LORA" \
        --tensor-parallel-size 1 \
        --max-model-len 32768 \
        --port "$VLLM_PORT" \
        --host 0.0.0.0 \
        --reasoning-parser qwen3 \
        --language-model-only \
        --default-chat-template-kwargs '{"enable_thinking": false}' \
        --served-model-name gymops_world_model \
        --kv-cache-dtype fp8 \
        --gpu-memory-utilization 0.92 \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --max-num-seqs 256 \
        --max-num-batched-tokens 8192 \
        >"$RUNTIME_DIR/vllm.log" 2>&1 & echo $! >"$RUNTIME_DIR/vllm.pid" )
  echo "[vllm]  launched pid $(cat "$RUNTIME_DIR/vllm.pid"); log: $RUNTIME_DIR/vllm.log"
fi

echo "[vllm]  waiting for /v1/models (model load can take minutes) ..."
for _ in $(seq 1 900); do
  if curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; then
    echo "[vllm]  up — $(curl -s "http://127.0.0.1:$VLLM_PORT/v1/models" | grep -o '"id":"[^"]*"' | head -1)"
    break
  fi
  sleep 2
done

# ---- 2. EWM predict MCP server ---------------------------------------------------------
if ! "$MCP_PYTHON" -c "import fastmcp, uvicorn" >/dev/null 2>&1; then
  echo "[mcp]   ERROR: '$MCP_PYTHON' cannot import fastmcp/uvicorn." >&2
  echo "        Install them (pip install fastmcp uvicorn) or set MCP_PYTHON to a venv that has them." >&2
  exit 1
fi

if _listening "$MCP_PORT"; then
  echo "[mcp]   already listening on :$MCP_PORT — skipping launch"
else
  echo "[mcp]   starting EWM predict MCP server on http://$MCP_HOST:$MCP_PORT/mcp (backend: vllm :$VLLM_PORT)"
  PYTHONPATH="$SRC_DIR:${PYTHONPATH:-}" \
  EWM_WORLD_MODEL_METHOD=vllm/gymops_world_model \
  WM_VLLM_SERVER_PORT="$VLLM_PORT" \
  WM_STATE="$WM_STATE" \
  MCP_HOST="$MCP_HOST" MCP_PORT="$MCP_PORT" \
    nohup "$MCP_PYTHON" -m ejepa_wm.server.ewm_predict \
      >"$RUNTIME_DIR/mcp.log" 2>&1 & echo $! >"$RUNTIME_DIR/mcp.pid"
  echo "[mcp]   launched pid $(cat "$RUNTIME_DIR/mcp.pid"); log: $RUNTIME_DIR/mcp.log"
fi

echo "[mcp]   waiting for /mcp ..."
for _ in $(seq 1 60); do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://$MCP_HOST:$MCP_PORT/mcp" 2>/dev/null || true)"
  if [ "$code" = "405" ]; then echo "[mcp]   up at http://$MCP_HOST:$MCP_PORT/mcp"; break; fi
  sleep 1
done

echo
echo "Ready. vLLM :$VLLM_PORT (log $RUNTIME_DIR/vllm.log) | MCP :$MCP_PORT (log $RUNTIME_DIR/mcp.log)"
echo "Next:  ./assets/EnterpriseOps-Gym/run_mcp.sh        (imagined trajectory over MCP)"
echo "Stop:  fuser -k $VLLM_PORT/tcp $MCP_PORT/tcp"
