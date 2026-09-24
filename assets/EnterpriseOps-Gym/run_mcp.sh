#!/usr/bin/env bash
#
# run_mcp.sh — run an EnterpriseOps-Gym task with the EWM imagined-trajectory tool over MCP.
#
# Lives in assets/EnterpriseOps-Gym/ but is meant to be run from the benchmarks repo root, e.g.
#
#     ./assets/EnterpriseOps-Gym/run_mcp.sh
#
# Default behavior:
#   * a fresh WM_IMAGINED_MAX_STEPS imagined rollout is forced BEFORE EVERY agent step
#     (EWM_IMAGINE_EVERY_STEP=1) and delivered as an `imagine_trajectory` tool result, so the
#     look-ahead never goes stale on long tasks (receding horizon);
#   * `imagine_trajectory` is also kept agent-callable (EWM_IMAGINE_TOOL=1);
#   * the rollout's world-model calls go over the EWM predict MCP server's `generate` tool
#     (WM_EWM_MCP_URL), i.e. the imagined trajectory is produced *over MCP*;
#   * per-step predict_state is dropped (auto-dropped while imagine is on).
# This matches the binary+error world-model behavior of the non-MCP path. Set
# EWM_IMAGINE_EVERY_STEP=0 to fall back to agent-timed (on-demand) re-imagination.
#
# Prereqs: ./assets/EnterpriseOps-Gym/build_mcp.sh is up (vLLM :9000 + MCP :12072), and a
# TOOL-CALLING agent LLM is configured (e.g. gpt-5.1 via ENTERPRISEOPS_LLM_* — set those
# yourself; secrets are never baked into this script).
#
set -uo pipefail

# Repo root = two levels up from this script (assets/EnterpriseOps-Gym/<script>), independent of CWD.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Optional: source a local, git-ignored env file for agent credentials (ENTERPRISEOPS_LLM_*).
[ -f "$REPO_ROOT/.env.ewm" ] && { echo "[env] sourcing .env.ewm"; set -a; . "$REPO_ROOT/.env.ewm"; set +a; }

# ---- EWM imagined-trajectory-over-MCP config (override via env) -------------------------
export EWM_IMAGINE_TOOL="${EWM_IMAGINE_TOOL:-1}"                       # register imagine_trajectory; drops predict_state
export EWM_IMAGINE_EVERY_STEP="${EWM_IMAGINE_EVERY_STEP:-1}"           # DEFAULT: force a fresh rollout (as a tool result) before every step
export WM_EWM_MCP_URL="${WM_EWM_MCP_URL:-http://127.0.0.1:12072}"      # rollout WM over MCP (generate tool)
export WM_EWM_MODEL="${WM_EWM_MODEL:-gymops_world_model}"
export WM_STATE="${WM_STATE:-binary_error}"                           # binary_error | binary_error_stage | tool_output
export WM_IMAGINED_MAX_STEPS="${WM_IMAGINED_MAX_STEPS:-3}"            # FORCED rollout depth
# EWM_IMAGINE_SUPERSEDE defaults to 1 (keep only the latest auto-trajectory in context).
# To revert to agent-timed re-imagination only, run with EWM_IMAGINE_EVERY_STEP=0.
# Optional rollout axes:
#   export ACTION_OPTIMIZER=topk_search WM_IMAGINED_CANDIDATE_ACTIONS=3 WM_IMAGINED_TOP_K=3
#   export WM_IMAGINED_TEMPERATURE=0.7

# Gym repo (benchmark/ + the seed-database SQL the verifiers read):
export ENTERPRISEOPS_GYM_REPO_PATH="${ENTERPRISEOPS_GYM_REPO_PATH:-$PWD/upstreams/EnterpriseOps-Gym}"

TARGET="${TARGET:-sample}"

# ---- sanity checks ---------------------------------------------------------------------
mcp_base="${WM_EWM_MCP_URL%/}"; mcp_base="${mcp_base%/mcp}"
code="$(curl -s -o /dev/null -w '%{http_code}' "$mcp_base/mcp" 2>/dev/null || true)"
[ "$code" = "405" ] || echo "[warn] EWM MCP server not reachable at $mcp_base/mcp — run ./assets/EnterpriseOps-Gym/build_mcp.sh first (got HTTP '$code')"

if [ ! -e "$ENTERPRISEOPS_GYM_REPO_PATH/benchmark/executor.py" ]; then
  echo "[warn] ENTERPRISEOPS_GYM_REPO_PATH=$ENTERPRISEOPS_GYM_REPO_PATH has no benchmark/executor.py"
fi

if [ -z "${ENTERPRISEOPS_LLM_CONFIG_FILE:-}" ] \
   && [ -z "${ENTERPRISEOPS_LLM_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[warn] no tool-calling agent LLM configured. Set ENTERPRISEOPS_LLM_PROVIDER/_MODEL/_API_KEY"
  echo "       (+ _API_ENDPOINT for an OpenAI-compatible gateway), or ENTERPRISEOPS_LLM_CONFIG_FILE."
fi

echo "[run] imagine_trajectory over MCP | every_step=$EWM_IMAGINE_EVERY_STEP | WM_EWM_MCP_URL=$WM_EWM_MCP_URL | steps=$WM_IMAGINED_MAX_STEPS | wm_state=$WM_STATE | target=$TARGET"
exec ejepa bench run EnterpriseOps-Gym --executor mcp_react --config "target=${TARGET}"
