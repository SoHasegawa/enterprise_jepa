#!/usr/bin/env bash
# Convenience driver for evaluating the `react_wm` family on
# EnterpriseOps-Gym tasks.
#
# This wraps src/evaluation.py with the right flags for the three new modes:
#
#   - react_wm           : fixed K foresight; default K = REACT_WM_K (=2).
#   - react_wm_decide_k  : agent picks K via a one-shot prompt; capped at KMAX.
#   - react_wm_rl_k      : trained K-controller picks K; requires K_CONTROLLER_PATH.
#
# Required env vars:
#   WM_METHOD or WM_PATH       -- the world model backend (vllm/... or local HF dir).
#   GYM_TASK_CONFIGS_DIR       -- folder of dumped EnterpriseOps-Gym task JSONs
#                                 (default: trajectories/enterpriseops_gym_task_configs).
#   AGENT_MODEL                -- action policy (default: gpt5.1).
#   WORLD_MODEL_TARGET         -- one of tool_execution_result_binary | state | tool_output.
#   METHOD                     -- one of react_wm | react_wm_decide_k | react_wm_rl_k | all_with_itp.
#   K_CONTROLLER_PATH          -- required when METHOD includes react_wm_rl_k.
#
# Optional knobs match the corresponding --react-wm-* / --k-controller-* flags.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if command -v uv >/dev/null 2>&1; then PYTHON="uv run python"
  elif command -v python3 >/dev/null 2>&1; then PYTHON=python3
  elif command -v python >/dev/null 2>&1; then PYTHON=python
  else echo "[FAIL] no python found (set PYTHON=/path/to/python)" >&2; exit 1; fi
fi

METHOD="${METHOD:-react_wm}"
AGENT_MODEL="${AGENT_MODEL:-gpt5.1}"
WORLD_MODEL_TARGET="${WORLD_MODEL_TARGET:-tool_execution_result_binary}"
GYM_TASK_CONFIGS_DIR="${GYM_TASK_CONFIGS_DIR:-trajectories/enterpriseops_gym_task_configs}"
OUTPUT_DIR="${OUTPUT_DIR:-results/evaluation/${METHOD}}"
REACT_WM_K="${REACT_WM_K:-2}"
REACT_WM_KMAX="${REACT_WM_KMAX:-3}"
REACT_WM_FORESIGHT_TEMP="${REACT_WM_FORESIGHT_TEMP:-0.0}"
REACT_WM_FORESIGHT_OBS_SOURCE="${REACT_WM_FORESIGHT_OBS_SOURCE:-world_model}"
TRAJECTORY_DATASET="${TRAJECTORY_DATASET:-enterpriseops_gym}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-9000}"
MAX_AGENT_TASKS="${MAX_AGENT_TASKS:-0}"

if [[ -z "${WM_METHOD:-}" && -z "${WM_PATH:-}" ]]; then
  echo "[FAIL] Set WM_METHOD (e.g. vllm/gymops_world_model) or WM_PATH (local HF dir)." >&2
  exit 1
fi

WM_FLAGS=()
[[ -n "${WM_METHOD:-}" ]] && WM_FLAGS+=( --world-model-method "${WM_METHOD}" )
[[ -n "${WM_PATH:-}" ]] && WM_FLAGS+=( --world-model-path "${WM_PATH}" )

K_CTRL_FLAGS=()
if [[ "${METHOD}" == "react_wm_rl_k" || "${METHOD}" == "all_with_itp" ]]; then
  : "${K_CONTROLLER_PATH:?Set K_CONTROLLER_PATH for METHOD=${METHOD}.}"
  K_CTRL_FLAGS+=(
    --k-controller-path "${K_CONTROLLER_PATH}"
    --k-controller-device "${K_CONTROLLER_DEVICE:-auto}"
    --k-controller-dtype "${K_CONTROLLER_DTYPE:-auto}"
    --k-controller-max-seq-len "${K_CONTROLLER_MAX_SEQ_LEN:-2048}"
  )
  [[ "${K_CONTROLLER_DO_SAMPLE:-0}" == "1" ]] && K_CTRL_FLAGS+=( --k-controller-do-sample )
fi

EXTRA=()
[[ "${INCLUDE_ERROR_MESSAGE_IN_TARGET:-0}" == "1" ]] && EXTRA+=( --include-error-message-in-target )
[[ "${INCLUDE_STAGE_IN_TARGET:-0}" == "1" ]] && EXTRA+=( --include-stage-in-target )
[[ "${INCLUDE_WORLD_MODEL_HISTORY:-0}" == "1" ]] && EXTRA+=( --include-world-model-history )
[[ "${MAX_AGENT_TASKS}" -gt 0 ]] && EXTRA+=( --max-agent-tasks "${MAX_AGENT_TASKS}" )

mkdir -p "${OUTPUT_DIR}"

echo "================================================================"
echo "[react_wm-eval] METHOD              : ${METHOD}"
echo "[react_wm-eval] AGENT_MODEL         : ${AGENT_MODEL}"
echo "[react_wm-eval] WORLD_MODEL_TARGET  : ${WORLD_MODEL_TARGET}"
echo "[react_wm-eval] WM_METHOD/PATH      : ${WM_METHOD:-}${WM_PATH:-}"
echo "[react_wm-eval] OUTPUT_DIR          : ${OUTPUT_DIR}"
echo "[react_wm-eval] REACT_WM_K          : ${REACT_WM_K}"
echo "[react_wm-eval] REACT_WM_KMAX       : ${REACT_WM_KMAX}"
echo "================================================================"

${PYTHON} -u src/evaluation.py \
  --trajectory-dataset "${TRAJECTORY_DATASET}" \
  --world-model-target "${WORLD_MODEL_TARGET}" \
  "${WM_FLAGS[@]}" \
  --vllm-server-port "${VLLM_SERVER_PORT}" \
  --agent-replay-mode "${METHOD}" \
  --react-wm-k "${REACT_WM_K}" \
  --react-wm-kmax "${REACT_WM_KMAX}" \
  --react-wm-foresight-temperature "${REACT_WM_FORESIGHT_TEMP}" \
  --react-wm-foresight-observation-source "${REACT_WM_FORESIGHT_OBS_SOURCE}" \
  --gym-task-configs "${GYM_TASK_CONFIGS_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --agent-model "${AGENT_MODEL}" \
  "${K_CTRL_FLAGS[@]}" \
  "${EXTRA[@]}"
