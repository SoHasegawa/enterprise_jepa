#!/usr/bin/env bash
# Train a K-controller (Stages I + II) for use with the `react_wm_rl_k`
# replay mode in src/evaluation.py.
#
# Ported from `ewm-state-design-and-model-training-experiments`. Runs only
# Stages I (label) and II (sft) -- Stage III (online A2C `rl_k`) is
# intentionally skipped because our action policy at inference time is
# typically an external API (GPT-5.1) that cannot sit inside the inner
# training loop.
#
# Pipeline:
#   1. label : for each expert (state, action) step in the ewm trajectory
#              JSON, score K=0..KMAX by `logprob(action | state, foresight_K)
#              - LAMBDA_K * K`, where foresight_K comes from the ewm world
#              model. Argmax K is the pseudo-optimal label.
#   2. sft   : SFT a small policy LM with a <CTRL>-positioned K-head on the
#              labelled JSONL. Joint loss = action LM loss + BETA_K *
#              cross_entropy(K-head logits, K_label).
#
# Required env vars:
#   TRAIN_TRAJECTORIES   -- one or more ewm trajectory JSON paths, separated
#                           by `:` (most users want
#                           trajectories/enterpriseops_gym_multi_model_world_model_train_trajectories.json
#                           or its `_enterprise_state_` variant).
#   WORLD_MODEL_TARGET   -- which ewm WM target the labeller scores against.
#                           One of:
#                             - tool_execution_result_binary  (binary + error message)
#                             - state                         (structured JSON state)
#                             - tool_output                   (raw tool output text)
#   WM_MODEL_PATH or WM_METHOD  -- the trained world model. Either a local HF
#                           dir (WM_MODEL_PATH) or a hosted backend like
#                           vllm/gymops_world_model (WM_METHOD).
#   OUT_DIR              -- output directory; receives labeled.jsonl and
#                           policy_sft_khead/.
#
# Optional knobs:
#   POLICY_BASE          -- small base LM for the K-controller
#                           (default: Qwen/Qwen2.5-1.5B-Instruct).
#   KMAX, LAMBDA_K, BETA_K, EPOCHS, LR, BATCH_SIZE, GRAD_ACCUM, MAX_SEQ_LEN
#   WM_MAX_NEW_TOKENS, SCORE_BATCH_SIZE
#   INCLUDE_ERROR_MESSAGE_IN_TARGET=1   (matches --include-error-message-in-target)
#   INCLUDE_STAGE_IN_TARGET=1           (matches --include-stage-in-target)
#   INCLUDE_INPUT_HISTORY=1             (include action/observation history in state text)
#   TORCH_DTYPE                         (bf16 | fp16 | fp32)
#   SKIP_LABEL=1                        (reuse existing OUT_DIR/labeled.jsonl)
#   SKIP_SFT=1                          (reuse existing OUT_DIR/policy_sft_khead/)
#
# Example (binary + error-message target with vllm-served WM):
#
#   TRAIN_TRAJECTORIES=trajectories/enterpriseops_gym_multi_model_world_model_train_trajectories.json \
#   WORLD_MODEL_TARGET=tool_execution_result_binary \
#   INCLUDE_ERROR_MESSAGE_IN_TARGET=1 \
#   WM_METHOD=vllm/gymops_world_model \
#   OUT_DIR=sessions/k_controller_binary \
#   KMAX=3 LAMBDA_K=0.2 \
#   bash scripts/run_k_controller_training.sh

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

# ---- Required inputs ------------------------------------------------------
: "${TRAIN_TRAJECTORIES:?Set TRAIN_TRAJECTORIES to one or more ewm trajectory JSON paths (colon-separated).}"
: "${OUT_DIR:?Set OUT_DIR to a fresh directory for the K-controller artifacts.}"
: "${WORLD_MODEL_TARGET:=tool_execution_result_binary}"

if [[ -z "${WM_MODEL_PATH:-}" && -z "${WM_METHOD:-}" ]]; then
  echo "[FAIL] Set WM_MODEL_PATH (local HF dir) or WM_METHOD (e.g. vllm/gymops_world_model)." >&2
  exit 1
fi

# Convert colon-separated TRAIN_TRAJECTORIES into space-separated array.
IFS=':' read -r -a TRAIN_TRAJ_ARR <<< "${TRAIN_TRAJECTORIES}"

# Normalise to absolute paths.
ABS_TRAJ=()
for path in "${TRAIN_TRAJ_ARR[@]}"; do
  case "${path}" in /*) ABS_TRAJ+=("${path}") ;; *) ABS_TRAJ+=("${ROOT_DIR}/${path}") ;; esac
done
case "${OUT_DIR}" in /*) ;; *) OUT_DIR="${ROOT_DIR}/${OUT_DIR}" ;; esac
mkdir -p "${OUT_DIR}"

# ---- Optional knobs -------------------------------------------------------
POLICY_BASE="${POLICY_BASE:-Qwen/Qwen2.5-1.5B-Instruct}"

# Stage I knobs.
KMAX="${KMAX:-3}"
LAMBDA_K="${LAMBDA_K:-0.2}"
WM_MAX_NEW_TOKENS="${WM_MAX_NEW_TOKENS:-128}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-2048}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-16}"
LOGPROB_NORM="${LOGPROB_NORM:-sum}"
STATE_HISTORY_SIZE="${STATE_HISTORY_SIZE:-3}"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-}"

# Stage II knobs.
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
LR="${LR:-2e-5}"
BETA_K="${BETA_K:-0.5}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"

# Skip toggles.
SKIP_LABEL="${SKIP_LABEL:-0}"
SKIP_SFT="${SKIP_SFT:-0}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LABELED_JSONL="${OUT_DIR}/labeled.jsonl"
POLICY_DIR="${OUT_DIR}/policy_sft_khead"

# Optional flags from env switches.
LABEL_EXTRA=()
[[ "${INCLUDE_ERROR_MESSAGE_IN_TARGET:-0}" == "1" ]] && LABEL_EXTRA+=( --include-error-message-in-target )
[[ "${INCLUDE_STAGE_IN_TARGET:-0}" == "1" ]] && LABEL_EXTRA+=( --include-stage-in-target )
[[ "${INCLUDE_INPUT_HISTORY:-0}" == "1" ]] && LABEL_EXTRA+=( --include-input-history )
[[ -n "${WM_MODEL_PATH:-}" ]] && LABEL_EXTRA+=( --wm-model-path "${WM_MODEL_PATH}" )
[[ -n "${WM_METHOD:-}" ]] && LABEL_EXTRA+=( --wm-method "${WM_METHOD}" )
[[ -n "${VLLM_SERVER_PORT}" ]] && LABEL_EXTRA+=( --vllm-server-port "${VLLM_SERVER_PORT}" )

SFT_EXTRA=()
case "${TORCH_DTYPE}" in
  bf16) SFT_EXTRA+=( --bf16 ) ;;
  fp16) SFT_EXTRA+=( --fp16 ) ;;
esac

echo "================================================================"
echo "[k-controller] POLICY_BASE           : ${POLICY_BASE}"
echo "[k-controller] WM_MODEL_PATH         : ${WM_MODEL_PATH:-(unset)}"
echo "[k-controller] WM_METHOD             : ${WM_METHOD:-(unset)}"
echo "[k-controller] WORLD_MODEL_TARGET    : ${WORLD_MODEL_TARGET}"
echo "[k-controller] TRAIN_TRAJECTORIES    : ${ABS_TRAJ[*]}"
echo "[k-controller] OUT_DIR               : ${OUT_DIR}"
echo "[k-controller] KMAX=${KMAX} LAMBDA_K=${LAMBDA_K} EPOCHS=${EPOCHS} LR=${LR}"
echo "================================================================"

# --------------------------------------------------------------------------
echo "[1/2] Stage I (label) -- compute pseudo-optimal K labels"
# --------------------------------------------------------------------------
if [[ "${SKIP_LABEL}" == "1" && -f "${LABELED_JSONL}" ]]; then
  echo "  SKIP_LABEL=1 -- reusing ${LABELED_JSONL}"
else
  ${PYTHON} -u -m src.itp.training.train_adaptive_k label \
    --train-trajectories "${ABS_TRAJ[@]}" \
    --world-model-target "${WORLD_MODEL_TARGET}" \
    --policy-model-path "${POLICY_BASE}" \
    --out-labeled-jsonl "${LABELED_JSONL}" \
    --kmax "${KMAX}" \
    --lambda-k "${LAMBDA_K}" \
    --wm-max-new-tokens "${WM_MAX_NEW_TOKENS}" \
    --score-batch-size "${SCORE_BATCH_SIZE}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --state-history-size "${STATE_HISTORY_SIZE}" \
    --padding-side left \
    --logprob-norm "${LOGPROB_NORM}" \
    "${LABEL_EXTRA[@]}"
fi
echo "  -> ${LABELED_JSONL}  ($(wc -l < "${LABELED_JSONL}") rows)"

# --------------------------------------------------------------------------
echo "[2/2] Stage II (sft) -- SFT policy LM with K-head"
# --------------------------------------------------------------------------
if [[ "${SKIP_SFT}" == "1" && -d "${POLICY_DIR}" ]]; then
  echo "  SKIP_SFT=1 -- reusing ${POLICY_DIR}"
else
  ${PYTHON} -u -m src.itp.training.train_adaptive_k sft \
    --train-jsonl "${LABELED_JSONL}" \
    --policy-model-path "${POLICY_BASE}" \
    --out-dir "${POLICY_DIR}" \
    --kmax "${KMAX}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum "${GRAD_ACCUM}" \
    --lr "${LR}" \
    --beta-k "${BETA_K}" \
    --max-seq-len "${MAX_SEQ_LEN}" \
    --padding-side left \
    "${SFT_EXTRA[@]}"
fi

echo "================================================================"
echo "[DONE] K-controller -> ${POLICY_DIR}"
echo
echo "Use this path with src/evaluation.py:"
echo "  --agent-replay-mode react_wm_rl_k \\"
echo "  --k-controller-path ${POLICY_DIR} \\"
echo "  --react-wm-kmax ${KMAX}"
echo "================================================================"
