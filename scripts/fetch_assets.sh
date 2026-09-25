#!/usr/bin/env bash
# Fetch the external data this repository deliberately does not vendor.
#
# The repo ships code, task definitions and the paper's derived results. Three classes
# of artefact are left out and fetched here:
#
#   1. upstream benchmark checkouts  (task content + verifiers, cloned from their origins)
#   2. CRMArena-Pro SQLite databases (extracted from the upstream baseline image)
#   3. model checkpoints             (the JEPA world model and the LoRA world model;
#                                     these are training outputs, see docs/training.md)
#
# Usage:
#   scripts/fetch_assets.sh                 # everything in 1 + 2
#   scripts/fetch_assets.sh benchmarks      # upstream checkouts only
#   scripts/fetch_assets.sh crm-db          # CRMArena-Pro databases only
#   scripts/fetch_assets.sh checkpoints     # print where checkpoints must be placed
#
# Environment:
#   UPSTREAM_ROOT   where upstream checkouts are cloned (default: <repo>/upstreams)
#   JEPA_CKPT       destination for the JEPA checkpoint (default: <repo>/checkpoints/jepa)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_ROOT="${UPSTREAM_ROOT:-$REPO_ROOT/upstreams}"
JEPA_CKPT="${JEPA_CKPT:-$REPO_ROOT/checkpoints/jepa}"

log() { echo "[fetch] $*"; }

clone_upstream() {  # name url [ref]
  local name="$1" url="$2" ref="${3:-}"
  local dest="$UPSTREAM_ROOT/$name"
  if [[ -d "$dest/.git" ]]; then
    log "$name already present: $dest"
    return 0
  fi
  log "cloning $name from $url"
  git clone "$url" "$dest"
  if [[ -n "$ref" ]]; then
    git -C "$dest" checkout "$ref"
  fi
}

fetch_benchmarks() {
  mkdir -p "$UPSTREAM_ROOT"
  # Pinned to the commits the paper's runs used. Drop the third argument to track HEAD.
  clone_upstream EnterpriseOps-Gym https://github.com/ServiceNow/EnterpriseOps-Gym.git
  clone_upstream WorkBench         https://github.com/olly-styles/WorkBench.git          49c7dfd
  clone_upstream AutomationBench   https://github.com/zapier/AutomationBench.git         4a8e106
  clone_upstream terminal-bench-2  https://github.com/laude-institute/terminal-bench-2.git 2fd12b8
  clone_upstream CRMArena          https://github.com/SalesforceAIResearch/CRMArena.git  32b609a

  cat <<EOF

[fetch] Export these before running the benchmarks (see docs/setup.md):

  export ENTERPRISEOPS_GYM_REPO_PATH=$UPSTREAM_ROOT/EnterpriseOps-Gym
  export WORKBENCH_REPO_PATH=$UPSTREAM_ROOT/WorkBench
  export AUTOMATIONBENCH_REPO_PATH=$UPSTREAM_ROOT/AutomationBench
  export TERMINAL_BENCH_TASK_REPO=$UPSTREAM_ROOT/terminal-bench-2

EnterpriseOps-Gym also needs its HuggingFace task corpus
(https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym) and its MCP tool
containers: see assets/EnterpriseOps-Gym/README.md (build_mcp.sh / run_mcp.sh).
EOF
}

fetch_crm_db() {
  local data_dir="$REPO_ROOT/assets/crmarenapro/purple-executors/baseline_crm_agent/data"
  local b2b="$data_dir/crmarenapro_b2b_data.db"
  local image="ghcr.io/rkstu/baseline-crm-agent:latest"
  if [[ -f "$b2b" ]]; then
    log "CRMArena-Pro databases already present: $data_dir"
    return 0
  fi
  if ! command -v docker >/dev/null 2>&1; then
    log "docker not found; cannot extract the CRMArena-Pro databases from $image" >&2
    return 1
  fi
  mkdir -p "$data_dir"
  log "pulling $image for the CRMArena-Pro SQLite databases"
  docker pull "$image"
  local cid
  cid="$(docker create "$image")"
  docker cp "$cid:/home/agent/data/crmarenapro_b2b_data.db" "$b2b"
  docker cp "$cid:/home/agent/data/crmarenapro_b2c_data.db" "$data_dir/crmarenapro_b2c_data.db"
  docker rm "$cid" >/dev/null
  log "installed $b2b"
}

show_checkpoints() {
  cat <<EOF
[fetch] The two trained world models are published as release assets; see model/README.md
[fetch] for their contents and checksums.

1. Enterprise-JEPA. Unpack the 'enterprise_jepa' asset to:

     $JEPA_CKPT

   It must contain text_leworldmodel.pt, jepa_data_manifest.json,
   canonical_event_vocab.json, the tokenizer files and backbone/. Pass it with
   --wm-ewm-jepa-checkpoint, or set JEPA_CKPT for the run scripts.

2. State-output LLM world model. Unpack the 'llm_wm_state_output' asset anywhere and
   serve it on an OpenAI-compatible endpoint:

     vllm serve <dir> --served-model-name world_model --port 9015

   then export WM_VLLM_BASE_URL / WM_VLLM_API_KEY for the run scripts.

3. EnterpriseOps LoRA world model (optional; only the 'ewm_predict' /
   wm_imagined_run.sh paths use it, which the paper does not). Place
   adapter_model.safetensors and tokenizer.json next to the adapter_config.json in:

     $REPO_ROOT/assets/EnterpriseOps-Gym/models/gymops_world_model
EOF
}

case "${1:-all}" in
  all)         fetch_benchmarks; fetch_crm_db || true; show_checkpoints ;;
  benchmarks)  fetch_benchmarks ;;
  crm-db)      fetch_crm_db ;;
  checkpoints) show_checkpoints ;;
  -h|--help)   sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//' ;;
  *)           echo "unknown component: $1 (expected all|benchmarks|crm-db|checkpoints)" >&2; exit 2 ;;
esac
