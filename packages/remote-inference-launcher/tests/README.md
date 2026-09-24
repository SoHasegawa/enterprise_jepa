# Tests

## Local Tests

Run the local pytest suite:

```bash
uv run pytest tests
```

Local tests must not require SSH, Slurm, GPUs, or remote hosts. Live integration
tests are skipped unless explicitly enabled.

## Live SSH/Slurm Integration Tests

Live tests are opt-in. Some submit remote jobs or start real model servers, so
run them only when you intend to allocate GPU, SSH, Slurm, disk, and network
resources.

Set `RIL_RUN_INTEGRATION=1` to enable the live tests. The configuration comes
from environment variables:

```bash
RIL_INTEGRATION_SSH_TARGETS=gpu-cluster,ssh://alice@gpu.example.internal:10001 \
RIL_INTEGRATION_BOOTSTRAP_SSH_TARGET=gpu-cluster \
RIL_INTEGRATION_BOOTSTRAP_PARTITION=batch-1gpu-short \
RIL_INTEGRATION_BOOTSTRAP_VENV_PATH='~/.ejepa-vllm-env' \
RIL_INTEGRATION_BOOTSTRAP_VLLM_PACKAGE=vllm==0.22.0+rocm722 \
RIL_INTEGRATION_BOOTSTRAP_BACKEND=rocm \
RIL_INTEGRATION_SERVING_SSH_TARGET=gpu-cluster \
RIL_INTEGRATION_SERVING_PARTITION=batch-1gpu-short \
RIL_INTEGRATION_SERVING_MODEL=Qwen/Qwen3.5-9B \
RIL_INTEGRATION_SERVING_PYTHON_BIN='~/.ejepa-vllm-env/bin/python' \
RIL_INTEGRATION_SERVING_TARGET_DEVICE=rocm \
RIL_INTEGRATION_SERVING_MAX_MODEL_LEN=8192 \
RIL_RUN_INTEGRATION=1 uv run pytest tests/integration -m integration
```

## Live Serving Mode Tests

`tests/integration/test_live_serving_modes.py` verifies end-to-end serving for
the launcher modes selected in `RIL_INTEGRATION_SERVING_MODES`:

- `existing`
- `generic_existing`
- `local`
- `ssh`
- `slurm`
- `fleet`
- `multinode_slurm`
- `all`

Each selected mode starts or checks an OpenAI-compatible endpoint, waits for
`/v1/models`, sends a `/v1/chat/completions` request, verifies non-empty text,
and then runs launcher cleanup.
`generic_existing` additionally runs the user-facing `remote-inference-launcher
start --config ... --env-file ...` path against an already-running endpoint and
uses the generated env file for the chat request.

Start from the example env file:

```bash
cp tests/integration/live-serving.example.env /tmp/ril-live-serving.env
$EDITOR /tmp/ril-live-serving.env
set -a
. /tmp/ril-live-serving.env
set +a
uv run pytest tests/integration/test_live_serving_modes.py -m integration -q
```

The serving mode tests skip unless both `RIL_RUN_INTEGRATION=1` and an explicit
`RIL_INTEGRATION_SERVING_MODES` selection are set.
