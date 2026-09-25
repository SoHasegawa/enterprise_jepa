# Terminal-Bench-2.0

[Terminal-Bench 2.0](https://github.com/laude-institute/terminal-bench-2) evaluates agents on real terminal tasks inside Docker environments. This asset ports the Harbor-based green orchestrator from `terminal-bench-green` and adds a routed purple agent with three executors: `llm_shell`, `terminus_2`, and `mcp_react`. `mcp_react` mirrors the EnterpriseOps-Gym `mcp_react` naming: because the green owns the `terminal-bench-shell-v1` loop (there are no MCP tool servers, only the shell), it reuses `llm_shell`'s turn-server and only reframes the prompt into an MCP-style single-tool (`run_shell`) ReAct catalog, still emitting the required `exec_request`/`final` JSON. Select with `--executor mcp_react` (same env/creds as `llm_shell`).

Upstream references:

- Task corpus: https://github.com/laude-institute/terminal-bench-2
- Green orchestration pattern: `terminal-bench-green` (Harbor + DooD Docker + verifier)
- Harness library: [Harbor](https://pypi.org/project/harbor/) (`harbor>=0.8.0`; required for `harbor.environments.capabilities`)

---

## Layout

```
Terminal-Bench-2.0/
├── benchmark.toml
├── README.md
├── tasks/
│   └── terminal-bench-2/           # clone upstream repo here (not vendored)
├── green/
│   ├── terminal_bench_green_agent.py   # A2A entrypoint
│   ├── agent.py                        # task download, Docker env, purple loop, verify
│   ├── executor.py
│   ├── messenger.py
│   ├── dood_environment.py
│   ├── dood_compose_base.yaml
│   ├── task_loader.py
│   └── tasks/
│       └── task_ids.toml               # target → task name list
├── purple/
│   └── terminal_bench_purple_agent.py
└── purple-executors/
    ├── llm_shell/                      # Single-command JSON protocol
    └── terminus_2/                     # Terminus-2 batch commands (LiteLLM)
```

Self-evolve wiring is intentionally deferred in this rollout.

---

## Prerequisites

1. **Docker** — green pulls task images and runs verifiers via `docker compose`. Your user must be in the `docker` group:

```bash
sudo usermod -aG docker "$USER"
# log out/in, then verify:
docker ps
scripts/check-docker-access.sh
```

If you see `permission denied while trying to connect to the docker API at unix:///var/run/docker.sock`, Docker is installed but your account lacks access — fix group membership above before running tasks.

On a local dev machine (outside the AI Lab cluster), set a writable storage root — `ejepa` defaults to `$BENCHMARK_HOME` when `BENCHMARK_HOME` is unset:

```bash
export BENCHMARK_HOME="$PWD/.cache/benchmark-home"
export TERMINAL_BENCH_WORKSPACE="$PWD/.cache/terminal-bench-workspace"
```

2. **Task repo** — clone the upstream corpus locally:

```bash
git clone --depth 1 https://github.com/laude-institute/terminal-bench-2.git \
  assets/Terminal-Bench-2.0/tasks/terminal-bench-2
```

Alternatively set `TERMINAL_BENCH_TASK_REPO` to an existing checkout.

3. **LLM credentials** (for `llm_shell` / `terminus_2`, not needed for oracle runs):

Azure OpenAI GPT-5.5 (recommended):

```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT_NAME="gpt-5.5"   # must match your Azure deployment name
# optional (defaults shown):
export AZURE_OPENAI_API_VERSION="2024-10-21"   # validated for gpt-5.5; 2025-04-01-preview also works
export TERMINAL_BENCH_LLM_MAX_TOKENS="16384"   # GPT-5 uses max_completion_tokens
export AZURE_OPENAI_REASONING_EFFORT="low"     # low | medium | high
```

`llm_shell` uses the official `openai` Azure client when `AZURE_OPENAI_ENDPOINT` is set.
`scripts/install.sh terminal-bench-2.0` installs the `llm_shell` extra; alternatively:
`uv sync --project assets/Terminal-Bench-2.0/purple --extra llm_shell`.
Set `TERMINAL_BENCH_LLM_USE_HTTP=true` to force raw HTTP instead of the SDK.

`terminus_2` ports the [Terminus-2](https://github.com/laude-institute/terminal-bench) agent
(JSON/XML batch commands, double `task_complete` confirmation) onto the same green shell loop.
Install with `uv sync --project assets/Terminal-Bench-2.0/purple --extra terminus_2`.
Optional: `TERMINUS_2_PARSER=json|xml` (default `json`), `TERMINUS_2_TEMPERATURE=0.7`.

`mcp_react` reuses the `llm_shell` turn-server with an MCP-style `run_shell` ReAct prompt.
Install with `uv sync --project assets/Terminal-Bench-2.0/purple --extra mcp_react`. It can also
consult a pluggable World Model (`src/ejepa_wm`) like EnterpriseOps-Gym's `wm_react`: when
`WM_STRATEGY` selects `prompt_injection` (e.g. `--wm-strategy imagined`, or a JEPA model via
`--wm-ewm-jepa-checkpoint`), the WM's imagined-lookahead is injected **transiently** into each
turn's prompt (the green owns the loop, so there is no internal loop to steer); any WM failure
degrades to plain prompting. The JEPA path needs the `jepa` extra (torch/transformers):
`uv sync --project assets/Terminal-Bench-2.0/purple --extra mcp_react --extra jepa`.

```bash
ejepa bench run Terminal-Bench-2.0 --executor mcp_react --wm-strategy imagined
```

OpenAI-compatible:

```bash
export LLM_API_KEY="your-api-key"
export LLM_MODEL="gpt-4.1"                    # optional
export LLM_BASE_URL="https://api.openai.com/v1"  # optional
```

---

## Setup

From the **benchmarks repo root**:

### 1. Install the `ejepa` CLI and common library

```bash
uv sync
# invokes ./ejepa (repo wrapper → .venv/bin/ejepa)
./ejepa --version
```

### 2. Install green/purple virtualenvs

Recommended — creates per-agent `.venv` directories and installs locked deps (including `llm_shell` / `terminus_2` purple extras):

```bash
scripts/install.sh terminal-bench-2.0
```

Or with `uv` directly:

```bash
uv sync --project assets/Terminal-Bench-2.0/green
uv sync --project assets/Terminal-Bench-2.0/purple --extra llm_shell
```

Green requires **Python 3.12+** (`requires-python` in `green/pyproject.toml`). Green also pins **`harbor>=0.8.0`**; older Harbor releases (for example `0.3.0`) lack `harbor.environments.capabilities` and the green agent will fail on import.

If the green agent exits immediately with “dynamic port” errors, the project venv is usually empty. Re-run `scripts/install.sh terminal-bench-2.0` or the `uv sync` commands above.

Lock files (package-age policy):

```bash
uv lock --project assets/Terminal-Bench-2.0/green --exclude-newer "7 days ago"
uv lock --project assets/Terminal-Bench-2.0/purple --exclude-newer "7 days ago"
```

### 3. Clone the task corpus

See [Prerequisites](#prerequisites) step 2.

### 4. Configure Docker and local storage

See [Prerequisites](#prerequisites) steps 1 and the storage note under Docker.

### 5. Export Azure credentials (for LLM runs)

See [Prerequisites](#prerequisites) step 3. When using Azure, unset OpenAI-compatible overrides left over from remote inference or other benchmarks:

```bash
unset OPENAI_BASE_URL OPENAI_API_BASE_URL INFERENCE_DEFAULT_BASE_URL INFERENCE_DEFAULT_MODEL
```

---

## Quick Start

Recommended local smoke (Azure GPT-5.5, trajectory capture, local storage paths):

```bash
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
```

Or run `ejepa` directly (set `BENCHMARK_HOME` on local machines — see [Prerequisites](#prerequisites)):

```bash
export BENCHMARK_HOME="$PWD/.cache/benchmark-home"
export TERMINAL_BENCH_WORKSPACE="$PWD/.cache/terminal-bench-workspace"

# Oracle smoke test (runs solve.sh locally, no LLM)
./ejepa bench run Terminal-Bench-2.0 \
  --task-id fix-git \
  --config oracle=true

# LLM agent on the sample target
./ejepa bench run Terminal-Bench-2.0 \
  --executor llm_shell \
  --task-id fix-git \
  --config capture_trajectories=true \
  --ready-timeout 600 \
  --show-logs

# All tasks listed under target "all" (requires full repo checkout)
./ejepa bench run Terminal-Bench-2.0 \
  --target all \
  --executor llm_shell

# Terminus-2 agent (batch commands, LiteLLM)
./ejepa bench run Terminal-Bench-2.0 \
  --executor terminus_2 \
  --task-id fix-git

Add `target = "all"` to `green/tasks/task_ids.toml` or pass `--target all` once the full repo is present.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `permission denied` on `/var/run/docker.sock` | Add your user to the `docker` group (`sudo usermod -aG docker $USER`), log out/in, then `scripts/check-docker-access.sh` |
| `PermissionError on the default result root` | Set `BENCHMARK_HOME` to a writable local path (see Prerequisites) |
| `ModuleNotFoundError: harbor.environments.capabilities` | Re-run `scripts/install.sh terminal-bench-2.0` — green needs `harbor>=0.8.0` |
| `task_repo_missing` / task not found | Clone upstream into `assets/Terminal-Bench-2.0/tasks/terminal-bench-2` |
| Azure 404 on deployment | Set `AZURE_OPENAI_DEPLOYMENT_NAME` to your Azure **deployment name**, not the raw model id |
| LLM hits wrong endpoint | Unset `OPENAI_BASE_URL` / inference env vars when using Azure |
| Green “dynamic port” exit | Empty green venv — run `uv sync --project assets/Terminal-Bench-2.0/green` |

---

## Protocol

Green and purple exchange JSON messages over A2A:

| Message | Direction | Purpose |
|---------|-----------|---------|
| `task` | green → purple | Task instruction |
| `exec_request` | purple → green | Shell command to run in the task container |
| `exec_result` | green → purple | `exit_code`, `stdout`, `stderr` |
| `final` | purple → green | Agent finished |

Green executes commands inside the Harbor-managed Docker environment and runs Harbor verification to produce the score.

## Configuration

| Key / env | Description |
|-----------|-------------|
| `config.target` | Group from `green/tasks/task_ids.toml` (default: `sample`) |
| `--task-id` | Single or multiple task names |
| `config.oracle` | `true` runs `solution/solve.sh` locally (no purple LLM) |
| `config.max_parallel` | Reserved for future parallel task runs (currently serial Docker lock) |
| `config.num_shards` / `shard_index` | Round-robin sharding across green instances |
| `TERMINAL_BENCH_WORKSPACE` | Host path for trials/tasks cache (default: `/tmp/tb-workspace`) |
| `TERMINAL_BENCH_TASK_REPO` | Path to `terminal-bench-2` checkout |
| `TERMINAL_BENCH_MAX_STEPS` | Max LLM turns per task (`llm_shell` default: `40`, `terminus_2`: `80`) |
| `TERMINUS_2_PARSER` | `json` or `xml` response format (default: `json`) |
| `TERMINUS_2_TEMPERATURE` | LiteLLM sampling temperature (default: `0.7`) |
| `AZURE_OPENAI_API_KEY` | Azure API key (auto-selected when endpoint is set) |
| `AZURE_OPENAI_ENDPOINT` | Azure resource endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT_NAME` | Azure deployment name (default with Azure: `gpt-5.5`) |
| `AZURE_OPENAI_API_VERSION` | Azure API version (default: `2024-10-21`) |
| `config.capture_trajectory` / `config.capture_trajectories` | `true` saves per-task JSONL trajectories under `{result_dir}/trajectories/` |
| `BENCHMARK_CAPTURE_TRAJECTORY` | Env fallback to enable trajectory capture (`1`, `true`, `yes`) |

Trajectory files include A2A client events, shell protocol messages (`task` / `exec_request` / `exec_result` / `final`), container command results, and optional `llm_shell` LLM message history (redacted secrets, truncated long strings). Failed tasks still write `{task_id}.jsonl` with any partial events plus a final `TaskFailed` record.

```bash
ejepa bench run Terminal-Bench-2.0 \
  --executor llm_shell \
  --task-id fix-git \
  --config capture_trajectory=true
```

---

## Scoring

Each task returns a Harbor verifier `reward` (typically `0` or `1`). The run summary reports total score, pass count, and per-task details under the standard `ejepa` result directory (`detail.json`, `manifest.json`).
