# EnterpriseOps-Gym

[EnterpriseOps-Gym](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym) is a containerized,
resettable enterprise simulation benchmark for evaluating LLM agents on stateful, multi-step planning
and tool use across realistic enterprise workflows. It comprises **1,150 expert-curated tasks** spanning
**8 enterprise domains** — Calendar, CSM, Drive, Email, HR, ITSM, Teams, and Hybrid — running against
live containerized MCP servers backed by realistic, fully synthetic databases. Tasks are evaluated on
**final environment state** using SQL verifiers, not on action sequences.

This benchmark wraps the upstream
[ServiceNow/EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym) evaluation framework so
that the Green/Purple agent harness can drive it end-to-end:

- **Green Agent** loads tasks (from a bundled local sample or directly from Hugging Face), forwards
each task to the Purple Agent, and aggregates per-task `overall_success` and `verifier_pass_rate`.
- **Purple Agent** routes each request to the `mcp_react` executor, which delegates to the upstream
`BenchmarkExecutor` to (1) connect to the live MCP server(s), (2) run the agent loop with the
configured orchestrator (`react` / `planner_react` / `decomposing`), and (3) execute the SQL
verifiers against the resulting database state.

---

## Layout

```
EnterpriseOps-Gym/
├── benchmark.toml
├── README.md
├── green/
│   ├── pyproject.toml
│   ├── enterpriseops_green_agent.py
│   ├── task_loader.py
│   └── tasks/
│       ├── task_ids.toml
│       └── Tasks_SAMPLE.json
├── purple/
│   ├── pyproject.toml
│   └── enterpriseops_purple_agent.py
└── purple-executors/
    └── mcp_react/
        ├── version.toml
        └── executor.py
```

---

## Prerequisites

- Linux host (tested on Ubuntu 22.04+) with **Python 3.11+**
- **Docker** or **Singularity/Apptainer** (for the per-domain MCP servers)
- `**uv`** (for syncing the per-component virtual environments)
- An LLM API key for at least one provider supported by the upstream framework, **or** remote
  inference via Slurm (see [Remote inference (Slurm cluster)](#remote-inference-slurm-login-cluster))

```bash
# Ubuntu — install system packages
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip git unzip curl docker.io
sudo usermod -aG docker "$USER" && newgrp docker

# Install uv if it isn't already on PATH
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setup

### 1. Install the green and purple environments

From the repository root (the directory that contains `ejepa`, `scripts/install.sh`, and `assets/`):

```bash
scripts/install.sh enterpriseops-gym
```

This creates:

- the root `.venv/` (the `ejepa` CLI lives here)
- `assets/EnterpriseOps-Gym/green/.venv`
- `assets/EnterpriseOps-Gym/purple/.venv`

Now add the dependencies the `mcp_react` executor needs into the **purple** venv. Pick the extra
that matches your provider:

```bash
# OpenAI / Azure OpenAI
uv sync --project assets/EnterpriseOps-Gym/purple --extra openai

# Other providers
# uv sync --project assets/EnterpriseOps-Gym/purple --extra anthropic
# uv sync --project assets/EnterpriseOps-Gym/purple --extra google
# uv sync --project assets/EnterpriseOps-Gym/purple --extra deepseek
```

And add the HuggingFace loader to the **green** venv (only needed for `target=hf_dataset`):

```bash
uv sync --project assets/EnterpriseOps-Gym/green --extra hf
```

### 2. Clone and prepare the upstream evaluation repository

The `mcp_react` executor imports the upstream `BenchmarkExecutor` in-process and reads seed-database
SQL files that live inside the upstream repo. Clone it and unzip the seed databases:

```bash
git clone https://github.com/ServiceNow/EnterpriseOps-Gym.git ~/EnterpriseOps-Gym
cd ~/EnterpriseOps-Gym
unzip gym_dbs.zip   # creates "Domain Wise DBs and Task-DB Mappings/<domain>/dbs/*.sql"
```

Tell our executor where it lives (consider adding this to `~/.bashrc`):

```bash
export ENTERPRISEOPS_GYM_REPO_PATH="$HOME/EnterpriseOps-Gym"
```

### 3. Pull and start the MCP server containers

Pull and start one container per domain you plan to evaluate. The `--name` flag lets you later run
`docker logs <name>`, `docker stop <name>`, `docker restart <name>` without looking up auto-generated
container IDs.

```bash
# Pull all 7 domain images (~2 GB total)
docker pull shivakrishnareddyma225/enterpriseops-gym-mcp-calendar:latest
docker pull shivakrishnareddyma225/enterpriseops-gym-mcp-teams:latest
docker pull shivakrishnareddyma225/enterpriseops-gym-mcp-csm:latest
docker pull shivakrishnareddyma225/enterpriseops-gym-mcp-email:latest
docker pull shivakrishnareddyma225/enterpriseops-gym-mcp-itsm:latest
docker pull shivakrishnareddyma225/enterpriseops-gym-mcp-hr:latest
docker pull shivakrishnareddyma225/enterpriseops-gym-mcp-drive:latest

# Start them on the host ports the dataset expects
docker run -d --name gym-calendar -p 8003:8003 shivakrishnareddyma225/enterpriseops-gym-mcp-calendar:latest
docker run -d --name gym-teams    -p 8002:8005 shivakrishnareddyma225/enterpriseops-gym-mcp-teams:latest
docker run -d --name gym-csm      -p 8001:8005 shivakrishnareddyma225/enterpriseops-gym-mcp-csm:latest
docker run -d --name gym-email    -p 8004:8005 shivakrishnareddyma225/enterpriseops-gym-mcp-email:latest
docker run -d --name gym-itsm     -p 8006:8005 shivakrishnareddyma225/enterpriseops-gym-mcp-itsm:latest
docker run -d --name gym-hr       -p 8008:8005 shivakrishnareddyma225/enterpriseops-gym-mcp-hr:latest
docker run -d --name gym-drive    -p 8009:8005 shivakrishnareddyma225/enterpriseops-gym-mcp-drive:latest
```

| Domain     | MCP server name        | Host port | Container port |
| ---------- | ---------------------- | --------- | -------------- |
| `calendar` | `gym-calendar`         | 8003      | 8003           |
| `teams`    | `gym-teams-mcp`        | 8002      | 8005           |
| `csm`      | `sn-csm-server`        | 8001      | 8005           |
| `email`    | `gym-email-mcp`        | 8004      | 8005           |
| `itsm`     | `gym-itsm-mcp`         | 8006      | 8005           |
| `hr`       | `sn-hr-internal`       | 8008      | 8005           |
| `drive`    | `gym-google-drive-mcp` | 8009      | 8005           |

Wait ~15 s for the containers to become healthy, then verify each port:

```bash
docker ps --filter name=gym- --format 'table {{.Names}}\t{{.Status}}'

for port in 8001 8002 8003 8004 8006 8008 8009; do
  printf "port %s -> " "$port"
  curl -s -o /dev/null -w "HTTP %{http_code}\n" --max-time 5 "http://localhost:${port}/" || echo "FAIL"
done
```

You should see `(healthy)` for each container and `HTTP 307` (or `HTTP 200`) for each port.

### 4. Configure the LLM used by the executor

Use hosted APIs (Azure OpenAI, OpenAI, Anthropic, …) **or** [remote inference on Slurm](#remote-inference-slurm-login-cluster)
with `--inference-config` and `ENTERPRISEOPS_LLM_PROVIDER=vllm`.

Either provide a JSON config file (recommended — same format as upstream's `conf/llm/*.json`) …

```bash
mkdir -p ~/EnterpriseOps-Gym/conf/llm
cat > ~/EnterpriseOps-Gym/conf/llm/my-model.json <<'EOF'
{
    "llm_provider": "azureopenai",
    "llm_model": "gpt-4.1",
    "llm_api_key": "<your-api-key>",
    "llm_api_endpoint": "https://<your-resource>.openai.azure.com",
    "llm_api_version": "2025-04-01-preview",
    "temperature": 0.0,
    "max_tokens": 16384
}
EOF
export ENTERPRISEOPS_LLM_CONFIG_FILE="$HOME/EnterpriseOps-Gym/conf/llm/my-model.json"
```

… or pass the values via environment variables:

```bash
export ENTERPRISEOPS_LLM_PROVIDER=openai
export ENTERPRISEOPS_LLM_MODEL=gpt-5.4-mini
export ENTERPRISEOPS_LLM_API_KEY="sk-..."
export ENTERPRISEOPS_LLM_TEMPERATURE=0.0
export ENTERPRISEOPS_LLM_MAX_TOKENS=4096
```

For `planner_react` or `decomposing` orchestrators, also export
`ENTERPRISEOPS_PLANNER_LLM_CONFIG_FILE=/path/to/planner_llm.json`.

### 5. Pick a writable result directory

By default Green writes results under `$BENCHMARK_HOME/experiments`. On a stock machine, override that:

```bash
export BENCHMARK_HOME="$HOME/eog-benchmark home"
# or, more directly:
# export BENCHMARK_RESULT_ROOT="$HOME/eog-experiments"
```

## Remote inference (Slurm cluster)

EnterpriseOps-Gym can drive a **remote vLLM server on the Slurm Slurm cluster** instead of
calling a hosted API directly. The benchmarks repo ships
[`remote-inference-launcher`](../../packages/remote-inference-launcher/README.md); `ejepa bench run`
starts the remote endpoint, opens an SSH tunnel to your laptop, and injects generic env vars
(`OPENAI_BASE_URL`, `OPENAI_MODEL_NAME`) that the `mcp_react` executor forwards to the upstream
`vllm` provider.

**Architecture**

```text
[Your laptop]                         [Slurm]
  ejepa bench run                          Slurm + vLLM (Qwen3.5)
  Green/Purple + MCP containers  ──►  OpenAI-compatible /v1 API
  (localhost:8001–8009)                  (GPU node via SSH tunnel)
```

MCP server containers still run **locally** on the machine that executes `ejepa`. Only the LLM
inference is remote.

### Prerequisites

Complete [Setup](#setup) sections 1–3 and 5 first. In addition:

- SSH access to Slurm (`ssh slurm-login`) — see
  [`remote-inference-launcher` SSH setup](../../packages/remote-inference-launcher/README.md#ssh-setup)
- Root workspace synced: `uv sync` at the benchmarks repo root
- A bootstrapped vLLM environment on Slurm (one-time)
- For full-corpus runs: all domain MCP containers running (section 3) and green HF extra installed
  (`uv sync --project assets/EnterpriseOps-Gym/green --extra hf`)

### One-time: bootstrap vLLM on Slurm

Create `bootstrap-slurm-login.yaml` at the repo root (**no `kind` field** — used with
`slurm-vllm-bootstrap` directly):

```yaml
ssh_target: slurm-login
environment_name: qwen35-vllm-cu129
venv_path: ~/.qwen35-vllm-cu129
partition: batch-2gpu
walltime: "1:00:00"
num_gpus: 1
memory: 32GB
cpus_per_task: 4
vllm_package: vllm==0.19.1
install_uv_if_missing: true
```

```bash
cd /path/to/benchmarks
uv run remote-inference-launcher slurm-vllm-bootstrap \
  --config bootstrap-slurm-login.yaml
```

Note the printed `python_bin` (for example `~/.qwen35-vllm-cu129/bin/python`) for the serving config.

### Inference config

The repo includes [`inference-slurm-login.yaml`](../../inference-slurm-login.yaml) for Qwen/Qwen3.5-27B.
Important fields:

| Field | Purpose |
| ----- | ------- |
| `kind: slurm_vllm` | Required for `ejepa --inference-config` / `remote-inference-launcher start` |
| `ssh_target: slurm-login` | SSH alias for the Slurm login node |
| `python_bin` | Path from bootstrap step |
| `partition`, `walltime`, `num_gpus`, `memory` | Slurm allocation — must fit partition limits (`sinfo` on Slurm) |
| `extra_args` | vLLM flags for tool calling (required by `mcp_react`) |
| `setup_cmd` | Installs `ninja` on the GPU node (FlashInfer JIT needs it at first inference) |

Tool-calling flags validated on vLLM 0.19.1:

```yaml
extra_args:
  - --enable-auto-tool-choice
  - --tool-call-parser
  - qwen3_coder
setup_cmd: |
  export PATH="$HOME/.qwen35-vllm-cu129/bin:$PATH"
  python -m pip install -q ninja
```

Do **not** use `--tool-call-parser qwen3` on vLLM 0.19.1 — that parser name is invalid and the
Slurm job will fail at startup.

Set `walltime` to the **maximum allowed by your partition** (for example `24:00:00` on
`batch-8gpu`). Values above the partition limit leave the job stuck in `PENDING`
(`PartitionTimeLimit`).

### Executor wiring

For remote vLLM, set the upstream provider to `vllm` and **clear any hosted-API overrides** so
`ejepa --inference-config` can inject the tunneled endpoint:

```bash
export ENTERPRISEOPS_LLM_PROVIDER=vllm
export ENTERPRISEOPS_LLM_TEMPERATURE=0.0
export ENTERPRISEOPS_LLM_MAX_TOKENS=8192
export BENCHMARK_A2A_CLIENT_TIMEOUT=1800

unset ENTERPRISEOPS_LLM_API_ENDPOINT ENTERPRISEOPS_LLM_MODEL ENTERPRISEOPS_LLM_API_KEY
unset ENTERPRISEOPS_LLM_API_VERSION ENTERPRISEOPS_LLM_CONFIG_FILE
unset AZURE_OPENAI_API_KEY AZURE_OPENAI_ENDPOINT 2>/dev/null || true
```

`ejepa` sets `OPENAI_BASE_URL` and `OPENAI_MODEL_NAME` from the inference config; `mcp_react` maps
those to the upstream `vllm` client automatically.

### Smoke test (one sample task)

Requires only `gym-calendar` on `localhost:8003`.

```bash
cd /path/to/benchmarks
export ENTERPRISEOPS_GYM_REPO_PATH="${ENTERPRISEOPS_GYM_REPO_PATH:-$PWD/.cache/EnterpriseOps-Gym}"
export BENCHMARK_HOME="${BENCHMARK_HOME:-$PWD/.cache/benchmark home}"
export ENTERPRISEOPS_LLM_PROVIDER=vllm
export ENTERPRISEOPS_LLM_TEMPERATURE=0.0
export ENTERPRISEOPS_LLM_MAX_TOKENS=8192

ejepa bench run EnterpriseOps-Gym \
  --executor mcp_react \
  --inference-config inference-slurm-login.yaml \
  --config target=sample \
  --task-id enterpriseops_sample_calendar_001 \
  --ready-timeout 600 \
  --show-logs
```

The first run waits several minutes while Slurm submits Slurm, loads Qwen3.5-27B, and opens the SSH
tunnel. `Connection refused` on the local port during that window is normal.

### Full corpus with trajectory capture

For all ~1,150 tasks across every domain, start **all** MCP containers (section 3), then either
use the helper script or run `ejepa` directly.

Helper script (checks MCP ports, logs to `.cache/benchmark home/logs/`):

```bash
scripts/run_enterpriseops_slurm_qwen_full.sh
```

Equivalent `ejepa` command:

```bash
ejepa bench run EnterpriseOps-Gym \
  --executor mcp_react \
  --inference-config inference-slurm-login.yaml \
  --config target=hf_dataset \
  --config mode=oracle \
  --config 'domains=["calendar","csm","drive","email","hr","hybrid","itsm","teams"]' \
  --config capture_trajectory=true \
  --config max_parallel=1 \
  --ready-timeout 600 \
  --show-logs
```

Omit `max_tasks_per_domain` to run the full Hugging Face split per domain. Trajectories are
written under `<result_dir>/trajectories/<task_id>.jsonl` (see [Trajectory Capture](#trajectory-capture)).

Tune concurrency with `MAX_PARALLEL=1 scripts/run_enterpriseops_slurm_qwen_full.sh` or
`--config max_parallel=N`. Keep `max_parallel=1` unless you run multiple vLLM endpoints (fleet
config) — a single remote server is easier to saturate with concurrent tool-calling requests.

### Two-terminal workflow (reuse a running endpoint)

To avoid submitting a new Slurm job on every `ejepa` invocation, keep inference alive in one
terminal and point the benchmark at it from another.

Terminal 1 — start and hold remote vLLM:

```bash
uv run remote-inference-launcher start \
  --config inference-slurm-login.yaml \
  --env-file /tmp/inference.env
```

Terminal 2 — run the benchmark without `--inference-config`:

```bash
source /tmp/inference.env
export ENTERPRISEOPS_LLM_PROVIDER=vllm
export ENTERPRISEOPS_LLM_TEMPERATURE=0.0
export ENTERPRISEOPS_LLM_MAX_TOKENS=8192

ejepa bench run EnterpriseOps-Gym \
  --executor mcp_react \
  --config target=sample \
  --task-id enterpriseops_sample_calendar_001 \
  --ready-timeout 600 \
  --show-logs
```

Alternatively, use an `existing_endpoint` config if the tunnel is already up:

```yaml
# inference-local.yaml
kind: existing_endpoint
api_base: http://127.0.0.1:36311/v1   # your local tunnel port
served_model_name: Qwen/Qwen3.5-27B
```

```bash
ejepa bench run EnterpriseOps-Gym \
  --executor mcp_react \
  --inference-config inference-local.yaml \
  ...
```

### Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Slurm job `PENDING (PartitionTimeLimit)` | `walltime` exceeds partition max | Lower `walltime` in `inference-slurm-login.yaml`; check `sinfo` on Slurm |
| `Connection refused` while waiting | vLLM still loading | Wait; tail remote `vllm.log` under the path printed by the launcher |
| `invalid tool call parser: qwen3` | Wrong parser name for vLLM 0.19.1 | Use `qwen3_coder` in `extra_args` |
| `auto tool choice requires --enable-auto-tool-choice` | Missing vLLM tool flags | Add `--enable-auto-tool-choice` and `--tool-call-parser qwen3_coder` |
| `FileNotFoundError: ninja` on first chat | FlashInfer JIT on GPU node | Keep `setup_cmd` that installs `ninja`, or bootstrap `ninja` into the venv |
| Benchmark hits Azure/hosted URL instead of tunnel | Stale `ENTERPRISEOPS_LLM_*` or Azure env | `unset` hosted-API vars before `ejepa bench run` (see above) |
| `unsupported Slurm vLLM bootstrap fields: kind` | `kind` in bootstrap YAML | Remove `kind` for `slurm-vllm-bootstrap`; use `kind: slurm_vllm` only with `start` / `ejepa --inference-config` |

Remote logs on Slurm:

```bash
ssh slurm-login 'tail -f ~/tmp/remote-inference-launcher/slurm-vllm/*/vllm.log'
```

More detail: [`packages/remote-inference-launcher/README.md`](../../packages/remote-inference-launcher/README.md).

---

## Quick Start

> Run `ejepa bench run` through this repository's `./ejepa` wrapper (or `.venv/bin/ejepa`).

```bash
# Smoke test: a single bundled task (real calendar/oracle row, copied locally).
# Requires gym-calendar running on localhost:8003.
ejepa bench run EnterpriseOps-Gym --executor mcp_react --config target=sample

# Real evaluation: stream from Hugging Face. Pick mode + domain + how many tasks per domain.
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --config target=hf_dataset \
  --config mode=oracle \
  --config domain=calendar \
  --config max_tasks_per_domain=5

# Multi-domain run (parallelism 3) — needs the corresponding MCP containers up.
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --config target=hf_dataset \
  --config mode=oracle \
  --config 'domains=["calendar","teams","csm","email","itsm","drive"]' \
  --config max_tasks_per_domain=5 \
  --config max_parallel=3

# Pick a specific real task by ID.
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --config target=hf_dataset \
  --config domain=email \
  --task-id task_20260107_132943_755_911d75d7_4c944103

# Increase tool-retrieval difficulty (oracle vs +5 / +10 / +15 distractor tools).
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --config target=hf_dataset \
  --config mode=plus_5_tools \
  --config domain=teams \
  --config max_tasks_per_domain=10

# Remote inference smoke test (Slurm + Qwen3.5) — see Remote inference section.
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --inference-config inference-slurm-login.yaml \
  --config target=sample \
  --task-id enterpriseops_sample_calendar_001 \
  --ready-timeout 600 \
  --show-logs
```

## Trajectory Capture

Enable per-task trajectory export with either `--config capture_trajectory=true` or
`BENCHMARK_CAPTURE_TRAJECTORY=1`:

```bash
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --config target=sample \
  --config capture_trajectory=true
```

Green writes JSONL files under `<result_dir>/trajectories/`. The records include A2A request/response
events plus Purple executor internals forwarded as `purple_internal` events: the system/user prompt,
final assistant response, upstream run records, tool usage, verifier results, and execution summary.

For Anthropic-compatible runs, the executor can read `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL`, and
`ANTHROPIC_MODEL`; equivalent `ENTERPRISEOPS_LLM_*` variables still take precedence.

---

## Targets

| `config.target`   | Description                                                                                                                                                                                                                                                     |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sample`          | A single bundled local task copied from the real Hugging Face corpus (calendar / oracle). Use it to confirm Green ↔ Purple ↔ MCP wiring against a real seed database. **Not** suitable as an actual benchmark — for measurement always use `target=hf_dataset`. |
| `sample_calendar` | Same single task, listed under its own target so you can pin it explicitly.                                                                                                                                                                                     |
| `hf_dataset`      | Stream tasks directly from `[ServiceNow-AI/EnterpriseOps-Gym](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym)`. Requires the optional `datasets` dependency on the green side. **This is the right target for real evaluation.**               |

### Hugging Face configuration keys

When `target = "hf_dataset"`, the following keys are honored:

| `config` key                | Description                                                                                                                              |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `mode` / `modes`            | HF dataset config name(s). Defaults to `oracle`. Supported: `oracle`, `plus_5_tools`, `plus_10_tools`, `plus_15_tools`.                  |
| `domain` / `domains`        | HF split name(s) (= enterprise domain names). Required. Singular accepts a string; plural accepts a JSON list or comma-separated string. |
| `max_tasks_per_domain`      | Max tasks to take per domain. Omit or set to `null` for the full split.                                                                  |
| `task_ids` (or `--task-id`) | When provided, restricts to the listed task IDs and ignores `max_tasks_per_domain`.                                                      |
| `orchestrator`              | One of `react` (default), `planner_react`, `decomposing`.                                                                                |
| `max_iterations`            | Max ReAct iterations forwarded to the upstream orchestrator.                                                                             |
| `max_parallel`              | Concurrent tasks Green will dispatch. Defaults to 1.                                                                                     |

Equivalent environment variables for headless usage: `ENTERPRISEOPS_HF_MODE`,
`ENTERPRISEOPS_HF_DOMAIN`, `ENTERPRISEOPS_HF_MAX_TASKS`, `ENTERPRISEOPS_HF_TASK_IDS`.

Supported domains: `calendar`, `csm`, `drive`, `email`, `hr`, `hybrid`, `itsm`, `teams`.

---

## Scoring

Scoring is **outcome-based**, identical to the upstream framework:

- Each task is given **score = 1.0** if every SQL verifier reported by the upstream
`BenchmarkExecutor` passes (`overall_success = True`); otherwise **score = 0.0**.
- Per-task `verifier_pass_rate` (`passed / total`) is recorded in `detail.json` for partial-credit
diagnostics. The benchmark also reports `avg_verifier_pass_rate` over all tasks.

This mirrors the **Avg Success Rate** column in the upstream `compute_score.py` table.

---

## EWM world model over MCP

Distinct from the **domain tool MCP servers** above (the `gym-calendar`/`gym-email`/… containers
the agent *acts* through), the **Enterprise World Model (EWM)** can be served as its own MCP
server and consulted by the agent to *predict* outcomes before acting. The server and engine live
under `src/ejepa_wm/` — this is just the EnterpriseOps entry point.

Two helper scripts in this directory drive it (run them from the repo root):

```bash
# 0. The EWM LoRA adapter ships in-repo via Git LFS at
#    assets/EnterpriseOps-Gym/models/gymops_world_model — fetch the weights on a fresh checkout:
git lfs pull

# 1. Start the EWM world model on vLLM (:9000) + the EWM predict MCP server (:12072).
#    The default EWM_LORA points at the in-repo adapter; MCP_PYTHON is a venv with fastmcp.
MCP_PYTHON=/path/to/venv/bin/python ./assets/EnterpriseOps-Gym/build_mcp.sh
#    (override EWM_LORA=gymops_world_model=/your/checkpoint to use a different adapter)

# 2. Run a task with the imagined-trajectory tool over MCP (drops per-step predict_state).
#    Needs a tool-calling agent LLM (ENTERPRISEOPS_LLM_* / .env.ewm).
./assets/EnterpriseOps-Gym/run_mcp.sh        # TARGET=sample by default
```

The EWM predict MCP server exposes three tools — `predict_state` (single-action feasibility),
`generate` (raw WM passthrough, used by imagined rollouts), and `info`. The `mcp_react` executor
wires them to the agent via env vars (no code change):

| Env var | Effect |
|---|---|
| `EWM_PREDICT_MCP_URL` | register `predict_state` as an agent tool (state injected from the flow). |
| `EWM_IMAGINE_TOOL=1` | register the forced K-step `imagine_trajectory` rollout tool; **drops** `predict_state` (keep both with `EWM_KEEP_PREDICT_WITH_IMAGINE=1`). |
| `EWM_IMAGINE_EVERY_STEP=1` | auto-run the rollout **before every step** and inject it through the `imagine_trajectory` tool channel (persisted tool result, not transient) — a guaranteed fresh look-ahead each step. `EWM_IMAGINE_SUPERSEDE=1` (default) stubs the prior auto-trajectory so context doesn't grow. |
| `WM_EWM_MCP_URL` | route the imagined rollout's WM calls over MCP (the `generate` tool). |
| `WM_IMAGINED_MAX_STEPS` | imagined-rollout depth (receding horizon — re-imagine as the task progresses). |
| `WM_STATE` | prediction mode: `binary_error` (default) / `binary_error_stage` / `tool_output`. |

For the always-on variant (a fresh rollout injected before *every* step), use
`--wm-strategy imagined` instead of the `imagine_trajectory` tool.

**Full details:** see [`src/ejepa_wm/README.md`](../../src/ejepa_wm/README.md) (server tools, backends,
the `ewm_predict`/`ewm_imagined` strategies, all env vars) and the verification procedure in
[`procedure_mcp.md`](../../procedure_mcp.md) (Steps 1–7, including the EWM predict-server smoke).

---

## Notes

- Loading the official corpus performs a `datasets.load_dataset(...)` call against Hugging Face on
the green side. The first run downloads the dataset into the standard `~/.cache/huggingface`
cache; subsequent runs are served from the cache. For air-gapped environments, pre-warm the cache
or configure a mirror via `HF_HOME` / `HF_ENDPOINT`.
- Each task in the corpus pins a specific `seed_database_file` (relative to the unzipped
`gym_dbs.zip` content). The `mcp_react` executor automatically rewrites these to absolute paths
under `ENTERPRISEOPS_GYM_REPO_PATH`, so always make sure the upstream repo has been unzipped and
that env var points at it. To verify:
`ls "${ENTERPRISEOPS_GYM_REPO_PATH}/Domain Wise DBs and Task-DB Mappings"`.
- Both the LLM-driven task execution and the SQL verification happen on the Purple side, since
verifiers must execute SQL through the same MCP backends the agent just mutated. The Green agent
purposefully does **not** hold a copy of the seed databases.

---

## References

- Upstream evaluation framework: [https://github.com/ServiceNow/EnterpriseOps-Gym](https://github.com/ServiceNow/EnterpriseOps-Gym)
- Hugging Face dataset: [https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym](https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym)
- Project page: [https://enterpriseops-gym.github.io/](https://enterpriseops-gym.github.io/)
- Paper: Malay et al., *EnterpriseOps-Gym: Environments and Evaluations for Stateful Agentic Planning
and Tool Use in Enterprise Settings*, arXiv:2603.13594.
