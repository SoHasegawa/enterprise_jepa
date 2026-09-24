# crmarenapro

CRMArenaPro benchmark asset, ported from an internal entropic-CRMArenaPro fork.

The Green agent implements the entropic CRMArenaPro evaluation flow with schema drift,
context rot, original CRMArena-Pro scoring, and 7D scoring. The Purple agent uses the
baseline CRM agent from `baseline-crm-agent` through the benchmark-local
`baseline_crm_agent` executor.

## Layout

```text
crmarenapro/
├── benchmark.toml
├── green/
│   ├── crmarenapro_green_agent.py
│   ├── agent.py
│   ├── executor.py
│   ├── messenger.py
│   ├── crm/
│   ├── original/
│   ├── shared/
│   └── data/crmarena_b2b_tasks.json
├── purple/
│   └── crmarenapro_purple_agent.py
└── purple-executors/
    └── baseline_crm_agent/
        ├── executor.py
        ├── agent.py
        ├── messenger.py
        └── version.toml
```

## Setup

```bash
scripts/install.sh crmarenapro
```

This installs the green/purple venvs, adds `ejepa` shims, and attempts to extract
the CRMArenaPro SQLite databases from `ghcr.io/rkstu/baseline-crm-agent:latest`
into `purple-executors/baseline_crm_agent/data/`.
After setup, these should work:

```bash
assets/crmarenapro/green/.venv/bin/ejepa --version
assets/crmarenapro/purple/.venv/bin/ejepa --version
```

For the baseline Purple agent, configure one of the supported LLM credentials:

```bash
export ANTHROPIC_API_KEY="..."
# or
export OPENAI_API_KEY="..."
# or
export NEBIUS_API_KEY="..."
```

Optional runtime settings:

```bash
export LLM_MODEL="claude-3-5-sonnet-latest"
export LLM_BASE_URL="https://api.anthropic.com"
export MAX_TURNS=8
```

The baseline Purple agent needs the CRMArenaPro SQLite database for real scoring.
If Docker extraction is unavailable, place `crmarenapro_b2b_data.db` under
`purple-executors/baseline_crm_agent/data/`, or set `CRMARENAPRO_DB_PATH` /
`CRMARENAPRO_DB_DIR`. When the database is absent, the agent returns a visible
setup message or a best-effort context-only answer instead of silently returning
`None` for every task.

For debugging only, `CRMARENAPRO_BASELINE_ENABLE_LOCAL_ANSWERS=1` enables an
oracle mode that reads gold answers from `green/data/crmarena_b2b_tasks.json`;
do not use it for scoring.

## Quick Start

```bash
ejepa bench run crmarenapro --executor baseline_crm_agent
```

The default config uses `task_limit=1` for a short smoke run. Omit or override it to run
more tasks from `green/data/crmarena_b2b_tasks.json`.

### Config keys

| key | default | meaning |
|---|---|---|
| `target` | `sample` | Named task split from `green/tasks/task_ids.toml` (e.g. `world_model_test`) |
| `task_limit` | `1` | Tasks to run; `0` = the whole target |
| `max_parallel` | `1` | Concurrent tasks dispatched to Purple |
| `capture_trajectory` | `false` | Write per-task JSONL trajectories |
| `leaderboard_mode` | `false` | Disable the adversarial perturbations (see below) |

`max_parallel > 1` is safe with a world model: Green opens a fresh A2A context per task and
Purple builds one world model per context, so no episode state (`beam_plan` MPC plan, critic
counters) is shared between concurrent tasks. Model weights *are* shared process-wide
(`WM_SHARE_MODEL_WEIGHTS`, defaulted on by the executor), so a JEPA checkpoint loads once
rather than once per task. Results are re-sorted into dataset order, so `detail.json` is
unaffected by completion order.

**`timeout` is not configurable unless `leaderboard_mode=true`.** Outside leaderboard mode
`drift_level`, `rot_level`, `max_steps` and `timeout` are hardcoded to `medium`/`medium`/`10`/
`300` seconds (`green/agent.py`), so `--config timeout=...` is silently ignored and a task
exceeding 300 s scores `0.0`. `leaderboard_mode=true` honours `timeout` (default 600) and
`max_steps` (default 20) but also turns the perturbations off, which changes what is measured.

## `mcp_react` executor

`purple-executors/mcp_react/` is a ReAct-over-tools executor presented as an MCP-style tool
catalog, mirroring the `mcp_react` executor on EnterpriseOps-Gym. CRMArenaPro has no MCP tool
servers — the tools are the CRM database operations — so it reuses the `baseline_crm_agent`
runtime (`crm_task` parsing, the SQLite CRM tool layer, LLM plumbing, and the `Answer` /
trajectory artifacts) and only reframes the system prompt into an explicit tool catalog +
ReAct loop (`execute` / `describe` / `respond`). It is functionally close to
`baseline_crm_agent`; use it when you want the ReAct/MCP framing.

```bash
ejepa bench run crmarenapro --executor mcp_react --config target=world_model_test
```

For a faster stress-oriented evaluation, `world_model_test_longest_100` selects
100 tasks from `world_model_test` with the longest observed baseline trajectories.
Ties at the eight-turn cap are resolved by numeric task ID.

```bash
ejepa bench run crmarenapro --executor mcp_react \
  --config target=world_model_test_longest_100 --config task_limit=0
```


### World-model advice (optional, incl. JEPA)

Like EnterpriseOps-Gym's `wm_react`, `mcp_react` can consult a pluggable World Model
(`src/ejepa_wm`) before each ReAct step when `WM_STRATEGY` selects the `prompt_injection`
path. `WorldModel.advise` returns an imagined-lookahead block that is injected
**transiently** into that turn's prompt. The WM is built once per executor (a JEPA
checkpoint loads a single time, shared across tasks); any WM failure degrades to plain
ReAct. Only `prompt_injection` (incl. the `imagined` alias) is supported here —
`selection` needs per-step candidate sampling the CRM loop doesn't do.

```bash
# text-LLM EWM world model (served over vLLM/MCP)
ejepa bench run crmarenapro --executor mcp_react --wm-strategy imagined

# JEPA world model instead of a text LLM (needs the `jepa` purple extra: torch/transformers)
ejepa bench run crmarenapro --executor mcp_react \
  --wm-strategy imagined --wm-ewm-jepa-checkpoint /models/jepa_ckpt
```

## Trajectory Capture

Set `capture_trajectory=true` to export one JSONL trajectory per evaluated task.
The Green agent records A2A events, and the `baseline_crm_agent` executor adds
Purple-internal records for LLM calls and CRM tools. The internal records include
`system`, `user`, `assistant`, and `tool` messages, including SQL `<execute>` /
`<describe>` actions and their database results.

```bash
ejepa bench run crmarenapro \
  --executor baseline_crm_agent \
  --config task_limit=20 \
  --config capture_trajectory=true
```

You can also enable it with `BENCHMARK_CAPTURE_TRAJECTORY=1`. The generated
`detail.json` includes `trajectory_capture` plus per-task `trajectory_file_path`
and `trajectory_event_count`; files are written under
`<result_dir>/trajectories/*.jsonl`.
