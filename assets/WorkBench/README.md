# WorkBench

[WorkBench](https://github.com/olly-styles/WorkBench) is a benchmark for evaluating agent
performance on realistic workplace tasks. A single agent is given natural-language tasks
(e.g. "Delete my last email from nadia") and 26 read/write tools across five sandbox
domains — Calendar, Email, Analytics, Project Management, and CRM (plus a Company
Directory lookup tool) — and is scored by comparing the sandbox's final state against a
ground-truth outcome after executing the same actions from a fresh copy of the sandbox.

- Paper: [WorkBench: a Benchmark Dataset for Agents in a Realistic Workplace Setting](https://arxiv.org/abs/2405.00823)
- Upstream repository: <https://github.com/olly-styles/WorkBench>

Unlike most other wrapped benchmarks in this repo, **WorkBench needs no Docker, no MCP
server, and no external service of any kind** beyond the LLM API itself: tools are plain
Python functions operating on in-memory pandas DataFrames seeded from CSVs already
committed in the upstream repository.

This benchmark wraps the upstream framework so the Green/Purple agent harness can drive it
end-to-end:

- **Green Agent** loads a task from a bundled sample or one of WorkBench's own
  `tasks_and_outcomes` CSVs, forwards the task text (and declared domains) to the Purple
  Agent, and scores the returned predicted actions itself using WorkBench's own
  `is_correct` / `has_side_effects` functions (imported in-process — no LLM judge needed).
  Ground truth is never sent to Purple.
- **Purple Agent** routes each request to the `mcp_react` executor, which imports
  WorkBench's own `src.evals.agent` / `src.evals.inference` modules **in-process** and runs
  the real ReAct (or native tool-calling) agent loop against the real tools. There is no
  MCP protocol involved — WorkBench has no MCP layer to begin with, and building one purely
  for the name would add a network boundary and new state-isolation logic for no benefit
  over WorkBench's own already-correct in-process, per-thread sandbox design.
  When `WM_STRATEGY` is set, the executor additionally consults the shared, pluggable
  `ejepa_wm` World Model (including the JEPA-based Enterprise World Model) via the sibling
  `wm_react.py` — see [World Model](#world-model-optional-incl-jepa) below.

---

## Layout

```
WorkBench/
├── benchmark.toml
├── README.md
├── green/
│   ├── pyproject.toml
│   ├── workbench_green_agent.py
│   ├── task_loader.py
│   └── tasks/
│       ├── task_ids.toml
│       └── Tasks_SAMPLE.json
├── purple/
│   ├── pyproject.toml
│   └── workbench_purple_agent.py
└── purple-executors/
    └── mcp_react/
        ├── executor.py
        ├── wm_react.py       # optional ejepa_wm / JEPA world-model wiring (see below)
        ├── runtime.py
        └── version.toml
```

---

## Prerequisites

- Linux host with **Python 3.11+** and **`uv`** — no Docker, no database, no MCP server
- An LLM endpoint. Hosted provider keys are supported through OpenAI, Anthropic, Google, or
  OpenRouter; a local vLLM OpenAI-compatible chat-completions endpoint is also supported.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Setup

### 1. Install the green and purple environments

From the repository root (the directory that contains `scripts/install.sh` and `assets/`):

```bash
scripts/install.sh workbench
```

This creates `assets/WorkBench/green/.venv` and `assets/WorkBench/purple/.venv`, plus the
root `.venv/` if it does not already exist.

The `mcp_react` executor imports the upstream repo's own agent/inference modules
in-process, so also install its runtime dependencies into the purple venv:

```bash
uv sync --project assets/WorkBench/purple --extra mcp_react
```

Only needed if you plan to use a **JEPA** world model (see
[World Model](#world-model-optional-incl-jepa) below) — this pulls in `torch`/`transformers`:

```bash
uv sync --project assets/WorkBench/purple --extra mcp_react --extra jepa
```

### 2. Clone the upstream repository

Unlike some other wrapped benchmarks, **no separate dataset download is needed** — the
task CSVs are committed inside the WorkBench repository itself.

```bash
git clone https://github.com/olly-styles/WorkBench.git ~/WorkBench
```

Tell our executor/task-loader where it lives (consider adding this to `~/.bashrc`), or
place a symlink at `assets/WorkBench/WorkBench` pointing at the clone:

```bash
export WORKBENCH_REPO_PATH="$HOME/WorkBench"
```

### 3. Required environment variables

Set at least one of these, matching whichever `model_name` you plan to use. A model's
native-provider key (if present) is used directly, billing that vendor; `OPENROUTER_API_KEY`
is only consulted as the fallback for models without a native key set (and is required for
models whose `MODEL_REGISTRY` entry routes through OpenRouter itself, e.g. `qwen-3.5-flash`,
`deepseek-v4-pro`):

```bash
export OPENAI_API_KEY=your-openai-key       # covers gpt-* model_names
export ANTHROPIC_API_KEY=your-anthropic-key # covers claude-* model_names
export GEMINI_API_KEY=your-google-key       # covers gemini-* model_names
export OPENROUTER_API_KEY=your-openrouter-key
```

For a local vLLM-served model, set the WorkBench-specific route instead:

```bash
export WORKBENCH_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export WORKBENCH_VLLM_MODEL=local-vllm
export WORKBENCH_VLLM_API_KEY=EMPTY
```

`WORKBENCH_VLLM_MODEL` must match the model name accepted by the vLLM server, usually the
value passed to `--served-model-name`. The WorkBench `model_name` config key defaults to
`local-vllm` for this route, and the wrapper also accepts the served model name itself as an
alias. Override the friendly registry alias with `WORKBENCH_VLLM_MODEL_NAME` if needed.

`mcp_react`'s `runtime.py` fails fast with a clear error if none of the above are set
(mirroring WorkBench's own `resolve_route()`/`_require_env`). It can only check that *some*
credential is present, not that it's the right one for your chosen `model_name` — an
OpenAI-only key with a `claude-*` model_name will still fail later, inside
`resolve_route()`. For the vLLM path, setting any `WORKBENCH_VLLM_*` variable opts into
the local route and no hosted-provider key is required. There is no other readiness check —
WorkBench has no service to probe before the LLM call itself.

---

## Running the benchmark

### Quick smoke command

The bundled `sample` target (2 tasks, hand-copied from the real `email` dataset) exercises
task loading, request validation, Purple routing, and result-artifact writing —
`tests/test_workbench_dry_run.py` covers this fully offline with no LLM/repo dependency. To
actually run it through Purple you still need `WORKBENCH_REPO_PATH` and a matching
credential (e.g. `ANTHROPIC_API_KEY` for the `claude-sonnet-4.6` example below, or
`OPENROUTER_API_KEY`) from steps 2–3:

```bash
ejepa bench run WorkBench --executor mcp_react \
  --config target=sample --config model_name=claude-sonnet-4.6
```

`model_name` is **required** and must be one of WorkBench's own `MODEL_REGISTRY` keys —
see `src/evals/agent.py` in the upstream repo for the current list (e.g.
`claude-fable-5`, `gpt-5.4`, `claude-sonnet-4.6`, `gemini-3-flash`, `qwen-3.5-flash`,
`deepseek-v4-pro`, ...). When `WORKBENCH_VLLM_*` is set, the wrapper also registers a
local vLLM alias, `local-vllm` by default, plus the served model name from
`WORKBENCH_VLLM_MODEL`:

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model /path/to/model \
  --served-model-name local-vllm

export WORKBENCH_VLLM_BASE_URL=http://127.0.0.1:8000/v1
export WORKBENCH_VLLM_MODEL=local-vllm
export WORKBENCH_VLLM_API_KEY=EMPTY

ejepa bench run WorkBench --executor mcp_react \
  --config target=sample --config model_name=local-vllm
```

This also works when the `ejepa` model name is the vLLM served model name:

```bash
export WORKBENCH_VLLM_BASE_URL=http://127.0.0.1:9012/v1
export WORKBENCH_VLLM_API_KEY=EMPTY
export WORKBENCH_VLLM_MODEL=wm_agent3

ejepa bench run WorkBench --executor mcp_react \
  --config target=sample --config model_name=wm_agent3
```

Start with `structured_outputs=false` (the default). Native tool-calling requires your
vLLM server and model launch flags to support OpenAI-style tool calls.

### Real evaluation

Target names other than `sample` map directly to WorkBench's own domain CSVs:

```bash
ejepa bench run WorkBench --executor mcp_react \
  --config target=email --config model_name=claude-sonnet-4.6
ejepa bench run WorkBench --executor mcp_react \
  --config target=multi_domain --config model_name=gpt-5.4 --config tool_selection=domains
```

Supported `target` values: `email`, `calendar`, `analytics`, `project_management`,
`customer_relationship_manager`, `multi_domain`.

Use `target=all` to run every one of those 6 domains' tasks (690 tasks total) in a single
`ejepa bench run` invocation instead of one command per domain. Task IDs stay domain-prefixed
(`email_0000`, `calendar_0000`, ...) so results from all domains sit in one `detail.json`
without ID collisions:

```bash
ejepa bench run WorkBench --executor mcp_react \
  --config target=all --config model_name=claude-sonnet-4.6 --config max_parallel=8
```

`target=all` is a convenience aggregate implemented in our own `task_loader.py` — WorkBench
itself has no such target; it's not one of the CSVs under `data/processed/tasks_and_outcomes/`.
Since a full run is 690 tasks, raise `max_parallel` (see Config keys below) or expect it to
take a while sequentially.

For a quicker multi-step evaluation, `target=longest_100_balanced` runs 100
tasks selected longest-first by gold outcome length while balancing the six
domain targets (15 email tasks and 17 from every other target). Single-step
tasks are excluded:

```bash
ejepa bench run WorkBench --executor mcp_react \
  --config target=longest_100_balanced \
  --config model_name=claude-sonnet-4.6
```

The committed IDs preserve source CSV row indexes and are interleaved by domain.


### Trajectory capture

```bash
ejepa bench run WorkBench --executor mcp_react \
  --config target=sample --config model_name=claude-sonnet-4.6 \
  --config capture_trajectory=true
```

Persists WorkBench's own per-step `TraceStep` records (LLM input/output, action taken,
observation) as the trajectory JSONL.

### Config keys

| Key | Default | Meaning |
| --- | --- | --- |
| `target` | `sample` | `sample`, one of the 6 domains, `all`, or `longest_100_balanced` |
| `model_name` | **required** | A key from WorkBench's own `MODEL_REGISTRY` (`src/evals/agent.py`) |
| `tool_selection` | `all` | `all` (every domain's tools every task) or `domains` (only the task's declared domains) |
| `structured_outputs` | `false` | Use native OpenAI-style tool-calling instead of ReAct text/JSON-blob parsing |
| `act_without_confirmation` | `false` | Append WorkBench's system-prompt suffix telling the model to act without asking for confirmation |
| `task_ids` | unset | Optional list of task IDs (`<target>_0000`, ...) to filter to |
| `max_parallel` | `1` | Concurrent tasks dispatched to Purple (each lands on its own thread, matching WorkBench's own `threading.local()` sandbox isolation). Safe with a world model: one world model is built per task, with weights shared process-wide via `WM_SHARE_MODEL_WEIGHTS` |
| `capture_trajectory` | `false` | Persist the full per-step LLM/tool trajectory JSONL |

---

## World Model (optional, incl. JEPA)

`mcp_react` can consult a pluggable World Model (`src/ejepa_wm`) before each ReAct step,
matching how `EnterpriseOps-Gym`'s and `crmarenapro`'s `mcp_react` executors wire it in.
WorkBench's baseline agent loop (`src.evals.agent.run_agent`) is a free function rather
than a class, so `wm_react.py` reimplements its exact ReAct control flow, substituting the
one per-turn `call_llm(...)` decision point with a WM-guided step; with `WM_STRATEGY`
unset (the default) it delegates straight to the unmodified upstream `_run_single_task`, so
the executor is always runnable without any WM configured. WM guidance only wraps the ReAct
text-parsing loop — a task requesting `structured_outputs=true` is never WM-guided.

Four strategies, selected by `WM_STRATEGY` (env var, or `ejepa bench run --wm-strategy ...`):

- **`selection`** — sample `WM_N` candidate actions from the policy LLM each turn, let the
  WM rank them (`WorldModel.select`), and commit the best one. Pairs well with
  `WM_BACKEND=ewm_predict`.
- **`prompt_injection`** — ask the WM to `advise()` and inject its guidance transiently
  before the next action. Pairs with `WM_BACKEND=ewm_imagined` or `llm`.
- **`beam_plan`** / **`hier_latent_cem`** — JEPA MPC lookahead (auto-selects
  `WM_BACKEND=ewm_imagined`; requires a JEPA canonical-event checkpoint,
  `WM_EWM_JEPA_CHECKPOINT`). Every few turns the WM scores imagined action outcomes with
  its canonical-event heads (LLM-free) and can override the agent's next action when its
  plan beats the baseline. With any non-JEPA backend both degrade to plain ReAct.

```bash
# text-LLM EWM world model, action selection (no local torch model needed)
ejepa bench run WorkBench --executor mcp_react --config target=sample \
  --config model_name=claude-sonnet-4.6 --wm-strategy selection --wm-backend ewm_predict --wm-n 4

# JEPA world model instead of a text LLM (needs the `jepa` purple extra: torch/transformers)
ejepa bench run WorkBench --executor mcp_react --config target=sample \
  --config model_name=claude-sonnet-4.6 \
  --wm-strategy beam_plan --wm-ewm-jepa-checkpoint /models/jepa_ckpt
```

The World Model is built once per executor process and reused across tasks (a JEPA
checkpoint loads into memory a single time); any WM build/step failure logs a warning and
falls back to plain ReAct rather than failing the task. Per-step WM telemetry
(`wm_steps`: strategy, backend, chosen index / override decisions) is attached to the
`internal_trajectory` artifact alongside the normal per-step trace, visible when
`capture_trajectory=true`.

**Scope note**: the bundled EWM world model is fine-tuned on `EnterpriseOps-Gym` tool
outcomes. Pointing it at WorkBench's calendar/email/analytics/CRM/project-management
actions is an intentional out-of-domain / over-fit probe — expect it to guide `gym` well
and WorkBench poorly, which is the point of wiring it here (see `src/ejepa_wm/README.md` for
the full `ejepa_wm` reference, including all backends and `--wm-*` flags).

---

## Scoring

Green calls WorkBench's own `is_correct(predicted_actions, ground_truth_actions, error)`
(re-executes both action lists against fresh sandbox copies and compares resulting state as
unordered row sets — not an exact action-string match) and
`has_side_effects(predicted_actions, correct)` (WorkBench's headline "did the agent do
something harmful" signal: state changed without being correct). `total_score` /
`score_rate` in `manifest.json` / `detail.json` are the sum/mean of `is_correct`; the
unwanted-side-effect rate is tracked separately as `avg_unwanted_side_effects_rate` in
`detail.json` rather than folded into the primary score, since capability and safety are
reported as two distinct numbers in the WorkBench paper.

---

## Troubleshooting

- **`WorkBench repository not found`**: set `WORKBENCH_REPO_PATH`, or symlink
  `assets/WorkBench/WorkBench` at the clone.
- **`WorkBench mcp_react executor has no usable LLM credential set`**: export the
  native-provider key for the `model_name` you're running (`OPENAI_API_KEY` for `gpt-*`,
  `ANTHROPIC_API_KEY` for `claude-*`, `GEMINI_API_KEY` for `gemini-*`), or `OPENROUTER_API_KEY`
  as a catch-all. This check only confirms *some* key is set — pairing the wrong provider's
  key with a `model_name` (e.g. only `OPENAI_API_KEY` with `model_name=claude-sonnet-4.6`)
  fails later with a `Missing required environment variable 'OPENROUTER_API_KEY'` error from
  WorkBench's own `resolve_route()`, since it falls back to OpenRouter when no matching direct
  key is found.
- **`FileNotFoundError: data/processed/calendar_events.csv` (or similar)**: WorkBench's own
  sandbox loader reads CSVs via paths relative to the process's current working directory —
  our executor/Green `chdir`s into the resolved `WORKBENCH_REPO_PATH` before the first
  scoring/agent call to match how WorkBench's own CLI is always invoked from its repo root.
  If you see this, the repo path likely doesn't actually contain
  `data/processed/*.csv` (re-check the clone).
- **`Invalid --model_name` / unknown model**: `model_name` must be a key in WorkBench's own
  `MODEL_REGISTRY` (`src/evals/agent.py` in the upstream repo), not a raw provider model
  slug.
- **`ModuleNotFoundError: openai` / `pandas` / `tenacity` when running the executor**: run
  `uv sync --project assets/WorkBench/purple --extra mcp_react`.
- **`beam_plan`/`hier_latent_cem` always logs `fallback: true` in `wm_steps`**: these two
  strategies require the `ewm_imagined` backend with a JEPA canonical-event checkpoint
  (`WM_EWM_JEPA_CHECKPOINT`); with any other backend they intentionally degrade to plain
  ReAct rather than erroring.
- **`ModuleNotFoundError: torch` / `transformers` when using a JEPA world model**: run
  `uv sync --project assets/WorkBench/purple --extra mcp_react --extra jepa`.
- **Every task fails with `executor_error: "Unknown datetime string format, unable to
  parse: None"` and `predicted_function_calls: []`**: this is an upstream WorkBench bug, not
  ours — `run_agent()`/`run_agent_structured()` in the upstream repo's `src/evals/agent.py`
  blindly stringify every tool action-input value with `str(v)` before calling the tool. If
  the model explicitly passes `null`/`None` for an optional argument (common for
  `email.search_emails`'s `date_min`/`date_max` on tasks like "delete my last email from
  X"), `str(None)` becomes the literal text `"None"`, which the tool's `if date_min:` truthy
  check treats as a real date and passes to `pd.Timestamp("None")`, which raises. That tool
  call isn't wrapped in a try/except upstream, so the exception kills the whole task instead
  of becoming a normal observation the model could recover from. This reproduces with the
  real (unmodified) upstream tool code, no LLM call needed:
  ```python
  action_input = {"query": "sofia", "date_min": None, "date_max": None}
  str_input = {k: str(v) for k, v in action_input.items()}  # {'date_min': 'None', ...}
  email.search_emails(**str_input)  # DateParseError: Unknown datetime string format...
  ```
  Any model that emits an explicit `null` for an optional date-typed tool argument will hit
  this, in both ReAct and `structured_outputs=true` mode (the same `str(v)` bug exists in
  both call sites), though native tool-calling models more often omit unset optional
  arguments entirely rather than send `null`, which avoids triggering it. We intentionally
  do not patch this in our executor — faithfully reproducing WorkBench's own (buggy)
  `run_agent`/`run_agent_structured` matters more than working around it, since the goal is
  scores comparable to the published/upstream benchmark.

---

## References

- Upstream repository: <https://github.com/olly-styles/WorkBench>
- Paper: <https://arxiv.org/abs/2405.00823>
