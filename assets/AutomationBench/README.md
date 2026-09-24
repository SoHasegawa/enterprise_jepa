# AutomationBench

[AutomationBench](https://github.com/zapier/AutomationBench) (Zapier) evaluates agents on
**cross-application business workflow orchestration over REST/Zapier-style tools**: 600 public
tasks across Sales, Marketing, Operations, Support, Finance and HR, plus 200 `simple`
foundational tasks, run against a simulated SaaS world of 47 tools. Every task ships a trigger
message, an initial world state, a per-task tool allow-list and a list of assertions over the
**final** state — so scoring is state-based, deterministic and needs no LLM judge.

- Paper: [AutomationBench](https://arxiv.org/abs/2604.18934)
- Upstream repository: <https://github.com/zapier/AutomationBench>

Text/API only: no GUI, no browser, no vision. Tasks are genuinely multi-step — upstream's own
runs use `max_steps=50`, and a published `simple`-domain run averaged 18 model steps and 28 tool
calls per task, with the six business domains longer still.

## Division of labour

| Component | Responsibility |
|---|---|
| **Green** (`green/automationbench_green_agent.py`) | Loads tasks (bundled sample or upstream's own `get_combined_dataset`), forwards trigger text + initial state + tool allow-list, **keeps the assertions**, scores the returned final state, writes `manifest.json` / `detail.json` / trajectories. |
| **Purple** (`purple/automationbench_purple_agent.py`) | Thin A2A router to one executor under `purple-executors/`. |
| **Executor** (`purple-executors/mcp_react/`) | Binds upstream tools to a fresh `WorldState`, runs the ReAct loop with the shared `ejepa_wm` world model, returns the final state + trajectory. |

Purple never receives ground truth: the payload carries `prompt`, `initial_state`,
`zapier_tools`, `toolset` and `max_turns` only.

## Scoring

Green scores with upstream's own rubric functions — `automationbench.rubric.partial_credit` and
`task_completed_correctly` — so the numbers match upstream's definitions, including its
"free assertion" rule (an assertion already satisfied by the initial state earns nothing, but
breaking it still counts as a failure).

- `score` / `score_rate` = **partial credit** (0..1), upstream's primary reward.
- `pass_rate` = **`task_completed_correctly`**, the strict all-assertions-pass metric.

For the bundled `sample` target (and any environment without the upstream package) a bundled
evaluator in `green/scoring.py` handles the generic `field_equals` / `collection_record_exists`
assertion forms with the same free-assertion rule. Which scorer ran is recorded per task as
`detail.details[].scorer`.

## Targets

| target | tasks | source |
|---|---|---|
| `sample` | 2 | bundled `green/tasks/Tasks_SAMPLE.json`, fully offline |
| `sales`, `marketing`, `operations`, `support`, `finance`, `hr` | 100 each | upstream |
| **`all_domains`** | **600** | upstream — **the full public benchmark**; matches upstream's `PUBLIC_DOMAINS` / `DEFAULT_DOMAINS`, i.e. what its own "all domains" runs report |
| `simple` | 200 | upstream — separate foundational/diagnostic domain, deliberately *not* part of `all_domains` |
| `all_with_simple` | 800 | upstream — the six public domains plus `simple` |

Toolsets (`--config toolset=`): `api` (generic REST interface), `zapier` (all tools),
`limited_zapier` (default — the per-task allow-list, upstream's headline setting).

## Setup

```bash
# Offline smoke test — no upstream checkout, no model
uv run pytest tests/test_automationbench_dry_run.py -v

# Upstream tasks
git clone https://github.com/zapier/AutomationBench ~/programs/tools/AutomationBench
export AUTOMATIONBENCH_REPO_PATH=~/programs/tools/AutomationBench
uv sync --project assets/AutomationBench/green  --extra upstream
uv sync --project assets/AutomationBench/purple --extra mcp_react
```

The repo path is resolved from `AUTOMATIONBENCH_REPO_PATH`, then an
`assets/AutomationBench/AutomationBench` symlink, then `${BENCHMARK_HOME}/repos/AutomationBench`,
then `../tools/AutomationBench`.

### Policy LLM

The acting agent talks to an OpenAI-compatible endpoint:

```bash
export AUTOMATIONBENCH_LLM_MODEL=wm_agent1
export AUTOMATIONBENCH_LLM_API_ENDPOINT=http://127.0.0.1:9011/v1
export AUTOMATIONBENCH_LLM_API_KEY=EMPTY          # vLLM ignores the value
export AUTOMATIONBENCH_LLM_TEMPERATURE=0          # >0 required for WM_STRATEGY=selection
export AUTOMATIONBENCH_LLM_MAX_TOKENS=4096
```

Fallbacks: `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY`, then `OPENAI_MODEL_NAME` /
`OPENAI_BASE_URL` / `OPENAI_API_KEY`.

## Running

```bash
# Baseline (no world model)
ejepa bench run AutomationBench --executor mcp_react \
  --config target=simple --config toolset=limited_zapier \
  --config max_turns=50 --config capture_trajectory=true
```

## World model (optional)

`purple-executors/mcp_react/wm_react.py` is structured after EnterpriseOps-Gym's `mcp_react`:
the policy LLM is handed to `ejepa_wm` as `chat_fn`, so every world model and harness in this repo
works here unchanged and the executor contains no per-backend code. Harnesses come from
`--wm-strategy`: `selection`, `prompt_injection`, `itp_i`, `revision`, `reference`, `beam_plan`
(`--wm-beam-plan-trigger interval|critic`), `hier_latent_cem`; unset / `none` is the baseline arm.

### Enterprise-JEPA (latent transition model, decode-free scoring)

```bash
ejepa bench run AutomationBench --executor mcp_react \
  --config target=simple --config capture_trajectory=true \
  --wm-strategy beam_plan --wm-beam-plan-trigger critic \
  --wm-ewm-jepa-checkpoint ../ewm/sessions/data_jepa_heads_partial_imb_terminal_3 \
  --wm-jepa-observation-backend canonical_event \
  --wm-beam-plan-samples 8 --wm-beam-plan-horizon 4 --wm-beam-mpc-execute-steps 4 \
  --wm-imagined-rollout-mode open_loop --wm-beam-plan-ssot-diversity
```

Needs `uv sync --project assets/AutomationBench/purple --extra mcp_react --extra jepa`
(torch/transformers). Pin the device with `CUDA_VISIBLE_DEVICES` — the JEPA generator takes the
first visible CUDA device.

### LLM-WM, state output (fine-tuned canonical-event model)

```bash
ejepa bench run AutomationBench --executor mcp_react --config target=simple \
  --wm-strategy beam_plan --wm-llm-ewm-mode llm_canonical_trained \
  --wm-ewm-llm-canonical-event-checkpoint ../ewm/sessions/llm_wm_beam_action_terminal_crmarenapro
```

Point it at a served model instead with `--wm-ewm-model <name>` plus `WM_VLLM_BASE_URL`.

### LLM-WM, tool output + judge / agent world model

```bash
# tool-output world model + judge
ejepa bench run AutomationBench --executor mcp_react --config target=simple \
  --wm-strategy beam_plan --wm-llm-ewm-mode llm_tool_output_judge \
  --wm-ewm-model world_model

# agent world model (Qwen-AgentWorld family)
WM_QWEN_AGENTWORLD=1 ejepa bench run AutomationBench --executor mcp_react --config target=simple \
  --wm-strategy beam_plan --wm-llm-ewm-mode llm_tool_output_judge \
  --wm-ewm-model Qwen/Qwen-AgentWorld-35B-A3B
```

The tool-output world model predicts each imagined tool result as text and a judge scores the
imagined trajectory, so it is the most expensive of the three per simulated transition.

## Harness sweeps

The shared runners work unchanged:

```bash
python scripts/run_wm_harnesses.py --label automationbench-jepa \
  --harnesses beam_critic revision --beam-samples 8 --beam-horizon 4 \
  -- ejepa bench run AutomationBench --executor mcp_react --config target=simple \
     --wm-ewm-jepa-checkpoint ../ewm/sessions/data_jepa_heads_partial_imb_terminal_3 \
     --wm-jepa-observation-backend canonical_event --config capture_trajectory=true
```

## Result artifacts

`detail.json` per-task records carry `partial_credit`, `task_completed_correctly`,
`assertions_passed` / `assertions_scored` / `assertions_total`, `assertion_results`, `scorer`,
`tool_calls`, `model_calls`, `steps`, `failed_tool_calls` and `purple_tools_used`. With
`--config capture_trajectory=true` each task also gets a JSONL trajectory that includes the
executor's `internal_trajectory` artifact (per-step tool calls and `wm_steps`), which is what
`scripts/summarize_wm_behavior_metrics.py` and `scripts/run_bench_repeated.py` read.

## Known limitations

- The `zapier` toolset registers upstream's full tool list without upstream's meta-tool
  discovery layer (`search_tools` / `execute_tool`); `limited_zapier` and `api` are the
  faithful paths. Prefer `limited_zapier`.
- Upstream's own `AutomationBenchEnv` (a `verifiers` `StatefulToolEnv`) is intentionally not
  used: this executor owns its loop so the world-model hook can sit at the decision point. Only
  upstream's tools, `WorldState` and rubric functions are imported, none of which need
  `verifiers`.
- Assertion types outside the bundled evaluator's two generic forms require the upstream
  package; without it they are reported as `excluded` with a reason rather than silently passing.
