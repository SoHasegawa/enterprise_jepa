# Reproducing the paper

Every number in the paper comes from one of five experiments. Each is a *run* step that
produces JSON summaries under `results/`, and a *summarize* step that turns those into the
reported value. The summarizers are the definition of the metric — read them before
disputing a number.

Read `docs/main_table_protocol.md` first: it fixes the configuration (and the reason each
value was fixed) that all of the below share.

## 0. The four harnesses

The world model is consulted through one of four harnesses, all driven by the same
`ejepa bench run` flags:

| Harness | What the world model does |
|---|---|
| `baseline` | nothing — the no-WM arm (`--wm-strategy none`) |
| `revision` | scores the action the policy just chose; the policy may revise it |
| `itp_i` | imagines `k` steps ahead, then the policy reflects before acting |
| `beam_interval` | plans: samples 8 candidate plans, rolls each out to horizon 3 in latent space, executes the arg-max plan for 2 steps, re-plans |

and one of four world models:

| World model | Flags |
|---|---|
| No WM | `--wm-strategy none` |
| State-output LLM-WM | `--wm-llm-ewm-mode llm_canonical_trained --wm-ewm-model world_model` |
| Tool-output LLM-WM | `--wm-llm-ewm-mode llm_tool_output_judge --wm-ewm-model world_model` |
| Enterprise-JEPA | `--wm-ewm-jepa-checkpoint checkpoints/jepa --wm-jepa-observation-backend canonical_event` |

`scripts/run_wm_harnesses.py` runs the harnesses for one (benchmark, world model) pair and
writes one summary covering all of them; the experiment scripts below wrap it.

## 1. Main success-rate table

One invocation per benchmark, so several can run concurrently against different agent
endpoints and GPUs:

```bash
BENCH=EnterpriseOps-Gym AGENT_URL=http://127.0.0.1:18045/v1 AGENT_NAME=Qwen3.6-27B JEPA_GPU=3 \
  REPEATS=3 scripts/run_main_table_repeats.sh
BENCH=WorkBench         AGENT_URL=http://127.0.0.1:9010/v1  AGENT_NAME=wm_agent    JEPA_GPU=5 \
  REPEATS=3 scripts/run_main_table_repeats.sh
# ...and for crmarenapro, AutomationBench, Terminal-Bench-2.0
```

Runs are labelled `tab-<slug>-<jepa|llmwm>-r<N>` and are resumable: a label whose summary
already exists is skipped. A cell whose tasks nearly all error is quarantined into
`results/wm_harness_summaries/failed/` rather than silently averaged.

```bash
uv run python scripts/summarize_main_table.py --verbose --csv results/analysis/main_table.csv
```

Each cell reports `mean ± sd (n)` over repeats. Only full-size targets count (EOPS 80,
CRM 428, WB 690, AB 600 scored on the four reported domains = 400, TB 89); ablation,
control and cost-matched labels are excluded by name. `results/analysis/main_table.csv` in
this repository is the paper's own output of that command.

## 2. Planning-budget ablation

One factor at a time around the centre point (8 candidates, horizon 3, open loop):

```bash
DRY_RUN=1 scripts/run_beam_ablation.sh      # print the plan
scripts/run_beam_ablation.sh                # 6 configs x 2 benchmarks, 3 repeats
uv run python scripts/summarize_beam_ablation.py
```

The write-up, including the caveat that the centre cell is quoted from the main table's
run set rather than the ablation's own, is `docs/ablation_analysis.md`.

## 3. Prediction controls ("is the planner doing the work?")

The identical planner with the JEPA per-step predictions replaced by shuffled rows, a
uniform distribution, or the training-set class priors
(`results/analysis/canonical_event_class_priors.json`):

```bash
DRY_RUN=1 scripts/run_prediction_controls.sh
scripts/run_prediction_controls.sh
uv run python scripts/summarize_wm_harness_summaries.py 'results/wm_harness_summaries/*control-*.json'
```

The control is implemented by `WM_JEPA_PREDICTION_CONTROL` in
`src/ejepa_wm/backends/_ewm_jepa.py`; the analysis is §10 of `docs/qualitative_analysis.md`.

## 4. Efficiency and cost-matching

```bash
# per-call latency vs rollout horizon, both world models
uv run --extra jepa python scripts/measure_wm_latency_vs_horizon.py \
    --jepa-checkpoint checkpoints/jepa \
    --llm-base-url http://127.0.0.1:9015/v1 --llm-tokenizer checkpoints/jepa/tokenizer.json
# end-to-end arms at equal wall-clock per re-plan
MATCHED_S=3 MATCHED_H=1 scripts/run_eops_cost_matched.sh
# per-policy-step latency (task length removed)
uv run python scripts/summarize_per_step_latency.py 'results/wm_harness_summaries/*.json'
# JEPA per-stage profile
uv run --extra jepa python scripts/profile_jepa_scoring.py --checkpoint checkpoints/jepa \
    --horizon 3 --candidates 8
```

Figures:

```bash
uv run --extra figures python scripts/plot_wm_latency_vs_horizon.py
uv run --extra figures python scripts/plot_wm_latency_hf_vs_production.py
uv run --extra figures python scripts/plot_wm_latency_scaling.py
uv run --extra figures python scripts/plot_success_latency_pareto.py
uv run --extra figures python scripts/plot_task_time_decomposition.py
```

Each writes a `.pdf` and the `.csv` it was drawn from into `results/figures/`; the versions
in this repository are the paper's. `docs/efficiency_analysis.md` is the full write-up,
including why the LLM world model is measured under vLLM rather than HF Transformers.

## 5. Behaviour and mechanism

```bash
uv run python scripts/summarize_wm_behavior_metrics.py   # -> results/analysis/wm_behavior_metrics.csv
```

Tool calls, world-model calls, advice injections, re-plans, critic fires and action
overrides per configuration. The qualitative attribution built on top of it — which task
classes the world model helps, which it hurts, and the worked examples — is
`docs/qualitative_analysis.md`.

## Cost

From `docs/main_table_protocol.md`, measured: a full block (baseline + 3 harnesses x 2 world
models on 4 benchmarks, 3 repeats) is 84 runs, roughly 170 hours of wall-clock on three
concurrent agent endpoints. Single cells range from 2.6 h (AutomationBench baseline at
`max_parallel=5`) to 12.5 h (Terminal-Bench ITP-I at `max_parallel=1`).
