# Reproducing the paper

Every number in the paper comes from one of the experiments below. Each has a *run* step
that writes JSON summaries under `results/`, and a *summarize* step that turns those into
the reported value. The summarizers are the definition of each metric — read them before
disputing a number.

| Paper | What it is | Run | Summarize |
|---|---|---|---|
| Table 2, Figure 3 | Next-state prediction (macro-F1 / macro-recall per field) | Stage-2 training eval — `docs/training.md` | the trainer's `run_summary.json`; `training/src/analysis/calculate_canonical_field_accuracy.py` for the LLM-WM row |
| Table 3 | Task success rates, 5 benchmarks × 3 harnesses × 2 world models + baseline | `scripts/run_main_table_repeats.sh` | `scripts/summarize_main_table.py`, `scripts/summarize_row_averages.py` (the *Ave.* column) |
| Table 4 | Prediction controls on EnterpriseOps-Gym | `scripts/run_prediction_controls.sh` | `scripts/summarize_wm_harness_summaries.py` |
| Table 5 | Enterprise-JEPA vs the tool-output LLM-WM (Qwen-AgentWorld) | `WORLD_MODELS=agentworld scripts/run_main_table_repeats.sh` | `scripts/summarize_main_table.py` |
| Table 6 | Stage-1 trajectory sources (expanded corpus) | `training/src/generation/generate_adp_world_model_trajectories.py` | counts printed by the generator |
| Table 9 | How the world model improves outcomes | (reuses the Table 3 runs) | `docs/qualitative_analysis.md` |
| Table 10 | Planning-budget ablation | `scripts/run_beam_ablation.sh` | `scripts/summarize_beam_ablation.py` |
| Figures 2, 4 | Beam-search latency vs rollout horizon, production and HF backends | `scripts/measure_wm_latency_vs_horizon.py` | `scripts/plot_wm_latency_hf_vs_production.py` |
| §5.5 per-step latency | 4.5 s (JEPA) / 9.3 s (LLM-WM) / 2.6 s (no WM) per policy step | (reuses the Table 3 runs) | `scripts/summarize_per_step_latency.py` |

`docs/main_table_protocol.md` fixes the configuration all of these share, and says why each
value was fixed.

## 0. The harnesses and world models

| Harness | What the world model does |
|---|---|
| Baseline | nothing — the no-WM arm (`--wm-strategy none`) |
| Revision | scores the action the policy just chose; the policy may revise it |
| ITP-I | imagines `k` steps ahead, then the policy reflects before acting |
| Beam search | samples 8 candidate plans, rolls each out to horizon 3 in latent space, executes the arg-max plan for 2 steps, re-plans |

| World model | Flags |
|---|---|
| No WM | `--wm-strategy none` |
| State-output LLM-WM | `--wm-llm-ewm-mode llm_canonical_trained --wm-ewm-model world_model` |
| Tool-output LLM-WM (Qwen-AgentWorld) | `--wm-llm-ewm-mode llm_tool_output_judge --wm-ewm-model world_model` |
| Enterprise-JEPA | `--wm-ewm-jepa-checkpoint checkpoints/jepa --wm-jepa-observation-backend canonical_event` |

`scripts/run_wm_harnesses.py` runs the harnesses for one (benchmark, world model) pair and
writes one summary covering all of them; the experiment scripts below wrap it.

### Single runs

One harness, one benchmark, straight from the CLI — useful for a smoke test before
committing to a full cell:

```bash
# no-WM baseline
ejepa bench run <BENCH> --executor mcp_react --config target=<target>

# Revision, with Enterprise-JEPA
ejepa bench run <BENCH> --executor mcp_react --config target=<target> \
  --wm-strategy revision \
  --wm-ewm-jepa-checkpoint checkpoints/jepa \
  --wm-jepa-observation-backend canonical_event

# ITP-I            : --wm-strategy itp_i --wm-itp-fixed-k 4
# Beam search      : --wm-strategy beam_plan --wm-beam-plan-samples 8 \
#                    --wm-beam-plan-horizon 3 --wm-beam-mpc-execute-steps 2

# state-output LLM-WM instead of JEPA: replace the two --wm-ewm-jepa-* flags with
#   --wm-llm-ewm-mode llm_canonical_trained \
#   --wm-ewm-llm-canonical-event-checkpoint <llm-wm checkpoint>

# every harness in one sweep, one summary
uv run python scripts/run_wm_harnesses.py --result-root "$BENCHMARK_HOME/experiments" \
  --label <run-label> -- ejepa bench run <BENCH> --executor mcp_react --config target=<target> ...
```

The ITP-I depth, beam horizon and re-plan interval above are the paper's
(k = 4, horizon 3, execute 2 — §5.4). `src/ejepa_wm/README.md` illustrates the same flags
with different values (`--wm-itp-max-k 5`, horizon 2, and the `WM_BEAM_MPC_EXECUTE_STEPS`
default of 3); those are documentation examples, not the reported configuration.
`scripts/run_main_table_repeats.sh` is the authority — it passes the paper's values
explicitly.

## 1. Table 3 — task success rates

One invocation per benchmark, so several can run concurrently against different agent
endpoints and GPUs:

```bash
BENCH=EnterpriseOps-Gym AGENT_URL=http://127.0.0.1:18045/v1 AGENT_NAME=Qwen3.6-27B JEPA_GPU=3 \
  REPEATS=3 scripts/run_main_table_repeats.sh
BENCH=WorkBench         AGENT_URL=http://127.0.0.1:9010/v1  AGENT_NAME=wm_agent    JEPA_GPU=5 \
  REPEATS=3 scripts/run_main_table_repeats.sh
# ...and for crmarenapro, AutomationBench, Terminal-Bench-2.0
```

Runs are labelled `tab-<slug>-<jepa|llmwm|agentworld>-r<N>` and are resumable: a label whose
summary already exists is skipped. A cell whose tasks mostly error is quarantined into
`results/wm_harness_summaries/failed/` rather than silently averaged.

```bash
uv run python scripts/summarize_main_table.py --verbose --csv results/analysis/main_table.csv
uv run python scripts/summarize_row_averages.py
```

Each cell reports `mean ± sd` over three runs. Only full-size targets count (EOPS 80,
CRM 428, WB 690, AB 600 scored on the four reported domains, TB 89). AutomationBench is run
over all six domains and scored excluding marketing and finance;
`results/analysis/main_table.csv` is the paper's own output of that command.

## 2. Table 4 — prediction controls

The identical beam-search planner with the JEPA per-step predictions replaced:

```bash
DRY_RUN=1 scripts/run_prediction_controls.sh
MODES="no_state shuffled" BENCHES=EnterpriseOps-Gym scripts/run_prediction_controls.sh
uv run python scripts/summarize_wm_harness_summaries.py 'results/wm_harness_summaries/*control-*.json'
```

`no_state` withholds every predicted state (the paper's *No world model feedback*) and
`shuffled` permutes the model's own rows across (plan, step) within a call (*Shuffled
predictions*). Both are implemented by `WM_JEPA_PREDICTION_CONTROL` in
`src/ejepa_wm/backends/_ewm_jepa.py`, which also supports `uniform` and `prior` — two
further controls that the paper does not report.

## 3. Table 10 — planning-budget ablation

One factor at a time around the centre point (8 candidates, horizon 3, open loop):

```bash
DRY_RUN=1 scripts/run_beam_ablation.sh      # print the plan
scripts/run_beam_ablation.sh                # 6 configs × 2 benchmarks, 3 repeats
uv run python scripts/summarize_beam_ablation.py
```

## 4. Figures 2 and 4, and the per-step latency numbers

```bash
# per-call latency vs rollout horizon, both world models
uv run --extra jepa python scripts/measure_wm_latency_vs_horizon.py \
    --jepa-checkpoint checkpoints/jepa \
    --llm-base-url http://127.0.0.1:9015/v1 --llm-tokenizer checkpoints/jepa/tokenizer.json
# the figure: left panel HF Transformers (Figure 4), right panel production serving (Figure 2)
uv run --extra figures python scripts/plot_wm_latency_hf_vs_production.py \
    --hf results/wm_latency/wm_latency_jepa.json results/wm_latency/wm_latency_llm_state.json \
    --production results/wm_latency/wm_latency_jepa_compiled_h1234.json \
                 results/wm_latency/wm_latency_jepa_compiled_h8.json \
                 results/wm_latency/wm_latency_llm_state_vllm.json \
                 results/wm_latency/wm_latency_llm_state_vllm_h8.json
# end-to-end latency per policy step, from the Table 3 runs
uv run python scripts/summarize_per_step_latency.py 'results/wm_harness_summaries/*.json'
```

The production panel runs Enterprise-JEPA with `WM_JEPA_COMPILE=1` (torch.compile + CUDA
graphs, bucketed shapes) and the LLM world model under vLLM; the HF panel runs both
in-process under Hugging Face Transformers on the same hardware. The figure and its CSV in
`results/figures/` are the paper's.

## 5. Table 9 — how the world model helps

`docs/qualitative_analysis.md` is the attribution behind Table 9 and the worked examples in
Appendix F.3: which task classes the world model helps, which it hurts, and one worked
example per help class, each traced to a task id in the Table 3 runs.

## Cost

From `docs/main_table_protocol.md`, measured: a full block (baseline + 3 harnesses × 2 world
models on 4 benchmarks, 3 repeats) is 84 runs, roughly 170 hours of wall-clock on three
concurrent agent endpoints. Single cells range from 2.6 h (AutomationBench baseline at
`max_parallel=5`) to 12.5 h (Terminal-Bench ITP-I at `max_parallel=1`).
