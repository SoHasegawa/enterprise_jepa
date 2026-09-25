# Enterprise-JEPA

Code and protocols for **Enterprise-JEPA**, a latent-dynamics (JEPA-style) world model for
enterprise tool-use agents, together with the planning harnesses and benchmark adapters used
to produce the paper's results.

The agent is always a text LLM. The world model never acts: it predicts the outcome of a
*planned* action and the harness turns those predictions into advice, a re-selection, or a
plan. Swapping Enterprise-JEPA for an LLM world model is a generator swap — the policy,
prompts and environment are unchanged — which is what makes the comparison in the paper a
comparison of world models rather than of agents.

```
policy LLM  ──candidate actions──▶  harness  ──imagined rollout──▶  world model
     ▲                                 │                          (Enterprise-JEPA
     └──────── advice / choice ────────┘                           or an LLM WM)
```

## What is here

| Path | Contents |
|---|---|
| `src/ejepa_wm/` | The world model: JEPA inference, the LLM world models, and the four harnesses (`baseline`, `revision`, `itp_i`, `beam_interval`) |
| `src/ejepa_cli/`, `src/common/` | The `ejepa` benchmark runner (agent/benchmark process orchestration, result store) |
| `assets/` | Adapters for the five benchmarks: EnterpriseOps-Gym, CRMArena-Pro, WorkBench, AutomationBench, Terminal-Bench 2.0 |
| `training/` | The JEPA training and data-preparation pipeline (canonical-event labelling, trajectory generation, fine-tuning) |
| `scripts/` | Run scripts for every experiment in the paper, plus the summarizers and figure scripts |
| `results/` | The paper's derived numbers: per-run CSVs, figure data, and the latency measurements |
| `docs/` | Setup, the reproduction protocol table by table, training, and provenance notes |

## Quick start

```bash
scripts/install.sh                 # root env + the five benchmark envs (uv)
scripts/fetch_assets.sh            # upstream benchmark checkouts + CRMArena-Pro databases
./ejepa bench list                 # should list the five benchmarks
```

The world model needs a checkpoint. Either train one (`docs/training.md`) or place an
existing one at `checkpoints/jepa` (`scripts/fetch_assets.sh checkpoints` prints the
expected layout).

A single run, world model on, beam-search harness:

```bash
JEPA_GPU=0 BENCH=AutomationBench AGENT_URL=http://127.0.0.1:18045/v1 AGENT_NAME=Qwen3.6-27B \
  scripts/run_main_table_repeats.sh
```

## Reproducing the paper

`docs/reproduction.md` maps every table and figure to the command that produces it and the
summarizer that turns runs into the reported number. In outline:

| Paper | Run | Summarize |
|---|---|---|
| Table 3 (task success rates) | `scripts/run_main_table_repeats.sh` (per benchmark) | `scripts/summarize_main_table.py` |
| Table 4 (prediction controls) | `scripts/run_prediction_controls.sh` | `scripts/summarize_wm_harness_summaries.py` |
| Table 5 (vs the tool-output LLM-WM) | `WORLD_MODELS=agentworld scripts/run_main_table_repeats.sh` | `scripts/summarize_main_table.py` |
| Table 10 (planning-budget ablation) | `scripts/run_beam_ablation.sh` | `scripts/summarize_beam_ablation.py` |
| Figures 2 and 4 (latency) | `scripts/measure_wm_latency_vs_horizon.py` | `scripts/plot_wm_latency_hf_vs_production.py` |
| Table 2 (next-state prediction) | Stage-2 training (`docs/training.md`) | the trainer's `run_summary.json` |

The fixed configuration behind every number — eight candidate plans, imagination horizon
three, two executed steps per re-plan, open-loop rollouts, score margin 0.10, temperature
0.7, one refinement round over the top four, terminal advice at 0.75 — is stated with its
flags in `docs/main_table_protocol.md`, along with the per-benchmark parallelism and the
measured cost of each run. The repository carries only what the paper reports; see
`docs/provenance.md`.

## Requirements

* Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/)
* An OpenAI-compatible endpoint serving the policy LLM (the paper uses Qwen3.6-27B under
  vLLM on one H200; see Appendix E.1 of the paper)
* A second endpoint for the LLM world-model arms
* One GPU for the JEPA world model (it runs in-process, not served)
* Docker for the CRMArena-Pro databases, the EnterpriseOps-Gym MCP tool servers, and
  Terminal-Bench 2.0's per-task containers

## Notes

`docs/provenance.md` records what this repository is: an extraction from a larger internal
benchmark monorepo, what was dropped in the extraction, and the handful of places where the
code here deliberately differs from the code that produced the recorded runs.
