# Provenance and deviations

This repository was extracted from a larger in-house benchmark monorepo and a separate
world-model research repository. Everything needed for the paper's claims is here; a number
of things that lived alongside it are not. This file records both, so that a difference between
this code and the code that produced the recorded runs is never a surprise.

## Scope: what the paper reports

The repository carries the experiments the paper reports and nothing else. Mapping:

| Paper | Here |
|---|---|
| Table 2, Figure 3 — next-state prediction | `training/` (Stage-2 eval metrics) |
| Table 3 — task success rates | `scripts/run_main_table_repeats.sh`, `summarize_main_table.py`, `summarize_row_averages.py` |
| Table 4 — prediction controls | `scripts/run_prediction_controls.sh` (`no_state`, `shuffled`) |
| Table 5 — vs the tool-output LLM-WM | the `agentworld` arm of `run_main_table_repeats.sh` |
| Table 6 — Stage-1 trajectory sources | `training/src/generation/generate_adp_world_model_trajectories.py` |
| Table 9 — help mechanisms | `docs/qualitative_analysis.md` |
| Table 10 — planning-budget ablation | `scripts/run_beam_ablation.sh`, `summarize_beam_ablation.py` |
| Figures 2, 4 — beam-search latency | `scripts/measure_wm_latency_vs_horizon.py`, `plot_wm_latency_hf_vs_production.py` |

Measurements that were made during the work but are **not** in the paper were removed with
their scripts, figures and write-ups: the cost-matched world-model comparison, the
success-latency Pareto figure, the task-time decomposition figure, the beam-width latency
sweep, the JEPA per-stage profile, per-task token accounting, the behavioural-metrics CSV,
and the critic-triggered (`beam_critic`) harness. Two prediction-control arms that the code
still supports (`uniform`, `prior`) are likewise not reported.

The remote-inference launcher (SSH/Slurm-managed vLLM) is also gone: the paper serves the
agent and the world models locally on one H200 (Appendix E.1), so the CLI now expects
endpoints that are already running.

## What was dropped

* **Other benchmarks.** The source monorepo carries ~15 benchmark assets. Only the five the
  paper reports are here: EnterpriseOps-Gym, CRMArena-Pro, WorkBench, AutomationBench and
  Terminal-Bench 2.0.
* **Other agent scaffolds.** A separate in-house agent scaffold and its test-time
  self-evolution machinery, together with the corresponding CLI options
  (`--max-self-evolutions`, `--max-benchmark-self-evolution-cycles`,
  `--self-evolve-strategy`) and result columns. The paper runs the `mcp_react` executor on
  every benchmark; the green agents now carry no-op stands-in for the self-evolve hooks, so
  they run without that runtime.
* **The in-house model gateway.** A world-model backend that reached an internal chat
  gateway through its model factory. The `served` backend now builds its chat client directly through
  the `openai` SDK (`src/ejepa_wm/backends/_openai_chat.py`), so it works against any
  OpenAI-compatible endpoint.
* **The graph/NLA world-model line.** `assets/EnterpriseOps-Gym/wm/` (workflow mining,
  success maps, the scorer sidecar) and the `sidecar` world-model backend. That is an
  earlier, different world model; none of the paper's numbers come from it.
* **The `mcp_react_ewm` executor**, which vendored a second copy of the EWM training code.
* **Raw run output.** `results/` carries the derived artefacts — the per-run CSVs, the
  figure data and the latency measurements. The ~260 GB of raw harness summaries, logs and
  trajectories behind them are not distributable here; the run scripts regenerate them.
* **Large data.** The CRMArena-Pro SQLite databases, the EnterpriseOps LoRA adapter and
  tokenizer, and the training trajectory corpus. `scripts/fetch_assets.sh` retrieves what
  can be retrieved and tells you where to put the rest.

## Renames

The CLI and its two packages were renamed: the command is `ejepa`, and the
packages are `ejepa_cli` and `ejepa_wm`. Run labels and output directories moved with them —
harness summaries are `ejepa-wm-harnesses-*.json`, repeat summaries are
`ejepa-bench-repeat-*.json`, and `out/` is `results/`. Summaries recorded before the rename
carry the old prefixes and will not match the summarizers' globs until renamed.

## Deviations from the code that produced the recorded runs

1. **Tool-output LLM world-model prompt.** `_ewm_qwen_agentworld.py` appends a section
   header to the adapted prompt. That header was renamed along with the CLI and now reads
   `# EJEPA Benchmark Integration (adapted)`; one token of the prompt therefore differs from
   the recorded runs for that arm.

6. **Names and paths.** Internal host names, cluster names, SSH host presets, storage roots,
   user names and gateway URLs were replaced with neutral placeholders or environment
   variables (`BENCHMARK_HOME`, `RIL_*_HOSTNAME`, `upstreams/`, `checkpoints/`). Results now
   default to the repository's own `results/` directory instead of a shared cluster path.
2. **JEPA training script version.** The vendored trainer predates the heads the paper's
   checkpoint uses. See the "Known gap" section of `docs/training.md`.
3. **Dropped stale test.** A Terminal-Bench world-model injection unit test asserted an
   executor shape that no longer exists and reached a live LLM call when adapted; it was
   removed rather than rewritten (noted in `tests/test_mcp_react_executor_ports.py`). Two
   sibling assertions in that file were updated to the current executor's attribute names.
4. **Paths.** Absolute paths to this machine's checkouts were replaced with `upstreams/`
   and `checkpoints/` defaults, overridable by the same environment variables as before.
5. **Lint scope.** `src/ejepa_wm` is kept at the upstream code style (it is a port, and
   diffing it against the research repo should stay easy), so style-only rule families are
   silenced for it in `pyproject.toml`. Correctness rules are not.

## Where the code and the paper disagree

Found while reconciling the recorded runs with the submission. None of these are changed in
the code — they are recorded so a reader who compares the two is not misled.

1. **Predictor depth.** Appendix E.1 says the predictor "comprises eight Transformer
   blocks". The shipped checkpoint's `jepa_data_manifest.json` records
   `predictor_transformer_layers: 6` (with `predictor_transformer_heads: 16` and
   `predictor_history_length: 8`, both as described), and the Stage-1 command passes
   `--predictor-transformer-layers 6`. Eight is the number of history tokens, not blocks.
2. **Small-corpus composition.** §5.2 describes the small corpus as 4,515 trajectories
   from EnterpriseOps-Gym and CRMArena-Pro. The `core` dataset the *JEPA, small* run
   actually used is EnterpriseOps-Gym + CRMArena-Pro + Terminal-Bench, 3,380 trajectories,
   so that row's Stage 1 saw the out-of-domain benchmark. The expanded corpus, which the
   agentic results use, is unaffected.
3. **Encoder learning rate.** Appendix E.1 gives 1e-6 for the text encoder; the Stage-1
   command passes `--backbone-learning-rate 5e-6`.
4. **AutomationBench size.** §5.2 says 392 tasks (and the 1,679 total is computed with
   392); Appendix A says all 600 are run and 400 reported after excluding marketing and
   finance. The data and `scripts/summarize_main_table.py` agree with the appendix — the
   summarizer requires exactly 400 scored tasks and drops a cell otherwise.

## Not reproducible from this repository

The scripts that assembled Table 2 and drew Figure 3 were written ad hoc outside either
repository (`/tmp/plot_percat.py`, `/tmp/table1_refresh.py`) and no longer exist. Their
inputs — each checkpoint's `canonical_event_training_metrics.json` and the LLM-WM
evaluation dump — are produced by the documented training commands, so the numbers are
recoverable; only the table/figure assembly would have to be rewritten.

## Code that is a port, not a copy

`src/ejepa_wm/backends/_ewm_jepa.py`, `_ewm_finetuning.py`, `_ewm_runtime.py`,
`_ewm_k_controller.py` and `_canonical_event_state.py` are inference-only ports of the EWM
research code. They deliberately have no imports from an EWM checkout: the evaluation side
stands alone, and `training/` is only needed to produce a checkpoint.
