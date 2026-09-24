# Provenance and deviations

This repository was extracted from a larger in-house benchmark monorepo and a separate
world-model research repository. Everything needed for the paper's claims is here; a number
of things that lived alongside it are not. This file records both, so that a difference between
this code and the code that produced the recorded runs is never a surprise.

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

## Code that is a port, not a copy

`src/ejepa_wm/backends/_ewm_jepa.py`, `_ewm_finetuning.py`, `_ewm_runtime.py`,
`_ewm_k_controller.py` and `_canonical_event_state.py` are inference-only ports of the EWM
research code. They deliberately have no imports from an EWM checkout: the evaluation side
stands alone, and `training/` is only needed to produce a checkpoint.
