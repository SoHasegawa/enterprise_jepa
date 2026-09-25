# Training Enterprise-JEPA

`training/` is the data-preparation and training pipeline, vendored from the EWM research
repository the checkpoints were produced in. It is a separate project with its own
`pyproject.toml`; run it from inside `training/`.

```bash
cd training && uv sync
```

## What the model is

A text backbone encodes the task context, the action, and the recent action/observation
history into a latent; an AdaLN transformer predictor maps `(z_t, action_t)` to a predicted
`z_{t+1}`; the loss is latent prediction against a stop-gradient target encoding, plus a
SIGReg Gaussian regularizer that prevents collapse, plus optional observation
reconstruction. Nothing is decoded during planning: the harness reads the predicted latent
through **classification heads** over a canonical event vocabulary
(`execution_status`, `risk_signal`, `error_signature`, `progress_signal`, `object_type`,
`side_effect_type`, `action_type`, and the nudge fields `information_gain`,
`information_sufficiency`, `recommended_abstract_action`, `missing_information_type`),
plus a terminal head. That is what makes a rollout cheap: no autoregressive decoding per
imagined transition.

The net is defined in `training/src/finetuning_jepa.py` (`TextLeWorldModel`); the
inference-only port used at evaluation time is `src/ejepa_wm/backends/_ewm_jepa.py`.

## Stage 1 — trajectories

Agent trajectories are harvested per benchmark and converted into world-model training
examples:

```bash
uv run python src/generation/generate_enterpriseops_gym_multi_model_world_model_trajectories.py
uv run python src/data_preparation/split_crmarenapro_trajectories.py
uv run python src/data_preparation/split_terminalbench_trajectories.py
uv run python src/data_preparation/materialize_aligned_world_model_trajectories.py
```

These read benchmark run output (`BENCHMARK_RESULT_ROOT`) and write into
`training/trajectories/`, which is not tracked — the trajectory corpus is several hundred
MB. Task-level train/eval disjointness is enforced by the split manifests, not by a random
row split, so the same task never appears on both sides.

That yields the paper's **small corpus** (4,515 trajectories from the two in-distribution
benchmarks). The **expanded corpus** (316,200 trajectories) adds filtered tool-use and
software-engineering trajectories from the Agent Data Protocol release; convert them with

```bash
uv run python src/generation/generate_adp_world_model_trajectories.py
```

one output file per ADP subset. Paper Table 6 lists the sources and their counts.

## Stage 2 — canonical-event labels

Each action gets a canonical event label, assigned by an LLM labeller and then cleaned:

```bash
uv run python src/data_preparation/label_canonical_events_with_llm.py
uv run python src/data_preparation/generate_canonical_event_state_examples.py
uv run python src/data_preparation/world_model_trajectory_cleanup.py
```

Output is one JSONL row per action carrying `system_prompt`, `task_prompt`, `action`,
`input_history`, `canonical_event_state` and `nudge`. Label accuracy can be audited with
`src/analysis/calculate_canonical_field_accuracy.py`.

## Stage 3 — JEPA latent dynamics

```bash
uv run python src/finetuning_jepa.py \
    --train-data-path trajectories/<train>.json \
    --eval-data-path  trajectories/<eval>.json \
    --output-dir checkpoints/jepa_pretrain \
    --memory-tokens 8 --predictor-hidden-multiplier 4.0 \
    --sigreg-coeff 0.05 --max-input-length 8192 --bf16
```

## Stage 4 — canonical-event heads

The heads are trained on top of a frozen JEPA checkpoint, which is what the paper's
checkpoint is (`initialized_from_jepa_checkpoint` in its
`canonical_event_data_manifest.json`):

```bash
uv run python src/finetuning_jepa.py \
    --train-canonical-event-heads-only \
    --canonical-event-train-jsonl trajectories/canonical_event_..._train_examples.jsonl \
    --canonical-event-eval-jsonl  trajectories/canonical_event_..._eval_examples.jsonl \
    --canonical-event-head-hidden-size 512 \
    --output-dir checkpoints/jepa
```

The result directory is what `--wm-ewm-jepa-checkpoint` consumes.

## The paper's checkpoint, exactly

`data_jepa_heads_partial_imb_terminal_3`, trained on 17,066 canonical-event examples and
evaluated on 5,854, over EnterpriseOps-Gym + CRMArena-Pro trajectories. Its
`run_summary.json` reports per-head accuracy — 0.875 `execution_status`, 0.963
`risk_signal`, 0.771 `terminal`, down to 0.246 `missing_information_type`. Those numbers
are the honest ceiling on how much signal the planner has to work with, and
`docs/qualitative_analysis.md` §6 argues from them.

**Known gap.** The vendored `finetuning_jepa.py` is the last committed snapshot of the
training script (EWM branch `jepa`, 2026-07-17). The paper's checkpoint was produced by a
later, uncommitted revision: its manifest records heads that this snapshot does not build
(`terminal_head`, `value_head`, `obs_grounding`, `fast_lewm`, `action_decoder`). Training
from this snapshot therefore reproduces the architecture and objective but not that
checkpoint bit for bit, and a checkpoint trained here will have no terminal head (so
`--wm-beam-plan-terminal-advice` has nothing to read). The authoritative definition of the
full net, including those heads and their checkpoint loaders, is the inference port
`src/ejepa_wm/backends/_ewm_jepa.py`; reconstructing the missing training code means adding
their losses to this script against that definition.

## LLM world models

The two LLM world-model baselines are fine-tuned with the same corpus in generative form:
`src/data_preparation/convert_canonical_nudge_to_llama_factory.py` emits LlamaFactory alpaca
datasets, and `src/analysis/evaluate_llama_factory_canonical_event.py` scores the result.
The state-output world model used in the paper is `llm_wm_beam_action_terminal_crmarenapro`;
at evaluation time it is served on an OpenAI-compatible endpoint rather than loaded
in-process.
