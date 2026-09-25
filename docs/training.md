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
cd training
uv run python src/generation/generate_enterpriseops_gym_multi_model_world_model_trajectories.py
uv run python src/data_preparation/convert_crmarenapro_jsonl_to_crmarena_results.py
uv run python src/data_preparation/convert_terminalbench_jsonl_to_trajectories.py
uv run python src/data_preparation/world_model_trajectory_cleanup.py
```

These read benchmark run output (`BENCHMARK_RESULT_ROOT`) and write into
`training/trajectories/`, which is not tracked — the trajectory corpus is several hundred
MB. Task-level train/eval disjointness is enforced by the split manifests, not by a random
row split, so the same task never appears on both sides.

The **expanded corpus** adds filtered tool-use and software-engineering trajectories from
the Agent Data Protocol release (paper Table 6 lists the sources and their counts):

```bash
uv run python src/generation/generate_adp_world_model_trajectories.py   # one file per ADP subset
```

`--trajectory-dataset` then selects the corpus. The presets that matter here are
`enterprise_tool_calling_plus_swe_25k` (expanded), `core` (EnterpriseOps-Gym +
CRMArena-Pro + Terminal-Bench) and `core_no_terminalbench` (the two in-distribution
benchmarks only); `src/finetuning_jepa.py` lists the rest.

## Stage 2 — canonical-event labels

Each action gets a canonical event label. The file the Stage-2 command consumes is named
for the chain that produces it — `..._cleaned_ensemble_value_scored.jsonl`:

```bash
cd training
# 1. clean the materialized LLM-labelled examples
uv run python src/data_preparation/clean_canonical_event_examples.py
# 2. re-annotate with a multi-model ensemble, then take majority vote
uv run python src/data_preparation/ensemble_relabel_canonical_events.py
uv run python src/data_preparation/apply_ensemble_consensus_labels.py
# 3. add the per-step value-head targets
uv run python src/data_preparation/annotate_step_value_scores.py
```

Each row carries `system_prompt`, `task_prompt`, `action`, `input_history`,
`canonical_event_state` and `nudge`. The ensemble step is what paper Appendix C reports
inter-rater agreement over (Table 7).

## Stage 3 — JEPA latent dynamics

> Reconstructed from the checkpoint's `jepa_data_manifest.json` and
> `jepa_training_metrics.json`, not re-run. Every flag below is corroborated by the
> manifest; the run itself was 8 GPUs × batch 8, 18,388 steps = exactly one epoch over
> 1,176,788 examples, 188,891,136 trainable parameters, top 4 of 28 encoder layers
> unfrozen.

Expanded corpus — the paper's *JEPA, expanded* row of Table 2:

```bash
cd training
uv run torchrun --nproc_per_node=8 src/finetuning_jepa.py \
  --model Qwen/Qwen3-Embedding-0.6B \
  --truncate-states-keep-newest --max-input-length 8192 \
  --backbone-type encoder --pooling last_token \
  --trajectory-dataset enterprise_tool_calling_plus_swe_25k \
  --predictor-arch transformer --predictor-transformer-layers 8 \
  --predictor-transformer-heads 16 --predictor-history-length 8 \
  --canonical-event-head-inputs state --event-target \
  --unfreeze-top-backbone-layers 4 --backbone-learning-rate 5e-6 \
  --sigreg-coeff 0.05 --latent-loss-coeff 1.0 --latent-loss-type smooth_l1_cosine \
  --num-train-epochs 1 --per-device-train-batch-size 8 --bf16 \
  --output-dir ../checkpoints/jepa_pretrain_expanded
```

Small corpus — the *JEPA, small* row — is the same command with
`--trajectory-dataset core`.

**Read this before comparing to the paper.** `core` is EnterpriseOps-Gym + CRMArena-Pro
+ **Terminal-Bench**, 3,380 trajectories. The paper describes the small corpus as 4,515
trajectories from the two in-distribution benchmarks only, which is the
`core_no_terminalbench` preset, not `core`. As run, the small arm saw the out-of-domain
benchmark during Stage 1, which weakens the out-of-domain claim for that row; the expanded
arm, which the agentic results use, is unaffected.

## Stage 4 — canonical-event heads

> Reconstructed from `canonical_event_data_manifest.json`, not re-run. 17,066 train /
> 5,854 eval examples, 17,354,840 trainable parameters — the numbers in the shipped
> checkpoint's `run_summary.json`.

The heads are trained on top of a frozen Stage-1 checkpoint:

```bash
cd training
uv run torchrun --nproc_per_node=2 src/finetuning_jepa.py \
  --train-canonical-event-heads-only \
  --model Qwen/Qwen3-Embedding-0.6B \
  --truncate-states-keep-newest \
  --backbone-type encoder --pooling last_token \
  --jepa-checkpoint-path ../checkpoints/jepa_pretrain_expanded \
  --canonical-event-train-jsonl trajectories/canonical_event_with_nudge_llm_enterpriseops_gym_crmarenapro_train_examples_cleaned_ensemble_value_scored.jsonl \
  --canonical-event-eval-jsonl  trajectories/canonical_event_with_nudge_llm_enterpriseops_gym_crmarenapro_eval_examples_cleaned_ensemble_value_scored.jsonl \
  --canonical-event-heads all --canonical-event-head-inputs state \
  --canonical-event-head-hidden-size 512 \
  --canonical-event-class-balance effective_num --canonical-event-cb-beta 0.9999 \
  --terminal-loss-coeff 1.0 --num-train-epochs 5 --learning-rate 5e-4 --terminal-class-balance effective_num --terminal-cb-beta 0.9999 \
  --per-device-train-batch-size 24 --bf16 \
  --canonical-event-dump-predictions ../checkpoints/jepa/eval_predictions_per_example.jsonl \
  --output-dir ../checkpoints/jepa
```

This writes `canonical_event_training_metrics.json`, which is the source of the three JEPA
columns of Table 2 and of Figure 3.

The result directory is what `--wm-ewm-jepa-checkpoint` consumes.

## The paper's checkpoint, exactly

`data_jepa_heads_partial_imb_terminal_3`, trained on 17,066 canonical-event examples and
evaluated on 5,854, over EnterpriseOps-Gym + CRMArena-Pro trajectories. Its
`run_summary.json` reports per-head accuracy — 0.875 `execution_status`, 0.963
`risk_signal`, 0.771 `terminal`, down to 0.246 `missing_information_type`. Those numbers
are the ceiling on how much signal the planner has to work with.

`training/` vendors the EWM `jepa` branch, the tree these commands were run from, so every
flag above resolves.

## The state-output LLM world model

> Verified: these two commands were run to produce the checkpoint the agentic results use.
> Training took 2 h 46 m on 2 GPUs (3,201 steps); the eval pass ~2 h at ~45 rows/min.

The baseline world model is a generative fine-tune on the same labelled corpus.

```bash
cd training
uv run torchrun --nproc_per_node=8 src/finetuning.py \
  --model ../checkpoints/llm_wm_base \
  --world-model-target canonical_event_with_nudge \
  --train-data-path trajectories/..._train_examples_cleaned_ensemble_value_scored.jsonl \
  --eval-data-path  trajectories/..._eval_examples_cleaned_ensemble_value_scored.jsonl \
  --state-history-size 8 \
  --include-world-model-history \
  --num-train-epochs 1 --learning-rate 2e-5 \
  --per-device-train-batch-size 2 --gradient-accumulation-steps 4 \
  --bf16 --gradient-checkpointing \
  --output-dir ../checkpoints/llm_wm_state
```

Three things will silently give you the wrong result here:

1. **`--include-world-model-history` is required.** Omitting it reproduces the original,
   defective baseline — the world model then predicts without the action/observation
   history the JEPA model gets, which is not the comparison the paper reports.
2. **Evaluation must be a separate single-process pass.** Under `torchrun` the eval block
   no-ops: you get `evaluation_metrics: null` and no metrics file, with no error. This is
   why the original checkpoint's metrics live in a sibling `_eval/` directory.
3. **`--world-model-eval-samples 0` means all rows.** The default of 2000 does not match
   the reported n = 5,854.

```bash
cd training
CUDA_VISIBLE_DEVICES=0 uv run python src/finetuning.py \
  --skip-training \
  --model            ../checkpoints/llm_wm_state \
  --world-model-path ../checkpoints/llm_wm_state \
  --world-model-target canonical_event_with_nudge \
  --train-data-path <train.jsonl> --eval-data-path <eval.jsonl> \
  --state-history-size 3 --include-world-model-history \
  --world-model-eval-samples 0 \
  --world-model-eval-dump-predictions ../checkpoints/llm_wm_state_eval/eval_predictions_per_example.jsonl \
  --output-dir ../checkpoints/llm_wm_state_eval
```

At evaluation time this checkpoint is served on an OpenAI-compatible endpoint rather than
loaded in-process; the agentic harnesses reach it with
`--wm-llm-ewm-mode llm_canonical_trained`.

Per-field accuracy for its Table 2 row comes from
`training/src/analysis/llm_world_model_field_eval.py`, run over the dumped predictions.

## Table 2 and Figure 3

Macro-F1, macro-recall and the per-category recall behind Figure 3 are computed from the
per-example prediction dumps that Stage 2 and the LLM-WM eval write:

```bash
cd training
uv run python src/analysis/canonical_event_prediction_report.py   # JEPA columns
uv run python src/analysis/llm_world_model_field_eval.py          # LLM-WM column
```

The training metrics files (`canonical_event_training_metrics.json`, `run_summary.json`)
carry only marginal class distributions, which is why per-class P/R/F1 needs the dumps.
The final matplotlib panels were drawn by an ad-hoc script outside either repository and
are not included; the numbers they plot come from these two commands.

The Stage-1 ablation behind the *JEPA, no Stage 1* row is
`src/analysis/jepa_readout_ablation.py`.
