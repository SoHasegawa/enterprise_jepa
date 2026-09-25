# Week 1 experiments (plan.md §"Week 1: diagnose the ceiling and lock the evaluation")

Three scripts, one per Week-1 deliverable. All operate on the **existing 11-field canonical
state schema** — plan.md's proposed `transition_state`/`task_state`/`policy_head`
restructuring is deliberately **not** applied.

Shared data: the recognition-probe JSONLs (non-terminal steps carrying a recoverable
observation), 13,099 train / 4,886 eval rows. Default trunk: `data_jepa_all_adp_06b_decoder_msp`
(highest-accuracy pretrain of the 31 surveyed).

---

## 1. Annotation audit — `week1_annotation_audit.py`

Answers: *are the GPT labels trustworthy, and which fields are mechanically derivable?*

```bash
# (a) no human effort — deterministic rules vs GPT labels over the whole eval split
uv run python src/analysis/week1_annotation_audit.py --stage rules

# (b) stratified adjudication packets (>=1 per benchmark x field x value, then sqrt-proportional)
uv run python src/analysis/week1_annotation_audit.py --stage sample --num-samples 250

# (c) after filling data/week1/audit_adjudication_template.jsonl
uv run python src/analysis/week1_annotation_audit.py --stage score \
    --adjudicated data/week1/audit_adjudication_template.jsonl
```

Packets carry everything plan.md asks the annotator to see: system+task prompt, full
pre-action history, action, next observation, subsequent trajectory, trajectory outcome, plus
the GPT and rule labels for reference. Metrics are accuracy, macro-F1, balanced accuracy and
Cohen's κ (not accuracy alone).

Deterministic labelers cover the five mechanically-derivable fields (`action_type`,
`object_type`, `error_signature`, `execution_status`, `side_effect_type`); the genuinely
semantic nudge fields are LLM-only by design and go straight to adjudication.

## 2. Ceiling ladder — `week1_ceiling_ladder.py`

Answers: *where is the bottleneck — imbalance, predictor, representation, pooling, or labels?*

```bash
uv run python src/analysis/week1_ceiling_ladder.py --stage encode --device cuda:2   # once
uv run python src/analysis/week1_ceiling_ladder.py --stage train  --device cuda:2
# optional annotation-noise row:
uv run python src/analysis/week1_ceiling_ladder.py --stage train \
    --adjudicated-labels data/week1/audit_adjudication_template.jsonl
```

Rows: `majority` (class prior) · `predicted` (deployed) · `target_obs` · `full_context_obs` ·
`attn_pooled_obs` (learned-query attention over observation tokens) · `backbone_direct`
(projector skipped) · `adjudicated_subset`. Identical head/protocol/seed across rows, so
differences are attributable to the input.

Metrics per plan.md: accuracy, **macro-F1, balanced accuracy, NLL, Brier, ECE**.

Decision gate (plan.md): low agreement or weak full-input supervised → relabel/simplify;
labels fine but global embeddings weak → change representation; target-state strong but
predicted-state weak → return to the predictor.

## 3. Decision-centric ranking — `week1_decision_ranking.py`

Answers: *does the model rank actions correctly — the quantity the planner actually consumes?*

```bash
uv run python src/analysis/week1_decision_ranking.py \
    --checkpoint checkpoints/jepa_msp_decoder_cls_head --device cuda:3 --max-states 800
```

Per logged state, builds `demonstrated` + `wrong_tool` + `wrong_args` (same tool, arguments
borrowed from another trajectory) + `redundant` (previous action replayed) + `risky`
(destructive tool from the same inventory). Scores every candidate through the **deployed**
path — `predict_latent` → canonical heads → `logits_to_field_probs` → `score_step` — so this
measures the planner's real ranking function.

Reports Recall@1, MRR, per-distractor pairwise preference accuracy, risky-action rejection
rate, and the demonstrated-vs-best-distractor score margin with a paired bootstrap 95% CI.

---

### Suggested order

`--stage rules` (minutes, no dependencies) → ceiling ladder encode+train (GPU) → decision
ranking (GPU) → human adjudication of the 250 packets → re-run audit `--stage score` and the
ladder's `--adjudicated-labels` row.
