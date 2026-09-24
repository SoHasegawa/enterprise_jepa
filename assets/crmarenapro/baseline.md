# crmarenapro `crmarena_100_test` baseline

Reference baseline for the fixed 100-task held-out split (`target=crmarena_100_test`,
`green/tasks/task_ids.toml`). The split is byte-identical to the seed-42
`task_limit=100` sample, drawn from the 2,140-task CRMArenaPro b2b corpus; the
remaining 2,040 tasks form `crmarena_train` (1,836) / `crmarena_valid` (204),
derived test-excluded at load time (`green/crm/splits.py`, sha256-ordered 10%
valid) — same design as EnterpriseOps-Gym `opsgym_80_test`.

## Cross-model summary

Same executor (`baseline_crm_agent`) and split (`crmarena_100_test`) across rows;
the **mode** column marks whether the adversarial perturbations are on.

| Model | Mode | Runs | Pass rate (entropic) | Original accuracy | Avg entropic score | Eval duration |
|-------|------|------|----------------------|-------------------|--------------------|---------------|
| `Qwen3.6-27B` on the GPU cluster AMD/ROCm (MI300X, TP=8) | adversarial (drift+rot=medium, 8–10 turns, temp 0.1) | 5 | **25.6%** (σ ≈ 1.5%) | **24.0%** (σ ≈ 1.6%) | 64.0 (σ ≈ 0.7) | ~37 min |
| `Qwen3.6-27B` on the GPU cluster AMD/ROCm (MI300X, TP=8) | leaderboard-replication (drift+rot=none, 20 turns, temp 0.0) | 1 | **46.0%** | **44.9%** | 72.0 | ~97 min |
| _placeholder: next model / config_ | – | – | – | – | – | – |

**Takeaways:**

- **Qwen3.6-27B (5-run mean)** is the reference baseline. Per-run pass rates were
  24 / 27 / 26 / 24 / 27% (original accuracy 23 / 25 / 24 / 22 / 26%). Run-to-run
  spread at `temperature=0.1` is tight (σ ≈ 1.5 pp).
- The gap between `pass_rate` (entropic `crm_reward == 1`) and `original`
  accuracy is small (~1.6 pp); both track the same fuzzy-match/exact-match
  reward per category.
- Avg entropic score (~64/100) is much higher than the pass rate because the
  seven-dimension scorer awards partial credit (efficiency, error recovery,
  drift adaptation) even when the final answer is wrong.

Measured 2026-07-06 PST (2026-07-07 JST on the GPU cluster). Benchmark branch
`crmarenapro-qwen36-baseline`, runs tagged
`qwen36_baseline_20260706_180406_r{1..5}` (Slurm jobs 236080–236083, 236087).

## Shared configuration

| Setting | Value |
|---------|-------|
| Executor | `baseline_crm_agent` (schema-aware SQL ReAct) |
| Split | `crmarena_100_test` (fixed 100 tasks, seed-42 sample) |
| Corpus | CRMArenaPro b2b (2,140 tasks) |
| Drift / rot | medium / medium (hardcoded by green agent) |
| Max agent turns | 10 (hardcoded) |
| Per-task timeout | 300 s (hardcoded) |
| LLM temperature | 0.1 (executor default) |
| Max tokens / call | 2,048 |
| Model serving | node-local vLLM, `Qwen/Qwen3.6-27B` served as `Qwen3.6-27B`, TP=8 |
| GPUs | 1× the GPU cluster node, 8× MI300X (ROCm 7.2) |
| Capture trajectory | `false` runs 1–4, `true` run 5 |

## Qwen3.6-27B baseline on the GPU cluster AMD/ROCm (5 runs)

| Run | Slurm job | Pass rate | Original accuracy | Avg score | Eval wall time |
|-----|-----------|-----------|-------------------|-----------|----------------|
| r1 | 236080 | 24% | 23% | 63.3 | 37.4 min |
| r2 | 236081 | 27% | 25% | 64.7 | 36.3 min |
| r3 | 236082 | 26% | 24% | 64.2 | 36.6 min |
| r4 | 236083 | 24% | 22% | 63.3 | 37.2 min |
| r5 | 236087 | 27% | 26% | 64.6 | 45.5 min |
| **mean** | | **25.6%** | **24.0%** | **64.0** | ~22–27 s/task |

All five runs score the same fixed 100 tasks (identical task set; variance is
sampling noise at temperature 0.1 only). r5 additionally captured full
trajectories (`capture_trajectory=true`); its slightly longer wall time includes
trajectory serialization.

### Noise floor

Run-to-run σ ≈ 1.5 pp over 5 runs. With ~2-standard-error thresholds as guides:

| Comparison | Minimum gain to treat as likely real |
|------------|--------------------------------------|
| One run vs one baseline run | **≥4–5 pp** (~4–5 tasks) |
| One run vs the 5-run baseline mean | **≥3–4 pp** |
| 5-run mean vs 5-run baseline mean | **≥2 pp** (~2 tasks) |

The more sensitive test is **paired per-task comparison**: the 100 tasks are
fixed, so diff per-task outcomes against the baseline runs and count flips
(McNemar or a bootstrap over tasks) rather than comparing aggregate rates.

### Failure anatomy

_Placeholder — to be filled from per-task outcome analysis across the 5 runs
(always-pass / flaky / never-solved counts, failing categories, ceilings)._

## Qwen3.6-27B leaderboard-replication on the GPU cluster AMD/ROCm (1 run)

To approximate the public
[CRMArena leaderboard](https://huggingface.co/spaces/Salesforce/CRMArena-Leaderboard)
setting as closely as this asset allows, a single run used
`--config target=crmarena_100_test leaderboard_mode=true task_limit=0` with
`TEMPERATURE=0.0` and `MAX_TURNS=20`.

`leaderboard_mode=true` turns off the adversarial perturbations that the strict
eval hardcodes: **schema drift → none**, **context rot → none**, and it raises
the interaction budget (green `max_steps` 10 → 20; the executor's own
`MAX_TURNS` must be raised alongside, 8 → 20). Temperature 0.0 makes the run
deterministic (one canonical number), matching how leaderboard entries are
reported.

| Metric | Adversarial (5-run mean) | Leaderboard-replication (1 run) | Δ |
|--------|--------------------------|---------------------------------|---|
| Pass rate (entropic) | 25.6% | **46.0%** | +20.4 pp |
| Original accuracy | 24.0% | **44.9%** | +20.9 pp |
| Avg entropic score | 64.0 | 72.0 | +8.0 |
| Eval duration | ~37 min | ~97 min | — |

Slurm job 236117, run tag `qwen36_leaderboard_test_20260706_195718`, config
hash `51e72a6dccb0`; log confirms `Drift: none, Rot: none` over all 100 tasks.

**Not directly comparable to the leaderboard.** Removing the perturbations
roughly doubles accuracy (24% → 45%), but the residual gap to frontier entries
(e.g. o1 ≈ 64%) comes from two differences this mode cannot close:

1. **Backend / action space.** This asset queries a bundled SQLite snapshot of
   the org with hand-written SQL (`<execute>…</execute>`); the official harness
   drives a live Salesforce sandbox through function-calling APIs. Different
   tools, observations, and judging path — no Salesforce credentials are used
   here at all.
2. **Model class.** `Qwen3.6-27B` (27B dense, greedy) vs frontier reasoning
   models. Being newer does not close the multi-hop reasoning gap.

Treat this as "the same executor/model with adversarial noise removed", i.e. an
upper bound for *this* asset — not a Salesforce-leaderboard-equivalent score.

## Train/valid trajectory generation (graph-building data)

Multi-temperature trajectory generation on `crmarena_train` / `crmarena_valid`
runs via:

```bash
assets/crmarenapro/submit_generate.sh \
  --splits "train valid" \
  --temps "0.2 0.4 0.6 0.8 1.0 1.2" \
  --reps 3 \
  --array-throttle 4 \
  --partition batch-8gpu \
  --slurm-time 23:59:00
```

First sweep: Slurm array 236092 (12 cells = 2 splits × 6 temps, reps in-cell,
manifest-based resume). Note a train rep is ~15 h of sequential eval, so 3 train
reps do not fit one 24 h allocation — resubmit the same command to resume
remaining reps (completed manifests are skipped).

_Placeholder — sweep completion status / trajectory counts per (split, temp,
rep) to be recorded here when the sweep finishes._

## Reproducibility

```bash
# 5-run baseline on the fixed test split (submits detached Slurm jobs,
# one 8-GPU node per run, node-local Qwen3.6-27B vLLM):
scripts/run_crmarenapro_qwen36_baseline_slurm.sh

# equivalently, per run (task_limit=0 = full split; benchmark.toml defaults
# task_limit=1, which would otherwise truncate the split to a single task):
ejepa --result-root <dir> bench run crmarenapro \
  --launcher local --ready-timeout 600 \
  --executor baseline_crm_agent \
  --config target=crmarena_100_test --config task_limit=0

# leaderboard-replication run (adversarial perturbations off, deterministic):
TEMPERATURE=0.0 MAX_TURNS=20 \
ejepa --result-root <dir> bench run crmarenapro \
  --launcher local --ready-timeout 600 \
  --executor baseline_crm_agent \
  --config target=crmarena_100_test --config task_limit=0 \
  --config leaderboard_mode=true
```

Results land in `<result-root>/bm-*/{manifest.json,detail.json}`;
`detail.json.summary.pass_rate` is the headline number,
`detail.json.original.scores.accuracy` the original CRMArena-Pro accuracy.
Expected duration per run: ~15 min vLLM boot (weights cached) + ~37 min eval.

_Placeholder — add new baseline entries above (cross-model summary + a per-model
section) rather than editing existing ones; keep prior sections untouched._
