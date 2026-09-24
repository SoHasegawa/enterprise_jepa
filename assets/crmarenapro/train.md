# crmarenapro trajectory generation (train / valid)

Multi-temperature, multi-rep trajectory generation over the fixed
`crmarena_train` (1,836 tasks) and `crmarena_valid` (204 tasks) splits, used to
build the World-Model graph. Both splits are **test-excluded by construction**
(`green/crm/splits.py`): they never overlap the 100-task `crmarena_100_test`
baseline set, so trajectories generated here are safe as WM training data.

The baseline/leaderboard *scores* live in [`baseline.md`](baseline.md); this file
is about **producing trajectories** at scale.

## Design

- **One split per submission.** The split is sharded into fixed-size task
  windows so each Slurm array cell finishes inside the 6h `batch-8gpu-short`
  wall.
- **Array cell = one `(temperature, shard)`.** Inside a cell, `--reps` runs are
  looped sequentially (default **5**). Each run is a full
  `ejepa bench run crmarenapro` with `capture_trajectory=true`.
- **Shard = task window** `[shard*shard_size, (shard+1)*shard_size)` of the
  split, passed to the green agent as `--config task_offset=… --config task_limit=…`.
- **Resumable.** Every `(shard, rep)` writes its own result dir; a rep whose
  `manifest.json` is `status=completed` is skipped. Re-running the exact same
  command continues where killed/failed cells left off.
- **Mode.** `--leaderboard` uses `leaderboard_mode=true` (schema drift + context
  rot OFF, `MAX_TURNS=20`) — clean trajectories. Omit it for the strict
  adversarial setting (`drift=rot=medium`, `MAX_TURNS=8`).

Cells = `#temps × ceil(split_size / shard_size)`. Sizing rule: keep
`shard_size × reps × per_task_seconds + vLLM_boot < wall`. On
`batch-8gpu-short` (6h), clean mode is ~58 s/task, so the default
`shard_size=50`, `reps=5` ⇒ ~4.3h/cell (safe margin).

## Reproduce

Run on the cluster login node from a repo checkout (e.g.
`~/repo/benchmarks-crmarenapro`). Requires the crmarenapro envs
(`scripts/install.sh crmarenapro`) and a vLLM at `~/repo/servers/.venv/bin/vllm`.

```bash
# valid split, clean (leaderboard) trajectories, 6 temps × 5 reps, sharded:
assets/crmarenapro/submit_generate.sh \
  --split valid --temps "0.2 0.4 0.6 0.8 1.0 1.2" \
  --reps 5 --shard-size 50 --leaderboard \
  --array-throttle 8 --partition batch-8gpu-short --slurm-time 05:59:00

# train split (same, larger — 37 shards × 6 temps = 222 cells):
assets/crmarenapro/submit_generate.sh \
  --split train --temps "0.2 0.4 0.6 0.8 1.0 1.2" \
  --reps 5 --shard-size 50 --leaderboard \
  --array-throttle 8 --partition batch-8gpu-short --slurm-time 05:59:00
```

Useful variants:

- **Adversarial trajectories:** drop `--leaderboard`.
- **Resume / top-up:** re-run the identical command — completed `(shard,rep)`
  cells are skipped, only missing ones run.
- **Bigger nodes (24h):** `--partition batch-8gpu --slurm-time 23:59:00
  --shard-size 200` (fewer, larger cells; fewer vLLM boots).
- **Dry run:** append `--dry-run` to print the `sbatch` line and shard math.
- **Smoke:** `--split valid --temps 0.7 --reps 1 --shard-size 2` (one tiny cell).

Monitor:

```bash
squeue -u "$USER" --name=crmgen-valid,crmgen-train -o "%.18i %.14j %.8T %R"
```

## Output layout (World-Model graph inputs)

```
<repo>/.cache/crmgen/
└── gen_<split>/                         # split = train | valid
    └── run_t<temp>/                     # temp = 0.2 … 1.2
        └── shardNN/                     # NN = 00 … (num_shards-1)
            └── repN/                    # N = 1 … reps
                └── bm-*/                # one ejepa run
                    ├── manifest.json    # status, config_hash, target
                    ├── detail.json      # per-task results + summary + scores
                    └── trajectories/    # <task_idx>.jsonl (A2A events +
                                         #   internal_trajectory artifact:
                                         #   format=crmarenapro_react, the SQL
                                         #   ReAct message log per task)
```

The per-task `trajectories/<idx>.jsonl` files are the graph-building inputs. Each
carries the executor's full turn sequence (`<thought>/<execute>/<describe>/<respond>`),
tool metrics, and the model/endpoint metadata. Note: each stored message is
truncated to `BENCHMARK_TRAJECTORY_MAX_TEXT_CHARS` (default 16,000) — raise the
env if you need longer observations verbatim.

To enumerate completed reps:

```bash
find <repo>/.cache/crmgen -path '*bm-*/manifest.json' \
  -exec grep -l '"status": *"completed"' {} + \
  | sed 's#.*/gen_#gen_#; s#/bm-.*##' | sort | uniq -c
```

## Current status / results

Scores below are per-rep binary correctness (`summary.pass_rate` = entropic
`crm_reward`; `original.scores.accuracy` = official-style accuracy). Generation
runs are not a "score" target — they exist to produce trajectories — but the
numbers are a useful sanity check.

### Clean (leaderboard) sweep — sharded, 5 reps — IN PROGRESS

- Jobs: `crmgen-valid` (236887, 30 cells), `crmgen-train` (236888, 222 cells),
  `batch-8gpu-short`, `shard_size=50`, launched 2026-07-07 PST.

| Split | Temps | Target reps/temp | Completed | Mean orig. acc | Notes |
|-------|-------|------------------|-----------|----------------|-------|
| valid | 0.2–1.2 | 5 | _pending_ | _tbd_ | _placeholder_ |
| train | 0.2–1.2 | 5 | _pending_ | _tbd_ | _placeholder_ |

### Earlier adversarial sweep (superseded layout, kept for reference)

Produced before the shard/clean rework, under the **old** unsharded path
`gen_<split>/run_t<temp>/repN/` (no `shardNN/`), in **adversarial** mode:

| Split | Temps with data | Reps each | Mean orig. acc | Tasks/rep |
|-------|-----------------|-----------|----------------|-----------|
| valid | 0.2–1.2 (all 6) | 3 | ~24% | 204 |
| train | 0.2, 0.4, 0.6, 0.8 | 2 | ~21% | 1,836 |

These adversarial trajectories remain on disk under the old layout and can be
used or discarded independently of the clean sweep above.

_Placeholder — update the "Clean sweep" table with completed rep counts, mean
accuracy per temp, and total trajectory counts once the sweep finishes; append
new sweeps as new subsections rather than editing prior ones._
