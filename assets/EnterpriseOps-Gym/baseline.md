# EnterpriseOps-Gym `opsgym_80_test` baseline

Reference baseline for the fixed 80-task held-out split (`target=opsgym_80_test`).
Task IDs match `ewm-enterprisearena` `enterpriseops_gym_80_test_task_split.json`.

## Cross-model summary

Same executor (`mcp_react`), orchestrator (`react`), HF mode (`oracle`), and sampling
settings across models. Config hash `89c6c6012051` unless noted.

| Model | Runs | Score rate | Tasks passed | Avg verifier pass | Eval duration |
|-------|------|------------|--------------|-------------------|---------------|
| `qwen3.6-27b` on NVIDIA H200 | 5 | **38.25%** (σ ≈ 2.6%) | 30.6 / 80 | 67.1% | ~4.2 min |
| `qwen3.6-27b` on the GPU cluster AMD/ROCm | 10 | **39.0%** (σ ≈ 2.8%) | 31.2 / 80 | 67.0% | 22.8 min mean; 22.9 min median |
| `gemma4-26b-a4b-it` | 1 | **31.25%** | 25 / 80 | 54.91% | 2.7 min |
| `nemotron-3-super-120b-a12b-nvfp4` | 5 | **19.50%** (σ ≈ 3.6%) | 15.6 / 80 | 42.9% | ~8.8 min |
| `nemotron-3-nano-30b-a3b` | 1 | **2.50%** | 2 / 80 | 9.98% | 3.3 min |

**Takeaways:**

- **Qwen3.6-27B on NVIDIA H200** is the reference baseline (5-run mean). Failure anatomy and noise
  floor below are measured on this model.
- **Qwen3.6-27B on the GPU cluster AMD/ROCm** scored **39.0%** over 10 same-node Slurm
  runs, essentially matching the prior NVIDIA H200 Qwen mean (**+0.75 pp**) with a
  similar verifier pass rate. Treat this as an AMD/ROCm runtime-stack result, not a
  clean hardware-only A/B: the the GPU cluster runs used the Slurm same-node wrapper, vLLM
  under ROCm, TP=8, and config hash `b1991ffc2738`; the earlier NVIDIA baseline used
  TP=4 on H200 and config hash `89c6c6012051`.
- **Gemma4-26B-A4B-IT** is ~7 pp below the Qwen 5-run mean on a single run — near the
  one-run noise threshold (≥7 pp vs one baseline run). A 5-run mean is needed before
  treating Gemma as clearly weaker.
- **Nemotron-3-Super-120B-A12B-NVFP4** (5-run mean **19.50%**) is **−18.8 pp** below
  Qwen and **−11.8 pp** below Gemma's single run. Tool calling works (`qwen3_coder` +
  `nemotron_v3`, `moe-backend marlin` on H200), but the 120B MoE model underperforms
  Qwen/Gemma on this agentic SQL benchmark despite higher capacity. Eval wall time is
  ~2× Qwen (~8.8 min/run). ~70 context-length 400 errors appeared across the 5 runs
  (`max-model-len=32768` with `max_tokens=8192`); raising context budget may recover a
  few tasks.
- **Nemotron-3-Nano-30B-A3B** required a vLLM 0.21 serve fix (`qwen3_coder` tool parser +
  built-in `nemotron_v3` reasoning parser). An earlier 1.25% run used the legacy HF
  `nano_v3` plugin and returned empty tool calls. After the fix, a single run scored
  **2.50%** — still well below Qwen/Gemma/Super, but tool use is confirmed working.

Measured 2026-06-12 – 2026-06-13 UTC. Benchmark commit `fa14b4bc` (branch
`feat/enterpriseops-opsgym-80-test`).

## Shared configuration

| Setting | Value |
|---------|-------|
| Executor | `mcp_react` |
| Orchestrator | `react` |
| HF mode | `oracle` |
| Max iterations | 50 |
| Max parallel | 40 |
| LLM temperature | 0.7 |
| LLM top_p | 0.95 |
| Capture trajectory | `true` |
| GPUs | `CUDA_VISIBLE_DEVICES=3,4,5,6` (4× H200, TP=4) |

`VLLM_MODEL` selects the local vLLM serve script (see Reproducibility).

## Qwen3.6-27B baseline on NVIDIA H200 (5 runs)

| Setting | Value |
|---------|-------|
| Model | `qwen3.6-27b` (local vLLM, served as `Qwen3.6-27B`) |

Run 1 used config hash `24ca0bee3cb8` (prior `run.sh` defaults). Runs 2–5 used config
hash `89c6c6012051` with the temperature / top_p settings above.

### Results

| Run | Run ID | Score rate | Total score | Avg verifier pass | Duration |
|-----|--------|------------|-------------|-------------------|----------|
| 1 | `20260612T193600Z-24ca0bee3cb8` | 40.00% | 32/80 | 68.63% | 4.6 min |
| 2 | `20260612T200136Z-89c6c6012051` | 35.00% | 28/80 | 64.02% | 4.2 min |
| 3 | `20260612T200855Z-89c6c6012051` | 36.25% | 29/80 | 67.44% | 4.0 min |
| 4 | `20260612T201507Z-89c6c6012051` | 41.25% | 33/80 | 69.41% | 4.4 min |
| 5 | `20260612T202144Z-89c6c6012051` | 38.75% | 31/80 | 66.18% | 4.0 min |

#### Aggregate (all 5 runs)

- Mean score rate: **38.25%** (σ ≈ 2.6%)
- Range: 35.0% – 41.25%
- Mean tasks passed: **30.6 / 80**
- Mean verifier pass rate: **67.1%**
- Mean eval duration: **~4.2 min/run** (~21 min total eval time)

#### Current config only (runs 2–5)

- Mean score rate: **37.81%** (σ ≈ 2.8%)
- Range: 35.0% – 41.25%

## Qwen3.6-27B on the GPU cluster AMD/ROCm (10 runs)

| Setting | Value |
|---------|-------|
| Model | `Qwen/Qwen3.6-27B` (same-node local vLLM, served as `Qwen3.6-27B`) |
| Cluster / accelerator stack | the GPU cluster AMD GPU nodes via ROCm vLLM |
| Slurm wrapper | a site-specific submit script, not included here |
| Slurm job | one `EWM_Test80` job per run, `--partition=batch-8gpu --gres=gpu:8 --nodes=1 --exclusive --time=01:00:00` |
| vLLM serve | TP=8 inferred from `SLURM_JOB_GPUS`; `--enable-auto-tool-choice`; `--tool-call-parser qwen3_xml`; `--reasoning-parser qwen3`; max model len 131072 |
| Config hash | `b1991ffc2738` |
| Run record | internal run log, not included here |

### Results

| Run | Slurm job | Node | Run ID | Score rate | Total score | Avg verifier pass | Benchmark duration | Slurm elapsed |
|-----|-----------|------|--------|------------|-------------|-------------------|--------------------|---------------|
| 1 | `232307` | `gpu25` | `20260627T001808Z-b1991ffc2738` | 35.0% | 28/80 | 66.36% | 951.480s | 00:21:29 |
| 2 | `232308` | `gpu27` | `20260627T001811Z-b1991ffc2738` | 37.5% | 30/80 | 65.58% | 1879.122s | 00:37:00 |
| 3 | `232309` | `gpu22` | `20260627T001808Z-b1991ffc2738` | 43.75% | 35/80 | 68.01% | 1413.271s | 00:29:14 |
| 4 | `232310` | `gpu23` | `20260627T001808Z-b1991ffc2738` | 37.5% | 30/80 | 66.62% | 1281.503s | 00:26:57 |
| 5 | `232311` | `gpu20` | `20260627T003738Z-b1991ffc2738` | 42.5% | 34/80 | 68.55% | 1397.143s | 00:28:59 |
| 6 | `232312` | `gpu25` | `20260627T003938Z-b1991ffc2738` | 38.75% | 31/80 | 67.74% | 1340.585s | 00:27:59 |
| 7 | `232313` | `gpu23` | `20260627T004515Z-b1991ffc2738` | 41.25% | 33/80 | 67.83% | 1544.119s | 00:31:30 |
| 8 | `232314` | `gpu22` | `20260627T004734Z-b1991ffc2738` | 40.0% | 32/80 | 68.69% | 1514.663s | 00:31:04 |
| 9 | `232315` | `gpu27` | `20260627T005522Z-b1991ffc2738` | 36.25% | 29/80 | 66.08% | 1349.709s | 00:28:21 |
| 10 | `232316` | `gpu20` | `20260627T010647Z-b1991ffc2738` | 37.5% | 30/80 | 64.83% | 1014.248s | 00:22:50 |

#### Aggregate (all 10 runs)

- Mean score rate: **39.0%** (σ ≈ 2.8%)
- Range: 35.0% – 43.75%
- Mean tasks passed: **31.2 / 80**
- Mean verifier pass rate: **67.0%**
- Mean benchmark duration: **1368.584s** (~22.8 min/run)
- Median benchmark duration: **1373.426s** (~22.9 min/run)
- Mean benchmark duration excluding the single slow run 2: **1311.858s** (~21.9 min/run)
- Slurm wall-clock window for the 10 submitted jobs: **1h05m36s**; the batch ran in
  two five-job waves on the current pool.

#### Comparison vs the previous NVIDIA H200 Qwen baseline

| Metric | NVIDIA H200 Qwen 5-run mean | the GPU cluster AMD/ROCm Qwen 10-run mean | Delta |
|--------|-----------------------------|------------------------------------|-------|
| Score rate | 38.25% | 39.0% | **+0.75 pp** |
| Tasks passed | 30.6 / 80 | 31.2 / 80 | **+0.6 tasks/run** |
| Avg verifier pass | 67.1% | 67.0% | **−0.1 pp** |
| Eval duration | ~4.2 min | 22.8 min mean; 22.9 min median | ~5.4× mean |

The the GPU cluster result is now statistically close to the NVIDIA H200 Qwen baseline and
well within the same broad performance band. The low run-to-run variance across 10
runs suggests stable runtime/configuration behavior rather than ordinary sampling
noise. However, the comparison is not a controlled hardware-only experiment because
the runtime stack and config hash changed. Before attributing the gap or parity to
AMD hardware, rerun one of:

- the GPU cluster AMD/ROCm with config hash aligned to the H200 baseline (`89c6c6012051`), if
  that older config is still reproducible.
- The current same-node Slurm wrapper/config on an NVIDIA H200 node.
- A paired per-task diff between the H200 Qwen details and the the GPU cluster details to see
  whether the same tasks are solved across both runs and which misses remain.

## Gemma4-26B-A4B-IT (1 run)

| Setting | Value |
|---------|-------|
| Model | `gemma4-26b-a4b-it` (local vLLM, served as `google/gemma-4-26B-A4B-it`) |

| Run | Run ID | Score rate | Total score | Avg verifier pass | Duration |
|-----|--------|------------|-------------|-------------------|----------|
| 1 | `20260612T222210Z-89c6c6012051` | 31.25% | 25/80 | 54.91% | 2.7 min |

Single-run delta vs Qwen 5-run mean: **−7.0 pp** score rate, **−12.2 pp** verifier
pass rate. Confirm with additional runs before drawing firm conclusions.

## Nemotron-3-Nano-30B-A3B (1 run)

| Setting | Value |
|---------|-------|
| Model | `nemotron-3-nano-30b-a3b` (local vLLM, served as `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`) |
| vLLM serve | `nemo3_serve.sh` — `tool-call-parser qwen3_coder`, `reasoning-parser nemotron_v3`, `max-num-seqs 40` |

| Run | Run ID | Score rate | Total score | Avg verifier pass | Duration | Notes |
|-----|--------|------------|-------------|-------------------|----------|-------|
| 1 (fixed) | `20260612T224132Z-89c6c6012051` | 2.50% | 2/80 | 9.98% | 3.3 min | vLLM 0.21 parsers |
| 0 (broken) | `20260612T221751Z-89c6c6012051` | 1.25% | 1/80 | 5.16% | 0.95 min | legacy `nano_v3` plugin; no tool calls |

Single-run delta vs Qwen 5-run mean: **−35.8 pp** score rate. Nemotron Nano is functional
after the serve-script fix but remains much weaker than Qwen/Gemma/Super on this benchmark.

## Nemotron-3-Super-120B-A12B-NVFP4 (5 runs)

| Setting | Value |
|---------|-------|
| Model | `nemotron-3-super-120b-a12b-nvfp4` (local vLLM, served as `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`) |
| vLLM serve | `serve-nemotron-3-super-120b-a12b-nvfp4.sh` — `tool-call-parser qwen3_coder`, `reasoning-parser nemotron_v3`, `moe-backend marlin`, `max-num-seqs 40`, `max-model-len 32768` |
| NVFP4 env | `VLLM_NVFP4_GEMM_BACKEND=marlin` (CUTLASS unsupported on H200) |

| Run | Run ID | Score rate | Total score | Avg verifier pass | Duration |
|-----|--------|------------|-------------|-------------------|----------|
| 1 | `20260613T024804Z-89c6c6012051` | 22.50% | 18/80 | 48.84% | 9.2 min |
| 2 | `20260613T025953Z-89c6c6012051` | 23.75% | 19/80 | 45.59% | 8.8 min |
| 3 | `20260613T031116Z-89c6c6012051` | 17.50% | 14/80 | 40.95% | 7.5 min |
| 4 | `20260613T032145Z-89c6c6012051` | 15.00% | 12/80 | 36.92% | 9.2 min |
| 5 | `20260613T033330Z-89c6c6012051` | 18.75% | 15/80 | 42.21% | 9.4 min |

#### Aggregate (all 5 runs)

- Mean score rate: **19.50%** (σ ≈ 3.6%)
- Range: 15.0% – 23.75%
- Mean tasks passed: **15.6 / 80**
- Mean verifier pass rate: **42.9%**
- Mean eval duration: **~8.8 min/run** (~44 min total eval time; vLLM kept warm after run 1)

5-run mean delta vs Qwen 5-run mean: **−18.8 pp** score rate, **−24.2 pp** verifier
pass rate. Super is clearly below Qwen and likely below Gemma (Gemma single run 31.25%
vs Super mean 19.50%, gap **−11.8 pp** — exceeds the ≥6 pp one-run-vs-mean threshold).

## Interpreting performance gains (Qwen baseline)

Scoring is binary: a task scores **1.0** only when **all** SQL verifiers pass, which is
why the score rate (~38%) sits well below the mean verifier pass rate (~67%). The
per-task results (`detail.json`) show where that gap actually comes from.

### Failure anatomy (Qwen, measured over 5 runs)

Of the 400 task-instances, 247 failed. By number of failing verifiers:

| Failing verifiers | Task-instances | Share of failures |
|-------------------|----------------|-------------------|
| Exactly 1 (near miss) | 113 | **45.7%** |
| 2 | 64 | 25.9% |
| 3 | 42 | 17.0% |
| ≥4 | 28 | 11.3% |

Failing tasks pass **46.8%** of their verifiers on average, and 19% of failures pass
none. Near misses are the largest single bucket, but not the majority of failures.

Per-task stability across the 5 runs:

| Outcome over 5 runs | Tasks |
|---------------------|-------|
| Solved in all 5 | 23 |
| Solved in 1–4 (flaky) | 16 |
| Never solved | 41 |

Three ceilings follow directly:

- **48.75% (39/80)** — variance-only ceiling: solving every flaky task every run while
  the 41 never-solved tasks stay unsolved. Sampling tricks, retries, and reranking
  cannot exceed this without making the policy solve new tasks.
- **11 of the 41 never-solved tasks fail exactly one verifier in all 5 runs** —
  systematic single-step misses, the cheapest capability headroom (~+1.25 pp each,
  up to ~+14 pp).
- **~66.5%** — ceiling if every single-verifier failure were converted to a pass.
  Beyond this requires progress on tasks failing ≥2 verifiers.

### Noise floor (Qwen baseline)

Run-to-run σ ≈ 2.6 pp over the 5 runs (2.8 pp on the current-config runs 2–5). With
only 4–5 runs the σ estimate is itself loose, so treat these ~2-standard-error
thresholds as guides, not exact cutoffs:

| Comparison | Minimum gain to treat as likely real |
|------------|--------------------------------------|
| One run vs one baseline run | **≥7 pp** (~6 tasks) |
| One run vs the 5-run baseline mean | **≥6 pp** |
| 5-run mean vs 5-run baseline mean | **≥3 pp** (~2–3 tasks) |

The more sensitive test is **paired per-task comparison**: the 80 tasks are fixed, so
diff per-task outcomes against the baseline runs and count flips (McNemar or a
bootstrap over tasks). Only 16 tasks are flaky at baseline — a real improvement shows
up as never-solved tasks flipping to solved (or flaky tasks stabilizing), not just as
a small shift in the aggregate rate.

Evaluating at `temperature=0.0` removes sampling noise but changes the operating
point; if used, compare temp-0 against a temp-0 baseline, not against this one.

### Reasonable expected gains by change

Priors, not measurements, except where noted:

| Change | Expected gain | Why |
|--------|---------------|-----|
| Prompt / orchestrator tuning | **+2 to +5 pp** | Cheap to test; addresses tool-use and stopping errors |
| Better agent loop (retry, decomposition) | **+3 to +8 pp** | Helps multi-step planning failures |
| Self-evolve / WM-guided reranking | **+0 to +5 pp** | Internal WM experiments on this benchmark measured +1.4 to +3.1 pp (not significant). Reranking only reorders what the policy already samples; the 41/80 never-solved tasks bound it |
| Stronger model (same setup) | **+5 to +12 pp** | Better reasoning and tool selection |
| Fine-tuning / RL on this slice | **+10 to +20 pp** | Requires aligned training data and eval protocol |

### Practical targets (same model + setup)

| Goal | Target score rate | Notes |
|------|-------------------|-------|
| Likely real, not noise | **≥41–42%** | ~+3 pp on a 5-run mean |
| Meaningful engineering win | **~45%** | ~+7 pp; ~6 more tasks solved |
| Above the variance-only ceiling | **>48.75%** | Requires solving tasks the baseline never solved |
| Strong result for Qwen3.6-27B + react | **~50–55%** | Needs several of the 11 systematic single-verifier tasks fixed |
| Likely needs a model upgrade | **>~65%** | Exceeds the convert-every-near-miss ceiling (~66.5%) |

A realistic medium-term target without changing the base model is **45–50%**: the
variance ceiling (48.75%) plus a few of the 11 systematic near-miss fixes. Above ~65%
exceeds what fixing single-verifier failures alone can deliver and likely needs both a
stronger model and more reliable planning.

## Reproducibility

Prerequisites: Docker (MCP servers), local vLLM with tool-calling support, upstream
`EnterpriseOps-Gym` repo prepared (`ENTERPRISEOPS_GYM_REPO_PATH`), and benchmark envs
installed (`scripts/install.sh enterpriseops-gym`).

Single run (any model):

```bash
cd /path/to/benchmarks

# Set VLLM_MODEL in assets/EnterpriseOps-Gym/run.sh (line ~18), or override before sourcing:
#   qwen3.6-27b | gemma4-26b-a4b-it | nemotron-3-nano-30b-a3b | nemotron-3-super-120b-a12b-nvfp4
export CUDA_VISIBLE_DEVICES=0,1,2,3   # adjust for your GPU layout
export MAX_PARALLEL=40
export ENTERPRISEOPS_LLM_TEMPERATURE=0.7
export ENTERPRISEOPS_LLM_TOP_P=0.95
START_VLLM=1 bash assets/EnterpriseOps-Gym/run.sh eval-opsgym-80
```

Example model switches:

```bash
# Gemma4
# (edit run.sh: VLLM_MODEL=gemma4-26b-a4b-it)
START_VLLM=1 bash assets/EnterpriseOps-Gym/run.sh eval-opsgym-80

# Nemotron-3-Nano
# (edit run.sh: VLLM_MODEL=nemotron-3-nano-30b-a3b)
START_VLLM=1 bash assets/EnterpriseOps-Gym/run.sh eval-opsgym-80

# Nemotron-3-Super NVFP4 (first run starts vLLM; keep warm for batch)
# (edit run.sh: VLLM_MODEL=nemotron-3-super-120b-a12b-nvfp4)
START_VLLM=1 STOP_VLLM_ON_EXIT=0 bash assets/EnterpriseOps-Gym/run.sh eval-opsgym-80
for i in 2 3 4; do START_VLLM=0 STOP_VLLM_ON_EXIT=0 bash assets/EnterpriseOps-Gym/run.sh eval-opsgym-80; done
START_VLLM=0 STOP_VLLM_ON_EXIT=1 bash assets/EnterpriseOps-Gym/run.sh eval-opsgym-80
```

Equivalent `ejepa` invocation:

```bash
ejepa bench run EnterpriseOps-Gym \
  --executor mcp_react \
  --config target=opsgym_80_test \
  --config mode=oracle \
  --config orchestrator=react \
  --config max_iterations=50 \
  --config max_parallel=40 \
  --config capture_trajectory=true
```

Five-run batch (matches this baseline):

```bash
for i in 1 2 3 4 5; do
  echo "=== run $i/5 ==="
  START_VLLM=1 bash assets/EnterpriseOps-Gym/run.sh eval-opsgym-80
done
```

Expected wall-clock time:

- Eval only (vLLM kept warm): **~20–23 min** for 5 runs
- Full `run.sh` with vLLM restart each run: **~30–33 min** for 5 runs

Results land under `${BENCHMARK_HOME}/experiments/` (default:
`.cache/benchmark-home/experiments/`). Each run writes `manifest.json` and `detail.json`.

Single-task smoke:

```bash
TASK_ID=task_20251201_113908_295_e81f6083_e770442c \
  bash assets/EnterpriseOps-Gym/run.sh eval-opsgym-80
```
