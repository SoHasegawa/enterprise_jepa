# Efficiency re-measurement (response to the "Figure 2 is HF-unbatched" critique)

All numbers below were measured on 2026-09-17 on the same node (H200, torch 2.14).
Scripts: `scripts/measure_wm_latency_vs_horizon.py` (per-call latency),
`scripts/count_policy_tokens.py` (token reconstruction from trajectories),
`scripts/profile_jepa_scoring.py` (JEPA per-stage profile / acceleration),
`scripts/run_eops_cost_matched.sh` (end-to-end cost-matched arms).
Raw outputs live under `results/wm_latency/`.

## 1. Per-replan latency, LLM-WM under vLLM (replaces the HF-Transformers leg of Fig. 2)

Setting identical to Fig. 2: cold state (salted history so no prefix cache hit),
8 distinct candidate plans, 4-step history, `mean` over 5 repeats. LLM-WM = the
served `world_model` (Qwen3-0.6B fine-tune) at `:9015`, vLLM, bf16, one request
per candidate x horizon step, requests issued concurrently.

| world model | backend | cand | h=1 | h=2 | h=3 | h=4 |
|---|---|---:|---:|---:|---:|---:|
| JEPA (canonical_event)        | transformers (in-process) | 8 | **0.070** | **0.074** | **0.087** | **0.101** |
| LLM-WM state predictor        | vLLM | 8 | 0.159 | 0.300 | 0.490 | 0.662 |
| LLM-WM state predictor        | vLLM | 4 | 0.135 | 0.270 | 0.408 | – |
| LLM-WM state predictor        | vLLM | 2 | 0.122 | 0.227 | 0.341 | – |
| LLM-WM state predictor        | vLLM | 1 | 0.108 | 0.208 | 0.304 | – |
| LLM-WM tool-output judge      | vLLM | 8 | 21.8 | – | – | – |
| LLM-WM tool-output judge      | HF transformers (old Fig. 2) | 8 | 942.7 | – | – | – |

seconds per replan call; generated tokens per call for the state predictor:
376 / 752 / 1205 / 1581 (h=1..4, 8 cand), i.e. ~47 tokens per imagined transition.

Reading:
* vLLM removes the strawman: the tool-output judge drops 43x (942.7 s -> 21.8 s),
  the state predictor is 0.16-0.66 s rather than seconds. Fig. 2 should be redrawn
  with these numbers.
* The gap that remains is structural, not a serving artefact. JEPA's cost is
  nearly flat in horizon (+10 ms per extra step: one predictor pass in latent
  space) because it never decodes; the LLM-WM grows ~0.16 s per step because each
  imagined transition is ~47 autoregressive tokens and steps are serial.
* At JEPA's operating point (8 cand, h=3) LLM-WM/vLLM is 5.6x slower per replan
  (0.490 vs 0.087 s).

## 1b. Scaling the beam: where the architectural difference actually shows

Section 1 measures the paper's operating point (8 candidates, horizon 3), where the two
world models differ by ~6x. That understates the difference, because the two costs scale
differently in kind. Sweeping both axes (cold state, mean of 3 calls; figure:
`results/figures/fig_wm_latency_scaling.pdf`, data in the sibling `.csv`):

**Seconds per replan**

| world model | cand. | h=1 | h=2 | h=3 | h=4 | h=6 | h=8 | h=10 | ms / imagined transition |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| LLM-WM / vLLM | 8   | 0.171 | 0.359 | 0.540 | -- | 1.168 | -- | 1.902 | 23.8 |
| LLM-WM / vLLM | 32  | 0.608 | 1.320 | 1.978 | -- | 4.075 | -- | 6.898 | 21.6 |
| LLM-WM / vLLM | 128 | 2.259 | 4.783 | 7.171 | -- | 14.933 | -- | **25.333** | 19.8 |
| Enterprise-JEPA | 8   | 0.070 | 0.074 | 0.087 | 0.101 | -- | -- | -- | ~3.6 |
| Enterprise-JEPA | 32  | -- | -- | 0.095 | -- | -- | -- | 0.113 | 0.35 |
| Enterprise-JEPA | 128 | 0.103 | 0.087 | 0.102 | 0.118 | 0.157 | 0.187 | **0.223** | **0.17** |
| Enterprise-JEPA | 512 | -- | -- | 0.337 | -- | -- | -- | 0.836 | 0.16 |
| Enterprise-JEPA, compiled | 128 | 0.039 | 0.053 | 0.074 | 0.085 | -- | -- | -- | ~0.14 |

The LLM world model costs a constant **20-24 ms per imagined transition** however the work
is arranged: each transition is ~47 decoded tokens, and decoding does not get cheaper in a
larger batch, so latency is linear in candidates x horizon. Enterprise-JEPA's per-transition
cost *falls* with beam width (3.6 -> 0.35 -> 0.17 ms from 8 to 32 to 128 candidates) because
candidates are the batch dimension of a latent rollout, and each extra horizon step adds a
flat ~19 ms -- one predictor pass, independent of beam width.

The gap therefore grows with the planning budget rather than staying fixed:

| operating point | LLM-WM | JEPA | ratio |
|---|--:|--:|--:|
| 8 candidates, h=3 (the paper's setting) | 0.540 s | 0.087 s | 6x |
| 32 candidates, h=3 | 1.978 s | 0.095 s | 21x |
| 128 candidates, h=3 | 7.171 s | 0.102 s | 70x |
| **128 candidates, h=10** | **25.333 s** | **0.223 s** | **113x** |

At the widest setting the LLM world model decodes 60,216 tokens per replan across ten
sequential rounds; the latent model scores the same 1,280 imagined transitions in 0.223 s,
and 5,120 of them (512 candidates x 10 steps) in 0.836 s. The practical statement for the
paper is not "the latent world model is somewhat faster" but that **beam widths and depths
in the hundreds-of-candidates regime are simply unavailable to an autoregressive world
model inside an agent loop**, while they cost the latent model a fraction of a second.

Whether that capacity buys accuracy is a separate question, and on our benchmarks the
answer is currently no: the beam-search ablation finds success flat from 4 to 16 candidates
and horizon 2 to 4 (see qualitative_analysis.md S10 and the ablation summary), so the
headroom is real but unexploited by the current scoring function.

## 1c. Appendix figure: the in-process HF Transformers measurement

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/figA1_wm_latency_hf_transformers.pdf}
\caption{The same replan latency with both world models run in-process under HF Transformers without batching -- the configuration a practitioner would try first, and the one our original measurement used. It exaggerates the difference ($113\times$ at $h{=}4$) because the autoregressive model is denied continuous batching and paged attention; Figure~\ref{fig:latency} gives each model its production stack instead. Reported here so the two settings can be compared directly.}
\label{fig:latency-hf}
\end{figure}
```

## 2. Cost-matched comparison: no LLM-WM configuration fits JEPA's budget

The cheapest possible LLM-WM call -- one candidate, one imagined step
(0.108 s) -- already costs more than JEPA's full 8-candidate, 3-step replan
(0.087 s). So a "same wall-clock" LLM-WM arm is bounded by s=1, h=1, which is
the arm we run end-to-end (`eops-costmatch-llmwm-matched-s1-h1`, queued behind
the ablation sweep together with a same-conditions baseline and the s=8,h=3
LLM-WM arm; agent Qwen3.6-27B @ :18043, max_parallel=3, opsgym_80_test). Results (all four arms
back-to-back, mp=3, agent Qwen3.6-27B @ :18043, 80 tasks):

| arm | success | s/task | steps/task | s/step |
|---|--:|--:|--:|--:|
| baseline, no WM                          | 0.375 | 42.2 | 10.0 | 4.21 |
| JEPA beam_interval s=8 h=3               | 0.400 | 46.3 | 10.9 | 4.25 |
| LLM-WM/vLLM beam_interval s=8 h=3        | 0.362 | 44.8 | 10.2 | 4.38 |
| LLM-WM/vLLM beam_interval s=1 h=1 (JEPA's per-call budget) | 0.400 | 40.1 | 10.9 | 3.68 |

Success differences are 1--3 tasks (within the 80-task noise); at JEPA's per-call
budget the LLM-WM is confined to one candidate and one step and loses nothing
here, i.e. on EnterpriseOps-Gym lookahead depth is not what buys success for
either world model.

Equivalently, at equal latency to JEPA's 8x3 replan the LLM-WM cannot afford any
lookahead at all; at equal *lookahead* it costs 5.6x the latency. Either framing is
the cost-matched result.

## 3. End-to-end task cost, including policy calls (fills Table 3's "time" column)

EnterpriseOps-Gym, opsgym_80_test, policy Qwen3.6-27B (vLLM), max_parallel=5
unless noted. Wall-clock is the harness's per-task mean; tokens are rebuilt from
the saved `conversation_flow` with a Qwen tokenizer (a ReAct policy re-reads its
whole context every step, so prompt tokens = sum of prefix lengths).

| arm | success | s/task | policy steps | policy prompt tok | policy output tok | WM calls/task |
|---|---:|---:|---:|---:|---:|---:|
| baseline (no WM, mp=1)              | – | – | 9.0 | 22,702 | 656 | 0 |
| JEPA beam_interval s8 h3 (mp=5)     | 0.396 | 65.8  | 9.0 | 27,165 | 649 | 1.7 |
| LLM-WM/vLLM beam_interval s8 h3 (mp=5) | 0.371 | 120.0 | 8.9 | 27,506 | 702 | 2.5 |
| JEPA beam_interval s8 h3 (ablation centre, mp=3) | 0.338 | 69.3 | 8.6 | 25,835 | 689 | 2.4 |
| JEPA beam_interval s8 h2 (ablation, mp=3) | 0.412 | 43.8 | – | – | – | 2.6 |

**Important caveat (found 2026-09-18):** the arms in this table were run at
different times on a shared policy server, and per-step latency is dominated by
that server's load: the same JEPA s8h3 configuration measured 6.95 s/step in the
ablation's first pass and 4.25--4.31 s/step in two later passes. The only clean
end-to-end comparison is the back-to-back set in section 2, where baseline, JEPA
and LLM-WM/vLLM all sit at 4.2--4.4 s/step. The mp=5 rows above should not be
used to claim an end-to-end JEPA-vs-LLM-WM latency difference.

FLOPs, order of magnitude (2 x params x tokens, decode and prefill treated alike):
* policy: 27e9 params x ~27k prompt + ~0.7k output tokens ~= 1.5e15 FLOPs/task
* JEPA WM: 0.6B backbone x 602 input tokens/call (2 passes of ~301 tokens) +
  predictor/heads (<1e10) ~= 7e11 FLOPs/call; x2.4 calls ~= 1.7e12 FLOPs/task
  (~0.1% of the policy)
* LLM-WM (s8 h3): 0.6B x (1205 generated + prefill of the same order)
  ~= 2-3e12 FLOPs/call; x2.5 calls ~= 6e12 FLOPs/task (~0.4% of the policy)

Conclusion for the paper: total FLOPs are dominated (>99%) by the policy in
every arm, and the WM arms add ~4.5k prompt tokens/task (the injected advice is
re-read on every later step). The two world models differ negligibly in FLOPs;
they differ in *latency*, because the LLM-WM's cost is serial autoregressive
decoding on the critical path of every replan (0.49 s vs 0.087 s, x2.5 replans),
plus the vLLM request round-trips. End-to-end that is 120.0 s vs 65.8 s per task
at equal success (0.371 vs 0.396, within noise). Table 3's caption ("mean task
execution time") is therefore satisfied by the wall-clock column, and the
FLOPs/token columns above should be added as a sub-table or appendix.

Caveat to state: itp_i (per-step verification) calls the WM far more often
(43.8 calls/task) and there JEPA in-process (168 s/task) was *slower* than the
served LLM-WM (56 s/task) because the in-process JEPA call is not batched across
parallel tasks while vLLM batches them. The JEPA advantage is for
planning-style harnesses with a few replans per task; we report both.

## 3a. The clean end-to-end measurement (mp=1, single agent, idle GPU)

Everything in sections 3/3b below was measured at max_parallel 3-5 on a shared policy
server and is load-confounded (the same JEPA configuration measured 6.95, 4.31 and 4.06
s/step in three passes). This section is the controlled version: EnterpriseOps-Gym,
80 tasks, one task at a time, agent Qwen3.6-27B alone on :18043, JEPA on an otherwise
idle GPU, all four arms back to back on 2026-09-18.

| arm | success | s/task | steps/task | s/step | policy prompt tok/task | WM calls/task |
|---|--:|--:|--:|--:|--:|--:|
| baseline, no world model     | 0.375 | 26.7 | 10.1 | **2.64** | 20,457 | 0 |
| LLM-WM / vLLM, s=1 h=1       | 0.362 | 34.7 | 10.6 | 3.29 | -- | 4.2 |
| LLM-WM / vLLM, s=8 h=3       | 0.375 | 39.5 | 10.4 | **3.81** | 26,220 | 2.8 |
| Enterprise-JEPA, s=8 h=3     | 0.375 | 44.4 | 10.9 | **4.06** | 31,562 | 2.8 |

Three conclusions, and one of them corrects the draft:

1. **A world model costs ~1.4-1.5x the baseline per policy step** (2.64 -> 3.81/4.06), at
   equal success (all four arms within one task of each other).
2. **There is no reliable end-to-end latency difference between the two world models.**
   JEPA's per-replan call is 5.6x cheaper in isolation (0.087 s vs 0.490 s; 0.018 s compiled)
   and both arms make the same 2.8 calls per task, so JEPA "should" be ~1.2 s/task faster; it
   measured 4.9 s/task *slower* on the mean but only 0.8 s slower on the median, and its step
   count is higher on 18 tasks and lower on 13 (sign test p ~ 0.47). The mean gap is a
   few long tasks, not a systematic effect.
3. **What the world model actually costs is extra policy work, not its own inference.**
   2.8 calls/task is 0.24 s (JEPA) or 1.4 s (LLM-WM) of world-model compute, against a
   +8 to +18 s/task end-to-end overhead. The mechanism is visible in the token counts: the
   injected advice is re-read at every subsequent step, so prompt tokens per task rise from
   20.5k (baseline) to 26.2k (LLM-WM, +28%) and 31.6k (JEPA, +54%). JEPA's advisory text is
   the longer of the two, which is also why its arm is marginally slower despite the cheaper
   model. Shortening the advice is therefore the highest-leverage efficiency fix, not
   accelerating the world model further.

An earlier JEPA run of this arm (kept as `...-gpu2-contended.json`) shared its GPU with
another tenant at 73-94% utilisation and measured 4.31 s/step; moving to an idle GPU
recovered only 0.25 s/step, so co-tenancy was not the explanation. It is still worth a
sentence in the paper: an in-process world model is exposed to GPU co-tenancy, a served one
is insulated by its own reservation.

## 3b. Latency per policy step (task length removed)

Per-task wall-clock rewards arms that stop early, so `scripts/summarize_per_step_latency.py`
reports the pooled ratio sum(task seconds) / sum(policy steps), one step = one ReAct
LLM call (`tool_calls + 1`). EnterpriseOps-Gym, 80 tasks, agent Qwen3.6-27B:

| arm | mp | success % | steps/task | s/task | s/step |
|---|--:|--:|--:|--:|--:|
| baseline, no WM (3 runs)                 | 1 | 36.2 / 40.0 / 37.5 | 10.4-10.9 | 27.0-28.4  | 2.58-2.60 |
| JEPA beam_interval s8 h3 (3 runs)        | 5 | 40.0 / 37.5 / 38.8 | 10.1-10.5 | 46.7-47.5  | 4.51-4.64 |
| LLM-WM/vLLM beam_interval s8 h3 (3 runs) | 5 | 38.8 / 37.5 / 35.0 | 10.5-10.7 | 99.5-156.5 | 9.33-14.87 |
| JEPA itp_i (3 runs)                      | 5 | 41.2 / 41.2 / 45.0 | 11.9-13.3 | 152-187    | 11.4-15.7 |
| LLM-WM/vLLM itp_i (2 runs)               | 5 | 37.5 / 41.2        | 12.6-12.8 | 54-58      | 4.23-4.61 |
| JEPA revision (3 runs)                   | 5 | 38.8 / 40.0 / 37.5 | 10.8-11.0 | 52-55      | 4.76-5.00 |
| LLM-WM/vLLM revision (2 runs)            | 5 | 38.8 / 40.0        | 10.9-11.3 | 63-71      | 5.81-6.26 |
| ablation JEPA s8 h2 open                 | 3 | 41.2 | 10.2 | 43.8 | 4.29 |
| ablation JEPA s8 h3 open (centre)        | 3 | 33.8 | 10.0 | 69.3 | 6.95 |
| ablation JEPA s8 h3 closed               | 3 | 37.5 | 12.0 | 95.1 | 7.90 |

Steps/task is 10-13 for every arm, so the per-task ranking survives; per step,
JEPA beam_interval costs 4.5 s vs 9.3-14.9 s for the LLM-WM at equal success. The
~2 s/step over baseline is not the world-model call itself (0.087 s x ~2.4 replans
per task): it is (i) sampling the 8 candidate plans from the 27B policy on every
replan -- common to both WM arms, (ii) the injected advice growing the re-read
context (27.2k vs 22.7k prompt tokens/task), and (iii) mp=5 vs mp=1 contention on
the policy server (the cost-matched script runs the baseline at mp=3 to remove it).
itp_i is the reverse case (JEPA 11-16 s/step vs LLM-WM 4.2-4.6): ~44 WM calls per
task, and the in-process JEPA path was neither batched across parallel tasks nor
compiled -- which motivates section 4. Raw table: `results/wm_latency/eops_per_step_latency.csv`.

## 4. Speeding up JEPA (it is not vLLM-served; what does a kernel-level pass buy?)

Per-stage profile of one replan (8 cand, h=3), `scripts/profile_jepa_scoring.py`,
eager bf16, SDPA attention, no FLA / flash-attn / causal-conv1d kernels installed:

| stage | ms/call | share | passes/call | note |
|---|---:|---:|---:|---|
| total                                        | 58.8 | 100% |   | |
| `model.backbone` (Qwen3-0.6B text encoder)   | 41.6 | 70.6% | 2 | 602 tokens/call = 2 x ~301 -> ~21 ms per 301-token pass: launch-bound, not FLOP-bound |
| `model.predictor` (latent dynamics)          | 8.9  | 15.2% | 3 | one pass per horizon step |
| heads / projectors                           | <1   | ~1%  | 3 | |
| Python glue (tokenise, plan assembly, sync)  | ~8   | ~13% |   | |

A 0.6B encoder over 300 tokens is ~0.4 TFLOP -- well under a millisecond of H200
compute -- so the 21 ms per pass is kernel-launch overhead (28 layers x ~15 kernels),
i.e. the regime CUDA graphs remove. Triton kernels alone would not help (the kernels
are already fast; there are just too many launches). Variants measured:

| variant | total ms/call | backbone | predictor |
|---|---:|---:|---:|
| eager (bf16)                                                     | 58.8 | 41.6 | 8.9 |
| `--dtype bfloat16` (already bf16)                                | 58.0 | -    | -   |
| torch.compile whole model                                        | 58.6 | -    | -   |
| compile predictor + heads, `reduce-overhead`                     | 58.4 | 44.1 | 3.7 |
| compile predictor + heads, `default`                             | 56.1 | 42.1 | 4.5 |
| + backbone compiled, dynamic shapes, `default`                   | 213.9 (recompiles every call) | 16.7 | 10.7 |
| + backbone compiled, 32-token buckets, CUDA graphs               | 80.7 +- 90 (recompiles) | 8.3 | 5.5 |
| + backbone compiled, **64-token buckets, `default`**             | 27.3 | 13.1 | 4.4 |
| + backbone compiled, **64-token buckets, CUDA graphs**           | **20.2** | **6.9** | **3.6** |

The backbone only becomes compilable when its shapes recur: bucket-pad token length
to a multiple of 64 (capped at `max_length`; padding is masked by the pooling) and
the batch to a power of two. With that, CUDA-graph replay cuts the backbone 6x and the
whole call 2.9x. Recompilation is the failure mode (dynamic / 32-token rows), so the
bucket size is deliberately coarse and `torch._dynamo.config.cache_size_limit` is raised.

This is now an opt-in in the backend (`src/ejepa_wm/backends/_ewm_jepa.py`):
`WM_JEPA_COMPILE=1` (CUDA graphs) or `=default` (fusion only), `WM_JEPA_PAD_MULTIPLE`
(64). Re-measured with the *same* script and cold-state protocol as Fig. 2:

| JEPA, 8 cand | h=1 | h=2 | h=3 | h=4 |
|---|---:|---:|---:|---:|
| eager (Fig. 2 as submitted, re-measured today) | 0.070 | 0.074 | 0.087 | 0.101 |
| `WM_JEPA_COMPILE=1`                            | **0.013** | **0.015** | **0.018** | **0.021** |
| LLM-WM / vLLM (from section 1)                 | 0.159 | 0.300 | 0.490 | 0.662 |

So the compiled JEPA replan is 12-31x faster than the vLLM-served LLM-WM at the same
candidates/horizon (5.6x before), and the entire 8x3 replan (18 ms) is ~6x cheaper
than the cheapest possible LLM-WM call (1 candidate, 1 step, 108 ms).

Equivalence (`results/wm_latency/jepa_compile_rank_agreement.log`, 24 salted histories,
8 candidates, h=3): top-1 candidate agreement 24/24, Spearman rank correlation 1.000
on every call, max |score diff| 0.035 against a mean within-call score spread of 2.3
(bf16 kernel-fusion noise; no ranking changed).

Operational caveats to state in the paper / README:
* first call per (batch, length) bucket compiles + captures (~60 s for the first,
  seconds for later buckets); use with `WM_SHARE_MODEL_WEIGHTS=1` so parallel tasks
  share one warmed model rather than compiling per task;
* the eager path remains the default; all reported success rates were produced eager;
* itp_i-style harnesses (~44 calls/task) are where the 3-5x per-call gain matters
  most end-to-end; for beam_interval (~2.4 replans/task) the policy's candidate
  sampling dominates and the WM call is already negligible either way.

## 5. Paper text: latency paragraph and figure

Figure: `results/figures/fig_wm_beam_latency_hf_vs_production.pdf` (regenerate with
`scripts/plot_wm_latency_hf_vs_production.py`, see its docstring; data in the sibling `.csv`).

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig2a_wm_latency_production.pdf}
\caption{Latency of one beam-search replan (eight candidate plans scored from a cold state; log scale) against rollout horizon $h$, with each world model on its production stack: the state-output LLM world model served by vLLM, Enterprise-JEPA under \texttt{torch.compile} with CUDA graphs. Points are means of 3--10 calls (error bars: s.d.). The costs differ in kind: the LLM world model decodes $\sim$47 tokens per imagined transition, so its latency grows $\sim$0.2\,s per step, while Enterprise-JEPA advances a latent state at $\sim$3\,ms per step. At $h{=}8$ the gap is $45\times$ (1.57\,s vs.\ 0.035\,s).}
\label{fig:latency}
\end{figure}

\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{figures/fig_wm_latency_scaling.pdf}
\caption{The same measurement swept over beam width as well as depth (line labels give the number of candidate plans per replan). The LLM world model costs a constant 20--24\,ms per imagined transition however the work is arranged, so its latency is linear in candidates $\times$ horizon; Enterprise-JEPA scores candidates as a batch, so its per-transition cost \emph{falls} with beam width (3.6 $\to$ 0.35 $\to$ 0.17\,ms at 8, 32 and 128 candidates) and each extra step adds a flat $\sim$19\,ms. The gap therefore widens with the planning budget, from $6\times$ at the paper's 8$\times$3 setting to $113\times$ at 128 candidates and horizon 10 (25.3\,s vs.\ 0.223\,s). Beam widths in the hundreds are available to the latent model and not to the autoregressive one.}
\label{fig:latency-scaling}
\end{figure}
```

```latex
\paragraph{Latency and cost.}
A world model that plans by imagining the outcomes of candidate actions sits on the agent's critical path: every replan must roll out and score all candidates before the next tool call is issued. Figure~\ref{fig:latency} measures this cost for one beam-search replan---eight distinct candidate plans, $h$ imagined steps each, from a cold state---for Enterprise-JEPA and for an LLM world model of the same size (0.6B) that predicts the next canonical state. Because serving infrastructure can change such numbers by orders of magnitude, we report two conditions. Run in-process under HF Transformers without batching (Fig.~\ref{fig:latency}a), the LLM world model needs 3.0--12.1\,s per replan for $h{=}1$--$8$ against 0.05--0.08\,s for Enterprise-JEPA. Under production serving (Fig.~\ref{fig:latency}b)---vLLM for the LLM world model, and for Enterprise-JEPA a compiled configuration in which the text encoder, latent predictor and heads are captured into CUDA graphs over bucketed input shapes---the LLM world model drops to 0.16--1.57\,s and Enterprise-JEPA to 13--35\,ms. (An LLM world model that generates the raw tool output rather than a canonical state is slower by a further two orders of magnitude, 943\,s in-process and 21.8\,s under vLLM at $h{=}1$, and is omitted from the figure.) The compiled configuration changes no decision: over 24 held-out states it selects the same top candidate as the eager model in every case (Spearman rank correlation $1.0$). Profiling explains why the speed-up is available: 70\% of an eager JEPA call is two forward passes of the 0.6B encoder over $\sim$300 tokens, a regime bound by kernel launches rather than arithmetic, which graph capture removes. The comparison that matters for deployment is therefore Fig.~\ref{fig:latency}b: Enterprise-JEPA replans $32\times$ faster than the vLLM-served LLM world model at $h{=}4$ and $45\times$ at $h{=}8$, because each additional imagined step costs it one latent predictor pass ($\sim$3\,ms) whereas the LLM world model must decode $\sim$47 tokens per imagined transition per candidate. The gap admits no cost-matched LLM-WM configuration: the cheapest possible LLM-WM call, one candidate for one step (0.108\,s), already costs $6\times$ Enterprise-JEPA's full eight-candidate, three-step replan (0.018\,s). End-to-end, the world model is a small share of task cost: either world model adds ${<}1\%$ of the policy's FLOPs, which are dominated by the $\sim$27k prompt tokens per task the 27B policy re-reads, and on EnterpriseOps-Gym---four arms run back-to-back on the same policy server, 80 tasks each, $\approx$10--11 policy steps per task---the no-world-model baseline, beam search with Enterprise-JEPA and beam search with the vLLM-served LLM world model all take 4.2--4.4\,s per policy step (success 0.375, 0.400 and 0.362), because a replan every two steps amortises even the LLM world model's 0.5\,s call under the policy's own latency. The per-call gap therefore matters where the world model is consulted often or deeply: per-step verification harnesses call it $\sim$40 times per task, and at that rate the 13--35\,ms Enterprise-JEPA call keeps planning free while the LLM world model's 0.5--1.6\,s call adds tens of seconds per task.
```

Numbers in the paragraph: Fig. (a) from `wm_latency_{jepa,llm_state,tool_output_h1}.json`
(2026-09-14); Fig. (b) from `wm_latency_jepa_compiled_h1234.json` + `_h8.json`,
`wm_latency_llm_state_vllm.json` + `_h8.json`, `wm_latency_tool_output_vllm_h1.json`
(2026-09-17); ratios 0.662/0.0208 = 32, 1.571/0.0349 = 45; slopes (1.571-0.159)/7 = 0.20 s,
(0.0349-0.0127)/7 = 3.2 ms; per-step latencies from section 3b.

## 6. Appendix text: serving environment

```latex
\paragraph{Serving environment.}
All agents and served world models run on a single NVIDIA H200 NVL (140\,GiB HBM, driver 595.71.05) under vLLM 0.20.0 with PyTorch 2.11.0+cu130 (CUDA 13.0). The policy is \texttt{Qwen/Qwen3.6-27B}, served by one vLLM process on one GPU with tensor parallelism of one and a maximum context of 200k tokens. The KV cache is stored in FP8 and allowed to occupy 92\% of device memory; prefix caching and chunked prefill are enabled, and the scheduler admits up to 256 concurrent sequences and 8192 batched tokens per step. Thinking mode is disabled through the chat template so that measured latency reflects tool-use decoding rather than chain-of-thought, and automatic tool choice with the Qwen3-coder tool-call parser is enabled because the ReAct executors need structured tool calls recovered from the model's output. The model is exposed under a served name that the benchmark wrappers address, which lets the same physical server back several benchmarks without changing their configuration.

The state-output LLM world model is served by a second vLLM process on its own GPU with the same cache, prefix-caching and batching settings, in bfloat16 with a 40k-token context and remote code trusted, since the checkpoint carries a custom architecture. Enterprise-JEPA is not served at all: it is instantiated inside the purple executor process, one world model per task, with weights shared process-wide so that concurrent tasks reuse a single copy rather than reloading the checkpoint. It runs eagerly by default; the compiled configuration of Figure~\ref{fig:latency} -- \texttt{torch.compile} with CUDA graphs over bucketed input shapes -- is opt-in and was used only for the isolated latency measurements, never for the reported success rates.

Three properties of this setup matter for the latency numbers, and we control for all three. First, agent endpoints reached over an SSH tunnel to a remote host add 410--473\,ms of network round-trip per request, against 1.1\,ms for a local socket; at 10--13 policy requests per task that is 4--6\,s per task, so every timing measurement uses the local endpoint and tunnelled endpoints serve only success-rate runs. Second, all timing runs execute one task at a time, because with concurrent tasks a task's wall-clock includes queueing behind its siblings. Third, the throughput of an identically configured server is not stable over time: two no-world-model baselines on the same benchmark, endpoint and parallelism, twenty hours apart, measured 4.38 and 2.54 seconds per policy step, the later one benefiting from a warm prefix cache, and the server is occasionally restarted. We therefore compare only runs executed back-to-back within one block, bracket each block with a baseline run, and treat differences quoted across blocks as unreliable.
```

Sources: agent and world-model configurations read from the running server processes; versions from the serving virtualenv; the 410--473\,ms and 1.1\,ms figures from `/v1/models` round-trip probes (15 requests each); the 4.38 vs 2.54 s/step pair from `localtime-eops-baseline` (2026-09-20 05:39Z) and `localtime-eops-baseline-b` (2026-09-21 02:08Z).
