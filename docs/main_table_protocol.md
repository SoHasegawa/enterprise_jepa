# Main table (tab:agent): canonical run protocol

Decision (2026-09-18): apart from EnterpriseOps-Gym and the no-world-model baselines,
the existing runs are **not reusable** for the main table. They mix parallelism
(WorkBench alone has mp=1, 3, 10 and 20) and beam settings (horizon 4 / execute 4 vs
horizon 3 / execute 2), and WorkBench success moves ~3 pp with parallelism alone
(0.8188 at mp=1, 0.7928 at mp=3, 0.7855 at mp=10), which is larger than most of the
effects the table reports. Every non-EOPS world-model cell is therefore re-run under
the fixed configuration below.

## Fixed configuration

| setting | value | flag |
|---|---|---|
| agent | Qwen3.6-27B | served as `wm_agent` (:9010) or `Qwen3.6-27B` (:18044/:18045) |
| beam candidates | 8 | `--beam-samples 8` |
| beam horizon | 3 | `--beam-horizon 3` |
| execute steps | 2 | `--beam-execute-steps 2`, `--wm-beam-mpc-execute-steps 2` |
| rollout mode | open loop | `--wm-imagined-rollout-mode open_loop` |
| score margin | 0.10 | `--beam-score-margin 0.10` |
| temperature | 0.7 | `--beam-temperature 0.7` |
| refinement | 1 round, top-k 4 | `--beam-refinement-rounds 1 --beam-refinement-top-k 4` |
| diversity / override | on / no hard override | `--ssot-diversity --no-beam-hard-override` |
| ITP-I depth | k = 4 | `--wm-itp-fixed-k 4` |
| terminal advice | on, threshold 0.75 | `--wm-beam-plan-terminal-advice ...` |
| JEPA checkpoint | `data_jepa_heads_partial_imb_terminal_3`, canonical_event | |
| state-output LLM-WM | `llm_wm_beam_action_terminal_crmarenapro`, served on :9015 | `--wm-llm-ewm-mode llm_canonical_trained --wm-ewm-model world_model` |
| JEPA inference | eager (not `WM_JEPA_COMPILE`) | rank-equivalent, but kept uniform |

## Parallelism, fixed per benchmark and never mixed within a column

| benchmark | tasks | max_parallel | why |
|---|--:|--:|---|
| EnterpriseOps-Gym | 80 | as run (reused) | 5 % churn, no measurable mp effect at mp<=5 |
| WorkBench | 690 | **1** | mp-sensitive (~3 pp); user requirement |
| AutomationBench | 600 | 5 | verified mp-insensitive (operations 31/100 at mp=3 vs 32/100 at mp=5) |
| CRMArena-Pro | 428 | 5 | matches the leaderboard-mode runs; timeout-bound above this |
| Terminal-Bench 2.0 | 89 | 1 | container-per-task, no evidence on mp |

## Cost per run (measured)

| benchmark | baseline | revision | ITP-I | beam search |
|---|--:|--:|--:|--:|
| WorkBench (mp=1) | 3.0 h | 5.3 h | 7.5 h | ~7 h |
| AutomationBench (mp=5) | 2.6 h | 4.5 h | 8.7 h | 4.5-6.7 h |
| CRMArena-Pro (mp=5) | ~5 h | 7.5 h | 11.4 h | 5.1-5.6 h |
| Terminal-Bench (mp=1) | 5.5 h | 7.9 h | 12.5 h | 9.5-10.4 h |

Three agent endpoints are available (:9010 local, :18044, :18045), so three repeats of
one cell run concurrently and a cell costs one run's wall-clock.

## Scope options (repeats = 3 unless stated)

| option | cells re-run | runs | wall-clock on 3 endpoints |
|---|---|--:|--:|
| A. beam only | baseline + JEPA beam + LLM-WM beam, 4 benchmarks | 36 | ~75 h |
| B. beam only, WB+AB | as A but only WorkBench and AutomationBench | 18 | ~30 h |
| C. full block | baseline + 3 harnesses x 2 world models, 4 benchmarks | 84 | ~170 h |
| D. WB only, full block | baseline + 3 harnesses x 2 world models, WorkBench | 21 | ~45 h |

Baselines are reusable in principle but WorkBench's must be re-run anyway: the three
mp=1 baselines used agent alias `wm_agent3`, and although `wm_agent` is verified to be
`Qwen/Qwen3.6-27B`, the alias-to-model mapping of the retired servers cannot be checked.
