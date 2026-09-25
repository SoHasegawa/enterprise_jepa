# Run protocol (paper Table 3)

The configuration every reported agentic run shares. `scripts/run_main_table_repeats.sh`
passes all of it; the values here are the ones to match if you drive the CLI yourself.

## Fixed configuration

| setting | value | flag |
|---|---|---|
| agent | Qwen3.6-27B | OpenAI-compatible endpoint, one per concurrent run |
| beam candidates | 8 | `--beam-samples 8` |
| beam horizon | 3 | `--beam-horizon 3` |
| execute steps | 2 | `--beam-execute-steps 2`, `--wm-beam-mpc-execute-steps 2` |
| rollout mode | open loop | `--wm-imagined-rollout-mode open_loop` |
| score margin | 0.10 | `--beam-score-margin 0.10` |
| temperature | 0.7 | `--beam-temperature 0.7` |
| refinement | 1 round, top-k 4 | `--beam-refinement-rounds 1 --beam-refinement-top-k 4` |
| diversity / override | on / no hard override | `--ssot-diversity --no-beam-hard-override` |
| ITP-I depth | k = 4 | `--wm-itp-fixed-k 4` |
| terminal advice | on, threshold 0.75 | `--wm-beam-plan-terminal-advice --wm-beam-plan-terminal-advice-threshold 0.75` |
| Enterprise-JEPA | `checkpoints/jepa`, canonical_event | `--wm-ewm-jepa-checkpoint`, `--wm-jepa-observation-backend` |
| state-output LLM-WM | served on an OpenAI-compatible endpoint | `--wm-llm-ewm-mode llm_canonical_trained --wm-ewm-model world_model` |
| JEPA inference | eager, not `WM_JEPA_COMPILE` | |
| repeats | 3 per cell | `REPEATS=3` |

## Parallelism, fixed per benchmark and never mixed within a column

| benchmark | tasks | max_parallel |
|---|--:|--:|
| EnterpriseOps-Gym | 80 | 5 |
| CRMArena-Pro | 428 | 3 |
| WorkBench | 690 | 1 |
| AutomationBench | 600 (400 scored) | 5 |
| Terminal-Bench 2.0 | 89 | 1 |

WorkBench is parallelism-sensitive — success moves about 3 points between `max_parallel`
1 and 10, which is larger than most effects the table reports — so a column must never mix
settings.

## Cost per run (measured)

| benchmark | baseline | revision | ITP-I | beam search |
|---|--:|--:|--:|--:|
| WorkBench (mp=1) | 3.0 h | 5.3 h | 7.5 h | ~7 h |
| AutomationBench (mp=5) | 2.6 h | 4.5 h | 8.7 h | 4.5-6.7 h |
| CRMArena-Pro (mp=5) | ~5 h | 7.5 h | 11.4 h | 5.1-5.6 h |
| Terminal-Bench (mp=1) | 5.5 h | 7.9 h | 12.5 h | 9.5-10.4 h |

A full block — baseline plus three harnesses times two world models on four benchmarks,
three repeats — is 84 runs, about 170 hours of wall-clock with three agent endpoints
running concurrently.
