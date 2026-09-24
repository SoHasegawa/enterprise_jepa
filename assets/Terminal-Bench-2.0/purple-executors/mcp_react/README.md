# `mcp_react` — WM-guided shell executor (Terminal-Bench-2.0)

Terminal-Bench analogue of EnterpriseOps-Gym's [`mcp_react`](../../../EnterpriseOps-Gym/purple-executors/mcp_react)
and CRMArenaPro's [`mcp_react`](../../../crmarenapro/purple-executors/mcp_react): a shell
agent with an optional, pluggable **World Model (WM)** decision hook backed by the shared
[`src/ejepa_wm`](../../../../src/ejepa_wm) package.

The gym variant reaches tools over **MCP servers**. Terminal-Bench has no MCP layer, so this
executor **reuses the sibling [`llm_shell`](../llm_shell) shell-command tool loop** verbatim
(its `terminal-bench-shell-v1` protocol, LLM client plumbing, session/prompt) and only adds
the `ejepa_wm` world-model hook at its single decision point (`_next_action`).

`llm_shell` runs a **per-step A2A protocol** (the green agent executes each command and
returns the `exec_result`), so the `conversation_flow` `ejepa_wm` consumes is reconstructed
from `session.history` on every step.

> **Scope / caveat.** The bundled EWM world model is fine-tuned on **EnterpriseOps-Gym**
> tool outcomes. Scoring shell commands with it is an intentional **out-of-domain over-fit
> probe** — expect it to help the gym and *not* Terminal-Bench. This executor exists to make
> that comparison runnable, not to improve Terminal-Bench accuracy.

## Behavior

| `WM_STRATEGY` | Effect |
|---|---|
| unset / `none` | Identical to `llm_shell` (no WM). Always runnable without an EWM server. |
| `selection` | Sample `WM_N` candidate commands per step; the WM ranks them and the best is committed. Pairs with `WM_BACKEND=ewm_predict`. |
| `prompt_injection` (alias `imagined`) | The WM `advise`s and its guidance block is injected before the next command. Pairs with `WM_BACKEND=ewm_imagined` / `llm`. |
| `beam_plan` | JEPA MPC lookahead. Requires `WM_BACKEND=ewm_imagined` **and** a JEPA canonical-event checkpoint (`--wm-ewm-jepa-checkpoint`). Every `WM_BEAM_MPC_EXECUTE_STEPS` steps the WM proposes `m` candidate commands per horizon step, scores the imagined outcomes with the canonical-event heads (LLM-free) and can override the agent's baseline command when it beats it by more than `WM_BEAM_PLAN_SCORE_MARGIN`. Degrades to plain `llm_shell` with any other backend. |
| `hier_latent_cem` | A second JEPA-only MPC lookahead. Same requirements as `beam_plan` (`WM_BACKEND=ewm_imagined` + JEPA canonical-event checkpoint) and the same per-step cadence/advisory contract (shares `WM_BEAM_MPC_EXECUTE_STEPS` / `WM_BEAM_PLAN_HARD_OVERRIDE`), but plans via **hierarchical latent-action CEM** instead of discrete LLM-proposed candidates per horizon step (`WM_HIER_CEM_*` knobs). Degrades to plain `llm_shell` with any other backend. |

The WM only sees the generic `conversation_flow` (`system_message` / `user_message` /
`ai_message{tool_calls=[run_shell]}` / `tool_result{exec_result}`).

> Candidate diversity for `selection` needs sampling temperature > 0. For non-GPT-5 models
> this executor raises it automatically (≥0.7) while sampling; GPT-5/reasoning models ignore
> `temperature`, so candidates may be identical and selection degrades to the first one.

## Env vars

Agent LLM (same as `llm_shell`): `AZURE_OPENAI_*`, `OPENAI_API_KEY` / `TERMINAL_BENCH_LLM_*`
/ `LLM_*`, `TERMINAL_BENCH_MAX_STEPS`, `TERMINAL_BENCH_LLM_TEMPERATURE`.

World model (read by `ejepa_wm`):

| Var | Meaning |
|---|---|
| `WM_STRATEGY` | `none` (default) \| `selection` \| `prompt_injection` (\| `imagined`) \| `beam_plan` \| `hier_latent_cem` |
| `WM_BACKEND` | `noop` \| `ewm_predict` \| `ewm_imagined` \| `llm` \| `served` |
| `WM_N` | candidates sampled per step (selection); default 1 |
| `WM_STATE` | `binary_error` (default) \| `binary_error_stage` \| `tool_output` |
| `WM_EWM_MCP_URL` | route EWM generation through the shared EWM MCP server (e.g. `http://127.0.0.1:12072`) |
| `WM_EWM_MODEL` | EWM model id (default `gymops_world_model`) |
| `WM_VLLM_BASE_URL` / `WM_VLLM_SERVER_PORT` | in-process vLLM EWM endpoint if not using MCP |
| `WM_IMAGINED_MAX_STEPS`, `ACTION_OPTIMIZER`, `WM_IMAGINED_TOP_K`, `WM_IMAGINED_CANDIDATE_ACTIONS` | imagined-rollout knobs (`ewm_imagined` only) |
| `WM_IMAGINED_ROLLOUTS` (`--wm-imagined-rollouts`), `WM_IMAGINED_SELECTION` (`--wm-imagined-selection` = `first` \| `llm_judge`) | decoded multi-rollout + LLM-judge selection (seq2seq JEPA; see below) |
| `WM_EWM_JEPA_CHECKPOINT` | JEPA canonical-event checkpoint dir (required for `beam_plan`; `--wm-ewm-jepa-checkpoint`) |
| `WM_BEAM_PLAN_SAMPLES`, `WM_BEAM_PLAN_HORIZON`, `WM_BEAM_MPC_EXECUTE_STEPS`, `WM_BEAM_PLAN_SCORE_MARGIN`, `WM_BEAM_PLAN_DIVERSITY_MULTIPLIER` | `beam_plan` MPC knobs (`--wm-beam-plan-samples` / `--wm-beam-plan-horizon` / `--wm-beam-mpc-execute-steps` / `--wm-beam-plan-score-margin` / `--wm-beam-plan-diversity-multiplier`) |
| `WM_BEAM_PLAN_HARD_OVERRIDE` | `beam_plan`: force the beam's confident recommendation over the agent's command (`--wm-beam-plan-hard-override`). Default off = **advisory** (agent acts; the confidence-gated plan is injected only as guidance) |
| `WM_JEPA_MERGE_CHECKPOINT`, `WM_BEAM_PLAN_DECODE_TOOL_OUTPUT`, `WM_BEAM_PLAN_DECODE_MAX_NEW_TOKENS` | merge in a decoder from another checkpoint and decode `beam_plan`'s predicted **tool output** text (not just labels) for the winning trajectory — see `src/ejepa_wm/README.md` |
| `WM_HIER_CEM_ANCHORS`, `WM_HIER_CEM_SAMPLES`, `WM_HIER_CEM_ELITES`, `WM_HIER_CEM_ITERS`, `WM_HIER_CEM_HORIZON`, `WM_HIER_CEM_INIT_STD`, `WM_HIER_CEM_MIN_STD`, `WM_HIER_CEM_SMOOTHING`, `WM_HIER_CEM_MIN_ELITE_AGREEMENT`, `WM_HIER_CEM_DECODE_STRATEGY`, `WM_HIER_CEM_DECODE_MAX_NEW_TOKENS` | `hier_latent_cem` hierarchical latent-action CEM knobs (`--wm-hier-cem-anchors` / `-samples` / `-elites` / `-iters` / `-horizon` / `-init-std` / `-min-std` / `-smoothing` / `-min-elite-agreement` / `-decode-strategy` / `-decode-max-new-tokens`). Shares `WM_BEAM_MPC_EXECUTE_STEPS` (`--wm-beam-mpc-execute-steps`) for planning cadence and `WM_BEAM_PLAN_HARD_OVERRIDE` (`--wm-beam-plan-hard-override`) for the advisory/hard-override toggle with `beam_plan` |

The EWM server (vLLM + MCP) is the **same** one EnterpriseOps-Gym uses — bring it up with
`assets/EnterpriseOps-Gym/build_mcp.sh`, then point `WM_EWM_MCP_URL` at it.

`beam_plan` is **JEPA-only** MPC lookahead: it needs `WM_STRATEGY=beam_plan` (which
auto-selects the `ewm_imagined` backend) together with a JEPA canonical-event checkpoint via
`--wm-ewm-jepa-checkpoint` (`WM_EWM_JEPA_CHECKPOINT`). Without a JEPA checkpoint that exposes
the canonical-event scoring heads the executor degrades to the plain shell agent.

`hier_latent_cem` is a second **JEPA-only** MPC lookahead with the same requirements
(`WM_STRATEGY=hier_latent_cem` auto-selects `ewm_imagined`; needs the same JEPA canonical-event
checkpoint) and the same degrade-to-plain-shell-agent fallback. It differs from `beam_plan`
only in how it produces the imagined trajectory internally: hierarchical latent-action CEM
(`WM_HIER_CEM_*`) instead of discrete LLM-proposed candidates per horizon step.

## Run

```bash
# plain shell agent (no WM) — sanity check
ejepa bench run Terminal-Bench-2.0 --executor mcp_react --config target=sample

# EWM-guided command selection over the shared EWM MCP server
export WM_STRATEGY=selection WM_BACKEND=ewm_predict WM_N=3 \
       WM_STATE=binary_error WM_EWM_MCP_URL=http://127.0.0.1:12072 \
       WM_EWM_MODEL=gymops_world_model
ejepa bench run Terminal-Bench-2.0 --executor mcp_react --config target=sample
```

## Files

- `executor.py` — A2A executor; `build_executor()` returns `WmShellExecutor`.
- `wm_react.py` — `WmShellExecutor` (subclasses `llm_shell`'s `LlmShellExecutor`); builds the
  WM from env and overrides `_next_action` to sample/rank or inject per step, rebuilding
  `conversation_flow` from `session.history`.
- `version.toml` — executor version.

## Decoded rollouts + LLM judge (seq2seq JEPA) — for terminal tasks

An alternative to the canonical-event `beam_plan` scoring, aimed at shell/terminal tasks
where the enterprise-trained classification heads don't discriminate commands well. Use an
**encoder-decoder (seq2seq) JEPA** checkpoint (e.g. `t5gemma`, trained on the broad
tool-use/SWE/OS corpus): the world model **decodes the predicted state as text** instead of
classifying it (no scoring heads). The executor generates several imagined rollouts of
`(action -> decoded state)` pairs and an **LLM-as-judge reranks them** to pick the trajectory
to inject — no world-model scoring is involved.

This is the `imagined` / `prompt_injection` strategy with `WM_IMAGINED_SELECTION=llm_judge`
and a seq2seq JEPA whose observation backend resolves to `decoder`:

```bash
export WM_EWM_JEPA_CHECKPOINT=checkpoints/data_jepa_all_adp06_grd_gemma
ejepa bench run <BENCHMARK> --executor mcp_react \
  --wm-strategy imagined \
  --wm-ewm-jepa-checkpoint "$WM_EWM_JEPA_CHECKPOINT" \
  --wm-jepa-observation-backend decoder \
  --wm-imagined-max-steps 3 \
  --wm-imagined-rollouts 4 --wm-imagined-selection llm_judge \
  --config target=sample
```

`--wm-imagined-rollouts N` runs N rollouts (the first deterministic, the rest at
`--wm-imagined-temperature`); `--wm-imagined-selection llm_judge` has the policy LLM pick the
best rollout by its action->state pairs (falls back to a heuristic if the judge output can't be
parsed). `first` keeps the deterministic rollout (no judge call). The chosen rollout is injected
as `[IMAGINED_TRAJECTORY_FOR_PLANNING_ONLY]` guidance; the agent's own action still executes.
