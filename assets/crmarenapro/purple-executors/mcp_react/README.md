# `mcp_react` — WM-guided ReAct executor (CRMArenaPro)

CRMArenaPro analogue of EnterpriseOps-Gym's [`mcp_react`](../../../EnterpriseOps-Gym/purple-executors/mcp_react)
executor: a ReAct agent with an optional, pluggable **World Model (WM)** decision hook
backed by the shared [`src/ejepa_wm`](../../../../src/ejepa_wm) package.

The gym variant reaches tools over **MCP servers**. CRMArenaPro has no MCP layer, so this
executor reuses the **native CRM-SQL tool loop** from the sibling
[`baseline_crm_agent`](../baseline_crm_agent) executor (`CRMDatabase`, `SYSTEM_PROMPT`,
the `<execute>`/`<describe>`/`<respond>` ReAct contract, the provider clients) and only
adds the `ejepa_wm` world-model hook per turn (`wm_react.WmReactAgent`).

> **Scope / caveat.** The bundled EWM world model is fine-tuned on **EnterpriseOps-Gym**
> tool outcomes. Pointing it at CRM-SQL actions is an intentional **out-of-domain
> over-fit probe** — expect it to help the gym and *not* CRMArenaPro. This executor exists
> to make that comparison runnable, not to improve CRMArenaPro accuracy.

## Behavior

| `WM_STRATEGY` | Effect |
|---|---|
| unset / `none` | Identical to `baseline_crm_agent` (no WM). The executor is always runnable without an EWM server. |
| `selection` | Sample `WM_N` candidate actions per turn; the WM ranks them and the best is committed. Pairs with `WM_BACKEND=ewm_predict`. |
| `prompt_injection` (alias `imagined`) | The WM `advise`s and its guidance block is injected before the next action. Pairs with `WM_BACKEND=ewm_imagined` or `llm`. |
| `beam_plan` | JEPA MPC lookahead. Every `WM_BEAM_MPC_EXECUTE_STEPS` turns the WM proposes `m` candidate actions per horizon step, scores the imagined outcomes with the JEPA canonical-event heads (LLM-free), and shows the agent the best imagined plan to follow; the step-0 action overrides the agent's baseline only when it beats it by more than `WM_BEAM_PLAN_SCORE_MARGIN`. **Requires** `WM_STRATEGY=beam_plan` + a JEPA canonical-event checkpoint (`--wm-ewm-jepa-checkpoint`, which auto-selects `WM_BACKEND=ewm_imagined`); with any non-JEPA backend it degrades to plain ReAct. |
| `hier_latent_cem` | A second JEPA-only MPC lookahead with the same requirements and per-turn cadence/advisory contract as `beam_plan` (shares `WM_BEAM_MPC_EXECUTE_STEPS` / `WM_BEAM_PLAN_HARD_OVERRIDE`), but instead of scoring discrete LLM-proposed candidates it opens the search with a handful of LLM-proposed anchor actions, then CEM-refines thousands of continuous latent-action trajectories around them (LLM-free, decode-free rollout) and only decodes the converged best plan (`WM_HIER_CEM_*` knobs). **Requires** `WM_STRATEGY=hier_latent_cem` + the same JEPA canonical-event checkpoint (auto-selects `WM_BACKEND=ewm_imagined`); with any non-JEPA backend it degrades to plain ReAct. |

The WM only ever sees the generic `conversation_flow` (`system_message` / `user_message` /
`ai_message{content,tool_calls}` / `tool_result{tool_name,result}`), so no benchmark
internals leak into `ejepa_wm`.

## Env vars

Agent LLM (same as `baseline_crm_agent`): `LLM_PROVIDER`, `LLM_MODEL`, `LLM_BASE_URL`,
`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `AZURE_OPENAI_*` / `NEBIUS_API_KEY`, `MAX_TURNS`,
`TEMPERATURE`. CRM DB: `CRMARENAPRO_DB_PATH` / `CRMARENAPRO_DB_DIR` (defaults to
`baseline_crm_agent/data/`).

World model (read by `ejepa_wm`, see `src/ejepa_wm`):

| Var | Meaning |
|---|---|
| `WM_STRATEGY` | `none` (default) \| `selection` \| `prompt_injection` (\| `imagined`) \| `beam_plan` \| `hier_latent_cem` |
| `WM_BACKEND` | `noop` \| `ewm_predict` \| `ewm_imagined` \| `llm` \| `served` |
| `WM_N` | candidates sampled per turn (selection); default 1 |
| `WM_STATE` | EWM output mode: `binary_error` (default) \| `binary_error_stage` \| `tool_output` |
| `WM_EWM_MCP_URL` | route EWM generation through the shared EWM MCP server (e.g. `http://127.0.0.1:12072`) |
| `WM_EWM_MODEL` | EWM model id (default `gymops_world_model`) |
| `WM_VLLM_BASE_URL` / `WM_VLLM_SERVER_PORT` | in-process vLLM EWM endpoint if not using MCP |
| `WM_IMAGINED_MAX_STEPS`, `ACTION_OPTIMIZER`, `WM_IMAGINED_TOP_K`, `WM_IMAGINED_CANDIDATE_ACTIONS` | imagined-rollout knobs (`ewm_imagined` only) |
| `WM_EWM_JEPA_CHECKPOINT` | JEPA canonical-event checkpoint dir; required for `beam_plan` (`--wm-ewm-jepa-checkpoint`) |
| `WM_BEAM_PLAN_SAMPLES` | `beam_plan`: `m` candidate next-actions proposed per horizon step (`--wm-beam-plan-samples`) |
| `WM_BEAM_PLAN_HORIZON` | `beam_plan`: `n` lookahead steps the WM rolls each plan forward (`--wm-beam-plan-horizon`) |
| `WM_BEAM_MPC_EXECUTE_STEPS` | `beam_plan`: MPC cadence — turns the agent follows the cached plan before re-planning (`--wm-beam-mpc-execute-steps`) |
| `WM_BEAM_PLAN_SCORE_MARGIN` | `beam_plan`: keep the baseline action unless the top beam candidate beats it by more than this margin (`--wm-beam-plan-score-margin`) |
| `WM_BEAM_PLAN_DIVERSITY_MULTIPLIER` | `beam_plan`: scales the anti-repetition penalty on already-executed tool-name sets (`--wm-beam-plan-diversity-multiplier`) |
| `WM_BEAM_PLAN_HARD_OVERRIDE` | `beam_plan`: force the beam's confident recommendation over the agent's action (`--wm-beam-plan-hard-override`). Default off = **advisory**: the agent acts, the (confidence-gated) plan is injected only as guidance |
| `WM_JEPA_MERGE_CHECKPOINT`, `WM_BEAM_PLAN_DECODE_TOOL_OUTPUT`, `WM_BEAM_PLAN_DECODE_MAX_NEW_TOKENS` | merge in a decoder from another checkpoint and decode `beam_plan`'s predicted **tool output** text (not just labels) for the winning trajectory — see `src/ejepa_wm/README.md` |
| `WM_HIER_CEM_ANCHORS`, `WM_HIER_CEM_SAMPLES`, `WM_HIER_CEM_ELITES`, `WM_HIER_CEM_ITERS`, `WM_HIER_CEM_HORIZON`, `WM_HIER_CEM_INIT_STD`, `WM_HIER_CEM_MIN_STD`, `WM_HIER_CEM_SMOOTHING`, `WM_HIER_CEM_MIN_ELITE_AGREEMENT`, `WM_HIER_CEM_DECODE_STRATEGY`, `WM_HIER_CEM_DECODE_MAX_NEW_TOKENS` | `hier_latent_cem`: hierarchical latent-action CEM knobs (`--wm-hier-cem-anchors` / `-samples` / `-elites` / `-iters` / `-horizon` / `-init-std` / `-min-std` / `-smoothing` / `-min-elite-agreement` / `-decode-strategy` / `-decode-max-new-tokens`). Shares `WM_BEAM_MPC_EXECUTE_STEPS` (`--wm-beam-mpc-execute-steps`) for planning cadence and `WM_BEAM_PLAN_HARD_OVERRIDE` (`--wm-beam-plan-hard-override`) for the advisory/hard-override toggle with `beam_plan` |

The EWM server itself (vLLM + MCP) is the **same** one used by EnterpriseOps-Gym — bring it
up with `assets/EnterpriseOps-Gym/build_mcp.sh`, then point `WM_EWM_MCP_URL` at it.
`beam_plan` / `hier_latent_cem` instead run the JEPA world model in-process from
`--wm-ewm-jepa-checkpoint` (no vLLM/MCP EWM server).

## Run

```bash
# plain ReAct (no WM) — sanity check
ejepa bench run crmarenapro --executor mcp_react --config target=sample

# EWM-guided action selection over the shared EWM MCP server
export WM_STRATEGY=selection WM_BACKEND=ewm_predict WM_N=3 \
       WM_STATE=binary_error WM_EWM_MCP_URL=http://127.0.0.1:12072 \
       WM_EWM_MODEL=gymops_world_model
ejepa bench run crmarenapro --executor mcp_react --config target=sample

# JEPA MPC beam_plan lookahead (requires a JEPA canonical-event checkpoint)
ejepa bench run crmarenapro --executor mcp_react --config target=sample \
    --wm-strategy beam_plan --wm-ewm-jepa-checkpoint /path/to/jepa_ckpt \
    --wm-beam-plan-samples 5 --wm-beam-plan-horizon 3 \
    --wm-beam-mpc-execute-steps 3 --wm-beam-plan-score-margin 0.0 \
    --wm-beam-plan-diversity-multiplier 1

# JEPA MPC hier_latent_cem lookahead (same JEPA checkpoint requirement as beam_plan)
ejepa bench run crmarenapro --executor mcp_react --config target=sample \
    --wm-strategy hier_latent_cem --wm-ewm-jepa-checkpoint /path/to/jepa_ckpt \
    --wm-hier-cem-anchors 8 --wm-hier-cem-samples 256 --wm-hier-cem-elites 16 \
    --wm-hier-cem-iters 3 --wm-hier-cem-horizon 5 \
    --wm-beam-mpc-execute-steps 3
```

## Files

- `executor.py` — A2A executor; `build_executor()` returns the routed `Executor`.
- `wm_react.py` — `WmReactAgent` (subclasses `baseline_crm_agent`'s `Agent`); builds the
  WM from env, samples/ranks or injects per turn, and tracks `conversation_flow`.
- `version.toml` — executor version.
