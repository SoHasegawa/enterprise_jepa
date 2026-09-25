# `ejepa_wm` — pluggable World Model for `ejepa`

A benchmark-agnostic **World Model (WM)**: a helper an agent/executor consults at its decision
points. It has swappable *strategies* and *backends*, surfaced as `ejepa` CLI args.

## Concepts

| Strategy           | Method     | What it does                                                       |
|--------------------|------------|--------------------------------------------------------------------|
| `prompt_injection` | `advise()` | WM returns a guidance block injected (transiently) into the prompt |
| `selection`        | `select()` | Agent samples *K* candidate actions; WM picks one of the *K*       |
| `itp_i`            | `advise()` | Adaptive-K WM imagination, then policy reflection before acting    |
| `none`             | both       | Baseline: `advise`→`""`, `select`→`0` (no-WM arm)                  |

| Backend   | Class                | Notes                                                        |
|-----------|----------------------|--------------------------------------------------------------|
| `noop`    | `NoopWorldModel`     | The no-WM baseline                                           |
| `served`  | `ServedWorldModel`   | **Served model as WM** (any OpenAI-compatible endpoint, e.g. local vLLM); reuses the EnterpriseOps baseline serving env |
| `ewm_predict` | `EwmPredictWorldModel` | **Direct EWM feasibility (no MCP)** — scores each candidate action's binary+error outcome via the EWM (vLLM/transformers); used with `selection` |
| `llm`     | `LlmWorldModel`      | Generic LLM-as-WM (inject your own `chat_fn`)                |

The *strategy* (how the WM is used) and *backend* (how it decides) are independent.

## Canonical action-feedback harnesses (`revision` and `reference`)

Both harnesses score the policy's proposed action with the configured canonical-state world
model. `revision` immediately returns the prediction to the policy and makes one additional
policy call to proceed with or revise the unexecuted action. `reference` executes the original
action unchanged, stores its prediction, and injects that prediction at the next policy step,
where it can be compared with the real tool result. Delayed reference therefore requires a
multi-step executor; Workspace-Bench's single-shot executor reports it as unsupported.

The action-feedback harness is shared by all scorers that implement the canonical scoring
contract:

- JEPA heads inject their predicted categorical state and terminal probability.
- `llm_canonical_trained` and `llm_canonical_zeroshot` inject the LLM-generated categorical
  state and terminal label (represented as one-hot categories by the scorer).
- `llm_tool_output_judge` injects the predicted tool output together with the acting agent
  model's step judgment (`can_proceed`, failure/progress probabilities, and finish judgment).
  Thus tool-output revision costs one WM generation, one same-agent judge call, and—when useful
  feedback is produced—one additional policy action call. Reference delays the same WM and judge
  evidence until the next ordinary policy call.

```bash
# Immediate pre-execution revision
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --config target=opsgym_80_test --config capture_trajectory=true \
  --wm-strategy revision \
  --wm-ewm-jepa-checkpoint /path/to/jepa-canonical-event-checkpoint \
  --wm-jepa-observation-backend canonical_event

# One-step-delayed reference
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --config target=opsgym_80_test --config capture_trajectory=true \
  --wm-strategy reference \
  --wm-ewm-jepa-checkpoint /path/to/jepa-canonical-event-checkpoint \
  --wm-jepa-observation-backend canonical_event
```

For a canonical-state LLM-WM served through vLLM, replace the JEPA options with:

```bash
WM_VLLM_BASE_URL=http://127.0.0.1:9000/v1 \
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --config target=opsgym_80_test --config capture_trajectory=true \
  --wm-strategy revision \
  --wm-llm-ewm-mode llm_canonical_trained \
  --wm-ewm-model world_model
```

For tool-output prediction plus the same-agent judge:

```bash
WM_VLLM_BASE_URL=http://127.0.0.1:9000/v1 \
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --config target=opsgym_80_test --config capture_trajectory=true \
  --wm-strategy revision \
  --wm-llm-ewm-mode llm_tool_output_judge \
  --wm-ewm-model world_model
```

Use `--wm-strategy reference` instead of `revision` for delayed next-step feedback in either
LLM-WM mode. Use `llm_canonical_zeroshot` when the served model was not trained on the canonical
schema.

### Single-model, single-benchmark harness comparison

`scripts/run_wm_harnesses.py` runs ITP-I, interval-triggered beam planning,
critic-triggered beam planning, revision, and reference exactly once each. Put the benchmark and
one model configuration after `--`; the wrapper adds the strategy-specific arguments and writes
an incrementally updated comparison JSON plus one log per harness.

```bash
python scripts/run_wm_harnesses.py \
  --result-root "$BENCHMARK_HOME/experiments" \
  --label enterpriseops-jepa \
  -- ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --config target=opsgym_80_test \
  --wm-ewm-jepa-checkpoint checkpoints/jepa \
  --wm-jepa-observation-backend canonical_event
```

Use `--dry-run` to inspect all five commands without executing them. The same wrapper accepts the
LLM canonical-state or tool-output model options shown above because model arguments remain in the
shared base command.

### Iterative score-conditioned beam refinement

Open-loop `beam_plan` can optionally run multiple generate-score-refine rounds. After each
non-final generation round, the world model scores the accumulated candidate trajectories. The
next shared generation prompt contains the top prior trajectories with their scores, veto status,
score explanation, and predicted states, allowing the action generator to preserve useful steps
and repair weak plans. Complete plans are deduplicated across rounds, and the final scoring pass
selects from all unique candidates produced in every round.

```bash
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --config target=opsgym_80_test --config capture_trajectory=true \
  --wm-strategy beam_plan \
  --wm-ewm-jepa-checkpoint /path/to/jepa-canonical-event-checkpoint \
  --wm-jepa-observation-backend canonical_event \
  --wm-beam-plan-refinement-rounds 3 \
  --wm-beam-plan-refinement-top-k 4 \
  --wm-beam-plan-samples 8 --wm-beam-plan-horizon 4
```

`--wm-beam-plan-refinement-rounds 1` is the default and exactly preserves single-pass behavior.
Values above one automatically select open-loop planning. Each refinement round still uses one
shared prompt for its samples, so a vLLM `n=k` sampler can share prefill within that round; prompts
necessarily differ between rounds because later rounds include prior scores. Telemetry records
per-round accepted/duplicate counts, generation time, request count, and WM scoring-pass count.

### Actionable beam scoring and per-horizon policy revision

`beam_plan` can prioritize immediately usable plans instead of allowing attractive later writes
to compensate for a generic first read. Its adjusted trajectory score adds three terms to the
canonical-state utility sum: an extra first-step score, coverage of operation types explicitly
requested by the task, and a bonus when the first action performs one of those operations. The
operation vocabulary is benchmark-agnostic (`create`, `update`, `delete`, `run`, `test`,
`communicate`, and `validate`); candidate action types come from the JEPA classification head,
with tool-name inference only as a fallback. Existing information-saturation penalties remain
active for `read` and `search`. Configure the terms with:

- `--wm-beam-plan-first-step-weight` (default `0.5`)
- `--wm-beam-plan-required-action-coverage-bonus` (default `1.0`)
- `--wm-beam-plan-first-required-action-bonus` (default `0.75`)
- `--wm-beam-plan-read-saturation-threshold` (default `2`)
- `--wm-beam-plan-read-penalty` (default `1.0`)
- `--wm-beam-plan-read-after-progress-scale` (default `0.5`)

Enable `--wm-beam-plan-revision` to make the acting policy explicitly choose between its fresh
action and the corresponding beam action at **every** live horizon step. The first decision is
between the original proposal and plan step 1. If the plan action is selected, the cache advances
and the next decision compares a fresh proposal with plan step 2, then step 3, and so on. If the
policy action is selected, the remaining imagined suffix is discarded because it was conditioned
on a transition that did not occur. Exact policy/plan agreement advances the cursor without an
extra arbitration request. A symbolic plan action is never executed unresolved; it is accepted
only when the fresh policy action supplies grounded arguments for the same tools, otherwise the
policy action wins safely. `--wm-beam-plan-hard-override` takes precedence when both modes are set.

```bash
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --config target=opsgym_80_test --config capture_trajectory=true \
  --wm-strategy beam_plan \
  --wm-ewm-jepa-checkpoint /path/to/jepa-canonical-event-checkpoint \
  --wm-jepa-observation-backend canonical_event \
  --wm-beam-plan-revision --wm-beam-plan-revision-temperature 0 \
  --wm-beam-plan-first-step-weight 0.5 \
  --wm-beam-plan-required-action-coverage-bonus 1.0 \
  --wm-beam-plan-first-required-action-bonus 0.75 \
  --wm-beam-plan-read-saturation-threshold 2 \
  --wm-beam-plan-read-penalty 1.0
```

Telemetry records the binary choice, whether arbitration was attempted, whether a planned action
was selected, the selected-action reason, and the extra acting-model call count for every step.

## Imagine-Then-Plan implicit-feedback baseline (`itp_i`)

`--wm-strategy itp_i` implements only the inference-time WM harness from
[Imagine-Then-Plan ITP-I](https://arxiv.org/abs/2601.08955):

1. The existing, unmodified policy selects an integer lookahead `K` in `[0, Kmax]`.
2. For `K>0`, a separate generative WM predicts one concise `K`-step action/observation
   trajectory in a single request.
3. The policy receives that trajectory as hypothetical implicit feedback, reflects, and makes
   the real tool-bound action. The imagined first action is not directly executed.

This does **not** train or fine-tune the policy. It adapts the released ITP-I inference loop to
each benchmark's native tool-calling policy. Its default `Kmax=5`, decision temperature `0.8`,
WM temperature `0.7`, and foresight budget `256` follow the released evaluation settings.

```bash
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --config target=opsgym_80_test --config capture_trajectory=true \
  --wm-strategy itp_i \
  --wm-itp-world-model-base-url http://127.0.0.1:9020/v1 \
  --wm-itp-world-model-model world_model \
  --wm-itp-max-k 5
```

Use `--wm-itp-fixed-k 0` for no lookahead or a positive value for a fixed-lookahead ablation;
the default `-1` is adaptive. Per-step trajectory metadata records `itp_i_k`, raw foresight,
policy decision calls, WM calls, and WM/advice elapsed time. The implementation follows the
released [ITP code](https://github.com/loyiv/ITP) prompt/control structure under its MIT license.

To use ITP-I with a JEPA canonical-state world model, provide the JEPA checkpoint instead of a
generative WM endpoint:

```bash
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --config target=opsgym_80_test --config capture_trajectory=true \
  --wm-strategy itp_i \
  --wm-ewm-jepa-checkpoint /path/to/jepa-canonical-event-checkpoint \
  --wm-jepa-observation-backend canonical_event \
  --wm-itp-max-k 5
```

In this mode policy and JEPA alternate for K imagined steps: the policy proposes one hypothetical
action, JEPA predicts its canonical next state, and the policy conditions its next hypothetical
action on that imagined prefix. The resulting action-to-canonical-state trajectory is returned as
ITP-I reflection feedback. The policy still independently chooses the real action afterward.

## Using it from `ejepa`

```bash
ejepa bench run EnterpriseOps-Gym \
    --wm-strategy selection --wm-backend served --wm-model Qwen3.6-27B --wm-n 4
```

`ejepa` exports the choice as `WM_*` env vars for the agent subprocess; the EnterpriseOps executor
then auto-selects the `wm_react` orchestrator and builds the WM via `wm_config_from_env()`.

### Served model as both agent **and** WM

Use the same served model an EnterpriseOps baseline runs (a local vLLM endpoint) as the WM. The `served`
backend reuses the baseline serving env, so if a run already exports `ENTERPRISEOPS_LLM_*` /
`LOCAL_VLLM_BASE`, no extra config is needed:

```bash
# baseline serves e.g. qwen3.6-27b at http://127.0.0.1:8020/v1 (run.sh START_VLLM=1)
ejepa bench run EnterpriseOps-Gym \
    --wm-strategy selection --wm-backend served --wm-model qwen3.6-27b --wm-n 4
```

Endpoint resolution (first non-empty wins): base_url = `WM_BASE_URL` → `ENTERPRISEOPS_LLM_BASE_URL`
→ `OPENAI_BASE_URL` → `LOCAL_VLLM_BASE`; model = `--wm-model` → `WM_MODEL` →
`ENTERPRISEOPS_LLM_MODEL`. Point it at a *different* served model than the agent simply by setting
`WM_BASE_URL` / `--wm-model`.

## Direct EWM feasibility, no MCP server (`backend=ewm_predict`)

The EWM's **binary + error** per-step feasibility can drive `selection` directly — no MCP
server, the world model runs in-process via vLLM or transformers. Each step the agent samples
`WM_N` candidate actions and the EWM scores each one's predicted tool-execution result against
the state reconstructed from the conversation flow; a candidate the WM predicts will **succeed**
is chosen. This is the recommended way to use EWM as a feasibility check (vs. exposing it as an
agent-called MCP tool, which relies on the agent passing state and one concrete action itself).

```bash
# world model on vLLM :9000 (or EWM_WORLD_MODEL_PATH=/ckpt for transformers); agent samples 4 candidates
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
    --wm-strategy selection --wm-backend ewm_predict --wm-n 4 \
    --config target=sample
# env the backend reads: EWM_WORLD_MODEL_METHOD=vllm/gymops_world_model WM_VLLM_SERVER_PORT=9000
#                        WM_STATE=binary_error  (and a sampling temperature so candidates differ)
```

Notes: this needs `WM_N>1` and a non-zero sampling temperature (else the candidates are
identical). Tool names are normalized (a leading `functions.`/namespace is stripped) and
arguments under either `args` or `arguments` are forwarded, so the WM sees the real call — do
**not** set `EWM_PREDICT_MCP_URL` (that switches on the separate MCP-tool path below).

## EWM prediction as an MCP tool (`server/ewm_predict.py`)

The Enterprise World Model's single-step prediction is also exposed as a standalone **MCP
server** so an agent can call the world model like any other tool (email, calendar). It wraps
the per-step WM call (`_ewm_runtime.predict_wm_feedback`), **not** a whole imagined execution —
the agent (client side) keeps any imagined-rollout loop and consults the WM one step at a time.

The server runs the EWM via either backend, selected by env:

| Backend        | Select with                                                        | Reaches the model |
|----------------|--------------------------------------------------------------------|-------------------|
| `vllm`         | `EWM_WORLD_MODEL_METHOD=vllm/<model>` (+ `WM_VLLM_SERVER_PORT` / `WM_VLLM_BASE_URL`) | OpenAI-compatible (vLLM) endpoint over the network |
| `transformers` | `EWM_WORLD_MODEL_PATH=/path/to/checkpoint` (or `EWM_WORLD_MODEL_METHOD=transformers`) | a local HuggingFace checkpoint loaded in-process (needs `torch`+`transformers`) |

```bash
MCP_PORT=12072 \
EWM_WORLD_MODEL_METHOD=vllm/gymops_world_model WM_VLLM_SERVER_PORT=9000 \
WM_STATE=binary_error \
bash src/ejepa_wm/server/run_local.sh        # serves streamable-HTTP at /mcp
```

Tool `predict_state(system_prompt, user_prompt, action, previous_state=None, state_history=None,
wm_state=None, interaction_index=0, conversation_flow=None)`. EWM is a **per-step** model, so
`action` is the **single next** tool call (`[{"name", "args"}]`) with concrete arguments — not a
multi-step plan. If a multi-step plan is passed, only the leading placeholder-free call(s) are
scored and a `warnings` entry says what was skipped (so the agent re-predicts each step once its
inputs are known). Pass `conversation_flow` (the running `system_message`/`user_message`/
`ai_message`/`tool_result` events) to have the server reconstruct the current state, history and
prompts — making the prediction state-aware without hand-assembling a state dict. For
`wm_state=binary_error` the `state` carries the predicted binary tool-execution result plus error:

```json
{"wm_state": "binary_error", "state": {...}, "success": 0, "error_message": "disk is full",
 "current_stage": null, "remaining_stages": null, "tool_output": "", "raw_prediction": "0,disk is full",
 "parse_error": null, "evaluated_action": [{"name": "...", "args": {...}}], "warnings": ["..."]}
```

`wm_state` modes: `binary_error` (default), `binary_error_stage`, `tool_output`, `canonical_nudge`
(schema-based categorical event state; weak nudge fields such as `recommended_abstract_action`
and `missing_information_type` are not scored or injected by `beam_plan`). The server also
exposes `info()` (resolved backend) and `generate(messages, temperature)` — a raw EWM
chat-completion passthrough used for imagined trajectories (below). This is distinct from
`assets/.../mcp_react_ewm/server/ewm/mcp_server.py`, which exposes the *whole*
`run_imagined_execution(task)` executor. Verification: see "Step 7" in `procedure_mcp.md`.

### Imagined trajectory over MCP (`ewm_imagined` + `generate`)

The imagined-trajectory backend (`ewm_imagined`) can reach a **remote** EWM as an MCP service
while keeping the agent↔WM rollout loop client-side (an MCP server can't call back to the
agent). Set `WM_EWM_MCP_URL` and `ewm_imagined` routes its world-model calls through the
server's `generate` tool (via `McpEwmGenerator`) instead of the in-process vLLM client; the
rollout is assembled and injected into the agent prompt exactly as in the direct path.

```bash
# EWM predict server up on :12072 (run_local.sh); agent is your client-side policy LLM
export WM_EWM_MCP_URL=http://127.0.0.1:12072      # → WM generation over MCP (tool: generate)
ejepa bench run EnterpriseOps-Gym --executor mcp_react --wm-strategy imagined --config target=sample
# optional: WM_EWM_MCP_TOOL=generate  WM_EWM_MCP_TOKEN=<token>; rollout axes as in the direct path
```

Unset `WM_EWM_MCP_URL` → the same backend runs the WM in-process over vLLM/transformers.

### JEPA world model instead of a text LLM (`ewm_imagined` + `WM_EWM_JEPA_CHECKPOINT`)

The imagined-trajectory replay loop is world-model-agnostic, so the world model can be a **text
JEPA** net instead of a text LLM. This is the replay path from `ewm/src/finetuning_jepa.py`
(`run_jepa_replay` → `JepaTextWorldModelGenerator`): the agent stays a text LLM proposing actions,
while a JEPA net predicts each imagined action's outcome **in latent space** and reconstructs the
imagined observation/state from its heads. Ported inference lives in `backends/_ewm_jepa.py`
(`JepaEwmGenerator`, self-contained — no external `ewm` checkout); `predict_wm_feedback` prefers a
generator's structured `predict_feedback` seam over the text-parsing path.

Point `ewm_imagined` at a JEPA checkpoint directory (containing `text_leworldmodel.pt` +
`backbone/`, and optionally `canonical_event_vocab.json` / manifests):

Each option is a first-class `ejepa bench run --wm-*` flag (mapped to the `WM_*` env var below), so
it works regardless of executor:

```bash
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --wm-strategy imagined \
  --wm-ewm-jepa-checkpoint /models/jepa_ckpt \    # (or --wm-ewm-backend jepa) → JEPA world model
  --wm-jepa-observation-backend auto \            # auto | canonical_event | success | decoder
  --wm-imagined-max-steps 3 \                     # rollout axes as in the text-LLM path
  --config target=sample
# optional: --wm-jepa-dtype bfloat16  --wm-jepa-trust-remote-code
#           --wm-jepa-arch-defaults '{"backbone_type":"seq2seq","goal_conditioning":true}'
#             (arch fallbacks used only when the checkpoint has no jepa_data_manifest.json;
#              a value present in the manifest always wins)
```

Equivalently via env (what the flags set, and what the backend reads directly):
`WM_EWM_JEPA_CHECKPOINT` (or `WM_EWM_BACKEND=jepa`), `WM_JEPA_OBSERVATION_BACKEND`, `WM_JEPA_DTYPE`,
`WM_JEPA_TRUST_REMOTE_CODE`, `WM_JEPA_ARCH_DEFAULTS`, `WM_EWM_MAX_NEW_TOKENS`.

`WM_JEPA_OBSERVATION_BACKEND` selects which JEPA head reconstructs the imagined observation:
`canonical_event` (classification heads → canonical outcome + nudge), `success` (P(action
succeeds)), or `decoder` (seq2seq decodes raw tool-output text); `auto` prefers them in that order
by what the checkpoint provides. Requires `torch`/`transformers` and (for canonical_event/success)
a checkpoint with the corresponding trained heads. No mcp_react changes are needed — `wm_react`
already builds the WM from these `WM_*` env vars.

**Predictor architecture is auto-detected from the checkpoint manifest** (`predictor_arch`:
`mlp` or `transformer`) and is completely transparent to every caller — `beam_plan`,
`hier_latent_cem`, `score_action_plans_canonical_event`, the decoder paths, all of it. `mlp` is
the original concat-MLP; `transformer` is a LeWorldModel-style causal-attention predictor with
AdaLN action conditioning (`AdaLNTransformerPredictor` in `backends/_ewm_jepa.py`), ported from
`ewm/src/finetuning_jepa.py`. Point `WM_EWM_JEPA_CHECKPOINT` at a `predictor_arch=transformer`
checkpoint and it just works — no flag, no executor change. The transformer predictor's full
design attends over a multi-step history of past latents (`frame_history`): `beam_plan`
(`score_action_plans_canonical_event`, `decode_plan_observations`) and `hier_latent_cem`
(`hierarchical_cem_plan`/`_rollout_and_score`) both build this from the real logged
action/observation history and autoregressively extend it every lookahead step with each newly
predicted event — see `predict_latent`'s docstring and `_extend_batch_frame_history` in
`backends/_ewm_jepa.py`. With no history yet (the very first step of an episode), it falls back
to the model's own documented degenerate case (`frame_history=None`, a 2-position
`[context, z_current]` sequence).

**`canonical_event_head_inputs`** (also manifest-driven, also transparent) controls which
latents the classification-head trunk reads: `all` (default) is `[z_current, z_action, z_context,
z_pred]`; `ctx_pred` is `[z_context, z_pred]`; `pred_only` is `[z_pred]`; `state` reads the
transformer predictor's own hidden state h_t directly (`predict_latent_with_state`'s `state`
return) instead of the projected latent — sized by the predictor's width (`predictor.dim`), not
`latent_dim`, and only valid with `predictor_arch=transformer`. Getting this wrong (e.g.
hardcoding `latent_dim * 4`) is exactly what produces a `size mismatch for
canonical_event_trunk.0.weight` error on checkpoints trained with `state`/`ctx_pred`/`pred_only`
readouts — `data_jepa_heads_ensemble` is one such checkpoint. `state_action` is a fifth mode:
`[h_t, z_action]` — the belief state plus the (already-projected) action latent, for fields that
are properties of the tool call itself (`action_type`, `object_type`, `side_effect_type`,
`risk_signal`) rather than of the predicted outcome — `h_t` is conditioned on the action but has
to spend capacity re-deriving it, so making it explicit is cheap, at the cost of reopening a
readout path around the predictor for action-identity information (not for the outcome). Trunk
width is `predictor.dim + latent_dim` (the two widths *add*, unlike `all`'s multiplicative
`latent_dim * 4`) — also only valid with `predictor_arch=transformer`.

### LLM canonical-event world model instead of JEPA (`ewm_imagined` + `WM_EWM_LLM_CANONICAL_EVENT_CHECKPOINT`)

`beam_plan`/its critic trigger (`WM_BEAM_PLAN_TRIGGER=critic`, below) only ever call one method
on `self._wm`: `score_action_plans_canonical_event(...)` (`supports_beam_plan()` just checks it's
callable). `backends/_ewm_llm_canonical_event.py` implements that same contract on top of a
fine-tuned causal LM instead of JEPA's classification heads, so it's a drop-in generator swap —
no planning-logic change, `beam_plan` with **or without** the critic trigger both work unmodified.

```bash
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --wm-strategy beam_plan \
  --wm-ewm-llm-canonical-event-checkpoint checkpoints/llm_wm_canonical_event \
  --wm-beam-plan-trigger critic \        # or the default "interval" -- both call the same method
  --config target=sample
# optional: --wm-jepa-dtype bfloat16  --wm-jepa-trust-remote-code  (reused knobs, same names)
```
Env: `WM_EWM_LLM_CANONICAL_EVENT_CHECKPOINT` (or `WM_EWM_BACKEND=llm_canonical_event`),
`WM_EWM_MAX_NEW_TOKENS` (default 128 here vs. 4096 for the plain text-LLM WM — one canonical-event
JSON object is short), `WM_JEPA_DTYPE`/`WM_JEPA_TRUST_REMOTE_CODE` (shared with the JEPA path;
there's nothing JEPA-specific about a torch dtype/`trust_remote_code` flag).

**Why this needs its own scoring mechanism.** A classifier head (JEPA) reads probabilities
straight off logits in one forward pass — no sampling, no text parsing. A causal LM by default
can only emit ONE generated JSON completion per call: an argmax over token sequences, not a
distribution over the 11 canonical-event fields `_ewm_canonical_event_scoring` needs. There is
no reference implementation for this anywhere in the EWM repo — `beam_plan`/the critic trigger
there are wired exclusively to JEPA. This backend invents the missing piece: per lockstep step,
(1) generate the completion once per active candidate (batched), (2) parse it into a field→value
dict (garbage generation degrades to neutral `unknown`/`["none"]` defaults, never a crash), (3)
re-serialize it into the checkpoint's exact fixed-key-order training format, (4) teacher-force
prompt+that canonical text through one extra forward pass per candidate and, at the token
position right before each field's value starts (found via the tokenizer's offset mapping),
restrict the next-token logits to that field's category vocabulary and softmax — reading off
`P(category | prompt, actual chosen earlier field values)` the way a classifier would, from a
model that has none. Cost is one `generate()` + one `forward()` per lockstep step, batched across
every active candidate — same shape as JEPA's per-step batched cost, not per-category-per-field.

The multi-label `missing_information_type` field is still decoded for compatibility, but it is
not part of the scoring signal. The scorer also ignores `recommended_abstract_action`, matching
the JEPA beam-planning path where these fields were too weak as next-state prediction signals.

Live-verified against `checkpoints/llm_wm_canonical_event/` (a 596M-param Qwen3 fine-tune)
on GPU: the 11-field probability vectors sum to ~1.0 per field as expected; scoring a real
held-out trajectory step at the critic's own batch-size-1 call pattern reproduced the gold
`execution_status`/`error_signature`/`risk_signal`/`action_type` exactly and gave a correctly
low `P(execution_status=failure)` for a genuinely-successful action; a 2-candidate `beam_plan`
comparison (a safe `create_channel` vs. a destructive `delete_team`) correctly scored the safe
plan positive/non-vetoed and hard-vetoed the destructive one on `P(failure)`. Note this is a much
smaller, less accurate model than JEPA's dedicated classification heads (a quick spot check
against one held-out example matched only 4/10 fields exactly) — expect noisier `beam_plan`
decisions than the JEPA backend, and prefer real long system prompts (matching training
distribution) over short/synthetic ones, which this checkpoint handles far less robustly.

### Scoring-path performance (memoized encodes, skipped goal encode, batched probability transfer)

`JepaEwmGenerator._encode_latent_text` memoizes by `(max_length, text)` (bounded LRU,
`LATENT_TEXT_CACHE_SIZE = 32`): the context and goal texts are constant for a whole task and the
state text recurs between scorers called at the same step, yet each encode is a full backbone
forward. `score_action_plans_canonical_event` additionally skips encoding the goal text at all
when `goal_conditioning=False` (previously a wasted backbone forward every scoring call), uses
`torch.inference_mode()` instead of `torch.no_grad()`, and converts per-step head logits to
probabilities via `logits_to_field_probs_batched` (one host transfer per field instead of one
per probability scalar — hundreds of device syncs per step at `B` plans × 11 fields × ~10
classes, collapsed to one transfer per field) in both `score_action_plans_canonical_event` and
`hier_latent_cem`'s `_rollout_and_score` (the CEM rollout, where `B` is the sample count —
typically 256, so this matters far more there). None of this changes the scores: verified
bit-identical (`0.000e+00` max diff) against a real checkpoint before/after, including a
same-input second call that hits the cache with zero additional backbone forwards.

### Compiled inference (`WM_JEPA_COMPILE`, opt-in)

Profiling one scoring call (8 plans, horizon 3, bf16, H200) puts ~70% of its ~59 ms in the
two Qwen3-0.6B backbone passes (~21 ms per ~300-token pass), which is kernel-launch overhead,
not compute — the regime CUDA graphs remove. `WM_JEPA_COMPILE=1` wraps the backbone,
predictor and heads in `torch.compile(mode="reduce-overhead")` and makes their shapes recur:
token lengths are right-padded to a multiple of `WM_JEPA_PAD_MULTIPLE` (default 64, never past
`max_length`; padded positions carry `attention_mask=0`, so the pooled latent is unchanged) and
encode batches to a power of two. The call drops to ~13–20 ms (h=1..4: 0.070/0.074/0.087/0.101 s
→ 0.013/0.015/0.018/0.021 s with `scripts/measure_wm_latency_vs_horizon.py`); over 24 salted
histories the eager and compiled paths agreed on the top plan 24/24 with Spearman 1.000
(max |score diff| 0.035 vs a within-call spread of 2.3). `WM_JEPA_COMPILE=default` compiles
without CUDA graphs (~27 ms). The first call per new (batch, length) bucket compiles/captures
(~60 s for the first), so pair it with `WM_SHARE_MODEL_WEIGHTS=1` rather than compiling per task;
the default remains eager.

### Open-loop planning (`WM_IMAGINED_ROLLOUT_MODE` / `--wm-imagined-rollout-mode`)

`closed_loop` (default) chains one agent call + one world-model call per horizon/rollout step —
step *k*'s action is chosen after seeing the WM's prediction for step *k-1*. `open_loop` instead
emits ALL candidate plans in **one** agent call up front (unknown values as symbolic
`"$stepK.field"` references, since actions never see a predicted state) and scores/rolls them
forward in one batched world-model pass. Trade-off: an open-loop plan can't react to a predicted
failure mid-trajectory, so quality rests on plan diversity and the same confidence gate that
already exists.

Affects two independent code paths, both gated on the same flag:
- **`beam_plan`** (`ewm_imagined.EwmImaginedWorldModel._beam_plan_open_loop`): replaces the
  per-depth candidate-proposal loop. *m* samples of a ONE-plan prompt (`_ewm_beam_plan.
  build_single_plan_prompt`/`parse_single_plan`, via `sample_many` — see below) build the
  candidate set; duplicate/unparseable samples are dropped. The seed (agent's own action) is
  scored alongside them as a one-step plan at index 0 but can never be selected as the winner;
  `score_action_plans_canonical_event`'s existing batched lockstep scoring (also returning
  `per_step_predicted_state`/`per_step_field_probs`, one entry per step rather than only the
  terminal one) handles all of it in one call. The override margin compares `per_step[0]`
  scores (winner vs. seed), not full trajectory sums, which are length-biased.
- **Plain imagined rollout** (`_ewm_runtime.imagine_trajectories_open_loop`, dispatched from
  `optimize_imagined_trajectory`): `WM_IMAGINED_ROLLOUTS` samples of a ONE-plan prompt
  (`build_react_open_loop_plan_messages`/`parse_open_loop_plans` with `num_plans=1`, via
  `sample_many`) build the plan set; each plan's per-step world-model prediction is looped
  directly through `predict_wm_feedback`'s local-adapter seam when the WM exposes
  `predict_feedback` (JEPA — a cheap in-process forward, no batching needed), or otherwise
  batched across all pending plans via `generate_many`. Falls through to a real closed-loop
  rollout for that cycle if every sample fails to parse.

Both used to ask for all *m*/`WM_IMAGINED_ROLLOUTS` plans in ONE response (`_ewm_beam_plan.
build_skeleton_prompt`/`parse_plans`, still present but unused by either driver, kept as a
fallback shape); sampling the SAME one-plan prompt *k* times instead is strictly cheaper when
the backend supports it — see `sample_many` below — because *k* plans in one response is *k*
times the output tokens on one serial decode stream, whereas *k* sampled responses decode
concurrently after a single shared prefill.

`hier_latent_cem` is unaffected by this flag — it's already structurally "1 LLM call (K diverse
anchors) + 1 batched world-model CEM pass", the same shape open-loop is trying to achieve.

### Sampling k plans in one request (`sample_many`, `EwmGenerator.generate_samples`)

`_ewm_runtime.sample_many(generator, messages, *, temperature, num_samples)` returns
`(texts, requests_issued)`. It prefers a backend exposing `generate_samples(messages,
temperature, num_samples) -> list[str]` — `EwmGenerator` implements this as one
`/chat/completions` request with `"n": k"` (vLLM/OpenAI-compatible servers decode the k
sequences concurrently after prefilling the shared prompt once), so `requests_issued == 1`
regardless of `k`. Any other backend (including a plain executor `chat_fn`) falls back to
`generate_many` issuing `k` separate requests — still correct, just not the single-request win.
Both `beam_plan`'s and the plain rollout's open-loop drivers report `requests_issued` as their
real LLM-call count (`beam_llm_calls` in beam_plan's detail dict) rather than assuming it's 1.

Set `WM_SAMPLE_TEMPERATURE_LADDER=1` (or pass `--sample-temperature-ladder` to `ejepa benchmark
run`) to spread open-loop beam samples over a bounded ladder. The first sample uses a lower
exploit temperature (`0.4 * base`, with a `0.05` floor) and the remaining samples increase from
`0.85 * base` toward `min(1.2, 1.7 * base)`. Override the ceiling with
`WM_SAMPLE_TEMPERATURE_LADDER_MAX` / `--sample-temperature-ladder-max`. This intentionally
forfeits `generate_samples(n=k)`, since one request cannot assign different temperatures to its
returned sequences; the default remains the one-request fast path.

For diversity without losing the one-request shared-prefill path, set
`WM_BEAM_PLAN_SSOT_DIVERSITY=1` / `--wm-beam-plan-ssot-diversity`. This follows the
String-Seed-of-Thought idea: the shared prompt asks each sampled continuation to internally
generate a random-looking string, map it to one option in the existing diversity menu, and emit
only the final JSON plan. Unlike the temperature ladder, this keeps one common prompt and one
`n=k` request; diversity comes from the generated suffix.

`EwmGenerator` also does a one-shot check that the vLLM endpoint it's talking to is actually
reusing shared prompt prefixes (`GET /metrics`, looking for `vllm:prefix_cache_queries_total`):
every planning call here repeats the agent's system prompt/conversation verbatim and only
appends a short instruction, so with prefix caching the prefill is nearly free and without it
every call re-reads the whole prompt. Logs a warning once if the counter is present and zero;
says nothing if the server doesn't report that metric at all (not a vLLM build that has it) or
if the probe itself fails for any reason — a diagnostic must never break generation.
`WM_EWM_MAX_NEW_TOKENS` was already correctly threaded through to the request's `max_tokens`
before this work (no equivalent of upstream's hardcoded-`max_tokens` bug existed here).

### LLM-EWM modes for beam_plan

`beam_plan` now supports three LLM-EWM modes in addition to JEPA. Select them with
`WM_LLM_EWM_MODE` or `--wm-llm-ewm-mode` (the older `--wm-ewm-backend` accepts the same values).

- `llm_canonical_trained`: for an LLM trained on the reduced EWM target from
  `~/program/ewm`: `execution_status`, `progress_signal`, `information_sufficiency`,
  `error_signature`, `side_effect_type`, and `terminal`. The generated category is treated as
  probability 1.0 and every other category as 0.0 for beam scoring and critic thresholds. Use
  `--wm-ewm-llm-canonical-event-checkpoint /path/to/checkpoint` for local HF loading, or serve
  the checkpoint and set `--wm-ewm-model` / `WM_VLLM_BASE_URL`.
- `llm_canonical_zeroshot`: asks a general served LLM to emit the same reduced canonical JSON
  zero-shot. Scoring and critic behavior are identical to `llm_canonical_trained`, including
  one-hot probabilities, but malformed or missing fields fall back to neutral labels and
  `terminal=not_finished`.
- `llm_tool_output_judge`: asks a general served LLM world model to predict tool-output text.
  The policy agent model is then used as the LLM judge for step failure/progress/finish signals
  and whole-trajectory selection. This mode does not use a separate judge model by default;
  judge calls go through the same `chat_fn` as the acting agent.

Examples:

```bash
# 1. Trained canonical LLM-EWM loaded locally
ejepa bench run EnterpriseOps-Gym --executor mcp_react --config target=opsgym_80_test \
  --wm-strategy beam_plan \
  --wm-llm-ewm-mode llm_canonical_trained \
  --wm-ewm-llm-canonical-event-checkpoint checkpoints/llm_wm_canonical_event_notb

# 2. General LLM zero-shot canonical EWM served over vLLM/OpenAI-compatible API
WM_VLLM_BASE_URL=http://127.0.0.1:9000/v1 \
ejepa bench run EnterpriseOps-Gym --executor mcp_react --config target=opsgym_80_test \
  --wm-strategy beam_plan \
  --wm-llm-ewm-mode llm_canonical_zeroshot \
  --wm-ewm-model Qwen/Qwen-AgentWorld-35B-A3B

# 3. General LLM predicts tool outputs; the acting agent model judges steps/trajectories
WM_VLLM_BASE_URL=http://127.0.0.1:9000/v1 \
ejepa bench run EnterpriseOps-Gym --executor mcp_react --config target=opsgym_80_test \
  --wm-strategy beam_plan \
  --wm-llm-ewm-mode llm_tool_output_judge \
  --wm-ewm-model Qwen/Qwen-AgentWorld-35B-A3B
```


### Dedicated diffusion action sampler

Open-loop `beam_plan` can generate candidate plans with a dedicated sampler while leaving the
policy agent and JEPA world model unchanged. Two backends are available:

- `openai`: set `--wm-beam-action-sampler-base-url`; the default model is
  `nvidia/diffusiongemma-26B-A4B-it-NVFP4`.
- `huggingface`: set `--wm-beam-action-sampler-backend huggingface`; the model is loaded inside
  the purple executor and defaults to `google/diffusiongemma-26B-A4B-it`.

The NVIDIA NVFP4 artifact is specific to vLLM/ModelOpt and cannot be loaded by the Hugging Face
backend. The EnterpriseOps `openai_jepa` extra already locks a compatible Transformers release;
install it with `uv sync --project assets/EnterpriseOps-Gym/purple --extra openai_jepa`. Other
executor environments need PyTorch, Accelerate, and Transformers 5.11 or newer. Select GPUs with
`CUDA_VISIBLE_DEVICES`; the default device map is `auto` and dtype is `bfloat16`.

Each sample is capped at 256 tokens by default
(`WM_BEAM_ACTION_SAMPLER_MAX_NEW_TOKENS`) so a DiffusionGemma plan fits one canvas. The optional
`WM_BEAM_ACTION_SAMPLER_MAX_DENOISING_STEPS` trades generation quality for latency. For the
OpenAI-compatible backend, `WM_BEAM_ACTION_SAMPLER_API_KEY` is environment-only and defaults to
`not-needed`.

Without `--sample-temperature-ladder`, all candidates use one `n=k` request against the dedicated
server, or one tensor-batched `generate()` call with Hugging Face. With the ladder, candidates
require separate generations; leave it disabled for the fastest local DiffusionGemma path.
Parsed plans are rejected before JEPA scoring if they contain a tool name outside the current
catalog or a non-object argument payload. Runtime tool-schema validation still occurs at execution
because symbolic `$stepN.field` references are unresolved during planning.

Example:

```bash
CUDA_VISIBLE_DEVICES=1,3 ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --config target=opsgym_80_test --config capture_trajectory=true \
  --wm-strategy beam_plan --wm-imagined-rollout-mode open_loop \
  --wm-beam-plan-ssot-diversity --wm-beam-plan-samples 8 \
  --wm-beam-action-sampler-backend huggingface \
  --wm-beam-action-sampler-model google/diffusiongemma-26B-A4B-it
```

### Batched closed-loop multi-rollout (`WM_IMAGINED_PARALLEL_ROLLOUTS`, `WM_IMAGINED_SINGLE_CALL_STEP`)

Only relevant when the multi-rollout selection path is active (`WM_IMAGINED_ROLLOUTS>1` or
`WM_IMAGINED_SELECTION=llm_judge`) — the default single-rollout path always calls
`imagine_trajectory` directly and ignores both knobs. Serial multi-rollout costs `N * steps *
(think + act + world_model)` sequential generations (one full rollout, then the next).
`WM_IMAGINED_PARALLEL_ROLLOUTS=1` (default) instead advances all N rollouts in LOCKSTEP via
`_ewm_runtime.imagine_trajectories_lockstep`: each step, every still-alive rollout's agent
decision is requested together through `generate_many`, then every rollout with a valid
action gets its world-model prediction requested together the same way (or looped through
`predict_feedback` for a JEPA-style local generator) — wall-clock becomes `~steps * phases`
batched calls regardless of N, with per-rollout semantics (prompts, temperatures, termination
rules) identical to the serial path. `WM_IMAGINED_SINGLE_CALL_STEP=1` (default) additionally
collapses each step's think+act into ONE combined generation (`build_react_step_messages`),
halving the per-step agent call count again — set `0` to A/B against the original two-call
think-then-act sequence.

### `beam_plan` re-plan trigger (`WM_BEAM_PLAN_TRIGGER` / `--wm-beam-plan-trigger`)

`interval` (default) re-plans every `WM_BEAM_MPC_EXECUTE_STEPS` steps regardless of need, so the
planning cost (an agent call plus the rollout) is paid on every step in amortized terms. `critic`
asks the cheaper question first: score the action the agent already produced — that generation
is sunk cost — as a one-step plan via **one world-model forward, no LLM call**
(`EwmImaginedWorldModel._beam_plan_critic`), and only escalate to a full planning cycle when the
prediction says the action is bad. Amortized cost becomes `critic + fire_rate * planning`, and
the fire rate is exposed (`critic_checks`/`critic_fires`/derivable fire rate in the `detail`
dict) so it can be measured and the thresholds calibrated from a real run.

Fires when the world model vetoes the action (the score config's own P(failure)/P(deleted)
safety limits), when P(execution_status=failure) reaches `WM_BEAM_PLAN_CRITIC_FAILURE_PROB`
(default 0.3), or when the action looks like it will not advance the task — `1 -
P(progress_signal=positive)` reaches `WM_BEAM_PLAN_CRITIC_STALL_PROB` (default 0.7).
`WM_BEAM_PLAN_CRITIC_MAX_QUIET_STEPS` (default 0 = disabled) is a safety valve that forces a
planning cycle after that many consecutive non-firing checks, so a mis-calibrated critic can't
disable lookahead for a whole episode. A critic malfunction (any exception scoring the seed
action) escalates to full planning rather than silently disabling it.

If the JEPA checkpoint includes a terminal head, `--wm-beam-plan-terminal-advice` enables a
non-binding finish advisory when `P(done)` reaches
`--wm-beam-plan-terminal-advice-threshold` (default 0.75). This never short-circuits execution;
it only tells the agent to stop calling tools and final-answer if the real observed state already
satisfies the task.

Not ported: a `WM_BEAM_PLAN_CRITIC_MIN_SCORE` threshold on the raw per-step score exists in the
EWM reference but is intentionally left out here — the reference itself ships it disabled by
default because "the scale is checkpoint-specific" (its own docs say to read real
`GYM_BEAM_PLAN_CRITIC`-equivalent telemetry from a run first to calibrate it), so there's no
default value to port faithfully; add it if/when real per-checkpoint calibration data exists.

Needs `score_action_plans_canonical_event`'s new `per_step_field_probs` field (raw
`{field: {category: prob}}` per step — the decoded `per_step_predicted_state` is an argmax and
throws away the confidence the critic needs).

### Merging in a decoder from a separate checkpoint (`WM_JEPA_MERGE_CHECKPOINT`)

A checkpoint's canonical-event-head training step (`--train-canonical-event-heads-only`) reads
the *base* checkpoint's `jepa_data_manifest.json` to know which optional modules
(`action_decoder`, `obs_grounding`, ...) to carry forward when it saves the head checkpoint. If
the base checkpoint has no manifest (e.g. a bare periodic checkpoint with only
`jepa_checkpoint_meta.json`), those modules are silently **not preserved** — the resulting `_head`
checkpoint has the canonical-event heads `beam_plan` needs, but not the `action_decoder_*` /
`obs_ground_*` weights a base checkpoint may have had trained.

`WM_JEPA_MERGE_CHECKPOINT` works around this without retraining: point it at the *other*
checkpoint (the one with the trained decoder), and its `action_decoder_*`/`obs_ground_*` weights
are loaded into the primary checkpoint's model at startup. Architecture dims (decoder width,
layers, memory tokens, max length) are **inferred directly from the merge checkpoint's own
tensor shapes** — no extra config needed. This assumes head-only training froze everything except
the canonical-event heads (true for `--train-canonical-event-heads-only`), so the merge
checkpoint's backbone/predictor/etc. are expected to match the primary checkpoint's.

```bash
ejepa bench run EnterpriseOps-Gym --executor mcp_react \
  --wm-strategy beam_plan \
  --wm-ewm-jepa-checkpoint /models/jepa_head_ckpt \        # has canonical_event heads
  --wm-jepa-merge-checkpoint /models/jepa_base_ckpt \      # has action_decoder_*/obs_ground_* weights
  --wm-beam-plan-decode-tool-output \                      # see below
  --config target=sample
```

Env: `WM_JEPA_MERGE_CHECKPOINT`, `WM_JEPA_MERGE_DECODE_MAX_NEW_TOKENS` (default per-call decode
length when using the merged decoder).

### Decoding predicted tool output for `beam_plan` (`WM_BEAM_PLAN_DECODE_TOOL_OUTPUT`)

`beam_plan`'s canonical-event heads only ever produce classification labels
(`execution_status=success`, `progress_signal=positive`, ...). `WM_BEAM_PLAN_DECODE_TOOL_OUTPUT`
additionally **decodes the predicted tool-output TEXT** for the winning, confidence-gated
trajectory (via the checkpoint's trained `obs_grounding` decoder — native seq2seq reconstruction,
or the encoder-only `obs_ground_transformer`) and injects it alongside the predicted state:

```
  step 1 action: create_event({"title": "Sync"})
  step 1 predicted state: execution_status=success, progress_signal=positive
  step 1 predicted tool output: {"event_id": "evt_...", "status": "created"}
```

This only runs on the ONE winning trajectory beam_plan already selected (not every scored
candidate) — decoding is a generative greedy loop, one `obs_grounding` forward pass per horizon
step, so it's opt-in (default off) and gated on `beam_confident` exactly like the injection
itself. Requires a checkpoint with a trained `obs_grounding` decoder (native, or merged in via
`WM_JEPA_MERGE_CHECKPOINT` above); without one it logs a warning and falls back to labels-only.

Env: `WM_BEAM_PLAN_DECODE_TOOL_OUTPUT` (`--wm-beam-plan-decode-tool-output`, default off),
`WM_BEAM_PLAN_DECODE_MAX_NEW_TOKENS` (`--wm-beam-plan-decode-max-new-tokens`, default 96).

### Forced imagined rollout as an agent tool (`imagine_trajectory`)

`--wm-strategy imagined` injects a rollout before *every* step (executor-forced, not agent-chosen).
If instead you want the **agent to decide when** to look ahead but to **force a full K-step
rollout** once it does (rather than relying on the agent to chain `predict_state` itself), enable
the `imagine_trajectory` tool with `EWM_IMAGINE_TOOL=1`. It is registered in the agent's tool
catalog; when the agent calls it, the `mcp_react`/`wm_react` executor runs a forced
`WM_IMAGINED_MAX_STEPS`-step imagined rollout — this agent proposes the imagined actions, the EWM
(over MCP when `WM_EWM_MCP_URL` is set) predicts each outcome — and returns the imagined trajectory
as the tool result, which the agent conditions its next real action on.

```bash
export EWM_IMAGINE_TOOL=1                          # register the imagine_trajectory agent tool
export WM_EWM_MCP_URL=http://127.0.0.1:12072       # the rollout's WM calls go over MCP
export WM_IMAGINED_MAX_STEPS=3                      # enforced rollout depth (not agent discretion)
ejepa bench run EnterpriseOps-Gym --executor mcp_react --config target=sample
```

When `EWM_IMAGINE_TOOL=1`, the per-step `predict_state` tool is **dropped** (even if
`EWM_PREDICT_MCP_URL` is set), so `imagine_trajectory` is the *only* EWM tool the agent sees —
the surest way to make it use the rollout. To offer both and let the agent choose, set
`EWM_KEEP_PREDICT_WITH_IMAGINE=1` (the system-prompt nudge then steers it to prefer the rollout).

Each call only looks `WM_IMAGINED_MAX_STEPS` ahead, so one call won't cover a long task. The tool
description, the system-prompt nudge, **and a footer appended to every returned trajectory** tell
the agent to **re-call `imagine_trajectory`** once it acts beyond the imagined horizon — a
receding-horizon, agent-timed re-imagination.

For a **guaranteed** fresh rollout every step while still delivering it through the tool channel,
set `EWM_IMAGINE_EVERY_STEP=1`: the executor auto-runs the rollout before each agent step and
injects a synthetic `imagine_trajectory` tool-call + result (persisted in context, unlike the
transient `--wm-strategy imagined` injection). `EWM_IMAGINE_SUPERSEDE=1` (default) stubs the
previous auto-trajectory in the prompt so context doesn't grow per step (the full history stays in
`conversation_flow`). Combine with `EWM_IMAGINE_TOOL=1` to also keep it agent-callable.

Difference vs. chaining `predict_state`: the agent *can* chain `predict_state` for lookahead, but
depth is its discretion; `imagine_trajectory` makes the K-step rollout **deterministic** once the
agent invokes it (`WM_IMAGINED_MAX_STEPS`). Optional: `EWM_IMAGINE_TOOL_NAME` renames the tool.

### Letting the EnterpriseOps-Gym agent use it

The `mcp_react` executor registers this server as an agent tool when `EWM_PREDICT_MCP_URL`
is set — it appends an `ewm-predict` entry to the task's `gym_servers_config`, adds
`predict_state` to the task's `selected_tools` allowlist (if any), and injects a usage nudge
into the agent's system prompt so the agent is encouraged to look ahead before consequential
actions. Unset by default (existing runs are unchanged).

```bash
# the predict server is up at http://127.0.0.1:12072/mcp (run_local.sh)
export EWM_PREDICT_MCP_URL=http://127.0.0.1:12072
# optional: EWM_PREDICT_MCP_TOKEN=<token>   EWM_PREDICT_MCP_NAME=ewm-predict
ejepa bench run EnterpriseOps-Gym --executor mcp_react --config target=sample
```

Unlike `mcp_react_ewm` (which auto-injects an imagined rollout every step), here EWM is just
another tool the agent may call — the same tool candidate as email/calendar. (`mcp_react_ewm`
is intentionally left as-is: it already consults the world model via imagined rollouts.)

## Plugging in a new WM

1. Add `backends/<name>.py` implementing `WorldModel` (`advise` + `select`) — or subclass
   `LlmWorldModel` with a `chat_fn`.
2. Add its prompts under `prompts/<name>/*.md` (**centralized — never inline prompt text in code**;
   load with `prompts.load_prompt`).
3. Register it in `factory.build_world_model`.
4. Run with `--wm-backend <name>`.

## With / without WM evaluation

The same WM interface makes A/B easy. Run the task set twice — once `--wm-strategy none` (off),
once with your WM (on) — then compare:

```bash
ejepa result wm-compare --off off/wm_eval_records.json --on on/wm_eval_records.json --out-dir report/
```

Each arm is a JSON list of `{"task_id": ..., "success": bool}` (or a dir containing
`wm_eval_records.json`). See `examples/eval_with_without_wm.py` for a self-contained, runnable
demonstration that produces the with/without-WM score table.
