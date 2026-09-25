# Release models

The two trained models the paper's agentic results use. They are staged here for upload as
release assets and are **not** tracked in git (`.gitignore` keeps everything in this
directory except this README).

| Directory | What it is | Paper | Size |
|---|---|---|--:|
| `enterprise_jepa/` | Enterprise-JEPA: Stage-1 latent dynamics + Stage-2 operational-state heads, expanded corpus | the Enterprise-JEPA rows of Tables 3, 4, 5, 10 and Figures 2, 4; its metric files are the source of Table 2 | 1.8 GB |
| `llm_wm_state_output/` | State-output LLM world model, a Qwen3 0.6B fine-tune that generates the operational state as text | the state-output LLM-WM rows of Table 3 and Figures 2, 4 | 1.2 GB |

Largest single file is 1.11 GB, so each directory fits GitHub's 2 GB per-asset limit when
archived on its own. Archive them separately, not as one tarball.

`SHA256SUMS` in each directory covers every file in that directory:

```bash
cd enterprise_jepa && sha256sum -c SHA256SUMS
```

## Using them

Enterprise-JEPA is loaded in-process by the world model; point the flag at the unpacked
directory:

```bash
ejepa bench run <BENCH> --executor mcp_react \
  --wm-strategy beam_plan \
  --wm-ewm-jepa-checkpoint model/enterprise_jepa \
  --wm-jepa-observation-backend canonical_event
```

The run scripts read the same path from `JEPA_CKPT` / `JEPA_CHECKPOINT`, defaulting to
`checkpoints/jepa`, so either unpack there or set the variable.

The state-output LLM world model is served rather than loaded in-process:

```bash
vllm serve model/llm_wm_state_output --served-model-name world_model --port 9015
export WM_VLLM_BASE_URL=http://127.0.0.1:9015/v1 WM_VLLM_API_KEY=EMPTY
```

and selected with `--wm-llm-ewm-mode llm_canonical_trained --wm-ewm-model world_model`.

## What is in each directory

**`enterprise_jepa/`** — `text_leworldmodel.pt` (predictor, projector and the seven
classification heads), `backbone/` (the Qwen3-Embedding-0.6B encoder, 28 layers, hidden
1024), the tokenizer files, `jepa_data_manifest.json` and `canonical_event_vocab.json`
(the architecture and label vocabularies the loader reads), and the two metric files
`canonical_event_training_metrics.json` and `run_summary.json`.

**`llm_wm_state_output/`** — `model.safetensors` with its `config.json`, generation config
and tokenizer, plus `training_metrics.json`, `run_summary.json` and `split_manifest.json`.

Mid-training checkpoint directories, optimizer state, the tokenized cache and the
training-data dumps were left out; nothing in either directory is needed for training, only
for inference and for reading the reported metrics.
