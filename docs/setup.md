# Setup

Everything below assumes the repository root as the working directory.

## 1. Environments

```bash
scripts/install.sh            # root env + all five benchmark envs
scripts/install.sh workbench  # or one benchmark at a time
```

`install.sh` creates one `uv` virtualenv per project: the root `.venv` (the `ejepa` CLI and
`ejepa_wm`), and a green/purple pair per benchmark. Benchmarks run their green (scorer) and
purple (agent) sides as separate processes, so they keep separate dependency sets.

The JEPA world model needs `torch` and `transformers`, which are an opt-in extra:

```bash
uv sync --extra jepa                                          # root env
uv sync --project assets/WorkBench/purple --extra jepa        # and per purple env that runs it
```

## 2. External data

```bash
scripts/fetch_assets.sh
```

This clones the upstream benchmark repositories into `upstreams/` and extracts the
CRMArena-Pro SQLite databases from the upstream baseline container image. It then prints the
environment variables that point the adapters at them:

| Variable | Points at |
|---|---|
| `ENTERPRISEOPS_GYM_REPO_PATH` | `upstreams/EnterpriseOps-Gym` |
| `WORKBENCH_REPO_PATH` | `upstreams/WorkBench` |
| `AUTOMATIONBENCH_REPO_PATH` | `upstreams/AutomationBench` |
| `TERMINAL_BENCH_TASK_REPO` | `upstreams/terminal-bench-2` |

EnterpriseOps-Gym additionally needs its HuggingFace task corpus and its MCP tool servers
(`assets/EnterpriseOps-Gym/build_mcp.sh`, then `run_mcp.sh`); Terminal-Bench 2.0 runs one
container per task. Each benchmark's own `README.md` under `assets/` is the authority on its
environment variables.

## 3. Checkpoints

Two checkpoints are training outputs and are not in the repository:

* **Enterprise-JEPA** — the paper uses `data_jepa_heads_partial_imb_terminal_3`: a JEPA net
  with canonical-event classification heads over a text backbone. Place or symlink it at
  `checkpoints/jepa`. The run scripts read `JEPA_CKPT` / `JEPA_CHECKPOINT`, defaulting to
  that path. Training it is `docs/training.md`.
* **State-output LLM world model** — `llm_wm_beam_action_terminal_crmarenapro`, served on an
  OpenAI-compatible endpoint (the protocol uses `:9015`) and selected with
  `--wm-llm-ewm-mode llm_canonical_trained --wm-ewm-model world_model`. Expected at
  `checkpoints/llm_wm_state` when a script needs it on disk (e.g. the latency sweep).

A JEPA checkpoint directory must contain `text_leworldmodel.pt`, `jepa_data_manifest.json`,
`canonical_event_vocab.json`, the tokenizer files, and `backbone/`.

## 4. Serving the models

The paper's runs serve the policy LLM (Qwen3.6-27B) and the LLM world model on separate
vLLM endpoints, and run the JEPA world model in-process on its own GPU:

```bash
export CUDA_VISIBLE_DEVICES=<jepa gpu>      # the JEPA net only
export WM_SHARE_MODEL_WEIGHTS=1             # share JEPA weights across tasks in a process
export WM_VLLM_BASE_URL=http://127.0.0.1:9015/v1 WM_VLLM_API_KEY=EMPTY   # LLM world model
```

`WM_JEPA_COMPILE=1` enables `torch.compile` + CUDA graphs for JEPA inference (≈3-4x faster
per scoring call, ~60 s of one-off compilation). It is rank-equivalent but was **off** for
the reported accuracy runs and should stay off when reproducing them; the production panel
of the latency figure (paper Figure 2) uses it deliberately.

## 5. Check the install

```bash
./ejepa bench list                 # the five benchmarks and their executors
make test                          # harness + world-model unit tests
make lint
```
