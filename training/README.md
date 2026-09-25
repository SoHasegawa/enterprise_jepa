# Enterprise-JEPA training

The world-model training and data-preparation pipeline, vendored from the EWM research
repository the paper's checkpoints were produced in. It is a separate project from the
evaluation side: the evaluation harness only needs a finished checkpoint directory.

**Start with [`../docs/training.md`](../docs/training.md)** — it is the pipeline in order
(trajectories → canonical-event labels → JEPA latent dynamics → classification heads), the
commands, and the known gap between this snapshot of the trainer and the checkpoint the
paper reports.

```bash
cd training && uv sync
```

| Path | Contents |
|---|---|
| `src/finetuning_jepa.py` | The JEPA net (`TextLeWorldModel`) and its training loop |
| `src/finetuning.py` | Shared example extraction, data loaders, SFT for the LLM world models |
| `src/data_preparation/` | Canonical-event labelling and cleanup, trajectory splits and alignment |
| `src/generation/` | Trajectory generation from benchmark runs |
| `src/evaluation.py`, `src/analysis/` | Replay evaluation and label/field accuracy analysis |
| `src/itp/` | The adaptive-k controller used by the ITP-I harness |
| `scripts/` | Wrappers for k-controller training and ReAct world-model evaluation |

Neither the trajectory corpus nor any checkpoint is tracked here; `training/trajectories/`
and `training/data/*.jsonl` are gitignored.
