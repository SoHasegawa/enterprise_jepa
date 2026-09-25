#!/usr/bin/env python3
"""Outcome-4 accuracy per ablation condition, for a --predictor-arch transformer checkpoint.

The transformer analogue of the table week1_action_sensitivity.py prints for the concat-MLP
predictor. Same conditions, same metric (mean accuracy over
OUTCOME_FIELDS = execution_status, error_signature, progress_signal, side_effect_type), so the
two are read the same way: if every row is within noise of `normal`, the predictor's output
carries no action information the heads can use.

Conditions, mapped onto the transformer's actual inputs (the action is AdaLN conditioning, not
an input vector, and z_current is not a token when a frame history is present):

  normal           h_t = F([ctx, o_{t-N+1..t}]; a_t)
  zero_action      the candidate action latent is zeroed
  shuffled_action  candidate actions permuted across rows (same states, wrong actions)
  persistence      the predictor is bypassed entirely; the heads read z_current
  context_only     no history, zeroed frame, zeroed action -- the context token alone
  action_only      the raw action embedding z_action, no predictor -- label leakage from the
                   action text itself

TWO tables are produced, because they answer different questions and are NOT interchangeable:

  frozen   the checkpoint's OWN trained heads, evaluated under each condition. "Does the
           deployed system's accuracy depend on the action?"
  probe    a fresh linear head fitted per condition on a held-out split. "Does the conditioned
           representation CONTAIN outcome information?" This is what week1_action_sensitivity
           reports, so only this row is numerically comparable to that table.

Usage:
  uv run python src/analysis/transformer_predictor_condition_table.py \
      --checkpoint checkpoints/data_jepa_heads_ensemble \
      --jsonl trajectories/..._eval_examples_cleaned_ensemble_value_scored.jsonl \
      --rows 1200 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

import src.finetuning_jepa as fj
from src.analysis.transformer_predictor_action_sensitivity import build_model, encode

OUTCOME_FIELDS = ["execution_status", "error_signature", "progress_signal", "side_effect_type"]
# --fields all: every head the checkpoint carries. missing_information_type is multi-label --
# scored as sigmoid>=0.5 EXACT MATCH, the same rule the trainer's eval uses, so its numbers are
# comparable with run_summary.json's.
SINGLE_FIELDS = list(fj.CANONICAL_EVENT_SINGLE_LABEL_FIELDS)
MULTI_FIELDS = list(fj.NUDGE_MULTI_LABEL_FIELDS)
ALL_FIELDS = SINGLE_FIELDS + MULTI_FIELDS


def build_batch_with_checkpoint_vocab(
    jsonl: Path, tokenizer: Any, manifest: dict, checkpoint: Path, rows: int,
    fields: list[str],
):
    """Tokenize rows AND map labels through the checkpoint's own vocabulary.

    Rebuilding the vocab from the sampled rows (as the sensitivity script does, where only
    argmax CHANGES matter) would index the label space differently from the trained heads and
    silently produce meaningless accuracies.
    """
    vocab_path = checkpoint / "canonical_event_vocab.json"
    if not vocab_path.is_file():
        raise SystemExit(f"{vocab_path} missing; cannot align labels with the trained heads.")
    vocab = json.loads(vocab_path.read_text())
    records = []
    with jsonl.open() as handle:
        for line in handle:
            if len(records) >= rows:
                break
            if line.strip():
                records.append(json.loads(line))
    examples = [e for e in (fj.build_canonical_event_example(r) for r in records) if e is not None]
    # Drop rows carrying a label outside the trained vocabulary rather than remapping them.
    def in_vocab(ex) -> bool:
        for f in fields:
            if f in MULTI_FIELDS:
                if any(v not in vocab.get(f, []) for v in ex.multi_labels.get(f, [])):
                    return False
            elif ex.single_labels.get(f) not in vocab.get(f, []):
                return False
        return True

    keep = [ex for ex in examples if in_vocab(ex)]
    dropped = len(examples) - len(keep)
    args = argparse.Namespace(
        max_input_length=int(manifest.get("max_input_length") or 8192),
        max_action_length=int(manifest.get("max_action_length") or 512),
        max_observation_length=int(manifest.get("max_observation_length") or 512),
        truncate_states_keep_newest=bool(manifest.get("truncate_states_keep_newest", False)),
        canonical_event_recognition_probe=False,
        joint_canonical_event_training=False,
        predictor_arch="transformer",
        recurrent_state_init=False,
    )
    dataset = fj.CanonicalEventDataset(keep, tokenizer, args, vocab)
    batch = fj.CanonicalEventCollator(tokenizer)([dataset[i] for i in range(len(keep))])
    return batch, keep, vocab, dropped


@torch.no_grad()
def condition_features(model, z_current, z_context, z_action, history, seed: int = 0):
    """{condition: features the heads read}. In `state` mode that is h_t; persistence bypasses
    the predictor entirely, which is the whole point of that row."""
    perm = torch.randperm(z_action.shape[0], generator=torch.Generator().manual_seed(seed)).to(z_action.device)
    zeros_a = torch.zeros_like(z_action)
    out = {}
    _, out["normal"] = model._predict_latent_transformer(z_current, z_action, z_context, None, history)
    _, out["zero_action"] = model._predict_latent_transformer(z_current, zeros_a, z_context, None, history)
    _, out["shuffled_action"] = model._predict_latent_transformer(z_current, z_action[perm], z_context, None, history)
    out["persistence"] = z_current
    _, out["context_only"] = model._predict_latent_transformer(
        torch.zeros_like(z_current), zeros_a, z_context, None, None)
    # Raw action embedding, no predictor: how much of each label is readable off the action
    # TEXT alone. The leakage counterpart of `persistence` (raw z_current) -- large values on
    # outcome fields would mean the heads could score actions without any world model.
    out["action_only"] = z_action
    return out


@torch.no_grad()
def frozen_head_accuracy(model, features, labels, vocab, fields) -> dict[str, float]:
    logits = {f: head(model.canonical_event_trunk(features)) for f, head in model.canonical_event_heads.items()}
    out = {}
    for field in fields:
        if field not in logits or field not in labels:
            continue
        if field in MULTI_FIELDS:
            predicted = (torch.sigmoid(logits[field].float()) >= 0.5)
            out[field] = float((predicted == labels[field].bool()).all(dim=-1).float().mean())
        else:
            out[field] = float((logits[field].argmax(-1) == labels[field]).float().mean())
    return out


def probe_head_accuracy(features, labels, vocab, split: float, epochs: int, seed: int, fields) -> dict[str, float]:
    """Fit a fresh linear head per field -- the week1_action_sensitivity methodology."""
    torch.manual_seed(seed)
    n = features.shape[0]
    cut = int(n * split)
    x_tr, x_ev = features[:cut].float(), features[cut:].float()
    accuracy = {}
    for field in fields:
        if field not in labels:
            continue
        y = labels[field]
        y_tr, y_ev = y[:cut], y[cut:]
        head = nn.Linear(x_tr.shape[-1], len(vocab[field])).to(x_tr.device)
        opt = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-4)
        multi = field in MULTI_FIELDS
        for _ in range(epochs):
            opt.zero_grad()
            if multi:
                nn.functional.binary_cross_entropy_with_logits(head(x_tr), y_tr.float()).backward()
            else:
                nn.functional.cross_entropy(head(x_tr), y_tr).backward()
            opt.step()
        with torch.no_grad():
            if multi:
                predicted = torch.sigmoid(head(x_ev)) >= 0.5
                accuracy[field] = float((predicted == y_ev.bool()).all(dim=-1).float().mean())
            else:
                accuracy[field] = float((head(x_ev).argmax(-1) == y_ev).float().mean())
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--encode-chunk", type=int, default=4)
    parser.add_argument("--probe-split", type=float, default=0.6)
    parser.add_argument("--probe-epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--fields", choices=("outcome4", "all"), default="outcome4",
                        help="outcome4 = the 4 outcome fields; all = every head the checkpoint has "
                             "(missing_information_type scored as sigmoid>=0.5 exact match).")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    fields = ALL_FIELDS if args.fields == "all" else OUTCOME_FIELDS

    device = torch.device(args.device)
    model, tokenizer, manifest = build_model(args.checkpoint, device, getattr(torch, args.dtype))
    batch, examples, vocab, dropped = build_batch_with_checkpoint_vocab(
        args.jsonl, tokenizer, manifest, args.checkpoint, args.rows, fields)
    print(f"[setup] {len(examples)} rows ({dropped} dropped for out-of-vocab labels) | "
          f"head_inputs={manifest.get('canonical_event_head_inputs')!r}", flush=True)

    z_current, z_context, z_action, history = encode(model, batch, device, args.encode_chunk)
    labels = {f: batch[f"label_{f}"].to(device) for f in fields if f"label_{f}" in batch}
    features = condition_features(model, z_current, z_context, z_action, history, args.seed)

    majority = {}
    for f in fields:
        if f in MULTI_FIELDS:
            # Exact-match majority: the most frequent full label SET.
            rows_as_tuples = [tuple(v) for v in labels[f].cpu().int().tolist()]
            majority[f] = max(rows_as_tuples.count(t) for t in set(rows_as_tuples)) / len(rows_as_tuples)
        else:
            majority[f] = float(torch.bincount(labels[f], minlength=len(vocab[f])).max() / labels[f].numel())
    mean_label = "outcome-4" if args.fields == "outcome4" else f"all-{len(fields)}"
    print(f"\nmajority-class baseline ({mean_label} mean): {sum(majority.values())/len(majority):.4f}")

    report: dict[str, Any] = {"rows": len(examples), "majority_baseline": majority, "conditions": {}}
    print(f"\n{'condition':22s} {'mean (frozen heads)':>26s} {'mean (fresh probe)':>25s}")
    for name, feat in features.items():
        frozen = frozen_head_accuracy(model, feat, labels, vocab, fields)
        probe = probe_head_accuracy(feat, labels, vocab, args.probe_split, args.probe_epochs, args.seed, fields)
        f4 = sum(frozen.values()) / len(frozen)
        p4 = sum(probe.values()) / len(probe)
        report["conditions"][name] = {"frozen_per_field": frozen, "frozen_outcome4": f4,
                                      "probe_per_field": probe, "probe_outcome4": p4}
        print(f"{name:22s} {f4:26.4f} {p4:25.4f}")

    for label, key in (("frozen heads", "frozen_per_field"), ("fresh probe", "probe_per_field")):
        print(f"\nper-field ({label}):")
        print(f"{'condition':22s} " + " ".join(f"{f[:14]:>15s}" for f in fields))
        for name in features:
            row = report["conditions"][name][key]
            print(f"{name:22s} " + " ".join(f"{row.get(f, float('nan')):15.4f}" for f in fields))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=1))
        print(f"\nreport -> {args.output}")


if __name__ == "__main__":
    main()
