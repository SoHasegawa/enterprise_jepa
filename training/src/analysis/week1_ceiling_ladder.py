#!/usr/bin/env python3
"""Week 1.2 -- ceiling ladder: locate the actual bottleneck behind the ~0.73 plateau.

Implements plan.md's ladder, one row per input configuration, all trained with an IDENTICAL
head/protocol/seed on identical rows so the differences are attributable to the input:

  row                     head input                              isolates
  ----------------------  --------------------------------------  ---------------------------
  majority                class prior (no model)                  class imbalance
  predicted               [z_cur,z_act,z_ctx,z_pred]              full model (deployed)
  target_obs              [z_obs]                                 representation/head ceiling
  full_context_obs        [z_cur,z_act,z_ctx,z_obs]               is oracle missing task context
  attn_pooled_obs         [z_cur,z_act,z_ctx, attn(obs tokens)]   global-vector bottleneck
  backbone_direct         raw pooled features, projector skipped  does JEPA projection lose info
  (+ --adjudicated-labels re-scores any row against human labels) annotation noise

Metrics per plan.md: macro-F1, balanced accuracy, NLL, Brier and ECE alongside accuracy.

SCOPE: uses the EXISTING 11-field schema unchanged (plan.md's schema restructuring is
deliberately not applied). Attention pooling is applied to the OBSERVATION tokens, which is
where plan.md points the global-vector question, and is cheap because observations are short.

Two stages (encode once, then train all rows):
  uv run python src/analysis/week1_ceiling_ladder.py --stage encode --device cuda:2
  uv run python src/analysis/week1_ceiling_ladder.py --stage train  --device cuda:2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

import src.finetuning_jepa as fj
from src.analysis.jepa_readout_ablation import load_rows as load_probe_rows

TRAJ = REPO_ROOT / "trajectories"
STEM = "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro"
DEFAULT_CHECKPOINT = Path("checkpoints/data_jepa_all_adp_06b_decoder_msp")
DEFAULT_WORKDIR = Path("checkpoints/week1_ceiling_ladder")

SINGLE_FIELDS = list(fj.CANONICAL_EVENT_SINGLE_LABEL_FIELDS)
MULTI_FIELDS = list(fj.NUDGE_MULTI_LABEL_FIELDS)
OUTCOME_FIELDS = ["execution_status", "error_signature", "progress_signal", "side_effect_type"]
OBS_MAX_TOKENS = 128  # observations are labeler-truncated to ~200 chars; 128 tokens covers them


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("encode", "train"), required=True)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--max-batch", type=int, default=48)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adjudicated-labels", type=Path, default=None,
                   help="Filled audit template; adds an annotation-noise row scored on those rows only.")
    return p.parse_args()


# --------------------------------------------------------------------------- encode
@torch.no_grad()
def encode(model, tokenizer, texts, max_length, keep_newest, device, max_batch, want_tokens=False):
    """Returns (pooled_pre_projector, projected_latent, [token_features, token_mask])."""
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    pooled_out = projected_out = None
    tok_out = mask_out = None
    prev = tokenizer.truncation_side
    tokenizer.truncation_side = "left" if keep_newest else "right"
    try:
        for start in range(0, len(order), max_batch):
            idx = order[start:start + max_batch]
            enc = tokenizer([texts[i] for i in idx], return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length, add_special_tokens=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            pooled = model._encode_pooled(input_ids, attention_mask)
            projected, _ = model._project_latent(pooled)
            if pooled_out is None:
                pooled_out = torch.empty(len(texts), pooled.shape[-1], dtype=torch.float32)
                projected_out = torch.empty(len(texts), projected.shape[-1], dtype=torch.float32)
            sel = torch.tensor(idx)
            pooled_out[sel] = pooled.float().cpu()
            projected_out[sel] = projected.float().cpu()
            if want_tokens:
                hidden = fj.backbone_encoder(model.backbone)(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state
                if tok_out is None:
                    tok_out = torch.zeros(len(texts), OBS_MAX_TOKENS, hidden.shape[-1], dtype=torch.float16)
                    mask_out = torch.zeros(len(texts), OBS_MAX_TOKENS, dtype=torch.bool)
                length = min(hidden.shape[1], OBS_MAX_TOKENS)
                tok_out[sel, :length] = hidden[:, :length].half().cpu()
                mask_out[sel, :length] = attention_mask[:, :length].bool().cpu()
    finally:
        tokenizer.truncation_side = prev
    return (pooled_out, projected_out, tok_out, mask_out)


def stage_encode(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    manifest = json.loads((args.checkpoint / "jepa_data_manifest.json").read_text())
    max_input = int(manifest.get("max_input_length") or 2048)
    max_action = int(manifest.get("max_action_length") or 512)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint))
    backbone = fj.load_jepa_backbone(args.checkpoint / "backbone", backbone_type=manifest["backbone_type"],
                                     trust_remote_code=True, dtype=torch.bfloat16)
    model = fj.TextLeWorldModel(
        backbone=backbone, latent_dim=int(manifest.get("latent_dim") or 0),
        memory_tokens=int(manifest.get("memory_tokens") or 8), dropout=0.0,
        predictor_hidden_multiplier=float(manifest.get("predictor_hidden_multiplier") or 4.0),
        goal_conditioning=bool(manifest.get("goal_conditioning", False)),
        latent_type=str(manifest.get("latent_type") or "continuous"),
        pooling=str(manifest.get("pooling") or "mean"),
    )
    fj.load_jepa_state_dict_for_training(model, args.checkpoint, allow_missing_success_head=True,
                                         allow_missing_canonical_event_heads=True)
    for param in model.parameters():          # true inference pass (see ablation script note)
        param.requires_grad = False
    model.to(device).eval()

    args.workdir.mkdir(parents=True, exist_ok=True)
    originals = []
    for split in ("train", "eval"):
        originals.extend(fj.build_canonical_event_examples(
            [json.loads(l) for l in (TRAJ / f"{STEM}_{split}_examples_cleaned_value_scored.jsonl").open() if l.strip()]))
    vocab = fj.build_canonical_event_vocabularies(originals)
    (args.workdir / "vocab.json").write_text(json.dumps(vocab, indent=1))

    for split in ("train", "eval"):
        examples, obs_texts = load_probe_rows(split)
        print(f"[{split}] {len(examples)} rows -> encoding", flush=True)
        pooled_cur, z_cur, _, _ = encode(model, tokenizer, [e.current_state_text for e in examples], max_input, False, device, args.max_batch)
        pooled_ctx, z_ctx, _, _ = encode(model, tokenizer, [e.context_text for e in examples], max_input, False, device, args.max_batch)
        pooled_act, z_act, _, _ = encode(model, tokenizer, [e.action_text for e in examples], max_action, False, device, args.max_batch)
        pooled_obs, z_obs, obs_tokens, obs_mask = encode(
            model, tokenizer, obs_texts, OBS_MAX_TOKENS, False, device, args.max_batch, want_tokens=True)
        with torch.no_grad():
            preds = []
            for start in range(0, len(examples), 1024):
                sl = slice(start, start + 1024)
                z, _ = model.predict_latent(z_cur[sl].to(device), z_act[sl].to(device), z_ctx[sl].to(device))
                preds.append(z.float().cpu())
            z_pred = torch.cat(preds)
        torch.save({
            "labels": {**{f: torch.tensor([{v: i for i, v in enumerate(vocab[f])}[e.single_labels[f]] for e in examples])
                          for f in SINGLE_FIELDS},
                       **{f: torch.stack([_multi_hot(e.multi_labels[f], vocab[f]) for e in examples]) for f in MULTI_FIELDS}},
            "z": {"cur": z_cur, "ctx": z_ctx, "act": z_act, "obs": z_obs, "pred": z_pred},
            "pooled": {"cur": pooled_cur, "ctx": pooled_ctx, "act": pooled_act, "obs": pooled_obs},
            "obs_tokens": obs_tokens, "obs_mask": obs_mask,
            "keys": [(e.trajectory_id, e.interaction_index) for e in examples],
        }, args.workdir / f"ladder_{split}.pt")
        print(f"[{split}] saved -> {args.workdir / f'ladder_{split}.pt'}", flush=True)


def _multi_hot(values: list[str], vocab: list[str]) -> torch.Tensor:
    hot = torch.zeros(len(vocab))
    index = {v: i for i, v in enumerate(vocab)}
    for v in values:
        if v in index:
            hot[index[v]] = 1.0
    return hot


# --------------------------------------------------------------------------- heads
class Heads(nn.Module):
    def __init__(self, input_dim: int, vocab: dict[str, list[str]], hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({f: nn.Linear(hidden, len(v)) for f, v in vocab.items()})

    def forward(self, x):
        t = self.trunk(x)
        return {f: h(t) for f, h in self.heads.items()}


class AttnPoolHeads(nn.Module):
    """Learned-query attention pooling over observation TOKEN features, concatenated with the
    (global) context/action/state latents. Tests plan.md's global-vector-bottleneck question:
    if attending over tokens beats the pooled observation vector, pooling is losing information."""

    def __init__(self, global_dim: int, token_dim: int, vocab, hidden: int = 512, dropout: float = 0.1, queries: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(queries, token_dim) * 0.02)
        self.attn = nn.MultiheadAttention(token_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)
        self.body = Heads(global_dim + queries * token_dim, vocab, hidden, dropout)

    def forward(self, globals_, tokens, mask):
        batch = tokens.shape[0]
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        pooled, _ = self.attn(query, tokens, tokens, key_padding_mask=~mask)
        pooled = self.norm(pooled).reshape(batch, -1)
        return self.body(torch.cat([globals_, pooled], dim=-1))


# --------------------------------------------------------------------------- metrics
def _macro_f1(gold: torch.Tensor, pred: torch.Tensor, num_classes: int) -> float:
    scores = []
    for c in range(num_classes):
        tp = int(((gold == c) & (pred == c)).sum())
        fp = int(((gold != c) & (pred == c)).sum())
        fn = int(((gold == c) & (pred != c)).sum())
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(scores) / len(scores) if scores else float("nan")


def _balanced_accuracy(gold: torch.Tensor, pred: torch.Tensor, num_classes: int) -> float:
    recalls = []
    for c in range(num_classes):
        support = int((gold == c).sum())
        if support:
            recalls.append(int(((gold == c) & (pred == c)).sum()) / support)
    return sum(recalls) / len(recalls) if recalls else float("nan")


def _ece(probs: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> float:
    confidence = probs.max(-1).values
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = (confidence > lo) & (confidence <= hi)
        n = int(sel.sum())
        if n:
            total += n / len(confidence) * abs(float(correct[sel].float().mean()) - float(confidence[sel].mean()))
    return total


def score_single(logits: torch.Tensor, gold: torch.Tensor) -> dict[str, float]:
    probs = torch.softmax(logits.float(), dim=-1)
    pred = probs.argmax(-1)
    correct = pred == gold
    one_hot = torch.zeros_like(probs).scatter_(1, gold.unsqueeze(1), 1.0)
    return {
        "accuracy": float(correct.float().mean()),
        "macro_f1": _macro_f1(gold, pred, probs.shape[-1]),
        "balanced_accuracy": _balanced_accuracy(gold, pred, probs.shape[-1]),
        "nll": float(-torch.log(probs.gather(1, gold.unsqueeze(1)).clamp_min(1e-12)).mean()),
        "brier": float(((probs - one_hot) ** 2).sum(-1).mean()),
        "ece": _ece(probs, correct),
    }


def score_multi(logits: torch.Tensor, gold: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits.float())
    pred = (probs >= 0.5).float()
    return {
        "accuracy": float((pred == gold).all(-1).float().mean()),   # exact match
        "label_accuracy": float((pred == gold).float().mean()),      # element-wise
        "macro_f1": float(sum(
            (lambda tp, fp, fn: 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)(
                float(((pred[:, c] == 1) & (gold[:, c] == 1)).sum()),
                float(((pred[:, c] == 1) & (gold[:, c] == 0)).sum()),
                float(((pred[:, c] == 0) & (gold[:, c] == 1)).sum()))
            for c in range(gold.shape[1])) / gold.shape[1]),
        "brier": float(((probs - gold) ** 2).mean()),
        "nll": float(nn.functional.binary_cross_entropy(probs.clamp(1e-6, 1 - 1e-6), gold)),
        "ece": float("nan"),
    }


def summarize(per_field: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        "outcome4_accuracy": sum(per_field[f]["accuracy"] for f in OUTCOME_FIELDS) / len(OUTCOME_FIELDS),
        "outcome4_macro_f1": sum(per_field[f]["macro_f1"] for f in OUTCOME_FIELDS) / len(OUTCOME_FIELDS),
        "macro_accuracy": sum(v["accuracy"] for v in per_field.values()) / len(per_field),
        "macro_f1": sum(v["macro_f1"] for v in per_field.values()) / len(per_field),
        "mean_ece": sum(v["ece"] for v in per_field.values() if not math.isnan(v["ece"]))
                    / max(sum(1 for v in per_field.values() if not math.isnan(v["ece"])), 1),
    }


# --------------------------------------------------------------------------- train
def train_and_eval(make_model, train_batch, eval_batch, labels_tr, labels_ev, vocab, args, device):
    torch.manual_seed(args.seed)
    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = train_batch[0].shape[0]
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed + epoch))
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            out = model(*[t[idx].to(device) for t in train_batch])
            loss = sum(nn.functional.cross_entropy(out[f], labels_tr[f][idx].to(device)) for f in SINGLE_FIELDS)
            loss = loss + sum(nn.functional.binary_cross_entropy_with_logits(out[f], labels_tr[f][idx].to(device)) for f in MULTI_FIELDS)
            loss = loss / (len(SINGLE_FIELDS) + len(MULTI_FIELDS))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    logits = {f: [] for f in SINGLE_FIELDS + MULTI_FIELDS}
    with torch.no_grad():
        for start in range(0, eval_batch[0].shape[0], 1024):
            out = model(*[t[start:start + 1024].to(device) for t in eval_batch])
            for f in logits:
                logits[f].append(out[f].cpu())
    logits = {f: torch.cat(v) for f, v in logits.items()}
    per_field = {f: score_single(logits[f], labels_ev[f]) for f in SINGLE_FIELDS}
    per_field.update({f: score_multi(logits[f], labels_ev[f]) for f in MULTI_FIELDS})
    return per_field, logits


def majority_row(labels_tr, labels_ev, vocab) -> dict[str, dict[str, float]]:
    per_field = {}
    for f in SINGLE_FIELDS:
        counts = torch.bincount(labels_tr[f], minlength=len(vocab[f])).float()
        prior = (counts / counts.sum()).unsqueeze(0).expand(labels_ev[f].shape[0], -1)
        per_field[f] = score_single(torch.log(prior.clamp_min(1e-12)), labels_ev[f])
    for f in MULTI_FIELDS:
        rate = labels_tr[f].mean(0, keepdim=True).expand(labels_ev[f].shape[0], -1)
        per_field[f] = score_multi(torch.logit(rate.clamp(1e-6, 1 - 1e-6)), labels_ev[f])
    return per_field


def stage_train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vocab = json.loads((args.workdir / "vocab.json").read_text())
    tr = torch.load(args.workdir / "ladder_train.pt", weights_only=False)
    ev = torch.load(args.workdir / "ladder_eval.pt", weights_only=False)
    ltr, lev = tr["labels"], ev["labels"]
    results: dict[str, Any] = {}

    def cat(d, parts, key="z"):
        return torch.cat([d[key][p] for p in parts], dim=-1)

    rows = [
        ("majority", None),
        ("predicted", (("z", ("cur", "act", "ctx", "pred")),)),
        ("target_obs", (("z", ("obs",)),)),
        ("full_context_obs", (("z", ("cur", "act", "ctx", "obs")),)),
        ("backbone_direct", (("pooled", ("cur", "act", "ctx", "obs")),)),
    ]
    for name, spec in rows:
        if spec is None:
            per_field = majority_row(ltr, lev, vocab)
        else:
            key, parts = spec[0]
            x_tr, x_ev = cat(tr, parts, key), cat(ev, parts, key)
            per_field, _ = train_and_eval(lambda d=x_tr.shape[-1]: Heads(d, vocab),
                                          (x_tr,), (x_ev,), ltr, lev, vocab, args, device)
        results[name] = {"per_field": per_field, **summarize(per_field)}
        s = results[name]
        print(f"[{name:18s}] outcome4_acc={s['outcome4_accuracy']:.4f} outcome4_F1={s['outcome4_macro_f1']:.4f} "
              f"macroF1={s['macro_f1']:.4f} ECE={s['mean_ece']:.4f}", flush=True)

    # attention-pooled observation tokens + global context/action/state
    g_tr = cat(tr, ("cur", "act", "ctx")); g_ev = cat(ev, ("cur", "act", "ctx"))
    per_field, _ = train_and_eval(
        lambda: AttnPoolHeads(g_tr.shape[-1], tr["obs_tokens"].shape[-1], vocab),
        (g_tr, tr["obs_tokens"].float(), tr["obs_mask"]),
        (g_ev, ev["obs_tokens"].float(), ev["obs_mask"]),
        ltr, lev, vocab, args, device)
    results["attn_pooled_obs"] = {"per_field": per_field, **summarize(per_field)}
    s = results["attn_pooled_obs"]
    print(f"[{'attn_pooled_obs':18s}] outcome4_acc={s['outcome4_accuracy']:.4f} outcome4_F1={s['outcome4_macro_f1']:.4f} "
          f"macroF1={s['macro_f1']:.4f} ECE={s['mean_ece']:.4f}", flush=True)

    # annotation-noise row: re-score the strongest arm against adjudicated labels
    if args.adjudicated_labels and args.adjudicated_labels.is_file():
        index = {tuple(k): i for i, k in enumerate(ev["keys"])}
        subset, human = [], {f: [] for f in SINGLE_FIELDS}
        for line in args.adjudicated_labels.open():
            entry = json.loads(line)
            key = (str(entry["trajectory_id"]), int(entry.get("interaction_index") or 0))
            labels = entry.get("adjudicated") or {}
            if key in index and all(labels.get(f) is not None for f in SINGLE_FIELDS):
                subset.append(index[key])
                for f in SINGLE_FIELDS:
                    human[f].append(vocab[f].index(labels[f]))
        if subset:
            sel = torch.tensor(subset)
            x_tr = cat(tr, ("cur", "act", "ctx", "obs")); x_ev = cat(ev, ("cur", "act", "ctx", "obs"))
            _, logits = train_and_eval(lambda d=x_tr.shape[-1]: Heads(d, vocab), (x_tr,), (x_ev,), ltr, lev, vocab, args, device)
            gpt_pf = {f: score_single(logits[f][sel], lev[f][sel]) for f in SINGLE_FIELDS}
            hum_pf = {f: score_single(logits[f][sel], torch.tensor(human[f])) for f in SINGLE_FIELDS}
            results["adjudicated_subset"] = {
                "n": len(subset),
                "vs_gpt_labels": {f: gpt_pf[f]["accuracy"] for f in SINGLE_FIELDS},
                "vs_human_labels": {f: hum_pf[f]["accuracy"] for f in SINGLE_FIELDS},
            }
            print(f"[adjudicated n={len(subset)}] mean acc vs GPT labels="
                  f"{sum(gpt_pf[f]['accuracy'] for f in SINGLE_FIELDS)/len(SINGLE_FIELDS):.4f} "
                  f"vs human={sum(hum_pf[f]['accuracy'] for f in SINGLE_FIELDS)/len(SINGLE_FIELDS):.4f}")

    args.workdir.mkdir(parents=True, exist_ok=True)
    out = args.workdir / "ceiling_ladder_results.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"\nresults -> {out}")


def main() -> None:
    args = parse_args()
    (stage_encode if args.stage == "encode" else stage_train)(args)


if __name__ == "__main__":
    main()
