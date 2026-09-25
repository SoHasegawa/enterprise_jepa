#!/usr/bin/env python3
"""Matched readout-ablation suite: what limits prediction-mode canonical-event accuracy?

Decomposes the prediction-vs-recognition gap into its components by training IDENTICAL
classification heads (same data rows, protocol, seed) on different input configurations,
all derived from one frozen JEPA checkpoint:

  arm            head input                      isolates
  -------------  ------------------------------  -------------------------------------------
  x_only         [z_cur, z_act, z_ctx]           predictability without the predictor
  pred_only      [z_pred]                        label info encoded by the predictor alone
  target_only    [z_target]                      label info in the TRUE next-state latent
  obs_only       [z_obs]                         label info in the raw observation latent
  x_pred         [z_cur, z_act, z_ctx, z_pred]   deployed setting
  x_target       [z_cur, z_act, z_ctx, z_target] oracle: perfect predictor substituted
  x_obs          [z_cur, z_act, z_ctx, z_obs]    recognition analog (matched protocol)

Eval-time controls on the trained x_pred heads (no retraining): z_pred permuted across rows
and z_pred zeroed -- if accuracy barely moves, the heads bypass the predictor and the deployed
setting cannot measure predictor quality (the concat-bypass critique).

Key readings:
  * x_target - x_pred   : headroom recoverable by a PERFECT predictor (predictor approximation
                          error as seen through this readout).
  * x_obs - x_target    : information the pooled next-STATE latent loses relative to the
                          observation latent (JEPA-target dilution).
  * x_pred - x_only     : marginal value of the current predictor.
  * x_pred vs shuffled  : whether the heads use z_pred at all.
McNemar paired tests are reported for these comparisons on execution_status.

Latents come from the frozen champion trunk (data_jepa_all_adp_06b_decoder_msp by default).
Encoding regimes: current/context/action use the trunk's NATIVE regime (max_input_length from
its manifest, right truncation -- deployment-faithful). z_target uses keep-newest (left)
truncation so the appended outcome text is guaranteed inside the window; without this the
known right-truncation defect would blind the oracle arm on ~1/3 of rows and understate
predictor headroom. Rows: the recognition-probe JSONLs (non-terminal steps with recoverable
observations), with next-state text taken from the ORIGINAL files' successor rows.

Usage:
  # 1) encode latents (GPU, ~30-60 min)
  uv run python src/analysis/jepa_readout_ablation.py --stage encode --device cuda:2
  # 2) train all arms + report (fast)
  uv run python src/analysis/jepa_readout_ablation.py --stage train
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

TRAJ = REPO_ROOT / "trajectories"
STEM = "canonical_event_with_nudge_llm_enterpriseops_gym_terminalbench_2_0_crmarenapro"
ORIGINAL = {s: TRAJ / f"{STEM}_{s}_examples_cleaned_value_scored.jsonl" for s in ("train", "eval")}
PROBE = {s: TRAJ / f"{STEM}_{s}_examples_cleaned_value_scored_recognition_probe.jsonl" for s in ("train", "eval")}
DEFAULT_CHECKPOINT = Path("checkpoints/data_jepa_all_adp_06b_decoder_msp")
DEFAULT_WORKDIR = Path("checkpoints/jepa_readout_ablation_msp")

SINGLE_FIELDS = list(fj.CANONICAL_EVENT_SINGLE_LABEL_FIELDS)
MULTI_FIELDS = list(fj.NUDGE_MULTI_LABEL_FIELDS)
OUTCOME_FIELDS = ["execution_status", "error_signature", "progress_signal", "side_effect_type"]

ARMS = {
    "x_only": ("cur", "act", "ctx"),
    "pred_only": ("pred",),
    "target_only": ("target",),
    "obs_only": ("obs",),
    "x_pred": ("cur", "act", "ctx", "pred"),
    "x_target": ("cur", "act", "ctx", "target"),
    "x_obs": ("cur", "act", "ctx", "obs"),
    # Context-conditioned readouts WITHOUT the z_cur/z_act skip. The x_* arms hand the head
    # the current state and the action directly, so a head can reach high accuracy while the
    # predicted latent contributes nothing -- exactly the confound the revised plan calls
    # "downstream skip connections concealing action-blindness". These two arms remove that
    # path: whatever they score is information genuinely carried by z_pred (or by the true
    # z_target), given only the task context.
    "ctx_pred": ("ctx", "pred"),
    "ctx_target": ("ctx", "target"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("encode", "train"), required=True)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    p.add_argument("--device", default="cuda:2")
    p.add_argument("--token-budget", type=int, default=32768, help="max tokens per encoding batch")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# --------------------------------------------------------------------------- data
def load_rows(split: str) -> tuple[list[Any], list[str]]:
    """Probe-subset examples with COMPLETE next_state_text (from the original file's
    successor rows) and observation_text (from the probe rows). Returns (examples, obs_texts)
    in probe-file order."""
    original = [json.loads(l) for l in ORIGINAL[split].open() if l.strip()]
    probe = [json.loads(l) for l in PROBE[split].open() if l.strip()]
    examples = fj.build_canonical_event_examples(original)  # populates next_state_text via successors
    by_key = {(e.trajectory_id, e.interaction_index): e for e in examples}
    out_examples, out_obs = [], []
    for row in probe:
        key = (str(row["trajectory_id"]), int(row.get("interaction_index") or 0))
        example = by_key.get(key)
        if example is None or not example.next_state_text:
            raise SystemExit(f"probe row {key} missing from original or lacks next_state_text")
        out_examples.append(example)
        out_obs.append(str(row.get("observation") or ""))
    return out_examples, out_obs


def build_labels(examples: list[Any], vocab: dict[str, list[str]]) -> dict[str, torch.Tensor]:
    labels: dict[str, torch.Tensor] = {}
    for field in SINGLE_FIELDS:
        index = {v: i for i, v in enumerate(vocab[field])}
        labels[field] = torch.tensor([index[e.single_labels[field]] for e in examples], dtype=torch.long)
    for field in MULTI_FIELDS:
        index = {v: i for i, v in enumerate(vocab[field])}
        hot = torch.zeros(len(examples), len(index))
        for i, e in enumerate(examples):
            for v in e.multi_labels[field]:
                hot[i, index[v]] = 1.0
        labels[field] = hot
    return labels


# --------------------------------------------------------------------------- encode
@torch.no_grad()
def encode_texts(model, tokenizer, texts, max_length, keep_newest, device, token_budget) -> torch.Tensor:
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = torch.empty(len(texts), model.latent_dim, dtype=torch.float32)
    prev_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left" if keep_newest else "right"
    try:
        batch: list[int] = []
        longest = 1

        def flush():
            nonlocal batch, longest
            if not batch:
                return
            enc = tokenizer([texts[i] for i in batch], return_tensors="pt", padding=True,
                            truncation=True, max_length=max_length, add_special_tokens=True)
            z, _ = model.encode_latent_and_logits(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            out[torch.tensor(batch)] = z.float().cpu()
            batch, longest = [], 1

        MAX_BATCH = 48  # hard cap: short-text batches otherwise balloon to 100s of padded rows
        for i in order:
            # conservative chars->tokens estimate (JSON-heavy text runs ~2-2.5 chars/token)
            estimate = min(max_length, max(8, len(texts[i]) // 2))
            if batch and ((len(batch) + 1) * max(longest, estimate) > token_budget or len(batch) >= MAX_BATCH):
                flush()
            batch.append(i)
            longest = max(longest, estimate)
        flush()
    finally:
        tokenizer.truncation_side = prev_side
    return out


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
        latent_type=str(manifest.get("latent_type") or "continuous"), pooling=str(manifest.get("pooling") or "mean"),
    )
    fj.load_jepa_state_dict_for_training(model, args.checkpoint, allow_missing_success_head=True,
                                         allow_missing_canonical_event_heads=True)
    # Freeze everything: _encode_pooled re-enables autograd via set_grad_enabled(
    # backbone_requires_grad()), which silently overrides an outer no_grad and accumulates
    # full activation graphs (39GB OOM). requires_grad=False makes it a true inference pass.
    for param in model.parameters():
        param.requires_grad = False
    model.to(device).eval()

    args.workdir.mkdir(parents=True, exist_ok=True)
    # full vocab from the ORIGINAL files so head sizes match the deployed models
    all_examples = []
    for split in ("train", "eval"):
        all_examples.extend(fj.build_canonical_event_examples([json.loads(l) for l in ORIGINAL[split].open() if l.strip()]))
    vocab = fj.build_canonical_event_vocabularies(all_examples)
    (args.workdir / "vocab.json").write_text(json.dumps(vocab, indent=1))

    for split in ("train", "eval"):
        examples, obs_texts = load_rows(split)
        print(f"[{split}] {len(examples)} rows; encoding on {device} (native max_input={max_input})", flush=True)
        payload: dict[str, Any] = {"labels": build_labels(examples, vocab)}
        payload["z"] = {
            "cur": encode_texts(model, tokenizer, [e.current_state_text for e in examples], max_input, False, device, args.token_budget),
            "ctx": encode_texts(model, tokenizer, [e.context_text for e in examples], max_input, False, device, args.token_budget),
            "act": encode_texts(model, tokenizer, [e.action_text for e in examples], max_action, False, device, args.token_budget),
            # keep-newest so the appended outcome is inside the window (see module docstring)
            "target": encode_texts(model, tokenizer, [e.next_state_text for e in examples], max_input, True, device, args.token_budget),
            "obs": encode_texts(model, tokenizer, obs_texts, max_action, False, device, args.token_budget),
        }
        with torch.no_grad():
            preds = []
            for start in range(0, len(examples), 1024):
                sl = slice(start, start + 1024)
                z, _ = model.predict_latent(payload["z"]["cur"][sl].to(device),
                                            payload["z"]["act"][sl].to(device),
                                            payload["z"]["ctx"][sl].to(device))
                preds.append(z.float().cpu())
            payload["z"]["pred"] = torch.cat(preds)
        torch.save(payload, args.workdir / f"latents_{split}.pt")
        print(f"[{split}] saved {args.workdir / f'latents_{split}.pt'}", flush=True)


# --------------------------------------------------------------------------- train
class Heads(nn.Module):
    def __init__(self, input_dim: int, vocab: dict[str, list[str]], hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({f: nn.Linear(hidden, len(v)) for f, v in vocab.items()})

    def forward(self, x):
        t = self.trunk(x)
        return {f: h(t) for f, h in self.heads.items()}


def arm_input(z: dict[str, torch.Tensor], parts) -> torch.Tensor:
    return torch.cat([z[p] for p in parts], dim=-1)


def loss_fn(logits, labels, idx):
    total = 0.0
    for f in SINGLE_FIELDS:
        total = total + nn.functional.cross_entropy(logits[f], labels[f][idx].to(logits[f].device))
    for f in MULTI_FIELDS:
        total = total + nn.functional.binary_cross_entropy_with_logits(logits[f], labels[f][idx].to(logits[f].device))
    return total / (len(SINGLE_FIELDS) + len(MULTI_FIELDS))


@torch.no_grad()
def evaluate(model, x, labels, device) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    model.eval()
    correct_rows: dict[str, torch.Tensor] = {}
    accuracy: dict[str, float] = {}
    logits = {f: [] for f in list(SINGLE_FIELDS) + MULTI_FIELDS}
    for start in range(0, x.shape[0], 2048):
        out = model(x[start:start + 2048].to(device))
        for f in logits:
            logits[f].append(out[f].cpu())
    for f in SINGLE_FIELDS:
        pred = torch.cat(logits[f]).argmax(-1)
        correct_rows[f] = pred == labels[f]
        accuracy[f] = correct_rows[f].float().mean().item()
    for f in MULTI_FIELDS:
        pred = (torch.sigmoid(torch.cat(logits[f])) >= 0.5).float()
        correct_rows[f] = (pred == labels[f]).all(-1)
        accuracy[f] = correct_rows[f].float().mean().item()
    return accuracy, correct_rows


def mcnemar(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int, float]:
    """(b_only_wrong, a_only_wrong, p) two-sided McNemar with continuity correction."""
    a_wrong_b_right = int(((~a) & b).sum())
    a_right_b_wrong = int((a & (~b)).sum())
    n = a_wrong_b_right + a_right_b_wrong
    if n == 0:
        return a_wrong_b_right, a_right_b_wrong, 1.0
    stat = (abs(a_wrong_b_right - a_right_b_wrong) - 1) ** 2 / n
    p = math.erfc(math.sqrt(stat / 2))
    return a_wrong_b_right, a_right_b_wrong, p


def stage_train(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vocab = json.loads((args.workdir / "vocab.json").read_text())
    train = torch.load(args.workdir / "latents_train.pt", weights_only=False)
    evald = torch.load(args.workdir / "latents_eval.pt", weights_only=False)

    results: dict[str, Any] = {}
    row_correct: dict[str, dict[str, torch.Tensor]] = {}
    x_pred_model = None
    for arm, parts in ARMS.items():
        torch.manual_seed(args.seed)
        x_train = arm_input(train["z"], parts)
        x_eval = arm_input(evald["z"], parts)
        model = Heads(x_train.shape[-1], vocab).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        n = x_train.shape[0]
        for epoch in range(args.epochs):
            model.train()
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed + epoch))
            for start in range(0, n, args.batch_size):
                idx = perm[start:start + args.batch_size]
                out = model(x_train[idx].to(device))
                loss = loss_fn(out, train["labels"], idx)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        acc, rows = evaluate(model, x_eval, evald["labels"], device)
        results[arm] = acc
        row_correct[arm] = rows
        if arm == "x_pred":
            x_pred_model = model
        outcome = sum(acc[f] for f in OUTCOME_FIELDS) / len(OUTCOME_FIELDS)
        print(f"[{arm:12s}] outcome4={outcome:.4f} exec={acc['execution_status']:.4f} "
              f"macro={sum(acc.values())/len(acc):.4f}", flush=True)

    # eval-time predictor-usage controls on the trained x_pred heads
    z = evald["z"]
    g = torch.Generator().manual_seed(args.seed)
    shuffled = z["pred"][torch.randperm(z["pred"].shape[0], generator=g)]
    for name, pred_sub in (("x_pred_shuffled", shuffled), ("x_pred_zeroed", torch.zeros_like(z["pred"]))):
        x_eval = torch.cat([z["cur"], z["act"], z["ctx"], pred_sub], dim=-1)
        acc, rows = evaluate(x_pred_model, x_eval, evald["labels"], device)
        results[name] = acc
        row_correct[name] = rows
        outcome = sum(acc[f] for f in OUTCOME_FIELDS) / len(OUTCOME_FIELDS)
        print(f"[{name:12s}] outcome4={outcome:.4f} exec={acc['execution_status']:.4f}", flush=True)

    print("\n=== McNemar (execution_status, paired on identical rows) ===")
    tests = [("x_pred", "x_only"), ("x_pred", "x_pred_shuffled"), ("x_target", "x_pred"),
             ("x_obs", "x_target"), ("x_obs", "x_pred")]
    stats = {}
    for a, b in tests:
        aw, bw, p = mcnemar(row_correct[a]["execution_status"], row_correct[b]["execution_status"])
        stats[f"{a}_vs_{b}"] = {"a_wrong_b_right": aw, "a_right_b_wrong": bw, "p": p}
        print(f"  {a:16s} vs {b:16s}: {a} fixes {bw:4d}, breaks {aw:4d}   p={p:.2e}")

    (args.workdir / "results.json").write_text(json.dumps(
        {"accuracy": results, "mcnemar_execution_status": stats,
         "arms": {k: list(v) for k, v in ARMS.items()},
         "outcome_fields": OUTCOME_FIELDS,
         "protocol": {"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "seed": args.seed}},
        indent=1))
    print(f"\nresults -> {args.workdir / 'results.json'}")


def main() -> None:
    args = parse_args()
    if args.stage == "encode":
        stage_encode(args)
    else:
        stage_train(args)


if __name__ == "__main__":
    main()
