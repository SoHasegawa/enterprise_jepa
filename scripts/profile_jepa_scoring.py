#!/usr/bin/env python3
"""Where does a JEPA beam-scoring call spend its time?

Wraps every public/private method of the JEPA generator with CUDA-synchronised
timers and runs the same cold-state scoring call that
``measure_wm_latency_vs_horizon.py`` times, so the per-call figure decomposes into
text encoding, latent prediction, head evaluation, and decode. Also reports the
runtime facts that decide which speed-ups apply (parameter dtype, attention
implementation, whether anything is compiled).

    CUDA_VISIBLE_DEVICES=2 assets/AutomationBench/purple/.venv/bin/python \
        scripts/profile_jepa_scoring.py --checkpoint checkpoints/jepa --horizon 3 --candidates 8

Optional experiments: --dtype bfloat16 | float16 | float32 (re-loads the model),
--compile (torch.compile the predictor/head modules found on the generator).
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import importlib.util
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def load_measure_module():
    path = Path(__file__).with_name("measure_wm_latency_vs_horizon.py")
    spec = importlib.util.spec_from_file_location("wm_measure", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--candidates", type=int, default=8)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--dtype", default=None, help="Override WM_JEPA_DTYPE (bfloat16/float16/float32)."
    )
    p.add_argument("--compile", action="store_true", help="torch.compile predictor/head modules.")
    p.add_argument(
        "--compile-backbone",
        action="store_true",
        help="Also torch.compile the text backbone (dynamic=False; pair with --static-shapes).",
    )
    p.add_argument(
        "--static-shapes",
        action="store_true",
        help="Bucket-pad every encode (see --pad-multiple) so backbone shapes recur (enables CUDA graphs).",
    )
    p.add_argument("--compile-mode", default="reduce-overhead")
    p.add_argument(
        "--pad-multiple",
        type=int,
        default=64,
        help="With --static-shapes, pad token length up to this multiple (capped at max_length).",
    )
    return p.parse_args(argv)


class Timer:
    def __init__(self) -> None:
        self.inclusive: dict[str, float] = collections.defaultdict(float)
        self.calls: collections.Counter = collections.Counter()
        self.tokens: collections.Counter = collections.Counter()  # input tokens seen per module
        self.depth = 0

    def wrap(self, owner: Any, name: str) -> None:
        original = getattr(owner, name)
        import torch

        def timed(*a: Any, **k: Any) -> Any:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            self.depth += 1
            t0 = time.perf_counter()
            try:
                return original(*a, **k)
            finally:
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                self.inclusive[name] += time.perf_counter() - t0
                self.calls[name] += 1
                self.depth -= 1

        setattr(owner, name, timed)

    def reset(self) -> None:
        self.inclusive.clear()
        self.calls.clear()
        self.tokens.clear()


def describe_runtime(generator: Any) -> None:
    import torch

    seen = set()
    print("\n--- runtime facts ---")
    print(
        f"  torch {torch.__version__}  cuda {torch.version.cuda}  device {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}"
    )
    for attr in dir(generator):
        if attr.startswith("__"):
            continue
        try:
            obj = getattr(generator, attr)
        except Exception:
            continue
        if isinstance(obj, torch.nn.Module) and id(obj) not in seen:
            seen.add(id(obj))
            params = list(obj.parameters())
            n = sum(p.numel() for p in params)
            dtypes = sorted({str(p.dtype) for p in params})
            compiled = "OptimizedModule" in type(obj).__name__ or hasattr(obj, "_orig_mod")
            cfg = getattr(obj, "config", None)
            attn = getattr(cfg, "_attn_implementation", None) or getattr(
                cfg, "attn_implementation", None
            )
            print(
                f"  module {attr:<28s} {type(obj).__name__:<32s} params={n / 1e6:8.1f}M dtypes={dtypes} attn={attn} compiled={compiled}"
            )


def compile_modules(generator: Any, mode: str, *, include_backbone: bool = False) -> list[str]:
    """torch.compile the model's first-level children in place.

    The predictor and heads get the requested mode. The HF text backbone is
    skipped unless ``include_backbone`` -- and then compiled with
    ``dynamic=False`` so that, together with ``--static-shapes``, its forward can be
    captured into CUDA graphs (the launch-overhead regime a 0.6B encoder on an
    H200 sits in at these sequence lengths).
    """
    import torch

    model = getattr(generator, "model", None)
    compiled: list[str] = []
    if not isinstance(model, torch.nn.Module):
        return compiled
    # CUDA-graph trees reuse output buffers across replays; the scorer keeps
    # backbone/predictor outputs alive across sub-calls, so clone user-visible
    # outputs (a small D2D copy) instead of letting them be overwritten.
    with contextlib.suppress(AttributeError, TypeError):
        torch._inductor.config.triton.cudagraph_trees_generation_cloning = "user_visible"
    for name, child in list(model.named_children()):
        is_backbone = "backbone" in name.lower()
        if is_backbone and not include_backbone:
            continue
        if sum(p.numel() for p in child.parameters()) == 0:
            continue
        setattr(
            model,
            name,
            _clone_outputs(torch.compile(child, mode=mode, dynamic=False if is_backbone else None)),
        )
        compiled.append(name)
    return compiled


def _clone_outputs(module: Any) -> Any:
    """Wrap a compiled module so its tensor outputs are cloned.

    Defensive complement to the inductor cloning flag (older torch builds lack
    it): under ``reduce-overhead`` the graph's static output buffers are
    overwritten by the next replay, and the scorer reads earlier outputs later.
    Returned object is an ``nn.Module`` so it can be re-assigned as a child.
    """
    import torch

    class CloneOutputs(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return _clone_tree(self.inner(*args, **kwargs))

        def get_encoder(self) -> Any:
            # decoder-only backbones report themselves as their encoder; keep the
            # compiled wrapper on the call path instead of leaking the inner model.
            return self

        def __getattr__(self, name: str) -> Any:  # config, dtype, device, ...
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    return CloneOutputs(module)


def _clone_tree(obj: Any) -> Any:
    import torch

    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
        return type(obj)(*(_clone_tree(o) for o in obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone_tree(o) for o in obj)
    if isinstance(obj, dict):  # includes HF ModelOutput (an OrderedDict)
        for key in list(obj.keys()):
            obj[key] = _clone_tree(obj[key])
        return obj
    return obj


def wrap_all_methods(generator: Any, timer: Timer) -> None:
    for name in dir(type(generator)):
        if name.startswith("__"):
            continue
        attr = getattr(type(generator), name, None)
        if callable(attr) and not isinstance(attr, property):
            timer.wrap(generator, name)


def _token_counter(timer: Timer, key: str):
    """Pre-hook counting ``input_ids`` elements per backbone pass (HF modules take kwargs)."""

    def hook(_module, args, kwargs):
        ids = kwargs.get("input_ids") if kwargs else None
        if ids is None and args and hasattr(args[0], "numel"):
            ids = args[0]
        if ids is not None and hasattr(ids, "numel"):
            timer.tokens[key] += int(ids.numel())

    return hook


def hook_submodules(generator: Any, timer: Timer) -> None:
    """Time each first-level child of the generator's model with forward hooks.

    Method-level timers miss work that happens inside ``nn.Module`` forwards
    (predictor rollout, classifier heads), so attribute that here under keys
    prefixed ``model.``.
    """
    import torch

    model = getattr(generator, "model", None)
    if not isinstance(model, torch.nn.Module):
        return
    starts: dict[str, float] = {}

    def pre(name):
        def hook(_module, _inputs):
            torch.cuda.synchronize()
            starts[name] = time.perf_counter()

        return hook

    def post(name):
        def hook(_module, _inputs, _output):
            torch.cuda.synchronize()
            timer.inclusive[f"model.{name}"] += time.perf_counter() - starts.pop(
                name, time.perf_counter()
            )
            timer.calls[f"model.{name}"] += 1

        return hook

    for name, child in model.named_children():
        child.register_forward_pre_hook(pre(name))
        child.register_forward_hook(post(name))
        if "backbone" in name.lower():
            child.register_forward_pre_hook(
                _token_counter(timer, f"model.{name}"), with_kwargs=True
            )


def apply_static_shapes(generator: Any, multiple: int) -> None:
    """Pad every tokenised batch up to the next multiple of ``multiple`` tokens (capped
    at the call's ``max_length``) so the backbone sees a handful of recurring shapes
    -- the precondition for CUDA-graph capture under torch.compile without paying
    for a full ``max_length`` forward on every call."""
    import torch
    import torch.nn.functional as F

    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    pad_id = int(getattr(getattr(generator, "tokenizer", None), "pad_token_id", 0) or 0)

    def padded(original):
        def call(*args: Any, **kwargs: Any) -> Any:
            out = original(*args, **kwargs)
            max_length = kwargs.get("max_length", args[-1] if args else None)
            ids = out.get("input_ids") if isinstance(out, dict) else None
            if ids is None:
                return out
            target = -(-ids.shape[-1] // multiple) * multiple
            if max_length is not None:
                target = min(target, int(max_length))
            if ids.shape[-1] >= target:
                return out
            extra = target - ids.shape[-1]
            out["input_ids"] = F.pad(ids, (0, extra), value=pad_id)
            if "attention_mask" in out:
                out["attention_mask"] = F.pad(out["attention_mask"], (0, extra), value=0)
            return out

        return call

    for name in ("_encode", "_encode_batch"):
        if hasattr(generator, name):
            setattr(generator, name, padded(getattr(generator, name)))
    torch.backends.cudnn.benchmark = True


def main(argv=None) -> int:
    args = parse_args(argv)
    import os

    if args.dtype:
        os.environ["WM_JEPA_DTYPE"] = args.dtype
    m = load_measure_module()
    import torch

    generator = m.build_jepa(args.checkpoint)
    describe_runtime(generator)
    if args.static_shapes:
        apply_static_shapes(generator, args.pad_multiple)
        print(
            f"  static shapes: every encode padded up to a multiple of {args.pad_multiple} tokens"
        )
    if args.compile or args.compile_backbone:
        names = compile_modules(
            generator, args.compile_mode, include_backbone=args.compile_backbone
        )
        print(f"  compiled {names} with mode={args.compile_mode}")

    timer = Timer()
    wrap_all_methods(generator, timer)
    hook_submodules(generator, timer)

    from ejepa_wm.backends._ewm_canonical_event_scoring import CanonicalEventScoreConfig

    plans = m.build_plans(horizon=args.horizon, candidates=args.candidates, distinct=True)

    def call(tag: str):
        return generator.score_action_plans_canonical_event(
            system_prompt=m.SYSTEM_PROMPT,
            user_prompt=m.USER_PROMPT,
            input_history=m.input_history(4, salt=f"prof-{tag}-{time.time_ns()}"),
            action_plans=plans,
            score_config=CanonicalEventScoreConfig(),
        )

    for i in range(args.warmup):
        call(f"w{i}")
    timer.reset()
    totals = []
    for i in range(args.repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        call(f"r{i}")
        torch.cuda.synchronize()
        totals.append(time.perf_counter() - t0)

    total = statistics.mean(totals)
    print(
        f"\n--- per-call breakdown: horizon={args.horizon} candidates={args.candidates} "
        f"dtype={args.dtype or 'auto'} compile={args.compile} ---"
    )
    print(
        f"  total {total * 1000:8.1f} ms/call  (stdev {statistics.stdev(totals) * 1000:.1f} ms, n={len(totals)})"
    )
    print(f"  {'method (inclusive)':<40s} {'ms/call':>9s} {'share':>7s} {'calls/call':>11s}")
    for name, secs in sorted(timer.inclusive.items(), key=lambda kv: -kv[1]):
        per = secs / args.repeats
        if per * 1000 < 0.05:
            continue
        print(
            f"  {name:<40s} {per * 1000:9.2f} {per / total:7.1%} {timer.calls[name] / args.repeats:11.1f}"
        )
    for key, n_tokens in sorted(timer.tokens.items()):
        passes = timer.calls.get(key, 0) / args.repeats
        print(
            f"  {key}: {n_tokens / args.repeats:,.0f} input tokens/call over "
            f"{passes:.1f} passes ({n_tokens / max(timer.calls.get(key, 1), 1):,.0f} tokens/pass)"
        )
    print(
        "  (inclusive: nested methods are counted inside their callers; the top-level entry"
        " score_action_plans_canonical_event equals the total)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
