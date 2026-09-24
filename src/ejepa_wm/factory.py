"""Construct a :class:`~ejepa_wm.base.WorldModel` from config / ``WM_*`` env.

Backend resolution (``WMConfig.backend``):

* ``noop``    — :class:`~ejepa_wm.backends.noop.NoopWorldModel` (no-WM baseline)
* ``served``  — :class:`~ejepa_wm.backends.served.ServedWorldModel` (served OpenAI-compatible model
  — any OpenAI-compatible endpoint; reuses the EnterpriseOps baseline serving env)
* ``ewm_predict`` — :class:`~ejepa_wm.backends.ewm_predict.EwmPredictWorldModel` (direct, no MCP:
  the EWM scores each candidate action's binary+error feasibility via vLLM/transformers)
* ``llm``     — :class:`~ejepa_wm.backends.llm.LlmWorldModel` (requires a ``chat_fn``)

If ``strategy=none`` the backend is forced to ``noop`` regardless of what was requested.
"""

from __future__ import annotations

import os

from ejepa_wm.backends.noop import NoopWorldModel
from ejepa_wm.base import ChatFn, WMConfig, WorldModel, env_float, normalize_strategy


def wm_config_from_env(environ: dict | None = None) -> WMConfig:
    """Build a :class:`WMConfig` from ``WM_*`` env vars.

    Precedence is documented in ``src/ejepa_wm/README.md``: a benchmark-level default (if any) is
    expected to be folded into the env before this is called, then explicit ``--wm-*`` CLI args
    (which the ``ejepa`` CLI exports as ``WM_*``) win. ``best_of_n`` is accepted as an alias for
    ``selection``.
    """
    env = environ if environ is not None else os.environ
    # Convenience alias: WM_STRATEGY=imagined => prompt_injection via ewm_imagined.
    imagined_alias = (env.get("WM_STRATEGY") or "").strip().lower() == "imagined"
    strategy = "prompt_injection" if imagined_alias else normalize_strategy(env.get("WM_STRATEGY"))
    backend = (env.get("WM_BACKEND") or "").strip().lower()
    if not backend:
        if imagined_alias or strategy in (
            "itp_i",
            "revision",
            "reference",
            "beam_plan",
            "hier_latent_cem",
        ):
            # ITP-I and both MPC planners live in the generative/imagined backend.
            backend = "ewm_imagined"
        else:
            backend = "noop"
    try:
        n = max(1, int(env.get("WM_N", "1")))
    except ValueError:
        n = 1
    return WMConfig(
        strategy=strategy,
        backend=backend,
        model=(env.get("WM_MODEL") or None),
        n=n,
        target_state=(env.get("WM_TARGET_STATE") or None),
        inject_template=(env.get("WM_INJECT_TEMPLATE") or "current"),
        # ``WM_SCORER_TIMEOUT`` is the name the recorded runs used; kept as a fallback so
        # a replayed command line keeps its timeout. ``_ewm_generators`` accepts both too.
        timeout=env_float("WM_TIMEOUT", env_float("WM_SCORER_TIMEOUT", 600.0)),
    )


def build_world_model(config: WMConfig, *, chat_fn: ChatFn | None = None) -> WorldModel:
    """Instantiate the configured World Model.

    Uniform, executor-agnostic contract: every backend is built from ``(config, chat_fn)``
    alone, so the World Model is pluggable regardless of which executor/benchmark drives it.
    ``chat_fn`` is the policy LLM, needed only by backends that consult it directly
    (``llm`` and ``ewm_imagined``); ``served`` builds its own chat callable; other
    backends ignore it.

    ``ewm_imagined`` derives the tool catalog for its imagined ReAct rollout from the
    ``conversation_flow`` it receives in :meth:`advise` (the agent's ``system_message``),
    not from an executor-supplied argument — keeping this contract uniform.
    """
    backend = config.backend
    if config.strategy == "none" or backend == "noop":
        return NoopWorldModel(config)
    if backend == "served":
        from ejepa_wm.backends.served import ServedWorldModel

        return ServedWorldModel(config)
    if backend == "ewm_predict":
        from ejepa_wm.backends.ewm_predict import EwmPredictWorldModel

        return EwmPredictWorldModel(config)
    if backend == "ewm_imagined":
        from ejepa_wm.backends.ewm_imagined import EwmImaginedWorldModel

        if chat_fn is None:
            raise ValueError("backend='ewm_imagined' requires a chat_fn (the policy LLM).")
        return EwmImaginedWorldModel(config, chat_fn)
    if backend == "llm":
        from ejepa_wm.backends.llm import LlmWorldModel

        if chat_fn is None:
            raise ValueError("backend='llm' requires a chat_fn.")
        return LlmWorldModel(config, chat_fn)
    raise ValueError(
        f"Unknown WM backend: {backend!r} "
        "(expected noop|served|ewm_predict|ewm_imagined|llm)."
    )
