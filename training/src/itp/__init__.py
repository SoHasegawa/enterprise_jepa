"""`react_wm` family integration for EnterpriseOps-Gym.

This package adds the ``react_wm`` family of agent strategies on top of the
existing ewm-enterprisearena pipeline without disturbing the original
``baseline`` / ``revision`` / ``imagined`` replay modes. It was ported from
the ``ewm-state-design-and-model-training-experiments`` repo and adapted to
reuse ewm-enterprisearena's trajectory format, world-model API, and agent
generator factory.

Three new replay modes are provided:

* ``react_wm`` -- standard ReAct, plus per-turn world-model foresight injection
  after turn 0 with a fixed depth ``K``.
* ``react_wm_decide_k`` -- ``react_wm`` but the action policy itself picks
  ``K`` per turn via a one-shot prompt.
* ``react_wm_rl_k`` -- ``react_wm`` but a separately trained K-controller
  (a small local LM with a K-head, trained via offline-RL pseudo-labelling +
  supervised K-head training) picks ``K`` per turn while the action policy
  (e.g. GPT-5.1) remains in charge of tool selection.

The K-controller training pipeline (Stages I label + II sft) lives in
:mod:`src.itp.training.train_adaptive_k` and reuses ewm's existing world-model
prompt format from :mod:`src.finetuning` so the runtime state text seen at
inference is byte-identical to the text used to train the K-controller.

Imports below are kept lazy because the runtime ``KController`` only needs
``torch`` / ``transformers`` when ``react_wm_rl_k`` is active.
"""

from src.itp.react_wm import (
    REACT_WM_MODES,
    ReactWMConfig,
    build_foresight_user_message,
    build_world_model_foresight,
)
from src.itp.k_decider import decide_k_via_agent

__all__ = [
    "REACT_WM_MODES",
    "ReactWMConfig",
    "build_foresight_user_message",
    "build_world_model_foresight",
    "decide_k_via_agent",
]
