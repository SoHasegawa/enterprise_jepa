"""Prediction-ablation controls for the beam planner (pure-Python, no torch needed)."""

from __future__ import annotations

import json
import random

import pytest

pytest.importorskip("torch")  # the backend module imports torch at import time
from ejepa_wm.backends import _ewm_jepa as jepa

VOCAB = {"execution_status": ["failure", "success"], "missing_information_type": ["a", "b"]}


def rows(values):
    return [
        {
            "execution_status": {"failure": 1 - v, "success": v},
            "missing_information_type": {"a": v, "b": v},
        }
        for v in values
    ]


def test_resolve_prediction_control():
    assert jepa.resolve_prediction_control("") is None
    assert jepa.resolve_prediction_control("none") is None
    assert jepa.resolve_prediction_control("shuffled") == "shuffled"
    with pytest.raises(ValueError):
        jepa.resolve_prediction_control("oracle")


def test_shuffled_keeps_marginals_and_moves_rows():
    traj = [rows([0.1, 0.2, 0.3]), rows([0.4, 0.5])]
    term = [[0.1, 0.2, 0.3], [0.4, 0.5]]
    new_traj, new_term = jepa.apply_prediction_control(
        traj, term, "shuffled", vocab=VOCAB, rng=random.Random(3)
    )
    flat_old = sorted(r["execution_status"]["success"] for t in traj for r in t)
    flat_new = sorted(r["execution_status"]["success"] for t in new_traj for r in t)
    assert flat_old == flat_new  # same marginal distribution
    assert [len(t) for t in new_traj] == [3, 2]  # same shape
    assert sorted(x for t in new_term for x in t) == sorted(x for t in term for x in t)
    # terminal probability moves with its row
    for t_rows, t_terms in zip(new_traj, new_term, strict=True):
        for r, p in zip(t_rows, t_terms, strict=True):
            assert r["execution_status"]["success"] == pytest.approx(p)
    old_flat = [r for t in traj for r in t]
    new_flat = [r for t in new_traj for r in t]
    assert any(a != b for a, b in zip(old_flat, new_flat, strict=True))  # something moved


def test_uniform_rows():
    traj = [rows([0.9, 0.9])]
    new_traj, new_term = jepa.apply_prediction_control(traj, [[0.9, 0.9]], "uniform", vocab=VOCAB)
    assert new_traj[0][0]["execution_status"] == {"failure": 0.5, "success": 0.5}
    assert new_traj[0][1]["missing_information_type"] == {"a": 0.5, "b": 0.5}
    assert new_term == [[0.5, 0.5]]


def test_prior_rows_from_file(tmp_path):
    path = tmp_path / "priors.json"
    path.write_text(
        json.dumps(
            {
                "fields": {"execution_status": {"failure": 0.2, "success": 0.8}},
                "terminal_probability": 0.3,
            }
        )
    )
    priors = jepa.load_class_priors(str(path))
    new_traj, new_term = jepa.apply_prediction_control(
        [rows([0.9])], [[0.9]], "prior", vocab=VOCAB, priors=priors
    )
    assert new_traj[0][0]["execution_status"] == {"failure": 0.2, "success": 0.8}
    assert new_traj[0][0]["missing_information_type"] == {
        "a": 0.5,
        "b": 0.5,
    }  # missing field -> uniform
    assert new_term == [[0.3]]


def test_none_is_identity():
    traj = [rows([0.1])]
    assert jepa.apply_prediction_control(traj, [[0.1]], None, vocab=VOCAB) == (traj, [[0.1]])


def test_no_state_ties_every_candidate_and_withholds_terminal_advice():
    """The no-predicted-state arm must leave candidate ranking to the policy's own order.

    Reviewer-requested ablation: keep candidate generation, the reflection instructions
    and the policy-call budget, remove the learned predictions. Every candidate must
    therefore receive an identical row, so the stable sort in ``rank_trajectories``
    falls through to the first candidate, and the terminal probability must be zero so
    no early-stop advice is derived from a prediction.
    """
    from ejepa_wm.backends._ewm_jepa import apply_prediction_control, resolve_prediction_control

    assert resolve_prediction_control("no_state") == "no_state"

    vocab = {"execution_status": ["success", "failure"], "action_kind": ["read", "write", "other"]}
    trajectories = [
        [{"execution_status": {"success": 0.9, "failure": 0.1}}, {"execution_status": {"success": 0.2, "failure": 0.8}}],
        [{"execution_status": {"success": 0.4, "failure": 0.6}}],
    ]
    terminal = [[0.7, 0.9], [0.1]]

    rows, terms = apply_prediction_control(trajectories, terminal, "no_state", vocab=vocab)

    flat = [row for traj in rows for row in traj]
    assert len(flat) == 3
    assert all(row == flat[0] for row in flat), "candidates must be indistinguishable"
    assert flat[0]["execution_status"] == {"success": 0.5, "failure": 0.5}
    assert flat[0]["action_kind"] == {"read": 1 / 3, "write": 1 / 3, "other": 1 / 3}
    assert terms == [[0.0, 0.0], [0.0]], "no terminal advice may survive"

    # uniform keeps terminal advice live; the two arms must not collapse into one
    _, uniform_terms = apply_prediction_control(trajectories, terminal, "uniform", vocab=vocab)
    assert uniform_terms == [[0.5, 0.5], [0.5]]
