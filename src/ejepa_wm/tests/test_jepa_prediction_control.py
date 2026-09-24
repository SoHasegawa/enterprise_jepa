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
