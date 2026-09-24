"""Process-wide sharing of world-model weights (``WM_SHARE_MODEL_WEIGHTS``).

Per-episode MPC state lives on the ``EwmImagined`` instance, so concurrent tasks
need one world model each. Sharing the multi-GB generator keeps that affordable.
"""

from __future__ import annotations

import pytest

from ejepa_wm.backends.ewm_imagined import (
    clear_shared_generators,
    shared_generator,
    weight_sharing_enabled,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_shared_generators()
    yield
    clear_shared_generators()


def test_sharing_is_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("WM_SHARE_MODEL_WEIGHTS", raising=False)
    assert weight_sharing_enabled() is False
    built = []

    def factory():
        built.append(1)
        return object()

    first = shared_generator(("jepa", "/ckpt", {}), factory)
    second = shared_generator(("jepa", "/ckpt", {}), factory)
    assert first is not second, "without opt-in every caller must get its own instance"
    assert len(built) == 2


def test_same_spec_is_built_once_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("WM_SHARE_MODEL_WEIGHTS", "1")
    built = []

    def factory():
        built.append(1)
        return object()

    first = shared_generator(("jepa", "/ckpt", {"dtype": "auto"}), factory)
    second = shared_generator(("jepa", "/ckpt", {"dtype": "auto"}), factory)
    assert first is second
    assert len(built) == 1, "the checkpoint must load exactly once per process"


def test_different_spec_gets_its_own_generator(monkeypatch) -> None:
    monkeypatch.setenv("WM_SHARE_MODEL_WEIGHTS", "1")
    a = shared_generator(("jepa", "/ckpt-a", {}), object)
    b = shared_generator(("jepa", "/ckpt-b", {}), object)
    c = shared_generator(("jepa", "/ckpt-a", {"dtype": "bfloat16"}), object)
    assert a is not b, "a different checkpoint is a different model"
    assert a is not c, "differing construction kwargs must not collide"
    assert shared_generator(("jepa", "/ckpt-a", {}), object) is a


def test_unhashable_spec_still_keys_cleanly(monkeypatch) -> None:
    """The spec carries a kwargs dict, so the key must not require hashability."""
    monkeypatch.setenv("WM_SHARE_MODEL_WEIGHTS", "1")
    spec = ("jepa", "/ckpt", {"arch_defaults": {"dim": 512}, "merge_checkpoint": None})
    first = shared_generator(spec, object)
    assert shared_generator(spec, object) is first


def test_clear_releases_cached_generators(monkeypatch) -> None:
    monkeypatch.setenv("WM_SHARE_MODEL_WEIGHTS", "1")
    first = shared_generator(("jepa", "/ckpt", {}), object)
    clear_shared_generators()
    assert shared_generator(("jepa", "/ckpt", {}), object) is not first
