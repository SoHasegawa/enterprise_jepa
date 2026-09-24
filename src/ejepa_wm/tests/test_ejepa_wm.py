"""Unit tests for the pluggable World Model package."""
from __future__ import annotations

import json

import pytest

from ejepa_wm import WMConfig, build_world_model, normalize_strategy, report, wm_config_from_env
from ejepa_wm.prompts import load_prompt


def test_normalize_strategy_aliases_and_defaults():
    assert normalize_strategy("best_of_n") == "selection"
    assert normalize_strategy("SELECTION") == "selection"
    assert normalize_strategy("prompt_injection") == "prompt_injection"
    assert normalize_strategy("bogus") == "none"
    assert normalize_strategy(None) == "none"


def test_wm_config_from_env_precedence_and_alias():
    cfg = wm_config_from_env({"WM_STRATEGY": "best_of_n", "WM_BACKEND": "served", "WM_N": "5"})
    assert (cfg.strategy, cfg.backend, cfg.n) == ("selection", "served", 5)
    # a strategy that needs the imagined backend selects it without an explicit WM_BACKEND
    cfg2 = wm_config_from_env({"WM_STRATEGY": "itp_i"})
    assert cfg2.backend == "ewm_imagined"
    # anything else with no backend falls back to the no-WM baseline
    cfg3 = wm_config_from_env({"WM_STRATEGY": "prompt_injection"})
    assert cfg3.backend == "noop"


def test_strategy_none_forces_noop_regardless_of_backend():
    wm = build_world_model(WMConfig(strategy="none", backend="served"))
    assert wm.name == "noop"
    assert wm.advise([]).text == ""
    assert wm.select([], [{"content": "a"}, {"content": "b"}]).index == 0


def test_noop_select_first():
    wm = build_world_model(WMConfig(strategy="selection", backend="noop", n=3))
    assert wm.select([], [{"content": "x"}, {"content": "y"}]).index == 0


def test_llm_select_parses_index():
    wm = build_world_model(
        WMConfig(strategy="selection", backend="llm", n=2), chat_fn=lambda m: "the answer is 1"
    )
    flow = [{"type": "user_message", "content": "task"}]
    cands = [{"content": "wrong"}, {"content": "CORRECT"}]
    assert wm.select(flow, cands).index == 1


def test_llm_select_malformed_falls_back_to_zero():
    wm = build_world_model(
        WMConfig(strategy="selection", backend="llm", n=2), chat_fn=lambda m: "no number here"
    )
    cands = [{"content": "a"}, {"content": "b"}]
    assert wm.select([], cands).index == 0


def test_llm_select_out_of_range_falls_back_to_zero():
    wm = build_world_model(
        WMConfig(strategy="selection", backend="llm", n=2), chat_fn=lambda m: "99"
    )
    assert wm.select([], [{"content": "a"}, {"content": "b"}]).index == 0


def test_llm_backend_requires_chat_fn():
    with pytest.raises(ValueError):
        build_world_model(WMConfig(strategy="selection", backend="llm"))


def test_llm_advise_returns_text():
    wm = build_world_model(
        WMConfig(strategy="prompt_injection", backend="llm"),
        chat_fn=lambda m: "[World-Model feedback] stay on task",
    )
    res = wm.advise([{"type": "user_message", "content": "do x"}])
    assert "World-Model feedback" in res.text


def test_load_prompt_fills_and_errors():
    text = load_prompt("llm", "select", flow="F", n=2, candidates="[0] a\n[1] b")
    assert "[0] a" in text and "2 candidate" in text
    with pytest.raises(FileNotFoundError):
        load_prompt("llm", "does_not_exist")


def test_served_resolve_reuses_baseline_env():
    from ejepa_wm.backends.served import resolve_served

    # explicit WM_* wins
    base, key, model = resolve_served(
        WMConfig(strategy="selection", backend="served", model="qwen3.6-27b"),
        {"WM_BASE_URL": "http://h:8020/v1", "WM_API_KEY": "k"},
    )
    assert (base, key, model) == ("http://h:8020/v1", "k", "qwen3.6-27b")
    # falls back to the EnterpriseOps baseline serving env; key defaults to EMPTY for vLLM
    base2, key2, model2 = resolve_served(
        WMConfig(strategy="selection", backend="served"),
        {"LOCAL_VLLM_BASE": "http://127.0.0.1:8020/v1", "ENTERPRISEOPS_LLM_MODEL": "gemma-4-26b-a4b-it"},
    )
    assert base2 == "http://127.0.0.1:8020/v1"
    assert key2 == "EMPTY"
    assert model2 == "gemma-4-26b-a4b-it"


def test_served_resolve_requires_base_url_and_model():
    from ejepa_wm.backends.served import resolve_served

    with pytest.raises(ValueError):
        resolve_served(WMConfig(strategy="selection", backend="served", model="m"), {})
    with pytest.raises(ValueError):
        resolve_served(WMConfig(strategy="selection", backend="served"), {"WM_BASE_URL": "http://x/v1"})


def test_report_compare_scores_and_delta(tmp_path):
    off = [{"task_id": "a", "success": False}, {"task_id": "b", "success": False}]
    on = [{"task_id": "a", "success": True}, {"task_id": "b", "success": False}]
    cmp = report.compare(off, on)
    assert cmp.off.score == pytest.approx(0.0)
    assert cmp.on.score == pytest.approx(0.5)
    assert cmp.delta_pp == pytest.approx(50.0)
    # round-trips through files / load_records
    d = tmp_path / "on"
    d.mkdir()
    (d / "wm_eval_records.json").write_text(json.dumps(on))
    assert report.load_records(d) == on
    report.write_report(tmp_path, cmp)
    assert json.loads((tmp_path / "wm_compare.json").read_text())["delta_pp"] == pytest.approx(50.0)
