"""Tests for the torch-free pure helpers in ``_ewm_hier_cem`` (family-key heuristics, action
normalization, decode validation). The module imports ``torch`` at top, so these are guarded by
``importorskip('torch')`` -- unlike ``test_ewm_hier_cem.py`` (which fakes the whole module via
sys.modules to test the ``ewm_imagined`` MPC controller without touching torch at all), here we
need the real module object to exercise its internal helpers directly.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ejepa_wm.backends import _ewm_hier_cem as hc


def test_family_key_plain_mcp_tool_uses_tool_name():
    action = {"tool_calls": [{"function": {"name": "create_event", "arguments": {"x": 1}}}]}
    assert hc._family_key(action) == "create_event"


def test_family_key_sql_backend_uses_verb():
    action = {"tool_calls": [{"function": {"name": "execute_crm_sql", "arguments": {"query": "SELECT * FROM leads"}}}]}
    assert hc._family_key(action) == "sql:SELECT"


def test_family_key_sql_detected_from_content_even_without_name_hint():
    action = {"tool_calls": [{"function": {"name": "run_query", "arguments": {"query": "DELETE FROM x"}}}]}
    assert hc._family_key(action) == "sql:DELETE"


def test_family_key_shell_backend_unwraps_and_skips_prefixes():
    action = {"tool_calls": [{"function": {
        "name": "run_shell", "arguments": {"command": "bash -lc 'sudo apt-get update && echo done'"}
    }}]}
    assert hc._family_key(action) == "shell:apt-get"


def test_family_key_shell_requires_name_hint_not_just_command_arg():
    # A generic "command" argument key on a non-shell-hinted tool name should NOT be treated as
    # shell (ambiguous) -- falls through to the plain tool-name family.
    action = {"tool_calls": [{"function": {"name": "some_mcp_tool", "arguments": {"command": "ls -la"}}}]}
    assert hc._family_key(action) == "some_mcp_tool"


def test_family_key_code_edit_backend_uses_sub_operation():
    action = {"tool_calls": [{"function": {"name": "str_replace_editor", "arguments": {"command": "view"}}}]}
    assert hc._family_key(action) == "code_edit:view"


def test_family_key_python_backend_uses_first_token():
    action = {"tool_calls": [{"function": {"name": "python_repl", "arguments": {"code": "import pandas as pd\ndf.head()"}}}]}
    assert hc._family_key(action) == "python:import"


def test_family_key_browser_backend_uses_verb_name():
    action = {"tool_calls": [{"function": {"name": "click", "arguments": {}}}]}
    assert hc._family_key(action) == "browser:click"


def test_family_key_noop_for_empty_calls():
    assert hc._family_key({"tool_calls": []}) == "noop"


def test_normalize_action_step_accepts_multiple_shapes():
    assert hc.normalize_action_step({"name": "ls", "arguments": {"path": "/"}})["tool_calls"][0]["function"]["name"] == "ls"
    assert hc.normalize_action_step({"function": {"name": "ls", "arguments": {}}})["tool_calls"][0]["function"]["name"] == "ls"
    assert hc.normalize_action_step({"tool": "ls", "args": {}})["tool_calls"][0]["function"]["name"] == "ls"
    assert hc.normalize_action_step({"tool_calls": [{"function": {"name": "ls", "arguments": {}}}]})["tool_calls"][0]["function"]["name"] == "ls"
    assert hc.normalize_action_step({}) is None
    assert hc.normalize_action_step("not a dict") is None


def test_validate_decoded_action():
    assert hc.validate_decoded_action('{"name":"ls","arguments":{}}') is not None
    assert hc.validate_decoded_action("not json") is None
    assert hc.validate_decoded_action("") is None
    assert hc.validate_decoded_action("   ") is None


def test_render_action():
    assert hc.render_action("raw text") == "raw text"
    assert hc.render_action({"name": "ls"}) == '{"name": "ls"}'
