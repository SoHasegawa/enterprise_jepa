"""
tests/test_automationbench_dry_run.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Offline dry-run test for AutomationBench.

Uses no internet, no vLLM, no model download, no Docker, no Hugging Face access
and no upstream checkout: it exercises the bundled ``sample`` target end to end
through the green agent, the ``mcp_react`` executor's tool binding / ReAct loop
(driven by a scripted fake policy client) and the bundled assertion evaluator.

    uv run pytest tests/test_automationbench_dry_run.py -v
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_BENCH_DIR = Path(__file__).resolve().parent.parent / "assets" / "AutomationBench"
_GREEN_DIR = _BENCH_DIR / "green"
_EXECUTOR_DIR = _BENCH_DIR / "purple-executors" / "mcp_react"

# The green module imports `scoring` / `task_loader` by bare name; purge them (and
# this asset's green dir) between tests so a sibling benchmark's identically-named
# modules cannot leak in through sys.modules/sys.path when the full suite runs in
# one process. Each green runs in its own process in production.
_SHARED_MODULES = ("task_loader", "scoring", "tools", "wm_react", "executor")


def _purge() -> None:
    for name in _SHARED_MODULES:
        sys.modules.pop(name, None)
    green_dir = str(_GREEN_DIR)
    while green_dir in sys.path:
        sys.path.remove(green_dir)


@pytest.fixture(autouse=True)
def _isolate_modules():
    _purge()
    yield
    _purge()


def _load_module(name: str, path: Path):
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"Cannot load: {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_task_loader():
    _purge()
    sys.path.insert(0, str(_GREEN_DIR))
    return _load_module("task_loader", _GREEN_DIR / "task_loader.py")


def _load_scoring():
    _purge()
    sys.path.insert(0, str(_GREEN_DIR))
    return _load_module("scoring", _GREEN_DIR / "scoring.py")


def _load_tools():
    return _load_module("automationbench_test_tools", _EXECUTOR_DIR / "tools.py")


def _load_wm_react():
    src_dir = _BENCH_DIR.parent.parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return _load_module("automationbench_test_wm_react", _EXECUTOR_DIR / "wm_react.py")


def _sample_tasks() -> list[dict]:
    loader = _load_task_loader().TaskLoader()
    return loader.load_tasks("sample")


class _ScriptedPolicy:
    """Fake policy client: replays a fixed list of assistant messages."""

    def __init__(self, script: list[dict]) -> None:
        self._script = list(script)
        self.calls = 0

    async def invoke_with_tools(self, messages, tools):
        self.calls += 1
        if self._script:
            return self._script.pop(0)
        return {"role": "assistant", "content": "done", "tool_calls": []}

    def complete(self, messages, *, temperature=None, response_format=None):
        return ""

    def complete_samples(self, messages, *, temperature=0.0, num_samples=1, response_format=None):
        return [""] * num_samples


def _assistant(name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


# ---------------------------------------------------------------------------
# task loading
# ---------------------------------------------------------------------------


def test_task_loader_reads_bundled_sample() -> None:
    tasks = _sample_tasks()
    assert [task["id"] for task in tasks] == [
        "sample_sf_contact_phone_update",
        "sample_slack_digest_post",
    ]
    first = tasks[0]
    assert first["prompt"]
    assert first["initial_state"]["salesforce"]["contacts"][0]["id"] == "003001"
    assert first["assertions"] and first["zapier_tools"]
    assert first["source"] == "bundled_sample"


def test_task_loader_filters_by_task_id_and_rejects_unknown() -> None:
    loader = _load_task_loader().TaskLoader()
    only = loader.load_tasks("sample", ["sample_slack_digest_post"])
    assert [task["id"] for task in only] == ["sample_slack_digest_post"]
    with pytest.raises(ValueError, match="Unknown task ids"):
        loader.load_tasks("sample", ["does_not_exist"])
    with pytest.raises(ValueError, match="Unknown target"):
        loader.load_tasks("not_a_target")


# ---------------------------------------------------------------------------
# scoring (bundled fallback evaluator)
# ---------------------------------------------------------------------------


def test_fallback_scorer_awards_credit_only_for_changed_state() -> None:
    scoring = _load_scoring()
    task = _sample_tasks()[0]
    initial = task["initial_state"]

    # Nothing done: the already-satisfied guard assertion is excluded, the real
    # one fails -> no credit for doing nothing.
    idle = scoring.score_task(
        assertions=task["assertions"],
        initial_state=initial,
        final_state=initial,
        prefer_upstream=False,
    )
    assert idle[scoring.PARTIAL_CREDIT_KEY] == 0.0
    assert idle[scoring.PASS_KEY] == 0.0
    assert idle["assertions_scored"] == 1  # the pre-satisfied one is not scored

    # Correct update: full credit and a strict pass.
    final = json.loads(json.dumps(initial))
    final["salesforce"]["contacts"][0]["Phone"] = "+1-555-0101"
    solved = scoring.score_task(
        assertions=task["assertions"],
        initial_state=initial,
        final_state=final,
        prefer_upstream=False,
    )
    assert solved[scoring.PARTIAL_CREDIT_KEY] == 1.0
    assert solved[scoring.PASS_KEY] == 1.0


def test_fallback_scorer_penalises_broken_guard_assertion() -> None:
    scoring = _load_scoring()
    task = _sample_tasks()[0]
    initial = task["initial_state"]
    final = json.loads(json.dumps(initial))
    final["salesforce"]["contacts"][0]["Phone"] = "+1-555-0101"  # requested change
    final["salesforce"]["contacts"][1]["Phone"] = "+1-555-8888"  # broke the guard
    scores = scoring.score_task(
        assertions=task["assertions"],
        initial_state=initial,
        final_state=final,
        prefer_upstream=False,
    )
    assert scores["assertions_scored"] == 2
    assert scores[scoring.PARTIAL_CREDIT_KEY] == 0.5
    assert scores[scoring.PASS_KEY] == 0.0


# ---------------------------------------------------------------------------
# executor: tool binding + ReAct loop (no upstream checkout, no LLM)
# ---------------------------------------------------------------------------


def test_sample_binding_exposes_schemas_and_mutates_world() -> None:
    tools = _load_tools()
    task = _sample_tasks()[0]
    binding = tools.build_binding(
        initial_state=task["initial_state"],
        zapier_tools=task["zapier_tools"],
        toolset="limited_zapier",
        prefer_upstream=False,
    )
    assert binding.source == "bundled_sample"
    names = {schema["function"]["name"] for schema in binding.schemas}
    assert names == {"records_search", "record_update", "record_create"}
    # Schemas are OpenAI function-calling shaped.
    schema = next(s for s in binding.schemas if s["function"]["name"] == "record_update")
    assert schema["function"]["parameters"]["required"] == [
        "app",
        "collection",
        "record_id",
        "field",
        "value",
    ]

    outcome = binding.call(
        "record_update",
        {
            "app": "salesforce",
            "collection": "contacts",
            "record_id": "003001",
            "field": "Phone",
            "value": "+1-555-0101",
        },
    )
    assert outcome["result"]["updated"] is True
    assert binding.final_state()["salesforce"]["contacts"][0]["Phone"] == "+1-555-0101"
    assert "error" in binding.call("no_such_tool", {})


def test_react_loop_executes_tools_and_returns_final_state() -> None:
    tools = _load_tools()
    wm_react = _load_wm_react()
    task = _sample_tasks()[0]
    binding = tools.build_binding(
        initial_state=task["initial_state"],
        zapier_tools=task["zapier_tools"],
        toolset="limited_zapier",
        prefer_upstream=False,
    )
    policy = _ScriptedPolicy(
        [
            _assistant(
                "records_search", {"app": "salesforce", "collection": "contacts", "query": "ada"}
            ),
            _assistant(
                "record_update",
                {
                    "app": "salesforce",
                    "collection": "contacts",
                    "record_id": "003001",
                    "field": "Phone",
                    "value": "+1-555-0101",
                },
            ),
            {"role": "assistant", "content": "Updated Ada's phone number.", "tool_calls": []},
        ]
    )
    result = asyncio.run(
        wm_react.run_single_task_with_wm(
            payload={
                "prompt": task["prompt"],
                "task_name": task["name"],
                "domain": task["domain"],
                "max_turns": 10,
            },
            llm_client=policy,
            binding=binding,
        )
    )
    # WM_STRATEGY is unset in the test environment -> plain ReAct baseline.
    assert result["wm_strategy"] == "none"
    assert result["error"] is None
    assert result["num_tool_calls"] == 2
    assert result["tools_used"] == ["records_search", "record_update"]
    assert result["final_state"]["salesforce"]["contacts"][0]["Phone"] == "+1-555-0101"
    assert result["final_response"] == "Updated Ada's phone number."

    # ... and the state Purple returns is exactly what Green scores.
    scoring = _load_scoring()
    scores = scoring.score_task(
        assertions=task["assertions"],
        initial_state=task["initial_state"],
        final_state=result["final_state"],
        prefer_upstream=False,
    )
    assert scores[scoring.PASS_KEY] == 1.0


def test_react_loop_stops_at_max_turns() -> None:
    tools = _load_tools()
    wm_react = _load_wm_react()
    task = _sample_tasks()[0]
    binding = tools.build_binding(
        initial_state=task["initial_state"],
        zapier_tools=task["zapier_tools"],
        toolset="limited_zapier",
        prefer_upstream=False,
    )
    looping = _ScriptedPolicy(
        [_assistant("records_search", {"app": "salesforce", "collection": "contacts"})] * 6
    )
    result = asyncio.run(
        wm_react.run_single_task_with_wm(
            payload={"prompt": task["prompt"], "max_turns": 3},
            llm_client=looping,
            binding=binding,
        )
    )
    assert result["num_model_calls"] == 3
    assert "max_turns" in (result["error"] or "")


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------


def test_benchmark_toml_declares_green_purple_and_executor() -> None:
    import tomllib

    config = tomllib.loads((_BENCH_DIR / "benchmark.toml").read_text(encoding="utf-8"))
    assert config["name"] == "AutomationBench"
    assert config["default_executor"] == "mcp_react"
    assert config["config"]["target"] == "sample"
    assert config["green_agent"]["entrypoint"] == "green/automationbench_green_agent.py"
    assert config["participants"][0]["role"] == "agent"
    assert (_BENCH_DIR / config["green_agent"]["entrypoint"]).is_file()
    assert (_BENCH_DIR / config["participants"][0]["entrypoint"]).is_file()
    assert (_EXECUTOR_DIR / "executor.py").is_file()
    assert (_EXECUTOR_DIR / "version.toml").is_file()


def test_sample_task_never_binds_upstream_tools() -> None:
    """A `bundled_sample` task uses the sample world even when upstream is importable.

    The bundled fixture stores Salesforce records under the API's own CamelCase names
    (`FirstName`, `Phone`); upstream's `WorldState` requires snake_case and forbids
    extras, so binding it upstream raised a pydantic ValidationError before the agent
    ever acted. Routing is by task origin, so importability cannot reintroduce that.
    """
    tools = _load_tools()
    task = _sample_tasks()[0]

    def _boom(**_kwargs):
        raise AssertionError("upstream binding must not be attempted for a sample task")

    monkey = {"upstream_available": lambda: True, "build_upstream_binding": _boom}
    saved = {name: getattr(tools, name) for name in monkey}
    for name, value in monkey.items():
        setattr(tools, name, value)
    try:
        binding = tools.build_binding(
            initial_state=task["initial_state"],
            zapier_tools=task["zapier_tools"],
            toolset="limited_zapier",
            source="bundled_sample",
        )
    finally:
        for name, value in saved.items():
            setattr(tools, name, value)
    assert binding.source == "bundled_sample"
    assert binding.final_state()["salesforce"]["contacts"][0]["Phone"] == "+1-555-9999"


def test_green_payload_tells_purple_the_task_source() -> None:
    green_dir_added = str(_GREEN_DIR) not in sys.path
    if green_dir_added:
        sys.path.insert(0, str(_GREEN_DIR))
    green = _load_module(
        "automationbench_green_under_test", _GREEN_DIR / "automationbench_green_agent.py"
    )
    task = _sample_tasks()[0]
    payload = json.loads(green._build_purple_payload(task, {"toolset": "limited_zapier"}))
    assert payload["source"] == "bundled_sample"
    assert "assertions" not in payload  # ground truth still stays with Green


def test_tool_schema_carries_parameter_docs_and_required_args() -> None:
    """Upstream defaults every argument to None but documents the real contract.

    Without parsing the `Args:` block the model saw a list of undocumented optional
    strings; in the 600-task baseline it omitted the spreadsheet id on every Google
    Sheets call and looped until max_turns (the `hr` domain scored 0.083).
    """
    tools = _load_tools()

    def google_sheets_find_worksheet(
        world=None, spreadsheet=None, title=None, spreadsheet_id=None, drive=None
    ) -> str:
        """Find a worksheet by title.

        Args:
            spreadsheet: Spreadsheet ID (required).
            title: Worksheet title to search for (required).
            spreadsheet_id: Alias for spreadsheet.
            drive: Google Drive location.

        Returns:
            JSON string with matching worksheet.
        """
        return "{}"

    schema = tools.tool_schema(google_sheets_find_worksheet)["function"]
    assert schema["description"] == "Find a worksheet by title."
    properties = schema["parameters"]["properties"]
    assert "world" not in properties
    assert properties["spreadsheet"]["description"] == "Spreadsheet ID (required)."
    assert properties["drive"]["description"] == "Google Drive location."
    # Documented "(required)" wins over the Optional=None signature default.
    assert schema["parameters"]["required"] == ["spreadsheet", "title"]
    # "Returns:" must not leak into the summary or the parameter map.
    assert "JSON string" not in schema["description"]


def test_dispatcher_decodes_json_string_container_arguments() -> None:
    """Models routinely serialise object arguments as a string; decode them."""
    tools = _load_tools()
    task = _sample_tasks()[1]
    binding = tools.build_binding(
        initial_state=task["initial_state"],
        zapier_tools=task["zapier_tools"],
        toolset="limited_zapier",
        source="bundled_sample",
    )
    outcome = binding.call(
        "record_create",
        {"app": "slack", "collection": "messages", "fields": '{"channel": "C100", "text": "hi"}'},
    )
    assert "error" not in outcome, outcome
    assert outcome["result"]["record"]["channel"] == "C100"
    # A string that is not JSON still reaches the tool, which raises its own error.
    broken = binding.call(
        "record_create", {"app": "slack", "collection": "messages", "fields": "not json"}
    )
    assert "error" in broken
