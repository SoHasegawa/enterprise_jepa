"""Tool binding for the AutomationBench `mcp_react` executor.

Upstream AutomationBench tools are plain typed Python callables whose first
parameter is ``world: WorldState`` (see ``automationbench/tools/api/impl/*.py``).
Upstream registers them into a ``verifiers`` StatefulToolEnv; this executor owns
its own ReAct loop instead, so it needs three things from them:

1. the tool list for the requested toolset (``api`` -> ``API_TOOLS``,
   ``zapier``/``limited_zapier`` -> ``ALL_TOOLS`` filtered per task),
2. OpenAI function schemas derived from the signatures, and
3. a dispatcher that injects the live ``WorldState`` and returns JSON-able output.

Importing only ``automationbench.tools`` / ``automationbench.schema.world`` keeps
the ``verifiers`` dependency out of the purple environment -- only upstream's
``runner``/``create_rubric`` need it.

A bundled fallback world (three generic tools over a dict of collections) runs the
offline ``sample`` target so the loop and the world-model wiring can be exercised
without the upstream checkout.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Any

_EXECUTOR_DIR = Path(__file__).resolve().parent
_BENCHMARK_DIR = _EXECUTOR_DIR.parents[1]

_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def candidate_repo_paths() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.getenv("AUTOMATIONBENCH_REPO_PATH")
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    link = _BENCHMARK_DIR / "AutomationBench"
    if link.exists():
        candidates.append(link.resolve())
    benchmark_home = os.getenv("BENCHMARK_HOME")
    if benchmark_home:
        candidates.append(
            (Path(benchmark_home) / "repos" / "AutomationBench").expanduser().resolve()
        )
    candidates.append((_BENCHMARK_DIR.parents[1] / "tools" / "AutomationBench").resolve())
    return candidates


def upstream_available() -> bool:
    return any((path / "automationbench" / "tools").is_dir() for path in candidate_repo_paths())


def ensure_upstream_importable() -> Path:
    for path in candidate_repo_paths():
        if (path / "automationbench" / "tools").is_dir():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return path
    raise RuntimeError(
        "AutomationBench repository not found. Clone "
        "https://github.com/zapier/AutomationBench and set AUTOMATIONBENCH_REPO_PATH."
    )


def resolve_signature(tool: Callable[..., Any]) -> inspect.Signature:
    """``inspect.signature`` with annotations evaluated to real types.

    This module (and upstream's newer tool modules) use ``from __future__ import
    annotations``, so raw annotations arrive as STRINGS. Without evaluating them
    ``dict[str, Any]`` fell through to ``{"type": "string"}``, which told the model
    to send the object as a JSON string -- and the tool then crashed on it.
    """
    try:
        return inspect.signature(tool, eval_str=True)
    except (NameError, TypeError, ValueError):
        return inspect.signature(tool)


def _json_type(annotation: Any) -> dict[str, Any]:
    """Best-effort JSON-schema fragment for a Python annotation."""
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}
    origin = typing.get_origin(annotation)
    if origin is typing.Union or str(origin) == "types.UnionType":
        args = [arg for arg in typing.get_args(annotation) if arg is not type(None)]
        return _json_type(args[0]) if args else {"type": "string"}
    if origin in (list, tuple, set):
        return {"type": "array", "items": {"type": "string"}}
    if origin is dict:
        return {"type": "object"}
    return {"type": _JSON_TYPES.get(annotation, "string")}


_ARGS_HEADINGS = ("args:", "arguments:", "parameters:")
_END_HEADINGS = ("returns:", "return:", "raises:", "yields:", "examples:", "example:", "note:")
_REQUIRED_MARKERS = ("(required)", "(required, ", "required.")


def parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into (summary, {parameter: description}).

    Upstream's tool signatures are almost entirely ``Optional[str] = None``, so the
    docstring's ``Args:`` block is the ONLY place the real contract lives -- e.g.
    ``spreadsheet: Spreadsheet ID (required).``. Dropping it left the model with a
    list of undocumented optional strings; it then omitted the spreadsheet id and
    every sheet lookup failed with ``not found in spreadsheet ''``.
    """
    summary_lines: list[str] = []
    params: dict[str, str] = {}
    section = "summary"
    current: str | None = None
    for raw in (doc or "").splitlines():
        line = raw.strip()
        lowered = line.lower()
        if lowered in _ARGS_HEADINGS:
            section, current = "args", None
            continue
        if lowered in _END_HEADINGS:
            section, current = "other", None
            continue
        if section == "summary":
            summary_lines.append(line)
        elif section == "args" and line:
            name, separator, description = line.partition(":")
            if separator and name and " " not in name.strip():
                current = name.strip()
                params[current] = description.strip()
            elif current:  # continuation of the previous parameter's description
                params[current] = f"{params[current]} {line}".strip()
    summary = " ".join(part for part in summary_lines if part).strip()
    return summary, params


def tool_schema(tool: Callable[..., Any]) -> dict[str, Any]:
    """OpenAI function schema for one upstream tool callable."""
    signature = resolve_signature(tool)
    summary, param_docs = parse_docstring(inspect.getdoc(tool) or "")
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if name == "world" or name.startswith("_"):
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        schema = _json_type(parameter.annotation)
        description = param_docs.get(name, "")
        if description:
            schema["description"] = description
        properties[name] = schema
        # Trust the docstring over the signature: upstream defaults everything to
        # None but documents which arguments the tool actually cannot work without.
        documented_required = any(marker in description.lower() for marker in _REQUIRED_MARKERS)
        if parameter.default is inspect.Parameter.empty or documented_required:
            required.append(name)
    return {
        "type": "function",
        "function": {
            "name": tool.__name__,
            "description": (summary or tool.__name__)[:900],
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


class ToolBinding:
    """The tools, their schemas and the live world for one task."""

    def __init__(
        self,
        *,
        world: Any,
        tools: dict[str, Callable[..., Any]],
        schemas: list[dict[str, Any]],
        world_dump: Callable[[Any], dict[str, Any]],
        source: str,
    ) -> None:
        self.world = world
        self.tools = tools
        self.schemas = schemas
        self._world_dump = world_dump
        self.source = source

    def final_state(self) -> dict[str, Any]:
        return self._world_dump(self.world)

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute one tool call, returning ``{"result": ...}`` or ``{"error": ...}``."""
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"unknown tool {name!r}"}
        kwargs = dict(arguments or {})
        try:
            signature = resolve_signature(tool)
            kwargs = _coerce_arguments(kwargs, signature)
            if "world" in signature.parameters:
                kwargs["world"] = self.world
            accepted = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters or _accepts_kwargs(signature)
            }
            result = tool(**accepted)
        except Exception as exc:  # noqa: BLE001 - tool errors are observations
            return {"error": f"{type(exc).__name__}: {exc}"}
        return {"result": _jsonable(result)}


def _accepts_kwargs(signature: inspect.Signature) -> bool:
    return any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values())


def _wants_container(annotation: Any) -> bool:
    """True when the parameter expects a dict/list rather than a scalar."""
    if annotation is inspect.Parameter.empty:
        return False
    origin = typing.get_origin(annotation)
    if origin is typing.Union or str(origin) == "types.UnionType":
        return any(_wants_container(arg) for arg in typing.get_args(annotation))
    return (origin or annotation) in (dict, list, tuple, set)


def _coerce_arguments(arguments: dict[str, Any], signature: inspect.Signature) -> dict[str, Any]:
    """Parse JSON-string arguments for parameters that expect a container.

    Tool-calling models routinely serialise object/array arguments as a *string*
    (``fields="{\"text\": \"hi\"}"``). Passing that straight through made every
    such call fail with ``ValueError: dictionary update sequence element #0 has
    length 1; 2 is required``, which the agent cannot recover from -- one sample
    task burned 14 of its 17 tool calls on it. Decode here so the tool sees the
    container its signature asks for; leave anything undecodable untouched so the
    tool still raises its own, more informative error.
    """
    coerced = dict(arguments)
    for name, parameter in signature.parameters.items():
        value = coerced.get(name)
        if not isinstance(value, str) or not _wants_container(parameter.annotation):
            continue
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, (dict, list)):
            coerced[name] = parsed
    return coerced


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def build_upstream_binding(
    *, initial_state: dict[str, Any], zapier_tools: list[str], toolset: str
) -> ToolBinding:
    """Bind upstream tools to a fresh ``WorldState`` for one task."""
    ensure_upstream_importable()
    from automationbench.schema.world import WorldState
    from automationbench.tools import ALL_TOOLS
    from automationbench.tools.api import API_TOOLS

    world = WorldState(**(initial_state or {}))
    pool: list[Callable[..., Any]] = list(API_TOOLS if toolset == "api" else ALL_TOOLS)
    if toolset == "limited_zapier" and zapier_tools:
        allowed = {str(name) for name in zapier_tools}
        filtered = [tool for tool in pool if tool.__name__ in allowed]
        # An unmatched allow-list would leave the agent with no tools at all; keep
        # the full pool in that case rather than silently crippling the task.
        pool = filtered or pool
    tools = {tool.__name__: tool for tool in pool}
    return ToolBinding(
        world=world,
        tools=tools,
        schemas=[tool_schema(tool) for tool in pool],
        world_dump=lambda w: json.loads(w.model_dump_json(exclude_defaults=True)),
        source=f"upstream:{toolset}",
    )


# --- bundled fallback world (offline `sample` target) ------------------------


class _SampleWorld:
    """Minimal dict-of-collections world: ``{app: {collection: [records]}}``."""

    def __init__(self, state: dict[str, Any]) -> None:
        self.state: dict[str, Any] = json.loads(json.dumps(state or {}))

    def collection(self, app: str, collection: str) -> list[dict[str, Any]]:
        return self.state.setdefault(app, {}).setdefault(collection, [])


def build_sample_binding(*, initial_state: dict[str, Any]) -> ToolBinding:
    """Three generic tools over the bundled sample world."""
    world = _SampleWorld(initial_state)

    def records_search(app: str, collection: str, query: str = "") -> list[dict[str, Any]]:
        """Search records in one app collection; empty query returns everything."""
        rows = world.collection(app, collection)
        if not query:
            return rows
        needle = str(query).lower()
        return [row for row in rows if needle in json.dumps(row).lower()]

    def record_update(
        app: str, collection: str, record_id: str, field: str, value: str
    ) -> dict[str, Any]:
        """Set one field on an existing record, matched by its ``id``."""
        for row in world.collection(app, collection):
            if str(row.get("id")) == str(record_id):
                row[field] = value
                return {"updated": True, "record": row}
        return {"updated": False, "error": f"no record {record_id!r} in {app}.{collection}"}

    def record_create(app: str, collection: str, fields: dict[str, Any]) -> dict[str, Any]:
        """Append a new record to an app collection."""
        rows = world.collection(app, collection)
        record = dict(fields or {})
        record.setdefault("id", f"{collection[:3]}{len(rows) + 1:04d}")
        rows.append(record)
        return {"created": True, "record": record}

    pool = [records_search, record_update, record_create]
    return ToolBinding(
        world=world,
        tools={tool.__name__: tool for tool in pool},
        schemas=[tool_schema(tool) for tool in pool],
        world_dump=lambda w: w.state,
        source="bundled_sample",
    )


BUNDLED_SAMPLE_SOURCE = "bundled_sample"


def build_binding(
    *,
    initial_state: dict[str, Any],
    zapier_tools: list[str],
    toolset: str,
    source: str = "",
    prefer_upstream: bool = True,
) -> ToolBinding:
    """Upstream tools for upstream tasks, the bundled sample world for sample tasks.

    Routing is by task ORIGIN, not by whether ``automationbench`` happens to be
    importable: the bundled ``sample`` fixture models Salesforce records with the
    API's own CamelCase field names, which upstream's ``WorldState`` rejects (it
    requires snake_case and forbids extras). Binding it to upstream tools raised a
    pydantic ``ValidationError`` before the agent ever acted.
    """
    if source == BUNDLED_SAMPLE_SOURCE or not (prefer_upstream and upstream_available()):
        return build_sample_binding(initial_state=initial_state)
    return build_upstream_binding(
        initial_state=initial_state, zapier_tools=zapier_tools, toolset=toolset
    )
