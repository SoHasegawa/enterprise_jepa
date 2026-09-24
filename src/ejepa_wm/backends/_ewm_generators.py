"""EWM model-backend factory — vLLM (served) and transformers (local HF).

The single-step EWM prediction primitive (:func:`ejepa_wm.backends._ewm_runtime.predict_wm_feedback`)
only needs a *generator* exposing ``generate_from_messages(messages, temperature=0.0) -> str``.
Two backends provide that, selected by env / config so the EWM MCP server can run the
same model the way it is served today:

* ``vllm``         — :class:`~ejepa_wm.backends._ewm_runtime.EwmGenerator`, a thin stdlib proxy
  to an OpenAI-compatible (vLLM) ``/chat/completions`` endpoint; and
* ``transformers`` — :class:`HFEwmGenerator`, a local HuggingFace checkpoint loaded
  in-process (``torch`` + ``transformers`` imported lazily, so this module stays importable
  without them when only the vLLM backend is used).

Resolution is split from construction on purpose: :func:`resolve_ewm_backend` is a pure,
side-effect-free read of the environment (unit-testable without loading any model), and
:func:`build_ewm_generator` does the actual (possibly heavy) instantiation.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ejepa_wm.backends._ewm_runtime import EwmGenerator

logger = logging.getLogger(__name__)

BACKEND_VLLM = "vllm"
BACKEND_TRANSFORMERS = "transformers"

_VLLM_PREFIXES = ("vllm/",)
_TRANSFORMERS_ALIASES = {"transformers", "hf", "huggingface", "local"}

_DEFAULT_MODEL = "gymops_world_model"
_DEFAULT_MAX_NEW_TOKENS = 512
_DEFAULT_TIMEOUT = 600.0


def _env(env: dict[str, str] | None, *names: str) -> str | None:
    """First non-empty value among ``names`` in ``env`` (falls back to ``os.environ``)."""
    source = env if env is not None else os.environ
    for name in names:
        value = source.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return None


def _env_int(env: dict[str, str] | None, default: int, *names: str) -> int:
    raw = _env(env, *names)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(env: dict[str, str] | None, default: float, *names: str) -> float:
    raw = _env(env, *names)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(env: dict[str, str] | None, default: bool, *names: str) -> bool:
    raw = _env(env, *names)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ResolvedEwmBackend:
    """A pure description of which EWM backend to build and with what parameters."""

    kind: str = BACKEND_VLLM            # vllm | transformers
    model: str = _DEFAULT_MODEL         # vLLM served model id
    base_url: str | None = None      # vLLM OpenAI-compatible base url (.../v1)
    api_key: str | None = None       # vLLM api key (usually "not-needed")
    model_path: str | None = None    # transformers checkpoint / LoRA adapter dir
    max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS
    timeout: float = _DEFAULT_TIMEOUT
    # transformers-only knobs
    dtype: str = "auto"
    trust_remote_code: bool = False
    device_map: str | None = None
    attn_implementation: str = "sdpa"

    def info(self) -> dict[str, Any]:
        """Compact, secret-free description for the server's ``info()`` tool."""
        base = {
            "backend": self.kind,
            "model": self.model,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.kind == BACKEND_VLLM:
            base["endpoint"] = self.base_url
        else:
            base.update(
                {
                    "model_path": self.model_path,
                    "dtype": self.dtype,
                    "device_map": self.device_map,
                }
            )
        return base


def _looks_like_local_path(value: str) -> bool:
    """Heuristic: a transformers checkpoint dir/file rather than a served-model id."""
    if not value:
        return False
    if value.endswith(".gguf"):
        return True
    return os.path.sep in value and Path(value).exists()


def resolve_ewm_backend(env: dict[str, str] | None = None) -> ResolvedEwmBackend:
    """Decide the EWM backend from ``EWM_*`` / ``WM_*`` env (no model is loaded here).

    Selection (mirrors the executor's ``build_agent_generator`` prefix detection):

    * ``EWM_WORLD_MODEL_METHOD=vllm/<model>``                      -> vLLM (model after the prefix)
    * ``EWM_WORLD_MODEL_METHOD`` in {transformers,hf,huggingface}  -> transformers
    * ``EWM_WORLD_MODEL_PATH`` set (and no vllm/ method)           -> transformers
    * a method that looks like a local path                       -> transformers
    * otherwise                                                    -> vLLM (default model)
    """
    method = (_env(env, "EWM_WORLD_MODEL_METHOD", "WM_EWM_METHOD") or "").strip()
    model_path = _env(env, "EWM_WORLD_MODEL_PATH", "WM_EWM_MODEL_PATH")
    max_new_tokens = _env_int(
        env, _DEFAULT_MAX_NEW_TOKENS, "WM_EWM_MAX_NEW_TOKENS", "EWM_MAX_NEW_TOKENS"
    )
    timeout = _env_float(env, _DEFAULT_TIMEOUT, "EWM_TIMEOUT", "WM_SCORER_TIMEOUT")

    method_lower = method.lower()
    is_vllm_method = any(method_lower.startswith(p) for p in _VLLM_PREFIXES)
    is_transformers_method = method_lower in _TRANSFORMERS_ALIASES or (
        bool(method) and not is_vllm_method and _looks_like_local_path(method)
    )

    use_transformers = is_transformers_method or (bool(model_path) and not is_vllm_method)

    if use_transformers:
        # method may itself be the checkpoint path (when not an alias keyword).
        path = model_path or (method if method_lower not in _TRANSFORMERS_ALIASES else None)
        if not path:
            raise ValueError(
                "transformers EWM backend requires EWM_WORLD_MODEL_PATH (the HF checkpoint dir)."
            )
        return ResolvedEwmBackend(
            kind=BACKEND_TRANSFORMERS,
            model=path,
            model_path=path,
            max_new_tokens=max_new_tokens,
            timeout=timeout,
            dtype=_env(env, "EWM_WORLD_MODEL_DTYPE") or "auto",
            trust_remote_code=_env_bool(env, False, "EWM_WORLD_MODEL_TRUST_REMOTE_CODE"),
            device_map=_env(env, "EWM_WORLD_MODEL_DEVICE_MAP"),
            attn_implementation=_env(env, "EWM_WORLD_MODEL_ATTN") or "sdpa",
        )

    # vLLM (default). Model id is the part after "vllm/" if given, else WM_EWM_MODEL.
    model = re.sub(r"^vllm/", "", method) if is_vllm_method else ""
    model = model or _env(env, "WM_EWM_MODEL") or _DEFAULT_MODEL
    port = _env(env, "WM_VLLM_SERVER_PORT", "EWM_VLLM_SERVER_PORT", "VLLM_SERVER_PORT") or "9000"
    base_url = (
        _env(env, "WM_VLLM_BASE_URL", "VLLM_BASE_URL") or f"http://127.0.0.1:{port}/v1"
    ).rstrip("/")
    api_key = _env(env, "WM_VLLM_API_KEY", "VLLM_API_KEY", "OPENAI_API_KEY") or "not-needed"
    return ResolvedEwmBackend(
        kind=BACKEND_VLLM,
        model=model,
        base_url=base_url,
        api_key=api_key,
        max_new_tokens=max_new_tokens,
        timeout=timeout,
    )


class HFEwmGenerator:
    """Local HuggingFace EWM generator exposing ``generate_from_messages``.

    Self-contained (no import from the vendored ``mcp_react_ewm`` package): ``torch`` and
    ``transformers`` are imported lazily at construction so this module loads without them
    when only the vLLM backend is used. Supports a plain checkpoint dir or a PEFT/LoRA
    adapter dir (detected via ``adapter_config.json``).
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = _DEFAULT_MAX_NEW_TOKENS,
        *,
        dtype: str = "auto",
        trust_remote_code: bool = False,
        device_map: str | None = None,
        attn_implementation: str = "sdpa",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional heavy deps
            raise RuntimeError(
                "The transformers EWM backend requires `torch` and `transformers`. "
                "Install them, or use the vllm backend (EWM_WORLD_MODEL_METHOD=vllm/<model>)."
            ) from exc

        self._torch = torch
        self.model_path = model_path
        self.max_new_tokens = max(1, int(max_new_tokens))
        resolved_dtype = getattr(torch, dtype) if dtype not in ("", "auto") else "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        adapter_config = Path(model_path) / "adapter_config.json"
        common = {
            "dtype": resolved_dtype,
            "trust_remote_code": trust_remote_code,
            "device_map": device_map or "auto",
            "attn_implementation": attn_implementation,
        }
        if adapter_config.is_file():
            import json as _json

            base = _json.loads(adapter_config.read_text(encoding="utf-8")).get(
                "base_model_name_or_path"
            )
            if not base:
                raise ValueError(f"Missing base_model_name_or_path in {adapter_config}")
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("Loading a LoRA adapter requires `peft`.") from exc
            base_model = AutoModelForCausalLM.from_pretrained(base, **common)
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **common)

        self._input_device = next(self.model.parameters()).device

    def generate_from_messages(
        self, messages: list[dict[str, str]], temperature: float = 0.0
    ) -> str:
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._input_device) for k, v in inputs.items()}
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": 0.95})
        else:
            gen_kwargs["do_sample"] = False
        with self._torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_ewm_generator(
    resolved: ResolvedEwmBackend | None = None, env: dict[str, str] | None = None
) -> Any:
    """Instantiate the EWM generator for the resolved backend (loads the model if local)."""
    resolved = resolved or resolve_ewm_backend(env)
    if resolved.kind == BACKEND_TRANSFORMERS:
        logger.info("EWM backend=transformers model_path=%s", resolved.model_path)
        return HFEwmGenerator(
            resolved.model_path or resolved.model,
            max_new_tokens=resolved.max_new_tokens,
            dtype=resolved.dtype,
            trust_remote_code=resolved.trust_remote_code,
            device_map=resolved.device_map,
            attn_implementation=resolved.attn_implementation,
        )
    logger.info("EWM backend=vllm model=%s endpoint=%s", resolved.model, resolved.base_url)
    return EwmGenerator(
        resolved.model,
        resolved.base_url or "http://127.0.0.1:9000/v1",
        resolved.api_key or "not-needed",
        max_new_tokens=resolved.max_new_tokens,
        timeout=resolved.timeout,
    )


def _parse_mcp_body(content_type: str, raw: str) -> dict[str, Any]:
    """Parse a streamable-HTTP MCP reply — plain JSON or SSE ``data:`` frames."""
    if "text/event-stream" in (content_type or ""):
        last: dict[str, Any] = {}
        for line in raw.splitlines():
            if line.startswith("data:"):
                with contextlib.suppress(ValueError):
                    last = json.loads(line[5:].strip())
        return last
    try:
        return json.loads(raw)
    except ValueError:
        return {}


def _extract_generated_text(data: dict[str, Any]) -> str:
    """Pull the ``generate`` tool's text out of a ``tools/call`` result."""
    result = (data or {}).get("result") or {}
    structured = result.get("structuredContent")
    if isinstance(structured, dict) and "text" in structured:
        return str(structured["text"])
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            try:
                obj = json.loads(text)
            except (ValueError, TypeError):
                return str(text)
            if isinstance(obj, dict) and "text" in obj:
                return str(obj["text"])
            return str(text)
    return ""


class McpEwmGenerator:
    """EWM generator that reaches the EWM predict MCP server's ``generate`` tool over HTTP.

    Lets the imagined-trajectory rollout (a client-side agent↔WM loop) call a **remote** EWM
    as an MCP service: ``generate_from_messages`` issues an MCP ``tools/call`` to ``generate``.
    stdlib-only (``urllib``), handshakes once, and parses both JSON and SSE responses — so it
    drops into the rollout helpers wherever an in-process :class:`EwmGenerator` would be used.
    """

    def __init__(
        self,
        base_url: str,
        *,
        mcp_endpoint: str | None = None,
        tool_name: str = "generate",
        api_key: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        base = base_url.rstrip("/")
        if mcp_endpoint is None:
            self.url = base if base.endswith("/mcp") else base + "/mcp"
        else:
            self.url = base + mcp_endpoint
        self.tool_name = tool_name
        self.api_key = api_key
        self.timeout = float(timeout)
        self._request_id = 0
        self._session_id: str | None = None
        self._initialized = False

    def _post(self, payload: dict[str, Any], *, notification: bool = False) -> dict[str, Any]:
        envelope: dict[str, Any] = {"jsonrpc": "2.0"}
        if not notification:
            self._request_id += 1
            envelope["id"] = self._request_id
        envelope.update(payload)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        request = urllib.request.Request(
            self.url, data=json.dumps(envelope).encode("utf-8"), headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            session_id = response.headers.get("mcp-session-id")
            if session_id:
                self._session_id = session_id
            content_type = response.headers.get("Content-Type", "")
            raw = response.read().decode("utf-8", "replace")
        return {} if notification else _parse_mcp_body(content_type, raw)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._post(
            {
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "ewm-imagined", "version": "0.1"},
                },
            }
        )
        self._post({"method": "notifications/initialized", "params": {}}, notification=True)
        self._initialized = True

    def generate_from_messages(
        self, messages: list[dict[str, str]], temperature: float = 0.0
    ) -> str:
        self._ensure_initialized()
        data = self._post(
            {
                "method": "tools/call",
                "params": {
                    "name": self.tool_name,
                    "arguments": {"messages": messages, "temperature": float(temperature)},
                },
            }
        )
        return _extract_generated_text(data)


def normalize_planned_calls(calls: Any) -> list[dict[str, Any]]:
    """Map agent/candidate tool calls to the ``{name, arguments}`` shape the WM expects.

    Fixes two mismatches that otherwise corrupt the prediction:

    * candidate/agent events carry arguments under ``args`` (or nested ``function.arguments``),
      but the runtime's ``normalize_tool_call`` reads ``arguments`` — so the WM would see
      **empty args**; and
    * policy LLMs namespace tool names (e.g. gpt-5.1's ``functions.create_calendar``), which
      is out-of-distribution for the WM — strip a leading ``<ns>.`` so the name matches the
      gym tool the WM was trained on.
    """
    out: list[dict[str, Any]] = []
    for call in calls or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else None
        name = str((fn or call).get("name") or "")
        if "." in name:
            name = name.rsplit(".", 1)[-1]
        args = call.get("arguments")
        if args is None and fn is not None:
            args = fn.get("arguments")
        if args is None:
            args = call.get("args", {})
        out.append({"name": name, "arguments": args})
    return out


__all__ = [
    "BACKEND_TRANSFORMERS",
    "BACKEND_VLLM",
    "HFEwmGenerator",
    "McpEwmGenerator",
    "ResolvedEwmBackend",
    "build_ewm_generator",
    "normalize_planned_calls",
    "resolve_ewm_backend",
]
