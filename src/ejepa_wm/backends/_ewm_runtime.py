"""Self-contained Enterprise World Model (EWM) helpers for the ``imagined`` strategy.

This module is a deliberate, trimmed **port** of the imagined-mode / world-model
path from the EWM repo (``ewm/src/evaluation.py`` + ``ewm/src/finetuning.py``),
rewritten to stand on its own. It has **no imports** from the sibling
``mcp_react_ewm`` package and does not need an external EWM checkout on
``sys.path`` — it depends only on the standard library.

``wm_react.py`` imports it by bare module name (``import wm_ewm``) and feeds it
two duck-typed generators, each exposing
``generate_from_messages(messages, temperature=0.0) -> str``:

* the **world-model generator** — :class:`EwmGenerator`, the fine-tuned
  ``gymops_world_model`` served over a vLLM OpenAI-compatible endpoint; and
* the **agent generator** — an adapter over ``wm_react``'s own
  ``self.llm_client`` (built in ``wm_react.py`` so this module stays
  LangChain-free).

Two axes are configurable from ``wm_react.py``:

* ``WM_STATE`` — what the WM predicts. ``parse_wm_prediction`` understands
  ``binary_error`` (``1`` / ``0,<error>``), ``binary_error_stage`` (JSON with
  stages), and ``tool_output`` (raw tool output, failure inferred heuristically).
* ``ACTION_OPTIMIZER`` — how the imagined trajectory is produced.
  :func:`optimize_imagined_trajectory` dispatches ``topk_search`` (beam search)
  vs. plain imagine (a single linear rollout); both return ``imagined_steps``
  that :func:`build_imagined_trajectory_message` injects into the agent prompt.

State is kept in the **compact** shape produced by
:func:`make_tool_execution_prediction_state`
(``{success, last_tool_execution_result, error_message, current_stage,
remaining_stages}``) so we avoid the heavy enterprise-state schema.
"""
from __future__ import annotations

import ast
import json
import logging
import re
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Tuple

# The real EWM training-time state + prompt machinery (vendored verbatim from the
# EWM repo's finetuning.py). Using these guarantees the WM sees byte-identical
# input to what it was fine-tuned on (enterprise state schema, sanitized, with the
# `[{interaction_index, state}]` history rendering) — per WM_STATE target mode.
from ejepa_wm.backends import _canonical_event_state as canonical
from ejepa_wm.backends import _ewm_finetuning as ft

logger = logging.getLogger(__name__)

# Supported WM_STATE values.
WM_STATE_BINARY_ERROR = "binary_error"
WM_STATE_BINARY_ERROR_STAGE = "binary_error_stage"
WM_STATE_TOOL_OUTPUT = "tool_output"
WM_STATE_CANONICAL_NUDGE = "canonical_nudge"
WM_STATES = (
    WM_STATE_BINARY_ERROR,
    WM_STATE_BINARY_ERROR_STAGE,
    WM_STATE_TOOL_OUTPUT,
    WM_STATE_CANONICAL_NUDGE,
)

# Soft caps mirroring evaluation.py's _REPLAY_LIMITS (overridable via configure_replay_limits).
_REPLAY_LIMITS: Dict[str, int] = {"observation_chars": 2000, "history_budget_chars": 60000}


def configure_replay_limits(observation_chars: int, history_budget_chars: int) -> None:
    _REPLAY_LIMITS["observation_chars"] = max(0, int(observation_chars))
    _REPLAY_LIMITS["history_budget_chars"] = max(0, int(history_budget_chars))


# ---------------------------------------------------------------------------
# Generative world-model client (OpenAI-compatible / vLLM)
# ---------------------------------------------------------------------------


class EwmGenerator:
    """Minimal OpenAI-compatible chat client exposing ``generate_from_messages``.

    Talks to a vLLM (or any OpenAI-compatible) ``/chat/completions`` endpoint via
    ``urllib`` so this module pulls in no third-party dependency. Used only for
    the world model; the imagining agent reuses ``wm_react``'s ``llm_client``.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "not-needed",
        max_new_tokens: int = 512,
        timeout: float = 600.0,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "not-needed"
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.timeout = float(timeout)
        self.top_p = top_p
        self.top_k = top_k
        self._prefix_cache_checked = False

    def _post_chat_completions(self, body: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def generate_from_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": self.max_new_tokens,
        }
        if self.top_p is not None:
            body["top_p"] = float(self.top_p)
        if self.top_k is not None:
            body["top_k"] = int(self.top_k)
        if response_format is not None:
            body["response_format"] = response_format
        payload = self._post_chat_completions(body)
        self._warn_if_prefix_caching_inactive()
        choices = payload.get("choices") or []
        if not choices:
            return ""
        return str((choices[0].get("message") or {}).get("content") or "")

    def generate_samples(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        num_samples: int = 1,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """``num_samples`` independent samples of the SAME prompt in ONE request (``n=k``).

        The server prefills the shared prompt once and decodes the k sequences concurrently,
        so k candidates cost about one call instead of k -- see :func:`sample_many`, which
        prefers this seam over issuing k separate requests.
        """
        k = max(1, int(num_samples))
        if k == 1:
            return [
                self.generate_from_messages(
                    messages, temperature=temperature, response_format=response_format
                )
            ]
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": self.max_new_tokens,
            "n": k,
        }
        if self.top_p is not None:
            body["top_p"] = float(self.top_p)
        if self.top_k is not None:
            body["top_k"] = int(self.top_k)
        if response_format is not None:
            body["response_format"] = response_format
        payload = self._post_chat_completions(body)
        self._warn_if_prefix_caching_inactive()
        choices = payload.get("choices") or []
        return [str((choice.get("message") or {}).get("content") or "") for choice in choices]

    def _warn_if_prefix_caching_inactive(self) -> None:
        """One-shot check that this vLLM endpoint is actually reusing shared prompt prefixes.

        Every planning/revision call in ejepa_wm's imagined rollouts repeats the agent's system
        prompt and conversation verbatim and only appends a short instruction, so with prefix
        caching the prefill is nearly free and without it every call re-reads the whole prompt.
        A server started without ``--enable-prefix-caching`` reports zero prefix-cache queries,
        which is worth saying out loud once rather than paying for it silently.
        """
        if self._prefix_cache_checked:
            return
        self._prefix_cache_checked = True
        try:
            metrics_base = self.base_url.rsplit("/v1", 1)[0]
            req = urllib.request.Request(f"{metrics_base}/metrics", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            queries = None
            for line in body.splitlines():
                if line.startswith("vllm:prefix_cache_queries_total"):
                    queries = float(line.rsplit(" ", 1)[1])
                    break
            if queries is None:
                return  # counter absent: not a vLLM build that reports it, say nothing
            if queries <= 0.0:
                logger.warning(
                    "ewm_runtime: %s reports no prefix-cache activity at %s. Planning calls "
                    "re-share the agent's prompt prefix, so restart the server with "
                    "--enable-prefix-caching --enable-chunked-prefill to skip the repeated "
                    "prefill (this is typically the single largest world-model-side overhead).",
                    self.model, self.base_url,
                )
        except Exception:
            return


def _generate_with_optional_temperature(
    generator: Any,
    messages: List[Dict[str, Any]],
    *,
    temperature: float,
    response_format: Optional[Dict[str, Any]] = None,
) -> str:
    if response_format is not None:
        try:
            return generator.generate_from_messages(
                messages, temperature=temperature, response_format=response_format
            )
        except TypeError:
            pass
    try:
        return generator.generate_from_messages(messages, temperature=temperature)
    except TypeError:
        return generator.generate_from_messages(messages)


def build_temperature_ladder(
    num_samples: int, base_temperature: float, ladder_max: float = 1.2
) -> List[float]:
    """Spread samples over temperatures while keeping one near-greedy candidate.

    Identical samples tend to collapse onto the modal plan.  The lowest-temperature slot keeps
    that exploit candidate, while the remaining slots add controlled exploration.  The upper
    bound is intentionally capped because malformed actions become more common at high
    temperatures and truncate open-loop plans.
    """
    k = max(1, int(num_samples))
    base = max(0.0, float(base_temperature))
    if k == 1 or base <= 0.0:
        return [base] * k
    ceiling = max(0.0, float(ladder_max))
    top = min(ceiling, base * 1.7)
    bottom = min(base, max(0.05, base * 0.4))
    if k == 2:
        return [bottom, top]
    span = max(0.0, top - base * 0.85)
    return [bottom] + [
        round(base * 0.85 + span * index / (k - 2), 4) for index in range(k - 1)
    ]


def _requests_for_temperatures(generator: Any, temperatures: List[float]) -> int:
    """Return the number of backend requests used for per-sample temperatures."""
    if hasattr(generator, "generate_from_messages_batch"):
        return len(set(temperatures))
    return len(temperatures)


def sample_many(
    generator: Any,
    messages: List[Dict[str, Any]],
    *,
    temperature: float,
    num_samples: int,
    temperatures: Optional[List[float]] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> "tuple[List[str], int]":
    """``num_samples`` independent samples of the SAME prompt. Returns ``(texts, requests_issued)``.

    Prefers a backend that can produce k samples in ONE request/forward pass (``n=k`` on
    vLLM/OpenAI via :meth:`EwmGenerator.generate_samples`): the shared prompt is prefilled once
    and the k sequences decode concurrently, so k candidates cost ~one generation instead of k.
    Backends without that capability fall back to :func:`generate_many`, which issues k parallel
    requests (or one padded batch) -- same result, k prefills.

    ``requests_issued`` is reported so callers can log the true LLM call count rather than
    assuming it equals the number of samples.
    """
    k = max(1, int(num_samples))
    if k == 1:
        return [
            _generate_with_optional_temperature(
                generator, messages, temperature=temperature, response_format=response_format
            )
        ], 1
    if temperatures:
        # A single n=k request accepts only one sampling configuration.  Use the existing
        # multi-request dispatcher when each candidate needs its own temperature.
        ladder = list(temperatures)[:k]
        ladder += [ladder[-1]] * (k - len(ladder))
        texts = generate_many(generator, [messages] * k, ladder, response_format=response_format)
        return texts, _requests_for_temperatures(generator, ladder)
    sampler = getattr(generator, "generate_samples", None)
    if sampler is not None:
        try:
            if response_format is not None:
                try:
                    texts = sampler(
                        messages,
                        temperature=temperature,
                        num_samples=k,
                        response_format=response_format,
                    )
                except TypeError:
                    texts = sampler(messages, temperature=temperature, num_samples=k)
            else:
                texts = sampler(messages, temperature=temperature, num_samples=k)
        except NotImplementedError:
            texts = None
        if texts:
            return list(texts)[:k], 1
    return (
        generate_many(
            generator, [messages] * k, [temperature] * k, response_format=response_format
        ),
        k,
    )


def generate_many(
    generator: Any,
    messages_list: List[List[Dict[str, Any]]],
    temperatures: List[float],
    *,
    max_workers: int = 8,
    response_format: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Run several independent generations as one batch instead of a serial loop (used by
    :func:`imagine_trajectories_open_loop` to fire every still-pending plan's world-model
    prediction for one horizon step together). Dispatch, in order of preference:

    1. ``generate_from_messages_batch`` (a local/HF-style backend): requests sharing a
       temperature are grouped into one padded batched ``generate`` call.
    2. ``supports_parallel_requests`` backends: a thread pool of concurrent requests, capped at
       ``max_workers``. A backend whose calls mutate shared client state exposes
       ``clone_for_parallel_requests()``; each worker thread gets its own clone via
       ``threading.local()``. vLLM/OpenAI-style continuous batching turns these into genuine
       server-side batches.
    3. Anything else: the original serial loop.

    A failed parallel request retries once serially, so batching never introduces a new failure
    mode; results are always returned in input order.

    ``_ChatFnAgent`` declares ``supports_parallel_requests`` by default so heterogeneous
    open-loop plan prompts can arrive together at a vLLM/OpenAI-compatible server and benefit
    from continuous batching. Other backends may opt into either real batching seam without this
    dispatcher changing.
    """
    if len(messages_list) != len(temperatures):
        raise ValueError("messages_list and temperatures must have equal length")
    if not messages_list:
        return []
    if len(messages_list) == 1:
        return [
            _generate_with_optional_temperature(
                generator,
                messages_list[0],
                temperature=temperatures[0],
                response_format=response_format,
            )
        ]

    if response_format is None and hasattr(generator, "generate_from_messages_batch"):
        results: List[Optional[str]] = [None] * len(messages_list)
        by_temperature: Dict[float, List[int]] = {}
        for index, temperature in enumerate(temperatures):
            by_temperature.setdefault(float(temperature), []).append(index)
        for temperature, indices in by_temperature.items():
            outputs = generator.generate_from_messages_batch(
                [messages_list[index] for index in indices], temperature=temperature
            )
            for index, output in zip(indices, outputs):
                results[index] = output
        return [result if result is not None else "" for result in results]

    if getattr(generator, "supports_parallel_requests", False):
        import concurrent.futures
        import threading

        thread_generators = threading.local()

        def call(index: int) -> str:
            worker = generator
            if hasattr(generator, "clone_for_parallel_requests"):
                worker = getattr(thread_generators, "generator", None)
                if worker is None:
                    worker = generator.clone_for_parallel_requests()
                    thread_generators.generator = worker
            return _generate_with_optional_temperature(
                worker,
                messages_list[index],
                temperature=temperatures[index],
                response_format=response_format,
            )

        workers = max(1, min(int(max_workers), len(messages_list)))
        results = [None] * len(messages_list)
        failed: List[int] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(call, index): index for index in range(len(messages_list))}
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    failed.append(index)
                    logger.warning("generate_many: parallel request %d failed, retrying serially (%s)", index, exc)
        for index in failed:
            results[index] = _generate_with_optional_temperature(
                generator,
                messages_list[index],
                temperature=temperatures[index],
                response_format=response_format,
            )
        return [result if result is not None else "" for result in results]

    return [
        _generate_with_optional_temperature(
            generator, messages, temperature=temperature, response_format=response_format
        )
        for messages, temperature in zip(messages_list, temperatures)
    ]


# ---------------------------------------------------------------------------
# Small parsing / formatting utilities (ported from finetuning.py)
# ---------------------------------------------------------------------------

_AGENT_DECISION_KEYS = ("action", "tool_calls", "final_answer", "name", "function")
_THOUGHT_FIELD_PATTERN = re.compile(
    r'["\']thought["\']\s*:\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^,}\n][^,}\n]*)',
    re.DOTALL,
)


def strip_code_fence(text: str) -> str:
    value = text.strip()
    fence = re.match(r"^```(?:json|python|text)?\s*(.*?)\s*```$", value, re.DOTALL)
    return fence.group(1).strip() if fence else value


def strip_action_wrappers(text: str) -> str:
    value = strip_code_fence(text).strip()
    for opener, closer in (("<tool_call>", "</tool_call>"), ("<answer>", "</answer>")):
        if opener in value and closer in value:
            start = value.find(opener) + len(opener)
            end = value.rfind(closer)
            if end > start:
                value = value[start:end].strip()
    if value.startswith("<tool_call>"):
        value = value[len("<tool_call>"):].strip()
    if value.endswith("</tool_call>"):
        value = value[: -len("</tool_call>")].strip()
    return value


def _try_parse_json_blob(blob: str) -> Any:
    candidate = blob.strip()
    if not candidate:
        return None
    attempts = [
        candidate,
        candidate.replace("None", "null").replace("True", "true").replace("False", "false"),
    ]
    for attempt in attempts:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(attempt)
        except (ValueError, SyntaxError):
            pass
    try:
        return json.loads(candidate.replace("'", '"'))
    except json.JSONDecodeError:
        return None


def parse_jsonish(text: str) -> Any:
    text = strip_code_fence(text)
    candidates = [text]
    first_brace, last_brace = text.find("{"), text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])
    first_bracket, last_bracket = text.find("["), text.rfind("]")
    if first_bracket != -1 and last_bracket > first_bracket:
        candidates.append(text[first_bracket : last_bracket + 1])
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue
        for attempt in (
            candidate,
            candidate.replace("None", "null").replace("True", "true").replace("False", "false"),
        ):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(attempt)
                except Exception:
                    continue
    raise ValueError(f"Unable to parse JSON from model output: {text[:200]}")


def iter_balanced_json_objects(text: str) -> Iterable[Any]:
    cursor, n = 0, len(text)
    while cursor < n:
        opener = text[cursor]
        if opener not in "{[":
            cursor += 1
            continue
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_string = False
        string_quote = ""
        escape = False
        end = cursor
        balanced = False
        while end < n:
            ch = text[end]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == string_quote:
                    in_string = False
            else:
                if ch in "\"'":
                    in_string = True
                    string_quote = ch
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        balanced = True
                        break
            end += 1
        if not balanced:
            return
        parsed = _try_parse_json_blob(text[cursor : end + 1])
        if parsed is not None:
            yield parsed
        cursor = end + 1


def _is_agent_decision_shaped(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in value for key in _AGENT_DECISION_KEYS)
    if isinstance(value, list) and value:
        return all(
            isinstance(item, dict) and any(key in item for key in _AGENT_DECISION_KEYS)
            for item in value
        )
    return False


def strip_model_thinking_output(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1]
    cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<\|channel>thought\s*.*?<channel\|>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def stringify_tool_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def truncate_for_replay(text: str, char_limit: int) -> str:
    if char_limit <= 0 or len(text) <= char_limit:
        return text
    keep_head = max(char_limit - 200, char_limit // 2)
    return f"{text[:keep_head]}... [truncated {len(text) - keep_head} chars to fit context]"


def preview_text(value: Any, limit: int = 320) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            value = str(value)
    value = " ".join(value.split()).strip()
    return value if len(value) <= limit else value[: limit - 3] + "..."


def preview_tool_calls(tool_calls: List[Dict[str, Any]], limit: int = 4) -> str:
    preview = [f"{c.get('name', '')}({preview_text(c.get('arguments', {}), limit=120)})" for c in tool_calls[:limit]]
    suffix = f" +{len(tool_calls) - limit} more" if len(tool_calls) > limit else ""
    return "; ".join(preview) + suffix


def normalize_tool_call(raw_call: Any) -> Dict[str, Any]:
    if raw_call is None:
        return {"name": "", "arguments": {}}
    if isinstance(raw_call, dict) and "function" in raw_call:
        raw_call = raw_call["function"]
    if not isinstance(raw_call, dict):
        return {"name": str(raw_call), "arguments": {}}
    name = str(raw_call.get("name", "")).strip()
    arguments = raw_call.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = parse_jsonish(arguments)
        except Exception:
            arguments = arguments.strip()
    return {"name": name, "arguments": arguments}


def to_openai_tool_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    formatted = []
    for call in calls:
        normalized = normalize_tool_call(call)
        formatted.append(
            {
                "type": "function",
                "function": {"name": normalized["name"], "arguments": normalized["arguments"]},
            }
        )
    return formatted


def normalize_agent_action_name(action: str) -> str:
    return re.sub(r"[\s_-]+", " ", action).strip().lower()


def parse_thought_payload(raw_text: str) -> Dict[str, Any]:
    cleaned = strip_action_wrappers(raw_text).strip()
    if not cleaned:
        return {"thought": ""}
    parsed: Any = None
    try:
        parsed = parse_jsonish(cleaned)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict) and "thought" in parsed:
        return {"thought": str(parsed.get("thought") or "")}
    for candidate in iter_balanced_json_objects(cleaned):
        if isinstance(candidate, dict) and "thought" in candidate:
            return {"thought": str(candidate.get("thought") or "")}
    match = _THOUGHT_FIELD_PATTERN.search(cleaned)
    if match:
        value = match.group(1).strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            try:
                value = json.loads('"' + value[1:-1].replace('"', '\\"') + '"')
            except json.JSONDecodeError:
                value = value[1:-1]
        return {"thought": value.strip()}
    return {"thought": cleaned}


# ---------------------------------------------------------------------------
# Agent decision parsing (ported from evaluation.py)
# ---------------------------------------------------------------------------


def parse_malformed_final_answer_decision(cleaned: str) -> Optional[Dict[str, Any]]:
    action_match = re.search(r"[\"']action[\"']\s*:\s*[\"']([^\"']+)[\"']", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if not action_match:
        return None
    if normalize_agent_action_name(action_match.group(1)) not in {"final answer", "final", "answer"}:
        return None
    input_match = re.search(r"[\"']action_input[\"']\s*:", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if not input_match:
        return {"final_answer": ""}
    cursor = input_match.end()
    while cursor < len(cleaned) and cleaned[cursor].isspace():
        cursor += 1
    if cursor >= len(cleaned):
        return {"final_answer": ""}
    quote = cleaned[cursor] if cleaned[cursor] in {'"', "'"} else ""
    if quote:
        tail = cleaned[cursor + 1 :].strip()
        closing = re.match(rf"(?s)(.*){re.escape(quote)}\s*}}\s*$", tail)
        if closing:
            answer = closing.group(1)
        else:
            answer = tail
            if answer.endswith("}"):
                answer = answer[:-1].rstrip()
            if answer.endswith(quote):
                answer = answer[:-1]
        return {"final_answer": answer}
    tail = cleaned[cursor:].strip()
    if tail.endswith("}"):
        tail = tail[:-1].rstrip()
    try:
        parsed_tail = parse_jsonish(tail)
    except Exception:
        parsed_tail = tail
    if isinstance(parsed_tail, dict):
        parsed_tail = json.dumps(parsed_tail, ensure_ascii=False)
    return {"final_answer": str(parsed_tail)}


def parse_agent_decision(raw_text: str) -> Dict[str, Any]:
    cleaned = strip_action_wrappers(raw_text)
    parsed: Any = None
    try:
        parsed = parse_jsonish(cleaned)
    except ValueError:
        parsed = None
    if not _is_agent_decision_shaped(parsed):
        first_parseable: Any = None
        for candidate in iter_balanced_json_objects(cleaned):
            if _is_agent_decision_shaped(candidate):
                parsed = candidate
                break
            if first_parseable is None:
                first_parseable = candidate
        else:
            if not _is_agent_decision_shaped(parsed) and first_parseable is not None:
                parsed = first_parseable
    if parsed is None:
        recovered = parse_malformed_final_answer_decision(cleaned)
        if recovered is not None:
            return recovered
        raise ValueError(f"Unable to parse JSON from model output: {raw_text[:200]}")
    if isinstance(parsed, list):
        return {"tool_calls": [normalize_tool_call(item) for item in parsed]}
    if isinstance(parsed, dict):
        if "action" in parsed:
            action = str(parsed.get("action", "")).strip()
            normalized_action = normalize_agent_action_name(action)
            action_input = parsed.get("action_input", {})
            if normalized_action in {"final answer", "final", "answer"}:
                if isinstance(action_input, dict):
                    action_input = json.dumps(action_input, ensure_ascii=False)
                return {"final_answer": str(action_input)}
            if normalized_action == "clarify":
                question = action_input.get("question", "") if isinstance(action_input, dict) else str(action_input)
                return {"clarify": question}
            return {"tool_calls": [normalize_tool_call({"name": action, "arguments": action_input})]}
        if "tool_calls" in parsed:
            tool_calls = parsed["tool_calls"] or []
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]
            return {"tool_calls": [normalize_tool_call(item) for item in tool_calls]}
        if "final_answer" in parsed:
            return {"final_answer": str(parsed["final_answer"])}
        if "name" in parsed or "function" in parsed:
            return {"tool_calls": [normalize_tool_call(parsed)]}
    raise ValueError(f"Unsupported next-action payload: {raw_text[:200]}")


def parse_agent_candidate_decisions(raw_text: str, expected_count: int) -> List[Tuple[Dict[str, Any], str]]:
    cleaned = strip_code_fence(strip_action_wrappers(raw_text)).strip()
    parsed: Any = None
    try:
        parsed = parse_jsonish(cleaned)
    except Exception:
        parsed = None
    raw_candidates: List[Any] = []
    if isinstance(parsed, dict):
        for key in ("candidates", "actions", "action_candidates"):
            value = parsed.get(key)
            if isinstance(value, list):
                raw_candidates = value
                break
        if not raw_candidates and _is_agent_decision_shaped(parsed):
            raw_candidates = [parsed]
    elif isinstance(parsed, list):
        raw_candidates = parsed
    if not raw_candidates:
        for candidate in iter_balanced_json_objects(cleaned):
            if isinstance(candidate, dict):
                for key in ("candidates", "actions", "action_candidates"):
                    value = candidate.get(key)
                    if isinstance(value, list):
                        raw_candidates.extend(value)
                        break
                else:
                    if _is_agent_decision_shaped(candidate):
                        raw_candidates.append(candidate)
            elif isinstance(candidate, list):
                raw_candidates.extend(candidate)
            if len(raw_candidates) >= expected_count:
                break
    if not raw_candidates:
        return [(parse_agent_decision(raw_text), raw_text)]
    decisions: List[Tuple[Dict[str, Any], str]] = []
    for raw_candidate in raw_candidates:
        if len(decisions) >= expected_count:
            break
        candidate_payload = raw_candidate.get("candidate") if isinstance(raw_candidate, dict) else raw_candidate
        if candidate_payload is None:
            candidate_payload = raw_candidate
        candidate_text = (
            candidate_payload if isinstance(candidate_payload, str) else json.dumps(candidate_payload, ensure_ascii=False)
        )
        try:
            decisions.append((parse_agent_decision(candidate_text), candidate_text))
        except Exception:
            if isinstance(raw_candidate, str):
                raise
            raise ValueError(f"Unable to parse candidate action: {candidate_text[:200]}")
    return decisions


def normalize_open_loop_plan_step(step: Any) -> Optional[Dict[str, Any]]:
    """One open-loop plan step -> the decision shape the imagined-rollout machinery already
    uses (``{"tool_calls": [...]}`` or ``{"final_answer": ...}``), reusing
    :func:`parse_agent_decision` rather than re-implementing its tolerance rules. Returns
    ``None`` if the step is unusable, which truncates the plan there (see
    :func:`parse_open_loop_plans`)."""
    if not isinstance(step, dict):
        return None
    try:
        decision = parse_agent_decision(json.dumps(step, ensure_ascii=False))
    except Exception:
        return None
    if "clarify" in decision:
        # An open-loop plan cannot pause for clarification -- nothing will answer it -- so
        # treat it as unusable and let the plan truncate here instead.
        return None
    return decision


def parse_open_loop_plans(raw_text: str, num_plans: int, max_steps: int) -> List[Dict[str, Any]]:
    """Tolerantly parse an open-loop plans payload into ``[{"strategy", "steps"}, ...]``.

    Accepts ``{"plans": [...]}`` (the "m plans in one response" shape), a bare JSON array of
    plans, a bare SINGLE plan ``{"strategy", "steps"}`` (what :func:`build_react_open_loop_plan_
    messages` asks each ``sample_many`` sample for -- ``num_plans=1`` in that case), and plans
    given either as ``{"strategy", "steps"}`` objects or bare step arrays. Steps are truncated at
    ``max_steps``, at the first unparseable/``clarify`` step, or at (and including) the first
    ``final_answer`` step. Unparseable plans are dropped; returns ``[]`` on total failure, which
    the caller (:func:`imagine_trajectories_open_loop`) treats as "fall back to a single
    closed-loop rollout for this cycle" rather than planning nothing.
    """
    cleaned = strip_action_wrappers(raw_text)
    payload: Any = None
    try:
        payload = parse_jsonish(cleaned)
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        # Either the multi-plan envelope {"plans": [...]} or a single plan {"steps": [...]}.
        payload = payload.get("plans") if "plans" in payload else [payload]
    if not isinstance(payload, list):
        for candidate in iter_balanced_json_objects(cleaned):
            if isinstance(candidate, dict) and isinstance(candidate.get("plans"), list):
                payload = candidate["plans"]
                break
            if isinstance(candidate, dict) and isinstance(candidate.get("steps"), list):
                payload = [candidate]
                break
    if not isinstance(payload, list):
        return []

    plans: List[Dict[str, Any]] = []
    for plan in payload[: max(1, int(num_plans))]:
        if isinstance(plan, dict):
            strategy = str(plan.get("strategy") or "").strip()
            raw_steps = plan.get("steps")
        else:
            strategy, raw_steps = "", plan
        if not isinstance(raw_steps, list):
            continue
        steps: List[Dict[str, Any]] = []
        for raw_step in raw_steps[: max(1, int(max_steps))]:
            decision = normalize_open_loop_plan_step(raw_step)
            if decision is None:
                break
            steps.append(decision)
            if "final_answer" in decision:
                break
        if steps:
            plans.append({"strategy": strategy, "steps": steps})
    return plans


# ---------------------------------------------------------------------------
# Compact state helpers
# ---------------------------------------------------------------------------


# Enterprise-state machinery — delegate to the real finetuning.py functions so the
# WM sees the same state schema/rendering it was trained on (NOT a compact shortcut).
blank_state = ft.make_blank_state
make_tool_execution_prediction_state = ft.make_tool_execution_prediction_state
state_current_stage = ft.state_current_stage
state_remaining_stages = ft.state_remaining_stages
state_is_finished = ft.state_is_finished
normalize_last_tool_execution_result = ft.normalize_last_tool_execution_result
append_state_history = ft.append_state_history


def update_state_from_actual_execution(
    previous_state: Dict[str, Any],
    execution_results: List[Dict[str, Any]],
    predicted_feedbacks: Optional[List[Dict[str, Any]]] = None,
    trust_predicted_state: bool = True,
) -> Dict[str, Any]:
    """Evolve the (enterprise/compact) state from a real tool execution.

    Verbatim port of ``evaluation.update_state_from_actual_execution`` so the
    state the WM conditions on matches training/replay exactly.
    """
    predicted_state = None
    if predicted_feedbacks:
        predicted_state = predicted_feedbacks[-1].get("predicted_state")
    state_seed = predicted_state if trust_predicted_state and predicted_state is not None else previous_state
    next_state = ft.sanitize_state_content(state_seed if state_seed is not None else ft.make_blank_state())
    final_result = execution_results[-1] if execution_results else None
    if final_result is not None:
        tool_name = final_result.get("resolved_name") or final_result.get("requested_name")
        success = ft.infer_execution_result_success(final_result)
        content = final_result.get("content", "")
        if ft.is_enterprise_state_payload(next_state):
            state_root = next_state.setdefault("state", {})
            target = state_root.get("diff_from_previous_state")
            if not isinstance(target, dict):
                target = state_root
            outcome = target.setdefault("outcome", {})
            if isinstance(outcome, dict):
                outcome["status"] = "success" if success else "failure"
                outcome["summary"] = str(content)[:1000] if content is not None else ""
                outcome.setdefault("failure_category", "none" if success else "tool_error")
                outcome.setdefault("recoverable", not success)
            history = target.setdefault("history_context", {})
            if isinstance(history, dict):
                events = history.setdefault("last_tool_events", [])
                if isinstance(events, list):
                    events.append({
                        "tool_name": tool_name,
                        "status": "success" if success else "failure",
                        "operation": "execute",
                        "summary": str(content)[:1000] if content is not None else "",
                        "error": "" if success else str(content),
                    })
        elif ft.is_compact_tool_execution_state(next_state):
            next_state["success"] = success
            next_state["last_tool_execution_result"] = 1 if success else 0
            next_state["last_tool_name"] = tool_name
            next_state["error_message"] = "" if success else str(content)
            next_state.setdefault("current_stage", state_current_stage(previous_state))
            next_state.setdefault("remaining_stages", state_remaining_stages(previous_state) or [])
        else:
            state_root = next_state.setdefault("state", {})
            context = state_root.setdefault("context", {})
            context["last_tool_execution_result"] = 1 if success else 0
            context["last_tool_name"] = tool_name
            context["error_message"] = "" if success else str(content)
    return next_state


def _execution_result_from_flow_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """Shape one ``tool_result`` flow event into the execution-result dict the updater reads."""
    tr = event.get("result")
    payload = tr.get("result", tr) if isinstance(tr, dict) else tr
    raw_success = tr.get("success") if isinstance(tr, dict) else None
    is_error = isinstance(payload, dict) and payload.get("isError") is True
    success = (bool(raw_success) and not is_error) if raw_success is not None else (not is_error)
    content = payload if isinstance(payload, str) else stringify_tool_output(payload)
    name = event.get("tool_name")
    return {"requested_name": name, "resolved_name": name, "success": success,
            "content": content, "raw_result": tr}


def _observation_payload(predicted_state: Any, error_message: Optional[str], raw_prediction: Optional[str]) -> Dict[str, Any]:
    return {
        "predicted_state": predicted_state,
        "predicted_error_message": error_message or None,
        "raw_world_model_prediction": raw_prediction or None,
    }


# ---------------------------------------------------------------------------
# ReAct prompt construction (ported from evaluation.py)
# ---------------------------------------------------------------------------

IMAGINED_TRAJECTORY_AGENT_POLICY = """

IMAGINED TRAJECTORY POLICY
You are generating hypothetical next steps for planning only. Your goal is to explore useful possible next actions, not to finish early.
Prefer information-gathering and validation actions before irreversible updates.
Do not produce a final answer unless all required task conditions are explicitly satisfied by observed or imagined evidence.
When a world-model prediction is uncertain, incomplete, generic, or lacks concrete IDs/counts, treat it as non-authoritative and continue exploring.
For each step, identify the unresolved requirement and choose one action that would reduce uncertainty.
Avoid repeating the same tool call with the same arguments.
If a tool call failed, try one alternative formulation or prerequisite lookup before giving up.
Return a tool call unless all requirements are explicitly satisfied, at least two distinct lookup/update strategies have failed, or the next step requires real user clarification.
If unsure, explore with a read-only lookup.
"""


def imagined_trajectory_system_prompt(system_prompt: str) -> str:
    if IMAGINED_TRAJECTORY_AGENT_POLICY.strip() in (system_prompt or ""):
        return system_prompt
    return (system_prompt or "").rstrip() + IMAGINED_TRAJECTORY_AGENT_POLICY


def _get_tool_schema_safe(tool: Any) -> Dict[str, Any]:
    try:
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is None:
            return {}
        if isinstance(args_schema, dict):
            return args_schema
        if hasattr(args_schema, "model_json_schema"):
            return args_schema.model_json_schema()
        if hasattr(args_schema, "schema"):
            return args_schema.schema()
    except Exception:
        return {}
    return {}


def build_react_tool_descriptions(tools: List[Any]) -> str:
    parts: List[str] = []
    for tool in tools:
        schema = _get_tool_schema_safe(tool)
        description = getattr(tool, "description", "") or ""
        chunk = [f"Tool: {getattr(tool, 'name', 'unknown')}", f"Description: {description}"]
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = set(schema.get("required", [])) if isinstance(schema, dict) else set()
        if properties:
            chunk.append("Parameters:")
            for name, info in properties.items():
                param_type = info.get("type", "string") if isinstance(info, dict) else "string"
                marker = "required" if name in required else "optional"
                chunk.append(f"- {name} ({param_type}, {marker})")
        parts.append("\n".join(chunk))
    return "\n\n".join(parts)


def build_react_tool_descriptions_from_gym(available_tools: List[Dict[str, Any]]) -> str:
    """Render gym MCP tool **dicts** into the react prompt's AVAILABLE TOOLS block.

    Verbatim port of ``evaluation.build_react_tool_descriptions_from_gym``: the
    imagined agent uses the *text* ReAct protocol (no native tool channel), so it
    must see the tool catalog in its system prompt to emit valid ``tool_calls``.
    ``mcp_react``'s ``available_tools`` are gym dicts (``name`` / ``description`` /
    ``inputSchema``), not LangChain objects, so :func:`build_react_tool_descriptions`
    (which reads ``.args_schema``) would yield empty entries here.
    """
    sections: List[str] = []
    for tool in available_tools or []:
        if not isinstance(tool, dict):
            # tolerate a LangChain object slipping through
            name = getattr(tool, "name", "unknown")
            description = getattr(tool, "description", "") or ""
            input_schema = _get_tool_schema_safe(tool)
        else:
            name = tool.get("name", "unknown")
            description = tool.get("description", "")
            input_schema = tool.get("inputSchema") or tool.get("input_schema") or {}
        schema_text = json.dumps(input_schema, ensure_ascii=False, indent=2) if input_schema else "{}"
        sections.append(f"- {name}: {description}\n  arguments schema: {schema_text}")
    return "\n".join(sections)


def build_react_system_prompt(tool_descriptions: str) -> str:
    return (
        "You are a ReAct reasoning agent. Follow this EXACT format:\n\n"
        f"AVAILABLE TOOLS:\n{tool_descriptions}\n\n"
        "RESPONSE FORMAT:\n"
        "Step 1: THINK NODE\n"
        "- Perform granular thinking.\n"
        "- Think about what you need to do next based on previous actions.\n"
        '- For THINK NODE return JSON: {"thought": "..."}\n\n'
        "Step 2: ACTION NODE\n"
        "- Output the best action.\n"
        "- For TOOL CALL return JSON:\n"
        '{"action": "<tool_name>", "action_input": {"param1": "value1"}}\n\n'
        "- For FINAL ANSWER return JSON:\n"
        '{"action": "Final Answer", "action_input": "Your complete response"}\n\n'
        "CRITICAL RULES:\n"
        "1. Use exact tool names.\n"
        "2. Use exact parameter names.\n"
        "3. Return JSON only.\n"
        "4. Call tools before answering when information is needed.\n"
        "5. Do not guess.\n"
        "6. Provide the complete final answer when done."
    )


def _trim_replay_messages_to_budget(messages: List[Dict[str, str]], char_budget: int) -> List[Dict[str, str]]:
    if char_budget <= 0:
        return messages
    total = sum(len(msg.get("content", "")) for msg in messages)
    if total <= char_budget:
        return messages
    pinned: List[Dict[str, str]] = []
    rest: List[Dict[str, str]] = []
    seen_first_user = False
    for msg in messages:
        role = msg.get("role")
        if role == "system" and len(pinned) == 0:
            pinned.append(msg)
        elif role == "user" and not seen_first_user:
            pinned.append(msg)
            seen_first_user = True
        else:
            rest.append(msg)
    used = sum(len(msg.get("content", "")) for msg in pinned)
    kept_tail: List[Dict[str, str]] = []
    for msg in reversed(rest):
        size = len(msg.get("content", ""))
        if used + size > char_budget and kept_tail:
            break
        kept_tail.append(msg)
        used += size
    kept_tail.reverse()
    if len(rest) > len(kept_tail):
        kept_tail.insert(
            0,
            {
                "role": "user",
                "content": (
                    f"[CONTEXT TRIMMED] Dropped {len(rest) - len(kept_tail)} earlier "
                    "thought/action/observation turns to fit the context window."
                ),
            },
        )
    return pinned + kept_tail


def filter_messages_for_react_replay(messages: List[Dict[str, Any]], system_prompt: str) -> List[Dict[str, str]]:
    observation_limit = _REPLAY_LIMITS["observation_chars"]
    history_budget = _REPLAY_LIMITS["history_budget_chars"]
    filtered: List[Dict[str, str]] = []
    system_added = False
    for message in messages:
        role = message.get("role")
        if role == "system":
            if not system_added:
                filtered.append({"role": "system", "content": system_prompt})
                system_added = True
            continue
        if role == "user":
            filtered.append({"role": "user", "content": str(message.get("content", ""))})
            continue
        if role == "assistant":
            if message.get("tool_calls"):
                tool_call = message["tool_calls"][0]
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                tool_args = tool_call.get("function", {}).get("arguments", {})
                filtered.append(
                    {
                        "role": "assistant",
                        "content": f"Action: Using {tool_name} with args: {json.dumps(tool_args, ensure_ascii=False)}",
                    }
                )
            elif message.get("content"):
                filtered.append({"role": "assistant", "content": str(message.get("content"))})
            continue
        if role == "tool":
            tool_output = truncate_for_replay(stringify_tool_output(message.get("content")), observation_limit)
            filtered.append(
                {
                    "role": "user",
                    "content": f"Observation: Tool '{message.get('name', 'unknown')}' returned: {tool_output}",
                }
            )
    if not system_added:
        filtered.insert(0, {"role": "system", "content": system_prompt})
    return _trim_replay_messages_to_budget(filtered, history_budget)


def build_conversation_summary_for_replay(messages: List[Dict[str, Any]]) -> str:
    summary_parts: List[str] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("content"):
            summary_parts.append(f"Thought/Action: {message['content']}")
        elif role == "assistant" and message.get("tool_calls"):
            tool_call = message["tool_calls"][0]
            summary_parts.append(
                f"Action: {tool_call.get('function', {}).get('name', 'unknown')}("
                f"{json.dumps(tool_call.get('function', {}).get('arguments', {}), ensure_ascii=False)})"
            )
        elif role == "tool":
            summary_parts.append(
                f"Observation: Tool {message.get('name', 'unknown')} returned "
                f"{stringify_tool_output(message.get('content'))}"
            )
    return "\n".join(summary_parts[-5:])


def build_react_think_messages(messages: List[Dict[str, Any]], current_query: str, system_prompt: str) -> List[Dict[str, str]]:
    filtered_messages = filter_messages_for_react_replay(messages, system_prompt)
    conversation_summary = build_conversation_summary_for_replay(messages)
    think_prompt = (
        f"Current Query: {current_query}\n"
        f"Conversation Summary: {conversation_summary}\n\n"
        "Status: Analyzing task...\n\n"
        "Think: What should I do next based on the current query and progress to complete the task?\n"
        'Respond in JSON format: {"thought": "your reasoning here"}'
    )
    return filtered_messages + [{"role": "user", "content": think_prompt}]


def build_react_action_messages(messages: List[Dict[str, Any]], current_query: str, system_prompt: str) -> List[Dict[str, str]]:
    filtered_messages = filter_messages_for_react_replay(messages, system_prompt)
    conversation_summary = build_conversation_summary_for_replay(messages)
    action_prompt = (
        f"Current Query: {current_query}\n"
        f"Conversation Summary: {conversation_summary}\n\n"
        "Based on your previous thought, select and execute the most appropriate action.\n"
        "Return JSON only in one of the allowed ACTION NODE formats. "
        "Do not return tool arguments by themselves; always include the tool name using "
        '{"action": "<tool_name>", "action_input": {...}} or {"tool_calls": [...]}.'
    )
    return filtered_messages + [{"role": "user", "content": action_prompt}]


def build_react_step_messages(messages: List[Dict[str, Any]], current_query: str, system_prompt: str) -> List[Dict[str, str]]:
    """Single-call variant of think-then-act: one generation returns the thought AND the
    action in one JSON object. Halves the per-imagined-step LLM call count; toggle with
    ``WM_IMAGINED_SINGLE_CALL_STEP=0`` to restore the two-call sequence for A/B comparison."""
    filtered_messages = filter_messages_for_react_replay(messages, system_prompt)
    conversation_summary = build_conversation_summary_for_replay(messages)
    step_prompt = (
        f"Current Query: {current_query}\n"
        f"Conversation Summary: {conversation_summary}\n\n"
        "Think about what to do next based on the current query and progress, then select "
        "and execute the most appropriate action.\n"
        "Return ONE JSON object only, containing BOTH your reasoning and the action:\n"
        '{"thought": "your reasoning here", "action": "<tool_name>", "action_input": {...}}\n'
        "Instead of action/action_input you may use one of the other allowed ACTION NODE "
        'formats alongside "thought": {"thought": "...", "tool_calls": [...]}, '
        '{"thought": "...", "final_answer": "..."}, or {"thought": "...", "clarify": "..."}. '
        "Do not return tool arguments by themselves; always include the tool name."
    )
    return filtered_messages + [{"role": "user", "content": step_prompt}]


def build_react_action_batch_messages(
    messages: List[Dict[str, Any]], current_query: str, system_prompt: str, candidate_action_count: int
) -> List[Dict[str, str]]:
    filtered_messages = filter_messages_for_react_replay(messages, system_prompt)
    conversation_summary = build_conversation_summary_for_replay(messages)
    candidate_action_count = max(1, int(candidate_action_count))
    same_action_name_count = max(1, candidate_action_count // 2)
    different_action_name_count = candidate_action_count - same_action_name_count
    action_prompt = (
        f"Current Query: {current_query}\n"
        f"Conversation Summary: {conversation_summary}\n\n"
        "Based on your previous thought, generate multiple candidate next actions for beam search.\n"
        f"Return exactly {candidate_action_count} candidate actions in one JSON object with this shape: "
        '{"candidates": [{"action": "<tool_name>", "action_input": {...}}]}.\n'
        f"Candidates 1 through {same_action_name_count} should use the same tool name/action family "
        "but materially different valid arguments.\n"
        f"Candidates {same_action_name_count + 1} through {candidate_action_count} should broaden the beam "
        f"with {different_action_name_count} different useful tool name(s)/action families whenever possible.\n"
        "Do not repeat the exact same tool name and arguments. Prefer concrete tool calls over final answers "
        "unless the task is already complete. Return JSON only."
    )
    return filtered_messages + [{"role": "user", "content": action_prompt}]


def build_react_open_loop_plan_messages(
    messages: List[Dict[str, Any]], current_query: str, system_prompt: str, max_steps: int,
) -> List[Dict[str, str]]:
    """Ask for ONE open-loop plan (used ``num_rollouts`` times in parallel via
    :func:`sample_many`, one plan per sample).

    Preferred over :func:`build_react_open_loop_plans_messages`'s "propose N plans in one
    response": N plans in one response is N times the output tokens on ONE serial decode
    stream, whereas N sampled responses decode CONCURRENTLY after a single shared prefill --
    same candidate set, ~N times less wall clock. Diversity comes from sampling temperature
    rather than from an instruction to differ."""
    filtered_messages = filter_messages_for_react_replay(messages, system_prompt)
    conversation_summary = build_conversation_summary_for_replay(messages)
    max_steps = max(1, int(max_steps))
    plan_prompt = (
        f"Current Query: {current_query}\n"
        f"Conversation Summary: {conversation_summary}\n\n"
        f"Plan ahead WITHOUT executing anything: propose ONE plan of up to {max_steps} next "
        "actions to complete the task.\n"
        'Return ONE JSON object only: {"strategy": "<one-line rationale>", '
        '"steps": [<action>, ...]}.\n'
        "Each <action> uses an allowed ACTION NODE format: "
        '{"action": "<tool_name>", "action_input": {...}} or {"tool_calls": [...]}; '
        'the plan may END with {"final_answer": "..."} when the task would be complete.\n'
        "Rules:\n"
        "- For any argument whose value depends on an earlier step's result, use a symbolic "
        'reference string like "$step1.field" instead of guessing a concrete value.\n'
        "- Do not return tool arguments by themselves; always include the tool name."
    )
    return filtered_messages + [{"role": "user", "content": plan_prompt}]


def build_react_open_loop_plans_messages(
    messages: List[Dict[str, Any]], current_query: str, system_prompt: str, num_plans: int, max_steps: int,
) -> List[Dict[str, str]]:
    """One-call prompt for ``num_plans`` COMPLETE open-loop plans of up to ``max_steps``
    ACTION-NODE steps each. Actions never see a predicted state (that's the whole point of
    open-loop), so a value that depends on an earlier, not-yet-known step's result must use a
    symbolic ``"$stepK.field"`` reference instead of a guessed concrete value -- the world model
    sees it verbatim when predicting each step's state, exactly like beam_plan's skeleton
    prompt (:func:`_ewm_beam_plan.build_skeleton_prompt`).

    Superseded by :func:`build_react_open_loop_plan_messages` + ``sample_many`` in
    :func:`imagine_trajectories_open_loop`, for the same reason ``build_skeleton_prompt`` was
    superseded in beam_plan -- kept here (unused by that driver) as a fallback shape for a
    backend with no sampling/parallel-request seam at all."""
    filtered_messages = filter_messages_for_react_replay(messages, system_prompt)
    conversation_summary = build_conversation_summary_for_replay(messages)
    num_plans = max(1, int(num_plans))
    max_steps = max(1, int(max_steps))
    plan_prompt = (
        f"Current Query: {current_query}\n"
        f"Conversation Summary: {conversation_summary}\n\n"
        f"Plan ahead WITHOUT executing anything: propose {num_plans} DISTINCT candidate plans, "
        f"each a sequence of up to {max_steps} next actions to complete the task.\n"
        'Return ONE JSON object only: {"plans": [{"strategy": "<one-line rationale>", '
        f'"steps": [<action>, ...]}}, ...]}} with exactly {num_plans} plans.\n'
        "Each <action> uses an allowed ACTION NODE format: "
        '{"action": "<tool_name>", "action_input": {...}} or {"tool_calls": [...]}; '
        'a plan may END with {"final_answer": "..."} when the task would be complete.\n'
        "Rules:\n"
        "- Make the plans genuinely different strategies, not paraphrases.\n"
        "- For any argument whose value depends on an earlier step's result, use a symbolic "
        'reference string like "$step1.field" instead of guessing a concrete value.\n'
        "- Do not return tool arguments by themselves; always include the tool name."
    )
    return filtered_messages + [{"role": "user", "content": plan_prompt}]


def build_imagined_trajectory_message(imagined_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "role": "user",
        "content": "[IMAGINED_TRAJECTORY_FOR_PLANNING_ONLY]\n" + json.dumps(imagined_steps, ensure_ascii=False, indent=2),
    }


# ---------------------------------------------------------------------------
# World-model feedback (the WM_STATE axis) — prompt + parsing delegate to ft so the
# WM input is byte-identical to training (tool-output failure heuristic too).
# ---------------------------------------------------------------------------


def _wm_target_params(wm_state: str) -> Tuple[str, bool, bool]:
    """WM_STATE -> (target_mode, include_error_message, include_stage) for the WM prompt.

    Mirrors the EWM training targets so the per-mode prompt is byte-identical:
    * binary_error        -> tool_execution_result_binary, +error
    * binary_error_stage  -> tool_execution_result_binary, +error, +stage (JSON)
    * tool_output         -> tool_output (raw text)
    * canonical_nudge     -> canonical_event_with_nudge (schema-based JSON + nudge)
    """
    if wm_state == WM_STATE_TOOL_OUTPUT:
        return ft.WORLD_MODEL_TARGET_TOOL_OUTPUT, False, False
    if wm_state == WM_STATE_CANONICAL_NUDGE:
        return ft.WORLD_MODEL_TARGET_CANONICAL_EVENT_WITH_NUDGE, False, False
    if wm_state == WM_STATE_BINARY_ERROR_STAGE:
        return ft.WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY, True, True
    return ft.WORLD_MODEL_TARGET_TOOL_EXECUTION_RESULT_BINARY, True, False


def _parse_canonical_prediction(
    raw: str, *, include_nudge: bool
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], bool, str]:
    """Parse a ``canonical_event(_with_nudge)`` WM prediction leniently.

    Returns ``(canonical_event, nudge_or_None, predicted_success, error_message)``. Lenient
    by design (mirrors the EWM evaluation path): a malformed/partial payload yields fewer
    fields and a conservative failure rather than raising, so a bad rollout degrades to
    no-injection upstream instead of erroring the task. Success is derived from the
    categorical ``execution_status``; the error message falls back to the categorical
    ``error_signature`` since the canonical schema carries no free text.
    """
    parsed = canonical.parse_jsonish(strip_code_fence(raw).strip())
    if not isinstance(parsed, dict):
        raise ValueError(f"Canonical prediction is not a JSON object: {raw[:200]}")
    if include_nudge:
        event = parsed.get("canonical_event_state")
        if not isinstance(event, dict):
            event = parsed  # tolerate a bare event-state dict without the wrapper
        nudge = parsed.get("nudge") if isinstance(parsed.get("nudge"), dict) else None
    else:
        event = parsed
        nudge = None
    execution_status = event.get("execution_status") if isinstance(event, dict) else None
    predicted_success = execution_status == "success"
    if predicted_success:
        error_message = ""
    else:
        signature = event.get("error_signature") if isinstance(event, dict) else None
        error_message = "" if signature in (None, "none", "") else str(signature)
    return event, nudge, predicted_success, error_message


def build_world_model_prediction_messages(
    system_prompt: str,
    user_prompt: str,
    previous_state: Dict[str, Any],
    planned_calls: List[Dict[str, Any]],
    state_history: Optional[List[Dict[str, Any]]],
    wm_state: str,
    interaction_index: int = 0,
) -> List[Dict[str, str]]:
    """Chat messages for one text-world-model prediction. Split out of what used to be an
    inline block inside :func:`predict_wm_feedback` so the serial path there and the batched
    path in :func:`imagine_trajectories_open_loop` (via :func:`generate_many`) share the exact
    same prompt -- otherwise the two rollout modes could silently diverge on what gets asked."""
    target_mode, include_err, include_stage = _wm_target_params(wm_state)
    example = ft.WorldModelStateExample(
        trajectory_id="0",
        trajectory_index=0,
        interaction_index=interaction_index,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        action={"tool_calls": to_openai_tool_calls(planned_calls)},
        state_history=list(state_history or []),
        input_history=[],
        previous_state=previous_state,
        state=ft.make_blank_state_like(previous_state),
    )
    return ft.build_state_prediction_chat_messages(
        example,
        target_mode=target_mode,
        include_error_message=include_err,
        include_stage=include_stage,
    )


def finish_world_model_text_feedback(
    raw: str,
    *,
    planned_calls: List[Dict[str, Any]],
    wm_state: str,
    interaction_index: int = 0,
) -> List[Dict[str, Any]]:
    """Parse one raw text-world-model completion into the feedback shape. Split out of
    :func:`predict_wm_feedback` alongside :func:`build_world_model_prediction_messages` for the
    same reason -- shared by the serial and batched prediction paths."""
    target_mode, _include_err, _include_stage = _wm_target_params(wm_state)
    raw = raw.split("</think>\n", 1)[-1].strip()
    logger.info(
        "[WORLD_MODEL_RAW] interaction_index=%s | target=%s | tool_calls=%s | raw_prediction=%s",
        interaction_index, target_mode, preview_tool_calls(planned_calls), raw[:300],
    )
    predicted_state = None
    predicted_tool_output = ""
    predicted_current_stage = None
    predicted_remaining_stages = None
    predicted_nudge = None
    predicted_canonical_event = None
    parse_error = None
    try:
        if target_mode == ft.WORLD_MODEL_TARGET_TOOL_OUTPUT:
            predicted_tool_output = raw
            looks_like_failure = ft.tool_output_looks_like_failure(raw)
            predicted_success = not looks_like_failure
            error_message = raw if looks_like_failure else ""
        elif ft.is_canonical_event_with_nudge_target(target_mode) or ft.is_canonical_event_state_target(target_mode):
            predicted_canonical_event, predicted_nudge, predicted_success, error_message = (
                _parse_canonical_prediction(
                    raw, include_nudge=ft.is_canonical_event_with_nudge_target(target_mode)
                )
            )
            # The canonical payload IS the predicted state: it carries forward into the
            # imagined conversation/history and the injected observation (nudge included).
            predicted_state = {"canonical_event_state": predicted_canonical_event}
            if predicted_nudge is not None:
                predicted_state["nudge"] = predicted_nudge
        else:
            parsed = ft.parse_binary_world_model_prediction(raw)
            label = ft.normalize_last_tool_execution_result(parsed.get("success"))
            if label is None:
                raise ValueError(f"Unable to parse tool-result prediction: {raw[:200]}")
            predicted_result = 1 if label == 1 else 0
            predicted_success = predicted_result == 1
            error_message = (parsed.get("error_message") or "").strip()
            predicted_current_stage = parsed.get("current_stage")
            predicted_remaining_stages = parsed.get("remaining_stages")
            predicted_state = ft.make_tool_execution_prediction_state(
                predicted_success=predicted_success,
                predicted_result=predicted_result,
                error_message=error_message,
                current_stage=predicted_current_stage,
                remaining_stages=predicted_remaining_stages,
            )
    except Exception as exc:
        predicted_success = False
        error_message = str(exc)
        parse_error = str(exc)
    return [
        {
            "tool_calls": planned_calls,
            "predicted_success": predicted_success,
            "predicted_state": predicted_state,
            "predicted_tool_output": predicted_tool_output,
            "predicted_error_message": error_message,
            "predicted_current_stage": predicted_current_stage,
            "predicted_remaining_stages": predicted_remaining_stages,
            "predicted_canonical_event": predicted_canonical_event,
            "predicted_nudge": predicted_nudge,
            "raw_prediction": raw,
            "parse_error": parse_error,
        }
    ]


def predict_wm_feedback(
    wm_generator: Any,
    system_prompt: str,
    user_prompt: str,
    previous_state: Dict[str, Any],
    planned_calls: List[Dict[str, Any]],
    state_history: Optional[List[Dict[str, Any]]],
    wm_state: str,
    interaction_index: int = 0,
) -> List[Dict[str, Any]]:
    """Ask the WM to predict the outcome of ``planned_calls``; return one feedback dict (in a list).

    The prompt is built with the **real** ``ft.build_state_prediction_chat_messages``
    so the WM input (enterprise state schema, sanitized, `[{interaction_index, state}]`
    history, per-mode instructions) is identical to what the WM was fine-tuned on.

    A world model may instead expose a structured ``predict_feedback`` seam (e.g.
    the JEPA generator in :mod:`_ewm_jepa`, which conditions on its heads in latent
    space rather than emitting parseable text). When present it is preferred and
    returns the feedback list directly, so this text-prompt path is bypassed. This
    mirrors ``finetuning.predict_world_model_feedback`` in the EWM repo.
    """
    if hasattr(wm_generator, "predict_feedback"):
        return wm_generator.predict_feedback(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            planned_calls=planned_calls,
            previous_state=previous_state,
            state_history=state_history,
            wm_state=wm_state,
            interaction_index=interaction_index,
        )
    messages = build_world_model_prediction_messages(
        system_prompt, user_prompt, previous_state, planned_calls, state_history, wm_state, interaction_index,
    )
    raw = wm_generator.generate_from_messages(messages)
    return finish_world_model_text_feedback(
        raw, planned_calls=planned_calls, wm_state=wm_state, interaction_index=interaction_index,
    )


# ---------------------------------------------------------------------------
# Imagined trajectory: plain rollout and top-k beam search (the ACTION_OPTIMIZER axis)
# ---------------------------------------------------------------------------


def _append_imagined_observation(
    conversation: List[Dict[str, Any]],
    planned_calls: List[Dict[str, Any]],
    feedbacks: List[Dict[str, Any]],
    predicted_state: Any,
) -> None:
    conversation.append({"role": "assistant", "tool_calls": to_openai_tool_calls(planned_calls)})
    last = feedbacks[-1] if feedbacks else {}
    predicted_tool_output = (last.get("predicted_tool_output") or "").strip()
    if predicted_tool_output:
        conversation.append(
            {
                "role": "tool",
                "name": planned_calls[0].get("name", "") if planned_calls else "",
                "content": "[IMAGINED_TOOL_OUTPUT_FROM_WORLD_MODEL]\n" + predicted_tool_output,
            }
        )
        return
    # canonical_nudge: surface the predicted event state AND the epistemic nudge
    # (recommended next action / missing information) as explicit planning guidance.
    nudge = last.get("predicted_nudge")
    canonical_event = last.get("predicted_canonical_event")
    if nudge is not None or canonical_event is not None:
        conversation.append(
            {
                "role": "user",
                "content": "[IMAGINED_CANONICAL_STATE_AND_NUDGE_FROM_WORLD_MODEL]\n"
                + json.dumps(
                    {"canonical_event_state": canonical_event, "nudge": nudge},
                    ensure_ascii=False,
                ),
            }
        )
        return
    conversation.append(
        {
            "role": "user",
            "content": "Imagined observation based on world-model prediction:\n"
            + json.dumps(
                _observation_payload(predicted_state, last.get("predicted_error_message"), last.get("raw_prediction")),
                ensure_ascii=False,
            ),
        }
    )


def imagine_trajectory(
    agent_generator: Any,
    wm_generator: Any,
    conversation: List[Dict[str, Any]],
    previous_state: Dict[str, Any],
    react_system_prompt: str,
    *,
    system_prompt: str,
    user_prompt: str,
    wm_state: str,
    max_imagined_steps: int,
    state_history: Optional[List[Dict[str, Any]]] = None,
    state_history_size: int = 3,
    temperature: float = 0.0,
) -> List[Dict[str, Any]]:
    """Roll a single linear imagined trajectory forward (plain imagine)."""
    imagined_conversation = [dict(m) for m in conversation]
    imagined_system_prompt = imagined_trajectory_system_prompt(react_system_prompt)
    imagined_state = previous_state
    imagined_history = append_state_history(list(state_history or []), imagined_state, max_items=state_history_size)
    imagined_steps: List[Dict[str, Any]] = []
    seen_signatures: Dict[str, int] = {}

    for index in range(max(0, int(max_imagined_steps))):
        raw_thought = _generate_with_optional_temperature(
            agent_generator,
            build_react_think_messages(imagined_conversation, user_prompt, imagined_system_prompt),
            temperature=temperature,
        )
        thought_payload = parse_thought_payload(strip_model_thinking_output(raw_thought))
        imagined_conversation.append({"role": "assistant", "content": json.dumps(thought_payload, ensure_ascii=False)})

        raw_action = _generate_with_optional_temperature(
            agent_generator,
            build_react_action_messages(imagined_conversation, user_prompt, imagined_system_prompt),
            temperature=temperature,
        )
        raw_action = strip_model_thinking_output(raw_action)
        try:
            decision = parse_agent_decision(raw_action)
        except Exception as exc:
            imagined_steps.append({"imagined_step": index + 1, "thought": thought_payload, "parse_error": str(exc)})
            break
        if "final_answer" in decision:
            imagined_steps.append(
                {"imagined_step": index + 1, "thought": thought_payload, "final_answer": decision["final_answer"]}
            )
            break
        if "clarify" in decision:
            imagined_steps.append(
                {"imagined_step": index + 1, "thought": thought_payload, "clarify": decision["clarify"]}
            )
            break
        planned_calls = [normalize_tool_call(c) for c in decision.get("tool_calls", [])]
        if not planned_calls:
            imagined_steps.append({"imagined_step": index + 1, "thought": thought_payload, "error": "empty_tool_calls"})
            break

        signature = json_compact(planned_calls)
        repeated = seen_signatures.get(signature, 0) + 1
        seen_signatures[signature] = repeated

        feedbacks = predict_wm_feedback(
            wm_generator, system_prompt, user_prompt, imagined_state, planned_calls, imagined_history, wm_state
        )
        predicted_state = feedbacks[-1].get("predicted_state")
        imagined_steps.append(
            {
                "imagined_step": index + 1,
                "thought": thought_payload,
                "tool_calls": planned_calls,
                "repeated_tool_call_loop": repeated > 1,
                "predicted_feedback": feedbacks,
                "predicted_state": predicted_state,
                "predicted_error_message": feedbacks[-1].get("predicted_error_message"),
                "predicted_tool_output": feedbacks[-1].get("predicted_tool_output"),
            }
        )
        if repeated > 1:
            break
        _append_imagined_observation(imagined_conversation, planned_calls, feedbacks, predicted_state)
        if predicted_state is not None:
            imagined_state = predicted_state
            imagined_history = append_state_history(imagined_history, imagined_state, max_items=state_history_size)
            if state_is_finished(predicted_state):
                break
    return imagined_steps


def imagine_trajectories_open_loop(
    agent_generator: Any,
    wm_generator: Any,
    conversation: List[Dict[str, Any]],
    previous_state: Dict[str, Any],
    react_system_prompt: str,
    *,
    system_prompt: str,
    user_prompt: str,
    wm_state: str,
    max_imagined_steps: int,
    num_rollouts: int,
    rollout_temperature: float = 0.7,
    state_history: Optional[List[Dict[str, Any]]] = None,
    state_history_size: int = 3,
    llm_batch_parallelism: int = 8,
) -> Optional[List[List[Dict[str, Any]]]]:
    """beam_plan-style open-loop rollouts for the plain (non-beam) imagined trajectory.

    ONE agent call proposes all ``num_rollouts`` plans up front (symbolic ``"$stepK.field"``
    references for values not yet known); the agent never sees a predicted state, so nothing
    serializes on the agent side. The world model then rolls each plan's state chain forward --
    state *k* feeds the step *k+1* prompt WITHIN a plan -- batched ACROSS plans per horizon step
    via :func:`generate_many` (or, for a local/cheap world model exposing ``predict_feedback``,
    looped with no batching -- see :func:`predict_wm_feedback`'s seam; batching text prompts is
    what pays off, batching a local in-process forward usually isn't the bottleneck).

    Returns one step list per plan, in the *exact* record shape :func:`imagine_trajectory`
    produces (plus an ``"open_loop": True`` marker), or ``None`` when no plan could be parsed at
    all -- the caller (:func:`optimize_imagined_trajectory`) falls back to a single closed-loop
    rollout for this cycle rather than planning nothing.
    """
    imagined_system_prompt = imagined_trajectory_system_prompt(react_system_prompt)
    num_rollouts = max(1, int(num_rollouts))
    max_imagined_steps = max(1, int(max_imagined_steps))

    # num_rollouts samples of a ONE-plan prompt rather than one response listing num_rollouts
    # plans -- same candidate set, but the samples decode CONCURRENTLY after one shared prefill
    # (n=k on a backend exposing generate_samples) instead of one serial decode stream emitting
    # every plan.
    sample_temperature = rollout_temperature if num_rollouts > 1 else 0.0
    raw_samples, _agent_requests = sample_many(
        agent_generator,
        build_react_open_loop_plan_messages(
            conversation, current_query=user_prompt, system_prompt=imagined_system_prompt,
            max_steps=max_imagined_steps,
        ),
        temperature=sample_temperature,
        num_samples=num_rollouts,
    )
    plans: List[Dict[str, Any]] = []
    seen_plan_signatures: set = set()
    for raw in raw_samples:
        parsed = parse_open_loop_plans(strip_model_thinking_output(raw), 1, max_imagined_steps)
        if not parsed:
            continue
        plan = parsed[0]
        # Sampling can repeat itself; a duplicate plan would only re-spend world-model compute
        # on an answer already in the candidate set.
        signature = json_compact(plan["steps"])
        if signature in seen_plan_signatures:
            continue
        seen_plan_signatures.add(signature)
        plans.append(plan)
    if not plans:
        return None

    rollouts: List[Dict[str, Any]] = []
    for plan_index, plan in enumerate(plans):
        rollouts.append({
            "rollout_index": plan_index,
            "plan": plan,
            "state": previous_state,
            "state_history": append_state_history(list(state_history or []), previous_state, max_items=state_history_size),
            "steps": [],
            "seen_signatures": {},
            "done": False,
        })

    for imagined_index in range(max_imagined_steps):
        pending: List[Dict[str, Any]] = []
        for rollout in rollouts:
            if rollout["done"] or imagined_index >= len(rollout["plan"]["steps"]):
                rollout["done"] = True
                continue
            decision = rollout["plan"]["steps"][imagined_index]
            # The plan-level strategy doubles as the first step's thought so records keep the
            # usual shape (imagine_trajectory always has a real per-step "thought").
            thought_payload = {"thought": rollout["plan"]["strategy"]} if imagined_index == 0 else {"thought": ""}

            if "final_answer" in decision:
                rollout["steps"].append({
                    "imagined_step": imagined_index + 1, "thought": thought_payload,
                    "final_answer": decision["final_answer"], "open_loop": True,
                })
                rollout["done"] = True
                continue

            planned_calls = [normalize_tool_call(c) for c in decision.get("tool_calls", [])]
            if not planned_calls:
                rollout["steps"].append({
                    "imagined_step": imagined_index + 1, "thought": thought_payload,
                    "error": "empty_tool_calls", "open_loop": True,
                })
                rollout["done"] = True
                continue

            signature = json_compact(planned_calls)
            repeated = rollout["seen_signatures"].get(signature, 0) + 1
            rollout["seen_signatures"][signature] = repeated
            pending.append({
                "rollout": rollout, "thought": thought_payload,
                "planned_calls": planned_calls, "repeated": repeated,
            })

        if not pending:
            break

        # One batched world-model pass for this horizon step across every still-pending plan.
        if hasattr(wm_generator, "predict_feedback"):
            for item in pending:
                rollout = item["rollout"]
                item["feedbacks"] = predict_wm_feedback(
                    wm_generator, system_prompt, user_prompt, rollout["state"], item["planned_calls"],
                    rollout["state_history"], wm_state,
                )
        else:
            predictions = generate_many(
                wm_generator,
                [
                    build_world_model_prediction_messages(
                        system_prompt, user_prompt, item["rollout"]["state"], item["planned_calls"],
                        item["rollout"]["state_history"], wm_state,
                    )
                    for item in pending
                ],
                [0.0] * len(pending),
                max_workers=llm_batch_parallelism,
            )
            for item, raw in zip(pending, predictions):
                item["feedbacks"] = finish_world_model_text_feedback(
                    raw, planned_calls=item["planned_calls"], wm_state=wm_state,
                )

        for item in pending:
            rollout = item["rollout"]
            planned_calls = item["planned_calls"]
            repeated = item["repeated"]
            feedbacks: List[Dict[str, Any]] = item.get("feedbacks") or []
            predicted_state = feedbacks[-1].get("predicted_state") if feedbacks else None
            rollout["steps"].append({
                "imagined_step": imagined_index + 1,
                "thought": item["thought"],
                "tool_calls": planned_calls,
                "repeated_tool_call_loop": repeated > 1,
                "predicted_feedback": feedbacks,
                "predicted_state": predicted_state,
                "predicted_error_message": feedbacks[-1].get("predicted_error_message") if feedbacks else None,
                "predicted_tool_output": feedbacks[-1].get("predicted_tool_output") if feedbacks else None,
                "open_loop": True,
            })
            if repeated > 1:
                rollout["done"] = True
                continue
            if predicted_state is not None:
                rollout["state"] = predicted_state
                rollout["state_history"] = append_state_history(
                    rollout["state_history"], predicted_state, max_items=state_history_size
                )
                if state_is_finished(predicted_state):
                    rollout["done"] = True

    return [rollout["steps"] for rollout in rollouts]


def imagine_trajectories_lockstep(
    agent_generator: Any,
    wm_generator: Any,
    conversation: List[Dict[str, Any]],
    previous_state: Dict[str, Any],
    react_system_prompt: str,
    *,
    system_prompt: str,
    user_prompt: str,
    wm_state: str,
    max_imagined_steps: int,
    rollout_temperatures: List[float],
    state_history: Optional[List[Dict[str, Any]]] = None,
    state_history_size: int = 3,
    single_call_step: bool = True,
    llm_batch_parallelism: int = 8,
) -> List[List[Dict[str, Any]]]:
    """Advance N independent CLOSED-loop imagined rollouts in lockstep, batching every LLM call.

    Serial rollouts (the ``for rollout_index in range(rollouts): imagine_trajectory(...)`` loop
    in :func:`optimize_imagined_trajectory`) cost ``N * steps * (think + act + world_model)``
    sequential generations. Here all alive rollouts issue each phase's generations together
    through :func:`generate_many` (one padded batched ``generate`` on HF backends; parallel
    requests on API/vLLM backends), so wall-clock is ``~steps * phases`` batched calls
    regardless of N. Per-rollout semantics match the serial :func:`imagine_trajectory` exactly:
    same prompts (in two-call mode), same temperatures, same termination rules -- this is a
    pure batching optimization, not a different rollout policy.

    With ``single_call_step`` the two think/act phases collapse into one combined generation
    per step (:func:`build_react_step_messages`), halving the agent call count again.
    """
    imagined_react_system_prompt = imagined_trajectory_system_prompt(react_system_prompt)
    root_state = ft.sanitize_state_content(previous_state)
    rollouts: List[Dict[str, Any]] = []
    for offset, temperature in enumerate(rollout_temperatures):
        rollouts.append({
            "rollout_index": offset,
            "temperature": float(temperature),
            "conversation": [dict(message) for message in conversation],
            "state": root_state,
            "state_history": append_state_history(list(state_history or []), root_state, max_items=state_history_size),
            "steps": [],
            "seen_signatures": {},
            "done": False,
        })

    for imagined_index in range(max(0, int(max_imagined_steps))):
        alive = [rollout for rollout in rollouts if not rollout["done"]]
        if not alive:
            break

        # --- Phase A: agent decision (batched across alive rollouts) -----------------
        if single_call_step:
            raw_steps = generate_many(
                agent_generator,
                [
                    build_react_step_messages(
                        rollout["conversation"], current_query=user_prompt, system_prompt=imagined_react_system_prompt,
                    )
                    for rollout in alive
                ],
                [rollout["temperature"] for rollout in alive],
                max_workers=llm_batch_parallelism,
            )
            raw_steps = [strip_model_thinking_output(raw) for raw in raw_steps]
            raw_actions = raw_steps
            thought_payloads = [parse_thought_payload(raw) for raw in raw_steps]
            for rollout, thought_payload in zip(alive, thought_payloads):
                rollout["conversation"].append(
                    {"role": "assistant", "content": json.dumps(thought_payload, ensure_ascii=False)}
                )
        else:
            raw_thoughts = generate_many(
                agent_generator,
                [
                    build_react_think_messages(rollout["conversation"], user_prompt, imagined_react_system_prompt)
                    for rollout in alive
                ],
                [rollout["temperature"] for rollout in alive],
                max_workers=llm_batch_parallelism,
            )
            thought_payloads = [
                parse_thought_payload(strip_model_thinking_output(raw)) for raw in raw_thoughts
            ]
            for rollout, thought_payload in zip(alive, thought_payloads):
                rollout["conversation"].append(
                    {"role": "assistant", "content": json.dumps(thought_payload, ensure_ascii=False)}
                )
            raw_actions = [
                strip_model_thinking_output(raw)
                for raw in generate_many(
                    agent_generator,
                    [
                        build_react_action_messages(rollout["conversation"], user_prompt, imagined_react_system_prompt)
                        for rollout in alive
                    ],
                    [rollout["temperature"] for rollout in alive],
                    max_workers=llm_batch_parallelism,
                )
            ]

        # --- Phase B: interpret decisions; collect world-model requests --------------
        pending_predictions: List[Dict[str, Any]] = []
        for rollout, thought_payload, raw_action in zip(alive, thought_payloads, raw_actions):
            try:
                decision = parse_agent_decision(raw_action)
            except Exception as exc:
                rollout["steps"].append(
                    {
                        "imagined_step": imagined_index + 1, "thought": thought_payload,
                        "raw_action": raw_action, "parse_error": str(exc),
                    }
                )
                rollout["done"] = True
                continue

            if "final_answer" in decision:
                rollout["steps"].append(
                    {
                        "imagined_step": imagined_index + 1, "thought": thought_payload,
                        "final_answer": decision["final_answer"],
                    }
                )
                rollout["done"] = True
                continue
            if "clarify" in decision:
                rollout["steps"].append(
                    {
                        "imagined_step": imagined_index + 1, "thought": thought_payload,
                        "clarify": decision["clarify"],
                    }
                )
                rollout["done"] = True
                continue

            planned_calls = [normalize_tool_call(call) for call in decision.get("tool_calls", [])]
            if not planned_calls:
                rollout["steps"].append(
                    {
                        "imagined_step": imagined_index + 1, "thought": thought_payload,
                        "raw_action": raw_action, "error": "empty_tool_calls",
                    }
                )
                rollout["done"] = True
                continue

            signature = json_compact(planned_calls)
            repeated_tool_call_count = rollout["seen_signatures"].get(signature, 0) + 1
            rollout["seen_signatures"][signature] = repeated_tool_call_count
            pending_predictions.append(
                {
                    "rollout": rollout, "thought": thought_payload,
                    "planned_calls": planned_calls, "repeated_tool_call_count": repeated_tool_call_count,
                }
            )

        # --- Phase C: world-model predictions (batched across pending rollouts) ------
        if pending_predictions:
            if hasattr(wm_generator, "predict_feedback"):
                # JEPA-style local generators expose their own prediction interface; keep the
                # existing per-rollout call (already a cheap local forward, not worth batching).
                for pending in pending_predictions:
                    rollout = pending["rollout"]
                    pending["feedbacks"] = predict_wm_feedback(
                        wm_generator, system_prompt, user_prompt, rollout["state"], pending["planned_calls"],
                        rollout["state_history"], wm_state,
                    )
            else:
                predictions = generate_many(
                    wm_generator,
                    [
                        build_world_model_prediction_messages(
                            system_prompt, user_prompt, pending["rollout"]["state"], pending["planned_calls"],
                            pending["rollout"]["state_history"], wm_state,
                        )
                        for pending in pending_predictions
                    ],
                    [0.0] * len(pending_predictions),
                    max_workers=llm_batch_parallelism,
                )
                for pending, prediction in zip(pending_predictions, predictions):
                    pending["feedbacks"] = finish_world_model_text_feedback(
                        prediction, planned_calls=pending["planned_calls"], wm_state=wm_state,
                    )

        # --- Phase D: fold predictions back into each rollout -------------------------
        for pending in pending_predictions:
            rollout = pending["rollout"]
            planned_calls = pending["planned_calls"]
            repeated_tool_call_count = pending["repeated_tool_call_count"]
            feedbacks: List[Dict[str, Any]] = pending.get("feedbacks") or []
            predicted_state = feedbacks[-1].get("predicted_state") if feedbacks else None
            rollout["steps"].append(
                {
                    "imagined_step": imagined_index + 1,
                    "thought": pending["thought"],
                    "tool_calls": planned_calls,
                    "repeated_tool_call_loop": repeated_tool_call_count > 1,
                    "predicted_feedback": feedbacks,
                    "predicted_state": predicted_state,
                    "predicted_error_message": feedbacks[-1].get("predicted_error_message") if feedbacks else None,
                    "predicted_tool_output": feedbacks[-1].get("predicted_tool_output") if feedbacks else None,
                }
            )
            if repeated_tool_call_count > 1:
                rollout["done"] = True
                continue
            _append_imagined_observation(rollout["conversation"], planned_calls, feedbacks, predicted_state)
            if predicted_state is not None:
                rollout["state"] = predicted_state
                rollout["state_history"] = append_state_history(
                    rollout["state_history"], predicted_state, max_items=state_history_size
                )
                if state_is_finished(predicted_state):
                    rollout["done"] = True

    return [rollout["steps"] for rollout in rollouts]


def _stage_value(state: Any) -> Any:
    return state_current_stage(state) if state is not None else None


def score_step(
    *,
    previous_state: Any,
    predicted_state: Any,
    feedbacks: List[Dict[str, Any]],
    repeated_tool_call_count: int,
    final_answer: bool = False,
) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    if final_answer:
        if state_is_finished(previous_state):
            return 1.0, ["final_answer_after_finished_stage=1"]
        return 0.0, ["final_answer_but_stage_remains=0"]
    if repeated_tool_call_count > 1:
        return 0.0, ["same_tool_names_and_arguments=0"]
    score = 0.0
    if feedbacks and all(bool(fb.get("predicted_success")) for fb in feedbacks):
        score += 1.0
        reasons.append("success=1")
    else:
        reasons.append("failure=0")
    previous_stage = _stage_value(previous_state)
    next_stage = _stage_value(predicted_state)
    if next_stage is None and feedbacks:
        next_stage = feedbacks[-1].get("predicted_current_stage")
    if next_stage is not None and next_stage != previous_stage:
        score += 1.0
        reasons.append("current_stage_change=1")
    else:
        reasons.append("current_stage_unchanged=0")
    return score, reasons


def imagine_trajectory_topk_search(
    agent_generator: Any,
    wm_generator: Any,
    conversation: List[Dict[str, Any]],
    previous_state: Dict[str, Any],
    react_system_prompt: str,
    *,
    system_prompt: str,
    user_prompt: str,
    wm_state: str,
    max_imagined_steps: int,
    candidate_action_count: int = 3,
    top_k: int = 3,
    temperature: float = 0.7,
    state_history: Optional[List[Dict[str, Any]]] = None,
    state_history_size: int = 3,
) -> List[Dict[str, Any]]:
    """Beam search over candidate actions; return the best beam's imagined_steps."""
    candidate_action_count = max(1, int(candidate_action_count))
    top_k = max(1, int(top_k))
    imagined_system_prompt = imagined_trajectory_system_prompt(react_system_prompt)
    beams: List[Dict[str, Any]] = [
        {
            "branch_id": "0",
            "conversation": [dict(m) for m in conversation],
            "state": previous_state,
            "state_history": append_state_history(list(state_history or []), previous_state, max_items=state_history_size),
            "imagined_steps": [],
            "seen_signatures": {},
            "score": 0.0,
            "terminal": False,
        }
    ]

    for index in range(max(0, int(max_imagined_steps))):
        expanded: List[Dict[str, Any]] = []
        for branch in beams:
            if branch.get("terminal"):
                expanded.append(branch)
                continue
            branch_conversation = [dict(m) for m in branch["conversation"]]
            raw_thought = _generate_with_optional_temperature(
                agent_generator,
                build_react_think_messages(branch_conversation, user_prompt, imagined_system_prompt),
                temperature=0.0,
            )
            thought_payload = parse_thought_payload(strip_model_thinking_output(raw_thought))
            thought_conversation = branch_conversation + [
                {"role": "assistant", "content": json.dumps(thought_payload, ensure_ascii=False)}
            ]
            raw_batch = _generate_with_optional_temperature(
                agent_generator,
                build_react_action_batch_messages(thought_conversation, user_prompt, imagined_system_prompt, candidate_action_count),
                temperature=temperature if candidate_action_count > 1 else 0.0,
            )
            raw_batch = strip_model_thinking_output(raw_batch)
            try:
                candidate_decisions = parse_agent_candidate_decisions(raw_batch, expected_count=candidate_action_count)
                batch_parse_error = None
            except Exception as exc:
                candidate_decisions = []
                batch_parse_error = str(exc)

            seen_candidate_signatures: set = set()
            for candidate_index in range(candidate_action_count):
                if candidate_index < len(candidate_decisions):
                    decision, raw_action = candidate_decisions[candidate_index]
                    parse_error = None
                else:
                    decision = None
                    parse_error = batch_parse_error or "missing candidate"
                child_id = f"{branch['branch_id']}.{candidate_index}"
                base_step = {"imagined_step": index + 1, "topk_branch_id": child_id, "thought": thought_payload}

                if decision is None:
                    step = {**base_step, "parse_error": parse_error, "topk_score_reasons": [f"parse_error=0:{parse_error}"]}
                    expanded.append({**branch, "branch_id": child_id, "imagined_steps": branch["imagined_steps"] + [step], "terminal": True})
                    continue
                if "final_answer" in decision:
                    step_score, reasons = score_step(previous_state=branch["state"], predicted_state=branch["state"], feedbacks=[], repeated_tool_call_count=0, final_answer=True)
                    step = {**base_step, "final_answer": decision["final_answer"], "topk_score_reasons": reasons}
                    expanded.append({**branch, "branch_id": child_id, "imagined_steps": branch["imagined_steps"] + [step], "score": branch["score"] + step_score, "terminal": True})
                    continue
                if "clarify" in decision:
                    step = {**base_step, "clarify": decision["clarify"], "topk_score_reasons": ["clarify=0"]}
                    expanded.append({**branch, "branch_id": child_id, "imagined_steps": branch["imagined_steps"] + [step], "terminal": True})
                    continue
                planned_calls = [normalize_tool_call(c) for c in decision.get("tool_calls", [])]
                if not planned_calls:
                    step = {**base_step, "error": "empty_tool_calls", "topk_score_reasons": ["empty_tool_calls=0"]}
                    expanded.append({**branch, "branch_id": child_id, "imagined_steps": branch["imagined_steps"] + [step], "terminal": True})
                    continue

                signature = json_compact(planned_calls)
                local_duplicate = signature in seen_candidate_signatures
                seen_candidate_signatures.add(signature)
                seen_signatures = dict(branch["seen_signatures"])
                repeated = seen_signatures.get(signature, 0) + 1
                seen_signatures[signature] = repeated
                if local_duplicate:
                    repeated = max(repeated, 2)
                    step_score, reasons = score_step(previous_state=branch["state"], predicted_state=branch["state"], feedbacks=[], repeated_tool_call_count=repeated)
                    step = {**base_step, "tool_calls": planned_calls, "repeated_tool_call_loop": True, "predicted_feedback": [], "topk_score_reasons": reasons}
                    expanded.append({**branch, "branch_id": child_id, "imagined_steps": branch["imagined_steps"] + [step], "score": branch["score"] + step_score, "terminal": True})
                    continue

                feedbacks = predict_wm_feedback(
                    wm_generator, system_prompt, user_prompt, branch["state"], planned_calls, branch["state_history"], wm_state
                )
                predicted_state = feedbacks[-1].get("predicted_state") if feedbacks else branch["state"]
                step_score, reasons = score_step(previous_state=branch["state"], predicted_state=predicted_state, feedbacks=feedbacks, repeated_tool_call_count=repeated)
                next_conversation = [dict(m) for m in thought_conversation]
                _append_imagined_observation(next_conversation, planned_calls, feedbacks, predicted_state)
                next_history = branch["state_history"]
                if predicted_state is not None:
                    next_history = append_state_history(next_history, predicted_state, max_items=state_history_size)
                terminal = repeated > 1 or state_is_finished(predicted_state)
                step = {
                    **base_step,
                    "tool_calls": planned_calls,
                    "repeated_tool_call_loop": repeated > 1,
                    "predicted_feedback": feedbacks,
                    "predicted_state": predicted_state,
                    "predicted_error_message": feedbacks[-1].get("predicted_error_message") if feedbacks else None,
                    "predicted_tool_output": feedbacks[-1].get("predicted_tool_output") if feedbacks else None,
                    "topk_score_reasons": reasons,
                }
                expanded.append(
                    {
                        "branch_id": child_id,
                        "conversation": next_conversation,
                        "state": predicted_state if predicted_state is not None else branch["state"],
                        "state_history": next_history,
                        "imagined_steps": branch["imagined_steps"] + [step],
                        "seen_signatures": seen_signatures,
                        "score": branch["score"] + step_score,
                        "terminal": terminal,
                    }
                )

        if not expanded:
            break
        expanded.sort(key=lambda item: (float(item.get("score", 0.0)), len(item.get("imagined_steps", [])), 0 if item.get("terminal") else 1), reverse=True)
        beams = expanded[:top_k]
        if all(b.get("terminal") for b in beams):
            break

    if not beams:
        return []
    return beams[0]["imagined_steps"]


IMAGINED_TRAJECTORY_JUDGE_SYSTEM_PROMPT = (
    "You are selecting the best imagined agent trajectory before any real tool execution.\n\n"
    "Each candidate is a sequence of (action -> predicted resulting state) steps produced by a "
    "world model. Choose the candidate most likely to succeed in the real environment. Prefer "
    "candidates that:\n"
    "- satisfy the user's requirements with the fewest missing dependencies\n"
    "- use tools and arguments coherently\n"
    "- avoid parse errors, empty actions, or repeated tool-call loops\n"
    "- make concrete progress toward a valid final answer\n\n"
    "Respond ONLY with JSON:\n"
    "{\n"
    '  "selected_index": int,\n'
    '  "scores": [{"index": int, "score": float, "reason": "brief explanation"}],\n'
    '  "comments": "brief explanation"\n'
    "}"
)


def build_imagined_trajectory_selection_judge_prompt() -> str:
    return IMAGINED_TRAJECTORY_JUDGE_SYSTEM_PROMPT


def score_imagined_trajectory_candidate(candidate_steps: List[Dict[str, Any]]) -> float:
    """Heuristic fallback score for one imagined rollout (used only when the LLM judge fails).
    Ported from ``finetuning.score_imagined_trajectory_candidate``."""
    score = 0.0
    for step in candidate_steps or []:
        if step.get("final_answer"):
            score += 3.0
        if step.get("clarify"):
            score -= 1.0
        if step.get("parse_error"):
            score -= 2.0
        if step.get("error") == "empty_tool_calls":
            score -= 2.0
        if step.get("repeated_tool_call_loop"):
            score -= 1.5
        feedbacks = step.get("predicted_feedback") or []
        if feedbacks:
            score += sum(0.5 if fb.get("predicted_success") else -0.5 for fb in feedbacks)
        if step.get("tool_calls"):
            score += 0.2
    return score


def _strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    match = re.search(r"```(?:json)?\s*(.+?)```", t, re.DOTALL)
    return match.group(1).strip() if match else t


def select_imagined_trajectory_with_llm_judge(
    agent_generator: Any,
    user_prompt: str,
    conversation: List[Dict[str, Any]],
    candidate_rollouts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """LLM reranker over independent imagined rollouts (port of
    ``finetuning.select_imagined_trajectory_with_llm_judge``). No world-model scoring is used —
    the judge reads each candidate's (action -> predicted state) pairs and picks the best. Falls
    back to the heuristic score on any judge/parse failure."""
    if not candidate_rollouts:
        return {"selected_index": 0, "scores": [], "comments": "No candidates provided", "fallback_used": True}
    messages = [
        {"role": "system", "content": build_imagined_trajectory_selection_judge_prompt()},
        {
            "role": "user",
            "content": (
                f"User task:\n{user_prompt}\n\n"
                f"Conversation so far:\n{_render_recent_conversation(conversation)}\n\n"
                "Candidate imagined trajectories:\n"
                + json.dumps(candidate_rollouts, ensure_ascii=False, indent=2)
            ),
        },
    ]
    try:
        raw = _generate_with_optional_temperature(agent_generator, messages, temperature=0.0)
        raw = str(raw).split("</think>", 1)[-1].strip()
        parsed = parse_jsonish(_strip_code_fence(raw))
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object from the imagined-trajectory judge.")
        selected_index = int(parsed.get("selected_index", 0))
        if not 0 <= selected_index < len(candidate_rollouts):
            raise ValueError(f"selected imagined rollout index out of range: {selected_index}")
        return {
            "selected_index": selected_index,
            "scores": parsed.get("scores") or [],
            "comments": str(parsed.get("comments", "")),
            "fallback_used": False,
        }
    except Exception as exc:
        scored = [
            {"index": i, "score": score_imagined_trajectory_candidate(r.get("imagined_steps") or []),
             "reason": "heuristic fallback"}
            for i, r in enumerate(candidate_rollouts)
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return {
            "selected_index": scored[0]["index"] if scored else 0,
            "scores": scored,
            "comments": f"LLM judge failed, used heuristic fallback: {exc}",
            "fallback_used": True,
        }


def optimize_imagined_trajectory(
    action_optimizer: str,
    agent_generator: Any,
    wm_generator: Any,
    conversation: List[Dict[str, Any]],
    previous_state: Dict[str, Any],
    react_system_prompt: str,
    *,
    system_prompt: str,
    user_prompt: str,
    wm_state: str,
    max_imagined_steps: int,
    candidate_action_count: int = 3,
    top_k: int = 3,
    temperature: float = 0.7,
    state_history: Optional[List[Dict[str, Any]]] = None,
    state_history_size: int = 3,
    num_rollouts: int = 1,
    selection_strategy: str = "first",
    rollout_mode: str = "closed_loop",
    llm_batch_parallelism: int = 8,
    parallel_rollouts: bool = True,
    single_call_step: bool = True,
) -> List[Dict[str, Any]]:
    """Dispatch the imagined-trajectory optimizer:

    * ``ACTION_OPTIMIZER=topk_search`` → per-step beam search (needs a scoring head); open-loop
      does not apply here (it replaces the plain rollout's per-step agent+WM calls, not the beam
      search's own scoring loop);
    * ``rollout_mode="open_loop"`` → ONE agent call proposes all ``num_rollouts`` plans up front
      (see :func:`imagine_trajectories_open_loop`); the resulting step lists then go through the
      *same* selection logic below (``llm_judge`` or ``first``) as closed-loop rollouts would.
      Falls through to the closed-loop path if no plan could be parsed at all;
    * ``selection_strategy=llm_judge`` (or ``num_rollouts>1``) → generate ``num_rollouts``
      independent rollouts (first deterministic, rest at ``temperature``) and pick one with the
      LLM judge over their (action -> predicted state) pairs. This is the **decoder-friendly**
      path (no world-model head scoring), used for the seq2seq JEPA on terminal tasks.
      ``parallel_rollouts=True`` (default) advances all of them in LOCKSTEP, batching every
      step's agent/world-model calls across rollouts via :func:`imagine_trajectories_lockstep`
      instead of running each rollout fully before starting the next; ``single_call_step=True``
      (default) additionally collapses each step's think+act into one combined generation
      (only read by the lockstep driver -- the plain single-rollout path below is unaffected by
      either toggle, matching the reference: they're wins for the *multi-rollout* path only);
    * otherwise → a single deterministic rollout.
    """
    if (action_optimizer or "").strip().lower() == "topk_search":
        return imagine_trajectory_topk_search(
            agent_generator,
            wm_generator,
            conversation,
            previous_state,
            react_system_prompt,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            wm_state=wm_state,
            max_imagined_steps=max_imagined_steps,
            candidate_action_count=candidate_action_count,
            top_k=top_k,
            temperature=temperature,
            state_history=state_history,
            state_history_size=state_history_size,
        )

    selection = (selection_strategy or "first").strip().lower()
    rollouts = max(1, int(num_rollouts or 1))

    if (rollout_mode or "closed_loop").strip().lower() == "open_loop":
        open_loop_plans = imagine_trajectories_open_loop(
            agent_generator, wm_generator, conversation, previous_state, react_system_prompt,
            system_prompt=system_prompt, user_prompt=user_prompt, wm_state=wm_state,
            max_imagined_steps=max_imagined_steps, num_rollouts=rollouts,
            rollout_temperature=temperature, state_history=state_history,
            state_history_size=state_history_size, llm_batch_parallelism=llm_batch_parallelism,
        )
        if open_loop_plans is not None:
            candidate_rollouts = [
                {"rollout_index": i, "imagined_steps": steps} for i, steps in enumerate(open_loop_plans)
            ]
            if selection == "llm_judge" and len(candidate_rollouts) > 1:
                decision = select_imagined_trajectory_with_llm_judge(
                    agent_generator, user_prompt, conversation, candidate_rollouts
                )
                chosen = int(decision.get("selected_index", 0))
            else:
                chosen = 0  # "first"
            chosen = chosen if 0 <= chosen < len(candidate_rollouts) else 0
            return candidate_rollouts[chosen].get("imagined_steps") or []
        # Total parse failure -- unlike beam_plan (which has no closed-loop rollout sitting
        # right here to fall back to), a real closed-loop rollout IS right below, so fall
        # through to it for this cycle instead of returning nothing.

    if selection == "llm_judge" or rollouts > 1:
        rollout_temperature_plan = [0.0 if index == 0 else temperature for index in range(rollouts)]
        if parallel_rollouts:
            # All rollouts advance in lockstep; each step's agent and world-model generations
            # are batched across the alive rollouts (see imagine_trajectories_lockstep).
            all_imagined_steps = imagine_trajectories_lockstep(
                agent_generator, wm_generator, conversation, previous_state, react_system_prompt,
                system_prompt=system_prompt, user_prompt=user_prompt, wm_state=wm_state,
                max_imagined_steps=max_imagined_steps, rollout_temperatures=rollout_temperature_plan,
                state_history=state_history, state_history_size=state_history_size,
                single_call_step=single_call_step, llm_batch_parallelism=llm_batch_parallelism,
            )
            candidate_rollouts = [
                {"rollout_index": i, "imagined_steps": steps} for i, steps in enumerate(all_imagined_steps)
            ]
        else:
            candidate_rollouts = []
            for rollout_index, roll_temp in enumerate(rollout_temperature_plan):
                steps = imagine_trajectory(
                    agent_generator,
                    wm_generator,
                    conversation,
                    previous_state,
                    react_system_prompt,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    wm_state=wm_state,
                    max_imagined_steps=max_imagined_steps,
                    state_history=state_history,
                    state_history_size=state_history_size,
                    temperature=roll_temp,
                )
                candidate_rollouts.append({"rollout_index": rollout_index, "imagined_steps": steps})
        if selection == "llm_judge" and len(candidate_rollouts) > 1:
            decision = select_imagined_trajectory_with_llm_judge(
                agent_generator, user_prompt, conversation, candidate_rollouts
            )
            chosen = int(decision.get("selected_index", 0))
        else:
            chosen = 0  # "first"
        chosen = chosen if 0 <= chosen < len(candidate_rollouts) else 0
        return candidate_rollouts[chosen].get("imagined_steps") or []

    return imagine_trajectory(
        agent_generator,
        wm_generator,
        conversation,
        previous_state,
        react_system_prompt,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        wm_state=wm_state,
        max_imagined_steps=max_imagined_steps,
        state_history=state_history,
        state_history_size=state_history_size,
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# Dynamic-K controllers — choose the imagined depth per step
# ---------------------------------------------------------------------------
#
# K_CONTROLLER selects how max_imagined_steps is decided each real step:
#   unset/none          → static WM_IMAGINED_MAX_STEPS
#   react_wm_decide_k   → the agent LLM picks K in [0, kmax] (decide_k_via_agent, below)
#   react_wm_rl_k       → a trained K-head picks K (vendored in k_controller.py)
# kmax = WM_IMAGINED_MAX_STEPS; K=0 means "skip imagination this step".

K_CONTROLLER_DECIDE_K = "react_wm_decide_k"
K_CONTROLLER_RL_K = "react_wm_rl_k"
K_CONTROLLERS = (K_CONTROLLER_DECIDE_K, K_CONTROLLER_RL_K)

_DECIDE_K_SYSTEM_PROMPT = (
    "You are an enterprise-operations agent's foresight planner. "
    "Given the current chat context and the agent's options, decide how many "
    "steps to look ahead with a learned world model BEFORE you pick the next "
    "tool call.\n\n"
    "Output a single JSON object with the key `k` whose value is an integer "
    "between 0 and {kmax} inclusive.\n"
    "* k=0 means: do NOT call the world model -- act immediately.\n"
    "* k>=1 means: ask the world model to imagine the next k steps and use "
    "that foresight to inform the next tool call.\n\n"
    "Pick a larger k when the task is long-horizon, irreversible, or has "
    "subtle multi-step dependencies; pick a smaller k when the next call is "
    "obvious or cheap to retry."
)
_DECIDE_K_USER_TEMPLATE = (
    "Conversation so far:\n{conversation}\n\n"
    "Most recent observed state summary:\n{state_summary}\n\n"
    "Respond with JSON ONLY in the form {{\"k\": <int>}}. /no_think"
)
_K_PATTERN = re.compile(r'"k"\s*:\s*(-?\d+)')


def _render_recent_conversation(conversation: List[Dict[str, Any]], max_chars: int = 4000) -> str:
    parts: List[str] = []
    for message in conversation[-12:]:
        role = message.get("role", "?")
        content = message.get("content")
        content_text = (
            json.dumps(content, ensure_ascii=False, default=str)
            if isinstance(content, (dict, list))
            else str(content or "")
        )
        parts.append(f"[{role}] {content_text}")
    text = "\n".join(parts)
    return text[-max_chars:] if len(text) > max_chars else text


def _parse_k_response(raw: str, kmax: int) -> Optional[int]:
    raw = (raw or "").strip()
    if not raw:
        return None
    raw = raw.split("</think>", 1)[-1].strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw).rstrip("`").strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "k" in obj:
            return max(0, min(int(obj["k"]), kmax))
    except Exception:
        pass
    match = _K_PATTERN.search(raw)
    if match:
        try:
            return max(0, min(int(match.group(1)), kmax))
        except Exception:
            return None
    bare = re.search(r"\b(\d+)\b", raw)
    if bare:
        try:
            return max(0, min(int(bare.group(1)), kmax))
        except Exception:
            return None
    return None


def decide_k_via_agent(
    agent_generator: Any,
    conversation: List[Dict[str, Any]],
    state: Any,
    kmax: int,
    fallback_k: int = 0,
) -> int:
    """react_wm_decide_k: ask the action LLM how many steps to look ahead (0..kmax)."""
    kmax = max(0, int(kmax))
    if kmax == 0:
        return 0
    state_summary = "(no state)"
    if state is not None:
        try:
            state_summary = json.dumps(state, ensure_ascii=False, default=str)[:2000]
        except Exception:
            state_summary = str(state)[:2000]
    messages = [
        {"role": "system", "content": _DECIDE_K_SYSTEM_PROMPT.format(kmax=kmax)},
        {
            "role": "user",
            "content": _DECIDE_K_USER_TEMPLATE.format(
                conversation=_render_recent_conversation(conversation),
                state_summary=state_summary,
            ),
        },
    ]
    try:
        raw = _generate_with_optional_temperature(agent_generator, messages, temperature=0.0)
    except Exception as exc:
        logger.warning("decide_k_via_agent generation failed: %s", exc)
        return max(0, min(fallback_k, kmax))
    parsed = _parse_k_response(raw, kmax)
    return parsed if parsed is not None else max(0, min(fallback_k, kmax))


def build_k_controller_state_text(
    system_prompt: str,
    user_prompt: str,
    previous_state: Dict[str, Any],
    state_history: Optional[List[Dict[str, Any]]],
) -> str:
    """State-side text for the trained K-controller.

    Mirrors ``train_adaptive_k.build_state_text_from_example`` (system + user +
    ``build_state_context_input_text``) so the K-head sees the same prompt shape
    and state rendering it was trained on.
    """
    example = ft.WorldModelStateExample(
        trajectory_id="0", trajectory_index=0,
        interaction_index=len(state_history or []),
        system_prompt=system_prompt, user_prompt=user_prompt, action="",
        state_history=list(state_history or []), input_history=[],
        previous_state=previous_state, state=ft.make_blank_state_like(previous_state),
    )
    return (
        f"System prompt:\n{system_prompt}\n\n"
        f"User prompt:\n{user_prompt}\n\n"
        f"{ft.build_state_context_input_text(example)}"
    )


# ---------------------------------------------------------------------------
# Conversation conversion (wm_react conversation_flow -> EWM role dicts)
# ---------------------------------------------------------------------------


def to_ewm_conversation(conversation_flow: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Map wm_react's ``conversation_flow`` events to role-dict messages."""
    converted: List[Dict[str, Any]] = []
    for event in conversation_flow or []:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if etype == "system_message":
            converted.append({"role": "system", "content": str(event.get("content", ""))})
        elif etype == "user_message":
            converted.append({"role": "user", "content": str(event.get("content", ""))})
        elif etype == "ai_message":
            message: Dict[str, Any] = {"role": "assistant", "content": str(event.get("content", "") or "")}
            tool_calls = event.get("tool_calls") or []
            if tool_calls:
                message["tool_calls"] = [
                    {"type": "function", "function": {"name": tc.get("name", ""), "arguments": tc.get("args", {})}}
                    for tc in tool_calls
                ]
            converted.append(message)
        elif etype == "tool_result":
            converted.append(
                {
                    "role": "tool",
                    "name": event.get("tool_name", "unknown"),
                    "content": stringify_tool_output((event.get("result") or {}).get("result", event.get("result"))),
                }
            )
    return converted


def enterprise_state_from_flow(
    conversation_flow: List[Dict[str, Any]],
    max_items: int = ft.DEFAULT_STATE_HISTORY_SIZE,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Reconstruct the **enterprise** state + multi-step history from the flow.

    Folds the real ``update_state_from_actual_execution`` over each ``tool_result``
    in the flow, seeded from the enterprise blank state — reproducing the state the
    WM was trained/replayed on (instead of a compact flat shortcut). Returns
    ``(previous_state, state_history)`` for the WM prompt.
    """
    state = ft.make_blank_state()
    history = ft.append_state_history([], state, max_items=max_items)
    for event in conversation_flow or []:
        if not isinstance(event, dict) or event.get("type") != "tool_result":
            continue
        state = update_state_from_actual_execution(
            state, [_execution_result_from_flow_event(event)], None, False
        )
        history = ft.append_state_history(history, state, max_items=max_items)
    return state, history
