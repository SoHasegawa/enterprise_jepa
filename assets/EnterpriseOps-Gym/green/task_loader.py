"""EnterpriseOps-Gym task loader (local JSON and HuggingFace `ServiceNow-AI/EnterpriseOps-Gym`).

Official dataset: https://huggingface.co/datasets/ServiceNow-AI/EnterpriseOps-Gym
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tomllib
from typing import Any


_HF_DATASET_ID = "ServiceNow-AI/EnterpriseOps-Gym"
_HF_MODES = ("oracle", "plus_5_tools", "plus_10_tools", "plus_15_tools")
_HF_DOMAINS = (
    "calendar",
    "csm",
    "drive",
    "email",
    "hr",
    "hybrid",
    "itsm",
    "teams",
)
_JSON_STRING_FIELDS = ("gym_servers_config", "verifiers")

# Fixed HF-backed evaluation slices (task IDs live in green/tasks/task_ids.toml).
HF_TARGET_DEFAULTS: dict[str, dict[str, Any]] = {
    "opsgym_80_test": {
        "mode": "oracle",
        "domains": list(_HF_DOMAINS),
        "use_task_ids": True,
    },
    "opsgym_domain_smoke": {
        "mode": "oracle",
        "domains": list(_HF_DOMAINS),
        "use_task_ids": True,
    },
    "opsgym_train": {
        "mode": "oracle",
        "domains": list(_HF_DOMAINS),
        "use_task_ids": False,
        "exclude_task_ids_target": "opsgym_80_test",
        "test_excluded_split": "train",
    },
    "opsgym_valid": {
        "mode": "oracle",
        "domains": list(_HF_DOMAINS),
        "use_task_ids": False,
        "exclude_task_ids_target": "opsgym_80_test",
        "test_excluded_split": "valid",
    },
}


def _normalize_local_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single local JSON entry into the benchmark's internal representation."""
    task_id = raw.get("task_id") or raw.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("Each task must have a non-empty string 'task_id' or 'id'")

    system_prompt = str(raw.get("system_prompt", "")).strip()
    user_prompt = str(raw.get("user_prompt", "")).strip()
    if not system_prompt or not user_prompt:
        raise ValueError(f"Task {task_id} must define both system_prompt and user_prompt")

    selected_tools_raw = raw.get("selected_tools") or []
    selected_tools = [str(item) for item in selected_tools_raw if isinstance(item, str) and item.strip()]
    restricted_tools_raw = raw.get("restricted_tools") or []
    restricted_tools = [str(item) for item in restricted_tools_raw if isinstance(item, str) and item.strip()]

    gym_servers_config = raw.get("gym_servers_config")
    if isinstance(gym_servers_config, str):
        gym_servers_config = json.loads(gym_servers_config)
    if not isinstance(gym_servers_config, list) or not gym_servers_config:
        raise ValueError(f"Task {task_id} must define a non-empty gym_servers_config list")

    verifiers = raw.get("verifiers")
    if isinstance(verifiers, str):
        verifiers = json.loads(verifiers)
    if not isinstance(verifiers, list):
        raise ValueError(f"Task {task_id} verifiers must be a list")

    domain = str(raw.get("domain", "")).strip()
    mcp_endpoint = str(raw.get("mcp_endpoint", "/mcp")).strip() or "/mcp"
    number_of_runs = int(raw.get("number_of_runs", 1) or 1)
    reset_database_between_runs = bool(raw.get("reset_database_between_runs", True))

    return {
        "id": task_id,
        "domain": domain,
        "mode": str(raw.get("mode", "")).strip(),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "selected_tools": selected_tools,
        "restricted_tools": restricted_tools,
        "gym_servers_config": gym_servers_config,
        "verifiers": verifiers,
        "mcp_endpoint": mcp_endpoint,
        "number_of_runs": number_of_runs,
        "reset_database_between_runs": reset_database_between_runs,
    }


def _coerce_string_or_list(value: Any) -> list[str]:
    """Normalize a config value into a list of unique non-empty strings.

    Accepts:
    - a list (each entry coerced to str)
    - a single string (treated as one entry, or comma-separated multi-entry)
    - anything else → empty list

    This lets ``--config domains=email`` (CLI parses to the string ``"email"``)
    and ``--config 'domains=["email","teams"]'`` (CLI parses to a list) both work
    when used under the plural key.
    """
    out: list[str] = []
    seen: set[str] = set()

    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
        return out

    if isinstance(value, str):
        for part in value.split(","):
            text = part.strip()
            if text and text not in seen:
                seen.add(text)
                out.append(text)
        return out

    return out


def resolve_hf_modes(config: dict[str, Any]) -> list[str]:
    """Return one or more HF `mode` values (HF dataset config names).

    Priority: ``config["modes"]`` → ``config["mode"]`` → ``ENTERPRISEOPS_HF_MODE``
    (comma-separated allowed). Defaults to ``["oracle"]``.
    """
    out = _coerce_string_or_list(config.get("modes"))
    if out:
        return out

    single = config.get("mode")
    if isinstance(single, str) and single.strip():
        return [single.strip()]

    env_modes = _coerce_string_or_list(os.getenv("ENTERPRISEOPS_HF_MODE", ""))
    if env_modes:
        return env_modes

    return ["oracle"]


def resolve_hf_domains(config: dict[str, Any]) -> list[str]:
    """Return one or more HF `split` values (= domain names).

    Priority: ``config["domains"]`` → ``config["domain"]`` → ``ENTERPRISEOPS_HF_DOMAIN``
    (comma-separated allowed).
    """
    out = _coerce_string_or_list(config.get("domains"))
    if out:
        return out

    single = config.get("domain")
    if isinstance(single, str) and single.strip():
        return [single.strip()]

    env_domains = _coerce_string_or_list(os.getenv("ENTERPRISEOPS_HF_DOMAIN", ""))
    if env_domains:
        return env_domains

    return []


def resolve_max_tasks_per_domain(config: dict[str, Any]) -> int | None:
    """Maximum number of tasks per domain. ``None`` means "all rows in that domain split"."""
    if "max_tasks_per_domain" in config:
        val = config["max_tasks_per_domain"]
        if val is None:
            return None
        n = int(val)
        if n < 1:
            raise ValueError("config.max_tasks_per_domain must be >= 1 when set")
        return n

    raw = os.getenv("ENTERPRISEOPS_HF_MAX_TASKS", "").strip()
    if raw:
        n = int(raw)
        if n < 1:
            raise ValueError("ENTERPRISEOPS_HF_MAX_TASKS must be >= 1 when set")
        return n
    return None


def is_supported_mode(mode: str) -> bool:
    """Whether the given mode (HF dataset config name) is supported."""
    return mode in _HF_MODES


def is_supported_domain(domain: str) -> bool:
    """Whether the given domain (HF dataset split name) is supported."""
    return domain in _HF_DOMAINS


def is_hf_corpus_target(target: str) -> bool:
    """Whether the target loads tasks from the Hugging Face EnterpriseOps-Gym corpus."""
    return target == "hf_dataset" or target in HF_TARGET_DEFAULTS


def resolve_hf_target_config(target: str, config: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Return an HF loader config for corpus-backed targets, or ``None`` for local targets."""
    cfg = dict(config or {})
    if target == "hf_dataset":
        return cfg

    defaults = HF_TARGET_DEFAULTS.get(target)
    if defaults is None:
        return None

    for key, value in defaults.items():
        if key not in cfg:
            cfg[key] = list(value) if isinstance(value, list) else value
    return cfg


def _order_tasks_by_ids(tasks: list[dict[str, Any]], task_ids: list[str]) -> list[dict[str, Any]]:
    by_id = {task["id"]: task for task in tasks}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise KeyError(f"Task IDs not found after HF load: {', '.join(missing)}")
    return [by_id[task_id] for task_id in task_ids]


def _stable_split_key(task: dict[str, Any]) -> str:
    return hashlib.sha256(str(task["id"]).encode("utf-8")).hexdigest()


def _test_excluded_split(tasks: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    """Return deterministic train/valid slices after held-out-test exclusion."""
    ordered = sorted(tasks, key=_stable_split_key)
    valid_count = max(1, round(len(ordered) * 0.1)) if ordered else 0
    valid_ids = {task["id"] for task in ordered[:valid_count]}
    if split == "valid":
        return [task for task in tasks if task["id"] in valid_ids]
    if split == "train":
        return [task for task in tasks if task["id"] not in valid_ids]
    raise ValueError(f"Unsupported test-excluded split: {split}")


class LocalTaskLoader:
    """Loader that reads from the bundled `Tasks_*.json` files and `task_ids.toml`."""

    def __init__(
        self,
        tasks_dir: Path | None = None,
        task_ids_path: Path | None = None,
    ) -> None:
        base = Path(__file__).resolve().parent / "tasks"
        self._tasks_dir = tasks_dir or base
        self._task_ids_path = task_ids_path or (base / "task_ids.toml")

    def _task_files(self) -> list[Path]:
        if self._tasks_dir.is_dir():
            files = sorted(self._tasks_dir.glob("Tasks_*.json"))
        else:
            files = [self._tasks_dir]
        if not files:
            raise FileNotFoundError(f"No task files found: {self._tasks_dir}")
        return files

    def load_task_ids(self, target: str) -> list[str]:
        if not self._task_ids_path.exists():
            raise FileNotFoundError(f"task_ids.toml not found: {self._task_ids_path}")
        with self._task_ids_path.open("rb") as fh:
            raw = tomllib.load(fh)
        ids = raw.get(target)
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"No task IDs defined for target '{target}' in {self._task_ids_path}")
        return [str(i) for i in ids]

    def _load_catalog(self) -> dict[str, dict[str, Any]]:
        catalog: dict[str, dict[str, Any]] = {}
        duplicates: list[str] = []
        for path in self._task_files():
            raw_list = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw_list, list):
                raise ValueError(f"Task file must contain a JSON array: {path}")
            for entry in raw_list:
                normalized = _normalize_local_entry(entry)
                tid = normalized["id"]
                if tid in catalog:
                    duplicates.append(tid)
                    continue
                catalog[tid] = normalized
        if duplicates:
            raise ValueError(f"Duplicate task IDs: {', '.join(sorted(set(duplicates)))}")
        return catalog

    def load_tasks(
        self,
        target: str,
        requested_task_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        task_ids = requested_task_ids or self.load_task_ids(target)
        catalog = self._load_catalog()
        missing = [tid for tid in task_ids if tid not in catalog]
        if missing:
            raise KeyError(f"Task IDs not found in catalog: {', '.join(missing)}")
        return [catalog[tid] for tid in task_ids]


class HuggingFaceTaskLoader:
    """Loads `ServiceNow-AI/EnterpriseOps-Gym` via the `datasets` library.

    Dependency: ``pip install datasets``

    Mode is taken from ``resolve_hf_modes(config)``, domain from
    ``resolve_hf_domains(config)``, and the per-domain row limit from
    ``resolve_max_tasks_per_domain(config)`` (unset means all rows).
    """

    @staticmethod
    def _row_to_task(row: dict[str, Any], *, mode: str, domain: str) -> dict[str, Any]:
        normalized: dict[str, Any] = {
            "task_id": row.get("task_id"),
            "domain": row.get("domain") or domain,
            "mode": mode,
            "system_prompt": row.get("system_prompt", ""),
            "user_prompt": row.get("user_prompt", ""),
            "selected_tools": row.get("selected_tools") or [],
            "restricted_tools": row.get("restricted_tools") or [],
            "gym_servers_config": row.get("gym_servers_config"),
            "verifiers": row.get("verifiers"),
            "mcp_endpoint": row.get("mcp_endpoint") or "/mcp",
            "number_of_runs": row.get("number_of_runs", 1) or 1,
            "reset_database_between_runs": bool(row.get("reset_database_between_runs", True)),
        }
        for key in _JSON_STRING_FIELDS:
            value = normalized.get(key)
            if isinstance(value, str):
                normalized[key] = json.loads(value)
        return _normalize_local_entry(normalized)

    def load_corpus_tasks(
        self,
        config: dict[str, Any],
        *,
        requested_task_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        try:
            from datasets import load_dataset  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceTaskLoader requires the 'datasets' package. "
                "Run: pip install datasets"
            ) from exc

        modes = resolve_hf_modes(config)
        domains = resolve_hf_domains(config)
        if not domains:
            raise ValueError(
                "target hf_dataset requires domains: set config.domains, config.domain, "
                "or ENTERPRISEOPS_HF_DOMAIN (e.g. calendar or calendar,teams). "
                f"Supported domains: {', '.join(_HF_DOMAINS)}."
            )

        for mode in modes:
            if not is_supported_mode(mode):
                raise ValueError(
                    f"Unsupported HF mode: {mode!r}. Supported modes: {', '.join(_HF_MODES)}."
                )
        for domain in domains:
            if not is_supported_domain(domain):
                raise ValueError(
                    f"Unsupported HF domain: {domain!r}. Supported domains: {', '.join(_HF_DOMAINS)}."
                )

        max_per = resolve_max_tasks_per_domain(config)

        all_tasks: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        found_ids: set[str] = set()

        for mode in modes:
            for domain in domains:
                ds = load_dataset(_HF_DATASET_ID, mode, split=domain)
                rows: list[dict[str, Any]] = list(ds)

                if requested_task_ids:
                    id_set = set(requested_task_ids)
                    rows = [r for r in rows if str(r.get("task_id", "")) in id_set]
                elif max_per is not None:
                    rows = rows[:max_per]

                for row in rows:
                    task = self._row_to_task(row, mode=mode, domain=domain)
                    if task["id"] in seen_ids:
                        continue
                    seen_ids.add(task["id"])
                    all_tasks.append(task)
                    found_ids.add(task["id"])

        if requested_task_ids:
            missing = set(requested_task_ids) - found_ids
            if missing:
                raise KeyError(
                    "Task IDs not found in HuggingFace corpus (any mode/domain): "
                    f"{', '.join(sorted(missing))}"
                )

        return all_tasks


class TaskLoader:
    """Dispatches to the local or HuggingFace loader based on the target."""

    def load_tasks(
        self,
        target: str,
        requested_task_ids: list[str] | None = None,
        *,
        hf_config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        resolved_hf_config = resolve_hf_target_config(target, hf_config)
        if resolved_hf_config is not None:
            target_defaults = HF_TARGET_DEFAULTS.get(target, {})
            env_ids = os.getenv("ENTERPRISEOPS_HF_TASK_IDS", "").strip()
            ids: list[str] | None = requested_task_ids
            if not ids and env_ids:
                ids = [part.strip() for part in env_ids.split(",") if part.strip()]
            if not ids and target_defaults.get("use_task_ids", False):
                ids = LocalTaskLoader().load_task_ids(target)
            tasks = HuggingFaceTaskLoader().load_corpus_tasks(
                resolved_hf_config,
                requested_task_ids=ids,
            )
            exclude_target = target_defaults.get("exclude_task_ids_target")
            if exclude_target:
                excluded_ids = set(LocalTaskLoader().load_task_ids(str(exclude_target)))
                if ids:
                    leaked_requested = sorted(set(ids) & excluded_ids)
                    if leaked_requested:
                        raise ValueError(
                            f"Requested held-out task IDs are not allowed for target {target}: "
                            f"{', '.join(leaked_requested[:10])}"
                        )
                tasks = [task for task in tasks if task["id"] not in excluded_ids]
            split = target_defaults.get("test_excluded_split")
            if split and not ids:
                tasks = _test_excluded_split(tasks, str(split))
            if ids:
                return _order_tasks_by_ids(tasks, ids)
            return tasks
        return LocalTaskLoader().load_tasks(target, requested_task_ids)
