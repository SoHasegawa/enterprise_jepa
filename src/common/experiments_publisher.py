"""
experiments_publisher.py
~~~~~~~~~~~~~~~~~~~~~~~~
BenchmarkRunManifest + detail.json を experiments リポジトリ形式へ変換し、
指定ディレクトリへ書き出すユーティリティ。

使用例::

    from common.experiments_publisher import publish_to_experiments
    from common.result_store import load_result_manifest

    manifest = load_result_manifest(Path("results/.../manifest.json"))
    with open("results/.../detail.json") as f:
        detail = json.load(f)

    out_dir = publish_to_experiments(
        manifest=manifest,
        detail=detail,
        experiments_root=Path("/path/to/experiments"),
        submitted_by="your-team",
    )
    print(f"Written to: {out_dir}")
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from common.models import BenchmarkRunManifest


def _run_dir_name(manifest: BenchmarkRunManifest) -> str:
    """experiments/evaluation/<Benchmark>/ 配下のディレクトリ名を生成する。

    形式: <YYYYMMDD>_<executor_slug>_<model_slug>
    """
    date_str = manifest.created_at_utc.strftime("%Y%m%d")
    executor_slug = _slug(manifest.executor_name, max_length=24)
    model = manifest.request_config.get("model", "") or manifest.request_config.get("llm_model", "")
    model_slug = _slug(model, max_length=32) if model else "unknown_model"
    return f"{date_str}_{executor_slug}_{model_slug}"


def _slug(value: str, *, max_length: int = 48) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]", "_", value.strip())
    collapsed = re.sub(r"_+", "_", normalized).strip("_")
    return (collapsed or "unknown")[:max_length]


def _build_results_json(manifest: BenchmarkRunManifest, detail: dict[str, Any]) -> dict:
    """results/results.json の内容を組み立てる。"""
    task_results: list[dict] = detail.get("task_results") or manifest.eval_result.task_results
    resolved: list[str] = []
    unresolved: list[str] = []
    no_generation: list[str] = []

    for tr in task_results:
        task_id = str(tr.get("task_id", tr.get("instance_id", "")))
        score = float(tr.get("score", tr.get("score_value", 0.0)))
        reason: str = str(tr.get("reason", "")).lower()

        if score >= 1.0:
            resolved.append(task_id)
        elif "no_generation" in reason or "no patch" in reason or "empty" in reason:
            no_generation.append(task_id)
        else:
            unresolved.append(task_id)

    total = len(task_results)
    resolved_count = len(resolved)
    rate = resolved_count / max(total, 1)

    return {
        "total_instances": total,
        "resolved_instances": resolved_count,
        "resolve_rate": round(rate, 6),
        "no_generation": no_generation,
        "resolved": resolved,
        "unresolved": unresolved,
    }


def _build_resolved_by_repo(results_json: dict) -> dict:
    """results/resolved_by_repo.json の内容を組み立てる。

    task_id の形式 "org__repo-NNNNN" を "org/repo" へ変換して集計する。
    """
    counts: dict[str, dict[str, int]] = {}

    def _to_repo(task_id: str) -> str:
        # SWE-bench 形式: "django__django-12345" → "django/django"
        if "__" in task_id:
            parts = task_id.split("__", 1)
            repo_issue = parts[1].rsplit("-", 1)
            return f"{parts[0]}/{repo_issue[0]}"
        return task_id.rsplit("-", 1)[0]

    all_ids = results_json["resolved"] + results_json["unresolved"] + results_json["no_generation"]
    for task_id in all_ids:
        repo = _to_repo(task_id)
        counts.setdefault(repo, {"resolved": 0, "total": 0})
        counts[repo]["total"] += 1

    for task_id in results_json["resolved"]:
        repo = _to_repo(task_id)
        counts[repo]["resolved"] += 1

    return counts


def _build_metadata_yaml(
    manifest: BenchmarkRunManifest,
    *,
    submitted_by: str,
    org: str,
    extra_info: dict | None = None,
) -> str:
    """metadata.yaml の文字列を生成する。"""
    info = extra_info or {}
    agent_image = (
        info.get("agent_image")
        or manifest.participants.get("purple")
        or manifest.participants.get("agent")
        or "unknown"
    )
    model = (
        info.get("llm_model")
        or manifest.request_config.get("model")
        or manifest.request_config.get("llm_model")
        or "unknown"
    )
    name = info.get("name") or f"{manifest.executor_name} ({model})"
    submitted_at = manifest.completed_at_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        "info:",
        f'  name: "{name}"',
        f'  benchmark: "{manifest.benchmark_name}"',
        f'  benchmark_version: "{manifest.benchmark_version or ""}"',
        f'  run_id: "{manifest.run_id}"',
        f'  executor: "{manifest.executor_name}"',
        f'  green_agent_version: "{manifest.green_agent_version or ""}"',
        f'  purple_agent_version: "{manifest.purple_agent_version or ""}"',
        f'  executor_version: "{manifest.executor_version or ""}"',
        f'  agent_image: "{agent_image}"',
        f'  llm_model: "{model}"',
        f'  submitted_by: "{submitted_by}"',
        f'  submitted_at: "{submitted_at}"',
        f'  report: "{info.get("report", "")}"',
        f'  site: "{info.get("site", "")}"',
        "tags:",
        "  checked: false",
        f'  org: "{org}"',
        "  os_model: false",
        "  os_system: true",
        "  system:",
        f'    attempts: "{info.get("attempts", "1")}"',
        "",
    ]
    return "\n".join(lines)


def _build_readme(manifest: BenchmarkRunManifest, results: dict, agent_readme: str | None) -> str:
    """run ディレクトリの README.md を生成する。"""
    if agent_readme:
        return agent_readme

    rate = results["resolve_rate"]
    resolved = results["resolved_instances"]
    total = results["total_instances"]
    model = manifest.request_config.get("model", "unknown")
    date = manifest.created_at_utc.strftime("%Y-%m-%d")

    return (
        f"# {manifest.executor_name}\n\n"
        f"**Benchmark**: {manifest.benchmark_name}  \n"
        f"**Run date**: {date}  \n"
        f"**Model**: {model}  \n\n"
        f"**Versions**: benchmark={manifest.benchmark_version or '—'}, "
        f"green={manifest.green_agent_version or '—'}, "
        f"purple={manifest.purple_agent_version or '—'}, "
        f"executor={manifest.executor_version or '—'}  \n\n"
        f"## Performance\n\n"
        f"```\n"
        f"Resolved {resolved} instances ({rate:.1%})\n"
        f"Total instances: {total}\n"
        f"```\n\n"
        f"## Score Summary\n\n"
        f"{manifest.score_summary}\n"
    )


def publish_to_experiments(
    manifest: BenchmarkRunManifest,
    detail: dict[str, Any],
    experiments_root: Path,
    *,
    submitted_by: str = "unspecified",
    org: str = "unspecified",
    agent_readme: str | None = None,
    extra_info: dict | None = None,
    run_dir_name: str | None = None,
) -> Path:
    """BenchmarkRunManifest を experiments リポジトリ形式へ変換して書き出す。

    Args:
        manifest: ``result_store.load_result_manifest()`` で読み込んだ manifest。
        detail: ``detail.json`` の dict（task_results リストを含む）。
        experiments_root: experiments リポジトリのルートディレクトリ。
        submitted_by: metadata.yaml の ``info.submitted_by`` に設定するユーザー/組織名。
        org: metadata.yaml の ``tags.org`` に設定する組織名。
        agent_readme: README.md のカスタム内容（None の場合は自動生成）。
        extra_info: metadata.yaml の ``info`` セクションに追加する dict。
        run_dir_name: 実行ディレクトリ名を上書きする場合に指定。

    Returns:
        書き出した run ディレクトリの Path。
    """
    experiments_root = Path(experiments_root).resolve()
    benchmark_dir = experiments_root / "evaluation" / manifest.benchmark_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    dirname = run_dir_name or _run_dir_name(manifest)
    run_dir = benchmark_dir / dirname
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_json = _build_results_json(manifest, detail)
    resolved_by_repo = _build_resolved_by_repo(results_json)
    metadata_yaml = _build_metadata_yaml(
        manifest, submitted_by=submitted_by, org=org, extra_info=extra_info
    )
    readme_md = _build_readme(manifest, results_json, agent_readme)

    _write_json(results_dir / "results.json", results_json)
    _write_json(results_dir / "resolved_by_repo.json", resolved_by_repo)
    (run_dir / "metadata.yaml").write_text(metadata_yaml, encoding="utf-8")
    (run_dir / "README.md").write_text(readme_md, encoding="utf-8")

    return run_dir


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
