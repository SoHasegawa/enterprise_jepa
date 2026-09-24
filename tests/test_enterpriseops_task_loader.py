from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_enterpriseops_task_loader_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "EnterpriseOps-Gym"
        / "green"
        / "task_loader.py"
    )
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location("enterpriseops_task_loader", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


enterpriseops_task_loader = _load_enterpriseops_task_loader_module()


def test_opsgym_80_test_target_defaults_to_oracle_and_all_domains() -> None:
    resolved = enterpriseops_task_loader.resolve_hf_target_config("opsgym_80_test", {})
    assert resolved is not None
    assert resolved["mode"] == "oracle"
    assert resolved["domains"] == list(enterpriseops_task_loader._HF_DOMAINS)


def test_opsgym_80_test_task_ids_manifest_has_80_entries() -> None:
    loader = enterpriseops_task_loader.LocalTaskLoader()
    task_ids = loader.load_task_ids("opsgym_80_test")
    assert len(task_ids) == 80
    assert len(set(task_ids)) == 80


def test_task_loader_routes_opsgym_80_test_to_hf_loader(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeHfLoader:
        def load_corpus_tasks(self, config, *, requested_task_ids=None):
            captured["config"] = config
            captured["requested_task_ids"] = requested_task_ids
            return [
                {"id": task_id, "domain": "calendar", "mode": "oracle"}
                for task_id in (requested_task_ids or [])
            ]

    monkeypatch.setattr(enterpriseops_task_loader, "HuggingFaceTaskLoader", _FakeHfLoader)

    tasks = enterpriseops_task_loader.TaskLoader().load_tasks("opsgym_80_test")

    assert len(tasks) == 80
    assert captured["config"]["mode"] == "oracle"
    requested_ids = captured["requested_task_ids"]
    assert isinstance(requested_ids, list)
    assert len(requested_ids) == 80
    assert tasks[0]["id"] == requested_ids[0]
    assert tasks[-1]["id"] == requested_ids[-1]
