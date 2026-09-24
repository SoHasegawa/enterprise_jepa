"""With-WM / without-WM evaluation scoring.

Given two arms of per-task results — ``off`` (WM disabled / baseline) and ``on`` (WM enabled) —
compute each arm's score and the delta. Benchmark-agnostic: an arm is just a list of
``{"task_id": str, "success": bool}`` records (or a results dir containing such a JSON file).

This backs the ``ejepa result wm-compare`` subcommand and the with/without-WM example.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ArmScore:
    arm: str
    n: int
    n_success: int
    score: float  # success rate in [0, 1]


@dataclass
class WMComparison:
    off: ArmScore
    on: ArmScore
    delta_pp: float  # (on.score - off.score) * 100

    def to_dict(self) -> dict[str, Any]:
        return {"off": asdict(self.off), "on": asdict(self.on), "delta_pp": self.delta_pp}


def _score(arm: str, records: list[dict[str, Any]]) -> ArmScore:
    n = len(records)
    n_success = sum(1 for r in records if bool(r.get("success")))
    return ArmScore(arm=arm, n=n, n_success=n_success, score=(n_success / n if n else 0.0))


def compare(off_records: list[dict[str, Any]], on_records: list[dict[str, Any]]) -> WMComparison:
    off = _score("off", off_records)
    on = _score("on", on_records)
    return WMComparison(off=off, on=on, delta_pp=round((on.score - off.score) * 100.0, 2))


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load per-task records from a JSON file, or from ``<dir>/wm_eval_records.json``."""
    p = Path(path)
    if p.is_dir():
        p = p / "wm_eval_records.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "records" in data:
        data = data["records"]
    return list(data)


def format_table(cmp: WMComparison) -> str:
    sign = "+" if cmp.delta_pp >= 0 else ""
    return (
        "WM evaluation (with / without)\n"
        f"  without WM (off): {cmp.off.n_success}/{cmp.off.n}  = {cmp.off.score * 100:.1f}%\n"
        f"  with WM    (on):  {cmp.on.n_success}/{cmp.on.n}  = {cmp.on.score * 100:.1f}%\n"
        f"  delta:            {sign}{cmp.delta_pp:.2f} pp"
    )


def write_report(out_dir: Path, cmp: WMComparison) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wm_compare.json").write_text(
        json.dumps(cmp.to_dict(), indent=2), encoding="utf-8"
    )
    md = out_dir / "wm_compare.md"
    md.write_text("```\n" + format_table(cmp) + "\n```\n", encoding="utf-8")
    return md
