"""Live forecast for one hub round: zero-shot Chronos-2 in the hub's model-output format.

Usage: python forecast.py <hub name> <hub checkout> <round date YYYY-MM-DD> <out dir>
Exit 3 when the round's data isn't out yet (newest week older than the hub's expected last week).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from hub_engine import core, hubs

EXIT_NO_DATA = 3


def allowed(tasks_path: Path, spec: core.HubSpec, key: str) -> set[str]:
    rounds = json.loads(tasks_path.read_text())["rounds"]
    values = set()
    for mt in (mt for r in rounds for mt in r["model_tasks"]):
        targets = set(mt["task_ids"]["target"].get("optional") or []) | set(mt["task_ids"]["target"].get("required") or [])
        if targets & set(spec.targets):
            ids = mt["task_ids"][key]
            values |= set(map(str, (ids.get("optional") or []) + (ids.get("required") or [])))
    return values


def main() -> None:
    name, hub, ref, out_dir = sys.argv[1], Path(sys.argv[2]), pd.Timestamp(sys.argv[3]), Path(sys.argv[4])
    spec, adapter = hubs.get(name)
    history = adapter.history(hub, spec, ref)
    expected_last = spec.target_end_date(ref, spec.last_data_horizon)
    if history.date.max() < expected_last:
        print(f"[forecast] {name} {ref.date()}: newest data {history.date.max().date()}, round data not out yet")
        sys.exit(EXIT_NO_DATA)
    tasks = hub / "hub-config" / "tasks.json"
    history = history[history.location.astype(str).isin(allowed(tasks, spec, "location"))]
    ends = [spec.target_end_date(ref, h) for h in spec.horizons]
    rows = core.to_hub_rows(core.chronos2_quantiles(history, ends, spec.quantiles), spec, ref)
    problems = core.check_against_tasks(rows, spec, tasks)
    if problems:
        raise ValueError(f"[forecast] {name} {ref.date()}: " + "; ".join(problems))
    path = core.write_rows(rows, spec, ref, out_dir)
    print(f"[forecast] {path.name}: {len(rows)} rows, {rows.location.nunique()} locations, data to {history.date.max().date()}")


if __name__ == "__main__":
    main()
