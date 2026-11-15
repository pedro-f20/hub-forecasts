"""Live 2026-27 scoreboard for pedro-Chronos2 on the AI4Casting hubs. The hubs publish no scores, so WIS is
computed here from the quantiles, against the latest (revised) truth, on cells both models forecast.

Usage: python scoreboard.py [out_dir]   (default: .) -> SEASON.md and scores.json, the card the AI4G tracker shows.
Clones the hubs into hub-<name>/; any fetch or parse problem raises, so no stale or empty file is written.
"""
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from hub_engine import hubs

HUBS = {"ai4c-hosp": "hospital beds", "ai4c-rvdss": "lab positivity"}
HUB_PREFIX = "AI4Casting_Hub-"
SEASON_START = "2026-10-24"  # first 2026-27 round
CELL = ["hub", "reference_date", "target", "horizon", "location", "target_end_date"]
MODEL_COLUMNS = {"reference_date", "target", "horizon", "location", "target_end_date", "output_type", "output_type_id", "value"}


def fetch(name: str) -> Path:
    spec, _ = hubs.get(name)
    hub = Path(f"hub-{name}")
    shutil.rmtree(hub, ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                    f"https://github.com/{spec.repo}.git", str(hub)], check=True)
    subprocess.run(["git", "-C", str(hub), "sparse-checkout", "set", "target-data", "auxiliary-data/target-data-archive",
                    "model-output"], check=True)
    return hub


def team_dirs(hub: Path, model: str) -> list[str]:
    """Every outside team's model: the hub's own models (ensemble, baselines, GPT_4o) share the hub prefix."""
    return sorted(d.name for d in (hub / "model-output").iterdir()
                  if d.is_dir() and d.name != model and not d.name.startswith(HUB_PREFIX))


def wis(quantiles: np.ndarray, levels: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Weighted interval score per row of `quantiles` (cells x levels): mean over levels of twice the pinball loss.
    Equals (|y - median|/2 + sum_k alpha_k/2 * IS_alpha_k) / (K + 1/2) for K central intervals plus the median."""
    y = truth[:, None]
    return (2 * ((y < quantiles) - levels) * (quantiles - y)).mean(axis=1)


def load_model(hub: Path, name: str, model: str) -> pd.DataFrame:
    spec, _ = hubs.get(name)
    files = sorted(f for f in (hub / "model-output" / model).glob(f"*-{model}.csv") if f.name[:10] >= SEASON_START)
    if not files:
        raise FileNotFoundError(f"[scoreboard] {name}: no {model} files from {SEASON_START}")
    raw = pd.concat([pd.read_csv(f) for f in files])
    missing = MODEL_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"[scoreboard] {name} {model}: missing columns {sorted(missing)}")
    # weekly quantiles only: the peak targets are also quantile-typed but have no horizon and are not evaluated
    df = raw[(raw.output_type == "quantile") & raw.target.isin(spec.targets)].copy()
    df["level"] = df.output_type_id.astype(float).round(4)
    df = df[df.level.isin(spec.quantiles)]
    q = df.pivot_table(index=CELL[1:], columns="level", values="value", aggfunc="first")
    if list(q.columns) != list(spec.quantiles) or q.isna().any().any():
        raise ValueError(f"[scoreboard] {name} {model}: cells without the full quantile set {spec.quantiles}")
    return q


def score_hub(name: str, models: list[str]) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """WIS per cell (rows) and model (columns); NaN where a model has no forecast or the truth is missing.
    `models` (ours, ensemble, baseline) must load; a team whose files don't is left out and reported."""
    spec, adapter = hubs.get(name)
    hub = fetch(name)
    truth = adapter.latest_truth(hub, spec).dropna(subset=["value"])
    truth = truth.rename(columns={"date": "target_end_date"}).set_index(["target", "location", "target_end_date"]).value
    levels = np.array(spec.quantiles)
    teams, skipped, out = team_dirs(hub, models[0]), {}, {}
    for model in models + teams:
        try:
            q = load_model(hub, name, model)
        except FileNotFoundError:
            if model in models:
                raise
            continue  # team not submitting this season: shows as 0 cells
        except ValueError as e:
            if model in models:
                raise
            skipped[f"{model} ({HUBS[name]})"] = str(e).split(": ", 1)[-1]
            continue
        q.index = q.index.set_levels(pd.to_datetime(q.index.levels[-1]), level="target_end_date")
        y = truth.reindex(q.index.droplevel(["reference_date", "horizon"])).to_numpy()
        out[model] = pd.Series(wis(q.to_numpy(float), levels, y), index=q.index)
    return pd.concat({name: pd.DataFrame(out)}, names=["hub"]), teams, skipped


def rank_teams(cells: pd.DataFrame, model: str, teams: list[str]) -> list[dict]:
    """Ratio vs each team on shared cells, keeping teams that share at least half our widest overlap with any
    team, so a team scored on a handful of cells doesn't count as a rival; strongest teams (highest ratio) first."""
    scored = [{"team": t, "pair": shared(cells, model, t)} for t in teams if t in cells.columns]
    scored = [s for s in scored if len(s["pair"])]
    widest = max((len(s["pair"]) for s in scored), default=0)
    return sorted(({"team": s["team"], "ratio": ratio(s["pair"], model, s["team"]), "cells": len(s["pair"])}
                   for s in scored if 2 * len(s["pair"]) >= widest), key=lambda r: (-r["ratio"], r["team"]))


def shared(cells: pd.DataFrame, model: str, other: str) -> pd.DataFrame:
    """Cells both models forecast and the truth covers."""
    return cells[[model, other]].dropna()


def ratio(pair: pd.DataFrame, model: str, other: str) -> float:
    """Geometric mean over (hub, target, location) of mean-WIS ratio, as in respicast-chronos2's scoreboard."""
    by_series = pair.groupby(["hub", "target", "location"])[[model, other]].mean()
    by_series = by_series[by_series[other] > 0]
    return float(np.exp(np.log(by_series[model] / by_series[other]).mean()))


def build(cells: pd.DataFrame, model: str, ensemble: str, baseline: str,
          teams: list[str] = (), skipped: dict[str, str] | None = None) -> str:
    lines = ["# 2026-27 season scoreboard (AI4Casting)", "",
             f"Interim. WIS from the 7 quantiles against the latest truth, cells both models forecast. Ratio = {model} WIS / "
             "comparator WIS, geometric mean over (hub, target, location) of mean WIS; below 1 means "
             f"{model} is better.", "", "| hub | comparator | ratio | cells | rounds |", "|---|---|---|---|---|"]
    for label, sub in [*((HUBS[h], cells.xs(h, drop_level=False)) for h in cells.index.unique("hub")), ("overall", cells)]:
        for other in (ensemble, baseline):
            pair = shared(sub, model, other)
            value = f"{ratio(pair, model, other):.2f}" if len(pair) else "-"
            lines.append(f"| {label} | {other} | {value} | {len(pair)} | {pair.index.unique('reference_date').size} |")
    lines += ["", f"By round, vs {ensemble}:", "", "| round | hub | ratio | cells |", "|---|---|---|---|"]
    for (ref, h), g in shared(cells, model, ensemble).groupby(level=["reference_date", "hub"]):
        lines.append(f"| {ref} | {HUBS[h]} | {ratio(g, model, ensemble):.2f} | {len(g)} |")
    lines += ["", "Vs every outside team, both hubs (ranked: shares at least half the widest overlap):", "",
              "| team | ratio | cells | ranked |", "|---|---|---|---|"]
    ranked = {r["team"] for r in rank_teams(cells, model, teams)}
    for team in teams:
        pair = shared(cells, model, team) if team in cells.columns else cells.iloc[0:0]
        value = f"{ratio(pair, model, team):.2f}" if len(pair) else "-"
        lines.append(f"| {team} | {value} | {len(pair)} | {'yes' if team in ranked else 'no'} |")
    for team, why in (skipped or {}).items():
        lines.append(f"\nNot scored: {team}: {why}")
    return "\n".join(lines) + "\n"


MAX_CARD_ROWS = 6  # AI4G's parse_scores limit


def build_scores(cells: pd.DataFrame, model: str, ensemble: str, baseline: str, today: str,
                 teams: list[str] = ()) -> dict:
    """Headline (rank among outside teams first, then the ensemble), period and rows, in the shape AI4G's
    status.parse_scores checks."""
    vs_ensemble = shared(cells, model, ensemble)
    if vs_ensemble.empty:
        return {"headline": "Live scores pending: no truth yet for our rounds",
                "period": f"2026-27 season, 0 rounds scored; updated {today}", "rows": []}
    vs_baseline = shared(cells, model, baseline)
    rows = [{"measure": label, "value": f"{ratio(pair, model, other):.2f}" if len(pair) else "-", "range": f"{len(pair)} cells"}
            for label, pair, other in [("Us vs hub ensemble", vs_ensemble, ensemble), ("Us vs baseline", vs_baseline, baseline)]]
    ranked = rank_teams(cells, model, list(teams))
    rows += [{"measure": f"Us vs {r['team']}"[:40], "value": f"{r['ratio']:.2f}", "range": f"{r['cells']} cells"}
             for r in ranked][:MAX_CARD_ROWS - len(rows)]
    r, rounds = ratio(vs_ensemble, model, ensemble), vs_ensemble.index.unique("reference_date").size
    vs_ens = f"{round(abs(1 - r) * 100)}% {'less' if r < 1 else 'more'} error than the hub ensemble"
    beaten = sum(t["ratio"] < 1 for t in ranked)
    # no outside team scored: the ensemble is only the hub's in-house models, so say so
    headline = f"Better than {beaten} of {len(ranked)} teams; {vs_ens}" if ranked else f"Only outside team so far; {vs_ens}"
    return {"headline": headline,
            "period": f"2026-27 season, {rounds} round{'s' if rounds != 1 else ''} scored, interim; updated {today}",
            "rows": rows}


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    spec, _ = hubs.get("ai4c-hosp")
    models = [spec.model, spec.ensemble, spec.baseline]
    scored = [score_hub(name, models) for name in HUBS]
    cells = pd.concat([s[0] for s in scored])
    teams = sorted({t for s in scored for t in s[1]})
    skipped = {k: v for s in scored for k, v in s[2].items()}
    (out / "SEASON.md").write_text(build(cells, *models, teams, skipped))
    scores = build_scores(cells, *models, date.today().isoformat(), teams)
    (out / "scores.json").write_text(json.dumps(scores, indent=2) + "\n")
    print(f"[scoreboard] {len(shared(cells, *models[:2]))} cells vs the ensemble; wrote {out / 'SEASON.md'} and {out / 'scores.json'}")


if __name__ == "__main__":
    main()
