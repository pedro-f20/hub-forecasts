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
SEASON_START = "2026-10-24"  # first 2026-27 round
CELL = ["hub", "reference_date", "target", "horizon", "location", "target_end_date"]
MODEL_COLUMNS = {"reference_date", "target", "horizon", "location", "target_end_date", "output_type", "output_type_id", "value"}


def fetch(name: str, models: list[str]) -> Path:
    spec, _ = hubs.get(name)
    hub = Path(f"hub-{name}")
    shutil.rmtree(hub, ignore_errors=True)
    subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                    f"https://github.com/{spec.repo}.git", str(hub)], check=True)
    subprocess.run(["git", "-C", str(hub), "sparse-checkout", "set", "target-data", "auxiliary-data/target-data-archive",
                    *[f"model-output/{m}" for m in models]], check=True)
    return hub


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


def score_hub(name: str, models: list[str]) -> pd.DataFrame:
    """WIS per cell (rows) and model (columns); NaN where a model has no forecast or the truth is missing."""
    spec, adapter = hubs.get(name)
    hub = fetch(name, models)
    truth = adapter.latest_truth(hub, spec).dropna(subset=["value"])
    truth = truth.rename(columns={"date": "target_end_date"}).set_index(["target", "location", "target_end_date"]).value
    levels = np.array(spec.quantiles)
    out = {}
    for model in models:
        q = load_model(hub, name, model)
        q.index = q.index.set_levels(pd.to_datetime(q.index.levels[-1]), level="target_end_date")
        y = truth.reindex(q.index.droplevel(["reference_date", "horizon"])).to_numpy()
        out[model] = pd.Series(wis(q.to_numpy(float), levels, y), index=q.index)
    return pd.concat({name: pd.DataFrame(out)}, names=["hub"])


def shared(cells: pd.DataFrame, model: str, other: str) -> pd.DataFrame:
    """Cells both models forecast and the truth covers."""
    return cells[[model, other]].dropna()


def ratio(pair: pd.DataFrame, model: str, other: str) -> float:
    """Geometric mean over (hub, target, location) of mean-WIS ratio, as in respicast-chronos2's scoreboard."""
    by_series = pair.groupby(["hub", "target", "location"])[[model, other]].mean()
    by_series = by_series[by_series[other] > 0]
    return float(np.exp(np.log(by_series[model] / by_series[other]).mean()))


def build(cells: pd.DataFrame, model: str, ensemble: str, baseline: str) -> str:
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
    return "\n".join(lines) + "\n"


def build_scores(cells: pd.DataFrame, model: str, ensemble: str, baseline: str, today: str) -> dict:
    """Headline, period and rows, in the shape AI4G's status.parse_scores checks."""
    vs_ensemble = shared(cells, model, ensemble)
    if vs_ensemble.empty:
        return {"headline": "Live scores pending: no truth yet for our rounds",
                "period": f"2026-27 season, 0 rounds scored; updated {today}", "rows": []}
    groups = [(f"Us vs hub ensemble, {HUBS[h]}", g, ensemble) for h, g in vs_ensemble.groupby(level="hub")]
    groups += [("Us vs hub ensemble, overall", vs_ensemble, ensemble),
               ("Us vs baseline, overall", shared(cells, model, baseline), baseline)]
    rows = [{"measure": label, "value": f"{ratio(pair, model, other):.2f}" if len(pair) else "-", "range": f"{len(pair)} cells"}
            for label, pair, other in groups]
    r, rounds = ratio(vs_ensemble, model, ensemble), vs_ensemble.index.unique("reference_date").size
    return {"headline": f"{round(abs(1 - r) * 100)}% {'less' if r < 1 else 'more'} error than the hub ensemble",
            "period": f"2026-27 season, {rounds} round{'s' if rounds != 1 else ''} scored, interim; updated {today}",
            "rows": rows}


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    spec, _ = hubs.get("ai4c-hosp")
    models = [spec.model, spec.ensemble, spec.baseline]
    cells = pd.concat([score_hub(name, models) for name in HUBS])
    (out / "SEASON.md").write_text(build(cells, *models))
    (out / "scores.json").write_text(json.dumps(build_scores(cells, *models, date.today().isoformat()), indent=2) + "\n")
    print(f"[scoreboard] {len(shared(cells, *models[:2]))} cells vs the ensemble; wrote {out / 'SEASON.md'} and {out / 'scores.json'}")


if __name__ == "__main__":
    main()
