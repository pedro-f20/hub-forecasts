"""Hub-agnostic part: zero-shot Chronos-2 on weekly series, hubverse model-output rows, format checks.

Model settings are the ones pre-registered for RespiCast (pedro-Chronos2): pinned weights, cross-learning
across every series of the hub in one call. Do not tune per hub: the point is one model everywhere.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ID = "amazon/chronos-2"
MODEL_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"  # weights unchanged since 2025-10-30
WEEK = pd.Timedelta(days=7)


@dataclass(frozen=True)
class HubSpec:
    name: str
    repo: str
    model: str  # team-model, as in model-output/<model>/
    targets: tuple[str, ...]
    quantiles: tuple[float, ...]
    horizons: tuple[int, ...]
    ensemble: str
    baseline: str
    round_col: str = "reference_date"
    output_type: str = "quantile"  # or "sample" (trajectories over horizons)
    n_samples: int = 100
    integer_values: bool = False
    value_max: float | None = None
    file_ext: str = "csv"
    has_horizon_col: bool = True
    end_offset_days: int = 0  # target_end_date = round date + offset + 7 * horizon
    last_data_horizon: int = -1  # the newest week a round's data covers, as a horizon

    def target_end_date(self, ref: pd.Timestamp, horizon: int) -> pd.Timestamp:
        return ref + pd.Timedelta(days=self.end_offset_days) + horizon * WEEK


def regular_context(history: pd.DataFrame) -> pd.DataFrame:
    """history[target, location, date, value] -> Chronos frame; missing weeks left as NaN."""
    history = history.dropna(subset=["value"])
    # some archives stamp a few weeks on another weekday; snap every date to the modal weekday
    modal = history.date.dt.day_of_week.mode()[0]
    shift = (modal - history.date.dt.day_of_week + 3) % 7 - 3
    history = history.assign(date=history.date + pd.to_timedelta(shift, unit="D"))
    frames = []
    for (target, location), g in history.groupby(["target", "location"]):
        s = g.set_index("date").value.sort_index()
        s = s[~s.index.duplicated(keep="last")]
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="7D"))
        frames.append(pd.DataFrame({"item_id": f"{target}|{location}", "timestamp": s.index, "target": s.to_numpy()}))
    return pd.concat(frames)


_PIPE = None


def _pipeline():
    global _PIPE
    if _PIPE is None:
        from chronos import Chronos2Pipeline
        _PIPE = Chronos2Pipeline.from_pretrained(MODEL_ID, revision=MODEL_REVISION, device_map="cpu")
    return _PIPE


def chronos2_quantiles(history: pd.DataFrame, end_dates: list[pd.Timestamp], quantiles) -> pd.DataFrame:
    """Quantiles for each series at each wanted week-ending date, whatever its last observed week.

    Returns [target, location, date, q, value]; series with no data are dropped.
    """
    context = regular_context(history)
    last = context.groupby("item_id").timestamp.max()
    steps = int(((max(end_dates) - last.min()) / WEEK))
    if steps < 1:
        raise ValueError(f"[chronos2] nothing to forecast: last data {last.min().date()} >= {max(end_dates).date()}")
    pred = _pipeline().predict_df(context, target="target", prediction_length=steps, quantile_levels=list(quantiles),
                                  cross_learning=True, batch_size=100)
    qcols = [c for c in pred.columns if _is_quantile(c)]
    long = pred.melt(id_vars=["item_id", "timestamp"], value_vars=qcols, var_name="q", value_name="value")
    long["q"] = long.q.astype(float)
    long["date"] = pd.to_datetime(long.pop("timestamp"))
    long = long[long.date.isin(end_dates)]
    long[["target", "location"]] = long.pop("item_id").str.split("|", expand=True)
    return long[["target", "location", "date", "q", "value"]]


def _is_quantile(col) -> bool:
    try:
        return 0 < float(col) < 1
    except (TypeError, ValueError):
        return False


def tidy_quantiles(fc: pd.DataFrame, spec: HubSpec) -> pd.DataFrame:
    """Clip to the hub's bounds, round counts, and re-sort so quantiles never cross."""
    fc = fc.copy()
    fc["value"] = fc.value.clip(lower=0, upper=spec.value_max)
    if spec.integer_values:
        fc["value"] = fc.value.round()
    fc = fc.sort_values(["target", "location", "date", "q"])
    fc["value"] = fc.groupby(["target", "location", "date"]).value.transform(np.sort)
    return fc


def quantiles_to_trajectories(fc: pd.DataFrame, n: int) -> pd.DataFrame:
    """Comonotone trajectories: sample i takes the same quantile level u_i at every horizon.

    The quantile function is interpolated linearly between the forecast levels and held flat beyond them.
    """
    u = (np.arange(n) + 0.5) / n
    rows = []
    for (target, location, date), g in fc.groupby(["target", "location", "date"]):
        g = g.sort_values("q")
        values = np.interp(u, g.q.to_numpy(), g.value.to_numpy())
        rows.append(pd.DataFrame({"target": target, "location": location, "date": date,
                                  "sample": np.arange(n), "value": values}))
    return pd.concat(rows)


def to_hub_rows(fc: pd.DataFrame, spec: HubSpec, ref: pd.Timestamp) -> pd.DataFrame:
    """fc[target, location, date, q, value] -> hubverse model-output rows."""
    horizon_of = {spec.target_end_date(ref, h): h for h in spec.horizons}
    fc = tidy_quantiles(fc[fc.date.isin(horizon_of)], spec)
    if spec.output_type == "sample":
        out = quantiles_to_trajectories(fc, spec.n_samples)
        # one id per (location, target) trajectory, shared across horizons (compound task id)
        key = out.location + "|" + out.target
        out["output_type_id"] = (key.map({k: i for i, k in enumerate(sorted(key.unique()))}) * spec.n_samples
                                 + out["sample"]).astype(str)
        out["output_type"] = "sample"
    else:
        out = fc.assign(output_type="quantile", output_type_id=fc.q.map(lambda q: f"{q:g}"))
    out[spec.round_col] = ref.strftime("%Y-%m-%d")
    out["target_end_date"] = out.date.dt.strftime("%Y-%m-%d")
    cols = [spec.round_col, "target", "horizon", "location", "target_end_date", "output_type", "output_type_id", "value"]
    if spec.has_horizon_col:
        out["horizon"] = out.date.map(horizon_of).astype(int)
    else:
        cols.remove("horizon")
    if spec.integer_values:
        out["value"] = out.value.round().astype(int)
    return out[cols].sort_values(cols[:7]).reset_index(drop=True)


def check_against_tasks(rows: pd.DataFrame, spec: HubSpec, tasks_path: Path) -> list[str]:
    """Format rules from the hub's tasks.json that a bad run could break. Empty list = valid."""
    rounds = json.loads(tasks_path.read_text())["rounds"]
    tasks = [mt for r in rounds for mt in r["model_tasks"]
             if set(mt["task_ids"]["target"].get("optional") or mt["task_ids"]["target"].get("required") or []) & set(spec.targets)]
    problems = []
    ref = rows[spec.round_col].iloc[0]
    for target, g in rows.groupby("target"):
        mts = [mt for mt in tasks if target in (mt["task_ids"]["target"].get("optional") or []) + (mt["task_ids"]["target"].get("required") or [])]
        if not mts:
            problems.append(f"target {target} not in tasks.json"); continue
        ids, out = mts[-1]["task_ids"], mts[-1]["output_type"]
        allowed = lambda k: set(map(str, (ids[k].get("optional") or []) + (ids[k].get("required") or [])))
        if ref not in allowed(spec.round_col):
            problems.append(f"{spec.round_col} {ref} is not a round")
        for col in ["location", "target_end_date"] + (["horizon"] if spec.has_horizon_col else []):
            bad = set(g[col].astype(str)) - allowed(col)
            if bad:
                problems.append(f"{target}: {col} not allowed: {sorted(bad)[:5]}")
        if spec.output_type not in out:
            problems.append(f"{target}: output type {spec.output_type} not accepted")
        elif spec.output_type == "quantile":
            want = set(float(x) for x in out["quantile"]["output_type_id"].get("required") or [])
            got = g.groupby(["location", "horizon"]).output_type_id.apply(lambda s: set(s.astype(float)))
            if want and (got.map(lambda s: s != want)).any():
                problems.append(f"{target}: quantile set differs from required {sorted(want)}")
        else:
            n = g.groupby(["location", "target_end_date"]).size()
            p = out["sample"]["output_type_id_params"]
            if not n.between(p["min_samples_per_task"], p["max_samples_per_task"]).all():
                problems.append(f"{target}: samples per task outside [{p['min_samples_per_task']}, {p['max_samples_per_task']}]")
        vmax = out.get(spec.output_type, {}).get("value", {}).get("maximum")
        if g.value.isna().any() or (g.value < 0).any() or (vmax is not None and (g.value > vmax).any()):
            problems.append(f"{target}: missing, negative or above-maximum values")
    if spec.output_type == "quantile":
        q = rows.assign(q=rows.output_type_id.astype(float)).sort_values("q")
        if (q.groupby(["target", "location", "target_end_date"]).value.diff() < 0).any():
            problems.append("quantiles cross")
    return problems


def write_rows(rows: pd.DataFrame, spec: HubSpec, ref: pd.Timestamp, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ref.strftime('%Y-%m-%d')}-{spec.model}.{spec.file_ext}"
    if spec.file_ext == "parquet":
        rows.to_parquet(path, index=False)
    else:
        rows.to_csv(path, index=False)
    return path
