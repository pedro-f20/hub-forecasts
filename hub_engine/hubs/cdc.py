"""CDC hubverse hubs (FluSight; COVID-19 and RSV hubs share the layout): Saturday reference_date,
due Wednesday 11 PM ET (R - 3). NHSN data released that Wednesday run to R - 7.

Vintages: target-data/time-series.csv, one full copy of the history per Wednesday `as_of`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from hub_engine.core import HubSpec

QUANTILES = (0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99)

FLUSIGHT = HubSpec(name="flusight", repo="cdcepi/FluSight-forecast-hub", model="pedro-Chronos2",
                   targets=("wk inc flu hosp",), quantiles=QUANTILES, horizons=(-1, 0, 1, 2, 3),
                   ensemble="FluSight-ensemble", baseline="FluSight-baseline", integer_values=True)
SPECS = {FLUSIGHT.name: FLUSIGHT}

_CACHE: dict[Path, pd.DataFrame] = {}


def _series(hub: Path) -> pd.DataFrame:
    if hub not in _CACHE:
        ts = pd.read_csv(hub / "target-data" / "time-series.csv", dtype={"location": str},
                         usecols=["as_of", "target", "target_end_date", "location", "observation"])
        _CACHE[hub] = ts.assign(as_of=pd.to_datetime(ts.as_of), date=pd.to_datetime(ts.pop("target_end_date")),
                                value=ts.pop("observation").astype(float))
    return _CACHE[hub]


def history(hub: Path, spec: HubSpec, ref: pd.Timestamp, backtest: bool = False) -> pd.DataFrame:
    """Newest as_of on or before the Wednesday deadline. Same rule live and in backtests."""
    ts = _series(hub)
    ts = ts[ts.target.isin(spec.targets)]
    seen = ts.as_of[ts.as_of <= ref - pd.Timedelta(days=3)]
    if seen.empty:
        raise FileNotFoundError(f"[{spec.name}] no as_of on or before {(ref - pd.Timedelta(days=3)).date()}")
    h = ts[ts.as_of == seen.max()]
    return h[h.date <= ref - pd.Timedelta(days=7)][["target", "location", "date", "value"]]


def latest_truth(hub: Path, spec: HubSpec) -> pd.DataFrame:
    ts = _series(hub)
    ts = ts[ts.target.isin(spec.targets)]
    return ts[ts.as_of == ts.as_of.max()][["target", "location", "date", "value"]]
