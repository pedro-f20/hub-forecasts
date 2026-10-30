"""RespiCast hubs (ECDC, european-modelling-hubs): Wednesday origin_date, ERVISS snapshots published Fridays.

target_end_date = origin - 10 days + 7 * horizon (week ending Sunday). Late-reporting countries make
horizons -1 and 0 real forecasts for some locations.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from hub_engine.core import HubSpec

QUANTILES = (0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99)

COVID = HubSpec(name="respicast-covid", repo="european-modelling-hubs/RespiCast-Covid19", model="pedro-Chronos2",
                targets=("hospital admissions",), quantiles=QUANTILES, horizons=(-1, 0, 1, 2, 3, 4),
                ensemble="respicast-hubEnsemble", baseline="respicast-quantileBaseline",
                round_col="origin_date", end_offset_days=-10, last_data_horizon=0)
SPECS = {COVID.name: COVID}
MAX_WEEKS_STALE = 3  # skip a country whose last week is > 3 weeks before the horizon-0 week


def _read(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, usecols=["target", "location", "truth_date", "value"])
    return x.assign(date=pd.to_datetime(x.pop("truth_date")), value=x.value.astype(float))


def history(hub: Path, spec: HubSpec, ref: pd.Timestamp, backtest: bool = False) -> pd.DataFrame:
    """Newest snapshot published before the origin (the deadline is the origin day, 23:59 CET).

    The same rule serves live and backtest: snapshots are real vintages.
    """
    snaps = sorted((hub / "target-data" / "ERVISS" / "snapshots").glob("*_admissions.csv"))
    seen = [p for p in snaps if pd.Timestamp(p.name[:10]) <= ref]
    if not seen:
        raise FileNotFoundError(f"[{spec.name}] no snapshot on or before {ref.date()}")
    h = _read(seen[-1])
    h = h[h.target.isin(spec.targets) & (h.date <= spec.target_end_date(ref, 0))]
    last = h.groupby("location").date.transform("max")
    return h[last >= spec.target_end_date(ref, 0) - pd.Timedelta(weeks=MAX_WEEKS_STALE)]


def latest_truth(hub: Path, spec: HubSpec) -> pd.DataFrame:
    return _read(hub / "target-data" / "latest-hospital_admissions.csv")
