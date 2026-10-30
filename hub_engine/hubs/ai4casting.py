"""AI4Casting Hub (U. Guelph + Public Health Ontario): two hubverse repos, same round calendar.

hosp: Ontario hospital bed occupancy (flu, RSV, COVID) for Ontario + 6 Ontario Health regions.
rvdss: % of lab tests positive (flu, RSV, SARS-CoV-2), Canada + regions + provinces (PHAC RVDSS).
Rounds: Saturday reference_date, due by the end of that Saturday (America/Toronto); data to R-7 lands Fri.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from hub_engine import vintages
from hub_engine.core import HubSpec

QUANTILES = (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
COMMON = dict(quantiles=QUANTILES, horizons=(-1, 0, 1, 2, 3), ensemble="AI4Casting_Hub-Ensemble_v1",
              baseline="AI4Casting_Hub-Quantile_Baseline", model="pedro-Chronos2")

HOSP = HubSpec(name="ai4c-hosp", repo="ai4castinghub/hospitalization-forecast",
               targets=("wk inc flu hosp", "wk inc rsv hosp", "wk inc covid hosp"), integer_values=True, **COMMON)
RVDSS = HubSpec(name="ai4c-rvdss", repo="ai4castinghub/rvdss-forecast",
                targets=("pct wk flu lab det", "pct wk rsv lab det", "pct wk covid lab det"), value_max=100.0, **COMMON)

_FILE = {"ai4c-hosp": "hospitalization-data.csv", "ai4c-rvdss": "target_rvdss_data.csv"}
_COLUMNS = {
    "ai4c-hosp": ("time", {"flu": "wk inc flu hosp", "rsv": "wk inc rsv hosp", "covid": "wk inc covid hosp"}),
    "ai4c-rvdss": ("time_value", {"flu_pct_positive": "pct wk flu lab det", "rsv_pct_positive": "pct wk rsv lab det",
                                  "sarscov2_pct_positive": "pct wk covid lab det"}),
}


def deadline(ref: pd.Timestamp) -> pd.Timestamp:
    """End of the reference Saturday in Toronto, as naive UTC."""
    return (ref + pd.Timedelta(days=1)).tz_localize("America/Toronto").tz_convert("UTC").tz_localize(None)


def _long(raw: pd.DataFrame, spec: HubSpec) -> pd.DataFrame:
    date_col, targets = _COLUMNS[spec.name]
    cols = [c for c in targets if c in raw.columns]
    out = raw.melt(id_vars=[date_col, "geo_value"], value_vars=cols, var_name="col", value_name="value")
    out["target"] = out.pop("col").map(targets)
    out["date"] = pd.to_datetime(out.pop(date_col))
    return out.rename(columns={"geo_value": "location"})[["target", "location", "date", "value"]]


def _archive(hub: Path, spec: HubSpec, before_season: str | None) -> pd.DataFrame:
    """Settled past seasons from auxiliary-data/target-data-archive (season dirs sort by name)."""
    seasons = sorted((hub / "auxiliary-data" / "target-data-archive").glob("season_*"))
    if before_season:
        seasons = [s for s in seasons if s.name < before_season]
    frames = [_long(pd.read_csv(s / _FILE[spec.name]), spec) for s in seasons if (s / _FILE[spec.name]).exists()]
    return pd.concat(frames)


def history(hub: Path, spec: HubSpec, ref: pd.Timestamp, backtest: bool = False) -> pd.DataFrame:
    """Data a forecaster had at the round's deadline.

    Backtest: settled seasons before 2025-26 + the newest 2025-26 file committed before the deadline
    (git-history vintages fetched by hub_engine.vintages into data/vintages/<hub name>).
    Live: archive + every file under target-data/ (the current season).
    """
    if backtest:
        vintage_dir = hub.parent.parent / "vintages" / spec.name
        current = vintages.as_of(vintage_dir, deadline(ref))
        if current is None:
            raise FileNotFoundError(f"[{spec.name}] no vintage before {deadline(ref)}")
        h = pd.concat([_archive(hub, spec, "season_2025_2026"), _long(pd.read_csv(current), spec)])
    else:
        live = [_long(pd.read_csv(f), spec) for f in sorted((hub / "target-data").glob(f"*/{_FILE[spec.name]}"))]
        h = pd.concat([_archive(hub, spec, None)] + live)
    h = h[h.date <= ref - pd.Timedelta(days=7)]
    return h.drop_duplicates(["target", "location", "date"], keep="last")


def latest_truth(hub: Path, spec: HubSpec) -> pd.DataFrame:
    return history(hub, spec, pd.Timestamp("2100-01-01"))


SPECS = {s.name: s for s in (HOSP, RVDSS)}
