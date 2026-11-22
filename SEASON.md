# 2026-27 season scoreboard (AI4Casting)

Interim. WIS from the 7 quantiles against the latest truth, cells both models forecast. Ratio = pedro-Chronos2 WIS / comparator WIS, geometric mean over (hub, target, location) of mean WIS; below 1 means pedro-Chronos2 is better.

| hub | comparator | ratio | cells | rounds |
|---|---|---|---|---|
| hospital beds | AI4Casting_Hub-Ensemble_v1 | 0.61 | 126 | 3 |
| hospital beds | AI4Casting_Hub-Quantile_Baseline | 0.77 | 126 | 3 |
| lab positivity | AI4Casting_Hub-Ensemble_v1 | 0.67 | 309 | 3 |
| lab positivity | AI4Casting_Hub-Quantile_Baseline | 0.60 | 303 | 3 |
| overall | AI4Casting_Hub-Ensemble_v1 | 0.65 | 435 | 3 |
| overall | AI4Casting_Hub-Quantile_Baseline | 0.65 | 429 | 3 |

By round, vs AI4Casting_Hub-Ensemble_v1:

| round | hub | ratio | cells |
|---|---|---|---|
| 2026-10-31 | hospital beds | 0.54 | 63 |
| 2026-10-31 | lab positivity | 0.64 | 153 |
| 2026-11-07 | hospital beds | 0.67 | 42 |
| 2026-11-07 | lab positivity | 0.59 | 102 |
| 2026-11-14 | hospital beds | 0.69 | 21 |
| 2026-11-14 | lab positivity | 1.00 | 54 |

Vs every outside team, both hubs (ranked: shares at least half the widest overlap):

| team | ratio | cells | ranked |
|---|---|---|---|
| JHU-DMD | - | 0 | no |
| JHU-DMD_memory | - | 0 | no |
| JHUAPL-DMD | - | 0 | no |
| JHUAPL-DMD_memory | - | 0 | no |
| UBCStat-ARX_Ensemble | - | 0 | no |
| UBCStat-Climate | - | 0 | no |
| UGuelph-CompositeCurve | - | 0 | no |
| UGuelph_AIDM-AdaptEns | - | 0 | no |
| VTSanghani-PRIME | - | 0 | no |
