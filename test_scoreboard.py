"""python -m unittest test_scoreboard"""
import unittest

import numpy as np
import pandas as pd

from scoreboard import build_scores, rank_teams, ratio, wis

LEVELS = np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
QUANTILES = np.array([[1.0, 2, 3, 4, 5, 6, 7]])


class WisTest(unittest.TestCase):
    def test_should_score_interval_widths_only_when_truth_is_the_median(self):
        # (0.025*6 + 0.1*4 + 0.25*2 + 0) / 3.5
        self.assertAlmostEqual(wis(QUANTILES, LEVELS, np.array([4.0]))[0], 0.3)

    def test_should_add_interval_and_median_penalties_when_truth_is_above_every_quantile(self):
        # (0.025*126 + 0.1*44 + 0.25*22 + 0.5*6) / 3.5, interval scores IS = width + 2/alpha * miss
        self.assertAlmostEqual(wis(QUANTILES, LEVELS, np.array([10.0]))[0], 16.05 / 3.5)

    def test_should_be_absolute_error_when_all_quantiles_agree(self):
        self.assertAlmostEqual(wis(np.full((1, 7), 2.0), LEVELS, np.array([5.0]))[0], 3.0)


class RatioTest(unittest.TestCase):
    def test_should_take_geometric_mean_of_per_series_mean_ratios(self):
        index = pd.MultiIndex.from_tuples([("h", "t", "a", 0), ("h", "t", "a", 1), ("h", "t", "b", 0)],
                                          names=["hub", "target", "location", "horizon"])
        pair = pd.DataFrame({"us": [1.0, 3.0, 1.0], "them": [4.0, 4.0, 0.25]}, index=index)
        self.assertAlmostEqual(ratio(pair, "us", "them"), np.sqrt(0.5 * 4))

    def test_should_report_pending_when_no_cell_has_truth(self):
        index = pd.MultiIndex.from_tuples([("h", "2026-11-14", "t", 0, "a", "2026-11-14")],
                                          names=["hub", "reference_date", "target", "horizon", "location", "target_end_date"])
        cells = pd.DataFrame({"us": [np.nan], "ens": [np.nan], "base": [np.nan]}, index=index)
        scores = build_scores(cells, "us", "ens", "base", "2026-11-15")
        self.assertEqual(scores["headline"], "Live scores pending: no truth yet for our rounds")
        self.assertEqual(scores["rows"], [])


def cells_frame(columns: dict) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    index = pd.MultiIndex.from_tuples([("h", "2026-11-07", "t", 0, f"l{i}", "2026-11-07") for i in range(n)],
                                      names=["hub", "reference_date", "target", "horizon", "location", "target_end_date"])
    return pd.DataFrame(columns, index=index)


class TeamsTest(unittest.TestCase):
    def test_should_lead_with_rank_among_teams_when_teams_are_scored(self):
        cells = cells_frame({"us": [1.0] * 4, "ens": [2.0] * 4, "base": [2.0] * 4,
                             "good": [0.5] * 4, "bad": [3.0] * 4, "worse": [4.0] * 4})
        scores = build_scores(cells, "us", "ens", "base", "2026-11-15", teams=["good", "bad", "worse"])
        self.assertEqual(scores["headline"], "Better than 2 of 3 teams; 50% less error than the hub ensemble")
        self.assertEqual([r["measure"] for r in scores["rows"]][:2], ["Us vs hub ensemble", "Us vs baseline"])

    def test_should_say_we_are_the_only_team_when_no_team_is_scored(self):
        cells = cells_frame({"us": [1.0] * 4, "ens": [2.0] * 4, "base": [2.0] * 4})
        scores = build_scores(cells, "us", "ens", "base", "2026-11-15", teams=["idle"])
        self.assertEqual(scores["headline"], "Only outside team so far; 50% less error than the hub ensemble")

    def test_should_leave_out_teams_with_little_overlap_when_ranking(self):
        cells = cells_frame({"us": [1.0] * 4, "ens": [2.0] * 4, "base": [2.0] * 4,
                             "wide": [2.0] * 4, "narrow": [9.0, np.nan, np.nan, np.nan]})
        ranked = rank_teams(cells, "us", ["wide", "narrow"])
        self.assertEqual([r["team"] for r in ranked], ["wide"])

    def test_should_cap_card_at_six_rows_when_many_teams(self):
        cells = cells_frame({"us": [1.0] * 4, "ens": [2.0] * 4, "base": [2.0] * 4,
                             **{f"t{i}": [2.0] * 4 for i in range(8)}})
        scores = build_scores(cells, "us", "ens", "base", "2026-11-15", teams=[f"t{i}" for i in range(8)])
        self.assertEqual(len(scores["rows"]), 6)


if __name__ == "__main__":
    unittest.main()
