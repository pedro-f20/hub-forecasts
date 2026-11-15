"""python -m unittest test_scoreboard"""
import unittest

import numpy as np
import pandas as pd

from scoreboard import build_scores, ratio, wis

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


if __name__ == "__main__":
    unittest.main()
