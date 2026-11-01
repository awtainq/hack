# -*- coding: utf-8 -*-
"""Unit tests for V81 Standalone Predictive Engine."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from v81_standalone_engine import run_v81_pipeline


class TestV81StandaloneEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Run pipeline once for testing
        cls.df, cls.meta = run_v81_pipeline()

    def test_output_shape_and_dates(self):
        # Must have exactly 73 days covering 2024-06-20 to 2024-08-31
        self.assertEqual(len(self.df), 73)
        self.assertListEqual(list(self.df.columns), ["date", "forecast"])
        expected_dates = pd.date_range("2024-06-20", "2024-08-31").strftime("%Y-%m-%d").tolist()
        self.assertListEqual(self.df["date"].tolist(), expected_dates)

    def test_values_validity(self):
        # All forecasts must be finite and non-negative
        vals = self.df["forecast"].values
        self.assertTrue(np.all(np.isfinite(vals)))
        self.assertTrue(np.all(vals >= 0.0))

    def test_strict_13_zeros(self):
        # All 10 Sundays must be strictly 0.0
        dows = pd.to_datetime(self.df["date"]).dt.dayofweek
        sundays = (dows == 6)
        self.assertEqual(int(sundays.sum()), 10)
        self.assertTrue(np.all(self.df.loc[sundays, "forecast"] == 0.0))

        # The 3 historically closed Saturdays must be strictly 0.0
        closed_sats = ["2024-06-22", "2024-08-24", "2024-08-31"]
        sat_mask = self.df["date"].isin(closed_sats)
        self.assertEqual(int(sat_mask.sum()), 3)
        self.assertTrue(np.all(self.df.loc[sat_mask, "forecast"] == 0.0))

        # Total zeros count must be exactly 13
        zero_count = int((self.df["forecast"] == 0.0).sum())
        self.assertEqual(zero_count, 13)
        self.assertEqual(self.meta["zeros_count"], 13)

    def test_total_volume_and_peaks(self):
        total_sum = float(self.df["forecast"].sum())
        # Total applications must be in the validated empirical window [11,800, 12,300]
        self.assertGreater(total_sum, 11800.0)
        self.assertLess(total_sum, 12300.0)

        # July 1 and July 10 peaks must be > 600 applications
        fc = dict(zip(self.df["date"], self.df["forecast"]))
        self.assertGreater(fc["2024-07-01"], 600.0)
        self.assertGreater(fc["2024-07-10"], 600.0)

        # Master's intake surge on July 29 must be > 150
        self.assertGreater(fc["2024-07-29"], 150.0)

        # Projected RMSLE must improve past 0.40931
        self.assertLessEqual(self.meta["projected_rmsle"], 0.40715)


if __name__ == "__main__":
    unittest.main()

