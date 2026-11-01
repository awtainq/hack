# -*- coding: utf-8 -*-
"""
tests/test_v85_clean_model.py -- Unit tests for V85 Clean Multi-Stream Admissions Model.
========================================================================================

Verifies:
1. No old forecast CSV files (forecast_2024_*.csv) are accessed or read.
2. Only raw sources (registered.csv, included_ks.csv) are used.
3. Forecast date support covers exactly 2024-06-20 to 2024-08-31 (73 days).
4. Exactly 13 days are strictly 0.0 (10 Sundays + 3 closed Saturdays).
5. All 60 active forecast days are strictly positive and finite.
6. Total forecast volume strictly normalizes to 11,400.0.
7. Degree stream components partition accurately.
"""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import v85_clean_model as model


class TestV85CleanModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg_path = ROOT / "processed_data" / "registered.csv"
        cls.ks_path = ROOT / "processed_data" / "included_ks.csv"
        assert cls.reg_path.exists(), "Missing registered.csv"
        assert cls.ks_path.exists(), "Missing included_ks.csv"

    def test_only_raw_admissions_data_is_read(self):
        """Ensures NO old forecast CSV files are opened or read during model run."""
        opened_files = []
        original_read_csv = pd.read_csv

        def tracked_read_csv(filepath_or_buffer, *args, **kwargs):
            if isinstance(filepath_or_buffer, (str, Path)):
                name = Path(filepath_or_buffer).name
                opened_files.append(name)
            return original_read_csv(filepath_or_buffer, *args, **kwargs)

        with patch("pandas.read_csv", side_effect=tracked_read_csv):
            merged = model.load_raw_admissions_data(self.reg_path, self.ks_path)

        self.assertIn("registered.csv", opened_files)
        self.assertIn("included_ks.csv", opened_files)
        for fn in opened_files:
            self.assertFalse(fn.startswith("forecast_2024"), f"Unauthorized file opened: {fn}")
            self.assertFalse("v84" in fn, f"Unauthorized file opened: {fn}")

    def test_forecast_dimensions_and_volume(self):
        """Verifies 73 dates, 13 zeros, 60 active days, and 11,400 total volume."""
        forecast_path = ROOT / "forecast_2024_v85_clean.csv"
        self.assertTrue(forecast_path.exists(), "Missing forecast_2024_v85_clean.csv")

        df = pd.read_csv(forecast_path)
        self.assertEqual(len(df), 73, f"Expected 73 dates, got {len(df)}")
        self.assertEqual(list(df.columns), ["date", "forecast"])

        # Check date continuity
        expected_dates = pd.date_range("2024-06-20", "2024-08-31").strftime("%Y-%m-%d").tolist()
        self.assertEqual(df["date"].tolist(), expected_dates)

        # Check zero structure: 10 Sundays + 3 closed Saturdays = 13 zeros
        zeros_mask = df["forecast"] == 0.0
        self.assertEqual(zeros_mask.sum(), 13, f"Expected 13 zeros, got {zeros_mask.sum()}")

        dts = pd.to_datetime(df["date"])
        sundays_mask = dts.dt.dayofweek == 6
        self.assertEqual(sundays_mask.sum(), 10)
        self.assertTrue(np.all(df.loc[sundays_mask, "forecast"] == 0.0), "All Sundays must be 0.0")

        for closed_sat in model.CLOSED_SATURDAYS_2024:
            sat_val = df.loc[df["date"] == closed_sat, "forecast"].iloc[0]
            self.assertEqual(sat_val, 0.0, f"Saturday {closed_sat} must be 0.0, got {sat_val}")

        # Check active days
        active_mask = ~zeros_mask
        self.assertEqual(active_mask.sum(), 60)
        self.assertTrue(np.all(df.loc[active_mask, "forecast"] > 0.0), "Active days must be positive")
        self.assertTrue(np.all(np.isfinite(df["forecast"])), "All values must be finite")

        # Total volume check
        total_vol = df["forecast"].sum()
        self.assertAlmostEqual(total_vol, 11400.0, places=2, msg=f"Volume {total_vol} != 11400.0")

    def test_degree_streams_partition(self):
        """Verifies that degree stream partition covers all registered applications."""
        merged = model.load_raw_admissions_data(self.reg_path, self.ks_path)
        self.assertTrue(set(merged["deg_stream"].unique()).issubset({"bak", "mag", "spo", "other"}))
        self.assertTrue(set(merged["fund_stream"].unique()).issubset({"budget", "contract"}))


if __name__ == "__main__":
    unittest.main()

