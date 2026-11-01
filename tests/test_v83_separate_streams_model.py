# -*- coding: utf-8 -*-
"""Unit tests for V83 Separate Streams Model."""
import inspect
import re
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import v83_separate_streams_model as v83


class TestV83SeparateStreamsModel(unittest.TestCase):
    def test_raw_data_and_stream_categorization(self):
        """Verifies raw data loading and stream categorization without any old forecast files."""
        merged = v83.load_and_preprocess_raw_data(v83.REG_FILE, v83.KS_FILE)
        
        # Check streams
        self.assertIn("deg_stream", merged.columns)
        self.assertIn("fund_stream", merged.columns)
        
        deg_types = set(merged["deg_stream"].unique())
        self.assertEqual(deg_types, {"bak", "mag", "spo", "other"})
        
        fund_types = set(merged["fund_stream"].unique())
        self.assertEqual(fund_types, {"budget", "contract"})
        
        # Non-empty
        self.assertGreater(len(merged), 100000)

    def test_feature_matrix_and_milestones(self):
        """Verifies feature engineering for bachelor, master, and calendar milestones."""
        merged = v83.load_and_preprocess_raw_data(v83.REG_FILE, v83.KS_FILE)
        df, feat_cols = v83.build_daily_feature_matrix(merged, start_year=2021, end_year=2024)
        
        # Check essential domain features
        essential_features = [
            "dist_bak_end", "is_bak_deadline", "after_bak_end", "bak_decay",
            "dist_mag_end", "is_mag_deadline", "after_mag_end", "is_mag_window", "is_mag_rush_sat",
            "dist_spo_end", "after_spo_end",
            "dist_consent", "dist_orders", "dist_contract_sat", "dist_contract_end",
            "progress", "sin_prog", "cos_prog", "sin_dow", "cos_dow",
        ]
        for feat in essential_features:
            self.assertIn(feat, feat_cols, f"Missing essential feature: {feat}")
            self.assertIn(feat, df.columns)

    def test_no_old_forecast_files_or_hardcoded_date_dict(self):
        """Strictly verifies that no old forecast CSVs are loaded and no hardcoded date dictionary is used."""
        source = inspect.getsource(v83)
        
        # No reading of old forecast CSVs
        forbidden_patterns = [
            r"read_csv\(.*forecast_2024_v",
            r"read_csv\(.*best_verified",
            r"read_csv\(.*forecast_2024_clean",
        ]
        for pat in forbidden_patterns:
            self.assertIsNone(
                re.search(pat, source),
                f"Source code must NOT read old forecast CSV files (matched {pat})"
            )
            
        # No hardcoded date delta dictionaries
        self.assertNotIn('"2024-06-20": +0.0812', source)
        self.assertNotIn('"2024-07-29": +0.3340', source)
        self.assertNotIn('DATE_DELTAS = {', source)

    def test_v83_forecast_integrity(self):
        """Verifies full execution and output shape, integrity, volume, and schedule."""
        df = pd.read_csv(v83.OUT_DEFAULT)
        
        # Shape
        self.assertEqual(len(df), 73)
        self.assertEqual(list(df.columns), ["date", "forecast"])
        
        # Date sequence
        expected_dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-06-20", "2024-08-31")]
        self.assertEqual(list(df["date"]), expected_dates)
        
        # Finite and non-negative
        self.assertTrue(np.all(np.isfinite(df["forecast"].values)))
        self.assertTrue(np.all(df["forecast"].values >= 0.0))
        
        # Strict 13 zeros
        zeros_mask = (df["forecast"] == 0.0)
        self.assertEqual(int(zeros_mask.sum()), 13)
        
        # All 10 Sundays must be zero
        dows = pd.to_datetime(df["date"]).dt.dayofweek
        self.assertEqual(int((df.loc[dows == 6, "forecast"] == 0.0).sum()), 10)
        
        # 3 designated closed Saturdays must be zero
        closed_sats = ["2024-06-22", "2024-08-24", "2024-08-31"]
        for cs in closed_sats:
            val = float(df.loc[df["date"] == cs, "forecast"].iloc[0])
            self.assertEqual(val, 0.0, f"Saturday {cs} must be strictly 0.0")
            
        # Bachelor deadline day (July 25) must be high
        bak_deadline_val = float(df.loc[df["date"] == "2024-07-25", "forecast"].iloc[0])
        self.assertGreater(bak_deadline_val, 250.0)
        
        # Master's deadline day (July 29) must be high
        mag_deadline_val = float(df.loc[df["date"] == "2024-07-29", "forecast"].iloc[0])
        self.assertGreater(mag_deadline_val, 100.0)
        
        # Total volume calibration
        total_volume = float(df["forecast"].sum())
        self.assertGreater(total_volume, 11800.0)
        self.assertLess(total_volume, 12300.0)


if __name__ == "__main__":
    unittest.main()

