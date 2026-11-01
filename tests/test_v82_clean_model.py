# -*- coding: utf-8 -*-
"""Unit tests for V82 Clean Model."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from v82_clean_model import run_clean_model


class TestV82CleanModel(unittest.TestCase):
    def test_mode_1d_execution(self):
        df, meta = run_clean_model(mode="1d")
        self.assertEqual(len(df), 73)
        self.assertListEqual(list(df.columns), ["date", "forecast"])
        self.assertTrue(np.all(np.isfinite(df["forecast"].values)))
        self.assertTrue(np.all(df["forecast"].values >= 0.0))

        # Check strict 13 zeros
        zero_count = int((df["forecast"] == 0.0).sum())
        self.assertEqual(zero_count, 13)
        self.assertEqual(meta["zeros_count"], 13)

        # Check total sum around 12,031
        self.assertAlmostEqual(meta["total_sum"], 12031.22, delta=5.0)

        # Check verified RMSLE
        self.assertEqual(meta["measured_rmsle"], 0.40546)

    def test_mode_2d_execution(self):
        df, meta = run_clean_model(mode="2d")
        self.assertEqual(len(df), 73)
        self.assertTrue(np.all(np.isfinite(df["forecast"].values)))
        self.assertTrue(np.all(df["forecast"].values >= 0.0))

        # Check strict 13 zeros
        zero_count = int((df["forecast"] == 0.0).sum())
        self.assertEqual(zero_count, 13)
        self.assertEqual(meta["zeros_count"], 13)

        # Check sub-0.40 projected score
        self.assertLess(meta["projected_rmsle"], 0.40000)
        self.assertGreater(meta["orthogonality"], 0.80)


if __name__ == "__main__":
    unittest.main()

