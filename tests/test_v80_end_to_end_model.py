# -*- coding: utf-8 -*-
"""Unit tests for V80 End-to-End Autonomous Model."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from v80_end_to_end_model import run_v80_model


class TestV80Model(unittest.TestCase):
    def test_v80_execution_and_integrity(self):
        df_tuned, meta_tuned = run_v80_model(saturday_mode="tuned")
        df_strict, meta_strict = run_v80_model(saturday_mode="strict")

        # Shape
        self.assertEqual(len(df_tuned), 73)
        self.assertEqual(len(df_strict), 73)

        # Non-negative, finite
        self.assertTrue(np.all(np.isfinite(df_tuned["forecast"].values)))
        self.assertTrue(np.all(np.isfinite(df_strict["forecast"].values)))
        self.assertTrue(np.all(df_tuned["forecast"].values >= 0.0))
        self.assertTrue(np.all(df_strict["forecast"].values >= 0.0))

        # Check Sundays are strictly 0.0
        dows = pd.to_datetime(df_tuned["date"]).dt.dayofweek
        sundays = (dows == 6)
        self.assertEqual(int(sundays.sum()), 10)
        self.assertTrue(np.all(df_tuned.loc[sundays, "forecast"] == 0.0))
        self.assertTrue(np.all(df_strict.loc[sundays, "forecast"] == 0.0))

        # Check zeros count
        self.assertEqual(meta_tuned["zeros_count"], 10)
        self.assertEqual(meta_strict["zeros_count"], 13)

        # Check projected score improvements
        self.assertLess(meta_tuned["projected_rmsle"], 0.40931)
        self.assertLess(meta_strict["projected_rmsle"], 0.40931)


if __name__ == "__main__":
    unittest.main()

