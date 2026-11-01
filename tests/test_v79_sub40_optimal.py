# -*- coding: utf-8 -*-
"""Unit tests for V79 sub-0.40 optimization reproducer."""
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
from reproducers import v79_sub40_optimal as model


class TestV79Sub40(unittest.TestCase):
    def test_forecast_generation_and_properties(self):
        f_strict, f_sat, s_strict, s_sat = model.build_v79()

        # Output dimensions
        self.assertEqual(len(f_strict), 73)
        self.assertEqual(len(f_sat), 73)

        # Non-negative and finite
        self.assertTrue(np.all(np.isfinite(f_strict)))
        self.assertTrue(np.all(np.isfinite(f_sat)))
        self.assertTrue(np.all(f_strict >= 0.0))
        self.assertTrue(np.all(f_sat >= 0.0))

        # Check dates and Sundays
        v61_df = pd.read_csv(model.V61_PATH)
        dows = pd.to_datetime(v61_df["date"]).dt.dayofweek
        sundays_mask = (dows == 6)

        # All 10 Sundays must be strictly zero
        self.assertEqual(int(sundays_mask.sum()), 10)
        self.assertTrue(np.all(f_strict[sundays_mask] == 0.0))
        self.assertTrue(np.all(f_sat[sundays_mask] == 0.0))

        # In strict version, exactly 13 days are zero
        self.assertEqual(int((f_strict == 0.0).sum()), 13)

        # In saturday-tuned version, exactly 10 days are zero
        self.assertEqual(int((f_sat == 0.0).sum()), 10)

        # Scores strictly below 0.40000
        self.assertLess(s_strict, 0.40000)
        self.assertLess(s_sat, 0.40000)
        self.assertLess(s_sat, s_strict)


if __name__ == "__main__":
    unittest.main()

