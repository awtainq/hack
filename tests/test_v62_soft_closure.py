"""Test transfer of a floor-only squared RMSLE effect between active-day models."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducers.v62_soft_closure import floor_curve, check_only_floor_changed


class FloorCurveTests(unittest.TestCase):
    def test_curve_transfers_to_different_active_day_predictions(self):
        truth = np.array([6., 8., 0., 3.])
        quiet = np.array([False, False, True, True])
        levels = np.array([.55, 1., 1.45])
        old = np.array([4., 5., 0., 0.])
        rmsle2 = lambda p: float(np.mean((np.log1p(p)-np.log1p(truth))**2))
        scores = [np.sqrt(rmsle2(np.where(quiet, f, old))) for f in levels]
        curve = floor_curve(levels, scores)
        self.assertAlmostEqual(curve["optimal_count_floor"], 1.)
        new = np.array([7., 9., 0., 0.])
        x = np.log1p(1.)
        predicted_loss = rmsle2(new)+curve["quadratic"]*x*x+curve["linear"]*x
        self.assertAlmostEqual(predicted_loss, rmsle2(np.where(quiet, 1., new)))

    def test_rejects_a_changed_non_quiet_forecast(self):
        quiet = np.array([False, True])
        a = pd.DataFrame({"date": ["2024-06-20", "2024-06-21"], "forecast": [10., .55]})
        b = a.copy()
        b.forecast = [11., 1.]
        with self.assertRaisesRegex(ValueError, "non-quiet"):
            check_only_floor_changed([a, b], [.55, 1.], quiet)


if __name__ == "__main__":
    unittest.main()
