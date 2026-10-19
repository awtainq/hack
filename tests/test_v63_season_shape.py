"""Held-out-label independence and volume/shape invariants of V63."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducers import v63_season_shape as model


class ShapeTests(unittest.TestCase):
    def test_monthly_weekday_level_is_preserved(self):
        te = pd.DataFrame({"month": [6, 6, 6, 7, 7, 7], "dow": [0, 1, 5, 0, 1, 5]})
        raw = np.array([1., 2., .5, 2., 3., .5])
        anchor = np.array([4., 5., 2., 5., 6., 3.])
        result = model.align_level(raw, anchor, te, "month")
        for month in [6, 7]:
            mask = (te.month == month) & (te.dow < 5)
            self.assertAlmostEqual(result[mask].mean(), anchor[mask].mean())

    def test_existing_anchor_correction_is_preserved(self):
        reference = np.array([4., 5., 0.])
        anchor = np.array([4.1, 4.9, 0.])
        shape = np.array([3.8, 5.2, 9.])
        closed = np.array([False, False, True])
        corrected = model.apply_shape(reference, anchor, shape, closed)
        ordinary = model.apply_shape(reference, reference, shape, closed)
        np.testing.assert_allclose(corrected-ordinary, anchor-reference)
        self.assertEqual(corrected[-1], 0.)

    def test_future_labels_cannot_affect_learned_shape(self):
        year = 2019
        g = model.base.create_daily_grid(model.base.REG_FILE, year, model.nominal_ends(year))
        tr, te = g[g.year < year], g[g.year == year].reset_index(drop=True)
        reference = np.full(len(te), 4.)
        first = model.shape_prediction(tr, te, reference)
        changed = te.copy()
        changed["y"] = 10**9
        changed["z"] = 40.
        changed["epgu"] = 10**8
        second = model.shape_prediction(tr, changed, reference)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
