"""Check the score-calibration algebra and pairing of scores with exact CSVs."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducers import v61_orthogonal_funding as model


class CalibrationTests(unittest.TestCase):
    def test_score_estimate_matches_known_truth_when_assumptions_hold(self):
        u = np.full(4, 5.)
        direction = np.array([1., -1., 1., -1.])*.1
        residual = np.array([1., 1., -1., -1.])*.2
        pool = u-direction
        candidate = u+residual-.25*direction
        truth = u+.4*residual+.05
        rmsle = lambda p: float(np.sqrt(np.mean((p-truth)**2)))
        z, meta = model.calibrate(u, candidate, pool, rmsle(u), rmsle(candidate))
        self.assertAlmostEqual(meta["estimated_optimal_step"], .4)
        self.assertAlmostEqual(meta["actual_step"], .15)
        self.assertAlmostEqual(meta["projection_inner_product"], 0, places=14)
        self.assertAlmostEqual(meta["estimated_rmsle_under_assumptions"], rmsle(z))
        self.assertLess(rmsle(z), rmsle(u))
        self.assertAlmostEqual(sum(meta["weights"].values()), 1.)
        w = meta["weights"]
        np.testing.assert_allclose(z, w["v59"]*u+w["v60"]*candidate+w["pooled"]*pool)
        self.assertLessEqual(meta["log_update_rmse"], .03+1e-12)

    def test_changed_csv_cannot_use_previous_score(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Path(folder)/"submission.csv"
            f.write_text("date,forecast\n2024-06-20,10\n")
            record = {"file": f.name, "sha256": model.digest(f),
                      "source": "user_reported", "public_rmsle": .42473}
            registry = Path(folder)/"scores.json"
            registry.write_text(json.dumps({"evaluations": [record]}))
            with patch.object(model, "SCORES", registry):
                self.assertEqual(model.measured_score(f), .42473)
                f.write_text("date,forecast\n2024-06-20,11\n")
                with self.assertRaisesRegex(ValueError, "exact bytes"):
                    model.measured_score(f)

    def test_dates_must_match_evaluation_campaign(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Path(folder)/"shifted.csv"
            pd.DataFrame({"date": pd.date_range("2024-06-21", periods=73),
                          "forecast": 10.}).to_csv(f, index=False)
            with self.assertRaisesRegex(ValueError, "dates must cover"):
                model.read_forecast(f)


if __name__ == "__main__":
    unittest.main()
