import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducers.v64_final_calibrated import calibrate


class FinalCalibrationTests(unittest.TestCase):
    def test_estimate_matches_actual_loss_on_known_full_calendar(self):
        rng = np.random.default_rng(48)
        truth = 3+rng.normal(0,.05,73)
        U = truth+rng.normal(0,.03,(5,73))
        scores = np.sqrt(np.mean((U-truth)**2,axis=1))
        prediction, meta = calibrate(U,scores)
        actual = float(np.sqrt(np.mean((prediction-truth)**2)))
        self.assertAlmostEqual(actual, meta["estimated_public_rmsle_under_full_calendar_assumption"])
        self.assertLess(actual, min(scores))
        self.assertLessEqual(meta["max_abs_log_update"], .15+1e-12)
        weights = np.asarray(meta["weights"])
        self.assertAlmostEqual(weights.sum(),1.)
        np.testing.assert_allclose(prediction,weights@U)

    def test_duplicate_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            calibrate(np.ones((3,73)),[.4,.5,.6])


if __name__ == "__main__":
    unittest.main()
