"""Data-boundary and persistence checks for the standalone predictor."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from admissions_model import AdmissionsModel, ModelConfig, read_daily_counts, season_dates


class AdmissionsModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = read_daily_counts(ROOT / "processed_data/registered.csv")
        cls.config = ModelConfig(architecture="hurdle", iterations=20, depth=3)

    def test_future_targets_cannot_change_fitted_predictions(self):
        altered = self.counts.copy()
        altered.loc[altered.index.year >= 2022] = 999_999
        first = AdmissionsModel(self.config).fit(self.counts, 2022).predict()
        second = AdmissionsModel(self.config).fit(altered, 2022).predict()
        pd.testing.assert_frame_equal(first, second)

    def test_activity_model_sees_zero_and_nonzero_days(self):
        with patch("admissions_model.CatBoostClassifier.fit", autospec=True) as fit:
            AdmissionsModel(self.config).fit(self.counts, 2022)
        labels = np.asarray(fit.call_args.args[2])
        weights = np.asarray(fit.call_args.kwargs["sample_weight"])
        self.assertEqual(set(labels), {0, 1})
        self.assertTrue((weights > 0).all())
        self.assertEqual(len(labels), 8 * 73)

    def test_selected_direct_model_learns_zero_days(self):
        config = ModelConfig(iterations=20)
        with patch("admissions_model.CatBoostRegressor.fit", autospec=True) as fit:
            AdmissionsModel(config).fit(self.counts, 2024)
        labels = np.asarray(fit.call_args.args[2])
        self.assertEqual(len(labels), 10 * 73)
        self.assertGreater((labels == 0).sum(), 0)

    def test_model_files_predict_without_source_data(self):
        model = AdmissionsModel(self.config).fit(self.counts, 2024)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model.save(directory / "model")
            restored = AdmissionsModel.load(directory / "model")
            pd.testing.assert_frame_equal(model.predict(), restored.predict())
            subprocess.run(
                [sys.executable, str(ROOT / "admissions_model.py"), "predict",
                 "--data", str(directory / "nonexistent.csv"),
                 "--model-dir", str(directory / "model"),
                 "--output", str(directory / "forecast.csv")],
                cwd=directory, check=True, capture_output=True,
            )
            forecast = pd.read_csv(directory / "forecast.csv", parse_dates=["date"])
            self.assertEqual(list(forecast.columns), ["date", "forecast"])
            pd.testing.assert_index_equal(pd.DatetimeIndex(forecast.date), season_dates(2024))
            self.assertTrue(np.isfinite(forecast.forecast).all())
            self.assertTrue((forecast.forecast >= 0).all())
            np.testing.assert_allclose(forecast.forecast, model.predict().forecast, atol=0.005)

    def test_missing_season_and_negative_counts_fail(self):
        missing = self.counts.loc[self.counts.index.year != 2020]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            AdmissionsModel(self.config).fit(missing, 2024)
        negative = self.counts.copy()
        negative.iloc[0] = -1
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            AdmissionsModel(self.config).fit(negative, 2024)


if __name__ == "__main__":
    unittest.main()
