"""Check chronological training, compositional constraints and analytic gradients."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize._numdiff import approx_derivative

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v86_volume_shape import ShapeConfig, STREAMS, VolumeShapeForecaster, campaign, load_counts, shape_objective


class CompositionalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = load_counts(ROOT / "processed_data/registered.csv")

    def test_loss_gradient_includes_softmax_normalization_and_structural_zeros(self):
        rng = np.random.default_rng(86)
        design = rng.normal(size=(2, 3, 7, 4))
        volumes = rng.uniform(20, 50, size=(2, 3))
        volumes[0, 1] = 0.0
        allowed = rng.uniform(size=(2, 3, 7)) > 0.15
        actual = rng.poisson(2, size=(2, 3, 7))
        point = rng.normal(size=12) * 0.1
        objective = lambda x: shape_objective(x, design, volumes, allowed, actual, np.array([0.7, 1.]), np.eye(4), .2, .3)
        numeric = approx_derivative(lambda x: objective(x)[0], point).ravel()
        np.testing.assert_allclose(objective(point)[1], numeric, atol=1e-6, rtol=1e-6)

    def test_unbiased_profiles_preserve_forecast_volumes(self):
        for channels in (False, True):
            model = VolumeShapeForecaster(ShapeConfig(knots=5, channels=channels)).fit(self.counts, 2024)
            forecast = model.predict(components=True)
            np.testing.assert_allclose(forecast[list(STREAMS)].sum().to_numpy(), model.future_volumes, rtol=1e-10)
            np.testing.assert_allclose(forecast[list(STREAMS)].sum(axis=1), forecast.forecast)
            self.assertTrue(np.isfinite(forecast.forecast).all())
            self.assertTrue(forecast.forecast.ge(0).all())
            self.assertEqual(len(forecast), 73)

    def test_future_targets_cannot_change_forecast_or_cache(self):
        altered = self.counts.copy()
        altered.loc[altered.index.year >= 2022] = 999999
        for calibration in (None, 100.):
            config = ShapeConfig(knots=5, calibration_penalty=calibration)
            cache = {}
            original = VolumeShapeForecaster(config).fit(self.counts, 2022, cache).predict()
            changed = VolumeShapeForecaster(config).fit(altered, 2022, cache).predict()
            pd.testing.assert_frame_equal(original, changed)

    def test_internal_forecasts_use_strictly_earlier_observations(self):
        calls = []
        original = VolumeShapeForecaster._fit_shape

        def tracked(model, counts, year):
            calls.append((int(counts.index.year.max()), year))
            return original(model, counts, year)

        with patch.object(VolumeShapeForecaster, "_fit_shape", tracked):
            model = VolumeShapeForecaster(ShapeConfig(knots=5, calibration_penalty=100.)).fit(self.counts, 2023)
        self.assertTrue(all(last < year for last, year in calls))
        self.assertEqual(model.calibration_years, [2016, 2017, 2018, 2019, 2021, 2022])

    def test_only_raw_registrations_are_loaded(self):
        original = pd.read_csv
        observed = []

        def tracked(path, *args, **kwargs):
            observed.append(Path(path).name)
            return original(path, *args, **kwargs)

        with patch("v86_volume_shape.pd.read_csv", side_effect=tracked):
            counts = load_counts(ROOT / "processed_data/registered.csv")
            VolumeShapeForecaster(ShapeConfig(knots=5)).fit(counts, 2024)
        self.assertEqual(observed, ["registered.csv"])
        rows = original(ROOT / "processed_data/registered.csv")
        target = pd.to_datetime(rows.reg_ts).dt.normalize().value_counts().reindex(counts.index, fill_value=0)
        np.testing.assert_array_equal(counts.sum(axis=1), target)

    def test_saved_weights_predict_without_source_data(self):
        model = VolumeShapeForecaster(ShapeConfig(knots=5, calibration_penalty=100.)).fit(self.counts, 2024)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model.save(directory / "model.joblib")
            pd.testing.assert_frame_equal(model.predict(), VolumeShapeForecaster.load(directory / "model.joblib").predict())
            subprocess.run([sys.executable, str(ROOT / "v86_volume_shape.py"), "predict", "--model", str(directory / "model.joblib"),
                            "--data", str(directory / "missing.csv"), "--output", str(directory / "forecast.csv")],
                           check=True, capture_output=True, cwd=directory)
            forecast = pd.read_csv(directory / "forecast.csv", parse_dates=["date"])
            np.testing.assert_array_equal(forecast.date, campaign(2024))
            np.testing.assert_allclose(forecast.forecast, model.predict().forecast, atol=0.0000005)

    def test_incomplete_and_negative_counts_fail(self):
        model = VolumeShapeForecaster(ShapeConfig(knots=5))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            model.fit(self.counts.iloc[1:], 2024)
        changed = self.counts.copy()
        changed.iloc[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "Invalid"):
            model.fit(changed, 2024)


if __name__ == "__main__":
    unittest.main()
