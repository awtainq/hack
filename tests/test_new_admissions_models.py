"""Check leakage boundaries, custom gradients, row accounting and saved models."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from activity_admissions_model import ActivityAdmissionsModel, ActivityConfig
from admissions_profile_model import load_counts, campaign
from coherent_admissions_model import (
    CoherentAdmissionsModel, CoherentConfig, STREAMS,
    load_stream_counts, partition_registrations,
)
from joint_admissions_model import loss_and_gradient


class ObjectiveTests(unittest.TestCase):
    def test_custom_gradient_matches_finite_differences(self):
        random = np.random.default_rng(73)
        raw = random.normal(0, 2, (3, 12))
        actual = random.poisson(3, raw.shape).astype(float)
        weights = np.array([0.2, 0.7, 1.0])
        _, gradient, curvature = loss_and_gradient(raw, actual, weights, 0.1)
        numerical = np.zeros_like(raw)
        step = 1e-5
        for index in np.ndindex(raw.shape):
            left, right = raw.copy(), raw.copy()
            left[index] -= step
            right[index] += step
            numerical[index] = (
                loss_and_gradient(right, actual, weights, 0.1)[0]
                - loss_and_gradient(left, actual, weights, 0.1)[0]
            ) / (2 * step)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-6, atol=1e-8)
        self.assertTrue((curvature > 0).all())

    def test_funding_join_counts_each_registration_once(self):
        registered = pd.DataFrame({
            "abitur_kod": [1, 2, 3, 4], "zayav_type_kod": [1, 1, 1, 1],
            "zayav_type_name": ["Бакалавриат"] * 4,
            "reg_ts": ["2023-07-01 10:00:00"] * 4,
        })
        included = pd.DataFrame({
            "abitur_kod": [1, 1, 1, 2, 4], "zayav_type_kod": [1] * 5,
            "reg_ts": ["2023-07-01 11:00:00"] * 4 + ["2024-07-01 11:00:00"],
            "ob_osnova_name": ["Договор", "Бюджет", "Бюджет", "Договор", "Бюджет"],
        })
        result = partition_registrations(registered, included)
        self.assertEqual(len(result), 4)
        self.assertEqual(result.funding.tolist(), ["budget", "contract", "unknown", "unknown"])
        self.assertNotIn("abitur_kod", result)


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.degrees = load_counts(ROOT / "processed_data/registered.csv")
        cls.streams = load_stream_counts(ROOT / "processed_data")

    def test_real_funding_partition_preserves_daily_totals(self):
        np.testing.assert_array_equal(self.degrees.sum(axis=1), self.streams.sum(axis=1))

    def test_future_labels_do_not_change_fitted_predictions(self):
        settings = [
            (ActivityAdmissionsModel, ActivityConfig(iterations=15, activity_iterations=15), self.degrees),
            (CoherentAdmissionsModel, CoherentConfig(iterations=15), self.streams),
        ]
        for model_type, config, counts in settings:
            with self.subTest(model=model_type.__name__):
                altered = counts.copy()
                altered.loc[altered.index.year >= 2022] = 987654
                first = model_type(config).fit(counts, 2022).predict()
                second = model_type(config).fit(altered, 2022).predict()
                pd.testing.assert_frame_equal(first, second)

    def test_models_predict_from_saved_weights_without_raw_inputs(self):
        settings = [
            (ActivityAdmissionsModel, ActivityConfig(iterations=15, activity_iterations=15),
             self.degrees, "activity_admissions_model.py"),
            (CoherentAdmissionsModel, CoherentConfig(iterations=15),
             self.streams, "coherent_admissions_model.py"),
        ]
        for model_type, config, counts, script in settings:
            with self.subTest(model=model_type.__name__), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                model = model_type(config).fit(counts, 2024)
                model.save(directory / "model")
                restored = model_type.load(directory / "model")
                pd.testing.assert_frame_equal(model.predict(), restored.predict())
                subprocess.run(
                    [sys.executable, str(ROOT / script), "predict",
                     "--data", str(directory / "does_not_exist"),
                     "--model-dir", str(directory / "model"),
                     "--output", str(directory / "forecast.csv")],
                    cwd=directory, check=True, capture_output=True,
                )
                forecast = pd.read_csv(directory / "forecast.csv", parse_dates=["date"])
                np.testing.assert_array_equal(forecast.date, campaign(2024))
                self.assertTrue(np.isfinite(forecast.forecast).all())
                self.assertTrue((forecast.forecast >= 0).all())
                np.testing.assert_allclose(forecast.forecast, model.predict().forecast, atol=0.005)

    def test_coherent_total_is_sum_of_predicted_streams(self):
        model = CoherentAdmissionsModel(CoherentConfig(iterations=15)).fit(self.streams, 2024)
        result = model.predict(components=True)
        np.testing.assert_allclose(result.forecast, result[list(STREAMS)].sum(axis=1), atol=1e-12)

    def test_incomplete_history_is_not_treated_as_zero(self):
        for model, counts in [
            (ActivityAdmissionsModel(), self.degrees),
            (CoherentAdmissionsModel(), self.streams),
        ]:
            with self.subTest(model=type(model).__name__):
                with self.assertRaisesRegex(ValueError, "73 dates"):
                    model.fit(counts.drop(counts.index[2]), 2024)


if __name__ == "__main__":
    unittest.main()
