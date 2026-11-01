"""Verify historical-only training and genuine out-of-fold reconciliation."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v85_historical_model import HistoricalForecaster, HistoricalHeads, ModelConfig, campaign, load_daily


class HistoricalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = load_daily(ROOT / "processed_data")

    def test_target_and_future_counts_do_not_change_forecast(self):
        altered = self.counts.copy()
        altered.loc[altered.index.year >= 2022] = 999999
        for penalty in (None, 10.0):
            config = ModelConfig(iterations=10, ridge_penalty=penalty)
            cache = {}
            original = HistoricalForecaster(config).fit(self.counts, 2022, cache).predict()
            changed = HistoricalForecaster(config).fit(altered, 2022, cache).predict()
            pd.testing.assert_frame_equal(original, changed)

    def test_every_calibration_prediction_uses_earlier_training_data(self):
        observed = []
        original = HistoricalHeads.fit

        def tracked(model, history, year):
            observed.append((int(history.index.year.max()), year))
            return original(model, history, year)

        with patch.object(HistoricalHeads, "fit", tracked):
            model = HistoricalForecaster(ModelConfig(iterations=10)).fit(self.counts, 2023)
        self.assertTrue(all(last < year for last, year in observed))
        self.assertEqual(model.calibration_years, [2016, 2017, 2018, 2019, 2021, 2022])

    def test_only_two_raw_sources_are_opened_and_partitions_reconcile(self):
        original = pd.read_csv
        observed = []

        def tracked(path, *args, **kwargs):
            observed.append(Path(path).name)
            return original(path, *args, **kwargs)

        with patch("v85_historical_model.pd.read_csv", side_effect=tracked):
            counts = load_daily(ROOT / "processed_data")
            HistoricalForecaster(ModelConfig(iterations=10)).fit(counts, 2024)
        self.assertEqual(observed, ["registered.csv", "included_ks.csv"])
        np.testing.assert_array_equal(counts.total, counts.budget + counts.other_funding)
        np.testing.assert_array_equal(counts.total, counts[["bachelor", "master", "college", "other_degree"]].sum(axis=1))

    def test_saved_model_predicts_without_training_sources(self):
        model = HistoricalForecaster(ModelConfig(iterations=10)).fit(self.counts, 2024)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model.save(directory / "model.joblib")
            subprocess.run([sys.executable, str(ROOT / "v85_historical_model.py"), "predict",
                            "--model", str(directory / "model.joblib"), "--output", str(directory / "forecast.csv"),
                            "--data", str(directory / "missing")], cwd=directory, check=True, capture_output=True)
            restored = HistoricalForecaster.load(directory / "model.joblib")
            pd.testing.assert_frame_equal(model.predict(), restored.predict())
            forecast = pd.read_csv(directory / "forecast.csv", parse_dates=["date"])
            np.testing.assert_array_equal(forecast.date, campaign(2024))
            np.testing.assert_allclose(forecast.forecast, model.predict().forecast, atol=0.0000005)


if __name__ == "__main__":
    unittest.main()
