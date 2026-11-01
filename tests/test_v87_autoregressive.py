"""Protect temporal lags, OOF residual training and standalone inference."""
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

from v87_autoregressive import AutoregressiveConfig, SeasonalAutoregressor
from v87_data import campaign, lag_features, load_daily


class AutoregressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = load_daily(ROOT / "processed_data")

    def test_lags_cannot_read_their_own_or_later_season(self):
        altered = self.counts.copy()
        altered.loc[altered.index.year >= 2022] = 999999
        before = lag_features(campaign(2022), self.counts)
        after = lag_features(campaign(2022), altered)
        pd.testing.assert_frame_equal(before, after)
        self.assertTrue(lag_features(campaign(2014), self.counts).isna().all().all())

    def test_future_targets_do_not_change_model_or_oof_predictions(self):
        altered = self.counts.copy()
        altered.loc[altered.index.year >= 2022] = 999999
        for family, correction in (("cat", False), ("cat", True), ("lgb", True)):
            config = AutoregressiveConfig(family=family, iterations=20, depth=2, correction=correction)
            cache = {}
            original = SeasonalAutoregressor(config).fit(self.counts, 2022, cache).predict()
            changed = SeasonalAutoregressor(config).fit(altered, 2022, cache).predict()
            pd.testing.assert_frame_equal(original, changed)

    def test_every_internal_fit_stops_before_its_forecast_year(self):
        observed = []
        original = SeasonalAutoregressor._fit_base

        def tracked(model, counts, year):
            observed.append((int(counts.index.year.max()), year))
            return original(model, counts, year)

        with patch.object(SeasonalAutoregressor, "_fit_base", tracked):
            model = SeasonalAutoregressor(AutoregressiveConfig(iterations=20, correction=True)).fit(self.counts, 2023)
        self.assertTrue(all(last < year for last, year in observed))
        self.assertEqual(model.calibration_years, [2016, 2017, 2018, 2019, 2021, 2022])

    def test_only_raw_inputs_are_read_and_channel_can_be_absent(self):
        observed = []
        original = pd.read_csv

        def tracked(path, *args, **kwargs):
            observed.append(Path(path).name)
            return original(path, *args, **kwargs)

        with patch("v87_data.pd.read_csv", side_effect=tracked):
            counts = load_daily(ROOT / "processed_data")
            model = SeasonalAutoregressor(AutoregressiveConfig(iterations=20)).fit(counts, 2017)
            prediction = model.predict(diagnostics=True)
        self.assertEqual(observed, ["registered.csv", "included_ks.csv"])
        np.testing.assert_array_equal(prediction.government_log, 0.0)
        np.testing.assert_array_equal(counts.total, counts.budget + counts.other_funding)

    def test_persistence_and_cli_need_no_raw_data(self):
        for family in ("cat", "lgb"):
            model = SeasonalAutoregressor(AutoregressiveConfig(family=family, iterations=20, correction=True)).fit(self.counts, 2024)
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                model.save(directory / "model.joblib")
                pd.testing.assert_frame_equal(model.predict(), SeasonalAutoregressor.load(directory / "model.joblib").predict())
                subprocess.run([sys.executable, str(ROOT / "v87_autoregressive.py"), "predict",
                                "--model", str(directory / "model.joblib"), "--data", str(directory / "missing"),
                                "--output", str(directory / "forecast.csv")], check=True, capture_output=True, cwd=directory)
                forecast = pd.read_csv(directory / "forecast.csv", parse_dates=["date"])
                np.testing.assert_array_equal(forecast.date, campaign(2024))
                np.testing.assert_allclose(forecast.forecast, model.predict().forecast, atol=.0000005)

    def test_incomplete_or_inconsistent_counts_fail(self):
        model = SeasonalAutoregressor(AutoregressiveConfig(iterations=20))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            model.fit(self.counts.iloc[1:], 2024)
        changed = self.counts.copy()
        changed.iloc[0, changed.columns.get_loc("budget")] += 1
        with self.assertRaisesRegex(ValueError, "partitions"):
            model.fit(changed, 2024)


if __name__ == "__main__":
    unittest.main()
