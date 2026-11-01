"""V75 uses only historical raw inputs and saved fitted weights."""

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

from v75_model import AdmissionsForecaster, CalendarHeads, ForecastConfig, campaign, load_counts


class CleanModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = load_counts(ROOT / "processed_data")

    def test_target_and_future_labels_do_not_change_predictions(self):
        altered = self.counts.copy()
        altered.loc[altered.index.year >= 2022] = 999999
        for mode in ("none", "global", "shape"):
            with self.subTest(calibration=mode):
                config = ForecastConfig(iterations=15, calibration=mode)
                cache = {}
                original = AdmissionsForecaster(config).fit(self.counts, 2022, cache).predict()
                changed = AdmissionsForecaster(config).fit(altered, 2022, cache).predict()
                pd.testing.assert_frame_equal(original, changed)

    def test_every_internal_calibration_fold_excludes_its_target_season(self):
        observed = []
        original = CalendarHeads.fit

        def tracked_fit(model, history, year):
            observed.append((int(history.index.year.max()), year))
            return original(model, history, year)

        with patch.object(CalendarHeads, "fit", tracked_fit):
            fitted = AdmissionsForecaster(ForecastConfig(iterations=15, calibration="shape"))
            fitted.fit(self.counts, 2023)
        self.assertEqual(fitted.calibration_years, [2016, 2017, 2018, 2019, 2021, 2022])
        self.assertTrue(all(last < target for last, target in observed))
        self.assertTrue(all(target <= 2023 for _, target in observed))

    def test_loading_training_data_opens_only_two_raw_sources(self):
        original = pd.read_csv
        observed = []

        def tracked_read(path, *args, **kwargs):
            observed.append(Path(path).name)
            return original(path, *args, **kwargs)

        with patch("v75_model.pd.read_csv", side_effect=tracked_read):
            counts = load_counts(ROOT / "processed_data")
            AdmissionsForecaster(ForecastConfig(iterations=15)).fit(counts, 2024)
        self.assertEqual(observed, ["registered.csv", "included_ks.csv"])
        np.testing.assert_array_equal(counts.total, counts.budget + counts.other)

    def test_persistence_and_cli_need_no_source_data(self):
        for mode in ("none", "shape"):
            with self.subTest(calibration=mode), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                model = AdmissionsForecaster(ForecastConfig(iterations=15, calibration=mode))
                model.fit(self.counts, 2024).save(directory / "model.joblib")
                restored = AdmissionsForecaster.load(directory / "model.joblib")
                pd.testing.assert_frame_equal(model.predict(), restored.predict())
                subprocess.run(
                    [sys.executable, str(ROOT / "v75_model.py"), "predict",
                     "--model", str(directory / "model.joblib"),
                     "--data", str(directory / "missing"),
                     "--output", str(directory / "forecast.csv")],
                    cwd=directory, check=True, capture_output=True,
                )
                forecast = pd.read_csv(directory / "forecast.csv", parse_dates=["date"])
                np.testing.assert_array_equal(forecast.date, campaign(2024))
                np.testing.assert_allclose(forecast.forecast, model.predict().forecast, atol=0.005)

    def test_missing_dates_and_inconsistent_funding_fail(self):
        config = ForecastConfig(iterations=15)
        with self.assertRaisesRegex(ValueError, "73 dates"):
            AdmissionsForecaster(config).fit(self.counts.iloc[1:], 2024)
        changed = self.counts.copy()
        changed.iloc[0, changed.columns.get_loc("budget")] += 1
        with self.assertRaisesRegex(ValueError, "partition"):
            AdmissionsForecaster(config).fit(changed, 2024)


if __name__ == "__main__":
    unittest.main()
