"""Historical isolation, counting integrity and reproducible forest inference."""

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

from v88_model import (
    AdmissionsForest,
    ForestConfig,
    annual_features,
    campaign,
    load_counts,
    load_exam_calendar,
)


class ForestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = load_counts(ROOT / "processed_data/registered.csv")
        cls.exams = load_exam_calendar(ROOT / "external_data/ege/exam_calendar.csv")

    def test_future_targets_do_not_change_forecast_or_context(self):
        altered = self.counts.copy()
        altered.loc[altered.index.year >= 2022] = 999999
        pd.testing.assert_frame_equal(
            annual_features(campaign(2022), self.counts), annual_features(campaign(2022), altered)
        )
        for family, schedule in (("forest", "none"), ("extra", "learned")):
            config = ForestConfig(family=family, trees=20, annual_context=True, schedule=schedule)
            before = AdmissionsForest(config).fit(self.counts, 2022).predict()
            after = AdmissionsForest(config).fit(altered, 2022).predict()
            pd.testing.assert_frame_equal(before, after)

    def test_only_raw_registrations_are_read_and_rows_count_once(self):
        observed = []
        original = pd.read_csv

        def tracked(path, *args, **kwargs):
            observed.append(Path(path).name)
            return original(path, *args, **kwargs)

        with patch("v88_model.pd.read_csv", side_effect=tracked):
            counts = load_counts(ROOT / "processed_data/registered.csv")
            model = AdmissionsForest(ForestConfig(trees=20)).fit(counts, 2024)
        self.assertEqual(observed, ["registered.csv"])
        self.assertLess(model.history.index.year.max(), 2024)
        self.assertTrue(all(year < 2024 for year, _ in model.breaks))
        np.testing.assert_array_equal(
            counts.total, counts[["bachelor", "master", "college", "other"]].sum(axis=1)
        )
        raw = original(ROOT / "processed_data/registered.csv")
        totals = (
            pd.to_datetime(raw.reg_ts)
            .dt.normalize()
            .value_counts()
            .reindex(counts.index, fill_value=0)
        )
        np.testing.assert_array_equal(counts.total, totals)

    def test_saved_weights_and_cli_do_not_need_raw_sources(self):
        for family in ("forest", "extra"):
            model = AdmissionsForest(
                ForestConfig(family=family, trees=20, annual_context=True, exams=family == "extra")
            ).fit(self.counts, 2024, self.exams)
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                model.save(directory / "model.joblib")
                pd.testing.assert_frame_equal(
                    model.predict(), AdmissionsForest.load(directory / "model.joblib").predict()
                )
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "v88_model.py"),
                        "predict",
                        "--model",
                        str(directory / "model.joblib"),
                        "--data",
                        str(directory / "missing.csv"),
                        "--output",
                        str(directory / "forecast.csv"),
                    ],
                    check=True,
                    capture_output=True,
                    cwd=directory,
                )
                forecast = pd.read_csv(directory / "forecast.csv", parse_dates=["date"])
                np.testing.assert_array_equal(forecast.date, campaign(2024))
                np.testing.assert_allclose(
                    forecast.forecast, model.predict().forecast, atol=0.0000005
                )
                self.assertTrue(np.isfinite(forecast.forecast).all())
                self.assertTrue(forecast.forecast.ge(0).all())

    def test_missing_dates_and_inconsistent_partitions_fail(self):
        model = AdmissionsForest(ForestConfig(trees=20))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            model.fit(self.counts.iloc[1:], 2024)
        changed = self.counts.copy()
        changed.iloc[0, changed.columns.get_loc("bachelor")] += 1
        with self.assertRaisesRegex(ValueError, "partition"):
            model.fit(changed, 2024)

    def test_exam_inputs_exclude_later_years_and_are_optional(self):
        altered = self.exams.copy()
        altered.loc[altered.index > 2022] += pd.Timedelta(days=70)
        config = ForestConfig(trees=20, exams=True)
        before = AdmissionsForest(config).fit(self.counts, 2022, self.exams).predict()
        after = AdmissionsForest(config).fit(self.counts, 2022, altered).predict()
        pd.testing.assert_frame_equal(before, after)
        with self.assertRaisesRegex(ValueError, "Missing public"):
            AdmissionsForest(config).fit(self.counts, 2024, self.exams.drop(index=2024))


if __name__ == "__main__":
    unittest.main()
