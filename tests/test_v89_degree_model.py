"""Separate target supervision, explicit teacher dependence and persistence."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v89_degree_model import DEGREES, DegreeConfig, SeparateDegreeForecaster
from v88_model import load_counts, load_exam_calendar


class DegreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = load_counts(ROOT / "processed_data/registered.csv")
        cls.exams = load_exam_calendar(ROOT / "external_data/ege/exam_calendar.csv")
        cls.reference = pd.read_csv(ROOT / "forecast_2024_v84.csv", parse_dates=["date"])

    def test_each_regressor_receives_its_own_historical_target(self):
        targets = []
        original = CatBoostRegressor.fit

        def tracked(model, frame, target, **kwargs):
            targets.append(np.asarray(target).copy())
            return original(model, frame, target, **kwargs)

        with patch.object(CatBoostRegressor, "fit", tracked):
            model = SeparateDegreeForecaster(DegreeConfig(iterations=20)).fit(
                self.counts, 2024, self.exams
            )
        history = self.counts[
            (self.counts.index.year < 2024)
            & (self.counts.index.year != 2020)
            & (self.counts.index.dayofweek != 6)
        ]
        for actual, expected in zip(
            targets, (history.bachelor, history.master, history.college + history.other)
        ):
            np.testing.assert_array_equal(actual, np.log1p(expected))
        self.assertEqual(len(targets), 3)
        self.assertEqual(len({id(value) for value in model.models.values()}), 3)
        self.assertIsNone(model.calibrator)
        self.assertIsNone(model.reference_provenance)

    def test_teacher_calibration_preserves_degree_shares_and_total(self):
        model = SeparateDegreeForecaster(DegreeConfig(iterations=20)).fit(
            self.counts, 2024, self.exams
        )
        base = model.base_prediction()
        model.adapt_to_reference(self.reference, {"file": "reference.csv"})
        predicted = model.predict(components=True).set_index("date")
        positive = base.sum(axis=1) > 0
        before = base.loc[positive].div(base.loc[positive].sum(axis=1), axis=0)
        after = predicted.loc[positive, list(DEGREES)].div(
            predicted.loc[positive, "forecast"], axis=0
        )
        np.testing.assert_allclose(before, after, atol=1e-12)
        np.testing.assert_allclose(predicted[list(DEGREES)].sum(axis=1), predicted.forecast)
        self.assertFalse(model.reference_provenance["independent_of_reference"])

    def test_future_raw_labels_do_not_change_base_or_guided_predictions(self):
        extra = pd.DataFrame(
            999999.0,
            index=pd.date_range("2024-06-20", "2024-08-31", name="date"),
            columns=self.counts.columns,
        )
        altered = pd.concat([self.counts, extra])
        config = DegreeConfig(iterations=20)
        first = SeparateDegreeForecaster(config).fit(self.counts, 2024, self.exams)
        second = SeparateDegreeForecaster(config).fit(altered, 2024, self.exams)
        pd.testing.assert_frame_equal(first.predict(), second.predict())
        first.adapt_to_reference(self.reference)
        second.adapt_to_reference(self.reference)
        pd.testing.assert_frame_equal(first.predict(), second.predict())

    def test_saved_guided_model_predicts_without_reference_or_raw_files(self):
        model = SeparateDegreeForecaster(DegreeConfig(iterations=20)).fit(
            self.counts, 2024, self.exams
        )
        model.adapt_to_reference(self.reference)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model.save(directory / "model.joblib")
            pd.testing.assert_frame_equal(
                model.predict(), SeparateDegreeForecaster.load(directory / "model.joblib").predict()
            )
            self.assertTrue(all((directory / f"{degree}_base.cbm").exists() for degree in DEGREES))
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "v89_degree_model.py"),
                    "predict",
                    "--model",
                    str(directory / "model.joblib"),
                    "--output",
                    str(directory / "forecast.csv"),
                    "--data",
                    str(directory / "missing.csv"),
                    "--reference",
                    str(directory / "missing_reference.csv"),
                ],
                check=True,
                capture_output=True,
                cwd=directory,
            )
            forecast = pd.read_csv(directory / "forecast.csv", parse_dates=["date"])
            np.testing.assert_allclose(forecast.forecast, model.predict().forecast, atol=0.0000005)

    def test_reference_dates_and_values_are_validated(self):
        model = SeparateDegreeForecaster(DegreeConfig(iterations=20)).fit(
            self.counts, 2024, self.exams
        )
        with self.assertRaisesRegex(ValueError, "dates"):
            model.adapt_to_reference(self.reference.iloc[1:])
        bad = self.reference.copy()
        bad.loc[0, "forecast"] = -1
        with self.assertRaisesRegex(ValueError, "Invalid reference"):
            model.adapt_to_reference(bad)


if __name__ == "__main__":
    unittest.main()
