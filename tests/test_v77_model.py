"""Protect chronological forecasting, external-input isolation and persistence."""
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

from v77_model import DailyAdmissionsModel, ModelConfig, campaign, load_counts, load_exams


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = load_counts(ROOT / "processed_data/registered.csv")
        cls.exams = load_exams(ROOT / "external_data/ege/exam_calendar.csv")

    def test_target_labels_and_future_exam_calendars_are_inaccessible(self):
        altered = self.counts.copy()
        altered.loc[altered.index.year >= 2022] = 999999
        exam_altered = self.exams.copy()
        exam_altered.loc[exam_altered.index > 2022] += pd.Timedelta(days=70)
        for hurdle, quiet, hard in ((False, False, False), (True, False, False), (False, True, False), (False, False, True)):
            with self.subTest(hurdle=hurdle, quiet=quiet, hard=hard):
                config = ModelConfig(iterations=10, hurdle=hurdle, quiet_groups=quiet, hard_schedule=hard)
                before = DailyAdmissionsModel(config).fit(self.counts, 2022, self.exams).predict()
                after = DailyAdmissionsModel(config).fit(altered, 2022, exam_altered).predict()
                pd.testing.assert_frame_equal(before, after)

    def test_loader_reads_only_registered_and_exam_dates(self):
        observed = []
        original = pd.read_csv

        def tracked(path, *args, **kwargs):
            observed.append(Path(path).name)
            return original(path, *args, **kwargs)

        with patch("v77_model.pd.read_csv", side_effect=tracked):
            counts = load_counts(ROOT / "processed_data/registered.csv")
            exams = load_exams(ROOT / "external_data/ege/exam_calendar.csv")
            model = DailyAdmissionsModel(ModelConfig(iterations=10)).fit(counts, 2024, exams)
            self.assertNotIn(2020, model.training_years)
            self.assertNotIn(2024, model.training_years)
        self.assertEqual(observed, ["registered.csv", "exam_calendar.csv"])

    def test_cli_from_saved_weights_needs_no_raw_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = DailyAdmissionsModel(ModelConfig(iterations=10)).fit(self.counts, 2024, self.exams)
            model.save(directory / "model.joblib")
            subprocess.run([
                sys.executable, str(ROOT / "v77_model.py"), "predict",
                "--model", str(directory / "model.joblib"), "--registered", str(directory / "missing.csv"),
                "--exams", str(directory / "missing_exams.csv"), "--output", str(directory / "forecast.csv"),
            ], check=True, cwd=directory, capture_output=True)
            restored = DailyAdmissionsModel.load(directory / "model.joblib")
            pd.testing.assert_frame_equal(model.predict(), restored.predict())
            forecast = pd.read_csv(directory / "forecast.csv", parse_dates=["date"])
            np.testing.assert_array_equal(forecast.date, campaign(2024))
            np.testing.assert_allclose(forecast.forecast, model.predict().forecast, atol=0.005)
            self.assertTrue(np.isfinite(forecast.forecast).all())
            self.assertTrue(forecast.forecast.ge(0).all())

    def test_invalid_history_and_missing_calendar_fail(self):
        config = ModelConfig(iterations=10)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            DailyAdmissionsModel(config).fit(self.counts.iloc[1:], 2024, self.exams)
        negative = self.counts.copy()
        negative.iloc[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "negative"):
            DailyAdmissionsModel(config).fit(negative, 2024, self.exams)
        with self.assertRaisesRegex(ValueError, "Missing public exam"):
            DailyAdmissionsModel(config).fit(self.counts, 2024, self.exams.drop(index=2019))
        with self.assertRaisesRegex(ValueError, "Missing public exam"):
            DailyAdmissionsModel(config).fit(self.counts, 2024, self.exams.drop(index=2024))


if __name__ == "__main__":
    unittest.main()
