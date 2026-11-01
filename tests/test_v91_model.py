"""Time boundaries, annual intensity and independent inference of V91."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v91_model import AdmissionsModel, DEGREES, ProfileConfig, forecast_level
from v88_model import campaign, load_counts, load_exam_calendar


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = load_counts(ROOT / "processed_data/registered.csv")
        cls.exams = load_exam_calendar(ROOT / "external_data/ege/exam_calendar.csv")
        cls.configs = {
            "bachelor": ProfileConfig(
                family="catboost", iterations=20, level_method="trend", exams=True
            ),
            "master": ProfileConfig(family="spline", level_method="mean"),
            "background": ProfileConfig(family="extra", iterations=12, level_method="none"),
        }

    def test_level_trend_uses_only_preceding_seasons(self):
        levels = pd.Series([4.0, 3.8, 3.6], index=[2021, 2022, 2023])
        trend = ProfileConfig(level_method="trend")
        predicted = forecast_level(levels, 2024, trend)
        extended = pd.concat([levels, pd.Series([9999.0], index=[2024])])
        self.assertEqual(predicted, forecast_level(extended, 2024, trend))
        self.assertLess(predicted, forecast_level(levels, 2024, ProfileConfig(level_method="mean")))
        self.assertEqual(forecast_level(levels, 2024, ProfileConfig(level_method="none")), 0.0)

    def test_future_raw_labels_cannot_change_either_level_or_profile(self):
        model = AdmissionsModel(self.configs).fit(self.counts, 2024, self.exams)
        future = pd.DataFrame(999999.0, index=campaign(2024), columns=self.counts.columns)
        altered = AdmissionsModel(self.configs).fit(
            pd.concat([self.counts, future]), 2024, self.exams
        )
        pd.testing.assert_frame_equal(
            model.predict(components=True), altered.predict(components=True)
        )
        for degree in DEGREES:
            self.assertEqual(model.models[degree].future_level, altered.models[degree].future_level)
            self.assertNotIn(2024, model.models[degree].training_years)

    def test_constant_campaign_is_recovered_by_level_and_profile(self):
        counts = self.counts.copy()
        counts.loc[:, :] = 0.0
        opened = counts.index.dayofweek != 6
        counts.loc[opened, ["bachelor", "master", "college"]] = [20.0, 10.0, 5.0]
        counts["total"] = counts[["bachelor", "master", "college", "other"]].sum(axis=1)
        configs = {d: ProfileConfig(level_method="trend") for d in DEGREES}
        prediction = AdmissionsModel(configs).fit(counts, 2024).predict(components=True)
        opened = prediction.date.dt.dayofweek != 6
        np.testing.assert_allclose(
            prediction.loc[opened, list(DEGREES)],
            np.tile([20.0, 10.0, 5.0], (int(opened.sum()), 1)),
            atol=1e-8,
        )
        self.assertTrue(prediction.loc[~opened, "forecast"].eq(0).all())
        np.testing.assert_allclose(prediction[list(DEGREES)].sum(axis=1), prediction.forecast)

    def test_cli_training_without_previous_forecast_files_and_round_trip(self):
        with tempfile.TemporaryDirectory(prefix="v91_clean_") as tmp:
            folder = Path(tmp)
            selection = {"degrees": {d: asdict(c) for d, c in self.configs.items()}}
            (folder / "selection.json").write_text(json.dumps(selection))
            common = [sys.executable, str(ROOT / "v91_model.py")]
            subprocess.run(
                common
                + [
                    "train",
                    "--selection",
                    str(folder / "selection.json"),
                    "--data",
                    str(ROOT / "processed_data/registered.csv"),
                    "--exams",
                    str(ROOT / "external_data/ege/exam_calendar.csv"),
                    "--model",
                    str(folder / "model.joblib"),
                    "--output",
                    str(folder / "trained.csv"),
                ],
                cwd=folder,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                common
                + [
                    "predict",
                    "--selection",
                    "/absent",
                    "--data",
                    "/absent",
                    "--exams",
                    "/absent",
                    "--model",
                    str(folder / "model.joblib"),
                    "--output",
                    str(folder / "loaded.csv"),
                ],
                cwd=folder,
                check=True,
                capture_output=True,
            )
            self.assertEqual(
                (folder / "trained.csv").read_bytes(), (folder / "loaded.csv").read_bytes()
            )
            predicted = AdmissionsModel.load(folder / "model.joblib").predict(components=True)
            self.assertEqual(len(predicted), 73)
            self.assertTrue(np.isfinite(predicted[list(DEGREES) + ["forecast"]]).all().all())


if __name__ == "__main__":
    unittest.main()
