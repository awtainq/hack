"""Deadline transfer, independent streams, temporal boundaries and persistence."""

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

from admissions_calendar import (
    DEGREES,
    campaign,
    features,
    load_counts,
    load_exams,
    nominal_deadline,
)
from v92_model import AdmissionsModel, DegreeConfig, DegreeModel


class DeadlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = load_counts(ROOT / "processed_data/registered.csv")
        cls.exams = load_exams(ROOT / "external_data/ege/exam_calendar.csv")
        cls.configs = {
            d: DegreeConfig(
                iterations=20,
                family="spline" if d == "background" else "catboost",
                level_method="mean",
                exams=d == "bachelor",
            )
            for d in DEGREES
        }

    def test_master_event_features_do_not_depend_on_bachelor_deadline(self):
        dates = campaign(2024)
        first = {(2024, "bachelor"): pd.Timestamp("2024-07-25")}
        second = {(2024, "bachelor"): pd.Timestamp("2024-07-01")}
        pd.testing.assert_frame_equal(
            features(dates, "master", first), features(dates, "master", second)
        )
        self.assertNotEqual(
            features(dates, "master", first, "calendar").distance_bachelor.iloc[0],
            features(dates, "master", second, "calendar").distance_bachelor.iloc[0],
        )
        last = features(pd.DatetimeIndex(["2024-07-26", "2024-07-29", "2024-07-30"]), "master")
        self.assertEqual(last.distance.tolist(), [-3, 0, 1])
        self.assertEqual(last.last_day.tolist(), [0, 1, 0])

    def test_deadline_profile_transfers_to_future_last_day(self):
        counts = self.counts.copy()
        for year in sorted(set(counts.index.year)):
            dates = campaign(year)
            for degree in DEGREES:
                delta = (dates - nominal_deadline(year, degree)).days
                values = np.where(delta <= 0, 100.0, 5.0)
                values = np.where((delta <= 0) & (delta >= -3), 200.0, values)
                counts.loc[dates, degree] = np.where(dates.dayofweek == 6, 0.0, values)
        model = DegreeModel("master", DegreeConfig(iterations=200, coordinates="event")).fit(
            counts, 2024
        )
        p = model.predict()
        self.assertGreater(p.loc["2024-07-29"], 5 * p.loc["2024-07-30"])
        self.assertGreater(p.loc["2024-07-26"], 5 * p.loc["2024-07-30"])

    def test_future_observations_cannot_change_training_or_inference(self):
        first = AdmissionsModel(self.configs).fit(self.counts, 2024, self.exams)
        future = pd.DataFrame(999999.0, index=campaign(2024), columns=DEGREES)
        second = AdmissionsModel(self.configs).fit(
            pd.concat([self.counts, future]), 2024, self.exams
        )
        pd.testing.assert_frame_equal(
            first.predict(components=True), second.predict(components=True)
        )
        self.assertEqual(len({id(m) for m in first.models.values()}), 3)
        self.assertTrue(all(2024 not in m.training_years for m in first.models.values()))
        p = first.predict(components=True)
        np.testing.assert_allclose(p[list(DEGREES)].sum(axis=1), p.forecast)

    def test_cli_trains_without_previous_forecast_files_and_round_trips(self):
        with tempfile.TemporaryDirectory(prefix="v92_clean_") as tmp:
            folder = Path(tmp)
            (folder / "selection.json").write_text(
                json.dumps({"degrees": {d: asdict(c) for d, c in self.configs.items()}})
            )
            base = [sys.executable, str(ROOT / "v92_model.py")]
            subprocess.run(
                base
                + [
                    "train",
                    "--data",
                    str(ROOT / "processed_data/registered.csv"),
                    "--exams",
                    str(ROOT / "external_data/ege/exam_calendar.csv"),
                    "--selection",
                    str(folder / "selection.json"),
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
                base
                + [
                    "predict",
                    "--data",
                    "/absent",
                    "--exams",
                    "/absent",
                    "--selection",
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
            p = AdmissionsModel.load(folder / "model.joblib").predict(components=True)
            self.assertEqual(len(p), 73)
            self.assertTrue(np.isfinite(p[list(DEGREES) + ["forecast"]]).all().all())


if __name__ == "__main__":
    unittest.main()
