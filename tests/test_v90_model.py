"""Historical supervision, time boundaries and standalone training."""

from dataclasses import asdict
import json
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

from v90_model import (
    AdmissionsModel,
    DEGREES,
    ModelConfig,
    degree_targets,
    fit_global_scale,
    load_counts,
    load_exam_calendar,
)


class HistoricalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = load_counts(ROOT / "processed_data/registered.csv")
        cls.exams = load_exam_calendar(ROOT / "external_data/ege/exam_calendar.csv")
        cls.configs = {d: ModelConfig(iterations=20) for d in DEGREES}

    def test_separate_supervision_and_no_future_label_effect(self):
        targets = []
        original = CatBoostRegressor.fit

        def tracked(model, frame, target, **kwargs):
            targets.append(np.asarray(target).copy())
            return original(model, frame, target, **kwargs)

        with patch.object(CatBoostRegressor, "fit", tracked):
            first = AdmissionsModel(self.configs).fit(self.counts, 2024, self.exams)
        history = self.counts[(self.counts.index.year != 2020) & (self.counts.index.dayofweek != 6)]
        for degree, observed in zip(DEGREES, targets):
            np.testing.assert_array_equal(observed, np.log1p(degree_targets(history)[degree]))
        self.assertEqual(len({id(m.regressor) for m in first.models.values()}), 3)
        future = pd.DataFrame(
            999999.0,
            index=pd.date_range("2024-06-20", "2024-08-31", name="date"),
            columns=self.counts.columns,
        )
        second = AdmissionsModel(self.configs).fit(
            pd.concat([self.counts, future]), 2024, self.exams
        )
        pd.testing.assert_frame_equal(first.predict(), second.predict())

    def test_scalar_recovers_known_scale_and_preserves_components(self):
        predicted = np.array([0.0, 2.0, 5.0, 100.0])
        self.assertAlmostEqual(fit_global_scale(predicted, predicted * 1.7), 1.7, places=5)
        model = AdmissionsModel(self.configs).fit(self.counts, 2024, self.exams)
        before = model.predict(components=True)
        historical = pd.DataFrame(
            {
                "date": pd.date_range("2023-06-20", periods=4),
                "forecast": predicted,
                "actual": predicted * 1.7,
            }
        )
        model.calibrate(historical)
        after = model.predict(components=True)
        np.testing.assert_allclose(after[list(DEGREES)], before[list(DEGREES)] * model.scale)
        np.testing.assert_allclose(after[list(DEGREES)].sum(axis=1), after.forecast)

    def test_calibration_rejects_current_year_and_duplicate_dates(self):
        model = AdmissionsModel(self.configs).fit(self.counts, 2024, self.exams)
        bad = pd.DataFrame({"date": ["2024-06-20"], "forecast": [10], "actual": [12]})
        with self.assertRaisesRegex(ValueError, "strictly before"):
            model.calibrate(bad)
        bad = pd.concat([bad.assign(date="2023-06-20")] * 2)
        with self.assertRaisesRegex(ValueError, "unique dates"):
            model.calibrate(bad)
        with self.assertRaises(ValueError):
            fit_global_scale([1.0], [-1.0])

    def test_forest_families_round_trip_with_same_features(self):
        for family in ("forest", "extra"):
            configs = {d: ModelConfig(family=family, iterations=12) for d in DEGREES}
            model = AdmissionsModel(configs).fit(self.counts, 2024, self.exams)
            with tempfile.TemporaryDirectory() as tmp:
                model.save(Path(tmp) / "model.joblib")
                loaded = AdmissionsModel.load(Path(tmp) / "model.joblib")
                pd.testing.assert_frame_equal(model.predict(), loaded.predict())
                self.assertTrue(
                    all((Path(tmp) / f"{degree}.joblib").exists() for degree in DEGREES)
                )

    def test_cli_trains_in_directory_with_no_previous_forecasts(self):
        with tempfile.TemporaryDirectory(prefix="v90_independent_") as tmp:
            folder = Path(tmp)
            selection = {
                "degrees": {d: asdict(c) for d, c in self.configs.items()},
                "calibration": "historical",
                "validation_years": [2017, 2018],
            }
            (folder / "selection.json").write_text(json.dumps(selection))
            common = [sys.executable, str(ROOT / "v90_model.py")]
            subprocess.run(
                common
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
                common
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
            model = AdmissionsModel.load(folder / "model.joblib")
            self.assertEqual(model.scale_provenance["years"], [2017, 2018])
            self.assertTrue(all((folder / f"{degree}.cbm").exists() for degree in DEGREES))


if __name__ == "__main__":
    unittest.main()
