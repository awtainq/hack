"""Reference-guided selection is separate from historical fitting and inference."""

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

from admissions_calendar import DEGREES, campaign, load_counts, load_exams
from tune_v93 import select_combination
from v93_model import AdmissionsModel, DegreeConfig


class TunedParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.counts = load_counts(ROOT / "processed_data/registered.csv")
        cls.exams = load_exams(ROOT / "external_data/ege/exam_calendar.csv")
        cls.configs = {
            "bachelor": DegreeConfig(
                iterations=20, history_window=3, level_method="mean", exams=True
            ),
            "master": DegreeConfig(iterations=20, history_window=5, positive_only=True, loss="MAE"),
            "background": DegreeConfig(family="spline", history_window=3, level_method="trend"),
        }

    def test_reference_changes_selection_and_is_not_independent_validation(self):
        candidates = {
            "bachelor": np.array([[0.0, 0.0], [8.0, 0.0]]),
            "master": np.array([[0.0, 0.0], [0.0, 8.0]]),
            "background": np.array([[0.0, 0.0], [1.0, 1.0]]),
        }
        (first_score, first), _ = select_combination(candidates, np.array([8.0, 8.0]))
        (second_score, second), _ = select_combination(candidates, np.array([1.0, 1.0]))
        self.assertEqual(first_score, 0.0)
        self.assertEqual(second_score, 0.0)
        self.assertNotEqual(first, second)

    def test_fixed_configuration_ignores_future_observations(self):
        first = AdmissionsModel(self.configs).fit(self.counts, 2024, self.exams)
        altered = pd.concat(
            [self.counts, pd.DataFrame(999999.0, index=campaign(2024), columns=DEGREES)]
        )
        second = AdmissionsModel(self.configs).fit(altered, 2024, self.exams)
        pd.testing.assert_frame_equal(
            first.predict(components=True), second.predict(components=True)
        )
        self.assertEqual(first.models["bachelor"].training_years, [2021, 2022, 2023])
        self.assertEqual(len({id(m) for m in first.models.values()}), 3)

    def test_selection_preserves_reference_closed_days_without_runtime_date_rules(self):
        candidates = {
            "bachelor": np.array([[0.1, 8.0], [0.0, 7.0]]),
            "master": np.array([[0.0, 0.0]]),
            "background": np.array([[0.0, 0.0]]),
        }
        (_, selected), _ = select_combination(candidates, np.array([0.0, 8.0]))
        self.assertEqual(selected["bachelor"], 1)

    def test_cli_fits_fixed_parameters_without_reference_and_round_trips(self):
        with tempfile.TemporaryDirectory(prefix="v93_historical_") as tmp:
            folder = Path(tmp)
            (folder / "parameters.json").write_text(
                json.dumps({"degrees": {d: asdict(c) for d, c in self.configs.items()}})
            )
            base = [sys.executable, str(ROOT / "v93_model.py")]
            subprocess.run(
                base
                + [
                    "train",
                    "--data",
                    str(ROOT / "processed_data/registered.csv"),
                    "--exams",
                    str(ROOT / "external_data/ege/exam_calendar.csv"),
                    "--selection",
                    str(folder / "parameters.json"),
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
                    "--selection",
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
            p = AdmissionsModel.load(folder / "model.joblib").predict(components=True)
            np.testing.assert_allclose(p[list(DEGREES)].sum(axis=1), p.forecast)
            self.assertEqual(len(p), 73)
            # Each export must match the already-trained component, and leave
            # the verified total forecast untouched.
            original = (folder / "trained.csv").read_bytes()
            for degree in [*DEGREES, "all"]:
                subprocess.run(
                    base + ["predict", "--model", str(folder / "model.joblib"), "--degree", degree],
                    cwd=folder,
                    check=True,
                    capture_output=True,
                )
                suffix = "components" if degree == "all" else degree
                exported = pd.read_csv(folder / f"forecast_2024_v93_{suffix}.csv")
                if degree == "all":
                    self.assertEqual(list(exported.columns), ["date", *DEGREES, "forecast"])
                    np.testing.assert_allclose(exported.forecast, p.forecast, atol=5e-7)
                else:
                    self.assertEqual(list(exported.columns), ["date", "forecast"])
                    np.testing.assert_allclose(exported.forecast, p[degree], atol=5e-7)
                self.assertEqual((folder / "trained.csv").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
