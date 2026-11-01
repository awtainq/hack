"""Separate degree models with an explicit V84-guided calibration stage.

Stage 1 fits bachelor, master and remaining registrations independently using
historical rows only. Stage 2 learns a smooth shared correction to their sum
from a reference forecast. It is teacher-guided approximation, not independent
validation against the unobserved 2024 registrations.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.linalg import solve
from sklearn.preprocessing import SplineTransformer

# Shared raw-data and calendar utilities only; no V88 weights or forecasts.
from v88_model import (
    CATEGORIES,
    activity_breaks,
    annual_features,
    calendar,
    campaign,
    load_counts,
    load_exam_calendar,
)


DEGREES = ("bachelor", "master", "background")


@dataclass(frozen=True)
class DegreeConfig:
    depth: int = 4
    iterations: int = 500
    half_life: float = 3.0
    correction_knots: int = 8
    correction_penalty: float = 0.1

    def __post_init__(self):
        if (
            min(self.depth, self.iterations, self.half_life, self.correction_penalty) <= 0
            or self.correction_knots < 3
        ):
            raise ValueError("Invalid model configuration")


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SeparateDegreeForecaster:
    def __init__(self, config=DegreeConfig()):
        self.config = config

    def features(self, dates, training=False):
        dates = pd.DatetimeIndex(dates)
        frame = calendar(dates, self.breaks if training else None)
        frame = frame.join(annual_features(dates, self.history))
        if self.exam_dates is not None:
            for subject in self.exam_dates:
                exams = pd.DatetimeIndex(self.exam_dates.loc[dates.year, subject])
                frame[f"since_exam_{subject}"] = np.clip((dates - exams).days, -14, 35)
        return frame

    def fit(self, counts, year, exam_dates=None):
        history = counts.loc[counts.index.year < year].sort_index().copy()
        years = sorted(history.index.year.unique())
        if len(years) < 2 or years != list(range(years[0], year)):
            raise ValueError("Need contiguous preceding seasons")
        expected = pd.DatetimeIndex(
            np.concatenate([campaign(y).values for y in years]), name="date"
        )
        if (
            not history.index.equals(expected)
            or not np.isfinite(history.to_numpy()).all()
            or (history < 0).any().any()
        ):
            raise ValueError("Invalid or incomplete counts")
        if not np.allclose(
            history.total, history[["bachelor", "master", "college", "other"]].sum(axis=1)
        ):
            raise ValueError("Education partition does not reconcile")
        self.year = int(year)
        self.history = history.loc[history.index.year != 2020].copy()
        self.training_years = sorted(set(map(int, self.history.index.year)))
        self.exam_dates = None
        if exam_dates is not None:
            if (set(self.training_years) | {year}) - set(exam_dates.index):
                raise ValueError("Missing exam calendar")
            self.exam_dates = exam_dates.loc[exam_dates.index <= year].copy()
        self.breaks = activity_breaks(self.history)
        frame = self.features(self.history.index, training=True)
        weights = np.exp2(-(year - self.history.index.year.to_numpy()) / self.config.half_life)
        schedule = calendar(self.history.index)
        self.closure_rates = {}
        for slot in ("first", "late", "august", "regular"):
            use = schedule.saturday_slot.eq(slot)
            self.closure_rates[slot] = float(
                np.average(self.history.total[use].eq(0), weights=weights[use])
            )
        targets = self.history[["bachelor", "master"]].copy()
        targets["background"] = self.history.college + self.history.other
        use = self.history.index.dayofweek != 6
        self.models = {}
        for index, degree in enumerate(DEGREES):
            model = CatBoostRegressor(
                loss_function="RMSE",
                depth=self.config.depth,
                iterations=self.config.iterations,
                learning_rate=0.035,
                l2_leaf_reg=10,
                cat_features=list(CATEGORIES),
                one_hot_max_size=16,
                allow_const_label=True,
                random_seed=89001 + index,
                thread_count=1,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(
                frame.loc[use], np.log1p(targets.loc[use, degree]), sample_weight=weights[use]
            )
            self.models[degree] = model
        self.feature_names = list(frame.columns)
        self.calibrator = None
        self.reference_provenance = None
        return self

    def base_prediction(self):
        dates = campaign(self.year)
        frame = self.features(dates)
        counts = np.column_stack(
            [np.maximum(0.0, np.expm1(self.models[degree].predict(frame))) for degree in DEGREES]
        )
        schedule = calendar(dates)
        closed = (dates.dayofweek == 6) | schedule.saturday_slot.map(self.closure_rates).gt(
            0.5
        ).to_numpy()
        counts[closed] = 0.0
        return pd.DataFrame(counts, index=dates, columns=DEGREES)

    def calibration_features(self, dates, fit=False):
        dates = pd.DatetimeIndex(dates)
        position = (
            (dates - pd.to_datetime(dates.year.astype(str) + "-06-20")).days / 72
        ).to_numpy()[:, None]
        if fit:
            self.spline = SplineTransformer(
                n_knots=self.config.correction_knots,
                degree=3,
                include_bias=False,
                extrapolation="linear",
            )
            smooth = self.spline.fit_transform(position)
        else:
            smooth = self.spline.transform(position)
        weekday = np.column_stack([(dates.dayofweek == day).astype(float) for day in range(1, 6)])
        frame = calendar(dates)
        phase = np.column_stack([frame.distance_bachelor.gt(0), frame.distance_master.gt(0)])
        # Common season curve, weekday offsets and weekday-specific season curves.
        interactions = (smooth[:, :, None] * weekday[:, None, :]).reshape(len(dates), -1)
        return np.column_stack([smooth, weekday, phase, interactions])

    def adapt_to_reference(self, reference, provenance=None):
        if list(reference.columns) != ["date", "forecast"]:
            raise ValueError("Reference must have date,forecast columns")
        reference = reference.copy()
        reference["date"] = pd.to_datetime(reference.date, errors="raise")
        if not np.array_equal(reference.date, campaign(self.year)):
            raise ValueError("Reference dates do not match forecast campaign")
        values = reference.forecast.to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid reference predictions")
        base = self.base_prediction().sum(axis=1).to_numpy()
        if ((base == 0) & (values > 0)).any():
            raise ValueError("Reference has activity on dates closed by the learned schedule")
        use = base > 0
        design = self.calibration_features(reference.date, fit=True)
        target = np.log1p(values) - np.log1p(base)
        matrix, response = design[use], target[use]
        center, mean_response = matrix.mean(axis=0), float(response.mean())
        centered = matrix - center
        gram = np.einsum("ni,nj->ij", centered, centered)
        gram += self.config.correction_penalty * np.eye(matrix.shape[1])
        rhs = np.einsum("ni,n->i", centered, response - mean_response)
        coefficients = solve(gram, rhs, assume_a="pos")
        intercept = mean_response - float(np.einsum("i,i->", center, coefficients))
        self.calibrator = {"coefficients": coefficients, "intercept": intercept}
        self.reference_provenance = {
            "role": "teacher_for_target_campaign",
            "independent_of_reference": False,
            "reference_year": self.year,
            "reference_rows_used": int(use.sum()),
            **(provenance or {}),
        }
        return self

    def predict(self, components=False):
        base = self.base_prediction()
        raw_total = base.sum(axis=1).to_numpy()
        if self.calibrator is not None:
            delta = np.einsum(
                "ni,i->n", self.calibration_features(base.index), self.calibrator["coefficients"]
            )
            delta += self.calibrator["intercept"]
            corrected = np.where(
                raw_total > 0, np.maximum(0.0, np.expm1(np.log1p(raw_total) + delta)), 0.0
            )
            multiplier = np.divide(
                corrected, raw_total, out=np.zeros_like(corrected), where=raw_total > 0
            )
            base *= multiplier[:, None]
        if not np.isfinite(base.to_numpy()).all():
            raise ValueError("Nonfinite forecast")
        base["forecast"] = base.sum(axis=1)
        result = base.reset_index()
        return result if components else result[["date", "forecast"]]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": asdict(self.config),
                "state": {k: v for k, v in vars(self).items() if k != "config"},
            },
            path,
        )
        for degree, model in self.models.items():
            model.save_model(str(path.parent / f"{degree}_base.cbm"))

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        model = cls(DegreeConfig(**payload["config"]))
        vars(model).update(payload["state"])
        return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data", type=Path, default=Path("processed_data/registered.csv"))
    parser.add_argument("--exams", type=Path, default=Path("external_data/ege/exam_calendar.csv"))
    parser.add_argument("--reference", type=Path, default=Path("forecast_2024_v84.csv"))
    parser.add_argument("--historical-only", action="store_true")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()
    model_path = args.model or Path(
        "output/v89/historical_model.joblib" if args.historical_only else "output/v89/model.joblib"
    )
    if args.command == "train":
        config = (
            DegreeConfig(**json.loads(args.config.read_text())) if args.config else DegreeConfig()
        )
        model = SeparateDegreeForecaster(config).fit(
            load_counts(args.data), args.year, load_exam_calendar(args.exams)
        )
        if not args.historical_only:
            reference = pd.read_csv(args.reference, parse_dates=["date"])
            model.adapt_to_reference(
                reference, {"file": str(args.reference), "sha256": checksum(args.reference)}
            )
        model.save(model_path)
    else:
        model = SeparateDegreeForecaster.load(model_path)
    guided = model.calibrator is not None
    output = args.output or Path(
        f"forecast_{model.year}_v89{'_historical' if not guided else ''}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction = model.predict(components=True)
    prediction[["date", "forecast"]].to_csv(output, index=False, float_format="%.6f")
    component_name = "degree_forecasts.csv" if guided else "degree_forecasts_historical.csv"
    prediction.to_csv(model_path.parent / component_name, index=False, float_format="%.6f")
    print(
        json.dumps(
            {
                "output": str(output),
                "totals": prediction[list(DEGREES) + ["forecast"]].sum().to_dict(),
                "reference_guided": model.calibrator is not None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
