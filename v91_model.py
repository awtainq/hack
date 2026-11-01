"""Separate degree forecasts with an explicitly learned annual intensity.

Historical daily log counts are decomposed into a campaign level and a
calendar profile. Both the level extrapolation and the profile use preceding
campaigns only. No previous forecast CSV or leaderboard score is an input.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import SplineTransformer

from v88_model import (
    CATEGORIES,
    activity_breaks,
    calendar,
    campaign,
    load_counts,
    load_exam_calendar,
)
from v90_model import DEGREES, degree_targets, historical_counts


@dataclass(frozen=True)
class ProfileConfig:
    family: str = "spline"
    half_life: float = 3.0
    level_method: str = "none"
    level_window: int = 3
    knots: int = 9
    penalty: float = 3.0
    depth: int = 3
    iterations: int = 700
    min_leaf: int = 8
    align_history: bool = False
    exams: bool = False

    def __post_init__(self):
        if self.family not in {"spline", "catboost", "extra"}:
            raise ValueError("Unknown profile family")
        if self.level_method not in {"none", "mean", "trend"}:
            raise ValueError("Unknown annual intensity method")
        if (
            min(self.half_life, self.penalty, self.depth, self.iterations, self.min_leaf) <= 0
            or self.level_window < 2
            or self.knots < 3
        ):
            raise ValueError("Invalid profile configuration")


def forecast_level(levels, year, config):
    """Ridge trend of preceding weekday mean log counts; no future values."""
    if config.level_method == "none":
        return 0.0
    recent = levels.loc[levels.index < year].tail(config.level_window)
    if recent.empty:
        raise ValueError("Missing previous intensity levels")
    years, values = recent.index.to_numpy(float), recent.to_numpy(float)
    weights = np.exp2(-(year - years) / config.half_life)
    center = float(np.average(years, weights=weights))
    mean = float(np.average(values, weights=weights))
    if config.level_method == "mean":
        return mean
    # A unit ridge penalty shrinks uncertain trends toward a constant level.
    slope = np.sum(weights * (years - center) * (values - mean))
    slope /= np.sum(weights * (years - center) ** 2) + 1.0
    return max(0.0, mean + float(slope) * (year - center))


class ProfileModel:
    def __init__(self, degree, config=ProfileConfig()):
        if degree not in DEGREES:
            raise ValueError("Unknown degree")
        self.degree, self.config = degree, config

    def frame(self, dates, training=False):
        dates = pd.DatetimeIndex(dates)
        result = calendar(dates, self.breaks if training else None)
        if self.config.exams:
            for subject in self.exams:
                exams = pd.DatetimeIndex(self.exams.loc[dates.year, subject])
                result[f"since_exam_{subject}"] = np.clip((dates - exams).days, -14, 35)
        return result

    def design(self, frame, fit=False):
        if self.config.family == "catboost":
            return frame
        if self.config.family == "extra":
            result = frame.copy()
            for column, categories in CATEGORIES.items():
                result[column] = pd.Categorical(result[column], categories=categories)
            return pd.get_dummies(result, columns=list(CATEGORIES)).fillna(-1.0)
        values = frame[["position", "distance_bachelor", "distance_master"]].to_numpy()
        if fit:
            # Fixed campaign-domain knots, independent of observed daily counts.
            knots = np.column_stack(
                [
                    np.linspace(0, 72, self.config.knots),
                    np.linspace(-45, 45, self.config.knots),
                    np.linspace(-45, 45, self.config.knots),
                ]
            )
            self.spline = SplineTransformer(
                knots=knots, degree=3, include_bias=False, extrapolation="linear"
            ).fit(values)
        smooth = self.spline.transform(values)
        weekday = np.column_stack([frame.weekday.eq(str(d)).to_numpy(float) for d in range(1, 6)])
        phase = np.column_stack(
            [frame.phase.eq(p).to_numpy(float) for p in CATEGORIES["phase"][1:]]
        )
        width = self.config.knots + 1
        interactions = (smooth[:, :width, None] * weekday[:, None, :]).reshape(len(frame), -1)
        extras = frame[["first_day", "after_bachelor", "after_master"]].to_numpy()
        columns = [np.ones((len(frame), 1)), smooth, weekday, phase, interactions, extras]
        if self.config.exams:
            columns.append(frame.filter(like="since_exam_").to_numpy() / 35)
        return np.column_stack(columns)

    def fit(self, counts, year, exams=None):
        self.year = int(year)
        history = historical_counts(counts, year)
        self.training_years = sorted(set(map(int, history.index.year)))
        self.breaks = activity_breaks(history) if self.config.align_history else {}
        self.exams = None
        if self.config.exams:
            if exams is None or (set(self.training_years) | {year}) - set(exams.index):
                raise ValueError("Missing exam dates")
            self.exams = exams.loc[exams.index <= year].copy()
        weights = np.exp2(-(year - history.index.year.to_numpy()) / self.config.half_life)
        schedule = calendar(history.index)
        self.closure_rates = {}
        for slot in ("first", "late", "august", "regular"):
            use = schedule.saturday_slot.eq(slot)
            self.closure_rates[slot] = float(
                np.average(history.total[use].eq(0), weights=weights[use])
            )
        target = np.log1p(degree_targets(history)[self.degree])
        weekdays = history.index.dayofweek < 5
        self.levels = target.loc[weekdays].groupby(history.index.year[weekdays]).mean()
        self.future_level = forecast_level(self.levels, year, self.config)
        if self.config.level_method != "none":
            target -= history.index.year.map(self.levels).to_numpy()
        use = history.index.dayofweek != 6
        frame = self.frame(history.index, training=True)
        matrix = self.design(frame, fit=True)
        response, weight = target.to_numpy()[use], weights[use]
        if self.config.family == "spline":
            matrix = matrix[use]
            gram = np.einsum("ni,n,nj->ij", matrix, weight, matrix)
            penalty = np.full(matrix.shape[1], self.config.penalty)
            penalty[0] = 1e-8  # Do not shrink the profile intercept.
            gram += np.diag(penalty)
            rhs = np.einsum("ni,n,n->i", matrix, weight, response)
            self.coefficients = solve(gram, rhs, assume_a="pos")
        else:
            if self.config.family == "catboost":
                self.regressor = CatBoostRegressor(
                    iterations=self.config.iterations,
                    depth=self.config.depth,
                    learning_rate=0.035,
                    l2_leaf_reg=10,
                    loss_function="RMSE",
                    cat_features=list(CATEGORIES),
                    one_hot_max_size=16,
                    random_seed=91001 + DEGREES.index(self.degree),
                    thread_count=1,
                    allow_const_label=True,
                    verbose=False,
                    allow_writing_files=False,
                )
            else:
                self.regressor = ExtraTreesRegressor(
                    n_estimators=self.config.iterations,
                    min_samples_leaf=self.config.min_leaf,
                    random_state=91001 + DEGREES.index(self.degree),
                    n_jobs=1,
                )
            self.regressor.fit(matrix.loc[use], response, sample_weight=weight)
        return self

    def predict(self):
        dates = campaign(self.year)
        frame = self.frame(dates)
        matrix = self.design(frame)
        profile = (
            np.einsum("ni,i->n", matrix, self.coefficients)
            if self.config.family == "spline"
            else self.regressor.predict(matrix)
        )
        prediction = np.expm1(np.maximum(0.0, profile + self.future_level))
        schedule = calendar(dates)
        closed = (dates.dayofweek == 6) | schedule.saturday_slot.map(self.closure_rates).gt(
            0.5
        ).to_numpy()
        prediction[closed] = 0.0
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite profile forecast")
        return pd.Series(prediction, index=dates, name=self.degree)


class AdmissionsModel:
    def __init__(self, configs):
        if set(configs) != set(DEGREES):
            raise ValueError("Need one configuration per degree")
        self.configs = dict(configs)

    def fit(self, counts, year, exams=None):
        self.year = int(year)
        self.models = {
            d: ProfileModel(d, self.configs[d]).fit(counts, year, exams) for d in DEGREES
        }
        return self

    def predict(self, components=False):
        result = pd.concat([self.models[d].predict() for d in DEGREES], axis=1)
        result["forecast"] = result.sum(axis=1)
        result = result.reset_index()
        return result if components else result[["date", "forecast"]]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "year": self.year,
            "configs": {d: asdict(c) for d, c in self.configs.items()},
            "models": {
                d: {k: v for k, v in vars(m).items() if k not in {"config", "degree"}}
                for d, m in self.models.items()
            },
        }
        joblib.dump(state, path)

    @classmethod
    def load(cls, path):
        state = joblib.load(path)
        model = cls({d: ProfileConfig(**c) for d, c in state["configs"].items()})
        model.year, model.models = state["year"], {}
        for degree, values in state["models"].items():
            stream = ProfileModel(degree, model.configs[degree])
            vars(stream).update(values)
            model.models[degree] = stream
        return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--data", type=Path, default=Path("processed_data/registered.csv"))
    parser.add_argument("--exams", type=Path, default=Path("external_data/ege/exam_calendar.csv"))
    parser.add_argument("--selection", type=Path, default=Path("output/v91/selected_config.json"))
    parser.add_argument("--model", type=Path, default=Path("output/v91/model.joblib"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        selection = json.loads(args.selection.read_text())
        model = AdmissionsModel({d: ProfileConfig(**selection["degrees"][d]) for d in DEGREES})
        model.fit(load_counts(args.data), args.year, load_exam_calendar(args.exams)).save(
            args.model
        )
    else:
        model = AdmissionsModel.load(args.model)
    output = args.output or Path(f"forecast_{model.year}_v91.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction = model.predict(components=True)
    prediction[["date", "forecast"]].to_csv(output, index=False, float_format="%.6f")
    prediction.to_csv(args.model.parent / "degree_forecasts.csv", index=False, float_format="%.6f")
    print(prediction[list(DEGREES) + ["forecast"]].sum().to_json(indent=2))


if __name__ == "__main__":
    main()
