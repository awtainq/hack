"""Independent degree forecasts and a scalar learned from historical errors.

Training and prediction never read previous forecasts or leaderboard scores.
Each degree has its own model and configuration. The optional scalar is fitted
to rolling predictions of preceding campaigns, using the same daily target.
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
from scipy.optimize import minimize_scalar
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

# Pure preparation functions only; no earlier model weights or predictions.
from v88_model import (
    CATEGORIES,
    GROUPS,
    activity_breaks,
    annual_features,
    calendar,
    campaign,
    load_counts,
    load_exam_calendar,
)

DEGREES = ("bachelor", "master", "background")


@dataclass(frozen=True)
class ModelConfig:
    family: str = "catboost"
    depth: int = 3
    iterations: int = 300
    half_life: float = 3.0
    feature_set: str = "exams"
    align_history: bool = False
    min_leaf: int = 6

    def __post_init__(self):
        if self.depth < 1 or self.iterations < 1 or self.half_life <= 0:
            raise ValueError("Invalid tree or weighting parameters")
        if self.feature_set not in {"calendar", "annual", "exams"}:
            raise ValueError("Unknown feature set")
        if self.family not in {"catboost", "forest", "extra"} or self.min_leaf < 1:
            raise ValueError("Unknown model family or invalid leaf size")


def historical_counts(counts, year):
    history = counts.loc[counts.index.year < year, list(GROUPS)].sort_index().copy()
    years = sorted(history.index.year.unique())
    if len(years) < 2 or years != list(range(years[0], year)):
        raise ValueError("Need contiguous preceding campaigns")
    expected = pd.DatetimeIndex(np.concatenate([campaign(y).values for y in years]), name="date")
    values = history.to_numpy()
    if not history.index.equals(expected) or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid or incomplete counts")
    if not np.allclose(
        history.total, history[["bachelor", "master", "college", "other"]].sum(axis=1)
    ):
        raise ValueError("Degree counts do not reconcile")
    return history.loc[history.index.year != 2020]


def degree_targets(counts):
    result = counts[["bachelor", "master"]].copy()
    result["background"] = counts.college + counts.other
    return result


def fit_global_scale(predicted, actual, weights=None):
    """Fit one nonnegative multiplier by daily log1p squared error."""
    predicted, actual = np.asarray(predicted, float), np.asarray(actual, float)
    weights = np.ones_like(actual) if weights is None else np.asarray(weights, float)
    if (
        predicted.shape != actual.shape
        or actual.shape != weights.shape
        or not np.isfinite([predicted, actual, weights]).all()
        or (predicted < 0).any()
        or (actual < 0).any()
        or (weights <= 0).any()
        or not (predicted > 0).any()
    ):
        raise ValueError("Invalid historical calibration samples")

    def objective(log_scale):
        residual = np.log1p(predicted * np.exp(log_scale)) - np.log1p(actual)
        return float(np.average(residual**2, weights=weights))

    solution = minimize_scalar(objective, bounds=(-5, 5), method="bounded")
    if not solution.success:
        raise ValueError("Historical scale fitting failed")
    return float(np.exp(solution.x))


class DegreeModel:
    """One independently supervised degree stream."""

    def __init__(self, degree, config=ModelConfig()):
        if degree not in DEGREES:
            raise ValueError("Unknown degree")
        self.degree, self.config = degree, config

    def features(self, dates, training=False):
        dates = pd.DatetimeIndex(dates)
        breaks = self.breaks if training and self.config.align_history else None
        result = calendar(dates, breaks)
        if self.config.feature_set != "calendar":
            result = result.join(annual_features(dates, self.history))
        if self.config.feature_set == "exams":
            for subject in self.exams:
                exam_dates = pd.DatetimeIndex(self.exams.loc[dates.year, subject])
                result[f"since_exam_{subject}"] = np.clip((dates - exam_dates).days, -14, 35)
        if self.config.family != "catboost":
            for column, categories in CATEGORIES.items():
                result[column] = pd.Categorical(result[column], categories=categories)
            result = pd.get_dummies(result, columns=list(CATEGORIES)).fillna(-1.0)
        return result

    def fit(self, counts, year, exams=None):
        self.year = int(year)
        self.history = historical_counts(counts, year)
        self.training_years = sorted(set(map(int, self.history.index.year)))
        self.exams = None
        if self.config.feature_set == "exams":
            if exams is None or (set(self.training_years) | {year}) - set(exams.index):
                raise ValueError("Missing exam calendar")
            self.exams = exams.loc[exams.index <= year].copy()
        self.breaks = activity_breaks(self.history) if self.config.align_history else {}
        weights = np.exp2(-(year - self.history.index.year.to_numpy()) / self.config.half_life)
        schedule = calendar(self.history.index)
        self.closure_rates = {}
        for slot in ("first", "late", "august", "regular"):
            use = schedule.saturday_slot.eq(slot)
            self.closure_rates[slot] = float(
                np.average(self.history.total[use].eq(0), weights=weights[use])
            )
        frame = self.features(self.history.index, training=True)
        target = np.log1p(degree_targets(self.history)[self.degree])
        use = self.history.index.dayofweek != 6
        if self.config.family == "catboost":
            self.regressor = CatBoostRegressor(
                loss_function="RMSE",
                depth=self.config.depth,
                iterations=self.config.iterations,
                learning_rate=0.035,
                l2_leaf_reg=10,
                cat_features=list(CATEGORIES),
                one_hot_max_size=16,
                allow_const_label=True,
                random_seed=90001 + DEGREES.index(self.degree),
                thread_count=1,
                verbose=False,
                allow_writing_files=False,
            )
        else:
            factory = (
                RandomForestRegressor if self.config.family == "forest" else ExtraTreesRegressor
            )
            self.regressor = factory(
                n_estimators=self.config.iterations,
                min_samples_leaf=self.config.min_leaf,
                max_features=1.0,
                random_state=90001 + DEGREES.index(self.degree),
                n_jobs=1,
            )
        self.regressor.fit(frame.loc[use], target.loc[use], sample_weight=weights[use])
        self.feature_names = list(frame.columns)
        return self

    def predict(self):
        dates = campaign(self.year)
        prediction = np.maximum(0.0, np.expm1(self.regressor.predict(self.features(dates))))
        schedule = calendar(dates)
        closed = (dates.dayofweek == 6) | schedule.saturday_slot.map(self.closure_rates).gt(
            0.5
        ).to_numpy()
        prediction[closed] = 0.0
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite degree forecast")
        return pd.Series(prediction, index=dates, name=self.degree)


class AdmissionsModel:
    def __init__(self, configs):
        if set(configs) != set(DEGREES):
            raise ValueError("Need one configuration per degree")
        self.configs = dict(configs)
        self.scale = 1.0
        self.scale_provenance = {"method": "none", "years": []}

    def fit(self, counts, year, exams=None):
        self.year = int(year)
        self.models = {
            degree: DegreeModel(degree, self.configs[degree]).fit(counts, year, exams)
            for degree in DEGREES
        }
        self.scale = 1.0
        self.scale_provenance = {"method": "none", "years": []}
        return self

    def calibrate(self, historical_predictions):
        required = {"date", "forecast", "actual"}
        if not required.issubset(historical_predictions.columns):
            raise ValueError("Need dated historical predictions and observations")
        frame = historical_predictions.copy()
        dates = pd.DatetimeIndex(pd.to_datetime(frame.date, errors="raise"))
        if not dates.is_unique or (dates.year >= self.year).any():
            raise ValueError("Calibration requires unique dates strictly before forecast year")
        weights = np.exp2(-(self.year - dates.year.to_numpy()) / 3.0)
        self.scale = fit_global_scale(frame.forecast, frame.actual, weights)
        self.scale_provenance = {
            "method": "rolling historical log1p least squares",
            "years": sorted(set(map(int, dates.year))),
            "rows": len(frame),
        }
        return self

    def predict(self, components=False):
        result = pd.concat([self.models[d].predict() for d in DEGREES], axis=1) * self.scale
        result["forecast"] = result.sum(axis=1)
        result = result.reset_index()
        return result if components else result[["date", "forecast"]]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Store state, rather than an __main__ class instance, for portable CLI loading.
        state = {
            "year": self.year,
            "scale": self.scale,
            "scale_provenance": self.scale_provenance,
            "configs": {d: asdict(c) for d, c in self.configs.items()},
            "models": {
                d: {k: v for k, v in vars(m).items() if k not in {"config", "degree"}}
                for d, m in self.models.items()
            },
        }
        joblib.dump(state, path)
        for degree, model in self.models.items():
            joblib.dump(model.regressor, path.parent / f"{degree}.joblib")
            if model.config.family == "catboost":
                model.regressor.save_model(str(path.parent / f"{degree}.cbm"))
            else:
                (path.parent / f"{degree}.cbm").unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        state = joblib.load(path)
        model = cls({d: ModelConfig(**c) for d, c in state["configs"].items()})
        model.year, model.scale, model.scale_provenance = (
            state["year"],
            state["scale"],
            state["scale_provenance"],
        )
        model.models = {}
        for degree, values in state["models"].items():
            stream = DegreeModel(degree, model.configs[degree])
            vars(stream).update(values)
            model.models[degree] = stream
        return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--data", type=Path, default=Path("processed_data/registered.csv"))
    parser.add_argument("--exams", type=Path, default=Path("external_data/ege/exam_calendar.csv"))
    parser.add_argument("--selection", type=Path, default=Path("output/v90/selected_config.json"))
    parser.add_argument("--model", type=Path, default=Path("output/v90/model.joblib"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        selection = json.loads(args.selection.read_text())
        configs = {d: ModelConfig(**selection["degrees"][d]) for d in DEGREES}
        counts, exams = load_counts(args.data), load_exam_calendar(args.exams)
        model = AdmissionsModel(configs).fit(counts, args.year, exams)
        if selection["calibration"] == "historical":
            folds = []
            for year in selection["validation_years"]:
                if year >= args.year:
                    continue
                forecast = AdmissionsModel(configs).fit(counts, year, exams).predict()
                forecast["actual"] = counts.loc[forecast.date, "total"].to_numpy()
                folds.append(forecast)
            model.calibrate(pd.concat(folds, ignore_index=True))
        model.save(args.model)
    else:
        model = AdmissionsModel.load(args.model)
    output = args.output or Path(f"forecast_{model.year}_v90.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction = model.predict(components=True)
    prediction[["date", "forecast"]].to_csv(output, index=False, float_format="%.6f")
    prediction.to_csv(args.model.parent / "degree_forecasts.csv", index=False, float_format="%.6f")
    print(
        json.dumps(
            {
                "output": str(output),
                "scale": model.scale,
                "totals": prediction[list(DEGREES) + ["forecast"]].sum().to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
