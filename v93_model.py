"""Daily degree models trained from historical registrations.

The fixed configuration is supplied separately; its selection provenance is
recorded in the experiment report. Training and prediction read no prior forecasts."""

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
from sklearn.preprocessing import SplineTransformer

from admissions_calendar import (
    DEGREES,
    campaign,
    categorical_columns,
    features,
    history_before,
    infer_deadlines,
    load_counts,
    load_exams,
    saturday_slots,
)


@dataclass(frozen=True)
class DegreeConfig:
    family: str = "catboost"
    coordinates: str = "event"
    depth: int = 4
    iterations: int = 700
    half_life: float = 3.0
    align_history: bool = True
    level_method: str = "none"
    level_window: int = 3
    exams: bool = False
    knots: int = 9
    penalty: float = 3.0
    history_window: int = 0
    learning_rate: float = 0.035
    l2_leaf_reg: float = 10.0
    positive_only: bool = False
    loss: str = "RMSE"

    def __post_init__(self):
        if self.family not in {"catboost", "spline"} or self.coordinates not in {
            "event",
            "calendar",
        }:
            raise ValueError("Unknown model family or coordinates")
        if self.level_method not in {"none", "mean", "trend"}:
            raise ValueError("Unknown annual level method")
        if (
            min(self.depth, self.iterations, self.half_life, self.penalty) <= 0
            or self.knots < 3
            or self.level_window < 2
        ):
            raise ValueError("Invalid degree configuration")
        if self.history_window not in {0, 3, 5, 7} or self.loss not in {"RMSE", "MAE"}:
            raise ValueError("Invalid history window or training loss")
        if min(self.learning_rate, self.l2_leaf_reg) <= 0:
            raise ValueError("Invalid boosting parameters")


def future_level(levels, year, config):
    if config.level_method == "none":
        return 0.0
    earlier = levels.loc[levels.index < year].tail(config.level_window)
    if earlier.empty:
        raise ValueError("Missing historical intensity levels")
    years, values = earlier.index.to_numpy(float), earlier.to_numpy(float)
    weight = np.exp2(-(year - years) / config.half_life)
    center = np.average(years, weights=weight)
    mean = np.average(values, weights=weight)
    if config.level_method == "trend":
        slope = np.sum(weight * (years - center) * (values - mean))
        slope /= np.sum(weight * (years - center) ** 2) + 1.0
        mean += slope * (year - center)
    return max(0.0, float(mean))


class DegreeModel:
    def __init__(self, degree, config=DegreeConfig()):
        if degree not in DEGREES:
            raise ValueError("Unknown degree")
        self.degree, self.config = degree, config

    def frame(self, dates, training=False):
        return features(
            dates,
            self.degree,
            self.deadlines if training else None,
            self.config.coordinates,
            self.exams if self.config.exams else None,
        )

    def design(self, frame, fit=False):
        if self.config.family == "catboost":
            return frame
        coordinates = ["opening_ramp", "distance"]
        if "position" in frame:
            coordinates.append("position")
        values = frame[coordinates].to_numpy()
        if fit:
            knots = [np.linspace(0, 14, self.config.knots), np.linspace(-60, 50, self.config.knots)]
            if "position" in frame:
                knots.append(np.linspace(0, 72, self.config.knots))
            self.spline = SplineTransformer(
                knots=np.array(knots).T, degree=3, include_bias=False, extrapolation="linear"
            ).fit(values)
        smooth = self.spline.transform(values)
        encoded = frame.copy()
        for column, categories in categorical_columns().items():
            encoded[column] = pd.Categorical(encoded[column], categories=categories)
        encoded = pd.get_dummies(encoded, columns=list(categorical_columns()))
        # All numeric/calendar effects plus separate weekday profiles around the deadline.
        relative = smooth[:, self.config.knots + 1 : 2 * (self.config.knots + 1)]
        weekday = np.column_stack([frame.weekday.eq(str(i)).to_numpy(float) for i in range(1, 6)])
        interaction = (relative[:, :, None] * weekday[:, None, :]).reshape(len(frame), -1)
        return np.column_stack(
            [np.ones((len(frame), 1)), smooth, encoded.to_numpy(float) / 35, interaction]
        )

    def fit(self, counts, year, exams=None):
        self.year = int(year)
        history = history_before(counts, year)
        if self.config.history_window:
            preceding = sorted(set(history.index.year))[-self.config.history_window :]
            history = history.loc[history.index.year.isin(preceding)].copy()
        self.training_years = sorted(set(map(int, history.index.year)))
        self.deadlines = infer_deadlines(history) if self.config.align_history else {}
        self.exams = None
        if self.config.exams and self.degree == "bachelor":
            if exams is None or (set(self.training_years) | {year}) - set(exams.index):
                raise ValueError("Missing exam dates")
            self.exams = exams.loc[exams.index <= year].copy()
        weights = np.exp2(-(year - history.index.year.to_numpy()) / self.config.half_life)
        slots = saturday_slots(history.index)
        total = history.sum(axis=1)
        self.closure_rates = {
            slot: float(np.average(total[slots == slot].eq(0), weights=weights[slots == slot]))
            for slot in ("first", "late", "august", "regular")
        }
        target = np.log1p(history[self.degree])
        weekdays = history.index.dayofweek < 5
        self.levels = target[weekdays].groupby(history.index.year[weekdays]).mean()
        self.level = future_level(self.levels, year, self.config)
        if self.config.level_method != "none":
            target -= history.index.year.map(self.levels).to_numpy()
        use = history.index.dayofweek != 6
        if self.config.positive_only:
            use &= history[self.degree].to_numpy() > 0
        frame = self.frame(history.index, training=True)
        design = self.design(frame, fit=True)
        if self.config.family == "catboost":
            self.regressor = CatBoostRegressor(
                iterations=self.config.iterations,
                depth=self.config.depth,
                learning_rate=self.config.learning_rate,
                l2_leaf_reg=self.config.l2_leaf_reg,
                loss_function=self.config.loss,
                cat_features=list(categorical_columns()),
                one_hot_max_size=16,
                random_seed=93001 + DEGREES.index(self.degree),
                thread_count=1,
                allow_const_label=True,
                verbose=False,
                allow_writing_files=False,
            )
            self.regressor.fit(design.loc[use], target.loc[use], sample_weight=weights[use])
        else:
            matrix, response, weight = design[use], target.to_numpy()[use], weights[use]
            gram = np.einsum("ni,n,nj->ij", matrix, weight, matrix)
            penalty = np.full(matrix.shape[1], self.config.penalty)
            penalty[0] = 1e-8
            gram += np.diag(penalty)
            rhs = np.einsum("ni,n,n->i", matrix, weight, response)
            self.coefficients = solve(gram, rhs, assume_a="pos")
        return self

    def predict(self):
        dates = campaign(self.year)
        design = self.design(self.frame(dates))
        log_counts = (
            self.regressor.predict(design)
            if self.config.family == "catboost"
            else np.einsum("ni,i->n", design, self.coefficients)
        ) + self.level
        prediction = np.expm1(np.maximum(0.0, log_counts))
        slots = saturday_slots(dates)
        closed = (dates.dayofweek == 6) | np.array(
            [self.closure_rates.get(s, 0) > 0.5 for s in slots]
        )
        prediction[closed] = 0.0
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite forecast")
        return pd.Series(prediction, index=dates, name=self.degree)


class AdmissionsModel:
    def __init__(self, configs):
        if set(configs) != set(DEGREES):
            raise ValueError("Need one configuration per degree")
        self.configs = dict(configs)

    def fit(self, counts, year, exams=None):
        self.year = int(year)
        self.models = {d: DegreeModel(d, self.configs[d]).fit(counts, year, exams) for d in DEGREES}
        return self

    def predict(self, components=False):
        result = pd.concat([self.models[d].predict() for d in DEGREES], axis=1)
        result["forecast"] = result.sum(axis=1)
        result = result.reset_index()
        return result if components else result[["date", "forecast"]]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "year": self.year,
                "configs": {d: asdict(c) for d, c in self.configs.items()},
                "models": {
                    d: {k: v for k, v in vars(m).items() if k not in {"config", "degree"}}
                    for d, m in self.models.items()
                },
            },
            path,
        )

    @classmethod
    def load(cls, path):
        state = joblib.load(path)
        model = cls({d: DegreeConfig(**c) for d, c in state["configs"].items()})
        model.year, model.models = state["year"], {}
        for degree, values in state["models"].items():
            component = DegreeModel(degree, model.configs[degree])
            vars(component).update(values)
            model.models[degree] = component
        return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--data", type=Path, default=Path("processed_data/registered.csv"))
    parser.add_argument("--exams", type=Path, default=Path("external_data/ege/exam_calendar.csv"))
    parser.add_argument("--selection", type=Path, default=Path("output/v93/selected_config.json"))
    parser.add_argument("--model", type=Path, default=Path("output/v93/model.joblib"))
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--degree",
        choices=["total", *DEGREES, "all"],
        default="total",
        help="Export the total, one education stream, or all components",
    )
    args = parser.parse_args()
    if args.command == "train":
        selection = json.loads(args.selection.read_text())
        model = AdmissionsModel({d: DegreeConfig(**selection["degrees"][d]) for d in DEGREES})
        model.fit(load_counts(args.data), args.year, load_exams(args.exams)).save(args.model)
    else:
        model = AdmissionsModel.load(args.model)
    prediction = model.predict(components=True)
    suffix = (
        ""
        if args.degree == "total"
        else f"_{'components' if args.degree == 'all' else args.degree}"
    )
    output = args.output or Path(f"forecast_{model.year}_v93{suffix}.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    column = "forecast" if args.degree == "total" else args.degree
    exported = (
        prediction
        if args.degree == "all"
        else prediction[["date", column]].rename(columns={column: "forecast"})
    )
    exported.to_csv(output, index=False, float_format="%.6f")
    prediction.to_csv(args.model.parent / "degree_forecasts.csv", index=False, float_format="%.6f")
    summary = {"output": str(output), "degree": args.degree}
    if args.degree == "all":
        summary["totals"] = prediction[list(DEGREES) + ["forecast"]].sum().to_dict()
    else:
        summary["total"] = float(exported.forecast.sum())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
