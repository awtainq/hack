"""V93 с фиксированными настройками. Их происхождение описано в V93_REPORT.md."""

import argparse
from pathlib import Path

from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.preprocessing import SplineTransformer

from admissions_calendar import (
    DEGREES, campaign, categorical_columns, features, history_before,
    infer_deadlines, load_counts, load_exams, saturday_slots,
)

ROOT = Path(__file__).resolve().parent
HALF_LIFE = {
    "bachelor": 2.4219521593767075,
    "master": 3.040427889891538,
    "background": 2.921803597533119,
}
CATBOOST_PARAMS = {
    "bachelor": dict(iterations=500, depth=4, learning_rate=0.04503094501267006,
                     l2_leaf_reg=2.948883755909228, loss_function="RMSE"),
    "master": dict(iterations=300, depth=7, learning_rate=0.14172324973151415,
                   l2_leaf_reg=2.184387671921391, loss_function="MAE"),
}


def calendar_features(dates, degree, exams, deadlines=None):
    return features(dates, degree, deadlines,
                    "event" if degree == "master" else "calendar",
                    exams if degree == "bachelor" else None)


def remove_annual_level(target, year, half_life):
    """Отделяем дневной профиль от уровня года и продолжаем тренд трёх лет."""
    weekdays = target.index.dayofweek < 5
    levels = target[weekdays].groupby(target.index.year[weekdays]).mean()
    recent = levels.loc[levels.index < year].tail(3)
    years, values = recent.index.to_numpy(float), recent.to_numpy(float)
    weights = np.exp2(-(year - years) / half_life)
    center = np.average(years, weights=weights)
    mean = np.average(values, weights=weights)
    slope = np.sum(weights * (years - center) * (values - mean))
    slope /= np.sum(weights * (years - center) ** 2) + 1.0
    level = max(0.0, float(mean + slope * (year - center)))
    return target - target.index.year.map(levels).to_numpy(), level


def spline_matrix(frame, spline):
    """Сплайны, календарные признаки и взаимодействия срока с днём недели."""
    smooth = spline.transform(frame[["opening_ramp", "distance", "position"]].to_numpy())
    encoded = frame.copy()
    for column, categories in categorical_columns().items():
        encoded[column] = pd.Categorical(encoded[column], categories=categories)
    encoded = pd.get_dummies(encoded, columns=list(categorical_columns()))
    width = spline.n_features_out_ // 3
    relative = smooth[:, width : 2 * width]
    weekday = np.column_stack([frame.weekday.eq(str(i)).to_numpy(float) for i in range(1, 6)])
    interaction = (relative[:, :, None] * weekday[:, None, :]).reshape(len(frame), -1)
    return np.column_stack([
        np.ones((len(frame), 1)), smooth, encoded.to_numpy(float) / 35, interaction,
    ])


def train(counts, exams, year=2024):
    """Обучаем три потока по истории; возвращаем всё нужное для predict()."""
    year = int(year)
    history = history_before(counts, year)
    if (set(history.index.year) | {year}) - set(exams.index):
        raise ValueError("Missing exam dates")
    model = {"year": year, "exams": exams.loc[exams.index <= year].copy(), "streams": {}}
    for index, degree in enumerate(DEGREES):
        data = history
        if degree == "master":
            years = sorted(set(history.index.year))[-5:]
            data = history.loc[history.index.year.isin(years)]
        weights = np.exp2(-(year - data.index.year.to_numpy()) / HALF_LIFE[degree])
        slots = saturday_slots(data.index)
        total = data.sum(axis=1)
        closed_slots = [
            slot for slot in ("first", "late", "august", "regular")
            if np.average(total[slots == slot].eq(0), weights=weights[slots == slot]) > 0.5
        ]
        target = np.log1p(data[degree])
        level = 0.0
        if degree != "background":
            target, level = remove_annual_level(target, year, HALF_LIFE[degree])
        use = data.index.dayofweek != 6
        if degree == "background":
            use &= data[degree].to_numpy() > 0
        deadlines = infer_deadlines(data) if degree == "master" else None
        frame = calendar_features(data.index, degree, model["exams"], deadlines)
        stream = {"level": level, "closed_slots": closed_slots}
        if degree in CATBOOST_PARAMS:
            regressor = CatBoostRegressor(
                **CATBOOST_PARAMS[degree], cat_features=list(categorical_columns()),
                one_hot_max_size=16, random_seed=93001 + index, thread_count=1,
                allow_const_label=True, verbose=False, allow_writing_files=False,
            )
            regressor.fit(frame.loc[use], target.loc[use], sample_weight=weights[use])
            stream["regressor"] = regressor
        else:
            knots = np.column_stack([
                np.linspace(0, 14, 5), np.linspace(-60, 50, 5), np.linspace(0, 72, 5),
            ])
            spline = SplineTransformer(knots=knots, degree=3, include_bias=False,
                                       extrapolation="linear")
            spline.fit(frame[["opening_ramp", "distance", "position"]].to_numpy())
            matrix = spline_matrix(frame, spline)[use]
            response, weight = target.to_numpy()[use], weights[use]
            gram = np.einsum("ni,n,nj->ij", matrix, weight, matrix)
            penalty = np.full(matrix.shape[1], 0.032036899931769114)
            penalty[0] = 1e-8
            gram += np.diag(penalty)
            rhs = np.einsum("ni,n,n->i", matrix, weight, response)
            stream["coefficients"] = solve(gram, rhs, assume_a="pos")
            stream["spline"] = spline
        model["streams"][degree] = stream
    return model


def predict(model, components=False):
    """Считаем каждый поток, обнуляем закрытые дни и складываем прогнозы."""
    dates = campaign(model["year"])
    slots = saturday_slots(dates)
    result = pd.DataFrame(index=dates)
    for degree in DEGREES:
        stream = model["streams"][degree]
        frame = calendar_features(dates, degree, model["exams"])
        if degree in CATBOOST_PARAMS:
            log_counts = stream["regressor"].predict(frame)
        else:
            matrix = spline_matrix(frame, stream["spline"])
            log_counts = np.einsum("ni,i->n", matrix, stream["coefficients"])
        values = np.expm1(np.maximum(0.0, log_counts + stream["level"]))
        values[(dates.dayofweek == 6) | np.isin(slots, stream["closed_slots"])] = 0.0
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite forecast")
        result[degree] = values
    result["forecast"] = result.sum(axis=1)
    result = result.reset_index()
    return result if components else result[["date", "forecast"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--data", type=Path, default=ROOT / "processed_data/registered.csv")
    parser.add_argument("--exams", type=Path, default=ROOT / "external_data/ege/exam_calendar.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "output/v93_simple/model.joblib")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--degree", choices=["total", *DEGREES, "all"], default="total")
    args = parser.parse_args()
    if args.command == "train":
        model = train(load_counts(args.data), load_exams(args.exams), args.year)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model)
    else:
        model = joblib.load(args.model)
    result = predict(model, components=True)
    suffix = "" if args.degree == "total" else f"_{args.degree}"
    output = args.output or Path(f"forecast_{model['year']}_v93_simple{suffix}.csv")
    if args.degree != "all":
        column = "forecast" if args.degree == "total" else args.degree
        result = result[["date", column]].rename(columns={column: "forecast"})
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False, float_format="%.6f")
    print(output)


if __name__ == "__main__":
    main()
