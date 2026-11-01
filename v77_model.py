"""A daily admissions model fitted to registrations and public exam calendars.

The hurdle variant estimates P(Y > 0 | calendar) and E[log(1+Y) | Y > 0,
calendar] separately. Their product minimizes expected squared log error.
No submitted forecasts, leaderboard scores or target-year totals are inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor


CATEGORIES = ["weekday", "phase", "saturday_slot"]
SEEDS = (77001, 77002, 77003)


@dataclass(frozen=True)
class ModelConfig:
    depth: int = 3
    iterations: int = 600
    half_life: float = 6.0
    hurdle: bool = True
    exams: bool = True
    quiet_groups: bool = False
    hard_schedule: bool = False

    def __post_init__(self):
        if self.depth < 1 or self.iterations < 1 or self.half_life <= 0:
            raise ValueError("Invalid training parameters")
        if self.hurdle and self.quiet_groups:
            raise ValueError("Choose hurdle or separate quiet groups")
        if self.hard_schedule and (self.hurdle or self.quiet_groups):
            raise ValueError("Hard schedule requires direct regression")


def campaign(year):
    return pd.date_range(f"{year}-06-20", f"{year}-08-31", name="date")


def load_counts(path):
    records = pd.read_csv(path, usecols=["reg_ts", "zayav_type_name"])
    records["date"] = pd.to_datetime(records.reg_ts, errors="raise").dt.normalize()
    if records.date.isna().any() or records.zayav_type_name.isna().any():
        raise ValueError("Missing date or degree")
    records["total"] = 1.0
    records["bachelor"] = records.zayav_type_name.str.startswith("Бакалавр").astype(float)
    records["master"] = records.zayav_type_name.str.startswith("Магистр").astype(float)
    years = sorted(records.date.dt.year.unique())
    if not years or years != list(range(years[0], years[-1] + 1)):
        raise ValueError("Historical seasons must be contiguous")
    dates = pd.DatetimeIndex(np.concatenate([campaign(year).values for year in years]), name="date")
    return records.groupby("date")[["total", "bachelor", "master"]].sum().reindex(dates, fill_value=0)


def load_exams(path):
    frame = pd.read_csv(path).set_index("year")
    if not frame.index.is_unique:
        raise ValueError("Duplicate exam years")
    for subject in ("russian", "math", "social", "english"):
        frame[subject] = pd.to_datetime(frame[subject], errors="raise")
        if frame[subject].isna().any() or not np.array_equal(frame[subject].dt.year, frame.index):
            raise ValueError("Exam date does not match its year")
    return frame[["russian", "math", "social", "english"]]


def historical_deadlines(history):
    deadlines = {}
    for year in sorted(history.index.year.unique()):
        for degree, fraction in (("bachelor", 0.20), ("master", 0.25)):
            series = history.loc[history.index.year == year, degree]
            threshold = fraction * series.loc[f"{year}-07-01":f"{year}-07-25"].max()
            if threshold <= 0:
                continue
            for date, value in series.loc[f"{year}-07-20":f"{year}-08-20"].items():
                following = series.loc[date + pd.Timedelta(days=1):date + pd.Timedelta(days=3)]
                if value >= threshold and len(following) == 3 and (following < threshold).all():
                    deadlines[int(year), degree] = date
                    break
    return deadlines


def calendar_features(dates, deadlines=None, exams=None):
    dates = pd.DatetimeIndex(dates)
    starts = pd.to_datetime(dates.year.astype(str) + "-06-20")
    position = (dates - starts).days
    frame = pd.DataFrame(index=dates)
    frame["day_of_campaign"] = position
    frame["day"] = dates.day
    frame["month"] = dates.month
    frame["week"] = position // 7
    frame["weekday"] = dates.dayofweek.astype(str)
    frame["first_day"] = (position == 0).astype(int)
    frame["first_week"] = (position < 4).astype(int)
    frame["sin_progress"] = np.sin(np.pi * position / 72)
    frame["cos_progress"] = np.cos(np.pi * position / 72)
    frame["workday"] = np.busday_count(starts.values.astype("datetime64[D]"), dates.values.astype("datetime64[D]"), weekmask="1111110")
    for degree, month_day in (("bachelor", "07-25"), ("master", "07-29")):
        nominal = pd.to_datetime(dates.year.astype(str) + "-" + month_day)
        if degree == "master":
            nominal -= pd.to_timedelta(np.maximum(nominal.dayofweek - 4, 0), unit="D")
        else:
            nominal = nominal.where(dates.year != 2021, pd.Timestamp("2021-07-29"))
        ends = pd.DatetimeIndex([(deadlines or {}).get((int(year), degree), end) for year, end in zip(dates.year, nominal)])
        distance = (dates - ends).days
        frame[f"distance_{degree}"] = distance
        frame[f"after_{degree}"] = (distance > 0).astype(int)
        frame[f"weekday_end_{degree}"] = ends.dayofweek
        frame[f"workdays_{degree}"] = np.busday_count(ends.values.astype("datetime64[D]"), dates.values.astype("datetime64[D]"), weekmask="1111110")
    frame["phase"] = np.select(
        [position < 4, dates.month == 6, frame.distance_bachelor <= 0,
         frame.distance_master <= 0, (dates.month == 8) & (dates.day > 21)],
        ["start", "ramp", "bachelor", "master", "late"], default="post",
    )
    frame["saturday_slot"] = np.select(
        [dates.dayofweek != 5, position < 7, (dates.month == 8) & (dates.day > 21), dates.month == 8],
        ["none", "first", "late", "august"], default="regular",
    )
    if exams is not None:
        if set(dates.year) - set(exams.index):
            raise ValueError("Missing public exam calendar")
        for subject in exams.columns:
            exam_dates = pd.DatetimeIndex(exams.loc[dates.year, subject])
            # Saturation prevents distant exam dates from acting as a year ID.
            frame[f"since_exam_{subject}"] = np.clip((dates - exam_dates).days, -14, 35)
    return frame


def activity_groups(frame):
    return np.where(frame.index.dayofweek == 6, "sunday", frame.saturday_slot)


class DailyAdmissionsModel:
    def __init__(self, config=ModelConfig()):
        self.config = config

    def fit(self, counts, year, exams=None):
        history = counts.loc[counts.index.year < year].sort_index().copy()
        years = sorted(history.index.year.unique())
        if not years or years != list(range(years[0], year)):
            raise ValueError("Forecast requires contiguous preceding seasons")
        expected = pd.DatetimeIndex(np.concatenate([campaign(y).values for y in years]), name="date")
        if not history.index.equals(expected) or not np.isfinite(history.to_numpy()).all() or (history < 0).any().any():
            raise ValueError("Invalid, negative or incomplete daily counts")
        self.year = int(year)
        self.training_years = [int(y) for y in years if y != 2020]
        # The postponed 2020 campaign has a different time origin and deadline.
        history = history.loc[history.index.year != 2020]
        self.exams = None
        if self.config.exams:
            if exams is None:
                raise ValueError("This configuration requires exam dates")
            self.exams = exams.loc[exams.index <= year].copy()
            if (set(self.training_years) | {year}) - set(self.exams.index):
                raise ValueError("Missing public exam calendar")
        self.deadlines = historical_deadlines(history)
        frame = calendar_features(history.index, self.deadlines, self.exams)
        weight = np.exp2(-(year - history.index.year.to_numpy()) / self.config.half_life)
        target = np.log1p(history.total.to_numpy())
        positive = target > 0
        if positive.sum() < 30 or (self.config.hurdle and positive.all()):
            raise ValueError("Insufficient positive/zero observations")
        self.regressors, self.classifiers = [], []
        self.quiet_expectations = {}
        self.closed_groups = []
        groups = activity_groups(frame)
        active = np.ones(len(history), dtype=bool)
        if self.config.hard_schedule:
            active &= history.index.dayofweek != 6
            for group in np.unique(groups):
                rows = groups == group
                if np.average(~positive[rows], weights=weight[rows]) > 0.5:
                    self.closed_groups.append(group)
        if self.config.quiet_groups:
            for group in np.unique(groups):
                rows = groups == group
                if np.average(~positive[rows], weights=weight[rows]) > 0.5:
                    active[rows] = False
            quiet_center = np.average(target[~active], weights=weight[~active]) if (~active).any() else 0.0
            for group in np.unique(groups[~active]):
                rows = groups == group
                # One pseudo-observation shrinks sparse groups toward the pooled
                # quiet-day log expectation; no future zero-day mask is supplied.
                self.quiet_expectations[group] = float((np.dot(weight[rows], target[rows]) + quiet_center) / (weight[rows].sum() + 1.0))
        for seed in SEEDS:
            common = dict(
                learning_rate=0.035, l2_leaf_reg=10.0, random_seed=seed,
                cat_features=CATEGORIES, one_hot_max_size=16, thread_count=1,
                verbose=False, allow_writing_files=False,
            )
            regressor = CatBoostRegressor(depth=self.config.depth, iterations=self.config.iterations, loss_function="RMSE", **common)
            use = positive if self.config.hurdle else active
            regressor.fit(frame.loc[use], target[use], sample_weight=weight[use])
            self.regressors.append(regressor)
            if self.config.hurdle:
                classifier = CatBoostClassifier(depth=3, iterations=250, loss_function="Logloss", **common)
                classifier.fit(frame, positive.astype(int), sample_weight=weight)
                self.classifiers.append(classifier)
        return self

    def predict(self, diagnostics=False):
        dates = campaign(self.year)
        frame = calendar_features(dates, exams=self.exams)
        predictions, probabilities = [], []
        for index, regressor in enumerate(self.regressors):
            conditional = np.maximum(0.0, regressor.predict(frame))
            probability = self.classifiers[index].predict_proba(frame)[:, 1] if self.config.hurdle else np.ones(len(frame))
            predictions.append(conditional * probability)
            probabilities.append(probability)
        log_forecast = np.mean(predictions, axis=0)
        groups = activity_groups(frame)
        for group, expectation in self.quiet_expectations.items():
            log_forecast[groups == group] = expectation
        log_forecast[np.isin(groups, self.closed_groups)] = 0.0
        if not np.isfinite(log_forecast).all():
            raise ValueError("Nonfinite predictions")
        result = pd.DataFrame({"date": dates, "forecast": np.expm1(log_forecast)})
        if diagnostics:
            result["activity_probability"] = np.mean(probabilities, axis=0)
            result["seed_log_std"] = np.std(predictions, axis=0)
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": asdict(self.config), "state": {k: v for k, v in vars(self).items() if k != "config"}}, path)

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        model = cls(ModelConfig(**payload["config"]))
        vars(model).update(payload["state"])
        return model


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--registered", type=Path, default=Path("processed_data/registered.csv"))
    parser.add_argument("--exams", type=Path, default=Path("external_data/ege/exam_calendar.csv"))
    parser.add_argument("--config", type=Path, default=Path("output/v77/selected_config.json"))
    parser.add_argument("--model", type=Path, default=Path("output/v77/model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("forecast_2024_v77.csv"))
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()
    if args.command == "train":
        config = ModelConfig(**json.loads(args.config.read_text()))
        model = DailyAdmissionsModel(config).fit(load_counts(args.registered), args.year, load_exams(args.exams) if config.exams else None)
        model.save(args.model)
    else:
        model = DailyAdmissionsModel.load(args.model)
    model.predict().to_csv(args.output, index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
