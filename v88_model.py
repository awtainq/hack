"""Daily admissions forecast from raw registrations and preceding seasons.

One randomized tree model predicts log1p(total daily registrations). Education
counts align historical activity transitions; optional lagged annual context
describes levels and government-channel adoption known before the forecast.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


GROUPS = ("total", "bachelor", "master", "college", "other", "government")
CATEGORIES = {
    "weekday": [str(day) for day in range(7)],
    "phase": ["start", "june", "bachelor", "master", "tail"],
    "saturday_slot": ["none", "first", "late", "august", "regular"],
}


@dataclass(frozen=True)
class ForestConfig:
    family: str = "forest"
    trees: int = 256
    min_leaf: int = 5
    feature_fraction: float = 0.8
    half_life: float = 4.0
    annual_context: bool = False
    schedule: str = "none"
    exams: bool = False

    def __post_init__(self):
        if self.family not in {"forest", "extra"} or self.schedule not in {"none", "learned"}:
            raise ValueError("Unknown model family or schedule")
        if (
            min(self.trees, self.min_leaf, self.half_life) <= 0
            or not 0 < self.feature_fraction <= 1
        ):
            raise ValueError("Invalid model parameters")


def campaign(year):
    return pd.date_range(f"{year}-06-20", f"{year}-08-31", name="date")


def load_counts(path):
    rows = pd.read_csv(path, usecols=["reg_ts", "zayav_type_name", "priem_doc_type_kod"])
    rows["date"] = pd.to_datetime(rows.reg_ts, errors="raise").dt.normalize()
    if rows[["date", "zayav_type_name", "priem_doc_type_kod"]].isna().any().any():
        raise ValueError("Missing date, degree or submission channel")
    degree = np.select(
        [rows.zayav_type_name.str.startswith(prefix) for prefix in ("Бакалавр", "Магистр", "СПО")],
        GROUPS[1:4],
        default="other",
    )
    for group in GROUPS[1:5]:
        rows[group] = (degree == group).astype(float)
    rows["total"] = 1.0
    rows["government"] = rows.priem_doc_type_kod.eq(4).astype(float)
    years = sorted(rows.date.dt.year.unique())
    if not years or years != list(range(years[0], years[-1] + 1)):
        raise ValueError("Source seasons must be contiguous")
    dates = pd.DatetimeIndex(np.concatenate([campaign(year).values for year in years]), name="date")
    return rows.groupby("date")[list(GROUPS)].sum().reindex(dates, fill_value=0)


def load_exam_calendar(path):
    frame = pd.read_csv(path).set_index("year")
    if not frame.index.is_unique:
        raise ValueError("Duplicate exam years")
    for subject in ("russian", "math", "social", "english"):
        frame[subject] = pd.to_datetime(frame[subject], errors="raise")
        if frame[subject].isna().any() or not np.array_equal(frame[subject].dt.year, frame.index):
            raise ValueError("Invalid exam dates")
    return frame[["russian", "math", "social", "english"]]


def activity_breaks(history):
    """Least-squares declining change points in past weekday log counts."""
    result = {}
    for year in sorted(history.index.year.unique()):
        part = history.loc[(history.index.year == year) & (history.index.dayofweek < 5)]
        for group in ("bachelor", "master"):
            values = np.log1p(part[group].to_numpy())
            losses = {}
            for split in range(10, len(values) - 10):
                before, after = values[:split], values[split:]
                if before.mean() > after.mean():
                    losses[part.index[split - 1]] = np.sum((before - before.mean()) ** 2) + np.sum(
                        (after - after.mean()) ** 2
                    )
            if losses:
                result[int(year), group] = min(losses, key=losses.get)
    return result


def calendar(dates, breaks=None):
    dates = pd.DatetimeIndex(dates)
    start = pd.to_datetime(dates.year.astype(str) + "-06-20")
    position = (dates - start).days
    frame = pd.DataFrame(
        {
            "position": position,
            "day": dates.day,
            "month": dates.month,
            "weekday": dates.dayofweek.astype(str),
            "first_day": (position == 0).astype(int),
        },
        index=dates,
    )
    frame["workdays_from_start"] = np.busday_count(
        start.values.astype("datetime64[D]"),
        dates.values.astype("datetime64[D]"),
        weekmask="1111110",
    )
    for group, month_day in (("bachelor", "07-25"), ("master", "07-29")):
        nominal = pd.to_datetime(dates.year.astype(str) + "-" + month_day)
        if group == "master":
            nominal -= pd.to_timedelta(np.maximum(nominal.dayofweek - 4, 0), unit="D")
        else:
            nominal = nominal.where(dates.year != 2021, pd.Timestamp("2021-07-29"))
        ends = pd.DatetimeIndex(
            [(breaks or {}).get((int(year), group), end) for year, end in zip(dates.year, nominal)]
        )
        frame[f"distance_{group}"] = (dates - ends).days
        frame[f"after_{group}"] = ((dates - ends).days > 0).astype(int)
        frame[f"end_weekday_{group}"] = ends.dayofweek
    frame["phase"] = np.select(
        [position < 4, dates.month == 6, frame.distance_bachelor <= 0, frame.distance_master <= 0],
        ["start", "june", "bachelor", "master"],
        default="tail",
    )
    frame["saturday_slot"] = np.select(
        [
            dates.dayofweek != 5,
            position < 7,
            (dates.month == 8) & (dates.day > 21),
            dates.month == 8,
        ],
        ["none", "first", "late", "august"],
        default="regular",
    )
    return frame


def annual_features(dates, history):
    dates = pd.DatetimeIndex(dates)
    annual = (
        history.loc[history.index.year != 2020]
        .groupby(history.loc[history.index.year != 2020].index.year)
        .mean()
    )
    frames = []
    for year in sorted(set(dates.year)):
        index = dates[dates.year == year]
        frame = pd.DataFrame(index=index)
        earlier = annual.loc[annual.index < year]
        for lag in (1, 2):
            level = earlier.iloc[-lag] if len(earlier) >= lag else pd.Series(np.nan, index=GROUPS)
            for group in GROUPS:
                frame[f"lag{lag}_{group}_level"] = np.log1p(level[group])
        frames.append(frame)
    return pd.concat(frames).reindex(dates)


class AdmissionsForest:
    def __init__(self, config=ForestConfig()):
        self.config = config

    def feature_frame(self, dates, training=False):
        frame = calendar(dates, self.breaks if training else None)
        if self.config.annual_context:
            frame = frame.join(annual_features(dates, self.history))
        if self.config.exams:
            dates = pd.DatetimeIndex(dates)
            for subject in self.exam_dates:
                exams = pd.DatetimeIndex(self.exam_dates.loc[dates.year, subject])
                frame[f"since_exam_{subject}"] = np.clip((dates - exams).days, -14, 35)
        for column, categories in CATEGORIES.items():
            frame[column] = pd.Categorical(frame[column], categories=categories)
        return pd.get_dummies(frame, columns=list(CATEGORIES)).fillna(-1.0)

    def fit(self, counts, year, exam_dates=None):
        history = counts.loc[counts.index.year < year, list(GROUPS)].sort_index().copy()
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
            raise ValueError("Invalid or incomplete daily counts")
        if not np.allclose(history.total, history[list(GROUPS[1:5])].sum(axis=1)):
            raise ValueError("Education partition does not reconcile")
        self.year = int(year)
        self.exam_dates = None
        if self.config.exams:
            if exam_dates is None or (set(map(int, years)) | {int(year)}) - set(exam_dates.index):
                raise ValueError("Missing public exam calendar")
            self.exam_dates = exam_dates.loc[exam_dates.index <= year].copy()
        self.history = history.loc[history.index.year != 2020]
        self.training_years = sorted(set(map(int, self.history.index.year)))
        self.breaks = activity_breaks(self.history)
        schedule = calendar(self.history.index)
        weight = np.exp2(-(year - self.history.index.year.to_numpy()) / self.config.half_life)
        self.closure_rates = {}
        for slot in ("first", "late", "august", "regular"):
            use = schedule.saturday_slot.eq(slot)
            self.closure_rates[slot] = float(
                np.average(self.history.total[use].eq(0), weights=weight[use])
            )
        use = np.ones(len(self.history), dtype=bool)
        if self.config.schedule == "learned":
            use &= self.history.index.dayofweek != 6
        factory = RandomForestRegressor if self.config.family == "forest" else ExtraTreesRegressor
        self.regressor = factory(
            n_estimators=self.config.trees,
            min_samples_leaf=self.config.min_leaf,
            max_features=self.config.feature_fraction,
            random_state=88001,
            n_jobs=1,
        )
        frame = self.feature_frame(self.history.index, training=True)
        self.regressor.fit(
            frame.loc[use], np.log1p(self.history.total[use]), sample_weight=weight[use]
        )
        self.feature_names = list(frame.columns)
        return self

    def predict(self):
        dates = campaign(self.year)
        values = np.maximum(0.0, self.regressor.predict(self.feature_frame(dates)))
        if self.config.schedule == "learned":
            frame = calendar(dates)
            closed = (dates.dayofweek == 6) | frame.saturday_slot.map(self.closure_rates).gt(
                0.5
            ).to_numpy()
            values[closed] = 0.0
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite forecast")
        return pd.DataFrame({"date": dates, "forecast": np.expm1(values)})

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

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        model = cls(ForestConfig(**payload["config"]))
        vars(model).update(payload["state"])
        return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data", type=Path, default=Path("processed_data/registered.csv"))
    parser.add_argument("--config", type=Path, default=Path("output/v88/selected_config.json"))
    parser.add_argument("--model", type=Path, default=Path("output/v88/model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("forecast_2024_v88.csv"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--exams", type=Path, default=Path("external_data/ege/exam_calendar.csv"))
    args = parser.parse_args()
    if args.command == "train":
        config = ForestConfig(**json.loads(args.config.read_text()))
        model = AdmissionsForest(config).fit(
            load_counts(args.data),
            args.year,
            load_exam_calendar(args.exams) if config.exams else None,
        )
        model.save(args.model)
    else:
        model = AdmissionsForest.load(args.model)
    model.predict().to_csv(args.output, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
