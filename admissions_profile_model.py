"""A standalone degree-panel model of daily registration profiles.

One CatBoost is shared across education levels. It learns log daily counts
relative to the size of the season. A separate historical volume estimate
restores the count scale. No old forecasts or leaderboard measurements are read.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


GROUPS = ("bachelor", "master_miep", "college", "postgraduate")
CATEGORIES = ["degree", "weekday", "saturday_slot"]


@dataclass(frozen=True)
class ProfileConfig:
    depth: int = 4
    iterations: int = 500
    half_life: float = 4.0
    volume_window: int = 3
    normalize: bool = True
    infer_closures: bool = True


def campaign(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-06-20", f"{year}-08-31")


def load_counts(path: Path) -> pd.DataFrame:
    records = pd.read_csv(path, usecols=["reg_ts", "zayav_type_name"])
    records["date"] = pd.to_datetime(records.reg_ts, errors="raise").dt.normalize()
    if records.date.isna().any() or records.empty:
        raise ValueError("Missing registration dates.")
    records["degree"] = np.select(
        [records.zayav_type_name.str.startswith(prefix) for prefix in ("Бакалавр", "Магистр", "СПО")],
        GROUPS[:3], default="postgraduate",
    )
    years = sorted(records.date.dt.year.unique())
    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError("Observed seasons must be contiguous.")
    index = pd.DatetimeIndex(np.concatenate([campaign(year).values for year in years]))
    return (records.groupby(["date", "degree"]).size().unstack(fill_value=0)
            .reindex(index=index, columns=GROUPS, fill_value=0).astype(float))


def calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.DatetimeIndex(dates)
    year = dates.year
    start = pd.to_datetime(year.astype(str) + "-06-20")
    bachelor_end = pd.to_datetime(year.astype(str) + "-07-25")
    bachelor_end = bachelor_end.where(year != 2021, pd.Timestamp("2021-07-29"))
    master_end = pd.to_datetime(year.astype(str) + "-07-29")
    master_end -= pd.to_timedelta(np.maximum(master_end.dayofweek - 4, 0), unit="D")
    features = pd.DataFrame(index=dates)
    features["day_of_campaign"] = (dates - start).days
    features["day_of_month"] = dates.day
    features["month"] = dates.month
    features["weekday"] = dates.dayofweek.astype(str)
    features["first_day"] = (dates == start).astype(int)
    features["days_from_bachelor_end"] = (dates - bachelor_end).days
    features["days_from_master_end"] = (dates - master_end).days
    features["bachelor_end_weekday"] = bachelor_end.dayofweek
    features["master_end_weekday"] = master_end.dayofweek
    features["saturday_slot"] = np.select(
        [dates.dayofweek != 5, features.day_of_campaign < 7,
         (dates.month == 8) & (dates.day > 21), dates.month == 8],
        ["none", "first", "late_august", "august"], default="regular",
    )
    for name, end in (("bachelor", bachelor_end), ("master", master_end)):
        features[f"workdays_from_{name}_end"] = np.busday_count(
            end.values.astype("datetime64[D]"), dates.values.astype("datetime64[D]"),
            weekmask="1111110",
        )
    return features


def make_panel(counts: pd.DataFrame) -> pd.DataFrame:
    features = calendar(counts.index)
    blocks = []
    for degree in GROUPS:
        block = features.copy()
        block["degree"] = degree
        block["count"] = counts[degree]
        block["year"] = counts.index.year
        blocks.append(block)
    return pd.concat(blocks).reset_index(names="date")


class ProfileModel:
    def __init__(self, config: ProfileConfig = ProfileConfig()) -> None:
        self.config = config

    def fit(self, counts: pd.DataFrame, year: int) -> ProfileModel:
        history = counts.loc[counts.index.year < year].copy()
        if history.empty or history.index.year.max() != year - 1:
            raise ValueError("The preceding season is required.")
        if not np.isfinite(history.to_numpy()).all() or (history < 0).any().any():
            raise ValueError("Counts must be finite and nonnegative.")
        self.year = year
        annual = history.groupby(history.index.year).sum()
        # The pandemic campaign had a different admissions calendar.
        history = history.loc[history.index.year != 2020]
        annual = annual.drop(index=2020, errors="ignore")
        recent = annual.tail(self.config.volume_window)
        volume_weights = np.exp2(-(year - recent.index.to_numpy()) / self.config.half_life)
        self.volumes = np.exp(np.average(np.log(recent + 1), axis=0, weights=volume_weights)) - 1
        self.volumes = dict(zip(GROUPS, self.volumes))

        daily = calendar(history.index)
        daily["zero"] = history.sum(axis=1).eq(0)
        daily["weight"] = np.exp2(-(year - history.index.year.to_numpy()) / self.config.half_life)
        self.closure_rates = {}
        for slot, rows in daily.loc[daily.saturday_slot != "none"].groupby("saturday_slot"):
            self.closure_rates[slot] = float(np.average(rows.zero, weights=rows.weight))

        panel = make_panel(history)
        self.features = list(calendar(history.index).columns) + ["degree"]
        annual_long = annual.rename_axis("year").reset_index().melt(
            id_vars="year", var_name="degree", value_name="volume",
        )
        panel = panel.merge(annual_long, on=["year", "degree"], validate="many_to_one")
        panel = panel.loc[panel.date.dt.dayofweek != 6]
        target = np.log1p(panel["count"].to_numpy())
        if self.config.normalize:
            target -= np.log1p(panel.volume.to_numpy())
        weights = np.exp2(-(year - panel.year.to_numpy()) / self.config.half_life)
        self.regressor = CatBoostRegressor(
            depth=self.config.depth, iterations=self.config.iterations,
            learning_rate=0.04, l2_leaf_reg=10, loss_function="RMSE",
            cat_features=CATEGORIES, one_hot_max_size=16, random_seed=69001,
            thread_count=1, verbose=False, allow_writing_files=False,
        )
        self.regressor.fit(panel[self.features], target, sample_weight=weights)
        return self

    def predict(self, volumes: dict[str, float] | None = None) -> pd.DataFrame:
        dates = campaign(self.year)
        panel = make_panel(pd.DataFrame(0.0, index=dates, columns=GROUPS))
        log_count = self.regressor.predict(panel[self.features])
        if self.config.normalize:
            levels = {**self.volumes, **(volumes or {})}
            log_count += np.log1p(panel.degree.map(levels).to_numpy())
        elif volumes is not None:
            raise ValueError("Volume inputs require the normalized architecture.")
        panel["forecast"] = np.maximum(np.expm1(log_count), 0)
        forecast = panel.pivot(index="date", columns="degree", values="forecast")[list(GROUPS)]
        closed = dates.dayofweek == 6
        if self.config.infer_closures:
            features = calendar(dates)
            closed = closed | features.saturday_slot.map(self.closure_rates).gt(0.5)
        forecast.loc[np.asarray(closed)] = 0
        forecast["forecast"] = forecast.sum(axis=1)
        return forecast.reset_index()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("processed_data/registered.csv"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--output", type=Path, default=Path("output/v69"))
    args = parser.parse_args()
    config = ProfileConfig(**json.loads(args.config.read_text())) if args.config else ProfileConfig()
    model = ProfileModel(config).fit(load_counts(args.data), args.year)
    model.save(args.output / "model.joblib")
    prediction = model.predict()
    prediction.to_csv(args.output / "degree_predictions.csv", index=False, float_format="%.6f")
    destination = Path(f"forecast_{args.year}_v69_profile.csv")
    prediction[["date", "forecast"]].to_csv(destination, index=False, float_format="%.2f")
    (args.output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    print(f"Saved {destination}, total {prediction.forecast.sum():.2f}")


if __name__ == "__main__":
    main()
