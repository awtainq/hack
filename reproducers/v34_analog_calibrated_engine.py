"""
V34: analog-calibrated ensemble for the 2024 registered daily flow.

The model keeps V30 as the main estimator and adds a leakage-safe historical
analog estimate. The analog layer uses the same campaign day and weekday from
previous campaigns, with recency weighting in log space. This gives the model
a direct estimate for recurring calendar positions while retaining the tree
ensemble for regulatory interactions.

The analog contribution is deliberately conservative. It is not a claim that
the hidden 2024 RMSLE will be below 0.5; that can only be established by the
external scorer.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "processed_data" / "registered.csv"
OUTPUT_FILE = ROOT / "forecast_2024_v34_analog_calibrated.csv"


def make_daily(df: pd.DataFrame) -> pd.DataFrame:
    grid = pd.DataFrame(
        {
            "date": [
                date
                for year in range(2014, 2025)
                for date in pd.date_range(f"{year}-06-20", f"{year}-08-31")
            ]
        }
    )
    grid["date"] = pd.to_datetime(grid["date"])
    grid["year"] = grid["date"].dt.year
    counts = (
        df.assign(date=pd.to_datetime(df["reg_ts"]).dt.normalize())
        .groupby("date")
        .size()
        .rename("actual_count")
    )
    grid = grid.merge(counts, on="date", how="left")
    grid["actual_count"] = grid["actual_count"].fillna(0.0)
    grid["dow"] = grid["date"].dt.dayofweek
    grid["day"] = grid["date"].dt.day
    grid["month"] = grid["date"].dt.month
    grid["iso_week"] = grid["date"].dt.isocalendar().week.astype(int)
    campaign_start = pd.to_datetime(grid["year"].astype(str) + "-06-20")
    grid["day_of_camp"] = (grid["date"] - campaign_start).dt.days
    grid["progress"] = grid["day_of_camp"] / 72.0
    grid["is_weekend"] = (grid["dow"] >= 5).astype(int)
    grid["is_saturday"] = (grid["dow"] == 5).astype(int)
    grid["is_sunday"] = (grid["dow"] == 6).astype(int)
    grid["is_monday"] = (grid["dow"] == 0).astype(int)

    for milestone, name in (
        ("06-20", "start"),
        ("07-08", "peak"),
        ("07-25", "deadline"),
        ("08-03", "consent"),
        ("08-09", "orders"),
        ("08-31", "end"),
    ):
        milestone_date = pd.to_datetime(grid["year"].astype(str) + "-" + milestone)
        grid[f"dist_{name}"] = (grid["date"] - milestone_date).dt.days

    flags = {
        "is_first_week": (grid["month"] == 6) & (grid["day"] <= 23),
        "is_diploma_week": (grid["month"] == 6) & (grid["day"] >= 24),
        "is_peak_period": (grid["month"] == 7) & (grid["day"] <= 15),
        "is_pre_deadline_3d": (grid["month"] == 7) & grid["day"].between(22, 25),
        "is_deadline_day": (grid["month"] == 7) & (grid["day"] == 25),
        "is_post_deadline": (grid["month"] == 7) & (grid["day"] > 25),
        "is_august": grid["month"] == 8,
        "sat_june": (grid["dow"] == 5) & (grid["month"] == 6),
        "sat_july_peak": (grid["dow"] == 5) & (grid["month"] == 7) & (grid["day"] <= 20),
        "sat_july_late": (grid["dow"] == 5) & (grid["month"] == 7) & (grid["day"] > 20),
        "sat_aug": (grid["dow"] == 5) & (grid["month"] == 8),
        "mon_july": (grid["dow"] == 0) & (grid["month"] == 7),
        "mon_aug": (grid["dow"] == 0) & (grid["month"] == 8),
    }
    for name, value in flags.items():
        grid[name] = value.astype(int)

    grid["sin_prog"] = np.sin(np.pi * grid["progress"])
    grid["sin_2prog"] = np.sin(2 * np.pi * grid["progress"])
    grid["cos_prog"] = np.cos(np.pi * grid["progress"])
    grid["sin_dow"] = np.sin(2 * np.pi * grid["dow"] / 7)
    grid["cos_dow"] = np.cos(2 * np.pi * grid["dow"] / 7)
    return grid


def weights(years: pd.Series) -> np.ndarray:
    return np.select(
        [years == 2020, years >= 2021, years >= 2017],
        [0.4, 2.5, 1.5],
        default=0.7,
    ).astype(float)


def analog_prediction(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    source = train[train["actual_count"] > 0].copy()
    source["log_count"] = np.log1p(source["actual_count"])
    source["analog_weight"] = weights(source["year"]) / (
        1.0 + 0.18 * (test["year"].iloc[0] - source["year"])
    )
    values = []
    for _, row in test.iterrows():
        same_position = source[
            (source["day_of_camp"] == row["day_of_camp"])
            & (source["dow"] == row["dow"])
        ]
        if len(same_position) < 2:
            same_position = source[source["day_of_camp"] == row["day_of_camp"]]
        if len(same_position) == 0:
            values.append(float(source["log_count"].median()))
            continue
        values.append(
            float(
                np.average(
                    same_position["log_count"],
                    weights=same_position["analog_weight"],
                )
            )
        )
    return np.expm1(np.asarray(values))


def generate_v34_forecast() -> pd.DataFrame:
    daily = make_daily(pd.read_csv(DATA_FILE))
    train = daily[daily["year"] < 2024].reset_index(drop=True)
    test = daily[daily["year"] == 2024].reset_index(drop=True)
    features = [
        column
        for column in daily.columns
        if column not in {"year", "date", "actual_count", "target_log"}
    ]
    target = np.log1p(train["actual_count"])
    sample_weight = weights(train["year"])

    lgb_predictions = []
    cat_predictions = []
    hgb_predictions = []
    for seed_index in range(10):
        model = lgb.LGBMRegressor(
            n_estimators=130, learning_rate=0.035, num_leaves=14,
            min_child_samples=4, subsample=0.85, colsample_bytree=0.90,
            random_state=seed_index * 17 + 42, verbosity=-1
        )
        model.fit(train[features], target, sample_weight=sample_weight)
        lgb_predictions.append(np.expm1(model.predict(test[features])))
        if seed_index < 5:
            cat = CatBoostRegressor(
                iterations=140, learning_rate=0.035, depth=4,
                l2_leaf_reg=4, random_seed=seed_index * 23 + 42, verbose=False
            )
            cat.fit(train[features], target, sample_weight=sample_weight)
            cat_predictions.append(np.expm1(cat.predict(test[features])))
        if seed_index < 3:
            hgb = HistGradientBoostingRegressor(
                max_iter=130, learning_rate=0.035, max_leaf_nodes=14,
                min_samples_leaf=4, random_state=seed_index * 31 + 42
            )
            hgb.fit(train[features], target, sample_weight=sample_weight)
            hgb_predictions.append(np.expm1(hgb.predict(test[features])))

    tree_prediction = (
        0.60 * np.mean(lgb_predictions, axis=0)
        + 0.10 * np.mean(hgb_predictions, axis=0)
        + 0.30 * np.mean(cat_predictions, axis=0)
    )
    analog = analog_prediction(train, test)
    prediction = 0.85 * tree_prediction + 0.15 * analog

    closed_dates = {"2024-06-22", "2024-08-24", "2024-08-31"}
    closed = (
        (test["is_sunday"] == 1)
        | test["date"].dt.strftime("%Y-%m-%d").isin(closed_dates)
    )
    prediction = np.where(closed, 0.0, np.clip(prediction, 0.55, None))
    result = pd.DataFrame(
        {
            "date": test["date"].dt.strftime("%Y-%m-%d"),
            "forecast": np.round(prediction, 2),
        }
    )
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"V34 successfully generated: {OUTPUT_FILE}")
    print(f"Total days: {len(result)}")
    print(f"Total zeros: {(result['forecast'] == 0.0).sum()}")
    print(f"Total sum: {result['forecast'].sum():.2f}")
    return result


if __name__ == "__main__":
    generate_v34_forecast()
