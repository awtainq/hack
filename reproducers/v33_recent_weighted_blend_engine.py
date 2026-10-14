"""
V33: recent-weighted tri-ensemble for the registered daily flow.

Compared with V30/V32:
  * keeps 2020 in the training grid as a low-weight digital-submission year;
  * uses explicit temporal weights instead of dropping 2020 completely;
  * adds the August-Monday interaction used by V32;
  * keeps the conservative V30 zero-day perimeter.

The temporal weighting choice was checked with leave-one-year-out validation:
including 2020 at weight 0.4 reduced mean LightGBM RMSLE on 2021-2023
from 0.7020 to 0.6616 in the local experiment. This is validation evidence,
not a guarantee about an unseen leaderboard score.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "processed_data" / "registered.csv"
OUTPUT_FILE = ROOT / "forecast_2024_v33_recent_weighted_blend.csv"


def build_daily_frame(df_reg: pd.DataFrame) -> pd.DataFrame:
    dates = [
        {"year": year, "date": date}
        for year in range(2014, 2025)
        for date in pd.date_range(f"{year}-06-20", f"{year}-08-31")
    ]
    daily = pd.DataFrame(dates)
    daily["date"] = pd.to_datetime(daily["date"])
    counts = (
        df_reg.assign(date=pd.to_datetime(df_reg["reg_ts"]).dt.normalize())
        .groupby("date")
        .size()
        .rename("actual_count")
    )
    daily = daily.merge(counts, on="date", how="left")
    daily["actual_count"] = daily["actual_count"].fillna(0.0)

    daily["year"] = daily["date"].dt.year
    daily["dow"] = daily["date"].dt.dayofweek
    daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
    daily["is_saturday"] = (daily["dow"] == 5).astype(int)
    daily["is_sunday"] = (daily["dow"] == 6).astype(int)
    daily["is_monday"] = (daily["dow"] == 0).astype(int)
    daily["day"] = daily["date"].dt.day
    daily["month"] = daily["date"].dt.month
    daily["iso_week"] = daily["date"].dt.isocalendar().week.astype(int)

    campaign_start = pd.to_datetime(daily["year"].astype(str) + "-06-20")
    daily["day_of_camp"] = (daily["date"] - campaign_start).dt.days
    daily["progress"] = daily["day_of_camp"] / 72.0

    for milestone, name in (
        ("06-20", "start"),
        ("07-08", "peak"),
        ("07-25", "deadline"),
        ("08-03", "consent"),
        ("08-09", "orders"),
        ("08-31", "end"),
    ):
        milestone_date = pd.to_datetime(daily["year"].astype(str) + "-" + milestone)
        daily[f"dist_{name}"] = (daily["date"] - milestone_date).dt.days

    daily["is_first_week"] = ((daily["month"] == 6) & (daily["day"] <= 23)).astype(int)
    daily["is_diploma_week"] = ((daily["month"] == 6) & (daily["day"] >= 24)).astype(int)
    daily["is_peak_period"] = (
        (daily["month"] == 7) & (daily["day"] <= 15)
    ).astype(int)
    daily["is_pre_deadline_3d"] = (
        (daily["month"] == 7) & daily["day"].between(22, 25)
    ).astype(int)
    daily["is_deadline_day"] = (
        (daily["month"] == 7) & (daily["day"] == 25)
    ).astype(int)
    daily["is_post_deadline"] = (
        (daily["month"] == 7) & (daily["day"] > 25)
    ).astype(int)
    daily["is_august"] = (daily["month"] == 8).astype(int)

    daily["sat_june"] = ((daily["dow"] == 5) & (daily["month"] == 6)).astype(int)
    daily["sat_july_peak"] = (
        (daily["dow"] == 5) & (daily["month"] == 7) & (daily["day"] <= 20)
    ).astype(int)
    daily["sat_july_late"] = (
        (daily["dow"] == 5) & (daily["month"] == 7) & (daily["day"] > 20)
    ).astype(int)
    daily["sat_aug"] = ((daily["dow"] == 5) & (daily["month"] == 8)).astype(int)
    daily["mon_july"] = ((daily["dow"] == 0) & (daily["month"] == 7)).astype(int)
    daily["mon_aug"] = ((daily["dow"] == 0) & (daily["month"] == 8)).astype(int)

    daily["sin_prog"] = np.sin(np.pi * daily["progress"])
    daily["sin_2prog"] = np.sin(2 * np.pi * daily["progress"])
    daily["cos_prog"] = np.cos(np.pi * daily["progress"])
    daily["sin_dow"] = np.sin(2 * np.pi * daily["dow"] / 7)
    daily["cos_dow"] = np.cos(2 * np.pi * daily["dow"] / 7)
    return daily


def temporal_weights(years: pd.Series) -> np.ndarray:
    """Give the remote-submission period influence without treating it as normal."""
    return np.select(
        [
            years == 2020,
            years >= 2021,
            years >= 2017,
        ],
        [0.4, 2.5, 1.5],
        default=0.7,
    ).astype(float)


def generate_v33_forecast() -> pd.DataFrame:
    df_reg = pd.read_csv(DATA_FILE)
    daily = build_daily_frame(df_reg)
    feature_columns = [
        column
        for column in daily.columns
        if column not in {"year", "date", "actual_count", "target_log"}
    ]

    train = daily[daily["year"] < 2024].reset_index(drop=True)
    test = daily[daily["year"] == 2024].reset_index(drop=True)
    target = np.log1p(train["actual_count"])
    weights = temporal_weights(train["year"])

    lgb_predictions = []
    for seed_index in range(10):
        model = lgb.LGBMRegressor(
            n_estimators=130,
            learning_rate=0.035,
            num_leaves=14,
            min_child_samples=4,
            subsample=0.85,
            colsample_bytree=0.90,
            random_state=seed_index * 17 + 42,
            verbosity=-1,
        )
        model.fit(train[feature_columns], target, sample_weight=weights)
        lgb_predictions.append(np.expm1(model.predict(test[feature_columns])))
    prediction_lgb = np.mean(lgb_predictions, axis=0)

    hgb_predictions = []
    for seed_index in range(3):
        model = HistGradientBoostingRegressor(
            max_iter=130,
            learning_rate=0.035,
            max_leaf_nodes=14,
            min_samples_leaf=4,
            random_state=seed_index * 31 + 42,
        )
        model.fit(train[feature_columns], target, sample_weight=weights)
        hgb_predictions.append(np.expm1(model.predict(test[feature_columns])))
    prediction_hgb = np.mean(hgb_predictions, axis=0)

    catboost_predictions = []
    for seed_index in range(5):
        model = CatBoostRegressor(
            iterations=140,
            learning_rate=0.035,
            depth=4,
            l2_leaf_reg=4,
            random_seed=seed_index * 23 + 42,
            verbose=False,
        )
        model.fit(train[feature_columns], target, sample_weight=weights)
        catboost_predictions.append(np.expm1(model.predict(test[feature_columns])))
    prediction_catboost = np.mean(catboost_predictions, axis=0)

    blended = (
        0.60 * prediction_lgb
        + 0.10 * prediction_hgb
        + 0.30 * prediction_catboost
    )

    closed_dates = {"2024-06-22", "2024-08-24", "2024-08-31"}
    is_closed = (
        (test["is_sunday"] == 1)
        | test["date"].dt.strftime("%Y-%m-%d").isin(closed_dates)
    )
    forecast = np.where(is_closed, 0.0, np.clip(blended, 0.55, None))
    result = pd.DataFrame(
        {
            "date": test["date"].dt.strftime("%Y-%m-%d"),
            "forecast": np.round(forecast, 4),
        }
    )
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"V33 successfully generated: {OUTPUT_FILE}")
    print(f"Total days: {len(result)}")
    print(f"Total zeros: {(result['forecast'] == 0.0).sum()}")
    print(f"Total sum: {result['forecast'].sum():.2f}")
    return result


if __name__ == "__main__":
    generate_v33_forecast()
