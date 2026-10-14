"""
V37: category-decomposed daily ensemble.

Instead of fitting one model to the aggregate daily count, this version fits
one log-count model per application type and sums the non-negative forecasts.
The decomposition lets the model learn that the composition of the campaign
changed over time, while the final metric is still evaluated on the aggregate.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from v34_analog_calibrated_engine import make_daily


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "processed_data" / "registered.csv"
OUTPUT_FILE = ROOT / "forecast_2024_v37_category_decomposed.csv"
CALIBRATION_SCALE = 1.16


def make_category_counts(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["reg_ts"]).dt.normalize()
    raw["year"] = raw["date"].dt.year
    counts = (
        raw.groupby(["date", "zayav_type_kod"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(1, 7), fill_value=0)
    )
    counts.columns = [f"type_{int(column)}" for column in counts.columns]
    return counts.reset_index()


def sample_weights(years: pd.Series) -> np.ndarray:
    return np.select(
        [years == 2020, years >= 2021, years >= 2017],
        [0.25, 2.5, 1.5],
        default=0.7,
    ).astype(float)


def predict_category(
    train: pd.DataFrame, test: pd.DataFrame, feature_columns: list[str], target: str
) -> np.ndarray:
    y = np.log1p(train[target].to_numpy())
    weights = sample_weights(train["year"])
    lgb_predictions = []
    cat_predictions = []
    for seed_index in range(4):
        lgb_model = lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.032,
            num_leaves=12,
            min_child_samples=5,
            subsample=0.88,
            colsample_bytree=0.9,
            random_state=seed_index * 29 + 101,
            verbosity=-1,
        )
        lgb_model.fit(train[feature_columns], y, sample_weight=weights)
        lgb_predictions.append(lgb_model.predict(test[feature_columns]))

        cat_model = CatBoostRegressor(
            iterations=160,
            learning_rate=0.032,
            depth=4,
            l2_leaf_reg=5,
            random_seed=seed_index * 37 + 101,
            verbose=False,
        )
        cat_model.fit(train[feature_columns], y, sample_weight=weights)
        cat_predictions.append(cat_model.predict(test[feature_columns]))

    prediction_log = 0.7 * np.mean(lgb_predictions, axis=0) + 0.3 * np.mean(
        cat_predictions, axis=0
    )
    return np.clip(np.expm1(prediction_log), 0.0, None)


def generate_v37_forecast() -> pd.DataFrame:
    raw = pd.read_csv(DATA_FILE)
    daily = make_daily(raw)
    category_counts = make_category_counts(raw)
    daily = daily.merge(category_counts, on="date", how="left")
    for column in [f"type_{index}" for index in range(1, 7)]:
        daily[column] = daily[column].fillna(0.0)

    train = daily[daily["year"] < 2024].reset_index(drop=True)
    test = daily[daily["year"] == 2024].reset_index(drop=True)
    feature_columns = [
        column
        for column in daily.columns
        if column
        not in {
            "year",
            "date",
            "actual_count",
            "target_log",
            *[f"type_{index}" for index in range(1, 7)],
        }
    ]

    category_predictions = []
    for index in range(1, 7):
        category_predictions.append(
            predict_category(train, test, feature_columns, f"type_{index}")
        )
    # Separate log models are conservative after summation; this factor was
    # selected from the 2021-2023 holdout scale sweep.
    prediction = np.sum(category_predictions, axis=0) * CALIBRATION_SCALE

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
    print(f"V37 successfully generated: {OUTPUT_FILE}")
    print(f"Total days: {len(result)}")
    print(f"Total zeros: {(result['forecast'] == 0.0).sum()}")
    print(f"Total sum: {result['forecast'].sum():.2f}")
    print(f"Peak day (2024-07-08): {result.loc[result['date'] == '2024-07-08', 'forecast'].iloc[0]:.2f}")
    return result


if __name__ == "__main__":
    generate_v37_forecast()
