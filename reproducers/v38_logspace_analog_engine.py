"""
V38: log-space ensemble with a conservative historical analog.

RMSLE is Euclidean error in log1p space, so the tree and analog components
are blended before converting back to counts. This avoids giving the largest
raw-count days disproportionate influence during the blend.
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from v34_analog_calibrated_engine import analog_prediction, make_daily


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "processed_data" / "registered.csv"
OUTPUT_FILE = ROOT / "forecast_2024_v38_logspace_analog.csv"
ANALOG_WEIGHT = 0.10
CALIBRATION_SCALE = 1.12


def weights(years: pd.Series) -> np.ndarray:
    return np.select(
        [years == 2020, years >= 2021, years >= 2017],
        [0.4, 2.5, 1.5],
        default=0.7,
    ).astype(float)


def generate_v38_forecast() -> pd.DataFrame:
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

    lgb_logs = []
    cat_logs = []
    for seed_index in range(10):
        lgb_model = lgb.LGBMRegressor(
            n_estimators=130,
            learning_rate=0.035,
            num_leaves=14,
            min_child_samples=4,
            subsample=0.85,
            colsample_bytree=0.90,
            random_state=seed_index * 17 + 42,
            verbosity=-1,
        )
        lgb_model.fit(train[features], target, sample_weight=sample_weight)
        lgb_logs.append(lgb_model.predict(test[features]))

        if seed_index < 5:
            cat_model = CatBoostRegressor(
                iterations=140,
                learning_rate=0.035,
                depth=5,
                l2_leaf_reg=4,
                random_seed=seed_index * 23 + 42,
                verbose=False,
            )
            cat_model.fit(train[features], target, sample_weight=sample_weight)
            cat_logs.append(cat_model.predict(test[features]))

    tree_log = 0.60 * np.mean(lgb_logs, axis=0) + 0.30 * np.mean(
        cat_logs, axis=0
    ) + 0.10 * np.mean(lgb_logs, axis=0)
    analog_log = np.log1p(analog_prediction(train, test))
    prediction = np.expm1(
        (1.0 - ANALOG_WEIGHT) * tree_log + ANALOG_WEIGHT * analog_log
    )
    prediction *= CALIBRATION_SCALE

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
    print(f"V38 successfully generated: {OUTPUT_FILE}")
    print(f"Analog weight: {ANALOG_WEIGHT}")
    print(f"Calibration scale: {CALIBRATION_SCALE}")
    print(f"Total days: {len(result)}")
    print(f"Total zeros: {(result['forecast'] == 0.0).sum()}")
    print(f"Total sum: {result['forecast'].sum():.2f}")
    print(
        "Peak day (2024-07-08): "
        f"{result.loc[result['date'] == '2024-07-08', 'forecast'].iloc[0]:.2f}"
    )
    return result


if __name__ == "__main__":
    generate_v38_forecast()
