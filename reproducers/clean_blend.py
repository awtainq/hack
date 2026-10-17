#!/usr/bin/env python3
"""
Clean blend: daily applications 06-20..08-31 with no leaderboard feedback.

z = log1p(count) is predicted as
  0.75 * mean(LightGBM, CatBoost, HistGB fitted on open days only)
  + 0.25 * CatBoost fitted on all days (closed days included as zeros).
Open-day models learn the level of a working day; the all-days model is pulled down where
history has closures and low-volume days, and its errors are only partly correlated with the
open-day trees. Sample weights per year come from regime statistics (EPGU share, July share).
The commission schedule is an input: Sundays are closed, other closed dates via --closed.

Backtest (each year predicted from earlier years only, Sundays forced to 0): RMSLE 0.488 mean over
2017-2019 and 2021-2023, vs 0.505 for reproducers/clean_baseline.py (lgb+analog+ridge, 0.53 public).

  python reproducers/clean_blend.py backtest
  python reproducers/clean_blend.py 2024 --closed 2024-06-22,2024-08-24,2024-08-31
"""

import argparse
from pathlib import Path
import warnings

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REG_FILE = ROOT / "processed_data" / "registered.csv"
BACKTEST_YEARS = (2017, 2018, 2019, 2021, 2022, 2023)
W_OPEN = 0.75

FEATS = [
    "dow", "day", "month", "iso_week", "day_of_camp", "progress",
    "is_weekend", "is_saturday", "is_sunday", "is_monday",
    "dist_start", "dist_peak", "dist_deadline", "dist_consent", "dist_orders", "dist_end",
    "is_first_week", "is_diploma_week", "is_peak_period", "is_pre_deadline_3d",
    "is_deadline_day", "is_post_deadline", "is_august",
    "sat_june", "sat_july_peak", "sat_july_late", "sat_aug", "mon_july", "mon_aug",
    "sin_prog", "sin_2prog", "cos_prog", "sin_dow", "cos_dow",
]


def daily_grid(reg_path: Path, last_year: int) -> pd.DataFrame:
    reg = pd.read_csv(reg_path, usecols=["reg_ts", "priem_doc_type_kod"])
    dt = pd.to_datetime(reg["reg_ts"]).dt.normalize()
    g = pd.DataFrame({"date": [d for y in range(2014, last_year + 1) for d in pd.date_range(f"{y}-06-20", f"{y}-08-31")]})
    g["y"] = g["date"].map(dt.value_counts()).fillna(0.0)
    g["epgu"] = g["date"].map(dt[reg["priem_doc_type_kod"] == 4].value_counts()).fillna(0.0)
    g["z"] = np.log1p(g["y"])
    g["year"] = g["date"].dt.year
    g["dow"] = g["date"].dt.dayofweek
    g["day"] = g["date"].dt.day
    g["month"] = g["date"].dt.month
    g["iso_week"] = g["date"].dt.isocalendar().week.astype(int)
    g["day_of_camp"] = (g["date"] - pd.to_datetime(g["year"].astype(str) + "-06-20")).dt.days
    g["progress"] = g["day_of_camp"] / 72.0
    g["is_weekend"] = (g["dow"] >= 5).astype(int)
    g["is_saturday"] = (g["dow"] == 5).astype(int)
    g["is_sunday"] = (g["dow"] == 6).astype(int)
    g["is_monday"] = (g["dow"] == 0).astype(int)
    for md, name in (("06-20", "start"), ("07-08", "peak"), ("07-25", "deadline"),
                     ("08-03", "consent"), ("08-09", "orders"), ("08-31", "end")):
        g[f"dist_{name}"] = (g["date"] - pd.to_datetime(g["year"].astype(str) + "-" + md)).dt.days
    jun, jul, aug = g["month"] == 6, g["month"] == 7, g["month"] == 8
    sat, mon = g["dow"] == 5, g["dow"] == 0
    flags = {
        "is_first_week": jun & (g["day"] <= 23), "is_diploma_week": jun & (g["day"] >= 24),
        "is_peak_period": jul & (g["day"] <= 15), "is_pre_deadline_3d": jul & g["day"].between(22, 25),
        "is_deadline_day": jul & (g["day"] == 25), "is_post_deadline": jul & (g["day"] > 25), "is_august": aug,
        "sat_june": sat & jun, "sat_july_peak": sat & jul & (g["day"] <= 20), "sat_july_late": sat & jul & (g["day"] > 20),
        "sat_aug": sat & aug, "mon_july": mon & jul, "mon_aug": mon & aug,
    }
    for k, v in flags.items():
        g[k] = v.astype(int)
    g["sin_prog"] = np.sin(np.pi * g["progress"])
    g["sin_2prog"] = np.sin(2 * np.pi * g["progress"])
    g["cos_prog"] = np.cos(np.pi * g["progress"])
    g["sin_dow"] = np.sin(2 * np.pi * g["dow"] / 7)
    g["cos_dow"] = np.cos(2 * np.pi * g["dow"] / 7)
    return g


def era_weights(tr: pd.DataFrame) -> np.ndarray:
    """Per-year weight from regime stats: EPGU era > standardized 2017-2019 > legacy; shifted campaigns down."""
    stats = tr.groupby("year").apply(lambda s: pd.Series({
        "july": s.loc[s["month"] == 7, "y"].sum() / max(1.0, s["y"].sum()),
        "epgu": s["epgu"].sum() / max(1.0, s["y"].sum()),
    }))
    normal_july = stats.loc[stats["july"] > 0.5, "july"].median()
    fidelity = np.exp(-18.0 * np.maximum(0.0, normal_july - stats["july"]) ** 2)
    base = np.where(stats["epgu"] > 0.01, 2.5 * fidelity,
                    np.where((stats.index >= 2017) & (fidelity > 0.8), 1.5 * fidelity,
                             np.where(fidelity <= 0.8, 0.4, 0.7)))
    return tr["year"].map(dict(zip(stats.index, base))).to_numpy(float)


def fit_mean(make, tr, te, seeds):
    w = era_weights(tr)
    return np.mean([make(s).fit(tr[FEATS], tr["z"], sample_weight=w).predict(te[FEATS]) for s in range(seeds)], axis=0)


def lgb(s):
    return LGBMRegressor(n_estimators=130, learning_rate=0.035, num_leaves=14, min_child_samples=4, subsample=0.85,
                         subsample_freq=1, colsample_bytree=0.9, random_state=s * 17 + 42, verbosity=-1)


def cat(s):
    return CatBoostRegressor(iterations=140, learning_rate=0.035, depth=4, l2_leaf_reg=4, random_seed=s * 23 + 42, verbose=False)


def hgb(s):
    return HistGradientBoostingRegressor(max_iter=130, learning_rate=0.035, max_leaf_nodes=14, min_samples_leaf=4, random_state=s * 31 + 42)


def forecast_log(tr, te, closed_dates):
    open_tr = tr[tr["y"] > 0]
    z_open = np.mean([fit_mean(lgb, open_tr, te, 5), fit_mean(cat, open_tr, te, 3), fit_mean(hgb, open_tr, te, 2)], axis=0)
    z_all = fit_mean(cat, tr, te, 3)
    z = np.clip(W_OPEN * z_open + (1 - W_OPEN) * z_all, 0, None)
    closed = te["date"].dt.strftime("%Y-%m-%d").isin(closed_dates).to_numpy() | (te["dow"] == 6).to_numpy()
    return np.where(closed, 0.0, z)


def backtest(reg_path: Path):
    g = daily_grid(reg_path, max(BACKTEST_YEARS))
    rows = []
    for ty in BACKTEST_YEARS:
        tr, te = g[g["year"] < ty], g[g["year"] == ty].reset_index(drop=True)
        closed = set(te.loc[te["y"] == 0, "date"].dt.strftime("%Y-%m-%d"))  # schedule known in advance
        rows.append({"year": ty, "rmsle": float(np.sqrt(np.mean((te["z"] - forecast_log(tr, te, closed)) ** 2)))})
    r = pd.DataFrame(rows)
    print(r.round(4).to_string(index=False))
    print(f"mean: {r['rmsle'].mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="'backtest' or the forecast year")
    ap.add_argument("--closed", default="", help="comma-separated closed dates; Sundays are always closed")
    ap.add_argument("--reg", default=str(REG_FILE))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.target == "backtest":
        return backtest(Path(a.reg))
    year = int(a.target)
    g = daily_grid(Path(a.reg), year)
    tr, te = g[g["year"] < year], g[g["year"] == year].reset_index(drop=True)
    pred = np.expm1(forecast_log(tr, te, [d for d in a.closed.split(",") if d]))
    out = pd.DataFrame({"date": te["date"].dt.strftime("%Y-%m-%d"), "forecast": np.round(pred, 2)})
    path = Path(a.out) if a.out else ROOT / f"forecast_{year}_clean_blend.csv"
    out.to_csv(path, index=False)
    print(f"{path.name}: sum={out['forecast'].sum():.0f}, zeros={(out['forecast'] == 0).sum()}")


if __name__ == "__main__":
    main()
