#!/usr/bin/env python3
"""
Clean baseline: daily applications 06-20..08-31 with no leaderboard feedback.

Forecast = mean in log1p space of three different models, all fitted on open days only:
  1) LightGBM on calendar features,
  2) analog: same weekday within +-3 campaign days of past years,
  3) ridge: smoothed campaign-day curve + weekday x phase effects + year level.
The commission schedule is an input: closed days are passed explicitly (Sundays are always
closed); without a schedule, weekend days are scaled by the historical P(open), which is the
RMSLE-optimal answer when the schedule is unknown.

  python reproducers/clean_baseline.py backtest
  python reproducers/clean_baseline.py 2024 --closed 2024-06-22,2024-08-24,2024-08-31
"""

import argparse
from pathlib import Path
import warnings

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
REG_FILE = ROOT / "processed_data" / "registered.csv"

FEATS = ["doc", "dow", "month", "day", "dist_deadline", "phase", "is_sat", "is_mon", "sat_phase", "mon_phase"]
PHASE_EDGES = [-1, 13, 35, 41, 52, 72]  # start / peak until 07-25 / lists / consents / tail


def daily_grid(reg_path: Path, last_year: int) -> pd.DataFrame:
    reg = pd.read_csv(reg_path, usecols=["reg_ts"])
    counts = pd.to_datetime(reg["reg_ts"]).dt.normalize().value_counts()
    days = [d for y in range(2014, last_year + 1) for d in pd.date_range(f"{y}-06-20", f"{y}-08-31")]
    g = pd.DataFrame({"date": days})
    g["y"] = g["date"].map(counts).fillna(0.0)
    g["year"] = g["date"].dt.year
    g["dow"] = g["date"].dt.dayofweek
    g["doc"] = (g["date"] - pd.to_datetime(g["year"].astype(str) + "-06-20")).dt.days
    g["month"] = g["date"].dt.month
    g["day"] = g["date"].dt.day
    g["dist_deadline"] = g["doc"] - 35
    g["phase"] = pd.cut(g["doc"], PHASE_EDGES, labels=False)
    g["is_sat"] = (g["dow"] == 5).astype(int)
    g["is_mon"] = (g["dow"] == 0).astype(int)
    g["sat_phase"] = g["is_sat"] * (g["phase"] + 1)
    g["mon_phase"] = g["is_mon"] * (g["phase"] + 1)
    g["z"] = np.log1p(g["y"])
    return g


def year_weight(years, target_year, decay=0.25):
    w = np.exp(-decay * (target_year - np.asarray(years, float)))
    return np.where(np.asarray(years) == 2020, 0.2 * w, w)  # 2020: COVID-shifted campaign


def m_lgb(tr, te):
    tr = tr[tr["y"] > 0]
    w = year_weight(tr["year"], te["year"].iloc[0])
    preds = [
        LGBMRegressor(n_estimators=150, learning_rate=0.035, num_leaves=14, min_child_samples=4,
                      subsample=0.85, subsample_freq=1, colsample_bytree=0.9, random_state=s, verbosity=-1)
        .fit(tr[FEATS], tr["z"], sample_weight=w).predict(te[FEATS])
        for s in range(5)
    ]
    return np.mean(preds, axis=0)


def m_analog(tr, te, win=3):
    src = tr[tr["y"] > 0]
    out = []
    for doc, dow in zip(te["doc"], te["dow"]):
        m = src[(src["dow"] == dow) & ((src["doc"] - doc).abs() <= win)]
        if m.empty:
            m = src[(src["doc"] - doc).abs() <= win]
        w = year_weight(m["year"], te["year"].iloc[0]) / (1 + (m["doc"] - doc).abs().to_numpy())
        out.append(np.average(m["z"], weights=w))
    return np.asarray(out)


def m_ridge(tr, te, lam_smooth=3.0):
    tr = tr[tr["y"] > 0]
    years = sorted(tr["year"].unique())
    yw = year_weight(years, te["year"].iloc[0]) * (np.array(years) != 2020)

    def design(df, fit):
        X = [np.eye(73)[df["doc"].to_numpy()]]
        for p in range(5):
            for d in (0, 4, 5, 6):
                X.append(((df["phase"] == p) & (df["dow"] == d)).to_numpy(float)[:, None])
        if fit:
            X.append(np.stack([(df["year"] == y).to_numpy(float) for y in years[:-1]], 1))
        else:
            X.append(np.tile(yw[:-1] / yw.sum(), (len(df), 1)))  # mean level of recent years
        return np.hstack(X)

    X = design(tr, True)
    w = year_weight(tr["year"], te["year"].iloc[0])
    D = np.diff(np.eye(73), 2, axis=0)
    D[[33, 34, 35], :] = 0  # allow a kink at the 07-25 deadline
    P = 1e-3 * np.eye(X.shape[1])
    P[:73, :73] += lam_smooth * D.T @ D
    beta = np.linalg.solve(X.T @ (w[:, None] * X) + P, X.T @ (w * tr["z"].to_numpy()))
    return design(te, False) @ beta


def p_open(tr, te, decay=0.35):
    src = tr[tr["dow"] >= 5]
    out = np.ones(len(te))
    for i, (dow, ph) in enumerate(zip(te["dow"], te["phase"])):
        if dow >= 5:
            m = src[(src["dow"] == dow) & (src["phase"] == ph)]
            out[i] = np.average(m["y"] > 0, weights=year_weight(m["year"], te["year"].iloc[0], decay)) if len(m) else 0.0
    return out


def forecast_log(tr, te, closed_dates=None):
    z = np.clip(np.mean([m_lgb(tr, te), m_analog(tr, te), m_ridge(tr, te)], axis=0), 0, None)
    if closed_dates is None:
        return p_open(tr, te) * z
    closed = te["date"].dt.strftime("%Y-%m-%d").isin(closed_dates).to_numpy() | (te["dow"] == 6).to_numpy()
    return np.where(closed, 0.0, z)


def rmsle_log(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def backtest(reg_path: Path):
    g = daily_grid(reg_path, 2023)
    rows = []
    for ty in (2019, 2021, 2022, 2023):
        tr, te = g[g["year"] < ty], g[g["year"] == ty].reset_index(drop=True)
        z = te["z"].to_numpy()
        oracle_closed = set(te.loc[te["y"] == 0, "date"].dt.strftime("%Y-%m-%d"))
        rows.append({"year": ty,
                     "schedule_known": rmsle_log(z, forecast_log(tr, te, oracle_closed)),
                     "schedule_soft": rmsle_log(z, forecast_log(tr, te))})
    r = pd.DataFrame(rows)
    print(r.round(4).to_string(index=False))
    print("mean:", r[["schedule_known", "schedule_soft"]].mean().round(4).to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="'backtest' or the forecast year")
    ap.add_argument("--closed", default=None, help="comma-separated closed dates; Sundays are always closed")
    ap.add_argument("--reg", default=str(REG_FILE))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.target == "backtest":
        return backtest(Path(a.reg))
    year = int(a.target)
    g = daily_grid(Path(a.reg), year)
    tr, te = g[g["year"] < year], g[g["year"] == year].reset_index(drop=True)
    closed = a.closed.split(",") if a.closed is not None else None
    pred = np.expm1(forecast_log(tr, te, closed))
    out = pd.DataFrame({"date": te["date"].dt.strftime("%Y-%m-%d"), "forecast": np.round(pred, 2)})
    path = Path(a.out) if a.out else ROOT / f"forecast_{year}_clean_{'sched' if closed is not None else 'soft'}.csv"
    out.to_csv(path, index=False)
    print(f"{path.name}: sum={out['forecast'].sum():.0f}, zeros={(out['forecast'] == 0).sum()}")


if __name__ == "__main__":
    main()
