#!/usr/bin/env python3
"""Compare leaderboard-best V61 (fixed 2024 weights) with V65 on 2022/2023.

V61 is reconstructed from the historical V59/V60 backtest predictions and a
fresh historical clean pooled-model prediction. V61's weights are frozen from
its 2024 calibration; they are not refit against 2022/2023 outcomes.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reproducers"))
import clean_deadline_blend as clean

YEARS = (2022, 2023)
DAILY_COMPONENTS = ROOT / "output" / "v60" / "backtest_daily.csv"
V65_METRICS = ROOT / "output" / "v65" / "search_metrics.csv"
V61_CALIBRATION = ROOT / "output" / "v61" / "calibration.json"
OUT = ROOT / "output" / "v61" / "backtest_v61_vs_v65.csv"


def v61_weights() -> dict[str, float]:
    meta = json.loads(V61_CALIBRATION.read_text())
    return meta["weights"]


def pooled_predictions(year: int) -> pd.DataFrame:
    mag_end = pd.Timestamp(f"{year}-07-29")
    while mag_end.dayofweek >= 5:
        mag_end -= pd.Timedelta(days=1)
    g = clean.daily_grid(clean.REG_FILE, year,
                         target_ends={"bak": "07-25", "mag": mag_end.strftime("%m-%d")})
    tr, te = g[g.year < year], g[g.year == year].reset_index(drop=True)
    z = clean.predict_open_log(tr, te)
    closed = clean.closed_from_history(tr, te)
    prediction = clean.apply_schedule(te, z, closed)
    return pd.DataFrame({"date": te.date.dt.strftime("%Y-%m-%d"),
                         "pooled": np.round(np.expm1(prediction), 2),
                         "truth": te.y.to_numpy()})


def main() -> None:
    component = pd.read_csv(DAILY_COMPONENTS, parse_dates=["date"])
    v65 = pd.read_csv(V65_METRICS)
    selected65 = v65[(v65.config_id == 7) & v65.year.isin(YEARS)].set_index("year")
    check65 = json.loads((ROOT / "output" / "v65" / "check_2023.json").read_text())
    weights = v61_weights()
    rows = []
    for year in YEARS:
        d = component[component.year == year].copy()
        pool = pooled_predictions(year)
        d["date_key"] = d.date.dt.strftime("%Y-%m-%d")
        pool = pool.set_index("date")
        d["pooled"] = d.date_key.map(pool.pooled)
        if d.pooled.isna().any() or len(d) != 73:
            raise ValueError(f"Missing or misaligned pooled predictions for {year}")

        # The first two terms come from V60's saved, earlier-year-only backtest;
        # this pooled model is trained only on seasons before the target year.
        logs = {name: np.log1p(d[col].to_numpy()) for name, col in
                (("v59", "v59_forecast"), ("v60", "v60_forecast"), ("pooled", "pooled"))}
        z = sum(weights[name] * logs[name] for name in weights)
        closed = ((d.v59_forecast == 0) & (d.v60_forecast == 0) & (d.pooled == 0))
        pred = np.where(closed, 0., np.maximum(0., z))
        rmsle61 = float(np.sqrt(np.mean((pred - np.log1p(d.y.to_numpy())) ** 2)))
        if year == 2023:
            rmsle65, weekday65 = float(check65["rmsle"]), float(check65["weekday_rmsle"])
        else:
            metrics65 = selected65.loc[year]
            rmsle65, weekday65 = float(metrics65.rmsle), float(metrics65.weekday_rmsle)
        rows.append({"year": year, "V61_RMSLE": rmsle61,
                     "V65_RMSLE": rmsle65,
                     "V61_weekday_RMSLE": float(np.sqrt(np.mean(
                         (pred[d.dow.to_numpy() < 5] - np.log1p(d.y.to_numpy()[d.dow.to_numpy() < 5]))**2))),
                     "V65_weekday_RMSLE": weekday65,
                     "closed_days_V61": int(closed.sum()),
                     "closed_days_V65": int((d.closed).sum())})
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Saved: {OUT}")
    print(f"Frozen V61 weights: {weights}")


if __name__ == "__main__":
    main()
