#!/usr/bin/env python3
"""
Clean deadline-aware blend: daily applications 06-20..08-31 with no leaderboard feedback.

Same blend as reproducers/clean_blend.py (0.75 * open-day LightGBM/CatBoost/HistGB + 0.25 * all-day
CatBoost, log1p space), plus the intake end date of each stream as features: days to the end of
bachelor intake and of budget master's intake. These dates are published in the admission rules
months ahead; for history they are recovered from the data as the last day of the intake rush,
for the target year they are passed explicitly. SPbGEU 2024: bachelor 07-25 (federal deadline),
budget master's 07-29 17:00 (t.me/askmag/121). SPbGEU 2026 rules: master's 08-01.

Schedule: Sundays are closed. Saturdays are closed by a majority vote of earlier years (era-weighted)
within their slot: first Saturday of the campaign / August up to 08-21 / after 08-21 / July. For 2024
this closes 06-22, 08-24, 08-31. An official schedule can be passed with --closed instead.

Backtest (each year predicted from earlier years only): RMSLE 0.453 with the true schedule and 0.533
with the history rule, mean over 2017-2019 and 2021-2023; clean_blend.py: 0.488 with the true schedule.
Public 2024: 0.458.

  python reproducers/clean_deadline_blend.py backtest
  python reproducers/clean_deadline_blend.py 2024 --bak-end 07-25 --mag-end 07-29
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
    "dist_end_bak", "after_end_bak", "dist_end_mag", "after_end_mag",
]


def intake_ends(reg: pd.DataFrame) -> dict:
    """Per year: last day of the bachelor and master's intake rush (count >= share of the July peak, then a drop)."""
    stream = np.select([reg["zayav_type_name"].str.startswith("Бакалавр"),
                        reg["zayav_type_name"].str.startswith("Магистр")], ["bak", "mag"], "other")
    counts = reg.groupby([stream, reg["dt"]]).size()
    out = {}
    for y in sorted(reg["dt"].dt.year.unique()):
        out[y] = {}
        for st, share in (("bak", 0.2), ("mag", 0.25)):
            ser = counts[st].reindex(pd.date_range(f"{y}-06-20", f"{y}-08-31"), fill_value=0)
            thr = share * ser[f"{y}-07-01":f"{y}-07-25"].max()
            for d, v in ser[f"{y}-07-20":f"{y}-08-20"].items():
                if v >= thr and (ser[d + pd.Timedelta(days=1):d + pd.Timedelta(days=3)] < thr).all():
                    out[y][st] = d.strftime("%m-%d")
                    break
    return out


def daily_grid(reg_path: Path, last_year: int, target_ends: dict) -> pd.DataFrame:
    reg = pd.read_csv(reg_path, usecols=["reg_ts", "priem_doc_type_kod", "zayav_type_name"])
    reg["dt"] = pd.to_datetime(reg["reg_ts"]).dt.normalize()
    ends = intake_ends(reg)
    if target_ends:
        ends[last_year] = target_ends
    g = pd.DataFrame({"date": [d for y in range(2014, last_year + 1) for d in pd.date_range(f"{y}-06-20", f"{y}-08-31")]})
    g["y"] = g["date"].map(reg["dt"].value_counts()).fillna(0.0)
    g["epgu"] = g["date"].map(reg.loc[reg["priem_doc_type_kod"] == 4, "dt"].value_counts()).fillna(0.0)
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
    for st in ("bak", "mag"):
        end = pd.to_datetime([f"{y}-{ends[y][st]}" for y in g["year"]])
        g[f"dist_end_{st}"] = (g["date"] - end).dt.days
        g[f"after_end_{st}"] = (g[f"dist_end_{st}"] > 0).astype(int)
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
    sat_n = g[g["dow"] == 5].groupby("year").cumcount().reindex(g.index)
    md = g["date"].dt.strftime("%m-%d")
    g["sat_slot"] = np.select([sat_n == 0, (g["dow"] == 5) & (md > "08-21"), (g["dow"] == 5) & (md >= "08-01"), g["dow"] == 5],
                              ["first", "late", "aug_mid", "july"], "")
    g["sin_prog"] = np.sin(np.pi * g["progress"])
    g["sin_2prog"] = np.sin(2 * np.pi * g["progress"])
    g["cos_prog"] = np.cos(np.pi * g["progress"])
    g["sin_dow"] = np.sin(2 * np.pi * g["dow"] / 7)
    g["cos_dow"] = np.cos(2 * np.pi * g["dow"] / 7)
    return g


def era_map(tr: pd.DataFrame) -> dict:
    """Per-year weight from regime stats: EPGU era > standardized 2017-2019 > legacy; shifted campaigns down."""
    total = tr.groupby("year")["y"].sum().clip(lower=1.0)
    stats = pd.DataFrame({"july": tr[tr["month"] == 7].groupby("year")["y"].sum() / total,
                          "epgu": tr.groupby("year")["epgu"].sum() / total})
    normal_july = stats.loc[stats["july"] > 0.5, "july"].median()
    fidelity = np.exp(-18.0 * np.maximum(0.0, normal_july - stats["july"]) ** 2)
    base = np.where(stats["epgu"] > 0.01, 2.5 * fidelity,
                    np.where((stats.index >= 2017) & (fidelity > 0.8), 1.5 * fidelity,
                             np.where(fidelity <= 0.8, 0.4, 0.7)))
    return dict(zip(stats.index, base))


def era_weights(tr: pd.DataFrame) -> np.ndarray:
    return tr["year"].map(era_map(tr)).to_numpy(float)


def closed_from_history(tr: pd.DataFrame, te: pd.DataFrame) -> list:
    """Saturdays whose slot had zero applications in most earlier years (era-weighted, 2020 excluded)."""
    sat = tr[(tr["dow"] == 5) & (tr["year"] != 2020)]
    w = sat["year"].map(era_map(tr))
    p_zero = (w * (sat["y"] == 0)).groupby(sat["sat_slot"]).sum() / w.groupby(sat["sat_slot"]).sum()
    closed = te["sat_slot"].map(lambda s: bool(s) and p_zero.get(s, 0.0) > 0.5)
    return list(te.loc[closed, "date"].dt.strftime("%Y-%m-%d"))


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


def predict_open_log(tr, te):
    open_tr = tr[tr["y"] > 0]
    z_open = np.mean([fit_mean(lgb, open_tr, te, 5), fit_mean(cat, open_tr, te, 3), fit_mean(hgb, open_tr, te, 2)], axis=0)
    z_all = fit_mean(cat, tr, te, 3)
    return np.clip(W_OPEN * z_open + (1 - W_OPEN) * z_all, 0, None)


def apply_schedule(te, z, closed_dates):
    closed = te["date"].dt.strftime("%Y-%m-%d").isin(closed_dates).to_numpy() | (te["dow"] == 6).to_numpy()
    return np.where(closed, 0.0, z)


def backtest(reg_path: Path):
    rows = []
    for ty in BACKTEST_YEARS:
        g = daily_grid(reg_path, ty, target_ends={})  # test-year ends recovered from data, as if read from its rules
        tr, te = g[g["year"] < ty], g[g["year"] == ty].reset_index(drop=True)
        z = predict_open_log(tr, te)
        true_closed = list(te.loc[te["y"] == 0, "date"].dt.strftime("%Y-%m-%d"))
        weekday_closed = list(te.loc[(te["y"] == 0) & (te["dow"] < 5), "date"].dt.strftime("%Y-%m-%d"))
        rmsle = lambda closed: float(np.sqrt(np.mean((te["z"] - apply_schedule(te, z, closed)) ** 2)))
        rows.append({"year": ty, "schedule_known": rmsle(true_closed),
                     "schedule_from_history": rmsle(closed_from_history(tr, te) + weekday_closed)})
    r = pd.DataFrame(rows)
    print(r.round(4).to_string(index=False))
    print("mean:", r[["schedule_known", "schedule_from_history"]].mean().round(4).to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="'backtest' or the forecast year")
    ap.add_argument("--bak-end", default="07-25", help="last day of bachelor intake (MM-DD)")
    ap.add_argument("--mag-end", default=None, help="last day of budget master's intake (MM-DD), from the admission rules")
    ap.add_argument("--closed", default=None, help="official closed dates, comma-separated; default: Sundays + history rule")
    ap.add_argument("--reg", default=str(REG_FILE))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.target == "backtest":
        return backtest(Path(a.reg))
    if a.mag_end is None:
        ap.error("--mag-end is required for a forecast")
    year = int(a.target)
    g = daily_grid(Path(a.reg), year, target_ends={"bak": a.bak_end, "mag": a.mag_end})
    tr, te = g[g["year"] < year], g[g["year"] == year].reset_index(drop=True)
    closed = [d for d in a.closed.split(",") if d] if a.closed is not None else closed_from_history(tr, te)
    print("closed besides Sundays:", ", ".join(closed) or "none")
    pred = np.expm1(apply_schedule(te, predict_open_log(tr, te), closed))
    out = pd.DataFrame({"date": te["date"].dt.strftime("%Y-%m-%d"), "forecast": np.round(pred, 2)})
    path = Path(a.out) if a.out else ROOT / f"forecast_{year}_clean_deadline_blend.csv"
    out.to_csv(path, index=False)
    print(f"{path.name}: sum={out['forecast'].sum():.0f}, zeros={(out['forecast'] == 0).sum()}")


if __name__ == "__main__":
    main()
