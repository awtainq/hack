"""
V51: Stream-Decomposed Standalone Engine (trained ONLY on raw data).

No previous forecast CSVs are used. Sources: processed_data/registered.csv
and processed_data/included_ks.csv.

Components (blended in log space):
  A. Total-flow Tri-Ensemble (LGBM/HGB/CatBoost) trained on active days only
     (Sundays and COVID-2020 removed -> no Sunday leakage into weekday leaves).
  B. Stream decomposition:
       - Budget Bachelor (zayav_type=1 and abiturient in a budget KS list),
         statutory zero after 25 July (2021 COVID extension excluded).
       - Everything else (contract bachelor, masters, SPO, PhD).
     Forecast = sum of two stream models.
  C. Clean kernel analog: weekdays matched to weekdays (+-3 campaign days,
     Gaussian kernel, same-DOW bonus), Saturdays matched to Saturdays only.

Run with --backtest to evaluate the identical pipeline on 2023 (train < 2023).
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v34_analog_calibrated_engine import make_daily  # feature builder only

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "forecast_2024_v51_stream_decomposed.csv"
W_TREE, W_STREAM, W_KERNEL = 0.40, 0.35, 0.25
CLOSED_2024 = {"2024-06-22", "2024-08-24", "2024-08-31"}


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    for ms, name in [("07-29", "masters_end"), ("07-27", "rank_list"), ("07-15", "mid_july")]:
        d[f"dist_{name}"] = (d["date"] - pd.to_datetime(d["year"].astype(str) + "-" + ms)).dt.days
    d["is_masters_wave"] = ((d["month"] == 7) & d["day"].between(26, 29)).astype(int)
    d["is_july_surge"] = ((d["month"] == 7) & d["day"].between(8, 15) & (d["dow"] < 5)).astype(int)
    d["sin_3prog"] = np.sin(3 * np.pi * d["progress"])
    return d


def year_weights(y: pd.Series, target: int) -> np.ndarray:
    age = target - y
    return np.select([age <= 2, age == 3, age <= 7], [3.2, 2.2, 1.3], default=0.6).astype(float)


def tri_ensemble(tr: pd.DataFrame, te: pd.DataFrame, feats, w) -> np.ndarray:
    y = np.log1p(tr["actual_count"].to_numpy())
    L, C, H = [], [], []
    for s in range(10):
        m = lgb.LGBMRegressor(n_estimators=130, learning_rate=0.035, num_leaves=14,
                              min_child_samples=4, subsample=0.85, subsample_freq=1,
                              colsample_bytree=0.9, random_state=s * 17 + 42, verbosity=-1)
        L.append(m.fit(tr[feats], y, sample_weight=w).predict(te[feats]))
        if s < 5:
            c = CatBoostRegressor(iterations=140, learning_rate=0.035, depth=4, l2_leaf_reg=4,
                                  random_seed=s * 23 + 42, verbose=False)
            C.append(c.fit(tr[feats], y, sample_weight=w).predict(te[feats]))
        if s < 3:
            h = HistGradientBoostingRegressor(max_iter=130, learning_rate=0.035, max_leaf_nodes=14,
                                              min_samples_leaf=4, random_state=s * 31 + 42)
            H.append(h.fit(tr[feats], y, sample_weight=w).predict(te[feats]))
    log_pred = 0.6 * np.mean(L, 0) + 0.1 * np.mean(H, 0) + 0.3 * np.mean(C, 0)
    return np.clip(np.expm1(log_pred), 0.0, None)


def kernel_analog(tr: pd.DataFrame, te: pd.DataFrame, target: int) -> np.ndarray:
    src = tr[(tr["dow"] < 6) & (tr["actual_count"] >= 2) & (tr["year"] != 2020)].copy()
    lc = np.log1p(src["actual_count"].to_numpy())
    wy = year_weights(src["year"], target) / (1 + 0.22 * (target - src["year"].to_numpy()))
    doc_s, dow_s = src["day_of_camp"].to_numpy(), src["dow"].to_numpy()
    out = []
    for doc, dow in zip(te["day_of_camp"], te["dow"]):
        if dow == 6:
            out.append(0.0)
            continue
        if dow == 5:
            m = (dow_s == 5) & (np.abs(doc_s - doc) <= 6)
            k = np.exp(-0.5 * ((doc_s[m] - doc) / 2.8) ** 2)
        else:
            m = (dow_s < 5) & (np.abs(doc_s - doc) <= 3)
            k = np.exp(-0.5 * ((doc_s[m] - doc) / 1.15) ** 2) * np.where(dow_s[m] == dow, 1.85, 1.0)
        out.append(np.expm1(np.average(lc[m], weights=wy[m] * k)) if m.any() else 0.0)
    return np.array(out)


def forecast(target: int) -> pd.DataFrame:
    reg = pd.read_csv(ROOT / "processed_data" / "registered.csv")
    ks = pd.read_csv(ROOT / "processed_data" / "included_ks.csv")
    budget_ids = set(ks.loc[ks["ob_osnova_kod"] == 1, "abitur_kod"])
    is_bb = (reg["zayav_type_kod"] == 1) & reg["abitur_kod"].isin(budget_ids)

    d_all = add_features(make_daily(reg))
    d_bb = add_features(make_daily(reg[is_bb]))
    d_nbb = add_features(make_daily(reg[~is_bb]))
    feats = [c for c in d_all.columns if c not in {"year", "date", "actual_count"}]

    def active(d):
        return d[(d["year"] < target) & (d["year"] != 2020) & (d["dow"] < 6)].reset_index(drop=True)

    te = d_all[d_all["year"] == target].reset_index(drop=True)
    post_deadline = ((te["month"] == 8) | ((te["month"] == 7) & (te["day"] > 25))).to_numpy()

    tr = active(d_all)
    p_tree = tri_ensemble(tr, te, feats, year_weights(tr["year"], target))

    tr_bb = active(d_bb)
    tr_bb = tr_bb[~((tr_bb["year"] == 2021) & (tr_bb["dist_deadline"] > 0))].reset_index(drop=True)
    p_bb = np.where(post_deadline, 0.0, tri_ensemble(tr_bb, te, feats, year_weights(tr_bb["year"], target)))
    tr_nbb = active(d_nbb)
    p_nbb = tri_ensemble(tr_nbb, te, feats, year_weights(tr_nbb["year"], target))
    p_stream = p_bb + p_nbb

    p_kern = kernel_analog(d_all[d_all["year"] < target], te, target)

    u = W_TREE * np.log1p(p_tree) + W_STREAM * np.log1p(p_stream) + W_KERNEL * np.log1p(p_kern)
    pred = np.expm1(u)

    ds = te["date"].dt.strftime("%Y-%m-%d")
    closed = (te["dow"] == 6).to_numpy()
    if target == 2024:
        closed |= ds.isin(CLOSED_2024).to_numpy()
    pred = np.where(closed, 0, np.maximum(np.rint(pred), 1)).astype(int)
    return pd.DataFrame({"date": ds, "forecast": pred, "actual": te["actual_count"].to_numpy()})


def rmsle(a, b):
    return float(np.sqrt(np.mean((np.log1p(a) - np.log1p(b)) ** 2)))


if __name__ == "__main__":
    if "--backtest" in sys.argv:
        bt = forecast(2023)
        print(f"Backtest 2023 RMSLE: {rmsle(bt['forecast'], bt['actual']):.5f}")
    res = forecast(2024)[["date", "forecast"]]
    res.to_csv(OUT, index=False)
    res.to_csv(ROOT / "forecast_2024.csv", index=False)
    print(f"Saved {OUT.name}: sum={res['forecast'].sum()}, zeros={(res['forecast'] == 0).sum()}")
    for d in ["2024-07-08", "2024-07-10", "2024-07-25", "2024-07-27", "2024-08-03", "2024-08-10", "2024-08-17"]:
        print(d, int(res.loc[res["date"] == d, "forecast"].iloc[0]))

