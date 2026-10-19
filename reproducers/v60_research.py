"""Experiments for V60. All validation forecasts use strictly earlier campaigns.

The nominal target deadlines are inputs, never recovered from validation labels.
Prediction caches contain no fitted models and can be deleted to refit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import v59_standalone_4_4 as base

CACHE = ROOT / "output" / "v60_research"
YEARS = (2017, 2018, 2019, 2021, 2022, 2023, 2024)


def build_grid(year: int) -> pd.DataFrame:
    # Known 2024 inputs; for backtesting the same nominal policy is declared
    # before seeing the held-out year's counts. Weekend master's dates move back.
    mag = pd.Timestamp(f"{year}-07-29")
    while mag.dayofweek >= 5:
        mag -= pd.Timedelta(days=1)
    g = base.create_stream_grid(base.REG_FILE, base.KS_FILE, year,
                                {"bak": "07-25", "mag": mag.strftime("%m-%d")})
    reg = pd.read_csv(base.REG_FILE)
    reg["date"] = pd.to_datetime(reg.reg_ts).dt.normalize()
    # Adding or changing held-out/future applicants cannot affect training data.
    reg = reg[reg.date.dt.year < year].copy()
    ks = pd.read_csv(base.KS_FILE)
    ks["date"] = pd.to_datetime(ks.reg_ts).dt.normalize()
    ks = ks[ks.date.dt.year < year]
    keys = ["abitur_kod", "date", "zayav_type_kod"]
    budget = ks.assign(budget=ks.ob_osnova_name.isin(["Бюджет", "Гослиния"]))
    budget = budget.groupby(keys).budget.max()
    reg = reg.join(budget, on=keys)
    degree = np.select([reg.zayav_type_name.str.startswith("Бакалавр"),
                        reg.zayav_type_name.str.startswith("Магистр")],
                       ["bak", "mag"], "other")
    funding = np.where(reg.budget.fillna(False), "budget", "contract")
    channel = np.where(reg.priem_doc_type_kod == 1, "offline", "remote")
    partitions = {
        "degree": degree,
        "fund": funding,
        "channel": channel,
        "degreefund": np.char.add(np.char.add(degree, "_"), funding),
    }
    for name, labels in partitions.items():
        counts = reg.groupby([reg.date, labels]).size().unstack(fill_value=0)
        for col in counts:
            g[f"z_{name}_{col}"] = np.log1p(g.date.map(counts[col]).fillna(0.0))
    return g


def predict_target(tr: pd.DataFrame, te: pd.DataFrame, target: str,
                   features: list[str] | None = None) -> np.ndarray:
    feats = features or base.FEATURE_COLUMNS
    op = tr[tr.y > 0]
    wo, wa = base.compute_era_weights(op), base.compute_era_weights(tr)
    families = [(base.make_lgbm, 5), (base.make_catboost, 3), (base.make_histgb, 2)]
    pred = []
    for make, seeds in families:
        ps = []
        for seed in range(seeds):
            model = make(seed)
            if model.__class__.__name__ == "CatBoostRegressor":
                model.set_params(thread_count=1, allow_writing_files=False)
            if model.__class__.__name__ == "LGBMRegressor":
                model.set_params(n_jobs=1)
            ps.append(model.fit(op[feats], op[target], sample_weight=wo).predict(te[feats]))
        pred.append(np.mean(ps, axis=0))
    allpred = []
    for seed in range(3):
        model = base.make_catboost(seed).set_params(thread_count=1, allow_writing_files=False)
        allpred.append(model.fit(tr[feats], tr[target], sample_weight=wa).predict(te[feats]))
    return np.maximum(0, .75 * np.mean(pred, axis=0) + .25 * np.mean(allpred, axis=0))


def generate(year: int, partitions: list[str]) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"predictions_{year}.csv"
    g = build_grid(year)
    tr, te = g[g.year < year], g[g.year == year].reset_index(drop=True)
    # Save truth solely for scoring; no downstream model is passed this column.
    out = pd.DataFrame({"date": te.date.dt.strftime("%Y-%m-%d"), "year": year,
                        "y": te.y, "dow": te.dow, "month": te.month,
                        "day_of_camp": te.day_of_camp})
    closed = base.determine_closed_days(tr, te)
    out["closed"] = te.date.dt.strftime("%Y-%m-%d").isin(closed) | (te.dow == 6)
    if path.exists():
        cached = pd.read_csv(path)
        if not cached.date.equals(out.date):
            raise ValueError("Prediction cache has different dates")
        for c in cached:
            if c not in out:
                out[c] = cached[c]
    targets = ["z", "z_Бюджет", "z_Договор"]
    targets += [c for c in g if any(c.startswith(f"z_{p}_") for p in partitions)]
    for target in targets:
        if target in out:
            continue
        start = time.monotonic()
        with threadpool_limits(limits=1):
            out[target] = predict_target(tr, te.drop(columns=["y", "z", "epgu"]), target)
        out.to_csv(path, index=False)
        print(f"{year} {target}: {time.monotonic()-start:.1f}s", flush=True)
    return out


def predictions(d: pd.DataFrame) -> dict[str, np.ndarray]:
    pool = d.z.to_numpy()
    stream = np.log1p(np.expm1(d["z_Бюджет"]) + np.expm1(d["z_Договор"]))
    preds = {"pool": pool, "legacy_streams": stream,
             "v59": -1.2 * pool + 2.2 * stream}
    for part in ("degree", "fund", "channel", "degreefund"):
        cols = [c for c in d if c.startswith(f"z_{part}_")]
        if not cols:
            continue
        summed = np.log1p(np.expm1(d[cols]).sum(axis=1)).to_numpy()
        preds[part] = summed
        preds[part + "_extrap"] = -1.2 * pool + 2.2 * summed
    return {k: np.where(d.closed, 0, np.maximum(0, v)) for k, v in preds.items()}


def evaluate() -> None:
    rows = []
    for year in YEARS[:-1]:
        path = CACHE / f"predictions_{year}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        truth = np.log1p(d.y.to_numpy())
        for name, p in predictions(d).items():
            rows.append({"year": year, "model": name,
                         "rmsle": float(np.sqrt(np.mean((p-truth)**2))),
                         "weekday_rmsle": float(np.sqrt(np.mean((p[d.dow<5]-truth[d.dow<5])**2)))})
    scores = pd.DataFrame(rows)
    scores.to_csv(CACHE / "scores.csv", index=False)
    print(scores.pivot(index="model", columns="year", values="rmsle").round(4).to_string())
    print("Means", scores.groupby("model").rmsle.mean().sort_values().round(4).to_string())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    p.add_argument("--partitions", nargs="*", default=["degree", "fund", "channel"])
    p.add_argument("--evaluate-only", action="store_true")
    a = p.parse_args()
    if not a.evaluate_only:
        for year in a.years:
            generate(year, a.partitions)
    evaluate()


if __name__ == "__main__":
    main()
