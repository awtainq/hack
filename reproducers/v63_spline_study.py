"""Forward validation of smooth calendar models with campaign-level nuisance effects.

No leaderboard scores are read. Predictions are made for complete unseen seasons.
"""
from __future__ import annotations

import itertools
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducers import v60_research as data
import v59_standalone_4_4 as base

OUT = ROOT / "output" / "v63_study"
CONFIGS = list(itertools.product((.1, 1., 10.), (5., 30.), (.5, None)))


def basis(values, knots, degree=3):
    return SplineTransformer(knots=np.asarray(knots, float)[:, None], degree=degree,
                             include_bias=False, extrapolation="linear").fit_transform(
                                 np.asarray(values, float)[:, None])


def matrix(tr, te, curve_penalty, year_penalty):
    d = pd.concat([tr, te], ignore_index=True)
    doc = basis(d.day_of_camp, [0, 5, 10, 17, 24, 30, 35, 38, 45, 55, 65, 72])
    bak = basis(d.dist_end_bak, [-60, -30, -15, -7, -3, -1, 0, 1, 3, 7, 15, 30, 60])
    mag = basis(d.dist_end_mag, [-60, -30, -15, -7, -3, -1, 0, 1, 3, 7, 15, 30, 60])
    slow = basis(d.day_of_camp, [0, 20, 40, 72], degree=1)
    blocks = [np.ones((len(d), 1)), doc, bak, mag]
    penalties = [np.array([1e-6]), np.full(doc.shape[1], curve_penalty),
                 np.full(bak.shape[1], curve_penalty*2), np.full(mag.shape[1], curve_penalty*2)]
    for dow in (0, 2, 3, 4, 5, 6):
        blocks.append(slow*(d.dow.to_numpy() == dow)[:, None])
        penalties.append(np.full(slow.shape[1], curve_penalty*2))
    # Year-by-period random effects absorb unusually high/low years during
    # curve fitting. The unseen year's prediction uses their era-weighted mean.
    phases = np.column_stack([d.dist_end_bak <= 0,
                               (d.dist_end_bak > 0) & (d.month == 7), d.month == 8]).astype(float)
    years = sorted(tr.year.unique())
    stats = tr.groupby("year").apply(lambda f: pd.Series({"weight": base.compute_era_weights(tr)[tr.year == f.name].mean()}), include_groups=False)
    yr_weights = stats.weight.to_numpy();yr_weights /= yr_weights.sum()
    for year, weight in zip(years, yr_weights):
        indicator = (d.year == year).to_numpy(float)
        indicator[len(tr):] = weight
        blocks.append(phases*indicator[:, None])
        penalties.append(np.full(3, year_penalty))
    X = np.column_stack(blocks)
    return X[:len(tr)], X[len(tr):], np.concatenate(penalties)


def predict(tr, te, target, curve_penalty, year_penalty, huber):
    tr = tr[tr.y > 0].reset_index(drop=True)
    X, T, penalty = matrix(tr, te, curve_penalty, year_penalty)
    weights = base.compute_era_weights(tr)
    target = tr[target].to_numpy()
    current = weights.copy()
    beta = None
    for _ in range(8 if huber else 1):
        beta = np.linalg.solve(X.T@(current[:, None]*X)+np.diag(penalty), X.T@(current*target))
        if huber:
            residual = np.abs(target-X@beta)
            current = weights*np.minimum(1., huber/np.maximum(residual, 1e-10))
    return np.maximum(0, T@beta)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for year in data.YEARS:
        path = OUT / f"predictions_{year}.csv"
        source = pd.read_csv(data.CACHE / f"predictions_{year}.csv")
        if path.exists():
            result = pd.read_csv(path)
        else:
            g = data.build_grid(year)
            tr, te = g[g.year < year], g[g.year == year].reset_index(drop=True)
            result = source[["date", "year", "y", "dow", "month", "day_of_camp", "closed"]].copy()
            for number, config in enumerate(CONFIGS):
                for target in ("z", "z_Бюджет", "z_Договор"):
                    with threadpool_limits(limits=1):
                        result[f"c{number}_{target}"] = predict(tr, te, target, *config)
            result.to_csv(path, index=False)
            print(f"{year}: fitted {len(CONFIGS)} configurations", flush=True)
        if year == 2024:
            continue
        truth = np.log1p(result.y.to_numpy())
        for number, config in enumerate(CONFIGS):
            pool = result[f"c{number}_z"].to_numpy()
            streams = np.log1p(np.expm1(result[f"c{number}_z_Бюджет"])+np.expm1(result[f"c{number}_z_Договор"]))
            for name, z in (("pool", pool), ("streams", streams), ("extrap", -1.2*pool+2.2*streams)):
                pred = np.where(result.closed, 0, np.maximum(0, z))
                err = pred-truth
                rows.append({"year": year, "config": number, "model": name,
                             "curve_penalty": config[0], "year_penalty": config[1], "huber": config[2],
                             "rmsle": float(np.sqrt(np.mean(err**2))),
                             "weekday_rmsle": float(np.sqrt(np.mean(err[result.dow<5]**2)))})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    table = metrics.pivot(index=["config", "model"], columns="year", values="rmsle")
    table["selection_mean_2017_2022"] = table.drop(columns=2023).mean(axis=1)
    print(table.sort_values("selection_mean_2017_2022").head(15).round(4).to_string())


if __name__ == "__main__":
    main()
