"""Learn within-period traffic shapes separately from campaign volume.

Counts are centred within historical year/period for fitting. Predictions for
an unseen year use only its calendar and a fixed prior forecast's period level.
No held-out labels or leaderboard scores enter fitting or local selection.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducers import v60_research as data
import v59_standalone_4_4 as base

OUT = ROOT / "output" / "v63_shape"


def phases(g, mode):
    if mode == "month":
        return g.month.to_numpy()
    return np.select([g.day_of_camp <= 10, g.day_of_camp <= 24,
                       g.dist_end_bak <= 0, g.dist_end_mag <= 0], [0, 1, 2, 3], 4)


def normalized_predictions(tr, te, mode="month", use_year=False, target="z"):
    tr = tr.copy();te = te.copy()
    tr["phase"] = phases(tr, mode);te["phase"] = phases(te, mode)
    levels = tr[(tr.dow < 5) & (tr.y > 0)].groupby(["year", "phase"])[target].mean()
    fallback = tr[tr.y > 0].groupby(["year", "phase"])[target].mean()
    levels = levels.combine_first(fallback)
    keys = pd.MultiIndex.from_frame(tr[["year", "phase"]])
    centre = levels.reindex(keys).to_numpy()
    if not np.isfinite(centre).all():
        raise ValueError("Missing historical normalisation level")
    tr["normalised"] = tr[target]-centre
    feats = base.FEATURE_COLUMNS + (["year"] if use_year else [])
    op = tr[tr.y > 0]
    po = []
    for make, n in ((base.make_lgbm, 5), (base.make_catboost, 3), (base.make_histgb, 2)):
        ps = []
        for seed in range(n):
            model = make(seed)
            if model.__class__.__name__ == "LGBMRegressor":
                model.set_params(n_jobs=1)
            if model.__class__.__name__ == "CatBoostRegressor":
                model.set_params(thread_count=1, allow_writing_files=False)
            ps.append(model.fit(op[feats], op.normalised, sample_weight=base.compute_era_weights(op)).predict(te[feats]))
        po.append(np.mean(ps, axis=0))
    pa = []
    for seed in range(3):
        model = base.make_catboost(seed).set_params(thread_count=1, allow_writing_files=False)
        pa.append(model.fit(tr[feats], tr.normalised, sample_weight=base.compute_era_weights(tr)).predict(te[feats]))
    return .75*np.mean(po, axis=0)+.25*np.mean(pa, axis=0)


def align_level(raw, anchor, te, mode):
    result = np.asarray(raw).copy()
    ph = phases(te, mode)
    for period in np.unique(ph):
        weekday = (ph == period) & (te.dow.to_numpy() < 5)
        all_period = ph == period
        result[all_period] += np.mean(anchor[weekday])-np.mean(raw[weekday])
    return np.maximum(0, result)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows=[]
    for year in data.YEARS:
        path = OUT / f"predictions_{year}.csv"
        source = pd.read_csv(data.CACHE / f"predictions_{year}.csv")
        legacy = np.log1p(np.expm1(source["z_Бюджет"])+np.expm1(source["z_Договор"]))
        anchor = np.maximum(0, -1.2*source.z.to_numpy()+2.2*legacy.to_numpy())
        if path.exists():
            result = pd.read_csv(path)
        else:
            g = data.build_grid(year)
            tr, te = g[g.year < year], g[g.year == year].reset_index(drop=True)
            result = source[["date", "year", "y", "dow", "month", "day_of_camp", "closed"]].copy()
            for mode, use_year in (("month", False), ("month", True), ("phase", False), ("phase", True)):
                name=f"{mode}_{int(use_year)}"
                with threadpool_limits(limits=1):
                    raw=normalized_predictions(tr, te, mode, use_year)
                result[name+"_raw"] = raw
                result[name] = np.where(source.closed, 0, align_level(raw, anchor, te, mode))
                print(year,name,flush=True)
            result.to_csv(path,index=False)
        if year==2024:
            continue
        truth=np.log1p(result.y.to_numpy())
        for name in ["month_0","month_1","phase_0","phase_1"]:
            for weight in [0,.25,.5,.75,1.]:
                pred=np.where(source.closed,0, (1-weight)*anchor+weight*result[name].to_numpy())
                rows.append({"year":year,"model":name,"weight":weight,
                             "rmsle":float(np.sqrt(np.mean((truth-pred)**2))),
                             "weekday_rmsle":float(np.sqrt(np.mean((truth[source.dow<5]-pred[source.dow<5])**2)))})
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT / "metrics.csv",index=False)
    table=metrics.pivot(index=["model","weight"],columns="year",values="rmsle")
    table["selection_mean_2017_2022"]=table.drop(columns=2023).mean(axis=1)
    print(table.sort_values("selection_mean_2017_2022").round(4).to_string())


if __name__=="__main__":
    main()
