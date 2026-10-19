"""Alternative calendar representations evaluated with forward campaign splits."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducers import v60_research as study
import v59_standalone_4_4 as base


def calendar_features(g: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    g = g.copy()
    date = g.date.values.astype("datetime64[D]")
    start = pd.to_datetime(g.year.astype(str) + "-06-20").values.astype("datetime64[D]")
    g["campaign_workday"] = np.busday_count(start, date, weekmask="1111110")
    g["campaign_start_dow"] = pd.to_datetime(start).dayofweek
    for st in ("bak", "mag"):
        end = (g.date - pd.to_timedelta(g[f"dist_end_{st}"], unit="D")).values.astype("datetime64[D]")
        g[f"workdays_end_{st}"] = np.busday_count(end, date, weekmask="1111110")
        g[f"end_dow_{st}"] = pd.to_datetime(end).dayofweek
        for bw in (2, 5):
            g[f"end_rbf_{st}_{bw}"] = np.exp(-.5*(g[f"dist_end_{st}"]/bw)**2)
    g["first_campaign_week"] = np.minimum(g.day_of_camp, 7)
    feats = ["dow", "day_of_camp", "dist_end_bak", "dist_end_mag",
             "campaign_workday", "campaign_start_dow", "first_campaign_week",
             "workdays_end_bak", "workdays_end_mag", "end_dow_bak", "end_dow_mag",
             "end_rbf_bak_2", "end_rbf_bak_5", "end_rbf_mag_2", "end_rbf_mag_5"]
    return g, feats


def rich_cat(tr, te, col, feats):
    op = tr[tr.y > 0]
    ps = []
    for rows in (op, tr):
        weights = base.compute_era_weights(rows)
        ps.append(np.mean([
            CatBoostRegressor(iterations=350, learning_rate=.035, depth=4,
                              l2_leaf_reg=12, random_seed=42+23*s, thread_count=1,
                              verbose=False, allow_writing_files=False)
            .fit(rows[feats], rows[col], sample_weight=weights).predict(te[feats])
            for s in range(3)], axis=0))
    return np.maximum(0, .75*ps[0]+.25*ps[1])


def main():
    for year in study.YEARS:
        path = study.CACHE / f"calendar_{year}.csv"
        if path.exists():
            continue
        g, feats = calendar_features(study.build_grid(year))
        tr, te = g[g.year < year], g[g.year == year].reset_index(drop=True)
        out = pd.DataFrame({"date": te.date.dt.strftime("%Y-%m-%d")})
        for col in ("z", "z_Бюджет", "z_Договор"):
            for name, predict in (("calendar", study.predict_target), ("richcat", rich_cat)):
                start = time.monotonic()
                with threadpool_limits(limits=1):
                    out[f"{name}_{col}"] = predict(tr, te.drop(columns=["y","z","epgu"]), col, feats)
                print(year, name, col, round(time.monotonic()-start,1), flush=True)
        out.to_csv(path, index=False)


if __name__ == "__main__":
    main()
