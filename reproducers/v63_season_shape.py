#!/usr/bin/env python3
"""V63: separate seasonal volume from the within-month daily calendar shape.

The additional model learns log daily counts centred within each historical
year/month, using only past seasons. Month levels in the forecast come from
V59; V61's existing funding correction is preserved. The shape correction's
weight .5 was selected on 2017-2019, 2021-2022; 2023 is reported separately.
This script does not read leaderboard scores or estimate a new public score.

    .venv/bin/python reproducers/v63_season_shape.py
    .venv/bin/python reproducers/v63_season_shape.py --backtest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import v59_standalone_4_4 as base
from reproducers.v60_consistent_streams import fit_predict, nominal_ends
from reproducers.v61_orthogonal_funding import read_forecast
from reproducers.v63_shape_study import normalized_predictions, align_level

REPORT = ROOT / "output" / "v63"


def shape_prediction(tr, te, reference):
    with threadpool_limits(limits=1):
        raw = normalized_predictions(tr, te, mode="month", use_year=True)
    return align_level(raw, np.asarray(reference, float), te, "month")


def apply_shape(reference, anchor, shape, closed, weight=.5):
    if not np.isfinite(weight) or not (0 <= weight <= 1):
        raise ValueError("Shape weight must lie in [0, 1]")
    reference, anchor, shape = [np.asarray(a, float) for a in (reference, anchor, shape)]
    if reference.shape != anchor.shape or reference.shape != shape.shape:
        raise ValueError("Forecasts must be aligned")
    if not all(np.isfinite(a).all() for a in (reference, anchor, shape)):
        raise ValueError("Forecasts must be finite")
    update = np.where(closed, 0., shape-reference)
    return np.where(closed, 0., np.maximum(0., anchor+weight*update))


def backtest(weight):
    REPORT.mkdir(parents=True, exist_ok=True)
    rows, daily = [], []
    for year in (2017, 2018, 2019, 2021, 2022, 2023):
        g = base.create_stream_grid(base.REG_FILE, base.KS_FILE, year, nominal_ends(year))
        tr, te = g[g.year < year], g[g.year == year].reset_index(drop=True)
        with threadpool_limits(limits=1):
            pooled, budget, contract = [fit_predict(tr, te[base.FEATURE_COLUMNS], c)
                                        for c in ("z", "z_Бюджет", "z_Договор")]
        streams = np.log1p(np.expm1(budget)+np.expm1(contract))
        reference = np.maximum(0, -1.2*pooled+2.2*streams)
        shape = shape_prediction(tr, te, reference)
        closed_dates = base.determine_closed_days(tr, te)
        closed = te.date.dt.strftime("%Y-%m-%d").isin(closed_dates).to_numpy() | (te.dow == 6).to_numpy()
        reference = np.where(closed, 0., reference)
        prediction = apply_shape(reference, reference, shape, closed, weight)
        truth = np.log1p(te.y.to_numpy())
        row = {"year": year, "reference_v59_rmsle": float(np.sqrt(np.mean((reference-truth)**2))),
               "v63_shape_rmsle": float(np.sqrt(np.mean((prediction-truth)**2)))}
        row["delta"] = row["v63_shape_rmsle"]-row["reference_v59_rmsle"]
        rows.append(row)
        daily.append(te[["date", "y", "year", "dow"]].assign(
            reference_forecast=np.round(np.expm1(reference), 2),
            shape_forecast=np.round(np.expm1(prediction), 2)))
        print(json.dumps(row), flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(REPORT / "backtest_metrics.csv", index=False)
    pd.concat(daily, ignore_index=True).to_csv(REPORT / "backtest_daily.csv", index=False)
    print("Mean seasonal RMSLE", r[["reference_v59_rmsle", "v63_shape_rmsle"]].mean().to_dict())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weight", type=float, default=.5)
    p.add_argument("--out", type=Path, default=ROOT / "forecast_2024_v63_season_shape.csv")
    p.add_argument("--backtest", action="store_true")
    a = p.parse_args()
    if a.backtest:
        backtest(a.weight)
        return
    reference_path = ROOT / "forecast_2024_v59_t4.4.csv"
    anchor_path = ROOT / "forecast_2024_v61_orthogonal_funding.csv"
    ref, anchor = read_forecast(reference_path), read_forecast(anchor_path)
    reference, prior = np.log1p(ref.forecast.to_numpy()), np.log1p(anchor.forecast.to_numpy())
    g = base.create_daily_grid(base.REG_FILE, 2024, {"bak": "07-25", "mag": "07-29"})
    tr, te = g[g.year < 2024], g[g.year == 2024].reset_index(drop=True)
    shape = shape_prediction(tr, te, reference)
    closed = (ref.forecast == 0).to_numpy() & (anchor.forecast == 0).to_numpy()
    prediction = apply_shape(reference, prior, shape, closed, a.weight)
    out = pd.DataFrame({"date": ref.date, "forecast": np.round(np.expm1(prediction), 2)})
    if not np.isfinite(out.forecast).all() or (out.forecast < 0).any():
        raise ValueError("Invalid submission values")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    out.assign(v61=anchor.forecast, delta_vs_v61=out.forecast-anchor.forecast).to_csv(
        REPORT / "comparison_2024.csv", index=False)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {"model": "V63 season shape", "shape_weight": a.weight, "normalization": "year/month weekday log mean",
            "selection_years": [2017, 2018, 2019, 2021, 2022], "additional_check_year": 2023,
            "anchor": "V61 with user-reported Public score 0.42430",
            "public_rmsle_2024": None, "estimated_public_rmsle_2024": None,
            "input_sha256": {path.name: sha(path) for path in (reference_path, anchor_path, base.REG_FILE)},
            "code_sha256": {path.name: sha(path) for path in (Path(__file__).resolve(),
                                ROOT / "reproducers" / "v63_shape_study.py", base.FILE_PATH)},
            "output_sha256": sha(a.out), "log_update_rmse": float(np.sqrt(np.mean((prediction-prior)**2))),
            "status": "Awaiting public leaderboard measurement"}
    (REPORT / "metadata_2024.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2)+"\n")
    print(f"Saved {a.out}: sum={out.forecast.sum():.2f}, rows={len(out)}, zeros={(out.forecast==0).sum()}")


if __name__ == "__main__":
    main()
