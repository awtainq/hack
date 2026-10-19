#!/usr/bin/env python3
"""V60: V59 with a deterministic, degree-aware funding partition.

Training rows are matched to competition lists on (applicant, day, degree).
An applicant with any budget/state-line entry is assigned to the budget stream;
contract-only and unmatched registrations go to the contract stream. Every
registration belongs to exactly one stream. This is a modelling convention,
not an inference of the applicant's eventual funding or enrolment.

The ensemble, era weights, closure rule and t=4.4 are preserved from V59.
No measured leaderboard scores are used by this pipeline.

    .venv/bin/python reproducers/v60_consistent_streams.py
    .venv/bin/python reproducers/v60_consistent_streams.py --backtest

Requires the existing v59_standalone_4_4.py and processed_data/*.csv.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import catboost
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import v59_standalone_4_4 as base

BACKTEST_YEARS = (2017, 2018, 2019, 2021, 2022, 2023)
REPORT_DIR = ROOT / "output" / "v60"


def nominal_ends(year: int) -> dict[str, str]:
    """Predeclared validation policy: July 25 / July 29 (master's weekends back).

    Historical official schedules are not available locally. Actual validation
    intake/closure dates are never estimated from the held-out year's counts.
    The policy can differ from the real historical admission calendar.
    """
    mag = pd.Timestamp(f"{year}-07-29")
    while mag.dayofweek >= 5:
        mag -= pd.Timedelta(days=1)
    return {"bak": "07-25", "mag": mag.strftime("%m-%d")}


def funding_counts(reg: pd.DataFrame, ks: pd.DataFrame, year: int) -> tuple[pd.DataFrame, dict]:
    """Partition only past registrations; a many-to-one join preserves row counts."""
    r, k = reg.copy(), ks.copy()
    r["date"] = pd.to_datetime(r.reg_ts).dt.normalize()
    k["date"] = pd.to_datetime(k.reg_ts).dt.normalize()
    r = r[r.date.dt.year < year].copy()
    k = k[k.date.dt.year < year].copy()
    keys = ["abitur_kod", "date", "zayav_type_kod"]
    k["has_budget"] = k.ob_osnova_name.isin(["Бюджет", "Гослиния"])
    mapping = k.groupby(keys).has_budget.max().reset_index()
    merged = r.merge(mapping, on=keys, how="left", validate="many_to_one")
    if len(merged) != len(r):
        raise ValueError("The funding join changed the number of registrations")
    merged["stream"] = np.where(merged.has_budget.eq(True), "budget", "contract")
    counts = merged.groupby(["date", "stream"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["budget", "contract"], fill_value=0)
    expected = r.groupby("date").size()
    if not counts.sum(axis=1).equals(expected):
        raise ValueError("Funding streams do not sum to total registrations")
    audit = {"training_registrations": len(r), "unmatched_registrations": int(merged.has_budget.isna().sum()),
             "budget_registrations": int((merged.stream == "budget").sum()),
             "contract_registrations": int((merged.stream == "contract").sum())}
    return counts, audit


def make_grid(year: int, ends: dict[str, str], reg_file: Path = base.REG_FILE,
              ks_file: Path = base.KS_FILE, legacy: bool = False) -> tuple[pd.DataFrame, dict]:
    reg = pd.read_csv(reg_file)
    ks = pd.read_csv(ks_file)
    max_year = int(pd.to_datetime(reg.reg_ts).dt.year.max())
    if year > max_year + 1:
        raise ValueError(f"The latest observed year is {max_year}; intervening seasons are required")
    if legacy:
        g = base.create_stream_grid(reg_file, ks_file, year, target_ends=ends)
    else:
        g = base.create_daily_grid(reg_file, year, target_ends=ends)
    counts, audit = funding_counts(reg, ks, year)
    for stream in ("budget", "contract"):
        g[f"z_{stream}"] = np.log1p(g.date.map(counts[stream]).fillna(0.0))
    # make_grid retains y for scoring. fit_predict only receives historical rows
    # and the explicit feature matrix of the target year.
    return g, audit


def fit_predict(tr: pd.DataFrame, target_features: pd.DataFrame, col: str) -> np.ndarray:
    op = tr[tr.y > 0]
    wo, wa = base.compute_era_weights(op), base.compute_era_weights(tr)
    family_predictions = []
    for make, n in ((base.make_lgbm, 5), (base.make_catboost, 3), (base.make_histgb, 2)):
        seeds = []
        for s in range(n):
            m = make(s)
            if isinstance(m, lightgbm.LGBMRegressor):
                m.set_params(n_jobs=1)
            if isinstance(m, catboost.CatBoostRegressor):
                m.set_params(thread_count=1, allow_writing_files=False)
            seeds.append(m.fit(op[base.FEATURE_COLUMNS], op[col], sample_weight=wo).predict(target_features))
        family_predictions.append(np.mean(seeds, axis=0))
    all_days = []
    for s in range(3):
        m = base.make_catboost(s).set_params(thread_count=1, allow_writing_files=False)
        all_days.append(m.fit(tr[base.FEATURE_COLUMNS], tr[col], sample_weight=wa).predict(target_features))
    return np.maximum(0.0, base.W_OPEN * np.mean(family_predictions, axis=0)
                      + (1-base.W_OPEN) * np.mean(all_days, axis=0))


def components(g: pd.DataFrame, year: int, legacy: bool = False) -> pd.DataFrame:
    tr, te = g[g.year < year], g[g.year == year].reset_index(drop=True)
    if len(te) != 73:
        raise ValueError("A campaign must have 73 calendar days")
    cols = ["z", "z_budget", "z_contract"] + (["z_Бюджет", "z_Договор"] if legacy else [])
    out = te[["date", "y", "dow", "month", "day_of_camp"]].copy()
    closed = base.determine_closed_days(tr, te)
    out["closed"] = te.date.dt.strftime("%Y-%m-%d").isin(closed) | (te.dow == 6)
    for col in cols:
        with threadpool_limits(limits=1):
            out[col] = fit_predict(tr, te[base.FEATURE_COLUMNS], col)
    return out


def predict_log(d: pd.DataFrame, t: float = 4.4, legacy: bool = False) -> np.ndarray:
    budget, contract = ("z_Бюджет", "z_Договор") if legacy else ("z_budget", "z_contract")
    streams = np.log1p(np.expm1(d[budget]) + np.expm1(d[contract]))
    z = (1-t/2) * d.z.to_numpy() + t/2 * streams.to_numpy()
    return np.where(d.closed, 0.0, np.maximum(0.0, z))


def checked_forecast(d: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
    forecast = pd.DataFrame({"date": d.date.dt.strftime("%Y-%m-%d"),
                             "forecast": np.round(np.expm1(z), 2)})
    if (len(forecast) != 73 or not forecast.date.is_unique
            or not np.isfinite(forecast.forecast).all() or (forecast.forecast < 0).any()):
        raise ValueError("Invalid submission")
    return forecast


def backtest(reg_file: Path, ks_file: Path) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows, daily = [], []
    for year in BACKTEST_YEARS:
        g, _ = make_grid(year, nominal_ends(year), reg_file, ks_file, legacy=True)
        d = components(g, year, legacy=True)
        truth = np.log1p(d.y.to_numpy())
        row = {"year": year}
        for name, old in (("v59", True), ("v60", False)):
            pred = predict_log(d, legacy=old)
            row[f"{name}_rmsle"] = float(np.sqrt(np.mean((pred-truth)**2)))
            d[f"{name}_forecast"] = np.round(np.expm1(pred), 2)
        row["delta"] = row["v60_rmsle"]-row["v59_rmsle"]
        rows.append(row)
        daily.append(d.assign(year=year))
        print(json.dumps(row), flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(REPORT_DIR / "backtest_metrics.csv", index=False)
    pd.concat(daily, ignore_index=True).to_csv(REPORT_DIR / "backtest_daily.csv", index=False)
    print("Mean seasonal RMSLE:", r[["v59_rmsle", "v60_rmsle"]].mean().to_dict())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--t", type=float, default=4.4)
    p.add_argument("--bak-end", default="07-25")
    p.add_argument("--mag-end", default="07-29")
    p.add_argument("--reg", type=Path, default=base.REG_FILE)
    p.add_argument("--ks", type=Path, default=base.KS_FILE)
    p.add_argument("--out", type=Path)
    p.add_argument("--backtest", action="store_true")
    a = p.parse_args()
    if a.backtest:
        backtest(a.reg, a.ks)
        return
    if not np.isfinite(a.t):
        p.error("--t must be finite")
    ends = {"bak": a.bak_end, "mag": a.mag_end}
    g, audit = make_grid(a.year, ends, a.reg, a.ks)
    d = components(g, a.year)
    result = checked_forecast(d, predict_log(d, a.t))
    out = a.out or ROOT / f"forecast_{a.year}_v60_consistent_streams.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    d.drop(columns="y").to_csv(REPORT_DIR / f"components_{a.year}.csv", index=False)
    meta = {"model": "V60", "year": a.year, "t": a.t, "ends": ends, "audit": audit,
            "rmsle_2024": None, "status": "Unscored on the 2024 leaderboard",
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__, "catboost": catboost.__version__,
            "files_sha256": {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): sha256(path)
                              for path in (a.reg.resolve(), a.ks.resolve(), Path(__file__).resolve(), base.FILE_PATH)}}
    (REPORT_DIR / f"metadata_{a.year}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2)+"\n")
    print(f"Saved {out}: {len(result)} rows, sum={result.forecast.sum():.2f}, zeros={(result.forecast==0).sum()}")


if __name__ == "__main__":
    main()
