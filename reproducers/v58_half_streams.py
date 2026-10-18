"""
Half-stream blend: clean_deadline_blend averaged with its budget/contract decomposition.

The V57 experiment (decay 0.9 + level shift + streams) scored 0.49667 on the leaderboard
against 0.458 for the plain blend, so the recency decay and the level shift were removed.
What remains is the one piece with a mechanism behind it: August is mostly contract traffic
and a pooled model smooths it toward the July mix. The streams prediction is averaged 50/50
with the pooled one rather than replacing it, which keeps the point close to the known-good
blend (mean |log step| 0.063, level within 0.5%).

Backtest: RMSLE 0.4396 mean over 2017-2019 and 2021-2023 (plain blend 0.4531).
Treat that number with suspicion -- V57 also had a good backtest and lost on the leaderboard.

  python reproducers/v58_half_streams.py backtest
  python reproducers/v58_half_streams.py 2024 --bak-end 07-25 --mag-end 07-29
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reproducers.clean_deadline_blend as base
from reproducers.clean_deadline_blend import (
    REG_FILE, apply_schedule, closed_from_history, daily_grid, era_weights,
)
from reproducers.v57_stream_calibrated_engine import FEATS

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
KS_FILE = ROOT / "processed_data" / "included_ks.csv"
W_POOLED = 0.5
STREAMS = ("Бюджет", "Договор")


def stream_grid(ty: int, target_ends: dict | None = None) -> pd.DataFrame:
    """daily_grid plus per-basis daily counts, joined from included_ks on (abitur, day)."""
    reg = pd.read_csv(REG_FILE, usecols=["abitur_kod", "reg_ts"])
    ks = pd.read_csv(KS_FILE, usecols=["abitur_kod", "reg_ts", "ob_osnova_name"])
    reg["dt"] = pd.to_datetime(reg["reg_ts"]).dt.normalize()
    ks["dt"] = pd.to_datetime(ks["reg_ts"]).dt.normalize()
    m = reg.merge(ks[["abitur_kod", "dt", "ob_osnova_name"]].drop_duplicates(["abitur_kod", "dt"]),
                  on=["abitur_kod", "dt"], how="left")
    m["ob_osnova_name"] = m["ob_osnova_name"].fillna("Договор")  # unmatched rows are contract traffic
    g = daily_grid(REG_FILE, ty, target_ends=target_ends or {}).set_index("date")
    piv = m.pivot_table(index="dt", columns="ob_osnova_name", values="abitur_kod", aggfunc="count").fillna(0)
    for col in STREAMS:
        g[f"z_{col}"] = np.log1p(g.index.map(piv[col]).fillna(0.0))
    return g.reset_index()


def zblend(tr: pd.DataFrame, te: pd.DataFrame, target: str) -> np.ndarray:
    """Base ensemble, unchanged weights, on a chosen target column."""
    open_tr = tr[tr["y"] > 0]
    w_all, w_open = era_weights(tr), era_weights(open_tr)
    fams = ((base.lgb, 5), (base.cat, 3), (base.hgb, 2))
    z_open = np.mean([np.mean([make(s).fit(open_tr[FEATS], open_tr[target], sample_weight=w_open)
                               .predict(te[FEATS]) for s in range(n)], axis=0) for make, n in fams], axis=0)
    z_all = np.mean([base.cat(s).fit(tr[FEATS], tr[target], sample_weight=w_all).predict(te[FEATS])
                     for s in range(3)], axis=0)
    return base.W_OPEN * z_open + (1 - base.W_OPEN) * z_all


def predict(tr, te, trs, tes):
    parts = [np.expm1(np.clip(zblend(trs, tes, f"z_{c}"), 0, None))
             for c in STREAMS if trs[f"z_{c}"].sum() > 0]
    z_streams = np.log1p(np.sum(parts, axis=0))
    return W_POOLED * base.predict_open_log(tr, te) + (1 - W_POOLED) * z_streams


def backtest():
    rows = []
    for ty in base.BACKTEST_YEARS:
        g = daily_grid(REG_FILE, ty, target_ends={})
        tr, te = g[g["year"] < ty], g[g["year"] == ty].reset_index(drop=True)
        gs = stream_grid(ty)
        trs, tes = gs[gs["year"] < ty], gs[gs["year"] == ty].reset_index(drop=True)
        closed = list(te.loc[te["y"] == 0, "date"].dt.strftime("%Y-%m-%d"))
        zp = apply_schedule(te, predict(tr, te, trs, tes), closed)
        rows.append({"year": ty, "rmsle": float(np.sqrt(np.mean((te["z"] - zp) ** 2)))})
    r = pd.DataFrame(rows)
    print(r.round(4).to_string(index=False))
    print(f"mean: {r['rmsle'].mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="'backtest' or the forecast year")
    ap.add_argument("--bak-end", default="07-25", help="last day of bachelor intake (MM-DD)")
    ap.add_argument("--mag-end", default=None, help="last day of budget master's intake (MM-DD)")
    ap.add_argument("--closed", default=None, help="official closed dates, comma-separated")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.target == "backtest":
        return backtest()
    if a.mag_end is None:
        ap.error("--mag-end is required for a forecast")

    year = int(a.target)
    ends = {"bak": a.bak_end, "mag": a.mag_end}
    g = daily_grid(REG_FILE, year, target_ends=ends)
    tr, te = g[g["year"] < year], g[g["year"] == year].reset_index(drop=True)
    gs = stream_grid(year, target_ends=ends)
    trs = gs[gs["year"] < year]
    tes = gs[gs["year"] == year].reset_index(drop=True)
    closed = [d for d in a.closed.split(",") if d] if a.closed is not None else closed_from_history(tr, te)
    print("closed besides Sundays:", ", ".join(closed) or "none")
    zp = apply_schedule(te, predict(tr, te, trs, tes), closed)
    out = pd.DataFrame({"date": te["date"].dt.strftime("%Y-%m-%d"), "forecast": np.round(np.expm1(zp), 2)})
    path = Path(a.out) if a.out else ROOT / f"forecast_{year}_v58_half_streams.csv"
    out.to_csv(path, index=False)
    print(f"{path.name}: sum={out['forecast'].sum():.0f}, zeros={(out['forecast'] == 0).sum()}")


if __name__ == "__main__":
    main()
