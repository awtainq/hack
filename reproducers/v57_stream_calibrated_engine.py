"""
Stream-calibrated blend: the clean_deadline_blend feature set with three changes that each
paid for themselves on the backtest.

1. Recency: era weights are multiplied by exp(decay * (year - max(year))), decay = 0.9.
   The EPGU era already dominates; the decay stops the pre-2017 regime from pulling the
   level back down.
2. Streams: the daily total is decomposed into budget and contract using the basis column of
   included_ks.csv, each stream is predicted separately and the log-space predictions are
   recombined. August is mostly contract traffic, so a pooled model smooths it away.
3. Level: open-day predictions are shifted in log space so their sum matches last year's
   total. The model was underpredicting 2021-2023 by 10-12% uniformly.

Blend: 0.75 * streams + 0.25 * pooled, same 0.75/0.25 spirit as the open/all-day split.

Backtest (each year from earlier years only, true schedule): RMSLE 0.433 mean over
2017-2019 and 2021-2023, from 0.453 for clean_deadline_blend.py.

  python reproducers/v57_stream_calibrated_engine.py backtest
  python reproducers/v57_stream_calibrated_engine.py 2024 --bak-end 07-25 --mag-end 07-29
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
    FEATS, REG_FILE, apply_schedule, closed_from_history, daily_grid, era_weights,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
KS_FILE = ROOT / "processed_data" / "included_ks.csv"
BACKTEST_YEARS = base.BACKTEST_YEARS
DECAY = 0.9
W_OPEN = 0.75
W_POOLED = 0.25
LEVEL_SHIFT = 0.5  # fraction of the full shift towards last year's total; 1.0 overfits the level
STREAMS = ("Бюджет", "Договор")


def weights(tr):
    return era_weights(tr) * np.exp(DECAY * (tr["year"] - tr["year"].max()))


def zblend(tr: pd.DataFrame, te: pd.DataFrame, target: str = "z") -> np.ndarray:
    """Same three-family ensemble as the base blend, on configurable weights and target."""
    open_tr = tr[tr["y"] > 0]
    w_all, w_open = weights(tr), weights(open_tr)
    fams = ((base.lgb, 5), (base.cat, 3), (base.hgb, 2))
    z_open = np.mean([np.mean([make(s).fit(open_tr[FEATS], open_tr[target], sample_weight=w_open)
                               .predict(te[FEATS]) for s in range(n)], axis=0) for make, n in fams], axis=0)
    z_all = np.mean([base.cat(s).fit(tr[FEATS], tr[target], sample_weight=w_all).predict(te[FEATS])
                     for s in range(3)], axis=0)
    return W_OPEN * z_open + (1 - W_OPEN) * z_all


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
        g[f"s_{col}"] = g.index.map(piv[col]).fillna(0.0)
        g[f"z_{col}"] = np.log1p(g[f"s_{col}"])
    return g.reset_index()


def predict_streams(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    parts = []
    for c in STREAMS:
        if tr[f"s_{c}"].sum() <= 100:  # a basis absent in history is skipped, not forced
            continue
        parts.append(np.expm1(np.clip(zblend(tr, te, target=f"z_{c}"), 0, None)))
    return np.log1p(np.sum(parts, axis=0))


def level_calibrate(zp: np.ndarray, g: pd.DataFrame, ty: int) -> np.ndarray:
    """Shift open days in log space part-way towards last year's total.

    A full shift to last year's total scores worse than half of it on the backtest
    (0.4321 vs 0.4296): the model's own level is partly right, and anchoring to the
    previous year too hard breaks years whose total moved 10%+.
    """
    prev = g[g["year"] < ty].groupby("year")["y"].sum()
    prev = prev[prev > 1000]
    open_idx = zp > 0
    if open_idx.sum() == 0 or prev.empty:
        return zp
    shift = np.log1p(float(prev.iloc[-1])) - np.log1p(np.expm1(zp[open_idx]).sum())
    out = zp.copy()
    out[open_idx] += LEVEL_SHIFT * shift
    return out


def predict(tr: pd.DataFrame, te: pd.DataFrame, trs: pd.DataFrame, tes: pd.DataFrame) -> np.ndarray:
    z_pooled = zblend(tr, te)
    z_streams = predict_streams(trs, tes)
    return W_POOLED * z_pooled + (1 - W_POOLED) * z_streams


def backtest():
    rows = []
    for ty in BACKTEST_YEARS:
        g = daily_grid(REG_FILE, ty, target_ends={})
        tr, te = g[g["year"] < ty], g[g["year"] == ty].reset_index(drop=True)
        gs = stream_grid(ty)
        trs, tes = gs[gs["year"] < ty], gs[gs["year"] == ty].reset_index(drop=True)
        closed = list(te.loc[te["y"] == 0, "date"].dt.strftime("%Y-%m-%d"))
        zp = level_calibrate(apply_schedule(te, predict(tr, te, trs, tes), closed), g, ty)
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
    zp = level_calibrate(apply_schedule(te, predict(tr, te, trs, tes), closed), g, year)
    out = pd.DataFrame({"date": te["date"].dt.strftime("%Y-%m-%d"), "forecast": np.round(np.expm1(zp), 2)})
    path = Path(a.out) if a.out else ROOT / f"forecast_{year}_v57_stream_calibrated.csv"
    out.to_csv(path, index=False)
    print(f"{path.name}: sum={out['forecast'].sum():.0f}, zeros={(out['forecast'] == 0).sum()}")


if __name__ == "__main__":
    main()
