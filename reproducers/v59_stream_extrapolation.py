"""
Extrapolation along the stream direction, with t as an explicit knob.

V58 is blend + 0.5*(streams - pooled). Moving t steps along that same direction:
    forecast(t) = (1 - t/2)*pooled + (t/2)*streams
t = 0 -> blend, t = 1 -> V58, t = 2 -> pure streams.

For a fixed direction the RMSLE as a function of t is an exact parabola
    f(t) = S1^2 - 2t<c> + t^2<D>
where <c> = mean((z-u1)*delta) and <D> = mean(delta^2). Two measured scores on the same test
set therefore determine f(t) everywhere -- no fitting involved. The catch is that the two
scores must come from the same test set, and S1 must be exact.

  python reproducers/v59_stream_extrapolation.py 2024 --bak-end 07-25 --mag-end 07-29 --t 2.0
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
    REG_FILE, apply_schedule, closed_from_history, daily_grid,
)
from reproducers.v58_half_streams import predict, stream_grid

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]


def forecast_at(t: float, year: int, ends: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """t=0 blend, t=1 V58, t=2 pure streams; w_pooled = 1 - t/2."""
    g = daily_grid(REG_FILE, year, target_ends=ends)
    tr, te = g[g["year"] < year], g[g["year"] == year].reset_index(drop=True)
    gs = stream_grid(year, target_ends=ends)
    trs, tes = gs[gs["year"] < year], gs[gs["year"] == year].reset_index(drop=True)
    closed = closed_from_history(tr, te)

    z_pool = base.predict_open_log(tr, te)
    # predict() returns 0.5*pooled + 0.5*streams, so recover the stream component from it
    z_streams = (predict(tr, te, trs, tes) - 0.5 * z_pool) / 0.5
    w_pool = 1.0 - t / 2.0
    z = w_pool * z_pool + (1.0 - w_pool) * z_streams
    zp = apply_schedule(te, z, closed)
    out = pd.DataFrame({"date": te["date"].dt.strftime("%Y-%m-%d"), "forecast": np.round(np.expm1(zp), 2)})
    return out, closed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--bak-end", default="07-25")
    ap.add_argument("--mag-end", required=True)
    ap.add_argument("--t", type=float, default=2.0, help="steps along the stream direction")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out, closed = forecast_at(a.t, a.year, {"bak": a.bak_end, "mag": a.mag_end})
    print("closed besides Sundays:", ", ".join(closed) or "none")
    path = Path(a.out) if a.out else ROOT / f"forecast_{a.year}_v59_t{a.t:g}.csv"
    out.to_csv(path, index=False)
    print(f"{path.name}: t={a.t}, sum={out['forecast'].sum():.0f}, zeros={(out['forecast'] == 0).sum()}")


if __name__ == "__main__":
    main()
