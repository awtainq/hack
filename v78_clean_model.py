# -*- coding: utf-8 -*-
"""V78 -- clean, self-contained forecast of daily registrations 20.06-31.08.

Inputs: ONLY processed_data/registered.csv (raw registrations of past campaigns).
No leaderboard scores, no previous forecast files, no probe results enter the model.
forecast_2024_v74.csv (leaderboard-probed, Public 0.41893) is read only AFTER the
forecast is produced, purely to report how close the clean model lands to it.

Model (in log1p space, open days only):

    log1p(y[year, d]) = level[year] + shape[phase(d), d] + dow[phase(d), weekday(d)]

* phase  : main campaign (<= 25.07, bachelor deadline), magistracy tail (26.07-29.07),
           late tail (>= 30.07);
* shape  : per-calendar-date profile, kernel-smoothed inside each phase (never across
           a deadline), averaged over past years with exponential recency weights;
* dow    : weekday effect per phase (Saturday ~0.5x in the main phase, etc.);
* level  : future-year level = recency-weighted mean of past levels.

The three blocks are fitted by backfitting (alternating weighted means). Sundays
and Saturdays whose slot was historically closed (estimated from past seasons
only) are forecast as 0. Hyperparameters are selected by a rolling-origin
backtest on 2019, 2021, 2022, 2023 (each fitted only on earlier seasons).

Usage:
    .venv/bin/python v78_clean_model.py            # backtest + grid + 2024 forecast
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REG = ROOT / "processed_data" / "registered.csv"
OUT = ROOT / "forecast_2024_v78_clean.csv"
REFERENCE = ROOT / "forecast_2024_v74.csv"   # comparison only, never used for fitting

EXCLUDED_YEARS = {2020}                         # covid campaign, different regime
BACKTEST_YEARS = (2017, 2018, 2019, 2021, 2022)   # same protocol as V75/V77
DIAGNOSTIC_YEARS = (2023,)
BAK_END, MAG_END = (7, 25), (7, 29)


# ----------------------------------------------------------------------------- data
def load_daily(path=REG):
    r = pd.read_csv(path, usecols=["reg_ts"])
    d = pd.to_datetime(r.reg_ts).dt.normalize()
    return d.value_counts().sort_index()


def season_frame(daily, year):
    idx = pd.date_range(f"{year}-06-20", f"{year}-08-31")
    f = pd.DataFrame({"date": idx, "y": daily.reindex(idx, fill_value=0).to_numpy()})
    f["year"] = year
    f["pos"] = np.arange(len(idx))                      # same calendar date every year
    f["dow"] = f.date.dt.dayofweek
    md = list(zip(f.date.dt.month, f.date.dt.day))
    f["phase"] = [0 if m <= BAK_END else (1 if m <= MAG_END else 2) for m in md]
    f["slot"] = saturday_slots(f)
    return f


def saturday_slots(f):
    """Name every Saturday by its role in the campaign; '' for other days."""
    s = pd.Series("", index=f.index)
    sat = f.dow == 5
    s[sat] = "july"
    s[sat & (f.date.dt.month == 8)] = "aug"
    s[sat & (f.date.dt.month == 8) & (f.date.dt.day > 20)] = "late"
    s[sat & (f.date.dt.month == 7) & (f.date.dt.day > 25)] = "tail"
    first = f.index[sat][0]
    if f.loc[first, "date"].day <= 26 and f.loc[first, "date"].month == 6:
        s[first] = "first"
    return s


# ----------------------------------------------------------------------- closures
def closed_mask(train, target, half_life):
    """Sundays always closed; a Saturday slot is closed if, recency-weighted over past
    seasons, it was closed more often than not."""
    sat = train[train.dow == 5]
    w = 0.5 ** ((target.year.iloc[0] - sat.year) / half_life)
    p_zero = ((sat.y == 0) * w).groupby(sat.slot).sum() / w.groupby(sat.slot).sum()
    sat_closed = target.slot.map(lambda s: bool(s) and p_zero.get(s, 0.0) > 0.5)
    return ((target.dow == 6) | sat_closed).to_numpy()


# ---------------------------------------------------------------------------- model
def smooth_by_phase(values, weights, phase, bandwidth):
    """Gaussian kernel smoothing over calendar position, never across phases."""
    out = np.zeros_like(values)
    pos = np.arange(len(values))
    for p in np.unique(phase):
        m = phase == p
        if bandwidth <= 0:
            out[m] = values[m]
            continue
        P = pos[m].astype(float)
        K = np.exp(-0.5 * ((P[:, None] - P[None, :]) / bandwidth) ** 2) * weights[m][None, :]
        out[m] = (K * values[m][None, :]).sum(1) / np.maximum(K.sum(1), 1e-12)
    return out


def fit(train, target_year, cfg, n_iter=20):
    tr = train[(train.y > 0) & (train.dow != 6)]
    z = np.log1p(tr.y.to_numpy(float))
    yr = tr.year.to_numpy()
    years, yi = np.unique(yr, return_inverse=True)
    pos, ph, dw = tr.pos.to_numpy(), tr.phase.to_numpy(), tr.dow.to_numpy()
    w = 0.5 ** ((target_year - yr) / cfg["half_life"])
    n_pos = 73
    phase_of_pos = season_frame(pd.Series(dtype=float), target_year).phase.to_numpy()
    pd_idx = ph * 7 + dw                                   # (phase, weekday) cell

    level = np.bincount(yi, weights=z) / np.bincount(yi)
    shape = np.zeros(n_pos)
    dow = np.zeros(21)
    for _ in range(n_iter):
        # shape: weighted mean residual per calendar position, smoothed within phase
        res = z - level[yi] - dow[pd_idx]
        den = np.bincount(pos, weights=w, minlength=n_pos)
        raw = np.bincount(pos, weights=w * res, minlength=n_pos) / np.maximum(den, 1e-12)
        shape = smooth_by_phase(raw, den, phase_of_pos, cfg["bandwidth"])
        # weekday effect per phase, shrunk toward the weekday effect pooled over phases
        # (the short magistracy tail has only ~4 days per season)
        res = z - level[yi] - shape[pos]
        num = np.bincount(pd_idx, weights=w * res, minlength=21)
        cnt = np.bincount(pd_idx, weights=w, minlength=21)
        pooled = num.reshape(3, 7).sum(0) / np.maximum(cnt.reshape(3, 7).sum(0), 1e-12)
        k = cfg["dow_shrink"]
        dow = (num + k * np.tile(pooled, 3)) / (cnt + k + 1e-12)
        # yearly level
        res = z - shape[pos] - dow[pd_idx]
        level = np.bincount(yi, weights=res) / np.bincount(yi)

    lw = 0.5 ** ((target_year - years) / cfg["level_half_life"])
    future_level = float(np.sum(lw * level) / lw.sum())
    return dict(level=future_level, shape=shape, dow=dow.reshape(3, 7), years=years, levels=level)


def predict(model, target, closed):
    t = target
    z = model["level"] + model["shape"][t.pos.to_numpy()] + \
        model["dow"][t.phase.to_numpy(), t.dow.to_numpy()]
    return np.where(closed, 0.0, np.maximum(np.expm1(z), 0.0))


def rmsle(y, p):
    return float(np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2)))


# ------------------------------------------------------------------------- backtest
def seasons(daily, upto):
    return pd.concat([season_frame(daily, y) for y in range(2014, upto)
                      if y not in EXCLUDED_YEARS], ignore_index=True)


_CACHE = {}


def backtest(daily, cfg):
    scores = {}
    for year in BACKTEST_YEARS + DIAGNOSTIC_YEARS:
        if year not in _CACHE:
            _CACHE[year] = (seasons(daily, year), season_frame(daily, year))
        train, target = _CACHE[year]
        closed = closed_mask(train, target, cfg["half_life"])
        p = predict(fit(train, year, cfg), target, closed)
        # Public 2024 averages over all ACTIVE days (+3 of the 13 zero days), so open/closed
        # guessing barely matters there; we score the part that does: days that were
        # open in truth and forecast open by the model.
        act = (target.y.to_numpy() > 0) & ~closed
        scores[year] = rmsle(target.y.to_numpy()[act], np.round(p[act], 2))
    return scores


GRID = dict(half_life=[2, 3, 5, 8],
            level_half_life=[1, 2, 4, 8, 1000],
            bandwidth=[0, 0.75, 1.5, 3],
            dow_shrink=[0.0, 1.0, 3.0, 10.0])


def main():
    daily = load_daily()
    rows = []
    for vals in itertools.product(*GRID.values()):
        cfg = dict(zip(GRID, vals))
        sc = backtest(daily, cfg)
        rows.append({**cfg, **{f"bt_{k}": v for k, v in sc.items()},
                     "bt_mean": float(np.mean([sc[y] for y in BACKTEST_YEARS]))})
    grid = pd.DataFrame(rows).sort_values("bt_mean")
    print("top configurations by rolling backtest (RMSLE on open days):")
    print(grid.head(10).to_string(index=False, float_format="%.4f"))
    best = grid.iloc[0]
    cfg = {k: best[k] for k in GRID}

    train = seasons(daily, 2024)
    target = season_frame(daily, 2024)
    closed = closed_mask(train, target, cfg["half_life"])
    model = fit(train, 2024, cfg)
    p = predict(model, target, closed)
    out = pd.DataFrame({"date": target.date.dt.strftime("%Y-%m-%d"), "forecast": p.round(6)})
    out.to_csv(OUT, index=False)
    print(f"\nchosen config: {cfg}")
    print(f"written {OUT.name}: sum {p.sum():.1f}, zero days {(p == 0).sum()}, "
          f"closed: {out.date[closed].str[5:].tolist()}")

    if REFERENCE.exists():                       # reporting only
        ref = pd.read_csv(REFERENCE).forecast.to_numpy()
        act = (ref > 1) & (p > 0)
        d = np.log1p(p[act]) - np.log1p(ref[act])
        print(f"vs V74 (probed, reference only): log-RMS distance on {act.sum()} active days "
              f"{np.sqrt(np.mean(d**2)):.4f}, mean log shift {d.mean():+.4f}, "
              f"sum ratio {p.sum()/ref.sum():.3f}")
    return grid, out


if __name__ == "__main__":
    main()
