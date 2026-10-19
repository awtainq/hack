#!/usr/bin/env python3
"""V62: replace the hard closure gate using a measured quiet-day loss curve.

Three historical submissions differ ONLY in their common quiet-day floor.
Their recorded public scores determine the quadratic squared-RMSLE effect of
that floor without assuming that all 73 dates are in the public evaluation.
This relies on those historical scores and the current scores belonging to the
same evaluation regime. The scores come from SUBMISSIONS.md, not a new user
measurement. The new V62 score remains unknown until it is submitted.

    .venv/bin/python reproducers/v62_soft_closure.py
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducers.v61_orthogonal_funding import read_forecast, measured_score

BASE = ROOT / "forecast_2024_v61_orthogonal_funding.csv"
HISTORY = [
    (ROOT / "forecast_2024_v7_3seed.csv", .55, .58862),
    (ROOT / "forecast_2024_v10_floor100.csv", 1., .58787),
    (ROOT / "forecast_2024_v11_floor145.csv", 1.45, .59158),
]
REPORT = ROOT / "output" / "v62"


def floor_curve(levels: np.ndarray, scores: np.ndarray) -> dict:
    levels, scores = np.asarray(levels, float), np.asarray(scores, float)
    if (levels.shape != (3,) or scores.shape != (3,) or len(np.unique(levels)) != 3
            or not np.isfinite(levels).all() or not np.isfinite(scores).all()
            or (levels < 0).any() or (scores < 0).any()):
        raise ValueError("Three distinct nonnegative floors and scores are required")
    x = np.log1p(levels)
    a, b, c = np.polyfit(x, scores**2, 2)
    if a <= 0 or b >= 0:
        raise ValueError("Historical scores do not support a positive quiet-day floor")
    opt_log = -b/(2*a)
    return {"quadratic": float(a), "linear": float(b), "intercept": float(c),
            "optimal_log_floor": float(opt_log), "optimal_count_floor": float(np.expm1(opt_log))}


def check_only_floor_changed(frames: list[pd.DataFrame], levels: list[float],
                             quiet: np.ndarray) -> None:
    if len(frames) != len(levels) or not np.any(quiet):
        raise ValueError("Invalid quiet-day comparison")
    first = frames[0]
    for d, level in zip(frames, levels):
        if not d.date.equals(first.date):
            raise ValueError("Historical dates are not aligned")
        if not np.array_equal(d.forecast.to_numpy()[~quiet], first.forecast.to_numpy()[~quiet]):
            raise ValueError("Historical models changed on non-quiet days; calibration is invalid")
        if not np.all(d.forecast.to_numpy()[quiet] == level):
            raise ValueError("Historical quiet-day predictions do not equal the declared floor")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--floor", type=float, help="Override fitted floor; default fitted optimum rounded to .01")
    p.add_argument("--out", type=Path, default=ROOT / "forecast_2024_v62_soft_closure.csv")
    args = p.parse_args()
    base = read_forecast(BASE)
    base_score = measured_score(BASE)
    quiet = (base.forecast == 0).to_numpy()
    frames = [read_forecast(path) for path, _, _ in HISTORY]
    levels = [level for _, level, _ in HISTORY]
    scores = [score for _, _, score in HISTORY]
    check_only_floor_changed(frames, levels, quiet)
    curve = floor_curve(levels, scores)
    level = float(np.round(curve["optimal_count_floor"], 2)) if args.floor is None else args.floor
    if not np.isfinite(level) or level < 0:
        p.error("--floor must be finite and nonnegative")
    result = base.copy()
    result.loc[quiet, "forecast"] = level
    if not np.array_equal(result.forecast[~quiet], base.forecast[~quiet]):
        raise ValueError("Non-quiet forecasts changed")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    x = np.log1p(level)
    delta = curve["quadratic"]*x*x+curve["linear"]*x
    estimated_score = float(np.sqrt(max(0, base_score**2+delta)))
    rounded_score_estimates = []
    for signs in itertools.product([-1, 1], repeat=3):
        alt = floor_curve(levels, np.asarray(scores)+np.asarray(signs)*.000005)
        rounded_score_estimates.append(float(np.sqrt(max(0, base_score**2+alt["quadratic"]*x*x+alt["linear"]*x))))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {
        "model": "V62", "base": BASE.name, "base_public_rmsle": base_score,
        "floor": level, "changed_days": int(quiet.sum()), "curve": curve,
        "estimated_public_rmsle_under_assumptions": estimated_score,
        "historical_score_rounding_sensitivity_only": [min(rounded_score_estimates), max(rounded_score_estimates)],
        "measured_v62_public_rmsle": None,
        "history_scores": [{"file": path.name, "floor": f, "score": s,
                            "source": "Existing SUBMISSIONS.md, not independently reconfirmed",
                            "sha256": sha(path)} for path, f, s in HISTORY],
        "input_base_sha256": sha(BASE), "output_sha256": sha(args.out),
        "code_sha256": sha(Path(__file__).resolve()),
        "assumptions": ["Historical scores in SUBMISSIONS.md refer to the unchanged floor-only CSV files",
                        "Those historical scores use the same public test set and scoring regime as the current user scores",
                        "Squared RMSLE is additive over evaluated dates"],
        "source": "Public-score calibration; not independent validation",
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "calibration.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2)+"\n")
    pd.DataFrame({"date": result.date[quiet], "previous": base.forecast[quiet],
                  "forecast": result.forecast[quiet]}).to_csv(REPORT / "changed_days.csv", index=False)
    print(f"Saved {args.out}: floor={level}, changed={quiet.sum()}, sum={result.forecast.sum():.2f}")
    print(f"Estimated Public RMSLE under the stated assumptions: {estimated_score:.6f}")
    print("Measured V62 Public score: unknown")


if __name__ == "__main__":
    main()
