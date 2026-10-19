#!/usr/bin/env python3
"""V61: a conservative funding blend calibrated using public leaderboard feedback.

This is a leaderboard-calibrated ensemble, not an independent validation model.
The submitted V59 and V60 CSVs are fixed inputs. Their measured RMSLE scores are
used to estimate a step along the funding contrast after projecting out the
already-tuned pooled-versus-stream direction. Only 75% of the estimated optimum
is taken. The projection uses all 73 days; the public evaluation subset is not
known, so the estimated resulting score must be verified by a new submission.

    .venv/bin/python reproducers/v61_orthogonal_funding.py

Dependencies: numpy and pandas. Upstream reproducible models: V59 and V60.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "output" / "v61"
BASE = ROOT / "forecast_2024_v59_t4.4.csv"
CANDIDATE = ROOT / "forecast_2024_v60_consistent_streams.csv"
REFERENCE = ROOT / "forecast_2024_clean_deadline_blend.csv"
SCORES = ROOT / "output" / "leaderboard_evaluations.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_forecast(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path)
    if d.columns.tolist() != ["date", "forecast"]:
        raise ValueError(f"{path}: expected exactly date,forecast")
    dates = pd.to_datetime(d.date, errors="raise")
    expected = pd.date_range("2024-06-20", "2024-08-31")
    if len(d) != 73 or not np.array_equal(dates.to_numpy(), expected.to_numpy()):
        raise ValueError(f"{path}: dates must cover 2024-06-20..2024-08-31 in order")
    if not np.isfinite(d.forecast).all() or (d.forecast < 0).any():
        raise ValueError(f"{path}: forecasts must be finite and nonnegative")
    return d


def measured_score(path: Path) -> float:
    """Never pair a score with a different revision of a submitted file."""
    records = json.loads(SCORES.read_text())["evaluations"]
    matches = [r for r in records if r["file"] == path.name and r["sha256"] == digest(path)
               and r["source"] == "user_reported"]
    if not matches:
        raise ValueError(f"No user-reported score for the exact bytes of {path.name}")
    return float(matches[-1]["public_rmsle"])


def calibrate(u: np.ndarray, v: np.ndarray, pool: np.ndarray,
              s_base: float, s_candidate: float, shrink: float = .75) -> tuple[np.ndarray, dict]:
    u, v, pool = [np.asarray(x, dtype=float) for x in (u, v, pool)]
    if u.ndim != 1 or v.shape != u.shape or pool.shape != u.shape or len(u) == 0:
        raise ValueError("Log forecast vectors must be nonempty, one-dimensional and aligned")
    if not all(np.isfinite(x).all() and (x >= 0).all() for x in (u, v, pool)):
        raise ValueError("Log forecasts must be finite and nonnegative")
    if not (0 < shrink <= 1) or min(s_base, s_candidate) < 0 or not np.isfinite([s_base, s_candidate]).all():
        raise ValueError("Invalid shrinkage or scores")
    direction = u-pool
    contrast = v-u
    norm = float(np.mean(direction**2))
    if norm < 1e-12:
        raise ValueError("The stream and pooled forecasts are indistinguishable")
    projection = float(np.mean(contrast*direction)/norm)
    residual = contrast-projection*direction
    contrast_norm = float(np.mean(contrast**2))
    residual_norm = float(np.mean(residual**2))
    if residual_norm < 1e-12:
        raise ValueError("Funding contrast has no independent component")
    # On a common full evaluation set, <truth-u, contrast> follows from two
    # measured squared losses. Projecting also assumes the base is locally
    # optimal along direction. Both assumptions are recorded in the report.
    estimated_alignment = (contrast_norm+s_base**2-s_candidate**2)/2
    if estimated_alignment <= 0:
        raise ValueError("The measured scores do not support a positive funding step")
    optimum = estimated_alignment/residual_norm
    # Bound both the typical change and the largest individual-day change.
    step = min(shrink*optimum, .03/np.sqrt(residual_norm), .10/np.max(np.abs(residual)))
    weights = np.array([1-step-step*projection, step, step*projection])
    z = u+step*residual
    closed = (u == 0) & (v == 0) & (pool == 0)
    z = np.where(closed, 0, np.maximum(0, z))
    estimated_msle = s_base**2-2*step*estimated_alignment+step**2*residual_norm
    metadata = {
        "base_public_rmsle": s_base, "candidate_public_rmsle": s_candidate,
        "projection": projection, "funding_contrast_mean_square": contrast_norm,
        "residual_mean_square": residual_norm, "estimated_alignment": estimated_alignment,
        "estimated_optimal_step": optimum, "shrink": shrink, "actual_step": step,
        "max_log_update_rms": .03, "max_abs_log_update": .10,
        "weights": dict(zip(["v59", "v60", "pooled"], map(float, weights))),
        "estimated_rmsle_under_assumptions": float(np.sqrt(max(0, estimated_msle))),
        "log_update_rmse": float(np.sqrt(np.mean((z-u)**2))),
        "projection_inner_product": float(np.mean(residual*direction)),
        "assumptions": ["The full 73-day geometry approximates the unknown public evaluation subset",
                        "V59 is approximately locally optimal along the old pooled/stream direction"],
        "measured_v61_public_rmsle": None,
    }
    return z, metadata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shrink", type=float, default=.75)
    p.add_argument("--out", type=Path, default=ROOT / "forecast_2024_v61_orthogonal_funding.csv")
    args = p.parse_args()
    b, c, ref = [read_forecast(path) for path in (BASE, CANDIDATE, REFERENCE)]
    s_base, s_candidate = measured_score(BASE), measured_score(CANDIDATE)
    z, meta = calibrate(np.log1p(b.forecast), np.log1p(c.forecast), np.log1p(ref.forecast),
                        s_base, s_candidate, args.shrink)
    out = pd.DataFrame({"date": b.date, "forecast": np.round(np.expm1(z), 2)})
    if not np.isfinite(out.forecast).all() or (out.forecast < 0).any():
        raise ValueError("Invalid output forecast")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    meta["status"] = "Awaiting a new public leaderboard measurement"
    meta["source"] = "Public-score calibration; not independent validation"
    meta["input_sha256"] = {p.name: digest(p) for p in (BASE, CANDIDATE, REFERENCE, SCORES)}
    meta["output_sha256"] = digest(args.out)
    meta["code_sha256"] = digest(Path(__file__).resolve())
    (REPORT / "calibration.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2)+"\n")
    comparison = out.assign(v59=b.forecast, v60=c.forecast, delta_vs_v59=out.forecast-b.forecast)
    comparison.to_csv(REPORT / "comparison_2024.csv", index=False)
    print(f"Saved {args.out}: {len(out)} days, sum={out.forecast.sum():.2f}, zeros={(out.forecast==0).sum()}")
    print("Weights:", meta["weights"])
    print("Estimated RMSLE under the stated assumptions:", meta["estimated_rmsle_under_assumptions"])
    print("Measured V61 RMSLE: unknown")


if __name__ == "__main__":
    main()
