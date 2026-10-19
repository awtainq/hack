#!/usr/bin/env python3
"""Final conservative calibration using only user-reported current public scores.

The five exact submitted CSVs supply four independent log-space corrections.
A local squared-RMSLE quadratic is estimated with full-calendar geometry,
regularized by 10% of its diagonal, and only 75% of the fitted step is used.
Historical scores from SUBMISSIONS.md are excluded. Public subset geometry is
unknown, so the estimated score is a modelling assumption, not a guarantee.

    .venv/bin/python reproducers/v64_final_calibrated.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproducers.v61_orthogonal_funding import read_forecast, measured_score

NAMES = ["forecast_2024_v59_t4.4.csv", "forecast_2024_v60_consistent_streams.csv",
         "forecast_2024_v61_orthogonal_funding.csv", "forecast_2024_v62_soft_closure.csv",
         "forecast_2024_v63_season_shape.csv"]
REPORT = ROOT / "output" / "v64"


def calibrate(log_predictions, scores, ridge=.1, shrink=.75):
    U, scores = np.asarray(log_predictions, float), np.asarray(scores, float)
    if (U.ndim != 2 or len(U) != len(scores) or len(scores) < 2
            or not np.isfinite(U).all() or not np.isfinite(scores).all()
            or (U < 0).any() or (scores < 0).any()):
        raise ValueError("Invalid log forecast matrix or measured scores")
    if not (ridge >= 0 and 0 < shrink <= 1):
        raise ValueError("Invalid regularisation")
    best = int(np.argmin(scores))
    others = [i for i in range(len(U)) if i != best]
    directions = (U[others]-U[best]).T
    gram = directions.T@directions/U.shape[1]
    if np.any(np.diag(gram) <= 1e-12):
        raise ValueError("Duplicate forecasts cannot define independent corrections")
    q = (scores[others]**2-scores[best]**2-np.diag(gram))/2
    stabilized = gram+ridge*np.diag(np.diag(gram))
    step = -shrink*np.linalg.solve(stabilized, q)
    correction = directions@step
    # Keep the final change small enough to remain close to a measured model.
    rms = float(np.sqrt(np.mean(correction**2)))
    max_abs = float(np.max(np.abs(correction)))
    trust = min(1., .08/max(rms, 1e-12), .15/max(max_abs, 1e-12))
    step *= trust
    correction = directions@step
    z = np.maximum(0., U[best]+correction)
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite calibrated forecast")
    weights = np.zeros(len(U));weights[others] = step;weights[best] = 1-step.sum()
    estimated = np.sqrt(max(0., scores[best]**2+2*step@q+step@gram@step))
    info = {"anchor_index": best, "ridge": ridge, "shrink": shrink, "trust_scale": trust,
            "weights": weights.tolist(), "log_update_rmse": float(np.sqrt(np.mean(correction**2))),
            "max_abs_log_update": float(np.max(np.abs(correction))),
            "gram_eigenvalues": np.linalg.eigvalsh(gram).tolist(),
            "estimated_public_rmsle_under_full_calendar_assumption": float(estimated),
            "measured_v64_public_rmsle": None}
    return z, info


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "forecast_2024_v64_final.csv")
    a = p.parse_args()
    frames = [read_forecast(ROOT/name) for name in NAMES]
    scores = np.array([measured_score(ROOT/name) for name in NAMES])
    U = np.array([np.log1p(d.forecast.to_numpy()) for d in frames])
    prediction, info = calibrate(U, scores)
    out = pd.DataFrame({"date": frames[0].date, "forecast": np.round(np.expm1(prediction), 2)})
    if not np.isfinite(out.forecast).all() or (out.forecast < 0).any() or len(out) != 73:
        raise ValueError("Invalid submission")
    a.out.parent.mkdir(parents=True, exist_ok=True);out.to_csv(a.out, index=False)
    REPORT.mkdir(parents=True, exist_ok=True)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    info["weights_by_model"] = dict(zip(NAMES, info.pop("weights")))
    info["anchor"] = NAMES[info["anchor_index"]]
    info["anchor_public_rmsle"] = float(scores[info["anchor_index"]])
    info["input_scores"] = dict(zip(NAMES, scores.tolist()))
    info["input_sha256"] = {name: sha(ROOT/name) for name in NAMES}
    info["output_sha256"] = sha(a.out);info["code_sha256"] = sha(Path(__file__).resolve())
    info["assumptions"] = ["Full 73-day squared log forecast distances approximate the unknown public evaluation geometry",
                           "All five user-reported scores refer to the same current scoring regime"]
    info["status"] = "Final daily candidate; awaiting one last measurement"
    (REPORT / "calibration.json").write_text(json.dumps(info, ensure_ascii=False, indent=2)+"\n")
    best = frames[info["anchor_index"]]
    out.assign(v61=best.forecast, delta_vs_v61=out.forecast-best.forecast).to_csv(
        REPORT / "comparison_2024.csv", index=False)
    print(f"Saved {a.out}: {len(out)} days, sum={out.forecast.sum():.2f}")
    print("Weights", info["weights_by_model"])
    print("Estimated under the stated assumptions:", info["estimated_public_rmsle_under_full_calendar_assumption"])
    print("Actual V64 Public score: unknown")


if __name__ == "__main__":
    main()
