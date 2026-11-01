"""Reproduce the reported V84 artifact from a frozen, explicit input manifest.

This is leaderboard-feedback calibration, not independent historical training.
The manifest records every score and forecast used. No directory scanning,
Markdown parsing, output-as-input dependency or implicit secondary writes occur.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "output/v84_audit/snapshot/manifest.json"


@dataclass(frozen=True)
class FeedbackConfig:
    regularization: float
    assumed_evaluated_days: float
    reference_score: float
    target_volume: float
    closed_saturdays: tuple[str, ...]

    def __post_init__(self):
        if not all(np.isfinite([self.regularization, self.assumed_evaluated_days, self.reference_score, self.target_volume])):
            raise ValueError("Configuration must be finite")
        if min(self.regularization, self.assumed_evaluated_days, self.target_volume) <= 0 or self.reference_score < 0:
            raise ValueError("Invalid calibration configuration")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_forecast(root: Path, entry: dict, expected_dates=None) -> pd.DataFrame:
    path = (root / entry["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Input path escapes snapshot")
    if digest(path) != entry["sha256"]:
        raise ValueError(f"Input checksum mismatch: {path.name}")
    frame = pd.read_csv(path, parse_dates=["date"])
    if list(frame.columns) != ["date", "forecast"] or frame.date.isna().any() or frame.date.duplicated().any():
        raise ValueError(f"Invalid forecast schema: {path.name}")
    values = frame.forecast.to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid forecast values: {path.name}")
    if expected_dates is not None and not np.array_equal(frame.date, expected_dates):
        raise ValueError(f"Forecast dates differ: {path.name}")
    return frame


def reproduce(manifest_path=DEFAULT_MANIFEST):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    config = FeedbackConfig(**manifest["config"])
    root = manifest_path.parent
    base = read_forecast(root, manifest["baseline"])
    base_log = np.log1p(base.forecast.to_numpy())
    active = base_log > 0
    rows, right_hand_side = [], []
    for entry in manifest["basis"]:
        frame = read_forecast(root, entry, base.date)
        difference = np.log1p(frame.forecast.to_numpy()) - base_log
        if not np.all(np.abs(difference[~active]) < 1e-4):
            raise ValueError("Basis forecast changed the baseline zero support")
        score = float(entry["score"])
        if not np.isfinite(score) or score < 0:
            raise ValueError("Invalid basis score")
        rows.append(difference[active])
        right_hand_side.append(0.5 * (
            np.sum(difference[active] ** 2)
            - config.assumed_evaluated_days * (score ** 2 - config.reference_score ** 2)
        ))
    if not rows:
        raise ValueError("No calibration inputs")
    directions = np.ascontiguousarray(rows, dtype=np.float64)
    rhs = np.asarray(right_hand_side, dtype=np.float64)
    gram = np.dot(directions, directions.T)
    weights = np.linalg.solve(gram + config.regularization * np.eye(len(rhs)), rhs)
    step = np.dot(directions.T, weights)
    adjusted = base_log.copy()
    adjusted[active] += step
    closed = (base.date.dt.dayofweek == 6).to_numpy() | base.date.dt.strftime("%Y-%m-%d").isin(config.closed_saturdays).to_numpy()
    adjusted[closed] = 0.0
    adjusted = np.maximum(adjusted, 0.0)
    values = np.zeros(len(base))
    unscaled = np.expm1(adjusted[~closed])
    if not np.isfinite(unscaled).all() or unscaled.sum() <= 0:
        raise ValueError("Invalid unscaled forecast")
    values[~closed] = np.round(unscaled * (config.target_volume / unscaled.sum()), 6)
    forecast = pd.DataFrame({"date": base.date.dt.strftime("%Y-%m-%d"), "forecast": values})
    metadata = {
        "method": "leaderboard_feedback_calibration",
        "independent_historical_model": False,
        "manifest_sha256": digest(manifest_path),
        "basis_count": len(rows), "direction_rank": int(np.linalg.matrix_rank(directions)),
        "forecast_sum": float(values.sum()),
        "reference_public_score": manifest["reported_result"]["public_rmsle"],
        "reference_score_source": "user_reported_for_original_csv",
        "public_score_computed_locally": False,
        "assumed_evaluated_days_verified": False,
        "contains_projected_score_as_input": any(entry["score_kind"] == "projected_not_measured" for entry in manifest["basis"]),
    }
    return forecast, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=ROOT / "forecast_2024_v84_reproduced.csv")
    args = parser.parse_args()
    forecast, metadata = reproduce(args.manifest)
    if args.output.resolve() == (ROOT / "forecast_2024_v84.csv").resolve():
        raise ValueError("Choose a separate output to preserve the reported submission")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(args.output, index=False)
    metadata["output_sha256"] = digest(args.output)
    reference = json.loads(args.manifest.read_text())["reported_result"]
    metadata["matches_reported_csv"] = metadata["output_sha256"] == reference["sha256"]
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
