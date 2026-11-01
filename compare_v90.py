"""Post-hoc comparison of the frozen independent model with a reference CSV.

This script writes diagnostics only. It never changes model selection, weights,
or the delivered forecast. The reference is not ground truth.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = Path("output/v90")
    frozen = json.loads((output / "selection_frozen.json").read_text())
    for path, digest in frozen["sha256"].items():
        if checksum(path) != digest:
            raise ValueError(f"Frozen artifact changed: {path}")
    reference_path = Path("forecast_2024_v84.csv")
    reference = pd.read_csv(reference_path, parse_dates=["date"])
    prediction = pd.read_csv("forecast_2024_v90.csv", parse_dates=["date"])
    if not reference.date.equals(prediction.date):
        raise ValueError("Forecast dates differ")
    comparison = pd.DataFrame(
        {"date": reference.date, "v84": reference.forecast, "v90": prediction.forecast}
    )
    comparison["difference"] = comparison.v90 - comparison.v84
    positive = comparison.v84 > 0
    comparison["absolute_percent_difference"] = np.where(
        positive, 100 * comparison.difference.abs() / comparison.v84.where(positive), np.nan
    )
    comparison.to_csv(output / "comparison_with_v84.csv", index=False, float_format="%.6f")
    summary = {
        "purpose": "Post-hoc agreement with V84 predictions, not measured accuracy",
        "reference_used_in_training_or_selection": False,
        "configuration_changed_after_comparison": False,
        "forecast_total": float(comparison.v90.sum()),
        "reference_total": float(comparison.v84.sum()),
        "total_percent_difference": float(100 * (comparison.v90.sum() / comparison.v84.sum() - 1)),
        "mean_absolute_difference_per_day": float(comparison.difference.abs().mean()),
        "mean_absolute_percent_difference_positive_days": float(
            comparison.loc[positive, "absolute_percent_difference"].mean()
        ),
        "median_absolute_percent_difference_positive_days": float(
            comparison.loc[positive, "absolute_percent_difference"].median()
        ),
        "max_absolute_difference": float(comparison.difference.abs().max()),
        "same_zero_dates": bool(comparison.v84.eq(0).equals(comparison.v90.eq(0))),
        "reference_sha256": checksum(reference_path),
        "freeze_sha256": checksum(output / "selection_frozen.json"),
        "public_rmsle": None,
    }
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    components = pd.read_csv(output / "degree_forecasts.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    axes[0].plot(comparison.date, comparison.v84, color="#182b49", label="V84 reference")
    axes[0].plot(comparison.date, comparison.v90, color="#dc6a26", label="V90 independent model")
    axes[0].set_title("Frozen V90 vs V84: agreement of forecasts, not actual admissions accuracy")
    axes[0].legend()
    axes[1].stackplot(
        components.date,
        components.bachelor,
        components.master,
        components.background,
        colors=["#277da8", "#62b59c", "#e9bc63"],
        labels=["Bachelor", "Master", "College + other"],
    )
    axes[1].set_title("V90 independently trained components")
    axes[1].legend(loc="upper right")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for axis in axes:
        axis.set_ylabel("Registrations / day")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    for path, digest in frozen["sha256"].items():
        assert checksum(path) == digest
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
