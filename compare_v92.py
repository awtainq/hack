"""Diagnostics only: compare frozen forecasts with V84 after training."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = Path("output/v92")
    frozen = json.loads((output / "selection_frozen.json").read_text())
    for name, digest in frozen["sha256"].items():
        if checksum(name) != digest:
            raise ValueError(f"Frozen artifact changed: {name}")
    reference = pd.read_csv("forecast_2024_v84.csv", parse_dates=["date"])
    frame = reference.rename(columns={"forecast": "v84"})
    summary = {
        "reference_role": "post-hoc comparison only; not ground truth or training input",
        "reference_used_for_selection": False,
        "public_rmsle": None,
        "comparisons": {},
    }
    positive = frame.v84.gt(0)
    for version in ["v90", "v91", "v92"]:
        prediction = pd.read_csv(f"forecast_2024_{version}.csv", parse_dates=["date"])
        if not prediction.date.equals(frame.date):
            raise ValueError("Mismatched dates")
        frame[version] = prediction.forecast
        difference = frame[version] - frame.v84
        frame[f"{version}_difference"] = difference
        summary["comparisons"][version] = {
            "total": float(frame[version].sum()),
            "reference_total": float(frame.v84.sum()),
            "total_percent_difference": float(100 * (frame[version].sum() / frame.v84.sum() - 1)),
            "mean_absolute_difference_per_day": float(difference.abs().mean()),
            "mean_absolute_percent_difference_positive_days": float(
                (100 * difference[positive].abs() / frame.v84[positive]).mean()
            ),
            "same_zero_dates": bool(frame[version].eq(0).equals(frame.v84.eq(0))),
        }
    for previous in ["v90", "v91"]:
        old = summary["comparisons"][previous]["mean_absolute_difference_per_day"]
        new = summary["comparisons"]["v92"]["mean_absolute_difference_per_day"]
        summary[f"mean_absolute_difference_reduction_vs_{previous}_percent"] = float(
            100 * (1 - new / old)
        )
    frame.to_csv(output / "comparison_with_v84.csv", index=False, float_format="%.6f")
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    parts = pd.read_csv(output / "degree_forecasts.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    for version, color in [("v84", "#182b49"), ("v91", "#b9bdc3"), ("v92", "#dc6a26")]:
        axes[0].plot(frame.date, frame[version], color=color, label=version.upper())
    axes[0].set_title("V92 deadline-based models: post-hoc comparison of forecasts")
    axes[0].legend()
    axes[1].stackplot(
        parts.date,
        parts.bachelor,
        parts.master,
        parts.background,
        labels=["Bachelor", "Master", "College + other"],
        colors=["#277da8", "#62b59c", "#e9bc63"],
    )
    axes[1].set_title("V92 independent degree components")
    axes[1].legend(loc="upper right")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for axis in axes:
        axis.set_ylabel("Registrations / day")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    for name, digest in frozen["sha256"].items():
        assert checksum(name) == digest
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
