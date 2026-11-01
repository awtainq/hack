"""Render the saved experiment diagnostics; this script does not train models."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def main():
    output = Path("output/v93")
    current = pd.read_csv(output / "reference_comparison.csv", parse_dates=["date"])
    previous = pd.read_csv("forecast_2024_v91.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    axes[0].plot(current.date, current.reference, color="#182b49", label="V84 selection reference")
    axes[0].plot(previous.date, previous.forecast, color="#b9bdc3", label="V91")
    axes[0].plot(
        current.date, current.forecast, color="#dc6a26", label="V93, reference-selected settings"
    )
    axes[0].set_title("Agreement with the selection reference, not measured 2024 accuracy")
    axes[0].legend(fontsize=9)
    axes[1].stackplot(
        current.date,
        current.bachelor,
        current.master,
        current.background,
        labels=["Bachelor", "Master", "College + other"],
        colors=["#277da8", "#62b59c", "#e9bc63"],
    )
    axes[1].set_title("V93 components fitted to historical registrations")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for axis in axes:
        axis.set_ylabel("Registrations / day")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.2)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
