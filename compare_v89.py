"""Compare daily predictions with V84, without treating V84 as ground truth."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v89_degree_model import DEGREES, SeparateDegreeForecaster, checksum


def main():
    output = Path("output/v89")
    model = SeparateDegreeForecaster.load(output / "model.joblib")
    reference_path = Path("forecast_2024_v84.csv")
    reference = pd.read_csv(reference_path, parse_dates=["date"])
    guided = model.predict(components=True)
    base = model.base_prediction().reset_index()
    if not reference.date.equals(guided.date) or not reference.date.equals(base.date):
        raise ValueError("Campaign dates differ")
    if checksum(reference_path) != model.reference_provenance["sha256"]:
        raise ValueError("Reference changed since calibration")
    comparison = pd.DataFrame(
        {
            "date": reference.date,
            "v84": reference.forecast,
            "historical_models_total": base[list(DEGREES)].sum(axis=1),
            "v89": guided.forecast,
            **{degree: guided[degree] for degree in DEGREES},
        }
    )
    comparison["difference_v89_minus_v84"] = comparison.v89 - comparison.v84
    positive = comparison.v84 > 0
    comparison["absolute_percent_difference"] = np.where(
        positive,
        100 * comparison.difference_v89_minus_v84.abs() / comparison.v84.where(positive),
        np.nan,
    )
    comparison.to_csv(output / "reference_comparison.csv", index=False, float_format="%.6f")
    summaries = {}
    for column in ("historical_models_total", "v89"):
        difference = comparison[column] - comparison.v84
        summaries[column] = {
            "total": float(comparison[column].sum()),
            "total_difference": float(difference.sum()),
            "mean_absolute_difference_per_day": float(difference.abs().mean()),
            "mean_absolute_percent_difference_positive_days": float(
                (100 * difference[positive].abs() / comparison.v84[positive]).mean()
            ),
            "max_absolute_difference": float(difference.abs().max()),
            "same_zero_dates": bool(comparison[column].eq(0).equals(comparison.v84.eq(0))),
        }
    summary = {
        "comparison_target": "V84 predictions, not observed registrations",
        "reference_total": float(comparison.v84.sum()),
        "zero_dates": int(comparison.v84.eq(0).sum()),
        "degree_totals_after_calibration": guided[list(DEGREES)].sum().to_dict(),
        "comparisons": summaries,
    }
    (output / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    metadata = {
        "model": "V89: independent historical degree models + V84-guided calibration",
        "year": model.year,
        "training_years": model.training_years,
        "targets": {
            "bachelor": "bachelor registrations",
            "master": "master registrations",
            "background": "college + other registrations",
        },
        "config": asdict(model.config),
        "reference_guided": True,
        "independent_of_reference": False,
        "public_rmsle": None,
        "reference_provenance": model.reference_provenance,
        "files_sha256": {
            str(path): checksum(path)
            for path in (
                Path("processed_data/registered.csv"),
                Path("external_data/ege/exam_calendar.csv"),
                Path("v89_degree_model.py"),
                Path("v88_model.py"),
                Path("forecast_2024_v89.csv"),
                reference_path,
            )
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(asdict(model.config), indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, layout="constrained")
    axes[0].plot(
        comparison.date, comparison.historical_models_total, color="0.7", label="Historical models"
    )
    axes[0].plot(comparison.date, comparison.v84, color="#182b49", label="V84 reference")
    axes[0].plot(
        comparison.date, comparison.v89, color="#dc6a26", label="V89 after reference calibration"
    )
    axes[0].set_title(
        "Daily forecasts: agreement with V84, not validation against actual admissions"
    )
    axes[0].set_ylabel("Registrations / day")
    axes[0].legend(fontsize=9)
    axes[1].stackplot(
        comparison.date,
        *[comparison[d] for d in DEGREES],
        labels=["Bachelor", "Master", "College + other"],
        colors=["#277da8", "#62b59c", "#e9bc63"]
    )
    axes[1].set_title("V89 components: independently trained streams, shared total calibration")
    axes[1].set_ylabel("Registrations / day")
    axes[1].legend(loc="upper right", fontsize=9)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_ylim(bottom=0)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
