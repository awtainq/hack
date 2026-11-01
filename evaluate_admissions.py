"""Select hyperparameters on earlier seasons; report 2023 separately.

This runner never reads previous model outputs or leaderboard scores. Every
fold fits a fresh model using only registrations before its target year.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from admissions_model import AdmissionsModel, ModelConfig, read_daily_counts, rmsle, write_forecast


SELECTION_YEARS = (2017, 2018, 2019, 2021, 2022)
DIAGNOSTIC_YEAR = 2023


def evaluate(counts: pd.Series, config: ModelConfig, year: int) -> tuple[dict, pd.DataFrame]:
    model = AdmissionsModel(config).fit(counts, year)
    prediction = model.predict()
    prediction["actual"] = counts.reindex(prediction.date).to_numpy()
    if prediction.actual.isna().any():
        raise ValueError(f"No complete ground truth for {year}.")
    # Score the same two-decimal values as the submitted CSV.
    prediction["forecast"] = prediction.forecast.round(2)
    metrics = {"year": year, "rmsle": rmsle(prediction.actual, prediction.forecast)}
    prediction["year"] = year
    return metrics, prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("processed_data/registered.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/v68"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = read_daily_counts(args.data)
    configs = [
        ModelConfig(
            architecture=architecture, depth=depth, iterations=iterations,
            half_life=half_life, exclude_pandemic=exclude_pandemic,
        )
        for architecture, depth, iterations, half_life, exclude_pandemic in itertools.product(
            ("direct", "hurdle", "hurdle_hard"), (3, 5), (400, 800), (2.0, 5.0), (False, True)
        )
    ]
    results = []
    forecasts = {}
    for config_id, config in enumerate(configs):
        for year in SELECTION_YEARS:
            metrics, forecast = evaluate(counts, config, year)
            results.append({"config_id": config_id, **asdict(config), **metrics})
            forecasts[config_id, year] = forecast
        latest = [row["rmsle"] for row in results if row["config_id"] == config_id]
        print(f"{config_id + 1}/{len(configs)} {config.architecture}: mean RMSLE {sum(latest) / len(latest):.5f}", flush=True)

    search = pd.DataFrame(results)
    search.to_csv(args.output_dir / "search_metrics.csv", index=False)
    ranking = search.groupby("config_id").rmsle.mean().sort_values()
    config_id = int(ranking.index[0])
    selected = configs[config_id]
    (args.output_dir / "selected_config.json").write_text(json.dumps(asdict(selected), indent=2) + "\n")

    diagnostic, diagnostic_forecast = evaluate(counts, selected, DIAGNOSTIC_YEAR)
    metrics = search.loc[search.config_id == config_id, ["year", "rmsle"]].copy()
    metrics["role"] = "hyperparameter_selection"
    metrics = pd.concat([metrics, pd.DataFrame([{**diagnostic, "role": "diagnostic_only"}])], ignore_index=True)
    metrics.to_csv(args.output_dir / "backtest_metrics.csv", index=False)
    daily = pd.concat([forecasts[config_id, year] for year in SELECTION_YEARS] + [diagnostic_forecast])
    daily.to_csv(args.output_dir / "backtest_daily.csv", index=False)

    fitted = AdmissionsModel(selected).fit(counts, 2024)
    fitted.save(args.output_dir / "model")
    output = Path("forecast_2024_v68_model.csv")
    write_forecast(fitted.predict(), output)
    metadata = {
        "selection_years": SELECTION_YEARS,
        "selection_rule": "minimum equal-weight mean RMSLE over selection years",
        "diagnostic_year": DIAGNOSTIC_YEAR,
        "diagnostic_caveat": "2023 was examined in earlier experiments; it is not a new blind holdout.",
        "selected_config": asdict(selected),
        "source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).with_name("admissions_model.py").read_bytes()).hexdigest(),
        "forecast_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "public_score": None,
        "external_volume_priors_used": False,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metrics.to_string(index=False))
    print(f"Selected: {asdict(selected)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
