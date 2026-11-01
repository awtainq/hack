"""Select V75 using nested chronological forecasts, with no leaderboard inputs."""

from dataclasses import asdict
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v75_model import AdmissionsForecaster, ForecastConfig, load_counts


def main():
    output = Path("output/v75")
    output.mkdir(parents=True, exist_ok=True)
    counts = load_counts(Path("processed_data"))
    years = (2017, 2018, 2019, 2021, 2022)
    configurations = []
    for depth, half_life in product((3, 5), (3., 6.)):
        configurations.append(ForecastConfig(depth=depth, half_life=half_life, calibration="none"))
        for mode, penalty in product(("global", "shape"), (10., 100.)):
            configurations.append(ForecastConfig(
                depth=depth, half_life=half_life, calibration=mode, ridge_penalty=penalty,
            ))
    cache, records, predictions = {}, [], {}

    def evaluate(config, year):
        model = AdmissionsForecaster(config).fit(counts, year, cache=cache)
        forecast = model.predict(diagnostics=True)
        forecast["forecast"] = forecast.forecast.round(2)
        forecast["actual"] = counts.loc[forecast.date, "total"].to_numpy()
        forecast["year"] = year
        score = float(np.sqrt(np.mean((np.log1p(forecast.actual) - np.log1p(forecast.forecast)) ** 2)))
        return score, forecast

    for index, config in enumerate(configurations):
        scores = []
        for year in years:
            score, forecast = evaluate(config, year)
            records.append({"config_id": index, "year": year, **asdict(config), "rmsle": score})
            predictions[index, year] = forecast
            scores.append(score)
        print(f"{index + 1}/{len(configurations)} {config.calibration}: {np.mean(scores):.5f}", flush=True)
    search = pd.DataFrame(records)
    search.to_csv(output / "search_metrics.csv", index=False)
    selected_id = int(search.groupby("config_id").rmsle.mean().idxmin())
    selected = configurations[selected_id]
    score, diagnostic = evaluate(selected, 2023)
    metrics = search.loc[search.config_id == selected_id, ["year", "rmsle"]]
    metrics = pd.concat([metrics, pd.DataFrame([{"year": 2023, "rmsle": score}])], ignore_index=True)
    metrics.to_csv(output / "backtest_metrics.csv", index=False)
    pd.concat([predictions[selected_id, year] for year in years] + [diagnostic]).to_csv(
        output / "backtest_daily.csv", index=False,
    )
    (output / "selected_config.json").write_text(json.dumps(asdict(selected), indent=2) + "\n")
    final = AdmissionsForecaster(selected).fit(counts, 2024, cache=cache)
    final.save(output / "model.joblib")
    forecast = final.predict(diagnostics=True)
    forecast.to_csv(output / "prediction_diagnostics.csv", index=False)
    destination = Path("forecast_2024_v75_clean.csv")
    forecast[["date", "forecast"]].to_csv(destination, index=False, float_format="%.2f")
    inputs = [Path("processed_data/registered.csv"), Path("processed_data/included_ks.csv")]
    paths = inputs + [Path("v75_model.py"), Path(__file__), destination]
    metadata = {
        "model": "V75", "forecast_year": 2024, "config": asdict(selected),
        "selection_years": years, "selection_rule": "minimum mean full-season RMSLE",
        "diagnostic_year": 2023, "diagnostic_caveat": "Previously examined; not a blind holdout.",
        "calibration_years": final.calibration_years,
        "calibration_rows": getattr(final, "calibration_rows", 0),
        "training_inputs": [str(path) for path in inputs],
        "old_predictions_used": False, "leaderboard_scores_used": False,
        "external_annual_totals_used": False, "public_rmsle": None,
        "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metrics.to_string(index=False))
    print(asdict(selected))


if __name__ == "__main__":
    main()
