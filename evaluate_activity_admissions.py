"""Chronological selection; daily labels from the target season are excluded."""

from dataclasses import asdict
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from activity_admissions_model import ActivityAdmissionsModel, ActivityConfig, load_counts


def main():
    output = Path("output/v72")
    output.mkdir(parents=True, exist_ok=True)
    source = Path("processed_data/registered.csv")
    counts = load_counts(source)
    years = (2017, 2018, 2019, 2021, 2022)
    configs = [
        ActivityConfig(depth=depth, iterations=iterations, half_life=half_life,
                       activity_half_life=activity_half_life)
        for depth, iterations, half_life, activity_half_life in product(
            (3, 5), (300, 600), (1.5, 4.0), (1.0, 4.0),
        )
    ]
    records, forecasts = [], {}

    def evaluate(config, year):
        forecast = ActivityAdmissionsModel(config).fit(counts, year).predict(diagnostics=True)
        forecast["forecast"] = forecast.forecast.round(2)
        forecast["actual"] = counts.reindex(forecast.date).sum(axis=1).to_numpy()
        forecast["year"] = year
        error = np.log1p(forecast.forecast) - np.log1p(forecast.actual)
        return float(np.sqrt(np.mean(error**2))), forecast

    for config_id, config in enumerate(configs):
        scores = []
        for year in years:
            score, forecast = evaluate(config, year)
            records.append({"config_id": config_id, "year": year, **asdict(config), "rmsle": score})
            forecasts[config_id, year] = forecast
            scores.append(score)
        print(f"{config_id+1}/{len(configs)} mean RMSLE={np.mean(scores):.5f}", flush=True)
    search = pd.DataFrame(records)
    search.to_csv(output / "search_metrics.csv", index=False)
    chosen_id = int(search.groupby("config_id").rmsle.mean().idxmin())
    config = configs[chosen_id]
    score, diagnostic = evaluate(config, 2023)
    selected = search.loc[search.config_id == chosen_id, ["year", "rmsle"]]
    selected = pd.concat([selected, pd.DataFrame([{"year": 2023, "rmsle": score}])], ignore_index=True)
    selected.to_csv(output / "backtest_metrics.csv", index=False)
    pd.concat([forecasts[chosen_id, year] for year in years] + [diagnostic]).to_csv(
        output / "backtest_daily.csv", index=False,
    )
    (output / "selected_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    final = ActivityAdmissionsModel(config).fit(counts, 2024)
    final.save(output / "model")
    final.predict(diagnostics=True).to_csv(output / "prediction_diagnostics.csv", index=False)
    destination = Path("forecast_2024_v72_activity.csv")
    final.predict().to_csv(destination, index=False, float_format="%.2f")
    metadata = {
        "selection_years": years, "diagnostic_year": 2023,
        "diagnostic_caveat": "2023 has been examined previously and is not a blind holdout.",
        "selection_rule": "minimum equal-weight mean full-season RMSLE",
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path("activity_admissions_model.py").read_bytes()).hexdigest(),
        "forecast_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "public_score": None, "external_volume_priors_used": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(selected.to_string(index=False))
    print(asdict(config))


if __name__ == "__main__":
    main()
