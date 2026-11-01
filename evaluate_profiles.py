"""Chronological validation of degree profiles; 2023 is diagnostic only."""

from dataclasses import asdict
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from admissions_profile_model import ProfileConfig, ProfileModel, load_counts


def score(actual, forecast):
    return float(np.sqrt(np.mean((np.log1p(actual) - np.log1p(forecast)) ** 2)))


def main():
    output = Path("output/v69")
    output.mkdir(parents=True, exist_ok=True)
    counts = load_counts(Path("processed_data/registered.csv"))
    years = (2017, 2018, 2019, 2021, 2022)
    configs = [ProfileConfig(depth=d, iterations=n, half_life=h, normalize=norm)
               for d, n, h, norm in product((3, 5), (350, 700), (3., 6.), (False, True))]
    records = []
    predictions = {}
    for index, config in enumerate(configs):
        for year in years:
            fitted = ProfileModel(config).fit(counts, year)
            forecast = fitted.predict()
            forecast["actual"] = counts.loc[counts.index.year == year].sum(axis=1).to_numpy()
            forecast["forecast"] = forecast.forecast.round(2)
            forecast["year"] = year
            records.append({"config_id": index, "year": year, **asdict(config),
                            "rmsle": score(forecast.actual, forecast.forecast)})
            predictions[index, year] = forecast
        values = [row["rmsle"] for row in records if row["config_id"] == index]
        print(f"{index+1}/{len(configs)} mean {np.mean(values):.5f} {asdict(config)}", flush=True)
    metrics = pd.DataFrame(records)
    metrics.to_csv(output / "search_metrics.csv", index=False)
    selected_id = int(metrics.groupby("config_id").rmsle.mean().idxmin())
    config = configs[selected_id]
    report = metrics.loc[metrics.config_id == selected_id, ["year", "rmsle"]].copy()
    diagnostic = ProfileModel(config).fit(counts, 2023).predict()
    diagnostic["actual"] = counts.loc[counts.index.year == 2023].sum(axis=1).to_numpy()
    diagnostic["forecast"] = diagnostic.forecast.round(2)
    diagnostic["year"] = 2023
    report = pd.concat([report, pd.DataFrame([{"year": 2023, "rmsle": score(diagnostic.actual, diagnostic.forecast)}])])
    report.to_csv(output / "backtest_metrics.csv", index=False)
    pd.concat([predictions[selected_id, year] for year in years] + [diagnostic]).to_csv(output / "backtest_daily.csv", index=False)
    (output / "selected_config.json").write_text(json.dumps(asdict(config), indent=2)+"\n")
    final = ProfileModel(config).fit(counts, 2024)
    final.save(output / "model.joblib")
    forecast = final.predict()
    forecast.to_csv(output / "degree_predictions.csv", index=False, float_format="%.6f")
    forecast[["date", "forecast"]].to_csv("forecast_2024_v69_profile.csv", index=False, float_format="%.2f")
    print(report.to_string(index=False))
    print("SELECTED", asdict(config))
    print("SUMS", forecast.select_dtypes("number").sum().to_dict())


if __name__ == "__main__":
    main()
