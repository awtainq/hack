"""Small chronological search for the calendar kernel model."""

from dataclasses import asdict
from itertools import product
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from admissions_profile_model import load_counts
from kernel_admissions_model import KernelAdmissionsModel, KernelConfig


def main():
    output = Path("output/v70")
    output.mkdir(parents=True, exist_ok=True)
    counts = load_counts(Path("processed_data/registered.csv"))
    years = (2017, 2018, 2019, 2021, 2022)
    configs = [KernelConfig(bandwidth=b, penalty=p, half_life=h, weekday_coupling=c)
               for b, p, h, c in product((4., 8., 16.), (.03, .3, 3.), (3., 6.), (.05, .5))]
    records, predictions = [], {}
    for i, config in enumerate(configs):
        for year in years:
            forecast = KernelAdmissionsModel(config).fit(counts, year).predict()
            forecast["actual"] = counts.loc[counts.index.year == year].sum(axis=1).to_numpy()
            forecast["forecast"] = forecast.forecast.round(2)
            forecast["year"] = year
            error = np.log1p(forecast.actual) - np.log1p(forecast.forecast)
            records.append({"config_id": i, "year": year, **asdict(config), "rmsle": np.sqrt(np.mean(error ** 2))})
            predictions[i, year] = forecast
        if i % 4 == 3:
            print(f"{i+1}/{len(configs)} completed", flush=True)
    metrics = pd.DataFrame(records)
    metrics.to_csv(output / "search_metrics.csv", index=False)
    selected_id = int(metrics.groupby("config_id").rmsle.mean().idxmin())
    config = configs[selected_id]
    selected = metrics.loc[metrics.config_id == selected_id, ["year", "rmsle"]].copy()
    diagnostic = KernelAdmissionsModel(config).fit(counts, 2023).predict()
    diagnostic["actual"] = counts.loc[counts.index.year == 2023].sum(axis=1).to_numpy()
    diagnostic["forecast"] = diagnostic.forecast.round(2)
    diagnostic["year"] = 2023
    error = np.log1p(diagnostic.actual) - np.log1p(diagnostic.forecast)
    selected = pd.concat([selected, pd.DataFrame([{"year": 2023, "rmsle": np.sqrt(np.mean(error**2))}])])
    selected.to_csv(output / "backtest_metrics.csv", index=False)
    pd.concat([predictions[selected_id, year] for year in years] + [diagnostic]).to_csv(output / "backtest_daily.csv", index=False)
    (output / "selected_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    final = KernelAdmissionsModel(config).fit(counts, 2024)
    joblib.dump(final, output / "model.joblib")
    final.predict().to_csv("forecast_2024_v70_kernel.csv", index=False, float_format="%.2f")
    print(selected.to_string(index=False))
    print(asdict(config))


if __name__ == "__main__":
    main()
