"""Chronological search for a smooth model of jointly fitted streams."""

from dataclasses import asdict
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v76_model import SmoothAdmissionsModel, SmoothConfig, load_counts


def main():
    output = Path("output/v76")
    output.mkdir(parents=True, exist_ok=True)
    counts = load_counts("processed_data")
    years = (2017, 2018, 2019, 2021, 2022)
    configs = [SmoothConfig(knots=k, penalty=p, auxiliary_weight=a, normalize_volume=n)
               for k, p, a, n in product((5, 7), (.03, .3, 3.), (.05, .5), (False, True))]
    records, predictions = [], {}

    def evaluate(config, year):
        model = SmoothAdmissionsModel(config).fit(counts, year)
        forecast = model.predict()
        forecast["forecast"] = forecast.forecast.round(2)
        forecast["actual"] = counts.loc[forecast.date].sum(axis=1).to_numpy()
        forecast["year"] = year
        score = float(np.sqrt(np.mean((np.log1p(forecast.actual) - np.log1p(forecast.forecast))**2)))
        return score, forecast, model.optimization

    for index, config in enumerate(configs):
        for year in years:
            score, forecast, optimizer = evaluate(config, year)
            records.append({"config_id": index, "year": year, **asdict(config), "rmsle": score,
                            "converged": optimizer["success"], "iterations_used": optimizer["iterations"]})
            predictions[index, year] = forecast
        search = pd.DataFrame(records)
        search.to_csv(output / "search_metrics.csv", index=False)
        current = search.loc[search.config_id == index]
        print(f"{index+1}/{len(configs)} RMSLE={current.rmsle.mean():.5f} converged={current.converged.all()}", flush=True)
    valid = search.groupby("config_id").converged.all()
    ranking = search.groupby("config_id").rmsle.mean().loc[valid]
    if ranking.empty:
        raise RuntimeError("No candidate converged on every fold.")
    selected_id = int(ranking.idxmin())
    config = configs[selected_id]
    score, diagnostic, optimizer = evaluate(config, 2023)
    metrics = search.loc[search.config_id == selected_id, ["year", "rmsle"]]
    metrics = pd.concat([metrics, pd.DataFrame([{"year": 2023, "rmsle": score}])], ignore_index=True)
    metrics.to_csv(output / "backtest_metrics.csv", index=False)
    pd.concat([predictions[selected_id, year] for year in years] + [diagnostic]).to_csv(
        output / "backtest_daily.csv", index=False,
    )
    (output / "selected_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    final = SmoothAdmissionsModel(config).fit(counts, 2024)
    final.save(output / "model.joblib")
    final.predict(components=True).to_csv(output / "stream_predictions.csv", index=False)
    final.predict().to_csv("forecast_2024_v76_smooth.csv", index=False, float_format="%.2f")
    (output / "optimization.json").write_text(json.dumps({"2023": optimizer, "2024": final.optimization}, indent=2) + "\n")
    print(metrics.to_string(index=False))
    print(asdict(config))


if __name__ == "__main__":
    main()
