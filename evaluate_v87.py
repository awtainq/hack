"""Chronological selection of fresh calendar/lag learners and OOF corrections."""
from dataclasses import asdict, replace
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v87_autoregressive import AutoregressiveConfig, SeasonalAutoregressor
from v87_data import load_daily


def main():
    output = Path("output/v87")
    output.mkdir(parents=True, exist_ok=True)
    years = (2017, 2018, 2019, 2021, 2022, 2023)
    protocol = {"years": years, "selection_rule": "recency-weighted mean full 73-day RMSLE",
                "validation_weight_half_life": 3.0, "independent_holdout_available": False,
                "leaderboard_inputs_allowed": False}
    (output / "validation_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    counts = load_daily(Path("processed_data"))
    configs = [AutoregressiveConfig(family=family, depth=depth, iterations=250, half_life=half,
                                    use_lags=lags, positive_only=positive, correction=correction, readout=readout)
               for family, depth, half, lags, positive, correction, readout in product(
                   ("cat", "lgb"), (3, 5), (3., 6.), (False, True), (False, True), (False, True),
                   ("total", "degree", "funding"))]
    records, predictions, cache = [], {}, {}
    for index, config in enumerate(configs):
        for year in years:
            model = SeasonalAutoregressor(config).fit(counts, year, cache=cache)
            forecast = model.predict(diagnostics=True)
            forecast["forecast"] = forecast.forecast.round(6)
            forecast["actual"] = counts.loc[forecast.date, "total"].to_numpy()
            forecast["year"] = year
            error = np.log1p(forecast.forecast) - np.log1p(forecast.actual)
            records.append({"config_id": index, "year": year, **asdict(config), "rmsle": float(np.sqrt(np.mean(error ** 2)))})
            predictions[index, year] = forecast
        part = pd.DataFrame(records).loc[lambda frame: frame.config_id == index]
        weights = np.exp2(-(2024 - part.year.to_numpy()) / 3.0)
        print(f"{index + 1}/{len(configs)} score={np.average(part.rmsle, weights=weights):.5f} {asdict(config)}", flush=True)
        pd.DataFrame(records).to_csv(output / "search_metrics.csv", index=False)
    search = pd.DataFrame(records)
    ranking = []
    for index, part in search.groupby("config_id"):
        weights = np.exp2(-(2024 - part.year.to_numpy()) / 3.0)
        ranking.append({"config_id": int(index), "criterion": float(np.average(part.rmsle, weights=weights)),
                        "mean_rmsle": float(part.rmsle.mean())})
    ranking = pd.DataFrame(ranking).sort_values("criterion")
    ranking.to_csv(output / "selection.csv", index=False)
    chosen = int(ranking.iloc[0].config_id)
    config = configs[chosen]
    search.loc[search.config_id == chosen, ["year", "rmsle"]].to_csv(output / "backtest_metrics.csv", index=False)
    pd.concat([predictions[chosen, year] for year in years]).to_csv(output / "backtest_daily.csv", index=False)
    (output / "selected_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    final = SeasonalAutoregressor(config).fit(counts, 2024, cache=cache)
    final.save(output / "model.joblib")
    forecast = final.predict(diagnostics=True)
    forecast.to_csv(output / "prediction_diagnostics.csv", index=False)
    destination = Path("forecast_2024_v87.csv")
    forecast[["date", "forecast"]].to_csv(destination, index=False, float_format="%.6f")
    paths = [Path("processed_data/registered.csv"), Path("processed_data/included_ks.csv"), Path("v87_data.py"),
             Path("v87_autoregressive.py"), Path(__file__), destination]
    metadata = {"model": "V87", "forecast_year": 2024, "config": asdict(config),
                "training_inputs": [str(path) for path in paths[:2]], "training_years": final.training_years,
                "calibration_years": final.calibration_years, "selection": ranking.iloc[0].to_dict(),
                "validation_protocol": protocol, "feature_names": final.feature_names,
                "public_rmsle": None, "old_forecasts_used": False, "leaderboard_scores_used": False,
                "target_year_annual_totals_used": False,
                "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(search.loc[search.config_id == chosen, ["year", "rmsle"]].to_string(index=False), flush=True)
    print("Selected:", asdict(config), flush=True)


if __name__ == "__main__":
    main()
