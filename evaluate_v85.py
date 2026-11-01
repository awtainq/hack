"""Choose readout and base-model parameters using chronological historical data."""
from dataclasses import asdict
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v85_historical_model import HistoricalForecaster, ModelConfig, load_daily


def main():
    output = Path("output/v85")
    output.mkdir(parents=True, exist_ok=True)
    counts = load_daily(Path("processed_data"))
    years = (2017, 2018, 2019, 2021, 2022)
    configs = [ModelConfig(depth=depth, half_life=half, ridge_penalty=penalty)
               for depth, half, penalty in product((3, 5), (3., 6.), (None, 10., 100.))]
    records, predictions, cache = [], {}, {}

    def evaluate(config, year):
        model = HistoricalForecaster(config).fit(counts, year, cache=cache)
        prediction = model.predict(diagnostics=True)
        prediction["forecast"] = prediction.forecast.round(6)
        prediction["actual"] = counts.loc[prediction.date, "total"].to_numpy()
        prediction["year"] = year
        error = np.log1p(prediction.forecast) - np.log1p(prediction.actual)
        return float(np.sqrt(np.mean(error ** 2))), prediction

    for index, config in enumerate(configs):
        scores = []
        for year in years:
            score, prediction = evaluate(config, year)
            records.append({"config_id": index, "year": year, **asdict(config), "rmsle": score})
            predictions[index, year] = prediction
            scores.append(score)
        print(f"{index + 1}/{len(configs)} {asdict(config)}: {np.mean(scores):.5f}", flush=True)
        pd.DataFrame(records).to_csv(output / "search_metrics.csv", index=False)
    search = pd.DataFrame(records)
    ranking = search.groupby("config_id").rmsle.mean().sort_values()
    ranking.to_csv(output / "selection.csv")
    chosen_id = int(ranking.index[0])
    config = configs[chosen_id]
    score, diagnostic = evaluate(config, 2023)
    metrics = search.loc[search.config_id == chosen_id, ["year", "rmsle"]]
    metrics = pd.concat([metrics, pd.DataFrame([{"year": 2023, "rmsle": score}])], ignore_index=True)
    metrics.to_csv(output / "backtest_metrics.csv", index=False)
    pd.concat([predictions[chosen_id, year] for year in years] + [diagnostic]).to_csv(output / "backtest_daily.csv", index=False)
    (output / "selected_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    final = HistoricalForecaster(config).fit(counts, 2024, cache=cache)
    final.save(output / "model.joblib")
    prediction = final.predict(diagnostics=True)
    prediction.to_csv(output / "prediction_diagnostics.csv", index=False)
    destination = Path("forecast_2024_v85_historical.csv")
    prediction[["date", "forecast"]].to_csv(destination, index=False, float_format="%.6f")
    paths = [Path("processed_data/registered.csv"), Path("processed_data/included_ks.csv"),
             Path("v85_historical_model.py"), Path(__file__), destination]
    metadata = {
        "model": "V85", "training_inputs": [str(path) for path in paths[:2]],
        "config": asdict(config), "selection_years": years,
        "selection_mean_rmsle": float(ranking.iloc[0]), "diagnostic_year": 2023,
        "diagnostic_caveat": "Examined before, not a blind holdout",
        "training_years": final.training_years, "calibration_years": final.calibration_years,
        "learned_readout_coefficients": final.coefficients.tolist(),
        "old_forecasts_used": False, "leaderboard_scores_used": False,
        "target_year_annual_total_used": False, "public_rmsle": None,
        "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
