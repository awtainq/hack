"""Select V86 using chronological forecasts, never competition feedback."""
from dataclasses import asdict
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v86_volume_shape import ShapeConfig, VolumeShapeForecaster, load_counts


def main():
    output = Path("output/v86")
    output.mkdir(parents=True, exist_ok=True)
    years = (2017, 2018, 2019, 2021, 2022, 2023)
    # All six years are development validation; none is called a blind holdout.
    # Recent seasons receive more weight because submission channels changed.
    protocol = {"years": years, "selection_rule": "recency-weighted mean full 73-day RMSLE",
                "validation_weight_half_life": 3.0, "independent_holdout_available": False,
                "leaderboard_inputs_allowed": False}
    (output / "validation_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    counts = load_counts(Path("processed_data/registered.csv"))
    configs = [ShapeConfig(knots=knots, penalty=penalty, half_life=half, volume_trend=trend,
                           channels=channels, calibration_penalty=calibration)
               for knots, penalty, half, trend, channels, calibration in
               product((5, 7, 9), (.03, .3), (3., 6.), (False, True), (False, True), (None, 100.))]
    records, predictions, cache = [], {}, {}
    for index, config in enumerate(configs):
        for year in years:
            model = VolumeShapeForecaster(config).fit(counts, year, cache=cache)
            forecast = model.predict(components=True)
            forecast["forecast"] = forecast.forecast.round(6)
            forecast["actual"] = counts.loc[forecast.date].sum(axis=1).to_numpy()
            forecast["year"] = year
            error = np.log1p(forecast.forecast) - np.log1p(forecast.actual)
            records.append({"config_id": index, "year": year, **asdict(config),
                            "rmsle": float(np.sqrt(np.mean(error ** 2))),
                            "optimization_success": model.optimization["success"],
                            "log_bias": model.log_bias})
            predictions[index, year] = forecast
        part = pd.DataFrame(records).loc[lambda frame: frame.config_id == index]
        weights = np.exp2(-(2024 - part.year.to_numpy()) / 3.0)
        score = np.average(part.rmsle, weights=weights)
        print(f"{index + 1}/{len(configs)} score={score:.5f} {asdict(config)}", flush=True)
        pd.DataFrame(records).to_csv(output / "search_metrics.csv", index=False)
    results = pd.DataFrame(records)
    ranking = []
    for index, part in results.groupby("config_id"):
        weights = np.exp2(-(2024 - part.year.to_numpy()) / 3.0)
        ranking.append({"config_id": int(index), "criterion": float(np.average(part.rmsle, weights=weights)),
                        "mean_rmsle": float(part.rmsle.mean())})
    ranking = pd.DataFrame(ranking).sort_values("criterion")
    ranking.to_csv(output / "selection.csv", index=False)
    chosen = int(ranking.iloc[0].config_id)
    config = configs[chosen]
    results.loc[results.config_id == chosen, ["year", "rmsle", "optimization_success", "log_bias"]].to_csv(output / "backtest_metrics.csv", index=False)
    pd.concat([predictions[chosen, year] for year in years]).to_csv(output / "backtest_daily.csv", index=False)
    (output / "selected_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    final = VolumeShapeForecaster(config).fit(counts, 2024, cache=cache)
    final.save(output / "model.joblib")
    forecast = final.predict(components=True)
    forecast.to_csv(output / "stream_forecasts.csv", index=False)
    destination = Path("forecast_2024_v86.csv")
    forecast[["date", "forecast"]].to_csv(destination, index=False, float_format="%.6f")
    source = [Path("processed_data/registered.csv"), Path("v86_volume_shape.py"), Path(__file__), destination]
    metadata = {"model": "V86", "forecast_year": 2024, "config": asdict(config),
                "training_inputs": [str(source[0])], "training_years": final.training_years,
                "calibration_years": final.calibration_years, "learned_log_bias": final.log_bias,
                "forecast_volumes_before_bias": dict(zip(STREAMS_FOR_REPORT, final.future_volumes.tolist())),
                "selection": ranking.iloc[0].to_dict(), "validation_protocol": protocol,
                "optimization": final.optimization, "public_rmsle": None,
                "old_forecasts_used": False, "leaderboard_scores_used": False,
                "target_year_annual_totals_used": False,
                "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source}}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(results.loc[results.config_id == chosen, ["year", "rmsle"]].to_string(index=False), flush=True)
    print("Selected:", asdict(config), flush=True)


STREAMS_FOR_REPORT = ("bachelor_other", "bachelor_gov", "master", "college", "other_degree")


if __name__ == "__main__":
    main()
