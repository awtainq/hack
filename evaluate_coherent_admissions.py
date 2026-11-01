"""Rolling evaluation of one jointly trained degree/funding model."""

from dataclasses import asdict
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from coherent_admissions_model import CoherentAdmissionsModel, CoherentConfig, load_stream_counts


def main():
    output = Path("output/v73")
    output.mkdir(parents=True, exist_ok=True)
    counts = load_stream_counts(Path("processed_data"))
    counts.to_csv(output / "training_stream_counts.csv", index_label="date")
    years = (2017, 2018, 2019, 2021, 2022)
    configs = [
        CoherentConfig(leaves=leaves, iterations=iterations, auxiliary_weight=auxiliary,
                       half_life=half_life, closure_policy=policy)
        for leaves, iterations, auxiliary, half_life, policy in product(
            (7, 15), (250, 500), (.01, .1), (3., 6.), ("learned", "none"),
        )
    ]
    records, forecasts = [], {}

    def evaluate(config, year):
        forecast = CoherentAdmissionsModel(config).fit(counts, year).predict()
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
    model = CoherentAdmissionsModel(config).fit(counts, 2024)
    model.save(output / "model")
    model.predict(components=True).to_csv(output / "stream_predictions.csv", index=False)
    destination = Path("forecast_2024_v73_coherent.csv")
    model.predict().to_csv(destination, index=False, float_format="%.2f")
    paths = [
        Path("processed_data/registered.csv"), Path("processed_data/included_ks.csv"),
        Path("coherent_admissions_model.py"), Path("joint_admissions_model.py"),
        Path("activity_admissions_model.py"), Path("admissions_profile_model.py"),
        Path(__file__), destination,
    ]
    metadata = {
        "selection_years": years, "diagnostic_year": 2023,
        "diagnostic_caveat": "2023 was previously examined and is not a blind holdout.",
        "selection_rule": "minimum equal-weight mean full-season RMSLE",
        "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "public_score": None, "external_volume_priors_used": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(selected.to_string(index=False))
    print(asdict(config))


if __name__ == "__main__":
    main()
