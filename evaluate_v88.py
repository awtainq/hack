"""Full-season rolling validation of one forest, without leaderboard inputs."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from itertools import product
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v88_model import AdmissionsForest, ForestConfig, load_counts, load_exam_calendar


def main():
    output = Path("output/v88")
    output.mkdir(parents=True, exist_ok=True)
    years = (2017, 2018, 2019, 2021, 2022, 2023)
    protocol = {
        "years": years,
        "selection_rule": "recency-weighted mean full 73-day RMSLE",
        "validation_weight_half_life": 3.0,
        "independent_holdout_available": False,
        "leaderboard_inputs_allowed": False,
    }
    (output / "validation_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    counts = load_counts(Path("processed_data/registered.csv"))
    exams = load_exam_calendar(Path("external_data/ege/exam_calendar.csv"))
    configs = [
        ForestConfig(
            family=family,
            min_leaf=leaf,
            feature_fraction=fraction,
            half_life=half,
            annual_context=context,
            schedule=schedule,
            exams=ege,
        )
        for family, leaf, fraction, half, context, schedule, ege in product(
            ("forest", "extra"),
            (3, 6, 10),
            (0.7, 1.0),
            (3.0, 6.0),
            (False, True),
            ("none", "learned"),
            (False, True),
        )
    ]

    def evaluate(index):
        config = configs[index]
        rows, predictions = [], []
        for year in years:
            model = AdmissionsForest(config).fit(counts, year, exams)
            forecast = model.predict()
            forecast["forecast"] = forecast.forecast.round(6)
            forecast["actual"] = counts.loc[forecast.date, "total"].to_numpy()
            forecast["year"] = year
            forecast["config_id"] = index
            error = np.log1p(forecast.forecast) - np.log1p(forecast.actual)
            rows.append(
                {
                    "config_id": index,
                    "year": year,
                    **asdict(config),
                    "rmsle": float(np.sqrt(np.mean(error**2))),
                }
            )
            predictions.append(forecast)
        return rows, pd.concat(predictions)

    records, forecasts = [], {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs = {executor.submit(evaluate, index): index for index in range(len(configs))}
        for completed, future in enumerate(as_completed(jobs), 1):
            index = jobs[future]
            rows, prediction = future.result()
            records.extend(rows)
            forecasts[index] = prediction
            part = pd.DataFrame(rows)
            criterion = np.average(
                part.rmsle, weights=np.exp2(-(2024 - part.year.to_numpy()) / 3.0)
            )
            print(f"{completed}/{len(configs)} config={index} score={criterion:.5f}", flush=True)
            pd.DataFrame(records).sort_values(["config_id", "year"]).to_csv(
                output / "search_metrics.csv", index=False
            )
    search = pd.DataFrame(records)
    ranking = []
    for index, part in search.groupby("config_id"):
        ranking.append(
            {
                "config_id": int(index),
                "criterion": float(
                    np.average(part.rmsle, weights=np.exp2(-(2024 - part.year.to_numpy()) / 3.0))
                ),
                "mean_rmsle": float(part.rmsle.mean()),
            }
        )
    ranking = pd.DataFrame(ranking).sort_values(["criterion", "config_id"])
    ranking.to_csv(output / "selection.csv", index=False)
    chosen = int(ranking.iloc[0].config_id)
    config = configs[chosen]
    search.loc[search.config_id == chosen, ["year", "rmsle"]].sort_values("year").to_csv(
        output / "backtest_metrics.csv", index=False
    )
    forecasts[chosen].drop(columns="config_id").to_csv(output / "backtest_daily.csv", index=False)
    (output / "selected_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    final = AdmissionsForest(config).fit(counts, 2024, exams)
    final.save(output / "model.joblib")
    destination = Path("forecast_2024_v88.csv")
    final.predict().to_csv(destination, index=False, float_format="%.6f")
    pd.DataFrame(
        {"feature": final.feature_names, "importance": final.regressor.feature_importances_}
    ).sort_values("importance", ascending=False).to_csv(
        output / "feature_importance.csv", index=False
    )
    paths = [
        Path("processed_data/registered.csv"),
        Path("v88_model.py"),
        Path(__file__),
        destination,
    ]
    inputs = [paths[0]]
    if config.exams:
        inputs.append(Path("external_data/ege/exam_calendar.csv"))
        paths += inputs[1:]
    metadata = {
        "model": "V88",
        "forecast_year": 2024,
        "config": asdict(config),
        "training_inputs": [str(path) for path in inputs],
        "training_years": final.training_years,
        "selection": ranking.iloc[0].to_dict(),
        "validation_protocol": protocol,
        "public_rmsle": None,
        "old_forecasts_used": False,
        "leaderboard_scores_used": False,
        "target_year_annual_totals_used": False,
        "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        search.loc[search.config_id == chosen, ["year", "rmsle"]]
        .sort_values("year")
        .to_string(index=False),
        flush=True,
    )
    print("Selected:", asdict(config), flush=True)


if __name__ == "__main__":
    main()
