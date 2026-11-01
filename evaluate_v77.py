"""Rolling forecast comparison, including a matched external-data ablation."""
from dataclasses import asdict
from itertools import product
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v77_model import DailyAdmissionsModel, ModelConfig, load_counts, load_exams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Reuse a compatible completed local search")
    args = parser.parse_args()
    output = Path("output/v77")
    output.mkdir(parents=True, exist_ok=True)
    counts = load_counts(Path("processed_data/registered.csv"))
    exams = load_exams(Path("external_data/ege/exam_calendar.csv"))
    years = (2017, 2018, 2019, 2021, 2022)
    configs = [ModelConfig(depth=depth, half_life=half, hurdle=hurdle, exams=ege)
               for depth, half, hurdle, ege in product((3, 5), (3., 6.), (False, True), (False, True))]
    configs += [ModelConfig(depth=depth, half_life=half, hurdle=False, exams=ege, quiet_groups=True)
                for depth, half, ege in product((3, 5), (3., 6.), (False, True))]
    configs += [ModelConfig(iterations=iterations, half_life=half, hurdle=False, exams=True)
                for iterations, half in product((250, 1200), (3., 6.))]
    configs += [ModelConfig(iterations=iterations, half_life=half, hurdle=False, exams=ege, hard_schedule=True)
                for iterations, half, ege in product((250, 600), (3., 6.), (False, True))]
    records, forecasts = [], {}
    # Resume completed configurations; the frozen initial experiment is retained.
    if args.resume and (output / "search_daily.csv").exists():
        prior_metadata = json.loads((output / "metadata.json").read_text())
        for path in (Path("processed_data/registered.csv"), Path("external_data/ege/exam_calendar.csv"), Path("v77_model.py")):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if prior_metadata["sha256"].get(str(path)) != digest:
                raise ValueError(f"Cached experiment no longer matches {path}; rerun without --resume")
        previous = pd.read_csv(output / "search_metrics.csv")
        previous["quiet_groups"] = previous.get("quiet_groups", False)
        previous["hard_schedule"] = previous.get("hard_schedule", False)
        daily = pd.read_csv(output / "search_daily.csv", parse_dates=["date"])
        for index in previous.config_id.unique():
            part = previous.loc[previous.config_id == index]
            same_config = int(index) < len(configs) and all(
                (part[key] == value).all() for key, value in asdict(configs[int(index)]).items()
            )
            if same_config and set(part.year) == set(years) and len(daily.loc[daily.config_id == index]) == len(years) * 73:
                records += part.to_dict("records")
                for year in years:
                    forecasts[index, year] = daily.loc[(daily.config_id == index) & (daily.year == year)].drop(columns="config_id").copy()

    def evaluate(config, year):
        model = DailyAdmissionsModel(config).fit(counts, year, exams)
        prediction = model.predict(diagnostics=True)
        prediction["forecast"] = prediction.forecast.round(2)
        prediction["actual"] = counts.loc[prediction.date, "total"].to_numpy()
        prediction["year"] = year
        error = np.log1p(prediction.forecast) - np.log1p(prediction.actual)
        return float(np.sqrt(np.mean(error ** 2))), prediction

    for index, config in enumerate(configs):
        if all((index, year) in forecasts for year in years):
            continue
        scores = []
        for year in years:
            score, prediction = evaluate(config, year)
            records.append({"config_id": index, "year": year, **asdict(config), "rmsle": score})
            forecasts[index, year] = prediction
            scores.append(score)
        pd.DataFrame(records).to_csv(output / "search_metrics.csv", index=False)
        print(f"{index + 1}/{len(configs)} {asdict(config)} mean={np.mean(scores):.5f}", flush=True)

    search = pd.DataFrame(records)
    ranking = search.groupby("config_id").rmsle.mean().sort_values()
    ranking.to_csv(output / "selection.csv")
    selected_id = int(ranking.index[0])
    selected = configs[selected_id]
    score, diagnostic = evaluate(selected, 2023)
    metrics = search.loc[search.config_id == selected_id, ["year", "rmsle"]]
    metrics = pd.concat([metrics, pd.DataFrame([{"year": 2023, "rmsle": score}])], ignore_index=True)
    metrics.to_csv(output / "backtest_metrics.csv", index=False)
    pd.concat([forecasts[selected_id, year] for year in years] + [diagnostic]).to_csv(output / "backtest_daily.csv", index=False)
    (output / "selected_config.json").write_text(json.dumps(asdict(selected), indent=2) + "\n")
    pd.concat([frame.assign(config_id=index) for (index, year), frame in forecasts.items()]).to_csv(output / "search_daily.csv", index=False)
    final = DailyAdmissionsModel(selected).fit(counts, 2024, exams)
    final.save(output / "model.joblib")
    prediction = final.predict(diagnostics=True)
    prediction.to_csv(output / "prediction_diagnostics.csv", index=False)
    destination = Path("forecast_2024_v77.csv")
    prediction[["date", "forecast"]].to_csv(destination, index=False, float_format="%.2f")
    inputs = [Path("processed_data/registered.csv")]
    if selected.exams:
        inputs.append(Path("external_data/ege/exam_calendar.csv"))
    paths = inputs + [Path("v77_model.py"), Path(__file__), destination]
    metadata = {
        "model": "V77", "forecast_year": 2024, "config": asdict(selected),
        "selection_years": years, "selection_rule": "minimum mean 73-day RMSLE",
        "selection_mean_rmsle": float(ranking.iloc[0]),
        "diagnostic_year": 2023, "diagnostic_caveat": "Previously examined, not a blind holdout",
        "training_years": final.training_years, "training_inputs": [str(path) for path in inputs],
        "old_predictions_used": False, "leaderboard_scores_used": False,
        "external_annual_totals_used": False, "public_rmsle": None,
        "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metrics.to_string(index=False), flush=True)
    print(asdict(selected), flush=True)


if __name__ == "__main__":
    main()
