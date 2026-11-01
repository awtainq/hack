"""Historical selection of degree level/profile models, without reference CSVs."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v91_model import AdmissionsModel, DEGREES, ProfileConfig, ProfileModel
from v88_model import load_counts, load_exam_calendar
from v90_model import degree_targets

YEARS = (2017, 2018, 2019, 2021, 2022, 2023)
OUTPUT = Path("output/v91")


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_configs():
    levels = (("none", 3), ("mean", 3), ("trend", 3), ("trend", 5))
    configs = [
        ProfileConfig(
            family="spline",
            knots=knots,
            penalty=penalty,
            half_life=half,
            level_method=method,
            level_window=window,
            align_history=aligned,
        )
        for knots, penalty, half, (method, window), aligned in product(
            (7, 11), (0.3, 3.0, 30.0), (2.0, 6.0), levels, (False, True)
        )
    ]
    configs.extend(
        ProfileConfig(
            family="catboost",
            depth=depth,
            half_life=half,
            level_method=method,
            level_window=window,
            align_history=aligned,
            exams=exams,
        )
        for depth, half, (method, window), aligned, exams in product(
            (3, 5), (2.0, 6.0), levels, (False, True), (False, True)
        )
    )
    configs.extend(
        ProfileConfig(
            family="extra",
            iterations=192,
            min_leaf=leaf,
            half_life=half,
            level_method=method,
            level_window=window,
        )
        for leaf, half, (method, window) in product((3, 8), (2.0, 6.0), levels)
    )
    return configs


def main():
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    configs = candidate_configs()
    protocol = {
        "validation_years": YEARS,
        "criterion": "pooled recency-weighted mean squared log1p error over complete campaigns",
        "validation_weight_half_life": 3.0,
        "selection": "individual initialization then up to six coordinate passes on total",
        "independent_holdout_available": False,
        "previous_forecast_inputs": False,
        "leaderboard_inputs": False,
        "target_year_campaign_totals": False,
        "candidates_per_degree": len(configs),
        "level_definition": "mean weekday log1p registrations per historical season",
    }
    (output / "validation_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "candidate_configs.json").write_text(
        json.dumps([asdict(c) for c in configs], indent=2) + "\n"
    )
    counts = load_counts(Path("processed_data/registered.csv"))
    exams = load_exam_calendar(Path("external_data/ege/exam_calendar.csv"))
    actual = np.stack([degree_targets(counts.loc[str(y)]).to_numpy().T for y in YEARS], axis=1)
    totals = actual.sum(axis=0)
    weights = np.exp2(-(2024 - np.asarray(YEARS)) / 3.0)
    predictions = np.empty((len(configs), len(DEGREES), len(YEARS), 73))

    def criterion(predicted, observed):
        squared = (np.log1p(predicted) - np.log1p(observed)) ** 2
        return float(np.average(squared.mean(axis=1), weights=weights))

    def evaluate(index, degree):
        values = np.stack(
            [
                ProfileModel(DEGREES[degree], configs[index])
                .fit(counts, y, exams)
                .predict()
                .to_numpy()
                for y in YEARS
            ]
        )
        return index, degree, values

    records = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs = [
            executor.submit(evaluate, index, degree)
            for index in range(len(configs))
            for degree in range(len(DEGREES))
        ]
        for completed, future in enumerate(as_completed(jobs), 1):
            index, degree, values = future.result()
            predictions[index, degree] = values
            for fold, year in enumerate(YEARS):
                mse = float(np.mean((np.log1p(values[fold]) - np.log1p(actual[degree, fold])) ** 2))
                records.append(
                    {
                        "config_id": index,
                        "degree": DEGREES[degree],
                        "year": year,
                        "mse_log1p": mse,
                        "rmsle": float(np.sqrt(mse)),
                        **asdict(configs[index]),
                    }
                )
            if completed % 12 == 0 or completed == len(jobs):
                pd.DataFrame(records).to_csv(output / "search_metrics.csv", index=False)
                print(f"Completed degree/config combinations {completed}/{len(jobs)}", flush=True)
    np.savez_compressed(output / "search_predictions.npz", predictions=predictions)
    selected = {
        degree: int(
            np.argmin(
                [criterion(predictions[index, i], actual[i]) for index in range(len(configs))]
            )
        )
        for i, degree in enumerate(DEGREES)
    }
    initial = selected.copy()

    def combined(indices):
        return sum(predictions[indices[d], i] for i, d in enumerate(DEGREES))

    trace = []
    for iteration in range(6):
        changed = False
        for degree in DEGREES:
            scores = [
                criterion(combined({**selected, degree: index}), totals)
                for index in range(len(configs))
            ]
            best = int(np.argmin(scores))
            changed |= selected[degree] != best
            selected[degree] = best
            trace.append(
                {
                    "pass": iteration + 1,
                    "degree": degree,
                    "config_id": best,
                    "pooled_rmsle": float(np.sqrt(scores[best])),
                }
            )
        if not changed:
            break
    selected_predictions = combined(selected)
    selection = {
        "degrees": {d: asdict(configs[selected[d]]) for d in DEGREES},
        "selected_indices": selected,
        "individual_initial_indices": initial,
        "validation_years": YEARS,
        "pooled_rmsle": float(np.sqrt(criterion(selected_predictions, totals))),
    }
    (output / "selected_config.json").write_text(json.dumps(selection, indent=2) + "\n")
    pd.DataFrame(trace).to_csv(output / "joint_selection_trace.csv", index=False)
    rows, daily = [], []
    for fold, year in enumerate(YEARS):
        prediction = selected_predictions[fold]
        rows.append(
            {
                "year": year,
                "rmsle": float(
                    np.sqrt(np.mean((np.log1p(prediction) - np.log1p(totals[fold])) ** 2))
                ),
            }
        )
        daily.append(
            pd.DataFrame(
                {
                    "date": counts.loc[str(year)].index,
                    "forecast": prediction,
                    "actual": totals[fold],
                    **{d: predictions[selected[d], i, fold] for i, d in enumerate(DEGREES)},
                }
            )
        )
    pd.DataFrame(rows).to_csv(output / "backtest_metrics.csv", index=False)
    pd.concat(daily, ignore_index=True).to_csv(output / "backtest_daily.csv", index=False)
    model = AdmissionsModel({d: configs[selected[d]] for d in DEGREES}).fit(counts, 2024, exams)
    model.save(output / "model.joblib")
    forecast = model.predict(components=True)
    destination = Path("forecast_2024_v91.csv")
    forecast[["date", "forecast"]].to_csv(destination, index=False, float_format="%.6f")
    forecast.to_csv(output / "degree_forecasts.csv", index=False, float_format="%.6f")
    sources = [
        Path("v91_model.py"),
        Path(__file__),
        Path("v88_model.py"),
        Path("v90_model.py"),
        Path("processed_data/registered.csv"),
        Path("external_data/ege/exam_calendar.csv"),
    ]
    metadata = {
        "model": "V91 historical level + daily profile",
        "selection": selection,
        "protocol": protocol,
        "training_years": model.models["bachelor"].training_years,
        "previous_forecasts_used": False,
        "leaderboard_scores_used": False,
        "target_year_annual_totals_used": False,
        "public_rmsle": None,
        "future_log_levels": {d: model.models[d].future_level for d in DEGREES},
        "files_sha256": {
            str(p): checksum(p) for p in sources + [destination, output / "selected_config.json"]
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    frozen_paths = sources + [
        destination,
        output / "selected_config.json",
        output / "metadata.json",
        output / "model.joblib",
    ]
    freeze = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Fixed before final reference comparison",
        "sha256": {str(p): checksum(p) for p in frozen_paths},
    }
    (output / "selection_frozen.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(json.dumps(selection, indent=2), flush=True)
    print("2024 totals", forecast[list(DEGREES) + ["forecast"]].sum().to_dict(), flush=True)


if __name__ == "__main__":
    main()
