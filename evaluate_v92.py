"""Select deadline-coordinate degree models on historical campaigns only."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from admissions_calendar import DEGREES, load_counts, load_exams
from v92_model import AdmissionsModel, DegreeConfig, DegreeModel

YEARS = (2017, 2018, 2019, 2021, 2022, 2023)
OUTPUT = Path("output/v92")


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidates(degree):
    coordinates = ("calendar",) if degree == "background" else ("event", "calendar")
    exam_options = (False, True) if degree == "bachelor" else (False,)
    levels = (("none", 3), ("mean", 3), ("trend", 3), ("trend", 5))
    result = [
        DegreeConfig(
            family="catboost",
            depth=depth,
            half_life=half,
            coordinates=coord,
            align_history=aligned,
            level_method=method,
            level_window=window,
            exams=exams,
        )
        for depth, half, coord, aligned, (method, window), exams in product(
            (3, 5), (2.0, 6.0), coordinates, (False, True), levels, exam_options
        )
    ]
    result.extend(
        DegreeConfig(
            family="spline",
            knots=knots,
            penalty=penalty,
            half_life=half,
            coordinates=coord,
            align_history=aligned,
            level_method=method,
        )
        for knots, penalty, half, coord, aligned, method in product(
            (7, 11), (0.3, 3.0, 30.0), (2.0, 6.0), coordinates, (False, True), ("none", "mean")
        )
    )
    return result


def main():
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    configs = {d: candidates(d) for d in DEGREES}
    protocol = {
        "validation_years": YEARS,
        "criterion": "pooled recency-weighted log1p MSE, full 73 days",
        "weight_half_life": 3.0,
        "candidates_per_degree": {d: len(c) for d, c in configs.items()},
        "future_observations_allowed": False,
        "previous_forecasts_allowed": False,
        "leaderboard_scores_allowed": False,
        "independent_holdout_available": False,
        "deadline_alignment": "last substantial historical day followed by three low days",
        "selection": "individual initialization, then at most six coordinate passes on total",
    }
    (output / "validation_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "candidate_configs.json").write_text(
        json.dumps({d: [asdict(c) for c in group] for d, group in configs.items()}, indent=2) + "\n"
    )
    counts = load_counts(Path("processed_data/registered.csv"))
    exams = load_exams(Path("external_data/ege/exam_calendar.csv"))
    actual = np.stack([counts.loc[str(y)].to_numpy().T for y in YEARS], axis=1)
    totals = actual.sum(axis=0)
    weight = np.exp2(-(2024 - np.array(YEARS)) / 3.0)
    predictions = {d: np.empty((len(configs[d]), len(YEARS), 73)) for d in DEGREES}

    def criterion(prediction, observed):
        return float(
            np.average(
                ((np.log1p(prediction) - np.log1p(observed)) ** 2).mean(axis=1), weights=weight
            )
        )

    def evaluate(degree, index):
        result = np.stack(
            [
                DegreeModel(degree, configs[degree][index])
                .fit(counts, y, exams)
                .predict()
                .to_numpy()
                for y in YEARS
            ]
        )
        return degree, index, result

    records = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs = [executor.submit(evaluate, d, i) for d in DEGREES for i in range(len(configs[d]))]
        for completed, future in enumerate(as_completed(jobs), 1):
            degree, index, result = future.result()
            predictions[degree][index] = result
            di = DEGREES.index(degree)
            for fold, year in enumerate(YEARS):
                mse = float(np.mean((np.log1p(result[fold]) - np.log1p(actual[di, fold])) ** 2))
                records.append(
                    {
                        "degree": degree,
                        "config_id": index,
                        "year": year,
                        "rmsle": float(np.sqrt(mse)),
                        **asdict(configs[degree][index]),
                    }
                )
            if completed % 12 == 0 or completed == len(jobs):
                pd.DataFrame(records).to_csv(output / "search_metrics.csv", index=False)
                print(f"Historical degree/config combinations {completed}/{len(jobs)}", flush=True)
    np.savez_compressed(output / "search_predictions.npz", **predictions)
    selected = {
        d: int(np.argmin([criterion(p, actual[i]) for p in predictions[d]]))
        for i, d in enumerate(DEGREES)
    }
    initial = selected.copy()

    def total(indices):
        return sum(predictions[d][indices[d]] for d in DEGREES)

    trace = []
    for iteration in range(6):
        changed = False
        for degree in DEGREES:
            scores = [
                criterion(total({**selected, degree: i}), totals)
                for i in range(len(configs[degree]))
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
    pd.DataFrame(trace).to_csv(output / "selection_trace.csv", index=False)
    combined = total(selected)
    selection = {
        "degrees": {d: asdict(configs[d][selected[d]]) for d in DEGREES},
        "selected_indices": selected,
        "individual_initial_indices": initial,
        "pooled_rmsle": float(np.sqrt(criterion(combined, totals))),
        "validation_years": YEARS,
    }
    (output / "selected_config.json").write_text(json.dumps(selection, indent=2) + "\n")
    metrics, daily = [], []
    for fold, year in enumerate(YEARS):
        metrics.append(
            {
                "year": year,
                "rmsle": float(
                    np.sqrt(np.mean((np.log1p(combined[fold]) - np.log1p(totals[fold])) ** 2))
                ),
            }
        )
        daily.append(
            pd.DataFrame(
                {
                    "date": counts.loc[str(year)].index,
                    "forecast": combined[fold],
                    "actual": totals[fold],
                    **{d: predictions[d][selected[d], fold] for d in DEGREES},
                }
            )
        )
    pd.DataFrame(metrics).to_csv(output / "backtest_metrics.csv", index=False)
    pd.concat(daily, ignore_index=True).to_csv(output / "backtest_daily.csv", index=False)
    model = AdmissionsModel({d: configs[d][selected[d]] for d in DEGREES}).fit(counts, 2024, exams)
    model.save(output / "model.joblib")
    forecast = model.predict(components=True)
    destination = Path("forecast_2024_v92.csv")
    forecast[["date", "forecast"]].to_csv(destination, index=False, float_format="%.6f")
    forecast.to_csv(output / "degree_forecasts.csv", index=False, float_format="%.6f")
    sources = [
        Path("admissions_calendar.py"),
        Path("v92_model.py"),
        Path(__file__),
        Path("processed_data/registered.csv"),
        Path("external_data/ege/exam_calendar.csv"),
    ]
    metadata = {
        "model": "V92 degree-specific deadline coordinates",
        "selection": selection,
        "protocol": protocol,
        "training_years": model.models["bachelor"].training_years,
        "previous_forecasts_used": False,
        "leaderboard_scores_used": False,
        "target_year_totals_used": False,
        "public_rmsle": None,
        "sha256": {
            str(p): checksum(p) for p in sources + [destination, output / "selected_config.json"]
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    paths = sources + [
        destination,
        output / "selected_config.json",
        output / "metadata.json",
        output / "model.joblib",
    ]
    freeze = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Fixed before post-hoc reference comparison",
        "sha256": {str(p): checksum(p) for p in paths},
    }
    (output / "selection_frozen.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(pd.DataFrame(metrics).to_string(index=False), flush=True)
    print(json.dumps(selection, indent=2), flush=True)
    print("2024 totals", forecast[list(DEGREES) + ["forecast"]].sum().to_dict(), flush=True)


if __name__ == "__main__":
    main()
