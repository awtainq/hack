"""Select V90 on preceding seasons only; no reference-forecast inputs."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v90_model import (
    AdmissionsModel,
    DEGREES,
    DegreeModel,
    ModelConfig,
    degree_targets,
    fit_global_scale,
    load_counts,
    load_exam_calendar,
)

YEARS = (2017, 2018, 2019, 2021, 2022, 2023)
DESTINATION = Path("output/v90")


def rmsle(prediction, actual):
    return float(np.sqrt(np.mean((np.log1p(prediction) - np.log1p(actual)) ** 2)))


def main():
    output = DESTINATION
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "validation_years": YEARS,
        "criterion": "recency-weighted mean of full 73-day RMSLE",
        "validation_weight_half_life": 3.0,
        "config_selection": "individual initialization, then up to four coordinate passes on total",
        "calibration_selection": "none vs multiplier fitted on strictly earlier rolling forecasts",
        "independent_holdout_available": False,
        "previous_forecasts_allowed": False,
        "leaderboard_scores_allowed": False,
        "target_campaign_totals_allowed": False,
    }
    (output / "validation_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    counts = load_counts(Path("processed_data/registered.csv"))
    exams = load_exam_calendar(Path("external_data/ege/exam_calendar.csv"))
    configs = [
        ModelConfig(
            depth=depth,
            iterations=iterations,
            half_life=half,
            feature_set=features,
            align_history=aligned,
        )
        for depth, iterations, half, features, aligned in product(
            (3, 5), (300, 700), (2.0, 6.0), ("calendar", "annual", "exams"), (False, True)
        )
    ]
    configs.extend(
        ModelConfig(
            family=family,
            iterations=192,
            half_life=half,
            feature_set=features,
            align_history=aligned,
            min_leaf=leaf,
        )
        for family, leaf, half, features, aligned in product(
            ("forest", "extra"), (3, 8), (2.0, 6.0), ("calendar", "exams"), (False, True)
        )
    )
    # Shape: configuration, degree, validation season, date within season.
    predictions = np.empty((len(configs), len(DEGREES), len(YEARS), 73))
    actual_degrees = np.stack(
        [degree_targets(counts.loc[str(y)]).to_numpy().T for y in YEARS], axis=1
    )
    actual_total = np.stack([counts.loc[str(y), "total"].to_numpy() for y in YEARS])
    weights = np.exp2(-(2024 - np.asarray(YEARS)) / 3.0)
    metrics = []

    def evaluate(index, degree_index):
        degree, config = DEGREES[degree_index], configs[index]
        result = []
        for year in YEARS:
            model = DegreeModel(degree, config).fit(counts, year, exams)
            result.append(model.predict().to_numpy())
        return index, degree_index, np.stack(result)

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
                metrics.append(
                    {
                        "config_id": index,
                        "degree": DEGREES[degree],
                        "year": year,
                        "rmsle": rmsle(values[fold], actual_degrees[degree, fold]),
                        **asdict(configs[index]),
                    }
                )
            if completed % 12 == 0 or completed == len(jobs):
                pd.DataFrame(metrics).to_csv(output / "search_metrics.csv", index=False)
                print(
                    f"Historical fits: {completed}/{len(jobs)} degree/config combinations",
                    flush=True,
                )
    np.savez_compressed(output / "search_predictions.npz", predictions=predictions)
    (output / "candidate_configs.json").write_text(
        json.dumps([asdict(c) for c in configs], indent=2) + "\n"
    )
    search = pd.DataFrame(metrics)
    selected = {}
    for degree_index, degree in enumerate(DEGREES):
        ranking = (
            search.loc[search.degree == degree]
            .groupby("config_id", sort=True)
            .apply(
                lambda part: np.average(part.sort_values("year").rmsle, weights=weights),
                include_groups=False,
            )
        )
        selected[degree] = int(ranking.idxmin())
    initial = selected.copy()

    def combined(indices):
        return sum(predictions[indices[d], i] for i, d in enumerate(DEGREES))

    def criterion(values):
        return float(
            np.average([rmsle(p, a) for p, a in zip(values, actual_total)], weights=weights)
        )

    trace = []
    for iteration in range(4):
        changed = False
        for degree in DEGREES:
            scores = [
                criterion(combined({**selected, degree: index})) for index in range(len(configs))
            ]
            best = int(np.argmin(scores))
            changed |= best != selected[degree]
            selected[degree] = best
            trace.append(
                {
                    "pass": iteration + 1,
                    "degree": degree,
                    "config_id": best,
                    "criterion": scores[best],
                }
            )
        if not changed:
            break
    pd.DataFrame(trace).to_csv(output / "joint_selection_trace.csv", index=False)
    raw = combined(selected)
    calibrated, factors = [], []
    for fold, year in enumerate(YEARS):
        scale = 1.0
        if fold:
            earlier_weights = np.repeat(np.exp2(-(year - np.asarray(YEARS[:fold])) / 3.0), 73)
            scale = fit_global_scale(
                raw[:fold].ravel(), actual_total[:fold].ravel(), earlier_weights
            )
        factors.append(scale)
        calibrated.append(raw[fold] * scale)
    calibrated = np.stack(calibrated)
    raw_comparable = np.average(
        [rmsle(p, a) for p, a in zip(raw[1:], actual_total[1:])], weights=weights[1:]
    )
    calibrated_comparable = np.average(
        [rmsle(p, a) for p, a in zip(calibrated[1:], actual_total[1:])], weights=weights[1:]
    )
    calibration = "historical" if calibrated_comparable < raw_comparable else "none"
    selection = {
        "degrees": {d: asdict(configs[selected[d]]) for d in DEGREES},
        "selected_indices": selected,
        "individual_initial_indices": initial,
        "validation_years": YEARS,
        "calibration": calibration,
        "uncalibrated_criterion": criterion(raw),
        "calibration_comparable_criterion": {
            "none": float(raw_comparable),
            "historical": float(calibrated_comparable),
        },
    }
    (output / "selected_config.json").write_text(json.dumps(selection, indent=2) + "\n")
    chosen = AdmissionsModel({d: configs[selected[d]] for d in DEGREES}).fit(counts, 2024, exams)
    daily, backtests = [], []
    for fold, year in enumerate(YEARS):
        dates = counts.loc[str(year)].index
        part = pd.DataFrame(
            {
                "date": dates,
                "forecast": raw[fold],
                "actual": actual_total[fold],
                **{d: predictions[selected[d], i, fold] for i, d in enumerate(DEGREES)},
            }
        )
        daily.append(part)
        backtests.append(
            {
                "year": year,
                "rmsle_raw": rmsle(raw[fold], actual_total[fold]),
                "historical_scale": factors[fold],
                "rmsle_with_past_calibration": rmsle(calibrated[fold], actual_total[fold]),
                "rmsle_selected": rmsle(
                    calibrated[fold] if calibration == "historical" else raw[fold],
                    actual_total[fold],
                ),
            }
        )
    historical = pd.concat(daily, ignore_index=True)
    historical.to_csv(output / "backtest_daily.csv", index=False)
    pd.DataFrame(backtests).to_csv(output / "backtest_metrics.csv", index=False)
    if calibration == "historical":
        chosen.calibrate(historical)
    chosen.save(output / "model.joblib")
    destination = Path("forecast_2024_v90.csv")
    final = chosen.predict(components=True)
    final[["date", "forecast"]].to_csv(destination, index=False, float_format="%.6f")
    final.to_csv(output / "degree_forecasts.csv", index=False, float_format="%.6f")
    metadata = {
        "model": "V90 independently trained degree models",
        "forecast_year": 2024,
        "selection": selection,
        "training_years": chosen.models["bachelor"].training_years,
        "scale": chosen.scale,
        "scale_provenance": chosen.scale_provenance,
        "validation_protocol": protocol,
        "previous_forecasts_used": False,
        "leaderboard_scores_used": False,
        "target_campaign_totals_used": False,
        "public_rmsle": None,
        "files_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path("v90_model.py"),
                Path(__file__),
                Path("v88_model.py"),
                Path("processed_data/registered.csv"),
                Path("external_data/ege/exam_calendar.csv"),
                output / "selected_config.json",
                destination,
            )
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(pd.DataFrame(backtests).to_string(index=False), flush=True)
    print("Selection:", json.dumps(selection, indent=2), flush=True)
    print("2024 totals:", final[list(DEGREES) + ["forecast"]].sum().to_dict(), flush=True)
    print("Historical multiplier:", chosen.scale, flush=True)


if __name__ == "__main__":
    main()
