"""Choose learner hyperparameters against a reference forecast, with provenance.

All candidate regressors fit historical registrations. The reference is used
only to rank candidate hyperparameters. This is reference-guided selection and
does not constitute independent validation or measured 2024 forecast accuracy.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from admissions_calendar import DEGREES, campaign, load_counts, load_exams
from evaluate_v92 import candidates as previous_grid
from v93_model import AdmissionsModel, DegreeConfig, DegreeModel

OUTPUT = Path("output/v93")
REFERENCE = Path("forecast_2024_v84.csv")


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidate_configs(degree):
    rng = np.random.default_rng(93001 + DEGREES.index(degree))
    result = [DegreeConfig(**asdict(c)) for c in previous_grid(degree)]
    for _ in range(220):
        result.append(
            DegreeConfig(
                family="catboost",
                depth=int(rng.choice([2, 3, 4, 5, 6])),
                iterations=int(rng.choice([250, 500, 900])),
                learning_rate=float(rng.choice([0.02, 0.04, 0.08])),
                l2_leaf_reg=float(np.exp(rng.uniform(np.log(0.3), np.log(80)))),
                half_life=float(np.exp(rng.uniform(np.log(0.75), np.log(12)))),
                coordinates=(
                    str(rng.choice(["event", "calendar"])) if degree != "background" else "calendar"
                ),
                align_history=bool(rng.integers(2)),
                level_method=str(rng.choice(["none", "mean", "trend"])),
                level_window=int(rng.choice([2, 3, 5])),
                history_window=int(rng.choice([0, 3, 5, 7])),
                exams=bool(rng.integers(2)) if degree == "bachelor" else False,
                positive_only=bool(rng.integers(2)),
                loss=str(rng.choice(["RMSE", "RMSE", "RMSE", "MAE"])),
            )
        )
    for _ in range(120):
        result.append(
            DegreeConfig(
                family="spline",
                knots=int(rng.choice([5, 7, 9, 13, 17, 21])),
                penalty=float(np.exp(rng.uniform(np.log(0.01), np.log(100)))),
                half_life=float(np.exp(rng.uniform(np.log(0.75), np.log(12)))),
                coordinates=(
                    str(rng.choice(["event", "calendar"])) if degree != "background" else "calendar"
                ),
                align_history=bool(rng.integers(2)),
                level_method=str(rng.choice(["none", "mean", "trend"])),
                level_window=int(rng.choice([2, 3, 5])),
                history_window=int(rng.choice([0, 3, 5, 7])),
                exams=bool(rng.integers(2)) if degree == "bachelor" else False,
                positive_only=bool(rng.integers(2)),
            )
        )
    unique = {json.dumps(asdict(c), sort_keys=True): c for c in result}
    return list(unique.values())


def select_combination(predictions, target, objective="balanced"):
    """Deterministic coordinate selection from eight initial combinations."""
    rng = np.random.default_rng(93999)

    def distance(values):
        error = np.abs(values - target)
        if objective == "balanced":
            error /= np.sqrt(1 + target)
        elif objective != "mae":
            raise ValueError("Unknown comparison objective")
        return error.mean(axis=-1)

    eligible = {
        d: np.flatnonzero(np.all(p[:, target == 0] <= 1e-10, axis=1))
        for d, p in predictions.items()
    }
    if any(len(indices) == 0 for indices in eligible.values()):
        raise ValueError("No candidate reproduces the reference closed-day schedule")
    starts = [{d: int(eligible[d][0]) for d in DEGREES}]
    starts.extend({d: int(rng.choice(eligible[d])) for d in DEGREES} for _ in range(7))
    best, trace = None, []
    for start, selected in enumerate(starts):
        for iteration in range(8):
            previous = selected.copy()
            for degree in DEGREES:
                other = sum(predictions[d][selected[d]] for d in DEGREES if d != degree)
                losses = distance(predictions[degree] + other[None, :])
                invalid = np.ones(len(losses), dtype=bool)
                invalid[eligible[degree]] = False
                losses[invalid] = np.inf
                selected[degree] = int(np.argmin(losses))
            if selected == previous:
                break
        score = float(distance(sum(predictions[d][selected[d]] for d in DEGREES)))
        trace.append({"start": start, "selection_error": score, "indices": selected.copy()})
        if best is None or score < best[0]:
            best = (score, selected.copy())
    # Check coordinated changes of two streams, keeping a beam of 64 pairs.
    # A single-stream search can miss a better pair of hyperparameter settings.
    for iteration in range(2):
        previous_score = best[0]
        for first, second, third in (
            ("bachelor", "master", "background"),
            ("bachelor", "background", "master"),
            ("master", "background", "bachelor"),
        ):
            left, right = eligible[first], eligible[second]
            other = predictions[third][best[1][third]]
            pair_scores = np.stack(
                [distance(predictions[first][i] + predictions[second][right] + other) for i in left]
            )
            size = min(64, pair_scores.size)
            positions = np.argpartition(pair_scores.ravel(), size - 1)[:size]
            for position in positions:
                a, b = np.unravel_index(position, pair_scores.shape)
                pair = predictions[first][left[a]] + predictions[second][right[b]]
                losses = distance(predictions[third][eligible[third]] + pair)
                k = int(np.argmin(losses))
                score = float(losses[k])
                if score < best[0] - 1e-12:
                    best = (
                        score,
                        {
                            first: int(left[a]),
                            second: int(right[b]),
                            third: int(eligible[third][k]),
                        },
                    )
        trace.append(
            {
                "stage": "pair_search",
                "pass": iteration + 1,
                "selection_error": best[0],
                "indices": best[1].copy(),
            }
        )
        if best[0] >= previous_score - 1e-12:
            break
    return best, trace


def refinement_configs(degree, center, round_number):
    rng = np.random.default_rng(94000 + 100 * round_number + DEGREES.index(degree))
    result = []
    for _ in range(180):
        updates = {
            "half_life": float(np.clip(center.half_life * np.exp(rng.normal(0, 0.35)), 0.5, 20)),
            "history_window": (
                int(rng.choice([0, 3, 5, 7])) if rng.random() < 0.12 else center.history_window
            ),
            "level_method": (
                str(rng.choice(["none", "mean", "trend"]))
                if rng.random() < 0.12
                else center.level_method
            ),
            "level_window": (
                int(rng.choice([2, 3, 5])) if rng.random() < 0.2 else center.level_window
            ),
            "align_history": (
                not center.align_history if rng.random() < 0.1 else center.align_history
            ),
            "positive_only": (
                not center.positive_only if rng.random() < 0.1 else center.positive_only
            ),
        }
        if center.family == "catboost":
            updates.update(
                {
                    "depth": int(np.clip(center.depth + rng.choice([-1, 0, 0, 1]), 2, 7)),
                    "iterations": int(
                        np.clip(center.iterations + rng.choice([-200, 0, 200]), 150, 1400)
                    ),
                    "learning_rate": float(
                        np.clip(center.learning_rate * np.exp(rng.normal(0, 0.3)), 0.01, 0.15)
                    ),
                    "l2_leaf_reg": float(
                        np.clip(center.l2_leaf_reg * np.exp(rng.normal(0, 0.7)), 0.1, 100)
                    ),
                    "loss": "MAE" if center.loss == "RMSE" and rng.random() < 0.1 else center.loss,
                }
            )
        else:
            updates.update(
                {
                    "knots": int(np.clip(center.knots + rng.choice([-2, -1, 0, 1, 2]), 3, 25)),
                    "penalty": float(
                        np.clip(center.penalty * np.exp(rng.normal(0, 1)), 0.001, 1000)
                    ),
                }
            )
        result.append(replace(center, **updates))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Reuse fitted candidates and search near the selected settings",
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="Rank cached candidates without additional fitting",
    )
    parser.add_argument("--objective", choices=["balanced", "mae"], default="balanced")
    args = parser.parse_args()
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    counts = load_counts(Path("processed_data/registered.csv"))
    exams = load_exams(Path("external_data/ege/exam_calendar.csv"))
    reference = pd.read_csv(REFERENCE, parse_dates=["date"])
    if (
        not np.array_equal(reference.date, campaign(2024))
        or not np.isfinite(reference.forecast).all()
        or reference.forecast.lt(0).any()
    ):
        raise ValueError("Invalid reference forecast")
    target = reference.forecast.to_numpy()
    previous_sizes = {d: 0 for d in DEGREES}
    cached = None
    round_number = 1
    if args.refine and args.select_only:
        parser.error("Choose either --refine or --select-only")
    if args.refine or args.select_only:
        old_metadata = json.loads((output / "metadata.json").read_text())
        for filename in (
            "v93_model.py",
            "admissions_calendar.py",
            "processed_data/registered.csv",
            "external_data/ege/exam_calendar.csv",
        ):
            if old_metadata["sha256"][filename] != checksum(filename):
                raise ValueError(f"Cached training input changed: {filename}")
        round_number = old_metadata.get("selection_round", 1) + int(args.refine)
        old_configs = json.loads((output / "candidate_configs.json").read_text())
        centers = json.loads((output / "selected_config.json").read_text())["degrees"]
        cached = np.load(output / "candidate_predictions.npz")
        configs = {}
        for degree in DEGREES:
            previous_sizes[degree] = len(old_configs[degree])
            group = [DegreeConfig(**c) for c in old_configs[degree]]
            if args.refine:
                group += refinement_configs(degree, DegreeConfig(**centers[degree]), round_number)
            unique = {json.dumps(asdict(c), sort_keys=True): c for c in group}
            configs[degree] = list(unique.values())
    else:
        configs = {d: candidate_configs(d) for d in DEGREES}
    (output / "candidate_configs.json").write_text(
        json.dumps({d: [asdict(c) for c in group] for d, group in configs.items()}, indent=2) + "\n"
    )
    protocol = {
        "reference_role": "hyperparameter selection only",
        "regressor_targets": "historical registrations only",
        "objective": (
            "mean absolute difference / sqrt(1 + reference), across all 73 dates"
            if args.objective == "balanced"
            else "mean absolute difference, across all 73 dates"
        ),
        "candidates_per_degree": {d: len(c) for d, c in configs.items()},
        "independent_validation": False,
        "teacher_independent": False,
        "coefficient_transfer_from_teacher": False,
        "closed_day_pattern": "candidate filter; no date-specific rules transferred to runtime",
        "combination_search": "eight coordinate starts, then two pair-search passes with beam 64",
    }
    (output / "selection_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    predictions = {d: np.empty((len(c), 73)) for d, c in configs.items()}
    if cached is not None:
        for degree in DEGREES:
            predictions[degree][: previous_sizes[degree]] = cached[degree]

    def evaluate(degree, index):
        model = DegreeModel(degree, configs[degree][index]).fit(counts, 2024, exams)
        return degree, index, model.predict().to_numpy()

    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs = [
            executor.submit(evaluate, d, i)
            for d in DEGREES
            for i in range(previous_sizes[d], len(configs[d]))
        ]
        for completed, future in enumerate(as_completed(jobs), 1):
            degree, index, values = future.result()
            predictions[degree][index] = values
            if completed % 60 == 0 or completed == len(jobs):
                print(f"Candidates fitted on history: {completed}/{len(jobs)}", flush=True)
    np.savez_compressed(output / "candidate_predictions.npz", **predictions)
    (score, selected), trace = select_combination(predictions, target, args.objective)
    (output / "selection_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    chosen = {d: configs[d][selected[d]] for d in DEGREES}
    (output / "selected_config.json").write_text(
        json.dumps({"degrees": {d: asdict(c) for d, c in chosen.items()}}, indent=2) + "\n"
    )
    model = AdmissionsModel(chosen).fit(counts, 2024, exams)
    model.save(output / "model.joblib")
    final = model.predict(components=True)
    np.testing.assert_allclose(
        final.forecast, sum(predictions[d][selected[d]] for d in DEGREES), rtol=0, atol=1e-10
    )
    destination = Path("forecast_2024_v93.csv")
    final[["date", "forecast"]].to_csv(destination, index=False, float_format="%.6f")
    final.to_csv(output / "degree_forecasts.csv", index=False, float_format="%.6f")
    comparison = final.copy()
    comparison["reference"] = target
    comparison["difference"] = comparison.forecast - target
    comparison.to_csv(output / "reference_comparison.csv", index=False, float_format="%.6f")
    positive = target > 0
    summary = {
        "selection_error": score,
        "mean_absolute_difference_per_day": float(
            np.abs(final.forecast.to_numpy() - target).mean()
        ),
        "mean_absolute_percent_difference_positive_days": float(
            np.mean(
                100
                * np.abs(final.forecast.to_numpy()[positive] - target[positive])
                / target[positive]
            )
        ),
        "total": float(final.forecast.sum()),
        "reference_total": float(target.sum()),
        "same_zero_dates": bool(np.array_equal(final.forecast.eq(0), target == 0)),
        "public_rmsle": None,
    }
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sources = [
        Path("v93_model.py"),
        Path("admissions_calendar.py"),
        Path(__file__),
        Path("processed_data/registered.csv"),
        Path("external_data/ege/exam_calendar.csv"),
    ]
    metadata = {
        "model": "V93 historical regressors with reference-selected hyperparameters",
        "selection_round": round_number,
        "selection_protocol": protocol,
        "selected_indices": selected,
        "reference": {"file": str(REFERENCE), "sha256": checksum(REFERENCE)},
        "reference_used_for_hyperparameter_selection": True,
        "teacher_targets_used_in_regressor_fit": False,
        "independent_of_reference": False,
        "public_rmsle": None,
        "sha256": {
            str(p): checksum(p) for p in sources + [destination, output / "selected_config.json"]
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(
        "Selected configurations:",
        json.dumps({d: asdict(c) for d, c in chosen.items()}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
