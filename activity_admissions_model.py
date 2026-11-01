"""Two-part daily registration model, fitted from raw registrations.

The first model estimates whether a date has registrations. The second learns
log(1 + count) conditional on activity. Their product estimates the conditional
mean in log space, which is the point forecast appropriate for RMSLE.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from admissions_profile_model import calendar, campaign, load_counts
from joint_admissions_model import training_deadlines


CATEGORIES = ["weekday", "saturday_slot", "phase"]


@dataclass(frozen=True)
class ActivityConfig:
    depth: int = 3
    iterations: int = 300
    half_life: float = 3.0
    activity_half_life: float = 2.0
    activity_iterations: int = 250
    activity_depth: int = 3
    l2_leaf_reg: float = 10.0

    def __post_init__(self):
        if min(self.half_life, self.activity_half_life, self.l2_leaf_reg) <= 0:
            raise ValueError("Decay periods and regularization must be positive.")
        if min(self.depth, self.activity_depth, self.iterations, self.activity_iterations) < 1:
            raise ValueError("Depths and iteration counts must be positive.")


def features(dates: pd.DatetimeIndex, deadlines: dict | None = None) -> pd.DataFrame:
    """Calendar features; inferred past breaks never describe future labels."""
    dates = pd.DatetimeIndex(dates)
    frame = calendar(dates)
    if deadlines:
        for group, name in (("bachelor", "bachelor"), ("master_miep", "master")):
            nominal = dates - pd.to_timedelta(frame[f"days_from_{name}_end"], unit="D")
            ends = pd.DatetimeIndex([
                deadlines.get((int(year), group), date)
                for year, date in zip(dates.year, nominal)
            ])
            frame[f"days_from_{name}_end"] = (dates - ends).days
            frame[f"{name}_end_weekday"] = ends.dayofweek
            frame[f"workdays_from_{name}_end"] = np.busday_count(
                ends.values.astype("datetime64[D]"),
                dates.values.astype("datetime64[D]"),
                weekmask="1111110",
            )
    start = pd.to_datetime(dates.year.astype(str) + "-06-20")
    frame["workdays_from_start"] = np.busday_count(
        start.values.astype("datetime64[D]"), dates.values.astype("datetime64[D]"),
        weekmask="1111110",
    )
    frame["phase"] = np.select(
        [
            frame.day_of_campaign < 4,
            dates.month == 6,
            frame.days_from_bachelor_end <= 0,
            frame.days_from_master_end <= 0,
            (dates.month == 8) & (dates.day > 21),
        ],
        ["start", "ramp", "bachelor", "master", "late"],
        default="post",
    )
    frame["digital_seasons"] = np.maximum(dates.year - 2019, 0)
    frame["week_of_campaign"] = frame.day_of_campaign // 7
    return frame


class ActivityAdmissionsModel:
    def __init__(self, config: ActivityConfig = ActivityConfig()):
        self.config = config
        self.forecast_year = None

    def fit(self, counts: pd.DataFrame, forecast_year: int) -> ActivityAdmissionsModel:
        history = counts.loc[counts.index.year < forecast_year].sort_index()
        years = sorted(history.index.year.unique())
        if len(years) < 2 or years != list(range(years[0], forecast_year)):
            raise ValueError("At least two contiguous preceding seasons are required.")
        expected = pd.DatetimeIndex(np.concatenate([campaign(year).values for year in years]))
        if not history.index.equals(expected):
            raise ValueError("Every training season must include all 73 dates.")
        if not np.isfinite(history.to_numpy()).all() or (history < 0).any().any():
            raise ValueError("Training counts must be finite and nonnegative.")
        # The 2020 campaign was postponed and cannot use the nominal calendar.
        history = history.loc[history.index.year != 2020]
        actual = history.sum(axis=1).to_numpy()
        active = actual > 0
        if not active.any():
            raise ValueError("Training history contains no registrations.")
        self.forecast_year = forecast_year
        self.deadlines = training_deadlines(history)
        design = features(history.index, self.deadlines)
        age = forecast_year - history.index.year.to_numpy()
        common = dict(
            learning_rate=0.035, l2_leaf_reg=self.config.l2_leaf_reg,
            random_seed=72001, thread_count=1, verbose=False,
            allow_writing_files=False, cat_features=CATEGORIES, one_hot_max_size=16,
        )
        self.activity = None
        if not active.all():
            self.activity = CatBoostClassifier(
                depth=self.config.activity_depth, iterations=self.config.activity_iterations,
                loss_function="Logloss", **common,
            )
            self.activity.fit(
                design, active.astype(int),
                sample_weight=np.exp2(-age / self.config.activity_half_life),
            )
        self.magnitude = CatBoostRegressor(
            depth=self.config.depth, iterations=self.config.iterations,
            loss_function="RMSE", **common,
        )
        self.magnitude.fit(
            design.loc[active], np.log1p(actual[active]),
            sample_weight=np.exp2(-age[active] / self.config.half_life),
        )
        return self

    def predict(self, diagnostics: bool = False) -> pd.DataFrame:
        if self.forecast_year is None:
            raise RuntimeError("Fit or load the model before prediction.")
        dates = campaign(self.forecast_year)
        design = features(dates)
        probability = self.activity.predict_proba(design)[:, 1] if self.activity else np.ones(len(dates))
        positive_log_count = np.maximum(self.magnitude.predict(design), 0.0)
        predicted = np.expm1(probability * positive_log_count)
        if not np.isfinite(predicted).all():
            raise ValueError("Nonfinite forecast.")
        result = pd.DataFrame({"date": dates, "forecast": predicted})
        if diagnostics:
            result["activity_probability"] = probability
            result["positive_count"] = np.expm1(positive_log_count)
        return result

    def save(self, directory: Path) -> None:
        if self.forecast_year is None:
            raise RuntimeError("Cannot save an unfitted model.")
        directory.mkdir(parents=True, exist_ok=True)
        self.magnitude.save_model(str(directory / "magnitude.cbm"))
        if self.activity:
            self.activity.save_model(str(directory / "activity.cbm"))
        metadata = {
            "config": asdict(self.config), "forecast_year": self.forecast_year,
            "has_activity_model": self.activity is not None,
            "features": list(features(campaign(self.forecast_year)).columns),
            "historical_phase_breaks": {
                f"{year}:{degree}": str(date.date())
                for (year, degree), date in self.deadlines.items()
            },
        }
        (directory / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path) -> ActivityAdmissionsModel:
        metadata = json.loads((directory / "model.json").read_text())
        model = cls(ActivityConfig(**metadata["config"]))
        model.forecast_year = metadata["forecast_year"]
        model.deadlines = {
            (int(key.split(":")[0]), key.split(":")[1]): pd.Timestamp(date)
            for key, date in metadata["historical_phase_breaks"].items()
        }
        if metadata["features"] != list(features(campaign(model.forecast_year)).columns):
            raise ValueError("Saved feature schema differs from the current implementation.")
        model.magnitude = CatBoostRegressor()
        model.magnitude.load_model(str(directory / "magnitude.cbm"))
        model.activity = None
        if metadata["has_activity_model"]:
            model.activity = CatBoostClassifier()
            model.activity.load_model(str(directory / "activity.cbm"))
        return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data", type=Path, default=Path("processed_data/registered.csv"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("output/v72/model"))
    parser.add_argument("--output", type=Path, default=Path("forecast_2024_v72_activity.csv"))
    args = parser.parse_args()
    if args.command == "train":
        config = ActivityConfig(**json.loads(args.config.read_text())) if args.config else ActivityConfig()
        model = ActivityAdmissionsModel(config).fit(load_counts(args.data), args.year)
        model.save(args.model_dir)
    else:
        model = ActivityAdmissionsModel.load(args.model_dir)
    model.predict().to_csv(args.output, index=False, float_format="%.2f")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
