"""Daily admissions forecasting from registration records only.

The selected model directly estimates log(1 + daily registrations). Optional
hurdle architectures are retained to reproduce the comparison experiments.
There is no forecast blend, closure mask, leaderboard input or volume correction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor


CATEGORICAL_FEATURES = ["weekday", "month", "saturday_position"]


@dataclass(frozen=True)
class ModelConfig:
    """Training settings selected by chronological validation."""

    architecture: str = "direct"
    depth: int = 3
    iterations: int = 400
    half_life: float = 5.0
    learning_rate: float = 0.04
    l2_leaf_reg: float = 10.0
    random_seed: int = 68001
    exclude_pandemic: bool = False

    def __post_init__(self) -> None:
        if self.architecture not in {"direct", "hurdle", "hurdle_hard"}:
            raise ValueError("Unknown model architecture.")
        if self.half_life <= 0 or self.iterations < 1 or self.depth < 1:
            raise ValueError("Half-life, iterations and depth must be positive.")


def season_dates(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-06-20", f"{year}-08-31", name="date")


def read_daily_counts(path: Path) -> pd.Series:
    """Count registration rows, retaining zero days within observed seasons."""
    records = pd.read_csv(path, usecols=["reg_ts"])
    dates = pd.to_datetime(records["reg_ts"], errors="raise").dt.normalize()
    if dates.empty or dates.isna().any():
        raise ValueError("Registration timestamps must be present and valid.")
    years = sorted(dates.dt.year.unique())
    if years != list(range(years[0], years[-1] + 1)):
        raise ValueError("The source has a missing season; it cannot mean zero demand.")
    calendar = pd.DatetimeIndex(np.concatenate([season_dates(y).values for y in years]))
    return dates.value_counts().reindex(calendar, fill_value=0).astype(float).rename("count")


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Features available before the campaign, independent of target counts.

    July 25 / July 29 are nominal deadlines, not inferred from future labels.
    The 2021 bachelor exception is documented in ADMISSIONS_MODEL.md.
    The master's nominal deadline moves to the preceding weekday on weekends.
    """
    dates = pd.DatetimeIndex(dates)
    years = dates.year
    starts = pd.to_datetime(years.astype(str) + "-06-20")
    bachelor = pd.to_datetime(years.astype(str) + "-07-25")
    bachelor = bachelor.where(years != 2021, pd.Timestamp("2021-07-29"))
    master = pd.to_datetime(years.astype(str) + "-07-29")
    master -= pd.to_timedelta(np.maximum(master.dayofweek - 4, 0), unit="D")
    progress = (dates - starts).days

    features = pd.DataFrame(index=dates)
    features["campaign_day"] = progress
    features["day_of_month"] = dates.day
    features["week_of_campaign"] = progress // 7
    features["weekday"] = dates.dayofweek.astype(str)
    features["month"] = dates.month.astype(str)
    features["pandemic_season"] = (years == 2020).astype(int)
    features["first_campaign_day"] = (progress == 0).astype(int)
    features["first_week"] = (progress < 7).astype(int)
    features["days_until_end"] = 72 - progress
    features["saturday_position"] = np.select(
        [dates.dayofweek != 5, progress < 7, (dates.month == 8) & (dates.day > 21), dates.month == 8],
        ["not_saturday", "first", "late_august", "august"],
        default="regular",
    )
    for name, deadline in (("bachelor", bachelor), ("master", master)):
        distance = (dates - deadline).days
        features[f"days_from_{name}_deadline"] = distance
        features[f"after_{name}_deadline"] = (distance > 0).astype(int)
        features[f"{name}_deadline_weekday"] = deadline.dayofweek
        features[f"workdays_from_{name}_deadline"] = np.busday_count(
            deadline.values.astype("datetime64[D]"),
            dates.values.astype("datetime64[D]"),
            weekmask="1111110",
        )
    return features


class AdmissionsModel:
    """A fitted daily-count model with an explicit training cutoff."""

    def __init__(self, config: ModelConfig = ModelConfig()) -> None:
        self.config = config
        self.regressor: CatBoostRegressor | None = None
        self.classifier: CatBoostClassifier | None = None
        self.forecast_year: int | None = None

    def fit(self, daily_counts: pd.Series, forecast_year: int) -> AdmissionsModel:
        """Ignore target-year and future observations, even if supplied."""
        history = daily_counts.loc[daily_counts.index.year < forecast_year].sort_index()
        observed_years = sorted(history.index.year.unique())
        if len(observed_years) < 2 or observed_years[-1] != forecast_year - 1:
            raise ValueError("At least two preceding seasons, including the last, are required.")
        if observed_years != list(range(observed_years[0], forecast_year)):
            raise ValueError("Training seasons must be contiguous.")
        expected = pd.DatetimeIndex(np.concatenate([season_dates(y).values for y in observed_years]))
        if not history.index.equals(expected):
            raise ValueError("Each training season must contain all 73 dates, including zeros.")
        if not np.isfinite(history).all() or (history < 0).any():
            raise ValueError("Training counts must be finite and nonnegative.")
        if self.config.exclude_pandemic:
            history = history.loc[history.index.year != 2020]

        features = calendar_features(history.index)
        ages = forecast_year - history.index.year.to_numpy()
        weights = np.exp2(-ages / self.config.half_life)
        positive = history.to_numpy() > 0
        if not positive.any():
            raise ValueError("Training history contains no registrations.")
        parameters = {
            "depth": self.config.depth,
            "iterations": self.config.iterations,
            "learning_rate": self.config.learning_rate,
            "l2_leaf_reg": self.config.l2_leaf_reg,
            "random_seed": self.config.random_seed,
            "cat_features": CATEGORICAL_FEATURES,
            "one_hot_max_size": 16,
            "thread_count": 1,
            "verbose": False,
            "allow_writing_files": False,
        }
        self.classifier = None
        if self.config.architecture != "direct" and not positive.all():
            self.classifier = CatBoostClassifier(loss_function="Logloss", **parameters)
            self.classifier.fit(features, positive.astype(int), sample_weight=weights)

        rows = positive if self.config.architecture != "direct" else np.ones(len(history), dtype=bool)
        self.regressor = CatBoostRegressor(loss_function="RMSE", **parameters)
        self.regressor.fit(features.loc[rows], np.log1p(history.loc[rows]), sample_weight=weights[rows])
        self.forecast_year = forecast_year
        return self

    def predict(self) -> pd.DataFrame:
        """Return one forecast for every day of the target campaign."""
        if self.regressor is None or self.forecast_year is None:
            raise RuntimeError("Call fit() or load() before predict().")
        dates = season_dates(self.forecast_year)
        features = calendar_features(dates)
        log_count = np.maximum(self.regressor.predict(features), 0.0)
        if self.classifier is not None:
            probability = self.classifier.predict_proba(features)[:, 1]
            if self.config.architecture == "hurdle_hard":
                log_count *= probability >= 0.5
            else:
                log_count *= probability
        predictions = np.expm1(log_count)
        if not np.isfinite(predictions).all():
            raise ValueError("The fitted model produced nonfinite predictions.")
        return pd.DataFrame({"date": dates, "forecast": predictions})

    def save(self, directory: Path) -> None:
        if self.regressor is None:
            raise RuntimeError("Cannot save an unfitted model.")
        directory.mkdir(parents=True, exist_ok=True)
        self.regressor.save_model(str(directory / "count_model.cbm"))
        if self.classifier is not None:
            self.classifier.save_model(str(directory / "activity_model.cbm"))
        metadata = {
            "config": asdict(self.config),
            "forecast_year": self.forecast_year,
            "has_classifier": self.classifier is not None,
            "features": list(calendar_features(season_dates(self.forecast_year)).columns),
        }
        (directory / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path) -> AdmissionsModel:
        metadata = json.loads((directory / "model.json").read_text())
        model = cls(ModelConfig(**metadata["config"]))
        model.forecast_year = metadata["forecast_year"]
        model.regressor = CatBoostRegressor()
        model.regressor.load_model(str(directory / "count_model.cbm"))
        if metadata["has_classifier"]:
            model.classifier = CatBoostClassifier()
            model.classifier.load_model(str(directory / "activity_model.cbm"))
        if metadata["features"] != list(calendar_features(season_dates(model.forecast_year)).columns):
            raise ValueError("Saved model and current feature schema differ.")
        return model


def rmsle(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2)))


def write_forecast(forecast: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(path, index=False, date_format="%Y-%m-%d", float_format="%.2f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data", type=Path, default=Path("processed_data/registered.csv"))
    parser.add_argument("--year", type=int, default=2024, help="Target year for training")
    parser.add_argument("--config", type=Path, help="JSON training settings; only used by train")
    parser.add_argument("--model-dir", type=Path, default=Path("output/v68/model"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "train":
        config = ModelConfig(**json.loads(args.config.read_text())) if args.config else ModelConfig()
        model = AdmissionsModel(config).fit(read_daily_counts(args.data), args.year)
        model.save(args.model_dir)
    else:
        model = AdmissionsModel.load(args.model_dir)
    forecast = model.predict()
    output = args.output or Path(f"forecast_{model.forecast_year}_v68_model.csv")
    write_forecast(forecast, output)
    print(f"Saved {len(forecast)} dates to {output}; sum = {forecast.forecast.sum():.2f}")


if __name__ == "__main__":
    main()
