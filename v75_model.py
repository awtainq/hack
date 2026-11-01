"""Admissions forecasting with chronological, out-of-fold calibration.

A shared CatBoost learns total, budget-associated and other registrations.
The readout learns stream reconciliation and optional seasonal corrections
from genuine earlier-year forecast errors. No submission CSV is a label.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler


TARGETS = ["total", "budget", "other"]
CATEGORIES = ["weekday", "phase", "saturday_slot"]


@dataclass(frozen=True)
class ForecastConfig:
    depth: int = 3
    iterations: int = 500
    half_life: float = 3.0
    calibration: str = "none"
    ridge_penalty: float = 100.0

    def __post_init__(self):
        if self.depth < 1 or self.iterations < 1 or self.half_life <= 0 or self.ridge_penalty <= 0:
            raise ValueError("Invalid training parameters.")
        if self.calibration not in {"none", "global", "shape"}:
            raise ValueError("Unknown calibration mode.")


def campaign(year: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-06-20", f"{year}-08-31", name="date")


def load_counts(directory: Path) -> pd.DataFrame:
    """Partition each registration once; unmatched rows belong to 'other'."""
    registered = pd.read_csv(directory / "registered.csv")
    included = pd.read_csv(directory / "included_ks.csv")
    for frame in (registered, included):
        frame["date"] = pd.to_datetime(frame.reg_ts, errors="raise").dt.normalize()
        if frame.date.isna().any():
            raise ValueError("Missing registration dates.")
    keys = ["abitur_kod", "date", "zayav_type_kod"]
    included["budget"] = included.ob_osnova_name.isin(["Бюджет", "Гослиния"])
    mapping = included.groupby(keys).budget.max()
    rows = registered.merge(mapping, on=keys, how="left", validate="many_to_one")
    if len(rows) != len(registered):
        raise ValueError("Funding join duplicated registration rows.")
    rows["budget"] = rows.budget.eq(True).astype(float)
    rows["other"] = 1.0 - rows.budget
    rows["total"] = 1.0
    rows["bachelor"] = rows.zayav_type_name.str.startswith("Бакалавр").astype(float)
    rows["master"] = rows.zayav_type_name.str.startswith("Магистр").astype(float)
    years = sorted(rows.date.dt.year.unique())
    if not years or years != list(range(years[0], years[-1] + 1)):
        raise ValueError("Source seasons must be contiguous.")
    dates = pd.DatetimeIndex(np.concatenate([campaign(year).values for year in years]), name="date")
    counts = rows.groupby("date")[TARGETS + ["bachelor", "master"]].sum()
    return counts.reindex(dates, fill_value=0)


def historical_deadlines(history: pd.DataFrame) -> dict:
    """Find past activity breaks; this function receives training data only."""
    result = {}
    for year in sorted(history.index.year.unique()):
        for degree, share in (("bachelor", 0.20), ("master", 0.25)):
            series = history.loc[history.index.year == year, degree]
            threshold = share * series.loc[f"{year}-07-01":f"{year}-07-25"].max()
            if threshold <= 0:
                continue
            for date, count in series.loc[f"{year}-07-20":f"{year}-08-20"].items():
                after = series.loc[date + pd.Timedelta(days=1):date + pd.Timedelta(days=3)]
                if count >= threshold and len(after) == 3 and (after < threshold).all():
                    result[int(year), degree] = date
                    break
    return result


def calendar_features(dates, deadlines=None):
    dates = pd.DatetimeIndex(dates)
    starts = pd.to_datetime(dates.year.astype(str) + "-06-20")
    progress = (dates - starts).days
    frame = pd.DataFrame(index=dates)
    frame["day_of_campaign"] = progress
    frame["day"] = dates.day
    frame["month"] = dates.month
    frame["week"] = progress // 7
    frame["weekday"] = dates.dayofweek.astype(str)
    frame["first_day"] = (progress == 0).astype(int)
    frame["first_week"] = (progress < 4).astype(int)
    frame["sin_progress"] = np.sin(np.pi * progress / 72)
    frame["cos_progress"] = np.cos(np.pi * progress / 72)
    frame["sin_double_progress"] = np.sin(2 * np.pi * progress / 72)
    frame["workday"] = np.busday_count(
        starts.values.astype("datetime64[D]"), dates.values.astype("datetime64[D]"), weekmask="1111110",
    )
    for degree, month_day in (("bachelor", "07-25"), ("master", "07-29")):
        nominal = pd.to_datetime(dates.year.astype(str) + "-" + month_day)
        if degree == "master":
            nominal -= pd.to_timedelta(np.maximum(nominal.dayofweek - 4, 0), unit="D")
        else:
            nominal = nominal.where(dates.year != 2021, pd.Timestamp("2021-07-29"))
        ends = pd.DatetimeIndex([
            (deadlines or {}).get((int(year), degree), end)
            for year, end in zip(dates.year, nominal)
        ])
        distance = (dates - ends).days
        frame[f"distance_{degree}"] = distance
        frame[f"after_{degree}"] = (distance > 0).astype(int)
        frame[f"weekday_end_{degree}"] = ends.dayofweek
        frame[f"workdays_{degree}"] = np.busday_count(
            ends.values.astype("datetime64[D]"), dates.values.astype("datetime64[D]"), weekmask="1111110",
        )
    frame["phase"] = np.select(
        [progress < 4, dates.month == 6, frame.distance_bachelor <= 0,
         frame.distance_master <= 0, (dates.month == 8) & (dates.day > 21)],
        ["start", "ramp", "bachelor", "master", "late"], default="post",
    )
    frame["saturday_slot"] = np.select(
        [dates.dayofweek != 5, progress < 7,
         (dates.month == 8) & (dates.day > 21), dates.month == 8],
        ["none", "first", "late", "august"], default="regular",
    )
    return frame


class CalendarHeads:
    """One shared tree ensemble with three supervised output heads."""

    def __init__(self, config: ForecastConfig):
        self.config = config

    def fit(self, history: pd.DataFrame, year: int):
        self.year = year
        self.deadlines = historical_deadlines(history)
        frame = calendar_features(history.index, self.deadlines)
        weight = np.exp2(-(year - history.index.year.to_numpy()) / self.config.half_life)
        weight *= np.where(history.index.year == 2020, 0.1, 1.0)
        self.closure_rates = {}
        for slot in ("first", "late", "august", "regular"):
            mask = frame.saturday_slot.eq(slot) & (history.index.year != 2020)
            self.closure_rates[slot] = float(np.average(history.total[mask].eq(0), weights=weight[mask]))
        use = history.index.dayofweek != 6
        target = np.log1p(history.loc[use, TARGETS].to_numpy())
        self.regressor = CatBoostRegressor(
            loss_function="MultiRMSE", depth=self.config.depth, iterations=self.config.iterations,
            learning_rate=0.035, l2_leaf_reg=10.0, random_seed=75001,
            cat_features=CATEGORIES, one_hot_max_size=16, thread_count=1,
            verbose=False, allow_writing_files=False,
        )
        self.regressor.fit(frame.loc[use], target, sample_weight=weight[use])
        return self

    def predict(self):
        dates = campaign(self.year)
        frame = calendar_features(dates)
        heads = np.maximum(self.regressor.predict(frame), 0.0)
        stream_sum = np.log1p(np.expm1(heads[:, 1:]).sum(axis=1))
        closed = (dates.dayofweek == 6) | frame.saturday_slot.map(self.closure_rates).gt(0.5)
        return pd.DataFrame({
            "pooled_log": heads[:, 0], "stream_log": stream_sum,
            "budget_log": heads[:, 1], "other_log": heads[:, 2], "closed": np.asarray(closed),
        }, index=dates)


class AdmissionsForecaster:
    """A complete predictor with an explicit cutoff and historical calibration."""

    def __init__(self, config=ForecastConfig()):
        self.config = config
        self.year = None

    def calibration_features(self, predictions, fit=False):
        direction = (predictions.stream_log - predictions.pooled_log).to_numpy()[:, None]
        if self.config.calibration == "global":
            return direction
        dates = predictions.index
        position = ((dates - pd.to_datetime(dates.year.astype(str) + "-06-20")).days / 72).to_numpy()[:, None]
        if fit:
            self.spline = SplineTransformer(n_knots=4, degree=3, include_bias=False)
            smooth = self.spline.fit_transform(position)
        else:
            smooth = self.spline.transform(position)
        return np.column_stack([direction, smooth, (dates.dayofweek == 5).astype(float)])

    def fit(self, counts, forecast_year, cache=None):
        history = counts.loc[counts.index.year < forecast_year].sort_index().copy()
        years = sorted(history.index.year.unique())
        if len(years) < 3 or years != list(range(years[0], forecast_year)):
            raise ValueError("At least three contiguous preceding seasons are required.")
        expected = pd.DatetimeIndex(np.concatenate([campaign(year).values for year in years]), name="date")
        if not history.index.equals(expected):
            raise ValueError("Every training season must include all 73 dates.")
        if not np.isfinite(history.to_numpy()).all() or (history < 0).any().any():
            raise ValueError("Counts must be finite and nonnegative.")
        if not np.allclose(history.total, history.budget + history.other):
            raise ValueError("Funding partition does not sum to total registrations.")
        self.year = forecast_year
        cache = {} if cache is None else cache

        def fit_heads(year):
            past = history.loc[history.index.year < year]
            digest = hashlib.sha256(pd.util.hash_pandas_object(past, index=True).values.tobytes()).hexdigest()
            key = (year, self.config.depth, self.config.iterations, self.config.half_life, digest)
            if key not in cache:
                fitted = CalendarHeads(self.config).fit(past, year)
                cache[key] = fitted, fitted.predict()
            return cache[key]

        self.calibrator = None
        self.calibration_years = []
        if self.config.calibration != "none":
            folds = []
            for year in years[2:]:
                if year == 2020:
                    continue
                _, predicted = fit_heads(year)
                predicted = predicted.copy()
                predicted["actual_log"] = np.log1p(history.loc[predicted.index, "total"])
                folds.append(predicted)
                self.calibration_years.append(int(year))
            calibration = pd.concat(folds)
            # The deployment mask is known from each fold's earlier history.
            calibration = calibration.loc[~calibration.closed]
            design = self.calibration_features(calibration, fit=True)
            self.scaler = StandardScaler().fit(design)
            weights = np.exp2(-(forecast_year - calibration.index.year.to_numpy()) / self.config.half_life)
            self.calibrator = Ridge(alpha=self.config.ridge_penalty)
            self.calibrator.fit(
                self.scaler.transform(design), calibration.actual_log - calibration.stream_log,
                sample_weight=weights,
            )
            self.calibration_rows = len(calibration)
        self.heads, _ = fit_heads(forecast_year)
        return self

    def predict(self, diagnostics=False):
        if self.year is None:
            raise RuntimeError("Fit or load the model before prediction.")
        prediction = self.heads.predict()
        correction = np.zeros(len(prediction))
        if self.calibrator is not None:
            design = self.scaler.transform(self.calibration_features(prediction))
            correction = self.calibrator.predict(design)
        log_count = np.maximum(prediction.stream_log.to_numpy() + correction, 0.0)
        log_count[prediction.closed.to_numpy()] = 0.0
        result = pd.DataFrame({"date": prediction.index, "forecast": np.expm1(log_count)})
        if not np.isfinite(result.forecast).all():
            raise ValueError("Nonfinite forecast.")
        if diagnostics:
            result["uncalibrated_forecast"] = np.expm1(prediction.stream_log.to_numpy())
            result["log_correction"] = correction
            result["closed"] = prediction.closed.to_numpy()
        return result

    def save(self, path):
        if self.year is None:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1, "config": asdict(self.config), "year": self.year,
            "regressor": self.heads.regressor, "closure_rates": self.heads.closure_rates,
            "deadlines": self.heads.deadlines, "calibrator": self.calibrator,
            "calibration_years": self.calibration_years,
            "calibration_rows": getattr(self, "calibration_rows", 0),
            "scaler": getattr(self, "scaler", None), "spline": getattr(self, "spline", None),
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        if payload.get("schema_version") != 1:
            raise ValueError("Unsupported saved model schema.")
        model = cls(ForecastConfig(**payload["config"]))
        model.year = payload["year"]
        model.heads = CalendarHeads(model.config)
        model.heads.year = model.year
        for name in ("regressor", "closure_rates", "deadlines"):
            setattr(model.heads, name, payload[name])
        for name in ("calibrator", "calibration_years", "calibration_rows", "scaler", "spline"):
            setattr(model, name, payload[name])
        return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data", type=Path, default=Path("processed_data"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path, default=Path("output/v75/model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("forecast_2024_v75_clean.csv"))
    args = parser.parse_args()
    if args.command == "train":
        config = ForecastConfig(**json.loads(args.config.read_text())) if args.config else ForecastConfig()
        model = AdmissionsForecaster(config).fit(load_counts(args.data), args.year)
        model.save(args.model)
    else:
        model = AdmissionsForecaster.load(args.model)
    model.predict().to_csv(args.output, index=False, float_format="%.2f")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
