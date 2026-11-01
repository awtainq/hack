"""Train a daily admissions forecast using historical observations only.

The base model jointly predicts total, education-level and funding counts.
A two-parameter ridge readout learns how to reconcile them from genuine
earlier-season out-of-fold predictions. Public scores are never inputs.
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


TARGETS = ("total", "bachelor", "master", "college", "other_degree", "budget", "other_funding")
CATEGORIES = ("weekday", "phase", "saturday_slot")


@dataclass(frozen=True)
class ModelConfig:
    depth: int = 3
    iterations: int = 400
    half_life: float = 3.0
    ridge_penalty: float | None = 10.0

    def __post_init__(self):
        if self.depth < 1 or self.iterations < 1 or self.half_life <= 0:
            raise ValueError("Invalid model configuration")
        if self.ridge_penalty is not None and self.ridge_penalty <= 0:
            raise ValueError("Ridge penalty must be positive")


def campaign(year):
    return pd.date_range(f"{year}-06-20", f"{year}-08-31", name="date")


def load_daily(directory):
    directory = Path(directory)
    keys = ["abitur_kod", "date", "zayav_type_kod"]
    records = pd.read_csv(directory / "registered.csv", usecols=["abitur_kod", "reg_ts", "zayav_type_kod", "zayav_type_name"])
    funding = pd.read_csv(directory / "included_ks.csv", usecols=["abitur_kod", "reg_ts", "zayav_type_kod", "ob_osnova_name"])
    for frame in (records, funding):
        frame["date"] = pd.to_datetime(frame.reg_ts, errors="raise").dt.normalize()
        if frame.date.isna().any() or frame[keys].isna().any().any():
            raise ValueError("Missing registration date or matching key")
    if records.zayav_type_name.isna().any():
        raise ValueError("Missing degree")
    funding["budget"] = funding.ob_osnova_name.isin(["Бюджет", "Гослиния"])
    mapping = funding.groupby(keys).budget.max()
    rows = records.merge(mapping, on=keys, how="left", validate="many_to_one")
    rows["budget"] = rows.budget.eq(True).astype(float)
    # 'Other' includes unmatched registrations; it is not asserted to be contract.
    rows["other_funding"] = 1.0 - rows.budget
    degree = np.select([rows.zayav_type_name.str.startswith(p) for p in ("Бакалавр", "Магистр", "СПО")], TARGETS[1:4], default="other_degree")
    for name in TARGETS[1:5]:
        rows[name] = (degree == name).astype(float)
    rows["total"] = 1.0
    years = sorted(rows.date.dt.year.unique())
    if not years or years != list(range(years[0], years[-1] + 1)):
        raise ValueError("Historical years must be contiguous")
    index = pd.DatetimeIndex(np.concatenate([campaign(year).values for year in years]), name="date")
    return rows.groupby("date")[list(TARGETS)].sum().reindex(index, fill_value=0)


def past_deadlines(history):
    """Estimate past activity breaks using only observations before the cutoff."""
    result = {}
    for year in history.index.year.unique():
        for degree, fraction in (("bachelor", 0.20), ("master", 0.25)):
            counts = history.loc[history.index.year == year, degree]
            threshold = fraction * counts.loc[f"{year}-07-01":f"{year}-07-25"].max()
            if threshold <= 0:
                continue
            for date, value in counts.loc[f"{year}-07-20":f"{year}-08-20"].items():
                after = counts.loc[date + pd.Timedelta(days=1):date + pd.Timedelta(days=3)]
                if value >= threshold and len(after) == 3 and (after < threshold).all():
                    result[int(year), degree] = date
                    break
    return result


def features(dates, deadlines=None):
    dates = pd.DatetimeIndex(dates)
    start = pd.to_datetime(dates.year.astype(str) + "-06-20")
    position = (dates - start).days
    frame = pd.DataFrame({"position": position, "day": dates.day, "month": dates.month,
                          "weekday": dates.dayofweek.astype(str), "first_day": (position == 0).astype(int)}, index=dates)
    frame["workdays_from_start"] = np.busday_count(start.values.astype("datetime64[D]"), dates.values.astype("datetime64[D]"), weekmask="1111110")
    for degree, month_day in (("bachelor", "07-25"), ("master", "07-29")):
        nominal = pd.to_datetime(dates.year.astype(str) + "-" + month_day)
        if degree == "master":
            nominal -= pd.to_timedelta(np.maximum(nominal.dayofweek - 4, 0), unit="D")
        else:
            nominal = nominal.where(dates.year != 2021, pd.Timestamp("2021-07-29"))
        ends = pd.DatetimeIndex([(deadlines or {}).get((int(year), degree), end) for year, end in zip(dates.year, nominal)])
        frame[f"distance_{degree}"] = (dates - ends).days
        frame[f"after_{degree}"] = ((dates - ends).days > 0).astype(int)
        frame[f"end_weekday_{degree}"] = ends.dayofweek
    frame["phase"] = np.select([position < 4, dates.month == 6, frame.distance_bachelor <= 0,
                                 frame.distance_master <= 0], ["start", "june", "bachelor", "master"], default="tail")
    frame["saturday_slot"] = np.select([dates.dayofweek != 5, position < 7,
                                         (dates.month == 8) & (dates.day > 21), dates.month == 8],
                                        ["none", "first", "late", "august"], default="regular")
    return frame


def validate_history(counts, year):
    history = counts.loc[counts.index.year < year].sort_index().copy()
    years = sorted(history.index.year.unique())
    if not years or years != list(range(years[0], year)):
        raise ValueError("Need contiguous preceding seasons")
    expected = pd.DatetimeIndex(np.concatenate([campaign(y).values for y in years]), name="date")
    if not history.index.equals(expected) or not np.isfinite(history.to_numpy()).all() or (history < 0).any().any():
        raise ValueError("Invalid or incomplete daily counts")
    if not np.allclose(history.total, history[list(TARGETS[1:5])].sum(axis=1)) or not np.allclose(history.total, history[list(TARGETS[5:])].sum(axis=1)):
        raise ValueError("Count partitions do not reconcile")
    return history


class HistoricalHeads:
    def __init__(self, config):
        self.config = config

    def fit(self, history, year):
        self.year = int(year)
        history = history.loc[(history.index.year < year) & (history.index.year != 2020)]
        frame = features(history.index, past_deadlines(history))
        weights = np.exp2(-(year - history.index.year.to_numpy()) / self.config.half_life)
        self.closure_rates = {}
        for slot in ("first", "late", "august", "regular"):
            use = frame.saturday_slot.eq(slot)
            self.closure_rates[slot] = float(np.average(history.total[use].eq(0), weights=weights[use]))
        use = history.index.dayofweek != 6
        self.regressor = CatBoostRegressor(
            loss_function="MultiRMSE", depth=self.config.depth, iterations=self.config.iterations,
            learning_rate=0.035, l2_leaf_reg=10, cat_features=list(CATEGORIES), one_hot_max_size=16,
            random_seed=85001, thread_count=1, verbose=False, allow_writing_files=False,
        )
        self.regressor.fit(frame.loc[use], np.log1p(history.loc[use, list(TARGETS)]), sample_weight=weights[use])
        return self

    def predict(self):
        dates = campaign(self.year)
        frame = features(dates)
        heads = np.maximum(0.0, self.regressor.predict(frame))
        closed = (dates.dayofweek == 6) | frame.saturday_slot.map(self.closure_rates).gt(0.5)
        return pd.DataFrame({
            "pooled_log": heads[:, 0],
            "degree_log": np.log1p(np.expm1(heads[:, 1:5]).sum(axis=1)),
            "funding_log": np.log1p(np.expm1(heads[:, 5:7]).sum(axis=1)),
            "closed": np.asarray(closed),
        }, index=dates)


def reconciliation_directions(prediction):
    return np.column_stack([prediction.degree_log - prediction.pooled_log,
                            prediction.funding_log - prediction.pooled_log])


class HistoricalForecaster:
    """Fit a base model and reconcile streams using earlier OOF forecast errors."""
    def __init__(self, config=ModelConfig()):
        self.config = config

    def fit(self, counts, year, cache=None):
        history = validate_history(counts, year)
        self.year = int(year)
        cache = {} if cache is None else cache
        self.training_years = sorted(set(map(int, history.index.year)) - {2020})

        def fitted_heads(cutoff):
            past = history.loc[history.index.year < cutoff]
            data_hash = hashlib.sha256(pd.util.hash_pandas_object(past, index=True).values.tobytes()).hexdigest()
            key = (self.config.depth, self.config.iterations, self.config.half_life, cutoff, data_hash)
            if key not in cache:
                cache[key] = HistoricalHeads(self.config).fit(past, cutoff)
            return cache[key]

        self.heads = fitted_heads(year)
        self.coefficients = np.zeros(2)
        self.calibration_years = []
        if self.config.ridge_penalty is not None:
            designs, errors, weights = [], [], []
            for past_year in range(self.training_years[0] + 2, year):
                if past_year == 2020:
                    continue
                prediction = fitted_heads(past_year).predict()
                use = ~prediction.closed.to_numpy()
                designs.append(reconciliation_directions(prediction)[use])
                observed = history.loc[prediction.index, "total"].to_numpy()
                errors.append((np.log1p(observed) - prediction.pooled_log.to_numpy())[use])
                weights.append(np.full(use.sum(), np.exp2(-(year - past_year) / self.config.half_life)))
                self.calibration_years.append(past_year)
            if designs:
                readout = Ridge(alpha=self.config.ridge_penalty, fit_intercept=False)
                readout.fit(np.concatenate(designs), np.concatenate(errors), sample_weight=np.concatenate(weights))
                self.coefficients = readout.coef_
        return self

    def predict(self, diagnostics=False):
        prediction = self.heads.predict()
        correction = np.einsum("ij,j->i", reconciliation_directions(prediction), self.coefficients)
        log_forecast = np.where(prediction.closed, 0.0, np.maximum(0.0, prediction.pooled_log + correction))
        if not np.isfinite(log_forecast).all():
            raise ValueError("Nonfinite model output")
        result = pd.DataFrame({"date": prediction.index, "forecast": np.expm1(log_forecast)})
        if diagnostics:
            for column in prediction.columns:
                result[column] = prediction[column].to_numpy()
            result["correction_log"] = correction
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": asdict(self.config), "year": self.year,
                     "training_years": self.training_years, "calibration_years": self.calibration_years,
                     "coefficients": self.coefficients, "regressor": self.heads.regressor,
                     "closure_rates": self.heads.closure_rates}, path)

    @classmethod
    def load(cls, path):
        state = joblib.load(path)
        model = cls(ModelConfig(**state["config"]))
        for name in ("year", "training_years", "calibration_years", "coefficients"):
            setattr(model, name, state[name])
        model.heads = HistoricalHeads(model.config)
        model.heads.year, model.heads.regressor = model.year, state["regressor"]
        model.heads.closure_rates = state["closure_rates"]
        return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data", type=Path, default=Path("processed_data"))
    parser.add_argument("--config", type=Path, default=Path("output/v85/selected_config.json"))
    parser.add_argument("--model", type=Path, default=Path("output/v85/model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("forecast_2024_v85_historical.csv"))
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()
    if args.command == "train":
        config = ModelConfig(**json.loads(args.config.read_text()))
        model = HistoricalForecaster(config).fit(load_daily(args.data), args.year)
        model.save(args.model)
    else:
        model = HistoricalForecaster.load(args.model)
    model.predict().to_csv(args.output, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
