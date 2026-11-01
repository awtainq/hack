"""Compositional admissions forecasting: historical volumes and normalized shapes.

Five disjoint streams partition registrations. Seasonal profiles are softmax
spline regressions trained jointly against total daily squared log error.
Future volumes and channel proportions use preceding campaigns only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits


STREAMS = ("bachelor_other", "bachelor_gov", "master", "college", "other_degree")


@dataclass(frozen=True)
class ShapeConfig:
    knots: int = 7
    penalty: float = 0.1
    auxiliary_weight: float = 0.2
    half_life: float = 4.0
    volume_trend: bool = False
    channels: bool = True
    calibration_penalty: float | None = None
    max_iterations: int = 1200

    def __post_init__(self):
        if self.knots < 3 or self.penalty <= 0 or self.half_life <= 0 or self.auxiliary_weight < 0:
            raise ValueError("Invalid model parameters")
        if self.calibration_penalty is not None and self.calibration_penalty <= 0:
            raise ValueError("Invalid calibration penalty")


def campaign(year):
    return pd.date_range(f"{year}-06-20", f"{year}-08-31", name="date")


def load_counts(path):
    rows = pd.read_csv(path, usecols=["reg_ts", "zayav_type_name", "priem_doc_type_kod"])
    rows["date"] = pd.to_datetime(rows.reg_ts, errors="raise").dt.normalize()
    if rows[["date", "zayav_type_name", "priem_doc_type_kod"]].isna().any().any():
        raise ValueError("Missing date, education level or submission channel")
    bachelor = rows.zayav_type_name.str.startswith("Бакалавр")
    rows["stream"] = np.select([
        bachelor & rows.priem_doc_type_kod.eq(4), bachelor,
        rows.zayav_type_name.str.startswith("Магистр"), rows.zayav_type_name.str.startswith("СПО"),
    ], ["bachelor_gov", "bachelor_other", "master", "college"], default="other_degree")
    years = sorted(rows.date.dt.year.unique())
    if not years or years != list(range(years[0], years[-1] + 1)):
        raise ValueError("Source seasons must be contiguous")
    index = pd.DatetimeIndex(np.concatenate([campaign(year).values for year in years]), name="date")
    counts = rows.groupby(["date", "stream"]).size().unstack(fill_value=0)
    return counts.reindex(index=index, columns=STREAMS, fill_value=0).astype(float)


def saturday_slots(dates):
    dates = pd.DatetimeIndex(dates)
    position = (dates - pd.to_datetime(dates.year.astype(str) + "-06-20")).days
    return np.select([dates.dayofweek != 5, position < 7, (dates.month == 8) & (dates.day > 21), dates.month == 8],
                     ["none", "first", "late", "august"], default="regular")


def historical_deadlines(history):
    result = {}
    for year in history.index.year.unique():
        for degree, columns, fraction in (("bachelor", STREAMS[:2], 0.2), ("master", ("master",), 0.25)):
            counts = history.loc[history.index.year == year, list(columns)].sum(axis=1)
            threshold = fraction * counts.loc[f"{year}-07-01":f"{year}-07-25"].max()
            if threshold <= 0:
                continue
            for date, count in counts.loc[f"{year}-07-20":f"{year}-08-20"].items():
                after = counts.loc[date + pd.Timedelta(days=1):date + pd.Timedelta(days=3)]
                if count >= threshold and len(after) == 3 and (after < threshold).all():
                    result[int(year), degree] = date
                    break
    return result


class ProfileDesign:
    def __init__(self, knots):
        self.absolute = SplineTransformer(knots=np.linspace(0, 72, knots)[:, None], degree=3,
                                          include_bias=False, extrapolation="linear").fit([[0.], [72.]])
        self.relative = SplineTransformer(knots=np.linspace(-50, 45, knots)[:, None], degree=3,
                                          include_bias=False, extrapolation="linear").fit([[-50.], [45.]])
        self.spline_size = self.absolute.n_features_out_

    def transform(self, dates, deadlines=None):
        dates = pd.DatetimeIndex(dates)
        position = (dates - pd.to_datetime(dates.year.astype(str) + "-06-20")).days.to_numpy()
        absolute = self.absolute.transform(position[:, None])
        weekday = np.column_stack([(dates.dayofweek == day).astype(float) for day in range(1, 7)])
        matrices = []
        for degree in ("bachelor", "bachelor", "master", "college", "other_degree"):
            month_day = {"bachelor": "07-25", "master": "07-29", "college": "08-15", "other_degree": "08-15"}[degree]
            nominal = pd.to_datetime(dates.year.astype(str) + "-" + month_day)
            if degree == "master":
                nominal -= pd.to_timedelta(np.maximum(nominal.dayofweek - 4, 0), unit="D")
            if degree == "bachelor":
                nominal = nominal.where(dates.year != 2021, pd.Timestamp("2021-07-29"))
            ends = pd.DatetimeIndex([(deadlines or {}).get((int(year), degree), end) for year, end in zip(dates.year, nominal)])
            distance = (dates - ends).days.to_numpy()
            matrix = np.column_stack([
                weekday, absolute, self.relative.transform(distance[:, None]),
                (position == 0), (position < 4), (distance > 0),
                ((distance >= -3) & (distance <= 0)),
                (dates.dayofweek == 5) * (distance > 0),
                weekday * (position / 72)[:, None],
            ])
            matrices.append(matrix)
        return np.stack(matrices, axis=1)

    def regularizer(self):
        size = 6 + 2 * self.spline_size + 5 + 6
        matrix = np.eye(size)
        for start in (6, 6 + self.spline_size):
            difference = np.diff(np.eye(self.spline_size), n=2, axis=0)
            matrix[start:start + self.spline_size, start:start + self.spline_size] += 10 * difference.T @ difference
        return matrix


def normalized_counts(coefficients, design, volumes, allowed):
    logits = np.einsum("ygdp,gp->ygd", design, coefficients)
    logits = np.where(allowed, logits, -np.inf)
    probabilities = np.exp(logits - logsumexp(logits, axis=2, keepdims=True))
    return volumes[:, :, None] * probabilities, probabilities


def shape_objective(flat, design, volumes, allowed, actual, weights, regularizer, penalty, auxiliary):
    coefficients = flat.reshape(design.shape[1], design.shape[3])
    counts, probabilities = normalized_counts(coefficients, design, volumes, allowed)
    totals = counts.sum(axis=1)
    error = np.log1p(totals) - np.log1p(actual.sum(axis=1))
    stream_error = np.log1p(counts) - np.log1p(actual)
    value = 0.5 * np.sum(weights[:, None] * error ** 2)
    value += 0.5 * auxiliary * np.sum(weights[:, None, None] * stream_error ** 2)
    derivative = weights[:, None, None] * (
        error[:, None, :] / (1 + totals[:, None, :]) + auxiliary * stream_error / (1 + counts)
    )
    centered = derivative - (derivative * probabilities).sum(axis=2, keepdims=True)
    gradient = np.einsum("ygd,ygdp->gp", counts * centered, design)
    regularized = np.einsum("pq,gq->gp", regularizer, coefficients)
    value += 0.5 * penalty * np.sum(coefficients * regularized)
    gradient += penalty * regularized
    return float(value), gradient.ravel()


class VolumeShapeForecaster:
    def __init__(self, config=ShapeConfig()):
        self.config = config

    def availability(self, dates):
        dates = pd.DatetimeIndex(dates)
        slots = saturday_slots(dates)
        closed = (dates.dayofweek == 6) | np.array([self.closure_rates.get(slot, 0) > 0.5 for slot in slots])
        allowed = np.repeat((~closed)[:, None], len(STREAMS), axis=1)
        if self.config.channels:
            allowed[:, 1] = True
        return allowed

    def fit(self, counts, year, cache=None):
        cache = {} if cache is None else cache
        base_config = replace(self.config, calibration_penalty=None)

        def base_model(cutoff):
            past = counts.loc[counts.index.year < cutoff]
            fingerprint = hashlib.sha256(pd.util.hash_pandas_object(past, index=True).values.tobytes()).hexdigest()
            key = (json.dumps(asdict(base_config), sort_keys=True), int(cutoff), fingerprint)
            if key not in cache:
                cache[key] = VolumeShapeForecaster(base_config)._fit_shape(past, cutoff)
            return cache[key]

        fitted = base_model(year)
        vars(self).update({k: v for k, v in vars(fitted).items() if k != "config"})
        self.calibration_years = []
        self.log_bias = 0.0
        if self.config.calibration_penalty is not None:
            numerator, denominator = 0.0, self.config.calibration_penalty
            for past_year in range(int(counts.index.year.min()) + 2, year):
                if past_year == 2020:
                    continue
                prediction = base_model(past_year).predict()
                use = (prediction.date.dt.dayofweek < 5) & prediction.forecast.gt(0)
                actual = counts.loc[prediction.date, list(STREAMS)].sum(axis=1).to_numpy()
                errors = np.log1p(actual) - np.log1p(prediction.forecast.to_numpy())
                weight = np.exp2(-(year - past_year) / self.config.half_life)
                numerator += weight * errors[use].sum()
                denominator += weight * int(use.sum())
                self.calibration_years.append(past_year)
            self.log_bias = float(numerator / denominator)
        return self

    def _fit_shape(self, counts, year):
        history = counts.loc[counts.index.year < year, list(STREAMS)].sort_index().copy()
        years = sorted(history.index.year.unique())
        if len(years) < 2 or years != list(range(years[0], year)):
            raise ValueError("Need contiguous preceding seasons")
        expected = pd.DatetimeIndex(np.concatenate([campaign(y).values for y in years]), name="date")
        if not history.index.equals(expected) or not np.isfinite(history.to_numpy()).all() or (history < 0).any().any():
            raise ValueError("Invalid or incomplete daily counts")
        history = history.loc[history.index.year != 2020]
        if not self.config.channels:
            history["bachelor_other"] += history.bachelor_gov
            history["bachelor_gov"] = 0.0
        self.year = int(year)
        self.log_bias = 0.0
        self.training_years = list(map(int, sorted(history.index.year.unique())))
        self.deadlines = historical_deadlines(history)
        weights = np.exp2(-(year - history.index.year.to_numpy()) / self.config.half_life)
        slots = saturday_slots(history.index)
        institutional = history.drop(columns="bachelor_gov").sum(axis=1)
        self.closure_rates = {slot: float(np.average(institutional[slots == slot].eq(0), weights=weights[slots == slot]))
                              for slot in ("first", "late", "august", "regular")}
        annual = history.groupby(history.index.year).sum()
        degree_annual = annual[["bachelor_other", "master", "college", "other_degree"]].copy()
        degree_annual["bachelor_other"] += annual.bachelor_gov
        recent = degree_annual.tail(3)
        x = recent.index.to_numpy(float)
        weight = np.exp2(-(year - x) / self.config.half_life)
        levels = np.log1p(recent.to_numpy())
        center = np.average(x, weights=weight)
        mean = np.average(levels, axis=0, weights=weight)
        if self.config.volume_trend:
            slope = np.einsum("n,n,ng->g", weight, x - center, levels - mean) / (np.sum(weight * (x - center) ** 2) + 1.0)
            mean += slope * (year - center)
        volumes = np.maximum(0.0, np.expm1(mean))
        last = annual.iloc[-1]
        share = float(last.bachelor_gov / max(last.bachelor_other + last.bachelor_gov, 1.0))
        self.future_volumes = np.array([volumes[0] * (1 - share), volumes[0] * share, *volumes[1:]])
        self.design = ProfileDesign(self.config.knots)
        flat_design = self.design.transform(history.index, self.deadlines)
        design = flat_design.reshape(len(annual), 73, len(STREAMS), -1).transpose(0, 2, 1, 3)
        actual = history.to_numpy().reshape(len(annual), 73, len(STREAMS)).transpose(0, 2, 1)
        allowed = self.availability(history.index).reshape(len(annual), 73, len(STREAMS)).transpose(0, 2, 1)
        volumes = annual.to_numpy()
        year_weights = np.exp2(-(year - annual.index.to_numpy()) / self.config.half_life)
        regularizer = self.design.regularizer()
        initial = []
        with threadpool_limits(limits=1):
            for group in range(len(STREAMS)):
                use_years = volumes[:, group] > 0
                matrix = design[use_years, group].reshape(-1, design.shape[-1])
                if not len(matrix):
                    initial.append(np.zeros(design.shape[-1]))
                    continue
                use_weights = np.repeat(year_weights[use_years], 73)
                target = (np.log(actual[use_years, group] + 0.1) - np.log(volumes[use_years, group, None] + 7.3)).ravel()
                gram = np.einsum("ni,n,nj->ij", matrix, use_weights, matrix) + self.config.penalty * regularizer
                rhs = np.einsum("ni,n,n->i", matrix, use_weights, target)
                initial.append(solve(gram, rhs, assume_a="pos"))
            fitted = minimize(shape_objective, np.array(initial).ravel(), jac=True, method="L-BFGS-B",
                              args=(design, volumes, allowed, actual, year_weights, regularizer,
                                    self.config.penalty, self.config.auxiliary_weight),
                              options={"maxiter": self.config.max_iterations, "ftol": 1e-9, "gtol": 1e-5, "maxcor": 20})
        if not np.isfinite(fitted.x).all() or not np.isfinite(fitted.fun):
            raise ValueError("Nonfinite optimization result")
        self.coefficients = fitted.x.reshape(len(STREAMS), -1)
        self.optimization = {"success": bool(fitted.success), "message": str(fitted.message),
                             "iterations": int(fitted.nit), "loss": float(fitted.fun),
                             "max_abs_gradient": float(np.max(np.abs(fitted.jac)))}
        return self

    def predict(self, components=False):
        dates = campaign(self.year)
        design = self.design.transform(dates).transpose(1, 0, 2)[None, ...]
        allowed = self.availability(dates).T[None, ...]
        values, _ = normalized_counts(self.coefficients, design, self.future_volumes[None, :], allowed)
        frame = pd.DataFrame(values[0].T, index=dates, columns=STREAMS)
        raw_total = frame.sum(axis=1).to_numpy()
        corrected = np.where(raw_total > 0, np.maximum(0.0, np.expm1(np.log1p(raw_total) + self.log_bias)), 0.0)
        factor = np.divide(corrected, raw_total, out=np.zeros_like(corrected), where=raw_total > 0)
        frame *= factor[:, None]
        frame["forecast"] = frame.sum(axis=1)
        frame = frame.reset_index()
        return frame if components else frame[["date", "forecast"]]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": asdict(self.config), "state": {k: v for k, v in vars(self).items() if k not in {"config", "design"}}}, path)

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        model = cls(ShapeConfig(**payload["config"]))
        vars(model).update(payload["state"])
        model.design = ProfileDesign(model.config.knots)
        return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data", type=Path, default=Path("processed_data/registered.csv"))
    parser.add_argument("--config", type=Path, default=Path("output/v86/selected_config.json"))
    parser.add_argument("--model", type=Path, default=Path("output/v86/model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("forecast_2024_v86.csv"))
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()
    if args.command == "train":
        model = VolumeShapeForecaster(ShapeConfig(**json.loads(args.config.read_text()))).fit(load_counts(args.data), args.year)
        model.save(args.model)
    else:
        model = VolumeShapeForecaster.load(args.model)
    model.predict().to_csv(args.output, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
