"""Smooth, jointly fitted registration intensities by degree and funding.

Six nonnegative streams share a weekday prior. Their sum is optimized directly
in log1p space. Optional season-volume offsets are estimated from past seasons.
Only raw registration and competition-list records are accepted as inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits


DEGREES = ("bachelor", "master", "other")
STREAMS = tuple(f"{degree}_{funding}" for degree in DEGREES for funding in ("budget", "other"))


@dataclass(frozen=True)
class SmoothConfig:
    knots: int = 5
    penalty: float = 0.3
    auxiliary_weight: float = 0.1
    normalize_volume: bool = True
    half_life: float = 3.0
    max_iterations: int = 1500

    def __post_init__(self):
        if self.knots < 3 or min(self.penalty, self.half_life) <= 0 or self.auxiliary_weight < 0:
            raise ValueError("Invalid model configuration.")


def campaign(year):
    return pd.date_range(f"{year}-06-20", f"{year}-08-31", name="date")


def load_counts(directory):
    directory = Path(directory)
    records = pd.read_csv(directory / "registered.csv")
    included = pd.read_csv(directory / "included_ks.csv")
    for frame in (records, included):
        frame["date"] = pd.to_datetime(frame.reg_ts, errors="raise").dt.normalize()
        if frame.date.isna().any():
            raise ValueError("Missing dates.")
    included["budget"] = included.ob_osnova_name.isin(["Бюджет", "Гослиния"])
    keys = ["abitur_kod", "date", "zayav_type_kod"]
    mapping = included.groupby(keys).budget.max()
    merged = records.merge(mapping, on=keys, how="left", validate="many_to_one")
    if len(merged) != len(records):
        raise ValueError("Registration rows were duplicated.")
    degree = np.select(
        [merged.zayav_type_name.str.startswith("Бакалавр"),
         merged.zayav_type_name.str.startswith("Магистр")],
        DEGREES[:2], default="other",
    )
    merged["stream"] = np.char.add(np.char.add(degree, "_"), np.where(merged.budget.eq(True), "budget", "other"))
    years = sorted(merged.date.dt.year.unique())
    if not years or years != list(range(years[0], years[-1] + 1)):
        raise ValueError("Source seasons must be contiguous.")
    dates = pd.DatetimeIndex(np.concatenate([campaign(year).values for year in years]), name="date")
    counts = merged.groupby(["date", "stream"]).size().unstack(fill_value=0)
    return counts.reindex(index=dates, columns=STREAMS, fill_value=0).astype(float)


def deadlines_from_history(history):
    """Estimate historical transitions, never transitions in a target season."""
    deadlines = {}
    for year in sorted(history.index.year.unique()):
        for degree, fraction in (("bachelor", .20), ("master", .25)):
            stream = history.loc[history.index.year == year, [f"{degree}_budget", f"{degree}_other"]].sum(axis=1)
            threshold = fraction * stream.loc[f"{year}-07-01":f"{year}-07-25"].max()
            if threshold <= 0:
                continue
            for date, value in stream.loc[f"{year}-07-20":f"{year}-08-20"].items():
                following = stream.loc[date + pd.Timedelta(days=1):date + pd.Timedelta(days=3)]
                if value >= threshold and len(following) == 3 and (following < threshold).all():
                    deadlines[int(year), degree] = date
                    break
    return deadlines


def saturday_slots(dates):
    dates = pd.DatetimeIndex(dates)
    progress = (dates - pd.to_datetime(dates.year.astype(str) + "-06-20")).days
    return np.select(
        [dates.dayofweek != 5, progress < 7, (dates.month == 8) & (dates.day > 21), dates.month == 8],
        ["none", "first", "late", "august"], default="regular",
    )


class SmoothDesign:
    def __init__(self, knots):
        self.absolute = SplineTransformer(
            knots=np.linspace(0, 72, knots)[:, None], degree=3, include_bias=False, extrapolation="linear",
        ).fit(np.array([[0.], [72.]]))
        self.relative = SplineTransformer(
            knots=np.linspace(-50, 45, knots)[:, None], degree=3, include_bias=False, extrapolation="linear",
        ).fit(np.array([[-50.], [45.]]))
        self.spline_size = self.absolute.n_features_out_

    def transform(self, dates, historical_deadlines=None):
        dates = pd.DatetimeIndex(dates)
        progress = (dates - pd.to_datetime(dates.year.astype(str) + "-06-20")).days.to_numpy()
        absolute = self.absolute.transform(progress[:, None])
        weekday = np.column_stack([(dates.dayofweek == day).astype(float) for day in range(1, 6)])
        groups = []
        for degree in DEGREES:
            month_day = {"bachelor": "07-25", "master": "07-29", "other": "08-15"}[degree]
            nominal = pd.to_datetime(dates.year.astype(str) + "-" + month_day)
            if degree == "master":
                nominal -= pd.to_timedelta(np.maximum(nominal.dayofweek - 4, 0), unit="D")
            if degree == "bachelor":
                nominal = nominal.where(dates.year != 2021, pd.Timestamp("2021-07-29"))
            ends = pd.DatetimeIndex([
                (historical_deadlines or {}).get((int(year), degree), end)
                for year, end in zip(dates.year, nominal)
            ])
            relative = (dates - ends).days.to_numpy()
            post = (relative > 0).astype(float)
            matrix = np.column_stack([
                np.ones(len(dates)), weekday, (progress == 0).astype(float),
                absolute, self.relative.transform(relative[:, None]), post,
                ((relative >= -3) & (relative <= 0)).astype(float),
                (dates.dayofweek == 5) * post,
            ])
            groups.extend([matrix, matrix])
        return np.stack(groups, axis=1)

    def penalty_matrix(self):
        size = 7 + 2 * self.spline_size + 3
        penalty = np.eye(size)
        penalty[0, 0] = .001
        for start in (7, 7 + self.spline_size):
            difference = np.diff(np.eye(self.spline_size), n=2, axis=0)
            penalty[start:start+self.spline_size, start:start+self.spline_size] += 10 * (difference.T @ difference)
        return penalty


def objective(coefficients, design, offset, actual, weights, penalty_matrix, penalty, auxiliary_weight):
    coefficients = coefficients.reshape(design.shape[1:])
    raw = offset + np.einsum("ngp,gp->ng", design, coefficients)
    log_total = logsumexp(np.column_stack([np.zeros(len(raw)), raw]), axis=1)
    total_error = log_total - np.log1p(actual.sum(axis=1))
    group_error = np.logaddexp(0., raw) - np.log1p(actual)
    value = .5 * np.sum(weights * (total_error**2 + auxiliary_weight * np.sum(group_error**2, axis=1)))
    derivative = weights[:, None] * (
        total_error[:, None] * np.exp(raw - log_total[:, None])
        + auxiliary_weight * group_error * expit(raw)
    )
    gradient = np.einsum("ng,ngp->gp", derivative, design)
    regularized = np.einsum("pq,gq->gp", penalty_matrix, coefficients)
    value += .5 * penalty * np.sum(coefficients * regularized)
    gradient += penalty * regularized
    # Partial pooling of weekday effects across the six streams.
    differences = coefficients[:, 1:6] - coefficients[:, 1:6].mean(axis=0)
    value += .5 * np.sum(differences**2)
    gradient[:, 1:6] += differences
    return float(value), gradient.ravel()


class SmoothAdmissionsModel:
    def __init__(self, config=SmoothConfig()):
        self.config = config
        self.year = None

    def fit(self, counts, forecast_year):
        history = counts.loc[counts.index.year < forecast_year, list(STREAMS)].sort_index()
        years = sorted(history.index.year.unique())
        if len(years) < 2 or years != list(range(years[0], forecast_year)):
            raise ValueError("Need contiguous preceding seasons.")
        expected = pd.DatetimeIndex(np.concatenate([campaign(year).values for year in years]), name="date")
        if not history.index.equals(expected):
            raise ValueError("Each season must include all 73 dates.")
        if not np.isfinite(history.to_numpy()).all() or (history < 0).any().any():
            raise ValueError("Counts must be finite and nonnegative.")
        self.year = forecast_year
        history = history.loc[history.index.year != 2020]
        self.deadlines = deadlines_from_history(history)
        ages = forecast_year - history.index.year.to_numpy()
        weights = np.exp2(-ages / self.config.half_life)
        slots = saturday_slots(history.index)
        self.closure_rates = {
            slot: float(np.average(history.sum(axis=1).to_numpy()[slots == slot] == 0, weights=weights[slots == slot]))
            for slot in ("first", "late", "august", "regular")
        }
        annual = history.groupby(history.index.year).sum()
        recent = annual.tail(3)
        volume_weights = np.exp2(-(forecast_year - recent.index.to_numpy()) / self.config.half_life)
        self.future_offset = np.zeros(len(STREAMS))
        if self.config.normalize_volume:
            self.future_offset = np.average(np.log((recent.to_numpy() + 1) / 73), axis=0, weights=volume_weights)
        keep = history.index.dayofweek != 6
        history, weights = history.loc[keep], weights[keep]
        offset = np.zeros(history.shape)
        if self.config.normalize_volume:
            offset = np.log((annual.loc[history.index.year].to_numpy() + 1) / 73)
        self.design = SmoothDesign(self.config.knots)
        features = self.design.transform(history.index, self.deadlines)
        regularizer = self.design.penalty_matrix()
        actual = history.to_numpy()
        coefficients = []
        with threadpool_limits(limits=1):
            for group in range(len(STREAMS)):
                matrix = features[:, group, :]
                target = np.log(actual[:, group] + .2) - offset[:, group]
                gram = np.einsum("ni,n,nj->ij", matrix, weights, matrix) + self.config.penalty * regularizer
                rhs = np.einsum("ni,n,n->i", matrix, weights, target)
                coefficients.append(solve(gram, rhs, assume_a="pos"))
            fit = minimize(
                objective, np.array(coefficients).ravel(), method="L-BFGS-B", jac=True,
                args=(features, offset, actual, weights, regularizer,
                      self.config.penalty, self.config.auxiliary_weight),
                options={"maxiter": self.config.max_iterations, "ftol": 1e-9, "gtol": 1e-5, "maxcor": 20},
            )
        if not np.isfinite(fit.x).all() or not np.isfinite(fit.fun):
            raise ValueError("Optimizer produced nonfinite parameters.")
        self.coefficients = fit.x.reshape(features.shape[1:])
        self.optimization = {"success": bool(fit.success), "message": str(fit.message),
                             "iterations": int(fit.nit), "loss": float(fit.fun),
                             "max_abs_gradient": float(np.max(np.abs(fit.jac)))}
        return self

    def predict(self, components=False):
        if self.year is None:
            raise RuntimeError("Fit or load the model first.")
        dates = campaign(self.year)
        raw = self.future_offset + np.einsum("ngp,gp->ng", self.design.transform(dates), self.coefficients)
        values = np.exp(raw)
        slots = saturday_slots(dates)
        closed = (dates.dayofweek == 6) | np.array([self.closure_rates.get(slot, 0) > .5 for slot in slots])
        values[closed] = 0
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite forecast.")
        result = pd.DataFrame(values, index=dates, columns=STREAMS)
        result["forecast"] = result.sum(axis=1)
        result = result.reset_index()
        return result if components else result[["date", "forecast"]]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": asdict(self.config), "year": self.year,
                   "coefficients": self.coefficients, "future_offset": self.future_offset,
                   "closure_rates": self.closure_rates, "deadlines": self.deadlines,
                   "optimization": self.optimization}
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        model = cls(SmoothConfig(**payload.pop("config")))
        for name, value in payload.items():
            setattr(model, name, value)
        model.design = SmoothDesign(model.config.knots)
        return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data", type=Path, default=Path("processed_data"))
    parser.add_argument("--config", type=Path, default=Path("output/v76/selected_config.json"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--model", type=Path, default=Path("output/v76/model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("forecast_2024_v76_smooth.csv"))
    args = parser.parse_args()
    if args.command == "train":
        model = SmoothAdmissionsModel(SmoothConfig(**json.loads(args.config.read_text())))
        model.fit(load_counts(args.data), args.year).save(args.model)
    else:
        model = SmoothAdmissionsModel.load(args.model)
    model.predict().to_csv(args.output, index=False, float_format="%.2f")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
