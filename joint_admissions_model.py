"""One shared boosted model trained on both daily totals and degree counts.

The objective couples rows belonging to the same day. Its main term is the
squared error of log(1 + sum of predicted streams), matching total-count RMSLE.
An auxiliary term learns the observed allocation across education levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from admissions_profile_model import GROUPS, calendar, campaign, load_counts


@dataclass(frozen=True)
class JointConfig:
    leaves: int = 7
    iterations: int = 350
    auxiliary_weight: float = 0.3
    half_life: float = 4.0
    align_deadlines: bool = True


def loss_and_gradient(raw, actual, weights, auxiliary_weight):
    """Return the weighted loss, exact gradient and positive GN diagonal.

    Arrays have shape (number of dates, number of streams). The diagonal
    Gauss-Newton approximation is used because trees require scalar Hessians.
    """
    with_constant = np.column_stack([np.zeros(len(raw)), raw])
    log_total = logsumexp(with_constant, axis=1)
    error = log_total - np.log1p(actual.sum(axis=1))
    total_jacobian = np.exp(raw - log_total[:, None])
    group_error = np.logaddexp(0, raw) - np.log1p(actual)
    group_jacobian = expit(raw)
    loss = 0.5 * np.sum(weights * (error**2 + auxiliary_weight * np.sum(group_error**2, axis=1)))
    gradient = error[:, None] * total_jacobian + auxiliary_weight * group_error * group_jacobian
    curvature = total_jacobian**2 + auxiliary_weight * group_jacobian**2
    return loss, gradient * weights[:, None], np.maximum(curvature * weights[:, None], 1e-6)


def training_deadlines(counts):
    """Estimate past phase breaks; no target-year observations are accepted."""
    result = {}
    for year in sorted(counts.index.year.unique()):
        for degree in ("bachelor", "master_miep"):
            stream = counts.loc[f"{year}-07-13":f"{year}-08-25", degree]
            stream = stream.loc[stream.index.dayofweek < 5]
            candidates = stream.loc[f"{year}-07-20":f"{year}-08-15"].index
            losses = {}
            for date in candidates:
                before = np.log1p(stream.loc[:date].to_numpy())
                after = np.log1p(stream.loc[stream.index > date].to_numpy())
                if len(before) >= 5 and len(after) >= 5 and before.mean() > after.mean():
                    losses[date] = np.sum((before-before.mean())**2) + np.sum((after-after.mean())**2)
            if losses:
                result[int(year), degree] = min(losses, key=losses.get)
    return result


def design(dates, deadlines=None):
    dates = pd.DatetimeIndex(dates)
    base = calendar(dates)
    if deadlines:
        for degree, name in (("bachelor", "bachelor"), ("master_miep", "master")):
            nominal = dates - pd.to_timedelta(base[f"days_from_{name}_end"], unit="D")
            ends = pd.DatetimeIndex([deadlines.get((int(y), degree), d) for y, d in zip(dates.year, nominal)])
            base[f"days_from_{name}_end"] = (dates - ends).days
            base[f"{name}_end_weekday"] = ends.dayofweek
            base[f"workdays_from_{name}_end"] = np.busday_count(
                ends.values.astype("datetime64[D]"), dates.values.astype("datetime64[D]"), weekmask="1111110")
    frame = base.loc[base.index.repeat(len(GROUPS))].reset_index(drop=True)
    frame["degree"] = np.tile(GROUPS, len(dates))
    for name, levels in {"degree": list(GROUPS), "weekday": [str(i) for i in range(7)],
                         "saturday_slot": ["none", "first", "late_august", "august", "regular"]}.items():
        frame[name] = pd.Categorical(frame[name], categories=levels)
    return frame


class JointAdmissionsModel:
    def __init__(self, config=JointConfig()):
        self.config = config

    def fit(self, counts, year):
        history = counts.loc[counts.index.year < year, list(GROUPS)].copy()
        if history.empty or history.index.year.max() != year - 1:
            raise ValueError("Need the preceding observed season.")
        if not np.isfinite(history.to_numpy()).all() or (history < 0).any().any():
            raise ValueError("Counts must be finite and nonnegative.")
        history = history.loc[history.index.year != 2020]
        self.year = year
        self.deadlines = training_deadlines(history) if self.config.align_deadlines else {}
        schedule = calendar(history.index)
        season_weight = np.exp2(-(year-history.index.year.to_numpy()) / self.config.half_life)
        self.closure_rates = {}
        for slot in ("first", "late_august", "august", "regular"):
            rows = schedule.saturday_slot.eq(slot).to_numpy()
            self.closure_rates[slot] = float(np.average(history.sum(axis=1).to_numpy()[rows] == 0,
                                                       weights=season_weight[rows]))
        history = history.loc[history.index.dayofweek != 6]
        weights = np.exp2(-(year-history.index.year.to_numpy()) / self.config.half_life)
        actual = history.to_numpy()
        self.base = np.log(np.maximum(np.average(actual, axis=0, weights=weights), 0.01))
        dataset = lgb.Dataset(design(history.index, self.deadlines), label=actual.ravel(),
                              init_score=np.tile(self.base, len(history)), free_raw_data=False)

        def objective(raw, data):
            _, gradient, curvature = loss_and_gradient(
                raw.reshape(actual.shape), actual, weights, self.config.auxiliary_weight)
            return gradient.ravel(), curvature.ravel()

        self.booster = lgb.train(
            {"objective": objective, "learning_rate": 0.035, "num_leaves": self.config.leaves,
             "min_data_in_leaf": 20, "lambda_l2": 5.0, "max_delta_step": 1.0,
             "verbosity": -1, "num_threads": 1, "seed": 71001, "deterministic": True,
             "force_col_wise": True},
            dataset, num_boost_round=self.config.iterations,
        )
        # Python closures used during fitting must not be stored with the model.
        self.booster.params.pop("objective", None)
        return self

    def predict(self):
        dates = campaign(self.year)
        raw = self.booster.predict(design(dates), raw_score=True, num_threads=1).reshape(len(dates), len(GROUPS))
        raw += self.base
        if not np.isfinite(raw).all():
            raise ValueError("Model returned nonfinite values.")
        forecast = pd.DataFrame(np.exp(raw), index=dates, columns=GROUPS)
        schedule = calendar(dates)
        closed = (dates.dayofweek == 6) | schedule.saturday_slot.map(self.closure_rates).gt(0.5)
        forecast.loc[np.asarray(closed)] = 0
        forecast["forecast"] = forecast.sum(axis=1)
        return forecast.rename_axis("date").reset_index()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
