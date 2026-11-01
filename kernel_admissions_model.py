"""Regularized kernel regression for an admissions calendar.

Fits log daily registration counts directly from observations. Similarity has
separate calendar, weekday, and campaign-stage components. Every fitted weight
is estimated by a regularized linear solve, without prior-model predictions.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits


@dataclass(frozen=True)
class KernelConfig:
    bandwidth: float = 7.0
    penalty: float = 0.3
    half_life: float = 4.0
    weekday_coupling: float = 0.05
    infer_closures: bool = True
    align_deadlines: bool = True


def features(dates, deadlines=None):
    dates = pd.DatetimeIndex(dates)
    years = dates.year
    start = pd.to_datetime(years.astype(str) + "-06-20")
    bachelor = pd.to_datetime(years.astype(str) + "-07-25")
    bachelor = bachelor.where(years != 2021, pd.Timestamp("2021-07-29"))
    master = pd.to_datetime(years.astype(str) + "-07-29")
    master -= pd.to_timedelta(np.maximum(master.dayofweek - 4, 0), unit="D")
    if deadlines:
        bachelor = pd.DatetimeIndex([deadlines.get((int(y), "bachelor"), d) for y, d in zip(years, bachelor)])
        master = pd.DatetimeIndex([deadlines.get((int(y), "master_miep"), d) for y, d in zip(years, master)])
    progress = (dates - start).days.to_numpy()
    bachelor_distance = (dates - bachelor).days.to_numpy()
    master_distance = (dates - master).days.to_numpy()
    phase = np.select([dates.month == 6, bachelor_distance <= 0, master_distance <= 0], [0, 1, 2], default=3)
    slot = np.select([dates.dayofweek != 5, progress < 7,
                      (dates.month == 8) & (dates.day > 21), dates.month == 8],
                     ["none", "first", "late", "august"], default="regular")
    return {"time": np.column_stack([progress, bachelor_distance, master_distance]),
            "weekday": dates.dayofweek.to_numpy(), "phase": phase, "slot": slot}


def kernel(left, right, config):
    distance = ((left["time"][:, None, :] - right["time"][None, :, :]) ** 2).mean(axis=2)
    temporal = np.exp(-0.5 * distance / config.bandwidth ** 2)
    same_weekday = left["weekday"][:, None] == right["weekday"][None, :]
    same_phase = left["phase"][:, None] == right["phase"][None, :]
    same_slot = left["slot"][:, None] == right["slot"][None, :]
    # Sum/product of positive-semidefinite kernels remains a valid kernel.
    weekday = config.weekday_coupling + (1 - config.weekday_coupling) * same_weekday
    return temporal * weekday * same_phase * same_slot


class KernelAdmissionsModel:
    def __init__(self, config=KernelConfig()):
        self.config = config

    def fit(self, daily_counts, year):
        history = daily_counts.loc[daily_counts.index.year < year]
        if history.empty or history.index.year.max() != year - 1:
            raise ValueError("Need preceding observed seasons.")
        history = history.loc[(history.index.year != 2020) & (history.index.dayofweek != 6)]
        deadlines = {}
        if isinstance(history, pd.DataFrame):
            if self.config.align_deadlines:
                for season in sorted(history.index.year.unique()):
                    for degree in ("bachelor", "master_miep"):
                        stream = history.loc[f"{season}-07-13":f"{season}-08-25", degree]
                        stream = stream.loc[stream.index.dayofweek < 5]
                        candidates = stream.loc[f"{season}-07-20":f"{season}-08-15"].index
                        losses = {}
                        for date in candidates:
                            before = np.log1p(stream.loc[:date].to_numpy())
                            after = np.log1p(stream.loc[stream.index > date].to_numpy())
                            if len(before) >= 5 and len(after) >= 5 and before.mean() > after.mean():
                                losses[date] = np.sum((before - before.mean())**2) + np.sum((after - after.mean())**2)
                        if losses:
                            deadlines[int(season), degree] = min(losses, key=losses.get)
            history = history.sum(axis=1)
        self.year = year
        self.deadlines = deadlines
        self.training = features(history.index, deadlines)
        weights = np.exp2(-(year - history.index.year.to_numpy()) / self.config.half_life)
        self.center = float(np.average(np.log1p(history), weights=weights))
        gram = kernel(self.training, self.training, self.config)
        gram.flat[::len(gram) + 1] += self.config.penalty / weights
        with threadpool_limits(limits=1):
            self.coefficients = cho_solve(cho_factor(gram), np.log1p(history) - self.center)
        self.closure_rates = {}
        for slot in ("first", "late", "august", "regular"):
            rows = self.training["slot"] == slot
            self.closure_rates[slot] = float(np.average(history.to_numpy()[rows] == 0, weights=weights[rows]))
        return self

    def predict(self):
        dates = pd.date_range(f"{self.year}-06-20", f"{self.year}-08-31")
        target = features(dates)
        similarity = kernel(target, self.training, self.config)
        log_counts = self.center + np.einsum("ij,j->i", similarity, self.coefficients)
        if not np.isfinite(log_counts).all():
            raise ValueError("Nonfinite kernel predictions.")
        closed = dates.dayofweek == 6
        if self.config.infer_closures:
            closed |= np.array([self.closure_rates.get(slot, 0) > 0.5 for slot in target["slot"]])
        return pd.DataFrame({"date": dates, "forecast": np.expm1(np.where(closed, 0, np.maximum(log_counts, 0)))})
