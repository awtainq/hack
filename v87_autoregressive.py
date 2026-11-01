"""Seasonal autoregression with jointly supervised admissions streams.

Calendar features explain known transitions. Lag features transfer daily
profiles and channel composition from preceding campaigns. The final readout
and weekend policy are selected only by chronological validation.
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
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from v87_data import CATEGORIES, TARGETS, calendar_features, campaign, history_before, lag_features, load_daily, past_deadlines


@dataclass(frozen=True)
class AutoregressiveConfig:
    family: str = "cat"
    depth: int = 3
    iterations: int = 500
    half_life: float = 4.0
    use_lags: bool = True
    readout: str = "total"
    schedule: str = "learned"
    correction: bool = False
    positive_only: bool = False
    min_leaf: int = 5
    feature_fraction: float = .8
    lag_levels_only: bool = False

    def __post_init__(self):
        if self.depth < 1 or self.iterations < 1 or self.half_life <= 0:
            raise ValueError("Invalid training parameters")
        if self.readout not in {"total", "degree", "funding"} or self.schedule not in {"learned", "none"}:
            raise ValueError("Invalid output configuration")
        if self.family not in {"cat", "lgb", "rf", "extra"} or (self.positive_only and self.schedule == "none"):
            raise ValueError("Invalid learner or conditional-count schedule")
        if self.min_leaf < 1 or not 0 < self.feature_fraction <= 1:
            raise ValueError("Invalid forest regularization")


class SeasonalAutoregressor:
    def __init__(self, config=AutoregressiveConfig()):
        self.config = config

    def fit(self, counts, year, cache=None):
        history = history_before(counts, year)
        cache = {} if cache is None else cache
        base_config = replace(self.config, correction=False, readout="total")

        def base_model(cutoff):
            past = history.loc[history.index.year < cutoff]
            fingerprint = hashlib.sha256(pd.util.hash_pandas_object(past, index=True).values.tobytes()).hexdigest()
            key = (json.dumps(asdict(base_config), sort_keys=True), int(cutoff), fingerprint)
            if key not in cache:
                cache[key] = SeasonalAutoregressor(base_config)._fit_base(past, cutoff)
            view = SeasonalAutoregressor(replace(base_config, readout=self.config.readout))
            vars(view).update({k: v for k, v in vars(cache[key]).items() if k != "config"})
            return view

        current = base_model(year)
        vars(self).update({k: v for k, v in vars(current).items() if k != "config"})
        self.calibration_years = []
        self.corrector = None
        if self.config.correction:
            frames, errors, weights = [], [], []
            for past_year in range(int(history.index.year.min()) + 2, year):
                if past_year == 2020:
                    continue
                model = base_model(past_year)
                prediction = model.predict(diagnostics=True)
                use = ~prediction.closed.to_numpy()
                frames.append(model.residual_features(prediction).loc[use])
                actual = history.loc[prediction.date, "total"].to_numpy()
                errors.append((np.log1p(actual) - np.log1p(prediction.forecast.to_numpy()))[use])
                weights.append(np.full(use.sum(), np.exp2(-(year - past_year) / self.config.half_life)))
                self.calibration_years.append(past_year)
            if frames:
                self.corrector = CatBoostRegressor(
                    depth=2, iterations=200, learning_rate=.03, l2_leaf_reg=30, loss_function="RMSE",
                    cat_features=list(CATEGORIES), one_hot_max_size=16, random_seed=87002,
                    thread_count=1, verbose=False, allow_writing_files=False,
                )
                self.corrector.fit(pd.concat(frames), np.concatenate(errors), sample_weight=np.concatenate(weights))
        return self

    def _fit_base(self, counts, year):
        self.year = int(year)
        history = history_before(counts, year)
        self.training_years = sorted(set(map(int, history.index.year)) - {2020})
        self.history = history.loc[history.index.year != 2020].copy()
        self.deadlines = past_deadlines(self.history)
        frame = calendar_features(self.history.index, self.deadlines)
        if self.config.use_lags:
            frame = frame.join(self.selected_lags(self.history.index))
        weight = np.exp2(-(year - self.history.index.year.to_numpy()) / self.config.half_life)
        self.closure_rates = {}
        for slot in ("first", "late", "august", "regular"):
            use = frame.saturday_slot.eq(slot)
            self.closure_rates[slot] = float(np.average(self.history.total[use].eq(0), weights=weight[use]))
        use = np.ones(len(frame), dtype=bool)
        if self.config.schedule == "learned":
            use &= self.history.index.dayofweek != 6
        if self.config.positive_only:
            use &= self.history.total.to_numpy() > 0
        if self.config.family == "cat":
            self.regressor = CatBoostRegressor(
                loss_function="MultiRMSE", depth=self.config.depth, iterations=self.config.iterations,
                learning_rate=.035, l2_leaf_reg=10, random_seed=87001, cat_features=list(CATEGORIES),
                one_hot_max_size=16, thread_count=1, verbose=False, allow_writing_files=False,
                allow_const_label=True,
            )
        elif self.config.family == "lgb":
            self.regressor = MultiOutputRegressor(LGBMRegressor(
                n_estimators=self.config.iterations, learning_rate=.035, num_leaves=2 ** self.config.depth,
                min_child_samples=10, reg_lambda=10, n_jobs=1, random_state=87001,
                verbosity=-1, deterministic=True, force_col_wise=True,
            ))
        else:
            forest = RandomForestRegressor if self.config.family == "rf" else ExtraTreesRegressor
            self.regressor = forest(n_estimators=self.config.iterations, min_samples_leaf=self.config.min_leaf,
                                    max_features=self.config.feature_fraction, n_jobs=1, random_state=87003)
        self.regressor.fit(self.model_features(frame.loc[use]), np.log1p(self.history.loc[use, list(TARGETS)]), sample_weight=weight[use])
        self.feature_names = list(frame.columns)
        self.corrector = None
        return self

    def model_features(self, frame):
        if self.config.family == "cat":
            return frame
        frame = frame.copy()
        levels = {"weekday": [str(day) for day in range(7)],
                  "phase": ["start", "june", "bachelor", "master", "tail"],
                  "saturday_slot": ["none", "first", "late", "august", "regular"]}
        for column, categories in levels.items():
            frame[column] = pd.Categorical(frame[column], categories=categories)
        if self.config.family in {"rf", "extra"}:
            frame = pd.get_dummies(frame, columns=list(CATEGORIES)).fillna(-1.0)
        return frame

    def selected_lags(self, dates):
        result = lag_features(dates, self.history)
        if self.config.lag_levels_only:
            result = result[[column for column in result if column.endswith("_level")]]
        return result

    def residual_features(self, prediction):
        dates = pd.DatetimeIndex(prediction.date)
        frame = calendar_features(dates).join(lag_features(dates, self.history))
        for column in ("total_log", "degree_log", "funding_log", "government_log"):
            frame[column] = prediction[column].to_numpy()
        return frame

    def predict(self, diagnostics=False):
        dates = campaign(self.year)
        frame = calendar_features(dates)
        if self.config.use_lags:
            frame = frame.join(self.selected_lags(dates))
        heads = np.maximum(0.0, self.regressor.predict(self.model_features(frame)))
        output = {"total": heads[:, 0], "degree": np.log1p(np.expm1(heads[:, 1:5]).sum(axis=1)),
                  "funding": np.log1p(np.expm1(heads[:, 5:7]).sum(axis=1))}
        log_forecast = output[self.config.readout]
        closed = np.zeros(len(frame), dtype=bool)
        if self.config.schedule == "learned":
            closed = (dates.dayofweek == 6) | frame.saturday_slot.map(self.closure_rates).gt(.5).to_numpy()
            log_forecast = np.where(closed, 0.0, log_forecast)
        correction = np.zeros(len(frame))
        if self.corrector is not None:
            raw = pd.DataFrame({"date": dates, **{f"{name}_log": values for name, values in output.items()}, "government_log": heads[:, 7]})
            correction = self.corrector.predict(self.residual_features(raw))
            log_forecast = np.where(closed, 0.0, np.maximum(0.0, log_forecast + correction))
        if not np.isfinite(log_forecast).all():
            raise ValueError("Nonfinite forecast")
        result = pd.DataFrame({"date": dates, "forecast": np.expm1(log_forecast)})
        if diagnostics:
            for name, values in output.items():
                result[f"{name}_log"] = values
            result["government_log"] = heads[:, 7]
            result["closed"] = closed
            result["correction_log"] = correction
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": asdict(self.config), "state": {k: v for k, v in vars(self).items() if k != "config"}}, path)

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        model = cls(AutoregressiveConfig(**payload["config"]))
        vars(model).update(payload["state"])
        return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data", type=Path, default=Path("processed_data"))
    parser.add_argument("--config", type=Path, default=Path("output/v87/selected_config.json"))
    parser.add_argument("--model", type=Path, default=Path("output/v87/model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("forecast_2024_v87.csv"))
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()
    if args.command == "train":
        model = SeasonalAutoregressor(AutoregressiveConfig(**json.loads(args.config.read_text()))).fit(load_daily(args.data), args.year)
        model.save(args.model)
    else:
        model = SeasonalAutoregressor.load(args.model)
    model.predict().to_csv(args.output, index=False, float_format="%.6f")


if __name__ == "__main__":
    main()
