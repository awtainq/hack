"""A single boosted model of degree/funding streams and their daily sum.

Funding is a partition of registration rows, not eventual enrolment. Applicants
with any matched budget entry belong to the budget stream; contract-only and
unmatched registrations remain distinct. Every registration is counted once.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from activity_admissions_model import features
from admissions_profile_model import GROUPS, campaign
from joint_admissions_model import loss_and_gradient, training_deadlines


FUNDING = ("budget", "contract", "unknown")
STREAMS = tuple(f"{degree}:{funding}" for degree in GROUPS for funding in FUNDING)


@dataclass(frozen=True)
class CoherentConfig:
    leaves: int = 15
    iterations: int = 350
    auxiliary_weight: float = 0.1
    half_life: float = 4.0
    closure_policy: str = "learned"

    def __post_init__(self):
        if self.leaves < 2 or self.iterations < 1 or self.half_life <= 0 or self.auxiliary_weight < 0:
            raise ValueError("Invalid training parameters.")
        if self.closure_policy not in {"learned", "none"}:
            raise ValueError("Unknown closure policy.")


def partition_registrations(registered: pd.DataFrame, included: pd.DataFrame) -> pd.DataFrame:
    records, lists = registered.copy(), included.copy()
    for frame in (records, lists):
        frame["date"] = pd.to_datetime(frame.reg_ts, errors="raise").dt.normalize()
        if frame.date.isna().any():
            raise ValueError("Missing dates.")
    keys = ["abitur_kod", "date", "zayav_type_kod"]
    lists["has_budget"] = lists.ob_osnova_name.isin(["Бюджет", "Гослиния"])
    lists["has_contract"] = lists.ob_osnova_name.eq("Договор")
    mapping = lists.groupby(keys)[["has_budget", "has_contract"]].max()
    joined = records.merge(mapping, on=keys, how="left", validate="many_to_one")
    if len(joined) != len(records):
        raise ValueError("Funding join changed the number of registrations.")
    joined["funding"] = np.select(
        [joined.has_budget.eq(True), joined.has_contract.eq(True)],
        ["budget", "contract"], default="unknown",
    )
    joined["degree"] = np.select(
        [joined.zayav_type_name.str.startswith(prefix) for prefix in ("Бакалавр", "Магистр", "СПО")],
        GROUPS[:3], default="postgraduate",
    )
    joined["stream"] = joined.degree + ":" + joined.funding
    # Return no applicant identifiers; all subsequent modelling uses counts.
    return joined[["date", "degree", "funding", "stream"]]


def load_stream_counts(directory: Path) -> pd.DataFrame:
    records = partition_registrations(
        pd.read_csv(directory / "registered.csv"),
        pd.read_csv(directory / "included_ks.csv"),
    )
    years = sorted(records.date.dt.year.unique())
    if not years or years != list(range(years[0], years[-1] + 1)):
        raise ValueError("Observed seasons must be contiguous.")
    dates = pd.DatetimeIndex(np.concatenate([campaign(year).values for year in years]))
    counts = records.groupby(["date", "stream"]).size().unstack(fill_value=0)
    return counts.reindex(index=dates, columns=STREAMS, fill_value=0).astype(float)


def panel(dates, deadlines=None):
    base = features(dates, deadlines)
    frame = base.loc[base.index.repeat(len(STREAMS))].reset_index(drop=True)
    frame["degree"] = np.tile(np.repeat(GROUPS, len(FUNDING)), len(dates))
    frame["funding"] = np.tile(FUNDING, len(dates) * len(GROUPS))
    levels = {
        "weekday": [str(day) for day in range(7)],
        "saturday_slot": ["none", "first", "late_august", "august", "regular"],
        "phase": ["start", "ramp", "bachelor", "master", "late", "post"],
        "degree": list(GROUPS), "funding": list(FUNDING),
    }
    for column, categories in levels.items():
        frame[column] = pd.Categorical(frame[column], categories=categories)
    return frame


class CoherentAdmissionsModel:
    def __init__(self, config=CoherentConfig()):
        self.config = config
        self.year = None

    def fit(self, counts, forecast_year):
        history = counts.loc[counts.index.year < forecast_year, list(STREAMS)].sort_index()
        years = sorted(history.index.year.unique())
        if len(years) < 2 or years != list(range(years[0], forecast_year)):
            raise ValueError("Need contiguous preceding seasons.")
        expected = pd.DatetimeIndex(np.concatenate([campaign(year).values for year in years]))
        if not history.index.equals(expected):
            raise ValueError("Each training season must contain all 73 dates.")
        if not np.isfinite(history.to_numpy()).all() or (history < 0).any().any():
            raise ValueError("Counts must be finite and nonnegative.")
        history = history.loc[history.index.year != 2020]
        self.year = forecast_year
        degree_counts = pd.DataFrame({
            degree: history[[f"{degree}:{funding}" for funding in FUNDING]].sum(axis=1)
            for degree in GROUPS
        })
        self.deadlines = training_deadlines(degree_counts)
        schedule = features(history.index)
        weights = np.exp2(-(forecast_year - history.index.year.to_numpy()) / self.config.half_life)
        self.closure_rates = {}
        for slot in ("first", "late_august", "august", "regular"):
            mask = schedule.saturday_slot.eq(slot).to_numpy()
            self.closure_rates[slot] = float(np.average(
                history.sum(axis=1).to_numpy()[mask] == 0, weights=weights[mask],
            ))
        if self.config.closure_policy == "learned":
            keep = history.index.dayofweek != 6
            history, weights = history.loc[keep], weights[keep]
        actual = history.to_numpy()
        self.offset = np.log(np.maximum(np.average(actual, axis=0, weights=weights), 0.01))
        dataset = lgb.Dataset(
            panel(history.index, self.deadlines), label=actual.ravel(),
            init_score=np.tile(self.offset, len(history)), free_raw_data=False,
        )

        def objective(raw, _):
            _, gradient, curvature = loss_and_gradient(
                raw.reshape(actual.shape), actual, weights, self.config.auxiliary_weight,
            )
            return gradient.ravel(), curvature.ravel()

        self.booster = lgb.train(
            {
                "objective": objective, "learning_rate": 0.035, "num_leaves": self.config.leaves,
                "min_data_in_leaf": 20, "lambda_l2": 5.0, "max_delta_step": 1.0,
                "verbosity": -1, "num_threads": 1, "seed": 73001,
                "deterministic": True, "force_col_wise": True,
            },
            dataset, num_boost_round=self.config.iterations,
        )
        self.booster.params.pop("objective", None)
        return self

    def predict(self, components=False):
        if self.year is None:
            raise RuntimeError("Fit or load the model first.")
        dates = campaign(self.year)
        values = self.booster.predict(panel(dates), raw_score=True, num_threads=1)
        values = np.exp(values.reshape(len(dates), len(STREAMS)) + self.offset)
        if not np.isfinite(values).all():
            raise ValueError("Model produced nonfinite predictions.")
        if self.config.closure_policy == "learned":
            schedule = features(dates)
            closed = (dates.dayofweek == 6) | schedule.saturday_slot.map(self.closure_rates).gt(0.5)
            values[np.asarray(closed)] = 0
        result = pd.DataFrame(values, index=dates, columns=STREAMS)
        result["forecast"] = result.sum(axis=1)
        result = result.rename_axis("date").reset_index()
        return result if components else result[["date", "forecast"]]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(directory / "counts.txt"))
        metadata = {
            "config": asdict(self.config), "year": self.year,
            "offset": self.offset.tolist(), "closure_rates": self.closure_rates,
            "streams": list(STREAMS), "features": list(panel(campaign(self.year)).columns),
        }
        (directory / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / "model.json").read_text())
        model = cls(CoherentConfig(**metadata["config"]))
        model.year = metadata["year"]
        if metadata["streams"] != list(STREAMS) or metadata["features"] != list(panel(campaign(model.year)).columns):
            raise ValueError("Saved model schema differs from the current implementation.")
        model.offset = np.array(metadata["offset"])
        model.closure_rates = metadata["closure_rates"]
        model.booster = lgb.Booster(model_file=str(directory / "counts.txt"))
        return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data", type=Path, default=Path("processed_data"))
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("output/v73/model"))
    parser.add_argument("--output", type=Path, default=Path("forecast_2024_v73_coherent.csv"))
    args = parser.parse_args()
    if args.command == "train":
        config = CoherentConfig(**json.loads(args.config.read_text())) if args.config else CoherentConfig()
        model = CoherentAdmissionsModel(config).fit(load_stream_counts(args.data), args.year)
        model.save(args.model_dir)
    else:
        model = CoherentAdmissionsModel.load(args.model_dir)
    model.predict().to_csv(args.output, index=False, float_format="%.2f")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
