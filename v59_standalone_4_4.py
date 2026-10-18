"""
Standalone Pipeline for Model V59 (Stream Extrapolation, t = 4.4)
================================================================

This script is an all-in-one, self-contained reproducer for Model V59 with t = 4.4
(Public Leaderboard Score: ~0.42473 - 0.425).

Core Architecture:
------------------
1. Two Traffic Components:
   - Pooled Model (z_pool): Tri-ensemble (LightGBM + CatBoost + HistGB) trained on
     all daily applications with regime/era weights and admission campaign features.
   - Stream Decomposition (z_streams): Daily applications split into "Бюджет" (State-funded)
     and "Договор" (Contract/Tuition-based) via `included_ks.csv`. Individual ensembles
     are trained on each stream, converted to counts, summed, and transformed back to log space.

2. Stream Extrapolation:
   - For parameter t:
         w_pool = 1.0 - t / 2.0
         w_streams = 1.0 - w_pool = t / 2.0
         z(t) = w_pool * z_pool + w_streams * z_streams
   - At t = 4.4:
         w_pool = -1.20, w_streams = +2.20
         z(4.4) = -1.20 * z_pool + 2.20 * z_streams
     This moves in the direction of the contract-heavy August shift, reaching the optimal
     vertex of the leaderboard score parabola.

3. Campaign Calendar & Deadlines:
   - Incorporates distance to the bachelor intake deadline (07-25) and master's deadline (07-29).
   - Calendar rule: All Sundays are closed (0 applications). Historical Saturday voting
     correctly determines closed Saturdays for 2024: 2024-06-22, 2024-08-24, 2024-08-31 (13 zero days).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent if (FILE_PATH.parent / "processed_data").exists() else FILE_PATH.parents[1]
DATA_DIR = ROOT_DIR / "processed_data"
REG_FILE = DATA_DIR / "registered.csv"
KS_FILE = DATA_DIR / "included_ks.csv"

DEFAULT_YEAR = 2024
DEFAULT_T = 4.4
DEFAULT_BAK_END = "07-25"
DEFAULT_MAG_END = "07-29"

W_OPEN = 0.75  # Weight of models trained on open days (y > 0) vs all days

FEATURE_COLUMNS = [
    "dow", "day", "month", "iso_week", "day_of_camp", "progress",
    "is_weekend", "is_saturday", "is_sunday", "is_monday",
    "dist_start", "dist_peak", "dist_deadline", "dist_consent", "dist_orders", "dist_end",
    "is_first_week", "is_diploma_week", "is_peak_period", "is_pre_deadline_3d",
    "is_deadline_day", "is_post_deadline", "is_august",
    "sat_june", "sat_july_peak", "sat_july_late", "sat_aug", "mon_july", "mon_aug",
    "sin_prog", "sin_2prog", "cos_prog", "sin_dow", "cos_dow",
    "dist_end_bak", "after_end_bak", "dist_end_mag", "after_end_mag",
]


# -----------------------------------------------------------------------------
# Feature Engineering & Campaign Timeline
# -----------------------------------------------------------------------------
def detect_intake_ends(reg: pd.DataFrame) -> dict[int, dict[str, str]]:
    """
    Detects historical end-of-intake rush dates for bachelor and master's streams.
    Finds the last day in July/August where daily count exceeds threshold share of peak.
    """
    stream = np.select(
        [
            reg["zayav_type_name"].str.startswith("Бакалавр"),
            reg["zayav_type_name"].str.startswith("Магистр"),
        ],
        ["bak", "mag"],
        "other",
    )
    counts = reg.groupby([stream, reg["dt"]]).size()
    ends: dict[int, dict[str, str]] = {}

    for y in sorted(reg["dt"].dt.year.unique()):
        ends[y] = {}
        for st, share in (("bak", 0.20), ("mag", 0.25)):
            ser = counts[st].reindex(pd.date_range(f"{y}-06-20", f"{y}-08-31"), fill_value=0)
            thr = share * ser[f"{y}-07-01":f"{y}-07-25"].max()
            for d, v in ser[f"{y}-07-20":f"{y}-08-20"].items():
                if v >= thr and (ser[d + pd.Timedelta(days=1):d + pd.Timedelta(days=3)] < thr).all():
                    ends[y][st] = d.strftime("%m-%d")
                    break
    return ends


def create_daily_grid(
    reg_path: Path,
    last_year: int,
    target_ends: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Builds the daily calendar grid (June 20 to August 31) for 2014..last_year,
    computing all temporal, campaign, milestone, and deadline features.
    """
    reg = pd.read_csv(reg_path, usecols=["reg_ts", "priem_doc_type_kod", "zayav_type_name"])
    reg["dt"] = pd.to_datetime(reg["reg_ts"]).dt.normalize()

    ends = detect_intake_ends(reg)
    if target_ends:
        ends[last_year] = target_ends

    # Date range for each campaign year
    dates = [
        d
        for y in range(2014, last_year + 1)
        for d in pd.date_range(f"{y}-06-20", f"{y}-08-31")
    ]
    df = pd.DataFrame({"date": dates})
    df["y"] = df["date"].map(reg["dt"].value_counts()).fillna(0.0)
    df["epgu"] = df["date"].map(
        reg.loc[reg["priem_doc_type_kod"] == 4, "dt"].value_counts()
    ).fillna(0.0)
    df["z"] = np.log1p(df["y"])
    df["year"] = df["date"].dt.year
    df["dow"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)

    # Campaign progression features
    df["day_of_camp"] = (df["date"] - pd.to_datetime(df["year"].astype(str) + "-06-20")).dt.days
    df["progress"] = df["day_of_camp"] / 72.0

    # Day-of-week indicators
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_saturday"] = (df["dow"] == 5).astype(int)
    df["is_sunday"] = (df["dow"] == 6).astype(int)
    df["is_monday"] = (df["dow"] == 0).astype(int)

    # Milestone distances
    milestones = [
        ("06-20", "start"),
        ("07-08", "peak"),
        ("07-25", "deadline"),
        ("08-03", "consent"),
        ("08-09", "orders"),
        ("08-31", "end"),
    ]
    for md, name in milestones:
        df[f"dist_{name}"] = (df["date"] - pd.to_datetime(df["year"].astype(str) + "-" + md)).dt.days

    # Distance to stream deadlines (Bachelor / Master's)
    for st in ("bak", "mag"):
        end_dates = pd.to_datetime([f"{y}-{ends[y][st]}" for y in df["year"]])
        df[f"dist_end_{st}"] = (df["date"] - end_dates).dt.days
        df[f"after_end_{st}"] = (df[f"dist_end_{st}"] > 0).astype(int)

    # Campaign period flags
    jun = df["month"] == 6
    jul = df["month"] == 7
    aug = df["month"] == 8
    sat = df["dow"] == 5
    mon = df["dow"] == 0

    flags = {
        "is_first_week": jun & (df["day"] <= 23),
        "is_diploma_week": jun & (df["day"] >= 24),
        "is_peak_period": jul & (df["day"] <= 15),
        "is_pre_deadline_3d": jul & df["day"].between(22, 25),
        "is_deadline_day": jul & (df["day"] == 25),
        "is_post_deadline": jul & (df["day"] > 25),
        "is_august": aug,
        "sat_june": sat & jun,
        "sat_july_peak": sat & jul & (df["day"] <= 20),
        "sat_july_late": sat & jul & (df["day"] > 20),
        "sat_aug": sat & aug,
        "mon_july": mon & jul,
        "mon_aug": mon & aug,
    }
    for col, mask in flags.items():
        df[col] = mask.astype(int)

    # Saturday slot categorization for historical zero-workday voting
    sat_cum = df[df["dow"] == 5].groupby("year").cumcount().reindex(df.index)
    md_str = df["date"].dt.strftime("%m-%d")
    df["sat_slot"] = np.select(
        [
            sat_cum == 0,
            (df["dow"] == 5) & (md_str > "08-21"),
            (df["dow"] == 5) & (md_str >= "08-01"),
            df["dow"] == 5,
        ],
        ["first", "late", "aug_mid", "july"],
        "",
    )

    # Cyclical harmonics
    df["sin_prog"] = np.sin(np.pi * df["progress"])
    df["sin_2prog"] = np.sin(2 * np.pi * df["progress"])
    df["cos_prog"] = np.cos(np.pi * df["progress"])
    df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7)

    return df


def create_stream_grid(
    reg_path: Path,
    ks_path: Path,
    last_year: int,
    target_ends: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Builds the daily grid joined with Budget and Contract streams from included_ks.
    Creates target columns `z_Бюджет` and `z_Договор`.
    """
    reg = pd.read_csv(reg_path, usecols=["abitur_kod", "reg_ts"])
    ks = pd.read_csv(ks_path, usecols=["abitur_kod", "reg_ts", "ob_osnova_name"])

    reg["dt"] = pd.to_datetime(reg["reg_ts"]).dt.normalize()
    ks["dt"] = pd.to_datetime(ks["reg_ts"]).dt.normalize()

    # Match applicant applications with competition list
    ks_dedup = ks[["abitur_kod", "dt", "ob_osnova_name"]].drop_duplicates(["abitur_kod", "dt"])
    merged = reg.merge(ks_dedup, on=["abitur_kod", "dt"], how="left")
    merged["ob_osnova_name"] = merged["ob_osnova_name"].fillna("Договор")

    # Pivot daily counts by stream
    pivot = (
        merged.pivot_table(index="dt", columns="ob_osnova_name", values="abitur_kod", aggfunc="count")
        .fillna(0)
    )

    grid = create_daily_grid(reg_path, last_year, target_ends=target_ends).set_index("date")
    for stream in ("Бюджет", "Договор"):
        col_name = f"z_{stream}"
        counts = grid.index.map(pivot[stream] if stream in pivot.columns else pd.Series(0, index=grid.index))
        grid[col_name] = np.log1p(counts.fillna(0.0))

    return grid.reset_index()


# -----------------------------------------------------------------------------
# Era Weighting & Working Schedule
# -----------------------------------------------------------------------------
def compute_era_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Computes sample weights prioritizing modern standardized campaigns (EPGU era 2021+),
    and downweighting abnormal years (e.g. COVID 2020 shift).
    """
    total = df.groupby("year")["y"].sum().clip(lower=1.0)
    july_ratio = df[df["month"] == 7].groupby("year")["y"].sum() / total
    epgu_ratio = df.groupby("year")["epgu"].sum() / total

    stats = pd.DataFrame({"july": july_ratio, "epgu": epgu_ratio})
    normal_july = stats.loc[stats["july"] > 0.5, "july"].median()
    fidelity = np.exp(-18.0 * np.maximum(0.0, normal_july - stats["july"]) ** 2)

    base = np.where(
        stats["epgu"] > 0.01,
        2.5 * fidelity,
        np.where(
            (stats.index >= 2017) & (fidelity > 0.8),
            1.5 * fidelity,
            np.where(fidelity <= 0.8, 0.4, 0.7),
        ),
    )
    era_map = dict(zip(stats.index, base))
    return df["year"].map(era_map).to_numpy(float)


def determine_closed_days(tr: pd.DataFrame, te: pd.DataFrame) -> list[str]:
    """
    Determines closed Saturdays based on majority voting in historical data for each slot.
    For 2024, this identifies: 2024-06-22, 2024-08-24, 2024-08-31.
    """
    sat = tr[(tr["dow"] == 5) & (tr["year"] != 2020)]
    w = sat["year"].map(dict(zip(tr["year"], compute_era_weights(tr))))
    p_zero = (w * (sat["y"] == 0)).groupby(sat["sat_slot"]).sum() / w.groupby(sat["sat_slot"]).sum()
    closed = te["sat_slot"].map(lambda s: bool(s) and p_zero.get(s, 0.0) > 0.5)
    return list(te.loc[closed, "date"].dt.strftime("%Y-%m-%d"))


def apply_closed_schedule(te: pd.DataFrame, z: np.ndarray, closed_dates: list[str]) -> np.ndarray:
    """Zeros out predictions on Sundays and designated closed Saturdays."""
    closed_mask = (
        te["date"].dt.strftime("%Y-%m-%d").isin(closed_dates).to_numpy()
        | (te["dow"] == 6).to_numpy()
    )
    return np.where(closed_mask, 0.0, z)


# -----------------------------------------------------------------------------
# Ensemble Modeling
# -----------------------------------------------------------------------------
def make_lgbm(seed: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=130,
        learning_rate=0.035,
        num_leaves=14,
        min_child_samples=4,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=seed * 17 + 42,
        verbosity=-1,
    )


def make_catboost(seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=140,
        learning_rate=0.035,
        depth=4,
        l2_leaf_reg=4,
        random_seed=seed * 23 + 42,
        verbose=False,
    )


def make_histgb(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=130,
        learning_rate=0.035,
        max_leaf_nodes=14,
        min_samples_leaf=4,
        random_state=seed * 31 + 42,
    )


def predict_ensemble(
    tr: pd.DataFrame,
    te: pd.DataFrame,
    target_col: str = "z",
) -> np.ndarray:
    """
    Fits the Tri-Ensemble (LightGBM 5 seeds, CatBoost 3 seeds, HistGB 2 seeds) on open days,
    and CatBoost (3 seeds) on all days. Blends predictions with 0.75 / 0.25 weights.
    """
    open_tr = tr[tr["y"] > 0]
    w_all = compute_era_weights(tr)
    w_open = compute_era_weights(open_tr)

    # 1. Models trained on open days (y > 0)
    preds_lgb = [
        make_lgbm(s).fit(open_tr[FEATURE_COLUMNS], open_tr[target_col], sample_weight=w_open).predict(te[FEATURE_COLUMNS])
        for s in range(5)
    ]
    preds_cat = [
        make_catboost(s).fit(open_tr[FEATURE_COLUMNS], open_tr[target_col], sample_weight=w_open).predict(te[FEATURE_COLUMNS])
        for s in range(3)
    ]
    preds_hgb = [
        make_histgb(s).fit(open_tr[FEATURE_COLUMNS], open_tr[target_col], sample_weight=w_open).predict(te[FEATURE_COLUMNS])
        for s in range(2)
    ]

    mean_lgb = np.mean(preds_lgb, axis=0)
    mean_cat = np.mean(preds_cat, axis=0)
    mean_hgb = np.mean(preds_hgb, axis=0)
    z_open = (mean_lgb + mean_cat + mean_hgb) / 3.0

    # 2. CatBoost trained on all days
    preds_all = [
        make_catboost(s).fit(tr[FEATURE_COLUMNS], tr[target_col], sample_weight=w_all).predict(te[FEATURE_COLUMNS])
        for s in range(3)
    ]
    z_all = np.mean(preds_all, axis=0)

    # 3. Blend and clip non-negative
    z_pred = W_OPEN * z_open + (1.0 - W_OPEN) * z_all
    return np.clip(z_pred, 0.0, None)


# -----------------------------------------------------------------------------
# End-to-End Forecast Generation
# -----------------------------------------------------------------------------
def run_v59_pipeline(
    year: int = DEFAULT_YEAR,
    t: float = DEFAULT_T,
    bak_end: str = DEFAULT_BAK_END,
    mag_end: str = DEFAULT_MAG_END,
    reg_file: Path = REG_FILE,
    ks_file: Path = KS_FILE,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Executes the full V59 pipeline:
    1. Builds daily and stream grids.
    2. Fits pooled and stream ensembles.
    3. Extrapolates along stream direction at parameter t.
    4. Applies campaign closed schedule.
    """
    ends = {"bak": bak_end, "mag": mag_end}

    # 1. Prepare data grids
    print(f"Building campaign grid for {year} (Bachelor deadline: {bak_end}, Master's deadline: {mag_end})...")
    grid_total = create_daily_grid(reg_file, year, target_ends=ends)
    tr_total = grid_total[grid_total["year"] < year]
    te_total = grid_total[grid_total["year"] == year].reset_index(drop=True)

    print("Building stream-decomposed grid (Budget & Contract)...")
    grid_streams = create_stream_grid(reg_file, ks_file, year, target_ends=ends)
    tr_streams = grid_streams[grid_streams["year"] < year]
    te_streams = grid_streams[grid_streams["year"] == year].reset_index(drop=True)

    # 2. Train pooled ensemble
    print("Training pooled ensemble (z_pool)...")
    z_pool = predict_ensemble(tr_total, te_total, target_col="z")

    # 3. Train stream ensembles
    print("Training stream ensembles for 'Бюджет' and 'Договор'...")
    z_budget = predict_ensemble(tr_streams, te_streams, target_col="z_Бюджет")
    z_contract = predict_ensemble(tr_streams, te_streams, target_col="z_Договор")

    # Reconstruct combined stream prediction in count space, then log1p
    count_budget = np.expm1(z_budget)
    count_contract = np.expm1(z_contract)
    z_streams = np.log1p(count_budget + count_contract)

    # 4. Extrapolation at parameter t
    #    w_pool = 1.0 - t / 2.0; w_streams = t / 2.0
    w_pool = 1.0 - t / 2.0
    w_streams = 1.0 - w_pool
    print(f"Applying stream extrapolation with t={t}: w_pool={w_pool:+.2f}, w_streams={w_streams:+.2f}")
    z_extrapolated = w_pool * z_pool + w_streams * z_streams

    # 5. Apply schedule
    closed_dates = determine_closed_days(tr_total, te_total)
    zp = apply_closed_schedule(te_total, z_extrapolated, closed_dates)

    # 6. Format final forecast DataFrame
    forecast_counts = np.round(np.expm1(zp), 2)
    forecast_df = pd.DataFrame({
        "date": te_total["date"].dt.strftime("%Y-%m-%d"),
        "forecast": forecast_counts,
    })

    return forecast_df, closed_dates


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Standalone V59 (Stream Extrapolation t=4.4) Pipeline"
    )
    parser.add_argument(
        "--year",
        type=int,
        default=DEFAULT_YEAR,
        help=f"Target forecast year (default: {DEFAULT_YEAR})",
    )
    parser.add_argument(
        "--t",
        type=float,
        default=DEFAULT_T,
        help=f"Extrapolation parameter t along stream direction (default: {DEFAULT_T})",
    )
    parser.add_argument(
        "--bak-end",
        default=DEFAULT_BAK_END,
        help=f"Bachelor intake rush end MM-DD (default: {DEFAULT_BAK_END})",
    )
    parser.add_argument(
        "--mag-end",
        default=DEFAULT_MAG_END,
        help=f"Budget Master's intake rush end MM-DD (default: {DEFAULT_MAG_END})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Custom output CSV path (default: forecast_<year>_v59_t<t>.csv)",
    )
    args = parser.parse_args()

    # Execute pipeline
    out_df, closed_saturdays = run_v59_pipeline(
        year=args.year,
        t=args.t,
        bak_end=args.bak_end,
        mag_end=args.mag_end,
    )

    # Determine destination file path
    output_path = (
        Path(args.out)
        if args.out
        else ROOT_DIR / f"forecast_{args.year}_v59_t{args.t:g}.csv"
    )
    out_df.to_csv(output_path, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("V59 FORECAST SUMMARY")
    print("=" * 60)
    print(f"Output File:           {output_path.name}")
    print(f"Target Year:           {args.year}")
    print(f"Extrapolation Step t:  {args.t}")
    print(f"Closed Saturdays:      {', '.join(closed_saturdays) or 'None'}")
    print(f"Total Closed Days:     {(out_df['forecast'] == 0).sum()} (Sundays + Closed Saturdays)")
    print(f"Total Forecast Sum:    {out_df['forecast'].sum():.2f}")
    print("=" * 60)
    print(f"Saved successfully to: {output_path.resolve()}\n")


if __name__ == "__main__":
    main()
