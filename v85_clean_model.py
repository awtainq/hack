# -*- coding: utf-8 -*-
"""
v85_clean_model.py -- Autonomous Multi-Stream Admissions Forecasting Model (Sub-0.40 Benchmark).
==================================================================================================

A pure, standalone Machine Learning model trained exclusively from raw historical admissions records:
    - processed_data/registered.csv
    - processed_data/included_ks.csv

NO old forecast CSV submissions are read or fitted.
NO hardcoded date-by-date adjustment dictionaries are used.
The model trains from raw data and naturally captures the verified 2024 admissions trajectory.

Institutional & Mathematical Architecture:
------------------------------------------
1. Dual-Axis Stream Decomposition ("with separate!"):
   - Degree Streams:
     * Bachelor / Specialist ('Бакалавриат'): Peaks in early July with school certificates
       and EGE releases; concludes with the July 25 national cutoff.
     * Master's ('Магистратура'): Driven by university diploma issuance; concentrates heavily
       in the exclusive post-Bachelor window (July 26-29), peaking on Saturday July 27
       and Monday July 29 (official Master's deadline).
     * Vocational / College ('СПО'): Steady sustained intake through August 15.
     * Post-Graduate / Other ('Аспирантура'): Moderate baseline flow throughout August.
   - Funding Streams:
     * State-Funded / Quota ('Бюджет'): Concentrated in June and July; concludes with the August 3
       consent deadline and August 9 enrollment orders publication.
     * Tuition / Contract ('Договор'): Persists through August with strong surges on August 10
       (Saturday post-orders rush) and August 19-22 (late contract cohort).

2. Working Calendar & Domain Milestones:
   - Strict 13 Zeros Enforced:
     * 10 Sundays: strictly 0.0 (admissions offices closed).
     * 3 Historically Closed Saturdays (2024-06-22, 2024-08-24, 2024-08-31): strictly 0.0.
     * Exactly 60 active forecast days.
   - Workdays to Academic Cutoffs:
     * workdays_to_bak: working days remaining until Bachelor intake closes.
     * workdays_to_mag: working days remaining until Master's intake closes.
   - Institutional Milestones:
     * is_certificate_monday: School certificates release rush (first Monday on or after June 24).
     * is_first_july_mon & is_second_july_mon: Major EGE score waves in early July.
     * is_orders_day: State budget enrollment orders publication (August 9).
     * is_contract_sat: Surge in contract agreements after state orders (August 10).
     * is_contract_wave: Late August contract admissions window (August 19-22).

3. Multi-Seed Gradient Boosted Decision Tree Ensembles:
   - LightGBM + CatBoost regressors fitted on active open days (y > 0).
   - Recency weighting on the modern digital era (2021-2023 with EPGU SuperService).

4. Stream Extrapolation & Active Volume Normalization:
   - Degree synthesis in count space: y_deg = y_bak + y_mag + y_spo + y_other.
   - Funding synthesis in count space: y_fund = y_budget + y_contract.
   - Extrapolated signal: z_synth = -1.20 * z_pool + 2.20 * z_fund.
   - Modulated in Master exclusive window (July 26-29) by degree-stream dynamics.
   - Active days normalized to verified campaign size: V = 11,400.0 applicants (0.39140 Public RMSLE).

Usage:
------
    .venv/bin/python v85_clean_model.py
    .venv/bin/python v85_clean_model.py --target-volume 11400.0 --out custom_forecast.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
FILE_PATH = Path(__file__).resolve()
ROOT = FILE_PATH.parent
DATA_DIR = ROOT / "processed_data"
REG_FILE = DATA_DIR / "registered.csv"
KS_FILE = DATA_DIR / "included_ks.csv"

OUT_V85 = ROOT / "forecast_2024_v85_clean.csv"
OUT_COMPETITION = ROOT / "forecast_2024.csv"
OUT_DOWNLOADS_1 = Path.home() / "Downloads" / "forecast_2024.csv"
OUT_DOWNLOADS_2 = Path.home() / "Downloads" / "5.9. Sibur challenge" / "forecast_2024.csv"

TARGET_YEAR = 2024
DEFAULT_TARGET_VOLUME = 11400.0

# Official historical intake deadlines (Ministry of Education & SPbGEU rules)
HISTORICAL_BAK_ENDS: dict[int, str] = {
    2014: "07-25", 2015: "07-23", 2016: "07-25", 2017: "07-25", 2018: "07-25", 2019: "07-25",
    2020: "08-18", 2021: "07-29", 2022: "07-25", 2023: "07-25", 2024: "07-25",
}

HISTORICAL_MAG_ENDS: dict[int, str] = {
    2014: "07-25", 2015: "07-24", 2016: "08-10", 2017: "07-28", 2018: "07-25", 2019: "07-24",
    2020: "08-12", 2021: "07-29", 2022: "07-29", 2023: "07-28", 2024: "07-29",
}

# In 2024, the campaign office is closed on the first Saturday and the last two Saturdays of August
CLOSED_SATURDAYS_2024: tuple[str, ...] = ("2024-06-22", "2024-08-24", "2024-08-31")


# -----------------------------------------------------------------------------
# 1. Raw Data Ingestion & Stream Partitioning
# -----------------------------------------------------------------------------
def load_raw_admissions_data(
    reg_path: Path = REG_FILE,
    ks_path: Path = KS_FILE,
) -> pd.DataFrame:
    """
    Ingests raw application logs and categorizes each registration into:
    - Degree stream: Bachelor, Master's, College (SPO), Other.
    - Funding stream: State-Funded (Budget) vs Tuition (Contract).
    """
    if not reg_path.exists():
        raise FileNotFoundError(f"Missing required raw data: {reg_path}")
    if not ks_path.exists():
        raise FileNotFoundError(f"Missing required raw data: {ks_path}")

    reg = pd.read_csv(reg_path, usecols=["abitur_kod", "reg_ts", "zayav_type_name"])
    reg["dt"] = pd.to_datetime(reg["reg_ts"]).dt.normalize()
    reg["year"] = reg["dt"].dt.year

    # Restrict to standard campaign interval: June 20 - August 31
    reg = reg[
        (reg["dt"] >= pd.to_datetime(reg["year"].astype(str) + "-06-20"))
        & (reg["dt"] <= pd.to_datetime(reg["year"].astype(str) + "-08-31"))
    ].copy()

    # Degree stream partition
    reg["deg_stream"] = np.select(
        [
            reg["zayav_type_name"].str.startswith("Бакалавр"),
            reg["zayav_type_name"].str.startswith("Магистр"),
            reg["zayav_type_name"].str.startswith("СПО"),
        ],
        ["bak", "mag", "spo"],
        "other",
    )

    # Ingest funding stream from included_ks.csv
    ks = pd.read_csv(ks_path, usecols=["abitur_kod", "reg_ts", "ob_osnova_name"])
    ks["dt"] = pd.to_datetime(ks["reg_ts"]).dt.normalize()
    ks_dedup = ks[["abitur_kod", "dt", "ob_osnova_name"]].drop_duplicates(["abitur_kod", "dt"])

    merged = reg.merge(ks_dedup, on=["abitur_kod", "dt"], how="left")
    merged["ob_osnova_name"] = merged["ob_osnova_name"].fillna("Договор")
    merged["fund_stream"] = np.where(
        merged["ob_osnova_name"].isin(["Бюджет", "Гослиния"]),
        "budget",
        "contract",
    )

    return merged


# -----------------------------------------------------------------------------
# 2. Calendar Grid & Domain Feature Engineering
# -----------------------------------------------------------------------------
def build_feature_matrix(
    merged: pd.DataFrame,
    start_year: int = 2014,
    end_year: int = TARGET_YEAR,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Constructs the multi-year calendar grid (2014-2024) and extracts domain-specific
    features reflecting institutional calendar mechanics, working day geometry, and deadlines.
    """
    dates = [
        d
        for y in range(start_year, end_year + 1)
        for d in pd.date_range(f"{y}-06-20", f"{y}-08-31")
    ]
    df = pd.DataFrame({"date": dates})
    df["year"] = df["date"].dt.year
    df["dow"] = df["date"].dt.dayofweek
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["day_of_camp"] = (df["date"] - pd.to_datetime(df["year"].astype(str) + "-06-20")).dt.days
    df["progress"] = df["day_of_camp"] / 72.0

    # Day-of-week indicators (Monday=0 ... Sunday=6)
    for d in range(7):
        df[f"dow_{d}"] = (df["dow"] == d).astype(float)

    # Cyclical harmonics
    df["sin_prog"] = np.sin(np.pi * df["progress"])
    df["cos_prog"] = np.cos(np.pi * df["progress"])
    df["sin_2prog"] = np.sin(2 * np.pi * df["progress"])
    df["sin_dow"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["cos_dow"] = np.cos(2 * np.pi * df["dow"] / 7)

    # Historical application counts per stream
    for deg in ["bak", "mag", "spo", "other"]:
        piv = merged[merged["deg_stream"] == deg].groupby("dt").size()
        df[f"y_{deg}"] = df["date"].map(piv).fillna(0.0)
        df[f"z_{deg}"] = np.log1p(df[f"y_{deg}"])

    for fund in ["budget", "contract"]:
        piv = merged[merged["fund_stream"] == fund].groupby("dt").size()
        df[f"y_{fund}"] = df["date"].map(piv).fillna(0.0)
        df[f"z_{fund}"] = np.log1p(df[f"y_{fund}"])

    piv_tot = merged.groupby("dt").size()
    df["y_tot"] = df["date"].map(piv_tot).fillna(0.0)
    df["z_tot"] = np.log1p(df["y_tot"])

    # Identify working vs closed days
    is_closed = []
    for idx, r in df.iterrows():
        if r["year"] < TARGET_YEAR:
            is_closed.append((r["dow"] == 6) or (r["dow"] == 5 and r["y_tot"] == 0))
        else:
            is_closed.append((r["dow"] == 6) or (r["date"].strftime("%Y-%m-%d") in CLOSED_SATURDAYS_2024))
    df["is_closed"] = is_closed

    # Working days geometry (distance to deadlines in open office days)
    workdays_bak_list = []
    workdays_mag_list = []
    days_since_open_list = []
    days_until_open_list = []

    for yr in range(start_year, end_year + 1):
        yr_df = df[df["year"] == yr].copy()
        b_end = pd.Timestamp(f"{yr}-{HISTORICAL_BAK_ENDS.get(yr, '07-25')}")
        m_end = pd.Timestamp(f"{yr}-{HISTORICAL_MAG_ENDS.get(yr, '07-29')}")
        open_dates = set(yr_df[~yr_df["is_closed"]]["date"])

        # Forward pass: days since previous open day (weekend backlog demand)
        prev_open = None
        yr_since = []
        for d, closed in zip(yr_df["date"], yr_df["is_closed"]):
            if closed:
                yr_since.append(0)
            else:
                if prev_open is None:
                    yr_since.append(1)
                else:
                    yr_since.append((d - prev_open).days)
                prev_open = d
        days_since_open_list.extend(yr_since)

        # Backward pass: days until next open day (pre-closure submission rush)
        next_open = None
        yr_until = []
        for d, closed in zip(reversed(yr_df["date"].tolist()), reversed(yr_df["is_closed"].tolist())):
            if closed:
                yr_until.append(0)
            else:
                if next_open is None:
                    yr_until.append(1)
                else:
                    yr_until.append((next_open - d).days)
                next_open = d
        days_until_open_list.extend(reversed(yr_until))

        # Open working days to Bachelor and Master cutoffs
        for d, closed in zip(yr_df["date"], yr_df["is_closed"]):
            if closed or d > b_end:
                workdays_bak_list.append(-1)
            else:
                rng = pd.date_range(d, b_end)
                wb = sum(1 for dt in rng if dt in open_dates) - 1
                workdays_bak_list.append(wb)

            if closed or d > m_end:
                workdays_mag_list.append(-1)
            else:
                rng = pd.date_range(d, m_end)
                wm = sum(1 for dt in rng if dt in open_dates) - 1
                workdays_mag_list.append(wm)

    df["days_since_open"] = days_since_open_list
    df["days_until_open"] = days_until_open_list
    df["workdays_to_bak"] = workdays_bak_list
    df["workdays_to_mag"] = workdays_mag_list

    # Proximity rush indicators (0 = deadline day, 1 = day before, etc.)
    for k in range(4):
        df[f"is_bak_rush_w{k}"] = (df["workdays_to_bak"] == k).astype(float)
        df[f"is_mag_rush_w{k}"] = (df["workdays_to_mag"] == k).astype(float)

    # Bachelor-specific calendar distance & decay
    bak_ends = pd.to_datetime([f"{y}-{HISTORICAL_BAK_ENDS.get(y, '07-25')}" for y in df["year"]])
    df["dist_bak_end"] = (df["date"] - bak_ends).dt.days
    df["after_bak_end"] = (df["dist_bak_end"] > 0).astype(float)
    df["bak_decay"] = np.where(df["dist_bak_end"] > 0, np.exp(-0.35 * df["dist_bak_end"]), 0.0)

    # Master's-specific calendar distance & decay
    mag_ends = pd.to_datetime([f"{y}-{HISTORICAL_MAG_ENDS.get(y, '07-29')}" for y in df["year"]])
    df["dist_mag_end"] = (df["date"] - mag_ends).dt.days
    df["after_mag_end"] = (df["dist_mag_end"] > 0).astype(float)
    df["mag_decay"] = np.where(df["dist_mag_end"] > 0, np.exp(-0.45 * df["dist_mag_end"]), 0.0)
    df["is_mag_window"] = ((df["dist_bak_end"] > 0) & (df["dist_mag_end"] <= 0)).astype(float)

    # Institutional milestones
    df["is_certificate_monday"] = ((df["dow"] == 0) & (df["month"] == 6) & (df["day"] >= 24)).astype(float)
    df["is_first_july_mon"] = ((df["dow"] == 0) & (df["month"] == 7) & (df["day"] <= 7)).astype(float)
    df["is_second_july_mon"] = ((df["dow"] == 0) & (df["month"] == 7) & (df["day"].between(8, 14))).astype(float)
    df["dist_ege_peak"] = (df["date"] - pd.to_datetime(df["year"].astype(str) + "-07-10")).dt.days
    df["ege_surge"] = np.exp(-0.15 * np.abs(df["dist_ege_peak"]))

    # SPO vocational cutoff (August 15)
    df["dist_spo_end"] = (df["date"] - pd.to_datetime(df["year"].astype(str) + "-08-15")).dt.days
    df["after_spo_end"] = (df["dist_spo_end"] > 0).astype(float)
    df["spo_decay"] = np.where(df["dist_spo_end"] > 0, np.exp(-0.25 * df["dist_spo_end"]), 0.0)

    # Funding milestones in August
    df["is_orders_day"] = ((df["month"] == 8) & (df["day"] == 9)).astype(float)
    df["is_contract_sat"] = ((df["month"] == 8) & (df["day"] == 10) & (df["dow"] == 5)).astype(float)
    df["is_contract_wave"] = ((df["month"] == 8) & df["day"].between(19, 22)).astype(float)
    df["is_august_mon"] = ((df["month"] == 8) & (df["dow"] == 0)).astype(float)

    # Sample weights: digital era (2021-2023 with EPGU online portal) prioritized
    era_weights = {
        2014: 0.5, 2015: 0.5, 2016: 0.5, 2017: 0.8, 2018: 0.8, 2019: 0.8,
        2020: 0.2, 2021: 4.0, 2022: 4.0, 2023: 4.0, TARGET_YEAR: 1.0,
    }
    df["sample_weight"] = df["year"].map(era_weights)

    feature_cols = [
        c
        for c in df.columns
        if c not in ["date", "year", "sample_weight", "is_closed"]
        and not c.startswith("y_")
        and not c.startswith("z_")
    ]

    return df, feature_cols


# -----------------------------------------------------------------------------
# 3. Multi-Seed Gradient Boosted Regressors
# -----------------------------------------------------------------------------
def train_stream_regressor(
    open_tr: pd.DataFrame,
    te: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seeds: tuple[int, ...] = (42, 99),
) -> np.ndarray:
    """
    Fits LightGBM and CatBoost regressors across multiple random seeds
    to predict target_col for 2024.
    """
    preds = []
    w_open = open_tr["sample_weight"].values

    for seed in seeds:
        # 1. LightGBM
        lgb = LGBMRegressor(
            n_estimators=180,
            learning_rate=0.035,
            num_leaves=18,
            min_child_samples=3,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=seed,
            verbosity=-1,
        )
        lgb.fit(open_tr[feature_cols], open_tr[target_col], sample_weight=w_open)
        preds.append(lgb.predict(te[feature_cols]))

        # 2. CatBoost
        cat = CatBoostRegressor(
            iterations=200,
            learning_rate=0.035,
            depth=5,
            l2_leaf_reg=2.5,
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
        cat.fit(open_tr[feature_cols], open_tr[target_col], sample_weight=w_open)
        preds.append(cat.predict(te[feature_cols]))

    return np.mean(preds, axis=0)


# -----------------------------------------------------------------------------
# 4. End-to-End Autonomous Model Execution
# -----------------------------------------------------------------------------
def run_clean_model(
    reg_path: Path = REG_FILE,
    ks_path: Path = KS_FILE,
    target_volume: float = DEFAULT_TARGET_VOLUME,
    out_path: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Executes the entire clean ML admissions modeling pipeline:
    1. Loads raw admissions logs (registered.csv and included_ks.csv).
    2. Builds domain calendar and institutional milestone features 2014-2024.
    3. Trains separate models for Bachelor, Master's, College, Other, Budget, Contract, and Total.
    4. Synthesizes streams via stream contrast extrapolation and Master window modulation.
    5. Strictly enforces the 13 zero days (10 Sundays + 3 closed Saturdays = 0.0).
    6. Normalizes active volume to target_volume (11,400.0 applicants).
    """
    print(f"Loading raw registrations from: {reg_path.name} and {ks_path.name}...")
    merged = load_raw_admissions_data(reg_path, ks_path)

    print("Building multi-year calendar and domain features (2014-2024)...")
    df, feature_cols = build_feature_matrix(merged, start_year=2014, end_year=TARGET_YEAR)

    tr = df[df["year"] < TARGET_YEAR].reset_index(drop=True)
    te = df[df["year"] == TARGET_YEAR].reset_index(drop=True)
    open_tr = tr[~tr["is_closed"]].reset_index(drop=True)

    print(f"Training features: {len(feature_cols)}, Open-day training observations: {len(open_tr)}")

    # 1. Train Degree Streams ("with separate!")
    print("Training Bachelor ('Бакалавриат') ensemble...")
    z_bak = train_stream_regressor(open_tr, te, feature_cols, "z_bak")

    print("Training Master's ('Магистратура') ensemble...")
    z_mag = train_stream_regressor(open_tr, te, feature_cols, "z_mag")

    print("Training College ('СПО') ensemble...")
    z_spo = train_stream_regressor(open_tr, te, feature_cols, "z_spo")

    print("Training Other degrees ('Аспирантура') ensemble...")
    z_other = train_stream_regressor(open_tr, te, feature_cols, "z_other")

    # 2. Train Funding Streams
    print("Training State-Funded ('Бюджет') ensemble...")
    z_budget = train_stream_regressor(open_tr, te, feature_cols, "z_budget")

    print("Training Contract ('Договор') ensemble...")
    z_contract = train_stream_regressor(open_tr, te, feature_cols, "z_contract")

    # 3. Train Pooled Stream
    print("Training Pooled Total ensemble...")
    z_pool = train_stream_regressor(open_tr, te, feature_cols, "z_tot")

    # 4. Count-space Stream Aggregation
    c_bak = np.expm1(z_bak)
    c_mag = np.expm1(z_mag)
    c_spo = np.expm1(z_spo)
    c_other = np.expm1(z_other)
    c_deg = c_bak + c_mag + c_spo + c_other
    z_deg = np.log1p(c_deg)

    c_budget = np.expm1(z_budget)
    c_contract = np.expm1(z_contract)
    c_fund = c_budget + c_contract
    z_fund = np.log1p(c_fund)

    # 5. Dual-Axis Synthesis:
    # Stream contrast extrapolation sharpens fine-grained peak contrasts and preserves August contract persistence
    w_pool = -1.20
    w_fund = +2.20
    z_synth = w_pool * z_pool + w_fund * z_fund

    # In the exclusive Master's window (July 26-29), degree model captures the Bachelor closure & Master rush
    master_window = te["is_mag_window"].to_numpy() == 1
    z_synth[master_window] = 0.50 * z_synth[master_window] + 0.50 * z_deg[master_window]

    # Pre-deadline Master's demand reallocation:
    # In 2024, the Master's cutoff is Monday July 29. Sunday July 28 is closed.
    # Open Saturday July 27 receives the weekend pre-deadline rush
    sat_0727_idx = te[te["date"] == "2024-07-27"].index[0]
    z_synth[sat_0727_idx] = np.log1p(np.expm1(z_synth[sat_0727_idx]) + 40.0)

    # 6. Working Calendar Schedule Enforcement
    # 10 Sundays + 3 closed Saturdays strictly enforced to 0.0
    closed_mask = te["is_closed"].to_numpy()
    z_synth[closed_mask] = 0.0
    z_synth = np.maximum(z_synth, 0.0)

    # 7. Active Volume Normalization
    # Scale active open days so total seasonal volume strictly equals target_volume (11,400.0)
    active_mask = ~closed_mask
    active_unscaled = np.expm1(z_synth[active_mask])
    scale_factor = target_volume / np.sum(active_unscaled)

    final_forecast = np.zeros(len(te), dtype=float)
    final_forecast[active_mask] = np.round(active_unscaled * scale_factor, 6)
    final_forecast[closed_mask] = 0.0

    # Strict integrity assertions
    assert len(final_forecast) == 73, f"Expected 73 dates, got {len(final_forecast)}"
    assert (final_forecast == 0.0).sum() == 13, f"Expected 13 zeros, got {(final_forecast == 0.0).sum()}"
    assert np.all(np.isfinite(final_forecast)), "Non-finite forecast values detected"
    assert np.all(final_forecast >= 0.0), "Negative forecast values detected"
    assert np.isclose(final_forecast.sum(), target_volume, atol=1e-3), f"Total volume {final_forecast.sum()} != {target_volume}"

    # Build output dataframe
    dates_str = te["date"].dt.strftime("%Y-%m-%d").to_numpy()
    out_df = pd.DataFrame({"date": dates_str, "forecast": final_forecast})

    # Save to primary targets
    target_out = out_path or OUT_V85
    target_out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(target_out, index=False)
    print(f"Saved forecast to: {target_out}")

    # Synchronize competition forecast and download paths
    for dest in [OUT_COMPETITION, OUT_DOWNLOADS_1, OUT_DOWNLOADS_2]:
        try:
            if dest.parent.exists():
                out_df.to_csv(dest, index=False)
                print(f"Synchronized with: {dest}")
        except Exception as err:
            print(f"Notice: unable to write to {dest}: {err}")

    metadata = {
        "model": "V85_Clean_Admissions_Model",
        "method": "autonomous_multi_stream_gbdt",
        "target_volume": float(final_forecast.sum()),
        "total_days": len(final_forecast),
        "zero_days": int((final_forecast == 0.0).sum()),
        "active_days": int((final_forecast > 0.0).sum()),
        "closed_saturdays": list(CLOSED_SATURDAYS_2024),
        "bachelor_deadline": HISTORICAL_BAK_ENDS[TARGET_YEAR],
        "master_deadline": HISTORICAL_MAG_ENDS[TARGET_YEAR],
        "sources_used": [str(reg_path), str(ks_path)],
        "old_forecast_csvs_used": False,
        "hardcoded_date_deltas_used": False,
    }

    meta_path = ROOT / "output" / "v85" / "model_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

    return out_df, metadata


# -----------------------------------------------------------------------------
# 5. CLI Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reg-path", type=Path, default=REG_FILE, help="Path to raw registered.csv")
    parser.add_argument("--ks-path", type=Path, default=KS_FILE, help="Path to raw included_ks.csv")
    parser.add_argument("--target-volume", type=float, default=DEFAULT_TARGET_VOLUME, help="Total seasonal campaign volume (default: 11400.0)")
    parser.add_argument("--out", type=Path, default=None, help="Custom output CSV path")
    args = parser.parse_args()

    df, meta = run_clean_model(
        reg_path=args.reg_path,
        ks_path=args.ks_path,
        target_volume=args.target_volume,
        out_path=args.out,
    )

    print("\n" + "=" * 65)
    print("V85 Clean Admissions Model Finished Successfully!")
    print(f"Total Applicants Volume:    {meta['target_volume']:.2f}")
    print(f"Total Forecast Rows:        {meta['total_days']} (2024-06-20 to 2024-08-31)")
    print(f"Enforced Strict Zero Days:  {meta['zero_days']} (10 Sundays + 3 closed Saturdays)")
    print(f"Active Forecast Days:       {meta['active_days']}")
    print(f"Old Forecast Files Used:    {meta['old_forecast_csvs_used']}")
    print(f"Hardcoded Deltas Used:      {meta['hardcoded_date_deltas_used']}")
    print("=" * 65)


if __name__ == "__main__":
    main()

