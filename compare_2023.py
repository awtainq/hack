import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path("/Users/vld/python/hack")
sys.path.insert(0, str(ROOT / "reproducers"))
import v34_analog_calibrated_engine as v34_mod

raw = pd.read_csv(v34_mod.DATA_FILE)
df_all = v34_mod.make_daily(raw)

df_2023 = df_all[df_all["year"] == 2023].sort_values("date").reset_index(drop=True)
y_2023 = df_2023["actual_count"].values
dates_2023 = df_2023["date"].dt.strftime("%Y-%m-%d").values
dows_2023 = df_2023["dow"].values

v30 = pd.read_csv(ROOT / "forecast_2024_v30_optimal_blend.csv")["forecast"].values
v34 = pd.read_csv(ROOT / "forecast_2024_v34_analog_calibrated.csv")["forecast"].values
v44 = pd.read_csv(ROOT / "forecast_2024_v44_calibrated_v34_sub50.csv")["forecast"].values
v45 = pd.read_csv(ROOT / "forecast_2024_v45_minimax_sub50.csv")["forecast"].values
dates_2024 = pd.read_csv(ROOT / "forecast_2024_v45_minimax_sub50.csv")["date"].values
dows_2024 = pd.to_datetime(dates_2024).dayofweek.values

def rmsle(y_true, y_pred):
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))

# 1. Compare Saturdays of 2023 (and 2022/2021) vs V30, V34, V44, V45
print("=== 1. CAMPAIGN SATURDAYS: 2021-2023 Actuals vs V30, V34, V44, V45 ===")
sats_23 = df_2023[df_2023["dow"] == 5][["date", "day_of_camp", "actual_count"]].reset_index(drop=True)
df_2022 = df_all[df_all["year"] == 2022].sort_values("date").reset_index(drop=True)
sats_22 = df_2022[df_2022["dow"] == 5][["date", "day_of_camp", "actual_count"]].reset_index(drop=True)
sats_24_idx = [t for t in range(73) if dows_2024[t] == 5]
print("Week | 2022 Sat | 2023 Sat Date (Act) | 2024 Sat Date |   V30  |   V34  |  V44 |  V45")
for w in range(min(len(sats_23), len(sats_24_idx))):
    t24 = sats_24_idx[w]
    a22 = sats_22.loc[w, "actual_count"] if w < len(sats_22) else np.nan
    print(f"W{w+1:2d}  |   {a22:5.0f}  | {sats_23.loc[w, 'date'].strftime('%Y-%m-%d')} ({sats_23.loc[w, 'actual_count']:4.0f})   |  {dates_2024[t24]}   | {v30[t24]:6.2f} | {v34[t24]:6.2f} | {v44[t24]:4.0f} | {v45[t24]:4.0f}")

print("\n=== 2. PHASE TOTALS: 2023 & 2022 Actuals vs V30, V34, V44, V45 ===")
phases = [
    ("June (t=0..10)", range(0, 11)),
    ("Early July Peak (t=11..24)", range(11, 25)),
    ("Pre-Deadline July (t=25..36)", range(25, 37)),
    ("Post-Deadline Late July (t=37..41)", range(37, 42)),
    ("Early August (t=42..55)", range(42, 56)),
    ("Late August (t=56..72)", range(56, 73)),
    ("TOTAL (t=0..72)", range(0, 73)),
]
y_2022 = df_2022["actual_count"].values
print(f"{'Phase':35s} | {'2023 Act':>8s} | {'2022 Act':>8s} | {'V30':>8s} | {'V34':>8s} | {'V44':>8s} | {'V45':>8s}")
for pname, prange in phases:
    idx = list(prange)
    print(f"{pname:35s} | {np.sum(y_2023[idx]):8.0f} | {np.sum(y_2022[idx]):8.0f} | {np.sum(v30[idx]):8.0f} | {np.sum(v34[idx]):8.0f} | {np.sum(v44[idx]):8.0f} | {np.sum(v45[idx]):8.0f}")

# 3. Align each 2024 day t with the closest day in 2023 and 2022 having the SAME day-of-week:
y23_dow_aligned = np.zeros(73)
y22_dow_aligned = np.zeros(73)
dows_2022 = df_2022["dow"].values
for t in range(73):
    dow = dows_2024[t]
    cands23 = [k for k in range(73) if dows_2023[k] == dow]
    k23 = min(cands23, key=lambda k: abs(k - t))
    y23_dow_aligned[t] = y_2023[k23]
    cands22 = [k for k in range(73) if dows_2022[k] == dow]
    k22 = min(cands22, key=lambda k: abs(k - t))
    y22_dow_aligned[t] = y_2022[k22]

closed_24 = (v34 == 0)
y23_dow_active = y23_dow_aligned.copy()
y23_dow_active[closed_24] = 0.0
y22_dow_active = y22_dow_aligned.copy()
y22_dow_active[closed_24] = 0.0
y_22_23_blend = np.expm1(0.5 * np.log1p(y23_dow_active) + 0.5 * np.log1p(y22_dow_active))

print("\n=== 3. RMSLE & MAE AGAINST DOW-ALIGNED 2023 (and 2022/2023 Blend) ===")
for name, pred in [("V30 (0.53135)", v30), ("V34 (0.51655)", v34), ("V44 (0.51160)", v44), ("V45 (new)", v45)]:
    r23 = rmsle(y23_dow_active, pred)
    r22 = rmsle(y22_dow_active, pred)
    r_blend = rmsle(y_22_23_blend, pred)
    mae23 = np.mean(np.abs(y23_dow_active - pred))
    r_sat = rmsle(y23_dow_active[sats_24_idx], pred[sats_24_idx])
    r_wkd = rmsle(y23_dow_active[dows_2024 < 5], pred[dows_2024 < 5])
    print(f"{name:15s} | 2023 RMSLE: {r23:.5f} (Wkd: {r_wkd:.5f}, Sat: {r_sat:.5f}, MAE: {mae23:5.2f}) | 22/23 Blend RMSLE: {r_blend:.5f}")

