"""
V35: leaderboard-informed calibration of V34.

V34 improved the public score to 0.51655, but its total forecast was lower
than the historical campaign totals. On the 2023 holdout, the 15% analog
blend had its best RMSLE near a 1.20 multiplicative calibration. V35 uses
1.18 as a conservative compromise and leaves V34 unchanged.
"""

from pathlib import Path

import pandas as pd

from v34_analog_calibrated_engine import generate_v34_forecast


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "forecast_2024_v35_scaled_analog.csv"
CALIBRATION_SCALE = 1.18


def generate_v35_forecast() -> pd.DataFrame:
    result = generate_v34_forecast().copy()
    active = result["forecast"] > 0
    result.loc[active, "forecast"] = (
        result.loc[active, "forecast"] * CALIBRATION_SCALE
    ).round(2)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"V35 successfully generated: {OUTPUT_FILE}")
    print(f"Calibration scale: {CALIBRATION_SCALE}")
    print(f"Total days: {len(result)}")
    print(f"Total zeros: {(result['forecast'] == 0.0).sum()}")
    print(f"Total sum: {result['forecast'].sum():.2f}")
    return result


if __name__ == "__main__":
    generate_v35_forecast()
