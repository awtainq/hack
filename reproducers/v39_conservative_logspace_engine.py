"""
V39: conservative rollback of V38.

V38 increased the 2024 campaign total to about 12.9k and scored 0.55 on the
public leaderboard. V39 keeps the log-space blend but removes the unverified
global scale that inflated the July peak.
"""

from pathlib import Path

import pandas as pd

from v38_logspace_analog_engine import generate_v38_forecast


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILE = ROOT / "forecast_2024_v39_conservative_logspace.csv"
V38_SCALE = 1.12


def generate_v39_forecast() -> pd.DataFrame:
    result = generate_v38_forecast().copy()
    active = result["forecast"] > 0
    result.loc[active, "forecast"] = (
        result.loc[active, "forecast"] / V38_SCALE
    ).round(2)
    result.to_csv(OUTPUT_FILE, index=False)
    print(f"V39 successfully generated: {OUTPUT_FILE}")
    print("Removed V38 global calibration scale: 1.12 -> 1.00")
    print(f"Total days: {len(result)}")
    print(f"Total zeros: {(result['forecast'] == 0.0).sum()}")
    print(f"Total sum: {result['forecast'].sum():.2f}")
    print(
        "Peak day (2024-07-08): "
        f"{result.loc[result['date'] == '2024-07-08', 'forecast'].iloc[0]:.2f}"
    )
    return result


if __name__ == "__main__":
    generate_v39_forecast()
