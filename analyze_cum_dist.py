import pandas as pd
import numpy as np

df_ks = pd.read_csv('processed_data/included_ks.csv')
df_ks['reg_ts'] = pd.to_datetime(df_ks['reg_ts'])
df_ks['year'] = df_ks['reg_ts'].dt.year
df_ks['date'] = df_ks['reg_ts'].dt.date

# Look at June 20 - August 31 for each year (excluding 2020 covid anomaly)
normal_years = [2017, 2018, 2019, 2021, 2022, 2023]
for yr in normal_years:
    df_yr = df_ks[df_ks['year'] == yr]
    june20 = pd.Timestamp(f'{yr}-06-20').date()
    df_main = df_yr[df_yr['date'] >= june20]
    total = len(df_main)
    # Check cumulative percentages
    daily = df_main.groupby('date').size()
    cum = daily.cumsum() / total
    
    # Days to 25%, 50%, 75%, 90%
    d25 = (cum >= 0.25).idxmax() - june20
    d50 = (cum >= 0.50).idxmax() - june20
    d75 = (cum >= 0.75).idxmax() - june20
    d90 = (cum >= 0.90).idxmax() - june20
    print(f"Year {yr}: Total={total} | 25% at day +{d25.days} | 50% at day +{d50.days} | 75% at day +{d75.days} | 90% at day +{d90.days}")

