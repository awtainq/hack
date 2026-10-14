import pandas as pd
import numpy as np

df_ks = pd.read_csv('processed_data/included_ks.csv')
df_ks['reg_ts'] = pd.to_datetime(df_ks['reg_ts'])
df_ks['year'] = df_ks['reg_ts'].dt.year
df_ks['date'] = df_ks['reg_ts'].dt.date
df_ks['hour'] = df_ks['reg_ts'].dt.hour
df_ks['dayofweek'] = df_ks['reg_ts'].dt.dayofweek
df_ks['dayofyear'] = df_ks['reg_ts'].dt.dayofyear

print("=== CAMPAIGN DATE RANGES BY YEAR ===")
for yr, grp in df_ks.groupby('year'):
    print(f"Year {yr}: {grp['reg_ts'].min()} to {grp['reg_ts'].max()} | Days: {grp['date'].nunique()} | Rows: {len(grp)} | Abitur: {grp['abitur_kod'].nunique()}")

print("\n=== OSNOVA COUNTS BY YEAR ===")
osnova_by_yr = pd.crosstab(df_ks['year'], df_ks['ob_osnova_name'])
print(osnova_by_yr)

print("\n=== PRIEM DOC TYPE BY YEAR ===")
priem_by_yr = pd.crosstab(df_ks['year'], df_ks['priem_doc_type_name'])
print(priem_by_yr)

print("\n=== ZAYAV TYPE BY YEAR ===")
zayav_by_yr = pd.crosstab(df_ks['year'], df_ks['zayav_type_name'])
print(zayav_by_yr)

print("\n=== HOURLY DISTRIBUTION ===")
hourly = df_ks['hour'].value_counts().sort_index()
print(hourly)

print("\n=== DAY OF WEEK DISTRIBUTION (0=Mon, 6=Sun) ===")
dow = df_ks['dayofweek'].value_counts().sort_index()
print(dow)

print("\n=== SAMPLE DAILY TIME SERIES (2022 & 2023) ===")
daily = df_ks.groupby(['year', 'date', 'ob_osnova_name']).size().unstack(fill_value=0)
print(daily.tail(20))

