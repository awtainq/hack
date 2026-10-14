import pandas as pd
import numpy as np

df_ks = pd.read_csv('processed_data/included_ks.csv')
df_ks['reg_ts'] = pd.to_datetime(df_ks['reg_ts'])
df_ks['date'] = df_ks['reg_ts'].dt.date
df_ks['year'] = df_ks['reg_ts'].dt.year
df_ks['month_day'] = df_ks['reg_ts'].dt.strftime('%m-%d')

print("=== START AND END DATES BY YEAR ===")
summary = []
for yr, grp in df_ks.groupby('year'):
    daily_counts = grp.groupby('date').size()
    # Find active period (e.g. days with at least 5 applications)
    active_days = daily_counts[daily_counts >= 5]
    summary.append({
        'year': yr,
        'min_date': grp['date'].min(),
        'max_date': grp['date'].max(),
        'active_min': active_days.index.min() if len(active_days) else None,
        'active_max': active_days.index.max() if len(active_days) else None,
        'total_rows': len(grp),
        'total_budget': (grp['ob_osnova_name'] == 'Бюджет').sum(),
        'total_contract': (grp['ob_osnova_name'] == 'Договор').sum(),
        'total_goslin': (grp['ob_osnova_name'] == 'Гослиния').sum()
    })
print(pd.DataFrame(summary).to_string())

print("\n=== DATE OF PEAK APPLICATIONS EACH YEAR ===")
for yr, grp in df_ks.groupby('year'):
    daily_counts = grp.groupby('date').size()
    peak_date = daily_counts.idxmax()
    peak_val = daily_counts.max()
    print(f"Year {yr}: Peak on {peak_date} ({peak_val} apps)")

