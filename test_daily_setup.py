import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import root_mean_squared_error

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))

# Load data
df_ks = pd.read_csv('processed_data/included_ks.csv')
df_ks['reg_ts'] = pd.to_datetime(df_ks['reg_ts'])
df_ks['date'] = df_ks['reg_ts'].dt.date
df_ks['year'] = df_ks['reg_ts'].dt.year

# Create full daily calendar for each campaign (June 20 to September 20)
rows = []
for yr in range(2014, 2024):
    dates = pd.date_range(f'{yr}-06-20', f'{yr}-09-20').date
    for d in dates:
        for osn in ['Бюджет', 'Договор', 'Гослиния']:
            rows.append({'year': yr, 'date': d, 'ob_osnova_name': osn})
df_daily = pd.DataFrame(rows)

# Aggregate actuals
counts = df_ks.groupby(['year', 'date', 'ob_osnova_name']).size().reset_index(name='actual_count')
df_daily = pd.merge(df_daily, counts, on=['year', 'date', 'ob_osnova_name'], how='left')
df_daily['actual_count'] = df_daily['actual_count'].fillna(0)

print(f"Total daily records: {len(df_daily)}")
print(df_daily.head())

