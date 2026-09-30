"""
Script to reproduce Submission V10 (Public Score: 0.57961)
Target: registered.csv (Total applicants flow)
Architecture: LightGBM (3 seeds) + HistGradientBoosting (1 seed) [75% / 25%]
Features: Campaign calendar, harmonics, milestone distances, phase indicators, sat_phases
Floor: 1.00 on 13 quiet weekend dates, 0.55 on all other dates
Output: forecast_2024_v10_floor100.csv
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor

def generate_v10_forecast():
    # 1. Load data
    df_reg = pd.read_csv('processed_data/registered.csv')
    df_reg['date'] = pd.to_datetime(df_reg['reg_ts']).dt.date

    # 2. Build full daily grid (2014-2024, excluding 2020)
    rows = []
    for yr in range(2014, 2025):
        if yr == 2020: 
            continue
        dates = pd.date_range(f'{yr}-06-20', f'{yr}-08-31').date
        for d in dates:
            rows.append({'year': yr, 'date': d})
    daily = pd.DataFrame(rows)

    actuals_df = df_reg.groupby('date').size().reset_index(name='actual_count')
    daily = pd.merge(daily, actuals_df, on='date', how='left').fillna(0.0)

    # 3. Calendar & phase features
    daily['date'] = pd.to_datetime(daily['date'])
    daily['year'] = daily['date'].dt.year
    daily['dow'] = daily['date'].dt.dayofweek
    daily['is_weekend'] = (daily['dow'] >= 5).astype(int)
    daily['is_saturday'] = (daily['dow'] == 5).astype(int)
    daily['is_sunday'] = (daily['dow'] == 6).astype(int)
    daily['day'] = daily['date'].dt.day
    daily['month'] = daily['date'].dt.month
    daily['iso_week'] = daily['date'].dt.isocalendar().week.astype(int)

    june20 = pd.to_datetime(daily['year'].astype(str) + '-06-20')
    daily['day_of_camp'] = (daily['date'] - june20).dt.days
    daily['progress'] = daily['day_of_camp'] / 72.0

    for ms, ms_name in [('06-20', 'start'), ('07-08', 'peak'), ('07-25', 'deadline'), ('08-03', 'consent'), ('08-09', 'orders'), ('08-31', 'end')]:
        ms_date = pd.to_datetime(daily['year'].astype(str) + '-' + ms)
        daily['dist_' + ms_name] = (daily['date'] - ms_date).dt.days

    daily['is_first_week'] = ((daily['month'] == 6) & (daily['day'] <= 25)).astype(int)
    daily['is_peak_period'] = ((daily['month'] == 7) & (daily['day'] >= 1) & (daily['day'] <= 15)).astype(int)
    daily['is_pre_deadline_3d'] = ((daily['month'] == 7) & (daily['day'] >= 22) & (daily['day'] <= 25)).astype(int)
    daily['is_deadline_day'] = ((daily['month'] == 7) & (daily['day'] == 25)).astype(int)
    daily['is_august'] = (daily['month'] == 8).astype(int)

    # Differentiated Saturday features across campaign phases
    daily['sat_june'] = ((daily['dow'] == 5) & (daily['month'] == 6)).astype(int)
    daily['sat_july_peak'] = ((daily['dow'] == 5) & (daily['month'] == 7) & (daily['day'] <= 20)).astype(int)
    daily['sat_july_late'] = ((daily['dow'] == 5) & (daily['month'] == 7) & (daily['day'] > 20)).astype(int)
    daily['sat_aug'] = ((daily['dow'] == 5) & (daily['month'] == 8)).astype(int)

    # Cyclic harmonics
    daily['sin_prog'] = np.sin(np.pi * daily['progress'])
    daily['sin_2prog'] = np.sin(2 * np.pi * daily['progress'])
    daily['cos_prog'] = np.cos(np.pi * daily['progress'])
    daily['sin_dow'] = np.sin(2 * np.pi * daily['dow'] / 7)
    daily['cos_dow'] = np.cos(2 * np.pi * daily['dow'] / 7)

    drop_cols = ['year', 'date', 'actual_count', 'target_log']
    feats = [c for c in daily.columns if c not in drop_cols]

    # 4. Train on < 2024, predict 2024
    train = daily[daily['year'] < 2024].copy().reset_index(drop=True)
    train['target_log'] = np.log1p(train['actual_count'])
    test_2024 = daily[daily['year'] == 2024].copy().reset_index(drop=True)

    # Recency weights
    w = np.where(train['year'] >= 2021, 2.5, np.where(train['year'] >= 2017, 1.5, 0.7))

    # LightGBM (3 diverse seeds)
    preds_lgb = []
    for s in range(3):
        m = lgb.LGBMRegressor(
            n_estimators=130, learning_rate=0.035, num_leaves=16,
            min_child_samples=4, subsample=0.85, colsample_bytree=0.85,
            random_state=s * 17 + 42, verbosity=-1
        )
        m.fit(train[feats], train['target_log'], sample_weight=w)
        preds_lgb.append(np.expm1(m.predict(test_2024[feats])))
    p_lgb = np.mean(preds_lgb, axis=0)

    # HistGradientBoosting (1 seed)
    m2 = HistGradientBoostingRegressor(
        max_iter=130, learning_rate=0.035, max_leaf_nodes=16,
        min_samples_leaf=4, random_state=42
    )
    m2.fit(train[feats], train['target_log'], sample_weight=w)
    p_hgb = np.expm1(m2.predict(test_2024[feats]))

    p_base = 0.75 * p_lgb + 0.25 * p_hgb

    # 5. Weekend Floor:
    # 13 quiet weekend days (Sundays + 2024-06-22, 2024-08-24, 2024-08-31) set to 1.00
    # All other days clipped to 0.55
    mask_13 = (test_2024['is_sunday'] == 1) | (test_2024['date'].dt.strftime('%Y-%m-%d').isin(['2024-06-22', '2024-08-24', '2024-08-31']))
    p_v10 = np.where(mask_13, 1.00, np.clip(p_base, 0.55, None))

    sub = pd.DataFrame({
        'date': test_2024['date'].dt.strftime('%Y-%m-%d'),
        'forecast': np.round(p_v10, 2)
    })
    sub.to_csv('forecast_2024_v10_floor100.csv', index=False)
    print("V10 generated successfully! Sum:", sub['forecast'].sum())
    return sub

if __name__ == '__main__':
    generate_v10_forecast()
