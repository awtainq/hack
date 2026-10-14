"""
Script to reproduce Submission V28 (Zero-Sunday & Modern 2023-Aligned Engine)
Target: registered.csv (Total applicants flow)

Key Innovations based on 2023 Campaign Analysis & User Hypothesis:
  1. Zero Sunday Strategy:
     - In 2023, 7/10 Sundays had exactly 0 registrations (2022 had 8/10 zeros).
     - Predicting 1.00 on zero days created an artificial RMSLE penalty of ~0.04 - 0.06.
     - Setting all 10 Sundays in 2024 to exactly 0.0 eliminates this squared log error penalty.
  2. Late August Saturday Shutdown:
     - Admissions commission office is closed on late August Saturdays (Aug 24, Aug 31).
     - Set to 0.0 (strictly 0 in 2022 and 2023).
  3. Monday Weekend Backlog Surge:
     - Monday absorbs all accumulated weekend demand, creating the largest volume day of each week (mean 655 in July 2023, peak 803 on July 10).
     - Explicit feature `mon_july` captures this operational dynamic.
  4. Post-Graduation Diploma Wave (Week 26):
     - `is_diploma_week` (June 24-30) models the surge following diploma issuance.
  5. Regularized Tri-Ensemble Core (Proven V24 architecture):
     - LightGBM (5 diverse seeds, num_leaves=14, colsample=0.90, weight 70%)
     - HistGradientBoosting (max_leaf_nodes=14, weight 20%)
     - CatBoost (depth=4, l2_leaf_reg=4, weight 10%)
  6. Organic Data-Driven Peak:
     - No artificial peak hacking. Deadline peak is organically modeled at ~240-250.

Output: forecast_2024_v28_zero_sunday.csv
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

def generate_v28_forecast():
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
    daily['is_monday'] = (daily['dow'] == 0).astype(int)
    daily['day'] = daily['date'].dt.day
    daily['month'] = daily['date'].dt.month
    daily['iso_week'] = daily['date'].dt.isocalendar().week.astype(int)

    june20 = pd.to_datetime(daily['year'].astype(str) + '-06-20')
    daily['day_of_camp'] = (daily['date'] - june20).dt.days
    daily['progress'] = daily['day_of_camp'] / 72.0

    # Signed distances to regulatory milestones
    for ms, ms_name in [('06-20', 'start'), ('07-08', 'peak'), ('07-25', 'deadline'), ('08-03', 'consent'), ('08-09', 'orders'), ('08-31', 'end')]:
        ms_date = pd.to_datetime(daily['year'].astype(str) + '-' + ms)
        daily['dist_' + ms_name] = (daily['date'] - ms_date).dt.days

    # Operational campaign phases
    daily['is_first_week'] = ((daily['month'] == 6) & (daily['day'] <= 23)).astype(int)
    daily['is_diploma_week'] = ((daily['month'] == 6) & (daily['day'] >= 24)).astype(int)
    daily['is_peak_period'] = ((daily['month'] == 7) & (daily['day'] >= 1) & (daily['day'] <= 15)).astype(int)
    daily['is_pre_deadline_3d'] = ((daily['month'] == 7) & (daily['day'] >= 22) & (daily['day'] <= 25)).astype(int)
    daily['is_deadline_day'] = ((daily['month'] == 7) & (daily['day'] == 25)).astype(int)
    daily['is_post_deadline'] = ((daily['month'] == 7) & (daily['day'] > 25)).astype(int)
    daily['is_august'] = (daily['month'] == 8).astype(int)

    # Operational staffing features
    daily['sat_june'] = ((daily['dow'] == 5) & (daily['month'] == 6)).astype(int)
    daily['sat_july_peak'] = ((daily['dow'] == 5) & (daily['month'] == 7) & (daily['day'] <= 20)).astype(int)
    daily['sat_july_late'] = ((daily['dow'] == 5) & (daily['month'] == 7) & (daily['day'] > 20)).astype(int)
    daily['sat_aug'] = ((daily['dow'] == 5) & (daily['month'] == 8)).astype(int)
    daily['mon_july'] = ((daily['dow'] == 0) & (daily['month'] == 7)).astype(int)

    # Cyclic progression & day of week harmonics
    daily['sin_prog'] = np.sin(np.pi * daily['progress'])
    daily['sin_2prog'] = np.sin(2 * np.pi * daily['progress'])
    daily['cos_prog'] = np.cos(np.pi * daily['progress'])
    daily['sin_dow'] = np.sin(2 * np.pi * daily['dow'] / 7)
    daily['cos_dow'] = np.cos(2 * np.pi * daily['dow'] / 7)

    drop_cols = ['year', 'date', 'actual_count', 'target_log']
    feats = [c for c in daily.columns if c not in drop_cols]

    # 4. Train on < 2024
    train = daily[daily['year'] < 2024].copy().reset_index(drop=True)
    train['target_log'] = np.log1p(train['actual_count'])
    test_2024 = daily[daily['year'] == 2024].copy().reset_index(drop=True)

    # Weight modern era higher
    w = np.where(train['year'] >= 2021, 2.5, np.where(train['year'] >= 2017, 1.5, 0.7))

    # Component 1: LightGBM (5 diverse seeds, num_leaves=14, colsample=0.90)
    preds_lgb = []
    for s in range(5):
        m = lgb.LGBMRegressor(
            n_estimators=130, learning_rate=0.035, num_leaves=14,
            min_child_samples=4, subsample=0.85, colsample_bytree=0.90,
            random_state=s * 17 + 42, verbosity=-1
        )
        m.fit(train[feats], train['target_log'], sample_weight=w)
        preds_lgb.append(np.expm1(m.predict(test_2024[feats])))
    p_lgb = np.mean(preds_lgb, axis=0)

    # Component 2: HistGradientBoosting (1 seed, max_leaf_nodes=14)
    m_hgb = HistGradientBoostingRegressor(
        max_iter=130, learning_rate=0.035, max_leaf_nodes=14, min_samples_leaf=4, random_state=42
    )
    m_hgb.fit(train[feats], train['target_log'], sample_weight=w)
    p_hgb = np.expm1(m_hgb.predict(test_2024[feats]))

    # Component 3: CatBoost (depth=4, l2_leaf_reg=4)
    m_cb = CatBoostRegressor(
        iterations=140, learning_rate=0.035, depth=4, l2_leaf_reg=4, random_seed=42, verbose=0
    )
    m_cb.fit(train[feats], train['target_log'], sample_weight=w)
    p_cb = np.expm1(m_cb.predict(test_2024[feats]))

    # Tri-Blend: 70% LightGBM + 20% HistGBM + 10% CatBoost
    p_blend = 0.70 * p_lgb + 0.20 * p_hgb + 0.10 * p_cb

    # 5. Zero Sunday & Quiet Saturday Override:
    # Set all 10 Sundays to 0.0 (closed admissions office, zero applicant processing)
    # Set late August Saturdays (2024-08-24, 2024-08-31) to 0.0
    # Set first opening Saturday (2024-06-22) to 0.0
    mask_zero = (test_2024['is_sunday'] == 1) | (test_2024['date'].dt.strftime('%Y-%m-%d').isin(['2024-06-22', '2024-08-24', '2024-08-31']))
    p_v28 = np.where(mask_zero, 0.0, np.clip(p_blend, 0.55, None))

    sub = pd.DataFrame({
        'date': test_2024['date'].dt.strftime('%Y-%m-%d'),
        'forecast': np.round(p_v28, 2)
    })
    out_file = 'forecast_2024_v28_zero_sunday.csv'
    sub.to_csv(out_file, index=False)
    print(f"V28 successfully generated: {out_file}")
    print(f"Total days: {len(sub)}")
    print(f"Total zeros: {(sub['forecast'] == 0.0).sum()}")
    print(f"Total sum: {sub['forecast'].sum():.2f}")
    print(f"Peak day (2024-07-08): {sub.loc[sub['date']=='2024-07-08', 'forecast'].values[0]:.2f}")
    print(f"Deadline day (2024-07-25): {sub.loc[sub['date']=='2024-07-25', 'forecast'].values[0]:.2f}")
    return sub

if __name__ == '__main__':
    generate_v28_forecast()

