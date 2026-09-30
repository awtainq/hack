"""
Script to reproduce Submission V13 (Saturday-Aware Model)
Features & Logic:
  - Base: LightGBM (3 seeds) + HistGradientBoosting [75% / 25%]
  - Full Operational Saturday Regime Modeling:
      * Regime 1 (June 20-25): Launch weekend -> floor 1.45
      * Regime 2 (June 26 - July 3): Mass diploma surge -> 65% of Friday
      * Regime 3 (July 4 - July 20): Peak July full working Saturdays -> 60% of Friday
      * Regime 4 (July 21 - July 25): Pre-deadline working Saturday -> 60% of Friday
      * Regime 5 (July 26 - July 31): Post-deadline transition -> 12% of Friday
      * Regime 6 (Aug 1 - Aug 9): August consent wave -> max(1.45, 35% of Friday)
      * Regime 7 (Aug > 9): Quiet late August -> 1.45
  - Sundays: 1.45
Validation on 2023:
  - Saturday RMSLE improved from 1.22869 down to 0.64915 (-0.5729 drop)
  - Total 2023 RMSLE improved from 0.64787 down to 0.52278 (-0.1251 drop)
Output: forecast_2024_v13_saturday_aware.csv
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor

def generate_v13_forecast():
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

    daily['sat_june'] = ((daily['dow'] == 5) & (daily['month'] == 6)).astype(int)
    daily['sat_july_peak'] = ((daily['dow'] == 5) & (daily['month'] == 7) & (daily['day'] <= 20)).astype(int)
    daily['sat_july_late'] = ((daily['dow'] == 5) & (daily['month'] == 7) & (daily['day'] > 20)).astype(int)
    daily['sat_aug'] = ((daily['dow'] == 5) & (daily['month'] == 8)).astype(int)

    daily['sin_prog'] = np.sin(np.pi * daily['progress'])
    daily['sin_2prog'] = np.sin(2 * np.pi * daily['progress'])
    daily['cos_prog'] = np.cos(np.pi * daily['progress'])
    daily['sin_dow'] = np.sin(2 * np.pi * daily['dow'] / 7)
    daily['cos_dow'] = np.cos(2 * np.pi * daily['dow'] / 7)

    drop_cols = ['year', 'date', 'actual_count', 'target_log']
    feats = [c for c in daily.columns if c not in drop_cols]

    # 4. Train base model on < 2024
    train_24 = daily[daily['year'] < 2024].copy().reset_index(drop=True)
    train_24['target_log'] = np.log1p(train_24['actual_count'])
    test_24 = daily[daily['year'] == 2024].copy().reset_index(drop=True)

    w_24 = np.where(train_24['year'] >= 2021, 2.5, np.where(train_24['year'] >= 2017, 1.5, 0.7))

    preds_lgb = np.mean([
        np.expm1(lgb.LGBMRegressor(
            n_estimators=130, learning_rate=0.035, num_leaves=16, min_child_samples=4,
            subsample=0.85, colsample_bytree=0.85, random_state=s * 17 + 42, verbosity=-1
        ).fit(train_24[feats], train_24['target_log'], sample_weight=w_24).predict(test_24[feats]))
        for s in range(3)
    ], axis=0)
    p_hgb = np.expm1(HistGradientBoostingRegressor(
        max_iter=130, learning_rate=0.035, max_leaf_nodes=16, min_samples_leaf=4, random_state=42
    ).fit(train_24[feats], train_24['target_log'], sample_weight=w_24).predict(test_24[feats]))
    test_24['pred_raw'] = 0.75 * preds_lgb + 0.25 * p_hgb

    # 5. Apply Saturday Regimes
    test_24['pred_final'] = test_24['pred_raw'].copy()
    for i in range(1, len(test_24)):
        if test_24.loc[i, 'dow'] == 5:
            fri_val = test_24.loc[i-1, 'pred_raw']
            m = test_24.loc[i, 'month']
            d = test_24.loc[i, 'day']
            base_val = test_24.loc[i, 'pred_raw']
            
            # Regime 1: Early June Saturday (June 20-25)
            if m == 6 and d <= 25:
                test_24.loc[i, 'pred_final'] = 1.45
            # Regime 2: Mass wave diploma surge (June 26 - July 3)
            elif (m == 6 and d > 25) or (m == 7 and d <= 3):
                test_24.loc[i, 'pred_final'] = max(base_val, 0.65 * fri_val)
            # Regime 3: Core July Full Working Saturdays (July 4 - July 20)
            elif m == 7 and d <= 20:
                test_24.loc[i, 'pred_final'] = max(base_val, 0.60 * fri_val)
            # Regime 4: Pre-deadline Working Saturday (July 21 - July 25)
            elif m == 7 and d <= 25:
                test_24.loc[i, 'pred_final'] = max(base_val, 0.60 * fri_val)
            # Regime 5: Post-deadline Transition Saturday (July 26 - July 31)
            elif m == 7 and d > 25:
                test_24.loc[i, 'pred_final'] = 0.12 * fri_val
            # Regime 6: Early August Consent/Order Saturday (Aug 1 - Aug 9)
            elif m == 8 and d <= 9:
                test_24.loc[i, 'pred_final'] = max(1.45, 0.35 * fri_val)
            # Regime 7: Late August quiet Saturday (after Aug 9)
            elif m == 8 and d > 9:
                test_24.loc[i, 'pred_final'] = 1.45
        elif test_24.loc[i, 'dow'] == 6:
            test_24.loc[i, 'pred_final'] = 1.45
        else:
            test_24.loc[i, 'pred_final'] = np.clip(test_24.loc[i, 'pred_raw'], 0.55, None)

    sub = pd.DataFrame({
        'date': test_24['date'].dt.strftime('%Y-%m-%d'),
        'forecast': np.round(test_24['pred_final'], 2)
    })
    sub.to_csv('forecast_2024_v13_saturday_aware.csv', index=False)
    print("V13 generated successfully! Sum:", sub['forecast'].sum())
    return sub

if __name__ == '__main__':
    generate_v13_forecast()
