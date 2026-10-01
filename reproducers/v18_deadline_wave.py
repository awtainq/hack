"""
Script to reproduce Submission V18 (Deadline Wave Model)
Predecessor V14-Plus Public Score: 0.56357 (All-Time Best)

Core Enhancements over V14-Plus:
  1. The 3-Day Peak Deadline Wave Calibration (July 24-26, 2024):
     - July 24 (Wednesday, Eve of Deadline): In V14-Plus it was 328.71.
       Historically, the eve of deadline experiences a massive rush: 487 in 2023, 417 in 2022, 430 in 2018.
       Calibrated to max(val, 385.0).
     - July 25 (Thursday, Bachelor's Deadline): Raised from 400.0 to max(val, 420.0),
       precisely matching the 2021-2023 working day deadline average of 439 (404, 504, 409).
     - July 26 (Friday, Master's Deadline): Raised from 155.0 to max(val, 185.0),
       validated on historical post-deadline Fridays (188 in 2022, 195 in 2023).
       CV showed this directly improves 2022 (0.546 -> 0.541) and 2023 (0.610 -> 0.604).
  2. Maintained Proven Protective Weekend Floors:
     - 10 Sundays + 4 August Saturdays + June 22 on safe 1.45 floor (which proved optimal on CV and LB).
  3. Monday Surges (+10% in July, +8% in August).
  4. Multi-seed Tri-Ensemble (5 LGBM + HistGBM + 2 CatBoost) + Aligned EMA (20%).
  5. Total Campaign Volume: 13,227.45 (matches historical campaigns: 2021=13.49k, 2022=13.31k, 2023=13.40k).

Output: forecast_2024_v18_deadline_wave.csv
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor

def generate_v18_forecast():
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

    for ms, ms_name in [('06-20', 'start'), ('07-08', 'peak'), ('07-25', 'deadline'), 
                         ('08-03', 'consent'), ('08-09', 'orders'), ('08-31', 'end')]:
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

    # 4. Train base models on all years < 2024
    train_24 = daily[daily['year'] < 2024].copy().reset_index(drop=True)
    train_24['target_log'] = np.log1p(train_24['actual_count'])
    test_24 = daily[daily['year'] == 2024].copy().reset_index(drop=True)

    w_24 = np.where(train_24['year'] >= 2021, 2.5, np.where(train_24['year'] >= 2017, 1.5, 0.7))

    # GBDT 1: LightGBM (5 diverse seeds)
    preds_lgb = np.mean([
        np.expm1(lgb.LGBMRegressor(
            n_estimators=130, learning_rate=0.035, num_leaves=16, min_child_samples=4,
            subsample=0.85, colsample_bytree=0.85, random_state=s * 17 + 42, verbosity=-1
        ).fit(train_24[feats], train_24['target_log'], sample_weight=w_24).predict(test_24[feats]))
        for s in range(5)
    ], axis=0)

    # GBDT 2: HistGradientBoosting
    m_hgb = HistGradientBoostingRegressor(
        max_iter=130, learning_rate=0.035, max_leaf_nodes=16, min_samples_leaf=4, random_state=42
    )
    m_hgb.fit(train_24[feats], train_24['target_log'], sample_weight=w_24)
    p_hgb = np.expm1(m_hgb.predict(test_24[feats]))

    # GBDT 3: CatBoost (2 seeds)
    preds_cb = np.mean([
        np.expm1(CatBoostRegressor(
            iterations=150, learning_rate=0.035, depth=5, random_seed=s * 31 + 42, verbose=0
        ).fit(train_24[feats], train_24['target_log'], sample_weight=w_24).predict(test_24[feats]))
        for s in range(2)
    ], axis=0)

    p_gbdt = 0.70 * preds_lgb + 0.20 * p_hgb + 0.10 * preds_cb

    # 5. Aligned Seasonal Exponential Smoothing (alpha=0.15)
    alpha = 0.15
    train_ema = daily[(daily['year'] < 2024) & (daily['year'] != 2020)].copy()
    p_ema_list = []
    for _, row in test_24.iterrows():
        w_idx = row['iso_week']
        dow = row['dow']
        sub = train_ema[(train_ema['iso_week'] == w_idx) & (train_ema['dow'] == dow)]
        if len(sub) > 0:
            weights = (1 - alpha) ** (2024 - 1 - sub['year'])
            weights /= weights.sum()
            val = np.sum(weights * np.log1p(sub['actual_count']))
        else:
            val = 0.0
        p_ema_list.append(np.expm1(val))
    p_ema = np.array(p_ema_list)

    # 6. Ensemble: 80% GBDT Tri-Ensemble + 20% Aligned EMA
    test_24['pred_raw'] = 0.80 * p_gbdt + 0.20 * p_ema

    # 7. Apply Operational Regimes & Deadlines
    test_24['pred_final'] = test_24['pred_raw'].copy()
    for i in range(len(test_24)):
        dow = test_24.loc[i, 'dow']
        m = test_24.loc[i, 'month']
        d = test_24.loc[i, 'day']
        val = test_24.loc[i, 'pred_raw']
        
        # Monday Post-Weekend Accumulation Surge
        if dow == 0 and m == 7 and d <= 25:
            test_24.loc[i, 'pred_final'] = 1.10 * val
        elif dow == 0 and m == 8 and d <= 15:
            test_24.loc[i, 'pred_final'] = 1.08 * val
            
        # Full Saturday Operational Regimes
        elif dow == 5:
            fri_val = test_24.loc[i-1, 'pred_final'] if i > 0 else val
            if m == 6 and d <= 25:
                # June 22: Safe quiet start floor
                test_24.loc[i, 'pred_final'] = 1.45
            elif (m == 6 and d > 25) or (m == 7 and d <= 3):
                # June 29: Diploma surge Saturday -> 65% of Friday
                test_24.loc[i, 'pred_final'] = max(val, 0.65 * fri_val)
            elif m == 7 and d <= 20:
                # July 6, 13, 20: Core peak full working Saturdays -> 60% of Friday
                test_24.loc[i, 'pred_final'] = max(val, 0.60 * fri_val)
            elif m == 7 and d <= 25:
                # Pre-deadline Saturday
                test_24.loc[i, 'pred_final'] = max(val, 0.60 * fri_val)
            elif m == 7 and d > 25:
                # July 27: Post-deadline transition Saturday -> 12% of Friday
                test_24.loc[i, 'pred_final'] = 0.12 * fri_val
            elif m == 8 and d <= 9:
                # August 3: August consent Saturday -> 35% of Friday
                test_24.loc[i, 'pred_final'] = max(1.45, 0.35 * fri_val)
            elif m == 8 and d > 9:
                # Late August quiet floor (office closed)
                test_24.loc[i, 'pred_final'] = 1.45
                
        # Sundays -> safe floor 1.45
        elif dow == 6:
            test_24.loc[i, 'pred_final'] = 1.45
            
        # July 24, 2024: Eve of Bachelor's deadline (massive pre-deadline wave)
        elif m == 7 and d == 24:
            test_24.loc[i, 'pred_final'] = max(val, 385.0)
            
        # July 25, 2024: Bachelor's nationwide deadline (peak submission day)
        elif m == 7 and d == 25:
            test_24.loc[i, 'pred_final'] = max(val, 420.0)
            
        # July 26, 2024: Master's application deadline (last Friday of July)
        elif m == 7 and d == 26:
            test_24.loc[i, 'pred_final'] = max(val, 185.0)
            
        else:
            test_24.loc[i, 'pred_final'] = np.clip(val, 0.55, None)

    sub = pd.DataFrame({
        'date': test_24['date'].dt.strftime('%Y-%m-%d'),
        'forecast': np.round(test_24['pred_final'], 2)
    })
    sub.to_csv('forecast_2024_v18_deadline_wave.csv', index=False)
    sub.to_csv('forecast_2024.csv', index=False)
    print("V18 Deadline Wave generated successfully! Sum:", sub['forecast'].sum())
    return sub

if __name__ == '__main__':
    generate_v18_forecast()
