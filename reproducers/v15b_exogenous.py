"""
V15b — Two-Stage Exogenous Model with Saturday Ratio Learning

Stage 1: Main GBDT model for all days (same as V14)
Stage 2: Separate Saturday Ratio Model
  - Predicts log(sat_actual / fri_actual) as the target
  - Uses campaign phase, gaussian envelope, regime code, gosuslugi era
  - Multiplies the learned ratio by Friday's prediction
  
This way the model only needs to learn the Saturday/Friday ratio
(a much simpler function) instead of the absolute Saturday count.

Sunday: Floor 1.45 (empirically validated on public LB)
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor


def generate_v15b_forecast():
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

    # Saturday exogenous features
    daily['sat_gaussian'] = np.exp(-0.5 * ((daily['day_of_camp'] - 18) / 12) ** 2) * daily['is_saturday']
    daily['gosuslugi_era'] = (daily['year'] >= 2021).astype(int)
    daily['is_post_deadline'] = ((daily['month'] == 7) & (daily['day'] >= 26) | (daily['month'] == 8)).astype(int)
    daily['weekend_accum'] = np.where(daily['dow'] == 0, 2, np.where(daily['dow'] == 6, 1, 0))
    daily['monday_july'] = ((daily['dow'] == 0) & (daily['month'] == 7) & (daily['day'] <= 25)).astype(int)
    daily['is_masters_deadline_zone'] = (
        (daily['month'] == 7) & (daily['day'] >= 24) & (daily['day'] <= 26) & (daily['dow'] == 4)
    ).astype(int)
    daily['sat_x_progress'] = daily['is_saturday'] * daily['progress']
    daily['sat_x_sinprog'] = daily['is_saturday'] * daily['sin_prog']
    daily['sun_x_gosuslugi'] = daily['is_sunday'] * daily['gosuslugi_era']
    daily['days_until_aug'] = np.clip(
        (pd.to_datetime(daily['year'].astype(str) + '-08-01') - daily['date']).dt.days, -10, 40)

    # Saturday regime code (ordinal)
    conditions = [
        (daily['dow'] != 5),
        (daily['dow'] == 5) & (daily['month'] == 6) & (daily['day'] <= 25),
        (daily['dow'] == 5) & (daily['month'] == 6) & (daily['day'] > 25),
        (daily['dow'] == 5) & (daily['month'] == 7) & (daily['day'] <= 10),
        (daily['dow'] == 5) & (daily['month'] == 7) & (daily['day'] <= 20),
        (daily['dow'] == 5) & (daily['month'] == 7) & (daily['day'] <= 25),
        (daily['dow'] == 5) & (daily['month'] == 7) & (daily['day'] > 25),
        (daily['dow'] == 5) & (daily['month'] == 8) & (daily['day'] <= 9),
        (daily['dow'] == 5) & (daily['month'] == 8) & (daily['day'] > 9),
    ]
    daily['sat_regime_code'] = np.select(conditions, list(range(9)), default=0)

    daily['target_log'] = np.log1p(daily['actual_count'])

    drop_cols = ['year', 'date', 'actual_count', 'target_log']
    feats = [c for c in daily.columns if c not in drop_cols]

    # ============================================================
    # STAGE 1: Main model for ALL days (including Saturdays)
    # ============================================================
    train = daily[daily['year'] < 2024].copy().reset_index(drop=True)
    test = daily[daily['year'] == 2024].copy().reset_index(drop=True)

    w = np.where(train['year'] >= 2021, 2.5, np.where(train['year'] >= 2017, 1.5, 0.7))

    preds_lgb = np.mean([
        np.expm1(lgb.LGBMRegressor(
            n_estimators=130, learning_rate=0.035, num_leaves=16, min_child_samples=4,
            subsample=0.85, colsample_bytree=0.85, random_state=s * 17 + 42, verbosity=-1
        ).fit(train[feats], train['target_log'], sample_weight=w).predict(test[feats]))
        for s in range(3)
    ], axis=0)

    m_hgb = HistGradientBoostingRegressor(
        max_iter=130, learning_rate=0.035, max_leaf_nodes=16, min_samples_leaf=4, random_state=42
    )
    m_hgb.fit(train[feats], train['target_log'], sample_weight=w)
    p_hgb = np.expm1(m_hgb.predict(test[feats]))

    cb = CatBoostRegressor(iterations=150, learning_rate=0.035, depth=5, random_seed=42, verbose=0)
    cb.fit(train[feats], train['target_log'], sample_weight=w)
    p_cb = np.expm1(cb.predict(test[feats]))

    p_gbdt = 0.70 * preds_lgb + 0.20 * p_hgb + 0.10 * p_cb

    # EMA
    alpha = 0.15
    train_ema = daily[(daily['year'] < 2024) & (daily['year'] != 2020)].copy()
    p_ema_list = []
    for _, row in test.iterrows():
        sub = train_ema[(train_ema['iso_week'] == row['iso_week']) &
                        (train_ema['dow'] == row['dow'])]
        if len(sub) > 0:
            wts = (1 - alpha) ** (2024 - 1 - sub['year'])
            wts /= wts.sum()
            val = np.sum(wts * np.log1p(sub['actual_count']))
        else:
            val = 0.0
        p_ema_list.append(np.expm1(val))
    p_ema = np.array(p_ema_list)

    test['pred_stage1'] = 0.80 * p_gbdt + 0.20 * p_ema

    # ============================================================
    # STAGE 2: Saturday Ratio Model
    # Build training data: for each Saturday, compute actual_ratio = sat_count / fri_count
    # The model predicts this ratio, then sat_pred = ratio * fri_pred
    # ============================================================
    print("\n=== Stage 2: Saturday Ratio Model ===")
    
    # Build Saturday training pairs: (Saturday features, Saturday/Friday actual ratio)
    sat_pairs = []
    for yr in sorted(daily['year'].unique()):
        if yr == 2020 or yr == 2024:
            continue
        yr_data = daily[daily['year'] == yr].sort_values('date').reset_index(drop=True)
        for i, row in yr_data.iterrows():
            if row['dow'] == 5:  # Saturday
                if i > 0 and yr_data.loc[i-1, 'dow'] == 4:  # Previous day is Friday
                    fri_count = yr_data.loc[i-1, 'actual_count']
                    sat_count = row['actual_count']
                    # Use log ratio as target: log1p(sat) - log1p(fri)
                    log_ratio = np.log1p(sat_count) - np.log1p(fri_count)
                    sat_pairs.append({
                        'year': yr,
                        'sat_idx': i,
                        'fri_count': fri_count,
                        'sat_count': sat_count,
                        'log_ratio': log_ratio,
                        **{f: row[f] for f in feats}
                    })
    
    sat_df = pd.DataFrame(sat_pairs)
    print(f"  Saturday training samples: {len(sat_df)}")
    
    # Saturday-specific features for the ratio model
    sat_feats = ['day_of_camp', 'progress', 'month', 'day', 'iso_week',
                 'sat_gaussian', 'gosuslugi_era', 'is_post_deadline',
                 'sat_regime_code', 'sat_x_progress', 'sat_x_sinprog',
                 'sin_prog', 'sin_2prog', 'cos_prog',
                 'dist_start', 'dist_peak', 'dist_deadline', 'dist_consent',
                 'days_until_aug', 'sat_june', 'sat_july_peak', 'sat_july_late', 'sat_aug']
    
    w_sat = np.where(sat_df['year'] >= 2021, 3.0, np.where(sat_df['year'] >= 2017, 1.5, 0.5))
    
    # Train ratio model (LightGBM)
    sat_model = lgb.LGBMRegressor(
        n_estimators=80, learning_rate=0.05, num_leaves=8, min_child_samples=3,
        subsample=0.9, colsample_bytree=0.8, random_state=42, verbosity=-1
    )
    sat_model.fit(sat_df[sat_feats], sat_df['log_ratio'], sample_weight=w_sat)
    
    # Print learned ratios for each regime
    for regime in sorted(sat_df['sat_regime_code'].unique()):
        regime_data = sat_df[sat_df['sat_regime_code'] == regime]
        actual_ratios = np.exp(regime_data['log_ratio'])
        predicted_ratios = np.exp(sat_model.predict(regime_data[sat_feats]))
        print(f"  Regime {regime}: n={len(regime_data)}, "
              f"actual_ratio mean={actual_ratios.mean():.3f} (std={actual_ratios.std():.3f}), "
              f"pred_ratio mean={predicted_ratios.mean():.3f}")
    
    # ============================================================
    # STAGE 3: Apply Saturday Ratio Model to 2024
    # ============================================================
    test['pred_final'] = test['pred_stage1'].copy()
    
    for i in range(len(test)):
        dow = test.loc[i, 'dow']
        
        if dow == 5:  # Saturday
            # Get Friday's stage1 prediction
            if i > 0 and test.loc[i-1, 'dow'] == 4:
                fri_pred = test.loc[i-1, 'pred_final']
            else:
                fri_pred = test.loc[i, 'pred_stage1']
            
            # Predict log ratio using the Saturday model
            sat_features = test.loc[i:i, sat_feats]
            log_ratio_pred = sat_model.predict(sat_features)[0]
            
            # Saturday prediction = exp(log_ratio) * Friday prediction
            # In log space: log1p(sat_pred) = log_ratio + log1p(fri_pred)
            sat_log_pred = log_ratio_pred + np.log1p(fri_pred)
            sat_pred = np.expm1(sat_log_pred)
            
            # Floor at 1.45 (empirically optimal for near-zero days under RMSLE)
            sat_pred = max(sat_pred, 1.45)
            
            test.loc[i, 'pred_final'] = sat_pred
            
        elif dow == 6:  # Sunday
            test.loc[i, 'pred_final'] = max(test.loc[i, 'pred_stage1'], 1.45)
            # For Sundays, typically the model itself should predict low
            # but we clip to 1.45 as safety net
            
        elif dow == 0 and test.loc[i, 'month'] == 7 and test.loc[i, 'day'] <= 25:
            # Monday surge — let exogenous feature monday_july handle most of it
            # but add a mild 5% boost (less aggressive than V14's 10%)
            test.loc[i, 'pred_final'] = max(test.loc[i, 'pred_stage1'] * 1.05, test.loc[i, 'pred_stage1'])
            
        else:
            test.loc[i, 'pred_final'] = max(test.loc[i, 'pred_stage1'], 0.55)
    
    # Print diagnostics
    print("\n=== V15b Diagnostics ===")
    for dow_name, dow_val in [('Mon', 0), ('Tue', 1), ('Wed', 2), ('Thu', 3),
                               ('Fri', 4), ('Sat', 5), ('Sun', 6)]:
        mask = test['dow'] == dow_val
        if mask.sum() > 0:
            vals = test.loc[mask, 'pred_final']
            print(f"  {dow_name}: mean={vals.mean():.1f}, min={vals.min():.1f}, max={vals.max():.1f}")

    print("\nSaturday predictions:")
    sat_test = test[test['dow'] == 5][['date', 'day', 'month', 'pred_stage1', 'pred_final', 'sat_regime_code']]
    for _, r in sat_test.iterrows():
        print(f"  {r['date'].strftime('%Y-%m-%d')} (regime={int(r['sat_regime_code'])}): "
              f"stage1={r['pred_stage1']:.1f}, final={r['pred_final']:.1f}")

    print("\nSunday predictions:")
    sun_test = test[test['dow'] == 6][['date', 'pred_stage1', 'pred_final']]
    for _, r in sun_test.iterrows():
        print(f"  {r['date'].strftime('%Y-%m-%d')}: stage1={r['pred_stage1']:.1f}, final={r['pred_final']:.1f}")

    sub = pd.DataFrame({
        'date': test['date'].dt.strftime('%Y-%m-%d'),
        'forecast': np.round(test['pred_final'], 2)
    })
    sub.to_csv('forecast_2024_v15b_exogenous.csv', index=False)
    print(f"\nV15b Exogenous generated! Sum: {sub['forecast'].sum():.1f}")
    return sub


def backtest_v15b(test_year=2023):
    """Run leave-one-year-out backtest for V15b."""
    df_reg = pd.read_csv('processed_data/registered.csv')
    df_reg['date'] = pd.to_datetime(df_reg['reg_ts']).dt.date

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

    daily['sat_gaussian'] = np.exp(-0.5 * ((daily['day_of_camp'] - 18) / 12) ** 2) * daily['is_saturday']
    daily['gosuslugi_era'] = (daily['year'] >= 2021).astype(int)
    daily['is_post_deadline'] = ((daily['month'] == 7) & (daily['day'] >= 26) | (daily['month'] == 8)).astype(int)
    daily['weekend_accum'] = np.where(daily['dow'] == 0, 2, np.where(daily['dow'] == 6, 1, 0))
    daily['monday_july'] = ((daily['dow'] == 0) & (daily['month'] == 7) & (daily['day'] <= 25)).astype(int)
    daily['is_masters_deadline_zone'] = (
        (daily['month'] == 7) & (daily['day'] >= 24) & (daily['day'] <= 26) & (daily['dow'] == 4)
    ).astype(int)
    daily['sat_x_progress'] = daily['is_saturday'] * daily['progress']
    daily['sat_x_sinprog'] = daily['is_saturday'] * daily['sin_prog']
    daily['sun_x_gosuslugi'] = daily['is_sunday'] * daily['gosuslugi_era']
    daily['days_until_aug'] = np.clip(
        (pd.to_datetime(daily['year'].astype(str) + '-08-01') - daily['date']).dt.days, -10, 40)

    conditions = [
        (daily['dow'] != 5),
        (daily['dow'] == 5) & (daily['month'] == 6) & (daily['day'] <= 25),
        (daily['dow'] == 5) & (daily['month'] == 6) & (daily['day'] > 25),
        (daily['dow'] == 5) & (daily['month'] == 7) & (daily['day'] <= 10),
        (daily['dow'] == 5) & (daily['month'] == 7) & (daily['day'] <= 20),
        (daily['dow'] == 5) & (daily['month'] == 7) & (daily['day'] <= 25),
        (daily['dow'] == 5) & (daily['month'] == 7) & (daily['day'] > 25),
        (daily['dow'] == 5) & (daily['month'] == 8) & (daily['day'] <= 9),
        (daily['dow'] == 5) & (daily['month'] == 8) & (daily['day'] > 9),
    ]
    daily['sat_regime_code'] = np.select(conditions, list(range(9)), default=0)

    daily['target_log'] = np.log1p(daily['actual_count'])

    drop_cols = ['year', 'date', 'actual_count', 'target_log']
    feats = [c for c in daily.columns if c not in drop_cols]
    
    sat_feats = ['day_of_camp', 'progress', 'month', 'day', 'iso_week',
                 'sat_gaussian', 'gosuslugi_era', 'is_post_deadline',
                 'sat_regime_code', 'sat_x_progress', 'sat_x_sinprog',
                 'sin_prog', 'sin_2prog', 'cos_prog',
                 'dist_start', 'dist_peak', 'dist_deadline', 'dist_consent',
                 'days_until_aug', 'sat_june', 'sat_july_peak', 'sat_july_late', 'sat_aug']

    from sklearn.metrics import mean_squared_error
    
    test_years = [2017, 2018, 2019, 2021, 2022, 2023] if test_year is None else [test_year]
    results = {}
    
    for ty in test_years:
        # Stage 1: Main model
        train_mask = (daily['year'] != ty) & (daily['year'] != 2020) & (daily['year'] < 2024)
        train = daily[train_mask].copy().reset_index(drop=True)
        test_data = daily[daily['year'] == ty].copy().reset_index(drop=True)
        
        w = np.where(train['year'] >= 2021, 2.5, np.where(train['year'] >= 2017, 1.5, 0.7))
        
        preds_lgb = np.mean([
            np.expm1(lgb.LGBMRegressor(
                n_estimators=130, learning_rate=0.035, num_leaves=16, min_child_samples=4,
                subsample=0.85, colsample_bytree=0.85, random_state=s * 17 + 42, verbosity=-1
            ).fit(train[feats], train['target_log'], sample_weight=w).predict(test_data[feats]))
            for s in range(3)
        ], axis=0)
        
        m_hgb = HistGradientBoostingRegressor(
            max_iter=130, learning_rate=0.035, max_leaf_nodes=16, min_samples_leaf=4, random_state=42
        )
        m_hgb.fit(train[feats], train['target_log'], sample_weight=w)
        p_hgb = np.expm1(m_hgb.predict(test_data[feats]))
        
        cb_m = CatBoostRegressor(iterations=150, learning_rate=0.035, depth=5, random_seed=42, verbose=0)
        cb_m.fit(train[feats], train['target_log'], sample_weight=w)
        p_cb = np.expm1(cb_m.predict(test_data[feats]))
        
        p_gbdt = 0.70 * preds_lgb + 0.20 * p_hgb + 0.10 * p_cb
        
        # EMA
        alpha = 0.15
        train_ema = daily[(daily['year'] != ty) & (daily['year'] != 2020) & (daily['year'] < 2024)]
        p_ema_list = []
        for _, row in test_data.iterrows():
            sub = train_ema[(train_ema['iso_week'] == row['iso_week']) &
                           (train_ema['dow'] == row['dow'])]
            if len(sub) > 0:
                wts = (1 - alpha) ** (ty - 1 - sub['year'])
                wts /= wts.sum()
                val = np.sum(wts * np.log1p(sub['actual_count']))
            else:
                val = 0.0
            p_ema_list.append(np.expm1(val))
        p_ema = np.array(p_ema_list)
        
        test_data['pred_stage1'] = 0.80 * p_gbdt + 0.20 * p_ema
        
        # Stage 2: Saturday Ratio Model
        sat_pairs = []
        for yr in sorted(daily['year'].unique()):
            if yr == 2020 or yr == ty or yr >= 2024:
                continue
            yr_data = daily[daily['year'] == yr].sort_values('date').reset_index(drop=True)
            for i in range(len(yr_data)):
                if yr_data.loc[i, 'dow'] == 5 and i > 0 and yr_data.loc[i-1, 'dow'] == 4:
                    fri_count = yr_data.loc[i-1, 'actual_count']
                    sat_count = yr_data.loc[i, 'actual_count']
                    log_ratio = np.log1p(sat_count) - np.log1p(fri_count)
                    sat_pairs.append({
                        'year': yr,
                        'log_ratio': log_ratio,
                        **{f: yr_data.loc[i, f] for f in sat_feats}
                    })
        
        sat_train_df = pd.DataFrame(sat_pairs)
        w_sat = np.where(sat_train_df['year'] >= 2021, 3.0, 
                np.where(sat_train_df['year'] >= 2017, 1.5, 0.5))
        
        sat_model = lgb.LGBMRegressor(
            n_estimators=80, learning_rate=0.05, num_leaves=8, min_child_samples=3,
            subsample=0.9, colsample_bytree=0.8, random_state=42, verbosity=-1
        )
        sat_model.fit(sat_train_df[sat_feats], sat_train_df['log_ratio'], sample_weight=w_sat)
        
        # Apply
        pred_final = test_data['pred_stage1'].values.copy()
        for i in range(len(test_data)):
            dow = test_data.loc[i, 'dow']
            if dow == 5:  # Saturday
                fri_pred = pred_final[i-1] if i > 0 and test_data.loc[i-1, 'dow'] == 4 else pred_final[i]
                log_ratio_pred = sat_model.predict(test_data.loc[i:i, sat_feats])[0]
                sat_pred = np.expm1(log_ratio_pred + np.log1p(fri_pred))
                pred_final[i] = max(sat_pred, 1.45)
            elif dow == 6:
                pred_final[i] = max(pred_final[i], 1.45)
            elif dow == 0 and test_data.loc[i, 'month'] == 7 and test_data.loc[i, 'day'] <= 25:
                pred_final[i] = pred_final[i] * 1.05
            else:
                pred_final[i] = max(pred_final[i], 0.55)
        
        actual = test_data['actual_count'].values
        rmsle = np.sqrt(np.mean((np.log1p(pred_final) - np.log1p(actual)) ** 2))
        results[ty] = rmsle
        print(f"  Year {ty}: RMSLE = {rmsle:.5f}")
        
        if ty == test_year or test_year is None:
            for _, r in test_data[test_data['dow'] == 5].iterrows():
                idx = r.name
                print(f"    Sat {r['date'].strftime('%Y-%m-%d')}: "
                      f"actual={r['actual_count']:.0f}, pred={pred_final[idx]:.1f}, "
                      f"stage1={r['pred_stage1']:.1f}, regime={int(r['sat_regime_code'])}")
    
    if len(results) > 1:
        print(f"\n  Mean RMSLE: {np.mean(list(results.values())):.5f}")
    
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'backtest':
        test_yr = int(sys.argv[2]) if len(sys.argv) > 2 else 2023
        print(f"=== V15b Exogenous Backtest (year={test_yr}) ===")
        backtest_v15b(test_yr)
    elif len(sys.argv) > 1 and sys.argv[1] == 'cv':
        print("=== V15b Full CV ===")
        backtest_v15b(test_year=None)
    else:
        generate_v15b_forecast()
