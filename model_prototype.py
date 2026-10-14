import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred))**2))

def create_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Campaign day (0 = June 20)
    df['june20'] = pd.to_datetime(df['year'].astype(str) + '-06-20')
    df['day_of_campaign'] = (df['date'] - df['june20']).dt.days
    
    # Calendar features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_saturday'] = (df['day_of_week'] == 5).astype(int)
    df['is_sunday'] = (df['day_of_week'] == 6).astype(int)
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['week_of_campaign'] = df['day_of_campaign'] // 7
    
    # Milestone distances
    df['july08'] = pd.to_datetime(df['year'].astype(str) + '-07-08')
    df['july25'] = pd.to_datetime(df['year'].astype(str) + '-07-25')
    df['aug03'] = pd.to_datetime(df['year'].astype(str) + '-08-03')
    df['aug09'] = pd.to_datetime(df['year'].astype(str) + '-08-09')
    df['aug20'] = pd.to_datetime(df['year'].astype(str) + '-08-20')
    df['sept20'] = pd.to_datetime(df['year'].astype(str) + '-09-20')
    
    df['dist_july08'] = (df['date'] - df['july08']).dt.days
    df['dist_july25'] = (df['date'] - df['july25']).dt.days
    df['dist_aug03'] = (df['date'] - df['aug03']).dt.days
    df['dist_aug09'] = (df['date'] - df['aug09']).dt.days
    df['dist_aug20'] = (df['date'] - df['aug20']).dt.days
    df['dist_sept20'] = (df['date'] - df['sept20']).dt.days
    
    df['is_after_july25'] = (df['date'] > df['july25']).astype(int)
    df['is_after_aug09'] = (df['date'] > df['aug09']).astype(int)
    df['is_after_aug20'] = (df['date'] > df['aug20']).astype(int)
    
    # Fourier terms for campaign progress (period = 93 days)
    df['sin_camp'] = np.sin(2 * np.pi * df['day_of_campaign'] / 93)
    df['cos_camp'] = np.cos(2 * np.pi * df['day_of_campaign'] / 93)
    df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Categorical
    df['ob_osnova_name'] = df['ob_osnova_name'].astype('category')
    
    drop_cols = ['june20', 'july08', 'july25', 'aug03', 'aug09', 'aug20', 'sept20']
    df = df.drop(columns=drop_cols)
    return df

# Load data
df_ks = pd.read_csv('processed_data/included_ks.csv')
df_ks['reg_ts'] = pd.to_datetime(df_ks['reg_ts'])
df_ks['date'] = df_ks['reg_ts'].dt.date
df_ks['year'] = df_ks['reg_ts'].dt.year

rows = []
for yr in range(2014, 2024):
    dates = pd.date_range(f'{yr}-06-20', f'{yr}-09-20').date
    for d in dates:
        for osn in ['Бюджет', 'Договор', 'Гослиния']:
            rows.append({'year': yr, 'date': d, 'ob_osnova_name': osn})
df_daily = pd.DataFrame(rows)

counts = df_ks.groupby(['year', 'date', 'ob_osnova_name']).size().reset_index(name='actual_count')
df_daily = pd.merge(df_daily, counts, on=['year', 'date', 'ob_osnova_name'], how='left')
df_daily['actual_count'] = df_daily['actual_count'].fillna(0)

# Feature engineering
df_feat = create_features(df_daily)
df_feat['target_log'] = np.log1p(df_feat['actual_count'])

feature_cols = [
    'day_of_campaign', 'day_of_week', 'is_weekend', 'is_saturday', 'is_sunday',
    'month', 'day', 'week_of_campaign',
    'dist_july08', 'dist_july25', 'dist_aug03', 'dist_aug09', 'dist_aug20', 'dist_sept20',
    'is_after_july25', 'is_after_aug09', 'is_after_aug20',
    'sin_camp', 'cos_camp', 'sin_dow', 'cos_dow',
    'ob_osnova_name'
]

print("Features:", feature_cols)

# Evaluate on 2022 (trained on 2014-2021) and 2023 (trained on 2014-2022)
for test_yr in [2022, 2023]:
    train = df_feat[(df_feat['year'] < test_yr) & (df_feat['year'] != 2020)] # exclude 2020 covid anomaly from train
    test = df_feat[df_feat['year'] == test_yr]
    
    # Train LightGBM
    model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    model.fit(train[feature_cols], train['target_log'])
    
    test_pred_log = model.predict(test[feature_cols])
    test_pred = np.clip(np.expm1(test_pred_log), 0, None)
    
    # Also evaluate historical baseline: average/median by (day_of_campaign, day_of_week, ob_osnova_name)
    base_group = train.groupby(['day_of_campaign', 'ob_osnova_name'])['actual_count'].mean().reset_index(name='base_pred')
    test_base = pd.merge(test, base_group, on=['day_of_campaign', 'ob_osnova_name'], how='left')
    base_pred = test_base['base_pred'].fillna(0).values
    
    test_actual = test['actual_count'].values
    
    score_lgb = rmsle(test_actual, test_pred)
    score_base = rmsle(test_actual, base_pred)
    
    print(f"\n--- Backtest Year {test_yr} ---")
    print(f"Overall RMSLE -> LightGBM: {score_lgb:.4f} | Baseline: {score_base:.4f}")
    
    # Breakdown by ob_osnova_name
    for osn in ['Бюджет', 'Договор', 'Гослиния']:
        idx = (test['ob_osnova_name'] == osn)
        osn_act = test_actual[idx]
        osn_pred = test_pred[idx]
        osn_score = rmsle(osn_act, osn_pred)
        print(f"  {osn:8s} RMSLE: {osn_score:.4f} | Total Actual: {osn_act.sum():.0f} | Total Pred: {osn_pred.sum():.0f}")
    
    # Evaluate Total (sum of 3 bases per date)
    test_res = test[['date', 'actual_count']].copy()
    test_res['pred'] = test_pred
    tot_act = test_res.groupby('date')['actual_count'].sum()
    tot_pred = test_res.groupby('date')['pred'].sum()
    score_tot = rmsle(tot_act.values, tot_pred.values)
    print(f"  TOTAL    RMSLE: {score_tot:.4f} | Total Actual: {tot_act.sum():.0f} | Total Pred: {tot_pred.sum():.0f}")

