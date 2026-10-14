import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

BASES = ['Бюджет', 'Договор', 'Гослиния']
CAMPAIGN_START_MONTH_DAY = '06-20'
CAMPAIGN_END_MONTH_DAY = '09-20'

def load_raw_data(data_path='processed_data/included_ks.csv'):
    df = pd.read_csv(data_path)
    df['reg_ts'] = pd.to_datetime(df['reg_ts'])
    df['date'] = df['reg_ts'].dt.date
    df['year'] = df['reg_ts'].dt.year
    return df

def generate_campaign_grid(years):
    """
    Generate complete date x basis grid for each year between June 20 and Sept 20.
    """
    rows = []
    for yr in years:
        start_date = pd.Timestamp(f'{yr}-{CAMPAIGN_START_MONTH_DAY}').date()
        end_date = pd.Timestamp(f'{yr}-{CAMPAIGN_END_MONTH_DAY}').date()
        dates = pd.date_range(start_date, end_date).date
        for d in dates:
            for osn in BASES:
                rows.append({
                    'year': int(yr),
                    'date': d,
                    'ob_osnova_name': osn
                })
    return pd.DataFrame(rows)

def build_daily_dataset(raw_df, years=None):
    if years is None:
        years = sorted(raw_df['year'].unique())
    grid = generate_campaign_grid(years)
    counts = raw_df.groupby(['year', 'date', 'ob_osnova_name']).size().reset_index(name='actual_count')
    df = pd.merge(grid, counts, on=['year', 'date', 'ob_osnova_name'], how='left')
    df['actual_count'] = df['actual_count'].fillna(0.0)
    return df

def extract_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Campaign timeline
    df['june20'] = pd.to_datetime(df['year'].astype(str) + '-06-20')
    df['day_of_campaign'] = (df['date'] - df['june20']).dt.days
    df['week_of_campaign'] = df['day_of_campaign'] // 7
    
    # Calendar features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_saturday'] = (df['day_of_week'] == 5).astype(int)
    df['is_sunday'] = (df['day_of_week'] == 6).astype(int)
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # Milestone dates
    df['july08'] = pd.to_datetime(df['year'].astype(str) + '-07-08')
    df['july25'] = pd.to_datetime(df['year'].astype(str) + '-07-25')
    df['aug03'] = pd.to_datetime(df['year'].astype(str) + '-08-03')
    df['aug09'] = pd.to_datetime(df['year'].astype(str) + '-08-09')
    df['aug20'] = pd.to_datetime(df['year'].astype(str) + '-08-20')
    df['sept20'] = pd.to_datetime(df['year'].astype(str) + '-09-20')
    
    # Distances to milestones (positive = after, negative = before)
    df['dist_july08'] = (df['date'] - df['july08']).dt.days
    df['dist_july25'] = (df['date'] - df['july25']).dt.days
    df['dist_aug03'] = (df['date'] - df['aug03']).dt.days
    df['dist_aug09'] = (df['date'] - df['aug09']).dt.days
    df['dist_aug20'] = (df['date'] - df['aug20']).dt.days
    df['dist_sept20'] = (df['date'] - df['sept20']).dt.days
    
    # Milestone flags
    df['is_after_july25'] = (df['date'] > df['july25']).astype(int)
    df['is_after_aug09'] = (df['date'] > df['aug09']).astype(int)
    df['is_after_aug20'] = (df['date'] > df['aug20']).astype(int)
    
    # Harmonic cyclical encodings
    df['sin_camp'] = np.sin(2 * np.pi * df['day_of_campaign'] / 93)
    df['cos_camp'] = np.cos(2 * np.pi * df['day_of_campaign'] / 93)
    df['sin_camp_semi'] = np.sin(4 * np.pi * df['day_of_campaign'] / 93)
    df['cos_camp_semi'] = np.cos(4 * np.pi * df['day_of_campaign'] / 93)
    df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Domain specific interaction features
    df['is_budget'] = (df['ob_osnova_name'] == 'Бюджет').astype(int)
    df['is_contract'] = (df['ob_osnova_name'] == 'Договор').astype(int)
    df['is_goslin'] = (df['ob_osnova_name'] == 'Гослиния').astype(int)
    
    df['budget_after_deadline'] = df['is_budget'] * df['is_after_july25']
    df['contract_after_deadline'] = df['is_contract'] * df['is_after_july25']
    
    # Categorical
    df['ob_osnova_cat'] = df['ob_osnova_name'].astype('category')
    
    # Cleanup temporary milestone columns
    drop_cols = ['june20', 'july08', 'july25', 'aug03', 'aug09', 'aug20', 'sept20']
    df = df.drop(columns=drop_cols)
    return df

FEATURE_COLS = [
    'day_of_campaign', 'day_of_week', 'is_weekend', 'is_saturday', 'is_sunday',
    'month', 'day', 'week_of_campaign',
    'dist_july08', 'dist_july25', 'dist_aug03', 'dist_aug09', 'dist_aug20', 'dist_sept20',
    'is_after_july25', 'is_after_aug09', 'is_after_aug20',
    'sin_camp', 'cos_camp', 'sin_camp_semi', 'cos_camp_semi',
    'sin_dow', 'cos_dow',
    'is_budget', 'is_contract', 'is_goslin',
    'budget_after_deadline', 'contract_after_deadline',
    'ob_osnova_cat'
]

