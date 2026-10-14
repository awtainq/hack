import pandas as pd
import numpy as np

def compute_hourly_profiles(raw_df):
    """
    Compute empirical hourly distribution p(hour | day_of_week) from historical data.
    Ensures that for each day_of_week, sum_h p(h | dow) == 1.0.
    """
    df = raw_df.copy()
    df['hour'] = df['reg_ts'].dt.hour
    df['dow'] = df['reg_ts'].dt.dayofweek
    
    # Exclude 2020 covid anomaly to have representative physical + electronic mix
    normal_df = df[df['year'] != 2020]
    
    # Cross-tabulate dow x hour
    dow_hour_counts = normal_df.groupby(['dow', 'hour']).size().unstack(fill_value=0)
    
    # Reindex full 0..6 days and 0..23 hours
    dow_hour_counts = dow_hour_counts.reindex(index=range(7), columns=range(24), fill_value=0)
    
    # For Sunday (dow=6), since commission is closed, keep profile as 0 except small trace or all 0
    # For normal days, normalize to sum=1.0
    profiles = {}
    for dow in range(7):
        row = dow_hour_counts.loc[dow].values.astype(float)
        s = row.sum()
        if s > 0:
            profiles[dow] = row / s
        else:
            profiles[dow] = np.zeros(24)
            
    # Explicitly enforce working hours structure
    # Weekdays: 10:00 to 18:00
    # Saturday (5): 10:00 to 15:00
    # Sunday (6): closed
    return profiles

def decompose_to_hourly(daily_predictions_df, hourly_profiles, apps_per_operator_hour=5.0, min_standby_operators=1):
    """
    Convert daily predictions by date and basis into hourly forecasts and operator requirements.
    
    Parameters:
    - daily_predictions_df: DataFrame with columns ['year', 'date', 'ob_osnova_name', 'predicted_count']
    - hourly_profiles: dict mapping dow -> array of 24 hourly fractions
    - apps_per_operator_hour: processing norm (default: 5 applications/hour/operator)
    - min_standby_operators: minimum staff on duty during open hours (default: 1)
    
    Returns:
    - hourly_df: DataFrame with hourly breakdown and recommended operator count
    """
    daily_df = daily_predictions_df.copy()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df['dow'] = daily_df['date'].dt.dayofweek
    
    records = []
    
    # Group by date and dow
    for (d, dow, yr), grp in daily_df.groupby(['date', 'dow', 'year']):
        date_str = d.strftime('%Y-%m-%d')
        profile = hourly_profiles[dow]
        
        # Totals by basis for this day
        budget_day = grp[grp['ob_osnova_name'] == 'Бюджет']['predicted_count'].sum() if 'Бюджет' in grp['ob_osnova_name'].values else 0.0
        contract_day = grp[grp['ob_osnova_name'] == 'Договор']['predicted_count'].sum() if 'Договор' in grp['ob_osnova_name'].values else 0.0
        goslin_day = grp[grp['ob_osnova_name'] == 'Гослиния']['predicted_count'].sum() if 'Гослиния' in grp['ob_osnova_name'].values else 0.0
        
        total_day = budget_day + contract_day + goslin_day
        
        for h in range(24):
            frac = profile[h]
            h_budget = budget_day * frac
            h_contract = contract_day * frac
            h_goslin = goslin_day * frac
            h_total = total_day * frac
            
            # Operator requirement calculation
            # During working hours:
            # Weekdays: 10..17
            # Saturday: 10..14
            # Sunday: closed
            is_open_hour = False
            if dow < 5 and 10 <= h <= 17:
                is_open_hour = True
            elif dow == 5 and 10 <= h <= 14:
                is_open_hour = True
                
            if is_open_hour:
                needed = int(np.ceil(h_total / apps_per_operator_hour))
                needed = max(min_standby_operators, needed)
            else:
                needed = 0
                
            records.append({
                'year': yr,
                'date': date_str,
                'day_of_week': int(dow),
                'hour': int(h),
                'time_slot': f'{h:02d}:00 - {h+1:02d}:00',
                'predicted_budget': round(h_budget, 2),
                'predicted_contract': round(h_contract, 2),
                'predicted_goslin': round(h_goslin, 2),
                'predicted_total': round(h_total, 2),
                'recommended_operators': needed
            })
            
    hourly_df = pd.DataFrame(records)
    return hourly_df

