import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from src.preprocess import (
    load_raw_data,
    build_daily_dataset,
    extract_features,
    generate_campaign_grid,
    FEATURE_COLS,
    BASES
)
from src.metrics import rmsle, mae, mape
from src.model import AdmissionsForecaster
from src.scheduler import compute_hourly_profiles, decompose_to_hourly

def run_backtesting(df_feat, raw_df, hourly_profiles):
    print("=" * 60)
    print(" 1. OUT-OF-TIME BACKTESTING VALIDATION (2022 & 2023)")
    print("=" * 60)
    
    metrics_summary = []
    
    for test_year in [2022, 2023]:
        # Train on prior years, excluding 2020 (severe COVID-19 lockdown anomaly)
        train_mask = (df_feat['year'] < test_year) & (df_feat['year'] != 2020)
        test_mask = (df_feat['year'] == test_year)
        
        train_data = df_feat[train_mask].copy()
        test_data = df_feat[test_mask].copy()
        
        forecaster = AdmissionsForecaster(use_ensemble=True)
        forecaster.fit(train_data, train_data['actual_count'])
        
        preds = forecaster.predict(test_data)
        test_data['predicted_count'] = preds
        
        # Calculate daily metrics
        y_true = test_data['actual_count'].values
        y_pred = preds
        
        overall_rmsle = rmsle(y_true, y_pred)
        overall_mae = mae(y_true, y_pred)
        
        print(f"\n>>> TEST YEAR: {test_year} <<<")
        print(f"Overall Daily RMSLE: {overall_rmsle:.4f} | MAE: {overall_mae:.2f}")
        
        # Breakdown by Basis
        for basis in BASES:
            idx = (test_data['ob_osnova_name'] == basis)
            b_true = y_true[idx]
            b_pred = y_pred[idx]
            b_rmsle = rmsle(b_true, b_pred)
            b_mae = mae(b_true, b_pred)
            print(f"  - {basis:8s} | RMSLE: {b_rmsle:.4f} | MAE: {b_mae:.2f} | Actual Total: {b_true.sum():.0f} | Pred Total: {b_pred.sum():.0f}")
            metrics_summary.append({
                'year': test_year,
                'target': basis,
                'rmsle': b_rmsle,
                'mae': b_mae,
                'actual_sum': b_true.sum(),
                'pred_sum': b_pred.sum()
            })
            
        # Total daily aggregate
        tot_pivot = test_data.pivot_table(index='date', columns='ob_osnova_name', values=['actual_count', 'predicted_count'])
        tot_act = tot_pivot['actual_count'].sum(axis=1).values
        tot_pred = tot_pivot['predicted_count'].sum(axis=1).values
        tot_rmsle = rmsle(tot_act, tot_pred)
        tot_mae = mae(tot_act, tot_pred)
        print(f"  * TOTAL    | RMSLE: {tot_rmsle:.4f} | MAE: {tot_mae:.2f} | Actual Total: {tot_act.sum():.0f} | Pred Total: {tot_pred.sum():.0f}")
        
        metrics_summary.append({
            'year': test_year,
            'target': 'TOTAL',
            'rmsle': tot_rmsle,
            'mae': tot_mae,
            'actual_sum': tot_act.sum(),
            'pred_sum': tot_pred.sum()
        })
        
        # Evaluate hourly on test year
        raw_test = raw_df[raw_df['year'] == test_year]
        hourly_act = raw_test.groupby([raw_test['reg_ts'].dt.date.rename('date'), raw_test['reg_ts'].dt.hour.rename('hour')]).size().reset_index(name='actual_hourly')
        
        hourly_pred_df = decompose_to_hourly(test_data, hourly_profiles)
        hourly_merged = pd.merge(
            hourly_pred_df,
            hourly_act,
            left_on=['date', 'hour'],
            right_on=[hourly_act['date'].astype(str), 'hour'],
            how='left'
        )
        hourly_merged['actual_hourly'] = hourly_merged['actual_hourly'].fillna(0)
        h_rmsle = rmsle(hourly_merged['actual_hourly'].values, hourly_merged['predicted_total'].values)
        print(f"  * HOURLY TOTAL RMSLE: {h_rmsle:.4f}")
        metrics_summary.append({
            'year': test_year,
            'target': 'HOURLY_TOTAL',
            'rmsle': h_rmsle,
            'mae': mae(hourly_merged['actual_hourly'].values, hourly_merged['predicted_total'].values),
            'actual_sum': hourly_merged['actual_hourly'].sum(),
            'pred_sum': hourly_merged['predicted_total'].sum()
        })
        
    return pd.DataFrame(metrics_summary)

def generate_target_forecast(forecaster, year, hourly_profiles, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Build grid & features for target year
    grid = generate_campaign_grid([year])
    feat = extract_features(grid)
    
    # 2. Predict daily
    preds = forecaster.predict(feat)
    grid['predicted_count'] = np.round(preds, 1)
    
    # Pivot for clean daily presentation
    daily_pivot = grid.pivot(index=['year', 'date'], columns='ob_osnova_name', values='predicted_count').reset_index()
    daily_pivot.columns.name = None
    daily_pivot['date'] = pd.to_datetime(daily_pivot['date'])
    daily_pivot['day_of_week'] = daily_pivot['date'].dt.day_name()
    daily_pivot['Всего_заявлений'] = np.round(daily_pivot['Бюджет'] + daily_pivot['Договор'] + daily_pivot['Гослиния'], 1)
    
    # Reorder columns
    cols_order = ['year', 'date', 'day_of_week', 'Бюджет', 'Договор', 'Гослиния', 'Всего_заявлений']
    daily_pivot = daily_pivot[cols_order]
    
    # 3. Generate hourly forecast & staffing
    hourly_df = decompose_to_hourly(grid, hourly_profiles, apps_per_operator_hour=5.0, min_standby_operators=1)
    
    # 4. Save to CSV and Excel
    daily_csv = os.path.join(output_dir, f'forecast_{year}_daily.csv')
    daily_xlsx = os.path.join(output_dir, f'forecast_{year}_daily.xlsx')
    hourly_csv = os.path.join(output_dir, f'forecast_{year}_hourly.csv')
    hourly_xlsx = os.path.join(output_dir, f'forecast_{year}_hourly.xlsx')
    
    daily_pivot.to_csv(daily_csv, index=False, encoding='utf-8-sig')
    daily_pivot.to_excel(daily_xlsx, index=False)
    hourly_df.to_csv(hourly_csv, index=False, encoding='utf-8-sig')
    hourly_df.to_excel(hourly_xlsx, index=False)
    
    total_apps = daily_pivot['Всего_заявлений'].sum()
    budget_apps = daily_pivot['Бюджет'].sum()
    contract_apps = daily_pivot['Договор'].sum()
    goslin_apps = daily_pivot['Гослиния'].sum()
    peak_row = daily_pivot.loc[daily_pivot['Всего_заявлений'].idxmax()]
    peak_date = peak_row['date'].strftime('%Y-%m-%d')
    peak_val = peak_row['Всего_заявлений']
    max_ops = hourly_df['recommended_operators'].max()
    
    print(f"\n==================================================")
    print(f" FORECAST SUMMARY FOR {year}")
    print(f"==================================================")
    print(f"Total Predicted Applications: {total_apps:,.0f}")
    print(f"  - Бюджет:   {budget_apps:,.0f} ({budget_apps/total_apps*100:.1f}%)")
    print(f"  - Договор:  {contract_apps:,.0f} ({contract_apps/total_apps*100:.1f}%)")
    print(f"  - Гослиния: {goslin_apps:,.0f} ({goslin_apps/total_apps*100:.1f}%)")
    print(f"Peak Day: {peak_date} ({peak_row['day_of_week']}) with {peak_val:.0f} applications")
    print(f"Max Operators Needed Simultaneously: {max_ops} operators (in peak hour)")
    print(f"Files saved: {daily_csv}, {hourly_csv}")
    
    return daily_pivot, hourly_df

def main():
    print("Starting Admissions Campaign Forecasting Pipeline...")
    os.makedirs('output', exist_ok=True)
    raw_df = load_raw_data('processed_data/included_ks.csv')
    hourly_profiles = compute_hourly_profiles(raw_df)
    
    # Prepare historical training dataset
    years = sorted(raw_df['year'].unique())
    daily_df = build_daily_dataset(raw_df, years=years)
    df_feat = extract_features(daily_df)
    
    # 1. Run backtesting validation
    val_results = run_backtesting(df_feat, raw_df, hourly_profiles)
    val_results.to_csv('output/validation_metrics.csv', index=False)
    
    # 2. Train final model on ALL historical data (2014-2023, excluding 2020 covid anomaly)
    print("\n" + "=" * 60)
    print(" 2. TRAINING FINAL PRODUCTION MODEL (2014-2023)")
    print("=" * 60)
    train_final = df_feat[df_feat['year'] != 2020].copy()
    final_forecaster = AdmissionsForecaster(use_ensemble=True)
    final_forecaster.fit(train_final, train_final['actual_count'])
    
    feat_imp = final_forecaster.get_feature_importances()
    print("\nTop 10 Feature Importances:")
    print(feat_imp.head(10).to_string(index=False))
    feat_imp.to_csv('output/feature_importances.csv', index=False)
    
    # 3. Generate Forecast for 2024
    generate_target_forecast(final_forecaster, 2024, hourly_profiles, output_dir='output')
    
    # 4. Generate Forecast for 2026
    generate_target_forecast(final_forecaster, 2026, hourly_profiles, output_dir='output')
    
    print("\n[SUCCESS] Pipeline completed successfully!")

if __name__ == '__main__':
    main()

