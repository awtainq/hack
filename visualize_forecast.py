import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Set styles for clean publication-quality figures
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linestyle'] = '--'

def plot_daily_forecasts(output_dir='output', reports_dir='reports'):
    os.makedirs(reports_dir, exist_ok=True)
    
    df_2024 = pd.read_csv(os.path.join(output_dir, 'forecast_2024_daily.csv'))
    df_2026 = pd.read_csv(os.path.join(output_dir, 'forecast_2026_daily.csv'))
    
    df_2024['date'] = pd.to_datetime(df_2024['date'])
    df_2026['date'] = pd.to_datetime(df_2026['date'])
    
    # 1. Plot 2024 Daily Forecast by Basis
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.plot(df_2024['date'], df_2024['Всего_заявлений'], label='Всего заявлений', color='#1f77b4', linewidth=2.5)
    ax.plot(df_2024['date'], df_2024['Бюджет'], label='Бюджет', color='#2ca02c', linewidth=1.8, linestyle='-')
    ax.plot(df_2024['date'], df_2024['Договор'], label='Договор (контракт)', color='#ff7f0e', linewidth=1.8, linestyle='-')
    ax.plot(df_2024['date'], df_2024['Гослиния'], label='Гослиния', color='#9467bd', linewidth=1.2, linestyle=':')
    
    # Annotate key milestones
    ax.axvline(pd.Timestamp('2024-06-20'), color='#888888', linestyle='--', alpha=0.7, label='Старт приёма (20 июня)')
    ax.axvline(pd.Timestamp('2024-07-25'), color='#d62728', linestyle='--', alpha=0.8, label='Дедлайн бюджета (25 июля)')
    ax.axvline(pd.Timestamp('2024-08-09'), color='#bcbd22', linestyle='--', alpha=0.7, label='Приказы бакалавриат (9 авг)')
    
    ax.set_title('Прогноз посуточного наплыва абитуриентов на 2024 год по основам обучения', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Дата', fontsize=11)
    ax.set_ylabel('Количество заявлений в сутки', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.grid(True)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(reports_dir, 'forecast_2024_daily_breakdown.png'))
    plt.close()
    
    # 2. Plot 2026 Daily Forecast by Basis
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    ax.plot(df_2026['date'], df_2026['Всего_заявлений'], label='Всего заявлений', color='#1f77b4', linewidth=2.5)
    ax.plot(df_2026['date'], df_2026['Бюджет'], label='Бюджет', color='#2ca02c', linewidth=1.8, linestyle='-')
    ax.plot(df_2026['date'], df_2026['Договор'], label='Договор (контракт)', color='#ff7f0e', linewidth=1.8, linestyle='-')
    ax.plot(df_2026['date'], df_2026['Гослиния'], label='Гослиния', color='#9467bd', linewidth=1.2, linestyle=':')
    
    ax.axvline(pd.Timestamp('2026-06-20'), color='#888888', linestyle='--', alpha=0.7, label='Старт приёма (20 июня)')
    ax.axvline(pd.Timestamp('2026-07-25'), color='#d62728', linestyle='--', alpha=0.8, label='Дедлайн бюджета (25 июля)')
    ax.axvline(pd.Timestamp('2026-08-09'), color='#bcbd22', linestyle='--', alpha=0.7, label='Приказы бакалавриат (9 авг)')
    
    ax.set_title('Прогноз посуточного наплыва абитуриентов на 2026 год по основам обучения', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Дата', fontsize=11)
    ax.set_ylabel('Количество заявлений в сутки', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.grid(True)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(reports_dir, 'forecast_2026_daily_breakdown.png'))
    plt.close()

def plot_operator_schedule(output_dir='output', reports_dir='reports'):
    os.makedirs(reports_dir, exist_ok=True)
    h_2024 = pd.read_csv(os.path.join(output_dir, 'forecast_2024_hourly.csv'))
    h_2024['date'] = pd.to_datetime(h_2024['date'])
    
    # Filter working hours: 10..17
    work_hours = h_2024[(h_2024['hour'] >= 10) & (h_2024['hour'] <= 17)].copy()
    
    # Max operators per day
    daily_ops = work_hours.groupby('date')['recommended_operators'].max().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    ax.bar(daily_ops['date'], daily_ops['recommended_operators'], color='#3498db', alpha=0.85, width=0.8, label='Пиковое число операторов в день')
    ax.axhline(daily_ops['recommended_operators'].mean(), color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Среднее по кампании ({daily_ops["recommended_operators"].mean():.1f} чел.)')
    
    ax.set_title('Потребность в операторах приёмной комиссии (пиковая численность в смену, 2024 г.)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Дата', fontsize=11)
    ax.set_ylabel('Рекомендуемое число операторов (чел.)', fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.grid(True, axis='y')
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(os.path.join(reports_dir, 'operator_staffing_curve_2024.png'))
    plt.close()
    
    # 2. Hourly Heatmap of Applications for a Peak Week (July 8 - July 14, 2024)
    peak_week = h_2024[(h_2024['date'] >= '2024-07-08') & (h_2024['date'] <= '2024-07-14')].copy()
    pivot_heat = peak_week.pivot(index='hour', columns='date', values='recommended_operators')
    pivot_heat.columns = [d.strftime('%a %d.%m') for d in pivot_heat.columns]
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    cax = ax.imshow(pivot_heat.loc[9:18], cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(pivot_heat.columns)))
    ax.set_xticklabels(pivot_heat.columns, rotation=0, fontsize=10)
    ax.set_yticks(range(len(range(9, 19))))
    ax.set_yticklabels([f'{h:02d}:00' for h in range(9, 19)], fontsize=10)
    
    # Annotate values inside cells
    for i in range(len(range(9, 19))):
        h = range(9, 19)[i]
        for j in range(len(pivot_heat.columns)):
            val = pivot_heat.iloc[h, j]
            color = 'white' if val > 15 else 'black'
            ax.text(j, i, int(val), ha='center', va='center', color=color, fontweight='bold', fontsize=9)
            
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label('Число операторов в смене', fontsize=10)
    ax.set_title('График дежурства операторов на пиковой неделе (8–14 июля 2024 г.)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('День недели и дата', fontsize=11)
    ax.set_ylabel('Час суток', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(reports_dir, 'peak_week_operator_heatmap.png'))
    plt.close()

if __name__ == '__main__':
    print("Generating forecast and staffing charts...")
    plot_daily_forecasts()
    plot_operator_schedule()
    print("Charts successfully saved to reports/ directory!")

