import pandas as pd

df_ks = pd.read_csv('processed_data/included_ks.csv')
df_ks['reg_ts'] = pd.to_datetime(df_ks['reg_ts'])
df_ks['year'] = df_ks['reg_ts'].dt.year
df_ks['date'] = df_ks['reg_ts'].dt.date

for yr in [2021, 2022, 2023]:
    df_yr = df_ks[df_ks['year'] == yr]
    june20 = pd.Timestamp(f'{yr}-06-20').date()
    
    for osn in ['Бюджет', 'Договор']:
        sub = df_yr[df_yr['ob_osnova_name'] == osn]
        tot = len(sub)
        daily = sub.groupby('date').size()
        cum = daily.cumsum() / tot
        d50 = (cum >= 0.50).idxmax() - june20
        d90 = (cum >= 0.90).idxmax() - june20
        print(f"Year {yr} | {osn:8s}: Total={tot:5d} | 50% at day +{d50.days:2d} | 90% at day +{d90.days:2d} | Peak={daily.idxmax()} ({daily.max()})")

