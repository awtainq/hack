"""Time validation for a model whose stream predictions are trained jointly."""

from dataclasses import asdict
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from joint_admissions_model import JointAdmissionsModel, JointConfig, load_counts


def main():
    output = Path("output/v71")
    output.mkdir(parents=True, exist_ok=True)
    counts = load_counts(Path("processed_data/registered.csv"))
    years = (2017, 2018, 2019, 2021, 2022)
    configs = [JointConfig(leaves=l, iterations=n, auxiliary_weight=a, half_life=h)
               for l,n,a,h in product((7,15), (250,500), (.1,.5), (3.,6.))]
    records, predictions = [], {}
    for i, config in enumerate(configs):
        for year in years:
            forecast = JointAdmissionsModel(config).fit(counts, year).predict()
            forecast["actual"] = counts.loc[counts.index.year == year].sum(axis=1).to_numpy()
            forecast["forecast"] = forecast.forecast.round(2)
            forecast["year"] = year
            error = np.log1p(forecast.actual) - np.log1p(forecast.forecast)
            records.append({"config_id": i, "year": year, **asdict(config), "rmsle": np.sqrt(np.mean(error**2))})
            predictions[i,year] = forecast
        average = np.mean([row["rmsle"] for row in records if row["config_id"]==i])
        print(f"{i+1}/{len(configs)}: {average:.5f}", flush=True)
    metrics = pd.DataFrame(records)
    metrics.to_csv(output/"search_metrics.csv",index=False)
    selected_id = int(metrics.groupby("config_id").rmsle.mean().idxmin())
    config = configs[selected_id]
    selected = metrics.loc[metrics.config_id==selected_id,["year","rmsle"]].copy()
    diagnostic = JointAdmissionsModel(config).fit(counts,2023).predict()
    diagnostic["actual"] = counts.loc[counts.index.year==2023].sum(axis=1).to_numpy()
    diagnostic["forecast"] = diagnostic.forecast.round(2)
    diagnostic["year"] = 2023
    error = np.log1p(diagnostic.actual)-np.log1p(diagnostic.forecast)
    selected = pd.concat([selected,pd.DataFrame([{"year":2023,"rmsle":np.sqrt(np.mean(error**2))}])])
    selected.to_csv(output/"backtest_metrics.csv",index=False)
    pd.concat([predictions[selected_id,year] for year in years]+[diagnostic]).to_csv(output/"backtest_daily.csv",index=False)
    (output/"selected_config.json").write_text(json.dumps(asdict(config),indent=2)+"\n")
    model = JointAdmissionsModel(config).fit(counts,2024)
    model.save(output/"model.joblib")
    forecast=model.predict()
    forecast.to_csv(output/"degree_predictions.csv",index=False,float_format="%.6f")
    forecast[["date","forecast"]].to_csv("forecast_2024_v71_joint.csv",index=False,float_format="%.2f")
    print(selected.to_string(index=False))
    print(asdict(config))


if __name__=="__main__":
    main()
