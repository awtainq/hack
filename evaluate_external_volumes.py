"""Audit a conditional reconstruction using published annual applicant totals.

This is explicitly NOT pre-campaign forecasting: the target year's published
aggregate is supplied. Daily target labels remain held out from model fitting.
The conversion between report totals and registration rows uses earlier years.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from admissions_profile_model import ProfileConfig, ProfileModel, load_counts


def published_totals(sources):
    # Keep a declared revision order; never choose the value closest to targets.
    result = {"bachelor": {}, "master_miep": {}}
    for source_name in ("report_2019", "report_2020", "report_2022", "report_2023", "report_2024"):
        for degree in result:
            result[degree].update({int(year): value for year, value in sources[source_name].get(degree, {}).items()})
    return result


def converted_volumes(counts, year, reports):
    annual = counts.loc[counts.index.year < year].groupby(lambda date: date.year).sum()
    estimates, audit = {}, []
    for degree in reports:
        preceding = [y for y in sorted(reports[degree]) if y < year and y in annual.index and y != 2020][-3:]
        ratios = [annual.loc[y, degree] / reports[degree][y] for y in preceding]
        conversion = float(np.exp(np.mean(np.log(ratios)))) if ratios else 1.0
        estimates[degree] = reports[degree][year] * conversion
        audit.append({"year": year, "degree": degree, "published": reports[degree][year],
                      "conversion": conversion, "volume_input": estimates[degree],
                      "conversion_years": preceding})
    return estimates, audit


def main():
    root = Path("output/v69")
    counts = load_counts(Path("processed_data/registered.csv"))
    config = ProfileConfig(**json.loads((root / "selected_config.json").read_text()))
    sources_path = Path("external_data/unecon/sources.json")
    reports = published_totals(json.loads(sources_path.read_text())["sources"])
    records, audits = [], []
    for year in (2019, 2021, 2022, 2023):
        model = ProfileModel(config).fit(counts, year)
        volumes, audit = converted_volumes(counts, year, reports)
        audits.extend(audit)
        actual = counts.loc[counts.index.year == year].sum(axis=1).to_numpy()
        for mode, forecast in (("preseason", model.predict()), ("known_annual_total", model.predict(volumes))):
            error = np.log1p(forecast.forecast.round(2)) - np.log1p(actual)
            records.append({"year": year, "mode": mode, "rmsle": float(np.sqrt(np.mean(error ** 2)))})
    metrics = pd.DataFrame(records)
    metrics.to_csv(root / "external_volume_metrics.csv", index=False)
    volumes, audit = converted_volumes(counts, 2024, reports)
    audits.extend(audit)
    model = ProfileModel(config).fit(counts, 2024)
    prediction = model.predict(volumes)
    prediction.to_csv(root / "conditional_degree_predictions.csv", index=False, float_format="%.6f")
    output = Path("forecast_2024_v69_conditional.csv")
    prediction[["date", "forecast"]].to_csv(output, index=False, float_format="%.2f")
    metadata = {
        "mode": "post_campaign_conditional_reconstruction",
        "config": asdict(config), "volume_inputs": volumes,
        "conversion_audit": audits,
        "sources_sha256": hashlib.sha256(sources_path.read_bytes()).hexdigest(),
        "forecast_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "public_score": None,
        "rules_allow_post_campaign_data": "unknown_pending_user_response",
    }
    (root / "conditional_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(metrics.pivot(index="year", columns="mode", values="rmsle").to_string())
    print("2024 volume inputs:", volumes)
    print("2024 forecast sums:", prediction.select_dtypes("number").sum().to_dict())


if __name__ == "__main__":
    main()
