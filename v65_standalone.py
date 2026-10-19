#!/usr/bin/env python3
"""One CatBoost model trained directly from raw admission records.

Inputs are ONLY processed_data/registered.csv and included_ks.csv. No previous
forecast or leaderboard-score file is read. The total count is predicted by
one fitted model. Optional auxiliary targets share its trees; they are never
summed or blended into the final prediction.

    .venv/bin/python v65_standalone.py --search
    .venv/bin/python v65_standalone.py --train
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
import time

import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
REG = ROOT / "processed_data" / "registered.csv"
KS = ROOT / "processed_data" / "included_ks.csv"
OUT = ROOT / "output" / "v65"
SELECTION_YEARS = (2017, 2018, 2019, 2021, 2022)
CHECK_YEAR = 2023
TARGETS = ["total", "budget", "contract", "bak", "mag", "spo", "other"]
CATS = ["weekday", "era"]
CALENDAR = ["weekday", "era", "year_num", "day", "month", "doc", "week", "is_mon", "is_sat",
            "dist_bak", "dist_mag", "after_bak", "after_mag", "bak_dow", "mag_dow",
            "workday", "workdist_bak", "workdist_mag", "dist_diploma", "dist_orders",
            "sin_progress", "cos_progress"]
LAGS = ["lag1_date", "lag1_weekday", "lag2_weekday", "lag1_week_mean", "lag2_week_mean",
        "lag1_bak_relative", "lag1_mag_relative", "lag1_relative_dow",
        "prior_total", "prior_july", "prior_august", "prior_epgu_share", "prior_growth",
        "prior_was_covid"]


@dataclass(frozen=True)
class Config:
    mode: str = "multi"
    depth: int = 5
    iterations: int = 600
    l2: float = 20.
    half_life: float = 4.
    lags: bool = True
    total_scale: float = 2.


def load_sources(reg_path=REG, ks_path=KS):
    r, k = pd.read_csv(reg_path), pd.read_csv(ks_path)
    r["date"] = pd.to_datetime(r.reg_ts).dt.normalize()
    k["date"] = pd.to_datetime(k.reg_ts).dt.normalize()
    return r, k


def historical_ends(history):
    ends = {}
    for year in sorted(history.date.dt.year.unique()):
        ends[int(year)] = {}
        for key, prefix, fraction in [("bak", "Бакалавр", .2), ("mag", "Магистр", .25)]:
            x = history[(history.date.dt.year == year) & history.zayav_type_name.str.startswith(prefix)]
            s = x.groupby("date").size().reindex(pd.date_range(f"{year}-06-20", f"{year}-08-31"), fill_value=0)
            threshold = fraction*s.loc[f"{year}-07-01":f"{year}-07-25"].max()
            answer = f"{year}-07-25" if key == "bak" else f"{year}-07-29"
            if threshold > 0:
                for date, value in s.loc[f"{year}-07-20":f"{year}-08-20"].items():
                    future = s.loc[date+pd.Timedelta(days=1):date+pd.Timedelta(days=3)]
                    if value >= threshold and len(future) == 3 and (future < threshold).all():
                        answer = date
                        break
            ends[int(year)][key] = pd.Timestamp(answer)
    return ends


def target_ends(year):
    mag = pd.Timestamp(f"{year}-07-29")
    while mag.dayofweek > 4:
        mag -= pd.Timedelta(days=1)
    return {"bak": pd.Timestamp(f"{year}-07-25"), "mag": mag}


def counts_by_target(reg, ks):
    keys = ["abitur_kod", "date", "zayav_type_kod"]
    mapping = ks.assign(is_budget=ks.ob_osnova_name.isin(["Бюджет", "Гослиния"]))
    mapping = mapping.groupby(keys).is_budget.max().reset_index()
    r = reg.merge(mapping, on=keys, how="left", validate="many_to_one")
    if len(r) != len(reg):
        raise ValueError("Funding merge changed the number of registrations")
    series = {"total": r.groupby("date").size()}
    budget = r.is_budget.eq(True)
    series["budget"] = r.loc[budget].groupby("date").size()
    series["contract"] = r.loc[~budget].groupby("date").size()
    degree = np.select([r.zayav_type_name.str.startswith("Бакалавр"),
                        r.zayav_type_name.str.startswith("Магистр"),
                        r.zayav_type_name.str.startswith("СПО")], ["bak", "mag", "spo"], "other")
    for name in ("bak", "mag", "spo", "other"):
        series[name] = r.loc[degree == name].groupby("date").size()
    return series


def make_dataset(reg, ks, year, ends_override=None):
    """Future labels can be retained for scoring but never enter feature construction."""
    latest = int(reg.date.dt.year.max())
    if year > latest+1:
        raise ValueError(f"Missing intervening seasons: latest observed year is {latest}")
    if year <= int(reg.date.dt.year.min()):
        raise ValueError("At least one earlier campaign is required")
    history = reg[reg.date.dt.year < year]
    ends = historical_ends(history)
    ends[year] = ends_override or target_ends(year)
    all_counts = counts_by_target(reg, ks)
    start_year = int(reg.date.dt.year.min())
    g = pd.DataFrame({"date": [d for y in range(start_year, year+1)
                              for d in pd.date_range(f"{y}-06-20", f"{y}-08-31")]})
    g["year"] = g.date.dt.year
    g["dow"] = g.date.dt.dayofweek
    g["weekday"] = g.dow.astype(str)
    g["era"] = np.select([g.year == 2020, g.year >= 2021], ["covid", "digital"], "earlier")
    g["year_num"] = g.year-2014
    g["day"] = g.date.dt.day;g["month"] = g.date.dt.month
    start = pd.to_datetime(g.year.astype(str)+"-06-20")
    g["doc"] = (g.date-start).dt.days;g["week"] = g.doc//7
    g["is_mon"] = (g.dow == 0).astype(int);g["is_sat"] = (g.dow == 5).astype(int)
    for stream in ("bak", "mag"):
        end = pd.to_datetime([ends[y][stream] for y in g.year])
        g[f"dist_{stream}"] = (g.date-end).dt.days
        g[f"after_{stream}"] = (g[f"dist_{stream}"] > 0).astype(int)
        g[f"{stream}_dow"] = end.dayofweek
        g[f"workdist_{stream}"] = np.busday_count(end.values.astype("datetime64[D]"),
                                                  g.date.values.astype("datetime64[D]"), weekmask="1111110")
    g["workday"] = np.busday_count(start.values.astype("datetime64[D]"),
                                    g.date.values.astype("datetime64[D]"), weekmask="1111110")
    for md, name in [("06-28", "diploma"), ("08-09", "orders")]:
        g[f"dist_{name}"] = (g.date-pd.to_datetime(g.year.astype(str)+"-"+md)).dt.days
    g["sin_progress"] = np.sin(np.pi*g.doc/72);g["cos_progress"] = np.cos(np.pi*g.doc/72)
    for name, counts in all_counts.items():
        g[name] = g.date.map(counts).fillna(0.).astype(float)
    # The annual lookup contains only earlier years for any feature row.
    total = all_counts["total"]
    valid_years = set(history.date.dt.year.unique())
    yearly = history.groupby(history.date.dt.year).size()
    july = history[history.date.dt.month == 7].groupby(history.date.dt.year).size()
    august = history[history.date.dt.month == 8].groupby(history.date.dt.year).size()
    epgu = history[history.priem_doc_type_kod == 4].groupby(history.date.dt.year).size()

    def at(date, prior_year):
        return float(total.get(date, 0.)) if prior_year in valid_years else np.nan

    lag_rows = []
    for row in g.itertuples():
        y, d = row.year, row.date
        prior = y-1
        if prior not in valid_years:
            lag_rows.append([np.nan]*len(LAGS))
            continue
        d1 = pd.Timestamp(year=prior, month=d.month, day=d.day)
        d2 = pd.Timestamp(year=y-2, month=d.month, day=d.day)
        shifted = lambda t: t+pd.Timedelta(days=(d.dayofweek-t.dayofweek+3)%7-3)
        w1 = np.mean([at(d1+pd.Timedelta(days=i), prior) for i in range(-3,4)])
        w2 = np.mean([at(d2+pd.Timedelta(days=i), y-2) for i in range(-3,4)])
        bak_date = ends[prior]["bak"]+pd.Timedelta(days=row.dist_bak)
        mag_date = ends[prior]["mag"]+pd.Timedelta(days=row.dist_mag)
        value = [np.log1p(at(d1, prior)), np.log1p(at(shifted(d1), prior)),
                 np.log1p(at(shifted(d2), y-2)), np.log1p(w1), np.log1p(w2),
                 np.log1p(at(bak_date, prior)), np.log1p(at(mag_date, prior)), mag_date.dayofweek,
                 np.log1p(yearly.get(prior, 0.)), np.log1p(july.get(prior, 0.)),
                 np.log1p(august.get(prior, 0.)), epgu.get(prior, 0.)/max(1, yearly.get(prior, 0.)),
                 np.log(yearly.get(prior, 1.)/yearly[y-2]) if y-2 in yearly else np.nan,
                 float(prior == 2020)]
        lag_rows.append(value)
    g[LAGS] = np.asarray(lag_rows)
    if not np.array_equal(g.budget+g.contract, g.total):
        raise ValueError("Funding targets do not sum to total registrations")
    if not np.array_equal(g[["bak","mag","spo","other"]].sum(axis=1), g.total):
        raise ValueError("Degree targets do not sum to total registrations")
    return g


def features(config):
    return CALENDAR+(LAGS if config.lags else [])


def closed_mask(tr, te):
    """Schedule estimated exclusively from past observed seasons; no test labels."""
    def slot(d):
        first = d[(d.dow == 5)].groupby("year").date.transform("min")
        s = pd.Series("", index=d.index)
        sat = d.dow == 5
        s.loc[sat] = "july"
        s.loc[sat & (d.month == 8)] = "aug"
        s.loc[sat & (d.month == 8) & (d.day > 21)] = "late"
        s.loc[first.index[first == d.loc[first.index, "date"]]] = "first"
        return s
    tr, te = tr.copy(), te.copy()
    tr["slot"], te["slot"] = slot(tr), slot(te)
    sat = tr[(tr.dow == 5) & (tr.year != 2020)]
    weights = np.exp(-np.log(2)*(te.year.iloc[0]-sat.year)/6)
    zeros = (weights*(sat.total == 0)).groupby(sat.slot).sum()/weights.groupby(sat.slot).sum()
    mask = (te.dow == 6) | te.slot.map(lambda s: bool(s) and zeros.get(s,0.) > .5)
    return mask.to_numpy()


def fit_model(tr, year, config):
    train = tr[(tr.total > 0) & (tr.dow != 6)]
    weights = np.exp(-np.log(2)*(year-train.year.to_numpy())/config.half_life)
    weights *= np.where(train.year == 2020, .15, 1.)
    if config.mode == "multi":
        target = np.log1p(train[TARGETS].to_numpy());target[:,0] *= config.total_scale
        loss = "MultiRMSE"
    else:
        target = np.log1p(train.total.to_numpy());loss = "RMSE"
    model = CatBoostRegressor(iterations=config.iterations, depth=config.depth, learning_rate=.035,
                              l2_leaf_reg=config.l2, loss_function=loss, random_seed=20261019,
                              thread_count=1, verbose=False, allow_writing_files=False,
                              random_strength=1., bootstrap_type="Bayesian", bagging_temperature=.5)
    with threadpool_limits(limits=1):
        model.fit(train[features(config)], target, cat_features=CATS, sample_weight=weights)
    return model


def predict_log(model, te, config):
    p = np.asarray(model.predict(te[features(config)]))
    return np.maximum(0, p[:,0]/config.total_scale if config.mode == "multi" else p)


def evaluate(model, tr, te, config):
    z = np.where(closed_mask(tr,te),0.,predict_log(model,te,config))
    error = z-np.log1p(te.total.to_numpy())
    return {"rmsle":float(np.sqrt(np.mean(error**2))),
            "weekday_rmsle":float(np.sqrt(np.mean(error[te.dow<5]**2)))}, z


def candidates():
    configs = [Config(mode=mode, depth=depth, iterations=iterations,
                      l2=10. if depth == 3 else 20.)
               for mode,depth,iterations in itertools.product(("single","multi"),(3,5,7),(300,700))]
    configs += [Config(mode=mode,depth=depth,iterations=600,l2=20.,half_life=half)
                for mode,depth,half in itertools.product(("single","multi"),(4,6),(3.,6.))]
    configs += [Config(mode="single",depth=4,iterations=300,l2=10.,lags=False),
                Config(mode="multi",depth=4,iterations=600,l2=20.,lags=False)]
    configs += [Config(mode="multi",depth=3,iterations=700,l2=10.,half_life=half,total_scale=scale)
                for scale,half in itertools.product((1.,4.),(3.,6.))]
    return configs


def search():
    OUT.mkdir(parents=True,exist_ok=True)
    reg,ks = load_sources()
    folds = {}
    for year in SELECTION_YEARS:
        g = make_dataset(reg,ks,year)
        folds[year] = (g[g.year<year],g[g.year==year].reset_index(drop=True))
    records = []
    configs = candidates()
    for index,config in enumerate(configs):
        started = time.monotonic()
        for year,(tr,te) in folds.items():
            model = fit_model(tr,year,config)
            metrics,_ = evaluate(model,tr,te,config)
            records.append({"config_id":index,"year":year,**asdict(config),**metrics})
        table = pd.DataFrame(records);table.to_csv(OUT/"search_metrics.csv",index=False)
        this = table[table.config_id==index]
        overall = this.rmsle.mean();recent=this[this.year>=2021].rmsle.mean()
        score = .5*overall+.5*recent
        print(f"{index+1}/{len(configs)} config={asdict(config)} selection={score:.5f} elapsed={time.monotonic()-started:.1f}s",flush=True)
    results = pd.DataFrame(records)
    ranked=[]
    for index,part in results.groupby("config_id"):
        ranked.append({"config_id":int(index),"selection":float(.5*part.rmsle.mean()+.5*part[part.year>=2021].rmsle.mean()),
                       "mean_rmsle":float(part.rmsle.mean())})
    ranking = pd.DataFrame(ranked).sort_values("selection")
    ranking.to_csv(OUT/"selection.csv",index=False)
    winner = int(ranking.iloc[0].config_id);config = configs[winner]
    (OUT/"selected_config.json").write_text(json.dumps(asdict(config),indent=2)+"\n")
    # The complete 2023 campaign is evaluated only AFTER choosing hyperparameters.
    g = make_dataset(reg,ks,CHECK_YEAR);tr,te=g[g.year<CHECK_YEAR],g[g.year==CHECK_YEAR].reset_index(drop=True)
    model = fit_model(tr,CHECK_YEAR,config);metrics,z=evaluate(model,tr,te,config)
    check={"year":CHECK_YEAR,"config_id":winner,**asdict(config),**metrics}
    (OUT/"check_2023.json").write_text(json.dumps(check,indent=2)+"\n")
    te[["date","total"]].assign(forecast=np.round(np.expm1(z),2)).to_csv(OUT/"check_2023_daily.csv",index=False)
    print("Selected",asdict(config),"2023:",metrics,flush=True)


def train(year=2024, reuse=False):
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"selected_config.json"
    existing = json.loads((OUT/"metadata.json").read_text()) if reuse else None
    if reuse:
        if existing["year"] != year:
            raise ValueError("Saved model belongs to a different forecast year")
        if existing["code_sha256"] != hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest():
            raise ValueError("Feature code changed; retrain the model")
        if existing.get("model_sha256") != hashlib.sha256((OUT/"model.cbm").read_bytes()).hexdigest():
            raise ValueError("Saved model checksum does not match its metadata")
        for source in (REG,KS):
            if hashlib.sha256(source.read_bytes()).hexdigest()!=existing["raw_input_sha256"][source.name]:
                raise ValueError("Raw data changed; retrain the model")
        config=Config(**existing["config"])
    else:
        config=Config(**json.loads(path.read_text())) if path.exists() else Config()
    reg,ks=load_sources();g=make_dataset(reg,ks,year);tr,te=g[g.year<year],g[g.year==year].reset_index(drop=True)
    if reuse:
        model=CatBoostRegressor();model.load_model(str(OUT/"model.cbm"))
    else:
        model=fit_model(tr,year,config)
    # A forecast has no observed target: do not compute a metric against the
    # placeholder zeros in the future calendar grid.
    z=np.where(closed_mask(tr,te),0.,predict_log(model,te,config))
    forecast=pd.DataFrame({"date":te.date.dt.strftime("%Y-%m-%d"),"forecast":np.round(np.expm1(z),2)})
    if len(forecast)!=73 or not np.isfinite(forecast.forecast).all() or (forecast.forecast<0).any():
        raise ValueError("Invalid forecast")
    output=ROOT/f"forecast_{year}_v65_standalone.csv";forecast.to_csv(output,index=False)
    if not reuse:
        model.save_model(str(OUT/"model.cbm"))
    pd.DataFrame({"feature":features(config),"importance":model.get_feature_importance()}).sort_values(
        "importance",ascending=False).to_csv(OUT/"feature_importance.csv",index=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    metadata={"model":"V65 single CatBoost","config":asdict(config),"year":year,
              "selection_years":list(SELECTION_YEARS),"check_year":CHECK_YEAR,
              "raw_input_sha256":{p.name:sha(p) for p in (REG,KS)},"code_sha256":sha(Path(__file__).resolve()),
              "output_sha256":sha(output),"public_score":None,
              "model_sha256":sha(OUT/"model.cbm"),
              "catboost_version":catboost.__version__,"numpy_version":np.__version__,"pandas_version":pd.__version__,
              "training_positive_non_sunday_days":int(((tr.total>0)&(tr.dow!=6)).sum()),
              "closed_dates":list(te.loc[closed_mask(tr,te),"date"].dt.strftime("%Y-%m-%d")),
              "feature_columns":features(config),
              "forecast_source":"Direct first output of one fitted CatBoost model; no previous forecast input"}
    (OUT/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(f"Saved {output.name}, sum={forecast.forecast.sum():.2f}, zeros={(forecast.forecast==0).sum()}",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--search",action="store_true");p.add_argument("--train",action="store_true")
    p.add_argument("--predict",action="store_true",help="Use the saved model without fitting")
    p.add_argument("--year",type=int,default=2024)
    a=p.parse_args()
    if a.predict and (a.train or a.search):
        p.error("--predict is separate from --search/--train")
    if a.predict:
        train(a.year,reuse=True)
        return
    if a.search:search()
    if a.train or not a.search:train(a.year)


if __name__=="__main__":main()
