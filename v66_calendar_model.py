#!/usr/bin/env python3
"""Direct calendar regression from raw registrations, with zero-day training.

Each candidate is one independent model of log1p(total daily registrations).
No previous forecast or leaderboard result is read. Selection uses rolling
historical seasons. Sundays / Saturday closures are estimated from history.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import catboost
import sklearn
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
REG = ROOT/'processed_data/registered.csv'
OUT = ROOT/'output/v66'
YEARS = (2017, 2018, 2019, 2021, 2022, 2023)
SELECT_YEARS = YEARS[:-1]
NUMERIC = ['day','doc','week','month','workday','dist_bak','dist_mag','workdist_bak',
           'workdist_mag','after_bak','after_mag','bak_dow','mag_dow','dist_diploma',
           'dist_peak','dist_orders','is_first_day','since_monday','is_first_week',
           'is_august','is_saturday','is_monday','sin_progress','cos_progress']
CATS = ['weekday','phase','sat_slot']


@dataclass(frozen=True)
class Config:
    family: str = 'cat'
    depth: int = 4
    iterations: int = 400
    penalty: float = 10.
    half_life: float = 6.
    zero_weight: float = 1.
    regime: bool = False
    fit_schedule: bool = False


def load_source(path=REG):
    r = pd.read_csv(path, usecols=['reg_ts','zayav_type_name'])
    r['date'] = pd.to_datetime(r.reg_ts).dt.normalize()
    return r


def nominal_ends(year):
    mag = pd.Timestamp(f'{year}-07-29')
    while mag.dayofweek >= 5:
        mag -= pd.Timedelta(days=1)
    # The 2021 federal calendar differed from the usual July 25 deadline.
    # Source and caveats for other years: output/v66/calendar_sources.json.
    bak_md='07-29' if year==2021 else '07-25'
    return {'bak':pd.Timestamp(f'{year}-{bak_md}'),'mag':mag}


def historical_ends(history):
    result = {}
    for year in sorted(history.date.dt.year.unique()):
        result[int(year)] = nominal_ends(year)
        for key,prefix,ratio in [('bak','Бакалавр',.2),('mag','Магистр',.25)]:
            r = history[(history.date.dt.year==year)&history.zayav_type_name.str.startswith(prefix)]
            s = r.groupby('date').size().reindex(pd.date_range(f'{year}-06-20',f'{year}-08-31'),fill_value=0)
            threshold = ratio*s.loc[f'{year}-07-01':f'{year}-07-25'].max()
            if threshold <= 0: continue
            for date,value in s.loc[f'{year}-07-20':f'{year}-08-20'].items():
                following = s.loc[date+pd.Timedelta(days=1):date+pd.Timedelta(days=3)]
                if value >= threshold and len(following)==3 and (following<threshold).all():
                    result[int(year)][key] = date
                    break
    return result


def make_grid(reg,year):
    first,last = int(reg.date.dt.year.min()),int(reg.date.dt.year.max())
    if year > last+1 or year <= first:
        raise ValueError('Target year requires contiguous preceding observed seasons')
    history = reg[reg.date.dt.year<year]
    ends = historical_ends(history)
    ends[year] = nominal_ends(year)
    if set(range(first,year))-set(ends):
        raise ValueError('Missing historical seasons')
    g = pd.DataFrame({'date':[d for y in range(first,year+1)
                             for d in pd.date_range(f'{y}-06-20',f'{y}-08-31')]})
    g['year']=g.date.dt.year;g['dow']=g.date.dt.dayofweek
    g['y']=g.date.map(reg.groupby('date').size()).fillna(0.).astype(float)
    g['day']=g.date.dt.day;g['month']=g.date.dt.month;g['weekday']=g.dow.astype(str)
    g['era']=np.select([g.year==2020,g.year>=2021],['covid','digital'],'earlier')
    start=pd.to_datetime(g.year.astype(str)+'-06-20')
    g['doc']=(g.date-start).dt.days;g['week']=g.doc//7
    g['workday']=np.busday_count(start.values.astype('datetime64[D]'),g.date.values.astype('datetime64[D]'),weekmask='1111110')
    for name in ('bak','mag'):
        end=pd.to_datetime([ends[y][name] for y in g.year])
        g[f'dist_{name}']=(g.date-end).dt.days
        g[f'workdist_{name}']=np.busday_count(end.values.astype('datetime64[D]'),g.date.values.astype('datetime64[D]'),weekmask='1111110')
        g[f'after_{name}']=(g[f'dist_{name}']>0).astype(int)
        g[f'{name}_dow']=end.dayofweek
    for name,md in [('diploma','06-28'),('peak','07-08'),('orders','08-09')]:
        g[f'dist_{name}']=(g.date-pd.to_datetime(g.year.astype(str)+'-'+md)).dt.days
    g['is_first_day']=(g.doc==0).astype(int);g['since_monday']=g.dow
    g['is_first_week']=(g.doc<4).astype(int);g['is_august']=(g.month==8).astype(int)
    g['is_saturday']=(g.dow==5).astype(int);g['is_monday']=(g.dow==0).astype(int)
    g['sin_progress']=np.sin(np.pi*g.doc/72);g['cos_progress']=np.cos(np.pi*g.doc/72)
    g['phase']=np.select([g.doc<4,g.month==6,g.dist_bak<=0,g.dist_mag<=0,(g.month==8)&(g.day>21)],
                         ['start','ramp','bachelor','master','late'],'post')
    first_sat=(g[g.dow==5].groupby('year').date.min())
    g['sat_slot']=np.select([(g.dow==5)&(g.date==g.year.map(first_sat)),
                            (g.dow==5)&(g.month==8)&(g.day>21),
                            (g.dow==5)&(g.month==8),g.dow==5],['first','late','aug','july'],'none')
    return g


def closed_mask(tr,te):
    sat=tr[(tr.dow==5)&(tr.year!=2020)]
    weights=np.exp(-np.log(2)*(int(te.year.iloc[0])-sat.year)/6)
    rates=(weights*(sat.y==0)).groupby(sat.sat_slot).sum()/weights.groupby(sat.sat_slot).sum()
    return ((te.dow==6)|te.sat_slot.map(lambda s:s!='none' and rates.get(s,0)>.5)).to_numpy()


def feature_frame(g,cfg):
    columns=NUMERIC+CATS+(['era'] if cfg.regime else [])
    x=g[columns].copy()
    if cfg.family=='lgb':
        levels={'weekday':[str(x) for x in range(7)],'phase':['start','ramp','bachelor','master','late','post'],
                'sat_slot':['first','late','aug','july','none'],'era':['earlier','covid','digital']}
        for name in CATS+(['era'] if cfg.regime else []):
            x[name]=pd.Categorical(x[name],categories=levels[name])
    return x


class SmoothCalendar:
    """Penalized spline regression, including seasonal weekday interactions."""
    def __init__(self,knots,alpha,regime):
        self.knots=knots;self.alpha=alpha;self.regime=regime

    def design(self,g,fit=False):
        values=g[['doc','dist_bak','dist_mag']].to_numpy()
        if fit:
            self.spline=SplineTransformer(n_knots=self.knots,degree=3,include_bias=False,extrapolation='linear')
            smooth=self.spline.fit_transform(values)
        else:smooth=self.spline.transform(values)
        dow=np.eye(7)[g.dow.to_numpy()][:,:6]
        # Degree of freedom is small, and ridge regularizes weekday curves.
        seasonal=smooth[:,:self.knots+1]
        weekday_curves=(seasonal[:,:,None]*dow[:,None,:]).reshape(len(g),-1)
        phase=np.column_stack([(g.phase==p).to_numpy(float) for p in ['start','ramp','bachelor','master','late','post']])
        extras=g[['after_bak','after_mag','is_first_day','is_first_week']].to_numpy()
        result=np.column_stack([smooth,dow,weekday_curves,phase,extras])
        if self.regime:
            result=np.column_stack([result,(g.year>=2021).to_numpy(float)])
        return result

    def fit(self,g,y,weights):
        self.reg=Ridge(alpha=self.alpha)
        self.reg.fit(self.design(g,True),y,sample_weight=weights)
        return self

    def predict(self,g):return self.reg.predict(self.design(g))


def fit_model(tr,year,cfg):
    # Rows are selected by calendar, never discarded because the target is zero.
    d=tr[tr.dow!=6].copy()
    if cfg.fit_schedule:
        d=d[~d.sat_slot.isin(['first','late'])].copy()
    weights=np.exp(-np.log(2)*(year-d.year.to_numpy())/cfg.half_life)
    weights*=np.where(d.year==2020,.1,1.)
    weights*=np.where(d.y==0,cfg.zero_weight,1.)
    y=np.log1p(d.y.to_numpy())
    if cfg.family=='spline':
        model=SmoothCalendar(cfg.depth,cfg.penalty,cfg.regime)
        with threadpool_limits(limits=1):model.fit(d,y,weights)
    elif cfg.family=='cat':
        model=CatBoostRegressor(iterations=cfg.iterations,depth=cfg.depth,learning_rate=.035,
                               l2_leaf_reg=cfg.penalty,loss_function='RMSE',random_seed=66123,
                               thread_count=1,verbose=False,allow_writing_files=False,
                               one_hot_max_size=16,bootstrap_type='Bayesian',bagging_temperature=.5)
        model.fit(feature_frame(d,cfg),y,sample_weight=weights,cat_features=CATS+(['era'] if cfg.regime else []))
    else:
        model=LGBMRegressor(n_estimators=cfg.iterations,learning_rate=.035,num_leaves=cfg.depth,
                            min_child_samples=12,reg_lambda=cfg.penalty,n_jobs=1,verbosity=-1,random_state=66123)
        model.fit(feature_frame(d,cfg),y,sample_weight=weights)
    return model


def predict_log(model,tr,te,cfg):
    values=model.predict(te if cfg.family=='spline' else feature_frame(te,cfg))
    if not np.isfinite(values).all():
        raise ValueError('Model produced nonfinite log predictions')
    return np.where(closed_mask(tr,te),0.,np.maximum(0.,values))


def metrics(te,z):
    err=z-np.log1p(te.y.to_numpy())
    return {'rmsle':float(np.sqrt(np.mean(err**2))),
            'weekday_rmsle':float(np.sqrt(np.mean(err[te.dow<5]**2))),
            'august_rmsle':float(np.sqrt(np.mean(err[(te.month==8)&(te.dow<5)]**2)))}


def candidates():
    configs=[Config('cat',depth,iters,10.,half,1.,False)
             for depth,iters,half in itertools.product((3,5,7),(300,700),(3.,6.))]
    configs += [Config('cat',depth,500,10.,6.,zw,era)
                for depth,zw,era in itertools.product((3,5),(.25,1.),(False,True))]
    configs += [Config('lgb',leaves,iters,5.,6.)
                for leaves,iters in itertools.product((7,15),(150,350))]
    configs += [Config('spline',knots,0,alpha,6.)
                for knots,alpha in itertools.product((4,6,8),(1.,10.,100.))]
    configs += [Config('cat',depth,iters,10.,half,1.,False,True)
                for depth,iters,half in itertools.product((3,5),(300,700),(3.,6.))]
    configs += [Config('spline',knots,0,alpha,6.,1.,False,True)
                for knots,alpha in itertools.product((4,8),(1.,10.))]
    return list(dict.fromkeys(configs))


def search():
    OUT.mkdir(parents=True,exist_ok=True)
    reg=load_source();folds={}
    for year in YEARS:
        g=make_grid(reg,year);folds[year]=(g[g.year<year],g[g.year==year].reset_index(drop=True))
    records=[];configs=candidates()
    for number,cfg in enumerate(configs):
        start=time.monotonic()
        for year,(tr,te) in folds.items():
            model=fit_model(tr,year,cfg);z=predict_log(model,tr,te,cfg)
            records.append({'config_id':number,'year':year,**asdict(cfg),**metrics(te,z)})
        results=pd.DataFrame(records);results.to_csv(OUT/'search_metrics.csv',index=False)
        part=results[(results.config_id==number)&results.year.isin(SELECT_YEARS)]
        criterion=.5*part.rmsle.mean()+.5*part[part.year>=2021].rmsle.mean()
        print(f'{number+1}/{len(configs)} {asdict(cfg)} selection={criterion:.5f} seconds={time.monotonic()-start:.1f}',flush=True)
    ranking=[]
    for number,part in results.groupby('config_id'):
        selection=part[part.year.isin(SELECT_YEARS)]
        score=.5*selection.rmsle.mean()+.5*selection[selection.year>=2021].rmsle.mean()
        ranking.append({'config_id':number,'selection':score,'check_2023':float(part[part.year==2023].rmsle.iloc[0])})
    rank=pd.DataFrame(ranking).sort_values('selection');rank.to_csv(OUT/'selection.csv',index=False)
    chosen=configs[int(rank.iloc[0].config_id)]
    (OUT/'selected_config.json').write_text(json.dumps(asdict(chosen),indent=2)+'\n')
    print(rank.head(12).to_string(index=False),flush=True)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def backtest():
    cfg=Config(**json.loads((OUT/'selected_config.json').read_text()))
    reg=load_source();rows=[];daily=[]
    for year in YEARS:
        g=make_grid(reg,year);tr=g[g.year<year];te=g[g.year==year].reset_index(drop=True)
        model=fit_model(tr,year,cfg);z=predict_log(model,tr,te,cfg)
        rows.append({'year':year,**metrics(te,z)})
        daily.append(te[['date','year','y','dow']].assign(log_forecast=z,forecast=np.round(np.expm1(z),2),closed=closed_mask(tr,te)))
    result=pd.DataFrame(rows);result.to_csv(OUT/'backtest_metrics.csv',index=False)
    pd.concat(daily,ignore_index=True).to_csv(OUT/'backtest_daily.csv',index=False)
    print(result.to_string(index=False),flush=True)


def train(year,reuse=False):
    OUT.mkdir(parents=True,exist_ok=True)
    if reuse:
        meta=json.loads((OUT/'metadata.json').read_text())
        expected={'input_sha256':sha256(REG),'code_sha256':sha256(Path(__file__).resolve()),
                  'model_sha256':sha256(OUT/'model.joblib')}
        if meta['year']!=year or any(meta.get(k)!=v for k,v in expected.items()):
            raise ValueError('Saved model, source, year or inputs changed; run --train')
        cfg=Config(**meta['config'])
    else:
        cfg=Config(**json.loads((OUT/'selected_config.json').read_text()))
    reg=load_source();g=make_grid(reg,year);tr=g[g.year<year];te=g[g.year==year].reset_index(drop=True)
    model=joblib.load(OUT/'model.joblib') if reuse else fit_model(tr,year,cfg)
    z=predict_log(model,tr,te,cfg)
    out=ROOT/f'forecast_{year}_v66_calendar.csv'
    forecast=pd.DataFrame({'date':te.date.dt.strftime('%Y-%m-%d'),'forecast':np.round(np.expm1(z),2)})
    if len(forecast)!=73 or not forecast.date.is_unique or not np.isfinite(forecast.forecast).all() or (forecast.forecast<0).any():
        raise ValueError('Invalid output forecast')
    if reuse and sha256(OUT/'model.joblib')!=meta['model_sha256']:
        raise ValueError('Model changed during prediction')
    forecast.to_csv(out,index=False)
    if not reuse:joblib.dump(model,OUT/'model.joblib')
    prior=json.loads((OUT/'metadata.json').read_text()) if (OUT/'metadata.json').exists() else {}
    score=prior.get('public_score') if prior.get('output_sha256')==sha256(out) else None
    fit_rows=tr[tr.dow!=6]
    if cfg.fit_schedule:fit_rows=fit_rows[~fit_rows.sat_slot.isin(['first','late'])]
    meta={'year':year,'config':asdict(cfg),'input_sha256':sha256(REG),'code_sha256':sha256(Path(__file__).resolve()),
          'model_sha256':sha256(OUT/'model.joblib'),'output_sha256':sha256(out),'public_score':score,
          'source':'Direct output of one model fitted to raw registrations',
          'training_days':len(fit_rows),'training_zero_days':int((fit_rows.y==0).sum()),
          'closed_dates':list(te.loc[closed_mask(tr,te),'date'].dt.strftime('%Y-%m-%d')),
          'total_forecast':float(forecast.forecast.sum()),'selection_years':list(SELECT_YEARS),
          'diagnostic_year':2023,'catboost_version':catboost.__version__,'sklearn_version':sklearn.__version__,
          'numpy_version':np.__version__,'pandas_version':pd.__version__}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    if cfg.family=='cat':
        pd.DataFrame({'feature':feature_frame(te,cfg).columns,'importance':model.get_feature_importance()}).sort_values('importance',ascending=False).to_csv(OUT/'feature_importance.csv',index=False)
        if not reuse:model.save_model(str(OUT/'model.cbm'))
    print('Saved',out,'sum',forecast.forecast.sum(),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--search',action='store_true');p.add_argument('--train',action='store_true');p.add_argument('--year',type=int,default=2024)
    p.add_argument('--backtest',action='store_true');p.add_argument('--predict',action='store_true')
    a=p.parse_args()
    if a.predict and (a.search or a.train or a.backtest):p.error('--predict must be used separately')
    if a.search:search()
    if a.backtest:backtest()
    if a.train:train(a.year)
    if a.predict:train(a.year,reuse=True)


if __name__=='__main__':main()
