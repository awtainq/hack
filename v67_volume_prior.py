#!/usr/bin/env python3
"""Soft, explicitly user-supplied volume priors on the verified V61 forecast.

A single calendar classifier learns degree shares from raw registrations.
V61 supplies the baseline daily forecast. Approximate aggregate volumes come
from a separate provenance-bearing JSON, never from hidden target labels.
This is a calibrated V61 scenario, not an independently discovered forecast.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import v66_calendar_model as calendar

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/v67'
BASE=ROOT/'forecast_2024_v61_orthogonal_funding.csv'
PRIORS=ROOT/'volume_priors_2024.json'
GROUPS=['bachelor','master_miep','other']
YEARS=(2017,2018,2019,2021,2022,2023)
SHARE_CONFIG=calendar.Config(depth=4,iterations=300,half_life=3.)


def load_registrations():
    reg=calendar.load_source()
    reg['group']=np.select([reg.zayav_type_name.str.startswith('Бакалавр'),
                            reg.zayav_type_name.str.startswith('Магистр')],GROUPS[:2],'other')
    return reg


def group_counts(reg,dates):
    return reg.groupby(['date','group']).size().unstack(fill_value=0).reindex(index=dates,columns=GROUPS,fill_value=0).fillna(0.)


def fit_shares(reg,year):
    # Dropping the target season from both feature construction and fitting is
    # intentional: the classifier never needs observed target-year totals.
    history=reg[reg.date.dt.year<year]
    grid=calendar.make_grid(history,year)
    tr=grid[(grid.year<year)&(grid.dow!=6)]
    te=grid[grid.year==year].reset_index(drop=True)
    counts=group_counts(history,tr.date).to_numpy(float)
    total=counts.sum(axis=1)
    keep=total>0;tr=tr.loc[keep].reset_index(drop=True);counts=counts[keep];total=total[keep]
    shares=counts/total[:,None]
    x=calendar.feature_frame(tr,SHARE_CONFIG)
    repeated=x.loc[x.index.repeat(len(GROUPS))].reset_index(drop=True)
    labels=np.tile(np.arange(len(GROUPS)),len(tr))
    season_weights=np.exp(-np.log(2)*(year-tr.year.to_numpy())/SHARE_CONFIG.half_life)
    season_weights*=np.where(tr.year==2020,.1,1.)
    weights=(shares*season_weights[:,None]).reshape(-1)
    # Each date contributes its fractional category labels; large July totals
    # do not drown out the different category mix at the end of the campaign.
    model=CatBoostClassifier(iterations=SHARE_CONFIG.iterations,depth=SHARE_CONFIG.depth,
                            learning_rate=.035,l2_leaf_reg=10.,loss_function='MultiClass',
                            classes_count=3,one_hot_max_size=16,random_seed=67123,
                            thread_count=1,verbose=False,allow_writing_files=False)
    model.fit(repeated,labels,sample_weight=weights,cat_features=calendar.CATS)
    predicted=model.predict_proba(calendar.feature_frame(te,SHARE_CONFIG))
    if not np.isfinite(predicted).all() or not np.allclose(predicted.sum(axis=1),1.):
        raise ValueError('Invalid category probabilities')
    return model,te,predicted


def historical_base(d,weights):
    pool=np.round(np.expm1(np.where(d.closed,0.,d.z)),2)
    z=(weights['v59']*np.log1p(d.v59_forecast.to_numpy())+
       weights['v60']*np.log1p(d.v60_forecast.to_numpy())+
       weights['pooled']*np.log1p(pool))
    return np.expm1(np.where(d.closed,0.,np.maximum(0.,z)))


def estimate_volume_uncertainty(reg):
    archived=pd.read_csv(ROOT/'output/v60/backtest_daily.csv',parse_dates=['date'])
    coefficients=json.loads((ROOT/'output/v61/calibration.json').read_text())['weights']
    rows=[]
    for year in YEARS:
        _,te,shares=fit_shares(reg,year)
        d=archived[archived.year==year].reset_index(drop=True)
        if not d.date.equals(te.date):raise ValueError('Historical forecast dates are not aligned')
        base=historical_base(d,coefficients)
        estimated=(base[:,None]*shares).sum(axis=0)
        actual=group_counts(reg,te.date).sum(axis=0).to_numpy()
        for group,predicted,truth in zip(GROUPS,estimated,actual):
            rows.append({'year':year,'group':group,'predicted_volume':predicted,'actual_volume':truth,
                         'log_error':float(np.log(predicted/truth))})
    diagnostics=pd.DataFrame(rows)
    sigma={}
    for group,part in diagnostics.groupby('group'):
        weights=np.exp(-np.log(2)*(2024-part.year.to_numpy())/3.)
        sigma[group]=float(max(.05,np.sqrt(np.average(part.log_error.to_numpy()**2,weights=weights))))
    return sigma,diagnostics


def reconcile(components,prior,sigma):
    uncertainty=float(prior['relative_uncertainty'])
    cap=float(prior['max_prior_weight'])
    if uncertainty<=0 or not 0<cap<=1:raise ValueError('Invalid prior uncertainty / weight cap')
    output=components.copy();rows=[]
    for group in GROUPS:
        baseline=float(components[group].sum())
        target=prior['totals'].get(group)
        if target is None:
            weight=0.;factor=1.
        else:
            if target<=0 or baseline<=0:raise ValueError('Volumes must be positive')
            prior_variance=np.log1p(uncertainty)**2
            weight=min(cap,sigma[group]**2/(sigma[group]**2+prior_variance))
            factor=float(np.exp(weight*np.log(target/baseline)))
        output[group]*=factor
        rows.append({'group':group,'baseline_volume':baseline,'user_prior':target,
                     'base_log_rmse':sigma[group],'prior_weight':weight,'scale_factor':factor,
                     'adjusted_volume':float(output[group].sum())})
    return output,pd.DataFrame(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--priors',type=Path,default=PRIORS)
    args=parser.parse_args();prior=json.loads(args.priors.read_text())
    if prior['year']!=2024 or prior.get('source')!='user_provided':
        raise ValueError('This scenario requires explicit user-provided 2024 volume hints')
    OUT.mkdir(parents=True,exist_ok=True)
    reg=load_registrations();sigma,diagnostics=estimate_volume_uncertainty(reg)
    diagnostics.to_csv(OUT/'historical_volume_errors.csv',index=False)
    model,te,shares=fit_shares(reg,2024)
    base=pd.read_csv(BASE,parse_dates=['date'])
    if not base.date.equals(te.date) or not np.isfinite(base.forecast).all() or (base.forecast<0).any():
        raise ValueError('Invalid or misaligned baseline forecast')
    calibration=json.loads((ROOT/'output/v61/calibration.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    if calibration['output_sha256']!=digest(BASE):
        raise ValueError('The expected verified V61 baseline changed')
    components=pd.DataFrame(base.forecast.to_numpy()[:,None]*shares,columns=GROUPS)
    adjusted,volumes=reconcile(components,prior,sigma)
    forecast=adjusted.sum(axis=1).round(2)
    if not np.isfinite(forecast).all() or (forecast<0).any() or not np.array_equal(forecast==0,base.forecast==0):
        raise ValueError('Calibration changed closed dates or produced invalid predictions')
    out=ROOT/'forecast_2024_v67_volume_prior.csv'
    pd.DataFrame({'date':base.date.dt.strftime('%Y-%m-%d'),'forecast':forecast}).to_csv(out,index=False)
    daily=adjusted.copy();daily.insert(0,'date',base.date.dt.strftime('%Y-%m-%d'))
    daily['baseline_total']=base.forecast;daily['forecast']=forecast
    daily.to_csv(OUT/'degree_breakdown.csv',index=False);volumes.to_csv(OUT/'volumes.csv',index=False)
    model.save_model(str(OUT/'degree_share_model.cbm'))
    meta={'model':'V61 with raw-data degree shares and explicit soft user volume priors',
          'source_of_volume_hints':'user_provided; not independently inferred',
          'uncertain_count_definition':prior['measurement_definition'],
          'working_assumption':prior['working_assumption'],'prior_config':prior,
          'base_file':BASE.name,'base_public_rmsle':calibration['measured_v61_public_rmsle'],
          'public_rmsle':None,'input_sha256':{str(p.resolve()):digest(p) for p in [BASE,calendar.REG,args.priors,ROOT/'output/v60/backtest_daily.csv',ROOT/'output/v61/calibration.json']},
          'code_sha256':digest(Path(__file__).resolve()),'calendar_code_sha256':digest(Path(calendar.__file__)),
          'share_model_sha256':digest(OUT/'degree_share_model.cbm'),
          'share_model_config':{'type':'CatBoostClassifier','depth':4,'iterations':300,'half_life':3.,'seed':67123},
          'base_log_volume_rmse':sigma,
          'output_sha256':digest(out),'total_forecast':float(forecast.sum()),'closed_days':int((forecast==0).sum()),
          'caveat':'Historical V61 components use frozen weights calibrated on 2024 feedback; volume-error estimates are retrospective, not independent validation of the new priors.'}
    (OUT/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(volumes.to_string(index=False),flush=True)
    print(f'Saved {out.name}; sum={forecast.sum():.2f}; public score unknown',flush=True)


if __name__=='__main__':main()
