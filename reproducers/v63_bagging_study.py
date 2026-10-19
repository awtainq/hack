"""Bagged calendar models as a distinct family for the shape/volume approach."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import v59_standalone_4_4 as base
from reproducers import v60_research as data
from reproducers.v63_shape_study import phases, align_level

OUT=ROOT/'output'/'v63_bagging'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for year in data.YEARS:
        source=pd.read_csv(data.CACHE/f'predictions_{year}.csv')
        legacy=np.log1p(np.expm1(source['z_Бюджет'])+np.expm1(source['z_Договор']))
        anchor=np.maximum(0,-1.2*source.z.to_numpy()+2.2*legacy.to_numpy())
        path=OUT/f'predictions_{year}.csv'
        if path.exists():
            result=pd.read_csv(path)
        else:
            g=data.build_grid(year);tr=g[g.year<year].copy();te=g[g.year==year].reset_index(drop=True)
            tr['phase']=phases(tr,'month')
            levels=tr[(tr.dow<5)&(tr.y>0)].groupby(['year','phase']).z.mean()
            centre=levels.reindex(pd.MultiIndex.from_frame(tr[['year','phase']])).to_numpy()
            tr['normalised']=tr.z-centre
            op=tr[tr.y>0];w=base.compute_era_weights(op)
            result=source[['date','year','y','dow','closed']].copy()
            for family,make,leaves in [('et',ExtraTreesRegressor,[1,3,8]),('rf',RandomForestRegressor,[3,8])]:
                for leaf in leaves:
                    for target in ['z','normalised']:
                        name=f'{family}{leaf}_{target}'
                        with threadpool_limits(limits=1):
                            model=make(n_estimators=300,min_samples_leaf=leaf,max_features=.8,n_jobs=1,random_state=42)
                            raw=model.fit(op[base.FEATURE_COLUMNS],op[target],sample_weight=w).predict(te[base.FEATURE_COLUMNS])
                        if target=='normalised':raw=align_level(raw,anchor,te,'month')
                        result[name]=np.where(source.closed,0,np.maximum(0,raw))
            result.to_csv(path,index=False);print(year,flush=True)
        if year==2024:continue
        truth=np.log1p(source.y.to_numpy())
        for name in [c for c in result if c.startswith(('et','rf'))]:
            for weight in [.25,.5,.75,1.]:
                pred=np.where(source.closed,0,(1-weight)*anchor+weight*result[name].to_numpy())
                rows.append({'year':year,'model':name,'weight':weight,'rmsle':float(np.sqrt(np.mean((pred-truth)**2)))})
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'metrics.csv',index=False)
    table=metrics.pivot(index=['model','weight'],columns='year',values='rmsle')
    table['selection_mean_2017_2022']=table.drop(columns=2023).mean(axis=1)
    print(table.sort_values('selection_mean_2017_2022').head(20).round(4).to_string())


if __name__=='__main__':main()
