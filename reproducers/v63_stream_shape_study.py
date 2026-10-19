"""Apply the shape/volume separation independently to both funding streams."""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from reproducers import v60_research as data
from reproducers.v63_shape_study import normalized_predictions,align_level
OUT=ROOT/'output'/'v63_stream_shape'


def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for year in data.YEARS:
        source=pd.read_csv(data.CACHE/f'predictions_{year}.csv')
        rawpool=pd.read_csv(ROOT/'output'/'v63_shape'/f'predictions_{year}.csv').month_1_raw.to_numpy()
        oldstream=np.log1p(np.expm1(source['z_Бюджет'])+np.expm1(source['z_Договор']))
        anchor=np.maximum(0,-1.2*source.z.to_numpy()+2.2*oldstream.to_numpy())
        path=OUT/f'predictions_{year}.csv'
        if path.exists():result=pd.read_csv(path)
        else:
            g=data.build_grid(year);tr,te=g[g.year<year],g[g.year==year].reset_index(drop=True)
            result=source[['date','year','y','dow','closed']].copy()
            newpool=align_level(rawpool,source.z.to_numpy(),te,'month')
            stream_parts=[]
            for col in ['z_Бюджет','z_Договор']:
                with threadpool_limits(limits=1):
                    raw=normalized_predictions(tr,te,'month',True,target=col)
                normal=align_level(raw,source[col].to_numpy(),te,'month')
                result[col+'_raw']=raw;result[col+'_new']=normal
                stream_parts.append(np.expm1(normal))
            newstream=np.log1p(np.sum(stream_parts,axis=0))
            new=np.maximum(0,-1.2*newpool+2.2*newstream)
            result['stream_shape']=np.where(source.closed,0,new)
            result['stream_shape_aligned']=np.where(source.closed,0,align_level(new,anchor,te,'month'))
            result.to_csv(path,index=False);print(year,flush=True)
        if year==2024:continue
        truth=np.log1p(source.y.to_numpy())
        for name in ['stream_shape','stream_shape_aligned']:
            for weight in [0,.25,.5,.75,1.]:
                pred=np.where(source.closed,0,(1-weight)*anchor+weight*result[name].to_numpy())
                rows.append(dict(year=year,model=name,weight=weight,rmsle=float(np.sqrt(np.mean((pred-truth)**2)))))
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'metrics.csv',index=False)
    table=metrics.pivot(index=['model','weight'],columns='year',values='rmsle')
    table['selection_mean_2017_2022']=table.drop(columns=2023).mean(axis=1)
    print(table.sort_values('selection_mean_2017_2022').round(4).to_string())


if __name__=='__main__':main()
