"""Exposure-only 2x2 coarsening diagnostic; not an outcome robustness test.

Same source/scenarios/wave years/sample. Area-proxy cosine-latitude weights;
aggregate monthly fields before annual mean/SD. Fixed nonoverlapping block
origin at the first native latitude/longitude index. No subdegree geolocation
or rounding mechanism is inferred. No raw or primary outputs are modified.
"""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from audit_temperature_provenance import ROOT,SCENARIOS,sha,nearest


def main():
    sys.path.insert(0,str(ROOT/'src/analyses/notebooks'))
    from SettingForFeatures import data_load_combine_dataset
    data=data_load_combine_dataset('tas');assert len(data)==273031
    vp=ROOT/'data/processed/revision_exposure_validation/fixed_sample_exposure_variants.parquet'
    v=pd.read_parquet(vp).set_index('analysis_row');assert v.index.equals(data.index)
    assert np.equal(data[['LATITUDE','LONGITUDE']],np.round(data[['LATITUDE','LONGITUDE']])).all().all()
    inputs={str(vp.relative_to(ROOT)):sha(vp)};fields=[];meta=[];native_errors=[]
    for name in ['GlobalFlourishingDataWithLonLit.parquet','df_out_2023.parquet','df_out_2024.parquet']:
        p=ROOT/'data/processed'/name;inputs[str(p.relative_to(ROOT))]=sha(p)
    for year in [2023,2024]:
        arrays=[]
        for scenario in SCENARIOS:
            p=ROOT/f'data/raw/cmip6/tas_Amon_EC-Earth3-Veg_{scenario}_r1i1p1f1_gr_{year}01-{year}12.nc';inputs[str(p.relative_to(ROOT))]=sha(p)
            with Dataset(p) as d:
                lat=np.asarray(d['lat'][:]);lon=np.asarray(d['lon'][:]);x=d['tas'][:]
                assert x.shape==(12,256,512) and not np.ma.getmaskarray(x).any()
                arrays.append(np.asarray(x,dtype=np.float64)-273.15)
                meta.append(dict(file=p.name,nominal_resolution=d.nominal_resolution,lat_step_min=float(np.diff(lat).min()),lat_step_max=float(np.diff(lat).max()),lon_step=float(np.diff(lon)[0])))
        # Match production coordinate ordering, including deterministic ties at the equator.
        native_lon=(lon+180)%360-180;order=np.argsort(native_lon);lon=native_lon[order];lat=lat[::-1]
        monthly=np.mean(arrays,axis=0)[:,::-1,:][:,:,order];weights=np.cos(np.deg2rad(lat)).reshape(128,2)
        coarse=(monthly.reshape(12,128,2,256,2)*weights[None,:,:,None,None]).sum(axis=(2,4))/(2*weights.sum(axis=1)[None,:,None])
        clat=(lat.reshape(128,2)*weights).sum(axis=1)/weights.sum(axis=1);clon=lon.reshape(256,2).mean(axis=1)
        sample=data.loc[v.WAVE_YEAR.eq(year)];a=nearest(clat,sample.LATITUDE.to_numpy());dist=np.abs((sample.LONGITUDE.to_numpy()[:,None]-clon[None,:]+180)%360-180);b=dist.argmin(axis=1)
        ni=nearest(lat,sample.LATITUDE.to_numpy());nd=np.abs((sample.LONGITUDE.to_numpy()[:,None]-lon[None,:]+180)%360-180);nj=nd.argmin(axis=1)
        for metric,grid in [('MEAN',monthly.mean(axis=0)),('STD',monthly.std(axis=0))]:
            err=float(np.max(np.abs(grid[ni,nj]-v.loc[sample.index,f'grid_TAS_{metric}'])));assert err<1e-4,(year,metric,err);native_errors.append(dict(year=year,metric=metric,max_absolute_error=err))
        f=pd.DataFrame(index=sample.index);f['coarse_TAS_MEAN']=coarse.mean(axis=0)[a,b];f['coarse_TAS_STD']=coarse.std(axis=0)[a,b];fields.append(f)
    result=pd.concat(fields).loc[data.index];assert result.index.equals(v.index) and np.isfinite(result.to_numpy()).all()
    records=[]
    for metric in ['MEAN','STD']:
        x=v[f'grid_TAS_{metric}'];y=result[f'coarse_TAS_{metric}'];diff=y-x
        for country,idx in [('all',data.index)]+[(str(k),g.index) for k,g in data.groupby('COUNTRY')]:
            d=diff.loc[idx];records.append(dict(metric=metric,country=country,n=len(idx),mean_change=float(d.mean()),mae=float(d.abs().mean()),p95_absolute_change=float(d.abs().quantile(.95)),max_absolute_change=float(d.abs().max()),pearson_correlation=float(x.loc[idx].corr(y.loc[idx]))))
    assert all(sha(ROOT/p)==h for p,h in inputs.items())
    out=ROOT/'data/exp/revision_audit/r1c4_spatial_coarsening';out.mkdir(parents=True,exist_ok=True)
    result.to_parquet(out/'candidate_exposures.parquet');pd.DataFrame(records).to_csv(out/'exposure_comparison.csv',index=False)
    report=dict(rows=len(data),countries=int(data.COUNTRY.nunique()),integer_degree_coordinates=True,source_metadata=meta,native_reproduction_checks=native_errors,input_sha256=inputs,inputs_unchanged=True,method='Fixed nonoverlapping2x2 native cells; cosine-latitude weighted monthly averages; then12-month mean/populationSD; nearest coarse centroid with circular longitude distance; five scenarios equally averaged; wave-year assignment retained.',scope='Exposure diagnostic only. Does not test health-profile robustness, finer reanalysis quality, true location error or classical measurement-error assumptions. Coarsening grid origin is a specified diagnostic choice, not a claimed GFS coordinate cell boundary.',script_sha256=sha(Path(__file__)),pooled=[r for r in records if r['country']=='all'])
    (out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report['pooled'],indent=2))

if __name__=='__main__':main()
