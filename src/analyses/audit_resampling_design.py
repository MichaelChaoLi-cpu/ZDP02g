"""Aggregate cluster feasibility for the fixed pooled sample; no model fitting."""
import json
import pandas as pd
from audit_temperature_provenance import ROOT, sha


def main():
    names = ['data/processed/GlobalFlourishingDataWithLonLit.parquet',
             'data/processed/df_out_2023.parquet','data/processed/df_out_2024.parquet',
             'data/processed/revision_exposure_validation/native_grid_location_year.parquet']
    hashes = {n:sha(ROOT/n) for n in names}
    survey=pd.read_parquet(ROOT/names[0]);exposures=[]
    for wave,year in [(1,2023),(2,2024)]:
        d=pd.read_parquet(ROOT/f'data/processed/df_out_{year}.parquet')
        d['WAVE']=float(wave)
        d=d[['LATITUDE','LONGITUDE','WAVE',f'tas_mean_{year}',f'tas_std_{year}']]
        d.columns=['LATITUDE','LONGITUDE','WAVE','TAS_MEAN','TAS_STD'];exposures.append(d)
    sample=survey.merge(pd.concat(exposures),on=['LATITUDE','LONGITUDE','WAVE'],validate='many_to_one')
    sample=sample.loc[sample.drop(columns='ID').notna().all(axis=1)].copy()
    cells=pd.read_parquet(ROOT/names[-1]);cells=cells.loc[cells.YEAR.eq(2023),['LATITUDE','LONGITUDE','native_lat_idx','native_lon_idx']]
    sample=sample.merge(cells,on=['LATITUDE','LONGITUDE'],how='left',validate='many_to_one')
    assert len(sample)==273031 and sample[['ID','native_lat_idx','native_lon_idx']].notna().all().all()
    sample['cell']=sample.groupby(['native_lat_idx','native_lon_idx'],sort=True).ngroup()
    sample['country_cell']=sample.groupby(['COUNTRY','cell'],sort=True).ngroup()
    sizes=sample.groupby('country_cell').size()
    person_cells=sample.groupby('ID').country_cell.nunique()
    person_countries=sample.groupby('ID').COUNTRY.nunique()
    per_country=sample.groupby('COUNTRY').agg(rows=('ID','size'),people=('ID','nunique'),cells=('cell','nunique'))
    result={'rows':len(sample),'people':sample.ID.nunique(),'countries':len(per_country),
        'native_cells':sample.cell.nunique(),'country_cell_clusters':len(sizes),
        'cluster_size_quantiles':sizes.quantile([0,.25,.5,.75,1]).to_dict(),
        'singleton_clusters':int(sizes.eq(1).sum()),
        'people_in_multiple_country_cells':int(person_cells.gt(1).sum()),
        'people_in_multiple_countries':int(person_countries.gt(1).sum()),
        'per_country':per_country.reset_index().to_dict(orient='records'),
        'inputs_unchanged':True,'input_sha256':hashes,
        'scope':'Feasibility only; no bootstrap design selected, no row-level identifiers exported.'}
    assert all(sha(ROOT/n)==v for n,v in hashes.items())
    out=ROOT/'data/exp/revision_audit/mediation_consistency/resampling_design.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['input_sha256','per_country']},indent=2))


if __name__=='__main__':main()
