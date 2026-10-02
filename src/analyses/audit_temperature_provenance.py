"""Read-only audit of legacy temperature provenance; writes aggregate audit outputs only.
Run from any directory: uv run python src/analyses/audit_temperature_provenance.py
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp585']
OUT = ROOT / 'data/exp/revision_audit/temperature_provenance'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def nearest(grid, points):
    return np.abs(points[:, None] - grid[None, :]).argmin(axis=1)

def summary(values):
    x = np.asarray(values, dtype=float)
    return {'n': int(x.size), 'mean': float(np.mean(x)), 'median': float(np.median(x)),
            'min': float(np.min(x)), 'max': float(np.max(x)), 'mean_abs': float(np.mean(np.abs(x)))}

def main():
    consumed = {}
    metadata, fields, grids = [], {}, {}
    inventory = {}
    for path in sorted((ROOT/'data/raw/cmip6').glob('*.nc')):
        parts = path.stem.split('_')
        key = parts[3]
        inventory.setdefault(key, []).append(int(parts[-1][:4]))
    inventory = {key: {'files': len(years), 'first_year': min(years), 'last_year': max(years)} for key,years in inventory.items()}
    for year in [2023,2024]:
        arrays = []
        for scenario in SCENARIOS:
            path = ROOT / f'data/raw/cmip6/tas_Amon_EC-Earth3-Veg_{scenario}_r1i1p1f1_gr_{year}01-{year}12.nc'
            consumed[str(path.relative_to(ROOT))] = sha(path)
            with Dataset(path) as ds:
                lat, lon = np.asarray(ds['lat'][:]), np.asarray(ds['lon'][:])
                values = ds['tas'][:]
                assert not np.ma.getmaskarray(values).any(), 'Masked source values require explicit handling'
                arr = np.asarray(values)
                t = ds['time']; dates = num2date(t[:], t.units, getattr(t,'calendar','standard'))
                assert arr.shape == (12,len(lat),len(lon))
                assert [x.month for x in dates] == list(range(1,13)) and all(x.year==year for x in dates)
                assert np.all(np.diff(lat)>0), 'Legacy unconditional latitude flip assumes ascending latitude'
                canonical_lon = ((lon+180)%360)-180 if lon.max()>180 else lon.copy()
                order = np.argsort(canonical_lon)
                actual_lat, actual_lon = lat[::-1], canonical_lon[order]
                if year in grids:
                    assert np.array_equal(grids[year][0],actual_lat) and np.array_equal(grids[year][1],actual_lon)
                grids[year] = (actual_lat,actual_lon)
                arrays.append(arr[:,::-1,:][:,:,order])
                attrs = {name:str(getattr(ds,name,'')) for name in ['source_id','activity_id','experiment_id','variant_label','grid_label','nominal_resolution','frequency','product','creation_date','history']}
                attrs.update({'path':str(path.relative_to(ROOT)),'shape':list(arr.shape),'units':str(ds['tas'].units),
                              'months':[str(x) for x in dates], 'latitude_ascending':True,'latitude_range':[float(lat.min()),float(lat.max())],
                              'longitude_range':[float(lon.min()),float(lon.max())]})
                metadata.append(attrs)
        # Reproduce the legacy calculation exactly: float32 K->C, then scenario and month means.
        monthly = (np.stack(arrays)-273.15).mean(axis=0)
        fields[year]=(monthly.mean(axis=0),monthly.std(axis=0))
    reports = {}
    saved = {}
    for year in [2023,2024]:
        path=ROOT/f'data/processed/df_out_{year}.parquet';consumed[str(path.relative_to(ROOT))]=sha(path)
        df=pd.read_parquet(path);saved[year]=df
        actual_lat,actual_lon=grids[year]
        synthetic_lat=np.linspace(90,-90,len(actual_lat));synthetic_lon=np.linspace(-180,180,len(actual_lon))
        la=df['LATITUDE'].to_numpy();lo=df['LONGITUDE'].to_numpy()
        old_i=nearest(synthetic_lat,la);old_j=nearest(synthetic_lon,lo)
        new_i=nearest(actual_lat,la)
        # Native longitude distances are circular at the dateline.
        delta=np.abs(lo[:,None]-actual_lon[None,:]);new_j=np.minimum(delta,360-delta).argmin(axis=1)
        mean,sd=fields[year]
        legacy_mean,legacy_sd=mean[old_i,old_j],sd[old_i,old_j]
        native_mean,native_sd=mean[new_i,new_j],sd[new_i,new_j]
        stats={'location_rows':len(df),'unique_locations':len(df[['LATITUDE','LONGITUDE']].drop_duplicates()),
               'integer_latitude_fraction':float(np.mean(np.isclose(la,np.round(la),rtol=0,atol=1e-9))),
               'integer_longitude_fraction':float(np.mean(np.isclose(lo,np.round(lo),rtol=0,atol=1e-9))),
               'latitude_coordinate_max_error_deg':float(np.max(np.abs(synthetic_lat-actual_lat))),
               'longitude_coordinate_max_error_deg':float(np.max(np.abs(synthetic_lon-actual_lon))),
               'native_longitude_step_deg':float(np.median(np.diff(actual_lon))),
               'latitude_cell_changes':int(np.sum(old_i!=new_i)), 'longitude_cell_changes':int(np.sum(old_j!=new_j)),
               'any_cell_changes':int(np.sum((old_i!=new_i)|(old_j!=new_j))),
               'legacy_mean_minus_saved':summary(legacy_mean-df[f'tas_mean_{year}'].to_numpy()),
               'legacy_sd_minus_saved':summary(legacy_sd-df[f'tas_std_{year}'].to_numpy()),
               'native_minus_legacy_mean_celsius':summary(native_mean-legacy_mean),
               'native_minus_legacy_monthly_sd_celsius':summary(native_sd-legacy_sd)}
        for col,idx in [('lat_idx',old_i),('lon_idx',old_j)]:
            if col in df:stats[f'{col}_matches_saved']=bool(np.array_equal(df[col].to_numpy(),idx))
        reports[str(year)]=stats
    waves=saved[2023].merge(saved[2024],on=['LATITUDE','LONGITUDE'],suffixes=('_2023idx','_2024idx'),validate='one_to_one')
    wave_comparison={}
    for metric in ['mean','std']:
        a=waves[f'tas_{metric}_2023'].to_numpy();b=waves[f'tas_{metric}_2024'].to_numpy()
        wave_comparison[metric]={'correlation':float(np.corrcoef(a,b)[0,1]),'difference_2024_minus_2023':summary(b-a),'exact_equal_count':int(np.sum(a==b))}
    # Aggregate-only survey precision audit; do not export personal data or coordinates.
    survey=ROOT/'data/processed/GlobalFlourishingDataWithLonLit.parquet';consumed[str(survey.relative_to(ROOT))]=sha(survey)
    coords=pd.read_parquet(survey,columns=['LATITUDE','LONGITUDE','WAVE'])
    survey_summary={'rows':len(coords),'unique_coordinate_pairs':len(coords[['LATITUDE','LONGITUDE']].drop_duplicates()),
                    'waves':sorted(coords['WAVE'].dropna().unique().tolist())}
    for col in ['LATITUDE','LONGITUDE']:
        x=coords[col].dropna().to_numpy();survey_summary[col.lower()+'_integer_fraction']=float(np.mean(np.isclose(x,np.round(x),rtol=0,atol=1e-9)))
    assert all(sha(ROOT/path)==digest for path,digest in consumed.items()),'An input changed during the audit'
    OUT.mkdir(parents=True,exist_ok=True)
    result={'inventory_by_filename':inventory,'metadata':metadata,'years':reports,'wave_comparison':wave_comparison,
            'survey_coordinate_summary':survey_summary,'input_sha256':consumed,'inputs_unchanged':True,
            'scope':'Unweighted unique-location diagnostics only. Native-coordinate comparison is diagnostic, not approved replacement exposure. No outcomes or models fitted.'}
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['metadata','input_sha256']},indent=2))

if __name__=='__main__':main()
