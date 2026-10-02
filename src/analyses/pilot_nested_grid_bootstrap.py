"""Run a bounded end-to-end implementation pilot; never report inferential CIs."""
import os
import sys
import json
import time
import platform
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier, XGBRegressor
from nested_grid_bootstrap import draw_grid_multiplicities,nested_predictions
from audit_temperature_provenance import ROOT,sha


def main():
    os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src/analyses/notebooks'))
    from SettingForFeatures import data_load_combine_dataset,return_always_input_variable_list,return_aim_mediate
    paths=[ROOT/f'data/processed/{n}.parquet' for n in ['GlobalFlourishingDataWithLonLit','df_out_2023','df_out_2024']]
    paths += [ROOT/'data/processed/revision_exposure_validation'/n for n in ['native_grid_location_year.parquet','fixed_sample_exposure_variants.parquet']]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    data=data_load_combine_dataset('tas')
    variants=pd.read_parquet(paths[-1]).set_index('analysis_row')
    assert variants.index.equals(data.index)
    for metric in ['MEAN','STD']:data[f'TAS_{metric}']=variants[f'grid_TAS_{metric}'].to_numpy()
    # Diagnostic subsample only; keep every country represented, then fill randomly.
    rng=np.random.default_rng(20260926)
    selected=[]
    for _,group in data.groupby('COUNTRY',sort=True):
        selected.extend(rng.choice(group.index,size=min(24,len(group)),replace=False).tolist())
    remainder=data.index.difference(selected)
    selected.extend(rng.choice(remainder,size=1024-len(selected),replace=False).tolist())
    subset=data.loc[sorted(selected)].copy()
    locations=pd.read_parquet(paths[-2]);locations=locations.loc[locations.YEAR.eq(2023),['LATITUDE','LONGITUDE','native_lat_idx','native_lon_idx']]
    metadata=subset[['COUNTRY','LATITUDE','LONGITUDE']].merge(locations,on=['LATITUDE','LONGITUDE'],how='left',sort=False,validate='many_to_one')
    assert np.array_equal(metadata[['LATITUDE','LONGITUDE']].to_numpy(),subset[['LATITUDE','LONGITUDE']].to_numpy())
    metadata.index=subset.index
    X=subset[return_always_input_variable_list()];M=subset[return_aim_mediate()];y=subset.PHYSICAL_HLTH
    parameters=dict(n_estimators=4,max_depth=2,learning_rate=0.1,tree_method='hist',device='cpu',n_jobs=1,random_state=42,subsample=1.)
    output=ROOT/'data/exp/revision_audit/nested_grid_bootstrap_pilot';output.mkdir(parents=True,exist_ok=True)
    records=[];fold_hash=None
    for replicate in [-1,0,1]:
        if replicate==-1:
            counts=pd.Series(1,index=X.index,dtype='int64');draws=[]
        else:
            counts,draws=draw_grid_multiplicities(metadata,2026092600+replicate)
            repeated,_=draw_grid_multiplicities(metadata,2026092600+replicate)
            pd.testing.assert_series_equal(counts,repeated)
            check=metadata.copy();check['multiplicity']=counts
            assert check.groupby(['COUNTRY','native_lat_idx','native_lon_idx']).multiplicity.nunique().eq(1).all()
        before=time.perf_counter()
        result,trace=nested_predictions(X,M,y,counts,
            lambda m:XGBClassifier(**parameters),lambda:XGBRegressor(**parameters),repeat_seed=42)
        duration=time.perf_counter()-before
        if fold_hash is None:fold_hash=trace['fold_assignment_sha256']
        assert fold_hash==trace['fold_assignment_sha256']
        zero_cols=[c for c in result if '_zero_' in c and c.endswith('_change')]
        assert np.abs(result[zero_cols].to_numpy()).max()==0
        means={c:float(np.average(result[c],weights=counts)) for c in result if c.endswith('_change')}
        # Keep predictions and multiplicities separate from legacy and publication results.
        result['bootstrap_multiplicity']=counts
        name='baseline' if replicate==-1 else f'bootstrap_{replicate:03d}'
        result.to_parquet(output/f'{name}.parquet')
        pd.testing.assert_frame_equal(result,pd.read_parquet(output/f'{name}.parquet'))
        record={'run':name,'seconds':duration,'expanded_rows':int(counts.sum()),'unique_selected_rows':int(counts.gt(0).sum()),
                'zero_max_abs':0.,'trace':trace,'strata_draws':draws,'diagnostic_means_not_paper_estimates':means}
        records.append(record)
        print(json.dumps({'run':name,'seconds':duration,'fits':trace['fits'],'zero_max_abs':0.}),flush=True)
    assert all(sha(ROOT/n)==v for n,v in hashes.items())
    report={'scope':'Implementation pilot only. 1024 real records, corrected native-grid/wave-year exposure, 4-tree models, one repeat seed, baseline plus TWO bootstrap replicates. No final interval or effect inference.',
        'decision':'KILA-D-20260926-004','rows':len(X),'countries':int(metadata.COUNTRY.nunique()),
        'outer_folds':10,'inner_folds':5,'resampling':'native-grid clusters within country',
        'bootstrap_replications':2,'model_fits':sum(r['trace']['fits'] for r in records),'parameters':parameters,
        'fixed_outer_folds_across_draws':True,'runs':records,'inputs_unchanged':True,'input_sha256':hashes,
        'software':{'python':platform.python_version(),'xgboost':xgboost.__version__},
        'code_sha256':{n:sha(ROOT/'src/analyses'/n) for n in ['paired_perturbation.py','nested_grid_bootstrap.py','pilot_nested_grid_bootstrap.py']},
        'limitations':['Cross-grid repeat-person dependence is not preserved as a single cluster.',
                      'Two bootstrap replicates cannot estimate reliable confidence intervals.',
                      'Small-sample shallow-tree timing cannot be extrapolated directly to full 1200-tree models.',
                      'Existing tuned hyperparameters are not retuned within bootstrap; full-scale tuning uncertainty remains outside this implementation.']}
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['runs','input_sha256','code_sha256']},indent=2))


if __name__=='__main__':main()
