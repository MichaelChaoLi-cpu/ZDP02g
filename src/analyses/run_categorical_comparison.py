"""Four-variable one-hot point sensitivity: corrected grid, seed42, no bootstrap.
Retains YAML settings and outer10/inner5 random folds in an isolated output.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import xgboost
from xgboost import XGBClassifier,XGBRegressor
from categorical_prediction import categorical_model, NOMINAL_COLUMNS
from nested_grid_bootstrap import nested_predictions,draw_grid_multiplicities
from audit_temperature_provenance import ROOT,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='data/exp/revision_experiments/categorical_seed42_20260928')
    parser.add_argument('--bootstrap-reps',type=int,default=0)
    parser.add_argument('--skip-point',action='store_true',help='Run only explicitly requested bootstrap replicates')
    parser.add_argument('--device',default='cpu',choices=['cpu','cuda:0'],help='Execution device; saved in immutable manifest')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--repeat-seeds',nargs='+',type=int,default=[42],
                        help='Fixed split seeds; each bootstrap statistic averages only these seeds')
    parser.add_argument('--variants',nargs='+',choices=['grid','calendar'],default=['grid'])
    args=parser.parse_args()
    if args.bootstrap_reps or args.skip_point or args.repeat_seeds != [42] or args.variants != ['grid']:
        parser.error('Approved scope: grid point estimate, seed42, no bootstrap')
    if args.bootstrap_reps<0 or args.threads<1:parser.error('Invalid replicate/thread count')
    if len(set(args.repeat_seeds))!=len(args.repeat_seeds) or any(s<0 or s>=2**32 for s in args.repeat_seeds):
        parser.error('Repeat seeds must be distinct integers in [0, 2**32)')
    if args.skip_point and args.bootstrap_reps==0:parser.error('--skip-point requires positive --bootstrap-reps')
    os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src/analyses/notebooks'))
    from SettingForFeatures import data_load_combine_dataset,return_always_input_variable_list,return_aim_mediate
    output=ROOT/args.output;output.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/f'data/processed/{n}.parquet' for n in ['GlobalFlourishingDataWithLonLit','df_out_2023','df_out_2024']]
    paths += [ROOT/'data/processed/revision_exposure_validation'/n for n in ['native_grid_location_year.parquet','fixed_sample_exposure_variants.parquet']]
    cfg=ROOT/'src/analyses/config'
    mediators=return_aim_mediate()
    paths += [cfg/f'{m}_params.yaml' for m in mediators]+[cfg/'params.yaml']
    data=data_load_combine_dataset('tas')
    variants=pd.read_parquet(paths[4]).set_index('analysis_row')
    assert variants.index.equals(data.index) and len(data)==273031
    cells=pd.read_parquet(paths[3]);cells=cells.loc[cells.YEAR.eq(2023),['LATITUDE','LONGITUDE','native_lat_idx','native_lon_idx']]
    metadata=data[['COUNTRY','LATITUDE','LONGITUDE']].merge(cells,on=['LATITUDE','LONGITUDE'],how='left',sort=False,validate='many_to_one')
    assert np.array_equal(metadata[['LATITUDE','LONGITUDE']].to_numpy(),data[['LATITUDE','LONGITUDE']].to_numpy())
    metadata.index=data.index
    original={m:yaml.safe_load((cfg/f'{m}_params.yaml').read_text()) for m in mediators}
    original['outcome']=yaml.safe_load((cfg/'params.yaml').read_text())
    effective={m:{**params,'device':args.device,'n_jobs':args.threads} for m,params in original.items()}
    code_names=['run_categorical_comparison.py','categorical_prediction.py','nested_grid_bootstrap.py','paired_perturbation.py','notebooks/SettingForFeatures.py']
    manifest={'rows':len(data),'repeat_seeds':args.repeat_seeds,'outer_folds':10,'inner_folds':5,
       'original_yaml_parameters':original,'effective_parameters':effective,
       'encoding':{'columns':NOMINAL_COLUMNS,'method':'training-fold one-hot, unknown ignored','employment_8':'retained uninterpreted'},
       'runtime_overrides':'Execution device and thread count only; no hyperparameter search or model-size reduction.',
       'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in paths},
       'code_sha256':{n:sha(ROOT/'src/analyses'/n) for n in code_names},
       'software':{'python':platform.python_version(),'xgboost':xgboost.__version__},
       'decisions':['KILA-D-20260926-003','KILA-D-20260926-004','KILA-D-20260926-005','KILA-D-20260928-020','KILA-D-20260928-021']}
    if args.repeat_seeds==[42]:manifest['decisions'].append('KILA-D-20260927-003')
    path=output/'manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=manifest:raise ValueError('Manifest mismatch; use a fresh output directory')
    else:path.write_text(json.dumps(manifest,indent=2)+'\n')
    def event(info):
        item={'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**info}
        with (output/'events.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
        print(json.dumps(item),flush=True)
    all_summary=[]
    for variant in args.variants:
        X=data[return_always_input_variable_list()].copy()
        for metric in ['MEAN','STD']:X[f'TAS_{metric}']=variants[f'{variant}_TAS_{metric}'].to_numpy()
        for replicate in range(0 if args.skip_point else -1,args.bootstrap_reps):
            label='point' if replicate<0 else f'bootstrap_{replicate:04d}'
            counts=pd.Series(1,index=data.index,dtype='int64') if replicate<0 else draw_grid_multiplicities(metadata,2026092600+replicate)[0]
            running=None
            for seed in args.repeat_seeds:
                destination=output/variant/label/f'seed_{seed}'
                destination.mkdir(parents=True,exist_ok=True)
                def progress(info):event({'variant':variant,'run':label,'seed':seed,**info})
                progress({'event':'repeat_start'})
                result,trace=nested_predictions(X,data[mediators],data.PHYSICAL_HLTH,counts,
                    lambda m:categorical_model(XGBClassifier(**effective[m])),lambda:categorical_model(XGBRegressor(**effective['outcome'])),
                    repeat_seed=seed,checkpoint_dir=destination/'folds',on_event=progress)
                zero=[c for c in result if '_zero_' in c and c.endswith('_change')]
                if np.max(np.abs(result[zero].to_numpy()))>1e-10:raise ValueError('Zero check failed')
                # Drop fold labels before averaging across repeats.
                prediction=result.drop(columns='fold')
                running=prediction/len(args.repeat_seeds) if running is None else running+prediction/len(args.repeat_seeds)
                result.to_parquet(destination/'predictions.parquet')
                (destination/'trace.json').write_text(json.dumps(trace,indent=2)+'\n')
                progress({'event':'repeat_complete'})
            folder=output/variant/label
            running.to_parquet(folder/'mean_predictions.parquet')
            for country,group in [('all',data.index)]+[(str(c),d.index) for c,d in data.groupby('COUNTRY')]:
                w=counts.loc[group]
                if w.sum()==0:raise ValueError('Empty country bootstrap draw')
                stats={'variant':variant,'run':label,'country':country,'weighted_rows':int(w.sum())}
                stats.update({c:float(np.average(running.loc[group,c],weights=w)) for c in running if c.endswith('_change')})
                all_summary.append(stats)
            pd.DataFrame(all_summary).to_csv(output/'completed_run_summaries.csv',index=False)
            event({'event':'run_complete','variant':variant,'run':label})
    assert all(sha(ROOT/n)==h for n,h in manifest['input_sha256'].items())
    (output/'completed.json').write_text(json.dumps({'variants':args.variants,'bootstrap_reps':args.bootstrap_reps,'point_estimates_included':not args.skip_point,'inputs_unchanged':True,'confidence_intervals_computed':False},indent=2)+'\n')
    event({'event':'batch_complete'})


if __name__=='__main__':main()
