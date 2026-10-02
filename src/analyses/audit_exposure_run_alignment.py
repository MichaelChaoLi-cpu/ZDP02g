"""Compare existing CPU exposure-sensitivity and GPU primary runs without fitting."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 old=ROOT/'data/exp/revision_experiments/fixed_yaml_20260926'
 new=ROOT/'data/exp/revision_uncertainty_audit'
 out=ROOT/'data/exp/revision_audit/exposure_run_alignment';out.mkdir(parents=True,exist_ok=True)
 paths={'cpu_manifest':old/'manifest.json','gpu_manifest':new/'manifest.json','cpu_grid':old/'grid/point/seed_42/predictions.parquet','cpu_calendar':old/'calendar/point/seed_42/predictions.parquet','gpu_grid':new/'point_predictions.parquet','gpu_summary':new/'completed_run_summaries.csv','cpu_trace':old/'grid/point/seed_42/trace.json','calendar_trace':old/'calendar/point/seed_42/trace.json'}
 hashes={k:sha(p) for k,p in paths.items()};a=json.loads(paths['cpu_manifest'].read_text());b=json.loads(paths['gpu_manifest'].read_text())
 assert a['input_sha256']==b['input_sha256']
 assert all(sha(ROOT/p)==h for p,h in a['input_sha256'].items())
 paramdiff={m:{k:[a['effective_parameters'][m].get(k),b['effective_parameters'][m].get(k)] for k in set(a['effective_parameters'][m])|set(b['effective_parameters'][m]) if a['effective_parameters'][m].get(k)!=b['effective_parameters'][m].get(k)} for m in a['effective_parameters']}
 assert all(set(v)=={'device'} for v in paramdiff.values())
 core=['nested_grid_bootstrap.py','paired_perturbation.py','notebooks/SettingForFeatures.py']
 assert all(a['code_sha256'][p]==b['code_sha256'][p]==sha(ROOT/'src/analyses'/p) for p in core)
 frames={k:pd.read_parquet(paths[k]) for k in ['cpu_grid','cpu_calendar','gpu_grid']}
 grid=frames['cpu_grid'];assert len(grid)==273031 and grid.index.is_unique
 cols=[f'{m}_positive_{c}_change' for m in ['mean','std'] for c in ['direct','total']]
 expected=np.zeros(len(grid),dtype=int)
 for fold,(_,test) in enumerate(KFold(10,shuffle=True,random_state=42).split(grid)):expected[test]=fold
 for key,df in frames.items():
  assert df.index.equals(grid.index),key
  assert np.isfinite(df[cols].to_numpy()).all(),key
  if 'fold' in df:assert np.array_equal(df['fold'].to_numpy(),expected),key
  else:assert key=='gpu_grid'
  zero=[c for c in df if '_zero_' in c and c.endswith('_change')];assert zero and (df[zero].to_numpy()==0).all(),key
 # Index-aligned country metadata from the unchanged production loader.
 import sys,os
 os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src/analyses/notebooks'))
 from SettingForFeatures import data_load_combine_dataset
 data=data_load_combine_dataset('tas');assert data.index.equals(grid.index)
 summary=pd.read_csv(paths['gpu_summary']);point=summary[summary.run.eq('point')];assert len(point)==23
 records=[];country=[]
 for c in cols:
  vals={k:df[c].to_numpy(dtype=np.float64) for k,df in frames.items()}
  row={'metric':c,**{k:float(v.mean()) for k,v in vals.items()},'cpu_calendar_minus_cpu_grid':float((vals['cpu_calendar']-vals['cpu_grid']).mean()),'gpu_grid_minus_cpu_grid':float((vals['gpu_grid']-vals['cpu_grid']).mean()),'gpu_vs_cpu_grid_row_mae':float(np.abs(vals['gpu_grid']-vals['cpu_grid']).mean()),'gpu_vs_cpu_grid_row_correlation':float(np.corrcoef(vals['gpu_grid'],vals['cpu_grid'])[0,1])}
  pooled=float(point.loc[point.country.astype(str).eq('all'),c].iloc[0]);assert abs(row['gpu_grid']-pooled)<1e-12
  row['gpu_aggregate_reproduction_abs_error']=abs(row['gpu_grid']-pooled);records.append(row)
  for code,group in data.groupby('COUNTRY'):
   means={k:float(df.loc[group.index,c].to_numpy(dtype=np.float64).mean()) for k,df in frames.items()}
   source=point[pd.to_numeric(point.country,errors='coerce').eq(code)];assert len(source)==1 and abs(float(source[c].iloc[0])-means['gpu_grid'])<1e-12
   country.append({'country_id':code,'records':len(group),'metric':c,**means,'cpu_calendar_minus_cpu_grid':means['cpu_calendar']-means['cpu_grid'],'cpu_calendar_sign_change':bool(np.sign(means['cpu_calendar'])!=np.sign(means['cpu_grid']))})
 pd.DataFrame(records).to_csv(out/'pooled_comparison.csv',index=False);pd.DataFrame(country).to_csv(out/'country_comparison.csv',index=False)
 report={'rows':len(grid),'countries':int(data.COUNTRY.nunique()),'inputs_identical_and_current_hashes_verified':True,'row_index_order_identical':True,'cpu_grid_calendar_outer_fold_labels_identical_and_match_seed42':True,'gpu_fold_labels':'not included in imported mean-prediction table; same manifest seed/folds and identical split code support expected equality, not direct label verification','inner_folds':'same hashed core code, sample order and seed; deterministic reconstruction, no GPU inner-fold trace locally imported','core_code_identical':core,'parameter_differences':paramdiff,'software':{'cpu':a['software'],'gpu':b['software']},'runner_hashes':{'cpu':a['code_sha256']['run_revision_experiments.py'],'gpu':b['code_sha256']['run_revision_experiments.py']},'historical_cpu_runner_snapshot_available':False,'point_weights':'all ones in current runner; GPU pooled and all country summaries reproduced from row-level predictions','zero_checks_exact':True,'source_hashes':hashes,'sources_unchanged':all(sha(p)==hashes[k] for k,p in paths.items()),'no_model_fits':True,'pooled':records,'inference_limit':'CPU versus GPU is a verified execution difference, not an experimentally isolated sole cause; historical CPU runner snapshot and full dependency inventory unavailable.','reuse_conclusion':'Existing CPU calendar versus CPU grid seed42 outputs support a separately labelled paired descriptive sensitivity analysis. Do not subtract CPU calendar from GPU primary as a pure year-assignment contrast; do not attach GPU bootstrap intervals to CPU results.'}
 assert report['sources_unchanged'];(out/'validation.json').write_text(json.dumps(report,indent=2));print(pd.DataFrame(records)[['metric','cpu_grid','cpu_calendar','gpu_grid','cpu_calendar_minus_cpu_grid']].to_string(index=False));print('Validation passed; no model fits.')
if __name__=='__main__':main()
