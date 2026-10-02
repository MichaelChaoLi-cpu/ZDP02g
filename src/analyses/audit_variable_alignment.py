"""Read-only raw-to-analysis variable audit; write aggregate diagnostics only."""
import ast,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/analyses/notebooks'))
from SettingForFeatures import return_always_input_variable_list,return_aim_mediate,data_load_combine_dataset

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 os.chdir(ROOT)
 paths=[ROOT/'data/raw/gfs_all-countries_wave2-with-midyear_sensitive.csv',ROOT/'data/processed/GlobalFlourishingDataWithLonLit.parquet',ROOT/'src/analyses/nbs/c01_gfs_data_preprocess.py']
 hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
 mapping=None
 for node in ast.parse(paths[2].read_text()).body:
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='income_mapping' for t in node.targets):mapping=ast.literal_eval(node.value)
 assert mapping is not None
 variables=[v for v in return_always_input_variable_list()+return_aim_mediate()+['PHYSICAL_HLTH'] if not v.startswith('TAS_')]
 header=pd.read_csv(paths[0],nrows=0).columns
 sources={}
 for v in variables:
  base={'INCOME_REAL':'INCOME','HAVE_CHILD':'NUM_CHILDREN'}.get(v,v)
  sources[v]=[base] if base in header else [base+'_Y1',base+'_Y2']
  assert all(c in header for c in sources[v])
 use=sorted(set(['ID']+[c for cols in sources.values() for c in cols]))
 raw=pd.read_csv(paths[0],usecols=use,na_values=['',' ','NA','N/A'],low_memory=False).apply(pd.to_numeric,errors='raise')
 assert raw.ID.is_unique
 raw=raw.set_index('ID');survey=pd.read_parquet(paths[1])
 fields=[]
 for wave,year in [(1,2023),(2,2024)]:
  f=pd.read_parquet(ROOT/f'data/processed/df_out_{year}.parquet');f['WAVE']=float(wave)
  fields.append(f[['LATITUDE','LONGITUDE','WAVE',f'tas_mean_{year}',f'tas_std_{year}']].rename(columns={f'tas_mean_{year}':'TAS_MEAN',f'tas_std_{year}':'TAS_STD'}))
 merged=survey.merge(pd.concat(fields),on=['LATITUDE','LONGITUDE','WAVE'],validate='many_to_one')
 sample=merged.loc[merged.drop(columns='ID').notna().all(axis=1)]
 pd.testing.assert_frame_equal(sample.drop(columns='ID'),data_load_combine_dataset('tas'))
 assert not sample.duplicated(['ID','WAVE']).any()
 records=[]
 for v in variables:
  for wave in [1,2]:
   part=sample.loc[sample.WAVE.eq(wave)];col=sources[v][0 if len(sources[v])==1 else wave-1]
   original=raw.loc[part.ID,col].to_numpy();expected=original.copy()
   if v=='INCOME_REAL':expected=pd.Series(original).map(mapping).to_numpy()
   elif v in ['HAVE_CHILD','DAYS_EXERCISE']:expected=(original>0).astype(float)
   elif v=='BODILY_PAIN':expected=np.where(original>2,0,1)
   elif v in ['HEALTH_PROB','CLOSE_TO']:expected=np.where(original==2,0,original)
   equal=np.isclose(expected,part[v].to_numpy(),rtol=0,atol=0,equal_nan=True)
   records.append(dict(variable=v,wave=wave,source_column=col,rows=len(part),mismatches=int((~equal).sum()),raw_codes=json.dumps(pd.Series(original).value_counts().sort_index().to_dict()) if len(np.unique(original))<=30 else 'continuous/count',processed_codes=json.dumps(part[v].value_counts().sort_index().to_dict()) if part[v].nunique()<=30 else 'continuous/count'))
 report=dict(rows=len(sample),variables=len(variables),all_value_mappings_exact=all(r['mismatches']==0 for r in records),duplicate_person_wave=0,source_sha256=hashes,scope='Value alignment verified, not equivalence of questionnaire wording across waves. Gender and country are shared unsuffixed fields; household size and housing are Y1-only.')
 assert all(sha(ROOT/p)==h for p,h in hashes.items());report['inputs_unchanged']=True
 out=ROOT/'data/exp/revision_audit/variable_alignment';out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(records).to_csv(out/'variable_wave_mapping.csv',index=False)
 (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
