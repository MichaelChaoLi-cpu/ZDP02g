"""Read-only income mapping audit; requires the saved public World Bank API snapshot.

Run from the repository root; exports aggregate diagnostics only.
"""
import ast,json,hashlib,sys
from pathlib import Path
import pandas as pd
import numpy as np
root=Path.cwd();source=root/'src/analyses/nbs/c01_gfs_data_preprocess.py';maps={}
for n in ast.parse(source.read_text()).body:
 if isinstance(n,ast.Assign):
  for t in n.targets:
   if isinstance(t,ast.Name) and t.id in ['income_mapping','exchange_rates']:maps[t.id]=ast.literal_eval(n.value)
rawp=root/'data/raw/gfs_all-countries_wave2-with-midyear_sensitive.csv';surveyp=root/'data/processed/GlobalFlourishingDataWithLonLit.parquet';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();before={str(p.relative_to(root)):sha(p) for p in [source,rawp,surveyp]}
raw=pd.read_csv(rawp,usecols=['ID','INCOME_Y1','INCOME_Y2'],low_memory=False)
for c in ['INCOME_Y1','INCOME_Y2']:raw[c]=pd.to_numeric(raw[c],errors='coerce')
survey=pd.read_parquet(surveyp)
long=pd.concat([raw[['ID',f'INCOME_Y{w}']].rename(columns={f'INCOME_Y{w}':'source_code'}).assign(WAVE=float(w)) for w in [1,2]],ignore_index=True)
x=survey[['ID','WAVE','INCOME_REAL']].merge(long,on=['ID','WAVE'],how='left',validate='one_to_one',indicator=True);assert x['_merge'].eq('both').all()
x['reconstructed']=x.source_code.map(maps['income_mapping']);mask=x.INCOME_REAL.notna();assert x.loc[mask,'reconstructed'].notna().all()
diff=np.abs(x.loc[mask,'reconstructed']-x.loc[mask,'INCOME_REAL']);assert (diff==0).all()
sys.path.insert(0,str(root/'src/analyses/notebooks'));from SettingForFeatures import data_load_combine_dataset
sample=data_load_combine_dataset('tas');out=root/'data/exp/revision_audit/income'
wb=json.loads((out/'worldbank_exchange_rates.json').read_text())[1];aliases={'Egypt, Arab Rep.':'Egypt','Hong Kong SAR, China':'Hong Kong','Turkiye':'Turkey'};checks=[]
for name,rate in maps['exchange_rates'].items():
 values=[r['value'] for r in wb if aliases.get(r['country']['value'],r['country']['value'])==name];assert len(values)==2
 checks.append({'country':name,'code_rate':rate,'worldbank_mean_rounded':round(sum(values)/2,4)})
assert all(v['code_rate']==v['worldbank_mean_rounded'] for v in checks)
report={'processed_rows':len(x),'nonmissing_income_rows':int(mask.sum()),'mapping_max_abs_difference':float(diff.max()),'mapping_entries':len(maps['income_mapping']),'mapping_maximum':max(maps['income_mapping'].values()),'codes_at_maximum':[int(k) for k,v in maps['income_mapping'].items() if v==240000.],'analysis_rows':len(sample),'analysis_income_maximum':float(sample.INCOME_REAL.max()),'analysis_rows_at_240000':int(sample.INCOME_REAL.eq(240000).sum()),'source_sha256':before,'inputs_unchanged':all(sha(root/n)==h for n,h in before.items()),'exchange_rates_verified':checks,'ppp_status':'No PPP adjustment in inspected mapping or construction path.'}
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');pd.DataFrame(sorted(maps['income_mapping'].items()),columns=['source_code','mapped_usd']).to_csv(out/'income_mapping.csv',index=False)
print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','exchange_rates_verified']},indent=2))
