"""Replay legacy preprocessing in memory; export aggregate sample-flow diagnostics.

Run from repository root with uv run python src/analyses/audit_sample_flow.py.
Never execute the source script's output-writing statements.
"""
import ast
import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
source = ROOT / 'src/analyses/nbs/c01_gfs_data_preprocess.py'
paths = [source, ROOT/'data/raw/gfs_all-countries_wave2-with-midyear_sensitive.csv',
         ROOT/'data/processed/GlobalFlourishingDataWithLonLit.parquet',
         ROOT/'data/processed/df_out_2023.parquet', ROOT/'data/processed/df_out_2024.parquet']
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
code = source.read_text(); tree = ast.parse(code)
env = {'pd':pd, 'np':np}; flow=[]; active=False; initial_wave_counts=None
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == 'pivot_raw_data_longer':
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='income_mapping' for t in node.targets):
        env['income_mapping']=ast.literal_eval(node.value)
    if isinstance(node, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='df_raw' for t in node.targets):active=True
    if not active:continue
    # Whitelist data transformations and list-building calls; never execute saves.
    allowed=isinstance(node,(ast.Assign,ast.For)) or (isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and isinstance(node.value.func.value,ast.Name) and node.value.func.value.id=='variable_list' and node.value.func.attr=='append')
    if not allowed:continue
    before=len(env['df_long']) if 'df_long' in env else None
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    if 'df_long' in env and (before is None or len(env['df_long'])!=before):
        if before is None:
            initial_wave_counts=env['df_long']['WAVE'].value_counts(dropna=False).to_dict()
        flow.append(dict(source_line=node.lineno,operation=ast.get_source_segment(code,node),before=before,remaining=len(env['df_long']),excluded=None if before is None else before-len(env['df_long'])))
replay=env['WahsedDataset'];stored=pd.read_parquet(paths[2])
try:
    pd.testing.assert_frame_equal(replay,stored,check_dtype=False); match=True; mismatch=None
except AssertionError as e:match=False;mismatch=str(e)[:1500]
exposures=[]
for wave,year in [(1,2023),(2,2024)]:
    x=pd.read_parquet(ROOT/f'data/processed/df_out_{year}.parquet');x['WAVE']=float(wave)
    x=x[['LATITUDE','LONGITUDE','WAVE',f'tas_mean_{year}',f'tas_std_{year}']]
    x.columns=['LATITUDE','LONGITUDE','WAVE','TAS_MEAN','TAS_STD'];exposures.append(x)
joined=stored.merge(pd.concat(exposures),on=['LATITUDE','LONGITUDE','WAVE'],how='left',validate='many_to_one',indicator=True)
matched=joined.loc[joined['_merge'].eq('both')].drop(columns='_merge')
missing=matched.drop(columns='ID').isna();sample=matched.loc[~missing.any(axis=1)].copy()
sys.path.insert(0,str(ROOT/'src/analyses/notebooks'))
from SettingForFeatures import data_load_combine_dataset
pd.testing.assert_frame_equal(sample.drop(columns='ID'),data_load_combine_dataset('tas'))
out=ROOT/'data/exp/revision_audit/sample_flow';out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(flow).to_csv(out/'legacy_replay_flow.csv',index=False)
missing.sum().rename('missing_rows').to_csv(out/'missing_by_variable.csv')
for name,frame in [('processed',stored),('final',sample)]:
    frame.groupby(['COUNTRY','WAVE']).size().rename('records').to_csv(out/f'{name}_country_wave.csv')
observed_waves={str(w):int(pd.to_numeric(env['df_raw'][f'WAVE_Y{w}'],errors='coerce').eq(w).sum()) for w in [1,2]}
observed_total=sum(observed_waves.values())
unmapped=env['df_long'].loc[env['df_long']['INCOME_REAL'].isna(),'INCOME'].value_counts().to_dict()
report=dict(observed_wave_records=observed_waves,observed_records=observed_total,absent_wave_slots=2*len(env['df_raw'])-observed_total,total_excluded_observed_records=observed_total-len(sample),unmapped_income_codes={str(k):int(v) for k,v in unmapped.items()},raw_wide_rows=len(env['df_raw']),raw_unique_people=env['df_raw']['ID'].nunique(),stacked_slots=2*len(env['df_raw']),legacy_replay_rows=len(replay),stored_rows=len(stored),legacy_replay_matches_stored=match,mismatch=mismatch,exposure_unmatched_rows=int(joined['_merge'].ne('both').sum()),matched_rows=len(matched),complete_case_excluded=int(missing.any(axis=1).sum()),final_rows=len(sample),final_people=sample.ID.nunique(),duplicate_person_wave=int(sample.duplicated(['ID','WAVE']).sum()),production_loader_exact_match=True,missing_pattern_counts=missing.apply(lambda row: ','.join(missing.columns[row]),axis=1).value_counts().to_dict(),source_sha256=hashes,inputs_unchanged=all(sha(ROOT/p)==h for p,h in hashes.items()),caution='Stacked slots include absent wave interviews. Replay exclusion counts must not be reported as the actual historical flow unless replay matches the stored artifact.')
(out/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
