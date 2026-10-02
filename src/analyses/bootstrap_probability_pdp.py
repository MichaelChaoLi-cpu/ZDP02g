"""Full-pipeline country/grid bootstrap PDPs, fixed seed 42 and hyperparameters.

Fixed exposure grid, bootstrap-weighted held-out averages, baseline mediator
features held fixed for the health PDP. Checkpoints contain curves, not models.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor
from threadpoolctl import threadpool_limits
from audit_temperature_provenance import ROOT, sha
from nested_grid_bootstrap import draw_grid_multiplicities


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True)
    ap.add_argument('--reps', type=int, default=100)
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    assert a.reps >= 2
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / 'src/analyses/notebooks'))
    from SettingForFeatures import data_load_combine_dataset, return_always_input_variable_list, return_aim_mediate
    out = ROOT / a.output
    out.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / 'data/processed' / (n + '.parquet') for n in
             ['GlobalFlourishingDataWithLonLit', 'df_out_2023', 'df_out_2024']]
    paths += [ROOT / 'data/processed/revision_exposure_validation' / n for n in
              ['native_grid_location_year.parquet', 'fixed_sample_exposure_variants.parquet']]
    data = data_load_combine_dataset('tas')
    v = pd.read_parquet(paths[-1]).set_index('analysis_row')
    assert v.index.equals(data.index)
    for metric in ['MEAN', 'STD']:
        data[f'TAS_{metric}'] = v[f'grid_TAS_{metric}'].to_numpy()
    if a.smoke:
        data = data.sample(800, random_state=42).sort_index()
    X = data[return_always_input_variable_list()].copy()
    mediators = return_aim_mediate()
    cells = pd.read_parquet(paths[-2])
    cells = cells.loc[cells.YEAR.eq(2023), ['LATITUDE', 'LONGITUDE', 'native_lat_idx', 'native_lon_idx']]
    metadata = data[['COUNTRY', 'LATITUDE', 'LONGITUDE']].merge(cells, on=['LATITUDE', 'LONGITUDE'], how='left', sort=False, validate='many_to_one')
    assert np.array_equal(metadata[['LATITUDE', 'LONGITUDE']], data[['LATITUDE', 'LONGITUDE']])
    metadata.index = data.index
    params = {}
    for m in mediators + ['outcome']:
        path = ROOT / 'src/analyses/config' / ('params.yaml' if m == 'outcome' else f'{m}_params.yaml')
        paths.append(path)
        params[m] = {**yaml.safe_load(path.read_text()), 'device': 'cpu', 'n_jobs': a.threads}
        if a.smoke:
            params[m].update(n_estimators=3, max_depth=2)
    reference = ROOT / 'data/exp/revision_probability_pdp_20260930/curves.csv'
    paths.append(reference)
    ref = pd.read_csv(reference, dtype={'wave': str})
    grids = {f: ref.loc[(ref.feature == f) & (ref.wave == 'pooled') & (ref.outcome == 'PHYSICAL_HLTH')].sort_values('grid_index').value.to_numpy()
             for f in ['TAS_MEAN', 'TAS_STD']}
    if a.smoke:
        grids = {f: g[[0, 20, 40]] for f, g in grids.items()}
    outer, inner = (2, 2) if a.smoke else (10, 5)
    manifest = dict(decision='KILA-D-20260930-011', smoke=a.smoke, reps=a.reps, seed=42,
                    rows=len(X), outer=outer, inner=inner, parameters=params,
                    grids={f:g.tolist() for f,g in grids.items()},
                    inputs={str(p.relative_to(ROOT)):sha(p) for p in paths},
                    code={n:sha(ROOT/'src/analyses'/n) for n in ['bootstrap_probability_pdp.py','nested_grid_bootstrap.py','notebooks/SettingForFeatures.py']},
                    statistic='bootstrap-multiplicity weighted pooled held-out PDP; fixed baseline generated mediators for health')
    mp = out / 'manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text()) == manifest, 'Manifest mismatch'
    else:
        mp.write_text(json.dumps(manifest, indent=2)+'\n')
    def event(**kw):
        line = json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), **kw))
        print(line, flush=True)
        with (out/'events.jsonl').open('a') as file:
            file.write(line+'\n')
    def profile(model, frame, outcome, weights, classification):
        rows = []
        for f, grid in grids.items():
            changed = frame.copy()
            for j, value in enumerate(grid):
                changed[f] = value
                prediction = model.predict_proba(changed)[:,list(model.classes_).index(1)] if classification else model.predict(changed)
                assert np.isfinite(prediction).all()
                if classification:
                    assert ((prediction >= 0) & (prediction <= 1)).all()
                rows.append(dict(outcome=outcome, feature=f, grid_index=j, value=float(value),
                                 weighted_sum=float(np.dot(prediction.astype(float), weights)), weight=int(weights.sum())))
        return pd.DataFrame(rows)
    keys = ['outcome','feature','grid_index','value']
    all_draws = []
    with threadpool_limits(limits=a.threads):
        for b in range(-1, a.reps):
            folder = out / ('point' if b < 0 else f'bootstrap_{b:04d}')
            folder.mkdir(exist_ok=True)
            counts = pd.Series(1, index=X.index) if b < 0 else draw_grid_multiplicities(metadata, 2026092600+b)[0]
            counts.to_csv(folder/'multiplicities.csv', index_label='analysis_row')
            check = metadata.assign(count=counts)
            assert check.groupby(['COUNTRY','native_lat_idx','native_lon_idx'])['count'].nunique().eq(1).all()
            folds = []
            for fold,(tr,te) in enumerate(KFold(outer,shuffle=True,random_state=42).split(X)):
                cp = folder / f'fold_{fold:02d}.csv'
                if cp.exists():
                    saved = pd.read_csv(cp, float_precision='round_trip')
                    assert len(saved)==4*2*len(grids['TAS_MEAN']) and np.isfinite(saved[['weighted_sum','weight']]).all().all()
                    folds.append(saved)
                    continue
                started = time.monotonic()
                event(event='fold_start', replicate=b, fold=fold)
                train, test = X.iloc[tr], X.iloc[te]
                w = counts.iloc[te].to_numpy()
                assert w.sum()>0
                oof = pd.DataFrame(np.nan,index=train.index,columns=mediators)
                mt = pd.DataFrame(0.,index=test.index,columns=mediators)
                components=[]
                for k,(it,iv) in enumerate(KFold(inner,shuffle=True,random_state=42).split(train)):
                    expanded = np.repeat(tr[it], counts.iloc[tr[it]].to_numpy(dtype=int))
                    assert len(expanded)>0 and not np.intersect1d(expanded,te).size
                    for m in mediators:
                        assert data.iloc[expanded][m].nunique()==2
                        model=XGBClassifier(**params[m])
                        model.fit(X.iloc[expanded],data.iloc[expanded][m])
                        pos=list(model.classes_).index(1)
                        oof.loc[train.iloc[iv].index,m]=model.predict_proba(train.iloc[iv])[:,pos]
                        mt[m]+=model.predict_proba(test)[:,pos]/inner
                        part=profile(model,test,m,w,True)
                        part['weighted_sum']/=inner
                        part['weight']=part['weight']/inner
                        components.append(part)
                        event(event='mediator_complete',replicate=b,fold=fold,inner=k,outcome=m)
                assert oof.notna().all().all()
                expanded=np.repeat(np.arange(len(tr)), counts.iloc[tr].to_numpy(dtype=int))
                model=XGBRegressor(**params['outcome'])
                model.fit(train.join(oof).iloc[expanded],data.iloc[tr].PHYSICAL_HLTH.iloc[expanded])
                components.append(profile(model,test.join(mt),'PHYSICAL_HLTH',w,False))
                saved=pd.concat(components).groupby(keys,as_index=False)[['weighted_sum','weight']].sum()
                tmp=cp.with_suffix('.tmp.csv');saved.to_csv(tmp,index=False);tmp.replace(cp)
                folds.append(saved)
                event(event='fold_complete',replicate=b,fold=fold,seconds=time.monotonic()-started)
            curve=pd.concat(folds).groupby(keys,as_index=False)[['weighted_sum','weight']].sum()
            assert np.allclose(curve.weight,counts.sum())
            curve['prediction']=curve.weighted_sum/curve.weight
            curve.to_csv(folder/'curves.csv',index=False)
            if b==-1 and not a.smoke:
                compare=curve.merge(ref[ref.wave.eq('pooled')],on=['outcome','feature','grid_index'],suffixes=('_new','_reference'),validate='one_to_one')
                error=float(np.max(np.abs(compare.prediction_new-compare.prediction_reference)))
                (out/'point_parity.json').write_text(json.dumps({'max_abs_difference':error,'tolerance':1e-6})+'\n')
                assert error<=1e-6, f'Point estimate differs from existing figure: {error}'
            if b>=0:
                all_draws.append(curve.assign(replicate=b))
            event(event='replicate_complete',replicate=b)
    draws=pd.concat(all_draws)
    draws.to_csv(out/'bootstrap_curves.csv',index=False)
    bands=draws.groupby(keys).prediction.agg(lower=lambda x:x.quantile(.025),upper=lambda x:x.quantile(.975),bootstrap_se='std',replicates='count').reset_index()
    assert bands.replicates.eq(a.reps).all()
    bands.to_csv(out/'pointwise_intervals.csv',index=False)
    assert all(sha(ROOT/p)==h for p,h in manifest['inputs'].items())
    (out/'completed.json').write_text(json.dumps(dict(smoke=a.smoke,reps=a.reps,inputs_unchanged=True,interval='pointwise percentile 95%; fixed hyperparameters/splits; not simultaneous',production_intervals=not a.smoke),indent=2)+'\n')
    event(event='complete')

if __name__=='__main__':
    main()
