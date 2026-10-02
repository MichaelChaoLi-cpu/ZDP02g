"""Replay validated CPU benchmark/Figure6 fits to extract normalized gain for Figure5.

Fixed sample, seed and parameters; no tuning or bootstrap. Each refit must
reproduce cached held-out predictions before its importance is retained.
Gain describes model splits, not causal effects or mediated contributions.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBClassifier,XGBRegressor
from threadpoolctl import threadpool_limits
from audit_temperature_provenance import ROOT,sha


def importance(model, fold, inner, outcome):
    booster = model.get_booster()
    names = booster.feature_names
    assert names and len(names) == len(set(names))
    raw = booster.get_score(importance_type="gain")
    values = np.array([raw.get(n, 0.) for n in names], dtype=float)
    assert np.isfinite(values).all() and (values >= 0).all() and values.sum() > 0
    return pd.DataFrame(dict(feature=names, gain=values,
        importance_percent=100*values/values.sum(), fold=fold, inner=inner, outcome=outcome))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',default='data/exp/revision_feature_importance_20261001')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--max-folds',type=int,default=10)
    a=ap.parse_args();out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'src/analyses/notebooks'))
    from SettingForFeatures import data_load_combine_dataset,return_always_input_variable_list,return_aim_mediate
    data=data_load_combine_dataset('tas');X=data[return_always_input_variable_list()].copy();ms=return_aim_mediate()
    vp=ROOT/'data/processed/revision_exposure_validation/fixed_sample_exposure_variants.parquet'
    v=pd.read_parquet(vp).set_index('analysis_row');assert v.index.equals(X.index)
    for metric in ['MEAN','STD']:X[f'TAS_{metric}']=v[f'grid_TAS_{metric}'].to_numpy()
    bp=ROOT/'data/exp/revision_benchmark_seed42_20260930';bm=json.loads((bp/'manifest.json').read_text())
    for p,h in bm['input_sha256'].items():assert sha(ROOT/p)==h
    for p,h in bm['code_sha256'].items():assert sha(ROOT/p)==h
    primary_manifest=ROOT/'data/exp/revision_experiments/fixed_yaml_20260926/manifest.json'
    pm=json.loads(primary_manifest.read_text());params=pm['effective_parameters']
    baseline=pd.read_parquet(bp/'all_predictions.parquet');assert baseline.index.equals(X.index)
    outer,inner,npoints=10,5,41
    if a.smoke:
        X=X.sample(800,random_state=42).sort_index();data=data.loc[X.index];outer,inner,npoints=2,2,3
        params={m:{**p,'n_estimators':3,'max_depth':2} for m,p in params.items()}
    grids={f:np.linspace(*X[f].quantile([.05,.95]),npoints) for f in ['TAS_MEAN','TAS_STD']}
    sources={**bm['input_sha256'],str((bp/'all_predictions.parquet').relative_to(ROOT)):sha(bp/'all_predictions.parquet'),str((bp/'manifest.json').relative_to(ROOT)):sha(bp/'manifest.json')}
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in sorted(bp.glob('fold_*/inner_*.parquet'))})
    manifest=dict(purpose='Figure5 current-model provenance repair; replay validated CPU benchmark models',smoke=a.smoke,rows=len(X),seed=42,outer=outer,inner=inner,parameters=params,grids={f:g.tolist() for f,g in grids.items()},input_sha256=sources,code_sha256=sha(Path(__file__)),importance='gain normalized to 100 within each model; average inner models within outer fold, then equally average outer folds',provenance='CPU benchmark/Figure6 models; not GPU bootstrap fits')
    mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Manifest mismatch'
    else:mp.write_text(json.dumps(manifest,indent=2)+'\n')
    def event(**kw):
        s=json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw));print(s,flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(s+'\n')
    allrows=[];checks=[]
    with threadpool_limits(limits=4):
        for fold,(tr,te) in enumerate(KFold(outer,shuffle=True,random_state=42).split(X)):
            if fold >= a.max_folds: break
            folder=out/f'fold_{fold:02d}';folder.mkdir(exist_ok=True)
            train,test=X.iloc[tr],X.iloc[te];oof=pd.DataFrame(np.nan,index=train.index,columns=ms);mt=pd.DataFrame(0.,index=test.index,columns=ms)
            if not a.smoke:assert (baseline.loc[test.index,'fold']==fold).all()
            for k,(it,iv) in enumerate(KFold(inner,shuffle=True,random_state=42).split(train)):
                for m in ms:
                    cp=folder/f'{m}_inner_{k}.csv';checkp=cp.with_suffix('.json')
                    cache=bp/f'fold_{fold:02d}'/f'inner_{k}_{m}.parquet'
                    expected=train.iloc[iv].index.append(test.index)
                    if not a.smoke:
                        saved=pd.read_parquet(cache);assert saved.index.equals(expected)
                        oof.loc[train.iloc[iv].index,m]=saved.loc[train.iloc[iv].index,'probability']
                        mt[m]+=saved.loc[test.index,'probability']/inner
                    if cp.exists() and checkp.exists() and not a.smoke:
                        df=pd.read_csv(cp,dtype={'wave':str});checks.append(json.loads(checkp.read_text()))
                    else:
                        event(event='mediator_start',fold=fold,inner=k,outcome=m)
                        model=XGBClassifier(**params[m]);model.fit(train.iloc[it],data.loc[train.iloc[it].index,m])
                        pred=model.predict_proba(pd.concat([train.iloc[iv],test]))[:,list(model.classes_).index(1)]
                        error=0.
                        if a.smoke:
                            oof.loc[train.iloc[iv].index,m]=pred[:len(iv)];mt[m]+=pred[len(iv):]/inner
                        else:
                            error=float(np.max(np.abs(pred-saved.probability.to_numpy())));assert error<=1e-7,(fold,k,m,error)
                        check=dict(fold=fold,inner=k,outcome=m,baseline_max_abs_difference=error);checks.append(check)
                        df=importance(model,fold,k,m)
                        df.to_csv(cp,index=False);checkp.write_text(json.dumps(check)+'\n')
                        event(event='mediator_complete',fold=fold,inner=k,outcome=m)
                    df['inner']=k;allrows.append(df)
            assert oof.notna().all().all()
            htrain=train.join(oof);htest=test.join(mt);hp=folder/'health.csv';hc=folder/'health.json'
            if hp.exists() and hc.exists() and not a.smoke:
                df=pd.read_csv(hp,dtype={'wave':str});checks.append(json.loads(hc.read_text()))
            else:
                model=XGBRegressor(**params['outcome']);model.fit(htrain,data.loc[train.index,'PHYSICAL_HLTH'])
                error=0.
                if not a.smoke:
                    error=float(np.max(np.abs(model.predict(htest)-baseline.loc[test.index,'primary_xgb'])));assert error<=1e-6,error
                check=dict(fold=fold,outcome='PHYSICAL_HLTH',baseline_max_abs_difference=error);checks.append(check)
                df=importance(model,fold,-1,'PHYSICAL_HLTH')
                df.to_csv(hp,index=False);hc.write_text(json.dumps(check)+'\n')
            df['inner']=-1;allrows.append(df);event(event='fold_complete',fold=fold)
    full=pd.concat(allrows,ignore_index=True)
    assert np.isfinite(full[['gain','importance_percent']]).all().all()
    assert np.allclose(full.groupby(['outcome','fold','inner']).importance_percent.sum(),100)
    folded=full.groupby(['outcome','feature','fold'],as_index=False).importance_percent.mean()
    summary=folded.groupby(['outcome','feature'],as_index=False).importance_percent.agg(['mean','std']).reset_index()
    summary['rank']=summary.groupby('outcome')['mean'].rank(ascending=False,method='min').astype(int)
    for p,h in sources.items():assert sha(ROOT/p)==h
    full.to_csv(out/'component_importance.csv',index=False)
    folded.to_csv(out/'fold_importance.csv',index=False)
    summary.to_csv(out/'summary.csv',index=False)
    complete=full['fold'].nunique()==outer
    (out/'validation.json').write_text(json.dumps(dict(smoke=a.smoke,complete=complete,
        rows=len(X),completed_folds=int(full['fold'].nunique()),component_checks=checks,
        expected_components=outer*(inner*len(ms)+1),actual_components=len(checks),
        inputs_unchanged=True,normalized_gain_sums_to_100=True,
        evidence_scope='CPU benchmark and Figure6 model replay; gain is not mediation evidence'),indent=2)+'\n')
    event(event='complete' if complete else 'partial',folds=int(full['fold'].nunique()),smoke=a.smoke)

if __name__=='__main__':main()
