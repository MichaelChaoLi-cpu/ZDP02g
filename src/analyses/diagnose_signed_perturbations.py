"""Replay fixed CPU seed42 models for signed perturbations and a plumbing placebo.

The excluded-column placebo is not a substantive negative-control exposure.
No tuning/bootstrap; preserve source results and compare replayed predictions.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBClassifier,XGBRegressor
from threadpoolctl import threadpool_limits
from audit_temperature_provenance import ROOT,sha
from paired_perturbation import Perturbation,evaluate_paired_fold


def placebo(model,x,classifier):
    """Perturb an unused auxiliary column before fixed-schema projection."""
    a=x.assign(diagnostic_unused=0.);b=x.assign(diagnostic_unused=1.)
    predict=(lambda z:model.predict_proba(z)[:,list(model.classes_).index(1)]) if classifier else model.predict
    pa=predict(a.loc[:,x.columns]);pb=predict(b.loc[:,x.columns])
    err=float(np.max(np.abs(pa-pb)));assert err==0
    return err


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--output',default='data/exp/revision_signed_diagnostics_20261001')
    args=ap.parse_args();out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'src/analyses/notebooks'))
    from SettingForFeatures import data_load_combine_dataset,return_always_input_variable_list,return_aim_mediate
    data=data_load_combine_dataset('tas');X=data[return_always_input_variable_list()].copy();ms=return_aim_mediate()
    vp=ROOT/'data/processed/revision_exposure_validation/fixed_sample_exposure_variants.parquet'
    v=pd.read_parquet(vp).set_index('analysis_row');assert v.index.equals(X.index)
    for metric in ['MEAN','STD']:X[f'TAS_{metric}']=v[f'grid_TAS_{metric}'].to_numpy()
    bp=ROOT/'data/exp/revision_benchmark_seed42_20260930';bm=json.loads((bp/'manifest.json').read_text())
    sources={**bm['input_sha256'],str((bp/'all_predictions.parquet').relative_to(ROOT)):sha(bp/'all_predictions.parquet')}
    for p,h in bm['code_sha256'].items():assert sha(ROOT/p)==h
    for p in bp.glob('fold_*/inner_*.parquet'):sources[str(p.relative_to(ROOT))]=sha(p)
    pm=ROOT/'data/exp/revision_experiments/fixed_yaml_20260926/manifest.json';sources[str(pm.relative_to(ROOT))]=sha(pm)
    params=json.loads(pm.read_text())['effective_parameters'];baseline=pd.read_parquet(bp/'all_predictions.parquet')
    positive_path=ROOT/'data/exp/revision_experiments/fixed_yaml_20260926/grid/point/seed_42/predictions.parquet'
    positive=pd.read_parquet(positive_path);sources[str(positive_path.relative_to(ROOT))]=sha(positive_path)
    for p,h in sources.items():assert sha(ROOT/p)==h
    outer,inner=10,5
    if args.smoke:
        X=X.sample(800,random_state=42).sort_index();data=data.loc[X.index];outer,inner=2,2
        params={m:{**p,'n_estimators':3,'max_depth':2} for m,p in params.items()}
    assert (X.TAS_STD-.1>=0).all(),'Negative SD perturbation needs a separately approved domain rule'
    specs=[(f'{short}_{sign}',f,delta) for short,f,step in [('mean','TAS_MEAN',1.),('std','TAS_STD',.1)] for sign,delta in [('zero',0.),('positive',step),('negative',-step)]]
    manifest=dict(smoke=args.smoke,rows=len(X),seed=42,outer=outer,inner=inner,parameters=params,input_sha256=sources,code_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__)), 'src/analyses/paired_perturbation.py':sha(ROOT/'src/analyses/paired_perturbation.py')},placebo='unused auxiliary column changed before fixed predictor-schema selection; computational invariance only',provenance='CPU Figure6/benchmark replay; GPU Table2 is a separate execution')
    mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:mp.write_text(json.dumps(manifest,indent=2))
    def event(**kw):
        s=json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw));print(s,flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(s+'\n')
    results=[];checks=[]
    with threadpool_limits(limits=4):
        for fold,(tr,te) in enumerate(KFold(outer,shuffle=True,random_state=42).split(X)):
            folder=out/f'fold_{fold:02d}';folder.mkdir(exist_ok=True)
            train,test=X.iloc[tr],X.iloc[te];oof=pd.DataFrame(np.nan,index=train.index,columns=ms)
            med={n:pd.DataFrame(0.,index=test.index,columns=ms) for n in ['baseline']+[s[0] for s in specs]}
            for k,(it,iv) in enumerate(KFold(inner,shuffle=True,random_state=42).split(train)):
                for m in ms:
                    cp=folder/f'inner_{k}_{m}.parquet';jp=cp.with_suffix('.json')
                    saved=None
                    if not args.smoke:
                        saved=pd.read_parquet(bp/f'fold_{fold:02d}'/f'inner_{k}_{m}.parquet');assert saved.index.equals(train.iloc[iv].index.append(test.index))
                        oof.loc[train.iloc[iv].index,m]=saved.loc[train.iloc[iv].index,'probability']
                    if cp.exists() and jp.exists() and not args.smoke:
                        pred=pd.read_parquet(cp);check=json.loads(jp.read_text());assert pred.index.equals(test.index)
                    else:
                        event(event='mediator_start',fold=fold,inner=k,outcome=m)
                        model=XGBClassifier(**params[m]);model.fit(train.iloc[it],data.loc[train.iloc[it].index,m])
                        base=model.predict_proba(pd.concat([train.iloc[iv],test]))[:,list(model.classes_).index(1)]
                        error=0.
                        if saved is not None:error=float(np.max(np.abs(base-saved.probability.to_numpy())));assert error<=1e-7
                        else:oof.loc[train.iloc[iv].index,m]=base[:len(iv)]
                        pred=pd.DataFrame({'baseline':base[len(iv):]},index=test.index)
                        for name,feature,delta in specs:
                            z=test.copy();z[feature]+=delta
                            pred[name]=model.predict_proba(z)[:,list(model.classes_).index(1)]
                            if delta==0:assert np.array_equal(pred[name],pred.baseline)
                        check=dict(fold=fold,inner=k,outcome=m,baseline_error=error,placebo_error=placebo(model,test,True))
                        pred.to_parquet(cp);jp.write_text(json.dumps(check))
                        event(event='mediator_complete',fold=fold,inner=k,outcome=m)
                    checks.append(check)
                    for name in med:med[name][m]+=pred[name]/inner
            assert oof.notna().all().all()
            htrain=train.join(oof);htest=test.join(med['baseline']);cp=folder/'predictions.parquet';jp=folder/'health.json'
            if cp.exists() and jp.exists() and not args.smoke:
                result=pd.read_parquet(cp);check=json.loads(jp.read_text());assert result.index.equals(test.index)
            else:
                event(event='health_start',fold=fold)
                model=XGBRegressor(**params['outcome']);model.fit(htrain,data.loc[train.index,'PHYSICAL_HLTH'])
                variants=[Perturbation(n,f,d,med[n]) for n,f,d in specs]
                result=pd.DataFrame(evaluate_paired_fold(model,htest,variants,ms),index=test.index);result['fold']=fold
                error=0.;positive_error=0.
                if not args.smoke:
                    error=float(np.max(np.abs(result.baseline-baseline.loc[test.index,'primary_xgb'])));assert error<=1e-6
                    cols=[c for c in positive if c!='fold'];positive_error=float(np.max(np.abs(result[cols].to_numpy()-positive.loc[test.index,cols].to_numpy())));assert positive_error<=1e-6
                check=dict(fold=fold,outcome='PHYSICAL_HLTH',baseline_error=error,positive_replay_error=positive_error,placebo_error=placebo(model,htest,False))
                result.to_parquet(cp);jp.write_text(json.dumps(check))
            checks.append(check);results.append(result);event(event='fold_complete',fold=fold)
    result=pd.concat(results).loc[X.index];assert result.index.equals(X.index) and np.isfinite(result.to_numpy()).all()
    rows=[]
    for exposure in ['mean','std']:
        for kind in ['direct','total','indirect']:
            def vals(sign):
                d=result[f'{exposure}_{sign}_direct_change'];t=result[f'{exposure}_{sign}_total_change']
                return d if kind=='direct' else t if kind=='total' else t-d
            plus,minus=vals('positive'),vals('negative');assert (vals('zero')==0).all()
            rows.append(dict(exposure=exposure,kind=kind,positive=float(plus.mean()),negative=float(minus.mean()),signed_sum=float((plus+minus).mean()),row_mean_abs_signed_sum=float((plus+minus).abs().mean())))
    for p,h in sources.items():assert sha(ROOT/p)==h
    result.to_parquet(out/'predictions.parquet');pd.DataFrame(rows).to_csv(out/'signed_summary.csv',index=False)
    (out/'validation.json').write_text(json.dumps(dict(complete=True,smoke=args.smoke,rows=len(X),checks=checks,inputs_unchanged=True,zero_exact=True,placebo_scope=manifest['placebo'],antisymmetry='descriptive signed sums; strict antisymmetry not imposed on nonlinear finite differences'),indent=2));event(event='complete')

if __name__=='__main__':main()
