"""One fixed-seed paired spatial aggregation sensitivity; no tuning/bootstrap.

Fit coarsened exposures, compare existing native CPU benchmark, evaluate health
PDPs at existing native grid points within both pooled5th–95th percentile ranges.
Same sample/folds/parameters. Marginal overlap is not joint support certification.
"""
import argparse,json,sys,time,platform
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score,log_loss,brier_score_loss
from threadpoolctl import threadpool_limits
from paired_perturbation import Perturbation,evaluate_paired_fold
from regenerate_probability_pdp import curves
from audit_temperature_provenance import ROOT,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--smoke',action='store_true');ap.add_argument('--output',default='data/exp/revision_spatial_coarsening_seed42_20261001');a=ap.parse_args()
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'src/analyses/notebooks'))
    from SettingForFeatures import data_load_combine_dataset,return_always_input_variable_list,return_aim_mediate
    data=data_load_combine_dataset('tas');ms=return_aim_mediate();X=data[return_always_input_variable_list()].copy()
    ep=ROOT/'data/exp/revision_audit/r1c4_spatial_coarsening';ev=json.loads((ep/'validation.json').read_text());assert ev['inputs_unchanged'] and len(ev['native_reproduction_checks'])==4
    for p,h in ev['input_sha256'].items():assert sha(ROOT/p)==h
    candidate=pd.read_parquet(ep/'candidate_exposures.parquet');assert candidate.index.equals(X.index) and len(X)==273031
    for f in ['TAS_MEAN','TAS_STD']:X[f]=candidate['coarse_'+f]
    bp=ROOT/'data/exp/revision_benchmark_seed42_20260930';bm=json.loads((bp/'manifest.json').read_text())
    for p,h in bm['input_sha256'].items():assert sha(ROOT/p)==h
    for p,h in bm['code_sha256'].items():assert sha(ROOT/p)==h
    nativep=ROOT/'data/exp/revision_experiments/fixed_yaml_20260926/grid/point/seed_42/predictions.parquet';native=pd.read_parquet(nativep);assert native.index.equals(X.index)
    mp=ROOT/'data/exp/revision_experiments/fixed_yaml_20260926/manifest.json';params=json.loads(mp.read_text())['effective_parameters']
    assert all(p['device']=='cpu' and p['random_state']==42 for p in params.values())
    cp=ROOT/'data/exp/revision_probability_pdp_20260930/curves.csv';native_curves=pd.read_csv(cp,dtype={'wave':str})
    grids={}
    for f in ['TAS_MEAN','TAS_STD']:
        old=native_curves.query("outcome == 'PHYSICAL_HLTH' and wave == 'pooled'");old=old[old.feature.eq(f)].sort_values('value');lo,hi=X[f].quantile([.05,.95]);g=old.value.to_numpy();grids[f]=g[(g>=lo)&(g<=hi)];assert len(grids[f])>=20
    sources={**ev['input_sha256'],**bm['input_sha256']}
    for p in [ep/'candidate_exposures.parquet',ep/'validation.json',bp/'manifest.json',nativep,mp,cp]:sources[str(p.relative_to(ROOT))]=sha(p)
    outer,inner=10,5
    if a.smoke:
        X=X.sample(800,random_state=42).sort_index();data=data.loc[X.index];outer,inner=2,2;params={m:{**p,'n_estimators':3,'max_depth':2} for m,p in params.items()};grids={f:g[[0,len(g)//2,-1]] for f,g in grids.items()}
    manifest=dict(decision='KILA-D-20261001-020',smoke=a.smoke,rows=len(X),seed=42,outer=outer,inner=inner,parameters=params,input_sha256=sources,code_sha256={p:sha(ROOT/p) for p in ['src/analyses/run_spatial_coarsening_sensitivity.py','src/analyses/paired_perturbation.py','src/analyses/regenerate_probability_pdp.py','src/analyses/notebooks/SettingForFeatures.py']},grids={f:g.tolist() for f,g in grids.items()},software=dict(python=platform.python_version(),xgboost=xgboost.__version__),scope='One specified2x2 coarsening; native reference is existing CPU result; no bootstrap intervals; common marginal PDP range is not joint support.')
    p=out/'manifest.json'
    if p.exists():assert json.loads(p.read_text())==manifest,'Manifest mismatch'
    else:p.write_text(json.dumps(manifest,indent=2))
    def event(**kw):
        s=json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw));print(s,flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(s+'\n')
    specs=[('mean_zero','TAS_MEAN',0.),('mean_positive','TAS_MEAN',1.),('std_zero','TAS_STD',0.),('std_positive','TAS_STD',.1)]
    allpred=[];allcurves=[];medmetrics=[]
    with threadpool_limits(limits=4):
        for fold,(tr,te) in enumerate(KFold(outer,shuffle=True,random_state=42).split(X)):
            folder=out/f'fold_{fold:02d}';folder.mkdir(exist_ok=True);train,test=X.iloc[tr],X.iloc[te]
            if not a.smoke:assert (native.loc[test.index,'fold']==fold).all()
            oof=pd.DataFrame(np.nan,index=train.index,columns=ms);mt={n:pd.DataFrame(0.,index=test.index,columns=ms) for n in ['baseline']+[s[0] for s in specs]}
            for k,(it,iv) in enumerate(KFold(inner,shuffle=True,random_state=42).split(train)):
                for m in ms:
                    cache=folder/f'inner_{k}_{m}.parquet';expected=train.iloc[iv].index.append(test.index)
                    if cache.exists():saved=pd.read_parquet(cache);assert saved.index.equals(expected)
                    else:
                        event(event='mediator_start',fold=fold,inner=k,outcome=m);model=XGBClassifier(**params[m]);model.fit(train.iloc[it],data.loc[train.iloc[it].index,m]);pos=list(model.classes_).index(1)
                        saved=pd.DataFrame({'baseline':model.predict_proba(pd.concat([train.iloc[iv],test]))[:,pos]},index=expected)
                        for n,f,d in specs:
                            z=test.copy();z[f]+=d;pr=model.predict_proba(z)[:,pos]
                            if d==0:assert np.array_equal(pr,saved.loc[test.index,'baseline'])
                            saved[n]=np.r_[np.repeat(np.nan,len(iv)),pr]
                        tmp=cache.with_suffix('.tmp.parquet');saved.to_parquet(tmp);tmp.replace(cache);event(event='mediator_complete',fold=fold,inner=k,outcome=m)
                    # NaN padding promotes variant columns to float64; restore model-output
                    # dtype before division so baseline and zero arms use identical arithmetic.
                    saved=saved.astype(np.float32)
                    for name in ['mean_zero','std_zero']:
                        assert np.array_equal(saved.loc[test.index,name],saved.loc[test.index,'baseline'])
                    oof.loc[train.iloc[iv].index,m]=saved.loc[train.iloc[iv].index,'baseline']
                    for n in mt:mt[n][m]+=saved.loc[test.index,n]/inner
            assert oof.notna().all().all()
            for name in ['mean_zero','std_zero']:
                assert np.array_equal(mt[name].to_numpy(),mt['baseline'].to_numpy())
            for m in ms:
                assert mt['baseline'][m].between(0,1).all()
            hp=folder/'predictions.parquet';hc=folder/'health_curves.csv'
            if hp.exists() and hc.exists():result=pd.read_parquet(hp);cur=pd.read_csv(hc,dtype={'wave':str});assert result.index.equals(test.index)
            else:
                event(event='health_start',fold=fold);model=XGBRegressor(**params['outcome']);model.fit(train.join(oof),data.loc[train.index,'PHYSICAL_HLTH']);htest=test.join(mt['baseline'])
                result=pd.DataFrame(evaluate_paired_fold(model,htest,[Perturbation(n,f,d,mt[n]) for n,f,d in specs],ms),index=test.index);result['fold']=fold
                cur=curves(model,htest,grids,False,data.loc[test.index,'WAVE'],fold,'PHYSICAL_HLTH');result.to_parquet(hp);cur.to_csv(hc,index=False)
            for m in ms:result['prob_'+m]=mt['baseline'][m]
            allpred.append(result);allcurves.append(cur);event(event='fold_complete',fold=fold)
    result=pd.concat(allpred).loc[X.index];assert result.index.equals(X.index) and np.isfinite(result.to_numpy()).all()
    zero=[c for c in result if '_zero_' in c and c.endswith('_change')];assert (result[zero]==0).all().all()
    summary=[];health=[]
    for label,pred in [('native',native.loc[X.index]),('coarse',result)]:
        health.append(dict(variant=label,n=len(X),rmse=float(np.sqrt(mean_squared_error(data.PHYSICAL_HLTH,pred.baseline))),r2=float(r2_score(data.PHYSICAL_HLTH,pred.baseline))))
        for country,idx in [('all',X.index)]+[(str(k),g.index) for k,g in data.groupby('COUNTRY')]:
            for exposure in ['mean','std']:
                direct=pred.loc[idx,f'{exposure}_positive_direct_change'].to_numpy(dtype=float);total=pred.loc[idx,f'{exposure}_positive_total_change'].to_numpy(dtype=float)
                summary.append(dict(variant=label,country=country,n=len(idx),exposure=exposure,direct=float(direct.mean()),total=float(total.mean()),indirect=float((total-direct).mean())))
    for m in ms:
        y=data[m];p=result['prob_'+m];medmetrics.append(dict(outcome=m,auc=roc_auc_score(y,p),log_loss=log_loss(y,p),brier=brier_score_loss(y,p)))
    folded=pd.concat(allcurves,ignore_index=True);folded['weighted']=folded.prediction*folded.n
    final=folded.groupby(['outcome','feature','grid_index','value','wave'],as_index=False).agg(weighted=('weighted','sum'),n=('n','sum'));final['prediction']=final.weighted/final.n
    assert (final.loc[final.wave.eq('pooled'),'n']==len(X)).all()
    for p,h in sources.items():assert sha(ROOT/p)==h
    result.to_parquet(out/'predictions.parquet');folded.to_csv(out/'fold_curves.csv',index=False);final.to_csv(out/'curves.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'contrasts.csv',index=False);pd.DataFrame(health).to_csv(out/'health_metrics.csv',index=False);pd.DataFrame(medmetrics).to_csv(out/'mediator_metrics.csv',index=False)
    # Each native comparison is an exact existing grid value, without interpolation.
    nc=native_curves.query("outcome == 'PHYSICAL_HLTH' and wave == 'pooled'")[['feature','value','prediction']].rename(columns={'prediction':'native'})
    comparison=final[final.wave.eq('pooled')][['feature','value','prediction']].rename(columns={'prediction':'coarse'}).merge(nc,on=['feature','value'],validate='one_to_one');assert len(comparison)==sum(map(len,grids.values()))
    comparison['difference']=comparison.coarse-comparison.native;comparison.to_csv(out/'pdp_comparison.csv',index=False)
    (out/'validation.json').write_text(json.dumps(dict(complete=True,smoke=a.smoke,rows=len(X),folds=outer,zero_exact=True,inputs_unchanged=True,fold_assignment_matches_native=not a.smoke,confidence_intervals=False,scope=manifest['scope']),indent=2));event(event='complete')

if __name__=='__main__':main()
