"""Fixed-seed held-out PDPs for Fig6; no tuning, bands or polynomial fits.

Classifiers average class-1 probabilities. Health PDP holds baseline generated
mediators fixed: a single-input prediction profile, not a total pathway effect.
All held-out records contribute; wave curves share the pooled fitted models.
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBClassifier,XGBRegressor
from threadpoolctl import threadpool_limits
from audit_temperature_provenance import ROOT,sha


def curves(model,X,grids,classification,wave,fold,outcome):
    rows=[]
    for feature,grid in grids.items():
        changed=X.copy()
        for j,value in enumerate(grid):
            changed[feature]=value
            if classification:
                pos=list(model.classes_).index(1)
                p=model.predict_proba(changed)[:,pos]
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
            else:p=model.predict(changed);assert np.isfinite(p).all()
            for label,mask in [('pooled',np.ones(len(X),dtype=bool))]+[(str(int(w)),wave.to_numpy()==w) for w in sorted(wave.unique())]:
                rows.append(dict(fold=fold,outcome=outcome,feature=feature,grid_index=j,value=float(value),wave=label,n=int(mask.sum()),prediction=float(np.mean(p[mask],dtype=np.float64))))
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',default='data/exp/revision_probability_pdp_20260930')
    ap.add_argument('--smoke',action='store_true')
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
    manifest=dict(decision='KILA-D-20260930-006',smoke=a.smoke,rows=len(X),seed=42,outer=outer,inner=inner,parameters=params,grids={f:g.tolist() for f,g in grids.items()},input_sha256=sources,code_sha256=sha(Path(__file__)),health_mediators='fixed at baseline generated probabilities',evaluation='all held-out records; sample-count weighting; pooled and wave-specific averages')
    mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Manifest mismatch'
    else:mp.write_text(json.dumps(manifest,indent=2)+'\n')
    def event(**kw):
        s=json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw));print(s,flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(s+'\n')
    allrows=[];checks=[]
    with threadpool_limits(limits=4):
        for fold,(tr,te) in enumerate(KFold(outer,shuffle=True,random_state=42).split(X)):
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
                        df=curves(model,test,grids,True,data.loc[test.index,'WAVE'],fold,m)
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
                df=curves(model,htest,grids,False,data.loc[test.index,'WAVE'],fold,'PHYSICAL_HLTH')
                df.to_csv(hp,index=False);hc.write_text(json.dumps(check)+'\n')
            df['inner']=-1;allrows.append(df);event(event='fold_complete',fold=fold)
    full=pd.concat(allrows,ignore_index=True);assert np.isfinite(full[['value','prediction','n']]).all().all()
    keys=['fold','outcome','feature','grid_index','value','wave','n']
    folded=full.groupby(keys,as_index=False).prediction.mean()
    folded['weighted']=folded.prediction*folded.n
    keys=['outcome','feature','grid_index','value','wave']
    final=folded.groupby(keys,as_index=False).agg(weighted=('weighted','sum'),n=('n','sum'));final['prediction']=final.weighted/final.n
    assert len(final)==4*2*npoints*3
    assert (final.loc[final.wave.eq('pooled'),'n']==len(X)).all()
    for p,h in sources.items():assert sha(ROOT/p)==h
    full.to_csv(out/'component_curves.csv',index=False);final.to_csv(out/'curves.csv',index=False)
    (out/'validation.json').write_text(json.dumps(dict(smoke=a.smoke,rows=len(X),component_checks=checks,inputs_unchanged=True,confidence_intervals=False),indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(10,13),layout='constrained')
    for i,m in enumerate(['PHYSICAL_HLTH']+ms):
        for j,feature in enumerate(grids):
            ax=axes[i,j]
            for wave,label,color in [('pooled','Pooled','#222222'),('1','Wave 1','#2878B5'),('2','Wave 2','#D65F24')]:
                g=final[(final.outcome==m)&(final.feature==feature)&(final.wave==wave)].sort_values('value')
                assert len(g)==npoints
                ax.plot(g.value,g.prediction,label=label,color=color,lw=1.7)
            ax.set_xlabel('Annual mean temperature (°C)' if j==0 else 'Annual temperature SD (°C)')
            ax.set_ylabel('Predicted health score' if i==0 else 'Predicted probability')
            ax.set_title(['Overall physical health','Bodily pain','Activity limitation','Weekly exercise'][i]);ax.grid(alpha=.15)
            if i:ax.set_ylim(0,1)
            ax.text(.02,.95,'abcdefgh'[i*2+j],transform=ax.transAxes,va='top',fontweight='bold')
    axes[0,0].legend(fontsize=8)
    fig.savefig(out/'figure6.png',dpi=250);fig.savefig(out/'figure6.svg');plt.close(fig)
    event(event='complete',smoke=a.smoke)

if __name__=='__main__':main()
