"""R2C6 fixed-split benchmarks; no search, perturbations, or bootstrap.

CPU XGBoost reproduces the primary inner-five ensemble on each outer test set.
Logistic is fitted on each full outer training set (a conventional, stronger
training-data baseline). The health spline uses the same XGBoost generated
mediators as the primary outcome model, never observed test mediators.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, log_loss,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from audit_temperature_provenance import ROOT, sha

CATEGORIES = ['COUNTRY', 'EMPLOYMENT', 'MARITAL_STATUS', 'GENDER',
              'OWN_RENT_HOME_Y1', 'URBAN_RURAL']
SMOOTH = ['INCOME_REAL', 'AGE', 'LATITUDE', 'LONGITUDE', 'TAS_MEAN', 'TAS_STD']


def transform(columns, spline=False):
    parts = [('categories', OneHotEncoder(drop='first', handle_unknown='ignore',
                                          sparse_output=False), CATEGORIES)]
    if spline:
        parts.append(('smooth', make_pipeline(SplineTransformer(n_knots=5,
                     degree=3, knots='quantile', include_bias=False,
                     extrapolation='linear'), StandardScaler()), SMOOTH))
    numeric = [c for c in columns if c not in CATEGORIES and (not spline or c not in SMOOTH)]
    parts.append(('numeric', StandardScaler(), numeric))
    return ColumnTransformer(parts, remainder='drop')


def save_frame(df, path):
    tmp = path.with_suffix('.tmp.parquet')
    df.to_parquet(tmp)
    tmp.replace(path)


def evaluate(frame, mediators, destination):
    rows, calibration = [], []
    for m in mediators:
        for model in ['prevalence', 'logistic', 'xgb']:
            key = f'{m}__{model}'
            if key not in frame:
                continue
            for label, group in [('all', frame)] + list(frame.groupby('fold')):
                y, p = group[f'{m}__y'], group[key]
                assert np.isfinite(p).all() and p.between(0, 1).all()
                rows.append(dict(outcome=m, model=model, fold=str(label), n=len(y),
                    auc=roc_auc_score(y,p), log_loss=log_loss(y,p,labels=[0,1]),
                    brier=brier_score_loss(y,p), accuracy=accuracy_score(y,p>=.5),
                    observed_prevalence=y.mean(), mean_probability=p.mean()))
            bins = np.minimum((frame[key].to_numpy()*10).astype(int), 9)
            for b in range(10):
                sub = frame.loc[bins==b]
                if len(sub):
                    calibration.append(dict(outcome=m, model=model, bin=b, n=len(sub),
                        mean_probability=sub[key].mean(), observed_rate=sub[f'{m}__y'].mean()))
    pd.DataFrame(rows).to_csv(destination/'probability_metrics.csv', index=False)
    pd.DataFrame(calibration).to_csv(destination/'calibration_bins.csv', index=False)
    health=[]
    for model in ['primary_xgb', 'spline_country_fe']:
        if model not in frame:
            continue
        for label, group in [('all',frame)]+list(frame.groupby('fold')):
            health.append(dict(model=model,fold=str(label),n=len(group),
                rmse=np.sqrt(mean_squared_error(group.health_y,group[model])),
                mae=mean_absolute_error(group.health_y,group[model]),
                r2=r2_score(group.health_y,group[model])))
    pd.DataFrame(health).to_csv(destination/'health_metrics.csv',index=False)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',default='data/exp/revision_benchmark_seed42_20260930')
    ap.add_argument('--stage',choices=['baselines','all'],default='all')
    ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--smoke',action='store_true',help='Separate small mechanics test, never research evidence')
    a=ap.parse_args()
    if a.threads<1: ap.error('threads must be positive')
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT/'src/analyses/notebooks'))
    from SettingForFeatures import data_load_combine_dataset, return_always_input_variable_list, return_aim_mediate
    data=data_load_combine_dataset('tas')
    variant_path=ROOT/'data/processed/revision_exposure_validation/fixed_sample_exposure_variants.parquet'
    variants=pd.read_parquet(variant_path).set_index('analysis_row')
    assert variants.index.equals(data.index) and len(data)==273031
    X=data[return_always_input_variable_list()].copy()
    for metric in ['MEAN','STD']: X[f'TAS_{metric}']=variants[f'grid_TAS_{metric}'].to_numpy()
    mediators=return_aim_mediate(); target=data.PHYSICAL_HLTH
    primary_path=ROOT/'data/exp/revision_experiments/fixed_yaml_20260926/grid/point/seed_42/predictions.parquet'
    primary=pd.read_parquet(primary_path)
    assert primary.index.equals(X.index)
    inputs=[variant_path,primary_path]+[ROOT/f'data/processed/{n}.parquet' for n in ['GlobalFlourishingDataWithLonLit','df_out_2023','df_out_2024']]
    cfg=ROOT/'src/analyses/config'
    inputs += [cfg/f'{m}_params.yaml' for m in mediators]
    params={m:{**yaml.safe_load((cfg/f'{m}_params.yaml').read_text()),'device':'cpu','n_jobs':a.threads} for m in mediators}
    source_manifest_path=ROOT/'data/exp/revision_experiments/fixed_yaml_20260926/manifest.json'
    source_manifest=json.loads(source_manifest_path.read_text())
    assert all(sha(ROOT/p)==h for p,h in source_manifest['input_sha256'].items())
    for m in mediators:
        assert params[m]=={**source_manifest['effective_parameters'][m],'n_jobs':a.threads}
    inputs.append(source_manifest_path)
    outer_n,inner_n=10,5
    if a.smoke:
        X=X.sample(800,random_state=42).sort_index();data=data.loc[X.index];target=target.loc[X.index];primary=primary.loc[X.index]
        outer_n,inner_n=2,2
        for p in params.values():p.update(n_estimators=3,max_depth=2)
    manifest=dict(decision='KILA-D-20260930-005',smoke=a.smoke,rows=len(X),seed=42,
        outer_folds=outer_n,inner_folds=inner_n,parameters=params,categories=CATEGORIES,
        spline_columns=SMOOTH,spline_knots=5,spline_degree=3,logistic='C=inf; lbfgs; tol=1e-6; max_iter=3000; outer training',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__).resolve(),ROOT/'src/analyses/notebooks/SettingForFeatures.py']},
        versions=dict(sklearn=sklearn.__version__,xgboost=xgboost.__version__,numpy=np.__version__,pandas=pd.__version__))
    mp=out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Manifest mismatch: use a new output'
    else:mp.write_text(json.dumps(manifest,indent=2)+'\n')
    def event(**kw):
        text=json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),**kw));print(text,flush=True)
        with (out/'events.jsonl').open('a') as f:f.write(text+'\n')
    frames=[]
    with threadpool_limits(limits=a.threads):
        for fold,(tr,te) in enumerate(KFold(outer_n,shuffle=True,random_state=42).split(X)):
            folder=out/f'fold_{fold:02d}';folder.mkdir(exist_ok=True)
            train,test=X.iloc[tr],X.iloc[te]
            if not a.smoke: assert (primary.loc[test.index,'fold']==fold).all()
            bp=folder/'baseline.parquet'
            if bp.exists():
                frame=pd.read_parquet(bp);assert frame.index.equals(test.index)
            else:
                frame=pd.DataFrame({'fold':fold,'health_y':target.loc[test.index],
                                    'primary_xgb':primary.loc[test.index,'baseline']},index=test.index)
                prep=transform(X.columns);xt=prep.fit_transform(train);xv=prep.transform(test)
                for m in mediators:
                    y=data.loc[train.index,m]
                    model=LogisticRegression(C=np.inf,solver='lbfgs',max_iter=3000,tol=1e-6)
                    with warnings.catch_warnings():
                        warnings.simplefilter('error',ConvergenceWarning);model.fit(xt,y)
                    assert model.n_iter_.max()<3000
                    frame[f'{m}__y']=data.loc[test.index,m]
                    frame[f'{m}__prevalence']=y.mean()
                    frame[f'{m}__logistic']=model.predict_proba(xv)[:,list(model.classes_).index(1)]
                    event(event='logistic_complete',fold=fold,mediator=m,iterations=int(model.n_iter_.max()))
                save_frame(frame,bp)
            if a.stage=='all':
                inner=pd.DataFrame(np.nan,index=train.index,columns=mediators)
                test_m=pd.DataFrame(0.,index=test.index,columns=mediators)
                for k,(it,iv) in enumerate(KFold(inner_n,shuffle=True,random_state=42).split(train)):
                    for m in mediators:
                        cp=folder/f'inner_{k}_{m}.parquet'
                        expected=train.iloc[iv].index.append(test.index)
                        if cp.exists():
                            pred=pd.read_parquet(cp);assert pred.index.equals(expected)
                        else:
                            model=xgboost.XGBClassifier(**params[m]);event(event='xgb_start',fold=fold,inner=k,mediator=m)
                            model.fit(train.iloc[it],data.loc[train.iloc[it].index,m])
                            pred=pd.DataFrame({'probability':model.predict_proba(pd.concat([train.iloc[iv],test]))[:,list(model.classes_).index(1)]},index=expected)
                            save_frame(pred,cp);event(event='xgb_complete',fold=fold,inner=k,mediator=m)
                        assert np.isfinite(pred.probability).all() and pred.probability.between(0,1).all()
                        inner.loc[train.iloc[iv].index,m]=pred.loc[train.iloc[iv].index,'probability']
                        test_m[m]+=pred.loc[test.index,'probability']/inner_n
                assert inner.notna().all().all()
                for m in mediators:frame[f'{m}__xgb']=test_m[m]
                htrain=train.join(inner);htest=test.join(test_m)
                health=make_pipeline(transform(htrain.columns,spline=True),LinearRegression())
                health.fit(htrain,target.loc[train.index]);frame['spline_country_fe']=health.predict(htest)
                save_frame(frame,folder/'complete.parquet')
            frames.append(frame);event(event='fold_complete',fold=fold,stage=a.stage)
    result=pd.concat(frames).loc[X.index]
    assert result.index.is_unique and result.index.equals(X.index) and np.isfinite(result.to_numpy()).all()
    assert all(sha(ROOT/p)==h for p,h in manifest['input_sha256'].items())
    save_frame(result,out/f'{a.stage}_predictions.parquet');evaluate(result,mediators,out)
    (out/f'{a.stage}_validated.json').write_text(json.dumps(dict(rows=len(result),smoke=a.smoke,inputs_unchanged=True,folds_match_primary=not a.smoke,uncertainty_computed=False),indent=2)+'\n')
    event(event='batch_complete',stage=a.stage,smoke=a.smoke)


if __name__=='__main__':main()
