"""Nested random-fold two-stage fitting and country-stratified grid resampling.

Draw multiplicities are expanded only inside fitting calls. Duplicate copies of an
original row always share its validation assignment. This is not causal identification.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from paired_perturbation import Perturbation, mediator_predictions, evaluate_paired_fold


def draw_grid_multiplicities(metadata, seed):
    required=['COUNTRY','native_lat_idx','native_lon_idx']
    if not metadata.index.is_unique or metadata[required].isna().any().any():
        raise ValueError('Unique row index and complete country/grid metadata required')
    rng=np.random.default_rng(seed)
    weights=pd.Series(0,index=metadata.index,dtype='int64')
    diagnostics=[]
    for country, group in metadata.groupby('COUNTRY',sort=True):
        keys=pd.MultiIndex.from_frame(group[['native_lat_idx','native_lon_idx']])
        codes,unique=pd.factorize(keys,sort=True)
        counts=np.bincount(rng.integers(0,len(unique),size=len(unique)),minlength=len(unique))
        weights.loc[group.index]=counts[codes]
        assert int(counts.sum())==len(unique)
        diagnostics.append({'country':float(country),'available_clusters':len(unique),
                            'cluster_draws':int(counts.sum()),'distinct_drawn':int((counts>0).sum())})
    assert weights.gt(0).any()
    return weights,diagnostics


def nested_predictions(features, mediators, target, counts, classifier_factory, regressor_factory,
                       repeat_seed=42, outer_folds=10, inner_folds=5, checkpoint_dir=None, on_event=None):
    """One repeat, all original rows evaluated, both stages refit on bootstrap copies.

    Factory signatures: classifier_factory(mediator_name), regressor_factory().
    Counts are frequency multiplicities, not survey weights. Aggregate returned
    contrasts using those same multiplicities for bootstrap statistics.
    """
    if not features.index.is_unique or len(features)<outer_folds:
        raise ValueError('Invalid feature index/sample size')
    for obj in [mediators,target,counts]:
        if not obj.index.equals(features.index):raise ValueError('Input index/order mismatch')
    if set(features.columns)&set(mediators.columns):raise ValueError('Mediators must be separate from base predictors')
    if not np.isfinite(features.to_numpy()).all() or not np.isfinite(target).all():raise ValueError('Nonfinite data')
    if not np.isfinite(counts).all() or (counts<0).any() or (counts%1!=0).any() or counts.sum()==0:raise ValueError('Invalid multiplicities')
    if not all(set(mediators[m].unique())=={0,1} for m in mediators):raise ValueError('Expected binary mediators')
    specs=[('mean_zero','TAS_MEAN',0.),('mean_positive','TAS_MEAN',1.),
           ('std_zero','TAS_STD',0.),('std_positive','TAS_STD',0.1)]
    checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir is not None else None
    if checkpoint_dir is not None:checkpoint_dir.mkdir(parents=True,exist_ok=True)
    restored_folds=0
    result=pd.DataFrame(index=features.index)
    seen=np.zeros(len(features),dtype=int)
    trace=[];fit_count=0;fold_digest=hashlib.sha256()
    for outer,(train,test) in enumerate(KFold(outer_folds,shuffle=True,random_state=repeat_seed).split(features)):
        assert not np.intersect1d(train,test).size
        fold_digest.update(np.asarray(test,dtype='<i8').tobytes())
        Xtrain=features.iloc[train];Xtest=features.iloc[test]
        cache=checkpoint_dir/f'fold_{outer:02d}.parquet' if checkpoint_dir is not None else None
        if cache is not None and cache.exists():
            saved=pd.read_parquet(cache)
            if not saved.index.equals(Xtest.index) or not np.isfinite(saved.to_numpy()).all():raise ValueError('Invalid fold checkpoint')
            for col in saved:result.loc[Xtest.index,col]=saved[col]
            seen[test]+=1;restored_folds+=1
            trace.append({'outer_fold':outer,'restored':True})
            if on_event:on_event({'event':'fold_restored','fold':outer})
            continue
        if on_event:on_event({'event':'fold_start','fold':outer})
        oof=pd.DataFrame(np.nan,index=Xtrain.index,columns=mediators.columns)
        test_m={name:pd.DataFrame(0.,index=Xtest.index,columns=mediators.columns) for name in ['baseline']+[s[0] for s in specs]}
        inner_seen=np.zeros(len(train),dtype=int)
        for inner,(it,iv) in enumerate(KFold(inner_folds,shuffle=True,random_state=repeat_seed).split(train)):
            global_train,global_val=train[it],train[iv]
            assert not np.intersect1d(global_train,test).size
            assert not np.intersect1d(global_train,global_val).size
            expanded=np.repeat(global_train,counts.iloc[global_train].to_numpy(dtype=int))
            if len(expanded)==0:raise ValueError('Empty bootstrap inner training sample')
            for m in mediators:
                if mediators[m].iloc[expanded].nunique()!=2:raise ValueError(f'One-class training sample: {m}')
                model=classifier_factory(m)
                model.fit(features.iloc[expanded],mediators[m].iloc[expanded]);fit_count+=1
                if on_event:on_event({'event':'mediator_fit','fold':outer,'inner_fold':inner,'mediator':m,'fits':fit_count})
                oof.loc[features.index[global_val],m]=mediator_predictions(model,features.iloc[global_val],'TAS_MEAN',0.)
                base=mediator_predictions(model,Xtest,'TAS_MEAN',0.)
                test_m['baseline'][m]+=base/inner_folds
                for name,feature,delta in specs:
                    prediction=mediator_predictions(model,Xtest,feature,delta)
                    if delta==0 and not np.array_equal(prediction,base):raise ValueError('First-stage zero check failed')
                    test_m[name][m]+=prediction/inner_folds
            inner_seen[iv]+=1
        assert (inner_seen==1).all() and oof.notna().all().all()
        train_out=Xtrain.join(oof);test_out=Xtest.join(test_m['baseline'])
        expanded_local=np.repeat(np.arange(len(train)),counts.iloc[train].to_numpy(dtype=int))
        outcome=regressor_factory()
        outcome.fit(train_out.iloc[expanded_local],target.iloc[train].iloc[expanded_local]);fit_count+=1
        variants=[Perturbation(name,feature,delta,test_m[name]) for name,feature,delta in specs]
        values=evaluate_paired_fold(outcome,test_out,variants,list(mediators.columns))
        for name,value in values.items():result.loc[Xtest.index,name]=value
        result.loc[Xtest.index,'fold']=outer
        trace.append({'outer_fold':outer,'unique_train_rows':len(train),'unique_test_rows':len(test),
                      'expanded_train_rows':len(expanded_local),'inner_oof_complete':True})
        seen[test]+=1
        if cache is not None:
            temporary=cache.with_suffix('.tmp.parquet')
            result.loc[Xtest.index].to_parquet(temporary)
            temporary.replace(cache)
        if on_event:on_event({'event':'fold_complete','fold':outer,'fits':fit_count})
    assert (seen==1).all() and np.isfinite(result.to_numpy()).all()
    return result,{'fits':fit_count,'restored_folds':restored_folds,'fold_assignment_sha256':fold_digest.hexdigest(),'folds':trace,
                   'outer_test_excluded_from_inner_fits':True,'bootstrap_copies_share_original_row_fold':True}
