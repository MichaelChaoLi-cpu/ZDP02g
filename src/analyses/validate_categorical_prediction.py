"""Bounded tests of unseen categories and nested paired prediction compatibility."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier,XGBRegressor
from categorical_prediction import categorical_model,NOMINAL_COLUMNS
from nested_grid_bootstrap import nested_predictions

def main():
    rng=np.random.default_rng(42);n=120
    X=pd.DataFrame({c:rng.integers(1,4,n).astype(float) for c in NOMINAL_COLUMNS})
    X['TAS_MEAN']=rng.normal(size=n);X['TAS_STD']=rng.uniform(.1,2,n)
    y=pd.Series(np.tile([0,1],n//2))
    test=X.iloc[-10:].copy();test['EMPLOYMENT']=999
    model=categorical_model(LogisticRegression(max_iter=500));model.fit(X.iloc[:-10],y.iloc[:-10]);p=model.predict_proba(test)
    assert np.isfinite(p).all() and 999 not in model.named_steps['encoding'].named_transformers_['nominal'].categories_[1]
    assert len(model.named_steps['encoding'].get_feature_names_out())==14
    med=pd.DataFrame({m:np.roll(y.to_numpy(),i) for i,m in enumerate(['a','b','c'])})
    target=pd.Series(rng.normal(size=n));counts=pd.Series(1,index=X.index)
    result,trace=nested_predictions(X,med,target,counts,lambda m:categorical_model(XGBClassifier(n_estimators=2,max_depth=2,n_jobs=1)),lambda:categorical_model(XGBRegressor(n_estimators=2,max_depth=2,n_jobs=1)),outer_folds=3,inner_folds=2)
    assert trace['fits']==21 and trace['outer_test_excluded_from_inner_fits']
    assert (result.filter(regex='_zero_.*change$').to_numpy()==0).all()
    assert np.isfinite(result.to_numpy()).all()
    out=Path('data/exp/revision_audit/comment5_categorical');out.mkdir(parents=True,exist_ok=True)
    report=dict(rows=n,fits=trace['fits'],unseen_level_excluded_from_training_vocabulary=True,unseen_predictions_finite=True,zero_perturbations_exact=True,outer_test_excluded=True)
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
if __name__=='__main__':main()
