"""Verify transferred data hashes and actual GPU execution before long experiments."""
import json
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from nested_grid_bootstrap import nested_predictions
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[2]


def main():
    manifest_path=ROOT/'transfer_data_manifest.json'
    if not manifest_path.exists():manifest_path=ROOT/'data/exp/sushi_sync/transfer_data_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    rng=np.random.default_rng(20260927)
    X=pd.DataFrame({'TAS_MEAN':rng.normal(size=512),'TAS_STD':rng.uniform(size=512),'other':rng.normal(size=512)})
    M=pd.DataFrame({m:rng.integers(0,2,len(X)) for m in ['a','b','c']})
    y=pd.Series(rng.normal(size=len(X)));counts=pd.Series(1,index=X.index)
    parameters=dict(n_estimators=4,max_depth=2,device='cuda:0',tree_method='hist',n_jobs=4,random_state=42)
    probe=XGBClassifier(**parameters);probe.fit(X,M.a)
    config=json.loads(probe.get_booster().save_config())
    device=config['learner']['generic_param']['device']
    assert device.startswith('cuda'),device
    result,trace=nested_predictions(X,M,y,counts,lambda m:XGBClassifier(**parameters),lambda:XGBRegressor(**parameters))
    zeros=[c for c in result if '_zero_' in c and c.endswith('_change')]
    assert np.max(np.abs(result[zeros].to_numpy()))==0
    report={'transfer_hashes_verified':len(manifest),'device':device,'synthetic_nested_smoke_rows':len(X),
            'fits':trace['fits'],'zero_max_abs':0.,'scope':'Deployment check, not research estimates.'}
    target=ROOT/'data/exp/sushi_sync/deployment_validation.json';target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
