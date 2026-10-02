"""Compare completed categorical and numeric seed42 points; no causal tests or CI."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/analyses/notebooks'))
from SettingForFeatures import data_load_combine_dataset

def main():
    import os
    os.chdir(ROOT)
    new=ROOT/'data/exp/revision_experiments/categorical_seed42_20260928'
    old=ROOT/'data/exp/revision_experiments/fixed_yaml_20260926'
    if not (new/'completed.json').exists():raise SystemExit('Categorical point is not complete; no comparison written.')
    nm=json.loads((new/'manifest.json').read_text());om=json.loads((old/'manifest.json').read_text())
    assert nm['input_sha256']==om['input_sha256'] and nm['effective_parameters']==om['effective_parameters']
    assert nm['outer_folds']==om['outer_folds']==10 and nm['inner_folds']==om['inner_folds']==5
    d=data_load_combine_dataset('tas')
    frames={label:pd.read_parquet(folder/'grid/point/seed_42/predictions.parquet') for label,folder in [('numeric',old),('categorical',new)]}
    assert all(p.index.equals(d.index) for p in frames.values())
    assert np.array_equal(frames['numeric']['fold'],frames['categorical']['fold'])
    rows=[]
    for label,p in frames.items():
        for country,idx in [('all',d.index)]+[(str(c),g.index) for c,g in d.groupby('COUNTRY')]:
            y=d.loc[idx,'PHYSICAL_HLTH'];pred=p.loc[idx,'baseline']
            r=dict(specification=label,country=country,rows=len(idx),rmse=float(np.sqrt(mean_squared_error(y,pred))),mae=float(mean_absolute_error(y,pred)),r2=float(r2_score(y,pred)))
            for metric in ['mean_positive','std_positive']:
                for term in ['direct_change','total_change']:r[metric+'_'+term]=float(p.loc[idx,metric+'_'+term].mean())
                r[metric+'_joint_difference']=r[metric+'_total_change']-r[metric+'_direct_change']
            rows.append(r)
    pd.DataFrame(rows).to_csv(new/'encoding_comparison.csv',index=False)
    (new/'comparison_validation.json').write_text(json.dumps(dict(same_inputs=True,same_yaml=True,same_fold_assignments=True,rows=len(d),scope='Four-variable one-hot sensitivity, fixed column-subsampling fraction; changes feature representation and the effective sampled feature sets. Descriptive point comparison only; no transfer of numeric bootstrap intervals.'),indent=2)+'\n')
    print(pd.DataFrame(rows).query("country == 'all'").to_string(index=False))
if __name__=='__main__':main()
