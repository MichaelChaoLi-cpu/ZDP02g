"""Bounded XGBoost implementation check, not revised-paper effect estimation."""
import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
from paired_perturbation import Perturbation, paired_predictions, mediator_predictions
from audit_temperature_provenance import ROOT, sha


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / 'src/analyses/notebooks'))
    from SettingForFeatures import data_load_combine_dataset, return_always_input_variable_list, return_aim_mediate
    mediators = return_aim_mediate()
    folder = ROOT / 'data/exp/model_outputs'
    paths = [ROOT / f'data/processed/{n}.parquet' for n in ['GlobalFlourishingDataWithLonLit','df_out_2023','df_out_2024']]
    paths += [folder / f'prediction_{m}.parquet' for m in mediators]
    paths += [folder / f'prediction_{feature}_{suffix}_{m}.parquet' for feature,suffix in [('TAS_MEAN','increase1degree'),('TAS_STD','increase0x1std')] for m in mediators]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    data = data_load_combine_dataset('tas')
    positions = np.sort(np.random.default_rng(42).choice(len(data), size=4096, replace=False))
    sample = data.iloc[positions]
    X = sample[return_always_input_variable_list()+mediators].copy()
    y = sample['PHYSICAL_HLTH'].copy()

    def read_mediators(prefix):
        frames = []
        for m in mediators:
            df = pd.read_parquet(folder / f'{prefix}{m}.parquet')
            if not df.index.equals(data.index):
                raise ValueError('Saved mediator row order differs from production sample')
            frames.append(df[m])
        return pd.concat(frames, axis=1).iloc[positions].copy()

    baseline = read_mediators('prediction_')
    variants = []
    for feature, delta, suffix in [('TAS_MEAN',1.,'increase1degree'),('TAS_STD',0.1,'increase0x1std')]:
        variants += [Perturbation(f'{feature}_zero',feature,0.,baseline.copy()),
                     Perturbation(f'{feature}_positive',feature,delta,read_mediators(f'prediction_{feature}_{suffix}_'))]
    folds = list(KFold(n_splits=10,shuffle=True,random_state=42).split(X))
    # Independently exercise each fitted first-stage model at zero perturbation.
    first_stage_zero = {}
    train, test = folds[0]
    base_features = X.drop(columns=mediators)
    for m in mediators:
        first = XGBClassifier(n_estimators=8,max_depth=2,random_state=42,device='cpu',n_jobs=2)
        first.fit(base_features.iloc[train],sample[m].iloc[train])
        reference = first.predict_proba(base_features.iloc[test])[:,list(first.classes_).index(1)]
        for feature in ['TAS_MEAN','TAS_STD']:
            zero_prediction = mediator_predictions(first,base_features.iloc[test],feature,0.)
            first_stage_zero[f'{m}_{feature}'] = float(np.max(np.abs(reference-zero_prediction)))
    assert max(first_stage_zero.values()) == 0
    original_X, original_m = X.copy(deep=True), baseline.copy(deep=True)
    fitted = []
    params = dict(n_estimators=24,max_depth=3,learning_rate=0.1,random_state=42,tree_method='hist',device='cpu',n_jobs=2)
    def factory():
        model = XGBRegressor(**params)
        fitted.append(model)
        return model
    result = paired_predictions(X,y,baseline,variants,folds,factory)
    assert len(fitted)==10
    pd.testing.assert_frame_equal(X,original_X)
    pd.testing.assert_frame_equal(baseline,original_m)
    zero = result[[c for c in result if '_zero_' in c and c.endswith('_change')]]
    assert np.max(np.abs(zero.to_numpy()))==0
    rejected = {}
    for name, bad_m, bad_variants, bad_folds in [
        ('reordered_mediator_index', baseline.iloc[::-1], variants, folds),
        ('changed_zero_mediators', baseline, [Perturbation('zero','TAS_MEAN',0.,baseline*0.9)], folds),
        ('incomplete_validation_coverage', baseline, variants, folds[:-1]),
    ]:
        try:
            paired_predictions(X,y,bad_m,bad_variants,bad_folds,lambda: XGBRegressor(**{**params,'n_estimators':1}))
        except ValueError as error:
            rejected[name]=str(error)
        else:
            raise AssertionError(f'{name} was not rejected')
    assert all(sha(ROOT/name)==digest for name,digest in hashes.items())
    output = ROOT / 'data/exp/revision_audit/paired_perturbation';output.mkdir(parents=True,exist_ok=True)
    result.to_parquet(output/'diagnostic_predictions.parquet')
    pd.testing.assert_frame_equal(result,pd.read_parquet(output/'diagnostic_predictions.parquet'))
    report={'scope':'Implementation validation only: 4096 real sample rows, reduced 24-tree outcome models, cached legacy first-stage predictions; not paper results or revised-exposure robustness. Outcome test uses cached probabilities; separate fitted first-stage zero checks use three small classifiers. This is component validation, not full nested two-stage estimation.',
            'rows':len(X),'folds':10,'repeat_seed':42,'outcome_model_fits':len(fitted),
            'zero_delta_max_abs_change':float(np.max(np.abs(zero.to_numpy()))),
            'negative_checks_rejected':rejected,'first_stage_zero_checks':first_stage_zero,'feature_and_mediator_inputs_unchanged':True,
            'parameters':params,'source_hashes_unchanged':True,'input_sha256':hashes,
            'source_sha256':{n:sha(ROOT/'src/analyses'/n) for n in ['paired_perturbation.py','validate_paired_perturbation.py']}}
    (output/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['input_sha256','source_sha256']},indent=2))


if __name__=='__main__':
    main()
