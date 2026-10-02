"""Read-only artifact consistency audit; never fit or deserialize models."""
import json
import os
import sys
import ast
from pathlib import Path
import numpy as np
import pandas as pd
from audit_temperature_provenance import ROOT, sha, summary


def main():
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / 'src/analyses/notebooks'))
    from SettingForFeatures import data_load_combine_dataset
    data = data_load_combine_dataset('tas')
    folder = ROOT / 'data/exp/model_outputs'
    inputs = list(folder.glob('*.parquet'))
    inputs += list((ROOT / 'src/analyses/nbs').glob('c*.py'))
    inputs += list((ROOT / 'src/analyses/config').glob('*.yaml'))
    inputs += [ROOT / 'src/analyses/notebooks/SettingForFeatures.py']
    inputs += [ROOT / f'data/processed/{name}.parquet' for name in ['GlobalFlourishingDataWithLonLit', 'df_out_2023', 'df_out_2024']]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    reports, frames = {}, {}
    for p in sorted(folder.glob('prediction_*.parquet')):
        df = pd.read_parquet(p)
        frames[p.name] = df
        target = next((t for t in ['PHYSICAL_HLTH', 'BODILY_PAIN', 'HEALTH_PROB', 'DAYS_EXERCISE'] if p.stem.endswith(t)), None)
        repetitions = [c for c in df.columns if str(c).isdigit()]
        reports[p.name] = {'rows': len(df), 'unique_index': df.index.is_unique,
            'index_equals_current_sample': df.index.equals(data.index),
            'finite_values': bool(np.isfinite(df.to_numpy(dtype=float)).all()),
            'repeat_columns': len(repetitions),
            'stored_mean_max_abs_error': float(np.abs(df[target]-df[repetitions].mean(axis=1)).max()) if target and repetitions else None}
    mediator = {}
    for target in ['BODILY_PAIN', 'HEALTH_PROB', 'DAYS_EXERCISE']:
        pred = frames[f'prediction_{target}.parquet'][target]
        assert pred.index.equals(data.index)
        mediator[target] = {'raw_values': sorted(data[target].unique().tolist()),
                            'prediction_range': [float(pred.min()), float(pred.max())],
                            'prediction_minus_raw': summary(pred.to_numpy()-data[target].to_numpy())}
    effects = {}
    baseline = frames['prediction_PHYSICAL_HLTH.parquet']
    for variable, suffix in [('TAS_MEAN', 'increase1degree'), ('TAS_STD', 'increase0x1std')]:
        for kind, infix in [('direct', ''), ('total', 'TotalEffect_')]:
            pred = frames[f'prediction_{variable}_{infix}{suffix}_PHYSICAL_HLTH.parquet']
            saved = pd.read_parquet(folder / f'{kind}_effect_of_{variable}.parquet')
            assert baseline.index.equals(pred.index)
            difference = pred['PHYSICAL_HLTH'].to_numpy()-baseline['PHYSICAL_HLTH'].to_numpy()
            effects[f'{kind}_{variable}'] = {'rows': len(saved), 'index_equals_sample': saved.index.equals(data.index),
                 'coordinates_equal_sample': np.array_equal(saved[['LATITUDE','LONGITUDE']].to_numpy(),data[['LATITUDE','LONGITUDE']].to_numpy()),
                 'saved_effect_max_abs_error': float(np.max(np.abs(saved['effect'].to_numpy()-difference))),
                 'difference': summary(difference)}
    p = next((ROOT / 'src/analyses/nbs').glob('c10_*.py'))
    tree = ast.parse(p.read_text())
    stores = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    assert 'mediate_variables' in loads and 'mediate_variables' not in stores
    inventory = [str(p.relative_to(ROOT)) for p in (ROOT / 'data/exp').rglob('*') if p.is_file() and p.suffix.lower() in {'.pkl','.pickle','.joblib','.ubj','.model','.bst'}]
    assert all(sha(ROOT / name)==value for name,value in hashes.items())
    result = {'predictions': reports, 'mediator_training_input_contrast': mediator, 'saved_effect_arithmetic': effects,
       'c10_undefined_mediate_variables': True, 'serialized_model_candidates_under_data_exp': inventory,
       'zero_perturbation_status': 'not numerically verified: inspected pipeline saves positive perturbations only and does not save fitted fold models; baseline and total branches train on different mediator inputs. Identity is not guaranteed. No outcome models fitted.',
       'input_sha256': hashes, 'inputs_unchanged': True}
    out = ROOT / 'data/exp/revision_audit/mediation_consistency';out.mkdir(parents=True,exist_ok=True)
    (out / 'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))


if __name__ == '__main__':
    main()
