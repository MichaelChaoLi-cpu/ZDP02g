"""Compare GPU-native predictions with the existing path on a full-size fold.

Writes diagnostics only; never changes production checkpoints or model parameters.
"""
import argparse
import hashlib
from datetime import datetime
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold
from xgboost import XGBClassifier, XGBRegressor
from gpu_prediction import GPUPredictionModel
from nested_grid_bootstrap import nested_predictions, draw_grid_multiplicities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-fold', action='store_true')
    parser.add_argument('--reference-run', type=Path, help='Reuse an existing bootstrap-0/seed-42/fold-0 as the exact reference')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / 'src/analyses/notebooks'))
    from SettingForFeatures import data_load_combine_dataset, return_always_input_variable_list, return_aim_mediate
    data = data_load_combine_dataset('tas')
    variants = pd.read_parquet(root / 'data/processed/revision_exposure_validation/fixed_sample_exposure_variants.parquet').set_index('analysis_row')
    assert variants.index.equals(data.index)
    X = data[return_always_input_variable_list()].copy()
    for metric in ['MEAN', 'STD']:
        X[f'TAS_{metric}'] = variants[f'grid_TAS_{metric}'].to_numpy()
    M = data[return_aim_mediate()]
    cfg = root / 'src/analyses/config'
    params = {m: {**yaml.safe_load((cfg / f'{m}_params.yaml').read_text()), 'device': 'cuda:0', 'n_jobs': 4} for m in M}
    params['outcome'] = {**yaml.safe_load((cfg / 'params.yaml').read_text()), 'device': 'cuda:0', 'n_jobs': 4}
    destination = root / 'data/exp/algorithm_optimization'
    destination.mkdir(parents=True, exist_ok=True)
    if not args.full_fold:
        train, test = next(KFold(10, shuffle=True, random_state=42).split(X))
        it, iv = next(KFold(5, shuffle=True, random_state=42).split(train))
        name = M.columns[0]
        model = GPUPredictionModel(XGBClassifier(**params[name]))
        start = time.perf_counter()
        model.fit(X.iloc[train[it]], M[name].iloc[train[it]])
        fit_seconds = time.perf_counter() - start
        batch = X.iloc[test]
        records = []
        for repeat in range(5):
            row = {}
            outputs = {}
            order = ['existing', 'gpu_native'] if repeat % 2 == 0 else ['gpu_native', 'existing']
            for mode in order:
                start = time.perf_counter()
                outputs[mode] = (model.model if mode == 'existing' else model).predict_proba(batch)
                row[mode] = time.perf_counter() - start
            row['max_abs_difference'] = float(np.max(np.abs(outputs['existing'] - outputs['gpu_native'])))
            np.testing.assert_array_equal(outputs['existing'], outputs['gpu_native'])
            records.append(row)
        try:
            model.predict_proba(batch[batch.columns[::-1]])
        except ValueError:
            pass
        else:
            raise AssertionError('Reordered columns must be rejected')
        report = {'fit_seconds': fit_seconds, 'training_rows': len(it), 'prediction_rows': len(test), 'timings': records, 'exact_equal': True, 'parameters': params[name]}
        (destination / 'prediction_benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
        return
    cells = pd.read_parquet(root / 'data/processed/revision_exposure_validation/native_grid_location_year.parquet')
    cells = cells.loc[cells.YEAR.eq(2023), ['LATITUDE', 'LONGITUDE', 'native_lat_idx', 'native_lon_idx']]
    metadata = data[['COUNTRY', 'LATITUDE', 'LONGITUDE']].merge(cells, on=['LATITUDE', 'LONGITUDE'], how='left', sort=False, validate='many_to_one')
    metadata.index = data.index
    counts = draw_grid_multiplicities(metadata, 2026092600)[0]
    class FoldFinished(Exception):
        pass
    def progress(event):
        print(json.dumps(event), flush=True)
        if event['event'] == 'fold_complete':
            raise FoldFinished
    results = {}
    timings = {}
    modes = ['existing', 'gpu_native']
    if args.reference_run:
        reference = args.reference_run
        manifest = json.loads((reference / 'manifest.json').read_text())
        assert manifest['effective_parameters'] == params
        assert manifest['outer_folds'] == 10 and manifest['inner_folds'] == 5
        assert manifest['repeat_seeds'] == list(range(42, 52))
        assert manifest['rows'] == len(X)
        for name, digest in manifest['input_sha256'].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name in ['nested_grid_bootstrap.py', 'paired_perturbation.py']:
            assert hashlib.sha256((root / 'src/analyses' / name).read_bytes()).hexdigest() == manifest['code_sha256'][name]
        events = [json.loads(line) for line in (reference / 'events.jsonl').read_text().splitlines()]
        events = [e for e in events if e.get('run') == 'bootstrap_0000' and e.get('seed') == 42 and e.get('fold') == 0]
        start = next(e['time'] for e in events if e['event'] == 'fold_start')
        end = next(e['time'] for e in events if e['event'] == 'fold_complete')
        timings['existing'] = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
        results['existing'] = pd.read_parquet(reference / 'grid/bootstrap_0000/seed_42/folds/fold_00.parquet')
        modes = ['gpu_native']
    for mode in modes:
        folder = destination / f'full_fold_{mode}'
        if folder.exists():
            raise ValueError(f'Use fresh benchmark folders: {folder}')
        def wrap(model):
            return GPUPredictionModel(model) if mode == 'gpu_native' else model
        start = time.perf_counter()
        try:
            nested_predictions(X, M, data.PHYSICAL_HLTH, counts,
                lambda m: wrap(XGBClassifier(**params[m])),
                lambda: wrap(XGBRegressor(**params['outcome'])),
                checkpoint_dir=folder, on_event=progress)
        except FoldFinished:
            pass
        timings[mode] = time.perf_counter() - start
        results[mode] = pd.read_parquet(folder / 'fold_00.parquet')
    pd.testing.assert_frame_equal(results['existing'], results['gpu_native'], check_exact=True)
    report = {'seconds': timings, 'speedup': timings['existing'] / timings['gpu_native'], 'exact_equal': True,
              'sample_rows': len(X), 'bootstrap': 0, 'seed': 42, 'outer_fold': 0, 'fits_per_path': 16,
              'parameters': params, 'reference_run': str(args.reference_run),
              'scope': 'One full-size bootstrap fold; compare timing only with no concurrent production.'}
    (destination / 'full_fold_benchmark.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
