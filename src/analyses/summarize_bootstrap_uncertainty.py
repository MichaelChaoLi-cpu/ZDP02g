"""Audit completed seed-42 draws and tabulate conditional percentile intervals.

No fitting. Run from the repository root with `uv run python
src/analyses/summarize_bootstrap_uncertainty.py`.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / 'data/exp/revision_uncertainty_audit'
    output = ROOT / 'data/results/revision_bootstrap_uncertainty'
    files = list(source.glob('*.json*')) + [source / 'completed_run_summaries.csv']
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    manifest = json.loads((source / 'manifest.json').read_text())
    completion = json.loads((source / 'completed.json').read_text())
    assert manifest['repeat_seeds'] == [42]
    assert (manifest['outer_folds'], manifest['inner_folds']) == (10, 5)
    assert completion['bootstrap_reps'] == 100 and completion['inputs_unchanged']
    assert completion['point_estimates_included']
    for name, digest in manifest['code_sha256'].items():
        assert sha(ROOT / 'src/analyses' / name) == digest, name
    for name, digest in manifest['input_sha256'].items():
        assert sha(ROOT / name) == digest, name

    events = [json.loads(line) for line in (source / 'events.jsonl').read_text().splitlines()]
    runs = ['point'] + [f'bootstrap_{i:04d}' for i in range(100)]
    for run in runs:
        folds = [e['fold'] for e in events if e.get('run') == run
                 and e['event'] in ('fold_complete', 'fold_restored')]
        assert sorted(folds) == list(range(10)), (run, folds)
        assert sum(e['event'] == 'run_complete' and e.get('run') == run for e in events) == 1
    data = pd.read_csv(source / 'completed_run_summaries.csv', dtype={'country': str})
    assert set(data.run) == set(runs) and set(data.variant) == {'grid'}
    assert not data.duplicated(['run', 'country']).any() and not data.isna().any().any()
    assert data.groupby('run').size().eq(23).all()
    assert data.groupby('country').size().eq(101).all()
    assert (data.weighted_rows > 0).all()
    zero = [c for c in data if '_zero_' in c]
    assert np.max(np.abs(data[zero].to_numpy())) == 0
    metrics = [f'{exposure}_positive_{kind}_change'
               for exposure in ('mean', 'std') for kind in ('direct', 'total')]
    max_weight_error = 0.0
    for run, group in data.groupby('run'):
        countries = group[group.country != 'all']
        pooled = group[group.country == 'all'].iloc[0]
        assert countries.weighted_rows.sum() == pooled.weighted_rows
        for metric in metrics:
            error = abs(np.average(countries[metric], weights=countries.weighted_rows) - pooled[metric])
            max_weight_error = max(max_weight_error, error)
            assert error < 1e-10
    for exposure in ('mean', 'std'):
        data[f'{exposure}_positive_indirect_change'] = (
            data[f'{exposure}_positive_total_change'] - data[f'{exposure}_positive_direct_change'])
    metrics += ['mean_positive_indirect_change', 'std_positive_indirect_change']
    sys.path.insert(0, str(ROOT / 'src/analyses/notebooks'))
    from SettingForFeatures import return_country_name_dict
    names = return_country_name_dict()
    rows = []
    for country, group in data.groupby('country'):
        point = group[group.run == 'point'].iloc[0]
        draws = group[group.run != 'point'].sort_values('run')
        for metric in metrics:
            values = draws[metric].to_numpy()
            assert len(values) == 100 and np.isfinite(values).all()
            low, high = np.quantile(values, [0.025, 0.975], method='linear')
            # Endpoint sensitivity diagnostic, not another confidence interval.
            loo = np.array([np.quantile(np.delete(values, i), [0.025, 0.975], method='linear')
                            for i in range(len(values))])
            rows.append(dict(country_id=country,
                country='Pooled sample' if country == 'all' else names[int(float(country))],
                metric=metric, point=float(point[metric]), bootstrap_mean=float(values.mean()),
                bootstrap_se=float(values.std(ddof=1)), lower_95=float(low), upper_95=float(high),
                contains_zero=bool(low <= 0 <= high), draws=100,
                records=int(point.weighted_rows),
                leave_one_out_lower_min=float(loo[:, 0].min()),
                leave_one_out_lower_max=float(loo[:, 0].max()),
                leave_one_out_upper_min=float(loo[:, 1].min()),
                leave_one_out_upper_max=float(loo[:, 1].max())))
    result = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / 'country_intervals.csv', index=False)
    display = result.copy()
    display['estimate_with_95_percentile_interval'] = display.apply(
        lambda row: f"{row.point:.5f} [{row.lower_95:.5f}, {row.upper_95:.5f}]", axis=1)
    display.pivot(index=['country_id', 'country'], columns='metric',
                  values='estimate_with_95_percentile_interval').to_csv(output / 'table2_candidate.csv')
    main_metrics = ['mean_positive_direct_change', 'mean_positive_total_change',
                    'std_positive_direct_change', 'std_positive_total_change']
    display.pivot(index=['country_id', 'country'], columns='metric',
                  values='estimate_with_95_percentile_interval')[main_metrics].to_csv(
                      output / 'table2_main_four_contrasts.csv')
    pooled = result[result.country_id == 'all']
    lines = ['# Conditional bootstrap uncertainty audit', '',
        'Review output; manuscript and Table 2 have not been replaced.', '',
        '100 country-stratified native-grid cluster draws; both prediction stages refitted; '
        'split seed 42 and YAML hyperparameters fixed. Point estimates use the matching seed-42 run.', '',
        'Perturbations: annual mean temperature +1°C; annual temperature standard deviation +0.1°C. '
        'Estimates are changes in predicted physical-health score (0–10), not standardized effect sizes.', '',
        '95% percentile endpoints use numpy linear quantiles at 0.025 and 0.975. Bootstrap SE is '
        'the sample SD of the 100 replicate statistics, not SD/sqrt(100). The indirect contrast is '
        'total minus direct within each draw, preserving their covariance.', '',
        '| Contrast | Point | Bootstrap SE | Lower | Upper |',
        '|---|---:|---:|---:|---:|']
    for row in pooled.itertuples():
        lines.append(f'| {row.metric} | {row.point:.6f} | {row.bootstrap_se:.6f} | {row.lower_95:.6f} | {row.upper_95:.6f} |')
    lines += ['', '## Country-level interval audit', '',
              '| Contrast | Intervals excluding zero / 22 | Countries (point-estimate sign) |',
              '|---|---:|---|']
    for metric, group in result[result.country_id != 'all'].groupby('metric'):
        selected = group[~group.contains_zero].sort_values('country')
        labels = ', '.join(f"{r.country} ({'+' if r.point > 0 else '−'})" for r in selected.itertuples())
        lines.append(f'| {metric} | {len(selected)} | {labels or "None"} |')
    lines += ['', '## Limits', '',
        'These are pointwise intervals conditional on fixed hyperparameters and fold assignments, '
        'not simultaneous country-wise intervals, causal-effect intervals, or uncertainty from tuning. '
        'The pooled statistic is record-weighted, not survey-weighted or an equal-country average. '
        'Single-grid clustering does not fully address respondents who cross grid/country boundaries. '
        '100 draws provide limited tail precision; leave-one-draw-out endpoint ranges are supplied '
        'as sensitivity diagnostics. Exclusion of zero is not a multiplicity-adjusted test. '
        'Aggregate completeness and fit-event coverage do not prove numerical convergence of every tree model. '
        'Figure 6 fold-SD shading is a separate descriptive quantity; no spatial pixel-wise intervals '
        'are inferred from these country summaries.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    assert all(sha(ROOT / name) == digest for name, digest in hashes.items())
    (output / 'validation.json').write_text(json.dumps(dict(
        source_hashes=hashes, source_inputs_match_manifest=True, source_code_matches_manifest=True,
        full_runs=101, bootstrap_runs=100, groups=23, intervals=len(result),
        fold_completion_or_restore_events=1010, zero_perturbation_exact=True,
        maximum_pooled_weight_identity_error=max_weight_error, source_files_unchanged=True,
        interval_method='percentile; numpy quantile method=linear; 2.5th–97.5th percentiles',
        status='validated aggregate review output; manuscript update pending'), indent=2) + '\n')
    print(pooled[['metric', 'point', 'bootstrap_se', 'lower_95', 'upper_95']].to_string(index=False))


if __name__ == '__main__':
    main()
