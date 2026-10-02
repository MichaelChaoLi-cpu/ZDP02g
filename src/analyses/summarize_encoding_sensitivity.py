"""Summarize the completed paired encoding experiment without refitting models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/exp/revision_experiments/categorical_seed42_20260928'
OUT = ROOT / 'data/results/revision_encoding_sensitivity'


def main():
    validation = json.loads((SOURCE / 'comparison_validation.json').read_text())
    assert all(validation[k] for k in ('same_inputs', 'same_yaml', 'same_fold_assignments'))
    data = pd.read_csv(SOURCE / 'encoding_comparison.csv', dtype={'country': str})
    numeric = data[data.specification == 'numeric'].set_index('country')
    onehot = data[data.specification == 'categorical'].set_index('country')
    assert numeric.index.is_unique and numeric.index.equals(onehot.index)
    assert numeric.rows.equals(onehot.rows)
    metrics = [c for c in numeric.columns if c not in ('specification', 'rows')]
    assert np.isfinite(numeric[metrics].to_numpy()).all()
    assert np.isfinite(onehot[metrics].to_numpy()).all()
    rows = []
    for country in numeric.index:
        for metric in metrics:
            a, b = float(numeric.at[country, metric]), float(onehot.at[country, metric])
            rows.append(dict(country=country, n=int(numeric.at[country, 'rows']),
                             metric=metric, numeric=a, one_hot=b,
                             difference=b-a, sign_reversal=bool(a*b < 0)))
    paired = pd.DataFrame(rows)
    countries = paired[paired.country != 'all']
    summary = countries.groupby('metric').agg(
        countries=('country', 'size'), sign_reversals=('sign_reversal', 'sum'),
        median_difference=('difference', 'median'),
        max_absolute_difference=('difference', lambda x: x.abs().max()))
    OUT.mkdir(parents=True, exist_ok=True)
    paired.to_csv(OUT / 'paired_encoding_results.csv', index=False)
    summary.to_csv(OUT / 'country_difference_summary.csv')
    countries[countries.sign_reversal].to_csv(OUT / 'country_sign_reversals.csv', index=False)
    pooled = paired[paired.country == 'all']
    lines = ['# Categorical-encoding sensitivity analysis', '',
             'Primary analysis: numeric encoding. Sensitivity: training-fold-fitted one-hot encoding of COUNTRY, EMPLOYMENT, MARITAL_STATUS, and GENDER.', '',
             'Both specifications use 273,031 observations, split seed 42, 10 outer folds, 5 inner folds, and the same YAML hyperparameters. This is a paired descriptive point comparison, not an equivalence test. The fixed column-subsampling fraction acts on different feature representations. Primary-model bootstrap intervals do not apply to the sensitivity model.', '',
             '| Metric | Numeric primary | One-hot sensitivity | Difference |',
             '|---|---:|---:|---:|']
    for r in pooled.itertuples():
        lines.append(f'| {r.metric} | {r.numeric:.8f} | {r.one_hot:.8f} | {r.difference:.8f} |')
    lines += ['', 'Predictive performance is nearly unchanged under one-hot encoding. Although pooled prediction contrasts differ little in absolute magnitude, some country-level contrasts change sign, indicating sensitivity in their directional interpretation. Sign reversals below are descriptive and are not tests of statistically significant changes.', '',
              '| Contrast | Countries with sign reversal | Maximum absolute difference |',
              '|---|---:|---:|']
    for metric, row in summary.iterrows():
        if metric not in ('rmse', 'mae', 'r2'):
            lines.append(f'| {metric} | {int(row.sign_reversals)} / {int(row.countries)} | {row.max_absolute_difference:.8f} |')
    lines += ['', 'Country values are source codes, not inferred labels. No observations were dropped or recoded by this summary. No additional model fits or bootstrap runs were performed.']
    (OUT / 'report.md').write_text('\n'.join(lines)+'\n')
    print(pooled.to_string(index=False))
    print(summary.to_string())


if __name__ == '__main__':
    main()
