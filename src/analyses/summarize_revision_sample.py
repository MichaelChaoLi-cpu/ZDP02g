"""Reproduce R2C14 descriptive candidates from the frozen analytical sample.
Run from any directory: uv run python src/analyses/summarize_revision_sample.py.
Exports aggregate tables only; never changes inputs or fits models.
"""
from pathlib import Path
import os
import sys
import json
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src/analyses/notebooks'))
from SettingForFeatures import data_load_combine_dataset, return_country_name_dict


def main():
    os.chdir(ROOT)
    inputs = [Path('data/processed') / (n + '.parquet') for n in
              ['GlobalFlourishingDataWithLonLit', 'df_out_2023', 'df_out_2024']]
    inputs += [Path('data/processed/revision_exposure_validation/fixed_sample_exposure_variants.parquet')]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {str(p): sha(p) for p in inputs}
    df = data_load_combine_dataset('tas')
    variants = pd.read_parquet(inputs[-1]).set_index('analysis_row')
    assert len(df) == 273031 and df.index.equals(variants.index)
    for metric in ['MEAN', 'STD']:
        df['TAS_' + metric] = variants['grid_TAS_' + metric].to_numpy()
    cats = ['COUNTRY', 'WAVE', 'GENDER', 'MARITAL_STATUS', 'EMPLOYMENT',
            'OWN_RENT_HOME_Y1', 'URBAN_RURAL', 'EDUCATION_3', 'HAVE_CHILD',
            'CLOSE_TO', 'BODILY_PAIN', 'HEALTH_PROB', 'DAYS_EXERCISE']
    continuous = ['PHYSICAL_HLTH', 'INCOME_REAL', 'LATITUDE', 'LONGITUDE', 'AGE',
                  'NUM_HOUSEHOLD_Y1', 'EXPENSES', 'TAS_MEAN', 'TAS_STD']
    assert not df[cats + continuous].isna().any().any()
    out = ROOT / 'data/results/revision_sample_descriptives'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for col in cats:
        counts = df[col].value_counts().sort_index()
        assert counts.sum() == len(df)
        for code, n in counts.items():
            rows.append(dict(variable=col, code=int(code), n=int(n), percent=100*n/len(df)))
    pd.DataFrame(rows).to_csv(out / 'categorical_frequencies.csv', index=False)
    cw = df.groupby(['COUNTRY', 'WAVE']).size().unstack(fill_value=0).astype(int)
    prior = pd.read_csv('data/exp/revision_audit/sample_flow/final_country_wave.csv').pivot(index='COUNTRY', columns='WAVE', values='records').fillna(0).astype(int)
    pd.testing.assert_frame_equal(cw, prior)
    cw.columns = ['Wave 1', 'Wave 2']; cw['Total'] = cw.sum(axis=1)
    cw.insert(0, 'Country', [return_country_name_dict()[int(c)] for c in cw.index])
    cw.to_csv(out / 'country_wave_counts.csv', index_label='Country code')
    df[continuous].describe().T.to_csv(out / 'continuous_summary.csv', index_label='Variable')
    assert all(sha(Path(p)) == h for p, h in hashes.items())
    report = dict(records=len(df), countries=len(cw), wave_records=df.WAVE.value_counts().sort_index().astype(int).to_dict(),
                  previous_sample_flow_exact_match=True, categorical_totals_verified=True,
                  employment_code8=int((df.EMPLOYMENT == 8).sum()),
                  units='Unweighted observation records, not unique respondents; both waves pooled.',
                  category_labels='Stored codes only; interpret labels using verified codebook before manuscript insertion.',
                  source_hashes=hashes, inputs_unchanged=True)
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_hashes'}, indent=2))


if __name__ == '__main__':
    main()
