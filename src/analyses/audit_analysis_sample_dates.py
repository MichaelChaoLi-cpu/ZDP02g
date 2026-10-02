"""Reconstruct the existing complete-case sample; export aggregate diagnostics only."""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    paths = {
        'raw': ROOT / 'data/raw/gfs_all-countries_wave2-with-midyear_sensitive.csv',
        'survey': ROOT / 'data/processed/GlobalFlourishingDataWithLonLit.parquet',
        '2023': ROOT / 'data/processed/df_out_2023.parquet',
        '2024': ROOT / 'data/processed/df_out_2024.parquet',
        'prediction': ROOT / 'data/exp/model_outputs/prediction_PHYSICAL_HLTH.parquet',
    }
    before = {key: sha256(path) for key, path in paths.items()}
    survey = pd.read_parquet(paths['survey'])
    exposures = []
    for wave, year in [(1, 2023), (2, 2024)]:
        frame = pd.read_parquet(paths[str(year)])
        frame['WAVE'] = float(wave)
        frame = frame[['LATITUDE', 'LONGITUDE', 'WAVE', f'tas_mean_{year}', f'tas_std_{year}']]
        frame.columns = ['LATITUDE', 'LONGITUDE', 'WAVE', 'TAS_MEAN', 'TAS_STD']
        exposures.append(frame)
    merged = survey.merge(pd.concat(exposures, ignore_index=True),
                          on=['LATITUDE', 'LONGITUDE', 'WAVE'], validate='many_to_one')
    sample = merged.loc[merged.drop(columns='ID').notna().all(axis=1)].copy()
    # Compare to the production loader, not merely an independently assumed N.
    sys.path.insert(0, str(ROOT / 'src/analyses/notebooks'))
    import os
    from SettingForFeatures import data_load_combine_dataset
    os.chdir(ROOT)
    pd.testing.assert_frame_equal(sample.drop(columns='ID'), data_load_combine_dataset('tas'))
    assert sample['ID'].notna().all()
    assert not sample.duplicated(['ID', 'WAVE']).any()
    raw = pd.read_csv(paths['raw'], usecols=['ID', 'DOI_ANNUAL_Y1', 'DOI_ANNUAL_Y2'], dtype=str)
    raw['ID'] = pd.to_numeric(raw['ID'], errors='raise')
    assert raw['ID'].notna().all() and raw['ID'].is_unique
    assert (raw['ID'] % 1 == 0).all() and (raw['ID'].abs() < 2**53).all()
    dates = []
    for wave in (1, 2):
        part = raw[['ID', f'DOI_ANNUAL_Y{wave}']].copy()
        part.columns = ['ID', 'interview_date']
        part['interview_date'] = pd.to_datetime(part['interview_date'].str.strip(), format='%m/%d/%Y', errors='raise')
        part['WAVE'] = float(wave)
        dates.append(part)
    linked = sample[['ID', 'WAVE']].merge(pd.concat(dates, ignore_index=True),
                     on=['ID', 'WAVE'], how='left', validate='one_to_one', indicator=True)
    assert len(linked) == len(sample) and linked['_merge'].eq('both').all()
    linked['year'] = linked['interview_date'].dt.year
    assert linked['year'].notna().all()
    counts = linked.groupby('ID').size()
    prediction = pd.read_parquet(paths['prediction'])
    out = {
        'sample_flow': {'processed_survey_rows': len(survey), 'matched_rows': len(merged),
                        'complete_case_rows': len(sample), 'rows_removed_by_complete_case_filter': len(merged)-len(sample)},
        'production_loader_exact_frame_match': True,
        'saved_prediction_rows': len(prediction),
        'saved_prediction_index_equals_sample': prediction.index.equals(sample.index),
        'date_join_all_rows_matched': True, 'missing_final_sample_dates': 0,
        'wave_year_counts': [dict(wave=int(w), year=int(y), rows=int(n)) for (w, y), n in linked.groupby(['WAVE', 'year']).size().items()],
        'assigned_year_mismatch_rows': int(linked['year'].ne(linked['WAVE'].map({1: 2023, 2: 2024})).sum()),
        'unique_people': len(counts), 'people_in_both_waves': int(counts.eq(2).sum()),
        'people_in_one_wave': int(counts.eq(1).sum()),
        'rows_from_repeated_people': int(counts[counts.eq(2)].sum()),
        'duplicate_person_wave_rows': 0,
        'limitations': 'Dates are annual-interview end dates. Index equality alone does not prove prediction values or model provenance. Repeated-person counts establish dependence risk, not actual cross-fold leakage. No individual-level data exported.',
        'input_sha256': before,
    }
    assert {key: sha256(path) for key, path in paths.items()} == before
    out['inputs_unchanged'] = True
    target = ROOT / 'data/exp/revision_audit/temperature_provenance/analysis_sample_dates.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
