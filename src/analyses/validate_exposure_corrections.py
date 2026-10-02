"""Build separate exposure variants; preserve legacy inputs and export aggregate checks."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from netCDF4 import Dataset, num2date
from audit_temperature_provenance import ROOT, SCENARIOS, sha, nearest, summary


def main():
    output = ROOT / 'data/processed/revision_exposure_validation'
    evidence = ROOT / 'data/exp/revision_audit/temperature_provenance'
    consumed = {}

    def track(path):
        consumed[str(path.relative_to(ROOT))] = sha(path)
        return path

    survey = pd.read_parquet(track(ROOT / 'data/processed/GlobalFlourishingDataWithLonLit.parquet'))
    legacy = []
    for wave, year in [(1, 2023), (2, 2024)]:
        frame = pd.read_parquet(track(ROOT / f'data/processed/df_out_{year}.parquet'))
        frame['WAVE'] = float(wave)
        legacy.append(frame[['LATITUDE', 'LONGITUDE', 'WAVE', f'tas_mean_{year}', f'tas_std_{year}']].rename(
            columns={f'tas_mean_{year}': 'TAS_MEAN', f'tas_std_{year}': 'TAS_STD'}))
    merged = survey.merge(pd.concat(legacy, ignore_index=True), on=['LATITUDE', 'LONGITUDE', 'WAVE'], validate='many_to_one')
    sample = merged.loc[merged.drop(columns='ID').notna().all(axis=1)].copy()
    sample['analysis_row'] = sample.index
    raw = pd.read_csv(track(ROOT / 'data/raw/gfs_all-countries_wave2-with-midyear_sensitive.csv'),
                      usecols=['ID', 'DOI_ANNUAL_Y1', 'DOI_ANNUAL_Y2'], dtype=str)
    raw['ID'] = pd.to_numeric(raw['ID'], errors='raise')
    dates = []
    for wave in (1, 2):
        part = raw[['ID', f'DOI_ANNUAL_Y{wave}']].copy()
        part.columns = ['ID', 'date']
        part['INTERVIEW_YEAR'] = pd.to_datetime(part.pop('date').str.strip(), format='%m/%d/%Y').dt.year
        part['WAVE'] = float(wave)
        dates.append(part)
    sample = sample.merge(pd.concat(dates), on=['ID', 'WAVE'], how='left', validate='one_to_one', sort=False)
    assert len(sample) == 273031 and sample['INTERVIEW_YEAR'].notna().all()
    sample['WAVE_YEAR'] = sample['WAVE'].map({1: 2023, 2: 2024})
    locations = survey[['LATITUDE', 'LONGITUDE']].drop_duplicates().reset_index(drop=True)
    all_fields, metadata = [], []
    reference_grid = None
    for year in (2022, 2023, 2024):
        arrays = []
        for scenario in SCENARIOS:
            path = track(ROOT / f'data/raw/cmip6/tas_Amon_EC-Earth3-Veg_{scenario}_r1i1p1f1_gr_{year}01-{year}12.nc')
            with Dataset(path) as ds:
                assert ds['tas'].units == 'K'
                for name, expected in [('source_id', 'EC-Earth3-Veg'), ('experiment_id', scenario),
                                       ('variant_label', 'r1i1p1f1'), ('grid_label', 'gr'), ('frequency', 'mon')]:
                    assert getattr(ds, name) == expected, (name, path)
                time = ds['time']
                months = num2date(time[:], time.units, getattr(time, 'calendar', 'standard'))
                assert [(d.year, d.month) for d in months] == [(year, m) for m in range(1, 13)]
                lat, lon = np.asarray(ds['lat'][:]), np.asarray(ds['lon'][:])
                assert np.all(np.diff(lat) > 0)
                lon = (lon + 180) % 360 - 180
                order = np.argsort(lon)
                lat, lon = lat[::-1], lon[order]
                if reference_grid is None:
                    reference_grid = (lat.copy(), lon.copy())
                assert all(np.array_equal(a, b) for a, b in zip(reference_grid, (lat, lon)))
                values = ds['tas'][:]
                assert values.shape == (12, len(lat), len(lon))
                assert not np.ma.getmaskarray(values).any() and np.isfinite(values).all()
                arrays.append(np.asarray(values)[:, ::-1, :][:, :, order])
                metadata.append({'year': year, 'scenario': scenario, 'months': 12, 'units': 'K', 'grid_verified': True})
        monthly = (np.stack(arrays) - 273.15).mean(axis=0)
        means, stds = monthly.mean(axis=0), monthly.std(axis=0)
        i = nearest(lat, locations['LATITUDE'].to_numpy())
        distance = np.abs(locations['LONGITUDE'].to_numpy()[:, None] - lon[None, :])
        j = np.minimum(distance, 360 - distance).argmin(axis=1)
        field = locations.copy()
        field['YEAR'] = year
        field['TAS_MEAN'] = means[i, j]
        field['TAS_STD'] = stds[i, j]
        field['native_lat_idx'], field['native_lon_idx'] = i, j
        all_fields.append(field)
    fields = pd.concat(all_fields, ignore_index=True)
    for label, year_col in [('grid', 'WAVE_YEAR'), ('calendar', 'INTERVIEW_YEAR')]:
        linked = sample[['LATITUDE', 'LONGITUDE', year_col]].merge(fields,
            left_on=['LATITUDE', 'LONGITUDE', year_col], right_on=['LATITUDE', 'LONGITUDE', 'YEAR'],
            how='left', validate='many_to_one', sort=False)
        assert len(linked) == len(sample) and linked[['TAS_MEAN', 'TAS_STD']].notna().all().all()
        for metric in ('MEAN', 'STD'):
            sample[f'{label}_TAS_{metric}'] = linked[f'TAS_{metric}'].to_numpy()
    same_year = sample['WAVE_YEAR'].eq(sample['INTERVIEW_YEAR'])
    contrasts = {}
    for metric in ('MEAN', 'STD'):
        assert np.array_equal(sample.loc[same_year, f'grid_TAS_{metric}'], sample.loc[same_year, f'calendar_TAS_{metric}'])
        contrasts[metric] = {
            'grid_minus_legacy_all_rows': summary(sample[f'grid_TAS_{metric}'] - sample[f'TAS_{metric}']),
            'calendar_minus_grid_all_rows': summary(sample[f'calendar_TAS_{metric}'] - sample[f'grid_TAS_{metric}']),
            'calendar_minus_grid_year_mismatch_rows': summary((sample[f'calendar_TAS_{metric}'] - sample[f'grid_TAS_{metric}'])[~same_year]),
        }
    assert all(sha(ROOT / name) == digest for name, digest in consumed.items())
    output.mkdir(parents=True, exist_ok=True)
    fields.to_parquet(output / 'native_grid_location_year.parquet', index=False)
    # Row-index linkage permits reproducible use without exporting IDs or survey dates.
    cols = ['analysis_row', 'WAVE_YEAR', 'INTERVIEW_YEAR', 'TAS_MEAN', 'TAS_STD',
            'grid_TAS_MEAN', 'grid_TAS_STD', 'calendar_TAS_MEAN', 'calendar_TAS_STD']
    sample[cols].to_parquet(output / 'fixed_sample_exposure_variants.parquet', index=False)
    for name, expected in [('native_grid_location_year.parquet', fields), ('fixed_sample_exposure_variants.parquet', sample[cols])]:
        pd.testing.assert_frame_equal(pd.read_parquet(output / name), expected.reset_index(drop=True))
    report = {'rows': len(sample), 'unique_people': sample['ID'].nunique(), 'year_mismatch_rows': int((~same_year).sum()),
              'source_metadata': metadata, 'sample_preserved': True, 'same_year_temporal_difference_exactly_zero': True,
              'contrasts_record_weighted': contrasts, 'inputs_unchanged': True, 'input_sha256': consumed,
              'output_sha256': {p.name: sha(p) for p in output.glob('*.parquet')},
              'scope': 'Exposure validation only; no outcome models fitted. Calendar years are a sensitivity definition, not observed-weather validation or a proven causal exposure window.'}
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'exposure_correction_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['source_metadata', 'input_sha256', 'output_sha256']}, indent=2))


if __name__ == '__main__':
    main()
