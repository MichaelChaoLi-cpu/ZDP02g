"""Map observed-location seed-42 prediction contrasts with one shared scale."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
import shapefile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--boundary-root', type=Path,
        default=ROOT / 'data/processed/map_boundaries/natural_earth')
    args = parser.parse_args()
    source = ROOT / 'data/exp/revision_uncertainty_audit'
    output = ROOT / 'data/results/revision_spatial_contrasts'
    sys.path.insert(0, str(ROOT / 'src/analyses/notebooks'))
    from SettingForFeatures import data_load_combine_dataset
    data = data_load_combine_dataset('tas')
    path = source / 'point_predictions.parquet'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pred = pd.read_parquet(path)
    assert pred.index.equals(data.index) and len(pred) == 273031
    metrics = [f'{v}_positive_{k}_change' for k in ('direct', 'total') for v in ('mean', 'std')]
    assert np.isfinite(pred[metrics]).all().all()
    summaries = pd.read_csv(source / 'completed_run_summaries.csv', dtype={'country': str})
    point = summaries[summaries.run == 'point'].set_index('country')
    errors = []
    for country, indices in [('all', data.index)] + [(str(c), g.index) for c, g in data.groupby('COUNTRY')]:
        for metric in metrics:
            error = abs(pred.loc[indices, metric].to_numpy(dtype='float64').mean() - point.loc[country, metric])
            assert error < 1e-10
            errors.append(error)
    joined = data[['LATITUDE', 'LONGITUDE']].join(pred[metrics].astype('float64'))
    locations = joined.groupby(['LATITUDE', 'LONGITUDE'], sort=True)[metrics].mean().reset_index()
    locations['records'] = joined.groupby(['LATITUDE', 'LONGITUDE'], sort=True).size().to_numpy()
    maximum = float(np.abs(locations[metrics]).to_numpy().max())
    # Round outward to two significant figures; never clip estimated locations.
    step = 10 ** (np.floor(np.log10(maximum)) - 1)
    limit = float(np.ceil(maximum / step) * step)
    cmap = LinearSegmentedColormap.from_list('legacy_diverging', ['blue', 'green', 'white', 'yellow', 'red'])
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    segments = []
    boundary_hashes = {}
    for relative in ['physical/ne_110m_coastline.shp', 'cultural/ne_110m_admin_0_boundary_lines_land.shp']:
        boundary = args.boundary_root / relative
        boundary_hashes[relative] = hashlib.sha256(boundary.read_bytes()).hexdigest()
        with shapefile.Reader(str(boundary)) as reader:
            for shape in reader.shapes():
                indices = list(shape.parts) + [len(shape.points)]
                segments.extend(np.asarray(shape.points[a:b]) for a, b in zip(indices[:-1], indices[1:]))
    output.mkdir(parents=True, exist_ok=True)
    locations.to_csv(output / 'observed_location_contrasts.csv', index=False)
    for number, kind in [(7, 'direct'), (8, 'total')]:
        fig, axes = plt.subplots(2, 1, figsize=(12, 9), layout='constrained')
        for ax, exposure, label in zip(axes, ['mean', 'std'],
                ['(a) Annual mean temperature +1°C', '(b) Annual temperature SD +0.1°C']):
            ax.add_collection(LineCollection(segments, colors='#777777', linewidths=0.35, zorder=1))
            dots = ax.scatter(locations.LONGITUDE, locations.LATITUDE,
                c=locations[f'{exposure}_positive_{kind}_change'], cmap=cmap, norm=norm,
                marker='s', s=10, linewidths=0, zorder=2, rasterized=True)
            ax.set(xlim=(-180, 180), ylim=(-60, 85), xlabel='Longitude', ylabel='Latitude')
            ax.set_title(label, loc='left', fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(alpha=0.18, linewidth=0.4)
        fig.colorbar(dots, ax=axes, orientation='horizontal', fraction=0.045, pad=0.04,
                     label='Change in predicted physical-health score (0–10)')
        fig.suptitle(f'{kind.capitalize()} prediction contrasts at observed survey locations', fontsize=15)
        fig.supxlabel('Same color scale in Figures 7 and 8. Blank areas: no displayed observations.\n'
                      'Record-averaged point estimates; no spatial interpolation or significance masking.', fontsize=9)
        for extension in ['png', 'svg', 'pdf']:
            fig.savefig(output / f'figure{number}_common_scale.{extension}', dpi=300)
        plt.close(fig)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    report = dict(rows=len(pred), locations=len(locations), point_sha256=digest,
        max_country_summary_error=max(errors), common_limits=[-limit, limit],
        maximum_absolute_location_value=maximum, clipped_locations=0,
        aggregation='Arithmetic mean of held-out record contrasts at each observed LATITUDE/LONGITUDE; pooled waves.',
        spatial_support='Only observed locations; square marker size is visual and does not denote native-cell footprint.',
        boundaries=boundary_hashes, source_unchanged=True)
    (output / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
