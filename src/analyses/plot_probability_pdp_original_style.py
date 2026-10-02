"""Restore legacy Figure 6 styling using corrected pooled PDPs and LOWESS.

The local linear smooth uses tricube weights, frac=0.4, no robust iterations.
It is a visual guide; it does not refit outcome models or estimate uncertainty.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from audit_temperature_provenance import ROOT, sha


def lowess(x, y, frac=0.4):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = max(3, int(np.ceil(frac * len(x))))
    result = []
    for x0 in x:
        distance = np.abs(x - x0)
        bandwidth = np.partition(distance, n - 1)[n - 1]
        weights = np.maximum(0, 1 - (distance / bandwidth) ** 3) ** 3
        design = np.column_stack([np.ones(len(x)), x - x0])
        beta = np.linalg.lstsq(design * np.sqrt(weights[:, None]),
                               y * np.sqrt(weights), rcond=None)[0]
        result.append(beta[0])
    return np.array(result)


def main():
    source = ROOT / 'data/exp/revision_probability_pdp_20260930'
    out = ROOT / 'data/results/revision_figure6'
    out.mkdir(parents=True, exist_ok=True)
    validation = json.loads((source / 'validation.json').read_text())
    assert not validation['smoke'] and validation['inputs_unchanged']
    path = source / 'curves.csv'
    before = sha(path)
    data = pd.read_csv(path, dtype={'wave': str})
    component_path = source / 'component_curves.csv'
    component_sha = sha(component_path)
    components = pd.read_csv(component_path, dtype={'wave': str})
    pooled = components[components.wave.eq('pooled')]
    fold_keys = ['fold', 'outcome', 'feature', 'grid_index']
    folded = pooled.groupby(fold_keys, as_index=False).agg(prediction=('prediction', 'mean'), n=('n', 'first'))
    assert folded.groupby(['outcome','feature','grid_index']).size().eq(10).all()
    spread = folded.groupby(['outcome','feature','grid_index']).prediction.std(ddof=1).rename('fold_sd').reset_index()
    data = data.merge(spread, on=['outcome','feature','grid_index'], validate='many_to_one')
    settings = [
        ('PHYSICAL_HLTH', 'Overall Physical Health (0–10)', (6.75, 7.20), .10),
        ('BODILY_PAIN', 'Bodily Pain in Past 4 Weeks', (.42, .48), .01),
        ('HEALTH_PROB', 'Health Problems Restricting Activity', (.18, .25), .01),
        ('DAYS_EXERCISE', 'Exercise at Least Once a Week', (.64, .73), .02),
    ]
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(4, 2, figsize=(12, 16), sharey='row')
    smooths = []
    for i, (outcome, title, limits, tick) in enumerate(settings):
        for j, feature in enumerate(['TAS_MEAN', 'TAS_STD']):
            ax = axes[i, j]
            g = data[(data.outcome == outcome) & (data.feature == feature)
                     & (data.wave == 'pooled')].sort_values('value')
            assert len(g) == 41 and g.prediction.between(*limits).all()
            fitted = lowess(g.value, g.prediction)
            assert np.isfinite(fitted).all() and ((fitted >= limits[0]) & (fitted <= limits[1])).all()
            ax.plot(g.value, g.prediction, color='#1f77b4', lw=1.8, label='Mean Prediction')
            ax.fill_between(g.value, g.prediction - 1.96*g.fold_sd,
                            g.prediction + 1.96*g.fold_sd, color='#1f77b4', alpha=.25,
                            label='±1.96 fold SD')
            ax.plot(g.value, fitted, color='red', ls='--', lw=1.8, label='LOWESS Trend')
            ax.set_title(f'{"abcdefgh"[i*2+j]}: {title}', loc='left')
            lower = min(limits[0], float((g.prediction-1.96*g.fold_sd).min()))
            upper = max(limits[1], float((g.prediction+1.96*g.fold_sd).max()))
            old_limits = axes[i,0].get_ylim() if j else limits
            ax.set_ylim(min(lower,old_limits[0]), max(upper,old_limits[1]))
            ax.yaxis.set_major_locator(MultipleLocator(tick))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.tick_params(axis='y', labelleft=True)
            ax.set_ylabel('Predicted health score' if i == 0 else 'Predicted probability')
            ax.set_xlabel('Annual mean temperature (°C)' if j == 0 else 'Annual temperature SD (°C)')
            ax.grid(color='#b0b0b0', alpha=.75, linewidth=.8)
            ax.legend(loc='best', fontsize=9, frameon=True, fancybox=False)
            smooths.append(g.assign(lowess_prediction=fitted))
    fig.tight_layout(rect=(0, .035, 1, 1), h_pad=1.5, w_pad=2)
    fig.text(.5, .015, 'Pooled profiles; shading: ±1.96 between-fold SD (not a 95% CI). LOWESS: visual guide.\n'
             'Y-axes are truncated; scales differ across outcomes and match within each row.',
             ha='center', fontsize=10)
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(out / f'figure6_original_style.{ext}', dpi=300)
    plt.close(fig)
    pd.concat(smooths).to_csv(out / 'figure6_original_style_curves.csv', index=False)
    assert sha(path) == before and sha(component_path) == component_sha
    (out / 'figure6_original_style_caption.md').write_text(
        'Figure 6. Model-based partial-dependence profiles for annual mean temperature and annual temperature standard deviation. '
        'Blue lines average held-out predictions over pooled records. Red dashed lines are local linear LOWESS smooths '
        '(fraction 0.4, tricube weights, no robust iterations), shown only as visual guides. '
        'Binary outcomes show class-1 probabilities; physical health holds baseline generated mediator inputs fixed. '
        'Exposure grids span the pooled fifth–95th percentiles. Y-axes do not start at zero; scales differ across outcomes '
        'but are identical within each row. Shading is the pooled profile plus/minus 1.96 times the sample standard deviation across ten outer-fold profiles (after averaging inner models within each outer fold). It describes fold-to-fold variability, not a 95% confidence interval. Profiles are descriptive, not causal effects.\n')
    (out / 'figure6_original_style_validation.json').write_text(json.dumps({
        'decision': 'KILA-D-20260930-012', 'band': 'pooled mean +/- 1.96 between-outer-fold sample SD; not confidence interval', 'component_sha256': component_sha, 'source_sha256': before,
        'source_unchanged': True, 'outcome_models_refitted': False, 'panels': 8,
        'points_per_panel': 41, 'all_curves_within_axes': True,
        'smooth': {'method': 'local linear LOWESS', 'frac': .4, 'robust_iterations': 0},
        'reference_style': 'Rev/origin/originsrc/media/image6.jpg',
        'script_sha256': sha(Path(__file__)),
    }, indent=2) + '\n')
    print(out / 'figure6_original_style.png')


if __name__ == '__main__':
    main()
