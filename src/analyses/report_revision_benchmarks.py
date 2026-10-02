"""Validate and summarize the completed R2C6 held-out benchmarks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from audit_temperature_provenance import ROOT, sha
from run_revision_benchmarks import evaluate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default='data/exp/revision_benchmark_seed42_20260930')
    p.add_argument('--output',default='data/results/revision_benchmarks')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();source=ROOT/a.input;dest=ROOT/a.output
    manifest=json.loads((source/'manifest.json').read_text())
    assert manifest['smoke']==a.smoke
    assert (source/'all_validated.json').exists(), 'Incomplete benchmark run'
    assert all(sha(ROOT/n)==h for n,h in manifest['input_sha256'].items())
    assert all(sha(ROOT/n)==h for n,h in manifest['code_sha256'].items())
    data=pd.read_parquet(source/'all_predictions.parquet')
    assert len(data)==manifest['rows'] and data.index.is_unique and np.isfinite(data.to_numpy()).all()
    mediators=['BODILY_PAIN','HEALTH_PROB','DAYS_EXERCISE']
    for m in mediators:
        assert set(data[f'{m}__y'].unique())=={0,1}
        for model in ['prevalence','logistic','xgb']:
            assert data[f'{m}__{model}'].between(0,1).all()
    if not a.smoke:
        primary=pd.read_parquet(ROOT/'data/exp/revision_experiments/fixed_yaml_20260926/grid/point/seed_42/predictions.parquet')
        assert primary.index.equals(data.index)
        np.testing.assert_array_equal(primary['fold'],data['fold'])
        np.testing.assert_array_equal(primary['baseline'],data['primary_xgb'])
    dest.mkdir(parents=True,exist_ok=True)
    evaluate(data,mediators,dest)
    metrics=pd.read_csv(dest/'probability_metrics.csv',dtype={'fold':str})
    health=pd.read_csv(dest/'health_metrics.csv',dtype={'fold':str})
    calibration=pd.read_csv(dest/'calibration_bins.csv')
    for (_, _),g in calibration.groupby(['outcome','model']):assert g.n.sum()==len(data)
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    titles=['Bodily pain','Activity limitation','Weekly exercise']
    colors={'prevalence':'#777777','logistic':'#2878B5','xgb':'#D65F24'}
    for ax,m,title in zip(axes,mediators,titles):
        ax.plot([0,1],[0,1],':',color='black',lw=1,label='Perfect calibration')
        for model in colors:
            g=calibration.loc[(calibration.outcome==m)&(calibration.model==model)]
            ax.plot(g.mean_probability,g.observed_rate,'o-',color=colors[model],label=model,ms=4,lw=1)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed proportion',title=title)
        ax.set_aspect('equal');ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    fig.savefig(dest/'calibration.png',dpi=240);fig.savefig(dest/'calibration.svg');plt.close(fig)
    pooled=metrics.loc[metrics.fold.eq('all')]
    lines=['# Held-out benchmark evaluation','',
        'SMOKE TEST ONLY — not manuscript evidence.' if a.smoke else f'Analytical records: {len(data):,}; fixed split seed 42; 10 outer folds.',
        '', '| Outcome | Model | AUC | Log-loss | Brier | Accuracy | Observed prevalence | Mean probability |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in pooled.itertuples():
        lines.append(f'| {r.outcome} | {r.model} | {r.auc:.4f} | {r.log_loss:.4f} | {r.brier:.4f} | {r.accuracy:.4f} | {r.observed_prevalence:.4f} | {r.mean_probability:.4f} |')
    lines+=['','| Health model | RMSE | MAE | R-squared |','|---|---:|---:|---:|']
    for r in health.loc[health.fold.eq('all')].itertuples():lines.append(f'| {r.model} | {r.rmse:.4f} | {r.mae:.4f} | {r.r2:.4f} |')
    lines+=['','## Interpretation limits','',
        'All evaluations use held-out records. Random folds do not isolate respondents, locations or countries. No confidence interval or significance test is computed; differences are descriptive. Logistic uses all outer training records; XGBoost uses the primary five-inner-model ensemble. The spline health model uses the same generated mediator inputs as the primary pipeline. Prevalence predictions vary between outer folds, so their pooled AUC need not be exactly 0.5 even though each within-fold constant predictor has AUC 0.5.',
        '', 'Calibration uses ten fixed-width probability bins; empty bins are omitted and bin counts are supplied in calibration_bins.csv. Small tail bins may be unstable. Curves do not represent recalibrated models or partial dependence.',
        '', 'Specification: Rev/docs/reviewer-2-comment-6-benchmark-protocol.md.']
    (dest/'report.md').write_text('\n'.join(lines)+'\n')
    (dest/'validation.json').write_text(json.dumps(dict(rows=len(data),smoke=a.smoke,
        source_sha256=sha(source/'all_predictions.parquet'),script_sha256=sha(Path(__file__)),
        inputs_unchanged=True,primary_predictions_preserved=not a.smoke,calibration_counts_complete=True),indent=2)+'\n')
    print(dest/'report.md')


if __name__=='__main__':main()
