"""Compare validated full-sample point experiments; no inferential intervals."""
import json
import sys
import numpy as np
import pandas as pd
from audit_temperature_provenance import ROOT,sha


def main():
    sys.path.insert(0,str(ROOT/'src/analyses/notebooks'))
    from SettingForFeatures import return_country_name_dict
    base=ROOT/'data/exp/revision_experiments/fixed_yaml_20260926'
    out=base/'comparison';out.mkdir(exist_ok=True)
    metrics=['mean_positive_direct_change','mean_positive_total_change','std_positive_direct_change','std_positive_total_change']
    inputs=[base/'completed_run_summaries.csv',base/'completed.json',base/'manifest.json']
    inputs += [base/v/'point/mean_predictions.parquet' for v in ['grid','calendar']]
    inputs += [base/v/'point'/f'seed_{s}/predictions.parquet' for v in ['grid','calendar'] for s in range(42,52)]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in inputs}
    completion=json.loads((base/'completed.json').read_text());assert completion['inputs_unchanged']
    summaries=pd.read_csv(base/'completed_run_summaries.csv',dtype={'country':str})
    assert not summaries.duplicated(['variant','run','country']).any() and len(summaries)==46
    means={v:pd.read_parquet(base/v/'point/mean_predictions.parquet') for v in ['grid','calendar']}
    assert means['grid'].index.equals(means['calendar'].index) and len(means['grid'])==273031
    overall=[];seeds=[]
    for variant in ['grid','calendar']:
        for seed in range(42,52):
            d=pd.read_parquet(base/variant/'point'/f'seed_{seed}/predictions.parquet')
            assert d.index.equals(means[variant].index) and np.isfinite(d.to_numpy()).all()
            seeds.append({'variant':variant,'seed':seed,**d[metrics].astype(float).mean().to_dict()})
    seed_table=pd.DataFrame(seeds);seed_table.to_csv(out/'seed_means.csv',index=False)
    for metric in metrics:
        a,b=means['grid'][metric].astype(float),means['calendar'][metric].astype(float)
        for v in ['grid','calendar']:
            saved=float(summaries.loc[(summaries.variant==v)&(summaries.country=='all'),metric].iloc[0])
            assert np.isclose(saved,means[v][metric].astype(float).mean(),atol=1e-12,rtol=0)
        overall.append({'metric':metric,'grid_mean':float(a.mean()),'calendar_mean':float(b.mean()),
            'calendar_minus_grid':float(b.mean()-a.mean()),'row_mean_absolute_difference':float((b-a).abs().mean()),
            'row_prediction_contrast_correlation':float(a.corr(b)),
            **{f'{v}_{name}':value for v in ['grid','calendar'] for name,value in {
                'seed_min':float(seed_table.loc[seed_table.variant==v,metric].min()),
                'seed_max':float(seed_table.loc[seed_table.variant==v,metric].max()),
                'positive_seeds':int(seed_table.loc[seed_table.variant==v,metric].gt(0).sum())}.items()}})
    overall=pd.DataFrame(overall);overall.to_csv(out/'overall_comparison.csv',index=False)
    a=summaries[(summaries.variant=='grid')&(summaries.country!='all')].set_index('country')
    b=summaries[(summaries.variant=='calendar')&(summaries.country!='all')].set_index('country').reindex(a.index)
    assert a.weighted_rows.equals(b.weighted_rows)
    countries=[]
    for country in a.index:
        for metric in metrics:
            av,bv=float(a.loc[country,metric]),float(b.loc[country,metric])
            countries.append({'country_id':country,'country':return_country_name_dict()[int(float(country))],
                'records':int(a.loc[country,'weighted_rows']),'metric':metric,'grid':av,'calendar':bv,
                'calendar_minus_grid':bv-av,'opposite_nonzero_sign':bool(av*bv<0)})
    countries=pd.DataFrame(countries);countries.to_csv(out/'country_comparison.csv',index=False)
    flips={m:countries.loc[countries.metric.eq(m)&countries.opposite_nonzero_sign,'country'].tolist() for m in metrics}
    largest={m:countries.loc[countries.metric.eq(m)].assign(abs_delta=lambda x:x.calendar_minus_grid.abs()).nlargest(3,'abs_delta')[['country','grid','calendar','calendar_minus_grid']].to_dict(orient='records') for m in metrics}
    assert all(sha(ROOT/n)==h for n,h in hashes.items())
    result={'scope':'Point predictions in PHYSICAL_HLTH coding units, not causal effects, p-values or confidence intervals. Seed ranges describe split variability only. Calendar minus grid combines changed year assignment and model refitting. Monthly SD is seasonal dispersion.',
            'records':273031,'countries':22,'overall':overall.to_dict(orient='records'),
            'country_sign_flips':flips,'largest_country_differences':largest,'input_sha256':hashes,'inputs_unchanged':True}
    (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},indent=2))


if __name__=='__main__':main()
