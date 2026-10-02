"""Quantitative descriptive comparison of Figure6 PDPs and paired shift contrasts.

PDP ranges and heterogeneous-baseline shifts are distinct functionals, not an
identity. Keep CPU paired comparisons separate from GPU primary Table2 values.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from audit_temperature_provenance import ROOT,sha


def main():
    paths={'curves':ROOT/'data/exp/revision_probability_pdp_20260930/curves.csv','cpu':ROOT/'data/exp/revision_experiments/fixed_yaml_20260926/grid/point/seed_42/predictions.parquet','gpu':ROOT/'data/exp/revision_uncertainty_audit/point_predictions.parquet'}
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths.values()}
    curves=pd.read_csv(paths['curves'],dtype={'wave':str});cpu=pd.read_parquet(paths['cpu']);gpu=pd.read_parquet(paths['gpu']);assert cpu.index.equals(gpu.index)
    rows=[]
    for short,feature,delta in [('mean','TAS_MEAN',1.),('std','TAS_STD',.1)]:
        p=curves[(curves.outcome=='PHYSICAL_HLTH') & (curves.feature==feature) & (curves.wave=='pooled')].sort_values('value');assert len(p)==41
        row=dict(feature=feature,delta_celsius=delta,pdp_xmin=float(p.value.min()),pdp_xmax=float(p.value.max()),pdp_ymin=float(p.prediction.min()),pdp_ymax=float(p.prediction.max()),pdp_range=float(p.prediction.max()-p.prediction.min()),pdp_endpoint_change=float(p.prediction.iloc[-1]-p.prediction.iloc[0]))
        for label,d in [('cpu',cpu),('gpu',gpu)]:
            direct=d[f'{short}_positive_direct_change'].to_numpy(dtype=float);total=d[f'{short}_positive_total_change'].to_numpy(dtype=float)
            row.update({f'{label}_direct':float(direct.mean()),f'{label}_total':float(total.mean()),f'{label}_indirect':float((total-direct).mean())})
        rows.append(row)
    out=ROOT/'data/exp/revision_audit/r2c2_scale';out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    note={'input_sha256':hashes,'rows':len(cpu),'no_model_fits':True,'sources_unchanged':all(sha(ROOT/p)==h for p,h in hashes.items()),'definitions':{'Figure6':'Held-out PDP: set one exposure to a common value for all rows, keeping baseline generated mediator inputs fixed. Displayed health-score range is max-min across 41 grid values spanning exposure percentiles5–95.','CPU direct':'Shift each row from its own exposure by +1 degree mean or +0.1 degree SD, retaining its baseline generated mediators. Paired same-model comparison.','CPU total':'Same row shift, also propagating changed first-stage probabilities.','GPU Table2':'Separate primary execution; CPU/GPU differences must not be attributed to a change of estimand or merged into one exact model identity.'},'interpretation':'PDP range is not the expected individual shift contrast or a theoretical bound on it. Smaller revised shifts no longer exhibit the legacy common positive offset; no mathematical equality between the two summaries is asserted. No local derivative from smoothed curves is used as a substitute for exact shifts.'}
    assert note['sources_unchanged'];(out/'validation.json').write_text(json.dumps(note,indent=2));print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
