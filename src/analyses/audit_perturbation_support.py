"""No-refit country marginal-range diagnostics; not joint support certification."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/analyses/notebooks'))
from SettingForFeatures import data_load_combine_dataset

def main():
    import os
    os.chdir(ROOT)
    sources=[ROOT/'data/processed/revision_exposure_validation/fixed_sample_exposure_variants.parquet',ROOT/'data/processed/GlobalFlourishingDataWithLonLit.parquet']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    d=data_load_combine_dataset('tas');v=pd.read_parquet(sources[0]).set_index('analysis_row')
    assert v.index.equals(d.index) and len(d)==273031
    rows=[];flags={}
    for feature,delta in [('TAS_MEAN',1.),('TAS_STD',.1)]:
        x=v['grid_'+feature];assert np.isfinite(x).all()
        lo=x.groupby(d.COUNTRY).transform('min');hi=x.groupby(d.COUNTRY).transform('max')
        outside=(x+delta<lo-1e-10)|(x+delta>hi+1e-10);flags[feature]=outside
        assert not ((x<lo-1e-10)|(x>hi+1e-10)).any()
        for country,idx in [('all',d.index)]+[(str(c),g.index) for c,g in d.groupby('COUNTRY')]:
            rows.append(dict(country=country,feature=feature,delta=delta,rows=len(idx),outside_rows=int(outside.loc[idx].sum()),outside_fraction=float(outside.loc[idx].mean())))
    out=ROOT/'data/exp/revision_audit/comment5_support';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'country_marginal_support.csv',index=False)
    summary={'rows':len(d),'any_metric_outside_rows':int((flags['TAS_MEAN']|flags['TAS_STD']).sum()),'scope':'Country-specific full-sample marginal min/max, corrected grid exposures; not fold-training or joint/location-conditional support. No trimming or refitting.', 'input_sha256':hashes}
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    (out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(pd.DataFrame(rows).query("country == 'all'").to_string(index=False));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
