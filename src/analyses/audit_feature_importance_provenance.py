"""Read-only audit of legacy Figure5; never treat legacy gains as current results."""
from pathlib import Path
from zipfile import ZipFile
import hashlib, json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=ROOT/'data/exp/revision_audit/feature_importance';out.mkdir(parents=True,exist_ok=True)
    sources=[ROOT/'src/analyses/nbs/c20_visualization_of_data.py',ROOT/'src/analyses/nbs/c08_check_explanation_physical_health_tas_relationship.py',ROOT/'src/analyses/notebooks/SettingForFeatures.py']
    figure=ROOT/'data/results/figures/fig07_feature_importance_3panel.png'
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources+[figure]}
    rows=[]
    for outcome,name in [('PHYSICAL_HLTH','importance.csv')]+[(m,f'{m}_importance.csv') for m in ['BODILY_PAIN','HEALTH_PROB','DAYS_EXERCISE']]:
        p=ROOT/'data/exp/model_outputs'/name;hashes[str(p.relative_to(ROOT))]=sha(p)
        df=pd.read_csv(p,index_col=0);cols=[f'fold_{i}' for i in range(1,11)]
        assert df.index.is_unique and np.isfinite(df[cols].to_numpy()).all()
        assert (df[cols]>=0).all().all() and (df[cols].sum()>0).all()
        normalized=df[cols].div(df[cols].sum(),axis=1)*100
        assert np.allclose(normalized.sum(),100)
        mean=normalized.mean(axis=1);ranks=mean.rank(ascending=False,method='min')
        rows.extend(dict(outcome=outcome,feature=f,legacy_mean_percent=float(mean[f]),legacy_rank=int(ranks[f])) for f in df.index)
    with ZipFile(ROOT/'Rev/revision/ZDP02g.rev.clean.docx') as z:
        matches=[n for n in z.namelist() if n.startswith('word/media/') and hashlib.sha256(z.read(n)).hexdigest()==sha(figure)]
    report=dict(source_sha256=hashes,embedded_exact_byte_matches=matches,
        literal_wrong_title="a: Mental Health" in sources[0].read_text(),
        outcome_code_calls_physical_health_helper='aim_variable = SettingForFeatures.return_aim_variable_ph()' in sources[1].read_text(),
        helper_returns_physical_health="def return_aim_variable_ph() -> str:\n    return 'PHYSICAL_HLTH'" in sources[2].read_text(),
        legacy_training='Global precomputed mediator predictions loaded before outcome KFold; not the current outer/inner generated-input procedure.',
        current_model_importance_validated=False,
        limitation='Code supports a title error; it does not authenticate historical CSV execution or current model provenance. A missing exact image match does not prove different pixels.',
        action='Replay fixed CPU benchmark/Figure6 fits, validate cached predictions, then extract gain; no tuning or bootstrap required.')
    pd.DataFrame(rows).to_csv(out/'legacy_ranks.csv',index=False)
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
