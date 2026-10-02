"""Render validated current-model normalized gain; reject partial or smoke evidence."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='data/exp/revision_feature_importance_20261001');p.add_argument('--output',default='data/results/revision_figure5');a=p.parse_args()
 src=ROOT/a.input;out=ROOT/a.output;v=json.loads((src/'validation.json').read_text())
 assert v['complete'] and not v['smoke'] and v['completed_folds']==10
 assert v['actual_components']==v['expected_components']==160
 assert all(x['baseline_max_abs_difference']<=1e-6 for x in v['component_checks'])
 df=pd.read_csv(src/'summary.csv');assert np.isfinite(df[['mean','std']]).all().all()
 import sys
 sys.path.insert(0,str(ROOT/'src/analyses/notebooks'))
 from SettingForFeatures import return_readable_variable_name
 labels=return_readable_variable_name()
 outcomes=['PHYSICAL_HLTH','BODILY_PAIN','HEALTH_PROB','DAYS_EXERCISE']
 titles=['a: Overall Physical Health','b: Bodily Pain','c: Health Problems Restricting Activity','d: Weekly Exercise']
 fig,axes=plt.subplots(4,1,figsize=(14,24),sharex=True,layout='constrained')
 for ax,m,title,color in zip(axes,outcomes,titles,['tab:blue','tab:green','tab:red','tab:purple']):
  g=df[df.outcome.eq(m)].sort_values('mean',ascending=False);assert len(g)>0 and np.isclose(g['mean'].sum(),100)
  names=[]
  for f in g.feature:
   name=labels.get(f,f)
   if m=='PHYSICAL_HLTH' and f in outcomes[1:]:name='Predicted probability: '+labels.get(f,f)
   names.append(name)
  ax.barh(names,g['mean'],color=color,alpha=.9)
  ax.invert_yaxis();ax.set_title(title,loc='left');ax.grid(axis='x',linestyle='--',alpha=.35);ax.set_axisbelow(True)
 axes[-1].set_xlabel('Mean normalized gain (%)')
 out.mkdir(parents=True,exist_ok=True)
 for ext in ['png','svg','pdf']:fig.savefig(out/f'figure5_current_models.{ext}',dpi=240)
 plt.close(fig)
 (out/'caption.txt').write_text('Figure 5: Feature importance in the fitted prediction models. Bars show gain normalized to 100% within each model. For each symptom or behavior classifier, normalized gains are averaged across the five inner models within each outer fold and then across the ten outer folds. Physical health gains are averaged across the ten outer-fold outcome models. These models use generated symptom and behavior probabilities. Models reproduce the CPU benchmark and Figure 6 configuration, with fixed parameters and seed 42. Importance describes model split usage; it does not quantify causal effects, mediation, or independent contributions.\n')
 print(out)
if __name__=='__main__':main()
