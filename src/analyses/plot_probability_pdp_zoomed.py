"""Plot approved Figure 6 axis ranges without refitting or changing PDP values."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
from audit_temperature_provenance import ROOT,sha

source=ROOT/'data/exp/revision_probability_pdp_20260930'
out=ROOT/'data/results/revision_figure6'
out.mkdir(parents=True,exist_ok=True)
validation=json.loads((source/'validation.json').read_text())
assert not validation['smoke'] and validation['inputs_unchanged']
p=source/'curves.csv';before=sha(p)
d=pd.read_csv(p,dtype={'wave':str})
settings=[('PHYSICAL_HLTH','Overall physical health',(6.75,7.20),.10),
          ('BODILY_PAIN','Bodily pain',(.42,.48),.01),
          ('HEALTH_PROB','Activity limitation',(.18,.25),.01),
          ('DAYS_EXERCISE','Weekly exercise',(.64,.73),.02)]
plt.rcParams.update({'font.size':11,'axes.titlesize':13,'svg.fonttype':'none'})
fig,axes=plt.subplots(4,2,figsize=(10,12.5),sharey='row')
for i,(outcome,title,limits,tick) in enumerate(settings):
    for j,feature in enumerate(['TAS_MEAN','TAS_STD']):
        ax=axes[i,j]
        for wave,label,color,style in [('pooled','Pooled','#222222','-'),('1','Wave 1','#2878B5','--'),('2','Wave 2','#D65F24',':')]:
            g=d[(d.outcome==outcome)&(d.feature==feature)&(d.wave==wave)].sort_values('value')
            assert len(g)==41 and g.prediction.between(*limits).all(),(outcome,feature,wave)
            ax.plot(g.value,g.prediction,label=label,color=color,ls=style,lw=1.9)
        ax.set_ylim(*limits)
        ax.yaxis.set_major_locator(MultipleLocator(tick))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(axis='y',labelleft=True)
        ax.set_title(title)
        ax.set_xlabel('Annual mean temperature (°C)' if j==0 else 'Annual temperature SD (°C)')
        ax.set_ylabel('Predicted health score' if i==0 else 'Predicted probability')
        ax.grid(alpha=.18)
        ax.text(.025,.94,'abcdefgh'[i*2+j],transform=ax.transAxes,va='top',fontweight='bold')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.998))
fig.text(.5,.018,'Y-axes do not start at zero; scales differ across outcomes and are shared within each row.',ha='center',fontsize=10)
fig.subplots_adjust(top=.945,bottom=.085,left=.105,right=.98,hspace=.50,wspace=.29)
for ext in ['png','svg','pdf']:fig.savefig(out/f'figure6_zoomed.{ext}',dpi=300)
plt.close(fig)
assert sha(p)==before
(out/'figure6_zoomed_caption.md').write_text('''Figure 6. Model-based partial-dependence profiles for annual mean temperature and annual temperature standard deviation.

Curves average held-out predictions over pooled records or records from each wave using the same pooled models. Binary outcomes show class-1 probabilities. For physical health, baseline predicted mediator inputs remain fixed as the temperature input varies. Y-axes do not start at zero; scales differ across outcomes but are identical within each row. Curves are descriptive model profiles, not causal effects; no confidence intervals are shown. Exposure grids span the pooled fifth–95th percentiles.
''')
(out/'figure6_zoomed_validation.json').write_text(json.dumps({'decision':'KILA-D-20260930-007','source':str(p.relative_to(ROOT)),'source_sha256':before,'source_unchanged':True,'all_curves_within_axes':True,'refitted':False,'axis_ranges':{x[0]:x[2] for x in settings},'script_sha256':sha(Path(__file__))},indent=2)+'\n')
print(out/'figure6_zoomed.png')
