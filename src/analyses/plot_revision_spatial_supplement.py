"""R2C22: display location N and country intervals from validated existing outputs."""
from pathlib import Path
import hashlib,json,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Patch
import shapefile

ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--figure-s3-only',action='store_true');args=parser.parse_args()
    out=ROOT/'data/results/revision_spatial_supplement';out.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/'data/results/revision_spatial_contrasts/observed_location_contrasts.csv',ROOT/'data/results/revision_bootstrap_uncertainty/country_intervals.csv',ROOT/'data/results/revision_sample_descriptives/country_wave_counts.csv']
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    loc=pd.read_csv(paths[0]);ci=pd.read_csv(paths[1]);counts=pd.read_csv(paths[2])
    assert len(loc)==2926 and loc.records.sum()==273031
    countries=ci[ci.country_id.ne('all')].copy()
    # Explicit numeric ID matching excludes pooled rows without relying on labels.
    countries['id']=pd.to_numeric(countries.country_id,errors='coerce');countries=countries[countries.id.notna()]
    assert countries.id.nunique()==22
    expected=counts.set_index('Country code').Total
    assert countries.apply(lambda r:r.records==expected.loc[r.id],axis=1).all()
    assert countries.draws.eq(100).all() and (countries.lower_95<=countries.upper_95).all()
    assert np.isfinite(countries[['point','lower_95','upper_95']]).all().all()
    plt.rcParams.update({'font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    boundaries=ROOT/'data/processed/map_boundaries/natural_earth';segments=[]
    for name in ['physical/ne_110m_coastline.shp','cultural/ne_110m_admin_0_boundary_lines_land.shp']:
        p=boundaries/name;hashes[str(p.relative_to(ROOT))]=sha(p)
        with shapefile.Reader(str(p)) as reader:
            for s in reader.shapes():
                ends=list(s.parts)+[len(s.points)];segments.extend(np.asarray(s.points[a:b]) for a,b in zip(ends[:-1],ends[1:]))
    bins=[1,2,10,50,100,500,np.inf];labels=['1','2–9','10–49','50–99','100–499','≥500'];colors=['#e0e0e0','#c6dbef','#9ecae1','#4292c6','#2171b5','#08306b']
    cat=pd.cut(loc.records,bins=bins,right=False,labels=False);assert cat.notna().all()
    fig,ax=plt.subplots(figsize=(12,5.8));ax.add_collection(LineCollection(segments,colors='#777777',linewidths=.35,zorder=1))
    # The supplied coordinates are integer-degree locations. Use half-degree
    # edges around each supplied coordinate, not fixed screen-size markers.
    assert np.array_equal(loc[['LATITUDE','LONGITUDE']].to_numpy(), np.rint(loc[['LATITUDE','LONGITUDE']].to_numpy()))
    grid=loc.groupby(['LATITUDE','LONGITUDE'],as_index=False).records.sum()
    pd.testing.assert_frame_equal(grid,loc[['LATITUDE','LONGITUDE','records']],check_dtype=False)
    grid['west']=grid.LONGITUDE-.5;grid['east']=grid.LONGITUDE+.5
    grid['south']=grid.LATITUDE-.5;grid['north']=grid.LATITUDE+.5
    vertices=[[(r.west,r.south),(r.east,r.south),(r.east,r.north),(r.west,r.north)] for r in grid.itertuples()]
    ax.add_collection(PolyCollection(vertices,facecolors=[colors[int(i)] for i in cat],edgecolors='none',zorder=2))
    assert grid.records.sum()==273031 and len(grid)==2926
    assert np.allclose(grid.east-grid.west,1) and np.allclose(grid.north-grid.south,1)
    grid.assign(count_bin=[labels[int(i)] for i in cat]).to_csv(out/'figureS3_grid_counts.csv',index=False)
    ax.set(xlim=(-180,180),ylim=(-60,85),xlabel='Longitude',ylabel='Latitude');ax.set_aspect('equal');ax.grid(alpha=.18,linewidth=.4)
    for spine in ax.spines.values():spine.set_visible(True)
    fig.legend(handles=[Patch(facecolor=c,edgecolor='#555555',label=l) for c,l in zip(colors,labels)],title='Records per 1° × 1° cell',loc='lower center',bbox_to_anchor=(.5,.015),ncol=6,frameon=False)
    fig.subplots_adjust(bottom=.21,top=.985,left=.065,right=.985)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'figureS3_location_counts.{ext}',dpi=300)
    plt.close(fig)
    if args.figure_s3_only:
        assert all(sha(ROOT/p)==h for p,h in hashes.items())
        report=dict(decision='KILA-D-20261002-014',title_removed=True,bottom_explanation_removed=True,style_reference='Figures 7/8: boundary color #777777, linewidth .35, coordinate extent and grid alpha .18 linewidth .4, full axes frame',source_hashes=hashes,source_unchanged=True,records=int(grid.records.sum()),occupied_cells=len(grid),cell_width_degrees=1,cell_height_degrees=1,cell_edges='supplied integer-degree coordinates +/- 0.5 degrees',counts_match_observed_locations_exactly=True,unsampled_cells='blank',native_bootstrap_grid_unchanged=True,figureS4_unchanged=True,model_refits=0)
        (out/'figureS3_grid_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));return
    order=countries[countries.metric.eq('mean_positive_total_change')].sort_values(['point','id']).id.tolist();assert len(order)==22
    metrics=['mean_positive_direct_change','mean_positive_total_change','std_positive_direct_change','std_positive_total_change']
    titles=['(a) Annual mean +1°C: direct','(b) Annual mean +1°C: total','(c) Temperature SD +0.1°C: direct','(d) Temperature SD +0.1°C: total']
    fig,axes=plt.subplots(2,2,figsize=(16,17));plotted=[]
    for ax,metric,title in zip(axes.flat,metrics,titles):
        g=countries[countries.metric.eq(metric)].set_index('id').loc[order].reset_index();assert len(g)==22
        y=np.arange(22);ax.hlines(y,g.lower_95,g.upper_95,color='#1f77b4',lw=1.4);ax.scatter(g.point,y,color='#1f77b4',s=26,zorder=3)
        ax.axvline(0,color='#666666',ls='--',lw=.8);ax.set_yticks(y,[f'{r.country.removeprefix("the ")}  (N={int(r.records):,})' for r in g.itertuples()],fontsize=10);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel('Change in predicted physical-health score (0–10)')
        pair=countries[countries.metric.isin(metrics[:2] if metric.startswith('mean') else metrics[2:])];lo=min(0,pair.lower_95.min(),pair.point.min());hi=max(0,pair.upper_95.max(),pair.point.max());pad=(hi-lo)*.06;ax.set_xlim(lo-pad,hi+pad)
        plotted.append(g)
    fig.suptitle('Country prediction contrasts and pointwise 95% bootstrap intervals',fontsize=16,y=.988)
    fig.tight_layout(rect=(0,.065,1,.97),h_pad=3,w_pad=3)
    fig.text(.5,.014,'100 country-stratified native-grid draws; both model stages refitted. Fixed seed 42 and hyperparameters.\nSame country order in every panel, sorted by annual-mean total point estimate; order does not establish distinct ranks.\nN counts analytical records. Intervals are pointwise, not multiplicity-adjusted; 100 draws limit tail precision.',ha='center',fontsize=11)
    for ext in ['png','svg','pdf']:fig.savefig(out/f'figureS4_country_intervals.{ext}',dpi=300)
    plt.close(fig);pd.concat(plotted).to_csv(out/'plotted_country_intervals.csv',index=False)
    loc.assign(count_bin=[labels[int(i)] for i in cat]).to_csv(out/'plotted_location_counts.csv',index=False)
    assert all(sha(ROOT/p)==h for p,h in hashes.items())
    report=dict(decision='KILA-D-20261002-012',source_hashes=hashes,source_unchanged=True,locations=2926,records=273031,country_panels=4,countries_per_panel=22,all_country_N_match_table1=True,all_estimates_and_intervals_copied_without_reestimation=True,location_bin_counts={labels[i]:int(cat.eq(i).sum()) for i in range(6)},rank_order=order,model_refits=0,location_intervals_available=False)
    (out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
