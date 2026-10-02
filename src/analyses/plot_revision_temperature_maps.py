"""Prepare corrected Figure 4 from validated native-grid exposures without refitting models."""
from pathlib import Path
import hashlib,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap,Normalize
import numpy as np
import pandas as pd
import shapefile
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 sys.path.insert(0,str(ROOT/'src/analyses/notebooks'))
 from SettingForFeatures import data_load_combine_dataset
 data=data_load_combine_dataset('tas')
 path=ROOT/'data/processed/revision_exposure_validation/fixed_sample_exposure_variants.parquet';digest=sha(path)
 exp=pd.read_parquet(path).set_index('analysis_row')
 assert data.index.equals(exp.index) and len(data)==273031
 assert np.array_equal(data[['TAS_MEAN','TAS_STD']].to_numpy(),exp[['TAS_MEAN','TAS_STD']].to_numpy())
 joined=data[['LATITUDE','LONGITUDE','WAVE']].join(exp[['grid_TAS_MEAN','grid_TAS_STD']])
 cols=['grid_TAS_MEAN','grid_TAS_STD'];keys=['LATITUDE','LONGITUDE','WAVE']
 assert joined.groupby(keys)[cols].nunique().to_numpy().max()==1
 loc=joined.groupby(keys)[cols].first().reset_index()
 out=ROOT/'data/results/revision_temperature_maps';out.mkdir(exist_ok=True,parents=True)
 loc.to_csv(out/'figure4_location_values.csv',index=False)
 segments=[];boundaries={}
 for relative in ['physical/ne_110m_coastline.shp','cultural/ne_110m_admin_0_boundary_lines_land.shp']:
  p=ROOT/'data/processed/map_boundaries/natural_earth'/relative;boundaries[relative]=sha(p)
  with shapefile.Reader(str(p)) as reader:
   for s in reader.shapes():
    ix=list(s.parts)+[len(s.points)];segments.extend(np.asarray(s.points[a:b]) for a,b in zip(ix[:-1],ix[1:]))
 cmap=LinearSegmentedColormap.from_list('legacy_temperature',['blue','green','yellow','red'],N=256)
 fig,axes=plt.subplots(2,2,figsize=(16,8),layout='constrained')
 limits=[(-10,35),(0,15)];labels=['Annual mean temperature','Annual SD of monthly mean temperature']
 for row,col in enumerate(cols):
  norm=Normalize(*limits[row])
  for c,wave in enumerate([1,2]):
   ax=axes[row,c];sub=loc[loc.WAVE==wave]
   ax.add_collection(LineCollection(segments,colors='black',linewidths=.3,zorder=1))
   dots=ax.scatter(sub.LONGITUDE,sub.LATITUDE,c=sub[col],cmap=cmap,norm=norm,marker='s',s=7,linewidths=0,zorder=2,rasterized=True)
   ax.set(xlim=(-180,180),ylim=(-90,90));ax.set_aspect('equal')
   ax.set_title(f'({"abcd"[row*2+c]}) {labels[row]} — Wave {wave}',loc='left',fontsize=11,fontweight='bold')
   ax.set_xticks([-180,-120,-60,0,60,120,180]);ax.set_yticks([-60,-30,0,30,60]);ax.grid(alpha=.3,linestyle='--',linewidth=.4)
   if row==1:ax.set_xlabel('Longitude (degrees)')
   if c==0:ax.set_ylabel('Latitude (degrees)')
  fig.colorbar(dots,ax=axes[row,:],fraction=.025,pad=.02,label='Temperature (°C)' if row==0 else 'Temperature SD (°C)')
 fig.supxlabel('Native-grid model values assigned to observed survey locations. Blank areas have no displayed records.\nWave 1: 2023 simulations; Wave 2: 2024 simulations. Each row uses the same color scale across waves.',fontsize=9)
 for ext in ['png','svg','pdf']:fig.savefig(out/f'figure4_native_grid.{ext}',dpi=300)
 plt.close(fig)
 assert sha(path)==digest
 report=dict(rows=len(data),location_wave_count=len(loc),input_sha256=digest,source_unchanged=True,sample_and_legacy_exposure_alignment_verified=True,within_location_wave_values_constant=True,limits=limits,clipped_values=int(sum(((loc[c]<lo)|(loc[c]>hi)).sum() for c,(lo,hi) in zip(cols,limits))),boundaries=boundaries,visual_support='Square markers locate supplied integer-degree survey coordinates; their size does not represent native grid-cell footprints.',status='Replacement asset prepared; not inserted into manuscript.')
 (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
