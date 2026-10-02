"""Read only aggregate coordinate precision and survey-date availability."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
def main():
    path=ROOT/'data/raw/gfs_all-countries_wave2-with-midyear_sensitive.csv'
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    cols=['LATITUDE_Y1','LONGITUDE_Y1','LATITUDE_Y2','LONGITUDE_Y2','DOI_ANNUAL_Y1','DOI_ANNUAL_Y2','DOI_RECRUIT_Y1','DOI_MY']
    df=pd.read_csv(path,usecols=cols,dtype=str)
    out={'raw_rows':len(df),'dates':{},'coordinates':{}}
    for col in cols:
        values=df[col].fillna('').str.strip();values=values[values.ne('')]
        if col.startswith('DOI'):
            dates=pd.to_datetime(values,format='%m/%d/%Y',errors='coerce')
            parts=values.str.split('/',expand=True)
            out['dates'][col]={'nonblank':len(values),'parsed':int(dates.notna().sum()),'unparsed':int(dates.isna().sum()),
                'first_component_over_12':int((pd.to_numeric(parts[0],errors='coerce')>12).sum()),
                'second_component_over_12':int((pd.to_numeric(parts[1],errors='coerce')>12).sum()),
                'counts_by_year':dates.dt.year.dropna().astype(int).value_counts().sort_index().to_dict(),
                'counts_by_month':dates.dt.to_period('M').dropna().astype(str).value_counts().sort_index().to_dict()}
        else:
            x=pd.to_numeric(values,errors='coerce').dropna();limit=90 if col.startswith('LAT') else 180
            valid=x[x.between(-limit,limit)]
            out['coordinates'][col]={'nonblank':len(values),'numeric':len(x),'in_physical_range':len(valid),
                'integer_fraction_in_range':float(np.mean(np.isclose(valid,np.round(valid),rtol=0,atol=1e-9)))}
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    out['raw_sha256']=before;out['input_unchanged']=True
    out['limitations']='Raw-file availability only, not final matched sample. Date format checked by parsing and component ranges; coordinate units/centroid/privacy provenance not established.'
    p=ROOT/'data/exp/revision_audit/temperature_provenance/survey_metadata.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
