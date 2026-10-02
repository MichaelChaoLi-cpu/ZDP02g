"""Build an editable SVG sample-flow figure and render PNG/PDF.

Reproduce: uv run --with cairosvg python src/analyses/plot_revision_sample_flow.py
On macOS with Homebrew Cairo, prefix the command with
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib.
Only aggregate audit outputs are read; no survey records are modified.
"""
from pathlib import Path
from html import escape
import json
import cairosvg

ROOT = Path(__file__).resolve().parents[2]
a = json.loads((ROOT / 'data/exp/revision_audit/sample_flow/audit.json').read_text())
assert a['legacy_replay_matches_stored'] and a['production_loader_exact_match']
assert a['observed_records'] - 20587 - 28670 - 14499 == a['final_rows'] == 273031
assert a['absent_wave_slots'] == 79051 and a['final_people'] == 182474
out = ROOT / 'data/results/revision'
out.mkdir(parents=True, exist_ok=True)
svg = ['''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1460" viewBox="0 0 1200 1460" role="img" aria-labelledby="title desc">
<title id="title">Selection of the analytical sample</title>
<desc id="desc">Flow of 336,787 observed survey records to 273,031 analytical records. Sequential exclusions total 63,756 records. Absent second-wave slots are not counted as exclusions.</desc>
<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M1 1 L7 4.5 L1 8" fill="none" stroke="#64748b" stroke-width="1.6"/></marker></defs>
<rect width="1200" height="1460" fill="white"/>
<g font-family="Arial, Helvetica, sans-serif" fill="#193247">''']

def text(x, y, value, size=20, fill='#193247', weight='400'):
    svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(value)}</text>')

def rect(x,y,w,h,fill='#f4f7fa',stroke='#cfdae3',radius=12):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')

def path(d):
    svg.append(f'<path d="{d}" fill="none" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>')

def mainbox(y,h,step,label,n,details,final=False):
    rect(64,y,560,h,'#eaf5f2' if final else '#f4f7fa','#75a99c' if final else '#cfdae3')
    text(88,y+31,step,14,'#40796c' if final else '#617789','700')
    text(88,y+63,label,21,weight='700')
    text(88,y+110,f'{n:,}',38,'#196452' if final else '#193247','700')
    for i,line in enumerate(details):text(88,y+140+26*i,line,18,'#526778')

def exclusion(y,h,n,lines):
    rect(708,y,428,h,'#fbf7f2','#e4d6c5')
    text(732,y+32,'EXCLUDED',13,'#8a6644','700')
    text(732,y+72,f'{n:,}',31,'#785735','700')
    for i,line in enumerate(lines):text(732,y+105+26*i,line,18,'#5f605e')

text(64,62,'Selection of the analytical sample',32,weight='700')
text(64,98,'Two survey waves · sequential record-level exclusions',19,'#617789')
text(64,151,'RETAINED RECORDS',13,'#617789','700')
text(708,151,'EXCLUSIONS AND SCOPE',13,'#617789','700')
mainbox(176,200,'01 / SOURCE','Observed survey records',336787,['Wave 1: 207,919  ·  Wave 2: 128,868','207,919 distinct respondents'])
rect(708,176,428,166,'#ffffff','#d9e1e8')
text(732,210,'NOT COUNTED AS EXCLUSIONS',13,'#617789','700')
text(732,249,'79,051 absent wave-2 slots',22,weight='700')
text(732,281,'These slots are not observed interviews.',18,'#617789')
text(732,308,'They are outside the starting record count.',18,'#617789')
mainbox(446,154,'02 / INCOME FILTER','After income-code filtering',316200,[])
path('M344 376 V446');path('M344 410 H708')
exclusion(366,158,20587,['Missing or designated nonresponse','income codes'])
mainbox(686,154,'03 / OTHER FILTERS','Processed survey records',287530,[])
path('M344 600 V686');path('M344 644 H708')
exclusion(570,210,28670,['Sequential coordinate, covariate,','mediator and outcome filters','Missing or out-of-range values'])
mainbox(906,178,'04 / EXPOSURE LINKAGE','Matched to temperature exposure',287530,['Unmatched records: 0'])
path('M344 840 V906')
mainbox(1180,190,'05 / COMPLETE-CASE SAMPLE','Final analytical sample',273031,['182,474 distinct respondents'],True)
path('M344 1084 V1180');path('M344 1132 H708')
exclusion(1000,270,14499,['Missing mapped income values at','the final complete-case filter:','','4,945  ·  nonresponse code −9998','9,554  ·  codes outside existing lookup'])
svg.append('<line x1="64" y1="1403" x2="1136" y2="1403" stroke="#d9e1e8"/>')
text(64,1437,'Total excluded observed records: 63,756',19,weight='700')
text(708,1437,'Counts refer to records, not unique people.',16,'#617789')
svg.append('</g></svg>')
svg_path = out / 'sample_flow.svg'
svg_path.write_text('\n'.join(svg)+'\n')
cairosvg.svg2png(url=str(svg_path),write_to=str(out/'sample_flow.png'),scale=2.5)
cairosvg.svg2pdf(url=str(svg_path),write_to=str(out/'sample_flow.pdf'))
print(f'Wrote SVG, 3000 × 3650 PNG, and PDF to {out}')
