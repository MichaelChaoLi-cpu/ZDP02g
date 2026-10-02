"""Render the validated Table 2 candidate as a readable review PNG."""
from pathlib import Path
import os
import textwrap

os.environ.setdefault('MPLCONFIGDIR', '/tmp/zdp02g-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/results/revision_bootstrap_uncertainty'


def main():
    data = pd.read_csv(OUT / 'table2_main_four_contrasts.csv', dtype={'country_id': str})
    data['order'] = pd.to_numeric(data.country_id, errors='coerce').fillna(999)
    data = data.sort_values('order')
    metrics = ['mean_positive_direct_change', 'mean_positive_total_change',
               'std_positive_direct_change', 'std_positive_total_change']
    rows = []
    for row in data.itertuples(index=False):
        values = row._asdict()
        name = values['country'].removeprefix('the ')
        rows.append([name] + [values[m].replace(' [', '\n[').replace('-', '−') for m in metrics])
    assert len(rows) == 23 and not data[metrics].isna().any().any()
    fig, ax = plt.subplots(figsize=(17, 18))
    fig.patch.set_facecolor('white')
    ax.axis('off')
    fig.text(.05, .965, 'Table 2. Country-level prediction contrasts', fontsize=21, weight='bold')
    fig.text(.05, .942, 'Review candidate • Point estimate [95% percentile bootstrap interval]', fontsize=12, color='#46515d')
    headers = ['Country / region', 'Mean temperature +1°C\nDirect contrast',
               'Mean temperature +1°C\nTotal contrast', 'Temperature SD +0.1°C\nDirect contrast',
               'Temperature SD +0.1°C\nTotal contrast']
    table = ax.table(cellText=rows, colLabels=headers, cellLoc='center',
                     colWidths=[.19, .2025, .2025, .2025, .2025], bbox=[0, .105, 1, .84])
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor('#d5dce3')
        cell.set_linewidth(.45)
        if r == 0:
            cell.set_facecolor('#203b53')
            cell.set_text_props(color='white', weight='bold', fontsize=11)
            cell.set_height(cell.get_height() * 1.25)
        else:
            cell.set_facecolor('#f2f5f8' if r % 2 == 0 else 'white')
            if c == 0:
                cell.set_text_props(ha='left', weight='medium')
                cell.PAD = .07
            if rows[r - 1][0] == 'Pooled sample':
                cell.set_facecolor('#e1ebf3')
                cell.set_text_props(weight='bold')
    notes = [
        'Units: change in predicted physical-health score (0–10). Temperature SD means annual temperature standard deviation.',
        'Direct contrasts hold generated mediator probabilities fixed; total contrasts update them. Pooled estimates average records, not countries.',
        'Intervals use 100 country-stratified native-grid bootstrap draws with both stages refitted; split seed 42 and hyperparameters remain fixed.',
        'Intervals are pointwise, not adjusted for multiple comparisons. Tail precision is limited with 100 draws. These are model-based prediction contrasts, not causal effects.',
    ]
    y = .104
    for note in notes:
        wrapped = textwrap.fill(note, width=145)
        fig.text(.05, y, wrapped, fontsize=10, va='top', color='#37434e', linespacing=1.5)
        y -= .019 * (wrapped.count('\n') + 1)
    fig.subplots_adjust(left=.05, right=.97, top=.945, bottom=.10)
    fig.savefig(OUT / 'table2_review.png', dpi=220, facecolor='white')
    plt.close(fig)
    print(OUT / 'table2_review.png')


if __name__ == '__main__':
    main()
