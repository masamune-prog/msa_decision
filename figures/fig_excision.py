"""Intact vs excised target, with and without MSA (seed 1, ipSAE_min, same 414 designs)."""
import csv, statistics as st, sys
from collections import defaultdict
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from msa_paths import ROOT, rpath
import make_figures as mf
csv.field_size_limit(10**9)
rows = list(csv.DictReader(open(rpath('all_af3_folds.csv'))))
ids = {r['binder_id'] for r in rows if r['experiment'] == 'ex_out'}
cell = defaultdict(lambda: defaultdict(list))   # (geom, msa) -> target -> [(score,label)]
for r in rows:
    if r['binder_id'] not in ids: continue
    key = {('e2_out','full'):('Intact','MSA'), ('e2_out','none'):('Intact','none'),
           ('ex_out','exfull'):('Excised','MSA'), ('ex_out','exnone'):('Excised','none')}.get((r['experiment'], r['condition']))
    if key: cell[key][r['target_id']].append((float(r['ipsae_min']), int(r['label'])))
T = ['IL7Ra', 'SARS_CoV2_RBD', 'Pdl1', 'sntx']
AP = {k: {t: mf.ap(v[t]) for t in T} for k, v in cell.items()}
crop3 = T[:3]
for g in ('Intact', 'Excised'):
    print(g, 'MSA %.3f none %.3f (mean of 3 cropped)' % tuple(st.mean(AP[(g, m)][t] for t in crop3) for m in ('MSA', 'none')))
    for t in T: print('  ', t, '%.3f %.3f' % (AP[(g,'MSA')][t], AP[(g,'none')][t]))

BLUE, ORANGE = '#2a78d6', '#d9822b'
names = {'IL7Ra': 'IL7Ra', 'SARS_CoV2_RBD': 'SARS-CoV-2 RBD', 'Pdl1': 'Pdl1', 'sntx': 'sntx (control, no crop)'}
fig, axes = plt.subplots(1, 2, figsize=(6.25, 3.0), sharey=True, sharex=True)
for ax, g in zip(axes, ('Intact', 'Excised')):
    for i, t in enumerate(T):
        a, b = AP[(g, 'MSA')][t], AP[(g, 'none')][t]
        ax.plot([a, b], [i, i], color='#bbb', lw=2, zorder=1)
        ax.plot(a, i, 'o', ms=8, color=BLUE, mec='white', mew=1.2, zorder=3, label='with MSA' if i == 0 else None)
        ax.plot(b, i, 'o', ms=8, color=ORANGE, mec='white', mew=1.2, zorder=3, label='no MSA' if i == 0 else None)
    ax.set_title(g + ' target', fontsize=9, loc='left')
    ax.set_xlim(0, 1); ax.grid(axis='x', color='#e5e5e5', lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.set_xlabel('average precision')
axes[0].set_yticks(range(4)); axes[0].set_yticklabels([names[t] for t in T]); axes[0].invert_yaxis()
h,l=axes[0].get_legend_handles_labels(); fig.legend(h,l,frameon=False,fontsize=8,loc='upper right',ncol=2)
fig.suptitle('Effect of removing the MSA, intact vs excised target (seed 1)', fontsize=9, x=0.02, ha='left')
fig.tight_layout(); fig.savefig('fig_excision_effect.png', dpi=200)
