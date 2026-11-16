"""Intact (all 13 receptors) vs excised (only the 4 receptors that were run), seed 1, ipSAE_min."""
import csv, sys
from collections import defaultdict
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from msa_paths import ROOT, rpath
import make_figures as mf
csv.field_size_limit(10**9)
cell = defaultdict(lambda: defaultdict(list))
M = {('e2_out','full'):('Intact','MSA'), ('e2_out','none'):('Intact','none'),
     ('ex_out','exfull'):('Excised','MSA'), ('ex_out','exnone'):('Excised','none'),
     ('ex2_out','exfull'):('Excised','MSA'), ('ex2_out','exnone'):('Excised','none')}
for r in csv.DictReader(open(rpath('all_af3_folds.csv'))):
    k = M.get((r['experiment'], r['condition']))
    if k: cell[k][r['target_id']].append((float(r['ipsae_min']), int(r['label'])))
T = sorted(cell[('Intact','MSA')], key=lambda t: mf.ap(cell[('Intact','none')][t]) - mf.ap(cell[('Intact','MSA')][t]), reverse=True)
AP = {k: {t: mf.ap(v[t]) for t in v} for k, v in cell.items()}
npos = {t: sum(l for _, l in cell[('Intact','MSA')][t]) for t in T}
BLUE, ORANGE = '#2a78d6', '#d9822b'
fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.6), sharey=True, sharex=True)
for ax, g in zip(axes, ('Intact', 'Excised')):
    for i, t in enumerate(T):
        if t not in AP[(g, 'MSA')]:
            ax.text(0.5, i, 'not run', ha='center', va='center', fontsize=7, color='#999'); continue
        a, b = AP[(g,'MSA')][t], AP[(g,'none')][t]
        ax.plot([a, b], [i, i], color='#bbb', lw=2, zorder=1)
        ax.plot(a, i, 'o', ms=7, color=BLUE, mec='white', mew=1.2, zorder=3, label='with MSA' if i == 0 else None)
        ax.plot(b, i, 'o', ms=7, color=ORANGE, mec='white', mew=1.2, zorder=3, label='no MSA' if i == 0 else None)
    ax.set_title(g + ' target', fontsize=9, loc='left')
    ax.set_xlim(0, 1); ax.grid(axis='x', color='#e5e5e5', lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.set_xlabel('average precision')
axes[0].set_yticks(range(len(T)))
axes[0].set_yticklabels([f"{t.replace('SARS_CoV2_RBD','SARS-CoV-2 RBD')} ({npos[t]})" for t in T], fontsize=8)
axes[0].invert_yaxis()
fig.suptitle('Effect of removing the MSA: intact (13 receptors) vs excised (all 13), seed 1', fontsize=9, x=0.02, ha='left')
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=8, loc='lower right', ncol=2)
fig.text(0.02, 0.01, 'Number of confirmed binders in brackets. Sorted from least to most MSA loss on the intact target.', fontsize=7, color='#666')
fig.tight_layout(rect=(0, 0.05, 1, 0.96)); fig.savefig('fig_excision_all13.png', dpi=200)
for t in T: print('%-14s pos %3d  intact %.3f %.3f   excised %.3f %.3f' % (t, npos[t], AP[('Intact','MSA')][t], AP[('Intact','none')][t], AP[('Excised','MSA')][t], AP[('Excised','none')][t]))
