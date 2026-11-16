"""Per-seed effect of removing the target MSA on intact targets (ipSAE_min).
Mean of per-target dAP (AP_no_MSA - AP_full) with a paired bootstrap over targets."""
import random, statistics as st, sys
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from msa_paths import ROOT, rpath
import make_figures as mf          # reuses the report's tie-robust AP + per-target dAP

BLUE, GREY = '#2a78d6', '#888888'
res = {}
for s in (1, 2, 3):
    d = mf.per_target_dap(s, metric=0)
    d = {t: v for t, v in d.items() if v == v}
    vals = list(d.values()); rng = random.Random(0)
    boots = sorted(st.mean(rng.choice(vals) for _ in vals) for _ in range(2000))
    res[s] = (st.mean(vals), boots[int(.025 * 2000)], boots[int(.975 * 2000)], len(vals))
    print(s, '%+.3f [%+.3f, %+.3f] n=%d' % res[s])
mean = st.mean(r[0] for r in res.values()); print('mean %+.3f' % mean)

fig, ax = plt.subplots(figsize=(6.25, 3.0))
for s, (m, lo, hi, n) in res.items():
    ax.plot([lo, hi], [s, s], color=BLUE, lw=2, solid_capstyle='round', zorder=2)
    ax.plot(m, s, 'o', ms=8, color=BLUE, mec='white', mew=1.5, zorder=3)
    ax.annotate('%+.3f  [%+.3f, %+.3f]' % (m, lo, hi), (hi, s), xytext=(8, 0),
                textcoords='offset points', va='center', fontsize=8, color='#111')
ax.axvline(0, color='#333', lw=0.9, zorder=1)
ax.axvline(mean, color=GREY, lw=0.9, ls=':', zorder=1)
ax.text(mean, 3.62, 'mean of seeds %+.3f' % mean, ha='center', fontsize=7.5, color='#555')
ax.set_yticks([1, 2, 3]); ax.set_yticklabels(['Seed 1', 'Seed 2', 'Seed 3'])
ax.set_ylim(3.8, 0.4); ax.set_xlim(-0.55, 0.1)
ax.set_xlabel(r'$\Delta$AP when the target MSA is removed (ipSAE$_{min}$; below zero = MSA helps)')
ax.set_title('Effect of removing the target MSA on intact targets, per seed\n(mean per-target $\\Delta$AP, paired bootstrap 95% CI over 13 targets)', fontsize=8.5, loc='left')
for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
ax.grid(axis='x', color='#e5e5e5', lw=0.6, zorder=0); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig_seed_effect.png', dpi=200)
