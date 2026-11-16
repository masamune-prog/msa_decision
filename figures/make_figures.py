"""Generate the report's figures from the saved result files.

Run: .venv/bin/python make_figures.py  ->  fig1..fig4 .png

Sized for the report's 15.9 cm text column, Liberation Sans (metric-compatible
with Arial) so the figure type matches the body type.
"""
import glob
import os
import random
import statistics as st
import sys
from collections import defaultdict
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from msa_paths import ROOT, rpath
from ipsae_recompute import ipsae_from_confidences

# fold directories (ex_out/, e3_out/, ...) and the saved PNGs are addressed
# relative to the work directory, exactly as when these figures were made
if os.path.isdir(ROOT):
    os.chdir(ROOT)      # figures that re-score folds need them on relative paths

plt.rcParams.update({
    'font.family': 'Liberation Sans',
    'font.size': 8,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'axes.edgecolor': '#555555',
    'text.color': '#111111',
    'axes.labelcolor': '#111111',
    'xtick.color': '#555555',
    'ytick.color': '#555555',
})

BLUE, RED, GREY = '#2a78d6', '#e34948', '#888888'
W = 6.25          # inches ~ 15.9 cm
rng = random.Random(0)


TIE_DRAWS = 200


def _ap_once(pairs, r):
    pairs = sorted(((s + r.random() * 1e-9, l) for s, l in pairs), key=lambda x: -x[0])
    P = sum(l for _, l in pairs)
    if not P:
        return float('nan')
    tp = 0
    a = 0.0
    for i, (_, l) in enumerate(pairs, 1):
        if l:
            tp += 1
            a += tp / i
    return a / P


def ap(pairs):
    """Tie-robust AP: the mean over TIE_DRAWS independent tie-breaks.

    Needed because 71% of designs score exactly 0 without the MSA, so a single
    random tie-break moves a small-subset AP by up to 0.07. Averaging makes the
    estimate deterministic and order-independent.
    """
    vals = [_ap_once(pairs, random.Random(k)) for k in range(TIE_DRAWS)]
    vals = [v for v in vals if v == v]
    return st.mean(vals) if vals else float('nan')


# ---------------------------------------------------------------- shared data
S1 = {r.split('\t')[0]: r.split('\t') for r in
      open(rpath('e2_scores.tsv')).read().splitlines()[1:]}
TGT = {b: v[1] for b, v in S1.items()}
LAB = {b: int(v[2]) for b, v in S1.items()}

SEED = defaultdict(lambda: defaultdict(dict))
for line in open(rpath('e2_seeds_scores.tsv')).read().splitlines()[1:]:
    b, c, s, lo, hi = line.split('\t')
    SEED[int(s)][c][b] = (float(lo), float(hi))


def per_target_dap(seed, metric=0):
    """metric 0 = ipSAEmin, 1 = ipSAEmax -> {target: dAP}"""
    on, off = defaultdict(list), defaultdict(list)
    for b in S1:
        if seed == 1:
            o, f = float(S1[b][3 + metric * 2]), float(S1[b][4 + metric * 2])
        else:
            if b not in SEED[seed]['full'] or b not in SEED[seed]['none']:
                continue
            o, f = SEED[seed]['full'][b][metric], SEED[seed]['none'][b][metric]
        on[TGT[b]].append((o, LAB[b]))
        off[TGT[b]].append((f, LAB[b]))
    return {t: ap(off[t]) - ap(on[t]) for t in on}


def _score_one(d):
    """Module level so multiprocessing can pickle it."""
    n = os.path.basename(d)
    head, _, cond = n.rpartition('__')
    top = [p for p in glob.glob(d + '/*_confidences.json')
           if 'summary' not in os.path.basename(p)]
    if not top:
        return None
    try:
        lo, hi, _ = ipsae_from_confidences(top[0])
    except Exception:
        return None
    return (head, cond, lo, hi)


def score_dirs(pattern):
    with Pool(24) as pool:
        return [r for r in pool.map(_score_one, sorted(glob.glob(pattern)), chunksize=8)
                if r]


# ------------------------------------------------------- fig 1: per-receptor
def fig1():
    d = {s: per_target_dap(s) for s in (1, 2, 3)}
    targets = sorted(d[1], key=lambda t: st.mean(d[s][t] for s in (1, 2, 3)))
    means = [st.mean(d[s][t] for s in (1, 2, 3)) for t in targets]
    lo = [min(d[s][t] for s in (1, 2, 3)) for t in targets]
    hi = [max(d[s][t] for s in (1, 2, 3)) for t in targets]

    fig, ax = plt.subplots(figsize=(W, 3.3))
    y = np.arange(len(targets))
    ax.barh(y, means, height=0.6,
            color=[BLUE if m < 0 else RED for m in means], zorder=2)
    for i in range(len(targets)):
        ax.plot([lo[i], hi[i]], [y[i], y[i]], color=GREY, lw=1.1, zorder=3,
                solid_capstyle='round')
    ax.axvline(0, color='#333333', lw=0.9, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(targets)
    ax.set_xlim(-0.72, 0.30)
    ax.set_ylim(-0.7, len(targets) - 0.3)
    ax.set_xlabel('Δ average precision when the target MSA is removed\n'
                  '(further left = the alignment mattered more)')
    ax.grid(axis='x', color='#e3e3e3', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig('fig1_per_receptor.png', dpi=220)
    plt.close(fig)
    print('fig1: per-receptor ΔAP, 3 seeds')


# -------------------------------------------------- fig 2: depth vs. effect
def fig2():
    neff = {}
    for line in open(rpath('neff80.tsv')).read().splitlines()[1:]:
        k, L, n, v, p = line.split('\t')
        t = k.rsplit('_B_', 1)[0].rsplit('_C_', 1)[0].rsplit('_D_', 1)[0]
        neff[t] = max(neff.get(t, 0), float(v))
    d = {s: per_target_dap(s) for s in (1, 2, 3)}
    pos = defaultdict(int)
    for b in S1:
        pos[TGT[b]] += LAB[b]
    pts = [(neff[t], st.mean(d[s][t] for s in (1, 2, 3)), t)
           for t in d[1] if t in neff and pos[t] >= 5]

    fig, ax = plt.subplots(figsize=(W, 2.9))
    for x, yv, t in pts:
        ax.scatter(x, yv, s=34, color=BLUE if yv < 0 else RED, zorder=3,
                   edgecolors='white', linewidths=0.6)
    ax.axhline(0, color='#333333', lw=0.9, zorder=2)
    for x, yv, t in pts:
        ax.annotate(t, (x, yv), textcoords='offset points', xytext=(0, 7),
                    ha='center', fontsize=6.6, color='#444444')
    ax.set_xscale('log')
    ax.set_xlim(20, 4000)
    ax.set_ylim(-0.72, 0.18)
    ax.set_xlabel('effective alignment depth of the target, Neff80 (log scale)')
    ax.set_ylabel('Δ average precision\n(MSA removed)')
    ax.grid(color='#e3e3e3', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.text(0.98, 0.06, 'Spearman ρ = +0.055  (n = 11)', transform=ax.transAxes,
            ha='right', fontsize=7.5, color='#111111')
    fig.tight_layout()
    fig.savefig('fig2_depth.png', dpi=220)
    plt.close(fig)
    print('fig2: Neff80 vs ΔAP')


# -------------------------------------------------------- fig 3: the 2 x 2
def fig3():
    ex = defaultdict(dict)
    for bid, cond, lo, hi in score_dirs('ex_out/*__ex*'):
        ex[bid][cond] = lo
    cells = defaultdict(lambda: defaultdict(list))
    for bid, v in ex.items():
        if len(v) != 2 or bid not in S1:
            continue
        t, lab = TGT[bid], LAB[bid]
        cells[t]['fm'].append((float(S1[bid][3]), lab))
        cells[t]['fn'].append((float(S1[bid][4]), lab))
        cells[t]['em'].append((v['exfull'], lab))
        cells[t]['en'].append((v['exnone'], lab))
    crop = ['Pdl1', 'IL7Ra', 'SARS_CoV2_RBD']          # sntx retained 100%
    m = [st.mean(ap(cells[t][k]) for t in crop) for k in ('fm', 'fn', 'em', 'en')]

    fig, ax = plt.subplots(figsize=(W, 2.7))
    x = np.array([0, 1])
    ax.bar(x - 0.17, [m[0], m[2]], width=0.32, color=BLUE, label='with the target MSA', zorder=2)
    ax.bar(x + 0.17, [m[1], m[3]], width=0.32, color=RED, label='MSA removed', zorder=2)
    for xi, v in zip([-0.17, 0.83, 0.17, 1.17], [m[0], m[2], m[1], m[3]]):
        ax.annotate(f'{v:.3f}', (xi, v), textcoords='offset points', xytext=(0, 3),
                    ha='center', fontsize=7.2)
    ax.annotate('', xy=(0.83, m[2]), xytext=(-0.17, m[0]),
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.1, ls='--'))
    ax.text(0.33, (m[0] + m[2]) / 2 + 0.035, '−0.176', color=BLUE, fontsize=7.5, ha='center')
    ax.annotate('', xy=(1.17, m[3]), xytext=(0.17, m[1]),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.1, ls='--'))
    ax.text(0.67, (m[1] + m[3]) / 2 - 0.055, '+0.049', color=RED, fontsize=7.5, ha='center')
    ax.axhline(0.107, color=GREY, lw=0.8, ls=':', zorder=1)
    ax.text(1.42, 0.113, 'random ranking', fontsize=6.6, color='#666666')
    ax.set_xticks(x)
    ax.set_xticklabels(['intact target', 'excised pocket'])
    ax.set_ylabel('average precision\n(higher is better)')
    ax.set_ylim(0, 0.70)
    ax.set_xlim(-0.55, 1.55)
    ax.legend(frameon=False, fontsize=7.4, loc='upper right')
    ax.grid(axis='y', color='#e3e3e3', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig('fig3_excision.png', dpi=220)
    plt.close(fig)
    print('fig3: the 2x2')


# ----------------------------------------------------- fig 4: depth ladder
def fig4():
    sc = defaultdict(dict)
    for bid, cond, lo, hi in score_dirs('e3_out/*__ex*') + score_dirs('ex_out/*__ex*'):
        sc[bid][cond] = hi
    cells = defaultdict(lambda: defaultdict(list))
    for bid, v in sc.items():
        if bid not in S1:
            continue
        for c, val in v.items():
            cells[TGT[bid]][c].append((val, LAB[bid]))
    crop = ['Pdl1', 'IL7Ra', 'SARS_CoV2_RBD']
    xs = [1, 2, 5, 10, 30, 90]                 # 90 stands in for "full" on a log axis
    series = {
        'random selection': ['exnone', 'exrand2', 'exrand5', 'exrand10', 'exrand30', 'exfull'],
        'diversity-maximising': ['exnone', 'exdiv2', 'exdiv5', 'exdiv10', 'exdiv30', 'exfull'],
    }
    fig, ax = plt.subplots(figsize=(W, 2.8))
    for (name, conds), colour, style in zip(series.items(), (BLUE, RED), ('-', '--')):
        ys = [st.mean(ap(cells[t][c]) for t in crop) for c in conds]
        ax.plot(xs, ys, style, color=colour, lw=1.5, marker='o', ms=4.2,
                label=name, zorder=3)
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels(['1', '2', '5', '10', '30', 'full'])
    ax.minorticks_off()
    ax.set_xlabel('effective depth of the cropped alignment, Neff80')
    ax.set_ylabel('average precision\n(higher is better)')
    ax.grid(color='#e3e3e3', lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.4, loc='upper left')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig('fig4_ladder.png', dpi=220)
    plt.close(fig)
    print('fig4: depth ladder')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
