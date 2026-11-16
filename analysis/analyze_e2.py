"""E2 analysis: does removing the target MSA change binder ranking?

Takes two score tables (MSA-on, MSA-off) over the same designs and reports:
  * per-target AP and ROC-AUC for each arm
  * paired per-target delta with a bootstrap CI (target = unit of replication)
  * the H1 decomposition: does the change come from moving the POSITIVE class
    or the NEGATIVE class?
  * the sign distribution of per-target deltas (is the effect a mixture?)

Run with --selftest to validate the machinery with no folding required.
"""
import argparse
import math
import random
from collections import defaultdict


def average_precision(pairs):
    """pairs: (score, label). Ties broken by random jitter upstream."""
    pairs = sorted(pairs, key=lambda x: -x[0])
    P = sum(l for _, l in pairs)
    if P == 0:
        return float('nan')
    tp = 0
    s = 0.0
    for i, (_, l) in enumerate(pairs, 1):
        if l:
            tp += 1
            s += tp / i
    return s / P


def roc_auc(pairs):
    pairs = sorted(pairs, key=lambda x: x[0])
    P = sum(l for _, l in pairs)
    N = len(pairs) - P
    if P == 0 or N == 0:
        return float('nan')
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sp = sum(r for r, (_, l) in zip(ranks, pairs) if l)
    return (sp - P * (P + 1) / 2) / (P * N)


def jitter(pairs, rng, eps=1e-9):
    """Break exact ties at random so AP does not depend on input order."""
    return [(s + rng.random() * eps, l) for s, l in pairs]


def per_target(scores, labels, targets):
    """-> {target: [(score, label), ...]}"""
    out = defaultdict(list)
    for d in scores:
        out[targets[d]].append((scores[d], labels[d]))
    return out


def paired_bootstrap(on, off, labels, targets, n_boot=2000, seed=0):
    """Resample TARGETS (the unit of replication), recompute mean delta AP."""
    rng = random.Random(seed)
    g_on = per_target(on, labels, targets)
    g_off = per_target(off, labels, targets)
    keys = [k for k in g_on if not math.isnan(average_precision(g_on[k]))]

    def mean_delta(ks):
        ds = []
        for k in ks:
            a = average_precision(jitter(g_off[k], rng))
            b = average_precision(jitter(g_on[k], rng))
            if not (math.isnan(a) or math.isnan(b)):
                ds.append(a - b)
        return sum(ds) / len(ds) if ds else float('nan')

    point = mean_delta(keys)
    boots = []
    for _ in range(n_boot):
        ks = [rng.choice(keys) for _ in keys]
        v = mean_delta(ks)
        if not math.isnan(v):
            boots.append(v)
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    return point, lo, hi, keys


def h1_decomposition(on, off, labels):
    """Did the POSITIVE class move up, or the NEGATIVE class move down?"""
    dpos = [off[d] - on[d] for d in on if labels[d] == 1]
    dneg = [off[d] - on[d] for d in on if labels[d] == 0]
    m = lambda v: sum(v) / len(v) if v else float('nan')
    return m(dpos), m(dneg), len(dpos), len(dneg)


def report(on, off, labels, targets, n_boot=2000, seed=0):
    rng = random.Random(seed)
    g_on = per_target(on, labels, targets)
    g_off = per_target(off, labels, targets)

    print(f'{"target":<18}{"n":>6}{"pos":>5}{"AP_on":>9}{"AP_off":>9}'
          f'{"dAP":>9}{"AUC_on":>9}{"AUC_off":>9}')
    deltas = {}
    for k in sorted(g_on, key=lambda x: -sum(l for _, l in g_on[x])):
        a = average_precision(jitter(g_on[k], rng))
        b = average_precision(jitter(g_off[k], rng))
        if math.isnan(a) or math.isnan(b):
            continue
        deltas[k] = b - a
        print(f'{k:<18}{len(g_on[k]):>6}{sum(l for _,l in g_on[k]):>5}'
              f'{a:>9.3f}{b:>9.3f}{b-a:>+9.3f}'
              f'{roc_auc(g_on[k]):>9.3f}{roc_auc(g_off[k]):>9.3f}')

    point, lo, hi, keys = paired_bootstrap(on, off, labels, targets, n_boot, seed)
    print(f'\nmean per-target dAP = {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]'
          f'  (n={len(keys)} targets, {n_boot} paired bootstrap)')
    print(f'  CI {"EXCLUDES" if (lo > 0 or hi < 0) else "INCLUDES"} zero')

    pos = sum(1 for v in deltas.values() if v > 0)
    neg = sum(1 for v in deltas.values() if v < 0)
    print(f'\nsign of per-target effect: {pos} improved, {neg} worsened'
          f'  -> {"MIXTURE (effect flips sign)" if pos and neg else "consistent direction"}')

    dpos, dneg, npos, nneg = h1_decomposition(on, off, labels)
    print(f'\nH1 decomposition (mean score shift, off - on):')
    print(f'  positives (n={npos:>5}): {dpos:+.4f}')
    print(f'  negatives (n={nneg:>5}): {dneg:+.4f}')
    if abs(dpos) < 1e-12 and abs(dneg) < 1e-12:
        print('  -> no score change in either class')
    elif abs(dneg) > 2 * abs(dpos):
        print('  -> consistent with CALIBRATION: negatives moved, positives did not')
    elif abs(dpos) > 2 * abs(dneg):
        print('  -> consistent with ACCURACY: positives moved, negatives did not')
    else:
        print('  -> both classes moved; H1 not cleanly separable')


def selftest():
    """Validate the machinery without any folding."""
    rng = random.Random(7)
    N, T = 1200, 10
    labels, targets, on = {}, {}, {}
    for i in range(N):
        d = f'd{i}'
        targets[d] = f't{i % T}'
        labels[d] = 1 if rng.random() < 0.11 else 0
        on[d] = rng.random() * 0.4 + (0.3 if labels[d] else 0.0)

    print('=== CASE A: identical arms -> dAP must be ~0, CI must contain 0 ===')
    report(on, dict(on), labels, targets, n_boot=400)

    print('\n\n=== CASE B: negatives pushed DOWN only (pure calibration) ===')
    off = {d: (on[d] if labels[d] else on[d] - 0.15) for d in on}
    report(on, off, labels, targets, n_boot=400)

    print('\n\n=== CASE C: positives pushed UP only (pure accuracy) ===')
    off = {d: (on[d] + 0.15 if labels[d] else on[d]) for d in on}
    report(on, off, labels, targets, n_boot=400)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        print('Supply score tables, or run with --selftest.')
