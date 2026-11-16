"""Score every E2 fold and emit the two arms as a single table.

Walks e2_out/<binder_id>__<cond>/, computes ipSAE from the PAE matrix with our
own implementation (the one validated against the deposit in E0), and writes
e2_scores.tsv with one row per design carrying both conditions.
"""
import csv
import glob
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ipsae_recompute import ipsae_from_confidences

ROOT = os.environ.get('MSA_DECISION_ROOT', os.getcwd())
csv.field_size_limit(10**9)


def score_one(path):
    """path: an e2_out/<name> directory -> (binder_id, cond, ipsae_min, iptm)."""
    name = os.path.basename(path)
    if '__' not in name:
        return None
    binder_id, cond = name.rsplit('__', 1)
    top = [p for p in glob.glob(os.path.join(path, '*_confidences.json'))
           if 'summary' not in os.path.basename(p)]
    summ = glob.glob(os.path.join(path, '*_summary_confidences.json'))
    if not top or not summ:
        return None
    try:
        lo, hi, _ = ipsae_from_confidences(top[0])
        with open(summ[0]) as fh:
            iptm = json.load(fh)['iptm']
    except Exception as exc:          # a broken fold should not kill the run
        return ('ERROR', name, str(exc)[:80], 0.0)
    return (binder_id, cond, lo, hi, iptm)


def main():
    dirs = sorted(glob.glob(os.path.join(ROOT, 'e2_out', '*__*')))
    print(f'scoring {len(dirs)} folds...')
    with Pool(24) as pool:
        results = pool.map(score_one, dirs, chunksize=16)

    errors = [r for r in results if r and r[0] == 'ERROR']
    good = [r for r in results if r and r[0] != 'ERROR']
    if errors:
        print(f'  {len(errors)} folds failed to score, e.g. {errors[0][1]}: {errors[0][2]}')

    table = {}
    for binder_id, cond, lo, hi, iptm in good:
        table.setdefault(binder_id, {})[cond] = (lo, hi, iptm)

    both = {k: v for k, v in table.items() if 'full' in v and 'none' in v}
    print(f'  scored {len(good)} folds -> {len(both)} designs with BOTH arms')

    with open(os.path.join(ROOT, 'final_dataset.csv'), newline='') as fh:
        meta = {r['binder_id']: r for r in csv.DictReader(fh)}

    out = os.path.join(ROOT, 'e2_scores.tsv')
    with open(out, 'w') as fh:
        fh.write('binder_id\ttarget_id\tlabel\tfull_ipsae_min\tnone_ipsae_min'
                 '\tfull_ipsae_max\tnone_ipsae_max\tfull_iptm\tnone_iptm\n')
        for bid in sorted(both):
            m = meta.get(bid)
            if not m:
                continue
            f, n = both[bid]['full'], both[bid]['none']
            lab = 1 if m['binder'].strip().lower() == 'true' else 0
            fh.write(f"{bid}\t{m['target_id']}\t{lab}\t{f[0]:.6f}\t{n[0]:.6f}"
                     f"\t{f[1]:.6f}\t{n[1]:.6f}\t{f[2]:.4f}\t{n[2]:.4f}\n")
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
