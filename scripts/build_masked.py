"""Exp 2 missing arm: INTACT target + alignment restricted to the 11 A window.

Columns outside the window are replaced by gaps in every alignment row, and rows empty in the
window are dropped (the same row set the excised +MSA arm keeps). The target stays full length.
Comparing this with `full` isolates cropping the ALIGNMENT; comparing it with `exfull` isolates
cropping the TARGET."""
import csv, hashlib, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_excised as B
ROOT = B.ROOT
csv.field_size_limit(10**9)

def mask_a3m(a3m, lo, hi, qlen):
    out, name = [], None
    for line in a3m.splitlines():
        if line.startswith('>'): name = line; continue
        if name is None: continue
        al = ''.join(c for c in line if not c.islower())
        if len(al) != qlen: name = None; continue
        seg = al[lo - 1:hi]
        if seg.replace('-', ''):
            out.append(f'{name}\n' + '-' * (lo - 1) + seg + '-' * (qlen - hi))
        name = None
    return '\n'.join(out) + ('\n' if out else '')

if __name__ == '__main__':
    out_dir = sys.argv[1]; os.makedirs(out_dir, exist_ok=True)
    ids = sorted({l.strip() for f in ('ex_ids.txt', 'ex2_ids.txt') for l in open(os.path.join(ROOT, f)) if l.strip()})
    meta = {r['binder_id']: r for r in csv.DictReader(open(os.path.join(ROOT, 'final_dataset.csv')))}
    n = skipped = 0
    for bid in ids:
        row = meta[bid]
        win = B.crop_window(os.path.join(ROOT, 'e2_out', f'{bid}__full', f'{bid}__full_model.cif'))
        if not win: skipped += 1; continue
        chain, lo, hi = win
        tseq = row.get(f'target_subchain_{chain}_seq') or row.get('B_seq') or ''
        if not tseq or hi > len(tseq): skipped += 1; continue
        key = f'{row["target_id"]}_{chain}_{hashlib.sha1(tseq.encode()).hexdigest()[:8]}'
        up, _ = B.parent_msa(key)
        spec = {'name': f'{bid}__mskfull', 'modelSeeds': [1], 'dialect': 'alphafold3', 'version': 1,
                'sequences': [
                    {'protein': {'id': 'A', 'sequence': row['A_seq'], 'unpairedMsa': '', 'pairedMsa': '', 'templates': []}},
                    {'protein': {'id': chain, 'sequence': tseq, 'templates': [],
                                 'unpairedMsa': mask_a3m(up, lo, hi, len(tseq)), 'pairedMsa': ''}}]}
        json.dump(spec, open(os.path.join(out_dir, f'{bid}__mskfull.json'), 'w')); n += 1
    print('wrote', n, 'skipped', skipped)
