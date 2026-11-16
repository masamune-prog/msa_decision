"""Build EXCISED-target inputs: the cropping arm of the 2x2.

For each design we take our own full-target model, find the target residues
within a 3D halo of any binder atom, take the single contiguous window that
spans them (as Kim et al. do), and crop both the target sequence AND its MSA
columns to that window.

This keeps the crop self-consistent with our own folds rather than importing
someone else's interface definition.
"""
import argparse
import csv
import functools
import hashlib
import json
import os

import numpy as np

ROOT = os.environ.get('MSA_DECISION_ROOT', os.getcwd())
HALO = 11.0          # Angstrom, matching the source paper's halo
csv.field_size_limit(10**9)


def parse_cif_atoms(path):
    """-> list of (chain, resnum, x, y, z) using author numbering."""
    out = []
    with open(path) as fh:
        for line in fh:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            f = line.split()
            if len(f) < 18:
                continue
            try:
                out.append((f[16], int(f[15]), float(f[10]), float(f[11]),
                            float(f[12])))
            except ValueError:
                continue
    return out


def crop_window(cif_path, binder_chain='A', halo=HALO):
    """-> (target_chain, lo, hi) contiguous 1-based residue window, or None."""
    atoms = parse_cif_atoms(cif_path)
    if not atoms:
        return None
    binder = np.array([(a[2], a[3], a[4]) for a in atoms
                       if a[0] == binder_chain], dtype=float)
    tgt = [a for a in atoms if a[0] != binder_chain]
    if binder.size == 0 or not tgt:
        return None
    chain = tgt[0][0]
    tgt = [a for a in tgt if a[0] == chain]
    coords = np.array([(a[2], a[3], a[4]) for a in tgt], dtype=float)
    resnums = np.array([a[1] for a in tgt])
    # min distance from each target atom to any binder atom
    keep = np.zeros(len(coords), dtype=bool)
    for i in range(0, len(coords), 2048):
        block = coords[i:i + 2048]
        d = np.sqrt(((block[:, None, :] - binder[None, :, :]) ** 2).sum(-1))
        keep[i:i + 2048] = d.min(axis=1) <= halo
    if not keep.any():
        return None
    sel = resnums[keep]
    return chain, int(sel.min()), int(sel.max())


def crop_a3m(a3m, lo, hi, qlen):
    """Keep only alignment columns lo..hi (1-based, inclusive)."""
    if not a3m:
        return ''
    out = []
    name = None
    for line in a3m.splitlines():
        if line.startswith('>'):
            name = line
            continue
        if name is None:
            continue
        # drop insertions, then slice the aligned columns
        aligned = ''.join(c for c in line if not c.islower())
        if len(aligned) != qlen:
            name = None
            continue
        seg = aligned[lo - 1:hi]
        if seg.replace('-', ''):          # skip sequences empty in the window
            out.append(f'{name}\n{seg}')
        name = None
    return '\n'.join(out) + ('\n' if out else '')


@functools.lru_cache(maxsize=None)
def parent_msa(key):
    path = os.path.join(ROOT, 'msa_out', key, f'{key}_data.json')
    with open(path) as fh:
        d = json.load(fh)
    for c in d['sequences']:
        p = c.get('protein')
        if p:
            return p.get('unpairedMsa') or '', p.get('pairedMsa') or ''
    return '', ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids_file', required=True)
    ap.add_argument('--condition', required=True, choices=['full', 'none'],
                    help='MSA condition for the excised target')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    with open(args.ids_file) as fh:
        wanted = [l.strip() for l in fh if l.strip()]
    with open(os.path.join(ROOT, 'final_dataset.csv'), newline='') as fh:
        meta = {r['binder_id']: r for r in csv.DictReader(fh)}

    os.makedirs(args.out_dir, exist_ok=True)
    stats = []
    written = skipped = 0
    for bid in wanted:
        row = meta.get(bid)
        if not row:
            skipped += 1
            continue
        cif = os.path.join(ROOT, 'e2_out', f'{bid}__full',
                           f'{bid}__full_model.cif')
        if not os.path.exists(cif):
            skipped += 1
            continue
        win = crop_window(cif)
        if not win:
            skipped += 1
            continue
        chain, lo, hi = win
        tseq = row.get(f'target_subchain_{chain}_seq') or row.get('B_seq') or ''
        if not tseq or hi > len(tseq):
            skipped += 1
            continue
        cropped = tseq[lo - 1:hi]
        stats.append((len(tseq), len(cropped)))

        prot = {'id': chain, 'sequence': cropped, 'templates': []}
        if args.condition == 'full':
            key = f'{row["target_id"]}_{chain}_' \
                  f'{hashlib.sha1(tseq.encode()).hexdigest()[:8]}'
            up, _ = parent_msa(key)
            prot['unpairedMsa'] = crop_a3m(up, lo, hi, len(tseq))
            prot['pairedMsa'] = ''
        else:
            prot['unpairedMsa'] = ''
            prot['pairedMsa'] = ''

        spec = {
            'name': f'{bid}__ex{args.condition}',
            'modelSeeds': [1],
            'sequences': [
                {'protein': {'id': 'A', 'sequence': row['A_seq'],
                             'unpairedMsa': '', 'pairedMsa': '',
                             'templates': []}},
                {'protein': prot},
            ],
            'dialect': 'alphafold3',
            'version': 1,
        }
        with open(os.path.join(args.out_dir,
                               f'{bid}__ex{args.condition}.json'), 'w') as fh:
            json.dump(spec, fh)
        written += 1

    print(f'wrote {written} excised inputs ({args.condition}), skipped {skipped}')
    if stats:
        full = sum(s[0] for s in stats) / len(stats)
        crop = sum(s[1] for s in stats) / len(stats)
        print(f'  mean target length {full:.0f} aa -> cropped {crop:.0f} aa '
              f'({crop/full:.0%} retained)')


if __name__ == '__main__':
    main()
