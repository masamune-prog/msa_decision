"""Measure effective alignment depth (Neff80) for every target.

This is the quantity the source paper asserts a rule about but never reports.

Convention (fixed here, must not change later):
  * a3m lowercase columns are insertions relative to the query and are dropped,
    so every sequence is aligned to the query length L.
  * identity(i, j) = fraction of the L positions where the two characters match,
    gaps included as an ordinary symbol.
  * cluster size n_i = number of sequences with identity >= 0.8 to sequence i
    (including itself).
  * Neff80 = sum_i 1 / n_i.

Computed on the UNPAIRED MSA: the target's own homologues from the genetic
search. The paired MSA exists for cross-chain pairing and is reported for
reference only.
"""
import glob
import json
import os

import numpy as np

ROOT = os.environ.get('MSA_DECISION_ROOT', os.getcwd())
THRESH = 0.8
CHUNK = 256


def parse_a3m(text, qlen):
    """-> uint8 array (n_seq, qlen); insertion (lowercase) columns removed."""
    seqs = []
    cur = []
    for line in text.splitlines():
        if line.startswith('>'):
            if cur:
                seqs.append(''.join(cur))
                cur = []
        elif line:
            cur.append(line)
    if cur:
        seqs.append(''.join(cur))
    keep = []
    for s in seqs:
        stripped = ''.join(c for c in s if not c.islower())
        if len(stripped) == qlen:
            keep.append(stripped)
    if not keep:
        return np.zeros((0, qlen), dtype=np.uint8)
    arr = np.frombuffer(''.join(keep).encode('ascii'), dtype=np.uint8)
    return arr.reshape(len(keep), qlen)


def neff(arr, thresh=THRESH):
    """Sum of inverse cluster sizes at the given identity threshold."""
    n, L = arr.shape
    if n == 0:
        return 0.0, 0
    counts = np.zeros(n, dtype=np.int32)
    need = int(np.ceil(thresh * L))
    for i in range(0, n, CHUNK):
        block = arr[i:i + CHUNK]
        same = (block[:, None, :] == arr[None, :, :]).sum(axis=2)
        counts[i:i + CHUNK] = (same >= need).sum(axis=1)
    return float(np.sum(1.0 / counts)), n


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(ROOT, 'msa_out', '*', '*_data.json'))):
        key = os.path.basename(os.path.dirname(path))
        with open(path) as fh:
            data = json.load(fh)
        for chain in data['sequences']:
            prot = chain.get('protein')
            if not prot:
                continue
            qlen = len(prot['sequence'])
            arr = parse_a3m(prot.get('unpairedMsa') or '', qlen)
            value, n_used = neff(arr)
            paired = (prot.get('pairedMsa') or '').count('>')
            rows.append((key, qlen, n_used, value, paired))

    rows.sort(key=lambda r: -r[3])
    print(f'{"target sequence":<30}{"len":>5}{"n_seq":>8}{"Neff80":>10}'
          f'{"Neff/seq":>10}{"paired":>8}')
    for key, qlen, n, value, paired in rows:
        frac = value / n if n else 0.0
        print(f'{key[:28]:<30}{qlen:>5}{n:>8}{value:>10.1f}{frac:>10.3f}{paired:>8}')

    with open(os.path.join(ROOT, 'neff80.tsv'), 'w') as fh:
        fh.write('key\tquery_len\tn_seq\tneff80\tpaired\n')
        for r in rows:
            fh.write('\t'.join(str(x) for x in r) + '\n')
    print(f'\nwrote {ROOT}/neff80.tsv')


if __name__ == '__main__':
    main()
