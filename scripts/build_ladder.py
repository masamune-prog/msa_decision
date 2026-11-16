"""E3: depth x policy ladder on EXCISED targets.

For each design we take its cropped parent alignment and build subsets at
controlled effective depth, under two selection policies:

  random  -- shuffle, then take the shortest prefix reaching the target Neff80
  diverse -- greedily add the sequence most dissimilar to those already chosen,
             then take the shortest prefix reaching the target Neff80

The depth-matched random arm is not optional: without it, any claim about
*which* homologues matter is uninterpretable (see proposal 5.2).

Endpoints (Neff80 = 1 and full) already exist as ex_in_none / ex_in_full, so
only the intermediate rungs are generated here.
"""
import argparse
import json
import os
import random

import numpy as np

ROOT = os.environ.get('MSA_DECISION_ROOT', os.getcwd())
RUNGS = [2, 5, 10, 30]
MAX_POOL = 4000          # candidates considered for the greedy ordering
MAX_PICK = 400           # enough to reach Neff80 = 30 comfortably


def parse_a3m(text):
    """-> (names, aligned_seqs) with insertion columns removed."""
    names, seqs, cur, name = [], [], [], None
    for line in text.splitlines():
        if line.startswith('>'):
            if name is not None:
                seqs.append(''.join(cur))
                names.append(name)
            name, cur = line, []
        elif line:
            cur.append(line)
    if name is not None:
        seqs.append(''.join(cur))
        names.append(name)
    out_n, out_s = [], []
    if not seqs:
        return out_n, out_s
    L = len(''.join(c for c in seqs[0] if not c.islower()))
    for n, s in zip(names, seqs):
        a = ''.join(c for c in s if not c.islower())
        if len(a) == L:
            out_n.append(n)
            out_s.append(a)
    return out_n, out_s


def encode(seqs):
    if not seqs:
        return np.zeros((0, 0), dtype=np.uint8)
    arr = np.frombuffer(''.join(seqs).encode('ascii'), dtype=np.uint8)
    return arr.reshape(len(seqs), -1)


def neff80(arr):
    """Sum of inverse cluster sizes at 80% identity (same convention as
    compute_neff.py)."""
    n, L = arr.shape
    if n == 0:
        return 0.0
    need = int(np.ceil(0.8 * L))
    same = (arr[:, None, :] == arr[None, :, :]).sum(axis=2)
    counts = (same >= need).sum(axis=1)
    return float(np.sum(1.0 / counts))


def prefix_for_target(arr, order, target):
    """Shortest prefix of `order` whose Neff80 >= target (query always first)."""
    lo, hi = 1, len(order)
    best = None
    while lo <= hi:                      # binary search: Neff80 is monotone
        mid = (lo + hi) // 2
        if neff80(arr[order[:mid]]) >= target:
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return order[:best] if best else order[:len(order)]


def diverse_order(arr, limit=MAX_PICK):
    """Greedy max-min-distance ordering, starting from the query (row 0)."""
    n = arr.shape[0]
    chosen = [0]
    if n == 1:
        return chosen
    # hamming distance to the chosen set, maintained as a running minimum
    mind = (arr != arr[0]).sum(axis=1).astype(np.int32)
    mind[0] = -1
    for _ in range(min(limit, n) - 1):
        nxt = int(np.argmax(mind))
        if mind[nxt] <= 0:
            break
        chosen.append(nxt)
        d = (arr != arr[nxt]).sum(axis=1).astype(np.int32)
        np.minimum(mind, d, out=mind)
        mind[nxt] = -1
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src_dir', default=os.path.join(ROOT, 'ex_in_full'))
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    files = sorted(os.listdir(args.src_dir))
    written = 0
    for i, fn in enumerate(files):
        if not fn.endswith('.json'):
            continue
        bid = fn.replace('__exfull.json', '')
        with open(os.path.join(args.src_dir, fn)) as fh:
            spec = json.load(fh)
        prot = spec['sequences'][1]['protein']
        names, seqs = parse_a3m(prot.get('unpairedMsa') or '')
        if len(seqs) < 2:
            continue
        pool = min(len(seqs), MAX_POOL)
        arr = encode(seqs[:pool])

        rand_order = [0] + rng.sample(range(1, pool), pool - 1)
        div_order = diverse_order(arr)

        for policy, order in (('rand', rand_order), ('div', div_order)):
            for rung in RUNGS:
                idx = prefix_for_target(arr, order, rung)
                sub = '\n'.join(f'{names[j]}\n{seqs[j]}' for j in idx) + '\n'
                new = json.loads(json.dumps(spec))
                p = new['sequences'][1]['protein']
                p['unpairedMsa'] = sub
                p['pairedMsa'] = ''
                tag = f'{policy}{rung}'
                new['name'] = f'{bid}__ex{tag}'
                with open(os.path.join(args.out_dir,
                                       f'{bid}__ex{tag}.json'), 'w') as fh:
                    json.dump(new, fh)
                written += 1
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(files)} designs...', flush=True)
    print(f'wrote {written} ladder inputs to {args.out_dir}')


if __name__ == '__main__':
    main()
