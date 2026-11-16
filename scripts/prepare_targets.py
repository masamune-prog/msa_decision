"""Extract the unique target sequences that need a homology search.

A target's homologues do not depend on which binder is attached, so we search
once per distinct (target_id, chain, sequence) rather than once per design.
Writes:
  targets/manifest.tsv   one row per unique target sequence
  targets/<key>.json     AF3 input JSON for the data-pipeline run
"""
import csv
import json
import os
import hashlib
from collections import defaultdict

ROOT = os.environ.get('MSA_DECISION_ROOT', os.getcwd())
CSV = os.path.join(ROOT, 'final_dataset.csv')
OUT = os.path.join(ROOT, 'targets')
csv.field_size_limit(10**9)


def target_chains(row):
    """Yield (chain_id, sequence) for each target chain that carries an MSA."""
    try:
        info = json.loads(row.get('msa_info') or '[]')
    except ValueError:
        return
    for entry in info:
        chain, _, mode = entry.partition(':')
        if mode != 'run_msa':
            continue
        seq = row.get(f'target_subchain_{chain}_seq') or ''
        if not seq and chain == 'B':
            seq = row.get('B_seq') or ''
        if seq:
            yield chain, seq


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(CSV, newline='') as fh:
        rows = [r for r in csv.DictReader(fh) if r['binder_id']]

    # unique (target_id, chain, seq) -> how many designs rely on it
    uniq = defaultdict(int)
    for r in rows:
        for chain, seq in target_chains(r):
            uniq[(r['target_id'], chain, seq)] += 1

    manifest = []
    for (tid, chain, seq), ndesigns in sorted(uniq.items()):
        digest = hashlib.sha1(seq.encode()).hexdigest()[:8]
        key = f'{tid}_{chain}_{digest}'
        # AF3 input JSON: one protein chain, data pipeline will fill the MSA
        spec = {
            'name': key,
            'modelSeeds': [1],
            'sequences': [{'protein': {'id': 'A', 'sequence': seq}}],
            'dialect': 'alphafold3',
            'version': 1,
        }
        with open(os.path.join(OUT, f'{key}.json'), 'w') as fh:
            json.dump(spec, fh, indent=2)
        manifest.append((key, tid, chain, len(seq), ndesigns, digest))

    with open(os.path.join(OUT, 'manifest.tsv'), 'w') as fh:
        fh.write('key\ttarget_id\tchain\tlength\tn_designs\tsha1\n')
        for row in manifest:
            fh.write('\t'.join(str(x) for x in row) + '\n')

    print(f'unique target sequences needing a homology search: {len(manifest)}')
    print(f'{"key":<34}{"target":<16}{"ch":>3}{"len":>6}{"designs":>9}')
    for key, tid, chain, L, n, _ in manifest:
        print(f'{key:<34}{tid:<16}{chain:>3}{L:>6}{n:>9}')
    print(f'\nwrote {len(manifest)} input JSONs + manifest.tsv to {OUT}')


if __name__ == '__main__':
    main()
