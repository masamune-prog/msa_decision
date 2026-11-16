"""Build AlphaFold 3 input JSONs for the MSA ablation.

Every complex is written twice-capable: the binder chain is ALWAYS
single-sequence (it has no homologues by construction), and the target chain
carries whichever MSA condition we are testing.

Conditions (AF3 input.md semantics):
  full  -> unpairedMsa = parent a3m, pairedMsa = ""   (option 2)
  none  -> unpairedMsa = "",         pairedMsa = ""   (option 4, MSA-free)

Templates are disabled everywhere (`templates: []`).
"""
import argparse
import csv
import functools
import hashlib
import json
import os

ROOT = os.environ.get('MSA_DECISION_ROOT', os.getcwd())
csv.field_size_limit(10**9)


def target_chains(row):
    """Yield (chain_id, sequence) for target chains that carry an MSA."""
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


def msa_key(target_id, chain, seq):
    return f'{target_id}_{chain}_{hashlib.sha1(seq.encode()).hexdigest()[:8]}'


@functools.lru_cache(maxsize=None)
def load_parent_msa(key):
    """-> (unpairedMsa, pairedMsa) produced by our data-pipeline run.

    Cached: thousands of designs share one target alignment.
    """
    path = os.path.join(ROOT, 'msa_out', key, f'{key}_data.json')
    with open(path) as fh:
        data = json.load(fh)
    for chain in data['sequences']:
        prot = chain.get('protein')
        if prot:
            return prot.get('unpairedMsa') or '', prot.get('pairedMsa') or ''
    raise ValueError(f'no protein chain in {path}')


def single_seq_a3m(seq, name='query'):
    return f'>{name}\n{seq}\n'


def build(row, condition, seed=1):
    binder = row['A_seq']
    seqs = [{
        'protein': {
            'id': 'A',
            'sequence': binder,
            # the binder is novel: no homologues, ever
            'unpairedMsa': '',
            'pairedMsa': '',
            'templates': [],
        }
    }]
    for chain, seq in target_chains(row):
        prot = {'id': chain, 'sequence': seq, 'templates': []}
        if condition == 'full':
            # Reproduce the deposit's condition: BOTH alignments exactly as the
            # data pipeline produced them (input.md option 5). Dropping the
            # paired MSA loses most of the alignment -- for FGFR2 it is 26,516
            # paired vs 5,948 unpaired sequences -- and measurably lowers ipSAE.
            unpaired, paired = load_parent_msa(
                msa_key(row['target_id'], chain, seq))
            if unpaired and not unpaired.startswith('>'):
                unpaired = single_seq_a3m(seq) + unpaired
            prot['unpairedMsa'] = unpaired
            prot['pairedMsa'] = paired
        elif condition == 'none':
            prot['unpairedMsa'] = ''
            prot['pairedMsa'] = ''
        else:
            raise ValueError(condition)
        seqs.append({'protein': prot})

    return {
        'name': f"{row['binder_id']}__{condition}__s{seed}",
        'modelSeeds': [seed],
        'sequences': seqs,
        'dialect': 'alphafold3',
        'version': 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids_file', required=True,
                    help='file with one binder_id per line')
    ap.add_argument('--condition', required=True, choices=['full', 'none'])
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    with open(args.ids_file) as fh:
        wanted = {l.strip() for l in fh if l.strip()}
    with open(os.path.join(ROOT, 'final_dataset.csv'), newline='') as fh:
        rows = {r['binder_id']: r for r in csv.DictReader(fh)
                if r['binder_id'] in wanted}

    os.makedirs(args.out_dir, exist_ok=True)
    written = skipped = 0
    for bid in sorted(wanted):
        row = rows.get(bid)
        if row is None:
            print(f'  [miss] {bid} not in CSV')
            skipped += 1
            continue
        try:
            spec = build(row, args.condition, args.seed)
        except FileNotFoundError:
            print(f'  [miss] {bid}: no parent MSA for its target')
            skipped += 1
            continue
        safe = bid.replace('/', '_')
        with open(os.path.join(args.out_dir, f'{safe}__{args.condition}.json'),
                  'w') as fh:
            json.dump(spec, fh)
        written += 1

    print(f'wrote {written} input JSONs ({args.condition}) to {args.out_dir}')
    if skipped:
        print(f'skipped {skipped}')


if __name__ == '__main__':
    main()
