"""Build the SELF-TEMPLATE arm: full target, +/- MSA, with the target's own
structure supplied as a template.

Kim et al. fold with an "inline self-template" in every arm; our E2 disabled
templates entirely. A template and an MSA are partly redundant sources of the
same information (how the target folds), so this arm tests whether the MSA
penalty we measured is an artefact of having no structural prior at all.

The template is the target chain extracted from OUR OWN full+MSA model, so the
comparison stays self-consistent.
"""
import argparse
import csv
import hashlib
import functools
import json
import os

ROOT = os.environ.get('MSA_DECISION_ROOT', os.getcwd())
csv.field_size_limit(10**9)


def single_chain_cif(src_cif, keep_chain):
    """Return an mmCIF containing only `keep_chain`.

    Both _atom_site and _pdbx_poly_seq_scheme must be filtered consistently --
    AF3's template parser cross-references them, and renaming the chain in one
    but not the other makes the lookup fail. The chain keeps its original id.
    """
    out = []
    section = None          # which loop's data rows we are currently inside
    for raw in open(src_cif):
        line = raw.rstrip('\n')
        s = line.strip()
        if s.startswith('_pdbx_poly_seq_scheme.'):
            section = 'scheme'
            out.append(line)
            continue
        if s.startswith('_atom_site.'):
            section = 'atom'
            out.append(line)
            continue
        if s.startswith(('_', 'loop_', '#', 'data_')) or not s:
            if s.startswith(('loop_', '#', 'data_')) or not s:
                section = None
            out.append(line)
            continue
        if section == 'scheme':
            f = s.split()
            # asym_id is the first column in this loop
            if f and f[0] != keep_chain:
                continue
        elif section == 'atom':
            f = s.split()
            # auth_asym_id
            if len(f) > 16 and f[16] != keep_chain:
                continue
        out.append(line)
    # AF3 filters templates by release date, which our predicted models lack.
    # Use a fixed early date so the template is never date-filtered out.
    out += [
        '#',
        'loop_',
        '_pdbx_audit_revision_history.ordinal',
        '_pdbx_audit_revision_history.data_content_type',
        '_pdbx_audit_revision_history.major_revision',
        '_pdbx_audit_revision_history.minor_revision',
        '_pdbx_audit_revision_history.revision_date',
        "1 'Structure model' 1 0 1970-01-01",
        '#',
    ]
    return '\n'.join(out) + '\n'


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
    ap.add_argument('--condition', required=True, choices=['full', 'none'])
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    ids = [l.strip() for l in open(args.ids_file) if l.strip()]
    with open(os.path.join(ROOT, 'final_dataset.csv'), newline='') as fh:
        meta = {r['binder_id']: r for r in csv.DictReader(fh)}

    os.makedirs(args.out_dir, exist_ok=True)
    tdir = os.path.join(ROOT, 'tmpl_cif')
    os.makedirs(tdir, exist_ok=True)

    written = skipped = 0
    for bid in ids:
        row = meta.get(bid)
        src = os.path.join(ROOT, 'e2_out', f'{bid}__full',
                           f'{bid}__full_model.cif')
        if not row or not os.path.exists(src):
            skipped += 1
            continue
        chain = 'B'
        tseq = row.get('target_subchain_B_seq') or row.get('B_seq') or ''
        if not tseq:
            skipped += 1
            continue

        cif_path = os.path.join(tdir, f'{bid}_tgt.cif')
        if not os.path.exists(cif_path):
            with open(cif_path, 'w') as fh:
                fh.write(single_chain_cif(src, chain))

        idx = list(range(len(tseq)))
        prot = {
            'id': chain,
            'sequence': tseq,
            'templates': [{
                'mmcifPath': cif_path,
                'queryIndices': idx,
                'templateIndices': idx,
            }],
        }
        if args.condition == 'full':
            key = f'{row["target_id"]}_{chain}_' \
                  f'{hashlib.sha1(tseq.encode()).hexdigest()[:8]}'
            up, pa = parent_msa(key)
            prot['unpairedMsa'], prot['pairedMsa'] = up, pa
        else:
            prot['unpairedMsa'], prot['pairedMsa'] = '', ''

        spec = {
            'name': f'{bid}__tm{args.condition}',
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
                               f'{bid}__tm{args.condition}.json'), 'w') as fh:
            json.dump(spec, fh)
        written += 1

    print(f'wrote {written} self-template inputs ({args.condition}), '
          f'skipped {skipped}')


if __name__ == '__main__':
    main()
