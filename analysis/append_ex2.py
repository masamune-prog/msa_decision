"""Append ex2_out (excision, remaining 9 receptors) to all_af3_folds.csv with the same columns."""
import csv, glob, json, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compile_all_folds as C
import add_features as F
ROOT = C.ROOT; csv.field_size_limit(10**9)

def work(path):
    row = C.one(('ex2_out', path))
    if row is None: return None
    row.update(F.feat(path))
    d = json.load(open(glob.glob(f'{path}/*_data.json')[0]))
    row['receptor_seq'] = ':'.join(s['protein']['sequence'] for s in d['sequences'] if 'protein' in s and s['protein']['id'] != 'A')
    return row

if __name__ == '__main__':
    old = list(csv.DictReader(open(f'{ROOT}/all_af3_folds.csv')))
    cols = list(old[0])
    dirs = sorted(p for p in glob.glob(f'{ROOT}/ex2_out/*') if os.path.isdir(p))
    with Pool(40) as p: new = [r for r in p.imap(work, dirs, chunksize=8) if r]
    meta = {}
    for r in csv.DictReader(open(f'{ROOT}/final_dataset.csv')): meta[r['binder_id']] = r
    lab = {r['binder_id']: (r['target_id'], r['label']) for r in csv.DictReader(open(f'{ROOT}/e2_scores.tsv'), delimiter='\t')}
    for r in new:
        r['target_id'], r['label'] = lab[r['binder_id']]
        r['binder_seq'] = meta[r['binder_id']]['A_seq']
        r['fold_dir'] = os.path.relpath(r['fold_dir'], ROOT) if os.path.isabs(r['fold_dir']) else r['fold_dir']
    with open(f'{ROOT}/all_af3_folds.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols, extrasaction='ignore'); w.writeheader(); w.writerows(old + new)
    print('appended', len(new), 'total', len(old) + len(new), 'errors', sum(1 for r in new if r.get('error') or r.get('feat_error')))
